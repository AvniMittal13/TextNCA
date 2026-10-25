"""Training loop for TextNCA experiments.

Supports: wandb logging, gradient accumulation, mixed precision, crash recovery.
"""

import os
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Callable

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from textnca.training.scheduler import WarmupCosineScheduler


@dataclass
class TrainConfig:
    """Training hyperparameters."""
    # Optimization
    lr: float = 3e-4
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.98)
    eps: float = 1e-8
    grad_clip: float = 1.0
    grad_accum_steps: int = 1

    # Schedule
    warmup_steps: int = 1000
    total_steps: int = 50000
    min_lr: float = 0.0

    # Precision
    use_amp: bool = True  # Mixed precision (bf16 on A100, fp16 on T4)
    amp_dtype: str = "bfloat16"  # "bfloat16" or "float16"

    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    checkpoint_every: int = 2000
    resume_from: Optional[str] = None

    # Logging
    log_every: int = 100
    eval_every: int = 1000
    use_wandb: bool = False
    wandb_project: str = "textnca"
    wandb_run_name: Optional[str] = None

    # Misc
    seed: int = 42
    max_eval_batches: int = 50


class Trainer:
    """Generic trainer for TextNCA and baselines.

    Handles checkpointing, crash recovery, mixed precision and wandb logging.
    Supports DDP via rank/world_size arguments.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dl: DataLoader,
        eval_dl: Optional[DataLoader],
        config: TrainConfig,
        loss_fn: Optional[Callable] = None,
        device: str = "cuda",
        rank: int = 0,
        world_size: int = 1,
    ):
        self.model = model.to(device)
        self.train_dl = train_dl
        self.eval_dl = eval_dl
        self.config = config
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.is_main = (rank == 0)
        self.loss_fn = loss_fn or nn.CrossEntropyLoss(ignore_index=-100)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
        )

        # Scheduler
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_steps=config.warmup_steps,
            total_steps=config.total_steps,
            min_lr=config.min_lr,
        )

        # Mixed precision
        self.amp_dtype = getattr(torch, config.amp_dtype, torch.float16)
        self.use_amp = config.use_amp and torch.cuda.is_available()
        self.scaler = GradScaler("cuda", enabled=self.use_amp and self.amp_dtype == torch.float16)

        # State
        self.global_step = 0
        self.best_eval_loss = float("inf")
        self.train_losses = []
        self.eval_losses = []

        # Checkpointing
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Resume
        if config.resume_from:
            self._load_checkpoint(config.resume_from)

        # Wandb
        self.wandb_run = None
        if config.use_wandb and self.is_main:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=config.wandb_project,
                    name=config.wandb_run_name,
                    config=vars(config),
                    resume="allow",
                )
            except ImportError:
                print("wandb not installed, skipping logging")

    def _save_checkpoint(self, tag: str = "latest"):
        if not self.is_main:
            return None
        path = os.path.join(self.config.checkpoint_dir, f"checkpoint_{tag}.pt")
        # Save the unwrapped model state for DDP compatibility
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        torch.save({
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "global_step": self.global_step,
            "best_eval_loss": self.best_eval_loss,
            "train_losses": self.train_losses[-1000:],  # Keep last 1000
            "eval_losses": self.eval_losses,
        }, path)
        return path

    def _load_checkpoint(self, path: str):
        if not os.path.exists(path):
            print(f"Checkpoint not found: {path}")
            return
        print(f"Resuming from {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        # Handle DDP-wrapped model: load into unwrapped model
        model_to_load = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_load.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.global_step = ckpt.get("global_step", 0)
        self.best_eval_loss = ckpt.get("best_eval_loss", float("inf"))
        self.train_losses = ckpt.get("train_losses", [])
        self.eval_losses = ckpt.get("eval_losses", [])
        print(f"Resumed at step {self.global_step}")

    @torch.no_grad()
    def evaluate(self) -> dict:
        """Run evaluation and return metrics."""
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        n_batches = 0

        for batch in self.eval_dl:
            if n_batches >= self.config.max_eval_batches:
                break
            inputs, labels = batch[0].to(self.device), batch[1].to(self.device)

            with autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                output = self.model(inputs)
                logits = output["logits"]
                loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

            mask = labels != -100
            n_tokens = mask.sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
            n_batches += 1

        avg_loss = total_loss / max(total_tokens, 1)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        self.model.train()
        return {"eval_loss": avg_loss, "eval_ppl": perplexity}

    def _set_sampler_epoch(self, epoch: int):
        """Set epoch on DistributedSampler if present."""
        sampler = getattr(self.train_dl, 'sampler', None)
        if sampler is not None and hasattr(sampler, 'set_epoch'):
            sampler.set_epoch(epoch)

    def train(self) -> dict:
        """Main training loop.

        Returns:
            Final metrics dict.
        """
        self.model.train()
        self._dl_epoch = 0
        self._set_sampler_epoch(self._dl_epoch)
        train_iter = iter(self.train_dl)
        accum_loss = 0.0
        step_start = time.time()

        pbar = tqdm(range(self.global_step, self.config.total_steps), desc="Training",
                    disable=not self.is_main)

        try:
            for step in pbar:
                if step < self.global_step:
                    continue  # Skip already-done steps after resume

                # Get batch (cycle dataloader, update sampler epoch for DDP)
                try:
                    batch = next(train_iter)
                except StopIteration:
                    self._dl_epoch += 1
                    self._set_sampler_epoch(self._dl_epoch)
                    train_iter = iter(self.train_dl)
                    batch = next(train_iter)

                inputs, labels = batch[0].to(self.device), batch[1].to(self.device)

                # Forward
                with autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                    output = self.model(inputs)
                    logits = output["logits"]
                    loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
                    loss = loss / self.config.grad_accum_steps

                # Backward
                self.scaler.scale(loss).backward()
                accum_loss += loss.item()

                # Step (after accumulation)
                if (step + 1) % self.config.grad_accum_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.scheduler.step()

                    # accum_loss already sums loss / grad_accum_steps, i.e. the mean micro-batch loss
                    self.train_losses.append(accum_loss)
                    accum_loss = 0.0

                self.global_step = step + 1

                # Logging (all ranks compute, only main prints)
                if self.global_step % self.config.log_every == 0 and self.is_main:
                    lr = self.scheduler.get_last_lr()[0]
                    elapsed = time.time() - step_start
                    tokens_per_sec = (self.config.log_every * inputs.shape[0] * inputs.shape[1] * self.world_size) / elapsed
                    recent_loss = sum(self.train_losses[-self.config.log_every:]) / min(len(self.train_losses), self.config.log_every)

                    pbar.set_postfix({
                        "loss": f"{recent_loss:.4f}",
                        "lr": f"{lr:.2e}",
                        "tok/s": f"{tokens_per_sec:.0f}",
                    })

                    if self.wandb_run:
                        import wandb
                        wandb.log({
                            "train/loss": recent_loss,
                            "train/lr": lr,
                            "train/tokens_per_sec": tokens_per_sec,
                            "train/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                        }, step=self.global_step)

                    step_start = time.time()

                # Evaluation (main rank only)
                if self.eval_dl and self.global_step % self.config.eval_every == 0 and self.is_main:
                    metrics = self.evaluate()
                    self.eval_losses.append((self.global_step, metrics["eval_loss"]))

                    print(f"\n  Step {self.global_step}: eval_loss={metrics['eval_loss']:.4f}, "
                          f"eval_ppl={metrics['eval_ppl']:.2f}")

                    if metrics["eval_loss"] < self.best_eval_loss:
                        self.best_eval_loss = metrics["eval_loss"]
                        self._save_checkpoint("best")

                    if self.wandb_run:
                        import wandb
                        wandb.log(metrics, step=self.global_step)

                # Checkpointing
                if self.global_step % self.config.checkpoint_every == 0:
                    self._save_checkpoint(f"step_{self.global_step}")
                    self._save_checkpoint("latest")

        except KeyboardInterrupt:
            print("\nTraining interrupted. Saving checkpoint...")
            self._save_checkpoint("interrupted")

        except Exception as e:
            print(f"\nError during training: {e}. Saving emergency checkpoint...")
            self._save_checkpoint("emergency")
            raise

        # Final save
        self._save_checkpoint("final")
        if self.wandb_run:
            self.wandb_run.finish()

        return {
            "final_train_loss": self.train_losses[-1] if self.train_losses else None,
            "best_eval_loss": self.best_eval_loss,
            "total_steps": self.global_step,
        }
