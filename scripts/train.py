"""Train Hierarchical TextNCA on SlimPajama, evaluating on WikiText-103.

Usage (paper flagship, 30.8M params):
    python scripts/train.py --total_steps 60000 --batch_size 32 --grad_accum 2 --seed 42
"""
import argparse
import sys
import os
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from textnca.utils import set_seed, count_parameters, get_device_info
from textnca.models.build import add_arch_args, arch_from_args, build_model, default_run_name
from textnca.data.language_modeling import get_slimpajama_dataloader, get_eval_dataloader
from textnca.training.trainer import Trainer, TrainConfig


def main():
    parser = argparse.ArgumentParser(description="Train Hierarchical TextNCA")
    add_arch_args(parser)
    parser.add_argument("--data_dir", type=str, default="data/lm")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--data_tokens", type=str, default="500M",
                        help="Token count suffix for data file, e.g. 500M or 2000M")
    parser.add_argument("--total_steps", type=int, default=60000,
                        help="Number of micro-batches (optimizer steps = total_steps / grad_accum)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--warmup", type=int, default=2000)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--checkpoint_every", type=int, default=2000)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    arch = arch_from_args(args)
    model = build_model(arch)
    model_name = args.run_name or f"{default_run_name(arch)}_s{args.seed}"

    print("=" * 70, flush=True)
    print(f"Training: {model_name} (seed={args.seed})", flush=True)
    print("=" * 70, flush=True)
    params = count_parameters(model)
    print(f"Parameters: {params['total']:,} total, {params['trainable']:,} trainable", flush=True)
    print(f"Device: {device}", flush=True)
    if device == "cuda":
        info = get_device_info()
        print(f"GPU: {info.get('gpu_name', 'unknown')}, bf16={info.get('bf16_supported', False)}", flush=True)

    # Load data
    train_path = os.path.join(args.data_dir, f"slimpajama_{args.data_tokens}_seq512.npy")
    eval_path = os.path.join(args.data_dir, "wikitext103_test_seq512.npy")

    if not os.path.exists(train_path):
        print(f"ERROR: Training data not found at {train_path}", flush=True)
        print("Run scripts/prepare_data.py first!", flush=True)
        sys.exit(1)

    print(f"\nLoading training data: {train_path}", flush=True)
    train_dl = get_slimpajama_dataloader(train_path, seq_len=512, batch_size=args.batch_size, num_workers=4)
    print(f"Training batches: {len(train_dl):,}", flush=True)

    eval_dl = None
    if os.path.exists(eval_path):
        print(f"Loading eval data: {eval_path}", flush=True)
        eval_dl = get_eval_dataloader(eval_path, seq_len=512, batch_size=args.batch_size, num_workers=2)
        print(f"Eval batches: {len(eval_dl):,}", flush=True)
    else:
        print("WARNING: Eval data not found, skipping eval during training", flush=True)

    # Setup results dir
    run_dir = os.path.join(args.results_dir, model_name)
    os.makedirs(run_dir, exist_ok=True)

    # Save config (the "arch" block is what scripts/evaluate.py rebuilds the model from)
    config_info = {
        "model_name": model_name,
        "arch": arch,
        "total_params": params["total"],
        "trainable_params": params["trainable"],
        "total_steps": args.total_steps,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch_tokens": args.batch_size * args.grad_accum * 512,
        "lr": args.lr,
        "warmup": args.warmup,
        "seed": args.seed,
        "data_tokens": args.data_tokens,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config_info, f, indent=2)

    train_config = TrainConfig(
        lr=args.lr,
        weight_decay=0.1,
        betas=(0.9, 0.95),
        eps=1e-8,
        grad_clip=1.0,
        grad_accum_steps=args.grad_accum,
        warmup_steps=args.warmup,
        total_steps=args.total_steps,
        min_lr=args.lr * 0.1,
        use_amp=True,
        amp_dtype="bfloat16",
        checkpoint_dir=os.path.join(run_dir, "checkpoints"),
        checkpoint_every=args.checkpoint_every,
        resume_from=args.resume,
        log_every=100,
        eval_every=args.eval_every,
        use_wandb=args.wandb,
        wandb_project="textnca-lm",
        wandb_run_name=model_name,
        seed=args.seed,
        max_eval_batches=50,
    )

    trainer = Trainer(
        model=model,
        train_dl=train_dl,
        eval_dl=eval_dl,
        config=train_config,
        device=device,
    )

    print(f"\nStarting training for {args.total_steps} steps...", flush=True)
    print(f"Effective batch size: {args.batch_size * args.grad_accum} sequences "
          f"= {args.batch_size * args.grad_accum * 512:,} tokens/step", flush=True)
    print(f"Checkpoints: {train_config.checkpoint_dir}", flush=True)
    print(flush=True)

    result = trainer.train()

    result["model_name"] = model_name
    result["total_params"] = params["total"]
    with open(os.path.join(run_dir, "final_results.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'=' * 70}", flush=True)
    print(f"Training complete: {model_name}", flush=True)
    print(f"Final train loss: {result.get('final_train_loss', 'N/A')}", flush=True)
    print(f"Best eval loss: {result.get('best_eval_loss', 'N/A')}", flush=True)
    print(f"Results saved to: {run_dir}", flush=True)
    print(f"{'=' * 70}", flush=True)


if __name__ == "__main__":
    main()
