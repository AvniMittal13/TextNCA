"""Hierarchical (staged) TextNCA: the final model reported in the paper."""

import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from typing import List, Optional

from textnca.models.config import ModelConfig
from textnca.models.components import (
    get_perception_module, get_gate, FFN, get_step_embedding,
)


class HierarchicalTextNCA2(nn.Module):
    """Staged Hierarchical NCA: run T iterations at each scale before moving to the next.

    Runs a full NCA (T_per_stage iterations) at w=8 first,
    then passes the output to a full NCA at w=32, then w=128.

    Architecture:
        h = TokenEmbed(x) + PosEmbed(x)

        # Stage 1: local syntax (w=8)
        for t in range(T_per_stage):
            h = Gate_1(h, FFN_1(LN_1(Attn_w8(h + step_emb_1(t)))))

        # Stage 2: phrase structure (w=32) — initialized from stage 1 output
        for t in range(T_per_stage):
            h = Gate_2(h, FFN_2(LN_2(Attn_w32(h + step_emb_2(t)))))

        # Stage 3: discourse (w=128) — initialized from stage 2 output
        for t in range(T_per_stage):
            h = Gate_3(h, FFN_3(LN_3(Attn_w128(h + step_emb_3(t)))))

        logits = LMHead(LN(h))

    Each scale gets T full NCA iterations to converge
    before the next scale begins. The hidden state flows between stages
    without reset — stage 2 starts from the refined output of stage 1.

    Total NCA steps = K * T_per_stage (e.g., 3 * 4 = 12)
    """

    def __init__(self, config: ModelConfig, window_schedule: Optional[List[int]] = None,
                 t_per_stage: int = 4, share_across_stages: bool = False):
        """
        Args:
            config: Model config
            window_schedule: Window sizes per stage, e.g. [8, 32, 128]
            t_per_stage: Number of NCA iterations per stage
            share_across_stages: If True, all stages share perception+FFN weights
                                 (true NCA spirit). If False, each stage has unique weights.
        """
        super().__init__()
        self.config = config
        self.share_across_stages = share_across_stages

        if window_schedule is None:
            window_schedule = [8, 32, 128]
        self.window_schedule = window_schedule
        self.t_per_stage = t_per_stage
        self.num_stages = len(window_schedule)

        # Embeddings
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.d_model)

        # Per-stage NCA blocks
        if share_across_stages:
            # One shared perception+FFN, but different window sizes need different attention modules
            # So we share FFN/LN but have separate attention per scale
            shared_ffn = FFN(config.d_model, config.d_ff, config.dropout)
            shared_norm = nn.LayerNorm(config.d_model)
            shared_gate = get_gate(config.gate_type, config.d_model)

            self.stages = nn.ModuleList()
            for w in window_schedule:
                self.stages.append(nn.ModuleDict({
                    "perception": get_perception_module(
                        config.kernel_type, config.d_model, config.n_heads,
                        config.head_dim, w, config.dropout,
                    ),
                    "norm": shared_norm,
                    "ffn": shared_ffn,
                    "gate": shared_gate,
                }))
        else:
            # Unique weights per stage
            self.stages = nn.ModuleList()
            for w in window_schedule:
                self.stages.append(nn.ModuleDict({
                    "perception": get_perception_module(
                        config.kernel_type, config.d_model, config.n_heads,
                        config.head_dim, w, config.dropout,
                    ),
                    "norm": nn.LayerNorm(config.d_model),
                    "ffn": FFN(config.d_model, config.d_ff, config.dropout),
                    "gate": get_gate(config.gate_type, config.d_model),
                }))

        # Per-stage, per-step embeddings
        total_steps = self.num_stages * t_per_stage * 4  # extra headroom
        self.step_embed = get_step_embedding(
            config.step_embed_type, total_steps, config.d_model)

        # Output
        self.ln_final = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embed.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _one_step(self, h: torch.Tensor, stage_idx: int,
                  step_emb: torch.Tensor) -> torch.Tensor:
        """One NCA step within a single stage."""
        stage = self.stages[stage_idx]
        x = h + step_emb.unsqueeze(0).unsqueeze(0)
        x = stage["perception"](x)
        x = stage["norm"](x)
        x = stage["ffn"](x)
        return stage["gate"](h, x)

    def forward(self, input_ids, num_steps=None, stochastic_rate=0.0,
                return_all_steps=False):
        t_per_stage = num_steps if num_steps is not None else self.t_per_stage
        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0)
        h = self.token_embed(input_ids) + self.pos_embed(positions)

        all_hidden = [h] if return_all_steps else None
        global_step = 0

        # Iterate through each stage (scale level)
        for stage_idx in range(self.num_stages):
            # Run T iterations at this scale
            for t in range(t_per_stage):
                step_emb = self.step_embed(global_step).to(h.device)

                if self.training and self.config.use_gradient_checkpointing:
                    h_new = cp.checkpoint(
                        self._one_step, h, stage_idx, step_emb,
                        use_reentrant=False)
                else:
                    h_new = self._one_step(h, stage_idx, step_emb)

                if stochastic_rate > 0.0 and self.training:
                    mask = torch.rand(B, L, 1, device=h.device) > stochastic_rate
                    h = torch.where(mask, h_new, h)
                else:
                    h = h_new

                global_step += 1
                if return_all_steps:
                    all_hidden.append(h)

        h = self.ln_final(h)
        logits = self.lm_head(h)

        result = {"logits": logits}
        if return_all_steps:
            result["all_hidden"] = all_hidden
        return result


class HierarchicalTextNCA2_Stochastic(HierarchicalTextNCA2):
    """HierarchicalTextNCA2 with stochastic per-batch T_s sampling during training.

    During training, samples t_per_stage uniformly from `t_range` per batch.
    During eval, uses the configured t_per_stage. This explicitly trains the
    model to be robust to varying iteration counts, so the same checkpoint can
    be run with different T_s at inference.
    """

    def __init__(self, config, window_schedule=None, t_per_stage=4,
                 share_across_stages=False, t_range=(2, 4, 6)):
        super().__init__(config, window_schedule=window_schedule,
                         t_per_stage=t_per_stage,
                         share_across_stages=share_across_stages)
        self.t_range = tuple(t_range)
        # Pre-allocate enough step embedding capacity for the largest T_s
        max_total = self.num_stages * max(self.t_range) * 4
        if hasattr(self.step_embed, 'embeddings'):
            current = self.step_embed.embeddings.num_embeddings
            if current < max_total:
                # Re-initialise with a larger table
                from textnca.models.components import get_step_embedding
                self.step_embed = get_step_embedding(
                    config.step_embed_type, max_total, config.d_model)

    def forward(self, input_ids, num_steps=None, stochastic_rate=0.0,
                return_all_steps=False):
        if self.training and num_steps is None:
            # Sample t_per_stage per batch from t_range
            import random
            t_eff = random.choice(self.t_range)
        else:
            t_eff = num_steps if num_steps is not None else self.t_per_stage
        return super().forward(input_ids, num_steps=t_eff,
                               stochastic_rate=stochastic_rate,
                               return_all_steps=return_all_steps)
