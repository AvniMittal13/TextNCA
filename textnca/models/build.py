"""Build Hierarchical TextNCA from command-line / saved architecture arguments.

Defaults reproduce the paper's flagship: staged [8, 32, 128] x T_s=4, local
softmax attention, GRU gate, learned step embeddings (30.8M parameters).
"""

import argparse

from textnca.models.config import ModelConfig
from textnca.models.hier_nca import HierarchicalTextNCA2, HierarchicalTextNCA2_Stochastic

ARCH_KEYS = [
    "d_model", "d_ff", "n_heads", "head_dim", "vocab_size", "max_seq_len",
    "window_schedule", "t_per_stage", "kernel_type", "gate_type",
    "step_embed_type", "stochastic_t_range",
]


def add_arch_args(parser: argparse.ArgumentParser):
    g = parser.add_argument_group("architecture")
    g.add_argument("--d_model", type=int, default=512)
    g.add_argument("--d_ff", type=int, default=2048)
    g.add_argument("--n_heads", type=int, default=8)
    g.add_argument("--head_dim", type=int, default=64)
    g.add_argument("--vocab_size", type=int, default=32000)
    g.add_argument("--max_seq_len", type=int, default=512)
    g.add_argument("--window_schedule", type=int, nargs="+", default=[8, 32, 128],
                   help="Attention window per stage, e.g. 8 32 128")
    g.add_argument("--t_per_stage", type=int, default=4,
                   help="NCA iterations per stage (T_s)")
    g.add_argument("--kernel_type", type=str, default="local_softmax",
                   choices=["local_softmax", "local_linear", "local_gla", "conv"])
    g.add_argument("--gate_type", type=str, default="gru",
                   choices=["gru", "sigmoid", "sigmoid_dropout", "highway", "residual", "overwrite"])
    g.add_argument("--step_embed_type", type=str, default="learned",
                   choices=["learned", "sinusoidal", "mlp", "none"])
    g.add_argument("--stochastic_t_range", type=int, nargs="+", default=None,
                   help="If set, sample T_s per batch from these values during training (e.g. 2 4 6)")


def arch_from_args(args: argparse.Namespace) -> dict:
    return {k: getattr(args, k) for k in ARCH_KEYS}


def build_model(arch: dict, use_gradient_checkpointing: bool = True):
    cfg = ModelConfig(
        d_model=arch["d_model"], d_ff=arch["d_ff"],
        n_heads=arch["n_heads"], head_dim=arch["head_dim"],
        window_size=32, num_steps=arch["t_per_stage"],
        vocab_size=arch["vocab_size"], max_seq_len=arch["max_seq_len"],
        kernel_type=arch["kernel_type"],
        gate_type=arch["gate_type"],
        step_embed_type=arch["step_embed_type"],
        pos_embed_type="learned",
        use_gradient_checkpointing=use_gradient_checkpointing,
        tie_embeddings=True,
    )
    if arch.get("stochastic_t_range"):
        return HierarchicalTextNCA2_Stochastic(
            cfg, window_schedule=arch["window_schedule"], t_per_stage=arch["t_per_stage"],
            share_across_stages=False, t_range=tuple(arch["stochastic_t_range"]),
        )
    return HierarchicalTextNCA2(
        cfg, window_schedule=arch["window_schedule"], t_per_stage=arch["t_per_stage"],
        share_across_stages=False,
    )


def default_run_name(arch: dict) -> str:
    windows = "_".join(str(w) for w in arch["window_schedule"])
    name = f"Hier2_{arch['kernel_type']}_{arch['gate_type']}_{arch['step_embed_type']}_{windows}_T{arch['t_per_stage']}"
    if arch.get("stochastic_t_range"):
        name += "_stochT" + "-".join(str(t) for t in arch["stochastic_t_range"])
    if arch["d_model"] != 512 or arch["d_ff"] != 2048:
        name += f"_d{arch['d_model']}_ff{arch['d_ff']}"
    return name
