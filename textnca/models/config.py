"""Model configuration for Hierarchical TextNCA."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for TextNCA models."""

    # Architecture
    d_model: int = 128
    d_ff: int = 512
    n_heads: int = 4
    head_dim: int = 32
    window_size: int = 16
    num_steps: int = 16  # T: number of iterative steps
    vocab_size: int = 8192

    # Gating
    gate_type: str = "gru"  # "gru", "sigmoid", "highway", "residual", "overwrite"

    # Perception kernel
    kernel_type: str = "local_softmax"  # "local_softmax", "local_linear", "local_gla", "conv"

    # Step embedding
    step_embed_type: str = "learned"  # "learned", "sinusoidal", "mlp", "none"

    # Position encoding
    pos_embed_type: str = "learned"  # "learned", "rope", "rope_local", "alibi"
    max_seq_len: int = 512

    # Training
    dropout: float = 0.0
    use_gradient_checkpointing: bool = False

    # LM head
    tie_embeddings: bool = True
