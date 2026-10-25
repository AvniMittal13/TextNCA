"""Core building blocks for TextNCA: attention, gating, FFN, embeddings."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ─── Local Windowed Attention ────────────────────────────────────────────────

class LocalWindowAttention(nn.Module):
    """Causal local windowed multi-head attention.

    Each token attends to at most `window_size` preceding tokens (including itself).
    Compatible with T4 GPUs (no FlashAttention required — uses masking).
    """

    def __init__(self, d_model: int, n_heads: int, head_dim: int,
                 window_size: int, dropout: float = 0.0, causal: bool = True):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.window_size = window_size
        self.causal = causal
        self.scale = head_dim ** -0.5

        self.W_q = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.W_k = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.W_v = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.W_o = nn.Linear(n_heads * head_dim, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)

    def _build_local_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Build a [seq_len, seq_len] bool mask: True = attend, False = masked."""
        row_idx = torch.arange(seq_len, device=device).unsqueeze(1)
        col_idx = torch.arange(seq_len, device=device).unsqueeze(0)
        mask = col_idx <= row_idx  # causal: can only attend to current and past
        mask = mask & (col_idx >= row_idx - self.window_size + 1)  # local: within window
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, seq_len, d_model] -> [batch, seq_len, d_model]"""
        B, L, _ = x.shape

        q = rearrange(self.W_q(x), "b l (h d) -> b h l d", h=self.n_heads)
        k = rearrange(self.W_k(x), "b l (h d) -> b h l d", h=self.n_heads)
        v = rearrange(self.W_v(x), "b l (h d) -> b h l d", h=self.n_heads)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, L, L]

        mask = self._build_local_causal_mask(L, x.device)  # [L, L]
        attn = attn.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = attn @ v  # [B, H, L, D]
        out = rearrange(out, "b h l d -> b l (h d)")
        return self.W_o(out)


class LocalLinearAttention(nn.Module):
    """Local linear attention with Taylor-2 feature map (φ(x) = [1, x, x⊗x/√2]).

    Approximates softmax attention in a local window using a polynomial kernel.
    """

    def __init__(self, d_model: int, n_heads: int, head_dim: int,
                 window_size: int, dropout: float = 0.0, causal: bool = True):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.window_size = window_size
        self.causal = causal

        self.W_q = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.W_k = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.W_v = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.W_o = nn.Linear(n_heads * head_dim, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)

    def _taylor2_features(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Taylor-2 feature map: φ(x) = concat([1], x, x⊗x/√2)."""
        # x: [B, H, L, D]
        ones = torch.ones(*x.shape[:-1], 1, device=x.device, dtype=x.dtype)
        quad = (x.unsqueeze(-1) * x.unsqueeze(-2)) / math.sqrt(2.0)
        quad = quad.flatten(-2)  # [B, H, L, D*D]
        return torch.cat([ones, x, quad], dim=-1)

    def _build_local_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        row_idx = torch.arange(seq_len, device=device).unsqueeze(1)
        col_idx = torch.arange(seq_len, device=device).unsqueeze(0)
        mask = col_idx <= row_idx
        mask = mask & (col_idx >= row_idx - self.window_size + 1)
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape

        q = rearrange(self.W_q(x), "b l (h d) -> b h l d", h=self.n_heads)
        k = rearrange(self.W_k(x), "b l (h d) -> b h l d", h=self.n_heads)
        v = rearrange(self.W_v(x), "b l (h d) -> b h l d", h=self.n_heads)

        # Normalize before feature map for stability
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
        k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)

        # Apply Taylor-2 feature map
        q_feat = self._taylor2_features(q)
        k_feat = self._taylor2_features(k)

        # Still use explicit attention matrix for local window (small window_size)
        attn = q_feat @ k_feat.transpose(-2, -1)  # [B, H, L, L]
        mask = self._build_local_causal_mask(L, x.device)
        attn = attn.masked_fill(~mask.unsqueeze(0).unsqueeze(0), 0.0)

        # Normalize
        attn_sum = attn.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        attn = attn / attn_sum
        attn = self.attn_dropout(attn)

        out = attn @ v
        out = rearrange(out, "b h l d -> b l (h d)")
        return self.W_o(out)


class ConvPerception(nn.Module):
    """1D causal convolution as perception kernel (for TextNCA-Conv variant)."""

    def __init__(self, d_model: int, kernel_size: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size
        # Causal: pad on the left only
        self.conv = nn.Conv1d(d_model, d_model, kernel_size, padding=0, groups=1)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, D] -> [B, L, D]"""
        # Causal padding
        x_t = x.transpose(1, 2)  # [B, D, L]
        x_t = F.pad(x_t, (self.kernel_size - 1, 0))
        out = self.conv(x_t).transpose(1, 2)  # [B, L, D]
        return self.proj(self.dropout(out))


class LocalGatedLinearAttention(nn.Module):
    """Local Gated Linear Attention with data-dependent decay.

    GLA (Yang et al., ICML 2024) uses a data-dependent gating factor per head
    to control how much each key-value pair decays over distance. This provides
    a middle ground between softmax attention (sharp, position-invariant) and
    linear attention (too smooth).

    Within a local window, computes:
        alpha_t = sigmoid(W_alpha * x_t)     # per-position gate [B, H, L, 1]
        attn_{i,j} = (q_i^T k_j) * prod_{m=j+1}^{i} alpha_m
    Approximated as:
        decay_{i,j} = exp( sum_{m=j+1}^{i} log(alpha_m) )
        attn = (Q K^T) * decay_mask
    """

    def __init__(self, d_model: int, n_heads: int, head_dim: int,
                 window_size: int, dropout: float = 0.0, causal: bool = True):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.window_size = window_size
        self.scale = head_dim ** -0.5

        self.W_q = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.W_k = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.W_v = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.W_o = nn.Linear(n_heads * head_dim, d_model, bias=False)
        # Data-dependent gate: one scalar per head per position
        self.W_alpha = nn.Linear(d_model, n_heads, bias=True)
        self.attn_dropout = nn.Dropout(dropout)

    def _build_local_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        row_idx = torch.arange(seq_len, device=device).unsqueeze(1)
        col_idx = torch.arange(seq_len, device=device).unsqueeze(0)
        mask = col_idx <= row_idx
        mask = mask & (col_idx >= row_idx - self.window_size + 1)
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape

        q = rearrange(self.W_q(x), "b l (h d) -> b h l d", h=self.n_heads)
        k = rearrange(self.W_k(x), "b l (h d) -> b h l d", h=self.n_heads)
        v = rearrange(self.W_v(x), "b l (h d) -> b h l d", h=self.n_heads)

        # Data-dependent decay gate in (0, 1) per head per position
        # Use log-sigmoid for numerical stability; clamp to avoid -inf
        log_alpha = F.logsigmoid(self.W_alpha(x))  # [B, L, H], values in (-inf, 0)
        log_alpha = log_alpha.clamp(min=-5.0)  # prevent extreme decay
        log_alpha = rearrange(log_alpha, "b l h -> b h l")  # [B, H, L]

        # Compute in float32 for numerical stability
        log_alpha_f32 = log_alpha.float()
        cumlog = torch.cumsum(log_alpha_f32, dim=-1)  # [B, H, L]

        # decay_matrix[i,j] = exp(cumlog[i] - cumlog[j]) for i >= j
        # This equals prod(alpha[j+1..i])
        decay_matrix = cumlog.unsqueeze(-1) - cumlog.unsqueeze(-2)  # [B, H, L, L]

        # Clamp before exp to prevent overflow/underflow
        decay_matrix = decay_matrix.clamp(min=-20.0, max=0.0)
        decay_matrix = torch.exp(decay_matrix)  # [B, H, L, L]
        decay_matrix = decay_matrix.to(q.dtype)  # back to original precision

        # Raw attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, L, L]

        # Apply decay
        attn = attn * decay_matrix

        # Apply local causal mask and softmax (not L1 norm — use proper softmax)
        mask = self._build_local_causal_mask(L, x.device)  # [L, L]
        attn = attn.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = F.softmax(attn, dim=-1)
        # Replace any NaN from all-masked rows with 0
        attn = attn.nan_to_num(0.0)
        attn = self.attn_dropout(attn)

        out = attn @ v
        out = rearrange(out, "b h l d -> b l (h d)")
        return self.W_o(out)


def get_perception_module(kernel_type: str, d_model: int, n_heads: int, head_dim: int,
                          window_size: int, dropout: float = 0.0) -> nn.Module:
    """Factory function to instantiate the appropriate perception kernel."""
    if kernel_type == "local_softmax":
        return LocalWindowAttention(d_model, n_heads, head_dim, window_size, dropout)
    elif kernel_type == "local_linear":
        return LocalLinearAttention(d_model, n_heads, head_dim, window_size, dropout)
    elif kernel_type == "local_gla":
        return LocalGatedLinearAttention(d_model, n_heads, head_dim, window_size, dropout)
    elif kernel_type == "conv":
        return ConvPerception(d_model, kernel_size=window_size, dropout=dropout)
    else:
        raise ValueError(f"Unknown kernel_type: {kernel_type}")


# ─── Gating Mechanisms ──────────────────────────────────────────────────────

class GRUGate(nn.Module):
    """GRU-style gated residual update for iterative refinement."""

    def __init__(self, d_model: int):
        super().__init__()
        self.W_z = nn.Linear(2 * d_model, d_model)
        self.W_r = nn.Linear(2 * d_model, d_model)
        self.W_h = nn.Linear(2 * d_model, d_model)

    def forward(self, h_prev: torch.Tensor, h_update: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([h_prev, h_update], dim=-1)
        z = torch.sigmoid(self.W_z(combined))
        r = torch.sigmoid(self.W_r(combined))
        h_tilde = torch.tanh(self.W_h(torch.cat([r * h_prev, h_update], dim=-1)))
        return (1 - z) * h_prev + z * h_tilde


class SigmoidGate(nn.Module):
    """Simple sigmoid gating: h = σ(g) ⊙ update + (1-σ(g)) ⊙ h_prev."""

    def __init__(self, d_model: int):
        super().__init__()
        self.gate = nn.Linear(2 * d_model, d_model)

    def forward(self, h_prev: torch.Tensor, h_update: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.gate(torch.cat([h_prev, h_update], dim=-1)))
        return g * h_update + (1 - g) * h_prev


class HighwayGate(nn.Module):
    """Highway gate: h = g ⊙ update + (1-g) ⊙ h_prev, g depends only on h_prev."""

    def __init__(self, d_model: int):
        super().__init__()
        self.gate = nn.Linear(d_model, d_model)

    def forward(self, h_prev: torch.Tensor, h_update: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.gate(h_prev))
        return g * h_update + (1 - g) * h_prev


class ResidualGate(nn.Module):
    """Simple additive residual: h = h_prev + update."""

    def __init__(self, d_model: int):
        super().__init__()

    def forward(self, h_prev: torch.Tensor, h_update: torch.Tensor) -> torch.Tensor:
        return h_prev + h_update


class OverwriteGate(nn.Module):
    """Pure overwrite (classic NCA, no residual): h = update."""

    def __init__(self, d_model: int):
        super().__init__()

    def forward(self, h_prev: torch.Tensor, h_update: torch.Tensor) -> torch.Tensor:
        return h_update


class SigmoidDropoutGate(nn.Module):
    """Sigmoid gate with dropout on h_prev — simulates stochastic reset.

    The GRU gate's reset gate r selectively suppresses dimensions of h_prev
    before computing the candidate. Dropout on h_prev achieves a similar effect
    stochastically: during training, random dimensions of h_prev are zeroed,
    forcing the model to not over-rely on any single feature from previous state.

    h_dropped = Dropout(h_prev)
    g = σ(W[h_dropped; h_update])
    h_new = g ⊙ h_update + (1-g) ⊙ h_prev  (note: full h_prev for residual)

    This is cheaper than GRU (1 linear vs 3) while providing regularization
    similar to the reset gate's selective forgetting.
    """

    def __init__(self, d_model: int, dropout_rate: float = 0.1):
        super().__init__()
        self.gate = nn.Linear(2 * d_model, d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, h_prev: torch.Tensor, h_update: torch.Tensor) -> torch.Tensor:
        # Dropout on h_prev before gate computation (stochastic reset)
        h_dropped = self.dropout(h_prev)
        g = torch.sigmoid(self.gate(torch.cat([h_dropped, h_update], dim=-1)))
        # Use full h_prev for the residual blend (not dropped)
        return g * h_update + (1 - g) * h_prev


def get_gate(gate_type: str, d_model: int) -> nn.Module:
    """Factory function to instantiate the appropriate gating mechanism."""
    gates = {
        "gru": GRUGate,
        "sigmoid": SigmoidGate,
        "sigmoid_dropout": SigmoidDropoutGate,
        "highway": HighwayGate,
        "residual": ResidualGate,
        "overwrite": OverwriteGate,
    }
    if gate_type not in gates:
        raise ValueError(f"Unknown gate_type: {gate_type}. Choose from {list(gates.keys())}")
    return gates[gate_type](d_model)


# ─── Feed-Forward Network ───────────────────────────────────────────────────

class FFN(nn.Module):
    """2-layer MLP with GELU activation: d_model → d_ff → d_model."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_ff)
        self.W2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W2(self.dropout(F.gelu(self.W1(x))))


# ─── Step Embeddings ────────────────────────────────────────────────────────

class LearnedStepEmbedding(nn.Module):
    """Learnable per-step embedding vector added to hidden states."""

    def __init__(self, max_steps: int, d_model: int):
        super().__init__()
        self.embeddings = nn.Embedding(max_steps, d_model)

    def forward(self, t: int) -> torch.Tensor:
        return self.embeddings(torch.tensor(t, dtype=torch.long, device=self.embeddings.weight.device))


class SinusoidalStepEmbedding(nn.Module):
    """Sinusoidal step embedding (like positional encoding but over step dimension)."""

    def __init__(self, max_steps: int, d_model: int):
        super().__init__()
        pe = torch.zeros(max_steps, d_model)
        position = torch.arange(0, max_steps, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, t: int) -> torch.Tensor:
        return self.pe[t]


class MLPStepEmbedding(nn.Module):
    """MLP-based step embedding (like diffusion timestep encoding)."""

    def __init__(self, max_steps: int, d_model: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.max_steps = max_steps

    def forward(self, t: int) -> torch.Tensor:
        t_norm = torch.tensor([t / self.max_steps], dtype=torch.float,
                              device=self.mlp[0].weight.device)
        return self.mlp(t_norm).squeeze(0)


class NoStepEmbedding(nn.Module):
    """No step embedding — returns zero."""

    def __init__(self, max_steps: int, d_model: int):
        super().__init__()
        self.d_model = d_model
        # Register a dummy buffer so .device works
        self.register_buffer("_zero", torch.zeros(d_model))

    def forward(self, t: int) -> torch.Tensor:
        return self._zero


def get_step_embedding(step_type: str, max_steps: int, d_model: int) -> nn.Module:
    """Factory for step embedding modules."""
    types = {
        "learned": LearnedStepEmbedding,
        "sinusoidal": SinusoidalStepEmbedding,
        "mlp": MLPStepEmbedding,
        "none": NoStepEmbedding,
    }
    if step_type not in types:
        raise ValueError(f"Unknown step_embed_type: {step_type}")
    return types[step_type](max_steps, d_model)
