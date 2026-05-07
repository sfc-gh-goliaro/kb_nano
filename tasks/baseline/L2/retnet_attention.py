"""RetNet Multi-Scale Retention attention layer.

Implements the full retention block:
  q/k/v projection -> rotary -> fixed-decay recurrence -> per-head norm + swish gate -> o projection

Weight names match the FLA checkpoint format exactly.
No gk_proj (unlike GLA) — decay is fixed per head: γ_h = 1 - 2^(-5-h).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..L1.retnet_recurrence import naive_recurrent_retention
from ..L1.rms_norm import RMSNorm


def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embedding (non-interleaved). x: [..., D], cos/sin: broadcastable to [..., D/2]."""
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class MultiScaleRetention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        expand_k: float = 1.0,
        expand_v: float = 2.0,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        # Per-head RMSNorm for output gating
        self.g_norm_swish_gate = RMSNorm(self.head_v_dim, eps=norm_eps)

        # Precompute rotary inverse frequencies (not saved in checkpoint)
        inv_freq = 1.0 / (
            10000.0 ** (torch.arange(0, self.head_k_dim, 2, dtype=torch.float32) / self.head_k_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _get_cos_sin(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))  # [T, D/2]
        return freqs.cos(), freqs.sin()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, T, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        g = self.g_proj(hidden_states)

        # Reshape to [B, T, H, D]
        q = q.view(B, T, self.num_heads, self.head_k_dim)
        k = k.view(B, T, self.num_heads, self.head_k_dim)

        # Apply rotary embeddings
        cos, sin = self._get_cos_sin(T, q.device)
        # cos, sin: [T, D/2] -> [1, T, 1, D/2] for broadcasting
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)
        q = _apply_rotary(q, cos, sin)
        k = _apply_rotary(k, cos, sin)

        # Transpose for recurrence: [B, T, H, D] -> [B, H, T, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_v_dim).transpose(1, 2)

        # Recurrence
        o, _ = naive_recurrent_retention(q, k, v)
        # o: [B, H, T, V] -> [B, T, H, V]
        o = o.transpose(1, 2)

        # Per-head RMSNorm + swish gate
        o = self.g_norm_swish_gate(o.reshape(-1, self.head_v_dim))
        o = o.view(B, T, self.value_dim)
        o = o * F.silu(g)

        return self.o_proj(o)
