"""GLA (Gated Linear Attention) attention layer.

Implements the full GLA attention block:
  q/k/v projection -> gk gate -> recurrence -> per-head norm + swish gate -> o projection

Weight names match the FLA checkpoint format exactly.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..L1.gla_recurrence import naive_recurrent_gla
from ..L1.rms_norm import RMSNorm


class GatedLinearAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        expand_k: float = 0.5,
        expand_v: float = 1.0,
        gate_low_rank_dim: int = 16,
        gate_logit_normalizer: int = 16,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        self.gate_logit_normalizer = gate_logit_normalizer

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        # 2-layer gate projection: hidden_size -> low_rank -> key_dim
        self.gk_proj = nn.Sequential(
            nn.Linear(hidden_size, gate_low_rank_dim, bias=False),
            nn.Linear(gate_low_rank_dim, self.key_dim, bias=True),
        )

        # Per-head RMSNorm for output gating
        self.g_norm_swish_gate = RMSNorm(self.head_v_dim, eps=norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, T, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        gk = self.gk_proj(hidden_states)
        g = self.g_proj(hidden_states)

        # Reshape to multi-head: [B, T, H, D] -> [B, H, T, D]
        q = q.view(B, T, self.num_heads, self.head_k_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_k_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_v_dim).transpose(1, 2)
        gk = gk.view(B, T, self.num_heads, self.head_k_dim).transpose(1, 2)

        # Gate: logsigmoid normalized
        gk = F.logsigmoid(gk) / self.gate_logit_normalizer

        # Recurrence
        o, _ = naive_recurrent_gla(q, k, v, gk)
        # o: [B, H, T, V] -> [B, T, H, V]
        o = o.transpose(1, 2)

        # Per-head RMSNorm + swish gate
        o = self.g_norm_swish_gate(o.reshape(-1, self.head_v_dim))
        o = o.view(B, T, self.value_dim)
        o = o * F.silu(g)

        return self.o_proj(o)
