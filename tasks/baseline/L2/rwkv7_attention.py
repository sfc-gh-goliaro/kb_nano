"""RWKV7 attention layer.

Implements the full RWKV7 attention block:
  token_shift -> addcmul -> projections + LoRA gates -> L2-norm + k_update
  -> DPLR recurrence -> GroupNorm -> gate output correction -> o_proj

Weight names match the FLA checkpoint format exactly.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..L1.rwkv7_recurrence import naive_recurrent_rwkv7


class LoRA(nn.Module):
    """Low-rank adapter: Linear -> activation -> Linear."""

    def __init__(self, input_dim: int, output_dim: int, low_rank_dim: int,
                 activation: str | None = 'tanh', bias: bool = True):
        super().__init__()
        if activation is None:
            act = nn.Identity()
        elif activation == 'tanh':
            act = nn.Tanh()
        elif activation == 'sigmoid':
            act = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        self.lora = nn.Sequential(
            nn.Linear(input_dim, low_rank_dim, bias=False),
            act,
            nn.Linear(low_rank_dim, output_dim, bias=bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora(x)


class RWKV7Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        head_dim: int = 64,
        num_heads: int | None = None,
        decay_low_rank_dim: int = 96,
        gate_low_rank_dim: int = 320,
        a_low_rank_dim: int = 96,
        v_low_rank_dim: int = 64,
        norm_eps: float = 1e-5,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        if num_heads is not None:
            self.num_heads = num_heads
        else:
            self.num_heads = hidden_size // head_dim
        self.key_dim = hidden_size
        self.value_dim = hidden_size
        self.head_v_dim = self.value_dim // self.num_heads
        self.layer_idx = layer_idx

        # Token-shift mixing parameters
        self.x_r = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_w = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_k = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_v = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_a = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_g = nn.Parameter(torch.zeros(1, 1, hidden_size))

        # Per-head/dim parameters
        self.k_k = nn.Parameter(torch.zeros(self.key_dim))
        self.k_a = nn.Parameter(torch.zeros(self.key_dim))
        self.r_k = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))

        # Projections
        self.r_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        # LoRA modules
        self.w_lora = LoRA(hidden_size, self.key_dim, decay_low_rank_dim, activation='tanh', bias=True)
        if layer_idx != 0:
            self.v_lora = LoRA(hidden_size, self.value_dim, v_low_rank_dim, activation=None, bias=True)
        self.a_lora = LoRA(hidden_size, self.key_dim, a_low_rank_dim, activation=None, bias=True)
        self.g_lora = LoRA(hidden_size, self.value_dim, gate_low_rank_dim, activation='sigmoid', bias=False)

        # GroupNorm (num_groups=num_heads, eps=head_dim*norm_eps)
        self.g_norm = nn.GroupNorm(
            num_groups=self.num_heads,
            num_channels=self.value_dim,
            eps=self.head_dim * norm_eps,
            affine=True,
        )

    def forward(
        self, hidden_states: torch.Tensor, v_first: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = hidden_states.shape

        # Token shift: delta = prev_token - current_token
        shifted = torch.zeros_like(hidden_states)
        shifted[:, 1:] = hidden_states[:, :-1]
        delta = shifted - hidden_states

        # Fused addcmul: xi = hidden_states + delta * x_i
        xr = torch.addcmul(hidden_states, delta, self.x_r)
        xw = torch.addcmul(hidden_states, delta, self.x_w)
        xk = torch.addcmul(hidden_states, delta, self.x_k)
        xv = torch.addcmul(hidden_states, delta, self.x_v)
        xa = torch.addcmul(hidden_states, delta, self.x_a)
        xg = torch.addcmul(hidden_states, delta, self.x_g)

        # Projections
        r = self.r_proj(xr)
        w = -0.6065306597126334 * self.w_lora(xw).sigmoid()
        k = self.k_proj(xk)
        v = self.v_proj(xv)

        # v_first cross-layer mixing
        if self.layer_idx == 0:
            v_first = v
        else:
            v = torch.lerp(v, v_first, self.v_lora(xv).sigmoid())

        a = self.a_lora(xa).sigmoid()
        g = self.g_lora(xg)

        # L2-normalized key (per-head)
        kk = F.normalize(
            (k * self.k_k).view(B, T, self.num_heads, self.head_dim),
            dim=-1, p=2.0,
        )

        # Key update: k = k * (1 + (a - 1) * k_a)
        k = k.addcmul(k * (a - 1), self.k_a)

        # Reshape to multi-head [B, T, H, D]
        r_mh = r.view(B, T, self.num_heads, self.head_dim)
        w_mh = w.view(B, T, self.num_heads, self.head_dim)
        k_mh = k.view(B, T, self.num_heads, self.head_dim)
        a_mh = a.view(B, T, self.num_heads, self.head_dim)
        v_mh = v.view(B, T, self.num_heads, self.head_v_dim)

        # Recurrence (transpose to [B, H, T, D] for our loop)
        o, _ = naive_recurrent_rwkv7(
            r_mh.transpose(1, 2), w_mh.transpose(1, 2),
            k_mh.transpose(1, 2), v_mh.transpose(1, 2),
            kk.transpose(1, 2), a_mh.transpose(1, 2),
            scale=1.0,
        )
        o = o.transpose(1, 2)  # [B, T, H, V]

        # GroupNorm
        o = self.g_norm(o.reshape(B * T, -1)).view(B, T, -1)

        # Gate output correction: (o + correction) * g
        correction = (
            (r_mh * k_mh * self.r_k[None, None]).sum(-1, keepdim=True) * v_mh
        ).reshape(B, T, -1)
        o = (o + correction) * g

        return self.o_proj(o), v_first
