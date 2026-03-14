"""BitNet attention: GQA with RoPE, activation quantization, and sub-norm.

Architecture (per the Microsoft BitNet-b1.58-2B-4T reference):
    x → BitLinear(q,k,v) → RoPE → SDPA → RMSNorm(sub) → BitLinear(o)

Uses consecutive-pairs (interleaved) RoPE convention to match the GPU
checkpoint weight format: within each head, dimensions are ordered as
[real_0, imag_0, real_1, imag_1, ...].

Weight names match HuggingFace checkpoint convention:
    self_attn.q_proj.weight, self_attn.k_proj.weight, ...
    self_attn.attn_sub_norm.weight
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..L1.bitnet_linear import BitLinear


def _apply_rope_interleaved(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
) -> torch.Tensor:
    """Apply RoPE with consecutive-pairs (interleaved) convention.

    Args:
        x:   [..., D] where D is even (pairs: (d0,d1), (d2,d3), ...)
        cos: [..., D//2] cos for each pair
        sin: [..., D//2] sin for each pair
    """
    x_pairs = x.view(*x.shape[:-1], -1, 2)  # [..., D//2, 2]
    x1, x2 = x_pairs.unbind(-1)             # each [..., D//2]
    o1 = x1 * cos - x2 * sin
    o2 = x1 * sin + x2 * cos
    return torch.stack([o1, o2], dim=-1).flatten(-2)


class BitNetAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_position_embeddings: int,
        rope_theta: float,
        rms_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads

        self.q_proj = BitLinear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = BitLinear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = BitLinear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = BitLinear(hidden_size, hidden_size, bias=False)

        self.attn_sub_norm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)

        # Precompute RoPE cos/sin cache (one value per pair)
        inv_freq = 1.0 / (
            rope_theta
            ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim)
        )
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.outer(t, inv_freq)  # [T, D//2]
        self.register_buffer("_rope_cos", freqs.cos(), persistent=False)
        self.register_buffer("_rope_sin", freqs.sin(), persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, T, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(B, T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(B, T, self.num_kv_heads, self.head_dim)

        # Transpose to [B, H, T, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply RoPE (consecutive-pairs / interleaved convention)
        cos = self._rope_cos[:T].to(q.dtype)  # [T, D//2]
        sin = self._rope_sin[:T].to(q.dtype)
        # Broadcast to [1, 1, T, D//2]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        q = _apply_rope_interleaved(q, cos, sin)
        k = _apply_rope_interleaved(k, cos, sin)

        # GQA: expand KV heads
        if self.num_kv_groups > 1:
            k = k.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1)
            k = k.reshape(B, self.num_heads, T, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1)
            v = v.reshape(B, self.num_heads, T, self.head_dim)

        # Scaled dot-product attention (causal)
        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # [B, H, T, D] → [B, T, hidden_size]
        attn_output = attn_output.transpose(1, 2).reshape(B, T, -1)

        # Sub-norm then output projection
        attn_output = self.attn_sub_norm(attn_output)
        return self.o_proj(attn_output)
