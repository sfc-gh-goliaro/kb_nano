"""YARN-scaled rotary position embeddings (RoPE) for DeepSeek-style models.

Uses sgl_kernel.apply_rope_with_cos_sin_cache_inplace with ``is_neox=False`` for
interleaved layout, matching vLLM's ``DeepseekScalingRotaryEmbedding``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from sgl_kernel import apply_rope_with_cos_sin_cache_inplace as _sgl_rope


def yarn_find_correction_dim(num_rotations: float, dim: int, base: float, max_position_embeddings: int) -> float:
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def yarn_find_correction_range(
    low_rot: int, high_rot: int, dim: int, base: float, max_position_embeddings: int
) -> tuple[int, int]:
    low = math.floor(yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)


def yarn_linear_ramp_mask(low: float, high: float, dim: int, dtype=torch.float) -> torch.Tensor:
    if low == high:
        high += 0.001
    linear_func = (torch.arange(dim, dtype=dtype) - low) / (high - low)
    return torch.clamp(linear_func, 0, 1)


def yarn_get_mscale(scale: float, mscale: float) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class YarnRotaryEmbedding(nn.Module):
    """YARN (Yet Another RoPE extensioN) scaled RoPE for extended context."""

    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int,
        rope_theta: float,
        scaling_factor: float,
        extrapolation_factor: float = 1,
        attn_factor: float = 1,
        beta_fast: int = 32,
        beta_slow: int = 1,
        mscale: float = 1,
        mscale_all_dim: float = 0,
        is_neox_style: bool = False,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.is_neox_style = is_neox_style
        rotary_dim = head_dim
        base = rope_theta

        softmax_mscale = (
            yarn_get_mscale(scaling_factor, mscale)
            / yarn_get_mscale(scaling_factor, mscale_all_dim)
            * attn_factor
        )
        self.softmax_mscale = softmax_mscale

        pos_freqs = base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)
        low, high = yarn_find_correction_range(beta_fast, beta_slow, rotary_dim, base, max_position_embeddings)
        inv_freq_mask = (1 - yarn_linear_ramp_mask(low, high, rotary_dim // 2)) * extrapolation_factor
        inv_freq = inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask

        t = torch.arange(max_position_embeddings * scaling_factor, dtype=torch.float32)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos() * self.softmax_mscale
        sin = freqs.sin() * self.softmax_mscale
        cache = torch.cat((cos, sin), dim=-1).float()
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def forward(self, positions, query, key):
        cache = self.cos_sin_cache
        if cache.dtype != torch.float32:
            cache = cache.float()
            self.cos_sin_cache = cache
        _sgl_rope(positions, query, key, self.head_dim, cache, is_neox=self.is_neox_style)
        return query, key
