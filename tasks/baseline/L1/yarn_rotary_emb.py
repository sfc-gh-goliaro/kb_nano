"""YaRN rotary position embeddings for DeepSeek-V3.

Implements the YaRN (Yet another RoPE extensioN) method with mscale
correction, used by DeepSeek-V2/V3/V3.2 for extended context windows.

Uses sgl_kernel.apply_rope_with_cos_sin_cache_inplace for in-place RoPE.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from sgl_kernel import apply_rope_with_cos_sin_cache_inplace as _sgl_rope


def _yarn_find_correction_dim(
    num_rotations: int, dim: int, base: float, max_position_embeddings: int,
) -> float:
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


def _yarn_find_correction_range(
    beta_fast: int, beta_slow: int, dim: int, base: float,
    max_position_embeddings: int,
) -> tuple[int, int]:
    low = _yarn_find_correction_dim(beta_slow, dim, base, max_position_embeddings)
    high = _yarn_find_correction_dim(beta_fast, dim, base, max_position_embeddings)
    low = max(math.floor(low), 0)
    high = min(math.ceil(high), dim - 1)
    return low, high


def _yarn_linear_ramp_mask(low: float, high: float, dim: int) -> torch.Tensor:
    if low == high:
        high += 0.001
    linear_func = (torch.arange(dim, dtype=torch.float) - low) / (high - low)
    return torch.clamp(linear_func, 0, 1)


def yarn_get_mscale(scale: float, mscale: float = 1.0) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class YaRNRotaryEmbedding(nn.Module):
    """YaRN RoPE with mscale for DeepSeek models.

    The cos/sin cache is pre-scaled by mscale so the attention kernel
    receives corrected embeddings without extra per-token math.
    """

    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int,
        rope_theta: float,
        scaling_factor: float = 1.0,
        original_max_position_embeddings: int = 4096,
        beta_fast: int = 32,
        beta_slow: int = 1,
        mscale: float = 1.0,
        mscale_all_dim: float = 0.0,
        attn_factor: float = 1.0,
    ):
        super().__init__()
        self.head_dim = head_dim

        mscale_val = (
            yarn_get_mscale(scaling_factor, mscale)
            / yarn_get_mscale(scaling_factor, mscale_all_dim)
            * attn_factor
        )

        pos_freqs = rope_theta ** (
            torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim
        )
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)

        low, high = _yarn_find_correction_range(
            beta_fast, beta_slow, head_dim, rope_theta,
            original_max_position_embeddings,
        )
        inv_freq_mask = 1 - _yarn_linear_ramp_mask(low, high, head_dim // 2)
        inv_freq = (
            inv_freq_interpolation * (1 - inv_freq_mask)
            + inv_freq_extrapolation * inv_freq_mask
        )

        max_len = int(max_position_embeddings * scaling_factor)
        t = torch.arange(max_len, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos() * mscale_val
        sin = freqs.sin() * mscale_val
        cache = torch.cat((cos, sin), dim=-1).float()
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def forward(self, positions, query, key):
        cache = self.cos_sin_cache
        if cache.dtype != torch.float32:
            cache = cache.float()
            self.cos_sin_cache = cache
        _sgl_rope(positions, query, key, self.head_dim, cache, is_neox=False)
        return query, key
