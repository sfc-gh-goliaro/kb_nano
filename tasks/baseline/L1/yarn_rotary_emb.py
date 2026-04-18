"""YaRN (Yet Another RoPE extensioN) rotary position embeddings.

Extends RoPE with frequency scaling and magnitude correction for
long-context models. Used by gpt-oss.

Reuses the same L1 CUDA kernel (pos_enc.cu) as RotaryEmbedding; only the
cos/sin cache computation differs (YaRN scaling + mscale correction).

References:
  - Peng et al., "YaRN: Efficient Context Window Extension of Large Language Models"
  - vLLM: vllm/model_executor/layers/rotary_embedding/yarn_scaling_rope.py
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .rotary_emb import RotaryEmbedding


def _yarn_find_correction_dim(
    num_rotations: int, dim: int, base: float, max_position_embeddings: int,
) -> float:
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


def _yarn_find_correction_range(
    low_rot: float, high_rot: float, dim: int, base: float,
    max_position_embeddings: int, truncate: bool = True,
) -> tuple[float | int, float | int]:
    low = _yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    high = _yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
    if truncate:
        low = math.floor(low)
        high = math.ceil(high)
    return max(low, 0), min(high, dim - 1)


def _yarn_linear_ramp_mask(
    low: float, high: float, dim: int, dtype: torch.dtype,
) -> torch.Tensor:
    if low == high:
        high += 0.001
    linear_func = (torch.arange(dim, dtype=dtype) - low) / (high - low)
    return torch.clamp(linear_func, 0, 1)


def _yarn_get_mscale(scale: float) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * math.log(scale) + 1.0


class YaRNRotaryEmbedding(nn.Module):
    """YaRN RoPE with precomputed cos/sin cache.

    Uses the same L1 CUDA kernel as RotaryEmbedding for the rotation step.
    """

    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int,
        rope_theta: float,
        scaling_factor: float,
        original_max_position_embeddings: int,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        truncate: bool = True,
    ):
        super().__init__()
        self.head_dim = head_dim
        rotary_dim = head_dim

        pos_freqs = rope_theta ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim
        )
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)

        low, high = _yarn_find_correction_range(
            beta_fast, beta_slow, rotary_dim, rope_theta,
            original_max_position_embeddings, truncate,
        )
        inv_freq_mask = (
            1 - _yarn_linear_ramp_mask(low, high, rotary_dim // 2, dtype=torch.float)
        )
        inv_freq = (
            inv_freq_interpolation * (1 - inv_freq_mask)
            + inv_freq_extrapolation * inv_freq_mask
        )

        mscale = _yarn_get_mscale(scaling_factor)

        max_t = int(max_position_embeddings * scaling_factor)
        t = torch.arange(max_t, dtype=torch.float32)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos() * mscale
        sin = freqs.sin() * mscale
        cache = torch.cat((cos, sin), dim=-1).float()
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def forward(self, positions, query, key):
        cache = self.cos_sin_cache
        if cache.dtype != query.dtype:
            cache = cache.to(query.dtype)
        if torch.compiler.is_compiling():
            return RotaryEmbedding.forward_native(
                positions, query, key, self.head_dim, cache,
            )
        torch.ops.kb_nano_rope.rotary_embedding(
            positions, query, key, self.head_dim, cache, True,
        )
        return query, key
