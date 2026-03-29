"""Rotary position embeddings (RoPE), with optional Llama 3.1-style frequency scaling.

Uses a custom CUDA kernel for high-performance in-place RoPE.
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load as _load_ext

_CSRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")
_C = _load_ext(
    name="kb_nano_L1_ops",
    sources=[os.path.join(_CSRC, f) for f in [
        "binding.cpp", "rmsnorm.cu", "activation.cu", "pos_enc.cu",
        "moe_sum.cu", "moe_align.cu", "moe_topk_softmax.cu",
    ]],
    extra_cuda_cflags=["-O3", "--use_fast_math",
                       "-DFLASHINFER_ENABLE_BF16", "-DFLASHINFER_ENABLE_F16"],
    extra_cflags=["-O3"],
    verbose=False,
)


def _compute_scaled_inv_freq(
    inv_freq: torch.Tensor,
    scaling_factor: float,
    low_freq_factor: float,
    high_freq_factor: float,
    original_max_position_embeddings: int,
) -> torch.Tensor:
    low_wl = original_max_position_embeddings / low_freq_factor
    high_wl = original_max_position_embeddings / high_freq_factor
    wl = 2 * math.pi / inv_freq
    if low_freq_factor != high_freq_factor:
        smooth = (original_max_position_embeddings / wl - low_freq_factor) / (
            high_freq_factor - low_freq_factor
        )
    else:
        smooth = torch.zeros_like(inv_freq)
    return torch.where(
        wl < high_wl,
        inv_freq,
        torch.where(
            wl > low_wl,
            inv_freq / scaling_factor,
            (1 - smooth) * inv_freq / scaling_factor + smooth * inv_freq,
        ),
    )


class RotaryEmbedding(nn.Module):
    """RoPE with optional Llama 3.1-style frequency scaling.

    When rope_scaling_factor == 1.0 (default), behaves as standard RoPE.
    When rope_scaling_factor != 1.0, applies the Llama 3.1 piecewise
    frequency scaling controlled by low/high freq factors.
    """

    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int,
        rope_theta: float,
        rope_scaling_factor: float = 1.0,
        rope_low_freq_factor: float = 1.0,
        rope_high_freq_factor: float = 1.0,
        rope_original_max_position_embeddings: int | None = None,
    ):
        super().__init__()
        self.head_dim = head_dim
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))

        if rope_scaling_factor != 1.0 and rope_original_max_position_embeddings is not None:
            inv_freq = _compute_scaled_inv_freq(
                inv_freq,
                rope_scaling_factor,
                rope_low_freq_factor,
                rope_high_freq_factor,
                rope_original_max_position_embeddings,
            )

        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1).float()
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def forward(self, positions, query, key):
        cache = self.cos_sin_cache
        if cache.dtype != query.dtype:
            cache = cache.to(query.dtype)
        _C.rotary_embedding(positions, query, key, self.head_dim, cache, True)
        return query, key
