"""Rotary position embeddings (RoPE), with optional Llama 3.1-style frequency scaling.

Uses a custom CUDA kernel for high-performance in-place RoPE.  The pybind11
op is wrapped into a ``torch.library`` custom op with in-place mutation
annotations so ``torch.compile`` handles functionalization automatically.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .csrc import _C

# ---------------------------------------------------------------------------
# Register in-place rotary embedding op for torch.compile compatibility.
# Uses Tensor(a!) annotations so Inductor auto-functionalizes correctly.
# ---------------------------------------------------------------------------

_lib = torch.library.Library("kb_nano_rope", "DEF")

_lib.define(
    "rotary_embedding(Tensor positions, Tensor(a!) query, "
    "Tensor(b!)? key, int head_size, Tensor cos_sin_cache, "
    "bool is_neox) -> ()"
)

def _rotary_embedding_impl(
    positions, query, key, head_size, cos_sin_cache, is_neox,
):
    _C.rotary_embedding(positions, query, key, head_size, cos_sin_cache, is_neox)

_lib.impl("rotary_embedding", _rotary_embedding_impl, "CUDA")

@torch.library.impl(_lib, "rotary_embedding", "Meta")
def _rotary_embedding_meta(positions, query, key, head_size, cos_sin_cache, is_neox):
    pass


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


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


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

    @staticmethod
    def forward_native(positions, query, key, head_dim, cos_sin_cache):
        """Pure PyTorch RoPE — visible to Inductor for optimization."""
        cos_sin = cos_sin_cache[positions]
        cos = cos_sin[..., :cos_sin.shape[-1] // 2]
        sin = cos_sin[..., cos_sin.shape[-1] // 2:]

        rot_dim = cos.shape[-1]
        q_shape = query.shape
        k_shape = key.shape

        q = query.view(q_shape[0], -1, head_dim)
        k = key.view(k_shape[0], -1, head_dim)

        q_rot, q_pass = q[..., :rot_dim], q[..., rot_dim:]
        k_rot, k_pass = k[..., :rot_dim], k[..., rot_dim:]

        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        q_rot = q_rot * cos + _rotate_half(q_rot) * sin
        k_rot = k_rot * cos + _rotate_half(k_rot) * sin

        query = torch.cat([q_rot, q_pass], dim=-1).view(q_shape)
        key = torch.cat([k_rot, k_pass], dim=-1).view(k_shape)
        return query, key

    def forward_cuda(self, positions, query, key):
        """CUDA kernel path for eager mode."""
        cache = self.cos_sin_cache
        if cache.dtype != query.dtype:
            cache = cache.to(query.dtype)
        torch.ops.kb_nano_rope.rotary_embedding(
            positions, query, key, self.head_dim, cache, True,
        )
        return query, key

    def forward(self, positions, query, key):
        if torch.compiler.is_compiling():
            cache = self.cos_sin_cache
            if cache.dtype != query.dtype:
                cache = cache.to(query.dtype)
            return self.forward_native(
                positions, query, key, self.head_dim, cache,
            )
        return self.forward_cuda(positions, query, key)
