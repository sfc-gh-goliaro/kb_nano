"""Dense (non-varlen) Flash Attention forward (L1).

Wraps ``flash_attn.flash_attn_func`` for the simple ``[B, T, H, D]``
input layout used when KV-cache management is done outside the kernel
(e.g. Qwen3-Next's full attention layers, which carry their cache in
per-sequence ``layer_state`` dicts rather than the engine's paged pool).

Falls back to vLLM's bundled FA3 wrapper on Hopper if available, for
numerical alignment with vLLM. Otherwise uses upstream ``flash_attn``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from flash_attn import flash_attn_func as _fa_flash_attn_func


class FlashAttnDense(nn.Module):
    """Dense Flash Attention.

    Inputs (q, k, v) follow ``[B, T, H, D]`` layout. ``causal`` controls
    triangular masking; with ``q_len < k_len`` the kernel uses bottom-right
    alignment, which is what we want for cached decode.
    """

    def __init__(self, softmax_scale: float, causal: bool = True):
        super().__init__()
        self.softmax_scale = softmax_scale
        self.causal = causal

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        return _fa_flash_attn_func(
            q, k, v,
            softmax_scale=self.softmax_scale,
            causal=self.causal,
        )
