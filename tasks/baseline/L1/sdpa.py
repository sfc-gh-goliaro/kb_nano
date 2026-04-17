"""Scaled dot-product attention primitive (L1).

Thin wrapper around ``torch.nn.functional.scaled_dot_product_attention``
so L2 callers needing PyTorch's fused SDPA (e.g. MLA, which carries its
own KV cache outside the engine's paged pool) don't import
``torch.nn.functional`` directly.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SDPA(nn.Module):
    """Scaled dot-product attention.

    Inputs follow ``[B, H, T, D]`` layout (PyTorch SDPA convention).
    ``scale`` is the softmax temperature; ``is_causal`` enables the
    triangular causal mask.
    """

    def __init__(self, scale: float | None = None, is_causal: bool = False):
        super().__init__()
        self.scale = scale
        self.is_causal = is_causal

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        is_causal: bool | None = None,
        scale: float | None = None,
    ) -> torch.Tensor:
        return F.scaled_dot_product_attention(
            q, k, v,
            is_causal=self.is_causal if is_causal is None else is_causal,
            scale=self.scale if scale is None else scale,
        )
