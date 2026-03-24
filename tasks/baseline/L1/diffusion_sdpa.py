"""Non-causal scaled dot-product attention for diffusion models.

Unlike the LLM attention ops (which use paged KV cache), diffusion models use
standard bidirectional attention without any KV cache.

Input layout: (batch, seq_len, num_heads, head_dim).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionSDPA(nn.Module):
    """Non-causal scaled dot-product attention for diffusion transformers."""

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        softmax_scale: float | None = None,
        causal: bool = False,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        query, key, value : (B, S, H, D)
        softmax_scale : float, optional
        causal : bool
        """
        q = query.permute(0, 2, 1, 3)
        k = key.permute(0, 2, 1, 3)
        v = value.permute(0, 2, 1, 3)
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=0.0,
            is_causal=causal,
            scale=softmax_scale,
        )
        return out.permute(0, 2, 1, 3)
