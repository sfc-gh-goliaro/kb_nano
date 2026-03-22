"""Non-causal scaled dot-product attention for diffusion models.

Unlike the LLM attention ops (which use paged KV cache), diffusion models use
standard bidirectional attention without any KV cache. This L1 op provides
both Flash Attention and PyTorch SDPA backends.

Input layout: (batch, seq_len, num_heads, head_dim).

Mirrors vllm-omni's diffusion attention backends.
"""

from __future__ import annotations

from importlib.util import find_spec

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionSDPA(nn.Module):
    """Non-causal scaled dot-product attention for diffusion transformers.

    Automatically selects Flash Attention when available, otherwise falls back
    to PyTorch SDPA.
    """

    def __init__(self) -> None:
        super().__init__()
        self._has_flash_attn = find_spec("flash_attn") is not None

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
        if self._has_flash_attn and query.dtype != torch.float32:
            return self._flash_forward(query, key, value, softmax_scale, causal)
        return self._sdpa_forward(query, key, value, softmax_scale, causal)

    def _flash_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        softmax_scale: float | None,
        causal: bool,
    ) -> torch.Tensor:
        from flash_attn import flash_attn_func
        out = flash_attn_func(query, key, value, causal=causal, softmax_scale=softmax_scale)
        if isinstance(out, tuple):
            out = out[0]
        return out

    def _sdpa_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        softmax_scale: float | None,
        causal: bool,
    ) -> torch.Tensor:
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
