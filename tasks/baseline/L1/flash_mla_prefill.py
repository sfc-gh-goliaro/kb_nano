"""FlashMLA prefill kernel for MLA (Multi-head Latent Attention)."""

from __future__ import annotations

import torch
import torch.nn as nn

from flash_mla import flash_attn_varlen_func


class FlashMLAPrefill(nn.Module):
    """Wraps flash_mla.flash_attn_varlen_func for variable-length MLA prefill."""

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        softmax_scale: float,
        causal: bool = True,
    ) -> torch.Tensor:
        return flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
        )
