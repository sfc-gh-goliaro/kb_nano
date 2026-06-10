"""Variable-length Flash Attention (no KV cache lookup).

Thin ``nn.Module`` wrapper around ``flash_attn_varlen_func`` with the same
3-way fallback as :mod:`flash_attn_prefill`: vLLM's bundled FA3 on Hopper,
then upstream ``flash_attn`` (FA2), then ``flash_mla``.

Used by MLA prefill and chunked-context paths where Q, K, V are dense
``[total_tokens, num_heads, head_dim]`` tensors (no paged cache lookup,
no ``block_table``).  Supports ``return_softmax_lse`` for MLA chunked
prefix merging.
"""

from __future__ import annotations

import torch
import torch.nn as nn

_VLLM_FA_AVAILABLE = False
_fa3_varlen_func = None
_get_fa_version = None
try:
    from vllm.v1.attention.backends.fa_utils import (
        flash_attn_varlen_func as _vllm_fa_varlen,
        get_flash_attn_version as _vllm_get_fa_version,
    )
    if torch.cuda.is_available():
        _VLLM_FA_AVAILABLE = True
        _fa3_varlen_func = _vllm_fa_varlen
        _get_fa_version = _vllm_get_fa_version
except ImportError:
    pass

_fa2_varlen_func = None
_flashmla_varlen_func = None
if not _VLLM_FA_AVAILABLE:
    try:
        from flash_attn import flash_attn_varlen_func as _fa2_varlen_func
    except ImportError:
        # vLLM vendors FlashMLA; fall back to the standalone ``flash_mla``
        # package when the vendored copy is unavailable.
        try:
            from vllm.third_party.flashmla.flash_mla_interface import (
                flash_attn_varlen_func as _flashmla_varlen_func,
            )
        except ImportError:  # pragma: no cover
            from flash_mla import (  # type: ignore[no-redef]
                flash_attn_varlen_func as _flashmla_varlen_func,
            )


class FlashAttnVarlen(nn.Module):
    """Variable-length Flash Attention without paged KV cache lookup."""

    def __init__(self, head_dim: int | None = None, has_sinks: bool = False):
        super().__init__()
        self.fa_version = (
            _get_fa_version(head_size=head_dim, has_sinks=has_sinks)
            if _get_fa_version is not None else None
        )

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
        return_softmax_lse: bool = False,
    ):
        if _VLLM_FA_AVAILABLE and self.fa_version is not None:
            kwargs = dict(
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=softmax_scale,
                causal=causal,
                return_softmax_lse=return_softmax_lse,
            )
            kwargs["fa_version"] = self.fa_version
            return _fa3_varlen_func(q, k, v, **kwargs)
        fn = _fa2_varlen_func if _fa2_varlen_func is not None else _flashmla_varlen_func
        kwargs = dict(
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
        )
        if return_softmax_lse:
            kwargs["return_softmax_lse"] = return_softmax_lse
        return fn(q, k, v, **kwargs)
