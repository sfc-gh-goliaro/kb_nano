"""Flash attention prefill kernel (variable-length sequences).

On Hopper (SM90) when vLLM's bundled FA3 is available, uses FA3 to match
vLLM's numerical behavior. Falls back to FA2 otherwise.
"""

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

if not _VLLM_FA_AVAILABLE:
    from flash_attn import flash_attn_varlen_func as _fa2_varlen_func


class FlashAttnPrefill(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        has_sinks: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sm_scale = head_dim ** -0.5
        self.fa_version = (
            _get_fa_version(head_size=head_dim, has_sinks=has_sinks)
            if _get_fa_version is not None else None
        )

    def forward(self, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, **kwargs):
        if _VLLM_FA_AVAILABLE and self.fa_version is not None:
            # vLLM's wrapper parameter order differs from standard flash_attn — use kwargs.
            # FA3 requires seqused_k (not cu_seqlens_k) when block_table is provided.
            fa3_kw = dict(
                max_seqlen_q=max_seqlen_q,
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_k=max_seqlen_k,
                fa_version=self.fa_version,
            )
            if kwargs.get("block_table") is not None:
                seqused_k = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
                fa3_kw["seqused_k"] = seqused_k
            else:
                fa3_kw["cu_seqlens_k"] = cu_seqlens_k
            fa3_kw.update(kwargs)
            return _fa3_varlen_func(q, k, v, **fa3_kw)
        return _fa2_varlen_func(
            q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, **kwargs,
        )
