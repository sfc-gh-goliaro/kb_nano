"""Flash attention prefill kernel (variable-length sequences).

Uses vLLM's bundled FlashAttention when available so numerics match vLLM:
FA3 on Hopper (SM90), FA4 on Blackwell (SM100+). Falls back to FA2 otherwise,
including for head sizes FA4 cannot serve on Blackwell.
"""

import torch
import torch.nn as nn

from ._fa_backend import VLLM_FA_AVAILABLE as _FA3_AVAILABLE
from ._fa_backend import fa_version_for_head_dim as _fa_version_for_head_dim
from ._fa_backend import vllm_fa_varlen_func as _fa3_varlen_func

if not _FA3_AVAILABLE:
    from flash_attn import flash_attn_varlen_func as _fa2_varlen_func


class FlashAttnPrefill(nn.Module):
    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sm_scale = head_dim ** -0.5
        # FA4 cannot serve every head size on Blackwell; resolve per layer.
        self._fa_version = _fa_version_for_head_dim(head_dim)

    def forward(self, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, **kwargs):
        if _FA3_AVAILABLE:
            # vLLM's wrapper parameter order differs from standard flash_attn — use kwargs.
            # FA3 requires seqused_k (not cu_seqlens_k) when block_table is provided.
            fa3_kw = dict(
                max_seqlen_q=max_seqlen_q,
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_k=max_seqlen_k,
                fa_version=self._fa_version,
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
