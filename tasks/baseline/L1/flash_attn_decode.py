"""Flash attention decode kernel (with paged KV cache).

On Hopper (SM90) when vLLM's bundled FA3 is available, uses the unified
``flash_attn_varlen_func(fa_version=3)`` interface which is significantly
faster.  Falls back to ``flash_attn_with_kvcache`` (FA2) otherwise.
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
    from flash_attn import flash_attn_with_kvcache


class FlashAttnDecode(nn.Module):
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
        self._cu_seqlens_q = None
        self.fa_version = (
            _get_fa_version(head_size=head_dim, has_sinks=has_sinks)
            if _get_fa_version is not None else None
        )

    def _get_cu_seqlens_q(self, n: int, device: torch.device) -> torch.Tensor:
        needed = n + 1
        if self._cu_seqlens_q is None or self._cu_seqlens_q.numel() < needed:
            self._cu_seqlens_q = torch.arange(
                needed, dtype=torch.int32, device=device,
            )
        return self._cu_seqlens_q[:needed]

    def forward(self, q, k_cache, v_cache, cache_seqlens=None, **kwargs):
        max_seq_len = kwargs.pop("max_seq_len", None)
        if _VLLM_FA_AVAILABLE and self.fa_version is not None:
            block_table = kwargs.pop("block_table", None)
            softmax_scale = kwargs.pop("softmax_scale", None)
            num_splits = kwargs.pop("num_splits", None)
            kwargs.pop("causal", None)

            n = q.shape[0]
            cu_seqlens_q = self._get_cu_seqlens_q(n, q.device)
            if max_seq_len is not None:
                max_seqlen_k = max_seq_len
            else:
                max_seqlen_k = int(cache_seqlens.max().item()) if cache_seqlens.numel() > 0 else 0

            fa3_kw = dict(
                q=q,
                k=k_cache,
                v=v_cache,
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=1,
                seqused_k=cache_seqlens,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=softmax_scale,
                causal=True,
                block_table=block_table,
                fa_version=self.fa_version,
            )
            if num_splits is not None:
                fa3_kw["num_splits"] = num_splits
            fa3_kw.update(kwargs)
            return _fa3_varlen_func(**fa3_kw)

        return flash_attn_with_kvcache(
            q.unsqueeze(1), k_cache, v_cache,
            cache_seqlens=cache_seqlens, **kwargs,
        ).squeeze(1)
