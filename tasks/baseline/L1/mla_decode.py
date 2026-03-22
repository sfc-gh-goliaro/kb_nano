"""MLA paged decode attention using FlashInfer's MLA kernel.

Uses BatchMLAPagedAttentionWrapper for efficient paged decode without
materializing the full KV cache per head (avoids GQA broadcast OOM).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from flashinfer.mla import BatchMLAPagedAttentionWrapper


class MLADecode(nn.Module):
    def __init__(self, num_heads: int, kv_lora_rank: int, kv_cache_head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = kv_cache_head_dim - kv_lora_rank
        self.kv_cache_head_dim = kv_cache_head_dim
        self._wrapper: BatchMLAPagedAttentionWrapper | None = None
        self._workspace: torch.Tensor | None = None

    def _get_wrapper(self, device: torch.device) -> BatchMLAPagedAttentionWrapper:
        if self._wrapper is None:
            self._workspace = torch.empty(
                128 * 1024 * 1024, dtype=torch.int8, device=device
            )
            self._wrapper = BatchMLAPagedAttentionWrapper(
                self._workspace, backend="fa2"
            )
        return self._wrapper

    def forward(self, q, k_cache, v_cache, cache_seqlens=None,
                block_table=None, softmax_scale=None, **kwargs):
        """
        q: [batch, num_heads, head_dim]   (head_dim = kv_lora_rank + qk_rope_head_dim)
        k_cache: [num_blocks, page_size, num_kv_heads=1, head_dim]
        v_cache: not used (same data as k_cache for MLA)
        cache_seqlens: [batch]
        block_table: [batch, max_num_blocks_per_seq]
        """
        batch = q.shape[0]
        page_size = k_cache.shape[1]

        if softmax_scale is None:
            softmax_scale = self.kv_cache_head_dim ** -0.5

        wrapper = self._get_wrapper(q.device)

        q_nope = q[..., :self.kv_lora_rank]
        q_pe = q[..., self.kv_lora_rank:]

        kc = k_cache.squeeze(2)
        ckv_cache = kc[..., :self.kv_lora_rank]
        kpe_cache = kc[..., self.kv_lora_rank:]

        qo_indptr = torch.arange(batch + 1, dtype=torch.int32, device=q.device)

        pages_per_seq = (cache_seqlens + page_size - 1) // page_size
        kv_indptr = torch.zeros(batch + 1, dtype=torch.int32, device=q.device)
        kv_indptr[1:] = pages_per_seq.to(torch.int32).cumsum(0)

        max_pages = int(pages_per_seq.max().item())
        bt_trimmed = block_table[:, :max_pages].to(torch.int32)
        mask = torch.arange(max_pages, device=q.device).unsqueeze(0) < pages_per_seq.unsqueeze(1)
        kv_indices = bt_trimmed[mask]

        wrapper.plan(
            qo_indptr, kv_indptr, kv_indices, cache_seqlens.to(torch.int32),
            self.num_heads, self.kv_lora_rank, self.qk_rope_head_dim,
            page_size, False, softmax_scale,
            q.dtype, ckv_cache.dtype,
        )

        out = wrapper.run(q_nope, q_pe, ckv_cache, kpe_cache)
        return out
