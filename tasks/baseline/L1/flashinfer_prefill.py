"""FlashInfer paged attention prefill kernel.

Accepts the same cu_seqlens-based interface as FlashAttnPrefill so that
LlamaAttention can dispatch to either backend without branch logic.
"""

import torch
import torch.nn as nn
from flashinfer.prefill import (
    BatchPrefillWithPagedKVCacheWrapper,
    trtllm_batch_context_with_kv_cache,
)


def _gather_paged_cache_hnd(cache: torch.Tensor, block_table: torch.Tensor,
                            seq_idx: int, seq_len: int) -> torch.Tensor:
    pieces = []
    remaining = int(seq_len)
    for block in block_table[seq_idx]:
        if remaining <= 0:
            break
        block_cache = cache[int(block.item())].transpose(0, 1)
        take = min(remaining, block_cache.shape[0])
        pieces.append(block_cache[:take])
        remaining -= take
    if not pieces:
        return cache.new_empty((0, cache.shape[1], cache.shape[-1]))
    return torch.cat(pieces, dim=0)


def _fallback_flash_attn_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k,
                                max_seqlen_q, max_seqlen_k, softmax_scale,
                                causal, block_table):
    if block_table is not None and k.ndim == 4:
        k_parts = []
        v_parts = []
        cu_k = [0]
        for i in range(cu_seqlens_k.numel() - 1):
            seq_len = int((cu_seqlens_k[i + 1] - cu_seqlens_k[i]).item())
            k_seq = _gather_paged_cache_hnd(k, block_table, i, seq_len)
            v_seq = _gather_paged_cache_hnd(v, block_table, i, seq_len)
            k_parts.append(k_seq)
            v_parts.append(v_seq)
            cu_k.append(cu_k[-1] + k_seq.shape[0])
        k = torch.cat(k_parts, dim=0) if k_parts else k.new_empty((0, k.shape[1], k.shape[-1]))
        v = torch.cat(v_parts, dim=0) if v_parts else v.new_empty((0, v.shape[1], v.shape[-1]))
        cu_seqlens_k = torch.tensor(cu_k, device=cu_seqlens_k.device, dtype=cu_seqlens_k.dtype)
        max_seqlen_k = max((cu_k[i + 1] - cu_k[i] for i in range(len(cu_k) - 1)), default=0)

    from flash_attn import flash_attn_varlen_func
    return flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=causal,
    )


class TRTLLMPrefill(nn.Module):
    def __init__(self, num_qo_heads: int, num_kv_heads: int, head_dim: int,
                 workspace: torch.Tensor | None = None):
        super().__init__()
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sm_scale = head_dim ** -0.5
        if workspace is None:
            workspace = torch.zeros(
                512 * 1024 * 1024, dtype=torch.uint8, device="cuda"
            )
        self._workspace = workspace
        self._paged_wrapper = None

    def _run_flashinfer_paged_prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        scale: float,
        causal: bool,
        block_table: torch.Tensor,
    ) -> torch.Tensor:
        page_size = k.shape[2]
        seq_lens = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).to(torch.int32)
        num_pages = torch.div(seq_lens + page_size - 1, page_size, rounding_mode="floor")
        paged_kv_indptr = torch.empty(seq_lens.numel() + 1, dtype=torch.int32, device=seq_lens.device)
        paged_kv_indptr[0] = 0
        paged_kv_indptr[1:] = torch.cumsum(num_pages, dim=0)
        page_offsets = torch.arange(block_table.shape[1], device=block_table.device)
        paged_kv_indices = block_table[page_offsets.unsqueeze(0) < num_pages.unsqueeze(1)].contiguous()
        paged_kv_last_page_len = ((seq_lens - 1) % page_size + 1).to(torch.int32)

        if self._paged_wrapper is None:
            self._paged_wrapper = BatchPrefillWithPagedKVCacheWrapper(
                self._workspace, kv_layout="HND", backend="auto",
            )
        self._paged_wrapper.plan(
            cu_seqlens_q,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            self.num_qo_heads,
            self.num_kv_heads,
            self.head_dim,
            page_size,
            causal=causal,
            sm_scale=scale,
            q_data_type=q.dtype,
            kv_data_type=k.dtype,
            o_data_type=q.dtype,
            seq_lens=seq_lens,
            seq_lens_q=(cu_seqlens_q[1:] - cu_seqlens_q[:-1]).to(torch.int32),
            block_tables=block_table,
        )
        return self._paged_wrapper.run(q.contiguous(), (k, v))

    def forward(self, q, k, v, cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k, softmax_scale=None,
                causal=True, block_table=None, **kwargs):
        scale = softmax_scale if softmax_scale is not None else self.sm_scale
        if block_table is not None:
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 10:
                return self._run_flashinfer_paged_prefill(
                    q, k, v,
                    cu_seqlens_q, cu_seqlens_k,
                    max_seqlen_q, max_seqlen_k,
                    scale,
                    causal,
                    block_table,
                )
            q = q.contiguous()
            seq_lens = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
            batch_size = seq_lens.shape[0]
            return trtllm_batch_context_with_kv_cache(
                query=q,
                kv_cache=(k, v),
                workspace_buffer=self._workspace,
                block_tables=block_table,
                seq_lens=seq_lens,
                max_q_len=max_seqlen_q,
                max_kv_len=max_seqlen_k,
                bmm1_scale=scale,
                bmm2_scale=1.0,
                batch_size=batch_size,
                cum_seq_lens_q=cu_seqlens_q,
                cum_seq_lens_kv=cu_seqlens_k,
                kv_layout="HND",
            )
        from flash_attn import flash_attn_varlen_func
        return flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
            softmax_scale=scale,
            causal=causal,
        )
