"""Semantic PyTorch reference for attention_impl.

This file is used for specification/prompting and optional validation only.
It is not the production baseline and should not be used for reported speed.

Limitations: this reference keeps kb-nano's Context-driven interface but routes
the attention math through the semantic L1 references. Chunked local attention
metadata is intentionally approximated by the same Python remap helpers as the
baseline.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from kb_nano.infra.context import get_attn_backend_config, get_context
from kb_nano.tasks.reference.L1.flash_attn_decode import FlashAttnDecode
from kb_nano.tasks.reference.L1.flash_attn_prefill import FlashAttnPrefill
from kb_nano.tasks.reference.L1.flashinfer_decode import TRTLLMDecode
from kb_nano.tasks.reference.L1.flashinfer_prefill import TRTLLMPrefill
from kb_nano.tasks.reference.L1.store_kvcache import StoreKVCache, StoreKVCacheHND


def _chunked_prefill_remap(
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    block_tables: torch.Tensor | None,
    attention_chunk_size: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, int, int, torch.Tensor | None]:
    device = cu_seqlens_q.device
    cu_q_np = cu_seqlens_q.cpu().numpy()
    cu_k_np = cu_seqlens_k.cpu().numpy()
    q_seqlens = cu_q_np[1:] - cu_q_np[:-1]
    k_seqlens = cu_k_np[1:] - cu_k_np[:-1]
    q_tokens_in_first_block = np.minimum(
        attention_chunk_size - ((k_seqlens - q_seqlens) % attention_chunk_size),
        q_seqlens,
    ).astype(np.int32)
    tokens_in_last_block = (
        attention_chunk_size + (k_seqlens % -attention_chunk_size)
    ).astype(np.int32)
    local_blocks = (
        1 + np.ceil(
            np.maximum(q_seqlens - q_tokens_in_first_block, 0) / attention_chunk_size
        ).astype(np.int32)
    )
    cu_num_blocks = np.cumsum(local_blocks)
    virtual_batches = int(cu_num_blocks[-1])
    block_offsets = np.repeat(cu_num_blocks - local_blocks, local_blocks)
    arange = np.arange(virtual_batches, dtype=np.int32) - block_offsets
    rarange = np.repeat(local_blocks, local_blocks) - arange - 1
    seqlens_q_local = np.repeat(
        q_seqlens - q_tokens_in_first_block, local_blocks,
    ).astype(np.int32)
    seqlens_q_local[arange == 0] = q_tokens_in_first_block
    seqlens_q_local[arange > 0] = np.minimum(
        seqlens_q_local - attention_chunk_size * (arange - 1),
        attention_chunk_size,
    )[arange > 0]
    cu_q_out = np.empty(virtual_batches + 1, dtype=np.int32)
    np.cumsum(seqlens_q_local, out=cu_q_out[1:])
    cu_q_out[0] = 0
    seqlens_k_local = np.full(virtual_batches, attention_chunk_size, dtype=np.int32)
    seqlens_k_local[cu_num_blocks - 1] = tokens_in_last_block
    cu_k_out = np.empty(virtual_batches + 1, dtype=np.int32)
    np.cumsum(seqlens_k_local, out=cu_k_out[1:])
    cu_k_out[0] = 0
    block_tables_out = None
    if block_tables is not None and block_size > 0:
        pages_per_chunk = attention_chunk_size // block_size
        k_seqstarts_absolute = np.repeat(k_seqlens, local_blocks) - (
            rarange * attention_chunk_size
            + np.repeat(tokens_in_last_block, local_blocks)
        )
        block_starts = k_seqstarts_absolute // block_size
        block_indices = (
            block_starts[:, None]
            + np.arange(pages_per_chunk, dtype=np.int32)
        ).reshape(-1).clip(max=block_tables.shape[1] - 1)
        batch_indices = np.repeat(
            np.arange(len(q_seqlens), dtype=np.int32),
            local_blocks * pages_per_chunk,
        )
        block_tables_out = block_tables[
            torch.from_numpy(batch_indices),
            torch.from_numpy(block_indices),
        ].view(virtual_batches, -1)
    return (
        torch.from_numpy(cu_q_out).to(device=device),
        torch.from_numpy(cu_k_out).to(device=device),
        int(seqlens_q_local.max()) if virtual_batches > 0 else 0,
        int(seqlens_k_local.max()) if virtual_batches > 0 else 0,
        block_tables_out,
    )


def _chunked_decode_remap(
    cache_seqlens: torch.Tensor,
    block_tables: torch.Tensor | None,
    attention_chunk_size: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor | None, int]:
    local_seqlens = torch.clamp(cache_seqlens, max=attention_chunk_size)
    max_context_len = int(local_seqlens.max().item()) if local_seqlens.numel() > 0 else 0
    if block_tables is not None and block_size > 0:
        pages_per_chunk = attention_chunk_size // block_size
        chunk_start_page = (cache_seqlens - local_seqlens) // block_size
        offsets = torch.arange(pages_per_chunk, device=block_tables.device)
        page_indices = (chunk_start_page.unsqueeze(1) + offsets).clamp(
            max=block_tables.shape[1] - 1,
        )
        block_tables = torch.gather(block_tables, 1, page_indices)
    return local_seqlens, block_tables, max_context_len


class Attention(nn.Module):
    def __init__(self, num_heads: int, head_size: int, scale: float,
                 num_kv_heads: int | None = None,
                 sliding_window: int | None = None,
                 sinks: torch.nn.Parameter | None = None,
                 attention_chunk_size: int | None = None):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.sliding_window = sliding_window
        self.sinks = sinks
        self.attention_chunk_size = attention_chunk_size
        self.k_cache = self.v_cache = torch.tensor([])
        attn_cfg = get_attn_backend_config()
        self._use_trtllm = attn_cfg.use_trtllm
        self._block_size = attn_cfg.block_size
        self._fa3_sinks = sinks
        self._fa3_window_size = (
            (sliding_window - 1, 0) if sliding_window is not None else (-1, -1)
        )
        if self._use_trtllm:
            self.store_kvcache = StoreKVCacheHND(page_size=attn_cfg.block_size)
            self.prefill_op = TRTLLMPrefill(self.num_heads, self.num_kv_heads, head_size)
            self.decode_op = TRTLLMDecode(self.num_heads, self.num_kv_heads, head_size)
        else:
            self.store_kvcache = StoreKVCache()
            self.prefill_op = FlashAttnPrefill(self.num_heads, self.num_kv_heads, head_size)
            self.decode_op = FlashAttnDecode(self.num_heads, self.num_kv_heads, head_size)
        self._use_custom_op = False
        self._layer_name = ""

    def set_trtllm_workspace(self, workspace: torch.Tensor):
        if self._use_trtllm:
            self.decode_op._workspace = workspace
            self.prefill_op._workspace = workspace

    def forward_impl(self, query: torch.Tensor, key: torch.Tensor,
                     value: torch.Tensor) -> torch.Tensor:
        ctx = get_context()
        n = query.shape[0]
        q = query.view(n, self.num_heads, self.head_size)
        k = key.view(n, self.num_kv_heads, self.head_size)
        v = value.view(n, self.num_kv_heads, self.head_size)
        if self.k_cache.numel() and self.v_cache.numel():
            self.store_kvcache(k, v, self.k_cache, self.v_cache, ctx.slot_mapping)
        if ctx.is_mixed:
            out = self._forward_mixed(q, self.k_cache, self.v_cache, ctx)
        else:
            out = self._forward_pure(q, k, v, self.k_cache, self.v_cache, ctx)
        return out.reshape(n, self.num_heads * self.head_size)

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor) -> torch.Tensor:
        return self.forward_impl(query, key, value)

    def _forward_pure(self, q, k, v, k_cache, v_cache, ctx):
        fa_extra = {}
        if self._fa3_sinks is not None:
            fa_extra["s_aux"] = self._fa3_sinks
        if self._fa3_window_size != (-1, -1):
            fa_extra["window_size"] = self._fa3_window_size

        if ctx.is_prefill:
            cu_q, cu_k = ctx.cu_seqlens_q, ctx.cu_seqlens_k
            msq, msk = ctx.max_seqlen_q, ctx.max_seqlen_k
            bt = ctx.block_tables
            if self.attention_chunk_size is not None:
                cu_q, cu_k, msq, msk, bt = _chunked_prefill_remap(
                    cu_q, cu_k, bt, self.attention_chunk_size, self._block_size,
                )
            return self.prefill_op(
                q, k_cache if bt is not None else k, v_cache if bt is not None else v,
                cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                max_seqlen_q=msq, max_seqlen_k=msk,
                softmax_scale=self.scale, causal=True, block_table=bt, **fa_extra,
            )
        cache_seqlens, bt, max_ctx = ctx.context_lens, ctx.block_tables, ctx.max_context_len
        if self.attention_chunk_size is not None:
            cache_seqlens, bt, max_ctx = _chunked_decode_remap(
                cache_seqlens, bt, self.attention_chunk_size, self._block_size,
            )
        return self.decode_op(
            q, k_cache, v_cache,
            cache_seqlens=cache_seqlens, block_table=bt,
            softmax_scale=self.scale, causal=True, max_seq_len=max_ctx, **fa_extra,
        )

    def _forward_mixed(self, q, k_cache, v_cache, ctx):
        fa_extra = {}
        if self._fa3_sinks is not None:
            fa_extra["s_aux"] = self._fa3_sinks
        if self._fa3_window_size != (-1, -1):
            fa_extra["window_size"] = self._fa3_window_size

        np_ = ctx.num_prefill_tokens
        nd = ctx.num_decode_tokens
        out = torch.empty_like(q)
        if np_ > 0:
            out[:np_] = self.prefill_op(
                q[:np_], k_cache, v_cache,
                cu_seqlens_q=ctx.prefill_cu_seqlens_q,
                cu_seqlens_k=ctx.prefill_cu_seqlens_k,
                max_seqlen_q=ctx.prefill_max_seqlen_q,
                max_seqlen_k=ctx.prefill_max_seqlen_k,
                softmax_scale=self.scale, causal=True,
                block_table=ctx.prefill_block_tables, **fa_extra,
            )
        if nd > 0:
            out[np_:] = self.decode_op(
                q[np_:], k_cache, v_cache,
                cache_seqlens=ctx.decode_context_lens,
                block_table=ctx.decode_block_tables,
                softmax_scale=self.scale, causal=True,
                max_seq_len=ctx.decode_max_context_len, **fa_extra,
            )
        return out
