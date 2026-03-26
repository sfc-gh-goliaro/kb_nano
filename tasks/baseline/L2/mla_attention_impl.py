"""MLA attention implementation with FP8 paged KV cache.

MLA equivalent of attention_impl.py's Attention class. Handles:
- FP8 KV cache storage (656 bytes/token)
- Dense prefill via FlashMLAPrefill
- Dense/sparse decode via FlashMLADecode (FP8 sparse kernel)
- Sparse prefill via FlashMLASparsePrefill (BF16 workspace)
- Mixed batch (prefill + decode) with separate FP8/BF16 paths

Matches vllm's FlashMLASparseBackend with FP8 separate prefill/decode mode.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ....infra.context import get_context
from ..L1.store_kvcache_fp8_mla import StoreKVCacheFP8MLA, GatherKVCacheFP8MLA
from ..L1.flash_mla_decode import FlashMLADecode, FlashMLAGetMetadata
from ..L1.flash_mla_prefill import FlashMLAPrefill
from ..L1.flash_mla_sparse_prefill import FlashMLASparsePrefill
from ..L1.convert_indices import ConvertIndicesToGlobal

_MLA_HEAD_DIM_V = 512
_MLA_WORKSPACE_HEAD_SIZE = 576  # 512 NoPE + 64 RoPE = 576 BF16 dims
MIN_HEADS_FOR_BF16_PREFILL = 32


def _compute_fp8_decode_padded_heads(num_heads: int) -> int:
    return 64 if num_heads <= 64 else 128


class MLAAttention(nn.Module):
    """MLA attention with FP8 paged KV cache.

    Unlike standard Attention which has separate k_cache and v_cache,
    MLA uses a single unified cache since kv_c_normed + k_pe are stored together.

    Attributes:
        k_cache, v_cache: both point to the same tensor for engine discovery
        _num_kv_heads: always 1 (MLA = multi-query on the latent)
        _head_dim: kv_lora_rank + qk_rope_head_dim (for cache slot size)
    """

    def __init__(self, num_heads: int, scale: float,
                 qk_nope_head_dim: int, qk_rope_head_dim: int,
                 v_head_dim: int, kv_lora_rank: int,
                 is_sparse: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.scale = scale
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.is_sparse = is_sparse

        self._num_kv_heads = 1
        self._head_dim = 656

        self.k_cache = self.v_cache = torch.tensor([])

        self.fp8_decode_padded_heads = _compute_fp8_decode_padded_heads(num_heads)

        # W_UV: absorbed V projection from kv_b_proj, computed after weight loading.
        # Shape: [num_heads, kv_lora_rank, v_head_dim] — used to project
        # FlashMLA's kv_lora_rank-dim output to v_head_dim per head.
        self.W_UV: torch.Tensor | None = None

        self.store_kvcache = StoreKVCacheFP8MLA()
        self.gather_kvcache = GatherKVCacheFP8MLA()
        self.decode_op = FlashMLADecode()
        self.prefill_op = FlashMLAPrefill()
        self.sparse_prefill_op = FlashMLASparsePrefill()
        self.get_metadata = FlashMLAGetMetadata()
        self.convert_indices = ConvertIndicesToGlobal()

    def forward(self, q: torch.Tensor, kv_c_normed: torch.Tensor,
                k_pe: torch.Tensor, kv_b_proj: nn.Module,
                topk_indices: torch.Tensor | None = None,
                output_shape: tuple | None = None) -> torch.Tensor:
        ctx = get_context()
        N = q.shape[0]

        kv_cache = self.k_cache

        if kv_cache.numel() and ctx.slot_mapping is not None:
            self.store_kvcache(kv_c_normed, k_pe, kv_cache, ctx.slot_mapping)

        if self.is_sparse and topk_indices is not None:
            o = self._forward_sparse(q, kv_c_normed, k_pe, kv_b_proj, kv_cache, ctx, topk_indices)
        elif ctx.is_mixed:
            o = self._forward_mixed(q, kv_c_normed, k_pe, kv_b_proj, kv_cache, ctx)
        else:
            o = self._forward_pure(q, kv_c_normed, k_pe, kv_b_proj, kv_cache, ctx)

        if output_shape is not None:
            o = o.view(*output_shape)
        return o

    def _forward_pure(self, q, kv_c_normed, k_pe, kv_b_proj, kv_cache, ctx):
        if ctx.is_prefill:
            return self._forward_dense_prefill(q, kv_c_normed, k_pe, kv_b_proj, ctx)
        return self._forward_dense_decode(q, kv_cache, ctx)

    def _forward_dense_prefill(self, q, kv_c_normed, k_pe, kv_b_proj, ctx):
        N = q.shape[0]
        kv = kv_b_proj(kv_c_normed)
        kv = kv.view(N, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k = torch.empty(N, self.num_heads, self.qk_head_dim, dtype=q.dtype, device=q.device)
        k[..., :self.qk_nope_head_dim] = k_nope
        k[..., self.qk_nope_head_dim:] = k_pe.expand(-1, self.num_heads, -1)

        o = self.prefill_op(
            q, k, v,
            cu_seqlens_q=ctx.cu_seqlens_q,
            cu_seqlens_k=ctx.cu_seqlens_k,
            max_seqlen_q=ctx.max_seqlen_q,
            max_seqlen_k=ctx.max_seqlen_k,
            softmax_scale=self.scale,
            causal=True,
        )
        return o.view(N, self.num_heads * self.v_head_dim)

    def _v_up_proj(self, attn_out: torch.Tensor) -> torch.Tensor:
        """Project FlashMLA output from kv_lora_rank to v_head_dim per head."""
        if self.W_UV is None:
            return attn_out[..., :self.v_head_dim]
        N = attn_out.shape[0]
        o = attn_out.view(N, self.num_heads, self.kv_lora_rank)
        o = o.transpose(0, 1)
        o = torch.bmm(o, self.W_UV)
        return o.transpose(0, 1).reshape(N, self.num_heads * self.v_head_dim)

    def _forward_dense_decode(self, q, kv_cache, ctx):
        N = q.shape[0]
        cache_seqlens = ctx.context_lens
        block_table = ctx.block_tables

        tile_sched_meta, _ = self.get_metadata(
            cache_seqlens, self.num_heads, num_heads_k=1,
            is_fp8_kvcache=True)

        o, _ = self.decode_op(
            q, kv_cache.view(torch.uint8).unsqueeze(-2),
            block_table, cache_seqlens,
            head_dim_v=_MLA_HEAD_DIM_V,
            tile_scheduler_metadata=tile_sched_meta,
            softmax_scale=self.scale,
            is_fp8_kvcache=True,
        )
        return self._v_up_proj(o)

    def _forward_sparse(self, q, kv_c_normed, k_pe, kv_b_proj, kv_cache, ctx, topk_indices):
        """Sparse attention: FP8 decode kernel for decode, BF16 workspace for prefill."""
        N = q.shape[0]

        if ctx.is_prefill:
            return self._forward_sparse_separate(
                q, kv_c_normed, k_pe, kv_b_proj, kv_cache, ctx, topk_indices,
                num_prefill_tokens=N, num_decode_tokens=0)
        elif ctx.is_mixed:
            np_ = ctx.num_prefill_tokens
            nd = ctx.num_decode_tokens
            return self._forward_sparse_separate(
                q, kv_c_normed, k_pe, kv_b_proj, kv_cache, ctx, topk_indices,
                num_prefill_tokens=np_, num_decode_tokens=nd)
        else:
            return self._forward_sparse_decode(q, kv_cache, ctx, topk_indices)

    def _pad_q_for_fp8(self, q: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Pad num_heads to 64 or 128 as required by the FP8 sparse decode kernel."""
        actual_heads = q.shape[-2]
        padded_heads = self.fp8_decode_padded_heads
        if actual_heads >= padded_heads:
            return q, actual_heads
        pad_shape = list(q.shape)
        pad_shape[-2] = padded_heads
        q_padded = q.new_zeros(pad_shape)
        q_padded[..., :actual_heads, :] = q
        return q_padded, actual_heads

    def _forward_sparse_decode(self, q, kv_cache, ctx, topk_indices):
        """Sparse FP8 decode: per-request batching with padded heads."""
        N = q.shape[0]
        block_size = int(kv_cache.shape[1])
        num_decodes = ctx.block_tables.shape[0]

        req_ids = getattr(ctx, 'req_id_per_token', None)
        if req_ids is None:
            req_ids = torch.arange(N, dtype=torch.int32, device=q.device)

        topk_indices = self.convert_indices(
            topk_indices, ctx.block_tables, block_size, req_ids=req_ids)

        decode_query_len = N // num_decodes if num_decodes > 0 else N
        q_4d = q.view(num_decodes, decode_query_len, self.num_heads, q.shape[-1])
        topk_4d = topk_indices.view(num_decodes, decode_query_len, -1)

        q_4d, actual_heads = self._pad_q_for_fp8(q_4d)
        padded_heads = q_4d.shape[-2]

        topk = topk_indices.shape[-1]
        topk_tensor = torch.full(
            (num_decodes,), topk, dtype=torch.int32, device=q.device)
        dummy_bt = torch.empty(
            (num_decodes, 1), dtype=torch.int32, device=q.device)

        tile_sched_meta, _ = self.get_metadata(
            topk_tensor, decode_query_len * padded_heads,
            topk=topk, num_heads_q=padded_heads,
            num_heads_k=1, is_fp8_kvcache=True)

        o, _ = self.decode_op(
            q_4d, kv_cache.view(torch.uint8).unsqueeze(-2),
            dummy_bt, topk_tensor,
            head_dim_v=_MLA_HEAD_DIM_V,
            tile_scheduler_metadata=tile_sched_meta,
            softmax_scale=self.scale,
            is_fp8_kvcache=True,
            indices=topk_4d,
        )

        o = o.view(-1, padded_heads, o.shape[-1])
        if actual_heads < padded_heads:
            o = o[:, :actual_heads, :]
        return self._v_up_proj(o)

    def _forward_sparse_separate(self, q, kv_c_normed, k_pe, kv_b_proj,
                                 kv_cache, ctx, topk_indices,
                                 num_prefill_tokens, num_decode_tokens):
        """Separate prefill (BF16 workspace) and decode (FP8 kernel)."""
        N = q.shape[0]
        block_size = int(kv_cache.shape[1])

        num_seqs_total = ctx.block_tables.shape[0]
        num_decode_seqs = getattr(ctx, 'num_decode_seqs', num_seqs_total if num_decode_tokens > 0 else 0)
        num_prefill_seqs = num_seqs_total - num_decode_seqs

        req_ids = getattr(ctx, 'req_id_per_token', None)
        if req_ids is None:
            req_ids = torch.arange(N, dtype=torch.int32, device=q.device)

        prefill_request_ids = None
        prefill_workspace_starts = None
        has_prefill = num_prefill_tokens > 0

        if has_prefill:
            if ctx.is_mixed:
                pf_bt = ctx.prefill_block_tables if hasattr(ctx, 'prefill_block_tables') else ctx.block_tables[num_decode_seqs:]
                pf_seq_lens = ctx.prefill_seq_lens if hasattr(ctx, 'prefill_seq_lens') else (ctx.cu_seqlens_k[1:] - ctx.cu_seqlens_k[:-1])
            else:
                pf_bt = ctx.block_tables
                pf_cu = ctx.cu_seqlens_k
                pf_seq_lens = pf_cu[1:] - pf_cu[:-1]

            prefill_request_ids = torch.full((N,), -1, dtype=torch.int32, device=q.device)
            prefill_workspace_starts = torch.zeros(num_prefill_seqs, dtype=torch.int32, device=q.device)

            if num_prefill_seqs > 1:
                prefill_workspace_starts[1:] = torch.cumsum(pf_seq_lens[:-1], dim=0).int()

            if ctx.is_mixed:
                pf_cu_q = ctx.prefill_cu_seqlens_q if hasattr(ctx, 'prefill_cu_seqlens_q') else ctx.cu_seqlens_q
                for req_idx in range(num_prefill_seqs):
                    global_idx = num_decode_seqs + req_idx
                    qs = int(pf_cu_q[req_idx].item()) if hasattr(ctx, 'prefill_cu_seqlens_q') else int(ctx.cu_seqlens_q[global_idx].item())
                    qe = int(pf_cu_q[req_idx + 1].item()) if hasattr(ctx, 'prefill_cu_seqlens_q') else int(ctx.cu_seqlens_q[global_idx + 1].item())
                    prefill_request_ids[qs:qe] = req_idx
            else:
                cu_q = ctx.cu_seqlens_q
                for req_idx in range(num_prefill_seqs):
                    qs = int(cu_q[req_idx].item())
                    qe = int(cu_q[req_idx + 1].item())
                    prefill_request_ids[qs:qe] = req_idx

        topk_global = self.convert_indices(
            topk_indices, ctx.block_tables, block_size,
            req_ids=req_ids,
            prefill_request_ids=prefill_request_ids,
            prefill_workspace_starts=prefill_workspace_starts,
        )

        out = torch.empty(N, self.num_heads, self.kv_lora_rank,
                          dtype=q.dtype, device=q.device)

        if num_decode_tokens > 0:
            nd = num_decode_tokens
            q_dc = q[:nd]
            topk_dc = topk_global[:nd]
            num_decodes = num_decode_seqs

            q_dc_4d = q_dc.view(num_decodes, -1, self.num_heads, q.shape[-1])
            topk_dc_4d = topk_dc.view(num_decodes, -1, topk_dc.shape[-1])
            q_dc_4d, actual_heads = self._pad_q_for_fp8(q_dc_4d)
            padded_heads = q_dc_4d.shape[-2]
            decode_query_len = q_dc_4d.shape[1]

            topk = topk_dc.shape[-1]
            topk_tensor = torch.full(
                (num_decodes,), topk, dtype=torch.int32, device=q.device)
            dummy_bt = torch.empty(
                (num_decodes, 1), dtype=torch.int32, device=q.device)

            tile_sched_meta, _ = self.get_metadata(
                topk_tensor, decode_query_len * padded_heads,
                topk=topk, num_heads_q=padded_heads,
                num_heads_k=1, is_fp8_kvcache=True)

            o_dc, _ = self.decode_op(
                q_dc_4d, kv_cache.view(torch.uint8).unsqueeze(-2),
                dummy_bt, topk_tensor,
                head_dim_v=_MLA_HEAD_DIM_V,
                tile_scheduler_metadata=tile_sched_meta,
                softmax_scale=self.scale,
                is_fp8_kvcache=True,
                indices=topk_dc_4d,
            )
            o_dc = o_dc.view(-1, padded_heads, o_dc.shape[-1])
            if actual_heads < padded_heads:
                o_dc = o_dc[:, :actual_heads, :]
            out[:nd] = o_dc

        if num_prefill_tokens > 0:
            np_ = num_prefill_tokens
            q_pf = q[num_decode_tokens:] if ctx.is_mixed else q
            topk_pf = topk_global[num_decode_tokens:] if ctx.is_mixed else topk_global

            total_seq_len = int(pf_seq_lens.sum().item())
            workspace = torch.empty(total_seq_len, _MLA_WORKSPACE_HEAD_SIZE,
                                    dtype=torch.bfloat16, device=q.device)
            self.gather_kvcache(
                kv_cache, pf_bt, pf_seq_lens,
                prefill_workspace_starts, num_prefill_seqs, workspace,
            )

            workspace_kv = workspace.view(-1, 1, _MLA_WORKSPACE_HEAD_SIZE)

            prefill_padding = 64
            actual_h = q_pf.shape[1]
            q_pf_3d = q_pf
            if actual_h % prefill_padding != 0:
                pad_h = prefill_padding
                q_padded = q_pf_3d.new_empty(q_pf_3d.shape[0], pad_h, q_pf_3d.shape[2])
                q_padded[:, :actual_h, :] = q_pf_3d
                q_pf_3d = q_padded

            topk_pf_3d = topk_pf.view(np_, 1, -1)
            pf_out = self.sparse_prefill_op(
                q_pf_3d, workspace_kv, topk_pf_3d, self.scale)

            if isinstance(pf_out, tuple):
                pf_out = pf_out[0]
            pf_out = pf_out[:, :actual_h, :]

            if ctx.is_mixed:
                out[num_decode_tokens:] = pf_out
            else:
                out[:] = pf_out

        return self._v_up_proj(out.view(N, self.num_heads, self.kv_lora_rank))

    def _forward_mixed(self, q, kv_c_normed, k_pe, kv_b_proj, kv_cache, ctx):
        """Mixed batch for dense (non-sparse) attention."""
        np_ = ctx.num_prefill_tokens
        nd = ctx.num_decode_tokens
        out = torch.empty(np_ + nd, self.num_heads * self.v_head_dim,
                          dtype=q.dtype, device=q.device)

        if np_ > 0:
            q_pf = q[:np_]
            kv_c_pf = kv_c_normed[:np_]
            k_pe_pf = k_pe[:np_]

            kv = kv_b_proj(kv_c_pf)
            kv = kv.view(np_, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            k = torch.empty(np_, self.num_heads, self.qk_head_dim, dtype=q.dtype, device=q.device)
            k[..., :self.qk_nope_head_dim] = k_nope
            k[..., self.qk_nope_head_dim:] = k_pe_pf.expand(-1, self.num_heads, -1)

            pf_out = self.prefill_op(
                q_pf, k, v,
                cu_seqlens_q=ctx.prefill_cu_seqlens_q,
                cu_seqlens_k=ctx.prefill_cu_seqlens_k,
                max_seqlen_q=ctx.prefill_max_seqlen_q,
                max_seqlen_k=ctx.prefill_max_seqlen_k,
                softmax_scale=self.scale,
                causal=True,
            )
            out[:np_] = pf_out.view(np_, self.num_heads * self.v_head_dim)

        if nd > 0:
            q_dc = q[np_:]
            cache_seqlens = ctx.decode_context_lens
            block_table = ctx.decode_block_tables
            tile_sched_meta, _ = self.get_metadata(
                cache_seqlens, self.num_heads, num_heads_k=1,
                is_fp8_kvcache=True)

            o, _ = self.decode_op(
                q_dc, kv_cache.view(torch.uint8).unsqueeze(-2),
                block_table, cache_seqlens,
                head_dim_v=_MLA_HEAD_DIM_V,
                tile_scheduler_metadata=tile_sched_meta,
                softmax_scale=self.scale,
                is_fp8_kvcache=True,
            )
            out[np_:] = self._v_up_proj(o)

        return out
