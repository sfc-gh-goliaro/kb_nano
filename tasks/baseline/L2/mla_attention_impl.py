"""MLA attention implementation with FP8 paged KV cache.

MLA equivalent of attention_impl.py's Attention class. Handles:
- FP8 KV cache storage (656 bytes/token)
- Dense prefill via FlashMLAPrefill
- Dense/sparse decode via FlashMLADecode
- Sparse prefill via FlashMLASparsePrefill
- Mixed batch (prefill + decode)
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
        self._head_dim = 656  # Total bytes per token in FP8 MLA cache

        # Both point to same tensor (engine discovers via hasattr)
        self.k_cache = self.v_cache = torch.tensor([])

        self.store_kvcache = StoreKVCacheFP8MLA()
        self.gather_kvcache = GatherKVCacheFP8MLA()  # for sparse prefill / gather paths
        self.decode_op = FlashMLADecode()
        self.prefill_op = FlashMLAPrefill()
        self.sparse_prefill_op = FlashMLASparsePrefill()
        self.get_metadata = FlashMLAGetMetadata()
        self.convert_indices = ConvertIndicesToGlobal()

    def forward(self, q: torch.Tensor, kv_c_normed: torch.Tensor,
                k_pe: torch.Tensor, kv_b_proj: nn.Module,
                topk_indices: torch.Tensor | None = None,
                output_shape: tuple | None = None) -> torch.Tensor:
        """
        Args:
            q: [N, num_heads, qk_head_dim] query after RoPE
            kv_c_normed: [N, kv_lora_rank] compressed KV after layernorm
            k_pe: [N, 1, qk_rope_head_dim] RoPE key component
            kv_b_proj: ColumnParallelLinear for expanding compressed KV
            topk_indices: [N, topk_tokens] sparse attention indices (DSA)
            output_shape: desired output shape
        """
        ctx = get_context()
        N = q.shape[0]

        kv_cache = self.k_cache  # [num_blocks, block_size, 656] uint8

        # Store to cache
        if kv_cache.numel() and ctx.slot_mapping is not None:
            self.store_kvcache(kv_c_normed, k_pe, kv_cache, ctx.slot_mapping)

        if ctx.is_mixed:
            o = self._forward_mixed(q, kv_c_normed, k_pe, kv_b_proj, kv_cache, ctx, topk_indices)
        else:
            o = self._forward_pure(q, kv_c_normed, k_pe, kv_b_proj, kv_cache, ctx, topk_indices)

        if output_shape is not None:
            o = o.view(*output_shape)
        return o

    def _forward_pure(self, q, kv_c_normed, k_pe, kv_b_proj, kv_cache, ctx, topk_indices):
        if ctx.is_prefill:
            return self._forward_prefill(q, kv_c_normed, k_pe, kv_b_proj, kv_cache, ctx, topk_indices)
        return self._forward_decode(q, kv_cache, ctx, kv_b_proj, topk_indices)

    def _forward_prefill(self, q, kv_c_normed, k_pe, kv_b_proj, kv_cache, ctx, topk_indices):
        """Dense prefill: expand KV, run FlashAttn varlen."""
        N = q.shape[0]

        # Expand compressed KV to full K, V
        kv = kv_b_proj(kv_c_normed)  # [N, num_heads * (qk_nope_head_dim + v_head_dim)]
        kv = kv.view(N, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # Build full key: [k_nope, k_pe_expanded]
        k = torch.empty(N, self.num_heads, self.qk_head_dim, dtype=q.dtype, device=q.device)
        k[..., :self.qk_nope_head_dim] = k_nope
        k[..., self.qk_nope_head_dim:] = k_pe.expand(-1, self.num_heads, -1)

        if topk_indices is not None and self.is_sparse:
            raise NotImplementedError(
                "Sparse MLA prefill with topk_indices is not implemented yet; "
                "use FlashMLASparsePrefill with gathered KV when ready."
            )

        # Dense prefill
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

    def _forward_decode(self, q, kv_cache, ctx, kv_b_proj, topk_indices):
        """Decode against FP8 KV cache via FlashMLA."""
        N = q.shape[0]
        cache_seqlens = ctx.context_lens
        block_table = ctx.block_tables

        # Get FlashMLA metadata
        tile_sched_meta, _ = self.get_metadata(cache_seqlens, self.num_heads)

        # FlashMLA decode
        if topk_indices is not None and self.is_sparse:
            # Sparse decode with indices
            o, _ = self.decode_op(
                q, kv_cache.view(torch.uint8).unsqueeze(-2),
                block_table, cache_seqlens,
                head_dim_v=self.v_head_dim,
                tile_scheduler_metadata=tile_sched_meta,
                softmax_scale=self.scale,
                is_fp8_kvcache=True,
                indices=topk_indices,
            )
        else:
            # Dense decode
            o, _ = self.decode_op(
                q, kv_cache.view(torch.uint8).unsqueeze(-2),
                block_table, cache_seqlens,
                head_dim_v=self.v_head_dim,
                tile_scheduler_metadata=tile_sched_meta,
                softmax_scale=self.scale,
                is_fp8_kvcache=True,
            )

        return o.view(N, self.num_heads * self.v_head_dim)

    def _forward_mixed(self, q, kv_c_normed, k_pe, kv_b_proj, kv_cache, ctx, topk_indices):
        """Mixed batch: split into prefill and decode portions."""
        np_ = ctx.num_prefill_tokens
        nd = ctx.num_decode_tokens
        out = torch.empty(np_ + nd, self.num_heads * self.v_head_dim,
                          dtype=q.dtype, device=q.device)

        if np_ > 0:
            q_pf = q[:np_]
            kv_c_pf = kv_c_normed[:np_]
            k_pe_pf = k_pe[:np_]
            topk_pf = topk_indices[:np_] if topk_indices is not None else None

            # Expand KV for prefill
            kv = kv_b_proj(kv_c_pf)
            kv = kv.view(np_, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            k = torch.empty(np_, self.num_heads, self.qk_head_dim, dtype=q.dtype, device=q.device)
            k[..., :self.qk_nope_head_dim] = k_nope
            k[..., self.qk_nope_head_dim:] = k_pe_pf.expand(-1, self.num_heads, -1)

            if topk_pf is not None and self.is_sparse:
                raise NotImplementedError("Sparse MLA prefill in mixed batches is not implemented yet.")

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
            topk_dc = topk_indices[np_:] if topk_indices is not None else None

            cache_seqlens = ctx.decode_context_lens
            block_table = ctx.decode_block_tables
            tile_sched_meta, _ = self.get_metadata(cache_seqlens, self.num_heads)

            if topk_dc is not None and self.is_sparse:
                o, _ = self.decode_op(
                    q_dc, kv_cache.view(torch.uint8).unsqueeze(-2),
                    block_table, cache_seqlens,
                    head_dim_v=self.v_head_dim,
                    tile_scheduler_metadata=tile_sched_meta,
                    softmax_scale=self.scale,
                    is_fp8_kvcache=True,
                    indices=topk_dc,
                )
            else:
                o, _ = self.decode_op(
                    q_dc, kv_cache.view(torch.uint8).unsqueeze(-2),
                    block_table, cache_seqlens,
                    head_dim_v=self.v_head_dim,
                    tile_scheduler_metadata=tile_sched_meta,
                    softmax_scale=self.scale,
                    is_fp8_kvcache=True,
                )
            out[np_:] = o.view(nd, self.num_heads * self.v_head_dim)

        return out
