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
        """Sparse FP8 decode: pad heads, use dummy block_table/cache_seqlens."""
        N = q.shape[0]

        topk_indices = self.convert_indices(
            topk_indices, ctx.block_tables, block_size=int(kv_cache.shape[1]))

        q_4d = q.unsqueeze(0)
        topk_4d = topk_indices.unsqueeze(0)

        q_4d, actual_heads = self._pad_q_for_fp8(q_4d)
        padded_heads = q_4d.shape[-2]

        topk = topk_indices.shape[-1]
        topk_tensor = torch.full((1,), topk, dtype=torch.int32, device=q.device)
        dummy_bt = torch.empty((1, 1), dtype=torch.int32, device=q.device)

        tile_sched_meta, _ = self.get_metadata(
            topk_tensor, N * padded_heads,
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

        o = o.squeeze(0)
        if actual_heads < padded_heads:
            o = o[:, :actual_heads, :]
        return self._v_up_proj(o)

    def _forward_sparse_separate(self, q, kv_c_normed, k_pe, kv_b_proj,
                                 kv_cache, ctx, topk_indices,
                                 num_prefill_tokens, num_decode_tokens):
        """Separate prefill (BF16 workspace) and decode (FP8 kernel)."""
        N = q.shape[0]
        out = torch.empty(N, self.num_heads * self.v_head_dim,
                          dtype=q.dtype, device=q.device)

        if num_decode_tokens > 0:
            nd = num_decode_tokens
            q_dc = q[:nd] if ctx.is_mixed else q
            topk_dc = topk_indices[:nd] if ctx.is_mixed else topk_indices

            dc_ctx_lens = ctx.decode_context_lens if ctx.is_mixed else ctx.context_lens
            dc_bt = ctx.decode_block_tables if ctx.is_mixed else ctx.block_tables

            topk_dc_global = self.convert_indices(
                topk_dc, dc_bt, block_size=int(kv_cache.shape[1]))

            q_4d = q_dc.unsqueeze(0)
            topk_4d = topk_dc_global.unsqueeze(0)
            q_4d, actual_heads = self._pad_q_for_fp8(q_4d)
            padded_heads = q_4d.shape[-2]

            topk = topk_dc_global.shape[-1]
            topk_tensor = torch.full((1,), topk, dtype=torch.int32, device=q.device)
            dummy_bt = torch.empty((1, 1), dtype=torch.int32, device=q.device)

            tile_sched_meta, _ = self.get_metadata(
                topk_tensor, nd * padded_heads,
                topk=topk, num_heads_q=padded_heads,
                num_heads_k=1, is_fp8_kvcache=True)

            o_dc, _ = self.decode_op(
                q_4d, kv_cache.view(torch.uint8).unsqueeze(-2),
                dummy_bt, topk_tensor,
                head_dim_v=_MLA_HEAD_DIM_V,
                tile_scheduler_metadata=tile_sched_meta,
                softmax_scale=self.scale,
                is_fp8_kvcache=True,
                indices=topk_4d,
            )
            o_dc = o_dc.squeeze(0)
            if actual_heads < padded_heads:
                o_dc = o_dc[:, :actual_heads, :]
            v_proj_out = self._v_up_proj(o_dc)
            if ctx.is_mixed:
                out[:nd] = v_proj_out
            else:
                out[:] = v_proj_out

        if num_prefill_tokens > 0:
            np_ = num_prefill_tokens
            if ctx.is_mixed:
                q_pf = q[num_decode_tokens:]
                topk_pf = topk_indices[num_decode_tokens:]
                pf_bt = ctx.prefill_block_tables if hasattr(ctx, 'prefill_block_tables') else ctx.block_tables
                pf_cu = ctx.prefill_cu_seqlens_k if hasattr(ctx, 'prefill_cu_seqlens_k') else ctx.cu_seqlens_k
            else:
                q_pf = q
                topk_pf = topk_indices
                pf_bt = ctx.block_tables
                pf_cu = ctx.cu_seqlens_k

            head_size = kv_cache.shape[-1]
            num_seqs = pf_cu.shape[0] - 1
            total_seq_len = int(pf_cu[-1].item())

            workspace = torch.empty(total_seq_len, head_size,
                                    dtype=torch.bfloat16, device=q.device)
            self._gather_and_upconvert(kv_cache, workspace, pf_bt, pf_cu, num_seqs)

            workspace_kv = workspace.view(-1, 1, head_size)

            workspace_starts = torch.zeros(num_seqs, dtype=torch.int32, device=q.device)
            seq_lens_cpu = pf_cu.cpu()
            for i in range(1, num_seqs):
                workspace_starts[i] = int(seq_lens_cpu[i].item())

            topk_pf_ws = self._convert_prefill_indices_to_workspace(
                topk_pf, workspace_starts, pf_cu)

            prefill_padding = 64
            q_pf_3d = q_pf
            actual_h = q_pf_3d.shape[1]
            if actual_h % prefill_padding != 0:
                pad_h = prefill_padding
                q_padded = q_pf_3d.new_empty(q_pf_3d.shape[0], pad_h, q_pf_3d.shape[2])
                q_padded[:, :actual_h, :] = q_pf_3d
                q_pf_3d = q_padded

            topk_pf_ws_3d = topk_pf_ws.view(np_, 1, -1)
            pf_out = self.sparse_prefill_op(
                q_pf_3d, workspace_kv, topk_pf_ws_3d, self.scale)

            if isinstance(pf_out, tuple):
                pf_out = pf_out[0]
            pf_out = pf_out[:, :actual_h, :]

            pf_v_out = self._v_up_proj(pf_out)
            if ctx.is_mixed:
                out[num_decode_tokens:] = pf_v_out
            else:
                out[:] = pf_v_out

        return out

    def _gather_and_upconvert(self, kv_cache, workspace, block_table, cu_seq_lens, num_seqs):
        """Gather FP8 KV cache and upconvert to BF16 workspace."""
        block_size = kv_cache.shape[1]
        head_size = kv_cache.shape[2]
        cache_flat = kv_cache.view(-1, head_size)

        from ..L1.store_kvcache_fp8_mla import (
            _KV_C_FP8_BYTES, _KV_C_SCALE_BYTES, _K_PE_BYTES,
            _KV_C_DIM, _K_PE_DIM, _GROUP_SIZE, _NUM_GROUPS,
        )

        out_idx = 0
        for seq_i in range(num_seqs):
            seq_start = int(cu_seq_lens[seq_i].item())
            seq_end = int(cu_seq_lens[seq_i + 1].item())
            seq_len = seq_end - seq_start
            for t in range(seq_len):
                block_idx = t // block_size
                slot_in_block = t % block_size
                physical_block = int(block_table[seq_i, block_idx].item())
                slot = physical_block * block_size + slot_in_block
                raw = cache_flat[slot]

                for g in range(_NUM_GROUPS):
                    fp8_offset = g * _GROUP_SIZE
                    fp8_bytes = raw[fp8_offset:fp8_offset + _GROUP_SIZE]
                    fp8_vals = fp8_bytes.view(torch.float8_e4m3fn).float()
                    scale_offset = _KV_C_FP8_BYTES + g * 4
                    scale = raw[scale_offset:scale_offset + 4].view(torch.float32)
                    workspace[out_idx, fp8_offset:fp8_offset + _GROUP_SIZE] = (
                        fp8_vals * scale).to(torch.bfloat16).view(torch.uint8)

                pe_offset = _KV_C_FP8_BYTES + _KV_C_SCALE_BYTES
                workspace[out_idx, pe_offset:pe_offset + _K_PE_BYTES] = raw[pe_offset:pe_offset + _K_PE_BYTES]
                out_idx += 1

    def _convert_prefill_indices_to_workspace(self, topk_indices, workspace_starts, cu_seq_lens):
        """Convert per-request logical indices to workspace offsets for prefill."""
        M = topk_indices.shape[0]
        out = topk_indices.clone()
        num_seqs = cu_seq_lens.shape[0] - 1
        for seq_i in range(num_seqs):
            seq_start = int(cu_seq_lens[seq_i].item())
            seq_end = int(cu_seq_lens[seq_i + 1].item())
            ws_start = int(workspace_starts[seq_i].item())
            for row in range(seq_start, min(seq_end, M)):
                valid = out[row] >= 0
                out[row] = torch.where(valid, out[row] + ws_start, out[row])
        return out

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
