"""DeepSeek Multi-head Latent Attention (MLA) with FlashMLA Sparse.

Uses FlashMLA sparse attention with FP8 656-byte KV cache format.
Decode uses flash_mla_with_kvcache (FP8 sparse decode kernel).
Prefill uses flash_mla_sparse_fwd (BF16 sparse prefill kernel).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ....infra.tp import _tp_size
from ....infra.context import get_context, get_attn_backend_config
from .parallel_linear import ColumnParallelLinear, RowParallelLinear
from ..L1.rms_norm import RMSNorm
from ..L1.linear import Linear
from ..L1.fp8_linear import Fp8Linear
from ..L1.store_kvcache_fp8_mla import StoreKVCacheFP8MLA

from flash_mla import get_mla_metadata, flash_mla_with_kvcache, flash_mla_sparse_fwd
from flash_mla.flash_mla_interface import FlashMLASchedMeta

BYTES_PER_TOKEN = 656


class ReplicatedLinear(nn.Module):
    """Linear layer replicated across TP ranks (no sharding)."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 quant_config: dict | None = None):
        super().__init__()
        self.use_fp8 = quant_config is not None
        if self.use_fp8:
            import math
            _FP8_BLOCK = 128
            self.weight = nn.Parameter(
                torch.empty(out_features, in_features, dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            self.weight_scale_inv = nn.Parameter(
                torch.empty(
                    math.ceil(out_features / _FP8_BLOCK),
                    math.ceil(in_features / _FP8_BLOCK),
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            self.weight.weight_loader = lambda p, w: p.data.copy_(w)
            self.weight_scale_inv.weight_loader = lambda p, w: p.data.copy_(w)
            self.linear_op = Fp8Linear()
        else:
            self.weight = nn.Parameter(torch.empty(out_features, in_features))
            self.weight.weight_loader = lambda p, w: p.data.copy_(w)
            self.linear_op = Linear()
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

    def forward(self, x):
        if self.use_fp8:
            return self.linear_op(x, self.weight, self.weight_scale_inv, self.bias)
        return self.linear_op(x, self.weight, self.bias)


class DeepSeekMLA(nn.Module):
    """Multi-head Latent Attention for DeepSeek V3.2 with FlashMLA Sparse."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        rms_norm_eps: float,
        rotary_emb: nn.Module,
        attn_scaling: float = 1.0,
        quant_config: dict | None = None,
    ):
        super().__init__()
        tp = _tp_size()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_local_heads = num_heads // tp
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.rotary_emb = rotary_emb

        self.scaling = (self.qk_head_dim ** -0.5) * attn_scaling
        self.kv_cache_head_dim = kv_lora_rank + qk_rope_head_dim

        if q_lora_rank is not None and q_lora_rank > 0:
            self.q_a_proj = ReplicatedLinear(
                hidden_size, q_lora_rank, bias=False, quant_config=quant_config,
            )
            self.q_a_layernorm = RMSNorm(q_lora_rank, eps=rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(
                q_lora_rank, num_heads * self.qk_head_dim,
                bias=False, quant_config=quant_config,
            )
        else:
            self.q_a_proj = None
            self.q_proj = ColumnParallelLinear(
                hidden_size, num_heads * self.qk_head_dim,
                bias=False, quant_config=quant_config,
            )

        self.kv_a_proj_with_mqa = ReplicatedLinear(
            hidden_size, kv_lora_rank + qk_rope_head_dim,
            bias=False, quant_config=quant_config,
        )
        self.kv_a_layernorm = RMSNorm(kv_lora_rank, eps=rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim),
            bias=False, quant_config=quant_config,
        )

        self.o_proj = RowParallelLinear(
            num_heads * v_head_dim, hidden_size,
            bias=False, quant_config=quant_config,
        )

        # FP8 KV cache (assigned by engine): [num_blocks, block_size, 656] uint8
        self.k_cache = torch.tensor([])
        self.v_cache = torch.tensor([])

        self._store_kvcache = StoreKVCacheFP8MLA()

        attn_cfg = get_attn_backend_config()
        self._block_size = attn_cfg.block_size

        self._w_uk = None
        self._w_uv = None

        self._sched_meta_cache = {}
        self._topk_indices_buffer = None

        self.indexer = None

        cc = torch.cuda.get_device_capability()
        self._prefill_head_padding = 128 if cc[0] >= 10 else 64

    def set_topk_indices_buffer(self, buf):
        self._topk_indices_buffer = buf

    def set_indexer(self, indexer):
        self.indexer = indexer

    def _extract_absorption_weights(self):
        """Extract W_UK and W_UV from kv_b_proj for decode-path absorption."""
        if self._w_uk is not None:
            return
        import math
        _B = 128
        w_raw = self.kv_b_proj.weight.data
        if w_raw.dtype == torch.float8_e4m3fn:
            scale = self.kv_b_proj.weight_scale_inv.data
            N, K = w_raw.shape
            sN, sK = math.ceil(N / _B), math.ceil(K / _B)
            pN, pK = sN * _B, sK * _B
            w_f = w_raw.to(torch.float32)
            if pN != N or pK != K:
                w_f = torch.nn.functional.pad(w_f, (0, pK - K, 0, pN - N))
            w_f = w_f.view(sN, _B, sK, _B) * scale[:sN, None, :sK, None]
            w_dequant = w_f.reshape(pN, pK)[:N, :K].to(torch.bfloat16)
        else:
            w_dequant = w_raw.to(torch.bfloat16)

        w = w_dequant.view(self.num_local_heads,
                           self.qk_nope_head_dim + self.v_head_dim,
                           self.kv_lora_rank)
        w_uk = w[:, :self.qk_nope_head_dim, :]
        w_uv = w[:, self.qk_nope_head_dim:, :].transpose(1, 2)
        self._w_uk = w_uk.contiguous()
        self._w_uv = w_uv.contiguous()
        del w_dequant, w

    def forward(self, positions, hidden_states):
        N = hidden_states.shape[0]
        ctx = get_context()

        if self.q_a_proj is not None:
            q_c = self.q_a_proj(hidden_states)
            q_c = self.q_a_layernorm(q_c)
            q = self.q_b_proj(q_c).view(N, self.num_local_heads, self.qk_head_dim)
        else:
            q = self.q_proj(hidden_states).view(N, self.num_local_heads, self.qk_head_dim)
            q_c = None

        q_nope, q_pe = q.split(
            [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1,
        )

        latent_cache = self.kv_a_proj_with_mqa(hidden_states)
        kv_a, k_pe = latent_cache.split(
            [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1,
        )
        kv_c_normed = self.kv_a_layernorm(kv_a)

        q_pe_flat = q_pe.reshape(N, self.num_local_heads * self.qk_rope_head_dim)
        k_pe_flat = k_pe
        q_pe_flat, k_pe_flat = self.rotary_emb(positions, q_pe_flat, k_pe_flat)
        q_pe = q_pe_flat.view(N, self.num_local_heads, self.qk_rope_head_dim)
        k_pe = k_pe_flat.view(N, self.qk_rope_head_dim)

        q[..., self.qk_nope_head_dim:] = q_pe

        kv_cache = self.k_cache
        if kv_cache.numel():
            self._store_kvcache(kv_c_normed, k_pe, kv_cache, ctx.slot_mapping)

        if self.indexer is not None and self._topk_indices_buffer is not None:
            q_c_for_idx = q_c if q_c is not None else self.q_a_layernorm(self.q_a_proj(hidden_states))
            self.indexer(hidden_states, q_c_for_idx, positions, self._topk_indices_buffer)

        self._extract_absorption_weights()
        ql_nope = torch.einsum('bhd,hdc->bhc', q_nope, self._w_uk)
        q_absorbed = torch.cat([ql_nope, q_pe], dim=-1)

        if ctx.is_mixed:
            attn_output = self._forward_mixed(q_absorbed, kv_c_normed, k_pe,
                                              kv_cache, ctx, N)
        elif ctx.is_prefill:
            attn_output = self._forward_prefill(q_absorbed, kv_c_normed, k_pe,
                                                kv_cache, ctx, N)
        else:
            attn_output = self._forward_decode(q_absorbed, kv_cache, ctx, N)

        return self.o_proj(attn_output)

    def _logical_to_physical(self, topk_indices, block_tables, block_size):
        """Convert logical token indices to physical cache slot indices.

        Args:
            topk_indices: [N, topk] int32, logical token positions (-1 = invalid)
            block_tables: [N, max_blocks_per_seq] int32
        Returns:
            physical_indices: [N, topk] int32
            where physical_index = physical_block_id * block_size + offset_in_block
        """
        valid_mask = topk_indices >= 0
        safe_indices = topk_indices.clone()
        safe_indices[~valid_mask] = 0

        page_idx = safe_indices // block_size
        offset_in_page = safe_indices % block_size

        max_pages = block_tables.shape[1]
        page_idx_clamped = page_idx.clamp(max=max_pages - 1)

        physical_blocks = torch.gather(
            block_tables, dim=1, index=page_idx_clamped.long(),
        )

        physical_indices = physical_blocks * block_size + offset_in_page
        physical_indices[~valid_mask] = -1
        return physical_indices.to(torch.int32)

    def _forward_decode(self, q_absorbed, kv_cache, ctx, N, idx_offset=0):
        """Decode using FlashMLA FP8 sparse decode kernel."""
        topk_indices = self._topk_indices_buffer[idx_offset:idx_offset + N]

        block_tables = ctx.decode_block_tables if ctx.decode_block_tables is not None else ctx.block_tables
        physical_indices = self._logical_to_physical(
            topk_indices, block_tables, self._block_size,
        )

        # q: (batch=N, seq_q=1, num_heads_q, head_dim=576)
        q_4d = q_absorbed.unsqueeze(1)
        # indices: (batch=N, seq_q=1, topk)
        indices_3d = physical_indices.unsqueeze(1)

        # k_cache: (num_blocks, block_size, num_heads_k=1, 656)
        kv_cache_view = kv_cache.unsqueeze(2)

        sched_key = (N, 1, self.num_local_heads, topk_indices.shape[1])
        if sched_key not in self._sched_meta_cache:
            self._sched_meta_cache[sched_key], _ = get_mla_metadata()

        out, lse = flash_mla_with_kvcache(
            q=q_4d,
            k_cache=kv_cache_view,
            block_table=None,
            cache_seqlens=None,
            head_dim_v=self.kv_lora_rank,
            tile_scheduler_metadata=self._sched_meta_cache[sched_key],
            softmax_scale=self.scaling,
            causal=False,
            is_fp8_kvcache=True,
            indices=indices_3d,
        )

        # out: (N, 1, num_heads_q, kv_lora_rank) -> squeeze seq dim
        o_latent = out.squeeze(1)
        o = torch.einsum('bhc,hcd->bhd', o_latent, self._w_uv)
        return o.reshape(N, self.num_local_heads * self.v_head_dim)

    def _dequant_cache_to_bf16(self, kv_cache, page_ids, total_tokens):
        """Gather FP8 cache pages and dequantize to BF16 [total_tokens, 576]."""
        block_size = self._block_size
        num_pages = page_ids.shape[0]

        raw = kv_cache[page_ids.long()]  # (num_pages, block_size, 656)
        flat = raw.reshape(num_pages * block_size, 656)[:total_tokens]

        nope_fp8 = flat[:, :512].contiguous().view(torch.float8_e4m3fn)
        scales = flat[:, 512:528].contiguous().view(torch.float32).reshape(total_tokens, 4)
        rope_bf16 = flat[:, 528:656].contiguous().view(torch.bfloat16).reshape(total_tokens, 64)

        nope_f32 = nope_fp8.to(torch.float32).view(total_tokens, 4, 128)
        nope_dequant = (nope_f32 * scales.unsqueeze(-1)).view(total_tokens, 512).to(torch.bfloat16)

        return torch.cat([nope_dequant, rope_bf16], dim=-1)

    def _forward_prefill(self, q_absorbed, kv_c_normed, k_pe, kv_cache, ctx, N):
        """Prefill using FlashMLA sparse prefill kernel (BF16).

        flash_mla_sparse_fwd expects:
            q: [s_q, h_q, d_qk] bf16
            kv: [s_kv, h_kv=1, d_qk] bf16
            indices: [s_q, h_kv=1, topk] int32 — global indices into kv
        """
        topk_indices = self._topk_indices_buffer[:N].clone()

        kv_bf16 = torch.cat([kv_c_normed, k_pe], dim=-1)

        cu_seqlens_k = ctx.prefill_cu_seqlens_k if ctx.prefill_cu_seqlens_k is not None else ctx.cu_seqlens_k
        cu_seqlens_q = ctx.prefill_cu_seqlens_q if ctx.prefill_cu_seqlens_q is not None else ctx.cu_seqlens_q
        if kv_cache.numel() and cu_seqlens_k is not None:
            cu_k = cu_seqlens_k.cpu().tolist()
            cu_q = cu_seqlens_q.cpu().tolist()
            num_seqs = len(cu_q) - 1
            all_kv = []
            kv_offset = 0

            for s in range(num_seqs):
                q_start, q_end = cu_q[s], cu_q[s + 1]
                q_len = q_end - q_start
                kv_len = cu_k[s + 1] - cu_k[s]
                cached_len = kv_len - q_len

                bt = ctx.prefill_block_tables if ctx.prefill_block_tables is not None else ctx.block_tables
                if cached_len > 0 and bt is not None:
                    num_pages = (cached_len + self._block_size - 1) // self._block_size
                    page_ids = bt[s, :num_pages]
                    cached_kv = self._dequant_cache_to_bf16(kv_cache, page_ids, cached_len)
                    all_kv.append(cached_kv)

                all_kv.append(kv_bf16[cu_q[s] - cu_q[0]:cu_q[s + 1] - cu_q[0]])

                valid_mask = topk_indices[q_start:q_end] >= 0
                topk_indices[q_start:q_end] = torch.where(
                    valid_mask,
                    topk_indices[q_start:q_end] + kv_offset,
                    topk_indices[q_start:q_end],
                )
                kv_offset += kv_len

            kv_flat = torch.cat(all_kv, dim=0)
        else:
            cu_q = (ctx.prefill_cu_seqlens_q if ctx.prefill_cu_seqlens_q is not None else ctx.cu_seqlens_q)
            if cu_q is not None:
                cu_q_list = cu_q.cpu().tolist()
                num_seqs = len(cu_q_list) - 1
                kv_offset = 0
                for s in range(num_seqs):
                    q_start, q_end = cu_q_list[s], cu_q_list[s + 1]
                    q_len = q_end - q_start
                    valid_mask = topk_indices[q_start:q_end] >= 0
                    topk_indices[q_start:q_end] = torch.where(
                        valid_mask,
                        topk_indices[q_start:q_end] + kv_offset,
                        topk_indices[q_start:q_end],
                    )
                    kv_offset += q_len
            kv_flat = kv_bf16

        kv_3d = kv_flat.unsqueeze(1)
        q_mqa = q_absorbed
        topk_3d = topk_indices.unsqueeze(1).to(torch.int32)

        out, max_logits, lse = flash_mla_sparse_fwd(
            q_mqa, kv_3d, topk_3d, self.scaling,
        )

        out = out[:, :self.num_local_heads, :self.kv_lora_rank]

        o = torch.einsum('bhc,hcd->bhd', out, self._w_uv)
        return o.reshape(N, self.num_local_heads * self.v_head_dim)

    def _forward_mixed(self, q_absorbed, kv_c_normed, k_pe, kv_cache, ctx, N):
        """Mixed prefill + decode batch."""
        np_ = ctx.num_prefill_tokens
        nd = ctx.num_decode_tokens
        out = torch.empty(N, self.num_local_heads * self.v_head_dim,
                          dtype=q_absorbed.dtype, device=q_absorbed.device)

        if np_ > 0:
            p_out = self._forward_prefill(
                q_absorbed[:np_], kv_c_normed[:np_], k_pe[:np_],
                kv_cache, ctx, np_,
            )
            out[:np_] = p_out

        if nd > 0:
            d_out = self._forward_decode(
                q_absorbed[np_:], kv_cache, ctx, nd, idx_offset=np_,
            )
            out[np_:] = d_out

        return out
