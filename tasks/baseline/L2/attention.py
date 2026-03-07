"""Multi-head attention with GQA, paged KV cache, and selectable backend.

Blackwell (sm_100+): TRTLLM-gen kernels via FlashInfer package.
Hopper and below:    flash_attn (unchanged).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ....infra.context import get_context
from ....infra.tp import _tp_size
from .parallel_linear import QKVParallelLinear, RowParallelLinear
from ..L1.store_kvcache import StoreKVCache, StoreKVCacheHND


def _use_trtllm() -> bool:
    from ....engine import USE_TRTLLM
    return USE_TRTLLM


def _block_size() -> int:
    from ....engine import BLOCK_SIZE
    return BLOCK_SIZE


class Attention(nn.Module):
    """Multi-head attention with GQA and paged KV cache."""

    def __init__(self, hidden_size: int, num_attention_heads: int,
                 num_key_value_heads: int, head_dim: int):
        super().__init__()
        tp = _tp_size()
        self.num_heads = num_attention_heads // tp
        self.num_kv_heads = num_key_value_heads // tp
        self.head_dim = head_dim
        self.scaling = head_dim ** -0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size, head_dim,
            num_attention_heads, num_key_value_heads,
        )
        self.o_proj = RowParallelLinear(
            num_attention_heads * head_dim, hidden_size,
        )
        self.k_cache = self.v_cache = torch.tensor([])

        self._use_trtllm = _use_trtllm()
        if self._use_trtllm:
            self.store_kvcache = StoreKVCacheHND(page_size=_block_size())
            from ..L1.flashinfer_prefill import TRTLLMPrefill
            from ..L1.flashinfer_decode import TRTLLMDecode
            self.trtllm_prefill = TRTLLMPrefill(
                self.num_heads, self.num_kv_heads, head_dim,
            )
            self.trtllm_decode = TRTLLMDecode(
                self.num_heads, self.num_kv_heads, head_dim,
            )
        else:
            self.store_kvcache = StoreKVCache()
            from ..L1.flash_attn_prefill import FlashAttnPrefill
            from ..L1.flash_attn_decode import FlashAttnDecode
            self.flash_attn_prefill = FlashAttnPrefill()
            self.flash_attn_decode = FlashAttnDecode()

    def set_trtllm_workspace(self, workspace: torch.Tensor):
        """Share a single workspace buffer across all layers (TRTLLM path only)."""
        if self._use_trtllm:
            self.trtllm_decode._workspace = workspace
            self.trtllm_prefill._workspace = workspace

    def forward(self, positions, hidden_states, rotary_emb):
        ctx = get_context()
        N = hidden_states.shape[0]
        qkv = self.qkv_proj(hidden_states)
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
        q = q.view(N, self.num_heads, self.head_dim)
        k = k.view(N, self.num_kv_heads, self.head_dim)
        v = v.view(N, self.num_kv_heads, self.head_dim)
        q, k = rotary_emb(positions, q, k)

        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            self.store_kvcache(k, v, k_cache, v_cache, ctx.slot_mapping)

        if ctx.is_mixed:
            if self._use_trtllm:
                o = self._forward_trtllm_mixed(q, k_cache, v_cache, ctx)
            else:
                o = self._forward_flash_attn_mixed(q, k_cache, v_cache, ctx)
        elif self._use_trtllm:
            o = self._forward_trtllm(q, k, v, k_cache, v_cache, ctx)
        else:
            o = self._forward_flash_attn(q, k, v, k_cache, v_cache, ctx)
        return self.o_proj(o.reshape(N, self.num_heads * self.head_dim))

    # --- Pure prefill / decode paths ---

    def _forward_flash_attn(self, q, k, v, k_cache, v_cache, ctx):
        if ctx.is_prefill:
            if ctx.block_tables is not None:
                return self.flash_attn_prefill(
                    q, k_cache, v_cache,
                    cu_seqlens_q=ctx.cu_seqlens_q,
                    cu_seqlens_k=ctx.cu_seqlens_k,
                    max_seqlen_q=ctx.max_seqlen_q,
                    max_seqlen_k=ctx.max_seqlen_k,
                    softmax_scale=self.scaling, causal=True,
                    block_table=ctx.block_tables,
                )
            return self.flash_attn_prefill(
                q, k, v,
                cu_seqlens_q=ctx.cu_seqlens_q, cu_seqlens_k=ctx.cu_seqlens_k,
                max_seqlen_q=ctx.max_seqlen_q, max_seqlen_k=ctx.max_seqlen_k,
                softmax_scale=self.scaling, causal=True,
            )
        return self.flash_attn_decode(
            q.unsqueeze(1), k_cache, v_cache,
            cache_seqlens=ctx.context_lens, block_table=ctx.block_tables,
            softmax_scale=self.scaling, causal=True,
        ).squeeze(1)

    def _forward_trtllm(self, q, k, v, k_cache, v_cache, ctx):
        if ctx.is_prefill:
            if ctx.block_tables is not None:
                seq_lens = ctx.cu_seqlens_k[1:] - ctx.cu_seqlens_k[:-1]
                batch_size = seq_lens.shape[0]
                return self.trtllm_prefill(
                    q, k_cache, v_cache,
                    block_tables=ctx.block_tables,
                    seq_lens=seq_lens,
                    max_q_len=ctx.max_seqlen_q,
                    max_kv_len=ctx.max_seqlen_k,
                    batch_size=batch_size,
                    cum_seq_lens_q=ctx.cu_seqlens_q,
                    cum_seq_lens_kv=ctx.cu_seqlens_k,
                )
            from flash_attn import flash_attn_varlen_func
            return flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=ctx.cu_seqlens_q, cu_seqlens_k=ctx.cu_seqlens_k,
                max_seqlen_q=ctx.max_seqlen_q, max_seqlen_k=ctx.max_seqlen_k,
                softmax_scale=self.scaling, causal=True,
            )
        return self.trtllm_decode(
            q, k_cache, v_cache,
            cache_seqlens=ctx.context_lens,
            block_table=ctx.block_tables,
            max_seq_len=ctx.max_context_len,
        )

    # --- Mixed batch paths (chunked prefill) ---

    def _forward_flash_attn_mixed(self, q, k_cache, v_cache, ctx):
        np = ctx.num_prefill_tokens
        nd = ctx.num_decode_tokens
        out = torch.empty_like(q)

        if np > 0:
            out[:np] = self.flash_attn_prefill(
                q[:np], k_cache, v_cache,
                cu_seqlens_q=ctx.prefill_cu_seqlens_q,
                cu_seqlens_k=ctx.prefill_cu_seqlens_k,
                max_seqlen_q=ctx.prefill_max_seqlen_q,
                max_seqlen_k=ctx.prefill_max_seqlen_k,
                softmax_scale=self.scaling, causal=True,
                block_table=ctx.prefill_block_tables,
            )

        if nd > 0:
            out[np:] = self.flash_attn_decode(
                q[np:].unsqueeze(1), k_cache, v_cache,
                cache_seqlens=ctx.decode_context_lens,
                block_table=ctx.decode_block_tables,
                softmax_scale=self.scaling, causal=True,
            ).squeeze(1)
        return out

    def _forward_trtllm_mixed(self, q, k_cache, v_cache, ctx):
        np_ = ctx.num_prefill_tokens
        nd = ctx.num_decode_tokens
        out = torch.empty_like(q)

        if np_ > 0:
            pq = q[:np_].contiguous()
            seq_lens = (ctx.prefill_cu_seqlens_k[1:]
                        - ctx.prefill_cu_seqlens_k[:-1])
            out[:np_] = self.trtllm_prefill(
                pq, k_cache, v_cache,
                block_tables=ctx.prefill_block_tables,
                seq_lens=seq_lens,
                max_q_len=ctx.prefill_max_seqlen_q,
                max_kv_len=ctx.prefill_max_seqlen_k,
                batch_size=ctx.num_prefill_seqs,
                cum_seq_lens_q=ctx.prefill_cu_seqlens_q,
                cum_seq_lens_kv=ctx.prefill_cu_seqlens_k,
            )

        if nd > 0:
            out[np_:] = self.trtllm_decode(
                q[np_:], k_cache, v_cache,
                cache_seqlens=ctx.decode_context_lens,
                block_table=ctx.decode_block_tables,
                max_seq_len=ctx.decode_max_context_len,
            )
        return out
