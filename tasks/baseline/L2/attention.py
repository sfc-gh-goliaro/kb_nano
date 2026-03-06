"""Multi-head attention with GQA, paged KV cache, and selectable backend.

Uses FlashInfer on Blackwell/Hopper (sm>=90) for paged decode/prefill,
falls back to flash_attn on older GPUs.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ....infra.context import get_context
from ....infra.tp import _tp_size
from .parallel_linear import QKVParallelLinear, RowParallelLinear
from ..L1.store_kvcache import StoreKVCache


def _use_flashinfer() -> bool:
    from ....engine import USE_FLASHINFER
    return USE_FLASHINFER


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

        self.store_kvcache = StoreKVCache()

        self._flashinfer = _use_flashinfer()
        if self._flashinfer:
            from ..L1.flashinfer_prefill import FlashInferPrefill
            from ..L1.flashinfer_decode import FlashInferDecode
            from ....engine import BLOCK_SIZE
            self.fi_prefill = FlashInferPrefill(
                self.num_heads, self.num_kv_heads, head_dim, BLOCK_SIZE,
            )
            self.fi_decode = FlashInferDecode(
                self.num_heads, self.num_kv_heads, head_dim, BLOCK_SIZE,
            )
        else:
            from ..L1.flash_attn_prefill import FlashAttnPrefill
            from ..L1.flash_attn_decode import FlashAttnDecode
            self.flash_attn_prefill = FlashAttnPrefill()
            self.flash_attn_decode = FlashAttnDecode()

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

        if self._flashinfer:
            o = self._forward_flashinfer(q, k, v, k_cache, v_cache, ctx)
        else:
            o = self._forward_flash_attn(q, k, v, k_cache, v_cache, ctx)
        return self.o_proj(o.reshape(N, self.num_heads * self.head_dim))

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
        )

    def _forward_flashinfer(self, q, k, v, k_cache, v_cache, ctx):
        if ctx.is_prefill:
            if ctx.fi_planned:
                return self.fi_prefill(q, k_cache, v_cache)
            # Warmup (no block tables yet) — use flash_attn_varlen_func
            from flash_attn import flash_attn_varlen_func
            return flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=ctx.cu_seqlens_q, cu_seqlens_k=ctx.cu_seqlens_k,
                max_seqlen_q=ctx.max_seqlen_q, max_seqlen_k=ctx.max_seqlen_k,
                softmax_scale=self.scaling, causal=True,
            )
        # Decode: wrapper.run expects q shape [batch_size, num_heads, head_dim]
        return self.fi_decode(q, k_cache, v_cache)
