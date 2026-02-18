"""Multi-head attention with GQA, flash_attn, and paged KV cache."""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

from .context import get_context
from .tp import QKVParallelLinear, RowParallelLinear, _tp_size


# ---------------------------------------------------------------------------
# Triton KV cache store kernel
# ---------------------------------------------------------------------------
@triton.jit
def _store_kvcache_kernel(
    key_ptr, key_stride, value_ptr, value_stride,
    k_cache_ptr, v_cache_ptr, slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key, value, k_cache, v_cache, slot_mapping):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    _store_kvcache_kernel[(N,)](
        key, key.stride(0), value, value.stride(0),
        k_cache, v_cache, slot_mapping, D,
    )


# ---------------------------------------------------------------------------
# Attention block (shared by Llama and Mixtral)
# ---------------------------------------------------------------------------
class Attention(nn.Module):
    """Multi-head attention with GQA, flash_attn, and paged KV cache."""

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
            store_kvcache(k, v, k_cache, v_cache, ctx.slot_mapping)

        if ctx.is_prefill:
            o = flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=ctx.cu_seqlens_q, cu_seqlens_k=ctx.cu_seqlens_k,
                max_seqlen_q=ctx.max_seqlen_q, max_seqlen_k=ctx.max_seqlen_k,
                softmax_scale=self.scaling, causal=True,
                block_table=ctx.block_tables,
            )
        else:
            o = flash_attn_with_kvcache(
                q.unsqueeze(1), k_cache, v_cache,
                cache_seqlens=ctx.context_lens, block_table=ctx.block_tables,
                softmax_scale=self.scaling, causal=True,
            )
        return self.o_proj(o.reshape(N, self.num_heads * self.head_dim))
