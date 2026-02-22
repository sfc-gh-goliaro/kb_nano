"""Qwen3 attention with QK-norm and M-RoPE support."""

from __future__ import annotations

import torch
import torch.nn as nn

from ...infra.context import get_context
from ...infra.tp import _tp_size
from .parallel_linear import QKVParallelLinear, RowParallelLinear
from ..L1.rms_norm import RMSNorm
from ..L1.store_kvcache import StoreKVCache
from ..L1.flash_attn_prefill import FlashAttnPrefill
from ..L1.flash_attn_decode import FlashAttnDecode


class Qwen3Attention(nn.Module):
    """Multi-head attention for Qwen3 language model.

    Key differences from LlamaAttention:
    - Per-head QK-norm (RMSNorm on Q and K before RoPE)
    - Uses M-RoPE (3D positions) via MRotaryEmbedding
    - No QKV bias
    """

    def __init__(self, hidden_size: int, num_attention_heads: int,
                 num_key_value_heads: int, head_dim: int,
                 rms_norm_eps: float = 1e-6):
        super().__init__()
        tp = _tp_size()
        self.num_heads = num_attention_heads // tp
        self.num_kv_heads = num_key_value_heads // tp
        self.head_dim = head_dim
        self.scaling = head_dim ** -0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size, head_dim,
            num_attention_heads, num_key_value_heads,
            bias=False,
        )
        self.o_proj = RowParallelLinear(
            num_attention_heads * head_dim, hidden_size,
        )

        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)

        self.k_cache = self.v_cache = torch.tensor([])

        self.store_kvcache = StoreKVCache()
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

        q = self.q_norm(q.reshape(-1, self.head_dim)).view(N, self.num_heads, self.head_dim)
        k = self.k_norm(k.reshape(-1, self.head_dim)).view(N, self.num_kv_heads, self.head_dim)

        q, k = rotary_emb(positions, q, k)

        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            self.store_kvcache(k, v, k_cache, v_cache, ctx.slot_mapping)

        if ctx.is_prefill:
            if ctx.block_tables is not None:
                o = self.flash_attn_prefill(
                    q, k_cache, v_cache,
                    cu_seqlens_q=ctx.cu_seqlens_q,
                    cu_seqlens_k=ctx.cu_seqlens_k,
                    max_seqlen_q=ctx.max_seqlen_q,
                    max_seqlen_k=ctx.max_seqlen_k,
                    softmax_scale=self.scaling, causal=True,
                    block_table=ctx.block_tables,
                )
            else:
                o = self.flash_attn_prefill(
                    q, k, v,
                    cu_seqlens_q=ctx.cu_seqlens_q, cu_seqlens_k=ctx.cu_seqlens_k,
                    max_seqlen_q=ctx.max_seqlen_q, max_seqlen_k=ctx.max_seqlen_k,
                    softmax_scale=self.scaling, causal=True,
                )
        else:
            o = self.flash_attn_decode(
                q.unsqueeze(1), k_cache, v_cache,
                cache_seqlens=ctx.context_lens, block_table=ctx.block_tables,
                softmax_scale=self.scaling, causal=True,
            )
        return self.o_proj(o.reshape(N, self.num_heads * self.head_dim))
