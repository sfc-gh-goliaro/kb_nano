"""Llama 4 attention with NoPE, QK norm, and temperature tuning.

Differences from standard Llama attention:
  - NoPE layers: skip RoPE, apply position-dependent temperature scaling to Q
  - RoPE layers: apply RoPE then weight-less QK RMSNorm (after RoPE, no learnable weight)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ....infra.context import get_context
from ....infra.tp import _tp_size
from .parallel_linear import QKVParallelLinear, RowParallelLinear
from ..L1.store_kvcache import StoreKVCache
from ..L1.flash_attn_prefill import FlashAttnPrefill
from ..L1.flash_attn_decode import FlashAttnDecode


class Llama4Attention(nn.Module):
    """Multi-head attention with optional NoPE, QK norm, and temperature tuning."""

    def __init__(self, hidden_size: int, num_attention_heads: int,
                 num_key_value_heads: int, head_dim: int,
                 nope: bool = False,
                 use_qk_norm: bool = False,
                 attn_temperature_tuning: bool = False,
                 floor_scale: float = 8192.0,
                 attn_scale: float = 0.1,
                 rms_norm_eps: float = 1e-5):
        super().__init__()
        tp = _tp_size()
        self.num_heads = num_attention_heads // tp
        self.num_kv_heads = num_key_value_heads // tp
        self.head_dim = head_dim
        self.scaling = head_dim ** -0.5
        self.nope = nope
        # QK norm on RoPE layers only; temp tuning on NoPE layers only
        self.use_qk_norm = use_qk_norm and not nope
        self.attn_temperature_tuning = attn_temperature_tuning and nope
        self.floor_scale = floor_scale
        self.attn_scale = attn_scale
        self.rms_norm_eps = rms_norm_eps

        self.qkv_proj = QKVParallelLinear(
            hidden_size, head_dim,
            num_attention_heads, num_key_value_heads,
        )
        self.o_proj = RowParallelLinear(
            num_attention_heads * head_dim, hidden_size,
        )

        self.k_cache = self.v_cache = torch.tensor([])
        self.store_kvcache = StoreKVCache()
        self.flash_attn_prefill = FlashAttnPrefill()
        self.flash_attn_decode = FlashAttnDecode()

    def _qk_rms_norm(self, x, n_heads):
        """Weight-less per-head RMSNorm in float32."""
        orig_dtype = x.dtype
        x = x.float().view(-1, self.head_dim)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.rms_norm_eps)
        return x.view(-1, n_heads * self.head_dim).to(orig_dtype)

    def _get_attn_scale(self, positions):
        """Position-dependent attention temperature scaling for NoPE layers."""
        floor = torch.floor((positions.float() + 1.0) / self.floor_scale)
        scale = torch.log(floor + 1.0) * self.attn_scale + 1.0
        return scale.unsqueeze(-1)

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

        # Apply RoPE only on non-NoPE layers
        if not self.nope:
            q, k = rotary_emb(positions, q, k)

        # Weight-less QK norm (after RoPE, only on RoPE layers)
        if self.use_qk_norm:
            q = q.view(N, -1)
            q = self._qk_rms_norm(q, self.num_heads)
            q = q.view(N, self.num_heads, self.head_dim)
            k = k.view(N, -1)
            k = self._qk_rms_norm(k, self.num_kv_heads)
            k = k.view(N, self.num_kv_heads, self.head_dim)

        # Temperature tuning (only on NoPE layers)
        if self.attn_temperature_tuning:
            scale = self._get_attn_scale(positions)
            q = (q.view(N, -1) * scale).to(q.dtype).view(N, self.num_heads, self.head_dim)

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
