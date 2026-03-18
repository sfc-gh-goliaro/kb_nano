"""Llama 4 attention with NoPE, QK norm, and temperature tuning.

Mirrors vLLM's ``Llama4Attention`` from
``vllm/model_executor/models/llama4.py``:
separate class from ``LlamaAttention`` because of NoPE layers,
weight-less QK norm, and temperature tuning, but delegates to the
same shared ``Attention`` backend for KV cache and kernel dispatch.

Differences from standard Llama attention:
  - NoPE layers: skip RoPE, apply position-dependent temperature scaling to Q
  - RoPE layers: apply RoPE then weight-less QK RMSNorm (after RoPE, no learnable weight)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ....infra.tp import _tp_size
from .parallel_linear import QKVParallelLinear, RowParallelLinear
from .attention_impl import Attention


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
        self.nope = nope
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

        self.attn = Attention(
            self.num_heads, head_dim, head_dim ** -0.5,
            num_kv_heads=self.num_kv_heads,
        )

    def _qk_rms_norm(self, x):
        """Weight-less per-head RMSNorm in float32."""
        orig_dtype = x.dtype
        x = x.float().view(-1, self.head_dim)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.rms_norm_eps)
        return x.view(x.shape[0], -1).to(orig_dtype)

    def _get_attn_scale(self, positions):
        """Position-dependent attention temperature scaling for NoPE layers."""
        floor = torch.floor((positions.float() + 1.0) / self.floor_scale)
        scale = torch.log(floor + 1.0) * self.attn_scale + 1.0
        return scale.unsqueeze(-1)

    def forward(self, positions, hidden_states, rotary_emb):
        N = hidden_states.shape[0]
        qkv = self.qkv_proj(hidden_states)
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)

        if not self.nope:
            q, k = rotary_emb(positions, q, k)

        if self.use_qk_norm:
            q = self._qk_rms_norm(q)
            k = self._qk_rms_norm(k)

        if self.attn_temperature_tuning:
            q = (q * self._get_attn_scale(positions)).to(q.dtype)

        attn_output = self.attn(q, k, v)
        return self.o_proj(attn_output)
