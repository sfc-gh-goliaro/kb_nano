"""Model-level multi-head attention (thin wrapper).

Mirrors vLLM's ``LlamaAttention`` from
``vllm/model_executor/models/llama.py``:  QKV projection, RoPE, then
delegates to ``Attention`` for KV cache storage and kernel dispatch.
"""

from __future__ import annotations

import torch.nn as nn

from ....infra.tp import _tp_size
from .parallel_linear import QKVParallelLinear, RowParallelLinear
from .attention_impl import Attention


class LlamaAttention(nn.Module):
    """Model-level attention: qkv_proj -> rope -> Attention -> o_proj."""

    def __init__(self, hidden_size: int, num_attention_heads: int,
                 num_key_value_heads: int, head_dim: int,
                 rotary_emb: nn.Module | None = None):
        super().__init__()
        tp = _tp_size()
        self.num_heads = num_attention_heads // tp
        self.num_kv_heads = num_key_value_heads // tp
        self.head_dim = head_dim
        self.rotary_emb = rotary_emb

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

    def forward(self, positions, hidden_states):
        N = hidden_states.shape[0]
        qkv = self.qkv_proj(hidden_states)
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        return self.o_proj(attn_output)
