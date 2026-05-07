"""GPT-OSS attention: GQA with bias, YaRN RoPE, sliding window, and attention sinks.

Attention sinks are virtual attention drains: they add exp(sink) to the
softmax denominator without contributing any value to the output.

Delegates to the shared Attention backend for KV cache and kernel dispatch.
Sliding window and sinks are handled by the backend's SDPA path.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ....infra.tp import _tp_size
from .parallel_linear import QKVParallelLinear, RowParallelLinear
from .attention_impl import Attention


class GptOssAttention(nn.Module):
    """GQA attention with bias, sliding window, and attention sinks."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        layer_idx: int,
        sliding_window: int | None = None,
    ):
        super().__init__()
        tp = _tp_size()
        self.num_heads = num_attention_heads // tp
        self.num_kv_heads = num_key_value_heads // tp
        self.head_dim = head_dim
        self.layer_idx = layer_idx

        per_layer_sw = sliding_window if layer_idx % 2 == 0 else None

        self.qkv_proj = QKVParallelLinear(
            hidden_size, head_dim,
            num_attention_heads, num_key_value_heads,
            bias=True,
        )
        self.o_proj = RowParallelLinear(
            num_attention_heads * head_dim, hidden_size,
            bias=True,
        )

        self.sinks = nn.Parameter(torch.zeros(self.num_heads))
        self.sinks.weight_loader = self._sinks_weight_loader

        self.attn = Attention(
            self.num_heads, head_dim, head_dim ** -0.5,
            num_kv_heads=self.num_kv_heads,
            sliding_window=per_layer_sw,
            sinks=self.sinks,
        )

    def _sinks_weight_loader(self, param, loaded_weight):
        from ....infra.tp import _tp_rank
        tp, rank = _tp_size(), _tp_rank()
        heads_per_rank = param.data.size(0)
        start = rank * heads_per_rank
        param.data.copy_(loaded_weight.narrow(0, start, heads_per_rank))

    def forward(self, positions, hidden_states, rotary_emb):
        N = hidden_states.shape[0]
        qkv = self.qkv_proj(hidden_states)
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)

        q, k = rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        return self.o_proj(attn_output)
