"""Model-level multi-head attention (thin wrapper).

Mirrors vLLM's ``LlamaAttention`` / ``Qwen3Attention`` from
``vllm/model_executor/models/llama.py`` / ``qwen3.py``:
QKV projection, optional QK-norm, RoPE, then delegates to ``Attention``
for KV cache storage and kernel dispatch.

Unified across Llama, Qwen2, and Qwen3 architectures:
  - bias:    Qwen2 uses bias=True on QKV projection; others use False.
  - qk_norm: Qwen3 applies per-head RMSNorm to Q and K before RoPE.
"""

from __future__ import annotations

import torch.nn as nn

from ....infra.tp import _tp_size
from .parallel_linear import QKVParallelLinear, RowParallelLinear
from .attention_impl import Attention
from ..L1.rms_norm import RMSNorm


class LlamaAttention(nn.Module):
    """Model-level attention: qkv_proj -> [qk_norm] -> rope -> Attention -> o_proj."""

    def __init__(self, hidden_size: int, num_attention_heads: int,
                 num_key_value_heads: int, head_dim: int,
                 rotary_emb: nn.Module | None = None,
                 bias: bool = False,
                 qk_norm: bool = False,
                 rms_norm_eps: float = 1e-6,
                 quant_config: dict | None = None):
        super().__init__()
        tp = _tp_size()
        self.num_heads = num_attention_heads // tp
        self.num_kv_heads = num_key_value_heads // tp
        self.head_dim = head_dim
        self.rotary_emb = rotary_emb

        self.qkv_proj = QKVParallelLinear(
            hidden_size, head_dim,
            num_attention_heads, num_key_value_heads,
            bias=bias,
            quant_config=quant_config,
        )
        self.o_proj = RowParallelLinear(
            num_attention_heads * head_dim, hidden_size,
            quant_config=quant_config,
        )

        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps) if qk_norm else None
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps) if qk_norm else None

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

        if self.q_norm is not None:
            q = q.view(N, self.num_heads, self.head_dim)
            k = k.view(N, self.num_kv_heads, self.head_dim)
            q = self.q_norm(q.reshape(-1, self.head_dim)).view(N, self.num_heads * self.head_dim)
            k = self.k_norm(k.reshape(-1, self.head_dim)).view(N, self.num_kv_heads * self.head_dim)

        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        return self.o_proj(attn_output)
