"""Qwen3-Next full attention with per-head QK-norm, partial RoPE, output gating, KV cache (L2).

GQA attention: 16 query heads, 2 KV heads, head_dim=256.
Q projection outputs 2x: [Q, gate] interleaved per head.
Partial RoPE (25% of head_dim = 64 dims rotated).
Output: attn_output * sigmoid(gate).

KV cache is stored in ``layer_state`` dict for decode steps (Qwen3-Next's
hybrid decoder threads per-sequence state through both linear and full
attention layers, so we don't use the engine's paged KV pool here).

Composes only L1 ops (``GemmaRMSNorm``, ``FlashAttnDense``) and the
canonical TP linears in ``parallel_linear``; no direct ``flash_attn``
imports.

Weight names match HuggingFace checkpoint:
  self_attn.q_proj.weight   [2 * num_heads * head_dim, hidden_size]  (Q + gate)
  self_attn.k_proj.weight   [num_kv_heads * head_dim, hidden_size]
  self_attn.v_proj.weight   [num_kv_heads * head_dim, hidden_size]
  self_attn.o_proj.weight   [hidden_size, num_heads * head_dim]
  self_attn.q_norm.weight   [head_dim]
  self_attn.k_norm.weight   [head_dim]
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ....infra.tp import _tp_size
from ..L1.flash_attn_dense import FlashAttnDense
from ..L1.gemma_rms_norm import GemmaRMSNorm
from .parallel_linear import QKVParallelLinear, RowParallelLinear


class Qwen3NextAttention(nn.Module):
    """Full attention with per-head QK-norm, partial RoPE, output gating, and KV cache."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        tp = _tp_size()
        self.num_heads = num_attention_heads // tp
        self.num_kv_heads = num_key_value_heads // tp if num_key_value_heads % tp == 0 else num_key_value_heads
        self.head_dim = head_dim
        self.scaling = head_dim ** -0.5

        # QKV projection: Q outputs 2x heads (Q + gate)
        self.qkv_proj = QKVParallelLinear(
            hidden_size, head_dim,
            num_attention_heads * 2,  # doubled for output gate
            num_key_value_heads,
        )

        self.o_proj = RowParallelLinear(
            num_attention_heads * head_dim, hidden_size,
        )

        # Per-head QK norms (GemmaRMSNorm)
        self.q_norm = GemmaRMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = GemmaRMSNorm(head_dim, eps=rms_norm_eps)

        # Dense flash-attention (L1 op) — KV cache lives in layer_state.
        self.attn = FlashAttnDense(softmax_scale=self.scaling, causal=True)

    def forward(self, hidden_states, rotary_emb=None, positions=None,
                layer_state=None):
        """
        Args:
            hidden_states: [B, T, hidden_size]
            rotary_emb: RotaryEmbedding module
            positions: [B, T] or [N] position tensor
            layer_state: dict for KV cache (keys: "k_cache", "v_cache")
        Returns:
            output: [B, T, hidden_size]
        """
        shape = hidden_states.shape
        x = hidden_states.reshape(-1, shape[-1])
        N = x.shape[0]

        qkv = self.qkv_proj(x)
        q_gate_size = self.num_heads * 2 * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        q_gate, k, v = qkv.split([q_gate_size, kv_size, kv_size], dim=-1)

        # Split Q and gate
        q_gate = q_gate.view(N, self.num_heads, 2 * self.head_dim)
        q = q_gate[:, :, :self.head_dim].contiguous()
        gate = q_gate[:, :, self.head_dim:].contiguous()

        k = k.view(N, self.num_kv_heads, self.head_dim)
        v = v.view(N, self.num_kv_heads, self.head_dim)

        # Per-head QK-norm (applied before RoPE)
        q = self.q_norm(q.reshape(-1, self.head_dim)).view(N, self.num_heads, self.head_dim)
        k = self.k_norm(k.reshape(-1, self.head_dim)).view(N, self.num_kv_heads, self.head_dim)

        # Partial RoPE (only rotates first rotary_dim dimensions)
        if rotary_emb is not None and positions is not None:
            pos_flat = positions.reshape(-1) if positions.dim() > 1 else positions
            q, k = rotary_emb(pos_flat, q, k)

        # Reshape to [B, T, H, D] for flash_attn
        B = shape[0]
        T = N // B if len(shape) == 3 else N
        q = q.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, T, self.num_kv_heads, self.head_dim)
        v = v.view(B, T, self.num_kv_heads, self.head_dim)
        gate = gate.view(B, T, self.num_heads, self.head_dim)

        # KV cache: append new K/V to cached K/V
        if layer_state is not None:
            if "k_cache" in layer_state:
                # Decode: concatenate with cached K/V
                k = torch.cat([layer_state["k_cache"], k], dim=1)
                v = torch.cat([layer_state["v_cache"], v], dim=1)
            # Save updated cache
            layer_state["k_cache"] = k
            layer_state["v_cache"] = v

        # FlashAttnDense (L1): [B, T, H, D] layout, handles GQA natively;
        # causal=True with Q_len < K_len uses bottom-right alignment for decode.
        o = self.attn(q, k, v)  # [B, T_q, num_heads, head_dim]

        # Output gating: o * sigmoid(gate)
        o = o * torch.sigmoid(gate)

        # Output projection
        o = o.reshape(B * T, self.num_heads * self.head_dim)
        result = self.o_proj(o)
        return result.view(shape)
