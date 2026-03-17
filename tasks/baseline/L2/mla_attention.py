"""MLA (Multi-head Latent Attention) for Kimi-Linear.

DeepSeek-V2 style compressed KV attention (NO RoPE rotation):
  q = q_proj(x)                     -> split to (q_nope, q_rope)
  kv_a = kv_a_proj_with_mqa(x)      -> split to (kv_compressed, k_rope_shared)
  kv_compressed = kv_a_layernorm(kv_compressed)
  kv_b = kv_b_proj(kv_compressed)   -> split to (k_nope, v)
  k = cat(k_nope, k_rope)           (no rotation applied)
  q = cat(q_nope, q_rope)           (no rotation applied)
  attn = SDPA(q, k, v)
  output = o_proj(attn)

Weight names:
  self_attn.q_proj.weight
  self_attn.kv_a_proj_with_mqa.weight
  self_attn.kv_a_layernorm.weight
  self_attn.kv_b_proj.weight
  self_attn.o_proj.weight
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....infra.tp import _tp_size
from .parallel_linear import ColumnParallelLinear, RowParallelLinear
from .kda_attention import ReplicatedLinear


class MLAAttention(nn.Module):
    """Multi-head Latent Attention (DeepSeek-V2 style)."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        kv_lora_rank: int,
        rms_norm_eps: float = 1e-5,
    ):
        super().__init__()
        tp = _tp_size()
        self.num_heads = num_attention_heads
        self.local_num_heads = num_attention_heads // tp
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.scaling = self.qk_head_dim ** -0.5

        # Q projection: hidden -> num_heads * qk_head_dim
        self.q_proj = ColumnParallelLinear(
            hidden_size, num_attention_heads * self.qk_head_dim
        )
        # Compressed KV + shared rope key
        self.kv_a_proj_with_mqa = ReplicatedLinear(
            hidden_size, kv_lora_rank + qk_rope_head_dim
        )
        self.kv_a_layernorm = nn.RMSNorm(kv_lora_rank, eps=rms_norm_eps)
        # Expand compressed KV
        self.kv_b_proj = ColumnParallelLinear(
            kv_lora_rank, num_attention_heads * (qk_nope_head_dim + v_head_dim)
        )
        self.o_proj = RowParallelLinear(
            num_attention_heads * v_head_dim, hidden_size
        )

    def forward(self, hidden_states, layer_state=None):
        """
        Args:
            hidden_states: [B, T, hidden_size]
            layer_state: dict with 'k_cache', 'v_cache' or None
        Returns:
            output: [B, T, hidden_size]
        """
        B, T, _ = hidden_states.shape

        # Q projection
        q = self.q_proj(hidden_states)
        q = q.view(B, T, self.local_num_heads, self.qk_head_dim)
        q_nope = q[..., :self.qk_nope_head_dim]
        q_rope = q[..., self.qk_nope_head_dim:]

        # KV compression
        kv_a = self.kv_a_proj_with_mqa(hidden_states)
        kv_compressed = kv_a[..., :self.kv_lora_rank]
        k_rope_shared = kv_a[..., self.kv_lora_rank:]  # [B, T, rope_dim]

        kv_compressed = self.kv_a_layernorm(kv_compressed)

        # Expand
        kv_b = self.kv_b_proj(kv_compressed)
        kv_b = kv_b.view(
            B, T, self.local_num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        k_nope = kv_b[..., :self.qk_nope_head_dim]
        v = kv_b[..., self.qk_nope_head_dim:]

        # No RoPE rotation for Kimi-Linear MLA — just concatenate
        k_rope = k_rope_shared.unsqueeze(2).expand(-1, -1, self.local_num_heads, -1)
        q_full = torch.cat([q_nope, q_rope], dim=-1)  # [B, T, H, qk_head_dim]
        k_full = torch.cat([k_nope, k_rope], dim=-1)  # [B, T, H, qk_head_dim]

        # Handle KV cache for decode
        if layer_state is not None:
            if "k_cache" not in layer_state:
                layer_state["k_cache"] = k_full
                layer_state["v_cache"] = v
            else:
                layer_state["k_cache"] = torch.cat(
                    [layer_state["k_cache"], k_full], dim=1
                )
                layer_state["v_cache"] = torch.cat(
                    [layer_state["v_cache"], v], dim=1
                )
            k_full = layer_state["k_cache"]
            v = layer_state["v_cache"]

        # SDPA: [B, T, H, D] -> [B, H, T, D]
        q_full = q_full.transpose(1, 2)
        k_full = k_full.transpose(1, 2)
        v = v.transpose(1, 2)

        is_causal = (q_full.shape[2] == k_full.shape[2] and q_full.shape[2] > 1)

        attn_out = F.scaled_dot_product_attention(
            q_full, k_full, v, is_causal=is_causal,
            scale=self.scaling,
        )

        # [B, H, T, v_dim] -> [B, T, H*v_dim]
        attn_out = attn_out.transpose(1, 2).reshape(
            B, T, self.local_num_heads * self.v_head_dim
        )
        return self.o_proj(attn_out)
