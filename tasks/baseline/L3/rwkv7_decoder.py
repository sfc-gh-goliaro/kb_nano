"""RWKV7 decoder block.

Pre-norm residual pattern with LayerNorm (with bias):
  Layer 0: pre_norm -> attn_norm -> RWKV7Attention -> residual -> ffn_norm -> FFN -> residual
  Other:              attn_norm -> RWKV7Attention -> residual -> ffn_norm -> FFN -> residual
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L2.rwkv7_attention import RWKV7Attention
from ..L2.rwkv7_ffn import RWKV7FeedForward


class RWKV7Block(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        if layer_idx == 0:
            self.pre_norm = nn.LayerNorm(
                config.hidden_size, bias=config.norm_bias, eps=config.norm_eps,
            )

        self.attn_norm = nn.LayerNorm(
            config.hidden_size, bias=config.norm_bias, eps=config.norm_eps,
        )
        self.attn = RWKV7Attention(
            hidden_size=config.hidden_size,
            head_dim=config.head_dim,
            num_heads=config.num_heads,
            decay_low_rank_dim=config.decay_low_rank_dim,
            gate_low_rank_dim=config.gate_low_rank_dim,
            a_low_rank_dim=config.a_low_rank_dim,
            v_low_rank_dim=config.v_low_rank_dim,
            norm_eps=config.norm_eps,
            layer_idx=layer_idx,
        )

        self.ffn_norm = nn.LayerNorm(
            config.hidden_size, bias=config.norm_bias, eps=config.norm_eps,
        )
        self.ffn = RWKV7FeedForward(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
        )

    def forward(
        self, hidden_states: torch.Tensor, v_first: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Layer 0 pre-norm
        residual = self.pre_norm(hidden_states) if hasattr(self, 'pre_norm') else hidden_states

        # Attention sub-block
        hidden_states = self.attn_norm(residual)
        hidden_states, v_first = self.attn(hidden_states, v_first)
        hidden_states = residual + hidden_states

        # FFN sub-block
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, v_first
