"""BitNet decoder block: pre-norm residual with attention and MLP.

Weight names match HuggingFace checkpoint convention:
    layers.{i}.input_layernorm.weight
    layers.{i}.self_attn.*
    layers.{i}.post_attention_layernorm.weight
    layers.{i}.mlp.*
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L2.bitnet_attention import BitNetAttention
from ..L2.bitnet_mlp import BitNetMLP


class BitNetBlock(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps,
        )
        self.self_attn = BitNetAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            rms_norm_eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps,
        )
        self.mlp = BitNetMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            rms_norm_eps=config.rms_norm_eps,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Attention block
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        # MLP block
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
