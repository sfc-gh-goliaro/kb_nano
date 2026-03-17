"""GPT-OSS decoder layer: attention + MoE with RMSNorm residual connections."""

from __future__ import annotations

import torch.nn as nn

from ..L1.rms_norm import RMSNorm
from ..L2.gpt_oss_attention import GptOssAttention
from ..L2.gpt_oss_moe import GptOssMoE


class GptOssDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.self_attn = GptOssAttention(
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            layer_idx=layer_idx,
            sliding_window=config.sliding_window,
        )
        self.mlp = GptOssMoE(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions, hidden_states, residual, rotary_emb):
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states, rotary_emb)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual
