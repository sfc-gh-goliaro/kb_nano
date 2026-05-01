"""Attention blocks for encoder models."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.dense_attention import DenseAttention
from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear


class EncoderSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size={config.hidden_size} must be divisible by "
                f"num_attention_heads={config.num_attention_heads}",
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size, bias=True)
        self.key = Linear(config.hidden_size, self.all_head_size, bias=True)
        self.value = Linear(config.hidden_size, self.all_head_size, bias=True)
        self.attn = DenseAttention(backend="sdpa")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        query = self.query(hidden_states).view(
            batch_size,
            seq_len,
            self.num_attention_heads,
            self.attention_head_size,
        )
        key = self.key(hidden_states).view(
            batch_size,
            seq_len,
            self.num_attention_heads,
            self.attention_head_size,
        )
        value = self.value(hidden_states).view(
            batch_size,
            seq_len,
            self.num_attention_heads,
            self.attention_head_size,
        )
        context = self.attn(query, key, value, causal=False, attn_mask=attention_mask)
        return context.contiguous().view(batch_size, seq_len, self.all_head_size)


class EncoderSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = Linear(config.hidden_size, config.hidden_size, bias=True)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        return self.LayerNorm(self.dense(hidden_states) + input_tensor)


class EncoderAttention(nn.Module):
    self_attention_cls = EncoderSelfAttention
    self_output_cls = EncoderSelfOutput

    def __init__(self, config):
        super().__init__()
        self.self = self.self_attention_cls(config)
        self.output = self.self_output_cls(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attention_output = self.self(hidden_states, attention_mask=attention_mask)
        return self.output(attention_output, hidden_states)
