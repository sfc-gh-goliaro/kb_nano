"""Self-attention block for encoder models."""

from __future__ import annotations

import torch
import torch.nn as nn

from .encoder_self_attention import EncoderSelfAttention
from .encoder_self_output import EncoderSelfOutput


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
