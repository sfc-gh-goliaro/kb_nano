"""XLM-RoBERTa encoder layer."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L2.encoder_attention import EncoderAttention
from ..L2.encoder_intermediate import EncoderIntermediate
from ..L2.encoder_output import EncoderOutput


class XLMRobertaLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = EncoderAttention(config)
        self.intermediate = EncoderIntermediate(config)
        self.output = EncoderOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attention_output = self.attention(hidden_states, attention_mask=attention_mask)
        intermediate_output = self.intermediate(attention_output)
        return self.output(intermediate_output, attention_output)
