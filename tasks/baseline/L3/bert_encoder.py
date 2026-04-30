"""BERT encoder stack."""

from __future__ import annotations

import torch
import torch.nn as nn

from .bert_layer import BertLayer


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask=attention_mask)
        return hidden_states
