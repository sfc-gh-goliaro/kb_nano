"""Feed-forward intermediate projection for encoder models."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.gelu import GELU
from ..L1.linear import Linear


class EncoderIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.intermediate_act_fn = GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.intermediate_act_fn(self.dense(hidden_states))
