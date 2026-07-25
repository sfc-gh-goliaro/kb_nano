"""LLaDA SwiGLU MLP."""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F

from .parallel_linear import ReplicatedLinear


class LLaDAMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, bias: bool = False):
        super().__init__()
        self.ff_proj = ReplicatedLinear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = ReplicatedLinear(hidden_size, intermediate_size, bias=bias)
        self.ff_out = ReplicatedLinear(intermediate_size, hidden_size, bias=bias)

    def forward(self, hidden_states):
        return self.ff_out(F.silu(self.ff_proj(hidden_states)) * self.up_proj(hidden_states))
