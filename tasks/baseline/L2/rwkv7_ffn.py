"""RWKV7 feed-forward network: token-shift + key -> sqrelu -> value.

Built exclusively from L1 ops:
  ``token_shift`` (zero-pad + lerp), ``Linear`` x2, ``SquaredReLU``.

Weight names match the FLA checkpoint format:
  ``x_k`` (per-channel mix vector), ``key.weight``, ``value.weight``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.linear import Linear
from ..L1.squared_relu import SquaredReLU


class RWKV7FeedForward(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.x_k = nn.Parameter(torch.zeros(hidden_size))
        self.key = Linear(hidden_size, intermediate_size, bias=False)
        self.value = Linear(intermediate_size, hidden_size, bias=False)
        self.act = SquaredReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Token shift: delta = prev_token - current_token (zero-pad on left)
        shifted = torch.zeros_like(x)
        shifted[:, 1:] = x[:, :-1]
        delta = shifted - x
        xk = torch.addcmul(x, delta, self.x_k)
        return self.value(self.act(self.key(xk)))
