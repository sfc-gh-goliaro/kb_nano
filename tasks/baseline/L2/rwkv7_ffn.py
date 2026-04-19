"""RWKV7 feed-forward network with token shift and squared ReLU.

Uses token shift mixing and sqrelu activation:
  delta = prev_token - current_token
  xk = hidden + delta * x_k
  output = value(sqrelu(key(xk)))
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RWKV7FeedForward(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.x_k = nn.Parameter(torch.zeros(hidden_size))
        self.key = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.value = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Token shift
        shifted = torch.zeros_like(x)
        shifted[:, 1:] = x[:, :-1]
        delta = shifted - x

        xk = x.addcmul(delta, self.x_k)
        return self.value(F.relu(self.key(xk)).square())
