"""Vision MLP for Qwen vision transformer blocks.

Supports QuickGELU (Qwen2-VL) and SiLU (Qwen3-VL) activations.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .parallel_linear import ColumnParallelLinear, RowParallelLinear


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class VisionMLP(nn.Module):
    """MLP for Qwen2-VL vision encoder (QuickGELU activation)."""

    def __init__(self, in_features: int, hidden_features: int,
                 act_layer: type[nn.Module] = QuickGELU):
        super().__init__()
        self.fc1 = ColumnParallelLinear(in_features, hidden_features, bias=True)
        self.act = act_layer()
        self.fc2 = RowParallelLinear(hidden_features, in_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Qwen3VisionMLP(nn.Module):
    """MLP for Qwen3-VL vision encoder (configurable activation, with bias)."""

    def __init__(self, in_features: int, hidden_features: int,
                 act_fn: Callable[[torch.Tensor], torch.Tensor] = F.silu,
                 bias: bool = True):
        super().__init__()
        self.fc1 = ColumnParallelLinear(in_features, hidden_features, bias=bias)
        self.fc2 = RowParallelLinear(hidden_features, in_features, bias=bias)
        self.act_fn = act_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act_fn(self.fc1(x)))
