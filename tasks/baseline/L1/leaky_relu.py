"""LeakyReLU wrapping F.leaky_relu.

Explicit L1 op for benchmark scaffolding (per mentor guidance: even when an
op is bit-equivalent to a torch builtin, having a dedicated kb-nano L1 file
gives the benchmark suite a clean per-op target and matches the existing
kb-nano pattern of one L1 file per torch.nn activation class — see
silu.py, gelu.py, relu.py, sigmoid.py, tanh.py).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeakyReLU(nn.Module):
    def __init__(self, negative_slope: float = 0.01, inplace: bool = False):
        super().__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(x, negative_slope=self.negative_slope, inplace=self.inplace)
