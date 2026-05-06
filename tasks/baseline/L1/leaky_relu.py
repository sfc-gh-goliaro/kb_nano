"""LeakyReLU wrapping F.leaky_relu."""

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
