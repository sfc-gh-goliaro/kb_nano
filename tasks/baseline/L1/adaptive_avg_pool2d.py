"""AdaptiveAvgPool2d wrapping F.adaptive_avg_pool2d."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveAvgPool2d(nn.Module):
    def __init__(self, output_size: int | tuple[int, int]):
        super().__init__()
        self.output_size = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(x, self.output_size)
