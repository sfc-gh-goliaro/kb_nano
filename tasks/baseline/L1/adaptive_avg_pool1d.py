"""AdaptiveAvgPool1d wrapping F.adaptive_avg_pool1d."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveAvgPool1d(nn.Module):
    def __init__(self, output_size: int):
        super().__init__()
        self.output_size = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool1d(x, self.output_size)
