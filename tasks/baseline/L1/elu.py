"""ELU wrapping F.elu."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ELU(nn.Module):
    def __init__(self, alpha: float = 1.0, inplace: bool = False):
        super().__init__()
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.elu(x, alpha=self.alpha, inplace=self.inplace)
