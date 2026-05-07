"""Hardsigmoid wrapping F.hardsigmoid (explicit L1 op for benchmark scaffolding)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Hardsigmoid(nn.Module):
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.hardsigmoid(x, inplace=self.inplace)
