"""Hardswish wrapping F.hardswish."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Hardswish(nn.Module):
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.hardswish(x, inplace=self.inplace)
