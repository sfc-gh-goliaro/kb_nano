"""SiLU-and-Mul activation: silu(x) * y where [x, y] = chunk(input, 2)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):
    @torch.compile
    def forward(self, x):
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
