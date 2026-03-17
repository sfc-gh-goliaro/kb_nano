"""GELU activation (exact, via torch.nn.functional.gelu)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x)
