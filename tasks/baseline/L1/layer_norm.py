"""Standard LayerNorm wrapper for vision encoder blocks."""

from __future__ import annotations

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)
