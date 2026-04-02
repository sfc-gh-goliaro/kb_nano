"""ReLU activation kernel."""

from __future__ import annotations

import torch
import torch.nn as nn


class ReLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)
