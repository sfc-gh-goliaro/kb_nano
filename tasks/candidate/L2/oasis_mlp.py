"""Oasis feed-forward blocks."""

from __future__ import annotations

from pathlib import Path
import sys
_L2_DIR = Path(__file__).resolve().parent
_L1_DIR = _L2_DIR.parent / "L1"
for _p in (str(_L2_DIR), str(_L1_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


import torch
import torch.nn as nn

from gelu import GELU
from linear import Linear


class OasisMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        *,
        approximate_tanh: bool = False,
    ):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = Linear(in_features, hidden_features, bias=True)
        self.act = GELU(approximate="tanh" if approximate_tanh else "none")
        self.fc2 = Linear(hidden_features, out_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))
