"""Softplus activation (L1).

Stateless wrapper around ``torch.nn.functional.softplus`` so L2 callers
can compose activations without importing ``torch.nn.functional``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Softplus(nn.Module):
    """Element-wise softplus: log(1 + exp(beta * x)) / beta."""

    def __init__(self, beta: float = 1.0, threshold: float = 20.0):
        super().__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x, beta=self.beta, threshold=self.threshold)
