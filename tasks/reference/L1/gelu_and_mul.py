"""Semantic PyTorch reference for gelu_and_mul.

This file is used for specification/prompting and optional validation only.
It is not the production baseline and should not be used for reported speed.

The input is a concatenation of [gate, up] along the last dimension; the
output is ``gelu(gate) * up`` with the chosen approximation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeluAndMul(nn.Module):
    """Apply GELU to the gate half and multiply by the up half."""

    def __init__(self, approximate: str = "none"):
        super().__init__()
        if approximate not in ("none", "tanh"):
            raise ValueError(f"Unsupported GELU approximation: {approximate}")
        self.approximate = approximate

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        return F.gelu(x[..., :d], approximate=self.approximate) * x[..., d:]

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_native(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_native(x)
