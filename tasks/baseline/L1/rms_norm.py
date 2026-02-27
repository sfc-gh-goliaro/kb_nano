"""RMSNorm using sgl_kernel for high-performance fused normalization."""

from __future__ import annotations

import torch
import torch.nn as nn

from sgl_kernel import rmsnorm as _sgl_rmsnorm
from sgl_kernel import fused_add_rmsnorm as _sgl_fused_add_rmsnorm


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x, residual=None):
        if residual is None:
            return _sgl_rmsnorm(x, self.weight, self.eps)
        else:
            _sgl_fused_add_rmsnorm(x, residual, self.weight, self.eps)
            return x, residual
