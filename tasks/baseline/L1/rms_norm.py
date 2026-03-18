"""RMSNorm using sgl_kernel for high-performance fused normalization."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from sgl_kernel import rmsnorm as _sgl_rmsnorm
from sgl_kernel import fused_add_rmsnorm as _sgl_fused_add_rmsnorm


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x, residual=None):
        if self.elementwise_affine:
            if residual is None:
                return _sgl_rmsnorm(x, self.weight, self.eps)
            else:
                _sgl_fused_add_rmsnorm(x, residual, self.weight, self.eps)
                return x, residual
        else:
            if residual is None:
                x = F.rms_norm(x, (self.hidden_size,), eps=self.eps)
                return x
            else:
                x = x + residual
                residual = x
                x = F.rms_norm(x, (self.hidden_size,), eps=self.eps)
                return x, residual
