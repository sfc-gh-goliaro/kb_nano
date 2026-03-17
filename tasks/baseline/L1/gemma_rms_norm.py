"""GemmaRMSNorm: RMSNorm where stored weight is an offset (add 1 at runtime).

Checkpoint stores weight values near zero; runtime applies (1 + weight) as scale.
Uses sgl_kernel for the actual normalization, adjusting weight at load time.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from sgl_kernel import rmsnorm as _sgl_rmsnorm
from sgl_kernel import fused_add_rmsnorm as _sgl_fused_add_rmsnorm


class GemmaRMSNorm(nn.Module):
    """RMSNorm with weight stored as offset from 1.0 (Gemma convention).

    The checkpoint stores weights near zero. We add 1.0 at load time so that
    the sgl_kernel RMSNorm (which uses weight directly as scale) produces
    correct results.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.weight.weight_loader = self._weight_loader

    @staticmethod
    def _weight_loader(param, loaded_weight):
        param.data.copy_(loaded_weight + 1.0)

    def forward(self, x, residual=None):
        if residual is None:
            return _sgl_rmsnorm(x, self.weight, self.eps)
        else:
            _sgl_fused_add_rmsnorm(x, residual, self.weight, self.eps)
            return x, residual
