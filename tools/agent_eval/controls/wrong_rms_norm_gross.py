"""NEGATIVE CONTROL (decisive): rms_norm with sum() where mean() belongs.

The 'skip normalization' control in ``wrong_rms_norm.py`` is a ~1/sqrt(2H)
relative perturbation on unit-variance input, which lands *at* the runner's bf16
gate (atol 1e-2 + rtol 1e-2) and is therefore only caught once a scenario has
enough rows. This control is unambiguous instead: using ``sum(x^2)`` rather than
``mean(x^2)`` scales every output by 1/sqrt(H) (~1/50 at H=2560), which no
tolerance can absorb. Every scenario must fail.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6,
                 elementwise_affine: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        else:
            self.register_buffer("_unit_weight", torch.ones(hidden_size),
                                 persistent=False)

    def forward(self, x, residual=None):
        weight = self.weight if self.elementwise_affine else self._unit_weight
        orig_dtype = x.dtype
        y = x.float()
        if residual is not None:
            y = y + residual.float()
            residual = y.to(orig_dtype)
        # WRONG ON PURPOSE: sum instead of mean -> output scaled by 1/sqrt(H).
        variance = y.pow(2).sum(dim=-1, keepdim=True)
        y = (y * torch.rsqrt(variance + self.eps)).to(orig_dtype)
        y = y * weight.to(device=x.device, dtype=orig_dtype)
        if residual is None:
            return y
        return y, residual
