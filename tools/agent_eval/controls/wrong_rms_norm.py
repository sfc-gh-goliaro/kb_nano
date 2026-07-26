"""NEGATIVE CONTROL: an rms_norm candidate that is mathematically wrong.

Same class name, same ``__init__`` signature, same parameter name/shape as
``tasks/baseline/L1/rms_norm.py`` -- so the strict weight transfer succeeds and
the ONLY difference the harness can see is the numerics: the normalization step
(``x * rsqrt(mean(x^2) + eps)``) is dropped, leaving a plain ``x * weight``.

If the harness reports this as PASSED, the harness is broken.
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
        if weight.dtype != x.dtype or weight.device != x.device:
            weight = weight.to(device=x.device, dtype=x.dtype)
        if residual is not None:
            x = x + residual
            residual = x
            return x * weight, residual
        # WRONG ON PURPOSE: no rsqrt(mean(x^2) + eps) factor.
        return x * weight
