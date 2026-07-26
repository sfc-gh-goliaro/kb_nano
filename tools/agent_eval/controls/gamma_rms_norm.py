"""WEIGHT-MISMATCH CONTROL: numerically correct rms_norm, wrong parameter name.

The math matches ``tasks/baseline/L1/rms_norm.py``'s ``forward_native`` exactly,
but the learnable scale is called ``self.gamma`` instead of ``self.weight``. A
``load_state_dict(baseline.state_dict(), strict=False)`` therefore reports
``missing_keys=['gamma']`` / ``unexpected_keys=['weight']`` and silently leaves
this module on its own initialisation.

The release runner swallows that (runner.py:496-500 is ``try/except pass``), and
because the baseline initialises ``weight = ones`` the un-transferred candidate
still matches numerically -- i.e. it PASSES for the wrong reason. The entrypoint
must report an error status instead.
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
            self.gamma = nn.Parameter(torch.ones(hidden_size))
        else:
            self.register_buffer("_unit_gamma", torch.ones(hidden_size),
                                 persistent=False)

    def forward(self, x, residual=None):
        weight = self.gamma if self.elementwise_affine else self._unit_gamma
        orig_dtype = x.dtype
        y = x.float()
        if residual is not None:
            y = y + residual.float()
            residual = y.to(orig_dtype)
        variance = y.pow(2).mean(dim=-1, keepdim=True)
        y = y * torch.rsqrt(variance + self.eps)
        y = y.to(orig_dtype)
        y = y * weight.to(device=x.device, dtype=orig_dtype)
        if residual is None:
            return y
        return y, residual
