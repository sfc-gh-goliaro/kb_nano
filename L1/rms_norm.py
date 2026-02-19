"""RMSNorm: fused RMS normalization kernel."""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def _rms_forward(self, x, weight, eps):
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + eps)
        return x.to(orig_dtype) * weight

    @torch.compile
    def _add_rms_forward(self, x, residual, weight, eps):
        orig_dtype = x.dtype
        x = x.float() + residual.float()
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + eps)
        return x.to(orig_dtype) * weight, residual

    def forward(self, x, residual=None):
        if residual is None:
            return self._rms_forward(x, self.weight, self.eps)
        else:
            return self._add_rms_forward(x, residual, self.weight, self.eps)
