"""Seed kernel for the kb_rms_norm task — a self-contained pure-PyTorch RMSNorm.

This is the agent's STARTING POINT (spawn.py copies it to solution/kernel.py) and
also the definition's `reference` string. It must satisfy the kb candidate
contract exactly, because the harness transfers the baseline's weights into it
and calls it with the baseline's forward signature:

  * class name          : RMSNorm  (matches tasks/baseline/L1/rms_norm.py)
  * __init__            : (hidden_size, eps=1e-6, elementwise_affine=True, **kwargs)
                          — **kwargs swallows scenario init_args the baseline
                          filters out (e.g. `training`).
  * parameter name      : `weight` when elementwise_affine, else a NON-persistent
                          `_unit_weight` buffer — the baseline's state_dict must
                          load with no missing/unexpected keys.
  * forward             : (x, residual=None) -> Tensor | (Tensor, Tensor)

Numerics mirror the baseline's `forward_native` (fp32 variance, rsqrt, cast back
to the input dtype, then multiply by weight — vLLM's rms_norm kernel applies the
weight AFTER the narrowing cast, so the order matters).

The residual path is IN-PLACE on both `x` and `residual`, mirroring vLLM's
`fused_add_rms_norm` which the baseline calls: residual := x + residual, then
x := rmsnorm(residual) * weight. The kb harness compares mutated inputs as well
as returned outputs, so a functional (non-mutating) residual path would be
reported as INCORRECT_NUMERICAL even if the returned tensors were right.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6,
                 elementwise_affine: bool = True, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        else:
            self.register_buffer("_unit_weight", torch.ones(hidden_size),
                                 persistent=False)

    def _scale(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight if self.elementwise_affine else self._unit_weight
        if w.dtype != x.dtype or w.device != x.device:
            w = w.to(device=x.device, dtype=x.dtype)
        return w

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        weight = self._scale(x)
        orig_dtype = x.dtype

        if residual is not None:
            residual.add_(x)                       # residual := x + residual
            acc = residual.float()
            var = acc.pow(2).mean(dim=-1, keepdim=True)
            normed = (acc * torch.rsqrt(var + self.eps)).to(orig_dtype) * weight
            x.copy_(normed)                        # x := rmsnorm(residual)
            return x, residual

        acc = x.float()
        var = acc.pow(2).mean(dim=-1, keepdim=True)
        return (acc * torch.rsqrt(var + self.eps)).to(orig_dtype) * weight
