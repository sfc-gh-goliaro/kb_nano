"""RMSNorm using custom CUDA kernels for high-performance fused normalization."""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load as _load_ext

_CSRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")
_C = _load_ext(
    name="kb_nano_L1_ops",
    sources=[os.path.join(_CSRC, f) for f in [
        "binding.cpp", "rmsnorm.cu", "activation.cu", "pos_enc.cu",
        "moe_sum.cu", "moe_align.cu", "moe_topk_softmax.cu",
    ]],
    extra_cuda_cflags=["-O3", "--use_fast_math",
                       "-DFLASHINFER_ENABLE_BF16", "-DFLASHINFER_ENABLE_F16"],
    extra_cflags=["-O3"],
    verbose=False,
)


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
                out = torch.empty_like(x)
                _C.rmsnorm(out, x, self.weight, self.eps)
                return out
            else:
                _C.fused_add_rmsnorm(x, residual, self.weight, self.eps)
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
