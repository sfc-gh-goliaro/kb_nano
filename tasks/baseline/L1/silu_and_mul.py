"""SiLU-and-Mul activation with CUDA eager and pure-PyTorch compiled paths."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import vllm._C  # noqa: F401  - registers torch.ops._C.silu_and_mul
    _silu_and_mul_kernel = torch.ops._C.silu_and_mul
except (AttributeError, ImportError):
    from .csrc import _C
    _silu_and_mul_kernel = _C.silu_and_mul

try:
    _silu_and_mul_with_clamp_kernel = torch.ops._C.silu_and_mul_with_clamp
except AttributeError:
    _silu_and_mul_with_clamp_kernel = None


class SiluAndMul(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward_native(x: torch.Tensor) -> torch.Tensor:
        """Pure PyTorch implementation — visible to Inductor for fusion."""
        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]

    @staticmethod
    def forward_cuda(x: torch.Tensor) -> torch.Tensor:
        d = x.size(-1) // 2
        output_shape = x.shape[:-1] + (d,)
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        _silu_and_mul_kernel(out, x)
        return out

    def forward(self, x):
        if torch.compiler.is_compiling():
            return self.forward_native(x)
        return self.forward_cuda(x)


class SiluAndMulWithClamp(nn.Module):
    """SwiGLU with input clamping: clamp gate (max only) and up (both), then SiLU*up.

    Reference: vllm/model_executor/layers/activation.py:SiluAndMulWithClamp
    """

    def __init__(self, swiglu_limit: float):
        super().__init__()
        self.swiglu_limit = float(swiglu_limit)

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        gate = torch.clamp(x[..., :d], max=self.swiglu_limit)
        up = torch.clamp(x[..., d:], min=-self.swiglu_limit, max=self.swiglu_limit)
        return F.silu(gate) * up

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        d = x.size(-1) // 2
        output_shape = x.shape[:-1] + (d,)
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        _silu_and_mul_with_clamp_kernel(out, x, self.swiglu_limit)
        return out

    def forward(self, x):
        if torch.compiler.is_compiling() or _silu_and_mul_with_clamp_kernel is None:
            return self.forward_native(x)
        return self.forward_cuda(x)
