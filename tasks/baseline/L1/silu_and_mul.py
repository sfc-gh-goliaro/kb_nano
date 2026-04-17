"""SiLU-and-Mul activation with dual dispatch: CUDA (eager) and pure-PyTorch (compiled).

Mirrors vLLM's ``CustomOp`` dispatch:
  - ``forward_cuda``: fast ``_C.silu_and_mul`` kernel with shared output buffer
  - ``forward_native``: pure PyTorch ``F.silu(x[..., :d]) * x[..., d:]`` so
    Inductor can fuse with adjacent ops (e.g. FP8 quantization)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .csrc import _C

# ---------------------------------------------------------------------------
# Register in-place silu_and_mul op for eager/CUDA-graph path.
# ---------------------------------------------------------------------------

_lib = torch.library.Library("kb_nano_act", "DEF")

_lib.define("silu_and_mul(Tensor! result, Tensor input) -> ()")

def _silu_and_mul_impl(result, input):
    _C.silu_and_mul(result, input)

_lib.impl("silu_and_mul", _silu_and_mul_impl, "CUDA")

@torch.library.impl(_lib, "silu_and_mul", "Meta")
def _silu_and_mul_meta(result, input):
    pass


class _ActivationBuffer:
    """Mutable container so multiple SiluAndMul layers can share one buffer."""
    __slots__ = ("buf",)

    def __init__(self):
        self.buf: torch.Tensor | None = None

    def get(self, rows: int, cols: int, device, dtype) -> torch.Tensor:
        b = self.buf
        if b is None or b.size(0) < rows or b.size(1) < cols:
            self.buf = b = torch.empty(rows, cols, device=device, dtype=dtype)
        return b[:rows, :cols]


class SiluAndMul(nn.Module):
    def __init__(self):
        super().__init__()
        self._act_buf = _ActivationBuffer()

    def set_shared_buffer(self, shared: _ActivationBuffer):
        self._act_buf = shared

    @staticmethod
    def forward_native(x: torch.Tensor) -> torch.Tensor:
        """Pure PyTorch implementation — visible to Inductor for fusion."""
        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        """CUDA kernel with pre-allocated output buffer."""
        num_tokens = x.numel() // x.size(-1)
        half = x.size(-1) // 2
        out = self._act_buf.get(num_tokens, half, x.device, x.dtype)
        torch.ops.kb_nano_act.silu_and_mul(out, x)
        return out

    def forward(self, x):
        if torch.compiler.is_compiling():
            return self.forward_native(x)
        return self.forward_cuda(x)
