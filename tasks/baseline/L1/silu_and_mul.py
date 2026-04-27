"""SiLU-and-Mul activation with dual dispatch: CUDA (eager) and pure-PyTorch (compiled).

Mirrors vLLM's ``CustomOp`` dispatch:

* ``forward_cuda``: calls ``vllm._custom_ops.silu_and_mul`` (the packed
  bfloat162/half2 kernel in ``vllm/csrc/activation_kernels.cu``) so that
  kb_nano and vLLM produce bit-identical activations for the same input.
  Previously we used a kb_nano-local scalar silu kernel which introduced
  a ~1 ulp per-element drift vs vLLM — small, but it compounded through
  the downstream FP8 MLP and eventually showed up as a handful of MoE
  expert-id boundary flips at layers 3/4.
* ``forward_native``: pure PyTorch ``F.silu(x[..., :d]) * x[..., d:]`` so
  Inductor can fuse with adjacent ops (e.g. FP8 quantization).

Important: vLLM's kernel writes to ``out`` assuming a contiguous
``[num_tokens, d]`` row layout (row stride == ``d``). We therefore
allocate a fresh output tensor per call (mirrors vLLM's
``layers/activation.py:SiluAndMul``) and never reuse a shared buffer
across instances with different ``d`` values.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import vllm._C  # noqa: F401  — registers torch.ops._C.silu_and_mul


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
        """CUDA kernel with a freshly-allocated, contiguous output tensor.

        Dispatches to vLLM's ``_C.silu_and_mul`` (the packed kernel) so
        the per-element output bits match vLLM exactly.
        """
        d = x.size(-1) // 2
        output_shape = x.shape[:-1] + (d,)
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        torch.ops._C.silu_and_mul(out, x)
        return out

    def forward(self, x):
        if torch.compiler.is_compiling():
            return self.forward_native(x)
        return self.forward_cuda(x)
