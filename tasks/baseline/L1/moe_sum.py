"""Fused MoE sum kernel: reduces top-k expert outputs into final output.

Uses a custom CUDA kernel for high-performance reduction.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
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


class MoeSum(nn.Module):
    """Fused top-k reduction for MoE outputs using sgl_kernel."""

    def __init__(self):
        super().__init__()
        self._output = None

    def forward(
        self,
        input: torch.Tensor,
        topk: int,
    ) -> torch.Tensor:
        """Sum over the topk dimension.

        Args:
            input: [M * topk, D] tensor
            topk: number of experts per token

        Returns:
            output: [M, D] tensor
        """
        total = input.size(0)
        M = total // topk
        D = input.size(1)

        if self._output is None or self._output.size(0) < M or self._output.size(1) < D:
            self._output = torch.empty(M, D, device=input.device, dtype=input.dtype)
        output = self._output[:M, :D]

        _C.moe_sum(input.view(M, topk, D), output)

        return output
