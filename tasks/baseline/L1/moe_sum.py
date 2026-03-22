"""Fused MoE sum kernel: reduces top-k expert outputs into final output.

Uses sgl_kernel.moe_sum for high-performance reduction.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from sgl_kernel import moe_sum as _sgl_moe_sum


class _MoeSumBuf:
    """Shared mutable container for MoeSum output buffer."""
    __slots__ = ("output",)

    def __init__(self):
        self.output: torch.Tensor | None = None

    def get(self, M: int, D: int, device, dtype) -> torch.Tensor:
        o = self.output
        if o is None or o.size(0) < M or o.size(1) < D:
            self.output = o = torch.empty(M, D, device=device, dtype=dtype)
        return o[:M, :D]


class MoeSum(nn.Module):
    """Fused top-k reduction for MoE outputs using sgl_kernel."""

    def __init__(self):
        super().__init__()
        self._buf = _MoeSumBuf()

    def set_shared_buf(self, buf: _MoeSumBuf):
        self._buf = buf

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

        output = self._buf.get(M, D, input.device, input.dtype)
        _sgl_moe_sum(input.view(M, topk, D), output)

        return output
