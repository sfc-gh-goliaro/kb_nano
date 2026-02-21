"""Fused MoE sum kernel: reduces top-k expert outputs into final output.

Uses sgl_kernel.moe_sum for high-performance reduction.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from sgl_kernel import moe_sum as _sgl_moe_sum


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

        _sgl_moe_sum(input.view(M, topk, D), output)

        return output
