"""Fused MoE sum kernel: reduces top-k expert outputs into final output.

Replaces torch .sum(dim=1) with an optimized Triton kernel for topk=2.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _moe_sum_kernel(
    input_ptr,
    output_ptr,
    d: tl.constexpr,
    topk: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_idx = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    mask = offs_d < d

    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for k in range(topk):
        vals = tl.load(
            input_ptr + token_idx * topk * d + k * d + offs_d,
            mask=mask, other=0.0,
        )
        acc += vals.to(tl.float32)

    tl.store(output_ptr + token_idx * d + offs_d, acc.to(tl.bfloat16), mask=mask)


class MoeSum(nn.Module):
    """Fused top-k reduction for MoE outputs."""

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

        if D > 8192:
            return input.view(M, topk, D).sum(dim=1)

        if self._output is None or self._output.size(0) < M or self._output.size(1) < D:
            self._output = torch.empty(M, D, device=input.device, dtype=input.dtype)
        output = self._output[:M, :D]

        BLOCK_D = triton.next_power_of_2(D)

        _moe_sum_kernel[(M,)](
            input, output, D, topk,
            BLOCK_D=BLOCK_D,
        )

        return output
