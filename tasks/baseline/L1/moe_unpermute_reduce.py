"""Fused MoE unpermute + weighted reduce via Triton.

Gathers expert outputs using an inverse permutation, applies routing weights,
and reduces across top-k in a single fused kernel. Replaces PyTorch fancy
indexing + weighted scatter + .sum(dim=1).

Matches vllm's ``ep_gather`` / ``deepgemm_unpermute_and_reduce``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _unpermute_reduce_kernel(
    total_token_num,
    input_ptr,
    input_stride0,
    topk_ids_ptr,
    topk_ids_stride0,
    topk_ids_stride1,
    topk_weights_ptr,
    topk_weights_stride0,
    topk_weights_stride1,
    inv_perm_ptr,
    inv_perm_stride0,
    inv_perm_stride1,
    output_ptr,
    output_stride0,
    topk_num: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    cur_block = tl.program_id(0)
    start_token = tl.program_id(1)
    grid_num = tl.num_programs(1)

    for cur_token in range(start_token, total_token_num, grid_num):
        off_d = tl.arange(0, BLOCK_D)
        accumulator = tl.zeros([BLOCK_D], dtype=tl.float32)
        for topk_index in range(0, topk_num):
            expert_id = tl.load(
                topk_ids_ptr + cur_token * topk_ids_stride0 + topk_index
            )
            if expert_id >= 0:
                source_index = tl.load(
                    inv_perm_ptr + cur_token * inv_perm_stride0 + topk_index
                )
                weight = tl.load(
                    topk_weights_ptr + cur_token * topk_weights_stride0
                    + topk_index
                )
                tmp = tl.load(
                    input_ptr
                    + source_index * input_stride0
                    + cur_block * BLOCK_D
                    + off_d
                )
                accumulator += tmp.to(tl.float32) * weight

        tl.store(
            output_ptr
            + cur_token * output_stride0
            + cur_block * BLOCK_D
            + off_d,
            accumulator.to(output_ptr.dtype.element_ty),
        )


class MoeUnpermuteReduce(nn.Module):
    """Fused unpermute + weighted reduce for MoE expert outputs.

    Given GEMM2 output in expert-contiguous layout, gathers rows using
    inv_perm, multiplies by routing weights, and sums across top-k — all
    in one Triton kernel.
    """

    def forward(
        self,
        expert_output: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        inv_perm: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """
        Args:
            expert_output: [M_sum, K] bf16 — GEMM2 output in permuted layout.
            topk_ids: [M, top_k] int32 — expert IDs per token.
            topk_weights: [M, top_k] — routing weights.
            inv_perm: [M, top_k] int32 — index into expert_output per (token, topk).
            output: [M, K] — pre-allocated output buffer.
        """
        M = output.shape[0]
        K = expert_output.shape[1]
        BLOCK_D = min(K, 1024)
        assert K % BLOCK_D == 0

        grid = (triton.cdiv(K, BLOCK_D), min(M, 1024))
        _unpermute_reduce_kernel[grid](
            M,
            expert_output, expert_output.stride(0),
            topk_ids, topk_ids.stride(0), topk_ids.stride(1),
            topk_weights, topk_weights.stride(0), topk_weights.stride(1),
            inv_perm, inv_perm.stride(0), inv_perm.stride(1),
            output, output.stride(0),
            topk_num=topk_ids.shape[1],
            num_warps=2,
            BLOCK_D=BLOCK_D,
        )
