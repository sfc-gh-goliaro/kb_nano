"""Fused MoE permute for DeepGEMM contiguous layout.

Permutes FP8 activations and scales into expert-contiguous order, computes
per-row expert IDs, and records inverse permutation — all via Triton kernels.
Replaces PyTorch fancy indexing + torch.arange + torch.where.

Matches vllm's ``deepgemm_moe_permute`` (ep_scatter_1 + ep_scatter_2).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from .moe_align import MoeAlign


@triton.jit
def _count_and_layout_kernel(
    expert_num_tokens_ptr,
    expert_start_loc_ptr,
    m_indices_ptr,
    num_experts: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_EXPERT_NUM: tl.constexpr,
):
    """Compute cumulative expert start locations (aligned to 128) and
    fill m_indices with expert IDs."""
    cur_expert = tl.program_id(0)

    offset = tl.arange(0, BLOCK_EXPERT_NUM)
    tokens_per_expert = tl.load(
        expert_num_tokens_ptr + offset,
        mask=offset < num_experts, other=0,
    )
    rounded = ((tokens_per_expert.to(tl.int64) + 127) // 128) * 128
    cumsum = tl.cumsum(rounded) - rounded
    tl.store(expert_start_loc_ptr + offset, cumsum,
             mask=offset < num_experts)

    cur_start = tl.load(expert_start_loc_ptr + cur_expert)
    cur_count = tl.load(expert_num_tokens_ptr + cur_expert)

    off = tl.arange(0, BLOCK_E)
    for start_m in tl.range(0, cur_count, BLOCK_E, num_stages=4):
        offs = start_m + off
        mask = offs < cur_count
        tl.store(m_indices_ptr + cur_start + offs, cur_expert, mask=mask)


@triton.jit
def _scatter_kernel(
    total_tokens,
    expert_start_loc_ptr,
    aq_ptr, aq_stride0,
    aq_scale_ptr, aq_scale_stride0,
    topk_ids_ptr, topk_ids_stride0, topk_ids_stride1,
    out_ptr, out_stride0,
    out_scale_ptr, out_scale_stride0,
    inv_perm_ptr, inv_perm_stride0, inv_perm_stride1,
    topk_num: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    HIDDEN_SIZE_PAD: tl.constexpr,
    SCALE_SIZE: tl.constexpr,
    SCALE_SIZE_PAD: tl.constexpr,
):
    start_token = tl.program_id(0)
    grid_num = tl.num_programs(0)

    offs_h = tl.arange(0, HIDDEN_SIZE_PAD)
    mask_h = offs_h < HIDDEN_SIZE

    offs_s = tl.arange(0, SCALE_SIZE_PAD)
    mask_s = offs_s < SCALE_SIZE

    for token_id in range(start_token, total_tokens, grid_num):
        row = tl.load(aq_ptr + token_id * aq_stride0 + offs_h, mask=mask_h)
        row_s = tl.load(aq_scale_ptr + token_id * aq_scale_stride0 + offs_s,
                        mask=mask_s)

        for topk_idx in tl.range(0, topk_num, 1, num_stages=4):
            expert_id = tl.load(
                topk_ids_ptr + token_id * topk_ids_stride0 + topk_idx
            )
            if expert_id >= 0:
                dest = tl.atomic_add(expert_start_loc_ptr + expert_id, 1)
                tl.store(
                    inv_perm_ptr + token_id * inv_perm_stride0 + topk_idx,
                    dest,
                )
                tl.store(out_ptr + dest * out_stride0 + offs_h, row,
                         mask=mask_h)
                tl.store(out_scale_ptr + dest * out_scale_stride0 + offs_s,
                         row_s, mask=mask_s)


@triton.jit
def _count_expert_num_tokens_kernel(
    topk_ids_ptr,
    expert_num_tokens_ptr,
    num_experts,
    topk_numel,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel: count tokens per expert. CUDA-graph compatible.
    Matches vllm's _count_expert_num_tokens."""
    curr_expert = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    topk_ids_ptrs = topk_ids_ptr + offsets
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    for _x in range(tl.cdiv(topk_numel, BLOCK_SIZE)):
        mask = offsets < (topk_numel - _x * BLOCK_SIZE)
        expert_ids = tl.load(topk_ids_ptrs, mask=mask, other=-1)
        has_curr_expert = tl.where(expert_ids == curr_expert, 1, 0)
        acc = acc + has_curr_expert
        topk_ids_ptrs += BLOCK_SIZE
    if curr_expert < num_experts:
        tl.store(expert_num_tokens_ptr + curr_expert, tl.sum(acc))


def _count_expert_tokens(topk_ids: torch.Tensor,
                         num_experts: int) -> torch.Tensor:
    """Count tokens per expert from topk_ids [M, top_k].
    Uses a Triton kernel for CUDA-graph compatibility
    (no boolean indexing or dynamic shapes).
    Matches vllm's count_expert_num_tokens.
    """
    expert_num_tokens = torch.empty(
        num_experts, device=topk_ids.device, dtype=torch.int32,
    )
    BLOCK_SIZE = min(topk_ids.numel(), 1024)
    BLOCK_SIZE = triton.next_power_of_2(BLOCK_SIZE)
    _count_expert_num_tokens_kernel[(num_experts,)](
        topk_ids,
        expert_num_tokens,
        num_experts,
        topk_ids.numel(),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return expert_num_tokens


class MoePermute(nn.Module):
    """Fused FP8 MoE permute into DeepGEMM contiguous layout.

    Permutes FP8 activations and their scales into expert-contiguous order
    suitable for ``deep_gemm.m_grouped_fp8_gemm_nt_contiguous``.
    """

    def forward(
        self,
        aq: torch.Tensor,
        aq_scale: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        m_sum: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            aq: [M, K] float8_e4m3fn — FP8 quantized activations.
            aq_scale: [M, K // 128] float32 — per-group scales.
            topk_ids: [M, top_k] int32 — expert assignments.
            num_experts: total number of experts.
            m_sum: aligned total rows for output.

        Returns:
            aq_out: [m_sum, K] float8_e4m3fn — permuted activations.
            aq_scale_out: [m_sum, K // 128] float32 — permuted scales.
            expert_ids: [m_sum] int32 — per-row expert ID (-1 for padding).
            inv_perm: [M, top_k] int32 — maps (token, topk_slot) -> row in aq_out.
        """
        M = aq.shape[0]
        K = aq.shape[1]
        device = aq.device
        scale_cols = aq_scale.shape[1]

        expert_num_tokens = _count_expert_tokens(topk_ids, num_experts)
        expert_start_loc = torch.empty(num_experts, dtype=torch.int32,
                                       device=device)
        expert_ids = torch.full((m_sum,), -1, dtype=torch.int32, device=device)

        BLOCK_E = 128
        _count_and_layout_kernel[(num_experts,)](
            expert_num_tokens, expert_start_loc, expert_ids,
            num_experts=num_experts,
            num_warps=8,
            BLOCK_E=BLOCK_E,
            BLOCK_EXPERT_NUM=triton.next_power_of_2(num_experts),
        )

        aq_out = torch.empty(m_sum, K, dtype=aq.dtype, device=device)
        aq_scale_out = torch.empty(m_sum, scale_cols, dtype=torch.float32,
                                   device=device)
        inv_perm = torch.empty_like(topk_ids, dtype=torch.int32)

        grid = min(M, 1024 * 8)
        _scatter_kernel[(grid,)](
            M,
            expert_start_loc,
            aq, aq.stride(0),
            aq_scale, aq_scale.stride(0),
            topk_ids, topk_ids.stride(0), topk_ids.stride(1),
            aq_out, aq_out.stride(0),
            aq_scale_out, aq_scale_out.stride(0),
            inv_perm, inv_perm.stride(0), inv_perm.stride(1),
            topk_num=topk_ids.shape[1],
            num_warps=8,
            HIDDEN_SIZE=K,
            HIDDEN_SIZE_PAD=triton.next_power_of_2(K),
            SCALE_SIZE=scale_cols,
            SCALE_SIZE_PAD=triton.next_power_of_2(scale_cols),
        )

        return aq_out, aq_scale_out, expert_ids, inv_perm
