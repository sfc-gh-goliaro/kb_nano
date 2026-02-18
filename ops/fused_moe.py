"""
Fused Mixture-of-Experts Triton kernels.

Implements a grouped GEMM approach matching vLLM's fused MoE:
  1. moe_align_block_size  — sort tokens by expert, pad to block boundaries
  2. fused_moe_kernel      — Triton grouped GEMM with optional weight multiply
  3. fused_experts         — two-pass driver: GEMM1 -> SiLU*mul -> GEMM2 -> sum

No vLLM imports.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# moe_align_block_size — Triton kernel for GPU-native sorting
# ---------------------------------------------------------------------------
@triton.jit
def _moe_align_kernel(
    topk_ids_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    tokens_per_expert_ptr,
    numel,
    num_experts,
    block_size,
    BLOCK: tl.constexpr,
):
    """Count tokens per expert and build sorted/padded output arrays."""
    pid = tl.program_id(0)

    if pid == 0:
        # Phase 1: count tokens per expert
        for i in range(numel):
            expert = tl.load(topk_ids_ptr + i)
            cur = tl.load(tokens_per_expert_ptr + expert)
            tl.store(tokens_per_expert_ptr + expert, cur + 1)

        # Phase 2: compute offsets and fill sorted_token_ids + expert_ids
        offset = 0
        total_blocks = 0
        for e in range(num_experts):
            count = tl.load(tokens_per_expert_ptr + e)
            padded = ((count + block_size - 1) // block_size) * block_size
            # Store offset for this expert (reuse tokens_per_expert as temp)
            tl.store(tokens_per_expert_ptr + num_experts + e, offset)
            tl.store(tokens_per_expert_ptr + 2 * num_experts + e, 0)  # write cursor

            n_blocks = padded // block_size
            for b in range(n_blocks):
                tl.store(expert_ids_ptr + total_blocks + b, e)
            total_blocks += n_blocks
            offset += padded

        tl.store(num_tokens_post_padded_ptr, offset)

        # Fill sorted_token_ids with padding value initially
        for i in range(offset):
            tl.store(sorted_token_ids_ptr + i, numel)

        # Phase 3: scatter tokens into sorted positions
        for i in range(numel):
            expert = tl.load(topk_ids_ptr + i)
            base = tl.load(tokens_per_expert_ptr + num_experts + expert)
            cursor = tl.load(tokens_per_expert_ptr + 2 * num_experts + expert)
            tl.store(sorted_token_ids_ptr + base + cursor, i)
            tl.store(tokens_per_expert_ptr + 2 * num_experts + expert, cursor + 1)


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    numel = topk_ids.numel()
    max_padded = numel + num_experts * (block_size - 1)
    max_blocks = triton.cdiv(max_padded, block_size)

    sorted_token_ids = torch.full(
        (max_padded,), numel, dtype=torch.int32, device=topk_ids.device,
    )
    expert_ids = torch.full(
        (max_blocks,), 0, dtype=torch.int32, device=topk_ids.device,
    )
    num_tokens_post_padded = torch.zeros(1, dtype=torch.int32, device=topk_ids.device)
    # Extra workspace: [counts | offsets | cursors] for num_experts each
    tokens_per_expert = torch.zeros(
        3 * num_experts, dtype=torch.int32, device=topk_ids.device,
    )

    _moe_align_kernel[(1,)](
        topk_ids.view(-1).contiguous(),
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        tokens_per_expert,
        numel, num_experts, block_size,
        BLOCK=1,
    )

    return sorted_token_ids, expert_ids, num_tokens_post_padded


# ---------------------------------------------------------------------------
# Triton fused MoE grouped GEMM kernel
# ---------------------------------------------------------------------------
@triton.jit
def _fused_moe_kernel(
    a_ptr, b_ptr, c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N, K, EM,
    num_valid_tokens,
    stride_am, stride_ak,
    stride_be, stride_bk, stride_bn,
    stride_cm, stride_cn,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    compute_type: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_m = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token_id = pid_m * BLOCK_SIZE_M + offs_m
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    off_expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K).to(tl.int64)

    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (off_expert * stride_be + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_SIZE_K):
        k_mask = (k_start + offs_k) < K
        a = tl.load(a_ptrs, mask=token_mask[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)
        accumulator = tl.dot(a.to(compute_type), b.to(compute_type), accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_token[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


# ---------------------------------------------------------------------------
# Kernel invocation
# ---------------------------------------------------------------------------
def _invoke_fused_moe_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    topk_weights: torch.Tensor | None,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: dict,
):
    EM = sorted_token_ids.size(0)
    if A.size(0) < config["BLOCK_SIZE_M"]:
        EM = min(EM, A.size(0) * top_k * config["BLOCK_SIZE_M"])

    grid = (
        triton.cdiv(EM, config["BLOCK_SIZE_M"]) * triton.cdiv(B.size(1), config["BLOCK_SIZE_N"]),
    )

    if A.dtype == torch.bfloat16:
        compute_type = tl.bfloat16
    elif A.dtype == torch.float16:
        compute_type = tl.float16
    else:
        compute_type = tl.float32

    _fused_moe_kernel[grid](
        A, B, C,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.size(1), B.size(2), EM,
        A.size(0) * top_k,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(2), B.stride(1),
        C.stride(0), C.stride(1),
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        **config,
    )


# ---------------------------------------------------------------------------
# Default config heuristic
# ---------------------------------------------------------------------------
def _get_default_config(M: int) -> dict:
    if M <= 4:
        return {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
        }
    return {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def fused_experts(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """Run fused MoE: two grouped GEMMs with SiLU-mul in between.

    Args:
        hidden_states: [M, K]
        w13: [E, 2*intermediate, K] — gate (w1) and up (w3) stacked on dim 1
        w2:  [E, K, intermediate]
        topk_weights: [M, top_k]
        topk_ids:     [M, top_k]
        num_experts: E

    Returns:
        output: [M, K]
    """
    M, K = hidden_states.size()
    E, N2, _ = w13.size()
    N = N2 // 2
    top_k = topk_ids.size(1)

    config = _get_default_config(M)

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, config["BLOCK_SIZE_M"], num_experts,
    )

    # GEMM1: hidden_states [M, K] x w13 [E, 2N, K]^T -> intermediate1 [M*top_k, 2N]
    intermediate1 = torch.empty(
        M * top_k, N2, device=hidden_states.device, dtype=hidden_states.dtype,
    )

    _invoke_fused_moe_kernel(
        hidden_states, w13, intermediate1,
        topk_weights, sorted_token_ids, expert_ids,
        num_tokens_post_padded,
        mul_routed_weight=False, top_k=top_k, config=config,
    )

    # SiLU-and-Mul: split gate/up, apply silu(gate)*up -> [M*top_k, N]
    gate = intermediate1[:, :N]
    up = intermediate1[:, N:]
    intermediate2 = F.silu(gate) * up

    # GEMM2: intermediate2 [M*top_k, N] x w2 [E, K, N]^T -> intermediate3 [M*top_k, K]
    # Apply routing weights in this kernel
    intermediate3 = torch.empty(
        M * top_k, K, device=hidden_states.device, dtype=hidden_states.dtype,
    )

    _invoke_fused_moe_kernel(
        intermediate2, w2, intermediate3,
        topk_weights, sorted_token_ids, expert_ids,
        num_tokens_post_padded,
        mul_routed_weight=True, top_k=1, config=config,
    )

    # Sum over top_k: [M, top_k, K] -> [M, K]
    return intermediate3.view(M, top_k, K).sum(dim=1)
