"""Triton fused MoE grouped GEMM kernel with FP8 W8A8 block-scaled quantization.

Extends the BF16 MoE kernel to accept FP8 expert weights with per-block
scale factors and FP8-quantized activations with per-token-group scales.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused_moe_fp8_kernel(
    a_ptr, b_ptr, c_ptr,
    a_scale_ptr, b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N, K, EM,
    num_valid_tokens,
    stride_am, stride_ak,
    stride_asm, stride_ask,
    stride_be, stride_bk, stride_bn,
    stride_bse, stride_bsk, stride_bsn,
    stride_cm, stride_cn,
    group_k: tl.constexpr,
    group_n: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NAIVE_BLOCK_ASSIGNMENT: tl.constexpr = False,
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

    if NAIVE_BLOCK_ASSIGNMENT:
        offs_token = tl.where(offs_m == 0, pid_m, num_valid_tokens)
    else:
        offs_token_id = pid_m * BLOCK_SIZE_M + offs_m
        offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    off_expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K).to(tl.int64)

    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (off_expert * stride_be + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Scale pointers: a_scale is [M, K//group_k], b_scale is [E, N//group_n, K//group_k]
    a_scale_base = a_scale_ptr + (offs_token[:, None] // top_k) * stride_asm
    offs_bsn = offs_bn // group_n
    b_scale_base = b_scale_ptr + off_expert * stride_bse + offs_bsn * stride_bsn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_SIZE_K):
        k_mask = (k_start + offs_k) < K
        a = tl.load(a_ptrs, mask=token_mask[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)

        offs_ks = k_start // group_k
        a_s = tl.load(a_scale_base + offs_ks * stride_ask, mask=token_mask[:, None], other=1.0)
        b_s = tl.load(b_scale_base + offs_ks * stride_bsk)

        accumulator += tl.dot(a, b) * a_s * b_s[None, :]

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(tl.bfloat16)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_token[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def _get_fp8_moe_config(M: int, N: int) -> dict:
    """Default config for FP8 MoE kernel.

    Uses BLOCK_SIZE_K=128 to align with the 128x128 FP8 block quantization.
    """
    if M <= 4:
        return {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 16, "num_warps": 4, "num_stages": 4,
        }
    if M <= 64:
        return {
            "BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 32, "num_warps": 4, "num_stages": 3,
        }
    return {
        "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 32, "num_warps": 4, "num_stages": 3,
    }


class FP8MoeGroupedGemm(nn.Module):
    """FP8 W8A8 block-scaled MoE grouped GEMM.

    forward(A, A_scale, B, B_scale, C, topk_weights, sorted_token_ids,
            expert_ids, num_tokens_post_padded, mul_routed_weight, top_k,
            block_size, config):
        A:          [M, K] float8_e4m3fn -- pre-quantized activations
        A_scale:    [M, K//bk] float32 -- per-token-group activation scales
        B:          [E, N, K] float8_e4m3fn -- expert weights (stored as [E, N, K])
        B_scale:    [E, ceil(N/bn), ceil(K/bk)] float32 -- per-block weight scales
        C:          [M*topk, N] bfloat16 -- output buffer
        block_size: (bn, bk) quantization block dimensions
        config:     optional kernel config dict (auto-selected when None)
    """

    def forward(
        self,
        A: torch.Tensor,
        A_scale: torch.Tensor,
        B: torch.Tensor,
        B_scale: torch.Tensor,
        C: torch.Tensor,
        topk_weights: torch.Tensor | None,
        sorted_token_ids: torch.Tensor | None,
        expert_ids: torch.Tensor,
        num_tokens_post_padded: torch.Tensor,
        mul_routed_weight: bool,
        top_k: int,
        block_size: tuple[int, int],
        config: dict | None = None,
    ):
        group_n, group_k = block_size
        naive = sorted_token_ids is None

        if config is None:
            config = _get_fp8_moe_config(A.size(0), B.size(1))

        if naive:
            EM = expert_ids.numel() * config["BLOCK_SIZE_M"]
        else:
            EM = sorted_token_ids.size(0)
            if A.size(0) < config["BLOCK_SIZE_M"]:
                EM = min(EM, A.size(0) * top_k * config["BLOCK_SIZE_M"])

        grid = (
            triton.cdiv(EM, config["BLOCK_SIZE_M"]) * triton.cdiv(B.size(1), config["BLOCK_SIZE_N"]),
        )

        launch_kwargs = {}
        if "num_warps" in config:
            launch_kwargs["num_warps"] = config["num_warps"]
        if "num_stages" in config:
            launch_kwargs["num_stages"] = config["num_stages"]

        sorted_ids_ptr = sorted_token_ids if sorted_token_ids is not None else A

        _fused_moe_fp8_kernel[grid](
            A, B, C,
            A_scale, B_scale,
            topk_weights,
            sorted_ids_ptr,
            expert_ids,
            num_tokens_post_padded,
            B.size(1), B.size(2), EM,
            A.size(0) * top_k,
            A.stride(0), A.stride(1),
            A_scale.stride(0), A_scale.stride(1),
            B.stride(0), B.stride(2), B.stride(1),
            B_scale.stride(0), B_scale.stride(2), B_scale.stride(1),
            C.stride(0), C.stride(1),
            group_k=group_k,
            group_n=group_n,
            MUL_ROUTED_WEIGHT=mul_routed_weight,
            top_k=top_k,
            BLOCK_SIZE_M=config["BLOCK_SIZE_M"],
            BLOCK_SIZE_N=config["BLOCK_SIZE_N"],
            BLOCK_SIZE_K=config["BLOCK_SIZE_K"],
            GROUP_SIZE_M=config["GROUP_SIZE_M"],
            NAIVE_BLOCK_ASSIGNMENT=naive,
            **launch_kwargs,
        )
