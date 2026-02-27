"""Triton fused MoE grouped GEMM kernel."""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl


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


_TUNED_CONFIGS_N14336 = {
    1: {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 16, "num_warps": 4, "num_stages": 5},
    2: {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 5},
    4: {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 5},
    8: {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 5},
    16: {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 16, "num_warps": 4, "num_stages": 3},
    24: {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 16, "num_warps": 4, "num_stages": 2},
    32: {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 16, "num_warps": 8, "num_stages": 3},
    48: {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 16, "num_warps": 4, "num_stages": 3},
    64: {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 64, "num_warps": 4, "num_stages": 3},
    96: {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 64, "num_warps": 4, "num_stages": 3},
    128: {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 4},
    256: {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 4},
    512: {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 16, "num_warps": 8, "num_stages": 4},
    1024: {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 16, "num_warps": 8, "num_stages": 4},
    2048: {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 32, "num_warps": 8, "num_stages": 4},
    4096: {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 16, "num_warps": 8, "num_stages": 4},
}

_TUNED_CONFIGS_N4096 = {
    1: {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 16, "num_warps": 4, "num_stages": 5},
    2: {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 32, "num_warps": 4, "num_stages": 5},
    4: {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 16, "num_warps": 8, "num_stages": 3},
    8: {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 64, "num_warps": 4, "num_stages": 5},
    16: {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 16, "num_warps": 8, "num_stages": 2},
    24: {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 64, "num_warps": 4, "num_stages": 2},
    32: {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2},
    48: {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 64, "num_warps": 4, "num_stages": 2},
    64: {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 4},
    96: {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 4},
    128: {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 4},
    256: {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 4},
    512: {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 64, "num_warps": 8, "num_stages": 4},
    1024: {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 32, "num_warps": 8, "num_stages": 4},
    2048: {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 64, "num_warps": 8, "num_stages": 3},
    4096: {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 32, "num_warps": 8, "num_stages": 4},
}

_TUNED_GRID_KEYS_N14336 = sorted(_TUNED_CONFIGS_N14336.keys())
_TUNED_GRID_KEYS_N4096 = sorted(_TUNED_CONFIGS_N4096.keys())


def _get_default_config(M: int, N: int = 0) -> dict:
    """Select best kernel config based on batch size M and output dim N.

    Uses autotuned configs from vLLM (H200) when N matches known dimensions.
    Falls back to a reasonable generic config otherwise.
    """
    if N > 0:
        if N >= 14336 or (N >= 7168 and N < 8192):
            configs = _TUNED_CONFIGS_N14336
            grid_keys = _TUNED_GRID_KEYS_N14336
        elif N <= 4096:
            configs = _TUNED_CONFIGS_N4096
            grid_keys = _TUNED_GRID_KEYS_N4096
        else:
            configs = _TUNED_CONFIGS_N14336
            grid_keys = _TUNED_GRID_KEYS_N14336
        best_key = min(grid_keys, key=lambda x: abs(x - M))
        return dict(configs[best_key])

    if M <= 4:
        return {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 16,
            "num_warps": 4,
            "num_stages": 5,
        }
    if M <= 64:
        return {
            "BLOCK_SIZE_M": 32,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 64,
            "num_warps": 4,
            "num_stages": 3,
        }
    return {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 16,
        "num_warps": 8,
        "num_stages": 4,
    }


class MoeGroupedGemm(nn.Module):
    def forward(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        topk_weights: torch.Tensor | None,
        sorted_token_ids: torch.Tensor | None,
        expert_ids: torch.Tensor,
        num_tokens_post_padded: torch.Tensor,
        mul_routed_weight: bool,
        top_k: int,
        config: dict,
    ):
        naive = sorted_token_ids is None
        if naive:
            EM = expert_ids.numel() * config["BLOCK_SIZE_M"]
        else:
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

        launch_kwargs = {}
        if "num_warps" in config:
            launch_kwargs["num_warps"] = config["num_warps"]
        if "num_stages" in config:
            launch_kwargs["num_stages"] = config["num_stages"]

        sorted_ids_ptr = sorted_token_ids if sorted_token_ids is not None else A

        _fused_moe_kernel[grid](
            A, B, C,
            topk_weights,
            sorted_ids_ptr,
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
            BLOCK_SIZE_M=config["BLOCK_SIZE_M"],
            BLOCK_SIZE_N=config["BLOCK_SIZE_N"],
            BLOCK_SIZE_K=config["BLOCK_SIZE_K"],
            GROUP_SIZE_M=config["GROUP_SIZE_M"],
            NAIVE_BLOCK_ASSIGNMENT=naive,
            **launch_kwargs,
        )
