"""Semantic PyTorch reference for MoE grouped GEMM."""

from __future__ import annotations

import torch
import torch.nn as nn


_DEFAULT_CONFIG_HEURISTIC = {
    "small": {
        "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 16, "num_warps": 4, "num_stages": 5,
    },
    "medium": {
        "BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 64, "num_warps": 4, "num_stages": 3,
    },
    "large": {
        "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 16, "num_warps": 8, "num_stages": 4,
    },
}


def _get_default_config(M: int, E: int = 0, N: int = 0,
                        block_shape: list[int] | None = None) -> dict:
    """Mirror of the production M-tier heuristic.  BLOCK_SIZE_M matters for
    correctness, not just speed: in naive mode the per-block gate is
    ``block * BLOCK_SIZE_M >= num_tokens_post_padded``."""
    del E, N, block_shape
    if M <= 4:
        return dict(_DEFAULT_CONFIG_HEURISTIC["small"])
    if M <= 64:
        return dict(_DEFAULT_CONFIG_HEURISTIC["medium"])
    return dict(_DEFAULT_CONFIG_HEURISTIC["large"])


def get_triton_config(M: int, w1_shape: tuple[int, ...], w2_shape: tuple[int, ...],
                      top_k: int, use_fp8: bool,
                      block_shape: list[int] | None = None) -> dict:
    del w1_shape, w2_shape, top_k, use_fp8
    return _get_default_config(M, block_shape=block_shape)


def _valid_deep_gemm(hidden_states: torch.Tensor, w1: torch.Tensor,
                     w2: torch.Tensor) -> bool:
    del hidden_states, w1, w2
    return False


def m_grouped_fp8_gemm_nt_contiguous(a_and_scale, b_and_scale, output, expert_ids):
    raise RuntimeError("DeepGEMM is not available in the self-contained reference path")


def _expand_group_scale(
    x: torch.Tensor,
    scale: torch.Tensor | None,
    block_shape: list[int] | None,
) -> torch.Tensor:
    if scale is None:
        return torch.ones_like(x, dtype=torch.float32)
    scale = scale.float()
    if scale.numel() == 1:
        return scale.reshape(1, 1).expand_as(x.float())
    if scale.shape == x.shape:
        return scale
    if scale.ndim == 1 and scale.numel() == x.shape[-1]:
        return scale.view(1, -1).expand_as(x.float())
    if scale.ndim == 1 and scale.numel() == x.shape[0]:
        return scale.view(-1, 1).expand_as(x.float())
    if block_shape is not None and len(block_shape) == 2 and scale.ndim == 2:
        block_n, block_k = block_shape
        return scale.repeat_interleave(block_n, dim=0).repeat_interleave(block_k, dim=1)[
            : x.shape[0], : x.shape[1]
        ]
    if scale.ndim == 2 and scale.shape[0] == x.shape[0]:
        repeat = (x.shape[1] + scale.shape[1] - 1) // scale.shape[1]
        return scale.repeat_interleave(repeat, dim=1)[:, : x.shape[1]]
    return torch.ones_like(x, dtype=torch.float32) * scale.reshape(-1)[0]


def _dequant_a(
    A: torch.Tensor,
    a_scale: torch.Tensor | None,
    block_shape: list[int] | None,
) -> torch.Tensor:
    A_f = A.float()
    if a_scale is None:
        return A_f
    if a_scale.ndim == 2 and a_scale.shape[0] == A.shape[0]:
        # Activation scales are PER TOKEN-GROUP: row i belongs to token i and
        # column g scales its g-th group of K features (group = block_shape's
        # K block).  Never expand along the token axis.
        if block_shape is not None and len(block_shape) == 2:
            group = int(block_shape[1])
        else:
            group = (A.shape[1] + a_scale.shape[1] - 1) // a_scale.shape[1]
        scale = a_scale.float().repeat_interleave(group, dim=1)[:, : A.shape[1]]
    else:
        scale = _expand_group_scale(A, a_scale, block_shape)
    return A_f * scale


def _dequant_b(
    B_e: torch.Tensor,
    b_scale_e: torch.Tensor | None,
    block_shape: list[int] | None,
) -> torch.Tensor:
    B_f = B_e.float()
    if b_scale_e is None:
        return B_f
    scale = _expand_group_scale(B_e, b_scale_e, block_shape)
    return B_f * scale


class MoeGroupedGemm(nn.Module):
    @staticmethod
    def get_config(M: int, N: int = 0, E: int = 0,
                   use_fp8: bool = False,
                   block_shape: list[int] | None = None) -> dict:
        del N, E, use_fp8
        return _get_default_config(M, block_shape=block_shape)

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
        config: dict | None = None,
        a_scale: torch.Tensor | None = None,
        b_scale: torch.Tensor | None = None,
        use_fp8_w8a8: bool = False,
        block_shape: list[int] | None = None,
    ):
        del use_fp8_w8a8
        config = _get_default_config(A.size(0)) if config is None else config
        block_size = int(config.get("BLOCK_SIZE_M", 1))
        valid_tokens = A.size(0) * top_k
        # The kernel gates every block on num_tokens_post_padded
        # (``pid_m * BLOCK_SIZE_M >= ntpp -> return``) and NEVER zeroes C:
        # rows it does not compute keep whatever the output buffer held.
        if isinstance(num_tokens_post_padded, torch.Tensor):
            ntpp = int(num_tokens_post_padded.reshape(-1)[0].item())
        else:
            ntpp = int(num_tokens_post_padded)
        A_deq = _dequant_a(A, a_scale, block_shape)
        flat_weights = topk_weights.reshape(-1).float() if topk_weights is not None else None

        if sorted_token_ids is None:
            # naive mode: block ``row`` handles flat token ``row`` alone
            for row, expert in enumerate(expert_ids.reshape(-1).tolist()):
                if row * block_size >= ntpp:
                    continue
                flat_id = row
                if flat_id >= valid_tokens or expert < 0:
                    continue
                token = flat_id // top_k
                B_e = _dequant_b(
                    B[int(expert)],
                    b_scale[int(expert)] if b_scale is not None and b_scale.ndim >= 1 else b_scale,
                    block_shape,
                )
                out = torch.matmul(A_deq[token], B_e.t())
                if mul_routed_weight and flat_weights is not None:
                    out = out * flat_weights[flat_id]
                C[flat_id].copy_(out.to(C.dtype))
            return C

        sorted_ids = sorted_token_ids.reshape(-1).to(torch.int64)
        for block, expert in enumerate(expert_ids.reshape(-1).tolist()):
            if block * block_size >= ntpp:
                continue
            if expert < 0:
                continue
            start = block * block_size
            end = min(start + block_size, sorted_ids.numel())
            flat_ids = sorted_ids[start:end]
            flat_ids = flat_ids[flat_ids < valid_tokens]
            if flat_ids.numel() == 0:
                continue
            B_e = _dequant_b(
                B[int(expert)],
                b_scale[int(expert)] if b_scale is not None and b_scale.ndim >= 1 else b_scale,
                block_shape,
            )
            tokens = torch.div(flat_ids, top_k, rounding_mode="floor")
            out = torch.matmul(A_deq[tokens], B_e.t())
            if mul_routed_weight and flat_weights is not None:
                out = out * flat_weights[flat_ids, None]
            C[flat_ids] = out.to(C.dtype)
        return C
