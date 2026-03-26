"""FP8 grouped GEMM for MoE experts via DeepGEMM."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

import deep_gemm

from .fp8_linear import _per_token_group_quant_fp8

_GROUP_SIZE = 128


def _round_up(x: int, a: int) -> int:
    return (x + a - 1) // a * a


class Fp8MoeGroupedGemm(nn.Module):
    """FP8 grouped GEMM for MoE expert execution via deep_gemm.

    Uses ``m_grouped_fp8_gemm_nt_contiguous`` for contiguous expert layouts.
    Activations are dynamically quantized to FP8 per-token-group (group=128)
    via :func:`_per_token_group_quant_fp8`.

    The non-naive path (``sorted_token_ids`` from :class:`MoeAlign`) gathers
    rows, pads ``M`` to DeepGEMM alignment, runs grouped FP8 GEMM, then writes
    results back into ``C_bf16``. The naive align path (``sorted_token_ids is
    None``) is not supported here because it uses a block layout that does not
    map directly to DeepGEMM's per-row ``m_indices``; use BF16
    :class:`MoeGroupedGemm` or vLLM-style ``deepgemm_moe_permute`` instead.
    """

    def __init__(self) -> None:
        super().__init__()
        self._a_buf: torch.Tensor | None = None
        self._s_buf: torch.Tensor | None = None

    def forward(
        self,
        A_bf16: torch.Tensor,
        B_fp8: torch.Tensor,
        C_bf16: torch.Tensor,
        weight_scale: torch.Tensor,
        sorted_token_ids: torch.Tensor | None,
        expert_ids: torch.Tensor,
        num_tokens_post_padded: torch.Tensor,
        topk_weights: torch.Tensor | None,
        mul_routed_weight: bool,
        top_k: int,
        block_size_m: int,
    ) -> None:
        """Run grouped FP8 GEMM and write BF16 results into ``C_bf16``.

        Args:
            A_bf16: ``[M, K]`` BF16 activations.
            B_fp8: ``[E, N, K]`` FP8 expert weights (NT grouped layout).
            C_bf16: ``[M * top_k, N]`` BF16 output buffer.
            weight_scale: FP32 block scales for ``B_fp8`` (DeepGEMM layout).
            sorted_token_ids: ``[>= ntp]`` int32, from ``MoeAlign`` (non-naive).
            expert_ids: ``[>= num_blocks]`` int32, one expert id per align block.
            num_tokens_post_padded: scalar int32, padded token count ``ntp``.
            topk_weights: ``[M, top_k]`` router weights if ``mul_routed_weight``.
            mul_routed_weight: multiply routed outputs by ``topk_weights``.
            top_k: routing top-k.
            block_size_m: block size passed to ``MoeAlign`` (same as Triton MoE).
        """
        if sorted_token_ids is None:
            raise NotImplementedError(
                "Fp8MoeGroupedGemm does not support naive MoeAlign "
                "(sorted_token_ids=None); use non-naive alignment or BF16 "
                "MoeGroupedGemm.",
            )

        M, K = A_bf16.shape
        num_experts, N, Kb = B_fp8.shape
        if Kb != K:
            raise ValueError(f"K mismatch: A has K={K}, B has K={Kb}")
        if num_experts < 1 or N < 1:
            raise ValueError("B_fp8 must have positive E and N")

        num_groups = math.ceil(K / _GROUP_SIZE)
        ntp = int(
            num_tokens_post_padded.item()
            if num_tokens_post_padded.numel() == 1
            else int(num_tokens_post_padded),
        )
        align = int(deep_gemm.get_mk_alignment_for_contiguous_layout())
        m_sum = _round_up(ntp, align)

        if self._a_buf is not None and M <= self._a_buf.shape[0] and K == self._a_buf.shape[1]:
            _per_token_group_quant_fp8(A_bf16, self._a_buf[:M], self._s_buf[:M])
            a_fp8 = self._a_buf[:M]
            a_scale = self._s_buf[:M]
        else:
            a_fp8 = torch.empty(
                M, K, dtype=torch.float8_e4m3fn, device=A_bf16.device,
            )
            a_scale = torch.empty(
                M, num_groups, dtype=torch.float32, device=A_bf16.device,
            )
            _per_token_group_quant_fp8(A_bf16, a_fp8, a_scale)

        stp = sorted_token_ids[:ntp].long()
        num_valid = M * top_k
        valid = stp < num_valid
        input_rows = (stp // top_k).clamp(max=M - 1)

        a_perm = a_fp8[input_rows]
        s_perm = a_scale[input_rows]

        blk = torch.arange(ntp, device=A_bf16.device, dtype=torch.int64) // int(block_size_m)
        blk = blk.clamp(max=expert_ids.shape[0] - 1)
        expert_ids_row = expert_ids[blk].to(torch.int32)
        expert_ids_row = torch.where(
            valid,
            expert_ids_row,
            torch.full_like(expert_ids_row, -1),
        )

        if m_sum > ntp:
            pad = m_sum - ntp
            a_perm = torch.cat(
                [
                    a_perm,
                    torch.zeros(
                        pad, K, dtype=a_fp8.dtype, device=a_fp8.device,
                    ),
                ],
                dim=0,
            )
            s_perm = torch.cat(
                [
                    s_perm,
                    torch.zeros(
                        pad, num_groups, dtype=torch.float32, device=a_fp8.device,
                    ),
                ],
                dim=0,
            )
            expert_ids_row = torch.cat(
                [
                    expert_ids_row,
                    torch.full((pad,), -1, dtype=torch.int32, device=A_bf16.device),
                ],
                dim=0,
            )
        elif m_sum < a_perm.shape[0]:
            a_perm = a_perm[:m_sum]
            s_perm = s_perm[:m_sum]
            expert_ids_row = expert_ids_row[:m_sum]

        out = torch.empty(m_sum, N, dtype=torch.bfloat16, device=A_bf16.device)
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            (a_perm, s_perm),
            (B_fp8, weight_scale),
            out,
            expert_ids_row,
        )

        out_valid = out[:ntp]
        tok_slots = stp.clamp(max=C_bf16.shape[0] - 1)
        if mul_routed_weight and topk_weights is not None:
            w = topk_weights.view(-1)[tok_slots].unsqueeze(1)
            vals = out_valid * w
        else:
            vals = out_valid

        vals = torch.where(valid.unsqueeze(1), vals, torch.zeros_like(vals))
        C_bf16.zero_()
        C_bf16[tok_slots[valid]] = vals[valid]
