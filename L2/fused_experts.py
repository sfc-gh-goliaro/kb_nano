"""Fused MoE experts: two grouped GEMMs with SiLU-mul in between."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..L1.moe_align import MoeAlign
from ..L1.moe_grouped_gemm import MoeGroupedGemm, _get_default_config


class FusedExperts(nn.Module):
    """Fused MoE experts: two grouped GEMMs with SiLU-mul in between.

    Args (to forward):
        hidden_states: [M, K]
        w13: [E, 2*intermediate, K] -- gate (w1) and up (w3) stacked on dim 1
        w2:  [E, K, intermediate]
        topk_weights: [M, top_k]
        topk_ids:     [M, top_k]
        num_experts: E

    Returns:
        output: [M, K]
    """

    def __init__(self):
        super().__init__()
        self.moe_align = MoeAlign()
        self.moe_grouped_gemm = MoeGroupedGemm()

    def forward(
        self,
        hidden_states: torch.Tensor,
        w13: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
    ) -> torch.Tensor:
        M, K = hidden_states.size()
        E, N2, _ = w13.size()
        N = N2 // 2
        top_k = topk_ids.size(1)

        config = _get_default_config(M)

        sorted_token_ids, expert_ids, num_tokens_post_padded = self.moe_align(
            topk_ids, config["BLOCK_SIZE_M"], num_experts,
        )

        intermediate1 = torch.empty(
            M * top_k, N2, device=hidden_states.device, dtype=hidden_states.dtype,
        )

        self.moe_grouped_gemm(
            hidden_states, w13, intermediate1,
            topk_weights, sorted_token_ids, expert_ids,
            num_tokens_post_padded,
            mul_routed_weight=False, top_k=top_k, config=config,
        )

        gate = intermediate1[:, :N]
        up = intermediate1[:, N:]
        intermediate2 = F.silu(gate) * up

        intermediate3 = torch.empty(
            M * top_k, K, device=hidden_states.device, dtype=hidden_states.dtype,
        )

        self.moe_grouped_gemm(
            intermediate2, w2, intermediate3,
            topk_weights, sorted_token_ids, expert_ids,
            num_tokens_post_padded,
            mul_routed_weight=True, top_k=1, config=config,
        )

        return intermediate3.view(M, top_k, K).sum(dim=1)
