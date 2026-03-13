"""Fused MoE experts: two grouped GEMMs with SiLU-mul in between.

Supports both BF16 and FP8 W8A8 block-scaled paths. When block_size is
provided at init, FP8 quantization and GEMM ops are initialized and the
forward path expects scale tensors alongside the weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.moe_align import MoeAlign
from ..L1.moe_grouped_gemm import MoeGroupedGemm, _get_default_config
from ..L1.moe_sum import MoeSum
from ..L1.silu_and_mul import SiluAndMul

SPARSITY_FACTOR = 4


class FusedExperts(nn.Module):
    """Fused MoE experts: two grouped GEMMs with SiLU-mul in between.

    Args:
        block_size: optional (bn, bk) FP8 quantization block dimensions.
                    When set, initializes the FP8 activation quantization
                    and FP8 grouped GEMM ops.

    forward args:
        hidden_states: [M, K]
        w13: [E, 2*intermediate, K]
        w2:  [E, K, intermediate]
        topk_weights: [M, top_k]
        topk_ids:     [M, top_k]
        num_experts: E
        w13_scale_inv: optional [E, scale_rows, scale_cols] float32 (FP8 path)
        w2_scale_inv:  optional [E, scale_rows, scale_cols] float32 (FP8 path)

    Returns:
        output: [M, K]
    """

    def __init__(self, block_size: tuple[int, int] | None = None):
        super().__init__()
        self.block_size = block_size
        self.moe_align = MoeAlign()
        self.act_fn = SiluAndMul()
        self.moe_sum = MoeSum()
        self._cache1 = None
        self._cache3 = None
        self._naive_num_tokens_post_padded = None

        if block_size is not None:
            from ..L1.fp8_quant import PerTokenGroupQuantFP8
            from ..L1.fp8_moe_grouped_gemm import FP8MoeGroupedGemm
            self.quant = PerTokenGroupQuantFP8(group_size=block_size[1])
            self.fp8_moe_grouped_gemm = FP8MoeGroupedGemm()

        self.moe_grouped_gemm = MoeGroupedGemm()

    def _get_cache(self, name, size, device, dtype):
        cache = getattr(self, name)
        if cache is None or cache.size(0) < size[0] or cache.size(1) < size[1]:
            cache = torch.empty(size, device=device, dtype=dtype)
            setattr(self, name, cache)
        return cache[:size[0], :size[1]]

    def _naive_align(self, topk_ids, block_size_m, num_experts):
        """Fast path: skip full alignment when tokens * top_k is very small."""
        numel = topk_ids.numel()
        max_num_tokens_padded = numel * block_size_m
        expert_ids = topk_ids.view(-1).to(torch.int32)
        if (self._naive_num_tokens_post_padded is None
                or self._naive_num_tokens_post_padded.device != topk_ids.device):
            self._naive_num_tokens_post_padded = torch.empty(
                1, dtype=torch.int32, device=topk_ids.device,
            )
        self._naive_num_tokens_post_padded.fill_(max_num_tokens_padded)
        sorted_token_ids = None
        return sorted_token_ids, expert_ids, self._naive_num_tokens_post_padded

    def forward(
        self,
        hidden_states: torch.Tensor,
        w13: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        w13_scale_inv: torch.Tensor | None = None,
        w2_scale_inv: torch.Tensor | None = None,
    ) -> torch.Tensor:
        use_fp8 = w13_scale_inv is not None
        M, K = hidden_states.size()
        E, N2, _ = w13.size()
        N = N2 // 2
        top_k = topk_ids.size(1)

        if use_fp8:
            from ..L1.fp8_moe_grouped_gemm import _get_fp8_moe_config
            config = _get_fp8_moe_config(M, N2)
        else:
            config = _get_default_config(M, N2)

        use_naive = (M * top_k * SPARSITY_FACTOR <= num_experts)

        if use_naive:
            sorted_token_ids, expert_ids, num_tokens_post_padded = \
                self._naive_align(topk_ids, config["BLOCK_SIZE_M"], num_experts)
        else:
            sorted_token_ids, expert_ids, num_tokens_post_padded = self.moe_align(
                topk_ids, config["BLOCK_SIZE_M"], num_experts,
            )

        if use_fp8:
            return self._forward_fp8(
                hidden_states, w13, w13_scale_inv, w2, w2_scale_inv,
                topk_weights, sorted_token_ids, expert_ids,
                num_tokens_post_padded, top_k, M, K, N2, config,
            )
        else:
            return self._forward_bf16(
                hidden_states, w13, w2,
                topk_weights, sorted_token_ids, expert_ids,
                num_tokens_post_padded, top_k, M, K, N2, config,
            )

    def _forward_bf16(
        self, hidden_states, w13, w2,
        topk_weights, sorted_token_ids, expert_ids,
        num_tokens_post_padded, top_k, M, K, N2, config,
    ):
        intermediate1 = self._get_cache(
            "_cache1", (M * top_k, N2),
            hidden_states.device, hidden_states.dtype,
        )

        self.moe_grouped_gemm(
            hidden_states, w13, intermediate1,
            topk_weights, sorted_token_ids, expert_ids,
            num_tokens_post_padded,
            mul_routed_weight=False, top_k=top_k, config=config,
        )

        intermediate2 = self.act_fn(intermediate1)

        intermediate3 = self._get_cache(
            "_cache3", (M * top_k, K),
            hidden_states.device, hidden_states.dtype,
        )

        self.moe_grouped_gemm(
            intermediate2, w2, intermediate3,
            topk_weights, sorted_token_ids, expert_ids,
            num_tokens_post_padded,
            mul_routed_weight=True, top_k=1, config=config,
        )

        return self.moe_sum(intermediate3, top_k)

    def _forward_fp8(
        self, hidden_states, w13, w13_scale_inv, w2, w2_scale_inv,
        topk_weights, sorted_token_ids, expert_ids,
        num_tokens_post_padded, top_k, M, K, N2, config,
    ):
        a_fp8, a_scale = self.quant(hidden_states)

        intermediate1 = self._get_cache(
            "_cache1", (M * top_k, N2),
            hidden_states.device, torch.bfloat16,
        )

        self.fp8_moe_grouped_gemm(
            a_fp8, a_scale, w13, w13_scale_inv, intermediate1,
            topk_weights, sorted_token_ids, expert_ids,
            num_tokens_post_padded,
            mul_routed_weight=False, top_k=top_k,
            block_size=self.block_size, config=config,
        )

        intermediate2 = self.act_fn(intermediate1)

        a2_fp8, a2_scale = self.quant(intermediate2)

        intermediate3 = self._get_cache(
            "_cache3", (M * top_k, K),
            hidden_states.device, torch.bfloat16,
        )

        self.fp8_moe_grouped_gemm(
            a2_fp8, a2_scale, w2, w2_scale_inv, intermediate3,
            topk_weights, sorted_token_ids, expert_ids,
            num_tokens_post_padded,
            mul_routed_weight=True, top_k=1,
            block_size=self.block_size, config=config,
        )

        return self.moe_sum(intermediate3, top_k)
