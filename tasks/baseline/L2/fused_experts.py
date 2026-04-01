"""Fused MoE experts: two grouped GEMMs with SiLU-mul in between.

Supports both BF16 and FP8 W8A8 block-scaled expert weights.
FP8 path quantizes activations to FP8 per-token-group before each GEMM,
matching vLLM's fused_experts_impl.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..L1.fp8_linear import _per_token_group_quant_fp8
from ..L1.moe_align import MoeAlign
from ..L1.moe_grouped_gemm import MoeGroupedGemm
from ..L1.moe_sum import MoeSum
from ..L1.silu_and_mul import SiluAndMul

SPARSITY_FACTOR = 4
_FP8_GROUP_SIZE = 128


class FusedExperts(nn.Module):
    """Fused MoE experts: two grouped GEMMs with SiLU-mul in between.

    Supports FP8 W8A8 block-scaled expert weights.  When use_fp8_w8a8=True,
    activations are dynamically quantized to FP8 per-token-group (group=128)
    before each GEMM, and the Triton kernel does FP8 dot products with FP32
    accumulation + block-wise scale dequantization — matching vLLM exactly.
    """

    def __init__(self):
        super().__init__()
        self.moe_align = MoeAlign()
        self.moe_grouped_gemm = MoeGroupedGemm()
        self.act_fn = SiluAndMul()
        self.moe_sum = MoeSum()
        self._cache13 = None
        self._a_fp8_1 = None
        self._a_scale_1 = None
        self._a_fp8_2 = None
        self._a_scale_2 = None

    def _get_cache13(self, total_elems, device, dtype):
        if self._cache13 is None or self._cache13.numel() < total_elems:
            self._cache13 = torch.empty(total_elems, device=device, dtype=dtype)
        return self._cache13[:total_elems]

    def _get_fp8_bufs(self, buf_id, M, K, device):
        attr_a = f"_a_fp8_{buf_id}"
        attr_s = f"_a_scale_{buf_id}"
        num_groups = math.ceil(K / _FP8_GROUP_SIZE)
        existing_a = getattr(self, attr_a)
        if existing_a is None or existing_a.size(0) < M or existing_a.size(1) < K:
            setattr(self, attr_a, torch.empty(M, K, dtype=torch.float8_e4m3fn, device=device))
            setattr(self, attr_s, torch.empty(M, num_groups, dtype=torch.float32, device=device))
        a = getattr(self, attr_a)
        s = getattr(self, attr_s)
        return a[:M, :K], s[:M, :num_groups]

    def forward(
        self,
        hidden_states: torch.Tensor,
        w13: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        w13_scale: torch.Tensor | None = None,
        w2_scale: torch.Tensor | None = None,
        use_fp8_w8a8: bool = False,
        block_shape: list[int] | None = None,
    ) -> torch.Tensor:
        M, K = hidden_states.size()
        E, N2, _ = w13.size()
        N = N2 // 2
        top_k = topk_ids.size(1)

        config = self.moe_grouped_gemm.get_config(M, N2)

        use_naive = (M * top_k * SPARSITY_FACTOR <= num_experts)

        sorted_token_ids, expert_ids, num_tokens_post_padded = self.moe_align(
            topk_ids, config["BLOCK_SIZE_M"], num_experts, naive=use_naive,
        )

        # Reuse memory between cache1 and cache3 (vLLM optimization)
        cache13_size = M * top_k * max(N2, K)
        cache13_flat = self._get_cache13(cache13_size, hidden_states.device, hidden_states.dtype)
        intermediate1 = cache13_flat[:M * top_k * N2].view(M * top_k, N2)
        intermediate3 = cache13_flat[:M * top_k * K].view(M * top_k, K)

        # First GEMM: hidden_states x w13 -> intermediate1
        if use_fp8_w8a8:
            a_fp8, a_scale = self._get_fp8_bufs(1, M, K, hidden_states.device)
            _per_token_group_quant_fp8(hidden_states, a_fp8, a_scale)
            gemm1_input = a_fp8
            gemm1_a_scale = a_scale
        else:
            gemm1_input = hidden_states
            gemm1_a_scale = None

        self.moe_grouped_gemm(
            gemm1_input, w13, intermediate1,
            topk_weights, sorted_token_ids, expert_ids,
            num_tokens_post_padded,
            mul_routed_weight=False, top_k=top_k, config=config,
            a_scale=gemm1_a_scale, b_scale=w13_scale,
            use_fp8_w8a8=use_fp8_w8a8, block_shape=block_shape,
        )

        # Activation: SiLU-and-Mul on intermediate1 [M*top_k, 2*N] -> [M*top_k, N]
        intermediate2 = self.act_fn(intermediate1)

        # Second GEMM: intermediate2 x w2 -> intermediate3
        if use_fp8_w8a8:
            a2_fp8, a2_scale = self._get_fp8_bufs(2, M * top_k, N, hidden_states.device)
            _per_token_group_quant_fp8(intermediate2, a2_fp8, a2_scale)
            gemm2_input = a2_fp8
            gemm2_a_scale = a2_scale
        else:
            gemm2_input = intermediate2
            gemm2_a_scale = None

        self.moe_grouped_gemm(
            gemm2_input, w2, intermediate3,
            topk_weights, sorted_token_ids, expert_ids,
            num_tokens_post_padded,
            mul_routed_weight=True, top_k=1, config=config,
            a_scale=gemm2_a_scale, b_scale=w2_scale,
            use_fp8_w8a8=use_fp8_w8a8, block_shape=block_shape,
        )

        return self.moe_sum(intermediate3, top_k)
