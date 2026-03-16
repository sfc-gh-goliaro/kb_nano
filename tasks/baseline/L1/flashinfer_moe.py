"""FlashInfer native CUDA MoE kernels with GPU architecture dispatch.

Provides a monolithic MoE forward pass that replaces the entire
align -> quant -> GEMM1 -> SiLU-mul -> quant -> GEMM2 -> sum pipeline.

Dispatch:
  sm100+ (Blackwell): flashinfer TRT-LLM kernels
  sm90   (Hopper):    flashinfer CUTLASS kernels
  other:              returns None (caller falls back to Triton)
"""

from __future__ import annotations

import torch
import torch.nn as nn

_SM = None


def _get_sm() -> int:
    global _SM
    if _SM is None:
        _SM = torch.cuda.get_device_capability()[0] * 10 + torch.cuda.get_device_capability()[1]
    return _SM


def is_available() -> bool:
    """Check if FlashInfer MoE kernels are available for the current GPU."""
    try:
        import flashinfer.fused_moe  # noqa: F401
    except ImportError:
        return False
    return _get_sm() >= 90


def swap_w13_to_w31(x: torch.Tensor) -> torch.Tensor:
    """Swap gate/up halves: W13 [gate; up] -> W31 [up; gate]."""
    return x.reshape(-1, 2, x.shape[-2] // 2, x.shape[-1]).flip(dims=[1]).reshape(x.shape)


class FlashInferFusedMoE(nn.Module):
    """Monolithic FlashInfer MoE kernel with GPU architecture dispatch.

    On sm100+ uses TRT-LLM kernels; on sm90 uses CUTLASS kernels.
    """

    def __init__(self):
        super().__init__()
        self._output_buf = None
        self._quant = None

    def forward_fp8(
        self,
        hidden_states: torch.Tensor,
        w13: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        top_k: int,
        intermediate_size: int,
        w13_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        a_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """FP8 MoE forward using FlashInfer native kernels.

        Args:
            hidden_states: [M, K] bfloat16
            w13: [E, 2*N, K] float8_e4m3fn (W31 layout for TRT-LLM)
            w2:  [E, K, N] float8_e4m3fn
            topk_weights: [M, top_k] float32
            topk_ids: [M, top_k] int32
            num_experts: total number of experts
            top_k: experts per token
            intermediate_size: N (per-TP shard)
            w13_scale: [E, ceil(2N/128), ceil(K/128)] float32
            w2_scale:  [E, ceil(K/128), ceil(N/128)] float32
            a_scale: pre-computed [M, K//128] activation scales (optional,
                     only needed for TRT-LLM routed kernel)
        """
        sm = _get_sm()

        if sm >= 100:
            return self._forward_trtllm_fp8(
                hidden_states, w13, w2, topk_weights, topk_ids,
                num_experts, top_k, intermediate_size,
                w13_scale, w2_scale, a_scale,
            )
        else:
            return self._forward_cutlass_fp8(
                hidden_states, w13, w2, topk_weights, topk_ids,
                num_experts, top_k, intermediate_size,
                w13_scale, w2_scale,
            )

    def forward_bf16(
        self,
        hidden_states: torch.Tensor,
        w13: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        top_k: int,
        intermediate_size: int,
        router_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """BF16 MoE forward using FlashInfer native kernels.

        Args:
            hidden_states: [M, K] bfloat16
            w13: [E, 2*N, K] bfloat16 (W31 layout for TRT-LLM)
            w2:  [E, K, N] bfloat16
            topk_weights: [M, top_k] float32
            topk_ids: [M, top_k] int32
            num_experts: total number of experts
            top_k: experts per token
            intermediate_size: N (per-TP shard)
            router_logits: [M, num_experts] (needed for TRT-LLM BF16 path)
        """
        sm = _get_sm()

        if sm >= 100:
            return self._forward_trtllm_bf16(
                hidden_states, w13, w2,
                num_experts, top_k, intermediate_size,
                router_logits,
            )
        else:
            return self._forward_cutlass_bf16(
                hidden_states, w13, w2, topk_weights, topk_ids,
                num_experts, top_k, intermediate_size,
            )

    def _forward_trtllm_fp8(
        self, hidden_states, w13, w2, topk_weights, topk_ids,
        num_experts, top_k, intermediate_size, w13_scale, w2_scale, a_scale,
    ):
        from flashinfer.fused_moe import trtllm_fp8_block_scale_routed_moe

        M, K = hidden_states.shape

        # Quantize hidden_states to FP8 if not already
        if hidden_states.dtype != torch.float8_e4m3fn:
            if self._quant is None:
                from .fp8_quant import PerTokenGroupQuantFP8
                # MoE kernels require float32 scales, not packed E8M0 int32
                self._quant = PerTokenGroupQuantFP8(group_size=128, use_packed_e8m0=False)
            hidden_fp8, a_scale = self._quant(hidden_states)
        else:
            hidden_fp8 = hidden_states
            assert a_scale is not None

        # TRT-LLM expects hidden_states_scale as [K//128, M] (transposed)
        a_scale_t = a_scale.t().contiguous()

        # Pack topk_ids and topk_weights: (expert_id << 16) | weight_bf16_as_int16
        packed = (topk_ids.to(torch.int32) << 16) | topk_weights.to(torch.bfloat16).view(torch.int16).to(torch.int32)

        result = trtllm_fp8_block_scale_routed_moe(
            topk_ids=packed,
            routing_bias=None,
            hidden_states=hidden_fp8,
            hidden_states_scale=a_scale_t,
            gemm1_weights=w13,
            gemm1_weights_scale=w13_scale,
            gemm2_weights=w2,
            gemm2_weights_scale=w2_scale,
            num_experts=num_experts,
            top_k=top_k,
            n_group=None,
            topk_group=None,
            intermediate_size=intermediate_size,
            local_expert_offset=0,
            local_num_experts=num_experts,
            routed_scaling_factor=None,
            routing_method_type=1,
            use_shuffled_weight=False,
            weight_layout=0,
        )
        return result

    def _forward_cutlass_fp8(
        self, hidden_states, w13, w2, topk_weights, topk_ids,
        num_experts, top_k, intermediate_size, w13_scale, w2_scale,
    ):
        from flashinfer.fused_moe import cutlass_fused_moe
        from flashinfer.fused_moe.core import ActivationType

        M, K = hidden_states.shape

        output = self._get_output_buf(M, K, hidden_states.device, hidden_states.dtype)

        cutlass_fused_moe(
            input=hidden_states,
            token_selected_experts=topk_ids.to(torch.int32),
            token_final_scales=topk_weights,
            fc1_expert_weights=w13,
            fc2_expert_weights=w2,
            output_dtype=hidden_states.dtype,
            quant_scales=[w13_scale, w2_scale],
            output=output,
            use_deepseek_fp8_block_scale=True,
            activation_type=ActivationType.Swiglu,
        )
        return output

    def _forward_trtllm_bf16(
        self, hidden_states, w13, w2,
        num_experts, top_k, intermediate_size, router_logits,
    ):
        from flashinfer.fused_moe import trtllm_bf16_moe

        assert router_logits is not None, (
            "TRT-LLM BF16 MoE requires router_logits"
        )

        result = trtllm_bf16_moe(
            routing_logits=router_logits,
            routing_bias=None,
            hidden_states=hidden_states,
            gemm1_weights=w13,
            gemm2_weights=w2,
            num_experts=num_experts,
            top_k=top_k,
            n_group=None,
            topk_group=None,
            intermediate_size=intermediate_size,
            local_expert_offset=0,
            local_num_experts=num_experts,
            routed_scaling_factor=None,
            routing_method_type=0,
            use_shuffled_weight=True,
            weight_layout=2,
            enable_pdl=True,
        )
        return result

    def _forward_cutlass_bf16(
        self, hidden_states, w13, w2, topk_weights, topk_ids,
        num_experts, top_k, intermediate_size,
    ):
        from flashinfer.fused_moe import cutlass_fused_moe
        from flashinfer.fused_moe.core import ActivationType

        M, K = hidden_states.shape

        output = self._get_output_buf(M, K, hidden_states.device, hidden_states.dtype)

        cutlass_fused_moe(
            input=hidden_states,
            token_selected_experts=topk_ids.to(torch.int32),
            token_final_scales=topk_weights,
            fc1_expert_weights=w13,
            fc2_expert_weights=w2,
            output_dtype=hidden_states.dtype,
            quant_scales=None,
            output=output,
            activation_type=ActivationType.Swiglu,
        )
        return output

    def _get_output_buf(self, M, K, device, dtype):
        if self._output_buf is None or self._output_buf.size(0) < M or self._output_buf.size(1) < K:
            self._output_buf = torch.empty(M, K, device=device, dtype=dtype)
        return self._output_buf[:M, :K]
