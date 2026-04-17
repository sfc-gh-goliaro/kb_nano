"""DeepSeek MoE router gate matmul (BF16 x BF16 -> FP32) with vLLM parity.

Mirrors vLLM's
``vllm/model_executor/layers/fused_moe/router/gate_linear.py:GateLinear``
which has a three-tier dispatch:

1. **DSV3 specialized kernel** — Hopper/Blackwell, ``num_experts in {256, 384}``,
   ``hidden_size == 7168``, batch ``<= 16``. BF16 x BF16 -> FP32 fused kernel
   that internally accumulates in FP32. Routes to
   ``vllm._custom_ops.dsv3_router_gemm``.
2. **cuBLAS BF16 -> FP32** — Hopper/Blackwell + BF16 weight + FP32 out. Routes
   to ``vllm._custom_ops.router_gemm_bf16_fp32``.
3. **PyTorch fallback** — vanilla ``F.linear`` at the input dtype, with cast
   back to FP32 at the end.

Matching vLLM exactly here matters for **correctness** of the grouped-topk
router: kb_nano's previous "promote both to FP32 then matmul" path was
strictly more precise but used a different accumulation order than vLLM's
specialized kernels, which flipped near-tie expert / group selections in
the noaux_tc path (see audit notes).
"""

from __future__ import annotations

import functools

import torch


# Cache the resolved kernel handles + capability bits so we only pay the
# import / attribute lookups + capability query once per process.
@functools.cache
def _maybe_load_router_kernels() -> tuple[
    object | None,  # dsv3_router_gemm callable
    object | None,  # router_gemm_bf16_fp32 callable
    bool,           # is_hopper_or_blackwell
]:
    if not torch.cuda.is_available():
        return None, None, False

    try:
        from vllm import _custom_ops as _vllm_ops
    except Exception:
        return None, None, False

    dsv3 = getattr(_vllm_ops, "dsv3_router_gemm", None)
    bf16_fp32 = getattr(_vllm_ops, "router_gemm_bf16_fp32", None)

    # Same gate vLLM uses (see ``GateLinear.__init__``):
    # ``current_platform.is_device_capability((9, 0))`` (Hopper) or
    # ``current_platform.is_device_capability_family(100)`` (Blackwell).
    cap = torch.cuda.get_device_capability()
    is_hopper_or_blackwell = (cap[0], cap[1]) == (9, 0) or cap[0] == 10

    return dsv3, bf16_fp32, is_hopper_or_blackwell


def gate_linear_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Compute router logits with vLLM-parity dispatch.

    Args:
        x: ``(num_tokens, hidden_size)`` activations (BF16).
        weight: ``(num_experts, hidden_size)`` gate weight (BF16).
        out_dtype: Desired output dtype (FP32 for DeepSeek-V3 monolithic).

    Returns:
        ``(num_tokens, num_experts)`` router logits in ``out_dtype``.
    """
    num_tokens = x.shape[0]
    num_experts = weight.shape[0]
    hidden_size = weight.shape[1]

    dsv3, bf16_fp32, is_hopper_or_blackwell = _maybe_load_router_kernels()

    bf16_input = x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16

    # Tier 1: DSV3 specialized kernel. Matches vLLM's ``GateLinear.forward``
    # which dispatches here for any ``out_dtype`` (BF16 or FP32) as long as
    # ``batch<=16`` and the hidden/num_experts dims match. Previously we
    # also required ``out_dtype == torch.float32``, which diverted the BF16
    # router path (used by non-monolithic FP8 MoE in DeepSeek-V3.2) to the
    # fallback F.linear — producing slightly different router_logits and
    # boundary-flipping the grouped-topk at layers 3+.
    if (
        is_hopper_or_blackwell
        and dsv3 is not None
        and bf16_input
        and num_tokens <= 16
        and num_experts in (256, 384)
        and hidden_size == 7168
    ):
        return dsv3(x, weight, out_dtype)

    # Tier 2: cuBLAS BF16 x BF16 -> FP32.
    if (
        is_hopper_or_blackwell
        and bf16_fp32 is not None
        and bf16_input
        and out_dtype == torch.float32
    ):
        return bf16_fp32(x, weight)

    # Tier 3: F.linear fallback. Match vLLM's behaviour (cast input to
    # weight dtype, then cast output to out_dtype).
    if x.dtype != weight.dtype:
        x = x.to(weight.dtype)
    out = torch.nn.functional.linear(x, weight)
    if out.dtype != out_dtype:
        out = out.to(out_dtype)
    return out
