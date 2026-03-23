"""Fused MoE experts: two grouped GEMMs with SiLU-mul in between.

Supports FP8 w8a8 with block-scale quantization via two backends:
  1. FlashInfer CUTLASS (preferred on Hopper+): fused routing + GEMM + activation
  2. Triton fallback: manual align/quant/GEMM with per-block FP8 scales
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ..L1.moe_align import MoeAlign
from ..L1.moe_grouped_gemm import MoeGroupedGemm
from ..L1.moe_sum import MoeSum
from ..L1.silu_and_mul import SiluAndMul

SPARSITY_FACTOR = 4

_FP8_BLOCK = 128
_FP8_INFO = torch.finfo(torch.float8_e4m3fn)

_USE_FLASHINFER_CUTLASS: bool | None = None

def _check_flashinfer_cutlass() -> bool:
    try:
        from flashinfer.fused_moe import cutlass_fused_moe  # noqa: F401
        from flashinfer.fused_moe.core import ActivationType  # noqa: F401
        return True
    except (ImportError, ModuleNotFoundError):
        return False

def use_flashinfer_cutlass() -> bool:
    global _USE_FLASHINFER_CUTLASS
    if _USE_FLASHINFER_CUTLASS is None:
        if os.environ.get("KB_NANO_DISABLE_FLASHINFER", "0") == "1":
            _USE_FLASHINFER_CUTLASS = False
        else:
            _USE_FLASHINFER_CUTLASS = _check_flashinfer_cutlass()
    return _USE_FLASHINFER_CUTLASS


@triton.jit
def _moe_act_quant_kernel(
    x_ptr, out_ptr, scale_ptr,
    stride_x_row, stride_out_row, stride_s_row,
    num_cols,
    fp8_max: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    """Single-kernel per-token-group FP8 quantization for MoE activations."""
    pid = tl.program_id(0)
    groups_per_row = num_cols // GROUP_SIZE
    row = pid // groups_per_row
    group = pid % groups_per_row

    x_base = x_ptr + row * stride_x_row + group * GROUP_SIZE
    cols = tl.arange(0, GROUP_SIZE)
    x = tl.load(x_base + cols).to(tl.float32)

    absmax = tl.max(tl.abs(x))
    absmax = tl.maximum(absmax, 1e-10)
    scale = absmax / fp8_max

    x_scaled = x / scale
    x_fp8 = tl.clamp(x_scaled, -fp8_max, fp8_max).to(out_ptr.dtype.element_ty)

    out_base = out_ptr + row * stride_out_row + group * GROUP_SIZE
    tl.store(out_base + cols, x_fp8)

    scale_base = scale_ptr + row * stride_s_row + group
    tl.store(scale_base, scale)


class _MoESharedBufs:
    """Shared mutable container for MoE intermediate and FP8 buffers.

    All FusedExperts instances in a model share one _MoESharedBufs so that
    only a single set of buffers is allocated. Layers execute sequentially.
    """
    __slots__ = ("cache1", "cache3", "a1_q", "a1_s", "a2_q", "a2_s")

    def __init__(self):
        self.cache1: torch.Tensor | None = None
        self.cache3: torch.Tensor | None = None
        self.a1_q: torch.Tensor | None = None
        self.a1_s: torch.Tensor | None = None
        self.a2_q: torch.Tensor | None = None
        self.a2_s: torch.Tensor | None = None

    def get_cache(self, which: str, size: tuple, device, dtype) -> torch.Tensor:
        cache = getattr(self, which)
        if cache is None or cache.size(0) < size[0] or cache.size(1) < size[1]:
            cache = torch.empty(size, device=device, dtype=dtype)
            setattr(self, which, cache)
        return cache[:size[0], :size[1]]


def _quant_fp8_inplace(
    x: torch.Tensor, out_q: torch.Tensor, out_s: torch.Tensor,
) -> None:
    """Quantize activations to FP8 using pre-allocated buffers (zero temp memory)."""
    M, K = x.shape
    groups_per_row = K // _FP8_BLOCK
    _moe_act_quant_kernel[(M * groups_per_row,)](
        x, out_q, out_s,
        x.stride(0), out_q.stride(0), out_s.stride(0),
        K,
        fp8_max=_FP8_INFO.max,
        GROUP_SIZE=_FP8_BLOCK,
    )


class FusedExperts(nn.Module):
    """Fused MoE experts with FlashInfer CUTLASS (preferred) or Triton fallback.

    Args (to forward):
        hidden_states: [M, K]
        w13: [E, 2*intermediate, K] -- W31 order for FlashInfer, W13 for Triton
        w2:  [E, K, intermediate]
        topk_weights: [M, top_k]
        topk_ids:     [M, top_k]
        num_experts: E (global number of experts for Triton; local for FlashInfer)
        w13_scale: [E, ceil(2*intermediate/128), ceil(K/128)] or None
        w2_scale:  [E, ceil(K/128), ceil(intermediate/128)] or None
        ep_size: expert parallel world size (default 1)
        ep_rank: expert parallel rank (default 0)

    Returns:
        output: [M, K]
    """

    def __init__(self):
        super().__init__()
        self.moe_align = MoeAlign()
        self.moe_grouped_gemm = MoeGroupedGemm()
        self.act_fn = SiluAndMul()
        self.moe_sum = MoeSum()
        self._shared_bufs = _MoESharedBufs()
        self._use_flashinfer = use_flashinfer_cutlass()

    def set_shared_bufs(self, bufs: _MoESharedBufs):
        self._shared_bufs = bufs

    def _ensure_fp8_bufs(self, M: int, K: int, N: int, top_k: int, device: torch.device):
        """Ensure FP8 buffers are large enough. K = hidden_size, N = 2*intermediate."""
        bufs = self._shared_bufs
        n_groups_k = K // _FP8_BLOCK
        if bufs.a1_q is None or bufs.a1_q.size(0) < M:
            bufs.a1_q = torch.empty(M, K, dtype=torch.float8_e4m3fn, device=device)
            bufs.a1_s = torch.empty(M, n_groups_k, dtype=torch.float32, device=device)
        a2_rows = M * top_k
        n_groups_n2 = (N // 2) // _FP8_BLOCK
        if bufs.a2_q is None or bufs.a2_q.size(0) < a2_rows:
            bufs.a2_q = torch.empty(a2_rows, N // 2, dtype=torch.float8_e4m3fn, device=device)
            bufs.a2_s = torch.empty(a2_rows, n_groups_n2, dtype=torch.float32, device=device)

    def _forward_flashinfer(
        self,
        hidden_states: torch.Tensor,
        w13: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        w13_scale: torch.Tensor | None,
        w2_scale: torch.Tensor | None,
        ep_size: int,
        ep_rank: int,
    ) -> torch.Tensor:
        from flashinfer.fused_moe import cutlass_fused_moe
        from flashinfer.fused_moe.core import ActivationType

        M, K = hidden_states.size()
        is_fp8 = w13.dtype == torch.float8_e4m3fn

        if is_fp8 and w13_scale is not None:
            quant_scales = [w13_scale, w2_scale]
        else:
            quant_scales = None

        output = torch.empty(M, K, dtype=hidden_states.dtype,
                             device=hidden_states.device)

        cutlass_fused_moe(
            input=hidden_states,
            token_selected_experts=topk_ids.to(torch.int),
            token_final_scales=topk_weights.float(),
            fc1_expert_weights=w13,
            fc2_expert_weights=w2,
            output_dtype=hidden_states.dtype,
            quant_scales=quant_scales,
            ep_size=ep_size,
            ep_rank=ep_rank,
            output=output,
            activation_type=ActivationType.Swiglu,
            use_deepseek_fp8_block_scale=(is_fp8 and w13_scale is not None),
        )
        return output

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
        ep_size: int = 1,
        ep_rank: int = 0,
    ) -> torch.Tensor:
        if self._use_flashinfer:
            return self._forward_flashinfer(
                hidden_states, w13, w2, topk_weights, topk_ids,
                num_experts, w13_scale, w2_scale, ep_size, ep_rank,
            )

        M, K = hidden_states.size()
        E, N2, _ = w13.size()
        N = N2 // 2
        top_k = topk_ids.size(1)

        is_fp8 = w13.dtype == torch.float8_e4m3fn
        out_dtype = hidden_states.dtype
        block_shape = [_FP8_BLOCK, _FP8_BLOCK] if is_fp8 else None

        config = self.moe_grouped_gemm.get_config(M, N2)

        use_naive = (M * top_k * SPARSITY_FACTOR <= num_experts)

        sorted_token_ids, expert_ids, num_tokens_post_padded = self.moe_align(
            topk_ids, config["BLOCK_SIZE_M"], num_experts, naive=use_naive,
        )

        sbufs = self._shared_bufs
        intermediate1 = sbufs.get_cache(
            "cache1", (M * top_k, N2),
            hidden_states.device, out_dtype,
        )

        if is_fp8:
            self._ensure_fp8_bufs(M, K, N2, top_k, hidden_states.device)
            a1_q = sbufs.a1_q[:M]
            a1_s = sbufs.a1_s[:M]
            _quant_fp8_inplace(hidden_states, a1_q, a1_s)
        else:
            a1_q, a1_s = hidden_states, None

        self.moe_grouped_gemm(
            a1_q, w13, intermediate1,
            topk_weights, sorted_token_ids, expert_ids,
            num_tokens_post_padded,
            mul_routed_weight=False, top_k=top_k, config=config,
            A_scale=a1_s, B_scale=w13_scale, block_shape=block_shape,
        )

        intermediate2 = self.act_fn(intermediate1)

        intermediate3 = sbufs.get_cache(
            "cache3", (M * top_k, K),
            hidden_states.device, out_dtype,
        )

        Mt = M * top_k
        if is_fp8:
            a2_q = sbufs.a2_q[:Mt]
            a2_s = sbufs.a2_s[:Mt]
            _quant_fp8_inplace(intermediate2, a2_q, a2_s)
        else:
            a2_q, a2_s = intermediate2, None

        self.moe_grouped_gemm(
            a2_q, w2, intermediate3,
            topk_weights, sorted_token_ids, expert_ids,
            num_tokens_post_padded,
            mul_routed_weight=True, top_k=1, config=config,
            A_scale=a2_s, B_scale=w2_scale, block_shape=block_shape,
        )

        return self.moe_sum(intermediate3, top_k)
