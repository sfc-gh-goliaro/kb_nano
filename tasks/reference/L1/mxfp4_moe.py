"""MXFP4-native fused MoE primitive backed by the OAI Triton kernels.

This reference keeps a local copy of the kb-nano baseline implementation
instead of importing ``kb_nano.tasks.baseline.L1.mxfp4_moe``. It intentionally
stays in the native packed MXFP4 representation and relies on the
``triton_kernels`` helper package, but it does not call FlashAttention,
FlashInfer, or DeepGEMM.

This module is the L1 wrapper around ``triton_kernels.matmul_ogs`` for
MXFP4-quantized expert weights with OAI-style SwiGLU activation. It owns
all of the routing/quantization/swizzling logic that GPT-OSS needs so
that the L2 ``GptOssMoE`` module can stay pure-composition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import triton
import triton.language as tl
import triton_kernels.matmul_ogs_details.opt_flags as opt_flags
import triton_kernels.swiglu as tk_swiglu
from triton_kernels.matmul_ogs import (
    FlexCtx,
    FnSpecs,
    FusedActivation,
    GatherIndx,
    PrecisionConfig,
    RoutingData,
    ScatterIndx,
    matmul_ogs,
)
from triton_kernels.numerics import InFlexData
from triton_kernels.tensor import (
    FP4,
    SparseMatrix,
    convert_layout,
    make_ragged_tensor_metadata,
    wrap_torch_tensor,
)
from triton_kernels.tensor_details import layout
from triton_kernels.topk import topk


# ---------------------------------------------------------------------------
# Quant config
# ---------------------------------------------------------------------------


@dataclass
class Mxfp4MoEQuantConfig:
    """Minimal quant config carrying the per-MoE precision/bias tensors.

    Attribute names match the subset consumed by the fused expert wrapper
    (``w{1,2}_precision`` and ``w{1,2}_bias``).
    """

    w1_precision: Any  # triton_kernels.matmul_ogs.PrecisionConfig
    w2_precision: Any  # triton_kernels.matmul_ogs.PrecisionConfig
    w1_bias: torch.Tensor | None = None
    w2_bias: torch.Tensor | None = None


# ---------------------------------------------------------------------------
# Weight swizzling
# ---------------------------------------------------------------------------


def _swizzle_mxfp4(quant_tensor: torch.Tensor, scale: torch.Tensor, num_warps: int):
    """Swizzle MXFP4 weight + E8M0 scales into the layout matmul_ogs wants.

    Returns ``(packed_tensor, in_flex_data, scale_tensor)`` where the two
    tensor returns are ``triton_kernels.tensor.Tensor`` wrappers, ready
    to be plugged into a ``PrecisionConfig``.

    This prepares weights for the packed MXFP4 layout expected by
    ``triton_kernels.matmul_ogs``.
    """
    cap = torch.cuda.get_device_capability()

    value_layout_opts: dict[str, Any] = {}
    scale_layout_opts: dict[str, Any] = {}
    value_layout, value_layout_opts = layout.make_default_matmul_mxfp4_w_layout(
        mx_axis=1
    )
    scale_layout, scale_layout_opts = layout.make_default_matmul_mxfp4_w_scale_layout(
        mx_axis=1, num_warps=num_warps
    )

    if cap[0] == 9:
        opt_flags.update_opt_flags_constraints({"split_k": 1})
    elif cap[0] == 10:
        opt_flags.update_opt_flags_constraints(
            {"is_persistent": True, "epilogue_subtile": 1}
        )

    # transpose so the quantization axis is on dim 1
    quant_tensor = quant_tensor.transpose(-2, -1)
    scale = scale.transpose(-2, -1)
    quant_tensor = convert_layout(
        wrap_torch_tensor(quant_tensor, dtype=FP4),
        value_layout,
        **value_layout_opts,
    )
    scale = convert_layout(
        wrap_torch_tensor(scale), scale_layout, **scale_layout_opts
    )
    return quant_tensor, InFlexData(), scale


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


@triton.jit
def _pack_bitmatrix_kernel(
    bitmatrix,
    topk_ids,
    n_rows,
    bm_cols: tl.constexpr,
    n_expts_act,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Pack ``topk_ids`` into a bitmatrix.

    Original Triton reference:
    https://github.com/triton-lang/triton/blob/dd1bbc52b34d202dfe5ffea1e04fb16166c5c04e/python/triton_kernels/bench/distributed.py#L264
    """
    pid_m = tl.program_id(0)
    offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_k = tl.arange(0, BLOCK_SIZE_K)
    offsets = offsets_m[:, None] * n_expts_act + offsets_k[None, :]
    mask = (offsets_m < n_rows)[:, None] & (offsets_k < n_expts_act)[None, :]
    indices = tl.load(topk_ids + offsets, mask=mask, other=-1)
    div = indices // 32
    rem = indices % 32
    one = tl.cast(1, tl.uint32)

    for i in range(bm_cols):
        offs = tl.arange(0, BLOCK_SIZE_K // 32) + i * (BLOCK_SIZE_K // 32)
        x = tl.where(
            div[:, :, None] == offs[None, None, :], (one << rem)[:, :, None], 0
        )
        y = tl.reduce_or(x, axis=1)
        bitmatrix_ptrs = bitmatrix + offsets_m[:, None] * bm_cols + offs[None, :]
        tl.store(bitmatrix_ptrs, y, mask=offsets_m[:, None] < n_rows)


def _routing_from_bitmatrix(bitmatrix, expt_scal, expt_indx, n_expts_tot, n_expts_act):
    """Build (RoutingData, GatherIndx, ScatterIndx) from a packed bitmatrix."""
    sparse_logits = SparseMatrix(indx=expt_indx, vals=expt_scal, mask=bitmatrix)
    dispatch_indx = sparse_logits.mask_metadata.row_sorted_indx
    combine_indx = sparse_logits.mask_metadata.col_sorted_indx
    ragged_batch_metadata = make_ragged_tensor_metadata(
        sparse_logits.mask_metadata.col_sum,
        dispatch_indx.shape[0],
    )
    gate_scal = sparse_logits.vals.flatten()[combine_indx]
    routing_data = RoutingData(
        gate_scal,
        ragged_batch_metadata.block_sizes,
        n_expts_tot,
        n_expts_act,
        ragged_batch_metadata,
    )
    gather_idx = GatherIndx(combine_indx, dispatch_indx)
    scatter_idx = ScatterIndx(dispatch_indx, combine_indx)
    return routing_data, gather_idx, scatter_idx


def _routing_from_logits(logits: torch.Tensor, n_expts_act: int, sm_first: bool):
    """Compute routing data straight from gating logits."""
    if sm_first:
        logits = torch.softmax(logits, dim=-1)
    sparse_logits = topk(logits, n_expts_act, apply_softmax=not sm_first)
    return _routing_from_bitmatrix(
        sparse_logits.mask,
        sparse_logits.vals,
        sparse_logits.indx,
        logits.shape[-1],
        n_expts_act,
    )


# ---------------------------------------------------------------------------
# Fused experts
# ---------------------------------------------------------------------------


def _resize_cache(x: torch.Tensor, v: tuple[int, ...]) -> torch.Tensor:
    """Shrink ``x`` and reshape it to ``v``. Used for intermediate caches."""
    n = 1
    for d in v:
        n *= d
    assert n <= x.numel(), f"{v} ({n}) <= {x.shape} ({x.numel()})"
    return x.flatten()[:n].view(*v)


def _fused_experts(
    output_tensor: torch.Tensor,
    hidden_states: torch.Tensor,
    w1,
    w2,
    routing_data,
    gather_indx,
    scatter_indx,
    topk: int,
    quant_config: Mxfp4MoEQuantConfig,
    swiglu_alpha: float = 1.702,
    swiglu_limit: float = 7.0,
    apply_router_weight_on_input: bool = False,
) -> torch.Tensor:
    """Run the two fused MXFP4 matmuls with OAI SwiGLU in between."""
    assert hidden_states.dtype == torch.bfloat16
    assert quant_config.w1_bias is None or quant_config.w1_bias.dtype == torch.float32
    assert quant_config.w2_bias is None or quant_config.w2_bias.dtype == torch.float32
    assert hidden_states.ndim == 2
    assert hidden_states.shape[-1] == w1.shape[-2]
    assert w2.shape[-1] == w1.shape[1]

    batch_dim = 1
    M, K = hidden_states.shape[-2:]
    _, _, N = w1.shape

    intermediate_cache = torch.empty(
        (batch_dim, M * topk, N // 2),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache = _resize_cache(intermediate_cache, (batch_dim, M * topk, N // 2))
    output_tensor = _resize_cache(output_tensor, (batch_dim, M, K))

    act = FusedActivation(
        FnSpecs(
            "swiglu",
            tk_swiglu.swiglu_fn,
            ("alpha", "limit"),
            reduction_n=2,
        ),
        (swiglu_alpha, swiglu_limit),
    )
    gammas = routing_data.gate_scal if routing_data else None

    matmul_ogs(
        hidden_states,
        w1,
        quant_config.w1_bias,
        routing_data,
        gather_indx=gather_indx,
        precision_config=quant_config.w1_precision,
        gammas=gammas if apply_router_weight_on_input else None,
        fused_activation=act,
        y=intermediate_cache,
    )
    matmul_ogs(
        intermediate_cache.view(M * topk, N // 2),
        w2,
        quant_config.w2_bias,
        routing_data,
        scatter_indx=scatter_indx,
        precision_config=quant_config.w2_precision,
        gammas=None if apply_router_weight_on_input else gammas,
        y=output_tensor,
    )
    return output_tensor.view(M, K)


# ---------------------------------------------------------------------------
# Public nn.Module interface
# ---------------------------------------------------------------------------


class Mxfp4MoE(nn.Module):
    """MXFP4-quantized fused MoE primitive (routing + matmul_ogs experts).

    The module is stateless -- expert weights, biases, and the
    :class:`Mxfp4MoEQuantConfig` are passed to ``forward`` so a single
    instance can serve any number of MoE layers. Weight preparation is
    exposed as static helpers so the L2 caller does not need to import
    ``triton_kernels`` directly.
    """

    @staticmethod
    def prepare_weight(
        quant_tensor: torch.Tensor,
        scale: torch.Tensor,
        num_warps: int = 8,
    ):
        """Swizzle an MXFP4 expert weight and build its ``PrecisionConfig``.

        Returns ``(swizzled_weight, precision_config)`` ready to feed
        into :meth:`make_quant_config` and :meth:`forward`.
        """
        weight, flex, scale_tensor = _swizzle_mxfp4(quant_tensor, scale, num_warps)
        precision = PrecisionConfig(
            weight_scale=scale_tensor, flex_ctx=FlexCtx(rhs_data=flex)
        )
        return weight, precision

    @staticmethod
    def make_quant_config(
        w1_precision: Any,
        w2_precision: Any,
        w1_bias: torch.Tensor | None = None,
        w2_bias: torch.Tensor | None = None,
    ) -> Mxfp4MoEQuantConfig:
        """Construct an MXFP4 W4A16 quant config from per-expert precisions/biases."""
        return Mxfp4MoEQuantConfig(
            w1_precision=w1_precision,
            w2_precision=w2_precision,
            w1_bias=w1_bias,
            w2_bias=w2_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        w1,
        w2,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
        quant_config: Mxfp4MoEQuantConfig,
        apply_router_weight_on_input: bool = False,
    ) -> torch.Tensor:
        """End-to-end MXFP4 MoE forward (routing + fused experts).

        ``w1``/``w2`` must already be swizzled (see :meth:`prepare_weight`)
        and ``quant_config`` must carry the matching precision configs and
        expert biases. ``hidden_states`` must be bfloat16 and 2D.
        """
        routing_data, gather_idx, scatter_idx = _routing_from_logits(
            gating_output, topk, sm_first=not renormalize
        )
        output = torch.empty_like(hidden_states)
        return _fused_experts(
            output,
            hidden_states,
            w1,
            w2,
            routing_data,
            gather_idx,
            scatter_idx,
            topk=topk,
            quant_config=quant_config,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )


__all__ = ["Mxfp4MoE", "Mxfp4MoEQuantConfig"]
