"""MXFP4-native fused MoE primitive backed by the OAI Triton kernels.

This module is the L1 wrapper around ``triton_kernels.matmul_ogs`` for
MXFP4-quantized expert weights with OAI-style SwiGLU activation. It owns
all of the routing/quantization/swizzling logic that GPT-OSS needs so
that the L2 ``GptOssMoE`` module can stay pure-composition.

Why we copy this code: the implementations of weight swizzling, routing
data construction, and the fused matmul wrapper live inside vLLM. FastKernels
L2+ modules are not allowed to call into vLLM, so the relevant bits of
``vllm.model_executor.layers.fused_moe.gpt_oss_triton_kernels_moe`` and
``vllm.model_executor.layers.quantization.utils.mxfp4_utils`` are
re-implemented here verbatim (modulo cleanup of code paths FastKernels does
not exercise -- AITER/ROCm fallbacks, expert parallelism, w4a8, and the
``use_legacy_triton_kernels`` shim).

The underlying ``triton_kernels`` package is OpenAI's standalone Triton
helper library (https://github.com/triton-lang/triton/tree/main/python/triton_kernels);
it is bundled inside vLLM's ``third_party`` directory but is otherwise
an external dependency. We locate it via the vLLM install path purely
to extend ``sys.path`` -- we never invoke any vLLM function.
"""

from __future__ import annotations

import functools
import importlib.util
import os
import sys
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# triton_kernels availability
# ---------------------------------------------------------------------------


@functools.cache
def _ensure_triton_kernels_on_path() -> None:
    """Ensure the ``triton_kernels`` package is importable.

    Mirrors vLLM's ``import_triton_kernels`` shim but performs only
    filesystem / sys.path manipulation -- no vLLM functions are called.
    Prefers a top-level install if present and otherwise falls back to
    the copy bundled inside the installed vLLM package.
    """
    if importlib.util.find_spec("triton_kernels") is not None:
        return

    vllm_spec = importlib.util.find_spec("vllm")
    if vllm_spec is not None and vllm_spec.origin is not None:
        third_party = os.path.join(os.path.dirname(vllm_spec.origin), "third_party")
        if os.path.isdir(os.path.join(third_party, "triton_kernels")):
            if third_party not in sys.path:
                sys.path.insert(0, third_party)
            return

    raise ImportError(
        "triton_kernels is required for MXFP4 MoE. Install it from "
        "https://github.com/triton-lang/triton/tree/main/python/triton_kernels"
    )


# ---------------------------------------------------------------------------
# Quant config (replacement for vLLM's FusedMoEQuantConfig)
# ---------------------------------------------------------------------------


@dataclass
class Mxfp4MoEQuantConfig:
    """Minimal quant config carrying the per-MoE precision/bias tensors.

    Attribute names match the subset of ``FusedMoEQuantConfig`` consumed
    by ``triton_kernel_fused_experts`` (``w{1,2}_precision`` and
    ``w{1,2}_bias``), so the call sites stay essentially unchanged.
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

    Copied from ``vllm.model_executor.layers.quantization.utils.mxfp4_utils._swizzle_mxfp4``,
    minus the ROCm/Hopper-old-torch fallbacks that FastKernels does not exercise.
    """
    _ensure_triton_kernels_on_path()
    import triton_kernels.matmul_ogs_details.opt_flags as opt_flags
    from triton_kernels.numerics import InFlexData
    from triton_kernels.tensor import FP4, convert_layout, wrap_torch_tensor
    from triton_kernels.tensor_details import layout

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
        constraints = {"is_persistent": True, "epilogue_subtile": 1}
        # These match vLLM's mxfp4_utils._swizzle_mxfp4 exactly -- but vLLM only ever
        # reaches this kernel on Blackwell when FlashInfer is *absent*
        # (``_get_mxfp4_backend`` returns SM100_FI_MXFP4_BF16 otherwise), so the
        # sm100 constraints it publishes are effectively untested. Both GPT-OSS rows
        # fault inside matmul_ogs here: 20b in the persistent kernel under CUDA
        # graphs, 120b in the split-k ``reduce`` even eagerly. Hopper avoids the
        # reduce path entirely by pinning split_k=1, so allow overriding the sm100
        # constraints to test the same, e.g.
        # ``FASTKERNELS_MXFP4_SM100_CONSTRAINTS=split_k=1,is_persistent=0``.
        raw = os.environ.get("FASTKERNELS_MXFP4_SM100_CONSTRAINTS", "")
        for item in (p for p in raw.split(",") if p.strip()):
            key, _, val = item.partition("=")
            constraints[key.strip()] = bool(int(val)) if key.strip() in (
                "is_persistent",
            ) else int(val)
        opt_flags.update_opt_flags_constraints(constraints)

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
    _ensure_triton_kernels_on_path()
    from triton_kernels.matmul_ogs import GatherIndx, RoutingData, ScatterIndx
    from triton_kernels.tensor import SparseMatrix, make_ragged_tensor_metadata

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
    if os.environ.get("FASTKERNELS_CHECK_MOE_ROUTING") == "1":
        _check_routing_indices(dispatch_indx, combine_indx,
                              sparse_logits.mask_metadata.col_sum, n_expts_tot)
    return routing_data, gather_idx, scatter_idx


def _check_routing_indices(dispatch_indx, combine_indx, col_sum, n_expts_tot):
    """Validate the indices ``matmul_ogs`` will dereference.

    ``matmul_ogs`` gathers rows through these, so a single out-of-range entry reads
    outside the expert weights and reports as ``illegal memory access`` from
    whichever CUDA call happens to come next -- which is how the GPT-OSS crash
    presents (inside Triton's ``load_binary``, several frames from the cause).
    Off by default: this syncs and allocates, so it is a debugging aid, not a
    hot-path guard, and it cannot run inside a CUDA graph capture.
    """
    n = dispatch_indx.numel()
    for name, t in (("dispatch_indx", dispatch_indx), ("combine_indx", combine_indx)):
        lo, hi = int(t.min()), int(t.max())
        # -1 is the documented "no token" sentinel; anything else must index a row.
        if lo < -1 or hi >= n:
            raise RuntimeError(
                f"MoE routing {name} out of range: min {lo} max {hi} for {n} rows "
                f"(n_expts_tot={n_expts_tot}). matmul_ogs would read out of bounds."
            )
    total = int(col_sum.sum())
    if col_sum.numel() != n_expts_tot or total > n:
        raise RuntimeError(
            f"MoE routing histogram inconsistent: {col_sum.numel()} experts "
            f"(expected {n_expts_tot}), tokens {total} > {n} rows."
        )


def _routing_from_logits(logits: torch.Tensor, n_expts_act: int, sm_first: bool):
    """Compute routing data straight from gating logits."""
    _ensure_triton_kernels_on_path()
    from triton_kernels.topk import topk

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
    _ensure_triton_kernels_on_path()
    import triton_kernels.swiglu
    from triton_kernels.matmul_ogs import FnSpecs, FusedActivation, matmul_ogs

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
            triton_kernels.swiglu.swiglu_fn,
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
    def sm100_flashinfer_available() -> bool:
        """Should this run take the Blackwell FlashInfer path?

        vLLM's ``_get_mxfp4_backend`` returns ``SM100_FI_MXFP4_BF16`` for any
        capability-family-100 host that has FlashInfer, and only falls through to
        ``Mxfp4Backend.TRITON`` (the ``matmul_ogs`` path below) when FlashInfer is
        absent. We were taking the Triton path unconditionally, which is why gpt-oss
        faulted inside ``_p_matmul_ogs`` on B200 and nowhere else: those sm100
        ``opt_flags`` constraints are code the reference never executes on Blackwell.
        Mirroring the selection also makes the row 1.4-4.7x faster.

        ``cc[0] == 10`` matches vLLM's ``is_device_capability_family(100)``. H200
        reports (9, 0), so the Triton path is untouched there.
        """
        if os.environ.get("FASTKERNELS_MXFP4_FLASHINFER", "1") != "1":
            return False
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
            return False
        try:
            from flashinfer import trtllm_fp4_block_scale_moe  # noqa: F401
            from flashinfer.fp4_quantization import (  # noqa: F401
                nvfp4_block_scale_interleave,
            )
            from flashinfer.fused_moe.core import (  # noqa: F401
                get_w2_permute_indices_with_cache,
            )
        except ImportError:
            return False
        return True

    @staticmethod
    def _swap_every_two_rows(t: torch.Tensor, axis: int = -1) -> torch.Tensor:
        """Swap adjacent pairs along ``axis``.

        gpt-oss stores gate/up interleaved as (gate_0, up_0, gate_1, up_1, ...); the
        TRTLLM-gen kernel's fused SwiGLU expects the opposite order within each pair.
        This is load-bearing, not cosmetic -- omitting it moves the output by abs-max
        2.01 against a 0.015 reference scale.
        """
        shape = t.shape
        if axis < 0:
            axis = len(shape) + axis
        new = list(shape)
        new[axis] = shape[axis] // 2
        new.insert(axis + 1, 2)
        return t.reshape(*new).flip(axis + 1).reshape(*shape)

    @staticmethod
    def prepare_weight_flashinfer(w13_q, w13_s, w13_b, w2_q, w2_s, w2_b):
        """Lay MXFP4 expert weights out for ``trtllm_fp4_block_scale_moe``.

        Mirrors vLLM's SM100_FI_MXFP4_BF16 preparation, which is a completely
        different layout from ``_swizzle_mxfp4``: swap adjacent gate/up rows, apply
        the kernel's epilogue row shuffle per expert, interleave the block scales, and
        hand the scales over as float8_e4m3fn. Inputs must already be padded to the
        shapes the kernel wants (see ``GptOssMoE``); zero pad rows survive the
        permutation as zeros, and an E8M0 byte of 0 times FP4 0 contributes nothing.

        Returns ``(FW13, FS13, FB13, FW2, FS2, FB2)``.
        """
        from flashinfer.fp4_quantization import nvfp4_block_scale_interleave
        from flashinfer.fused_moe.core import get_w2_permute_indices_with_cache

        E = w13_q.shape[0]
        dev = w13_q.device
        swap = Mxfp4MoE._swap_every_two_rows
        s13 = swap(w13_s, -2)
        q13 = swap(w13_q, -2)
        b13 = swap(w13_b.float(), -1)
        b2 = w2_b.float()

        cache: dict = {}
        epilogue_tile_m = 128
        g1w, g1s, g1b, g2w, g2s, g2b = [], [], [], [], [], []
        for i in range(E):
            p = get_w2_permute_indices_with_cache(
                cache, q13[i].view(torch.uint8), epilogue_tile_m)
            g1w.append(q13[i].view(torch.uint8)[p.to(dev)].contiguous())
            ps = get_w2_permute_indices_with_cache(
                cache, s13[i].view(torch.uint8), epilogue_tile_m, num_elts_per_sf=16)
            g1s.append(nvfp4_block_scale_interleave(
                s13[i].view(torch.uint8)[ps.to(dev)].contiguous()))
            pb = get_w2_permute_indices_with_cache(
                cache, b13[i].clone().reshape(-1, 1), epilogue_tile_m)
            g1b.append(b13[i].clone().reshape(-1, 1)[pb.to(dev)].contiguous())

            p = get_w2_permute_indices_with_cache(
                cache, w2_q[i].view(torch.uint8), epilogue_tile_m)
            g2w.append(w2_q[i].view(torch.uint8)[p.to(dev)].contiguous())
            ps = get_w2_permute_indices_with_cache(
                cache, w2_s[i].view(torch.uint8), epilogue_tile_m, num_elts_per_sf=16)
            g2s.append(nvfp4_block_scale_interleave(
                w2_s[i].view(torch.uint8)[ps.to(dev)].contiguous()))
            p = get_w2_permute_indices_with_cache(
                cache, b2[i].clone().reshape(-1, 1), epilogue_tile_m)
            g2b.append(b2[i].clone().reshape(-1, 1)[p.to(dev)].contiguous())

        FW13 = torch.stack(g1w)
        FS13 = torch.stack(g1s).reshape(*w13_s.shape).view(torch.float8_e4m3fn)
        FW2 = torch.stack(g2w)
        FS2 = torch.stack(g2s).reshape(*w2_s.shape).view(torch.float8_e4m3fn)
        FB13 = torch.stack(g1b).reshape(E, -1)
        FB2 = torch.stack(g2b).reshape(E, -1)
        return FW13, FS13, FB13, FW2, FS2, FB2

    @staticmethod
    def forward_flashinfer(hidden_states, router_logits, *, w13, w13_scale, w13_bias,
                           w2, w2_scale, w2_bias, alpha, beta, clamp_limit,
                           num_experts, top_k, intermediate_size,
                           tune_max_num_tokens):
        """One ``trtllm_fp4_block_scale_moe`` call, replacing routing + two matmuls.

        ``routing_method_type=1`` is Renormalize (top-k then softmax), matching the
        ``renormalize=True`` / ``sm_first=False`` semantics of the Triton path.
        FlashInfer autotuning is left off, as vLLM also forces it off, so no probe
        kernels are launched -- which matters because this runs inside CUDA graphs.
        """
        from flashinfer import trtllm_fp4_block_scale_moe

        return trtllm_fp4_block_scale_moe(
            routing_logits=router_logits.to(torch.bfloat16), routing_bias=None,
            hidden_states=hidden_states, hidden_states_scale=None,
            gemm1_weights=w13, gemm1_weights_scale=w13_scale, gemm1_bias=w13_bias,
            gemm1_alpha=alpha, gemm1_beta=beta, gemm1_clamp_limit=clamp_limit,
            gemm2_weights=w2, gemm2_weights_scale=w2_scale, gemm2_bias=w2_bias,
            output1_scale_scalar=None, output1_scale_gate_scalar=None,
            output2_scale_scalar=None,
            num_experts=num_experts, top_k=top_k, n_group=None, topk_group=None,
            intermediate_size=intermediate_size,
            local_expert_offset=0, local_num_experts=num_experts,
            routed_scaling_factor=None, routing_method_type=1, do_finalize=True,
            tune_max_num_tokens=tune_max_num_tokens,
        )[0]

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
        _ensure_triton_kernels_on_path()
        from triton_kernels.matmul_ogs import FlexCtx, PrecisionConfig

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
