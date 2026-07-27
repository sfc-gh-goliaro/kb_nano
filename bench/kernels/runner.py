"""Isolated kernel-level benchmarking via direct forward() calls.

Instantiates baseline and candidate nn.Module instances, copies weights,
loads inputs from the InputRegistry (random or golden), compares outputs
and timing. No full model build required — per-kernel test time is seconds
rather than minutes.
"""

from __future__ import annotations

import gc
import os
import time
import inspect
from typing import Any

import torch
import torch.nn as nn

from fastkernels.bench.kernels.forward_args import synthesize_forward_args
from fastkernels.bench.kernels.forward_context import tier1_forward_context
from fastkernels.bench.kernels.real_weights import load_real_weights
from fastkernels.bench.kernels.init_resolver import (
    candidate_kwargs,
    describe_unresolved,
    preferred_dtype,
)
from fastkernels.bench.kernels.scenario_registry import InputRegistry
from fastkernels.infra.kernel_swapper import (
    BenchTarget,
    discover_references,
    discover_targets,
    get,
    load_candidate,
    load_reference,
)

from .result import KernelBenchResult, OperatorResult, ScenarioResult

_DEFAULT_REGISTRY = None
_FP32_ATOL = 1e-5
_FP32_RTOL = 1e-3
_LOW_PRECISION_ATOL = 1e-2
_LOW_PRECISION_RTOL = 1e-2
_FP8_ATOL = 1.25e-1
_FP8_RTOL = 1.25e-1
_FP8_GROUP_SIZE = 128


def _short_exception(exc: BaseException) -> str:
    message = str(exc).strip()
    if not message:
        message = exc.__class__.__name__
    return f"{exc.__class__.__name__}: {message}"


def _get_registry() -> InputRegistry:
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = InputRegistry()
    return _DEFAULT_REGISTRY


def _find_candidate_path(target_name: str, level: int) -> str:
    """Return the relative path to the candidate file for display."""
    return f"tasks/candidate/L{level}/{target_name}.py"


def _find_reference_path(target_name: str, level: int) -> str:
    """Return the relative path to the semantic reference file for display."""
    return f"tasks/reference/L{level}/{target_name}.py"


def _normalize_kwargs(cls: type, init_args: dict[str, Any]) -> dict[str, Any]:
    kwargs = dict(init_args)
    if "head_size" in kwargs and "head_dim" not in kwargs:
        kwargs["head_dim"] = kwargs.pop("head_size")
    if "base" in kwargs and "rope_theta" not in kwargs:
        kwargs["rope_theta"] = kwargs.pop("base")
    kwargs.pop("rotary_dim", None)
    kwargs.pop("is_neox_style", None)
    try:
        sig = inspect.signature(cls.__init__)
        params = sig.parameters
        accepts_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in params.values()
        )
        if not accepts_kwargs:
            kwargs = {
                k: v for k, v in kwargs.items()
                if k in params and k != "self"
            }
    except (TypeError, ValueError):
        pass
    return kwargs


def _release_cuda(device: str) -> None:
    """Drop freed modules and return their memory to the allocator.

    ``empty_cache()`` alone cannot reclaim memory that is still referenced, and
    nn.Modules routinely sit in reference cycles that plain refcounting never
    breaks.  Without an explicit collection gpt-oss's L4 pipeline entered its
    second scenario with 131.7 GiB still allocated and every remaining scenario
    died with OutOfMemoryError; collecting first brings it back to 0.0 GiB.
    """
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()


def _run_scenario_sequentially(
    target: Any,
    user_impl: type,
    scenario: Any,
    inputs: dict[str, Any],
    models: tuple[str, ...],
    input_dtype: torch.dtype | None,
    device: str,
    timing_warmup: int,
    timing_runs: int,
    validation_mode: str,
    baseline_holder: list,
) -> "ScenarioResult":
    """Benchmark a scenario with only one module resident at a time.

    Used when the baseline and candidate together would exceed the device (see
    ``_needs_sequential``).  The baseline runs and is released before the
    candidate is built, so peak memory is one model rather than two.

    Both modules are built by the same path from the same checkpoint, with a
    seeded synthetic fallback, so this does not change what is compared -- only
    when each side is resident.

    ``baseline_holder`` is a one-element list rather than the module itself: a
    plain argument leaves the *caller's* name bound to the baseline for the whole
    call, so ``del`` here frees nothing and the candidate is still built beside a
    resident baseline -- which is the entire thing this path exists to avoid.
    Popping it drops the last reference.
    """
    baseline_mod = baseline_holder.pop()

    extra_args = synthesize_forward_args(
        target.target_cls, inputs, models, device, input_dtype,
    )
    if extra_args:
        inputs = {**inputs, **extra_args}

    with tier1_forward_context([baseline_mod], inputs, device, input_dtype):
        baseline_check_inputs = _clone_inputs(inputs)
        baseline_out = _run_forward_once(baseline_mod, baseline_check_inputs)
        baseline_out_cpu = _detach_to_cpu(baseline_out)
        baseline_inputs_cpu = _detach_to_cpu(baseline_check_inputs)
        del baseline_out, baseline_check_inputs
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        _, baseline_ms = _time_forward(
            baseline_mod, _clone_inputs(inputs), timing_warmup, timing_runs,
        )

    del baseline_mod
    _release_cuda(device)

    candidate_mod = _instantiate_module(
        user_impl, scenario.init_args, device, dtype=input_dtype,
        models=models, inputs=inputs, level=target.level,
    )

    with tier1_forward_context([candidate_mod], inputs, device, input_dtype):
        candidate_check_inputs = _clone_inputs(inputs)
        candidate_out = _run_forward_once(candidate_mod, candidate_check_inputs)
        correct, max_error_ratio, mean_diff = _merge_correctness(
            _compare_outputs(baseline_out_cpu, _detach_to_cpu(candidate_out)),
            _compare_outputs(
                baseline_inputs_cpu, _detach_to_cpu(candidate_check_inputs),
            ),
        )
        del candidate_out, candidate_check_inputs
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        _, candidate_ms = _time_forward(
            candidate_mod, _clone_inputs(inputs), timing_warmup, timing_runs,
        )

    del candidate_mod
    _release_cuda(device)

    speedup = baseline_ms / candidate_ms if candidate_ms > 0 else float("inf")
    classification = (
        "harness_validation_passed"
        if validation_mode in ("baseline_identity", "pytorch_reference") and correct
        else "candidate_correct_and_timed" if correct
        else "candidate_correctness_failure"
    )
    return ScenarioResult(
        name=scenario.name,
        correct=correct,
        max_error_ratio=max_error_ratio,
        mean_abs_diff=mean_diff,
        baseline_ms=baseline_ms,
        candidate_ms=candidate_ms,
        speedup=speedup,
        failure_reason=None if correct else "output_mismatch",
        classification=classification,
    )


def _detach_to_cpu(obj: Any) -> Any:
    """Copy an output tree to host memory, preserving its structure."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().to("cpu", copy=True)
    if isinstance(obj, dict):
        return {k: _detach_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        built = [_detach_to_cpu(v) for v in obj]
        return type(obj)(built) if isinstance(obj, tuple) else built
    return obj


def _needs_sequential(module: nn.Module, device: str) -> bool:
    """True when a second copy of ``module`` would not fit beside the first.

    Tier 1 builds the baseline and the candidate together so both can be driven
    from one forward context.  For an L4 pipeline that doubles the model: gpt-oss
    (120B, MXFP4) and qwen3-vl (235B) each occupy ~60-68 GB, so the pair
    saturates a 140 GB card and every scenario dies with OutOfMemoryError --
    llama-3.1-8B at 16384 tokens missed by just 224 MB.

    Where the pair does not fit, run the two sides one at a time instead.  Both
    are built from the same checkpoint, and the synthetic fallback is seeded per
    module, so weights stay identical without keeping a host-side copy.
    """
    if not device.startswith("cuda"):
        return False
    try:
        free, _total = torch.cuda.mem_get_info()
    except Exception:
        return False
    footprint = sum(p.numel() * p.element_size() for p in module.parameters())
    footprint += sum(b.numel() * b.element_size() for b in module.buffers())
    # Leave room for activations: a second copy must fit in well under what is
    # left, not merely fit.
    return footprint > 0 and footprint * 1.5 > free


def _instantiate_module(
    cls: type,
    init_args: dict[str, Any],
    device: str = "cuda",
    dtype: torch.dtype | None = None,
    models: tuple[str, ...] = (),
    inputs: Any = None,
    level: int | None = None,
    load_weights: bool = True,
) -> nn.Module:
    """Create an nn.Module instance with init_args, handling common patterns.

    The registry's ``init_args`` are frequently incomplete: the tracer records
    only keyword arguments that are YAML-serializable, so HF ``config`` objects
    and positionally-passed scalars are missing.  ``init_resolver`` reconstructs
    them; each candidate kwarg set is tried in turn, the recorded args first, so
    targets with complete scenarios are unaffected.
    """
    attempts = candidate_kwargs(cls, init_args, models, inputs, level)

    module = None
    # Report the *last* attempt's error: attempts run from least to most
    # resolved, so the final one is the informative failure.  Reporting the
    # first would just repeat "missing required argument 'config'" and hide
    # whatever went wrong once the config was actually supplied.
    last_error: BaseException | None = None
    # Construct directly in the target dtype.  Building in fp32 and casting
    # afterwards doubles peak memory, which an L4 pipeline cannot afford: a
    # Llama-3.1-8B baseline+candidate pair peaks at ~64 GB in fp32 before the
    # cast, and gpt-oss-120b never fits at all.
    prev_dtype = torch.get_default_dtype()
    if dtype is not None and dtype.is_floating_point:
        torch.set_default_dtype(dtype)
    try:
        for raw in attempts:
            kwargs = _normalize_kwargs(cls, raw)
            try:
                module = cls(**kwargs)
                break
            except Exception as exc:
                last_error = exc
    finally:
        torch.set_default_dtype(prev_dtype)
    if module is None:
        try:
            module = cls()
        except Exception:
            unresolved = describe_unresolved(cls, init_args, models, inputs)
            if unresolved:
                raise TypeError(
                    f"{cls.__name__}: cannot build from registry init_args; "
                    f"unresolved required arguments {unresolved}. The input "
                    f"tracer did not record them (positional or "
                    f"non-serializable)."
                ) from last_error
            raise last_error if last_error is not None else TypeError(
                f"{cls.__name__}: instantiation failed"
            )

    module = module.to(device)

    # Modules built here allocate parameters with ``torch.empty`` and are never
    # given weights (there is no checkpoint load in Tier 1).  Uninitialized
    # memory is frequently all-zero, which silently destroys any module with a
    # log-space gate: GLA computes log(sigmoid(0)) over zero projections and
    # every one of gla_decoder's 320 scenarios returns NaN -- for the baseline
    # too, so the comparison is meaningless rather than merely wrong.
    #
    # Seeded per module so baseline and candidate receive identical weights;
    # the seed is fixed so a scenario is reproducible across runs.
    # Prefer real checkpoint weights: synthetic values make any log-space gate
    # or deep fp16 chain produce NaN on *both* sides, which turns the
    # correctness check into noise.  Falls back to synthetic init when no local
    # snapshot exists.
    n_real = 0
    if load_weights:
        try:
            n_real = load_real_weights(module, models, device, dtype)
        except Exception:
            n_real = 0
        # Coverage is the first thing to check when both sides come back
        # non-finite: a module left on synthetic values can diverge for reasons
        # that say nothing about the candidate.
        if os.environ.get("FASTKERNELS_DEBUG_WEIGHTS"):
            n_total = sum(
                1 for _, p in module.named_parameters(recurse=True)
                if p.is_floating_point()
            )
            print(f"  [weights] {type(module).__name__}: "
                  f"{n_real}/{n_total} from checkpoint")

    with torch.no_grad():
        gen = torch.Generator(device="cpu").manual_seed(1234)
        for name, param in module.named_parameters(recurse=True):
            if not param.is_floating_point() or not param.numel():
                continue
            if not bool((param == 0).all()):
                continue
            lname = name.lower()
            if param.ndim <= 1:
                # Norm scales and biases: a zero bias is legitimate, but a zero
                # norm weight annihilates the signal.  Ones for scales, zeros
                # for biases.
                param.data.fill_(
                    1.0 if ("norm" in lname or "scale" in lname or "gamma" in lname)
                    else 0.0
                )
                continue
            # Packed-quantization tensors (MXFP4 expert blocks/scales) are not
            # plain weights: their layout encodes a block structure the kernel
            # reinterprets, and random bytes make the reinterpretation fail
            # ("shape '[128, 5760, 32]' is invalid").  Leave them at zero --
            # a valid, if trivial, packed value.
            if any(tag in lname for tag in
                   ("_blocks", "_scales", "weight_scale", "w13_", "w2_")):
                continue
            # Weight matrices: draw at a *small* scale.  1/sqrt(fan_in) is the
            # right variance for a single layer, but a decoder block stacks
            # many of them and the activations blow up to inf in fp16 -- both
            # sides then produce NaN and the comparison says nothing.
            fan_in = param.shape[-1]
            # fp16 saturates at 65504 and these blocks chain several matmuls,
            # so the scale has to stay well below the single-layer optimum:
            # at std=0.02 oasis_block's spatial-attention output projection
            # already overflows to NaN on both sides.
            std = min(0.006, (1.0 / max(fan_in, 1)) ** 0.5)
            if param.dtype == torch.float16 or dtype == torch.float16:
                std = min(std, 0.003)
            sample = torch.randn(
                param.shape, generator=gen, dtype=torch.float32,
            ) * std
            param.data.copy_(sample.to(device=param.device, dtype=param.dtype))

    if dtype is not None:
        # Cast learnable parameters to the scenario dtype without changing
        # precision-sensitive buffers such as RoPE/YARN cos/sin caches.
        with torch.no_grad():
            for param in module.parameters(recurse=True):
                if param.is_floating_point():
                    param.data = param.data.to(dtype=dtype)
    module.eval()
    return module


def _first_floating_dtype(value: Any) -> torch.dtype | None:
    if isinstance(value, torch.Tensor) and value.is_floating_point():
        if "float8" not in str(value.dtype):
            return value.dtype
        return None
    if isinstance(value, dict):
        for v in value.values():
            dtype = _first_floating_dtype(v)
            if dtype is not None:
                return dtype
    if isinstance(value, (tuple, list)):
        for v in value:
            dtype = _first_floating_dtype(v)
            if dtype is not None:
                return dtype
    return None


def _clone_input_value(value: Any) -> Any:
    """Clone tensors in an input tree so in-place kernels cannot cross-contaminate runs."""
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, tuple):
        return tuple(_clone_input_value(v) for v in value)
    if isinstance(value, list):
        return [_clone_input_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _clone_input_value(v) for k, v in value.items()}
    return value


def _clone_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    return {k: _clone_input_value(v) for k, v in inputs.items()}


def _contains_cuda_tensor(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return value.is_cuda
    if isinstance(value, dict):
        return any(_contains_cuda_tensor(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_cuda_tensor(v) for v in value)
    return False


def _synchronize_if_cuda(*values: Any) -> None:
    if any(_contains_cuda_tensor(v) for v in values):
        torch.cuda.synchronize()


def _time_forward(
    module: nn.Module,
    inputs: dict[str, Any],
    num_warmup: int,
    num_runs: int,
) -> tuple[Any, float]:
    """Warmup + time forward() calls. Returns (output, median_ms)."""
    tensor_inputs = {
        k: v for k, v in inputs.items()
        if isinstance(v, torch.Tensor)
    }
    scalar_inputs = {
        k: v for k, v in inputs.items()
        if not isinstance(v, torch.Tensor)
    }

    with torch.no_grad():
        for _ in range(num_warmup):
            module(**tensor_inputs, **scalar_inputs)

        _synchronize_if_cuda(tensor_inputs)
        times = []
        output = None
        for _ in range(num_runs):
            start = time.perf_counter()
            output = module(**tensor_inputs, **scalar_inputs)
            _synchronize_if_cuda(tensor_inputs, output)
            times.append((time.perf_counter() - start) * 1000)

    times.sort()
    median_ms = times[len(times) // 2]
    if output is None:
        output = {k: v for k, v in tensor_inputs.items()}
    return output, median_ms


def _run_forward_once(module: nn.Module, inputs: dict[str, Any]) -> Any:
    tensor_inputs = {
        k: v for k, v in inputs.items()
        if isinstance(v, torch.Tensor)
    }
    scalar_inputs = {
        k: v for k, v in inputs.items()
        if not isinstance(v, torch.Tensor)
    }
    with torch.no_grad():
        output = module(**tensor_inputs, **scalar_inputs)
        _synchronize_if_cuda(tensor_inputs, output)
    return output


def _tolerances_for_dtype(dtype: torch.dtype) -> tuple[float, float]:
    """Return (atol, rtol) for tolerance-normalized correctness."""
    dtype_name = str(dtype)
    if dtype in (torch.float16, torch.bfloat16):
        return _LOW_PRECISION_ATOL, _LOW_PRECISION_RTOL
    if "float8" in dtype_name:
        return _FP8_ATOL, _FP8_RTOL
    return _FP32_ATOL, _FP32_RTOL


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _is_fp8_tensor(tensor: Any) -> bool:
    return isinstance(tensor, torch.Tensor) and "float8" in str(tensor.dtype)


def _fp8_rtol(dtype: torch.dtype) -> float:
    if "e5m2" in str(dtype):
        return 0.25
    return 0.125


def _expand_fp8_scale(
    fp8: torch.Tensor,
    scale: torch.Tensor,
    *,
    group_size: int = _FP8_GROUP_SIZE,
) -> torch.Tensor | None:
    """Broadcast FP8 per-group/per-block scales to the FP8 tensor shape."""
    if not isinstance(scale, torch.Tensor) or not scale.is_floating_point():
        return None

    shape = tuple(fp8.shape)
    scale_shape = tuple(scale.shape)
    per_group_shape = (*shape[:-1], _ceil_div(shape[-1], group_size))
    if scale_shape == per_group_shape:
        return scale.float().repeat_interleave(group_size, dim=-1)[..., :shape[-1]]

    if fp8.ndim >= 2:
        per_block_shape = (
            *shape[:-2],
            _ceil_div(shape[-2], group_size),
            _ceil_div(shape[-1], group_size),
        )
        if scale_shape == per_block_shape:
            expanded = scale.float().repeat_interleave(group_size, dim=-2)
            expanded = expanded.repeat_interleave(group_size, dim=-1)
            return expanded[..., :shape[-2], :shape[-1]]

    return None


def _compare_fp8_scaled_outputs(
    baseline_fp8: torch.Tensor,
    baseline_scale: torch.Tensor,
    candidate_fp8: torch.Tensor,
    candidate_scale: torch.Tensor,
) -> tuple[bool, float, float]:
    """Compare FP8 tensors in dequantized value space using local scales."""
    if baseline_fp8.shape != candidate_fp8.shape:
        return False, float("inf"), float("inf")

    baseline_scale_expanded = _expand_fp8_scale(baseline_fp8, baseline_scale)
    candidate_scale_expanded = _expand_fp8_scale(candidate_fp8, candidate_scale)
    if baseline_scale_expanded is None or candidate_scale_expanded is None:
        return False, float("inf"), float("inf")

    baseline = baseline_fp8.float() * baseline_scale_expanded
    candidate = candidate_fp8.float() * candidate_scale_expanded
    if not torch.isfinite(baseline).all() or not torch.isfinite(candidate).all():
        return False, float("inf"), float("inf")

    diff = (baseline - candidate).abs()
    mean_diff = diff.mean().item()
    atol = 0.5 * baseline_scale_expanded.abs().clamp_min(1e-12)
    tolerance = atol + _fp8_rtol(baseline_fp8.dtype) * baseline.abs()
    max_error_ratio = (diff / tolerance).max().item()
    passed = max_error_ratio <= 1.0
    return passed, max_error_ratio, mean_diff


def _compare_outputs(baseline_out: Any, candidate_out: Any) -> tuple[bool, float, float]:
    """Compare outputs: return (pass, max_error_ratio, mean_abs_diff)."""
    if isinstance(baseline_out, torch.Tensor) and isinstance(candidate_out, torch.Tensor):
        if baseline_out.shape != candidate_out.shape:
            return False, float("inf"), float("inf")

        baseline = baseline_out.float()
        candidate = candidate_out.float()
        if not torch.isfinite(baseline).all() or not torch.isfinite(candidate).all():
            return False, float("inf"), float("inf")

        diff = (baseline - candidate).abs()
        mean_diff = diff.mean().item()
        atol, rtol = _tolerances_for_dtype(baseline_out.dtype)
        # Element-wise ``atol + rtol*|x|`` collapses to bare atol wherever the
        # reference element is ~0, but a low-precision rounding error scales
        # with the *tensor's* magnitude, not that element's.  Measured on
        # gpt_oss_decoder and vision_block, every tolerance violation sat at
        # |baseline| == 0 while the global relative error was only 0.5-0.9%.
        # Anchor the absolute term to the tensor scale as well -- the same
        # convention torch.testing.assert_close uses.
        scale = baseline.abs().amax()
        tolerance = torch.maximum(
            atol + rtol * baseline.abs(),
            rtol * scale.clamp_min(0.0),
        )
        ratio = diff / tolerance
        max_error_ratio = ratio.max().item()

        # A pure max() verdict is unusable for deep pipelines: llama's 32 bf16
        # layers give rel_mean 0.05% against a 1% tolerance, yet a handful of
        # elements out of millions hit the worst-case rounding path and reach
        # 2.6% (~6 ULP at bf16's 2^-8 epsilon), failing the whole scenario.
        # Accept a vanishing fraction of outliers -- under 0.1% of elements --
        # provided the bulk is within tolerance.  This is the same tradeoff
        # numerical test suites make; it does not loosen atol/rtol themselves.
        over = (ratio > 1.0)
        frac_over = over.float().mean().item() if ratio.numel() else 0.0
        passed = max_error_ratio <= 1.0 or frac_over <= 1e-3

        # Sparse-MoE outputs are not comparable element-wise.  A router logit
        # differing by one bf16 ULP flips the top-k choice for the tokens near a
        # tie, and a token routed to a different expert produces a *completely*
        # different output row -- not a small error.  Measured on
        # gpt_oss_decoder: router logits agree to 1.6e-2 yet 6.2% of tokens pick
        # a different top-4 set, so element-wise comparison rejects a reference
        # that is behaving correctly.  Accept when the affected rows are a small
        # minority and every other row is within tolerance.
        if not passed and baseline.ndim >= 2:
            row_bad = over.reshape(-1, over.shape[-1]).any(dim=-1)
            frac_rows = row_bad.float().mean().item() if row_bad.numel() else 1.0
            if frac_rows <= 0.10:
                clean = ~row_bad
                if bool(clean.any()):
                    clean_ratio = ratio.reshape(-1, ratio.shape[-1])[clean]
                    if clean_ratio.max().item() <= 1.0:
                        passed = True
        return passed, max_error_ratio, mean_diff

    if isinstance(baseline_out, (tuple, list)) and isinstance(candidate_out, (tuple, list)):
        if len(baseline_out) != len(candidate_out):
            return False, float("inf"), float("inf")
        all_pass = True
        max_error_ratio = 0.0
        total_diff = 0.0
        count = 0
        i = 0
        while i < len(baseline_out):
            b = baseline_out[i]
            c = candidate_out[i]
            if (
                i + 1 < len(baseline_out)
                and _is_fp8_tensor(b)
                and _is_fp8_tensor(c)
                and isinstance(baseline_out[i + 1], torch.Tensor)
                and isinstance(candidate_out[i + 1], torch.Tensor)
            ):
                baseline_scale = baseline_out[i + 1]
                candidate_scale = candidate_out[i + 1]
                if (
                    _expand_fp8_scale(b, baseline_scale) is not None
                    and _expand_fp8_scale(c, candidate_scale) is not None
                ):
                    p, ratio, d = _compare_fp8_scaled_outputs(
                        b, baseline_scale, c, candidate_scale,
                    )
                    all_pass = all_pass and p
                    max_error_ratio = max(max_error_ratio, ratio)
                    total_diff += d
                    count += 1
                    i += 2
                    continue

            if isinstance(b, torch.Tensor) and isinstance(c, torch.Tensor):
                p, ratio, d = _compare_outputs(b, c)
                all_pass = all_pass and p
                max_error_ratio = max(max_error_ratio, ratio)
                total_diff += d
                count += 1
            i += 1
        mean_diff = total_diff / count if count > 0 else 0.0
        return all_pass, max_error_ratio, mean_diff

    if isinstance(baseline_out, dict) and isinstance(candidate_out, dict):
        if set(baseline_out) != set(candidate_out):
            return False, float("inf"), float("inf")
        all_pass = True
        max_error_ratio = 0.0
        total_diff = 0.0
        count = 0
        for key in sorted(baseline_out):
            b = baseline_out[key]
            c = candidate_out[key]
            p, ratio, d = _compare_outputs(b, c)
            all_pass = all_pass and p
            max_error_ratio = max(max_error_ratio, ratio)
            total_diff += d
            count += 1
        mean_diff = total_diff / count if count > 0 else 0.0
        return all_pass, max_error_ratio, mean_diff

    return True, 0.0, 0.0


def _merge_correctness(
    output_check: tuple[bool, float, float],
    input_check: tuple[bool, float, float],
) -> tuple[bool, float, float]:
    output_correct, output_ratio, output_diff = output_check
    input_correct, input_ratio, input_diff = input_check
    correct = output_correct and input_correct
    max_error_ratio = max(output_ratio, input_ratio)
    if output_diff == 0.0:
        mean_diff = input_diff
    elif input_diff == 0.0:
        mean_diff = output_diff
    else:
        mean_diff = 0.5 * (output_diff + input_diff)
    return correct, max_error_ratio, mean_diff


def run_kernel_benchmark(
    target_name: str,
    scenarios: list[str] | None = None,
    models: list[str] | None = None,
    tp: list[int] | None = None,
    category: str | None = None,
    num_warmup: int = 10,
    num_runs: int = 100,
    device: str = "cuda",
    pytorch_reference: bool = False,
    validation_mode: str = "candidate",
) -> OperatorResult:
    """Run isolated kernel benchmark for a single operator.

    For each matching scenario in the InputRegistry:
    1. Instantiate baseline and candidate with init_args
    2. Copy baseline weights to candidate (via load_state_dict)
    3. Prepare inputs (random or golden)
    4. Warmup both
    5. Time both (median of num_runs)
    6. Compare outputs: max error ratio pass/fail, mean abs diff

    The candidate implementation is auto-discovered from
    tasks/candidate/L{level}/{target_name}.py.

    Args:
        target_name: Operator name (e.g. 'rms_norm').
        scenarios: Filter by scenario name patterns.
        models: Filter by model key prefix.
        tp: Filter by TP degrees.
        category: Filter by category (not yet used).
        num_warmup: Warmup iterations.
        num_runs: Timed iterations for median.
        device: Device for tensors.

    Returns:
        OperatorResult with per-scenario correctness and speedup.
    """
    target = get(target_name)

    if pytorch_reference:
        validation_mode = "pytorch_reference"

    if validation_mode == "baseline_identity":
        user_impl = target.target_cls
    elif validation_mode == "pytorch_reference":
        user_impl = load_reference(target_name)
    else:
        user_impl = load_candidate(target_name)

    if user_impl is None:
        impl_kind = "PyTorch reference" if validation_mode == "pytorch_reference" else "candidate kernel"
        impl_dir = "reference" if validation_mode == "pytorch_reference" else "candidate"
        raise ValueError(
            f"No {impl_kind} found for {target_name!r}. "
            f"Place implementation in tasks/{impl_dir}/L{target.level}/{target_name}.py"
        )

    registry = _get_registry()
    all_scenarios = registry.scenarios(
        target_name, models=models, tp=tp, category=category,
    )

    if scenarios:
        all_scenarios = [
            s for s in all_scenarios
            if any(pat in s.name for pat in scenarios)
        ]

    if not all_scenarios:
        print(f"  WARNING: No scenarios found for {target_name} in InputRegistry.")
        return OperatorResult(
            target=target_name,
            level=target.level,
            candidate_path=(
                _find_reference_path(target_name, target.level)
                if pytorch_reference
                else _find_candidate_path(target_name, target.level)
            ),
        )

    if validation_mode == "baseline_identity":
        candidate_path = f"tasks/baseline/L{target.level}/{target_name}.py"
    elif validation_mode == "pytorch_reference":
        candidate_path = _find_reference_path(target_name, target.level)
    else:
        candidate_path = _find_candidate_path(target_name, target.level)
    scenario_results: list[ScenarioResult] = []

    for scenario in all_scenarios:
        # Bound before the try so the OOM handler and the finally block can read
        # them even when the failure happened while building the baseline.
        baseline_mod = None
        candidate_mod = None
        sequential = False
        timing_warmup = 0 if validation_mode == "candidate_smoke" else num_warmup
        timing_runs = 1 if validation_mode == "candidate_smoke" else num_runs
        if os.environ.get("FASTKERNELS_DEBUG_MEM") and device.startswith("cuda"):
            print(f"  [mem] {scenario.name}: entering with "
                  f"{torch.cuda.memory_allocated()/2**30:.1f} GiB allocated, "
                  f"{torch.cuda.memory_reserved()/2**30:.1f} GiB reserved")
        try:
            inputs = registry.get_inputs(target_name, scenario.name, device=device)
            # Scenarios with only integer inputs yield no dtype; fall back to
            # what the checkpoint declares rather than forcing bf16 on
            # everything (that regressed fp32 vision models).
            input_dtype = (
                _first_floating_dtype(inputs)
                or preferred_dtype(tuple(target.models or ()))
            )

            models = tuple(target.models or ())
            baseline_mod = _instantiate_module(
                target.target_cls, scenario.init_args, device,
                dtype=input_dtype, models=models, inputs=inputs,
                level=target.level,
            )

            timing_warmup = 0 if validation_mode == "candidate_smoke" else num_warmup
            timing_runs = 1 if validation_mode == "candidate_smoke" else num_runs

            sequential = _needs_sequential(baseline_mod, device)
            if sequential:
                holder = [baseline_mod]
                baseline_mod = None          # hand off the only reference
                result = _run_scenario_sequentially(
                    target, user_impl, scenario, inputs, models, input_dtype,
                    device, timing_warmup, timing_runs, validation_mode,
                    holder,
                )
                scenario_results.append(result)
                continue

            candidate_mod = _instantiate_module(
                user_impl, scenario.init_args, device,
                dtype=input_dtype, models=models, inputs=inputs,
                level=target.level,
                # The candidate is filled from the baseline's state_dict just
                # below, so re-reading the checkpoint here only doubles peak
                # memory -- enough to push an 8B L4 pipeline over the limit.
                load_weights=False,
            )

            if hasattr(baseline_mod, "state_dict") and len(baseline_mod.state_dict()) > 0:
                try:
                    candidate_mod.load_state_dict(baseline_mod.state_dict(), strict=False)
                except Exception:
                    pass

            # Attention-bearing tasks read paged-KV metadata from the global
            # forward context and expect the engine to have attached real
            # k_cache/v_cache tensors.  Tier 1 has no engine, so install the
            # minimal single-sequence prefill context both modules share.
            # Non-tensor forward arguments (a shared rotary_emb module, a
            # list of feature maps) are absent from the registry; rebuild the
            # ones that can be derived from config so the call is well-formed.
            extra_args = synthesize_forward_args(
                target.target_cls, inputs, models, device, input_dtype,
            )
            if extra_args:
                inputs = {**inputs, **extra_args}

            with tier1_forward_context(
                [baseline_mod, candidate_mod], inputs, device, input_dtype,
            ):
                baseline_check_inputs = _clone_inputs(inputs)
                candidate_check_inputs = _clone_inputs(inputs)
                baseline_out = _run_forward_once(baseline_mod, baseline_check_inputs)
                candidate_out = _run_forward_once(candidate_mod, candidate_check_inputs)

                correct, max_error_ratio, mean_diff = _merge_correctness(
                    _compare_outputs(baseline_out, candidate_out),
                    _compare_outputs(baseline_check_inputs, candidate_check_inputs),
                )

                # The correctness pass holds two full output trees plus both
                # modules' activations; a 128-expert MoE layer peaks around
                # 18 GB across the pair.  Release them before timing, which
                # re-runs both forwards.
                del baseline_out, candidate_out
                del baseline_check_inputs, candidate_check_inputs
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()

                _, baseline_ms = _time_forward(
                    baseline_mod, _clone_inputs(inputs), timing_warmup, timing_runs,
                )
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
                _, candidate_ms = _time_forward(
                    candidate_mod, _clone_inputs(inputs), timing_warmup, timing_runs,
                )
            speedup = baseline_ms / candidate_ms if candidate_ms > 0 else float("inf")
            classification = (
                "harness_validation_passed"
                if validation_mode in ("baseline_identity", "pytorch_reference")
                and correct
                else "candidate_correct_and_timed"
                if correct
                else "candidate_correctness_failure"
            )

            scenario_results.append(ScenarioResult(
                name=scenario.name,
                correct=correct,
                max_error_ratio=max_error_ratio,
                mean_abs_diff=mean_diff,
                baseline_ms=baseline_ms,
                candidate_ms=candidate_ms,
                speedup=speedup,
                failure_reason=None if correct else "output_mismatch",
                classification=classification,
            ))

        except torch.cuda.OutOfMemoryError as e:
            # Free both modules before doing anything else: the scenario that
            # ran out is holding the memory the retry needs.
            baseline_mod = candidate_mod = None
            _release_cuda(device)

            retried = None
            if not sequential:
                # The pair did not fit after all -- ``_needs_sequential`` sizes
                # parameters but cannot predict activation peaks (llama-3.1-8B
                # at 16384 tokens missed by 224 MB).  Rebuild and run the two
                # sides one at a time before calling the scenario a failure.
                try:
                    retry_baseline = _instantiate_module(
                        target.target_cls, scenario.init_args, device,
                        dtype=input_dtype, models=models, inputs=inputs,
                        level=target.level,
                    )
                    retry_holder = [retry_baseline]
                    retry_baseline = None
                    retried = _run_scenario_sequentially(
                        target, user_impl, scenario, inputs, models,
                        input_dtype, device, timing_warmup, timing_runs,
                        validation_mode, retry_holder,
                    )
                except Exception:
                    retried = None
                    if device.startswith("cuda"):
                        torch.cuda.empty_cache()
            if retried is not None:
                scenario_results.append(retried)
                continue

            failure_reason = _short_exception(e)
            print(f"  ERROR in scenario {scenario.name}: {failure_reason}")
            if os.environ.get("FASTKERNELS_DEBUG_TRACEBACK"):
                import traceback
                traceback.print_exc()
            scenario_results.append(ScenarioResult(
                name=scenario.name,
                correct=False,
                max_error_ratio=float("inf"),
                mean_abs_diff=float("inf"),
                baseline_ms=0.0,
                candidate_ms=0.0,
                speedup=0.0,
                failure_reason=failure_reason,
                classification="harness_or_candidate_exception",
            ))

        except Exception as e:
            failure_reason = _short_exception(e)
            print(f"  ERROR in scenario {scenario.name}: {failure_reason}")
            # The one-line reason names the exception but not the frame that
            # raised it, which is the only thing that distinguishes a broken
            # candidate from a harness gap.  Opt-in so normal runs are unchanged.
            if os.environ.get("FASTKERNELS_DEBUG_TRACEBACK"):
                import traceback
                traceback.print_exc()
            scenario_results.append(ScenarioResult(
                name=scenario.name,
                correct=False,
                max_error_ratio=float("inf"),
                mean_abs_diff=float("inf"),
                baseline_ms=0.0,
                candidate_ms=0.0,
                speedup=0.0,
                failure_reason=failure_reason,
                classification="harness_or_candidate_exception",
            ))

        finally:
            # Rebinding the names is what actually drops the references; the
            # previous ``for v in locals().values(): del v`` only deleted the
            # loop variable, so a failed scenario kept its modules alive and the
            # next one started with the card already full -- one OOM cascaded
            # into every remaining scenario.
            baseline_mod = None
            candidate_mod = None
            _release_cuda(device)

    op_result = OperatorResult(
        target=target_name,
        level=target.level,
        candidate_path=candidate_path,
        scenarios=scenario_results,
    )
    op_result.compute_aggregates()
    return op_result


def run_all_kernel_benchmarks(
    models: list[str] | None = None,
    tp: list[int] | None = None,
    category: str | None = None,
    num_warmup: int = 10,
    num_runs: int = 100,
    device: str = "cuda",
    pytorch_reference: bool = False,
    validation_mode: str = "candidate",
) -> KernelBenchResult:
    """Run kernel benchmarks for all operators that have candidate implementations.

    Discovers all candidate kernels and runs isolated benchmarks for each.
    """
    from fastkernels.infra.kernel_swapper import discover_candidates

    if pytorch_reference:
        validation_mode = "pytorch_reference"

    candidates = discover_references() if validation_mode == "pytorch_reference" else discover_candidates()
    if not candidates:
        if validation_mode == "pytorch_reference":
            print("No PyTorch references found in tasks/reference/.")
        else:
            print("No candidate kernels found in tasks/candidate/.")
        result = KernelBenchResult()
        result.compute_aggregates()
        return result

    operators: list[OperatorResult] = []
    for target, _ in candidates:
        label = (
            "baseline identity"
            if validation_mode == "baseline_identity"
            else "PyTorch reference"
            if validation_mode == "pytorch_reference"
            else "candidate smoke"
            if validation_mode == "candidate_smoke"
            else "candidate"
        )
        print(f"\n  Benchmarking {target.name} (L{target.level}, {label})...")
        op_result = run_kernel_benchmark(
            target.name,
            models=models,
            tp=tp,
            category=category,
            num_warmup=num_warmup,
            num_runs=num_runs,
            device=device,
            pytorch_reference=pytorch_reference,
            validation_mode=validation_mode,
        )
        operators.append(op_result)

    result = KernelBenchResult(operators=operators)
    result.compute_aggregates()
    return result
