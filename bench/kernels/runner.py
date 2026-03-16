"""Isolated kernel-level benchmarking via direct forward() calls.

Instantiates baseline and candidate nn.Module instances, copies weights,
loads inputs from the InputRegistry (random or golden), compares outputs
and timing. No full model build required — per-kernel test time is seconds
rather than minutes.
"""

from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn as nn

from kb_nano.bench.utils.input_registry import InputRegistry
from kb_nano.infra.kernel_swapper import BenchTarget, discover_targets, get, load_candidate

from .result import KernelBenchResult, NcuProfileResult, OperatorResult, ScenarioResult

_DEFAULT_REGISTRY = None


def _get_registry() -> InputRegistry:
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = InputRegistry()
    return _DEFAULT_REGISTRY


def _find_candidate_path(target_name: str, level: int) -> str:
    """Return the relative path to the candidate file for display."""
    return f"tasks/candidate/L{level}/{target_name}.py"


def _instantiate_module(
    cls: type,
    init_args: dict[str, Any],
    device: str = "cuda",
) -> nn.Module:
    """Create an nn.Module instance with init_args, handling common patterns."""
    try:
        module = cls(**init_args)
    except TypeError:
        module = cls()

    module = module.to(device)
    module.eval()
    return module


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

        torch.cuda.synchronize()
        times = []
        output = None
        for _ in range(num_runs):
            start = time.perf_counter()
            output = module(**tensor_inputs, **scalar_inputs)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)

    times.sort()
    median_ms = times[len(times) // 2]
    return output, median_ms


def _compare_outputs(baseline_out: Any, candidate_out: Any) -> tuple[bool, float]:
    """Compare outputs: return (allclose_pass, mean_abs_diff)."""
    if isinstance(baseline_out, torch.Tensor) and isinstance(candidate_out, torch.Tensor):
        if baseline_out.shape != candidate_out.shape:
            return False, float("inf")
        diff = (baseline_out.float() - candidate_out.float()).abs()
        mean_diff = diff.mean().item()
        passed = torch.allclose(
            baseline_out.float(), candidate_out.float(),
            atol=1e-5, rtol=1e-3,
        )
        return passed, mean_diff

    if isinstance(baseline_out, (tuple, list)) and isinstance(candidate_out, (tuple, list)):
        if len(baseline_out) != len(candidate_out):
            return False, float("inf")
        all_pass = True
        total_diff = 0.0
        count = 0
        for b, c in zip(baseline_out, candidate_out):
            if isinstance(b, torch.Tensor) and isinstance(c, torch.Tensor):
                p, d = _compare_outputs(b, c)
                all_pass = all_pass and p
                total_diff += d
                count += 1
        mean_diff = total_diff / count if count > 0 else 0.0
        return all_pass, mean_diff

    return True, 0.0


def run_kernel_benchmark(
    target_name: str,
    scenarios: list[str] | None = None,
    models: list[str] | None = None,
    tp: list[int] | None = None,
    category: str | None = None,
    num_warmup: int = 10,
    num_runs: int = 100,
    device: str = "cuda",
    profile: bool = False,
    num_ncu_launches: int = 20,
) -> OperatorResult:
    """Run isolated kernel benchmark for a single operator.

    For each matching scenario in the InputRegistry:
    1. Instantiate baseline and candidate with init_args
    2. Copy baseline weights to candidate (via load_state_dict)
    3. Prepare inputs (random or golden)
    4. Warmup both
    5. Time both (median of num_runs)
    6. Compare outputs: allclose pass/fail, mean abs diff

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
        profile: Run NCU profiling on baseline and candidate.
        num_ncu_launches: Number of forward() calls NCU captures.

    Returns:
        OperatorResult with per-scenario correctness and speedup.
    """
    target = get(target_name)

    user_impl = load_candidate(target_name)
    if user_impl is None:
        raise ValueError(
            f"No candidate kernel found for {target_name!r}. "
            f"Place kernel in tasks/candidate/L{target.level}/{target_name}.py"
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
            candidate_path=_find_candidate_path(target_name, target.level),
        )

    candidate_path = _find_candidate_path(target_name, target.level)
    scenario_results: list[ScenarioResult] = []

    for scenario in all_scenarios:
        try:
            baseline_mod = _instantiate_module(target.target_cls, scenario.init_args, device)
            candidate_mod = _instantiate_module(user_impl, scenario.init_args, device)

            if hasattr(baseline_mod, "state_dict") and len(baseline_mod.state_dict()) > 0:
                try:
                    candidate_mod.load_state_dict(baseline_mod.state_dict(), strict=False)
                except Exception:
                    pass

            inputs = registry.get_inputs(target_name, scenario.name, device=device)

            baseline_out, baseline_ms = _time_forward(
                baseline_mod, inputs, num_warmup, num_runs,
            )
            candidate_out, candidate_ms = _time_forward(
                candidate_mod, inputs, num_warmup, num_runs,
            )

            correct, mean_diff = _compare_outputs(baseline_out, candidate_out)
            speedup = baseline_ms / candidate_ms if candidate_ms > 0 else float("inf")

            baseline_ncu = None
            candidate_ncu = None
            if profile:
                try:
                    from kb_nano import KB_ROOT
                    from .ncu_profiler import profile_scenario as _profile_scenario

                    candidate_file = str(
                        KB_ROOT / "tasks" / "candidate" / f"L{target.level}" / f"{target_name}.py"
                    )
                    ncu_result = _profile_scenario(
                        target_name=target_name,
                        scenario=scenario,
                        target=target,
                        candidate_path=candidate_file,
                        num_launches=num_ncu_launches,
                    )
                    if ncu_result is not None:
                        baseline_ncu = NcuProfileResult(
                            metrics_json=ncu_result["baseline_metrics"],
                        )
                        candidate_ncu = NcuProfileResult(
                            metrics_json=ncu_result["candidate_metrics"],
                        )
                except Exception as e:
                    print(f"  WARNING: NCU profiling failed for {scenario.name}: {e}")

            scenario_results.append(ScenarioResult(
                name=scenario.name,
                correct=correct,
                mean_abs_diff=mean_diff,
                baseline_ms=baseline_ms,
                candidate_ms=candidate_ms,
                speedup=speedup,
                baseline_ncu=baseline_ncu,
                candidate_ncu=candidate_ncu,
            ))

        except Exception as e:
            print(f"  ERROR in scenario {scenario.name}: {e}")
            scenario_results.append(ScenarioResult(
                name=scenario.name,
                correct=False,
                mean_abs_diff=float("inf"),
                baseline_ms=0.0,
                candidate_ms=0.0,
                speedup=0.0,
            ))

        finally:
            for v in list(locals().values()):
                if isinstance(v, nn.Module):
                    del v

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
    profile: bool = False,
    num_ncu_launches: int = 20,
) -> KernelBenchResult:
    """Run kernel benchmarks for all operators that have candidate implementations.

    Discovers all candidate kernels and runs isolated benchmarks for each.
    """
    from kb_nano.infra.kernel_swapper import discover_candidates

    candidates = discover_candidates()
    if not candidates:
        print("No candidate kernels found in tasks/candidate/.")
        result = KernelBenchResult()
        result.compute_aggregates()
        return result

    operators: list[OperatorResult] = []
    for target, _ in candidates:
        print(f"\n  Benchmarking {target.name} (L{target.level})...")
        op_result = run_kernel_benchmark(
            target.name,
            models=models,
            tp=tp,
            category=category,
            num_warmup=num_warmup,
            num_runs=num_runs,
            device=device,
            profile=profile,
            num_ncu_launches=num_ncu_launches,
        )
        operators.append(op_result)

    result = KernelBenchResult(operators=operators)
    result.compute_aggregates()
    return result
