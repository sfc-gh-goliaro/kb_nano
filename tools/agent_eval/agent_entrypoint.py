#!/usr/bin/env python3
"""AKO4X -> fastkernels benchmark entrypoint (single-operator, subprocess CLI).

Usage
-----
    agent_entrypoint.py --op rms_norm --candidate /path/to/kernel.py
    agent_entrypoint.py --op rms_norm --baseline-identity
    agent_entrypoint.py --op rms_norm --candidate k.py --scenarios tokens-1,tokens-4

Contract
--------
Writes EXACTLY one JSON object to stdout (the AKO4X normalized result dict, see
``AKO4X/scripts/benchmark_adapter.py``)::

    {definition_name: {workload_uuid: {"status", "solution", "axes",
        "latency_ms", "reference_latency_ms", "speedup_factor",
        "max_abs_error", "max_rel_error", "error_log"}}}

definition_name = f"kb_{op}"; workload_uuid = the scenario name. All logging goes
to stderr (fd 1 is redirected to fd 2 for the whole run so that library banners
cannot corrupt the JSON; the real stdout fd is held aside and written at exit).

Exit codes: 0 when the benchmark ran (scenario failures are *data*, not errors);
2 on infrastructure errors (bad args, unknown op, tree/candidate import failure,
no CUDA, empty scenario selection).

Correctness / timing semantics are the release runner's
(``fastkernels/bench/kernels/runner.py``) -- its comparison helpers, its
tolerances, its median timing are imported and reused, not reimplemented. Two
deliberate differences from ``run_kernel_benchmark``:

1. **Strict weight transfer.** The runner wraps ``load_state_dict`` in a bare
   ``try/except pass`` (runner.py:496-500), so a candidate whose parameters do
   not line up with the baseline's silently runs on its own initialisation. Here
   the transfer is unconditional and any raise / non-empty ``missing_keys`` /
   non-empty ``unexpected_keys`` fails the scenario with RUNTIME_ERROR.
2. **Targeted baseline discovery.** ``kernel_swapper.get()`` calls
   ``discover_targets()``, which imports *every* baseline module in the tree and
   dies in this environment on an unrelated optional dependency
   (``tasks/baseline/L2/pointtransformerv3_layers.py`` -> ``import spconv``).
   ``_resolve_target`` below reproduces ``discover_targets``'s per-op logic
   (same module path, same ``kernel_swapper._find_module_class``) for one op only.

Tolerances are NOT settable from the CLI: they are read from the runner's module
constants. A calling agent must not be able to loosen its own correctness gate.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import math
import os
import shutil
import sys
import traceback
from typing import Any

# --- stdout quarantine -------------------------------------------------------
# Hold the real stdout fd aside and point fd 1 at stderr, so that anything the
# imported stack prints (vLLM/flashinfer banners, the runner's own print()s)
# lands on stderr instead of corrupting the single JSON object we emit.
_REAL_STDOUT_FD = os.dup(1)
os.dup2(2, 1)


def _log(msg: str) -> None:
    print(f"[agent_entrypoint] {msg}", file=sys.stderr, flush=True)


def _emit_json(obj: Any) -> None:
    payload = json.dumps(obj, allow_nan=False) + "\n"
    sys.stdout.flush()
    os.write(_REAL_STDOUT_FD, payload.encode())


class InfraError(RuntimeError):
    """Whole-run failure: nothing meaningful to report, exit nonzero."""


# --- AKO4X status constants (mirrored from benchmark_adapter.py) --------------
STATUS_PASSED = "PASSED"
STATUS_COMPILE_ERROR = "COMPILE_ERROR"
STATUS_INCORRECT_NUMERICAL = "INCORRECT_NUMERICAL"
STATUS_RUNTIME_ERROR = "RUNTIME_ERROR"
STATUS_TIMEOUT = "TIMEOUT"

# Runner defaults, pinned here so the CLI exposes no timing knob either.
NUM_WARMUP = 10
NUM_RUNS = 100


# --- fastkernels tree bootstrap ---------------------------------------------

def _tree_candidates() -> list[str]:
    roots: list[str] = []
    env_tree = os.environ.get("FASTKERNELS_TREE")
    if env_tree:
        roots.append(env_tree)
    roots.extend(p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p)
    roots.append(os.getcwd())
    seen, out = set(), []
    for r in roots:
        a = os.path.abspath(r)
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out


def _looks_like_tree(root: str) -> bool:
    return all(os.path.isfile(os.path.join(root, *p)) for p in (
        ("__init__.py",),
        ("bench", "kernels", "runner.py"),
        ("infra", "kernel_swapper.py"),
    ))


def _bootstrap_fastkernels() -> str:
    """Register the on-disk tree as the ``fastkernels`` package and return its root.

    The tree declares ``[tool.setuptools.package-dir] "fastkernels" = "."``, i.e.
    the repo root *is* the package. Rather than requiring a symlink named
    ``fastkernels`` on sys.path, bind the name explicitly here -- this also wins
    over any installed/editable copy, because the name is in sys.modules before
    anything imports it.
    """
    for root in _tree_candidates():
        if not _looks_like_tree(root):
            continue
        spec = importlib.util.spec_from_file_location(
            "fastkernels", os.path.join(root, "__init__.py"),
            submodule_search_locations=[root],
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["fastkernels"] = module
        spec.loader.exec_module(module)
        return root
    raise InfraError(
        "could not locate a fastkernels tree. Set FASTKERNELS_TREE or put the "
        f"repo root on PYTHONPATH. Looked in: {_tree_candidates()}"
    )


def _ensure_ninja_on_path() -> None:
    """L1 baselines JIT-build a CUDA extension; torch needs ninja on PATH."""
    if shutil.which("ninja") is None:
        bindir = os.path.dirname(os.path.abspath(sys.executable))
        os.environ["PATH"] = bindir + os.pathsep + os.environ.get("PATH", "")
        _log(f"ninja not on PATH; prepended {bindir}")


# --- target / candidate resolution ------------------------------------------

def _resolve_target(op: str):
    """BenchTarget for one op without importing the whole baseline corpus."""
    from fastkernels import KB_ROOT
    from fastkernels.infra.kernel_swapper import BenchTarget, _find_module_class

    for level in (1, 2, 3, 4):
        path = KB_ROOT / "tasks" / "baseline" / f"L{level}" / f"{op}.py"
        if not path.is_file():
            continue
        module_path = f"tasks.baseline.L{level}.{op}"
        mod = importlib.import_module(f"fastkernels.{module_path}")
        cls = _find_module_class(mod)
        if cls is None:
            raise InfraError(f"no nn.Module class found in {path}")
        _log(f"baseline module: {mod.__file__}")
        _log(f"baseline class : {cls.__name__} (L{level})")
        return BenchTarget(
            name=op, level=level, module_path=module_path, models=[],
            target_cls=cls, requires_recompile=(level == 1),
        )
    raise InfraError(f"no baseline file tasks/baseline/L*/{op}.py under {KB_ROOT}")


def _load_candidate_from_path(path: str, baseline_cls: type) -> type:
    """Path-taking port of ``kernel_swapper.load_candidate``.

    Copied (not imported) because the upstream function derives its path from
    ``CANDIDATE_DIR`` and takes no file argument. Selection logic is identical:
    exec the file, prefer the class named like the baseline, else the first
    nn.Module subclass.
    """
    import torch.nn as nn

    if not os.path.isfile(path):
        raise InfraError(f"candidate file not found: {path}")
    module_name = "_agent_candidate_impl"
    spec = importlib.util.spec_from_file_location(module_name, os.path.abspath(path))
    if spec is None or spec.loader is None:
        raise InfraError(f"cannot build an import spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        sys.modules.pop(module_name, None)
        raise InfraError(
            f"candidate import failed: {type(exc).__name__}: {exc}\n"
            + traceback.format_exc()
        ) from exc
    cls = getattr(mod, baseline_cls.__name__, None)
    if cls is None:
        for v in vars(mod).values():
            if isinstance(v, type) and issubclass(v, nn.Module) and v is not nn.Module:
                cls = v
                break
    if cls is None:
        raise InfraError(f"no nn.Module subclass found in {path}")
    return cls


# --- reporting helpers -------------------------------------------------------

def _scenario_axes(scenario) -> dict[str, Any]:
    """Flat, JSON-safe axes from the scenario's shape dict.

    AKO4X consumers format axes as ``k=v`` pairs and filter on scalar values, so
    each declared shape is flattened to ``<arg>_dim<i>`` plus ``<arg>_dtype``.
    """
    axes: dict[str, Any] = {}
    for arg, spec in scenario.inputs.items():
        if isinstance(spec, dict) and "shape" in spec:
            for i, dim in enumerate(spec["shape"]):
                axes[f"{arg}_dim{i}"] = int(dim)
            if spec.get("dtype") is not None:
                axes[f"{arg}_dtype"] = str(spec["dtype"])
        elif spec is None or isinstance(spec, (int, float, str, bool)):
            axes[arg] = spec
    return axes


def _num(x: Any) -> Any:
    """JSON-safe float: non-finite -> "NaN" (the contract's sentinel)."""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return "NaN"
    return f if math.isfinite(f) else "NaN"


def _abs_rel_errors(baseline_out: Any, candidate_out: Any) -> tuple[float, float]:
    """max |b-c| and max |b-c|/|b| over a matching output tree.

    Reported only; the pass/fail decision belongs entirely to the runner's
    tolerance-normalized ``_compare_outputs``.
    """
    import torch

    if isinstance(baseline_out, torch.Tensor) and isinstance(candidate_out, torch.Tensor):
        if baseline_out.shape != candidate_out.shape:
            return float("inf"), float("inf")
        b = baseline_out.float()
        c = candidate_out.float()
        diff = (b - c).abs()
        denom = b.abs()
        rel = torch.where(
            denom > 0, diff / denom.clamp_min(torch.finfo(torch.float32).tiny),
            torch.where(diff > 0, torch.full_like(diff, float("inf")),
                        torch.zeros_like(diff)),
        )
        return diff.max().item(), rel.max().item()

    if isinstance(baseline_out, (tuple, list)) and isinstance(candidate_out, (tuple, list)):
        if len(baseline_out) != len(candidate_out):
            return float("inf"), float("inf")
        pairs = zip(baseline_out, candidate_out)
    elif isinstance(baseline_out, dict) and isinstance(candidate_out, dict):
        if set(baseline_out) != set(candidate_out):
            return float("inf"), float("inf")
        pairs = ((baseline_out[k], candidate_out[k]) for k in sorted(baseline_out))
    else:
        return 0.0, 0.0

    max_abs = max_rel = 0.0
    for b, c in pairs:
        a, r = _abs_rel_errors(b, c)
        max_abs = max(max_abs, a)
        max_rel = max(max_rel, r)
    return max_abs, max_rel


# --- the benchmark ------------------------------------------------------------

def run(op: str, candidate_path: str | None, scenario_filters: list[str] | None,
        baseline_identity: bool) -> dict[str, Any]:
    import torch

    from fastkernels.bench.kernels import runner as R
    from fastkernels.bench.kernels.scenario_registry import InputRegistry

    _log(f"runner.__file__ = {R.__file__}")
    _log(f"scenario_registry -> {InputRegistry.__module__}")
    _log(f"tolerances (from runner constants): fp32 atol={R._FP32_ATOL} rtol={R._FP32_RTOL} "
         f"| low-precision atol={R._LOW_PRECISION_ATOL} rtol={R._LOW_PRECISION_RTOL} "
         f"| fp8 atol={R._FP8_ATOL} rtol={R._FP8_RTOL}")

    if not torch.cuda.is_available():
        raise InfraError("no CUDA device visible")
    _log(f"device: {torch.cuda.get_device_name(0)}")

    target = _resolve_target(op)

    if baseline_identity:
        user_impl = target.target_cls
        solution = "baseline_identity"
    else:
        user_impl = _load_candidate_from_path(candidate_path, target.target_cls)
        solution = os.path.abspath(candidate_path)
        _log(f"candidate class: {user_impl.__name__} from {candidate_path}")

    registry = InputRegistry()
    scenarios = registry.scenarios(op)
    if not scenarios:
        raise InfraError(f"no scenarios registered for operator {op!r}")
    if scenario_filters:
        names = {s.name for s in scenarios}
        selected, unmatched = [], []
        for pat in scenario_filters:
            hits = [s for s in scenarios if s.name == pat] or \
                   [s for s in scenarios if pat in s.name]
            if not hits:
                unmatched.append(pat)
            selected.extend(hits)
        if unmatched:
            raise InfraError(
                f"--scenarios patterns matched nothing for {op!r}: {unmatched}. "
                f"{len(names)} scenarios available."
            )
        seen: set[str] = set()
        deduped = []
        for s in selected:
            if s.name not in seen:
                seen.add(s.name)
                deduped.append(s)
        scenarios = deduped
    _log(f"operator {op!r}: {len(scenarios)} scenario(s) selected")

    definition = f"kb_{op}"
    results: dict[str, Any] = {}

    for scenario in scenarios:
        entry: dict[str, Any] = {
            "status": STATUS_RUNTIME_ERROR,
            "solution": solution,
            "axes": _scenario_axes(scenario),
        }
        try:
            inputs = registry.get_inputs(op, scenario.name, device="cuda")
            input_dtype = R._first_floating_dtype(inputs)

            baseline_mod = R._instantiate_module(
                target.target_cls, scenario.init_args, "cuda", dtype=input_dtype)
            candidate_mod = R._instantiate_module(
                user_impl, scenario.init_args, "cuda", dtype=input_dtype)

            # --- strict weight transfer (tightens runner.py:496-500) ---
            # Perturb the baseline's floating-point parameters (deterministic,
            # seeded) BEFORE the transfer. Several kb baselines initialise
            # parameters to identity values (rms_norm: weight = ones), which
            # makes "candidate forgot to apply the weight" invisible to the
            # comparison. Both modules see the SAME perturbed values, so
            # correct candidates are unaffected; degenerate-parameter blind
            # spots are closed. Measured before this fix: a no-weight-multiply
            # candidate PASSED every scenario with max_abs_error 0.0.
            _pgen = torch.Generator(device="cpu").manual_seed(0x5EED)
            with torch.no_grad():
                for _pname, _p in baseline_mod.named_parameters():
                    if _p.is_floating_point():
                        _scale = torch.empty(_p.shape, dtype=torch.float32)
                        _scale.uniform_(0.75, 1.25, generator=_pgen)
                        _shift = torch.empty(_p.shape, dtype=torch.float32)
                        _shift.uniform_(-0.05, 0.05, generator=_pgen)
                        _p.mul_(_scale.to(device=_p.device, dtype=_p.dtype))
                        _p.add_(_shift.to(device=_p.device, dtype=_p.dtype))
            baseline_sd = baseline_mod.state_dict()
            try:
                incompatible = candidate_mod.load_state_dict(baseline_sd, strict=False)
            except Exception as exc:
                entry["error_log"] = (
                    "weight_transfer_failed: load_state_dict raised "
                    f"{R._short_exception(exc)}. baseline keys="
                    f"{sorted(baseline_sd)} candidate keys="
                    f"{sorted(candidate_mod.state_dict())}"
                )
                results[scenario.name] = entry
                continue
            missing = list(getattr(incompatible, "missing_keys", []))
            unexpected = list(getattr(incompatible, "unexpected_keys", []))
            if missing or unexpected:
                entry["error_log"] = (
                    "weight_transfer_incomplete: load_state_dict(strict=False) "
                    f"reported missing_keys={missing} unexpected_keys={unexpected}. "
                    "The candidate would have run on its own initialisation, so "
                    "any correctness result would be meaningless."
                )
                results[scenario.name] = entry
                continue

            # --- correctness: the runner's own comparison, unmodified ---
            baseline_check_inputs = R._clone_inputs(inputs)
            candidate_check_inputs = R._clone_inputs(inputs)
            baseline_out = R._run_forward_once(baseline_mod, baseline_check_inputs)
            candidate_out = R._run_forward_once(candidate_mod, candidate_check_inputs)

            correct, max_error_ratio, mean_diff = R._merge_correctness(
                R._compare_outputs(baseline_out, candidate_out),
                R._compare_outputs(baseline_check_inputs, candidate_check_inputs),
            )
            out_abs, out_rel = _abs_rel_errors(baseline_out, candidate_out)
            in_abs, in_rel = _abs_rel_errors(baseline_check_inputs, candidate_check_inputs)
            max_abs, max_rel = max(out_abs, in_abs), max(out_rel, in_rel)

            # --- timing: the runner's median-of-N ---
            _, baseline_ms = R._time_forward(
                baseline_mod, R._clone_inputs(inputs), NUM_WARMUP, NUM_RUNS)
            _, candidate_ms = R._time_forward(
                candidate_mod, R._clone_inputs(inputs), NUM_WARMUP, NUM_RUNS)

            entry["status"] = STATUS_PASSED if correct else STATUS_INCORRECT_NUMERICAL
            entry["latency_ms"] = _num(candidate_ms)
            entry["reference_latency_ms"] = _num(baseline_ms)
            entry["speedup_factor"] = _num(
                baseline_ms / candidate_ms if candidate_ms > 0 else float("inf"))
            entry["max_abs_error"] = _num(max_abs)
            entry["max_rel_error"] = _num(max_rel)
            if not correct:
                atol, rtol = R._tolerances_for_dtype(
                    input_dtype or torch.float32)
                entry["error_log"] = (
                    "output_mismatch: max_error_ratio="
                    f"{max_error_ratio:.6g} > 1.0 (tolerance = atol {atol} + rtol "
                    f"{rtol} * |baseline|); mean_abs_diff={mean_diff:.6g}, "
                    f"max_abs_error={max_abs:.6g}, max_rel_error={max_rel:.6g}"
                )

            del baseline_mod, candidate_mod, baseline_out, candidate_out
            del baseline_check_inputs, candidate_check_inputs, inputs

        except Exception as exc:
            entry["status"] = STATUS_RUNTIME_ERROR
            entry["error_log"] = (
                f"scenario raised {type(exc).__name__}: {exc}\n"
                + traceback.format_exc()
            )
        results[scenario.name] = entry

    tally: dict[str, int] = {}
    for e in results.values():
        tally[e["status"]] = tally.get(e["status"], 0) + 1
    _log(f"status tally: {json.dumps(tally, sort_keys=True)}")
    _audit_loaded_trees()
    return {definition: results}


def _audit_loaded_trees() -> None:
    """Prove which on-disk tree the fastkernels modules actually came from.

    A second checkout can be installed (editable) in the same interpreter under a
    different distribution name, so 'the import worked' is not evidence that the
    intended tree was used. Count every loaded module by tree root instead.
    """
    root = os.path.abspath(str(sys.modules["fastkernels"].KB_ROOT))
    from_tree = [n for n, m in list(sys.modules.items())
                 if getattr(m, "__file__", None)
                 and os.path.abspath(m.__file__).startswith(root + os.sep)]
    _log(f"modules loaded from {root}: {len(from_tree)}")
    foreign = sorted(
        f"{n} <- {m.__file__}" for n, m in list(sys.modules.items())
        if (n == "fastkernels" or n.startswith("fastkernels."))
        and getattr(m, "__file__", None)
        and not os.path.abspath(m.__file__).startswith(root)
    )
    if foreign:
        _log(f"WARNING: fastkernels.* modules resolved outside {root}: {foreign}")
    else:
        _log(f"all fastkernels.* modules resolve under {root}: OK")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agent_entrypoint.py",
        description="Run one fastkernels operator's scenarios and emit the "
                    "AKO4X normalized result dict on stdout.",
    )
    parser.add_argument("--op", required=True, help="operator name, e.g. rms_norm")
    parser.add_argument("--candidate", help="path to the candidate kernel .py")
    parser.add_argument("--scenarios", help="comma-separated scenario names/substrings")
    parser.add_argument("--baseline-identity", action="store_true",
                        help="self-test: candidate := a second baseline instance")
    args = parser.parse_args(argv)

    try:
        if args.baseline_identity and args.candidate:
            raise InfraError("--baseline-identity and --candidate are mutually "
                             "exclusive; pass exactly one")
        if not args.baseline_identity and not args.candidate:
            raise InfraError("pass --candidate <path> or --baseline-identity")

        _ensure_ninja_on_path()
        root = _bootstrap_fastkernels()
        _log(f"fastkernels tree: {root}")
        _log(f"fastkernels.__file__ = {sys.modules['fastkernels'].__file__}")

        filters = [s for s in (args.scenarios or "").split(",") if s.strip()] or None
        payload = run(args.op, args.candidate, filters, args.baseline_identity)
    except InfraError as exc:
        _log(f"INFRASTRUCTURE ERROR: {exc}")
        return 2
    except Exception as exc:  # unexpected -> still an infrastructure error
        _log(f"INFRASTRUCTURE ERROR: {type(exc).__name__}: {exc}")
        traceback.print_exc(file=sys.stderr)
        return 2

    _emit_json(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
