#!/usr/bin/env python3
"""Multi-rank identity/grading harness for the ``allreduce`` op.

Why this exists
---------------
``agent_entrypoint.py`` is a single-process grader: it has no torch.distributed
process group, so every ``allreduce`` scenario dies with RUNTIME_ERROR before a
verdict (15/15 in the census). The paper's own allreduce number came from a
separate 4-rank harness whose source is lost. This file is the rebuild: a
standalone launcher + N-rank worker that grades the op with a real NCCL process
group, emitting the same JSON contract as the main grader so census tooling can
consume it unchanged.

Usage (wrapper mode -- the only mode callers need)
--------------------------------------------------
    allreduce_runner.py --baseline-identity
    allreduce_runner.py --candidate /path/to/kernel.py
    ... [--scenarios pat,pat] [--nranks 4] [--gpus 2,3,6,7] [--output out.json]
        [--no-custom-ar] [--log-dir DIR]

The wrapper picks ``--nranks`` idle GPUs via nvidia-smi (waiting and retrying if
the box is busy -- never colliding with other users' jobs), binds a free
rendezvous port (env ``KB_NANO_NCCL_PORT`` overrides), spawns one worker
subprocess per rank (this same file; worker mode is detected via the ``RANK``
env var, so ``torchrun --nproc_per_node=N`` also works), and prints exactly one
JSON object to stdout:

    {"kb_allreduce": {scenario_name: {"status", "solution", "axes",
        "latency_ms", "reference_latency_ms", "speedup_factor",
        "max_abs_error", "max_rel_error", ["error_log"], ...}}}

Additive keys (``world_size``, ``custom_ar_used``, ``*_per_rank``, component
error ratios) extend the contract; the core keys match agent_entrypoint.py.
Exit codes mirror the main grader: 0 when the benchmark ran (scenario failures
are data), 2 on infrastructure errors.

Correctness: analytic ground truth on an exact-summation grid
-------------------------------------------------------------
Unlike the main grader (candidate vs baseline on one process), a collective has
an ANALYTIC ground truth: with known per-rank inputs the expected output is
their elementwise sum. Each rank's input is drawn from a seed derived from
(op, scenario, rank) via the same sha256 construction as the grader's
``_stable_seed`` (agent_entrypoint.py:500-503), so every rank can regenerate
ALL ranks' inputs locally and compute the expected sum without communication.

Values are drawn on an exact-summation grid: k/16 with integer
k ~ randint[-64, 64]. Every partial sum satisfies |sum(k)| <= 4*64 = 256 = 2^8,
and any integer multiple of 2^-4 with <= 8 significant bits is exactly
representable in bfloat16 (8-bit significand), so EVERY summation order --
NCCL ring, the custom IPC kernel's two-shot, a candidate's tree -- produces the
bit-identical result. A correct implementation therefore shows
max_abs_error == 0.0, while wrong reductions (AVG: error 0.75*|sum|; identity:
error |sum - x_rank|) breach the tolerance gate by orders of magnitude. randn
inputs were rejected: vs an fp32-exact reference, bf16 accumulation-order
rounding on cancelling elements can exceed atol=1e-2 at the 67M-element
scenarios (half-ulp at partial magnitude 8 is 0.031/add), i.e. rare flaky
failures with zero added discrimination -- allreduce is value-independent.

The pass/fail rule is still the runner's own tolerance-normalized comparison:
``R._compare_outputs`` and ``R._tolerances_for_dtype`` are imported from
``fastkernels.bench.kernels.runner`` and applied with the analytic sum in the
reference slot, exactly as the main grader applies them with the baseline's
output in that slot. Tolerances are NOT settable from the CLI (same rationale
as agent_entrypoint.py:73-75).

What is graded per scenario
---------------------------
Baseline module A and module B (a second baseline instance under
``--baseline-identity``, the candidate class under ``--candidate``) run on
every rank; on every rank all three comparisons must pass:
    A_out vs analytic sum, B_out vs analytic sum, A_out vs B_out.
Status is the rank-worst (any-rank RUNTIME_ERROR > INCORRECT_NUMERICAL >
PASSED). Both modules are timed with the runner's ``_time_forward``
(median of NUM_RUNS after NUM_WARMUP, agent_entrypoint.py:123-124 constants);
the reported latency is the max over ranks' medians -- a collective is not
finished until its slowest rank is.

Deliberate deviations from the main grader, with reasons
--------------------------------------------------------
1. **Input mutation is NOT graded** (the grader merges an inputs comparison,
   agent_entrypoint.py:1773-1776). The baseline itself is path-dependent on
   this: the NCCL fallback reduces in place (tasks/baseline/L1/allreduce.py:47,
   ``dist.all_reduce(tensor)``) while the custom-IPC path writes a fresh
   ``torch.empty_like`` output and leaves the input untouched
   (tasks/baseline/L1/allreduce.py:177-190). A contract the baseline's own two
   paths disagree on cannot be graded; consumers use the return value.
2. **Custom IPC fast path is enabled for the BASELINE by default** (engine
   parity: infra/engine.py:344-350 builds ``CustomAllreduce`` on a gloo group
   with max_size=8MB and installs it via ``set_custom_ar``). The baseline
   timing denominator must be the production path. Candidates bring their own
   fast path or run whatever their file implements against the default NCCL
   group. ``--no-custom-ar`` or ``FASTKERNELS_DISABLE_CUSTOM_AR=1`` (the
   engine's own flag) disables it; if the JIT build fails on any rank, ALL
   ranks drop it (consensus) and fall back to NCCL-only, logged per entry via
   ``custom_ar_used``. With the 8MB cap (inclusive: 1024x4096 bf16 is exactly
   8MB and qualifies), 12 of the 15 registry scenarios take the IPC path and
   3 (>8MB) take NCCL -- measured 2026-07-26 -- so both baseline paths are
   exercised.
3. **Desync containment.** A rank that fails BEFORE its collective would hang
   the peers inside theirs, so every scenario runs a gloo-group consensus
   (the NCCL communicator is never touched unless all ranks are ready, and
   collective-call counts are cross-checked after each phase). On a detected
   mismatch the NCCL communicator is considered poisoned: the scenario and all
   remaining scenarios report RUNTIME_ERROR, no further NCCL calls are made,
   and results still flow back over gloo. Timeout ladder (all bounded):
   ``TORCH_NCCL_ASYNC_ERROR_HANDLING=2`` + the pg timeout (--nccl-timeout-s)
   for NCCL-detected failures; the per-scenario consensus gathers run on a
   SHORT-timeout gloo group (max(120, 2 x nccl timeout)) -- measured in the
   2026-07-26 fault injection, a rank wedged inside torch.cuda.synchronize()
   on an unmatched collective is NOT released by the watchdog, so the healthy
   ranks' consensus timeout is the effective containment clock; a separate
   LONG-timeout gloo group (--gloo-timeout-s) covers only CustomAllreduce
   setup (first-run JIT build); and the wrapper's job timeout (SIGTERM then
   SIGKILL on each worker's session) is the final backstop against zombies.

Weight transfer: ``AllReduce`` is stateless, but the grader's strict
state_dict check (agent_entrypoint.py:30-34) is kept: a candidate that
declares parameters/buffers the baseline does not have fails the scenario
rather than silently running its own state.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import inspect
import json
import math
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any

OP = "allreduce"
DEFINITION = f"kb_{OP}"

# Mirrors agent_entrypoint.py:117-124 (AKO4X status constants + timing budget).
STATUS_PASSED = "PASSED"
STATUS_INCORRECT_NUMERICAL = "INCORRECT_NUMERICAL"
STATUS_RUNTIME_ERROR = "RUNTIME_ERROR"
NUM_WARMUP = 10
NUM_RUNS = 100

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def _log(msg: str) -> None:
    rank = os.environ.get("RANK")
    tag = f"[allreduce_runner r{rank}]" if rank is not None else "[allreduce_runner]"
    print(f"{tag} {msg}", file=sys.stderr, flush=True)


class InfraError(RuntimeError):
    """Whole-run failure: nothing meaningful to report, exit nonzero."""


# --- deterministic seeds (copy of agent_entrypoint.py:500-503) ---------------

def _stable_seed(*parts: str) -> int:
    """Process-independent seed from a name (``hash()`` is salted per process)."""
    digest = hashlib.sha256("/".join(parts).encode()).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF


# --- fastkernels tree bootstrap (copy of agent_entrypoint.py:129-184) --------
# Copied, not imported: importing agent_entrypoint executes its module-level
# stdout quarantine (os.dup2(2, 1)), which this file must not inherit.

def _tree_candidates() -> list[str]:
    roots: list[str] = []
    env_tree = os.environ.get("FASTKERNELS_TREE")
    if env_tree:
        roots.append(env_tree)
    roots.extend(p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p)
    roots.append(os.getcwd())
    roots.append(_REPO_ROOT)  # this file lives at <root>/tools/agent_eval/
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
    """The custom-IPC baseline JIT-builds a CUDA extension; torch needs ninja."""
    if shutil.which("ninja") is None:
        bindir = os.path.dirname(os.path.abspath(sys.executable))
        os.environ["PATH"] = bindir + os.pathsep + os.environ.get("PATH", "")
        _log(f"ninja not on PATH; prepended {bindir}")


# --- CLI ---------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Multi-rank grader for the allreduce op (same CLI shape as "
                    "agent_entrypoint.py; wrapper mode unless RANK is set).")
    p.add_argument("--op", default=OP, help="must be 'allreduce' (CLI-shape parity)")
    p.add_argument("--candidate", help="path to the candidate kernel .py")
    p.add_argument("--baseline-identity", action="store_true",
                   help="grade the baseline against itself (and the analytic sum)")
    p.add_argument("--scenarios", help="comma-separated scenario names/substrings")
    p.add_argument("--nranks", type=int, default=4,
                   help="world size (default 4; custom AR supports 2/4/6/8)")
    p.add_argument("--gpus", help="comma-separated GPU indices; default: pick idle "
                                  "GPUs via nvidia-smi, waiting until enough are free")
    p.add_argument("--output", help="also write the result JSON here")
    p.add_argument("--log-dir", help="per-rank worker logs (default: a fresh tempdir)")
    p.add_argument("--no-custom-ar", action="store_true",
                   help="disable the baseline's custom IPC fast path (NCCL only)")
    p.add_argument("--wait-timeout-s", type=int, default=3600,
                   help="max seconds to wait for idle GPUs (default 3600)")
    p.add_argument("--poll-s", type=int, default=60,
                   help="idle-GPU poll interval (default 60)")
    p.add_argument("--job-timeout-s", type=int, default=7200,
                   help="hard wall clock for the whole distributed job")
    p.add_argument("--nccl-timeout-s", type=int, default=300,
                   help="NCCL process-group timeout (watchdog backstop)")
    p.add_argument("--gloo-timeout-s", type=int, default=900,
                   help="gloo consensus-group timeout (covers first JIT build)")
    return p


def _validate_args(args: argparse.Namespace) -> None:
    if args.op != OP:
        raise InfraError(f"this runner only grades {OP!r}, got --op {args.op!r}")
    if bool(args.candidate) == bool(args.baseline_identity):
        raise InfraError("exactly one of --candidate / --baseline-identity is required")
    if args.candidate and not os.path.isfile(args.candidate):
        raise InfraError(f"candidate file not found: {args.candidate}")
    if args.nranks < 2:
        raise InfraError("--nranks must be >= 2 (a 1-rank allreduce grades nothing)")


# =============================================================================
# Wrapper mode (no RANK in env): pick GPUs, spawn workers, aggregate.
# =============================================================================

def _idle_gpus() -> list[int]:
    """GPU indices that are safe to take: ~no memory held and ~no utilization."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=60)
    if out.returncode != 0:
        raise InfraError(f"nvidia-smi failed: {out.stderr.strip()}")
    idle = []
    for line in out.stdout.strip().splitlines():
        idx, mem, util = [f.strip() for f in line.split(",")]
        if int(mem) <= 2000 and int(util) <= 5:
            idle.append(int(idx))
    return idle


def _pick_gpus(n: int, wait_timeout_s: int, poll_s: int) -> list[int]:
    deadline = time.monotonic() + wait_timeout_s
    while True:
        idle = _idle_gpus()
        if len(idle) >= n:
            return idle[:n]
        if time.monotonic() >= deadline:
            raise InfraError(
                f"only {len(idle)} idle GPU(s) ({idle}) after waiting "
                f"{wait_timeout_s}s; need {n}. Not colliding with other jobs.")
        _log(f"{len(idle)} idle GPU(s) {idle}; need {n}. Retrying in {poll_s}s "
             f"({int(deadline - time.monotonic())}s left)")
        time.sleep(poll_s)


def _free_port() -> int:
    env_port = os.environ.get("KB_NANO_NCCL_PORT")
    if env_port:
        return int(env_port)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wrapper_main(args: argparse.Namespace, argv: list[str]) -> int:
    gpus = [int(g) for g in args.gpus.split(",")] if args.gpus else \
        _pick_gpus(args.nranks, args.wait_timeout_s, args.poll_s)
    if len(gpus) < args.nranks:
        raise InfraError(f"--gpus gave {len(gpus)} GPU(s); --nranks is {args.nranks}")
    gpus = gpus[:args.nranks]
    port = _free_port()
    log_dir = args.log_dir or tempfile.mkdtemp(prefix="allreduce_runner_")
    os.makedirs(log_dir, exist_ok=True)
    result_path = os.path.join(log_dir, "result.json")
    _log(f"world_size={args.nranks} gpus={gpus} port={port} logs={log_dir}")

    procs: list[subprocess.Popen] = []
    log_files = []
    try:
        for rank in range(args.nranks):
            env = dict(os.environ)
            env.update({
                "CUDA_VISIBLE_DEVICES": ",".join(str(g) for g in gpus),
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": str(port),
                "RANK": str(rank),
                "WORLD_SIZE": str(args.nranks),
                "LOCAL_RANK": str(rank),
                # 2 = CleanUpOnly: on watchdog timeout, abort the NCCL comm and
                # raise a catchable error in-process (the poison path handles
                # it) instead of 1 = TearDown, which SIGABRTs the whole rank
                # and only leaves the wrapper's reap-and-report backstop.
                "TORCH_NCCL_ASYNC_ERROR_HANDLING": "2",
                "KB_ALLREDUCE_RESULT": result_path,
                "FASTKERNELS_TREE": _REPO_ROOT,
                "PYTHONUNBUFFERED": "1",
            })
            lf = open(os.path.join(log_dir, f"rank{rank}.log"), "w")
            log_files.append(lf)
            procs.append(subprocess.Popen(
                [sys.executable, os.path.abspath(__file__)] + argv,
                env=env, stdout=lf, stderr=subprocess.STDOUT,
                start_new_session=True))

        deadline = time.monotonic() + args.job_timeout_s
        pending = set(range(args.nranks))
        rcodes: dict[int, int] = {}
        while pending:
            if time.monotonic() >= deadline:
                _log(f"job timeout ({args.job_timeout_s}s); killing workers")
                _kill_all(procs)
                raise InfraError(
                    f"distributed job exceeded --job-timeout-s={args.job_timeout_s}; "
                    f"workers killed. Logs: {log_dir}")
            for r in sorted(pending):
                rc = procs[r].poll()
                if rc is not None:
                    rcodes[r] = rc
                    pending.discard(r)
                    _log(f"rank {r} exited rc={rc}")
                    if rc != 0:
                        # One dead rank means the job cannot complete; reap the rest.
                        _log("nonzero worker exit; terminating remaining ranks")
                        _kill_all(procs)
                        for r2 in list(pending):
                            rcodes[r2] = procs[r2].wait()
                            pending.discard(r2)
            time.sleep(0.5)
    finally:
        _kill_all(procs)  # no-op for already-exited workers
        for lf in log_files:
            lf.close()

    bad = {r: rc for r, rc in rcodes.items() if rc != 0}
    if bad:
        for r in sorted(bad):
            _log(f"--- tail of rank {r} log ---")
            _tail_to_stderr(os.path.join(log_dir, f"rank{r}.log"))
        raise InfraError(f"worker exit codes {rcodes}; logs in {log_dir}")
    if not os.path.isfile(result_path):
        raise InfraError(f"all workers exited 0 but no result at {result_path}")

    with open(result_path) as f:
        result = json.load(f)
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        _log(f"result copied to {args.output}")
    tally: dict[str, int] = {}
    for e in result.get(DEFINITION, {}).values():
        tally[e["status"]] = tally.get(e["status"], 0) + 1
    _log(f"status tally: {json.dumps(tally, sort_keys=True)}")
    print(json.dumps(result))
    return 0


def _kill_all(procs: list[subprocess.Popen]) -> None:
    """SIGTERM the whole session of each live worker, then SIGKILL stragglers."""
    live = [p for p in procs if p.poll() is None]
    for p in live:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    t0 = time.monotonic()
    while any(p.poll() is None for p in live) and time.monotonic() - t0 < 10:
        time.sleep(0.2)
    for p in live:
        if p.poll() is None:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
    for p in live:
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass


def _tail_to_stderr(path: str, n: int = 30) -> None:
    try:
        with open(path) as f:
            for line in f.readlines()[-n:]:
                sys.stderr.write(line)
    except OSError as exc:
        _log(f"could not read {path}: {exc}")


# =============================================================================
# Worker mode (RANK in env): one rank of the distributed grading job.
# =============================================================================

_DTYPE_NAMES = {"float32", "float16", "bfloat16"}


def _scenario_axes(scenario) -> dict[str, Any]:
    """Copy of agent_entrypoint.py:1462-1478 (flat JSON-safe axes)."""
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
    """Copy of agent_entrypoint.py:1480-1486 (JSON-safe float)."""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return "NaN"
    return f if math.isfinite(f) else "NaN"


def _short_exc(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _load_candidate_class(path: str, baseline_cls: type):
    """Selection logic of agent_entrypoint.py:214-248 (_load_candidate_from_path):
    exec the file, prefer the class named like the baseline, else the first
    nn.Module subclass."""
    import torch.nn as nn

    module_name = "_allreduce_candidate_impl"
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
            f"candidate import failed: {_short_exc(exc)}\n" + traceback.format_exc()
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


def _instantiate(cls, init_args: dict[str, Any], device: str):
    """Signature-filtered construction (the registry's ``training: false`` is a
    trace artifact, not an __init__ kwarg -- same handling as the grader)."""
    kwargs = dict(init_args or {})
    kwargs.pop("training", None)
    try:
        params = inspect.signature(cls.__init__).parameters
        accepts_kwargs = any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
        if not accepts_kwargs:
            kwargs = {k: v for k, v in kwargs.items() if k in params}
    except (TypeError, ValueError):
        kwargs = {}
    mod = cls(**kwargs)
    return mod.to(device).eval()


def _build_fixture(torch, scenario, rank: int, world: int, device: str):
    """Per-rank seeded input + analytic expected sum (see module docstring).

    Returns ({"tensor": input}, expected) with both on ``device`` in the
    scenario dtype. Exact-summation grid: input = k/16, k ~ randint[-64, 64]
    from a CPU generator seeded by _stable_seed(op, scenario, rank).
    """
    spec = scenario.inputs.get("tensor")
    if not (isinstance(spec, dict) and "shape" in spec):
        raise InfraError(f"{scenario.name}: expected a single 'tensor' shape input, "
                         f"got {sorted(scenario.inputs)}")
    dtype_name = spec.get("dtype", "bfloat16")
    if dtype_name not in _DTYPE_NAMES:
        raise InfraError(f"{scenario.name}: unsupported dtype {dtype_name!r} "
                         f"(exactness bound derived for >=8-bit significands)")
    dtype = getattr(torch, dtype_name)
    shape = [int(d) for d in spec["shape"]]

    own = None
    ksum = None
    for r in range(world):
        g = torch.Generator(device="cpu").manual_seed(
            _stable_seed(OP, scenario.name, f"rank{r}"))
        k = torch.randint(-64, 65, shape, generator=g, dtype=torch.int32)
        ksum = k.clone() if ksum is None else ksum.add_(k)
        if r == rank:
            own = k
    inp = own.to(torch.float32).mul_(1.0 / 16.0).to(dtype).to(device)
    expected = ksum.to(torch.float32).mul_(1.0 / 16.0).to(dtype).to(device)
    return {"tensor": inp}, expected


def _gather(dist, gloo, world: int, payload: Any) -> list[Any]:
    """all_gather_object over the gloo side channel (usable when NCCL is not)."""
    out: list[Any] = [None] * world
    dist.all_gather_object(out, payload, group=gloo)
    return out


def _select_scenarios(scenarios, filters_arg: str | None):
    """Filter semantics of agent_entrypoint.py:1626-1646 (exact, then substring)."""
    if not filters_arg:
        return scenarios
    filters = [f for f in filters_arg.split(",") if f]
    selected, unmatched = [], []
    for pat in filters:
        hits = [s for s in scenarios if s.name == pat] or \
               [s for s in scenarios if pat in s.name]
        if not hits:
            unmatched.append(pat)
        selected.extend(hits)
    if unmatched:
        raise InfraError(
            f"--scenarios patterns matched nothing for {OP!r}: {unmatched}. "
            f"{len(scenarios)} scenarios available.")
    seen: set[str] = set()
    deduped = []
    for s in selected:
        if s.name not in seen:
            seen.add(s.name)
            deduped.append(s)
    return deduped


def worker_main(args: argparse.Namespace) -> int:
    from datetime import timedelta

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = f"cuda:{local_rank}"

    root = _bootstrap_fastkernels()
    _ensure_ninja_on_path()
    _log(f"fastkernels tree: {root}")

    import torch
    import torch.distributed as dist

    from fastkernels.bench.kernels import runner as R
    from fastkernels.bench.kernels.scenario_registry import InputRegistry
    from fastkernels.infra.kernel_swapper import _find_module_class, registry_class_pin

    if not torch.cuda.is_available():
        raise InfraError("no CUDA device visible")
    torch.cuda.set_device(local_rank)
    _log(f"device {device}: {torch.cuda.get_device_name(local_rank)}")
    _log(f"tolerances (runner constants): low-precision atol={R._LOW_PRECISION_ATOL} "
         f"rtol={R._LOW_PRECISION_RTOL} fp32 atol={R._FP32_ATOL} rtol={R._FP32_RTOL}")

    # Process groups: NCCL world for the collectives under test, plus TWO gloo
    # side channels (infra/engine.py:335-344 pattern for the first):
    #   * ``gloo``      -- long timeout (--gloo-timeout-s): CustomAllreduce
    #     setup only, because the first-ever run JIT-builds its CUDA extension.
    #   * ``consensus`` -- short timeout: per-scenario consensus/aggregation.
    #     Measured (fault-injection, 2026-07-26): when a candidate crashes on
    #     one rank, the peer wedges inside torch.cuda.synchronize() on its
    #     unmatched collective and the NCCL watchdog does NOT release it, so
    #     the healthy rank's consensus gather is the effective containment
    #     clock -- it must not wait the full JIT-build allowance.
    dist.init_process_group(
        "nccl", init_method="env://", world_size=world, rank=rank,
        device_id=torch.device(device),
        timeout=timedelta(seconds=args.nccl_timeout_s))
    gloo = dist.new_group(backend="gloo",
                          timeout=timedelta(seconds=args.gloo_timeout_s))
    consensus = dist.new_group(
        backend="gloo",
        timeout=timedelta(seconds=max(120, 2 * args.nccl_timeout_s)))

    exit_code = 1
    custom_ar = None
    try:
        # --- baseline class (targeted import, agent_entrypoint.py:189-211) ---
        baseline_mod_py = importlib.import_module(f"fastkernels.tasks.baseline.L1.{OP}")
        baseline_cls = _find_module_class(baseline_mod_py, pin=registry_class_pin(OP))
        if baseline_cls is None:
            raise InfraError(f"no nn.Module class found in {baseline_mod_py.__file__}")
        _log(f"baseline class: {baseline_cls.__name__} from {baseline_mod_py.__file__}")

        if args.baseline_identity:
            user_cls = baseline_cls
            solution = "baseline_identity"
        else:
            user_cls = _load_candidate_class(args.candidate, baseline_cls)
            solution = os.path.abspath(args.candidate)
            _log(f"candidate class: {user_cls.__name__} from {args.candidate}")

        # --- custom IPC fast path for the baseline (engine parity; docstring
        # deviation 2). All-ranks-or-none via gloo consensus.
        want_ar = (not args.no_custom_ar
                   and os.environ.get("FASTKERNELS_DISABLE_CUSTOM_AR", "0") != "1")
        if want_ar:
            ar_err = None
            try:
                custom_ar = baseline_mod_py.CustomAllreduce(
                    gloo, local_rank, max_size=8 * 1024 * 1024)
                if custom_ar.disabled:
                    ar_err = "CustomAllreduce reported disabled"
            except Exception as exc:
                ar_err = _short_exc(exc)
            peer_errs = [e for e in _gather(dist, gloo, world, ar_err) if e]
            if peer_errs:
                _log(f"custom AR unavailable, falling back to NCCL-only: {peer_errs}")
                if custom_ar is not None:
                    try:
                        custom_ar.close()
                    except Exception:
                        pass
                    custom_ar = None
            else:
                baseline_mod_py.set_custom_ar(custom_ar)
                _log("custom IPC allreduce enabled for the baseline")

        registry = InputRegistry()
        scenarios = _select_scenarios(registry.scenarios(OP), args.scenarios)
        if not scenarios:
            raise InfraError(f"no scenarios registered for operator {OP!r}")
        _log(f"{len(scenarios)} scenario(s) selected")

        results: dict[str, Any] = {}
        poisoned: str | None = None  # scenario name that broke the NCCL comm

        for scenario in scenarios:
            rec = _run_scenario(
                args, torch, dist, R, consensus, world, rank, device,
                scenario, baseline_cls, user_cls, custom_ar, poisoned)
            if rec.pop("_poison", False) and poisoned is None:
                poisoned = scenario.name
                _log(f"NCCL communicator poisoned at {scenario.name}; "
                     "remaining scenarios will be skipped")
            # Rank-0 merges the per-rank records into one contract entry.
            all_recs = _gather(dist, consensus, world, rec)
            if rank == 0:
                results[scenario.name] = _merge_records(
                    scenario, solution, world, all_recs)
                _log(f"{scenario.name}: {results[scenario.name]['status']}")

        if rank == 0:
            tally: dict[str, int] = {}
            for e in results.values():
                tally[e["status"]] = tally.get(e["status"], 0) + 1
            _log(f"status tally: {json.dumps(tally, sort_keys=True)}")
            payload = {DEFINITION: results}
            out_path = os.environ.get("KB_ALLREDUCE_RESULT")
            if out_path:
                tmp = out_path + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(payload, f, indent=2)
                os.replace(tmp, out_path)
                _log(f"result written to {out_path}")
            else:  # torchrun-direct invocation: rank 0 prints the JSON
                print(json.dumps(payload))
        exit_code = 0
    finally:
        if custom_ar is not None:
            try:
                custom_ar.close()
            except Exception:
                pass
        try:
            dist.destroy_process_group()
        except Exception as exc:
            _log(f"destroy_process_group: {_short_exc(exc)}")
    return exit_code


def _run_scenario(args, torch, dist, R, consensus, world, rank, device,
                  scenario, baseline_cls, user_cls, custom_ar, poisoned):
    """One scenario on this rank. Returns the per-rank record; ``_poison`` is
    set when this rank believes the NCCL communicator is no longer usable."""
    rec: dict[str, Any] = {
        "runtime_error": None,
        "correct": False,
        "ratios": None,          # (A vs analytic, B vs analytic, B vs A)
        "abs_rel": None,         # (max_abs, max_rel) worst of B-vs-analytic, B-vs-A
        "a_ms": None,
        "b_ms": None,
        "custom_ar_used": False,
        "_poison": False,
    }

    if poisoned is not None:
        rec["runtime_error"] = (
            f"skipped: NCCL communicator poisoned by desync at {poisoned!r}")
        return rec

    # --- phase 1: everything that needs NO collective ------------------------
    prep_err = None
    inputs = expected = mod_a = mod_b = None
    try:
        inputs, expected = _build_fixture(torch, scenario, rank, world, device)
        mod_a = _instantiate(baseline_cls, scenario.init_args, device)
        mod_b = _instantiate(user_cls, scenario.init_args, device)
        # Strict state transfer (grader parity; AllReduce is stateless so this
        # only fires for candidates that declare state the baseline lacks).
        sd = mod_a.state_dict()
        incompatible = mod_b.load_state_dict(sd, strict=False)
        missing = list(getattr(incompatible, "missing_keys", []))
        unexpected = list(getattr(incompatible, "unexpected_keys", []))
        if missing or unexpected:
            prep_err = (f"weight_transfer_incomplete: missing_keys={missing} "
                        f"unexpected_keys={unexpected}")
        if custom_ar is not None:
            rec["custom_ar_used"] = bool(
                custom_ar.should_custom_ar(inputs["tensor"]))
    except Exception as exc:
        prep_err = f"fixture/instantiation failed: {_short_exc(exc)}"

    peer_errs = [e for e in _gather(dist, consensus, world, prep_err) if e]
    if peer_errs:
        rec["runtime_error"] = "; ".join(sorted(set(peer_errs)))
        return rec  # no rank touched NCCL for this scenario: comm still clean

    # --- phase 2: correctness collectives (1 per module) ---------------------
    n_coll = 0
    coll_err = None
    a_out = b_out = None
    try:
        a_out = R._run_forward_once(mod_a, R._clone_inputs(inputs))
        n_coll += 1
        b_out = R._run_forward_once(mod_b, R._clone_inputs(inputs))
        n_coll += 1
    except Exception as exc:
        coll_err = f"forward failed: {_short_exc(exc)}"

    states = _gather(dist, consensus, world, (n_coll, coll_err))
    if _check_desync(rec, states):
        return rec

    # --- comparisons (local, no collectives) ---------------------------------
    # Runner's own tolerance rule with the analytic sum in the reference slot.
    ok_a, ratio_a, _ = R._compare_outputs(expected, a_out)
    ok_b, ratio_b, _ = R._compare_outputs(expected, b_out)
    ok_ab, ratio_ab, _ = R._compare_outputs(a_out, b_out)
    rec["correct"] = bool(ok_a and ok_b and ok_ab)
    rec["ratios"] = (_num(ratio_a), _num(ratio_b), _num(ratio_ab))
    abs_b, rel_b = _abs_rel(torch, expected, b_out)
    abs_ab, rel_ab = _abs_rel(torch, a_out, b_out)
    rec["abs_rel"] = (_num(max(abs_b, abs_ab)), _num(max(rel_b, rel_ab)))

    # --- phase 3: timing (runner's median-of-N; barrier-aligned) -------------
    n_coll = 0
    time_err = None
    try:
        dist.barrier()
        _, a_ms = R._time_forward(mod_a, R._clone_inputs(inputs), NUM_WARMUP, NUM_RUNS)
        n_coll += NUM_WARMUP + NUM_RUNS
        dist.barrier()
        _, b_ms = R._time_forward(mod_b, R._clone_inputs(inputs), NUM_WARMUP, NUM_RUNS)
        n_coll += NUM_WARMUP + NUM_RUNS
        rec["a_ms"], rec["b_ms"] = a_ms, b_ms
    except Exception as exc:
        time_err = f"timing failed: {_short_exc(exc)}"

    states = _gather(dist, consensus, world, (n_coll, time_err))
    if _check_desync(rec, states):
        return rec

    del mod_a, mod_b, a_out, b_out, inputs, expected
    return rec


def _check_desync(rec: dict[str, Any], states: list[tuple[int, str | None]]) -> bool:
    """Fold a gathered (collective_count, error) consensus into ``rec``.

    Returns True when the scenario must stop here (any rank errored or counts
    diverged). Poisons the communicator on either (a) diverging collective
    counts -- ranks left NCCL with mismatched op sequences -- or (b) an error
    that names NCCL/timeout/abort: the watchdog tears the communicator down
    even when counts still match, and without the flag every remaining
    scenario would burn a full watchdog timeout discovering the same corpse.
    """
    errs = sorted(set(e for _, e in states if e))
    counts = [n for n, _ in states]
    if not errs and len(set(counts)) == 1:
        return False
    rec["runtime_error"] = "; ".join(errs) or "collective-count mismatch"
    desynced = len(set(counts)) > 1
    aborted = any(("NCCL" in e or "nccl" in e or "imeout" in e or "abort" in e)
                  for e in errs)
    if desynced:
        rec["runtime_error"] += (
            f" (per-rank collective counts {counts}: NCCL comm is desynced)")
    if desynced or aborted:
        rec["_poison"] = True
    return True


def _abs_rel(torch, ref, out) -> tuple[float, float]:
    """max |ref-out| and max |ref-out|/|ref| (agent_entrypoint.py:1553-1573
    semantics, tensor case only -- allreduce returns a single tensor)."""
    if not (isinstance(ref, torch.Tensor) and isinstance(out, torch.Tensor)) \
            or ref.shape != out.shape:
        return float("inf"), float("inf")
    b = ref.float()
    c = out.float()
    diff = (b - c).abs()
    denom = b.abs()
    rel = torch.where(
        denom > 0, diff / denom.clamp_min(torch.finfo(torch.float32).tiny),
        torch.where(diff > 0, torch.full_like(diff, float("inf")),
                    torch.zeros_like(diff)),
    )
    return diff.max().item(), rel.max().item()


def _merge_records(scenario, solution: str, world: int,
                   recs: list[dict[str, Any]]) -> dict[str, Any]:
    """Fold per-rank records into one contract entry (rank-worst status)."""
    entry: dict[str, Any] = {
        "status": STATUS_RUNTIME_ERROR,
        "solution": solution,
        "axes": _scenario_axes(scenario),
        "world_size": world,
        "custom_ar_used": all(r.get("custom_ar_used") for r in recs),
    }
    runtime_errs = [f"rank{i}: {r['runtime_error']}"
                    for i, r in enumerate(recs) if r.get("runtime_error")]
    if runtime_errs:
        entry["error_log"] = "; ".join(runtime_errs)
        return entry

    a_ms = [r["a_ms"] for r in recs]
    b_ms = [r["b_ms"] for r in recs]
    have_timing = all(isinstance(v, (int, float)) for v in a_ms + b_ms)
    if have_timing:
        # A collective is not finished until its slowest rank is.
        ref_ms, cand_ms = max(a_ms), max(b_ms)
        entry["latency_ms"] = _num(cand_ms)
        entry["reference_latency_ms"] = _num(ref_ms)
        entry["speedup_factor"] = _num(
            ref_ms / cand_ms if cand_ms > 0 else float("inf"))
        entry["latency_ms_per_rank"] = [_num(v) for v in b_ms]
        entry["reference_latency_ms_per_rank"] = [_num(v) for v in a_ms]

    max_abs = max(float(r["abs_rel"][0]) if r["abs_rel"][0] != "NaN" else float("inf")
                  for r in recs)
    max_rel = max(float(r["abs_rel"][1]) if r["abs_rel"][1] != "NaN" else float("inf")
                  for r in recs)
    entry["max_abs_error"] = _num(max_abs)
    entry["max_rel_error"] = _num(max_rel)
    entry["error_ratios_per_rank"] = [r["ratios"] for r in recs]

    if all(r["correct"] for r in recs):
        entry["status"] = STATUS_PASSED
    else:
        entry["status"] = STATUS_INCORRECT_NUMERICAL
        bad = [f"rank{i}: ratios(baseline_vs_analytic, candidate_vs_analytic, "
               f"candidate_vs_baseline)={r['ratios']}"
               for i, r in enumerate(recs) if not r["correct"]]
        entry["error_log"] = (
            "output_mismatch vs analytic sum and/or baseline "
            "(tolerance = atol + rtol * |reference|, runner constants); "
            + "; ".join(bad))
    return entry


# =============================================================================

def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = _build_parser().parse_args(argv)
    try:
        _validate_args(args)
        if os.environ.get("RANK") is not None:
            return worker_main(args)
        return wrapper_main(args, argv)
    except InfraError as exc:
        _log(f"INFRA ERROR: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
