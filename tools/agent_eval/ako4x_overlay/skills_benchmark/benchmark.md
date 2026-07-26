# Benchmark reference

The active benchmark is **kb-nano / fastkernels**. AKO4X reaches it through
`scripts/benchmark_adapter.py`, which shells out to `agent_entrypoint.py` under
kb's own interpreter (kb needs a different venv: torch + vLLM + JIT-built CUDA
extensions). One subprocess per bench run; it prints one JSON result object.

## Task model

A kb task is **one `nn.Module` replacing one kb baseline module**. There is no
`run(...)` function and no kernel-signature JSON: the harness instantiates the kb
baseline and your class side by side with the same `init_args`, copies the
baseline's `state_dict()` into yours, and calls both `forward()`s with identical
inputs.

A workload is one *scenario* — a (init_args, input-shape, dtype) triple from kb's
shape registry. Its uuid IS the kb scenario name (e.g. `tokens-1/8f255483`), listed
in `docs/workloads.jsonl` with its axes. Per workload the harness:

1. Builds fresh random inputs at the scenario's declared shapes/dtypes.
2. Instantiates baseline + candidate, transfers weights, runs each `forward()`
   once on cloned inputs, compares (see Correctness).
3. Times each: 10 warmup calls, then the **median of 100** timed calls
   (`bench/kernels/runner.py::_time_forward`).

**Inputs are the same tensor objects across all 110 calls of a workload** and are
mutated in place by whatever your kernel does to them. Fresh tensors are created
per workload, not per call.

## Interface contract (FROZEN — read before writing code)

Read the baseline source named in `docs/definition.json`'s description before your
first edit. Your class must match it on all four points:

- **Class name** — identical to the baseline's class (`RMSNorm` for `kb_rms_norm`).
  The loader takes the class of that name; only if absent does it fall back to the
  first `nn.Module` subclass in the file.
- **`__init__` semantics** — same accepted keyword names. Accept `**kwargs` and
  ignore what you don't use: scenarios carry extra init args (e.g. `training`).
- **`forward` signature** — same parameter names in the same order. The harness
  calls **by keyword** (`module(x=..., residual=...)`), so a renamed parameter is a
  `TypeError`, not a silent positional match.
- **Parameter / buffer names** — the baseline's `state_dict()` is loaded into your
  module with `strict=False`, and **any missing or unexpected key fails the
  scenario** with `RUNTIME_ERROR: weight_transfer_incomplete`. Do not rename, add,
  split, or fuse parameters (no packed `qkv` if the baseline has three, no
  `weight_t`). Cache derived layouts in a non-persistent buffer or a plain
  attribute instead — non-persistent buffers stay out of `state_dict()`.

This gate is deliberately stricter than kb's own bench CLI, which wraps
`load_state_dict` in a bare `try/except pass` (`runner.py:496-500`) and would let a
mismatched candidate run on its own random initialisation.

**In-place mutation is part of the contract.** If the baseline mutates its inputs
(kb's fused add+norm does: `residual := x + residual`, `x := norm(residual)`), your
kernel must mutate the same tensors the same way — the harness compares the input
tensors after the call as well as the returned outputs. A functionally correct but
non-mutating implementation is reported `INCORRECT_NUMERICAL`. (The candidate
example in kb's own `docs/user_guide.md` has this bug; don't copy it.)

## Correctness

kb's Tier-1 comparison, imported unmodified from `bench/kernels/runner.py`:

- Compared: **the returned outputs AND the input tensors after the call** (merged —
  both must pass). Tuples/lists/dicts are walked element-wise; shape mismatch or
  any non-finite value fails immediately.
- Per element: `|baseline - candidate| <= atol + rtol * |baseline|`, expressed as
  `max_error_ratio = max(diff / tolerance)`; **PASSED iff `max_error_ratio <= 1.0`**
  (i.e. every element within tolerance — there is no partial-match ratio).
- Tolerances are **per input dtype**, from kb's module constants:
  fp32 `atol=1e-5, rtol=1e-3`; **fp16/bf16 `atol=1e-2, rtol=1e-2`**;
  fp8 `atol=rtol=0.125` (fp8 pairs compared after dequantization).
- `max_abs_error` / `max_rel_error` in the results are **reported only** — the
  pass/fail decision is the ratio above.

**Tolerances are not settable from `config.toml`.** `[benchmark].atol` / `rtol` /
`required_matched_ratio` exist in the schema (bench_utils requires the keys) but the
adapter drops them before invoking kb, and the entrypoint exposes no tolerance flag.
Editing them changes nothing. This is intentional: the correctness gate must not be
reachable from a file the agent under evaluation can edit.

## Status enum

- **`PASSED`** — within tolerance; latency recorded.
- **`INCORRECT_NUMERICAL`** — `max_error_ratio > 1.0`, or a shape/structure
  mismatch, or non-finite output. `error_log` carries the ratio, mean abs diff, and
  the tolerance that was applied.
- **`RUNTIME_ERROR`** — the scenario raised: `forward` threw, CUDA error, OOM, or
  the weight transfer was refused (`weight_transfer_failed` /
  `weight_transfer_incomplete` — see the contract above).
- **`COMPILE_ERROR`** — `solution/kernel.py` did not import (syntax error, bad
  import, module-level raise). Fanned out to every requested workload, since the
  failure happens once before any scenario runs. Triton/CUDA JIT failures that
  happen at *call* time surface as `RUNTIME_ERROR` instead.
- **`TIMEOUT`** — the run exceeded `timeout_seconds x <number of workloads>`
  (the subprocess covers all requested workloads, so the budget is scaled).

## Reference, expert baseline, and scoring

Two different "baselines" — keep them straight:

1. **The correctness oracle** is always the **kb baseline module** (e.g.
   `tasks/baseline/L1/rms_norm.py`). It is re-instantiated and re-timed inside every
   run; you are never compared against anything else.
2. **The score denominator** is whatever produced `baseline.json`. If
   `expert_baseline.json` exists at the child root, bench_utils profiles **that**
   (for kb tasks it is a thin import of the kb production baseline) and caches its
   latency. Otherwise it profiles `docs/definition.json`'s `reference` — the naive
   pure-PyTorch seed you started from.

`baseline.json` is cached per environment; re-profile with
`bash scripts/bench.sh --force-baseline`. It is invalidated when the workload-uuid
set changes, the source flips reference<->expert, or gpu/backend changes
(`scripts/bench_utils.py::load_baseline`).

**Score = arithmetic mean of `speedup_factor` over workloads**, where
`speedup_w = baseline_latency_w / your_latency_w`
(`scripts/bench_utils.py::compute_score`). If ANY workload is not PASSED the run
has no valid score. Against the expert baseline a score near 1.0 means "matched the
kb production kernel"; the seed starts well below 1.0 and that is the intended
starting point, not a bug.

Note the per-result `reference_latency_ms` from kb is the kb baseline measured in
the same process; bench_utils overwrites it with the cached baseline latency before
scoring, so both numbers are honest but only the cached one feeds the score.

## `config.toml` schema

```toml
[solution]
name = "<operator>-solution"
definition = "kb_<op>"          # kb_rms_norm -> the entrypoint's --op rms_norm
author = "user"

[build]
gpu = "<gpu-name>"
dataset_path = "/path/to/kb-trace"   # local backend only
language = "python"             # advisory metadata for kb (see below)
entry_point = "kernel.py::run"  # only the FILE part is used
destination_passing_style = false    # no kb analogue; ignored

[benchmark]
baseline_iterations = 5
solution_iterations = 100
num_trials = 3
warmup_runs = 3
timeout_seconds = 240           # PER WORKLOAD; subprocess budget = this x n_workloads
use_isolated_runner = false     # FIB-internal; our isolation is the subprocess
atol = 0.01                     # IGNORED (kb owns tolerances)
rtol = 0.01                     # IGNORED
required_matched_ratio = 1.0    # IGNORED
```

- **`[build].language`** is metadata only. kb always imports a Python module; a
  Triton `@triton.jit` kernel or a `load_inline` CUDA kernel JITs from inside it.
  There is no compile step the harness performs on your behalf, and therefore no
  language-specific build failure — an unbuildable kernel fails at import
  (`COMPILE_ERROR`) or at call time (`RUNTIME_ERROR`).
- **`[build].entry_point`**: the file part (`kernel.py`) names the module the
  harness imports from `solution/`; the `::run` suffix is vestigial (kb dispatches
  on the class, not a function). Helper modules are allowed — put them in
  `solution/`, they are staged next to `kernel.py` with that directory on
  `PYTHONPATH`, so `import my_helper` works.
- **Honored `[benchmark]` keys: `timeout_seconds` only.** `warmup_runs` /
  `iterations` / `num_trials` are ignored — kb pins warmup=10, median-of-100
  in-process. One consequence: the expert baseline and your solution are measured
  with the *same* protocol, so their latencies are directly comparable.
  `use_isolated_runner` is ignored (each run is already a fresh process).

## Valid solution: write your own kernel (no delegation)

The benchmark scores *your* kernel. A solution must **implement the operator
itself**. kb's own guidance is "Avoid importing `vllm` or `sgl_kernel` in your
replacement -- the point is to provide an alternative implementation"
(`docs/user_guide.md`); under this harness that is tightened to:

- **Banned**
  - Importing the thing you are replacing: `tasks.baseline.*`, `tasks.reference.*`,
    or `fastkernels.tasks.*` (in any spelling — this is the incumbent kernel and,
    for the expert baseline, literally the score denominator).
  - Vendor fused-op wrappers: `flash_attn`, `flashinfer`, `fla`, `vllm`
    (incl. `torch.ops._C.*`), `sgl_kernel`, `xformers`, `deepgemm`, `apex`, cuBLAS /
    cuDNN reached by any route.
  - A `torch` / `torch.nn.functional` call standing in **as** the target operator —
    e.g. `F.rms_norm` / `F.layer_norm` for a norm task,
    `F.scaled_dot_product_attention` for an attention task, `torch.matmul` as a
    whole GEMM solution.
- **Allowed**
  - Hand-written **Triton** (`@triton.jit`) and **CUDA** (`load_inline` /
    `cpp_extension`), CUTLASS/CuTe you instantiate yourself.
  - Plain `torch` ops as **glue** around a kernel you wrote — reshape, view, cast,
    output allocation, small elementwise fixups. The line is core-compute vs glue:
    torch orchestrating your kernel is fine; a torch op *being* the operator is not.
  - `torch.compile` on your own implementation.

`expert_baseline.json` at the child root is **infra, exempt from this rule**: it
imports the kb production baseline on purpose, because measuring the incumbent IS
its job. Never copy that import into `solution/kernel.py`.

## kb-specific failure modes

- **`weight_transfer_incomplete`** — your parameter/buffer names diverge from the
  baseline's. Fix the names; do not "work around" by re-initialising weights, which
  would make every correctness number meaningless.
- **Silent staleness** — because inputs are the same tensors for all 110 timed calls
  of a workload, a kernel that caches an output buffer keyed on `data_ptr()`, or
  captures a CUDA graph and replays it, can return a stale-but-matching result. The
  correctness pass happens on a *separate* cloned input set, so it does not protect
  you here. If a speedup looks implausible, prove the kernel still runs (perturb the
  input, expect a changed output) before believing it.
- **`INCORRECT_NUMERICAL` only on the residual/in-place workloads** — you
  implemented the fused path functionally instead of in place. See the contract.
- **Anything printed to stdout by your kernel is discarded**; the entrypoint
  redirects fd 1 to stderr so only its JSON reaches the harness. Use stderr, and
  `bash scripts/bench.sh --capture-logs` to see it for PASSED workloads.

## Frozen for bench comparability

Within a campaign these anchor cross-run comparability and must not be edited:

- **Scoring formula** — `bench_utils.py::compute_score` (mean of speedups).
- **Baseline freshness rule and write path** — `bench_utils.py::load_baseline` /
  `save_baseline`.
- **`[benchmark]` scoring config** — `timeout_seconds` may be raised if runs are
  genuinely being cut off; the tolerance keys are inert (above) and editing them is
  a no-op that only misleads a reader.
- **The baseline blobs** — `expert_baseline.json` at the child root and
  `docs/definition.json`'s `reference`. Changing either silently redefines the
  score denominator.
- **The entrypoint's tolerances and timing protocol** — kb's runner constants
  (`_FP32_ATOL` / `_LOW_PRECISION_ATOL` / ...) and its warmup/median-of-N. They live
  outside the child env; treat them as read-only.

To change any of these, start fresh with re-measured baselines.

## COUPLED references

- `scripts/benchmark_adapter.py` — the swap seam; owns the subprocess invocation,
  the params it honors/ignores, and the status mapping. Its docstring is the
  canonical spec.
- `agent_entrypoint.py` (outside the child, under kb's interpreter) — runs the
  scenarios, owns tolerances and timing.
- `scripts/bench_utils.py` — scoring / baseline caching (frozen segments above).
- `docs/definition.json` — the operator's interface description + the seed
  `reference`; `docs/workloads.jsonl` — uuid + axes listing.
- `expert_baseline.json` (child root) — the expert blob, when present.
