# Agent-eval pilot: validation report (B200, catalyst-fleet1, 2026-07-25)

Evidence, not claims: every item lists the command actually run and its key
output. Raw outputs live in `/raid/user_data/olu/scratch/agent_eval_pilot/`
(entrypoint test JSONs, driver/stub logs) and in the run dirs cited below.
Hardware: NVIDIA B200 (cc 10.0), driver 595.58.03, system nvcc 13.2.51.

## Environment (setup_agent_envs.sh final checklist, verbatim)

```
  git-lfs        PASS: git-lfs/3.7.0
  ako4x-clone    PASS: @0fd4b5f, deps pinned
  astra-clone    PASS: @34380e2
  dataset        PASS: @37c121a
  venv_ako4x     PASS: torch 2.9.1+cu128, flashinfer-bench 0.1.3.dev86, flashinfer 0.6.8
  venv_astra     PASS: torch 2.9.1+cu130, sgl-kernel imports, agents SDK imports
  astra-patch    PASS: applied
  gpu-cupti      PASS: CUPTI13 coexists with torch on GPU 0 (NVIDIA B200)
  gpu-sgl        PASS: sgl_kernel.fused_add_rmsnorm executed on GPU 0
RESULT: ALL CHECKS PASSED (or explicitly skipped)
```

Idempotency verified: second run over existing state skips completed steps and
re-passes all checks.

## AKO4X on its own benchmark (Smoke A) — PASSED

1. **No-LLM pipeline check** (`bash scripts/bench.sh --label
   "reference-baseline-validation"` in child `ako4x-run-smoke1`, GPU 7):
   seeded naive reference kernel scored **0.0989x vs the FlashInfer expert
   baseline, 14/14 workloads PASSED** with CUPTI timing
   (`Baseline: expert (flashinfer_wrapper_57c111)`). Sub-0.1x for a naive
   seed vs expert is the expected shape.
2. **Live agent run** (`claude -p ... --allowedTools "Bash,Read,Edit,Write,Glob,Grep"`,
   subscription auth, 3-iteration budget): agent produced a Triton kernel
   scoring **1.15x vs the expert, 14/14 PASSED**, committed as
   `bench(1.15): iter-1 tiered-rows Triton port of prior anchor`
   (child git log). Regime breakdown (huge-B 1.36-1.65x, mid-B 1.02-1.04x,
   tiny-B 0.93-0.99x) matches the archived campaign's structure; archived
   headline was 1.22-1.23x on Modal B200 / Triton 3.6.0 / CUDA 13.0 vs our
   local Triton 3.5.1 / CUDA 12.8. Agent stopped after 1 of 3 budgeted
   iterations (its choice); the full edit->bench->log->commit loop is proven.

## kb entrypoint (`tools/agent_eval/agent_entrypoint.py`) — PASSED

Runs under the kb main venv against the `agent-eval-pilot` worktree
(provenance printed and checked: `runner.__file__ =
/raid/user_data/olu/kb_agent_eval/bench/kernels/runner.py`; all
`fastkernels.*` modules resolve under the worktree).

| Test | Result |
|---|---|
| Baseline identity, 35/35 rms_norm scenarios | `{"PASSED": 35}`, speedup med ~1.00, max_abs_error 0.0; deterministic across reruns; independently re-run by the coordinator |
| Gross-wrong candidate (sum-vs-mean) | `{"INCORRECT_NUMERICAL": 35}` (max_error_ratio up to 78.3); independently re-run by the coordinator |
| Weight-mismatch candidate (param `gamma` vs `weight`) | `{"RUNTIME_ERROR": 35}` with `weight_transfer_incomplete` — the strict-transfer tightening is load-bearing: the release runner silently "passes" the same file 25/35 with ratio=0 because the baseline initializes weight=ones |
| Subtle-wrong candidate (normalization dropped, ~1% scale error) | 32-34/35 INCORRECT_NUMERICAL, **1-3 borderline PASSes on 1-4-token scenarios, non-reproducible across runs** — see finding below |
| CLI contract | correctness failures exit 0 (data); infra failures exit 2, empty stdout; no tolerance flags exist (tolerances fixed from runner constants) |

**Fixture finding (pre-existing, affects the release harness equally):** the
rms_norm scenario set cannot discriminate a ~1% pure-scale error at 1-4
tokens (`rms(x) = 1 +/- 1/sqrt(2H)` sits inside the bf16 gate
`atol 1e-2 + rtol 1e-2|x|`), and registry inputs are unseeded, so
near-tolerance candidates get flaky verdicts (measured: 60%/12% pass rates at
[1,2560]/[4,2560] over 200 draws, using the runner's own comparator; the
unmodified release runner shows the same behavior). Follow-up: seed registry
inputs or exclude <=4-token scenarios from the correctness gate.

## ASTRA -> Claude port — everything except the live call PASSED

- Patch (`tools/agent_eval/astra_claude.patch`): applies clean to pristine
  @34380e2 (re-verified by coordinator with `git apply --check`); adds
  `ASTRA_MODEL` env (default `claude-opus-4-7`), Anthropic OpenAI-compat
  client (chat-completions mode, tracing disabled), drops unused pycuda.
  openai-agents 0.18.3 / openai 2.48.0 — no SDK API adaptation needed.
- No-LLM driver (GPU 6): ASTRA's own compile/verify/bench on its bundled
  `rms_v1.cu`: compile OK (arch via `TORCH_CUDA_ARCH_LIST=10.0`,
  `-gencode=arch=compute_100,code=sm_100` in build.ninja), correctness 6/6 vs
  real sgl_kernel (max_abs_diff ~1.9e-6, the fp32 reduction floor), benchmark
  6/6 (e.g. 512x4096: 0.0495 ms).
- Negative controls: no-op / dropped-weight / wrong-eps / dropped-residual
  kernels all FAIL 0/2; a baseline-mutation probe confirms the comparison is
  live (not a mutually-no-op pass).
- Stub test (zero credits): localhost chat-completions stub; TEST A (no-tool
  agent) and TEST B (tool-call dispatch into ASTRA's real
  `generate_comprehensive_test_cases`, `optimization_state` mutated) both
  PASS. Re-run independently by the coordinator: exit 0.

## AKO4X kb overlay (the FastKernels port) — PASSED, real end-to-end

Overlay = exactly the three designed port files (`benchmark_adapter.py`,
`templates/skills/benchmark/`, `evaluation.toml`) + the kb task dataset
(`kb-trace/definitions/kb/kb_rms_norm.json`, 35 workloads with uuid == kb
scenario name) + an expert blob delegating to the kb production baseline.

- Adapter tests: 26/26 checks with the real entrypoint (GPU 4), 22/22 with a
  no-GPU mock; unknown-uuid raises; import failure -> COMPILE_ERROR fan-out;
  subprocess expiry -> TIMEOUT fan-out.
- `spawn.py --operator kb_rms_norm` produced a working child; seed extracted
  byte-identical from the definition's `reference`.
- **Full bench through the real AKO4X pipeline: 35/35 PASSED, 12.9 s,
  `FINAL SCORE (mean speedup): 0.264x` vs
  `Baseline: expert (kb_rms_norm-expert-baseline)`** — i.e. the naive seed
  vs the kb production kernel, exactly the intended semantics. baseline.json
  cached with `"source": "expert"`.
- Negative controls through the full pipeline: gross-wrong ->
  INCORRECT_NUMERICAL (ratio 21.3); non-in-place residual variant (the kb
  user_guide example!) -> INCORRECT_NUMERICAL on residual scenarios (ratio
  217); renamed param -> RUNTIME_ERROR.

**Second fixture blind spot, found and FIXED.** kb's rms_norm baseline
initializes `weight = ones` and the entrypoint transferred that state_dict
verbatim, so a candidate that never multiplies by `weight` PASSED all 35
scenarios with max_abs_error 0.0. Fix (in `agent_entrypoint.py`, coordinator,
post-overlay): deterministically perturb all floating-point baseline
parameters (seeded; x U[0.75,1.25] + U[-0.05,0.05]) BEFORE the transfer —
both modules still share identical weights, so correct candidates are
unaffected. Re-run after the fix:

```
identity                : {'PASSED': 35}            (unchanged)
no-weight-multiply      : {'INCORRECT_NUMERICAL': 35}   (was {'PASSED': 35})
renamed-param (gamma)   : {'RUNTIME_ERROR': 35}     (unchanged)
```

The no-weight control is preserved as `tools/agent_eval/controls/no_weight_rms.py`.

## Smoke B (live agent on the kb task) — PASSED (run truncated by billing, not by the stack)

Child `ako4x-run-kbsmoke`, GPU 4, subscription auth, 2-iteration budget. The
agent read the kb benchmark skill, honored the frozen contract (its kernel
header restates class/param/forward contract and implements the IN-PLACE
residual semantics), wrote a Triton kernel, and ran one labeled bench through
the full ported pipeline. From
`trajectory/20260725_164418_iter-1_triton_fused_row-tile_rmsnorm/results.json`:

```
label:  iter-1 triton fused row-tile rmsnorm
passed: 35 / 35   (kb correctness gate, post weight-perturbation fix)
final_score: 0.942x vs kb production baseline (expert blob)
per-tokens speedup: 1->0.67x ... 1600->0.73x | 16384->1.16x | 86960->2.40x | 262144->3.59x | 542000->2.11x
```

The session terminated early with "You're out of usage credits" (the
subscription plan's included usage was exhausted by the day's combined pilot
work) AFTER the bench completed but BEFORE the agent wrote its ITERATIONS.md
row — so the child's ITERATIONS table is empty and the evidence of record is
the trajectory snapshot plus `solution/kernel.py`. Nothing in the ported
stack failed.

Incidental but striking: a one-iteration bounded smoke landed at 0.94x vs the
kb production kernel — winning on large shapes, losing on launch-bound small
shapes — the same "beats eager-adjacent regimes, loses on production hot
paths" pattern as the paper's headline result.

## NOT validated (the complete list)

1. **ASTRA live Claude call** — needs an Anthropic Console API key (none on
   this machine). Run `tools/agent_eval/astra_live_smoke.sh` (~$1-3). Also
   unverified: whether `claude-opus-4-7` is the correct model ID (override
   with `ASTRA_MODEL`).
2. **Anything on H200** — no H200 exists on this machine. See H200_RUNBOOK.md
   preflight.
3. **Full-clean-state setup run** — the setup script was verified idempotent
   over existing state; a from-scratch run on a blank machine is exactly what
   the target H200 setup will be (script is the recipe; ~10 GB of wheel
   downloads were not re-downloaded here to conserve the shared /raid, which
   sits at 100% with ~156 GB free).
4. **ASTRA multi-iteration loop, merge/silu kernels, prompts.py contents** —
   LLM-driven or out of pilot scope.
5. **Coordinator's independent re-run of ASTRA's neg_control.py** hit a
   stale-module-cache issue (`gen_v1` not importable from a fresh session);
   the control's original run (captured in scratch logs) is the evidence of
   record. Re-verify by running driver_no_llm.py then neg_control.py in the
   same session.

## Environment fixes discovered (all encoded in setup_agent_envs.sh)

| Issue | Root cause (verified) | Fix |
|---|---|---|
| sgl-kernel 0.3.21 ImportError | needs `c10_cuda_check_implementation(int,...)` — torch >= 2.10 exports the uint32 variant (`nm -D libc10_cuda.so`) | torch 2.9.1 in venv_astra |
| CUPTI "Incompatible ... libcupti.so.12" | torch/kineto dlopens the system CUDA-12 cupti at import; cupti-python 13 then refuses; libcupti loads on first API CALL, not import | `.pth` preload that calls a CUPTI API before torch; plus uninstall of `nvidia-cuda-cupti-cu12` |
| venv `sitecustomize.py` ignored | shadowed by Ubuntu's `/usr/lib/python3.12/sitecustomize.py` | `.pth` executable-import instead |
| Dataset schema error (`Definition.reference missing`) | flashinfer-trace HEAD (2026-05-16+) adds definitions the pinned flashinfer-bench schema rejects | dataset pinned to 37c121a (2026-05-01) |
| AKO4X pip install downgrades torch to 2.9.1+cu12 wheels | resolver behavior of the pinned flashinfer set | accepted; works on B200 (verified) |
| Headless child claude ignores its settings.local.json | workspace not trusted; `-p` can't show the trust dialog | pass `--allowedTools` explicitly (or pre-trust in ~/.claude.json) |
| ASTRA defaults broken | `--baseline-func` default `sgl_fused_add_rmsnorm` doesn't exist in sgl-kernel 0.3.21 (export is `fused_add_rmsnorm`); ninja missing from requirements | smoke script passes both flags; ninja installed in venv_astra |
