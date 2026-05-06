# Candidate L1 Kernel Implementation Guide

This directory is intentionally empty at the start of a kernel-agent run. Add
new candidate implementations under `tasks/candidate/L1/` and benchmark them
against the production baselines.

## Scope

Implement the 48 L1 operators present in
`bench/kernels/benchmark_scenarios/small/shape_registry.yaml`:

```text
allreduce
batch_norm2d
chunk_gla
chunk_retention
conv2d
conv3d
dense_attention
diffusion_rope
embedding
flash_attn_decode
flash_attn_prefill
flash_attn_varlen
flashinfer_decode
flashinfer_prefill
flux_pos_embed
fp8_linear
fused_recurrent_gla
fused_recurrent_retention
gelu
gla_recurrence
interpolate
l2_norm
layer_norm
linear
log_sigmoid
max_pool2d
moe_align
moe_grouped_gemm
moe_sum
mrope
mrope_input_positions
mxfp4_moe
oasis_rotary
quickgelu
relu
rms_norm
rotary_emb
sigmoid
silu
silu_and_mul
silu_mul_quant_fp8
softmax
store_kvcache
t5_layer_norm
tensor_ops
topk_softmax
vision_rotary_emb
yarn_rotary_emb
```

## Goal

The final goal is to produce correct and fast handwritten L1 candidate kernels.
Correctness is mandatory; speed is the optimization target.

For the complete 48-op L1 run:

- Every implemented scenario must pass alignment/correctness against the production baseline.
- `ERR_RATIO` must be `<= 1.0` for every scenario.
- There should be no exception rows.
- Candidate latency should be as low as possible; optimize for per-scenario and per-op speedup.
- Report all blocked cases explicitly instead of hiding them. `allreduce` is blocked unless the benchmark is launched with a real distributed/NCCL process group.

In this isolated kernel benchmark, "alignment" means the candidate output and
mutated input tensors match the production baseline under the tolerances in
`bench/kernels/runner.py`. End-to-end text/model alignment is tested elsewhere;
this task is kernel-level alignment.

## Reference Material

Use these files to understand the required contract:

- `bench/kernels/benchmark_scenarios/small/shape_registry.yaml`: scenario names, tensor shapes, dtypes, scalar inputs, and current L1 workload coverage.
- `tasks/baseline/L1/<op>.py`: production baseline API. Match the public class name, `__init__` arguments, `forward()` signature, output structure, dtype/device behavior, and in-place side effects.
- `tasks/reference/L1/<op>.py`: semantic PyTorch reference when present. Use it only for understanding correctness, not as candidate code.
- `bench/kernels/scenario_registry.py`: how shape-only inputs are materialized.
- `bench/kernels/runner.py`: how baseline/candidate modules are instantiated, state dicts are copied, outputs are compared, and latency is measured.
- `infra/kernel_swapper.py`: how `tasks/candidate/L1/<op>.py` is discovered and loaded.

For GPT-OSS MXFP4 scenarios, the runner automatically uses real checkpoint
weights when available. It infers the model from the workload shape:

- `32` experts loads `openai/gpt-oss-20b`.
- `128` experts loads `openai/gpt-oss-120b`.

Use env vars only to override the automatic choice:

```bash
export KB_NANO_GPT_OSS_MXFP4_MODEL=openai/gpt-oss-120b
# or point directly at a local snapshot:
export KB_NANO_GPT_OSS_MXFP4_PATH=/path/to/openai/gpt-oss-120b/snapshot
```

The runner builds packed MXFP4 weights and calls each implementation's own
`prepare_weight()` method. Candidate code must not fall back to dense PyTorch
MoE for `mxfp4_moe`.

## Implementation Rules

Each candidate file must define an op as an `nn.Module` subclass:

```python
import torch
import torch.nn as nn


class RmsNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...
```

Hard requirements:

- Put each implementation at `tasks/candidate/L1/<op>.py`.
- Use the same class name as the baseline class when possible. The loader first looks for that exact class name.
- Subclass `torch.nn.Module`; do not expose a bare function as the candidate.
- Match baseline input names, scalar defaults, output type, output shape, dtype, device, and mutation behavior.
- Keep implementation source in `tasks/candidate/L1/`. If shared helpers are needed, put them in a private file such as `tasks/candidate/L1/_triton_utils.py`; that helper is part of the candidate and must contain handwritten code.
- During candidate implementation, modify only `tasks/candidate/` by default. Do not edit benchmark harness, baseline, reference, or infra files as part of the candidate implementation.
- Handwrite kernels in Triton or CUDA. Simple Python wrapper code is fine, but the target op's actual computation must not be delegated to an existing high-performance op.
- `allreduce` is the only communication exception: it may use `torch.distributed` collectives, but must not provide an identity/single-process fallback. True benchmarking requires an initialized process group.

Forbidden in candidate implementations:

- Do not import `tasks.baseline`, `tasks.reference`, or `kb_nano.tasks.baseline/reference`.
- Do not call external kernel libraries such as `flash_attn`, `flashinfer`, `fla`, `vllm`, `triton_kernels`, `xformers`, `bitsandbytes`, or vendor fused-op wrappers.
- Do not use `torch.nn.functional` or `torch.*` as the implementation of the target kernel when that call is the optimized op being replaced, e.g. no `F.layer_norm`, `F.softmax`, `F.conv2d`, `torch.topk`, or `torch.distributed` fallback for non-communication ops.
- Do not modify `tasks/baseline/`, `tasks/reference/`, `bench/kernels/runner.py`, or `infra/kernel_swapper.py` to make a candidate pass unless you have first proven a benchmark harness bug with `baseline_identity`.
- Do not introduce global environment-variable switches, monkey patches, network downloads, logging spam, or hidden cache state in candidate files.
- Do not change scenario shapes to avoid hard cases.

Allowed dependencies:

- Python standard library.
- `torch` for tensor allocation, dtype/device checks, and module plumbing.
- `triton` and `triton.language` for handwritten kernels.
- `torch.utils.cpp_extension` only if the candidate includes local CUDA/C++ source under `tasks/candidate/L1/`.

Do not add shared JIT/compiler helpers outside `tasks/candidate/`. If a CUDA
extension helper is needed, keep the loader/build helper and all CUDA/C++ source
inside `tasks/candidate/L1/`, for example `tasks/candidate/L1/_cuda_utils.py`.
The helper must only compile local candidate sources; it must not hide kernel
implementations in infra or import external kernel libraries.

## Development Flow

Set up the environment:

```bash
source /mnt/weka/home/hao.zhang/conda/miniconda/etc/profile.d/conda.sh
conda activate kb
export PYTHONPATH=/mnt/weka/home/hao.zhang/async_rl_bench/AsyncRL/bench-async/mps-demo:${PYTHONPATH:-}
```

Start from a clean candidate tree:

```bash
rm -rf tasks/candidate/L1
mkdir -p tasks/candidate/L1
```

Before implementing an op, verify the baseline harness:

```bash
python -m kb_nano.bench.kernels \
  --target <op> \
  --validation-mode baseline_identity \
  --num-warmup 1 \
  --num-runs 3
```

If `baseline_identity` fails, fix or document the harness/input issue before
working on the candidate. A candidate result is not meaningful until
baseline-vs-baseline passes.

To run the baseline harness check for all 48 L1 targets, use this script. It
intentionally skips `allreduce` unless you are running with an initialized
distributed process group.

```bash
python - <<'PY'
import yaml
from pathlib import Path
from kb_nano.bench.kernels.runner import run_kernel_benchmark

registry = yaml.safe_load(Path(
    "bench/kernels/benchmark_scenarios/small/shape_registry.yaml"
).read_text())
l1_targets = {
    path.stem
    for path in Path("tasks/baseline/L1").glob("*.py")
    if not path.name.startswith("_")
}
ops = sorted(name for name in registry if name in l1_targets)
failed = []
for op in ops:
    if op == "allreduce":
        print("SKIP allreduce: requires initialized distributed/NCCL process group")
        continue
    result = run_kernel_benchmark(
        op,
        validation_mode="baseline_identity",
        num_warmup=0,
        num_runs=1,
    )
    result.compute_aggregates()
    ok = result.passed == result.total_scenarios and result.failed == 0
    print(f"{op:<28} {'PASS' if ok else 'FAIL'} {result.passed}/{result.total_scenarios}")
    if not ok:
        failed.append(op)
print("failed_ops:", failed)
raise SystemExit(1 if failed else 0)
PY
```

Implement and smoke-test one op:

```bash
python -m kb_nano.bench.kernels \
  --target <op> \
  --validation-mode candidate_smoke \
  --num-warmup 0 \
  --num-runs 1
```

Run the real isolated benchmark for one op:

```bash
python -m kb_nano.bench.kernels \
  --target <op> \
  --num-warmup 10 \
  --num-runs 100 \
  --output-json bench/results/<op>_candidate.json
```

Run all implemented L1 candidates:

```bash
python -m kb_nano.bench.kernels \
  --validation-mode candidate_smoke \
  --num-warmup 0 \
  --num-runs 1 \
  --output-json bench/results/l1_candidate_smoke.json

python -m kb_nano.bench.kernels \
  --num-warmup 10 \
  --num-runs 100 \
  --output-json bench/results/l1_candidate_full.json
```

The no-`--target` command benchmarks all discovered files under
`tasks/candidate/`. Keep this directory limited to L1 when evaluating the 48 L1
targets.

Each `--output-json` file stores per-kernel and per-scenario results, including:

- `correct`: scenario-level alignment/correctness result.
- `max_error_ratio` and `mean_abs_diff`: numerical mismatch details.
- `baseline_ms` and `candidate_ms`: median latency for that scenario.
- `speedup`: `baseline_ms / candidate_ms` for that scenario.
- `classification`: whether the scenario was correct/timed, incorrect, or failed before timing.

Always keep the full JSON from the final full run locally so failures can be
traced to exact scenarios. Do not commit `bench/results/` unless explicitly
requested.

## Result Criteria

For each scenario:

- `CORRECT` must be `PASS`.
- `ERR_RATIO` must be `<= 1.0`.
- There must be no exception rows.
- `speedup = baseline_ms / candidate_ms`; values above `1.0x` are faster than baseline.

For the 48-op L1 task, report:

- Number of implemented candidate files.
- Number of ops and scenarios passing correctness.
- Per-op average speedup and worst failing scenario if any.
- Slowdowns as well as speedups; correctness-only but slower kernels are not the target.
- Ops blocked by harness/environment. Currently `allreduce` requires a real distributed/NCCL process group and should not be made to pass via identity fallback.

Useful summary snippet:

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("bench/results/l1_candidate_full.json")
data = json.loads(path.read_text())
for op in data.get("operators", []):
    total = op.get("total_scenarios", 0)
    passed = op.get("passed", 0)
    speedup = op.get("avg_speedup", 0.0)
    status = "PASS" if passed == total else "FAIL"
    print(f"{op['target']:<28} {status:<4} {passed}/{total:<4} avg_speedup={speedup:.2f}x")
PY
```

## Commit Hygiene

For a candidate-only commit, add only:

```bash
git add -f tasks/candidate/L1
git add tasks/candidate/README.md
```

Do not add `bench/results/` unless a PR explicitly asks for benchmark artifacts.
Do not mix benchmark harness fixes with candidate kernels; keep harness changes in
a separate commit so candidate quality can be reviewed independently.

If an agent believes a non-candidate file must change, stop and prove it with a
failing `baseline_identity` result first. That change is a harness/baseline fix,
not a candidate implementation, and should be reviewed/committed separately.
