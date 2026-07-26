# Agent-eval runbook: AKO4X campaigns on your cluster

Everything here is machine-neutral. Set these once and every command below
works verbatim:

```bash
export KB_REPO=~/kb_agent_eval          # this repo, branch agent-eval-pilot
export AGENTS_DIR=<big-storage>/agents  # AKO4X + ASTRA clones + datasets
export VENVS_DIR=<big-storage>          # venv_ako4x / venv_astra live here
export KB_MAIN_PY=<kb-main-venv>/bin/python   # kb-nano stack: torch 2.10+cu128-era,
                                              # flash_attn, flashinfer (build from
                                              # this repo's pyproject if absent)
export GPUS=0,1,2,3                     # the GPUs campaigns may use
```

Everything was validated end-to-end on an 8xB200 machine (evidence:
VALIDATION_REPORT.md and the pilot scratch — reference material only, not
needed to run). H200-specific caveats are marked H200:.

## 0. Preflight (minutes)

```bash
nvidia-smi          # H200: driver branch >= 580, else CUPTI-13 timing silently
                    # degrades and numbers are not comparable to published AKO4X
nvcc --version      # >= 13.x
python3 --version   # 3.12 recommended (3.10+ ok)
claude --version && claude -p "Reply OK"    # CLI installed AND authenticated
df -h $VENVS_DIR    # ~25 GB needed
```

Billing (decided): campaigns run on an Anthropic API key —
`export ANTHROPIC_API_KEY=...` and `ENABLE_PROMPT_CACHING_1H=1` (AKO4X
respawns child sessions constantly; the 1-hour cache TTL is a first-order
cost lever).

## 1. Setup (one command, idempotent)

```bash
git clone <kb_nano remote> $KB_REPO && cd $KB_REPO && git checkout agent-eval-pilot
AGENTS_DIR=$AGENTS_DIR VENVS_DIR=$VENVS_DIR KB_REPO=$PWD KB_MAIN_PY=$KB_MAIN_PY \
  bash tests/setup_agent_envs.sh
```

Expect the final checklist all-PASS; re-run after fixing anything. Every
pin and environment quirk is documented inside the script.

## 2. The kb campaign run (the main event)

```bash
cd $KB_REPO
# (a) generate the task menu from the registry — every seed is verified
#     through the real grader before shipping; the authoritative task list
#     and per-op skip reasons land in $AGENTS_DIR/kb-trace/PACKAGING_REPORT.md
PYTHONPATH=$PWD $KB_MAIN_PY tools/agent_eval/package_tasks.py --gpus $GPUS

# (b) smoke: one bounded campaign proves the whole stack on your cluster
bash tools/agent_eval/run_campaigns.sh --ops rms_norm --iters 1 \
  --gpu-list "$GPUS" --tag smoke

# (c) the sweep: one AK campaign per packaged op (L1+L2+L3), resumable,
#     round-robin over GPUs; per-op artifacts in each child run dir,
#     summary CSV per tag
bash tools/agent_eval/run_campaigns.sh --ops all --iters <N> \
  --gpu-list "$GPUS" --tag h200-r1
```

Budgeting: per-bench cost varies ~1000x across ops (gelu seconds,
flashinfer_decode minutes) — pick `--iters` per budget, not uniformly.
AKO4X's published campaigns ran 4-50 GPU-hours per op family; bounded
runs are proportionally cheaper.

H200: paste this steering prompt into each campaign's first message and
disclose it (the shipped skills were tuned on B200):

> Target GPU is H200 (Hopper, sm_90a) — not B200. Skill guidance tagged
> B200/CUDA 13.2 does not transfer: no tmem/tcgen05; Blackwell donor
> kernels will not compile. Prefer sm_90a paths (wgmma/FA3-style
> pipelines, PDL). The fp8->bf16 pairwise cvt instruction IS available.

Correctness inside campaigns is `tools/agent_eval/agent_entrypoint.py` —
the kb Tier-1 runner with strict weight transfer, seeded discriminating
fixtures, and tolerances the agent cannot override. Anti-gaming layers:
grader design + master-agent code inspection + the negative-control
battery in `tools/agent_eval/controls/` (each file is a known-wrong
kernel that must FAIL; re-run them after any grader edit).

The `allreduce` op is graded by its own multi-rank harness (the only op
needing >1 GPU): `tools/agent_eval/allreduce_runner.py --baseline-identity`
(same CLI/JSON contract; `--nranks 4` is the paper-parity configuration —
use it on H200; our pilot verified at 2 ranks under cluster contention).

## 3. Optional: end-to-end deployment numbers

Accepted agent kernels dropped into `tasks/candidate/L*/<op>.py` are
picked up automatically by the existing e2e benchmarks (kernel_swapper
discovery) — run the standard `bench/` e2e suites on the traced models
for end-to-end throughput with agent kernels in place.

## 4. AKO4X on its own benchmark (FlashInfer-Bench), if wanted

```bash
cd $AGENTS_DIR/AKO4X
PATH=$VENVS_DIR/venv_ako4x/bin:$PATH AKO_DATASET_PATH=$AGENTS_DIR/flashinfer-trace \
CUDA_VISIBLE_DEVICES=<free> python spawn.py --operator rmsnorm_h128 --name h200-r1 --backend local
cd ../ako4x-run-h200-r1
claude -p "Read CLAUDE.md and optimize the kernel using Triton. Budget: N labeled bench iterations." \
  --allowedTools "Bash,Read,Edit,Write,Glob,Grep"
```

Before a new family: verify the FlashInfer expert baseline runs on sm_90a
(`bash scripts/bench.sh --first 1`); check `baseline.json` provenance
before comparing to archived numbers. AKO4X's hardware lock means H200
campaigns start fresh families (by design).

## 5. Troubleshooting (all hit and fixed during the pilot)

| Symptom | Fix |
|---|---|
| `Incompatible CUPTI Library ... libcupti.so.12` | venv not built by setup script (it installs a `.pth` preload) |
| `ValidationError: Definition reference Field required` | dataset not at pin 37c121a |
| sgl_kernel `undefined symbol ...c10_cuda_check...` | torch >= 2.10 in venv_astra; must be 2.9.1 |
| headless claude does nothing / "not trusted" | pass `--allowedTools "Bash,Read,Edit,Write,Glob,Grep"` |
| ASTRA `no attribute 'sgl_fused_add_rmsnorm'` | pass `--baseline-func fused_add_rmsnorm --generated-export-func sgl_fused_add_rmsnorm` (smoke script does) |
| `Ninja is required` | keep venv_astra bin on PATH |
| matmul_ogs illegal memory access (mxfp4 ops) | known upstream ragged-TMA OOB (SM100); grader auto-sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for those ops; re-check on H200 (M8) |

## 6. What needs your decision (not blocking the run)

See OPEN_DECISIONS.md — a decision RECORD, mostly already decided by the
author; what remains with you: the merge-PR review itself, routing the M8
upstream Triton bug report, and M7's outcome if its investigation
confirms a baseline fix.
