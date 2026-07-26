# H200 runbook: AKO4X + ASTRA agent evaluation

Goal: reproduce the validated B200 pilot on the H200 cluster and run real
campaigns. Everything here was verified on B200 (see VALIDATION_REPORT.md);
H200-specific caveats are marked. Branch: `agent-eval-pilot` (from
`origin/release`).

## 0. Preflight (10 minutes, run before anything)

```bash
nvidia-smi                    # driver branch >= 580 (CUDA >= 13.0)?
nvcc --version                # >= 13.x present?
python3 --version             # 3.12 recommended (3.10+ ok)
claude --version && claude -p "Reply OK"   # CLI installed AND authenticated
df -h <big-storage>           # ~25 GB for venvs+repos+dataset
```

- **Driver < 580**: CUPTI-13 timing silently degrades to CUDA events —
  numbers then are NOT methodologically comparable to AKO4X's published
  results or to our B200 pilot. Get the driver upgraded first.
- **claude auth**: campaigns bill someone. For real campaigns use an API key
  (`export ANTHROPIC_API_KEY=...`) — no five-hour subscription windows
  mid-campaign, clean lab attribution — and set `ENABLE_PROMPT_CACHING_1H=1`
  (restores the 1-hour cache TTL that subscription auth gets by default;
  AKO4X respawns child sessions constantly, so cache TTL is a first-order
  cost lever).

## 1. Setup (one command)

```bash
git clone <kb_nano remote> kb_agent_eval && cd kb_agent_eval && git checkout agent-eval-pilot
AGENTS_DIR=<storage>/agents VENVS_DIR=<storage> KB_REPO=$PWD \
  KB_MAIN_PY=<kb-main-venv>/bin/python bash tests/setup_agent_envs.sh
```

Expect the final checklist to be all PASS. The script is idempotent — re-run
it after fixing anything. Pins and the reasons for every environment quirk
are documented inside the script and in VALIDATION_REPORT.md ("Environment
fixes discovered").

Note `KB_MAIN_PY`: the kb entrypoint needs a venv with kb-nano's stack
(torch 2.10+cu128 era, flash_attn, flashinfer 0.6.6). On catalyst-fleet1
that's `/raid/user_data/olu/venv/bin/python`; on the H200 cluster build the
equivalent from the repo's pyproject first if it doesn't exist.

## 2. First live step (needs the API key, costs cents)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
bash tools/agent_eval/astra_live_smoke.sh        # ~$1-3, 3-iteration ASTRA run
```

If it errors on the model ID within seconds: list models (command in the
script header) and re-run with `ASTRA_MODEL=<correct-opus-4.7-id>`. This is
the ONLY unvalidated step from the B200 pilot.

## 3. AKO4X campaign quickstart

### On its own benchmark (FlashInfer-Bench task)

```bash
cd <AGENTS_DIR>/AKO4X
PATH=<VENVS_DIR>/venv_ako4x/bin:$PATH AKO_DATASET_PATH=<AGENTS_DIR>/flashinfer-trace \
CUDA_VISIBLE_DEVICES=<free> python spawn.py --operator rmsnorm_h128 --name h200-r1 --backend local
cd ../ako4x-run-h200-r1
# interactive (first time — also dismisses the trust dialog):
claude    # then: "Read CLAUDE.md and optimize the kernel using Triton."
# headless (after trusting, or always with explicit tools):
claude -p "Read CLAUDE.md and optimize the kernel using Triton. Budget: N labeled bench iterations." \
  --allowedTools "Bash,Read,Edit,Write,Glob,Grep"
```

**H200 steering prompt — paste into the first message and disclose in the
paper** (the shipped skills were tuned on B200):

> Target GPU is H200 (Hopper, sm_90a) — not B200. Skill guidance tagged
> B200/CUDA 13.2 does not transfer: there is no tmem/tcgen05 on this GPU, and
> Blackwell donor kernels in the cute-dsl skill will not compile. Prefer
> sm_90a paths (wgmma/FA3-style pipelines, PDL). The fp8->bf16 pairwise cvt
> instruction IS available on sm_90a.

**Before any campaign on a new family**: verify the FlashInfer expert
baseline actually runs on sm_90a — spawn the child and run
`bash scripts/bench.sh --first 1`; if the expert profile fails, that family
has no denominator on H200 (known: dsa-topk-indexer's expert is unrunnable
even on B200/cu132). Check the family's `baseline.json` provenance before
comparing against archived numbers (some were measured under CUDA 12.8).

**Hardware lock**: AKO4X refuses to reuse a family archive measured on other
hardware. H200 campaigns start fresh families (e.g. suffix `-h200`) — no
B200 archive seeds. This is by design (MASTER.md).

Where results land: `ITERATIONS.md` (one row per labeled bench), `git log`
(`bench(<score>): ...` commits), `trajectory/<timestamp>/` (kernel +
results.json snapshots).

### On kb tasks (the FastKernels port)

The setup script applies the overlay (adapter + benchmark skill +
evaluation.toml) to the AKO4X clone and installs the kb task dataset at
`<AGENTS_DIR>/kb-trace`. Then:

```bash
cd <AGENTS_DIR>/AKO4X
PATH=<VENVS_DIR>/venv_ako4x/bin:$PATH AKO_DATASET_PATH=<AGENTS_DIR>/kb-trace \
KB_EVAL_PYTHON=<kb-main-venv>/bin/python KB_EVAL_REPO=<path-to-this-checkout> \
CUDA_VISIBLE_DEVICES=<free> python spawn.py --operator kb_rms_norm --name kb-h200-r1 --backend local
# copy the expert blob (kb production baseline as the score denominator):
cp <this-repo>/tools/agent_eval/ako4x_overlay/expert_baseline.json ../ako4x-run-kb-h200-r1/
```

Correctness inside these campaigns is computed by
`tools/agent_eval/agent_entrypoint.py` — the kb Tier-1 runner with a
STRICT weight-transfer check, tolerances fixed from the runner's constants
(deliberately not overridable by the agent's config.toml).

## 4. Budget anchors

AKO4X's published campaigns: 4-50 h GPU wall-clock per operator family,
Claude Opus, 1M context, max thinking. Our bounded 1-iteration smoke was
minutes. Plan family count accordingly; per-token costs land on whatever key
is exported.

## 5. Troubleshooting (all hit and fixed during the B200 pilot)

| Symptom | Cause / fix |
|---|---|
| `Incompatible CUPTI Library with soname libcupti.so.12` | See VALIDATION_REPORT "Environment fixes"; the setup script installs a `.pth` preload — if you see this, the venv wasn't built by the script |
| `ValidationError: Definition reference Field required` | Dataset not at pin 37c121a (`git -C flashinfer-trace checkout 37c121a`, with git-lfs on PATH) |
| sgl_kernel `undefined symbol ...c10_cuda_check_implementation...` | torch >= 2.10 in venv_astra; must be 2.9.1 |
| Headless claude does nothing / "not trusted" warnings | Pass `--allowedTools "Bash,Read,Edit,Write,Glob,Grep"` or trust the child dir interactively once |
| ASTRA `sgl_kernel has no attribute 'sgl_fused_add_rmsnorm'` | Upstream default bug; pass `--baseline-func fused_add_rmsnorm --generated-export-func sgl_fused_add_rmsnorm` (the smoke script does) |
| `Ninja is required to load C++ extensions` | ninja is in venv_astra; keep the venv's bin on PATH |
| git clone/checkout fails with `git-lfs: not found` | `export PATH=<VENVS_DIR>/bin:$PATH` (user-level git-lfs installed by the script) |

## 6. Named follow-up tasks (not in the pilot)

1. **Populate the 61 CONFIG-blocked ops** — see `OP_CENSUS.md` for the
   measured per-op survey (33 runnable today / 61 config-blocked / 7 mixed /
   4 special) and the two-lane plan: Lane A (model-scoped ops: static
   config.json values into registry init_args + a one-time generic
   dict-to-object shim in the entrypoint; verify via identity), Lane B
   (dimension-generic ops: team decision required — replicate the
   experiments-codex fabricated dims for comparability with published
   numbers, or reconstruct real call-site dims for production fidelity).
   The experiments-codex runner's if/elif reconstructions
   (`bench/kernels/runner.py` on that branch, ~lines 88-360) remain useful
   as a cross-check and as ready-made code for several classes.
2. **rms_norm fixture sharpening** — seed registry inputs or drop <=4-token
   scenarios from the correctness gate (see VALIDATION_REPORT fixture
   finding) before trusting per-scenario verdicts near tolerance.
3. **L3 seeded-vs-scratch experiment** — two arms via `spawn.py --kernel`
   (seed dir with the agent's accepted L1/L2 kernels vs baseline-only);
   needs (1) first.
4. **Decision: re-run Codex/KernelAgent/Dr. Kernel on the new hardware** —
   any new agent's number is only comparable to Table 1 if the original
   three ran on the same GPU. (Paper's Table 1 is H200; Appendix E's "H100"
   is an erratum to fix in the revision.)
5. **ASTRA**: add ninja to its requirements upstream-style; consider a
   modern-model second row; note o4-mini API retires 2026-10-23 (published
   config unrunnable after that).

---

## FINAL: full-run quickstart (supersedes the per-op commands above)

After setup + preflight, the whole kb evaluation is two commands:

```bash
# 1. (Re)generate the task menu from the current registry — 67 tasks,
#    every seed verified through the real grader before shipping:
PYTHONPATH=<this-repo> <kb-main-venv>/bin/python tools/agent_eval/package_tasks.py

# 2. Run campaigns (resumable; one campaign per GPU, round-robin):
bash tools/agent_eval/run_campaigns.sh --ops all --iters <N> \
  --gpu-list "0,1,2,3" --tag h200-r1
# -> summary CSV per tag; per-op artifacts in each child run dir.
```

Budget note: per-bench cost varies ~1000x by op (gelu seconds,
flashinfer_decode minutes) — budget per-op, not uniformly. 18 ops are not
packaged (PACKAGING_REPORT.md in the dataset dir lists each with its reason;
12 trace to defective tasks/reference implementations — fixing those files
unlocks the tasks).

## ASTRA: final decision + reasoning

Run ASTRA AS PUBLISHED on its own three kernels (astra_live_smoke.sh, needs
an API key); do NOT port it to kb. Reasoning, not preference: (a) reviewer
Q3 explicitly accepts disclosure of practical barriers — running as published
exceeds that bar; (b) the run-on-our-benchmark requirement is satisfied by
AKO4X (67 packaged tasks); (c) ASTRA's candidate format is one compiled CUDA
function per run (verified in source: codegen prompt "Generate ... CUDA
source code (.cu)", extract/is_valid_cuda_code filters, single PyBind
export) — fits single-kernel L1s only; a composite L2/L3 candidate is not
representable without rewriting its core loop; (d) an L1-subset fork is
feasible (~2-4 days: _import_callable fix + per-op testgen scaffolding +
naive .cu seeds) and documented here in case the team wants the extra row —
but the same days buy more as AKO4X campaign coverage.
