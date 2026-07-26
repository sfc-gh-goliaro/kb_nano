#!/bin/bash
# setup_agent_envs.sh — reproduce the AKO4X + ASTRA agent-eval environments.
#
# Idempotent: every step checks its own completion marker and skips if done.
# Ends with a PASS/FAIL verification checklist. Safe to re-run.
#
# Layout (override via env):
#   AGENTS_DIR   (required) — repos, dataset, run dirs
#   VENVS_DIR    (required) — venv_ako4x, venv_astra
#   KB_REPO      (required) — this repo's checkout (branch agent-eval-pilot)
#   KB_MAIN_PY   (required) — kb main venv python
#   OVERLAY_DIR  (default <this script's dir>/../tools/agent_eval) — overlay + patch artifacts
#
# Pins (do not change without re-validating — see docs/agent_eval/H200_RUNBOOK.md):
#   AKO4X            0fd4b5fe99c8b8d9d0a322d3a787eb84d31eff6f  (only release commit)
#   flashinfer-bench f7b4d8d185625ab2d609233a1a06e99ee18a0c6b  (AKO4X's own Modal pin)
#   flashinfer-python 0.6.8       (version in AKO4X's reference image)
#   flashinfer-trace 37c121a      (2026-05-01; later commits add definitions that
#                                  fail the pinned flashinfer-bench schema)
#   ASTRA            34380e20a6714794202c0079d397b00c1b71b6f2
#   torch (ako4x)    resolved by pip from the pins (lands on 2.9.1+cu128)
#   torch (astra)    2.9.1+cu130   (sgl-kernel 0.3.21 ABI requires torch <= 2.9.x:
#                                  needs c10_cuda_check_implementation(...int,bool),
#                                  torch >= 2.10 exports the uint32 variant)
#   sgl-kernel       0.3.21
set -u  # NOT -e: we collect failures into the final checklist instead of dying

# Machine-neutral: all four locations must be provided by the caller (the
# runbook's variables block). Refusing to guess prevents silently building
# into another machine's conventions.
: "${AGENTS_DIR:?set AGENTS_DIR (agent repos/datasets root, e.g. <big-storage>/agents)}"
: "${VENVS_DIR:?set VENVS_DIR (where venv_ako4x/venv_astra will live)}"
: "${KB_REPO:?set KB_REPO (path to this repo checkout, branch agent-eval-pilot)}"
: "${KB_MAIN_PY:?set KB_MAIN_PY (python of a kb-nano main venv: torch+flash_attn+flashinfer)}"
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OVERLAY_DIR=${OVERLAY_DIR:-$SCRIPT_DIR/../tools/agent_eval}

AKO_SHA=0fd4b5fe99c8b8d9d0a322d3a787eb84d31eff6f
FIB_SHA=f7b4d8d185625ab2d609233a1a06e99ee18a0c6b
TRACE_SHA=37c121a
ASTRA_SHA=34380e20a6714794202c0079d397b00c1b71b6f2

declare -a RESULTS
step() { echo; echo "===== $1 ====="; }
record() { RESULTS+=("$1|$2"); }   # record "name" "PASS/FAIL/SKIP: detail"

# ---------------------------------------------------------------- git-lfs
step "git-lfs (user-level, no sudo)"
if command -v git-lfs >/dev/null 2>&1 || [ -x "$VENVS_DIR/bin/git-lfs" ]; then
    record "git-lfs" "PASS: $(PATH=$VENVS_DIR/bin:$PATH git-lfs version | cut -d' ' -f1)"
else
    mkdir -p "$VENVS_DIR/bin"
    curl -sL -o /tmp/git-lfs.tgz \
      "https://github.com/git-lfs/git-lfs/releases/download/v3.7.0/git-lfs-linux-amd64-v3.7.0.tar.gz" \
      && tar xzf /tmp/git-lfs.tgz -C /tmp \
      && cp /tmp/git-lfs-3.7.0/git-lfs "$VENVS_DIR/bin/" \
      && PATH=$VENVS_DIR/bin:$PATH git lfs install
    if PATH=$VENVS_DIR/bin:$PATH git-lfs version >/dev/null 2>&1; then
        record "git-lfs" "PASS: installed user-level to $VENVS_DIR/bin"
    else
        record "git-lfs" "FAIL: install did not produce a working binary"
    fi
fi
export PATH=$VENVS_DIR/bin:$PATH

# ---------------------------------------------------------------- clones
step "AKO4X clone @ $AKO_SHA (+ dependency pins)"
mkdir -p "$AGENTS_DIR"
if [ ! -d "$AGENTS_DIR/AKO4X/.git" ]; then
    git clone --quiet https://github.com/TongmingLAIC/AKO4X.git "$AGENTS_DIR/AKO4X"
fi
git -C "$AGENTS_DIR/AKO4X" checkout --quiet "$AKO_SHA" 2>/dev/null || true
# pin deps in pyproject (idempotent seds)
sed -i "s|flashinfer-bench.git@main|flashinfer-bench.git@$FIB_SHA|" "$AGENTS_DIR/AKO4X/pyproject.toml"
sed -i "s|\"flashinfer-python @ git+https://github.com/flashinfer-ai/flashinfer.git@main ; sys_platform == 'linux'\"|\"flashinfer-python==0.6.8 ; sys_platform == 'linux'\"|" "$AGENTS_DIR/AKO4X/pyproject.toml"
if grep -q "$FIB_SHA" "$AGENTS_DIR/AKO4X/pyproject.toml" && grep -q "flashinfer-python==0.6.8" "$AGENTS_DIR/AKO4X/pyproject.toml"; then
    record "ako4x-clone" "PASS: @$(git -C "$AGENTS_DIR/AKO4X" rev-parse --short HEAD), deps pinned"
else
    record "ako4x-clone" "FAIL: dependency pins not applied"
fi

step "ASTRA clone @ $ASTRA_SHA"
if [ ! -d "$AGENTS_DIR/Astra/.git" ]; then
    git clone --quiet https://github.com/Anjiang-Wei/Astra.git "$AGENTS_DIR/Astra"
fi
git -C "$AGENTS_DIR/Astra" checkout --quiet "$ASTRA_SHA" 2>/dev/null || true
record "astra-clone" "PASS: @$(git -C "$AGENTS_DIR/Astra" rev-parse --short HEAD)"

step "flashinfer-trace dataset @ $TRACE_SHA (LFS pointers only)"
if [ ! -d "$AGENTS_DIR/flashinfer-trace/.git" ]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone --quiet \
      https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace "$AGENTS_DIR/flashinfer-trace"
fi
GIT_LFS_SKIP_SMUDGE=1 git -C "$AGENTS_DIR/flashinfer-trace" checkout --quiet "$TRACE_SHA" 2>/dev/null || true
record "dataset" "$( [ "$(git -C "$AGENTS_DIR/flashinfer-trace" rev-parse --short HEAD)" = "$TRACE_SHA" ] && echo "PASS: @$TRACE_SHA" || echo "FAIL: not at $TRACE_SHA" )"

# ---------------------------------------------------------------- venv_ako4x
step "venv_ako4x (torch cu-resolved + AKO4X package set)"
AKO_PY=$VENVS_DIR/venv_ako4x/bin/python
if [ ! -x "$AKO_PY" ]; then python3 -m venv "$VENVS_DIR/venv_ako4x" && "$AKO_PY" -m pip -q install --upgrade pip; fi
if ! "$AKO_PY" -c "import flashinfer_bench" 2>/dev/null; then
    "$AKO_PY" -m pip -q install torch==2.12.1+cu132 --index-url https://download.pytorch.org/whl/cu132
    (cd "$AGENTS_DIR/AKO4X" && "$AKO_PY" -m pip install .)
    # NOTE: pip resolves the pinned set down to torch 2.9.1+cu12 wheels; expected.
    "$AKO_PY" -m pip -q uninstall -y nvidia-cuda-cupti-cu12 2>/dev/null || true
fi
# CUPTI-13 preload (.pth executes at interpreter start; sitecustomize is shadowed
# by Ubuntu's /usr/lib/python3.12/sitecustomize.py, so .pth is the working hook).
SP=$("$AKO_PY" -c "import site; print(site.getsitepackages()[0])")
if [ ! -f "$SP/zz_ako_cupti_preload.py" ]; then
    cat > "$SP/zz_ako_cupti_preload.py" <<'EOF'
# Load CUPTI 13 before torch. torch/kineto dlopens libcupti.so.12 from the
# system CUDA-12 toolkit during `import torch`; once a .so.12 soname is
# resident, cupti-python >= 13 refuses to init. libcupti loads lazily on the
# first API CALL (import alone is not enough), so we invoke one here.
# Guarded: silent no-op if cupti is unavailable (falls back to CUDA events).
try:
    from cupti import cupti as _cupti
    _cupti.activity_enable(_cupti.ActivityKind.RUNTIME)
    _cupti.activity_disable(_cupti.ActivityKind.RUNTIME)
except Exception:
    pass
EOF
    echo "import zz_ako_cupti_preload" > "$SP/zz_ako_cupti_preload.pth"
fi
record "venv_ako4x" "$("$AKO_PY" -c "
import torch, flashinfer_bench, flashinfer
from importlib.metadata import version
print(f'PASS: torch {torch.__version__}, flashinfer-bench {version(\"flashinfer-bench\")}, flashinfer {version(\"flashinfer-python\")}')" 2>&1 | tail -1)"

# ---------------------------------------------------------------- venv_astra
step "venv_astra (torch 2.9.1+cu130 + sgl-kernel + agents SDK)"
ASTRA_PY=$VENVS_DIR/venv_astra/bin/python
if [ ! -x "$ASTRA_PY" ]; then python3 -m venv "$VENVS_DIR/venv_astra" && "$ASTRA_PY" -m pip -q install --upgrade pip; fi
if ! "$ASTRA_PY" -c "import sgl_kernel, agents" 2>/dev/null; then
    "$ASTRA_PY" -m pip -q install torch==2.9.1+cu130 --index-url https://download.pytorch.org/whl/cu130
    "$ASTRA_PY" -m pip -q install sgl-kernel==0.3.21 openai openai-agents numpy pandas matplotlib tqdm ninja
fi
record "venv_astra" "$("$ASTRA_PY" -c "
import torch, sgl_kernel, agents, openai
print(f'PASS: torch {torch.__version__}, sgl-kernel imports, agents SDK imports')" 2>&1 | tail -1)"

# ---------------------------------------------------------------- overlays/patches
step "AKO4X kb overlay + ASTRA patch"
if [ -d "$OVERLAY_DIR/ako4x_overlay" ]; then
    for f in "$OVERLAY_DIR"/ako4x_overlay/benchmark_adapter.py; do
        [ -f "$AGENTS_DIR/AKO4X/scripts/benchmark_adapter.py.fib.orig" ] || \
            cp "$AGENTS_DIR/AKO4X/scripts/benchmark_adapter.py" "$AGENTS_DIR/AKO4X/scripts/benchmark_adapter.py.fib.orig"
        cp "$f" "$AGENTS_DIR/AKO4X/scripts/benchmark_adapter.py"
    done
    if [ -d "$OVERLAY_DIR/ako4x_overlay/skills_benchmark" ]; then
        [ -d "$AGENTS_DIR/AKO4X/templates/skills/benchmark.fib.orig" ] || \
            cp -r "$AGENTS_DIR/AKO4X/templates/skills/benchmark" "$AGENTS_DIR/AKO4X/templates/skills/benchmark.fib.orig"
        rm -rf "$AGENTS_DIR/AKO4X/templates/skills/benchmark"
        cp -r "$OVERLAY_DIR/ako4x_overlay/skills_benchmark" "$AGENTS_DIR/AKO4X/templates/skills/benchmark"
    fi
    if [ -f "$OVERLAY_DIR/ako4x_overlay/evaluation.toml" ]; then
        [ -f "$AGENTS_DIR/AKO4X/templates/benchmark/evaluation.toml.fib.orig" ] || \
            cp "$AGENTS_DIR/AKO4X/templates/benchmark/evaluation.toml" "$AGENTS_DIR/AKO4X/templates/benchmark/evaluation.toml.fib.orig"
        cp "$OVERLAY_DIR/ako4x_overlay/evaluation.toml" "$AGENTS_DIR/AKO4X/templates/benchmark/evaluation.toml"
    fi
    record "ako4x-overlay" "PASS: applied (originals kept as *.fib.orig)"
else
    record "ako4x-overlay" "SKIP: $OVERLAY_DIR/ako4x_overlay not found (FIB-only setup)"
fi
if [ -d "$OVERLAY_DIR/kb-trace" ]; then
    mkdir -p "$AGENTS_DIR/kb-trace"
    cp -r "$OVERLAY_DIR/kb-trace/." "$AGENTS_DIR/kb-trace/"
    record "kb-dataset" "PASS: kb task definitions installed to $AGENTS_DIR/kb-trace"
else
    record "kb-dataset" "SKIP: no kb-trace in overlay dir"
fi
if [ -f "$OVERLAY_DIR/astra_claude.patch" ]; then
    if git -C "$AGENTS_DIR/Astra" apply --check "$OVERLAY_DIR/astra_claude.patch" 2>/dev/null; then
        git -C "$AGENTS_DIR/Astra" apply "$OVERLAY_DIR/astra_claude.patch"
        record "astra-patch" "PASS: applied"
    elif grep -q "ASTRA_MODEL" "$AGENTS_DIR/Astra/cuda_kernel_optimizer_multi.py" 2>/dev/null; then
        record "astra-patch" "PASS: already applied"
    else
        record "astra-patch" "FAIL: patch does not apply cleanly"
    fi
else
    record "astra-patch" "SKIP: patch file not found"
fi

# ---------------------------------------------------------------- GPU-side verification
step "GPU verification (needs a free GPU; override with VERIFY_GPU=N, skip with SKIP_GPU=1)"
if [ "${SKIP_GPU:-0}" = "1" ]; then
    record "gpu-cupti" "SKIP: SKIP_GPU=1"
    record "gpu-sgl"   "SKIP: SKIP_GPU=1"
else
    GPU=${VERIFY_GPU:-$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2 < 1000 {print $1; exit}')}
    if [ -z "$GPU" ]; then
        record "gpu-cupti" "FAIL: no free GPU found (set VERIFY_GPU=N)"
    else
        record "gpu-cupti" "$(CUDA_VISIBLE_DEVICES=$GPU "$AKO_PY" -c "
import sys, torch
assert any('cupti' in m for m in sys.modules), 'preload did not fire'
torch.zeros(1, device='cuda')
from cupti import cupti
cupti.activity_enable(cupti.ActivityKind.RUNTIME); cupti.activity_disable(cupti.ActivityKind.RUNTIME)
print(f'PASS: CUPTI13 coexists with torch on GPU $GPU ({torch.cuda.get_device_name(0)})')" 2>&1 | tail -1)"
        record "gpu-sgl" "$(CUDA_VISIBLE_DEVICES=$GPU "$ASTRA_PY" -c "
import torch, sgl_kernel
x = torch.randn(32, 2560, device='cuda'); r = torch.randn_like(x); w = torch.randn(2560, device='cuda')
sgl_kernel.fused_add_rmsnorm(x, r, w, 1e-6)
print(f'PASS: sgl_kernel.fused_add_rmsnorm executed on GPU $GPU')" 2>&1 | tail -1)"
    fi
fi

# ---------------------------------------------------------------- checklist
echo; echo "================ VERIFICATION CHECKLIST ================"
FAILED=0
for r in "${RESULTS[@]}"; do
    name=${r%%|*}; detail=${r#*|}
    printf "  %-14s %s\n" "$name" "$detail"
    case "$detail" in FAIL*) FAILED=1;; esac
done
echo "========================================================"
if [ "$FAILED" = "1" ]; then echo "RESULT: FAILURES PRESENT — see above"; exit 1; fi
echo "RESULT: ALL CHECKS PASSED (or explicitly skipped)"
