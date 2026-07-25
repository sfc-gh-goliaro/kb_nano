#!/bin/bash
# astra_live_smoke.sh — the ONE validation that needs an Anthropic API key.
#
# Everything else about the ASTRA->Claude port is already validated credit-free
# (see docs/agent_eval/VALIDATION_REPORT.md): GPU machinery via the no-LLM
# driver, and the SDK->chat-completions wiring via a localhost stub. This
# script runs the real 3-iteration ASTRA loop against the real Anthropic
# endpoint. Expected cost: roughly $1-3. Fails loudly within seconds if the
# model ID or the tool-call JSON path is wrong.
#
# Usage:
#   export ANTHROPIC_API_KEY=sk-ant-...        # Console key (platform.claude.com)
#   bash astra_live_smoke.sh [GPU_INDEX]
#
# Optional: export ASTRA_MODEL=<model-id> to override the default
# (claude-opus-4-7). If the default errors with a model-not-found, list
# models first:
#   curl -s https://api.anthropic.com/v1/models \
#     -H "x-api-key: $ANTHROPIC_API_KEY" -H "anthropic-version: 2023-06-01" \
#     | python3 -m json.tool | grep '"id"'
set -euo pipefail

GPU=${1:-$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2 < 1000 {print $1; exit}')}
ASTRA_DIR=${ASTRA_DIR:-/raid/user_data/olu/agents/Astra}
ASTRA_PY=${ASTRA_PY:-/raid/user_data/olu/venv_astra/bin/python}

[ -n "${ANTHROPIC_API_KEY:-}" ] || { echo "ERROR: ANTHROPIC_API_KEY not set"; exit 1; }
grep -q "ASTRA_MODEL" "$ASTRA_DIR/cuda_kernel_optimizer_multi.py" \
  || { echo "ERROR: astra_claude.patch not applied to $ASTRA_DIR (run tests/setup_agent_envs.sh)"; exit 1; }

# Known ASTRA quirks (pre-existing upstream bugs, documented in the runbook):
#  - default --baseline-func is wrong for sgl-kernel 0.3.21: must be
#    fused_add_rmsnorm (the .cu export name sgl_fused_add_rmsnorm is separate)
#  - torch cpp_extension needs ninja on PATH (installed in venv_astra)
cd "$ASTRA_DIR"
PATH=$(dirname "$ASTRA_PY"):$PATH CUDA_VISIBLE_DEVICES=$GPU "$ASTRA_PY" \
  cuda_kernel_optimizer_multi.py \
  --api-key "dummy-not-used-openai-key" \
  --initial-kernel-path test/rms/rms_v1.cu \
  --max-iterations 3 \
  --compare-kind rmsnorm \
  --baseline-func fused_add_rmsnorm \
  --generated-export-func sgl_fused_add_rmsnorm \
  2>&1 | tee /tmp/astra_live_smoke_$(date +%Y%m%d_%H%M%S).log

echo
echo "Success criteria: the run completes >=1 optimization iteration, generated"
echo "kernels compile, and correctness verification runs against sgl_kernel."
echo "Output dir: $ASTRA_DIR/cuda_optimization_runs/ (timestamped)."
