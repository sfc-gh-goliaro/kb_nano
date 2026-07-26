#!/usr/bin/env bash
#
# run_campaigns.sh — drive one AKO4X optimization campaign per kb operator.
#
# For every requested op it (1) spawns an isolated child env from the kb-trace
# dataset, (2) drops the op's expert blob at the child root as the score
# denominator, (3) runs a headless Claude Code campaign with a fixed iteration
# budget inside it, and (4) folds the best trajectory result into a summary CSV.
#
# Ops are dealt round-robin across --gpu-list; each GPU runs its share
# sequentially, so at most one campaign per GPU is live at a time. Re-running is
# safe: an op whose child already holds .campaign_done is skipped.
#
#   tools/agent_eval/run_campaigns.sh --ops gelu,rms_norm --gpu-list "1,5" --iters 3
#   tools/agent_eval/run_campaigns.sh --ops ops.txt --dry-run
#
# Env passthrough (defaults shown) — these reach the child's benchmark_adapter,
# which shells out to the kb grader:
#   KB_EVAL_PYTHON=/raid/user_data/olu/venv/bin/python
#   KB_EVAL_REPO=/home/olu/kb_nano
#   KB_EVAL_ENTRYPOINT=$KB_EVAL_REPO/tools/agent_eval/agent_entrypoint.py
#   AKO_HOME=/raid/user_data/olu/agents/AKO4X
#   AKO_DATASET_PATH=<AKO_HOME>/../kb-trace
#   AKO_VENV=/raid/user_data/olu/venv_ako4x
#   CLAUDE_BIN=claude
#
# NOTE on KB_EVAL_ENTRYPOINT: the adapter's compiled-in default points at a
# scratch copy of the grader that has since drifted from the repo. We always
# export the repo copy so campaigns are graded by the checked-in entrypoint.

set -uo pipefail

# --------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------
KB_EVAL_PYTHON="${KB_EVAL_PYTHON:-/raid/user_data/olu/venv/bin/python}"
KB_EVAL_REPO="${KB_EVAL_REPO:-/home/olu/kb_nano}"
KB_EVAL_ENTRYPOINT="${KB_EVAL_ENTRYPOINT:-$KB_EVAL_REPO/tools/agent_eval/agent_entrypoint.py}"
AKO_HOME="${AKO_HOME:-/raid/user_data/olu/agents/AKO4X}"
AGENTS_DIR="$(dirname "$AKO_HOME")"
AKO_DATASET_PATH="${AKO_DATASET_PATH:-$AGENTS_DIR/kb-trace}"
AKO_VENV="${AKO_VENV:-/raid/user_data/olu/venv_ako4x}"
CLAUDE_BIN="${CLAUDE_BIN:-claude}"

OPS_ARG=""
ITERS=3
GPU_LIST="0"
TAG="r1"
SUMMARY=""
LOGDIR=""
DRY_RUN=0
ALLOWED_TOOLS="Bash,Read,Edit,Write,Glob,Grep"

usage() {
    cat >&2 <<'EOF'
usage: run_campaigns.sh --ops <file-or-comma-list> [options]

  --ops <arg>        comma-separated op names, or a file with one op per line
                     (blank lines and #-comments ignored). Names are bare kb
                     op names (gelu), not definition names (kb_gelu).
  --iters N          benchmarked-iteration budget given to each agent (default 3)
  --gpu-list "a,b"   GPUs to deal campaigns across (default "0")
  --tag NAME         campaign tag; child dir = ako4x-run-kb-<op>-<tag> (default r1)
  --summary PATH     summary CSV (default <agents>/campaign_summary_<tag>.csv)
  --logdir PATH      per-campaign logs (default <agents>/campaign_logs_<tag>)
  --allowed-tools S  --allowedTools value for claude (default Bash,Read,Edit,Write,Glob,Grep)
  --dry-run          print every command instead of running it
  -h, --help         this message
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --ops)           OPS_ARG="$2"; shift 2 ;;
        --iters)         ITERS="$2"; shift 2 ;;
        --gpu-list)      GPU_LIST="$2"; shift 2 ;;
        --tag)           TAG="$2"; shift 2 ;;
        --summary)       SUMMARY="$2"; shift 2 ;;
        --logdir)        LOGDIR="$2"; shift 2 ;;
        --allowed-tools) ALLOWED_TOOLS="$2"; shift 2 ;;
        --dry-run)       DRY_RUN=1; shift ;;
        -h|--help)       usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

[ -n "$OPS_ARG" ] || { echo "error: --ops is required" >&2; usage; exit 2; }
SUMMARY="${SUMMARY:-$AGENTS_DIR/campaign_summary_$TAG.csv}"
LOGDIR="${LOGDIR:-$AGENTS_DIR/campaign_logs_$TAG}"

# --------------------------------------------------------------------------
# Resolve the op list
# --------------------------------------------------------------------------
OPS=()
if [ -f "$OPS_ARG" ]; then
    while IFS= read -r line; do
        line="${line%%#*}"
        line="$(echo "$line" | tr -d '[:space:]')"
        [ -n "$line" ] && OPS+=("$line")
    done < "$OPS_ARG"
else
    IFS=',' read -r -a OPS <<< "$OPS_ARG"
fi
[ "${#OPS[@]}" -gt 0 ] || { echo "error: --ops resolved to no operators" >&2; exit 2; }

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
[ "${#GPUS[@]}" -gt 0 ] || { echo "error: --gpu-list resolved to no GPUs" >&2; exit 2; }

# Preflight: every op must be packaged in the dataset, with an expert blob.
MISSING=()
for op in "${OPS[@]}"; do
    [ -f "$AKO_DATASET_PATH/definitions/kb/kb_$op.json" ] || MISSING+=("kb_$op: definition")
    [ -f "$AKO_DATASET_PATH/workloads/kb/kb_$op.jsonl" ]  || MISSING+=("kb_$op: workloads")
    [ -f "$AKO_DATASET_PATH/blobs/kb_$op.json" ]          || MISSING+=("kb_$op: expert blob")
done
if [ "${#MISSING[@]}" -gt 0 ]; then
    echo "error: dataset at $AKO_DATASET_PATH is missing:" >&2
    printf '  %s\n' "${MISSING[@]}" >&2
    echo "run tools/agent_eval/package_tasks.py first" >&2
    exit 2
fi

# --------------------------------------------------------------------------
# Campaign prompt
# --------------------------------------------------------------------------
campaign_prompt() {
    local op="$1"
    cat <<EOF
You are optimizing the kb-nano kernel task kb_$op in this directory. Read
CLAUDE.md first (task contract, bench workflow), then docs/definition.json (the
frozen interface: class name, __init__/forward signatures, state_dict keys) and
ITERATIONS.md.

Budget: $ITERS benchmarked iterations. One iteration = edit solution/kernel.py,
run ./scripts/bench.sh --label iter-<n>_<short-slug>, append the Summary row to
ITERATIONS.md. Use ./scripts/bench.sh --first 1 for quick smoke checks; those do
not count against the budget. Stop after $ITERS labeled benches and write the
end-of-session synthesis in ITERATIONS.md.

Rules that decide whether the run counts:
  * Correctness is a gate, not a tradeoff. Every workload must report PASSED;
    a single non-PASSED workload makes final_score null no matter how fast the
    rest were.
  * solution/kernel.py must NOT import fastkernels, tasks.baseline, or otherwise
    delegate to the kb production kernel. That is the one banned move — the
    incumbent is the thing you are being measured against.
  * The score denominator is expert_baseline.json (the kb production kernel), so
    final_score >= 1.0 means you matched the incumbent on average.

Start from the seed already in solution/kernel.py: it is correct and slow.
Profile before rewriting; report what you measured, not what you assume.
EOF
}

# --------------------------------------------------------------------------
# Score collection: best (highest final_score) results.json in trajectory/
# --------------------------------------------------------------------------
collect_score() {
    local child="$1"
    "$KB_EVAL_PYTHON" - "$child" <<'PY'
import json, sys
from pathlib import Path

child = Path(sys.argv[1])
best = None
for res in sorted(child.glob("trajectory/*/results.json")):
    try:
        data = json.loads(res.read_text())
    except Exception:
        continue
    score = data.get("score") or {}
    fs = score.get("final_score")
    key = (fs is not None, fs if fs is not None else -1.0, res.parent.name)
    if best is None or key > best[0]:
        best = (key, data, res)
if best is None:
    print("no_result,,,,")
    sys.exit(0)
_, data, res = best
score = data.get("score") or {}
passed = score.get("passed", "")
total = score.get("total", "")
fs = score.get("final_score")
label = (data.get("label") or res.parent.name).replace(",", ";")
print("%s,%s,%s,%s,%s" % (
    "scored" if fs is not None else "incorrect",
    passed, total, "" if fs is None else "%.4f" % fs, label))
PY
}

# --------------------------------------------------------------------------
# One campaign
# --------------------------------------------------------------------------
run_one() {
    local op="$1" gpu="$2"
    local label="kb-$op-$TAG"
    local child="$AGENTS_DIR/ako4x-run-$label"
    local log="$LOGDIR/$op.log"
    local marker="$child/.campaign_done"
    local t0 elapsed line status

    if [ -f "$marker" ]; then
        echo "[$op] SKIP (resume): $marker exists"
        return 0
    fi

    local -a spawn_env=(
        "PATH=$AKO_VENV/bin:$PATH"
        "AKO_DATASET_PATH=$AKO_DATASET_PATH"
        "KB_EVAL_PYTHON=$KB_EVAL_PYTHON"
        "KB_EVAL_REPO=$KB_EVAL_REPO"
        "KB_EVAL_ENTRYPOINT=$KB_EVAL_ENTRYPOINT"
        "CUDA_VISIBLE_DEVICES=$gpu"
    )

    if [ "$DRY_RUN" -eq 1 ]; then
        # Display form: keep $PATH symbolic (the expanded value is unreadable
        # and identical for every op).
        local envline="PATH=\$AKO_VENV/bin:\$PATH AKO_DATASET_PATH=$AKO_DATASET_PATH"
        envline="$envline KB_EVAL_PYTHON=$KB_EVAL_PYTHON KB_EVAL_REPO=$KB_EVAL_REPO"
        envline="$envline KB_EVAL_ENTRYPOINT=$KB_EVAL_ENTRYPOINT CUDA_VISIBLE_DEVICES=$gpu"
        echo "# ---- $op   gpu $gpu   -> $child"
        echo "env $envline \\"
        echo "    $KB_EVAL_PYTHON $AKO_HOME/spawn.py --operator kb_$op \\"
        echo "        --name $label --backend local"
        echo "cp $AKO_DATASET_PATH/blobs/kb_$op.json $child/expert_baseline.json"
        echo "cd $child && env $envline \\"
        echo "    $CLAUDE_BIN -p \"\$CAMPAIGN_PROMPT(kb_$op, budget=$ITERS iters)\" \\"
        echo "        --allowedTools \"$ALLOWED_TOOLS\" >> $log 2>&1"
        echo "date -Iseconds > $child/.campaign_done"
        echo "# fold best trajectory/*/results.json into $SUMMARY"
        echo
        return 0
    fi

    mkdir -p "$LOGDIR"
    t0=$(date +%s)

    if [ ! -d "$child" ]; then
        echo "[$op] spawning $child (gpu $gpu)"
        if ! env "${spawn_env[@]}" "$KB_EVAL_PYTHON" "$AKO_HOME/spawn.py" \
                --operator "kb_$op" --name "$label" --backend local >> "$log" 2>&1; then
            echo "[$op] SPAWN FAILED (see $log)"
            # 10 columns, same as the header: the 5 result columns stay empty.
            printf '%s,%s,%s,spawn_failed,,,,,,%s\n' \
                "$op" "$child" "$gpu" "$(( $(date +%s) - t0 ))" >> "$SUMMARY"
            return 1
        fi
    else
        echo "[$op] reusing existing child $child"
    fi

    cp "$AKO_DATASET_PATH/blobs/kb_$op.json" "$child/expert_baseline.json"

    echo "[$op] campaign start (gpu $gpu, $ITERS iters) -> $log"
    if ( cd "$child" && env "${spawn_env[@]}" "$CLAUDE_BIN" \
            -p "$(campaign_prompt "$op")" \
            --allowedTools "$ALLOWED_TOOLS" >> "$log" 2>&1 ); then
        status="ok"
    else
        status="agent_error"
    fi
    elapsed=$(( $(date +%s) - t0 ))

    line="$(collect_score "$child")"
    printf '%s,%s,%s,%s,%s,%s\n' "$op" "$child" "$gpu" "$status" "$line" "$elapsed" \
        >> "$SUMMARY"
    date -Iseconds > "$marker"
    echo "[$op] done ($status, ${elapsed}s): $line"
}

# --------------------------------------------------------------------------
# Deal ops round-robin, one worker per GPU
# --------------------------------------------------------------------------
echo "ops:        ${#OPS[@]} (${OPS[*]})"
echo "gpus:       ${GPUS[*]}"
echo "iters:      $ITERS"
echo "dataset:    $AKO_DATASET_PATH"
echo "grader:     $KB_EVAL_ENTRYPOINT ($KB_EVAL_PYTHON)"
echo "repo:       $KB_EVAL_REPO"
echo "summary:    $SUMMARY"
echo "dry-run:    $DRY_RUN"
echo

if [ "$DRY_RUN" -eq 0 ] && [ ! -f "$SUMMARY" ]; then
    mkdir -p "$(dirname "$SUMMARY")"
    echo "op,child_dir,gpu,status,result,scenarios_passed,scenarios_total,mean_speedup,best_label,seconds" \
        > "$SUMMARY"
fi

rc=0
if [ "$DRY_RUN" -eq 1 ]; then
    # Serial, in deal order, so the printed plan is readable. The GPU each op
    # gets is the same one the real run would deal it.
    echo "===== \$CAMPAIGN_PROMPT (identical per op except the task name) ====="
    campaign_prompt "<op>" | sed 's/^/    /'
    echo "==================================================================="
    echo
    for oi in "${!OPS[@]}"; do
        run_one "${OPS[$oi]}" "${GPUS[$(( oi % ${#GPUS[@]} ))]}" || rc=1
    done
else
    pids=()
    for gi in "${!GPUS[@]}"; do
        gpu="${GPUS[$gi]}"
        (
            for oi in "${!OPS[@]}"; do
                [ $(( oi % ${#GPUS[@]} )) -eq "$gi" ] || continue
                run_one "${OPS[$oi]}" "$gpu"
            done
        ) &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do
        wait "$pid" || rc=1
    done
fi

if [ "$DRY_RUN" -eq 0 ]; then
    echo
    echo "===== summary ($SUMMARY) ====="
    cat "$SUMMARY"
fi
exit "$rc"
