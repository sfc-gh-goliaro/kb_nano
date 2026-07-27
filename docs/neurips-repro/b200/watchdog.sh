#!/bin/bash
# Keep the B200 work alive, in two phases:
#
#   1. the main sweep (all 8 GPUs) -- establishes coverage: every row runs and
#      aligns. next_jobs.py picks what still needs running (failures within their
#      retry budget, plus rows that exited 0 without a usable reference).
#   2. once phase 1 has nothing left, the low-concurrency re-measurement pass
#      over a 3-GPU pool. The host is 1.5x oversubscribed at 8-wide and
#      fastkernels is host-heavier than the references, so the numbers we quote
#      have to come from a pass where the host is not the bottleneck.
#
#   3. finally DeepSeek-V3.2 at TP=8, which needs all 8 GPUs at once. It runs
#      last on purpose: it would otherwise monopolise the pool and delay both
#      the retries and the clean re-measurement, and it is the one row the H200
#      reproduction also left blank (known decode regression).
#
# Never runs two schedulers at once.
set -uo pipefail
ROOT=/home/yak/b200_repro
LOG=$ROOT/logs/watchdog.log

export HF_HOME=/home/yak/data-fast/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false

log(){ echo "[$(date -u +%H:%M:%S)] $*" >> "$LOG"; }

launch(){  # jobs_file status_file pool
  local jobs="$1" status="$2" pool="$3" next n
  next=$(cd "$ROOT" && python next_jobs.py "$jobs" "$status")
  if [ -z "$next" ]; then return 1; fi
  n=$(echo "$next" | tr ',' '\n' | wc -l)
  log "launching $jobs for $n job(s) on pool $pool"
  ( cd "$ROOT" && nohup python sched.py "$jobs" --pool "$pool" \
      --status "$ROOT/$status" --resume --only "$next" \
      >> "$ROOT/logs/sched_$(basename "$jobs" .json).log" 2>&1 & )
  return 0
}

# GPUs listed in logs/reserved_gpus.txt (one index per line) are excluded from
# every pool. Long orphaned runs live there -- e.g. BitNet, whose Microsoft
# reference takes ~2.5h -- so a relaunch cannot drain the GPU out from under them.
pool_excluding_reserved(){
  # Optional $1 = how many GPUs to take (phase 2 wants a narrow pool).
  local want="${1:-8}" all="0 1 2 3 4 5 6 7" out=() g
  for g in $all; do
    if [ -f "$ROOT/logs/reserved_gpus.txt" ] && grep -qx "$g" "$ROOT/logs/reserved_gpus.txt"; then
      continue
    fi
    out+=("$g")
    [ "${#out[@]}" -ge "$want" ] && break
  done
  (IFS=,; echo "${out[*]}")
}

log "watchdog started (pid $$)"
# bash buffers the loop body, so editing this file while it runs takes effect
# unpredictably -- three times a phase I had added was silently skipped because the
# running process was still executing an older copy. Re-exec when the file changes.
_SELF_MTIME=$(stat -c %Y "$0")
while true; do
  if [ "$(stat -c %Y "$0")" != "$_SELF_MTIME" ]; then
    log "watchdog.sh changed on disk -- re-exec'ing to pick it up"
    exec bash "$0"
  fi
  if pgrep -f "sched.py jobs_" > /dev/null; then
    sleep 60; continue
  fi
  if launch jobs_b200_full.json logs/status_full.json "$(pool_excluding_reserved)"; then
    sleep 90; continue
  fi
  log "main sweep complete -> re-measurement pass"
  # Low concurrency on purpose (the host is 1.5x oversubscribed at 8-wide), but
  # still exclude reserved GPUs: hardcoding 0,1,2 here would have let the
  # pre-launch drain kill BitNet's orphan on GPU 0.
  if launch jobs_remeasure.json logs/status_remeasure.json "$(pool_excluding_reserved 3)"; then
    sleep 90; continue
  fi
  # Phase 2 rows that ran while my own diagnostics were also on the box did not get
  # the quiet host they exist to measure on; logs/overlapped_remeasure.txt lists
  # them and jobs_remeasure2.json re-runs exactly those.
  if launch jobs_remeasure2.json logs/status_remeasure2.json "$(pool_excluding_reserved 3)"; then
    sleep 90; continue
  fi
  # SwinV2 (0.874x vs 1.17x) is a line-for-line port of timm's, so parity is the
  # ceiling -- except that both sides recompute the continuous position bias every
  # forward though it is constant in inference. We now cache it; this re-run checks the
  # embedding cosine stays 1.0000 and measures what the cache buys.
  if launch jobs_swinv2.json logs/status_swinv2.json "$(pool_excluding_reserved 1)"; then
    sleep 90; continue
  fi
  # Blackwell's HND page size is 16; 64 measures 4-12% faster on both Llama and
  # Mixtral at identical alignment. Sweep the attention rows at 64 as a separate
  # tuned column -- after the clean re-measurement, never alongside it.
  # 4 wide, not 3: p64_mixtral inherits Mixtral's TP=4.
  if launch jobs_page64.json logs/status_page64.json "$(pool_excluding_reserved 4)"; then
    sleep 90; continue
  fi
  # Authoritative post-parity table. Split by GPU count on purpose: the single-GPU rows
  # run on a 1-wide pool so exactly one benchmark is on the box at a time (4-wide cost
  # Llama 6% -- 0.977x standalone versus 0.917x contended), and the multi-GPU rows follow
  # on a 4-wide pool where a TP=4 job occupies the machine anyway.
  if launch jobs_quiet1.json logs/status_quiet1.json "$(pool_excluding_reserved 1)"; then
    sleep 90; continue
  fi
  if launch jobs_quietN.json logs/status_quietN.json "$(pool_excluding_reserved 4)"; then
    sleep 90; continue
  fi
  # The block-table stride fix changes the numbers for every row that uses the paged
  # attention path, so the whole table needs re-measuring before anything is quoted.
  # This runs first and is the authoritative post-fix pass. 4 wide because Mixtral is TP=4.
  if launch jobs_postfix_all.json logs/status_postfix_all.json "$(pool_excluding_reserved 4)"; then
    sleep 90; continue
  fi
  # bench_fla / bench_jamba / bench_timm write to a FIXED results.json rather than a
  # timestamped run dir, so the page-64 and prefill-budget experiments silently
  # overwrote the default-configuration results for RWKV-7, GLA, Jamba and SwinV2 --
  # their as-published numbers now exist only in the bench logs. Restore them first;
  # the experiment job files now pass --output-dir so this cannot recur.
  if launch jobs_rebaseline.json logs/status_rebaseline.json "$(pool_excluding_reserved 3)"; then
    sleep 90; continue
  fi
  # Back-to-back SwinV2 A/B on one GPU: the first cached run looked like 0.874x ->
  # 0.955x, but timm's own throughput moved 25% between the two runs being compared, so
  # the pair has to be measured under the same conditions to mean anything.
  if launch jobs_swinv2b.json logs/status_swinv2b.json "$(pool_excluding_reserved 1)"; then
    sleep 90; continue
  fi
  # Mamba2's full-scale profile puts 24% of wall time in the mixed-prefill path (86
  # steps at 358 ms) against a 16384-token budget. The same knob was worth +33-53% on
  # RWKV-7, so try it here.
  if launch jobs_mamba2b.json logs/status_mamba2b.json "$(pool_excluding_reserved 2)"; then
    sleep 90; continue
  fi
  # RWKV-7/GLA/RetNet share their prefill kernel with the FLA reference, and the
  # reference gives itself a 196608-token prefill budget where our engine defaults to
  # max_num_batched_tokens=16384. For chunk-based linear attention that is a 12x
  # smaller launch, so try larger budgets.
  if launch jobs_fla.json logs/status_fla.json "$(pool_excluding_reserved 3)"; then
    sleep 90; continue
  fi
  # Mamba2 is the worst speed gap (0.53x vs 0.97x). Its decode kernels are vLLM's own
  # and host work is 1.2% of the loop, so the remaining lever is graph coverage: the
  # bucket cap is 256 while the engine runs 987 state slots.
  if launch jobs_mamba2.json logs/status_mamba2.json "$(pool_excluding_reserved 2)"; then
    sleep 90; continue
  fi
  # Both GPT-OSS rows fault inside triton_kernels matmul_ogs on sm100 -- a kernel
  # vLLM only selects on Blackwell when FlashInfer is missing. Try Hopper's
  # split_k=1 (and non-persistent) constraints, which dodge the two faulting paths.
  if launch jobs_gptoss.json logs/status_gptoss.json "$(pool_excluding_reserved 3)"; then
    sleep 90; continue
  fi
  # The big gap: jobs_remeasure only ever covered 17 rows, so ~30 table rows still carry
  # 8-wide coverage numbers, and the 8-wide pass is demonstrably off by up to +-40%
  # (RWKV-7 1.270 vs 0.710). This re-measures every remaining row on a quiet host.
  # 4 wide rather than 3 because Mixtral is TP=4 and a 3-wide pool could never place it
  # -- the same deadlock the TP=8 guard below exists to avoid.
  if launch jobs_quiet_all.json logs/status_quiet_all.json "$(pool_excluding_reserved 4)"; then
    sleep 90; continue
  fi
  # DeepSeek-V3.2 and Qwen3-VL-235B both need gpus=8. If any GPU is reserved the pool
  # is 7 wide and sched.py can never place them -- it sat with all 7 idle. Skip the
  # phase until the reservation clears rather than deadlock on it.
  _pool="$(pool_excluding_reserved)"
  if [ "$(echo "$_pool" | tr ',' '\n' | wc -l)" -lt 8 ]; then
    log "TP=8 phase deferred: pool is $_pool, reserved GPUs still busy"
    sleep 180; continue
  fi
  log "re-measurement complete -> DeepSeek-V3.2 (TP=8, needs the whole pool)"
  if launch jobs_deepseek.json logs/status_deepseek.json "$_pool"; then
    sleep 90; continue
  fi
  log "nothing left to run (all succeeded or retries exhausted); idling"
  sleep 180
done
