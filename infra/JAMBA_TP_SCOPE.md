# JambaEngine Tensor Parallelism — Engineering Scope

**Status**: not implemented. JambaEngine is single-rank only. This doc
specifies the work required to add TP=2 support, sized for a follow-up
PR.

## Why

Every other ≥10B LLM in kb-nano runs at TP>1: `gpt-oss-120b` TP=2,
`Llama-70B` TP=4, `Mixtral-8x7B` TP=4. `Jamba-v0.1` at 52B running at
TP=1 deviates from this convention. The current bench publishes both
sides (kb-nano + vLLM) capped at `max_num_seqs=256` on a single B200,
which fits but is the only ≥10B LLM in the project not at TP>1.

The CLAUDE.md guideline ("`use TP=1 for models under 10B parameters`")
prescribes TP=1 below the 10B threshold; for ≥10B the project pattern
is to use TP based on weight size and KV/state pressure.

## What's already in place

L2 attention + MLP are TP-aware via the project's standard parallel
linears:

- `tasks/baseline/L2/jamba_attention.py` uses `QKVParallelLinear` (col
  parallel) + `RowParallelLinear` (allreduces).
- `tasks/baseline/L2/jamba_mlp.py` uses `MergedColumnParallelLinear` +
  `RowParallelLinear` (allreduces).

These return the right shapes when `_tp_size() > 1`; they use the
`AllReduce` L1 op (custom IPC allreduce on small tensors, NCCL on
large) that the rest of the project shares.

`infra/tp.py` exposes `_tp_size()` / `_tp_rank()` reading from
`torch.distributed`. So the moment `dist.init_process_group` is called
in JambaEngine, the existing L2 modules will start sharding.

## What's missing

### 1. L2 `JambaMambaMixer` TP (4–6h)

The mixer at `tasks/baseline/L2/jamba_mamba_mixer.py` uses plain
`L1.linear.Linear` for `in_proj`, `x_proj`, `dt_proj`, `out_proj`,
plus a `Conv1d`-shaped weight `conv1d_weight`. None of these are
TP-aware. Pattern to mirror is `tasks/baseline/L2/mamba_mixer.py`:

- `in_proj`: `MergedColumnParallelLinear(hidden, [intermediate, intermediate])`
  — splits gate/x channels across ranks.
- `conv1d_weight`: shard along `intermediate` dimension (ColumnParallel
  semantics — every rank holds `intermediate / tp_size` channels). The
  `causal_conv1d_fn` and `causal_conv1d_update` kernels are channel-wise
  so per-rank computation is independent.
- `x_proj`: `ColumnParallelLinear(intermediate, time_step_rank + 2*ssm_state_size)`
  — outputs are replicated for the SSM transform.
- `dt_proj`: `ColumnParallelLinear(time_step_rank, intermediate)` — outputs
  dt per channel; rank-local.
- `out_proj`: `RowParallelLinear(intermediate, hidden)` — allreduce after.
- `A` log-parameter: shape `[intermediate, ssm_state]` — shard along
  `intermediate`.
- `D`: shape `[intermediate]` — shard along `intermediate`.
- `b_layernorm` / `c_layernorm`: replicated (operate on per-token B/C of
  size `ssm_state`, not sharded).
- `dt_layernorm`: shard along `time_step_rank` if rank-divisible (assert
  in `__init__`).

Constraint: `intermediate_size % tp_size == 0` (mirror `mamba_mixer.py`'s
assertion). Same `pad_slot_id=-1` sentinel pattern carries through.

The vendored vLLM Mamba kernels (`causal_conv1d_fn`, `selective_scan_fn`,
`causal_conv1d_update`, `selective_state_update`) are agnostic to which
slice of `intermediate` they receive, so per-rank invocation works.

### 2. L2 `JambaMoE` TP — expert parallelism (4–6h)

Currently `tasks/baseline/L2/jamba_moe.py` packs all experts' weights
into 3-D tensors `w13: [E, 2I, H]` and `w2: [E, H, I]`. Single-rank
only by construction (docstring: "No tensor parallelism").

Two TP strategies:

**(a) Tensor-parallel (shard the intermediate dim)** — every rank
keeps all experts but each holds `I/tp_size` channels:
- `w13: [E, 2*(I/tp_size), H]` per rank
- `w2: [E, H, I/tp_size]` per rank, allreduce after

This matches the dense MLP pattern. FusedMoE kernel needs to handle
the sharded shapes correctly; verify with the existing `triton_kernels`
backend.

**(b) Expert-parallel (shard the expert dim)** — every rank keeps a
disjoint subset of experts; gather expert IDs across ranks each step.

(a) is simpler and matches what gpt-oss-120b does at TP=2 (TP-shard the
expert intermediate dim). Recommend (a) first; (b) is a future optimisation.

Weight loader at `tasks/baseline/L4/jamba.py` lines ~378–392 needs
per-rank shard slicing for `w13` (split along `2I` axis) and `w2`
(split along `I` axis).

### 3. `JambaEngine` multi-rank fork (6–8h)

Today: single-process, single-CUDA-context. Need to mirror
`infra.engine.LlamaEngine`'s pattern:

- `__init__` spawns `tp_size - 1` worker subprocesses via
  `mp.get_context("spawn").Process(target=_JambaModelRunner, ...)`.
- Each rank runs its own `_JambaModelRunner` (new class, mirror of
  `infra.engine.ModelRunner` but Jamba-specific) which calls
  `dist.init_process_group("nccl", ...)` with rank/world_size.
- Rank 0 drives the scheduler (`waiting / prefilling / running`
  deques) and dispatches forward calls to all ranks via shm + `Event`.
- Workers listen on shm for next-step input arrays, run forward,
  produce next_tokens, broadcast to rank 0.
- Sampling: rank 0 runs `argmax` / `_sample_one`, broadcasts the
  sampled token IDs to all ranks (so each rank's KV cache + Mamba
  state advance with the same input next step).
- KV cache: each rank allocates `num_kv_heads / tp_size` heads worth
  of paged blocks. `JambaAttention` already handles this via
  `QKVParallelLinear.num_heads` shard math.
- `_MambaSlotPool`: per-rank slabs of size `[num_slots, K-1,
  intermediate / tp_size]`. State allocation/deallocation is broadcast
  from rank 0 in lock-step (same slot index on every rank).
- TRTLLM workspace: each rank gets its own; share across attention
  modules within rank like today.

The rank-0 scheduler stays sequential (Python). The forward/sampling
fan-out is the perf-sensitive bit and matches what LlamaEngine does.

### 4. L4 weight loader per-rank sharding (2–3h)

`tasks/baseline/L4/jamba.py:load_weights` currently loads all
weights replicated. Need to:

- For Parallel* params: invoke `param.weight_loader(param, tensor,
  shard_id)` which already handles per-rank narrowing (see
  `RowParallelLinear._weight_loader` etc.).
- For Mamba mixer params: add per-rank `narrow` calls along the
  `intermediate` axis.
- For MoE `w13`/`w2`: per-rank narrow along the appropriate axis.
- For replicated params (norms, embedding, lm_head): broadcast as-is
  (every rank holds the full tensor).
- Embedding + lm_head: TP-shard the vocab axis is optional; vLLM
  default is to keep these replicated. Match that.

### 5. `bench_jamba.py --tp` CLI (1h)

- Add `parser.add_argument("--tp", type=int, default=1)`.
- Plumb through to `JambaEngine(tensor_parallel_size=args.tp)`.
- vLLM reference already accepts a `tp` arg; mirror.
- Output dir naming: `<model>_jamba_tp<tp>` (already does this).

### 6. Re-bench + correctness verification at TP=2 (2–4h GPU time)

- v0.1 at TP=2 on 2× B200, full N=1000 paper-spec.
- Compare match-tokens to TP=1 (should be identical at TP=1; at TP=2
  may differ slightly due to allreduce ordering bf16 noise — verify
  it's within ~5–10 tokens/req of vLLM TP=2).
- Latency at TP=2.

## Order of work

1. Mamba mixer TP first (smallest blast radius; can verify at TP=2
   with a stub model).
2. MoE TP next.
3. Engine multi-rank last (the "wiring up the things" step).
4. Weight loader updates concurrently with engine work.
5. Bench + verify at the end.

Estimate **~20–28h focused engineering**, plus debug iterations.

## Why this is its own PR, not a follow-up commit on `add-jamba-support`

CLAUDE.md is explicit: "Correctness must hold at every step.
Performance with broken match-tokens is not progress; it's a
regression with a misleading speedup number." Half-implemented TP
with the multi-rank fork landed but the L2 mixers still single-rank
would silently break: the engine would run, but the mamba states /
MoE outputs would be inconsistent across ranks because the L2 mixers
wouldn't shard. Better to do all the L2 TP first, verify TP=2 against
TP=1 numerically at the L2 level (microbenches), then wire the engine.

That structure doesn't fit cleanly as commits on the current
`add-jamba-support` branch — each step needs its own correctness
gate before the next.
