# Op census: what is evaluable today (B200, 2026-07-25)

Method: for every operator in the shape registry (106), run the entrypoint's
baseline-identity check (build baseline twice from registry init_args -> strict
weight transfer -> feed recorded shapes -> compare -> time) in an isolated
subprocess; then rerun every all-scenarios-fail op once and classify its actual
error message. Raw data: `census.json` / `drilldown.json` in the pilot scratch
(`/raid/user_data/olu/scratch/agent_eval_pilot/`), summary tables committed
here.

## Headline

| Bucket | Count | Meaning |
|---|---|---|
| RUNNABLE | 33 | Full chain works today — ready for agent campaigns (rms_norm proven end-to-end) |
| CONFIG-blocked | 61 | Constructor args missing from registry `init_args` — the population worklist |
| Mixed identity results | 7 | Some scenarios pass, some fail identity — per-scenario input-semantics issues |
| Identity-nondeterministic | 1 | `flash_attn_varlen`: baseline vs itself fails numerically on ALL scenarios |
| Special | 4 | `moe_grouped_gemm` (deep_gemm CUDA crash on B200), `allreduce` (needs process group — use a distributed harness like the paper's 4-rank runner), `fp8_linear` (deep_gemm layout assertion), `oasis_patch_embed` (default 256x256 vs recorded 360x640 input — config-adjacent) |

RUNNABLE (33): batch_norm2d, chunk_retention, conv3d, dense_attention,
diffusion_rope, flash_attn_decode, flux_pos_embed, fused_experts,
fused_recurrent_gla, fused_recurrent_retention, gelu, gla_recurrence,
interpolate, l2_norm, log_sigmoid, max_pool2d, moe_sum, rms_norm, ... (full
list with per-op tallies in census.json).

Mixed (7): chunk_gla, flash_attn_prefill, flashinfer_decode (47/56 pass),
flashinfer_prefill (1/18), flux_attention, moe_align (1/10), store_kvcache
(14/15). Likely cause: randomly materialized structured inputs (block tables,
cu_seqlens, expert indices) are not always valid/deterministic for these ops —
investigate before scoring agents on them.

## The population plan for the 61 CONFIG-blocked ops

Two lanes with different work:

**Lane A — model-scoped ops (the majority: decoders, MLPs, MoEs, oasis_*,
vision_*, yolov10_*, gla_*, ...).** Deterministic recipe, fully static, no
runs: op name -> the traced checkpoint -> its config.json -> write the
constructor fields as per-scenario `init_args` (registry data), plus a
one-time generic shim in the entrypoint ("if an init arg named `config`
arrives as a dict, wrap it in an attribute object"). The authoritative list
of traced checkpoints is `bench/kernels/benchmark_scenarios/small/config.yaml`
(9 models: llama31-8b, gpt-oss-120b, gla-2.7b-100b, flux1-dev,
qwen3-vl-235b-a22b-fp8, yolov1on, openfold3, bge-m3, oasis-500m); op->model
matching uses the same family logic as `scenario_pipeline._resolve_targets`.
Verify each op with the identity check; it flips to RUNNABLE. Sanity
cross-check: diff derived values against the experiments-codex runner's
hardcoded reconstructions (`bench/kernels/runner.py` on that branch, the
if/elif chain at ~lines 88-360) — they should agree for the traced models.

**Lane B — dimension-generic ops (linear, embedding, conv2d,
parallel_linear; layer_norm turned out input-derivable and belongs to Lane
A).** The registry records inputs only (no output shapes, no call-site
provenance), so a Linear scenario's `out_features` is unrecoverable from its
record. Worse, the tracer deduplicated by input signature, so one record
BUNDLES several distinct layers (q/o/gate/up_proj all receive the same
tensor shape) — single-assignment of an `out` per old record is ill-posed,
not merely hard. The experiments-codex runner fabricated defaults here
(Linear: out=in i.e. square; Embedding: dim=128; Conv2d: 1x1) — meaning the
PUBLISHED per-op numbers for these ops were measured on non-production
dimensions.

RECOMMENDED RESOLUTION (ratify, then ~1 day of work): **static rebuild** —
walk each of the 9 traced models' module trees on the meta device (config ->
model class -> no weights, no GPU; works even for the 235B), enumerate every
real (in, out) call site, regenerate these ops' scenarios per distinct
(in, out) with leading shapes curated to the standard regimes, retire the
old bundled records, disclose the fixture correction in the revision.
Expected scenario growth for `linear`: 66 -> roughly 130-260 (exact count
falls out of the enumeration). Alternatives considered and rejected:
replicate the fabrications (comparability with a fictional fixture; keep
only if a transition table is wanted), heuristic single-assignment
(ill-posed due to bundling), runtime re-trace (dominated — same information,
plus GPU/download/disk costs; remains the right tool for future wholesale
registry regeneration), dropping the ops (needless coverage loss).

## Flags for anyone scoring agents

1. `flash_attn_varlen`'s baseline appears nondeterministic under the harness
   (identity fails numerically everywhere) — do not score agent kernels on it
   until understood.
2. The rms_norm <=4-token flaky-verdict finding (VALIDATION_REPORT.md) likely
   generalizes to other near-tolerance cases; the mixed-bucket ops above are
   the first places to look.
3. Registry init_args were produced by a scalar-attribute harvest
   (`bench/kernels/scenario_pipeline.py:_extract_init_args`) — the config gap
   is a designed limitation of that extractor, not data loss; nothing needs
   re-running to fix it (values are static in each checkpoint's config.json).
