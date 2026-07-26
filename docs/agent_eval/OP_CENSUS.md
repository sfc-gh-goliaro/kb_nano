# Op census — current state

**Scoreboard (2026-07-26, final pass): 103 of 106 registry ops fully
RUNNABLE** under the Tier-1 grader (baseline builds from registry
init_args -> strict weight transfer -> recorded inputs -> compare).
Started this effort at 33.

The 3 residuals are by design, not defects:

| op | why | where its correctness lives |
|---|---|---|
| llama | identity needs two whole-model copies on one GPU | e2e tier (token-matching vs vLLM) |
| gpt_oss | same | e2e tier |
| qwen3_vl | same | e2e tier |

Ops needing special handling, all closed and verified:
- `allreduce` — graded by its own multi-rank harness
  (`tools/agent_eval/allreduce_runner.py`), analytic ground truth; 15/15.
- `oasis_rollout` — forward takes whole sub-models; the grader builds the
  DiT+VAE fixture in trusted code.
- `gpt_oss_moe`, `chunk_gla`, `vision_block` — gradeable, but their
  cancellation structure means non-bit-identical implementations exceed
  the 1% band; see the fp64-oracle dual-gate policy in OPEN_DECISIONS
  history. Excluded from the packaged task menu until that lands.

Task menu: see `PACKAGING_REPORT.md` in the generated dataset dir (the
authoritative list; every seed is grader-verified before shipping).

Raw data: `census_v3.json` / `census_final.json` + per-op receipts in the
pilot scratch. Everything below this line is the historical record of how
the number moved 33 -> 103; it contains superseded statements and is kept
for audit only.

---

# [HISTORY] Census v1 (2026-07-25) — SUPERSEDED

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

RECOMMENDED RESOLUTION (ratify, then execute; done 2026-07-25): **static rebuild** —
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

---

# [HISTORY] Census v2 (2026-07-25) — SUPERSEDED

After registry population (1181 -> 1386 scenarios), the entrypoint upgrade
(no-mangling instantiation + config namespace-wrap + seeded valid structured
inputs ported/extended from the experiments-codex prep + weight/buffer repair
+ fp8 exemptions + non-finite policy), and three post-census data fixes:

**85 / 106 fully RUNNABLE** (was 33). Raw data: census_v2.json.

Supersedes three v1 flags above: `flash_attn_varlen` now PASSES identity (the
v1 "nondeterministic" symptom was invalid unseeded varlen metadata, fixed by
the entrypoint's seeded structured-input prep — the "do not score" flag is
lifted), and `moe_grouped_gemm` / `fp8_linear` both pass on B200 (the v1
deep_gemm crash/assert was garbage-weight-induced, fixed by fp8-aware weight
prep).

Codex cross-check (2026-07-26, scratch codex_crosscheck.py): replaying the
experiments-codex runner's init-arg reconstruction formulas over the
pre-population records and diffing against our populated init_args gives
43/49 Lane A field agreements. Every disagreement is a codex FABRICATION
default where ours holds the checkpoint-true value: gla_mlp intermediate
(codex hidden*4=10240 vs true 6912 — fla derives int(2560*4*2/3) rounded up
to a 256 multiple; config.json has intermediate_size=null + hidden_ratio=4),
yolov10 conv/scdown out-channels (codex c2:=c1 / c2:=2*c1 vs real yolov10n
head/backbone channels), parallel_linear 4096-in (codex square-only vs our
enumeration carrying BOTH real layers the old bundled record covered:
llama o_proj 4096->4096 AND gpt-oss o_proj 4096->2880). Lane B by design
diverges from codex's fabrications (square Linears: ours 21/177 square with
30 real (in,out) pairs; Embedding dim 128: ours {4096,2880,2560,1152,1024};
Conv2d 1x1: ours 31/59).

Near-runnable (1-7 failing scenarios each, cause known): attention 14/15
(one activation-nonfinite scenario), gla 9/10 (one cu_seqlens step-sign
scenario), gla_decoder 313/320, gla_attention 8-10/10 (intermittent,
GPU-dependent -- monitor), moe_align 2/10 (permutation contract; the codex
runner solved this with output canonicalization -- `_canonicalize_output_for_
target` -- adopt after ratification).

# [HISTORY] Census v2.2 (2026-07-26) — SUPERSEDED by the current-state header

**Effectively 89/106 runnable.** The census-v2 "near-runnable" paragraph
above is SUPERSEDED for four ops -- their failures were measured with the
pre-8c46aa8 repair (NaN/Inf/zero only) and are gone under the current
grader: attention 15/15, yolov10_backbone 1/1, gla_decoder 320/320,
gla_attention stable (21 consecutive identity runs clean across GPUs 1/5/7,
including 3 post-rebase).

**gla_attention root cause (proven; kernel-nondeterminism hypothesis
REFUTED).** GLA-family modules allocate weights UNINITIALIZED (repair logs
name q/k/v/g_proj across every layer; the whole GLAForCausalLM state dict
on the L4). Allocator garbage varies per run and per GPU-state; the old
repair passed huge-but-finite garbage -> bf16 overflow ->
baseline_output_nonfinite -> RUNTIME_ERROR on however many scenarios drew
bad bytes. A/B on demand: ca58784 entrypoint = 4/8 runs fail with varying
counts (10/10, 8/2, 2/8, 9/1); current entrypoint = 21/21 runs clean.
Receipts: pilot scratch gla_attn_repro_*.json.

**Remaining genuinely-failing:** gla 9/10 -- `tokens-70/2f8947b2` raises
"upper bound and lower bound inconsistent with step sign" deterministically,
including post-rebase (28cf517's per-segment varlen fix does NOT cover it).
This is a harness input-prep defect; fix stream E1 (OPEN_DECISIONS.md).

**Rebase note:** branch now sits on release @ 28cf517. attention_impl's
fast dispatch (_TRITON_UNIFIED_AVAILABLE) flipped False -> True in our venv
(vllm 0.18.0): correctness identity was never affected (same dispatch both
sides), but census `seconds` fields recorded for attention_impl-family ops
pre-rebase raced the slow path -- do not compare them to post-rebase
timings.

Remaining residuals by class:
1. L4 identity is memory-bound: llama, gpt_oss, qwen3_vl (two whole-model
   instances exceed single-GPU memory; L4 correctness belongs to Tier-2/3
   e2e as designed). yolov10 L4: detection postprocess on noise inputs
   (empty-tensor max) -- same conclusion.
2. L3 composite depth: yolov10_backbone (activation blowup through deep
   random-weight conv stack), llama_decoder / gpt_oss_decoder /
   qwen3_moe_decoder (decoder forward wiring: rotary_emb forward arg,
   TRTLLM decode-path cache API, deep_gemm fp8-in-module; recipes in
   drill2_merged.json + this file).
3. Class/scenario mismatches: rotary_emb, tensor_ops (discovery picks a
   later-added class than the one traced; needs an op->class resolution
   decision). NOTE: the experiments-codex CSV records bitwise-0 "PASS" for
   both -- those published rows likely evaluated the wrong class.
4. Constructor needs modules: oasis_block, oasis_spatial/temporal_axial_
   attention (builder recipe documented; the codex runner built these
   inline).
5. Fixture-independence: mxfp4_moe (codex used real-checkpoint weights
   prepared by the implementation under test -- rejected here by design;
   bespoke trusted-preprocessing lane documented).
6. Distributed: allreduce (codex CSV: SKIPPED distributed_not_run with an
   NCCL env failure note; paper number came from the separate 4-rank
   harness).

How the codex branch handled each residual it faced is recorded above with
CSV receipts; 4 of its 7 handlings used mechanisms we surface for
ratification instead of adopting silently.
