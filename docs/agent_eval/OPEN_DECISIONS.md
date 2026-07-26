# Agent-eval: execution status + open decisions

Rewritten 2026-07-26 after the branch was rebased onto `origin/release`
@ 28cf517 (mentor's H200 table-repro fixes). Supersedes the earlier
D1-D11 numbering; resolved items moved to the bottom. Glossary
(registry / identity check / Lane A-B) unchanged — see OP_CENSUS.md.

## IN EXECUTION (decided; fix streams running on this branch)

### E1. Harness fixes (grader-side, `tools/agent_eval/agent_entrypoint.py`)
- Decoder L3 wiring: llama_decoder, gpt_oss_decoder, qwen3_moe_decoder
  (rotary forward arg / decode-path cache API / fp8-in-module; recipes in
  pilot scratch drill2_merged.json).
- oasis_block + oasis spatial/temporal axial attention: constructor needs a
  rotary-embedding *module*; build it in the harness (codex runner precedent:
  its `_instantiate_module` OasisSpatial/Temporal branch).
- gla `tokens-70/2f8947b2`: "upper bound and lower bound inconsistent with
  step sign" in the varlen input path — SURVIVES the 28cf517 rebase
  (re-measured post-rebase: 9/10), so it is our harness/input-prep defect,
  not the upstream bug that commit fixed.
- Repair perf: `_repair_degenerate_parameters` re-inits ~350 tensors per
  scenario on 32-layer GLA models via CPU generator (minutes/scenario);
  batch it without changing repaired values (or fully re-verify if values
  change).

### E2. Op→class pins (was D4) — the "last class in the file" fix
Registry gains an explicit `class_name` per op; resolution
(`infra/kernel_swapper._find_module_class` and its consumers) honors it,
additively (unpinned ops keep the last-class rule). Evidence-backed pins:
rotary_emb→RotaryEmbedding, tensor_ops→OneHot, softmax→Softmax (traces are
YOLO DFL-head plain softmax), flux_transformer_block→FluxTransformerBlock
(recorded inputs are dual-stream). Audits still owed: yarn_rotary_emb
(reference may mirror the other spelling-variant class — reclassifies its
E3 entry), yolov10_c2f (single scenario, attribute C2f vs C2fCIB), and
scenario-fit confirmation for the other 14 multi-class files (sweep:
20 of 106 op files define >1 nn.Module class). Runtime identity checks
CANNOT catch a wrong-but-compatible pick (softmax and
flux_transformer_block passed while measuring the wrong class) — the pin
is the only structural fix. Merge-to-main needs mentor ratification
(touches shared infra).

### E3. tasks/reference repairs (was D9 + D3)
12 reference files fail against their own baseline when actually executed
(the original pipeline never ran them — reference paths were display-only;
the optional `pytorch_reference` runner mode that would have caught this
has no recorded run). 5 crash, 7 disagree numerically (1.18x to 16,654x
over tolerance). Fix each in place; the three packaging-time SEED_PATCHES
(softmax / flash_attn_varlen / flux_attention) are retired as their files
get fixed or their pin lands. Proactive sweep: verify EVERY existing
reference file through the grader, not just the previously-failing set —
the "12" is a lower bound from the 85-op packaging pass.

### E4. moe_align output canonicalization (was D2) — DONE
Implemented + verified (10/10 twice; wrong-expert negative control 10/10
INCORRECT). Original rationale below.

### E4-history. moe_align output canonicalization (was D2)
Adopted (user 2026-07-26): sort both sides within expert groups before
comparing, scoped to this op. Grounded in the consumer:
`fused_experts.py:309-354` uses `sorted_token_ids`/`expert_ids` purely as
gather/scatter indices — order within an expert's group is not part of the
contract. Port of codex `_canonicalize_output_for_target` (runner.py:955).

### E5. mxfp4_moe frozen real-checkpoint fixture (was D5)
Weight-side analog of the benchmark's golden-inputs mechanism: extract the
already-MXFP4-packed expert tensors from the gpt-oss checkpoint once,
freeze content-hashed blobs, grader pours identical bytes into both sides
(strict transfer unchanged). Precondition: verify checkpoint tensor layout
matches the module's state-dict layout; if a transform is needed it runs
once, trusted, frozen. Integrates after E1 (same file).

## MENTOR-GATED (not blocked on us)

- **M1 (was D1)**: ratify the Lane B fixture rebuild for merge + revision
  disclosure (codex numbers for linear/embedding/conv2d/parallel_linear
  were measured on fabricated dims).
- **M2 (was D6)**: IN EXECUTION (user 2026-07-26: build without waiting on
  the mentor) — a 4-rank harness with analytic ground truth is being built
  as fix stream; the original-source question survives only as a
  comparability footnote for the published 0.84x.
- **M3**: IN EXECUTION (user 2026-07-26: author them) — references for the
  6 L3 composites (flux_transformer_block, oasis_dit,
  oasis_vae_attention_block, vision_block, yolov10_head/neck) are being
  written + grader-verified; they become 6 new packageable agent tasks.
- **M4**: merge-to-main ratification of E2 (resolution semantics), E3
  (reference edits), E4 (comparison semantics).
- **M5**: ASTRA — verify GPT-5.5 model id via /v1/models; run
  astra_live_smoke.sh (dual-mode; OpenAI mode documented in header).
  AK campaigns: Anthropic API key billing (decided).
- **M6**: optional experiment knobs, disclose if used: baseline-seeded AK
  arm (only for ops with portable baselines); showing baseline source to
  the agent as reading material (codex-style).
- **M7 (baseline-file territory, found by fix stream E1)**:
  `TRTLLMPrefill.forward` silently drops the `s_aux` (attention sinks) and
  `window_size` kwargs on the no-block-table path
  (tasks/baseline/L1/flashinfer_prefill.py:26-55), so gpt-oss's
  sinks/sliding-window semantics never reach the prefill kernel in the
  Tier-1 harness context. Symmetric for baseline and candidate (identity
  unaffected) but those semantics are therefore UNTESTED by Tier-1 for the
  affected scenarios. Fixing means editing a baseline file — mentor call.
- **M8 (upstream, found by fix stream E1)**: triton_kernels' `matmul_ogs`
  ragged-TMA path performs an out-of-range read on every launch on
  SM100/B200 (compute-sanitizer receipt in stream A's report; faults only
  when it hits unmapped VA — order-dependent crashes). Grader mitigates
  with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` scoped to the
  four matmul_ogs ops. Worth escalating upstream (triton 3.6.0 /
  triton_kernels) and re-checking on H200 (SM90 uses a different path).

## RESOLVED (this session; receipts in OP_CENSUS.md + pilot scratch)

- Group 7 dissolved. Root cause of gla_attention "intermittency": GLA-family
  modules allocate weights UNINITIALIZED; allocator garbage varied per run;
  the pre-8c46aa8 repair missed huge-but-finite garbage -> bf16 overflow ->
  nonfinite guard. Proven by A/B: old entrypoint 4/8 runs fail variably;
  current 21 runs clean. Kernel-nondeterminism hypothesis REFUTED.
  Collateral fixes measured: attention 15/15, yolov10_backbone 1/1,
  gla_decoder 320/320, gla_attention stable. Census now effectively 89/106.
- Packaging re-verified post-fix: 68 packaged / 17 skipped (yolov10_c2f
  stale skip flipped).
- Rebase onto 28cf517 clean (no file overlap); post-rebase battery:
  gla_attention 3/3, rotary_emb crash unchanged (expected), gla 9/10
  (step-sign persists -> E1), attention_impl fast dispatch now ACTIVE in
  our env (vllm 0.18.0: pre-rebase False -> True; correctness unaffected,
  pre-rebase census timing fields for attention_impl-family are stale).
- flash_attn_varlen / moe_grouped_gemm / fp8_linear stale v1 flags lifted
  (see OP_CENSUS v2 supersession note).

## Late additions (2026-07-26, execution round 2)

- **Difference 9 (chunk_gla grading precision)**: the op's fp32-container
  outputs carry bf16-precision computation (kernel ex2.approx + bf16
  rounding; the BASELINE sits ~1e-4 from fp64 truth where the fp32 band is
  ~1e-5 — E3 stream measurement), so the grader now applies low-precision
  tolerances to this op (raw values, tolerance-scoped; a bf16 cast was
  tried and rejected — quantization-boundary cliffs). Negative control: a
  10% scale-error candidate fails 5/5. Residual o-path divergence under
  investigation (reference side).
- **yolov10_head / yolov10 closed**: two stacked causes, both fixed —
  (1) the repair loop crashed on the head's zero-element anchor/stride
  buffers (empty-tensor .max()); guard added. My earlier
  "postprocess-on-noise" classification of yolov10_head was WRONG — the
  crash frame was in the grader, not the model. (2) the head's production
  bias prior yields zero detections on random weights; the fixture now
  boosts cls biases so the decode/select path executes (comparing hidden
  states instead was considered and rejected: intermediates are not part
  of the op contract and fused candidates must remain free to skip them).
  Both ops PASS; rms_norm regression clean.
- **SEED_PATCHES retired** (all three; raw references verified through
  packaging: softmax + flash_attn_varlen + flux_attention SEED_PASS).
- **gpt_oss_moe reference reopened → resolved as M9 member**: its earlier
  PASS was vacuous (all-zero packed weights, pre-Fix-7). Re-fix found and
  repaired 3 real reference bugs (gate bf16 rounding, per-slot bf16
  scratch rounding, chunked dequant; ratio 110.5 → ~1.4), and the residual
  is PROVEN unreachable: at the cancellation offenders, bf16-rounding the
  fp64-TRUTH slot values reproduces the REFERENCE bit-exactly — the
  baseline is the side one ULP off truth (its fp32-accumulator noise).
  Fixture scale-bound tightening (2^0 → 2^-3) was tried, shrank ratios
  ~8x, cannot reach the band (the wall is relative-ULP vs fixed atol at
  baseline-zero elements); REVERTED to the verified bounds.

## M9 (new, consolidates three findings): cancellation-amplified grading

For three ops, a correct-but-not-bit-identical implementation exceeds the
1% band at cancellation elements, so candidate correctness is ungradeable
under current tolerances regardless of which side is "right":
- **gpt_oss_moe**: reference provably CLOSER to fp64 truth than the
  baseline at the offenders (receipt: C stream ADDENDUM).
- **chunk_gla o-path**: kernel and reference equidistant from fp64 truth
  (kernel uses TF32 intra-chunk per FLA's own comment); stored-state
  substitution control excludes state flips.
- **vision_block**: ten pure-torch attention variants all sit at the same
  2-ULP kernel deviation (UNDER band at the kernel), amplified 6x by
  residual cancellation to 3.08 at the output.
All three stay packaging-skipped with proofs; identity (baseline vs
itself) is unaffected. Mentor decision: per-op ratio allowance (~2.5-3x),
fp64-oracle band, or keep excluded. We recommend keeping them excluded
from agent scoring until decided — a relaxed band admits real 2-3% errors.
