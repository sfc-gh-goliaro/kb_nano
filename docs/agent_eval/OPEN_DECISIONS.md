# Decision record + remaining items

Most items were DECIDED by the author (olu, 2026-07-26) — recorded here so
the mentor sees the state, not a to-do list. Genuinely-open items are
marked OPEN.

Nothing here blocks running campaigns (see H200_RUNBOOK.md). These are
ratifications and policy calls. Evidence for every claim lives in the
commit messages on this branch and the pilot's stream reports (pointer at
the bottom) — reading it is optional.

## Decided by the author (ride into the merge PR)

- **R1 (DECIDED: adopt).** linear / embedding / conv2d /
  parallel_linear scenarios were rebuilt from real model call sites
  (meta-device enumeration). The previously published per-op numbers for
  these four ops used fabricated dimensions (square Linears, 128-dim
  embeddings, 1x1 convs); the revision should disclose the fixture
  correction and that those rows are not comparable.
- **R2 (DECIDED: keep).** The registry now names the benchmarked class
  per op (`class_name`), replacing the fragile last-class-in-file rule.
  Four published-era ops resolved to the wrong class (rotary_emb,
  tensor_ops, softmax, flux_transformer_block — the latter two silently);
  their published rows measured the wrong class.
- **R3 (DECIDED: ratified)** — (each scoped to one op,
  documented in the grader's docstring as Differences 8/9): moe_align
  output canonicalization (order within an expert group is not
  contractual — grounded in the consumer); chunk_gla graded at
  computation precision (the baseline itself is ~10x outside the fp32
  band vs fp64 truth).
- **R4 (DECIDED: adopt).** 13 files repaired, 6 L3
  references authored; the corpus is now fully executed and verified
  (nothing ever ran these files before — reference paths were
  display-only in the original pipeline).

## Policy calls

- **M9 (DECIDED: fp64-oracle dual gate; exclusion rejected by the author).**
  For gpt_oss_moe, chunk_gla (output path), and vision_block, a
  correct-but-not-bit-identical implementation exceeds the 1% band at
  cancellation elements. Codex receipts show the band is passable by
  kernel-language candidates (its chunk_gla/gpt_oss_moe candidates were
  Triton, same tolerance constants) — only naive pure-torch mirrors hit
  the wall. Policy: scoped to these ops, a candidate passes if within the
  standard band of the baseline OR elementwise no farther from a
  harness-computed fp64 oracle than the baseline itself (+ low-precision
  margin). Never fails a candidate at-least-as-accurate as the baseline;
  never passes semantic wrongness (controls enforced); baseline-band
  unchanged for comparability; restores all three ops to the task menu.
- **M5 (DECIDED: ASTRA DROPPED from scope).** Running it as published adds
  no benchmark row; the reviewer response cites the source-level receipts
  instead (its candidate format is one self-contained CUDA function per
  run — composite production modules are not expressible). The Claude
  patch + smoke script remain in-tree as inert artifacts. AK campaigns
  bill an Anthropic API key (decided).
- **M6 (DECIDED: defaults — none used).** baseline-seeded
  AK arm; showing baseline source to the agent codex-style.

## Found in passing (yours to route)

- **M7 (IN EXECUTION: author directed fix-if-certain; investigation running)** — `TRTLLMPrefill.forward` silently drops sinks/window kwargs on
  the no-block-table path (tasks/baseline/L1/flashinfer_prefill.py:26-55):
  symmetric, so grading is unaffected, but those semantics are untested at
  Tier-1. Fix = baseline edit.
- **M8** — upstream triton_kernels `matmul_ogs` ragged-TMA out-of-range
  read on SM100 (compute-sanitizer receipt in the stream A report). The
  grader mitigates with `expandable_segments:True` scoped to 4 ops;
  worth reporting upstream and re-checking on H200 (SM90 = different path).
- **M2 (CLOSED: our verified harness supersedes; original source moot)** — allreduce is now graded by our own multi-rank harness
  with analytic ground truth. If the paper's original 4-rank harness
  source still exists anywhere, running it would make numbers directly
  comparable to the published 0.84x; otherwise ours (at `--nranks 4`)
  is the successor.

## Census residuals (final; none block anything)

- llama, gpt_oss, qwen3_vl: identity needs two whole-model copies on one
  GPU — physically impossible; correctness lives in the e2e tier (by
  design).
- oasis_rollout: pure orchestration (forward takes whole sub-models);
  closed/attempted per the pilot's final state — see OP_CENSUS.md current
  state for the last word.
- vision_block / chunk_gla / gpt_oss_moe tasks: see M9.

## Evidence pointers (optional reading)

Branch commit messages carry per-change verification tallies. Deep
receipts: the pilot machine's scratch (`streams/{A..F}/REPORT.md`,
census/packaging JSONs) and docs/agent_eval/VALIDATION_REPORT.md +
OP_CENSUS.md (kept as the audit trail; the top of OP_CENSUS states the
final scoreboard).
