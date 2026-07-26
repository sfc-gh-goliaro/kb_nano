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

### E4. moe_align output canonicalization (was D2)
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
- **M2 (was D6)**: allreduce — does the paper's 4-rank NCCL+Gloo harness
  source still exist? (Not in any branch; only the Table-1 caption and
  §6.2 reference it.) If lost: rebuild spec is in OP_CENSUS/D6 history —
  torchrun 4 ranks, broadcast inputs, rank-0 compare + collective timing.
- **M3**: 6 composite ops have NO reference file (flux_transformer_block,
  oasis_dit, oasis_vae_attention_block, vision_block, yolov10_head/neck) —
  authoring work; priority call.
- **M4**: merge-to-main ratification of E2 (resolution semantics), E3
  (reference edits), E4 (comparison semantics).
- **M5**: ASTRA — verify GPT-5.5 model id via /v1/models; run
  astra_live_smoke.sh (dual-mode; OpenAI mode documented in header).
  AK campaigns: Anthropic API key billing (decided).
- **M6**: optional experiment knobs, disclose if used: baseline-seeded AK
  arm (only for ops with portable baselines); showing baseline source to
  the agent as reading material (codex-style).

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
