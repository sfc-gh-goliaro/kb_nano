# Open decisions — agent-eval pilot (branch `agent-eval-pilot`)

One entry per decision that is NOT ours to finalize (mentor/team ratifies) or
that we deliberately deferred. Everything else in this pilot is done and
verified — see VALIDATION_REPORT.md (evidence), OP_CENSUS.md (per-op status),
H200_RUNBOOK.md (setup + reproduce).

Plain-language glossary used below:
- **shape registry** (`bench/kernels/benchmark_scenarios/small/shape_registry.yaml`):
  the benchmark's fixture catalog. Each operator has *scenarios*; a scenario =
  constructor arguments (`init_args`, i.e. how to BUILD the module) + input
  specs (what to FEED it).
- **identity check**: build the reference module twice from a scenario, feed
  identical inputs, compare. If a module can't pass against *itself*, the
  scenario can't grade an agent kernel.
- **Lane A / Lane B** (names from the population pass): the original trace
  only captured scalar constructor args, so 61 ops had incomplete `init_args`.
  *Lane A* = ops whose missing values are readable off a traced checkpoint's
  config.json (e.g. llama_mlp's intermediate_size) — filled statically,
  no judgment calls. *Lane B* = dimension-generic ops (linear, embedding,
  conv2d, parallel_linear) where the registry never stored output dims AND
  one traced record bundled several distinct layers with identical input
  shapes — unrecoverable from the record, so scenarios were REBUILT by
  walking the real models' module trees (meta device, no weights) and
  enumerating actual (in, out) call sites.

## D1. Lane B fixture rebuild — ratify for merge

What changed: linear 66→177, parallel_linear 43→45, embedding 13→40,
conv2d 39→59 scenarios; old bundled records retired. Why: the
experiments-codex runner *fabricated* these dims (square Linears, 128-dim
Embeddings, 1x1 Convs), so the published per-op numbers for these four ops
used non-production shapes. Our rebuild = the real call-site dims.
Consequence to state in the revision: per-op numbers for these ops are not
comparable to the published table (the published fixture was fictional).
Receipts: OP_CENSUS.md census-v2 section + codex cross-check paragraph;
proposals + verify JSONs in the pilot scratch.

## D2. moe_align: adopt codex's output canonicalization?

`moe_align` outputs a token permutation; several orderings are equally
valid, so element-wise comparison fails correct outputs (our census: 2/10).
The codex runner passed 10/10 by canonicalizing (sorting) both sides before
comparing (`_canonicalize_output_for_target`, experiments-codex
runner.py:955). Adopting changes the definition of "equal" for this op —
surfaced rather than silently adopted. Recommendation: adopt, scoped to
this op, with a comment citing the contract.

## D3. Two seed-kernel patches — upstream review

Two packaged task seeds needed small patches to run under the grader
(details in the dataset's PACKAGING_REPORT.md). Grader-verified, not yet
reviewed by a kb maintainer. Ratify or replace the seeds.

## D4. rotary_emb / tensor_ops: op→class resolution

The registry op name resolves (by discovery order) to a class added later,
not the one that was traced. Our census refuses to grade the mismatch.
NOTE: the codex CSV records PASS for both with bitwise-zero error — the
signature of comparing a class against itself — so those published rows
likely evaluated the wrong class. Decide which class each op means, then
either fix discovery or rename the ops.

## D5. mxfp4_moe: trusted-preprocessing lane

The reference needs real-checkpoint quantized weights *prepared by the
implementation under test* — the grader would trust candidate code for its
own fixture. Codex accepted that; we rejected it by design. Options: build
a trusted one-time preprocessing step (weights prepared once by the
reference, frozen as fixture data), or leave the op out of agent scoring.

## D6. allreduce: distributed harness or documented skip

Multi-GPU communication op; the single-process grader has no process
group (15/15 RUNTIME_ERROR by construction). Reconciling the record
(checked 2026-07-26):
- The committed codex CSV run did NOT produce the paper's number: its row
  is SKIPPED ("requires a working distributed/NCCL process group;
  singleton NCCL failed... CUDA driver/runtime mismatch" — an env
  breakage on that run's machine, not a hardware limit).
- The paper's allreduce 0.84x comes from what its Table 1 caption calls
  "a separate 4-rank NCCL+Gloo harness" (also referenced in §6.2, where
  it catches a KernelAgent Triton kernel whose staged single-process
  checker had reduced all-reduce to identity).
- That harness's source is in NO branch we have (searched
  experiments-codex/release/main for gloo/torchrun/nproc; only hit is
  the TP engine's own `dist.new_group(backend="gloo")` in
  infra/engine.py — reusable infra, not a bench harness). The paper
  number is disclosed but not reproducible from committed artifacts.

So: running allreduce IS possible on any >=4-GPU box with working NCCL
(incl. our B200 node and the H200 cluster); what's missing is harness
code. ASK THE MENTOR whether the 4-rank harness source still exists
privately; if not, options: rebuild it (fully specified, no discovery
needed: torchrun 4 ranks, broadcast inputs, rank-0 compare + collective
timing, reusing the engine's process-group setup), or keep the op out of
agent campaigns with the paper's own disclosure.

## D7. Decoder L3 wiring (llama_decoder, gpt_oss_decoder, qwen3_moe_decoder)

Identity fails on harness plumbing, not configs: rotary-embedding forward
argument, decode-path KV-cache API, fp8-inside-module. Recipes in
drill2_merged.json (pilot scratch) + OP_CENSUS residuals. Recipes are
written; remaining work is implementation + identity verification, no
unknowns identified. Unlocks 3 high-value composite ops. Do, or accept
the coverage gap.

## D8. oasis_block + oasis spatial/temporal attention: module-arg builders

Their constructors take a *module* (a rotary-embedding object), which
registry data can't express. Codex built these objects inline in its
runner. Our equivalent builder recipe is documented (OP_CENSUS residuals);
adopting it is a small entrypoint change. Recommendation: adopt.

## D9. 12 defective tasks/reference implementations

18 ops are not packaged as agent tasks; 12 trace to broken reference
implementations in `tasks/reference/` (per-file reasons in the dataset's
PACKAGING_REPORT.md). Fixing them unlocks the tasks AND is a
paper-relevant finding on its own. Decide: fix now vs disclose.

## D10. ASTRA scope

Decided (2026-07-26): run ASTRA as published on its own kernels, second
row with GPT-5.5 (see runbook); do NOT port to kb. Reopen only if the
team wants the extra benchmark row enough to fund an L1-subset fork
(scope: _import_callable fix + per-op test scaffolding + naive .cu seeds;
single-.cu-function candidate format is the wall; reasoning in the
runbook's ASTRA section).

## D11. Model IDs to verify at run time (cannot be verified from here)

- ASTRA/GPT-5.5: exact model id via OpenAI `/v1/models` before the run.
- AK/Claude: `claude --version && claude -p "Reply OK"` preflight; API-key
  billing decided (runbook preflight).
