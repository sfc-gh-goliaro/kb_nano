# Open items (mentor)

Everything previously listed here was decided by the author and executed
on this branch — full decision record and rationale live in the git
history of this file (see `git log --follow docs/agent_eval/OPEN_DECISIONS.md`,
esp. commits 3478fb6, 22a55c2, 2e8f59c) and the commit messages carrying
per-change verification tallies. Only genuinely open items remain below.

## OPEN-1 — Route the upstream Triton bug (M8)

`triton_kernels`' `matmul_ogs` ragged-TMA path performs an out-of-range
read on every launch on SM100/B200 (compute-sanitizer receipt in the
pilot's stream-A report); it crashes only when the read lands on unmapped
memory, so failures look order-dependent. Our grader mitigates with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` scoped to the four
affected ops. Worth filing upstream (triton 3.6.0 / triton_kernels) with
the receipts, and re-checking on H200 — SM90 uses a different code path
and may be unaffected.

## OPEN-2 — M7 outcome (pending in-flight investigation)

`TRTLLMPrefill.forward` silently drops the `s_aux` (attention sinks) and
`window_size` kwargs on its no-block-table path
(tasks/baseline/L1/flashinfer_prefill.py:26-55). Symmetric for baseline
and candidate, so grading verdicts are unaffected — but those semantics
are untested at Tier-1. The author directed: fix only if confirmed a bug
with certainty; the investigation is running. If a fix lands it is a
baseline-file edit — review that hunk specifically in the merge PR.

## OPEN-3 — Merge-PR review

The branch carries author-decided changes that alter benchmark semantics
and therefore deserve your eyes at merge time: the Lane-B fixture rebuild
(real dims replacing fabricated ones), the op→class pins, two scoped
comparison differences (moe_align canonicalization; chunk_gla computation-
precision grading), the fp64-oracle dual gate for three cancellation-
amplified ops, the repaired+completed tasks/reference corpus, and the
multi-rank allreduce harness. Each commit message states what was
verified and how.
