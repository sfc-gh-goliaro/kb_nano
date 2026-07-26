# Open items

Everything previously listed here was decided by the author and executed
on this branch — full decision record and rationale live in the git
history of this file (see `git log --follow docs/agent_eval/OPEN_DECISIONS.md`,
esp. commits 3478fb6, 22a55c2, 2e8f59c) and the commit messages carrying
per-change verification tallies. Only genuinely open items remain below — for the team / whoever runs and merges this.

## OPEN-1 — Route the upstream Triton bug (M8)

`triton_kernels`' `matmul_ogs` ragged-TMA path performs an out-of-range
read on every launch on SM100/B200 (compute-sanitizer receipt in the
pilot's stream-A report); it crashes only when the read lands on unmapped
memory, so failures look order-dependent. Our grader mitigates with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` scoped to the four
affected ops. Worth filing upstream (triton 3.6.0 / triton_kernels) with
the receipts, and re-checking on H200 — SM90 uses a different code path
and may be unaffected.

## OPEN-2 — Rebase onto the newest release commit

This branch is based on release `28cf517`. A newer release commit
(`a863ded`, "Qwen3 h200 fix") changes two baselines we sit downstream of:
`moe_grouped_gemm`'s kernel-config heuristic (new block-wise FP8 branch)
and `moe_align`. Our repaired references mirror the OLD heuristic, so
after rebasing, re-grade `moe_grouped_gemm` and `fused_experts`
(reference-as-candidate) and `moe_align` (identity + the canonicalization
control) and update the mirrors if the selected config changed. Nothing
else in that commit touches our files.

## OPEN-3 — Merge-PR review

The branch carries author-decided changes that alter benchmark semantics
and therefore deserve reviewer attention at merge time: the Lane-B fixture rebuild
(real dims replacing fabricated ones), the op→class pins, two scoped
comparison differences (moe_align canonicalization; chunk_gla computation-
precision grading), the fp64-oracle dual gate for three cancellation-
amplified ops, the repaired+completed tasks/reference corpus, and the
multi-rank allreduce harness. Each commit message states what was
verified and how.
