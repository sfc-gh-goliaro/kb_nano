# Intermediate process docs

These two files document audit *process* rather than final decisions. They are
preserved here for methodology defense (showing how cross-agent
disagreements were resolved and how individual agent claims were
spot-checked) but **the final canonical state lives at the parent-directory
level** in `REAUDIT_NOTES.md` and `CAVEATS_AND_METHODOLOGY.md`.

| file | role |
|---|---|
| [`CONSISTENCY_AUDIT.md`](CONSISTENCY_AUDIT.md) | Phase-1 cross-agent disagreement resolution. For each pattern (ALiBi, partial-rotary, AutoBackbone, …) the table shows first-pass vs cross-verifier verdicts side-by-side and the chosen tie-breaker. Final = v4 era; subsequently superseded by v7 → v10 → v11 → v12. |
| [`VERIFIER_AUDIT.md`](VERIFIER_AUDIT.md) | Phase-2 verifier audit — for each verifier subagent's slice, I personally read source for a sample of folders and recorded agreement / overrides. Includes the evidence-claim hallucination check (0/10 hallucinated `file:line` refs across a random 10-row spot-check). |

If you only want the final v12 numbers + critical decisions, you do not need
these files. Use the top-level `REAUDIT_NOTES.md` (v12 RECONCILIATION
section) and `CAVEATS_AND_METHODOLOGY.md` instead.
