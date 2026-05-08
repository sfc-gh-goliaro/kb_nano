# Intermediate process docs

These three files document audit *process* and *historical reconciliation*
rather than final decisions. They are preserved here for methodology defense
(showing how cross-agent disagreements were resolved, how individual agent
claims were spot-checked, and how the audit moved through versions v4 → v7
→ v10 → v11 → v12). **The final canonical state lives at the parent-directory
level** in `CAVEATS_AND_METHODOLOGY.md`.

| file | role |
|---|---|
| [`REAUDIT_NOTES.md`](REAUDIT_NOTES.md) | Full reconciliation chain across all 5 audit versions. Contains the v4 / v7 / v10 / v11 / v12 RECONCILIATION sections side-by-side, plus per-pass coverage tables and round-by-round summaries. The final v12 RECONCILIATION section at the bottom is the authoritative version-bump record; everything earlier is intermediate state. The L4 promotion table and 12-unsupported per-folder rationale have been promoted into `../CAVEATS_AND_METHODOLOGY.md` §7-8 so this doc is no longer load-bearing for those decisions. |
| [`CONSISTENCY_AUDIT.md`](CONSISTENCY_AUDIT.md) | Phase-1 cross-agent disagreement resolution. For each pattern (ALiBi, partial-rotary, AutoBackbone, …) the table shows first-pass vs cross-verifier verdicts side-by-side and the chosen tie-breaker. Final = v4 era; subsequently superseded by v7 → v10 → v11 → v12. |
| [`VERIFIER_AUDIT.md`](VERIFIER_AUDIT.md) | Phase-2 verifier audit — for each verifier subagent's slice, the coordinator personally read source for a sample of folders and recorded agreement / overrides. Includes the evidence-claim hallucination check (0/10 hallucinated `file:line` refs across a random 10-row spot-check). |

If you only want the final v12 numbers + critical decisions, you do not need
these files. Use the top-level `README.md`, `CAVEATS_AND_METHODOLOGY.md`,
and `NUMBER_DRIFT_RECONCILIATION.md` instead.
