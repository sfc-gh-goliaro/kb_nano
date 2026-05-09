# External verification agent outputs

Outputs from independent verification agents the user ran on 2026-05-09
against the prior session's audit. Two distinct passes:

| pass | files | scope |
|---|---|---|
| First pass — "independent re-audit" | `independent_reaudit_report.md`, `independent_reaudit_verification_log.json`, `independent_reaudit_self_review_addendum.md` | Targeted re-verification of the prior session's claims, with the self-review addendum a follow-up after the user pushed back |
| Second pass — "paper-quality re-audit" | `paper_quality_reaudit_methodology.md`, `paper_quality_reaudit_source_manifest.json` (447 rows), `paper_quality_parent_review_working.md`, `paper_quality_reaudit_working_report.md`, `paper_quality_reaudit_evidence_skeleton.json` | A fresh from-scratch pass with its own R1–R8 ambiguity rules. Per the working report, it was in-progress at write time (only methodology, source manifest, parent-review notes, and skeleton were materialized — full per-folder evidence ledger and final report were still pending). |

Originally written under `/tmp/` by the verification agents; moved here so
they survive `/tmp/` being a shared, ephemeral workspace.

These files are **input** to any follow-up reconciliation; they have not
yet been merged into the canonical audit. Compare against:
- `../REAUDIT_NOTES.md` — prior session's per-version reconciliation
- `../../CAVEATS_AND_METHODOLOGY.md` — current canonical decisions
- `../../audit_evidence.csv` — per-folder evidence trail
