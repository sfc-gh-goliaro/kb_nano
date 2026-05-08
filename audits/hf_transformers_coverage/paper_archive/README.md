# Paper-version archive

The submitted-paper rendering of the HF coverage table, frozen as it was at
submission time. Used to reproduce the paper's headline number 409/425 = 96.24%.

| file | what it is |
|---|---|
| `hf_coverage_rows_paper_v1.tex` | The paper's appendix table — 425 rows: 22 L4 (`$\bullet$`) + 387 composable (`\cmark`) + 9 partial (`\textbf{P}`) + 7 unsupported (`\xmark`). Strict (L4 + composable) = **409/425 = 96.24%**, which matches the paper text. |

## Reproducing the paper number

```bash
cd audits/hf_transformers_coverage
grep -c '\\xmark' paper_archive/hf_coverage_rows_paper_v1.tex   # 7  unsupported
grep -c '\\$\\bullet\\$' paper_archive/hf_coverage_rows_paper_v1.tex   # 22 L4
grep -c '\\cmark' paper_archive/hf_coverage_rows_paper_v1.tex   # 387 composable
grep -c '\\textbf{P}' paper_archive/hf_coverage_rows_paper_v1.tex   # 9  partial
# Total = 425; strict = 22 + 387 = 409; 409/425 = 96.24%
```

## Why this is no longer canonical

The post-submission re-audit (`../intermediate/REAUDIT_NOTES.md`) found
two independent issues with the paper version:

1. **20 folders were dropped** from the original audit due to a sharding
   bug (e.g., `mask2former`, `mistral4`, `lw_detr`, `granite4_vision`).
   Recovered in v9 / v10 / v11.
2. **The classification was too lenient** — many folders that route
   through external libraries (timm, detectron2, kernels-community/*) or use
   bespoke compute (differential attention, integer-arithmetic quant,
   triangle attention) were marked composable but should be partial /
   unsupported. Tightened across v4 → v7 → v10 → v11 → v12.

The current canonical numbers (v12) are 27 L4 / 237 composable / 171 partial
/ 12 unsupported = 264/447 strict (59.06%), 435/447 loose (97.32%). See
`../README.md` and `../CAVEATS_AND_METHODOLOGY.md` for the v12 rationale and
per-pattern reasoning.
