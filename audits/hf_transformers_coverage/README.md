# HF Transformers × kb-nano Coverage Audit

This directory contains a paper-appendix-quality audit of how broadly Hugging
Face Transformers architectures can be supported by kb-nano's existing
L1/L2/L3/L4 operator surface, plus the post-submission re-audit that fixes
several errors in the original paper version.

## Headline result (v12 final, 447 folders)

| status | count | %     |
|--------|------:|------:|
| `kb_nano_l4` | 27 | 6.0% |
| `composable` | 237 | 53.0% |
| `partial`    | 171 | 38.3% |
| `unsupported`| 12  | 2.7% |

- **Strict** (L4 + composable, kb-nano kernel exists for every compute class): **264/447 = 59.06%**
- **Loose** (+ partial / torch fallback): **435/447 = 97.32%**
- **Unsupported** (genuinely needs custom CUDA / external lib): **12/447 = 2.68%**

The 12 unsupported (canonical):
`diffllama`, `dinat`, `fast_vlm`, `gemma3n`, `ibert`, `layoutlmv2`, `mra`,
`rwkv`, `timm_backbone`, `timm_wrapper`, `xlstm`, `yoso`.

## What you should read

### One doc has everything: `CAVEATS_AND_METHODOLOGY.md`

| section | content |
|---|---|
| §1 | Methodology: 4-status definitions, source pinning, audit pipeline, what was strict vs lenient, worked example |
| §2 | The 4 denominators (and why we use 447) |
| §3 | Post-v11 fixes — 11 file-mapping corrections + the 6 v12 partial-rotary demotions, with config:line evidence + the "why kb-nano L1/rotary_emb.py does not cover partial-rotary directly" deep-dive |
| §4 | 8 known limitations (read before citing) |
| §5–6 | Reproducibility command + trust footprint |
| §7 | **The 27 L4 promotions** — per-folder rationale (HF folder ↔ kb-nano L4 file) + 4 calls intentionally NOT promoted (qwen2_5_vl, falcon_mamba, gemma4_assistant, sam2_video) |
| §8 | **The 12 unsupported folders** — per-folder rationale with HF source line evidence |
| §9 | **Cross-pattern judgment calls (the ambiguous decisions)** — every recurring partial-vs-composable / composable-vs-unsupported call, with the consistency rule applied |
| §10 | **Per-folder partial gap rationale (all 171)** — every partial folder grouped by gap pattern (partial-rotary, AutoBackbone, T5 cross-attn, Conformer rel_shift, sliding-window, MoE bespoke routing, etc.) with a one-line specific reason for each folder |

### Companion docs

| file | purpose |
|---|---|
| [`CAVEATS_AND_METHODOLOGY.md`](CAVEATS_AND_METHODOLOGY.md) | All critical decisions in one place. **Start here.** |
| [`NUMBER_DRIFT_RECONCILIATION.md`](NUMBER_DRIFT_RECONCILIATION.md) | Why the denominator was 421 / 425 / 442 / 445 / 447 / 448 over time, and why **447** is canonical. |

### Data files (reproduce the numbers)

| file | purpose |
|---|---|
| [`audit_evidence.csv`](audit_evidence.csv) | Per-folder evidence trail (447 rows × 12 columns: shard verdict, cross-verifier verdict, phase-2 verdict, HF file:line, "I personally read" flag, p2_rationale). |
| [`hf_coverage_rows.tex`](hf_coverage_rows.tex) | Current paper-input LaTeX rows (447 entries, v12). |
| [`_reaudit_final_v11.json`](_reaudit_final_v11.json) | Machine-readable {folder: status} dict (447 entries; contains v12 final state). |
| [`paper_archive/hf_coverage_rows_paper_v1.tex`](paper_archive/hf_coverage_rows_paper_v1.tex) | Paper's original 425-row table, frozen — reproduces 409/425 = 96.24%. |

### Renderer + canonical shards

`tools/md_to_tex.py` parses `tools/manual_audit_shard_{01..17}.md` and writes
`hf_coverage_rows.tex`. The 17 shard markdowns are the per-folder per-class
breakdowns that are the source of truth for the rendered tex.

### Optional historical / process docs

- [`paper_archive/`](paper_archive/) — frozen paper-version table + a README that walks through reproducing the paper's 409/425.
- [`intermediate/`](intermediate/) — full v4 → v12 reconciliation chain (`REAUDIT_NOTES.md`), Phase-1 cross-agent disagreement log (`CONSISTENCY_AUDIT.md`), Phase-2 hallucination spot-check (`VERIFIER_AUDIT.md`). **Not needed for current numbers**; preserved for methodology defense.

## Pinning

- **HF source:** `huggingface/transformers` @ commit
  `da6c53e431f7c9ef0691239d4ce89b0f711ecad7`. Cloned (shallow) to
  `/tmp/hf_transformers_pinned/`.
- **kb-nano support surface:** branch `audit/hf-transformers-coverage` cut from
  `origin/experiments` @ commit `11aa838`.

## Reproducing the numbers

### Paper (409/425 = 96.24%)

```bash
cd audits/hf_transformers_coverage
grep -c '\$\\bullet\$' paper_archive/hf_coverage_rows_paper_v1.tex   # 22 L4
grep -c '\\cmark'      paper_archive/hf_coverage_rows_paper_v1.tex   # 387 composable
grep -c '\\textbf{P}'  paper_archive/hf_coverage_rows_paper_v1.tex   # 9 partial
grep -c '\\xmark'      paper_archive/hf_coverage_rows_paper_v1.tex   # 7 unsupported
# Total = 425; strict = 22 + 387 = 409 → 409/425 = 96.24%
```

### Current v12 (264/447 = 59.06%)

```bash
cd audits/hf_transformers_coverage
grep -c '\$\\bullet\$' hf_coverage_rows.tex   # 27 L4
grep -c '\\cmark'      hf_coverage_rows.tex   # 237 composable
grep -c '\\textbf{P}'  hf_coverage_rows.tex   # 171 partial
grep -c '\\xmark'      hf_coverage_rows.tex   # 12 unsupported
# Total = 447; strict = 27 + 237 = 264 → 264/447 = 59.06%
```

### Re-render v12 from the shards

```bash
# 1. Clone HF at the pinned commit (only needed if you want to re-verify HF source refs)
cd /tmp && rm -rf hf_transformers_pinned && mkdir hf_transformers_pinned && cd hf_transformers_pinned
git init -q && git remote add origin https://github.com/huggingface/transformers.git
git fetch --depth 1 origin da6c53e431f7c9ef0691239d4ce89b0f711ecad7
git checkout -q FETCH_HEAD

# 2. Re-render the v12 tex from the markdown shards
cd /home/olu/kb_nano
python audits/hf_transformers_coverage/tools/md_to_tex.py
# -> writes audits/hf_transformers_coverage/hf_coverage_rows.tex
```
