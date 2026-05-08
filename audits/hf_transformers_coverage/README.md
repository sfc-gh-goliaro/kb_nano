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

- **Strict** (L4 + composable, kb-nano kernel exists for every compute class): **264/447 = 59.1%**
- **Loose** (+ partial / torch fallback): **435/447 = 97.3%**
- **Unsupported** (genuinely needs custom CUDA / external lib): **12/447 = 2.7%**

v12 vs v11: 6 folders demoted composable → partial for partial-rotary
consistency (bamba, glm4v_moe, laguna, musicflamingo, recurrent_gemma,
solar_open). See REAUDIT_NOTES.md "v12 RECONCILIATION" for the full delta.

The 12 unsupported (canonical):
`diffllama`, `dinat`, `fast_vlm`, `gemma3n`, `ibert`, `layoutlmv2`, `mra`,
`rwkv`, `timm_backbone`, `timm_wrapper`, `xlstm`, `yoso`.

## What you should read

| file | purpose |
|---|---|
| [`REAUDIT_NOTES.md`](REAUDIT_NOTES.md) | Full reconciliation, methodology, per-version history, and self-assessment. **Start here.** v11 final state is in the section labelled "v11 RECONCILIATION". |
| [`NUMBER_DRIFT_RECONCILIATION.md`](NUMBER_DRIFT_RECONCILIATION.md) | Why the denominator was 421 / 425 / 442 / 445 / 447 / 448 over time, and why **447** is canonical. |
| [`VERIFIER_AUDIT.md`](VERIFIER_AUDIT.md) | Per-slice agent agreement audit + hallucination spot-checks (0/10 hallucinated file:line refs). |
| [`CONSISTENCY_AUDIT.md`](CONSISTENCY_AUDIT.md) | Phase-1 cross-pattern consistency groups (ALiBi, partial-rotary, AutoBackbone, etc.) — how identical patterns were forced to identical statuses. |
| [`audit_evidence.csv`](audit_evidence.csv) | Per-folder evidence trail (447 rows × 12 columns: shard verdict, cross-verifier verdict, phase-2 verdict, HF file:line, "I personally read" flag). |
| [`hf_coverage_rows.tex`](hf_coverage_rows.tex) | Paper-input LaTeX rows (447 entries). |
| [`MENTOR_REVIEW_full_audit.tex`](MENTOR_REVIEW_full_audit.tex) | Full standalone tex for review. |
| [`_reaudit_final_v11.json`](_reaudit_final_v11.json) | Machine-readable {folder: status} dict (447 entries). |

`tools/manual_audit_shard_{01..17}.md` are the per-shard markdown audit notes
that the renderer (`tools/md_to_tex.py`) reads to produce the tex.

`_stale_pre_reaudit/` holds the original pre-reaudit artifacts kept for
provenance (do not use; superseded by the v11 files above).

## Pinning

- **HF source:** `huggingface/transformers` @ commit
  `da6c53e431f7c9ef0691239d4ce89b0f711ecad7`. Cloned (shallow) to
  `/tmp/hf_transformers_pinned/`.
- **kb-nano support surface:** branch `audit/hf-transformers-coverage` cut from
  `origin/experiments` @ commit `11aa838`.

## Reproducing

```bash
# 1. Clone HF at the pinned commit
cd /tmp && rm -rf hf_transformers_pinned && mkdir hf_transformers_pinned && cd hf_transformers_pinned
git init -q && git remote add origin https://github.com/huggingface/transformers.git
git fetch --depth 1 origin da6c53e431f7c9ef0691239d4ce89b0f711ecad7
git checkout -q FETCH_HEAD

# 2. Re-render the paper-appendix tex from the markdown shards
cd /home/olu/kb_nano
python audits/hf_transformers_coverage/tools/md_to_tex.py
# -> writes hf_coverage_rows.tex
```

## Audit work

| pass | folders touched |
|---|---:|
| First-pass (16 parallel subagents) | 425 |
| Cross-verify round 1 (5 verifiers, partial/unsupported priority) | 239 |
| Phase 2 verifiers (slices 1–6, 8 — full source-read of edge cases) | 222 |
| Slice 7 (recovered 20 folders missed by sharding bug) | 20 |
| v11 additions (esmfold, donut_swin) | 2 |
| **Personally source-read by coordinator (cumulative)** | **107** |

Total folder-touches across all rounds: ~1010 (with significant overlap; many
folders were touched 2–4 times across rounds for cross-pattern consistency).

## Verification gates passed (v11)

- 447 audit rows = filesystem ground truth (448 modeling files − `auto/` non-model)
- `audit_evidence.csv` rows = `_reaudit_final_v11.json` entries (0 mismatches across 447)
- `hf_coverage_rows.tex` rows = `_reaudit_final_v11.json` entries (0 mismatches across 447)
- Status-marker mapping in tex matches json (27 / 237 / 171 / 12)
- Every kb-nano `L1/L2/L3/L4/<file>.py` referenced in the tex exists on disk (202 unique refs, 0 missing)
- Every folder in slice 7 + v11 (22 additions) exists in the HF clone
- Every L4 promotion (27) has a matching `tasks/baseline/L4/<file>.py` whose docstring header targets the corresponding HF folder
- Every `unsupported` (12) was source-verified by the coordinator: 5 use `kernels-community` CUDA kernels, 4 use timm/detectron2/natten external libs, 3 use bespoke compute (differential attn, integer arith, mLSTM)

## Status / Known limitations

- **Static audit only.** This audits compute primitives by reading source. It
  does not certify byte-correctness against HF, nor does it measure
  performance. A `composable` model may still run slowly via torch fallback
  until each L1/L2 op gets a tuned kernel.
- **`partial` is "decomposable, not yet wrapped".** A partial folder can run
  in PyTorch (the relevant op is in `torch.*`); kb-nano just doesn't have a
  dedicated kernel for that variant yet (e.g., partial-rotary RoPE,
  interleaved-RoPE, ALiBi slopes, Conformer rel_shift).
- **Multi-modeling folders** (`blip`, `data2vec`, `donut`, `esm`, `maskformer`,
  `rt_detr`) are audited per-modeling-file. Each PyTorch `modeling_*.py` gets
  its own row. This is why 442 folders → 447 rows.
- **Paper text vs this audit.** The submitted paper cites 421 / 96.2% / 7
  unsupported. Those numbers are not reproducible from the table that ships
  with the paper — see `NUMBER_DRIFT_RECONCILIATION.md` for the full story
  and v11 corrections. The `_hf_coverage_rows_pre_reaudit_*.tex` file is
  preserved as the original-state backup.
