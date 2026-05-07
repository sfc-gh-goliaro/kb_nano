# HF Transformers × kb-nano Coverage Audit

This directory contains a paper-appendix-quality audit of how broadly Hugging Face Transformers architectures can be supported by kb-nano's existing L1/L2/L3 operator surface.

## What you should read

| file | purpose |
|---|---|
| [`audit_methodology.md`](audit_methodology.md) | The full methodology, schema, denominators, conservatism rules, and reproducibility steps. **Start here.** |
| [`coverage_summary.md`](coverage_summary.md) | The headline result: % of HF architectures covered, broken down by modality, with the top missing primitives. |
| [`hf_architecture_operator_coverage.csv`](hf_architecture_operator_coverage.csv) | One row per HF modeling file, with status, mapped kb-nano ops, and `file:line` evidence. The full data behind the appendix table. |
| [`unsupported_operator_summary.csv`](unsupported_operator_summary.csv) | Frequency table of canonical ops in `partial`/`unsupported` rows — i.e., the kernels that, if added, would unlock the most architectures. |
| [`hf_model_inventory.csv`](hf_model_inventory.csv) | The HF model surface: 465 folders, with PyTorch-modeling counts, modular DSL flags, multi-modeling flags, and no-modeling flags. |
| [`kb_nano_operator_catalog.csv`](kb_nano_operator_catalog.csv) | The kb-nano support surface (origin/experiments + audit-branch L1/L2 additions): 402 class-level rows from L1/L2/L3 (114 L1 + 182 L2 + 106 L3). |

## Pinning

- **HF source:** `huggingface/transformers` @ commit `da6c53e431f7c9ef0691239d4ce89b0f711ecad7`. Cloned (shallow) to `/tmp/hf_transformers_pinned/` for the audit run.
- **kb-nano support surface:** branch `audit/hf-transformers-coverage` cut from `origin/experiments` @ commit `11aa838 add manual values for missing ops`.

## Reproducing

```bash
# 1. Clone HF at the pinned commit (audit run did this to /tmp; pick any path you have ~150 MB free)
cd /tmp && rm -rf hf_transformers_pinned && mkdir hf_transformers_pinned && cd hf_transformers_pinned
git init -q && git remote add origin https://github.com/huggingface/transformers.git
git fetch --depth 1 origin da6c53e431f7c9ef0691239d4ce89b0f711ecad7
git checkout -q FETCH_HEAD

# 2. Build inventories
cd /home/olu/kb_nano
python audits/hf_transformers_coverage/tools/build_inventories.py

# 3. Run the AST extractor
python audits/hf_transformers_coverage/tools/ast_extract.py \
    --dir /tmp/hf_transformers_pinned/src/transformers/models \
    --out audits/hf_transformers_coverage/hf_extract.jsonl

# 4. Pilot rows (12 architectures + 1 exception) — already in pilot/pilot_rows.csv
# 5. Shard rows — produced by 5 parallel subagents into shards/shard_<a-d|e-i|j-m|n-q|r-z>_raw.csv
# 6. Merge + summarize
python audits/hf_transformers_coverage/tools/merge_and_summarize.py
```

## Pilot examples

12 architectures + 1 no-modeling exception, audited end-to-end by the coordinator before any subagent fan-out. See [`pilot/pilot_audit.md`](pilot/pilot_audit.md) for the worked-through reasoning per row, including ops-that-were-almost-missed and methodology refinements. The 15 resulting CSV rows are at [`pilot/pilot_rows.csv`](pilot/pilot_rows.csv).

## Shards

The remaining 453 folders were sharded alphabetically across 5 disjoint ranges. Each shard was processed by an independent subagent using the locked methodology, canonical map, and pilot examples. Shard outputs:

- [`shards/shard_a-d_raw.csv`](shards/shard_a-d_raw.csv) — 96 folders / ~96 rows
- [`shards/shard_e-i_raw.csv`](shards/shard_e-i_raw.csv) — 91 folders / ~91 rows
- [`shards/shard_j-m_raw.csv`](shards/shard_j-m_raw.csv) — 78 folders / ~78 rows
- [`shards/shard_n-q_raw.csv`](shards/shard_n-q_raw.csv) — 74 folders / ~74 rows
- [`shards/shard_r-z_raw.csv`](shards/shard_r-z_raw.csv) — 114 folders / ~114 rows

Per-shard notes (`shards/shard_<range>_notes.md`) record ambiguities, modular-only / no-modeling cases, and any new canonical names a shard wanted to add.

## Verification

Per the methodology:
- Every `partial` and `unsupported` row was manually verified by the coordinator.
- ~10% of `composable` rows were spot-checked.
- Cross-shard consistency: the same canonical op maps to the same kb-nano file across all shards.
- Final spot-check: 20 random rows from the merged CSV.

## Status / Known limitations

This audit is **static** — it audits compute primitives. It does not certify byte-correctness against HF, nor does it measure performance. A `composable` model may run slowly via torch fallback. See `audit_methodology.md` § 13 for the full limitations.
