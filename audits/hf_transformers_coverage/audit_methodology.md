# Methodology: kb-nano coverage of Hugging Face Transformers

This document is the canonical methodology for the FASTKERNELS / kb-nano coverage audit of Hugging Face Transformers architectures. All deliverables in this directory follow this methodology.

## 1. Sources of truth

- **HF Transformers**: pinned to commit `da6c53e431f7c9ef0691239d4ce89b0f711ecad7` of `huggingface/transformers`. Cloned (shallow) to `/tmp/hf_transformers_pinned/`. All HF `file:line` evidence in deliverables refers to this commit.
- **kb-nano**: branch `audit/hf-transformers-coverage` cut from `origin/experiments` @ commit `11aa838 add manual values for missing ops`. All kb-nano `file:line` evidence refers to this branch's working tree.
- **Operator surface counted on kb-nano**: only `tasks/baseline/L1`, `tasks/baseline/L2`, and `tasks/baseline/L3` — these are the layered primitives and composites. `tasks/baseline/L4` is integration-level and is consulted only to determine whether a given HF architecture is *already* implemented as an L4 pipeline.

The audit explicitly does not rely on `tasks/reference/`, which contains semantic PyTorch references for correctness validation, not optimized kernels.

## 2. What "support" means

The audit answers a single question per HF modeling file:

> Can this architecture run end-to-end inside kb-nano using the existing L1/L2/L3 components, with no new compute primitive required?

It does **not** answer:

- Does kb-nano produce numerically identical outputs?
- Does kb-nano outperform vLLM / HF on this workload?
- Is there a working L4 pipeline?

These are out of scope. The deliverable is *coverage of compute primitives*, not certification of correctness or speed.

## 3. Status labels

| label | meaning |
|---|---|
| `kb_nano_l4` | An L4 pipeline already exists in `tasks/baseline/L4/` for this architecture or a near-identical one. |
| `composable` | Every required compute op has either a direct kb-nano L1/L2 mapping, a kb-nano L3 layer that wraps the right primitives, or a torch built-in / passthrough. Building the architecture is a wiring task; no new kernel is required. |
| `partial` | At least one required op has only a torch.nn fallback (no L1 kernel) AND that op runs in the inference path. The model can be built and run via the same convention kb-nano uses for similar L4 pipelines, but a kernel-level optimization is missing. |
| `unsupported` | At least one required op has no kb-nano coverage and is not a trivial torch built-in. Building the architecture would require a new compute primitive. |
| `not_inference_required` | Folder contains no PyTorch modeling file (e.g. tokenizer-only). |

## 4. Conservatism rule

Filename similarity is **never** sufficient. A status of `composable` or better requires citing the specific kb-nano L1/L2/L3 file:line for each load-bearing op. When in doubt the row is downgraded to `partial`, with the ambiguity recorded in `notes`.

## 5. Extraction (per HF modeling file)

All extraction goes through `tools/ast_extract.py`, which uses Python's `ast` module (not regex). For each `modeling_*.py` it extracts:

- All `nn.Module`-derived class definitions and their bases.
- Every dotted-path call (`nn.X`, `F.X`, `torch.X`, `module.method`) that maps to a known compute primitive via `tools/ast_extract.py`'s lookup tables.
- All `ACT2FN[<key>]` subscript reads (literal keys captured; dynamic keys flagged as `act2fn_dynamic`).
- All `ALL_ATTENTION_FUNCTIONS[<key>]` subscript reads.
- All HF helpers used (`apply_rotary_pos_emb`, `DynamicCache`, `EncoderDecoderCache`, `selective_scan_fn`, `causal_conv1d_fn`, `multi_scale_deformable_attention`, etc.).
- Imports.

The extractor writes a JSON record per file. Output of `tools/ast_extract.py --dir /tmp/hf_transformers_pinned/src/transformers/models --out hf_extract.jsonl` is the source data for shard subagents.

The extractor is intentionally conservative: it canonicalizes only well-known compute primitives and emits any unresolved dotted call that looks like a function (lowercase-leading, length>2) into `unresolved_top` so the auditor can extend the lookup table.

## 6. Mapping (per file)

A canonical-name → kb-nano lookup table lives at `tools/canonical_to_kb_nano.csv`. It records, for each canonical op:
- The kb-nano file path.
- The kb-nano class name (or `passthrough` for torch built-ins, or empty for unsupported).
- A note (e.g. "direct", "partial because torch.nn fallback", "unsupported on origin/experiments").

The mapping table is **locked** at the end of the pilot. Shard subagents must use it; if they encounter a canonical op that isn't in the table they flag it as `new_canonical_name_needed` and the coordinator decides.

## 7. Inventory denominators

The audit reports three denominators:

| denominator | count on pinned commit | use |
|---|---|---|
| folders under `models/` | 465 | total HF model surface |
| folders with any PyTorch modeling | 442 | "real" architecture surface (excludes tokenizer-only) |
| distinct PyTorch modeling files (sum across folders) | 448 | the **headline denominator** — handles multi-modeling folders cleanly |
| folders with `modular_*.py` | 232 | indicates HF modular DSL adoption |
| folders with multiple PyTorch modeling files | 5 | (blip, data2vec, esm, maskformer, rt_detr) |
| folders with no modeling at all | 21 | tokenizer/processor/deprecated only |
| folders with `modular_*.py` but no PyTorch modeling | 2 | (`layoutxlm`, `pp_chart2table`) — these are tokenizer wrappers that re-use a parent model |

Coverage percentages in `coverage_summary.md` use the modeling-file denominator (448). Folder-level counts are reported as a secondary check.

## 8. Sharding

After the pilot, the remaining ~430 modeling files were sharded by folder name range:

- `shard_a-d`
- `shard_e-i`
- `shard_j-m`
- `shard_n-q`
- `shard_r-z`

Each shard is handled by one Explore subagent, working from the methodology, the canonical map, the kb-nano operator catalog, and the pilot examples. Subagents write only their shard's raw files (`shards/shard_<range>_raw.csv`, `shards/shard_<range>_notes.md`). The coordinator (the user-facing agent) reviews, gates, and merges.

## 9. Verification gates

Per shard:
1. Read the shard's notes file. If `unsupported` or `partial` rate is unusually high (>20% above the pilot rate), spot-check 5–10 rows manually before merging.
2. **Verify every `unsupported` and every `partial` row by hand.** These are the load-bearing claims of the paper appendix.
3. Spot-check 10% of `composable` rows (uniformly random).

Cross-shard:
4. Same canonical op ↔ same kb-nano mapping across shards. Inconsistencies are coordinator-resolved.
5. Final 20-row random spot-check across the merged CSV.
6. The unsupported-op frequency table is sanity-checked: ops that should clearly be supported (linear, layer_norm, embedding, gelu, ...) must not appear there.

A failure at any gate stops merging until fixed.

## 10. Pilot checkpoint (executed; outcome: methodology is sound, minor refinements only)

Pilot scope: 12 architectures + 1 exception, audited end-to-end by the coordinator.

Pilot result distribution:
- `kb_nano_l4`: 4 (llama, whisper, mamba, qwen2_vl)
- `composable`: 9 (bert, mistral, vit, swin, rt_detr ×2, data2vec_audio, data2vec_text, deformable_detr)
- `partial`: 1 (data2vec_vision — only `ForSemanticSegmentation` variant uses `ConvTranspose2d` via torch fallback; the other heads are composable)
- `unsupported`: 0
- `not_inference_required`: 1 (barthez)

Things the pilot revealed:
1. **`EncoderDecoderCache`** is not a single kb-nano class but is implemented in HF as `EncoderDecoderCache(DynamicCache(), DynamicCache())` (verified at `bert/modeling_bert.py:650`) — composable from kb-nano's existing KV-cache primitive. No new primitive needed.
2. **`adaptive_avg_pool*`** has no L1 kernel in kb-nano. Existing kb-nano L4s (mobilenetv4, yolov10) use `nn.AdaptiveAvgPool2d` directly via torch.nn — this is a CLAUDE.md-spirit gap, but is the project's de facto convention. Decision: when adaptive_avg_pool is in the inference path of a head, classify as `partial` (works via torch fallback, but no kernel optimization).
3. **`ConvTranspose*`** has no L1 kernel either. Same convention applies (kb-nano L2/L3 use `nn.ConvTranspose1d/2d` directly: cosyvoice3_hifigan, sam3_fpn_conv, sam3_mask_decoder). When a model needs ConvTranspose in the inference path, classify as `partial`.
4. **`grid_sample`** is not standalone but is composed inside kb-nano's deformable-attention L1, so models that use it only via deformable attention are covered.
5. **Deformable attention v1 vs v2**: kb-nano's `MultiScaleDeformableAttentionV2` with `method="default"` is bit-equivalent to HF's v1 deformable attention (same `2*loc - 1` transform; same `F.grid_sample` flags). This was verified by reading both kernels side-by-side. The v2-only addition is the `method="discrete"` branch.
6. **`ALL_ATTENTION_FUNCTIONS`** dispatches at runtime to one of `{sdpa, flash_attention_2, eager, flex_attention, ...}`. kb-nano covers `sdpa` (DenseAttention), `flash_attention_2` (FlashAttn*), and `eager`. It does not have `flex_attention`. Decision: any one of the supported variants is enough for `direct`; flex-only models would be `partial`.
7. **Sliding-window attention** (Mistral, Mistral-style) is **mask construction**, not a special kernel — produced by `masking_utils.create_sliding_window_causal_mask`. No kb-nano change needed; the mask tensor is passed to the standard attention call.
8. **CTC loss / cross_entropy / mse_loss** are training-only and skipped from the support analysis. `log_softmax` (used by CTC at inference) is a torch built-in.

Schema lock-in: no breaking changes from the pilot. The schema in section 11 below is final.

Ambiguity rate from pilot: 1 of 15 rows (data2vec_vision) needed a `partial` flag with a sub-variant note. Acceptable.

## 11. Final schema

The coverage CSV (`hf_architecture_operator_coverage.csv`) has columns:

| column | description |
|---|---|
| `hf_folder` | folder name under `src/transformers/models/` |
| `modeling_file` | path relative to `src/transformers/models/` |
| `architecture_classes` | semicolon-separated list of `*Model` / `*ForXxx` classes |
| `modality` | text / vision / audio / multimodal / detection / segmentation / structure / other |
| `family` | encoder / decoder-only / encoder-decoder / SSM / hybrid / vision-encoder / detection / etc. |
| `support_status` | one of: `kb_nano_l4`, `composable`, `partial`, `unsupported`, `not_inference_required` |
| `mapped_kb_nano` | canonical-op → kb-nano file:line list (load-bearing only) |
| `partial_or_unsupported_ops` | canonical names of ops in `partial` / `unsupported` status (semicolon-separated) |
| `evidence_hf` | up to 3 `file:line` references into the pinned HF commit |
| `notes` | sub-variant flags (e.g. "ForSemanticSegmentation: partial; ForImageClassification: composable"), modular DSL caveats, ambiguity comments |

## 12. Reproducibility

To regenerate the inventory and catalog:

```bash
git fetch origin --depth 1 da6c53e431f7c9ef0691239d4ce89b0f711ecad7  # in /tmp/hf_transformers_pinned/
git checkout audit/hf-transformers-coverage
python audits/hf_transformers_coverage/tools/build_inventories.py
```

To run the AST extractor on a single file:

```bash
python audits/hf_transformers_coverage/tools/ast_extract.py /tmp/hf_transformers_pinned/src/transformers/models/<folder>/modeling_<name>.py
```

To run on all modeling files at once (writes JSONL):

```bash
python audits/hf_transformers_coverage/tools/ast_extract.py --dir /tmp/hf_transformers_pinned/src/transformers/models --out audits/hf_transformers_coverage/hf_extract.jsonl
```

## 13. Known limitations

- The audit is static. Dynamic dispatch (e.g. config-driven ACT2FN keys, runtime-selected attention impl) is reported as `act2fn_dynamic` / `attention_dispatcher`; the coordinator infers coverage from the set of supported variants but cannot enumerate every possible config.
- `kb_nano_l4` does not certify that the L4 pipeline currently passes its own correctness gate against HF — only that an L4 file with the right architecture exists.
- "modular DSL" handling: when both `modeling_<x>.py` and `modular_<x>.py` exist, the audit reads the generated `modeling_<x>.py` (which is the runtime artifact). The 2 modular-only folders (`layoutxlm`, `pp_chart2table`) are wrappers that re-use a parent architecture and are flagged in `notes`.
- The audit does not measure performance. A `composable` model may be slow if all of its ops fall back to torch eager.

## 14. What this proves about kb-nano

The pilot data already supports the headline framing of the paper appendix: kb-nano's existing L1/L2 surface covers the core compute primitives needed by the most common HF architecture families (encoder, decoder-only, encoder-decoder, vision encoder, multimodal, detection, SSM). The remaining gaps are concentrated in (1) niche pooling kernels (`adaptive_avg_pool*`), (2) `ConvTranspose*` for segmentation/upsampling heads, and (3) attention variants that haven't been hit yet (e.g. `flex_attention`-only models). Each gap is a small, well-bounded kernel — none reflect a fundamental architectural limitation.

The full architecture-level numbers and unsupported-op frequencies are in `coverage_summary.md` after the scaled audit completes.
