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

## 14. Re-audit pass — addressing inconsistencies and adding the trivially-fixable L1 wrappers

After the first audit pass landed (77.5% coverage), a deeper re-audit identified two classes of issues:

**(a) Subagent inconsistency on `nn.MultiheadAttention`.** The same call site was classified `composable` by the r-z subagent (siglip) but `partial` by the e-i / a-d / j-m subagents (idefics2, bridgetower, aria, mask2former). `nn.MultiheadAttention` is the legacy PyTorch wrapper around 3×Linear + scaled_dot_product_attention + 1×Linear — the underlying compute is fully present in kb-nano via `linear.py:Linear` and `dense_attention.py:DenseAttention`. Per mentor guidance, kb-nano will not add a literal wrapper for the deprecated class API; the rows are reclassified `composable` because the math primitives are all present.

**(b) "Partial" was used for any flagged op that lacked a dedicated L1 wrapper, even when the op was a thin torch.nn module that's a one-line `F.x` call.** This included `adaptive_avg_pool_*` (CNN classifier heads), `conv_transpose_{1,2,3}d` (audio vocoders, segmentation upsamplers), `batch_norm_{1,3}d`, `max_pool_1d` / `avg_pool_1d`, `leaky_relu`, `elu`, `hardsigmoid`, `hardswish`, `grid_sample`. Each of these is a 10-line wrapper around the corresponding `F.x` torch built-in.

**Resolution.** In this audit branch I added 16 new L1 wrappers (14 around torch built-ins + `nn.LSTM` for encodec + `fla.ops.gated_delta_rule` for Qwen3.5/Qwen3-Next/OLMo-Hybrid, mirroring `chunk_gla.py`'s pattern). Each new op was numerically verified against `torch.nn.X` on random input — every test produced bit-identical output (max-abs diff 0.00e+00) and matching state_dict keys for the parameter-bearing ones. The test script lives at `tools/test_new_l1_ops.py`.

The merge tool then auto-reclassifies any `partial` row whose flagged ops are *all* in the now-supported set → `composable`, preserving the original flagged-op text in a notes field for traceability. The reclassification logic is in `tools/merge_and_summarize.py:NEWLY_SUPPORTED_OPS`.

**Effect on numbers (denominator: 447 PT-modeling rows, NIR excluded):**

| status | before re-audit (post-coordinator-overrides at commit 1f0c60a) | after re-audit | delta |
|---|---:|---:|---:|
| `kb_nano_l4` | 17 | 17 | 0 |
| `composable` | 330 | 422 | **+92** |
| `partial` | 96 | 4 | **−92** |
| `unsupported` | 4 | 4 | 0 |

Coverage (L4 + composable): **77.9% → 98.2%**.

(Earlier draft of this table claimed `composable` went `326 → 422 (+96)`. That was an arithmetic error — I used the *pre-coordinator-overrides* baseline (326) for composable but the *post-overrides* baseline (96) for partial. The correct deltas are `+92` and `−92`. The post-re-audit numbers (422 / 4) are unaffected.)

The 4 remaining `partial` rows all have at least one flagged op that genuinely cannot be wrapped trivially:
- `layoutlmv2` — needs `detectron2_backbone` (external library, runtime-loaded).
- `recurrent_gemma` — needs RG-LRU recurrent kernel (custom recurrence with per-head gates; could be added but non-trivial).
- `timm_backbone`, `timm_wrapper` — runtime-loaded `timm` model; coverage is undecidable from static analysis.

The 4 `unsupported` rows are unchanged (mra/reformer/rwkv-v4/xlstm — each requires a custom non-SDPA kernel).

### Re-re-audit: removed 8 stylistic L1 wrappers (true minimum is 8 new files, not 16)

Original re-audit added 16 new L1 wrappers. On code-deep re-inspection it became clear that **8 of those 16 are unnecessary** because they are either (i) torch builtins available via the audit's passthrough mechanism, or (ii) trivially composable from pre-existing kb-nano L1 ops with no precision loss. They were removed from this audit branch.

Verification of compositions (`tools/test_composition_equivalence.py`, **243 tests across all dtypes, all shape ranks, all parameter configs, eval+train modes; 100% pass**):

- **`BatchNorm1d` / `BatchNorm3d`** → use kb-nano `BatchNorm2d` directly. Code-verified by reading `tasks/baseline/L1/batch_norm2d.py:42` — its forward is `F.batch_norm(x, ...)` with no rank check. Empirically verified that `F.batch_norm` accepts ranks 2/3/4/5 with same output shape. Tested across the actually-used HF patterns: rank-2 `[B,C]` (groupvit/levit linear-projection BatchNorm1d), rank-3 `[B,C,L]` (hubert/fastspeech conformer BatchNorm1d), rank-5 `[B,C,D,H,W]` (emu3 VQVAE BatchNorm3d).
- **`MaxPool1d` / `AvgPool1d`** → use kb-nano `MaxPool2d` / `AvgPool2d` with kernel `(1, k)` on `x.unsqueeze(-2)` then `squeeze(-2)`. Bit-identical for all kernel/stride/padding configurations HF uses.
- **`LeakyReLU` / `ELU` / `Hardsigmoid` / `Hardswish`** → torch builtins (`F.leaky_relu`, `F.elu`, `F.hardsigmoid`, `F.hardswish`). Bit-identical to `nn.X` by construction. Audit passthrough mechanism covers them (same as `cat`, `gather`, `where`).

The 8 ops that REMAIN added as L1 wrappers (genuinely new primitives, not compositions):

| L1 file kept | reference impl | why genuinely new (not composable) |
|---|---|---|
| `tasks/baseline/L1/adaptive_avg_pool1d.py` | `F.adaptive_avg_pool1d` | adaptive output sizing computes kernel/stride dynamically — non-trivial composition |
| `tasks/baseline/L1/adaptive_avg_pool2d.py` | `F.adaptive_avg_pool2d` | same |
| `tasks/baseline/L1/conv_transpose1d.py` | `F.conv_transpose1d` (+ weight/bias storage matching `nn.ConvTranspose1d` state_dict) | transposed convolution is a fundamentally different op direction; cannot be composed from `Conv` |
| `tasks/baseline/L1/conv_transpose2d.py` | `F.conv_transpose2d` (+ weight/bias storage) | same |
| `tasks/baseline/L1/conv_transpose3d.py` | `F.conv_transpose3d` (+ weight/bias storage) | same |
| `tasks/baseline/L1/grid_sample.py` | `F.grid_sample` | fundamental sampling op (also used standalone by `superpoint`, `videomt` outside deformable-attention contexts) |
| `tasks/baseline/L1/lstm.py` | `nn.LSTM` (cuDNN) | recurrent state machine with multi-layer / bidirectional / proj_size — not composable from primitives |
| `tasks/baseline/L1/chunk_gated_delta_rule.py` | `fla.ops.gated_delta_rule.{chunk,fused_recurrent}_gated_delta_rule` | specific FLA recurrent algorithm (Qwen3.5/Qwen3-Next/OLMo-Hybrid); same import pattern as kb-nano's existing `chunk_gla.py` |

The 8 ops that were REMOVED (proven composable, bit-identical to a composition of pre-existing kb-nano + torch builtins): `BatchNorm1d`, `BatchNorm3d`, `MaxPool1d`, `AvgPool1d`, `LeakyReLU`, `ELU`, `Hardsigmoid`, `Hardswish`. Their canonical map entries are now `composable` (no file path) with the composition recipe documented in `tools/canonical_to_kb_nano.csv`. The auto-reclassify logic still treats them as supported (a row whose only partial flag is one of these is correctly reclassified as `composable` — no kb-nano L1 file is needed because the existing primitives cover them).

**Headline numbers are unchanged** by this re-re-audit (422 composable, 4 partial, 4 unsupported) — the change is in *how* certain ops are supported (composition vs new L1), not whether they are supported.

### Final list: 8 new L1 wrappers in `tasks/baseline/L1/` (numerically verified — see `tools/test_keep_ops_thorough.py`, 297 tests, 100% pass)

| file | class | reference impl |
|---|---|---|
| `adaptive_avg_pool1d.py` | `AdaptiveAvgPool1d` | `F.adaptive_avg_pool1d` |
| `adaptive_avg_pool2d.py` | `AdaptiveAvgPool2d` | `F.adaptive_avg_pool2d` |
| `conv_transpose1d.py` | `ConvTranspose1d` | `F.conv_transpose1d` + state_dict-compat with `nn.ConvTranspose1d` |
| `conv_transpose2d.py` | `ConvTranspose2d` | `F.conv_transpose2d` + state_dict-compat |
| `conv_transpose3d.py` | `ConvTranspose3d` | `F.conv_transpose3d` + state_dict-compat |
| `grid_sample.py` | `GridSample` | `F.grid_sample` |
| `lstm.py` | `LSTM` | `torch.nn.LSTM` (cuDNN; encodec uses this) |
| `chunk_gated_delta_rule.py` | `ChunkGatedDeltaRule`, `FusedRecurrentGatedDeltaRule` | `fla.ops.gated_delta_rule.{chunk,fused_recurrent}_gated_delta_rule` (Qwen3.5 / Qwen3-Next / OLMo-Hybrid; mirrors kb-nano's existing `chunk_gla.py` import pattern) |

(Plus 8 ops removed as composable; see table above.)

**Auto-reclassification accounting.** Of the 96 originally-`partial` rows, 92 flipped to `composable` after the re-audit. Each row flipped because every flagged op falls in `tools/merge_and_summarize.py:NEWLY_SUPPORTED_OPS`, which contains:

- (A) **Genuinely-new L1 wrappers** (8 ops): `adaptive_avg_pool_1d/2d`, `conv_transpose1d/2d/3d`, `grid_sample`, `lstm`, `chunk_gated_delta_rule`.
- (B) **Composition-supported ops** (8 ops): `batch_norm_1d/3d`, `max_pool_1d`, `avg_pool_1d`, `leaky_relu`, `elu`, `hardsigmoid`, `hardswish`. No new file — uses kb-nano BatchNorm2d/MaxPool2d/AvgPool2d with reshape, or torch builtin F.x. Verified bit-identical (`tools/test_composition_equivalence.py`, 243 tests).
- (C) **Pre-existing kb-nano support** (3 cases): `multihead_attention` (deprecated nn class; per mentor, no new wrapper), `causal_conv1d` (already in `tasks/baseline/L2/mamba_mixer.py`), `deformable_attention_v1_normalization` (kb-nano L1 with `method="default"` is bit-identical).

The 4 remaining `partial` rows have at least one flag that is none of the above (e.g. `detectron2_backbone`, `rg_lru_scan`, `timm_dynamic_backbone`).

## 15. Pass v3: explicit op support (mentor: performance-faithful) + Conv1d narrowness fix + MHA L2 wrapper

A subsequent re-audit on this branch (after independent code-deep inspection per the audit prompt) found three separate issues with the previous pass:

### Conv1d narrowness — false-positive `composable` rows

I had marked `conv1d` → `tasks/baseline/L1/conv1d.py:Conv1d` "direct" in the canonical map without reading the wrapper code. Re-inspection showed the existing kb-nano `Conv1d` is a Whisper-specific narrow wrapper:

```python
class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=True):
        ...  # NO groups, NO dilation, NO padding_mode
```

But HF heavily uses `groups=` (granite_speech, vibevoice, squeezebert) and `dilation=` (dac, encodec, pe_audio, vibevoice_acoustic_tokenizer) for `nn.Conv1d`. ~10 HF rows had been over-classified as `composable` when their Conv1d use is actually narrow-uncovered.

**Fix.** Added an additive general-purpose wrapper at `tasks/baseline/L1/conv1d_native.py:Conv1dNative` with full `nn.Conv1d` kwarg coverage (groups, dilation, padding_mode for zeros/reflect/replicate/circular, bias, stride, padding). State_dict-compatible with `nn.Conv1d` (direct `weight`/`bias` parameters, not nested under `self.conv`). Existing narrow Conv1d preserved unchanged for Whisper. Verified bit-identical to `nn.Conv1d` across all HF kwarg patterns and dtypes (`tools/test_v3_ops.py`, 18 Conv1dNative tests, 100% pass).

### Mentor reversal on stylistic L1 ops — performance-faithfulness matters

The previous pass removed `MaxPool1d`, `AvgPool1d`, `LeakyReLU`, `ELU`, `Hardsigmoid`, `Hardswish` because they were "composable from existing primitives." Mentor pushback: the composed version (e.g. `MaxPool2d((1, k))(x.unsqueeze(-2)).squeeze(-2)` for 1D pool) is *semantically* equivalent but *benchmarks the wrong kernel family* (2D pool with degenerate H=1) and adds reshape overhead. Per the audit prompt's policy: "If a composition changes the kernel family, adds unnecessary memory movement, prevents fused dispatch, materializes intermediates, or uses an awkward higher-dimensional kernel, add an explicit op."

**Fix.** Re-added the 6 ops as explicit L1 wrappers (each dispatches directly to the matching `F.max_pool1d` / `F.avg_pool1d` / `F.leaky_relu` etc. — NOT through the 2D-composed workaround). Verified vs `torch.nn.X` on 100% of HF kwarg patterns + edge cases.

`BatchNorm1d` / `BatchNorm3d` are NOT re-added because kb-nano `BatchNorm2d.forward` calls `F.batch_norm` rank-agnostically — the "composition" is calling kb-nano BatchNorm2d directly with whatever rank input you have, no reshape, *same kernel*. So the mentor's performance-faithfulness rule does not apply (no different benchmark target, no extra movement).

### LayerNorm narrowness — `bias` kwarg missing

kb-nano `LayerNorm.__init__` had `create_scale`/`create_offset` (openfold3-style) but did NOT accept the `bias` kwarg `nn.LayerNorm` exposes. 9 HF folders pass `nn.LayerNorm(..., bias=False)` or `bias=config.norm_bias` (bark, dbrx, gemma4, lasr, modernbert, modernbert_decoder, modernvbert, moonshine, moonshine_streaming) — drop-in compatibility breaks with TypeError.

**Fix.** Extended `tasks/baseline/L1/layer_norm.py:LayerNorm.__init__` additively: added `bias` kwarg (when provided, takes precedence over `create_offset`) and `tuple/list normalized_shape` support (matches `nn.LayerNorm` semantics). Existing callers that pass `create_scale` / `create_offset` are unaffected. Verified vs `nn.LayerNorm` with `bias=True/False` and tuple shape (max-abs diff 0.00e+00).

### MultiheadAttention — proper L2 wrapper instead of "composable via existing primitives"

The previous pass declared `nn.MultiheadAttention` "composable via 3×Linear + DenseAttention + 1×Linear, no new wrapper needed." Per audit prompt: this is correct semantically, but a naive composition that *materializes the attention map* (which `nn.MultiheadAttention.forward` always does internally) defeats the SDPA fast path. Six HF folders use `nn.MultiheadAttention` directly (aria, bridgetower, idefics2, mask2former, oneformer, omdet_turbo, phi4_multimodal).

**Fix.** Added `tasks/baseline/L2/multihead_attention.py:MultiheadAttention` — a proper L2 wrapper that:
- Mirrors `torch.nn.MultiheadAttention.__init__` and `forward` signatures (so HF call sites are drop-in).
- Stores `in_proj_weight` / `in_proj_bias` / `out_proj.weight` / `out_proj.bias` matching torch's parameter names (HF reference checkpoints load with no remap).
- Uses `F.scaled_dot_product_attention` (= kb-nano `DenseAttention`) when `need_weights=False` (the common case — aria, idefics2 take `attention(...)[0]`, discarding weights).
- Materializes the attention map only when `need_weights=True` (matches `nn.MultiheadAttention`'s tuple-return semantics).
- Supports `attn_mask`, `key_padding_mask`, `batch_first`, `is_causal`, `average_attn_weights`, separate `kdim`/`vdim`.

Verified bit-identical to `nn.MultiheadAttention` across self-attention (no_weights / with_weights), cross-attention (different Q vs KV lengths), `attn_mask` (causal), `key_padding_mask`, `batch_first=False/True` — 15 tests in `tools/test_v3_ops.py`.

### Test coverage of v3 changes

`tools/test_v3_ops.py` — 227 tests, 100% PASS:
- 24 MaxPool1d + 18 AvgPool1d (direct 1D dispatch verified across 3 dtypes × 4 kernel/stride/padding × 4 shapes)
- 144 elementwise activations (LeakyReLU/ELU/Hardsigmoid/Hardswish across 3 dtypes × 4 shapes × multiple slope/alpha)
- 18 Conv1dNative across all HF kwarg patterns: depthwise (granite), strided+dilated+grouped (vibevoice), padded+dilated (dac/encodec), 1×1 grouped (squeezebert), narrow (whisper), padding_mode in {reflect, replicate, circular} — fp32 + bf16
- 1 Conv1dNative state_dict key match
- 15 MultiheadAttention tests covering state_dict compat, self-attn no_weights, self-attn with_weights, cross-attn, attn_mask, key_padding_mask, batch_first=False, multiple sizes/dtypes

### Final spot-check audit (10 critical rows by code-deep inspection)

Read each HF modeling file directly and verified the audit row's classification against the actual compute primitives used:

| HF folder | status | what HF uses | kb-nano coverage |
|---|---|---|---|
| `llama` | kb_nano_l4 | Linear, RMSNorm, RoPE, SDPA, KV cache, SiLU | L4 pipeline at `tasks/baseline/L4/llama.py` ✓ |
| `bert` | composable | Linear, LayerNorm, Embedding, Dropout, Tanh | all in kb-nano L1 ✓ |
| `bark` | composable | LayerNorm with `bias=config.bias`, GELU, Dropout, Embedding, Linear, F.softmax | now covered after LayerNorm `bias` kwarg extension ✓ |
| `dac` | composable | Conv1d with `dilation=` and `padding=` | now covered by new Conv1dNative ✓ |
| `mask2former` | composable | GroupNorm, LayerNorm, `nn.MultiheadAttention` (line 1585), MS-deformable attn v1 | MHA covered by new L2 wrapper; deformable v1 covered by kb-nano L1 with method="default" ✓ |
| `whisper` | kb_nano_l4 | Conv1d (stride+padding only), GELU, LayerNorm, Linear, MHA inside L2 whisper_attention | L4 pipeline exists; narrow Conv1d sufficient ✓ |
| `regnet` | composable | Conv2d, BatchNorm2d, AdaptiveAvgPool2d, ReLU | all in kb-nano L1 (AdaptiveAvgPool2d added in v3) ✓ |
| `mra` | unsupported | `mra_cuda_kernel.index_max` / `mm_to_sparse` / `sparse_dense_mm` from `kernels-community/mra` | confirmed: custom CUDA, no kb-nano analog ✓ |
| `recurrent_gemma` | partial | `RecurrentGemmaRglru` with `torch.baddbmm`-based custom recurrence | confirmed: no rg_lru kernel in kb-nano (FLA family covers GLA/Retention/RWKV7 not RG-LRU) ✓ |
| `reformer` | unsupported | `LSHSelfAttention` with `_hash_vectors` + bucketed attention | confirmed: hash-based bucketing IS the algorithm, not a kernel call ✓ |

10/10 verified. No misclassifications found in this spot-check pass.

### Final state after v3

- **8 explicit new L1 ops** kept from prior passes (AdaptiveAvgPool1d/2d, ConvTranspose1d/2d/3d, GridSample, LSTM, ChunkGatedDeltaRule)
- **6 explicit new L1 ops re-added** (MaxPool1d, AvgPool1d, LeakyReLU, ELU, Hardsigmoid, Hardswish)
- **1 new L1 op** (Conv1dNative — additive to existing narrow Conv1d)
- **1 existing L1 op extended** (LayerNorm: added `bias` kwarg + tuple-shape support, additive)
- **1 new L2 op** (MultiheadAttention — uses DenseAttention/SDPA fast path)

Coverage numbers UNCHANGED (the change is in HOW certain rows are supported and removing false positives, not in which models can run):
- 17 `kb_nano_l4` (3.8%)
- 422 `composable` (94.4%)
- 4 `partial` (0.9%) — layoutlmv2 (detectron2), recurrent_gemma (RG-LRU), timm_backbone, timm_wrapper
- 4 `unsupported` (0.9%) — mra, reformer, rwkv v4, xlstm
- Coverage = 439/447 = **98.21%**

## 16. Pass v3.1: systematic narrowness re-audit (4 more ops extended additively)

After the v3 commit (96f740e), user feedback: "is it just conv1d? any other things with a similar issue? please make sure everything is up to date and correct, do a full reaudit of the guidelines and instruction prompt." Did a systematic code-deep inspection of every kb-nano L1 wrapper's `__init__` signature against `torch.nn.X` and HF actual usage.

Found 4 more narrow ops with the same pattern as Conv1d (file existed in kb-nano L1 catalog but missed kwargs HF uses):

| op | what was missing | HF usage that breaks |
|---|---|---|
| `Conv3d` | `padding`, `dilation`, `groups`, `padding_mode` | `emu3/modeling_emu3.py: nn.Conv3d(..., padding=0)` (VQVAE temporal block) — TypeError on kb-nano narrow Conv3d |
| `ReLU` | `inplace` kwarg | `regnet/sam_hq/...: nn.ReLU(inplace=True)` — TypeError on kb-nano narrow ReLU |
| `AvgPool2d` | `count_include_pad`, `divisor_override` | `swin2sr/...: nn.AvgPool2d(pool_size, stride=1, padding=pool_size // 2, count_include_pad=False)` — TypeError |
| `Dropout` | `inplace` | `nn.Dropout(p, inplace=True)` (multiple HF) — TypeError |

**Fix.** Each extended additively (defaults match torch.nn.X so existing kb-nano callers are unaffected). Verified by:
- Backward-compat: kb-nano callers using only the original kwargs produce 0.00e+00 max-abs diff vs the prior wrapper.
- New-kwargs forward output matches `torch.nn.X` 0.00e+00 max-abs diff.
- All existing test suites still pass (v3: 227/227, keep_ops_thorough: 297/297, composition: 243/243).

**Files modified additively:**
- `tasks/baseline/L1/conv3d.py` — added `padding`, `dilation`, `groups`, `padding_mode`
- `tasks/baseline/L1/relu.py` — added `inplace`
- `tasks/baseline/L1/avg_pool2d.py` — added `count_include_pad`, `divisor_override`
- `tasks/baseline/L1/dropout.py` — added `inplace`

The original Conv1d narrowness was the same systemic mistake: I had built the canonical map by reading file *names*, not file *contents*. After v3 (Conv1d + LayerNorm) and v3.1 (Conv3d + ReLU + AvgPool2d + Dropout), I have now code-deep-verified every kb-nano L1 wrapper that maps to a canonical HF op against actual HF call sites in the pinned commit.

**Remaining ops that I did NOT extend** (verified their narrowness is acceptable):
- `Embedding`: no HF usage of `max_norm`/`scale_grad_by_freq`/`sparse` in pinned commit. Fine.
- `Conv2d`: no HF usage of `padding_mode`. Fine.
- `MaxPool2d`: no HF usage of `dilation` or `return_indices`. Fine.
- `Linear`: no HF use of `device`/`dtype` at architectural level. Fine.
- `silu/sigmoid/tanh`: stateless, no kwargs in HF use. Fine.
- `softmax`: only `dim` is needed. Fine.
- `GroupNorm`: kb-nano default eps is 1e-6, torch's is 1e-5. Most HF callers pass explicit eps; some don't (`nn.GroupNorm(min(8, out_channels), out_channels)`). Default mismatch is a subtle correctness issue (~1e-5 numeric difference) but changing the default would risk breaking existing kb-nano callers (sdxl, sam3, rwkv7, etc. some of which may rely on the 1e-6 default). Leaving alone but documented.

**Coverage numbers UNCHANGED** (the v3.1 change is in *how* HF kwargs are accepted; no model moves between status buckets — they were already classified as composable based on the math, the kwarg gap was a wrapper/API issue):
- 17 `kb_nano_l4` (3.8%) + 422 `composable` (94.4%) + 4 `partial` (0.9%) + 4 `unsupported` (0.9%) + 24 `not_inference_required`
- Coverage = 439/447 = **98.21%**

## 17. Pass v3.2: MHA L2 wrapper removed; persistent stale-prose cleanup; Conv3d default deviation documented

After the v3.1 audit and the cross-branch merge round, three more issues surfaced and were fixed:

**A. MHA L2 wrapper removed.** The `tasks/baseline/L2/multihead_attention.py` wrapper had two real problems: (i) it constructed `DenseAttention()` in `__init__`, which calls `torch.cuda.get_device_capability()` unconditionally — broke construction on CPU-only machines with `TypeError: '<' not supported between instances of 'NoneType' and 'int'`; the `_sdpa` attribute was also dead code (forward called `F.scaled_dot_product_attention` directly). (ii) `need_weights=True` is the default (matching `torch.nn.MultiheadAttention`'s API) — but HF callsites like `aria` and `idefics2` use `attention(q, k, v)[0]` *without* passing `need_weights=False`, so they take the materialize-attention-map branch, defeating the wrapper's claimed SDPA fast path. Combined with the original mentor instruction "no need to add multihead_attention" and the fact that no kb-nano L4/L3/L2 file uses `nn.MultiheadAttention` internally (verified by grep), the wrapper is removed. Canonical map now records `multihead_attention` as `composable` (`3×Linear + DenseAttention + Linear`); HF rows that flag it stay `composable` because the underlying compute is fully present in kb-nano L1.

**B. Persistent stale-prose cleanup.** Earlier cleanup passes (`tools/clean_stale_notes.py` one-shot script + `[v3.1: composable; …]` banner) were one-off — subsequent merge regenerations cleared them, so 90 composable rows still had phrases like `partial via torch.nn fallback` / `no L1 kernel` / `(no L1 wrapper)` after each pipeline rerun. Fix: bake the cleanup into `tools/merge_and_summarize.py:normalize_row` itself. It runs on every regeneration and is idempotent. Verified: 0 truly-stale phrases in the regenerated CSV. The list of patterns rewritten is in `_STALE_PROSE_REWRITES` (top of `merge_and_summarize.py`); each maps a stale phrase to current-state phrasing (e.g., `nn.AdaptiveAvgPool2d (no L1 kernel)` → `nn.AdaptiveAvgPool2d (covered by tasks/baseline/L1/adaptive_avg_pool2d.py:AdaptiveAvgPool2d)`).

**C. Conv3d default-stride deviation documented (NOT changed).** kb-nano `Conv3d.__init__` has `stride=stride or kernel_size` (vllm/patch-embed convention). torch.nn.Conv3d defaults to `stride=1`. Two kb-nano callsites — `L2/vision_patch_embed.py:25` and `L2/vjepa2_embeddings.py` — rely on the patch-embed default (one passes only `kernel`, no `stride`; the other passes `stride=kernel_size` explicitly). Changing the default to torch's `stride=1` would silently break `vision_patch_embed.py`. Decision: keep the patch-embed default; honestly document the deviation in `tasks/baseline/L1/conv3d.py`'s docstring. Earlier methodology claim that "defaults match torch.nn.Conv3d" was inaccurate and is corrected here.

**D. Stale canonical map entries fixed.** `mish` now points to `tasks/baseline/L1/mish.py:Mish` (file came in via `add-dp3-support` merge); `causal_conv1d` now points to `tasks/baseline/L1/causal_conv1d.py:CausalConv1d` (file came in via `add-kimi-qwen3next` merge). Previously both entries said the file didn't exist.

**E. Conv1d / Conv1dNative coexistence (NOT a duplicate).** After `add-dp3-support` merge, both files exist:
- `tasks/baseline/L1/conv1d.py:Conv1d` — extended additively in this audit (groups, dilation, padding_mode, full HF kwarg surface). Holds an inner `nn.Conv1d` as `self.conv`; state_dict keys nested under `conv.` (e.g. `conv.weight`). This is the **canonical map target for HF audit's `conv1d` op**.
- `tasks/baseline/L1/conv1d_native.py:Conv1dNative` — DP3-specific. Direct `weight`/`bias` parameters (state_dict-compat with `nn.Conv1d`); narrower kwargs (no padding_mode). Used by `tasks/baseline/L2/dp3_conv1d_block.py`. NOT a superset of `Conv1d`.

Both serve different purposes (different state_dict layouts). Removing `conv1d_native.py` would require either (i) refactoring `dp3_conv1d_block.py` to use `Conv1d` (breaks DP3's checkpoint loading because of the layout difference) or (ii) extending `Conv1d` to expose direct top-level `weight`/`bias` parameters in addition to the nested `self.conv` (complex, risks breaking Whisper). Decision: keep both, document each one's scope clearly in its docstring.

**Comprehensive scan for similar situations across merged branches.** I grepped all merged-in L2/L3/L4 files for `nn.X(…)` direct usage where a kb-nano L1 wrapper exists for `X`. Found 97 unique (file, nn-class) pairs across 16 merged branches. Top patterns: `nn.Dropout` (16 files, mostly in attention modules), `nn.Linear` (16 files), `nn.Conv2d` (13 files, mobilenetv4 family + sam3), `nn.ReLU` (11 files), `nn.LayerNorm` (9 files), `nn.Conv1d` (5 files), `nn.Embedding` (5 files), `nn.BatchNorm2d` (4 files). These are CLAUDE.md violations in spirit (L2+ should use kb-nano L1, not torch.nn directly) but they're **pre-existing code from the original model authors' branches**, not introduced by this audit. Refactoring all of them to consistently use kb-nano L1 wrappers is a large multi-branch sweep that's out of scope for the HF coverage audit. Documented for follow-up; not changing any of these in this branch.

## 18. What this proves about kb-nano

The pilot data already supports the headline framing of the paper appendix: kb-nano's existing L1/L2 surface covers the core compute primitives needed by the most common HF architecture families (encoder, decoder-only, encoder-decoder, vision encoder, multimodal, detection, SSM). The remaining gaps are concentrated in (1) niche pooling kernels (`adaptive_avg_pool*`), (2) `ConvTranspose*` for segmentation/upsampling heads, and (3) attention variants that haven't been hit yet (e.g. `flex_attention`-only models). Each gap is a small, well-bounded kernel — none reflect a fundamental architectural limitation.

The full architecture-level numbers and unsupported-op frequencies are in `coverage_summary.md` after the scaled audit completes.
