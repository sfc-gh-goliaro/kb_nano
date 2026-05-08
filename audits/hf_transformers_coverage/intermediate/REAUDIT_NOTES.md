# Re-audit notes (post-submission verification)

> **Canonical final state lives in the `v11 RECONCILIATION` section near the
> bottom of this document (search "v11 RECONCILIATION").** The headline
> tables in earlier sections (v4, v7, v10) are kept as a chronological audit
> trail and are **superseded**. v11 is the source of truth that matches
> `_reaudit_final_v11.json`, `audit_evidence.csv`, and `hf_coverage_rows.tex`.
>
> **v12 final (447 folders):** 27 L4 / 237 composable / 171 partial / 12 unsupported.
> Strict (L4 + composable) = 59.1%. Loose (+partial) = 97.3%. Unsupported = 2.7%.
>
> v12 vs v11: 6 folders demoted composable → partial for partial-rotary
> consistency (bamba, glm4v_moe, laguna, musicflamingo, recurrent_gemma,
> solar_open) — kb-nano L1/rotary_emb.py rotates the full head_dim, so
> partial_rotary_factor < 1.0 needs external slicing or a Gemma4-style
> proportional rotary. This matches how phi/persimmon/glm/etc. were already
> classified. See "v12" section below and CAVEATS_AND_METHODOLOGY.md §3.

This document captures the nuanced decisions made during the post-paper re-audit
of all 447 folders in `hf_coverage_rows.tex` (originally 425 + 20 missing-folder
recovery + 2 multi-modeling additions in v11). Each verdict was reached by reading
both HF source (`/tmp/hf_transformers_pinned/src/transformers/models/<folder>/`,
commit `da6c53e4`) and the relevant kb-nano kernel files
(`/home/olu/kb_nano/tasks/baseline/{L1,L2,L3,L4}/`). Cross-references list HF
file:line and kb-nano files so any reviewer can re-verify.

## Methodology decisions

The current paper uses a "loose composable" definition (any L1 op exists ⇒
composable, even without an L2 wrapper). The 16-shard re-audit used a stricter
"is there a kb-nano kernel that implements the same compute pattern" definition.

**Final reconciliation rule (used here):**
- `kb_nano_l4`: there is a kb-nano `tasks/baseline/L4/<file>.py` whose docstring
  header explicitly targets the same model family. Verified by reading the L4
  file's first 30 lines.
- `composable`: every compute class maps to an existing kb-nano L1/L2/L3 kernel
  (file is opened and verified — same activation, same norm variant, same
  projection layout, same attention type, same op sequence).
- `partial`: at least one compute class needs a torch op (`torch.fft`, custom
  index gather, weight-norm parametrization, etc.) that exists in PyTorch but
  has no kb-nano wrapper. Decomposable from L1 ops + standard PyTorch ops.
- `unsupported`: at least one compute class needs a custom CUDA kernel
  (`kernels-community/*`), an external library (timm, natten, detectron2), or
  a genuinely novel compute primitive (custom autograd Function with non-trivial
  math) that has no kb-nano OR standard-PyTorch equivalent.

Initial v4 totals (after all 16 shards + my source-verified overrides; **superseded
by v11 — see bottom**):

| status         | count | %     |
|----------------|------:|------:|
| `kb_nano_l4`   |    27 |  6.4% |
| `composable`   |   249 | 58.6% |
| `partial`      |   118 | 27.8% |
| `unsupported`  |    31 |  7.3% |
| **strict (L4 + composable)**     | **276** | **64.9%** |
| **loose (+partial)**             | **394** | **92.7%** |

The v4 numbers reflect a stricter "needs custom CUDA / external library"
definition that left 31 folders as unsupported. Subsequent rounds (v7 phase-2
deep source reads; v10 slice-7 missing-folder recovery; v11 multi-modeling
additions) re-classified 19 of those as `partial` once it was clear they
decompose from torch primitives + L1 ops, leaving the canonical 12.

Paper claim: 96.2% (409/425). Reconciled v4 strict: 64.9%. Reconciled v4 loose: 92.7%.
**Reconciled v11 strict: 60.4% / loose: 97.3% (superseded by v12 below: strict 59.1% / loose 97.3%).**

---

## L4 status changes

### Promote to `kb_nano_l4` (7 folders, currently `composable`)

Verified by reading kb-nano `L4/<file>.py` docstring header against HF
`modeling_<folder>.py`:

| HF folder        | kb-nano L4 file              | evidence |
|------------------|------------------------------|----------|
| `bitnet`         | `L4/bitnet.py`               | targets `microsoft/bitnet-b1.58-2B-4T` (W1.58A8, GQA, NeoX RoPE, squared-ReLU MLP). HF `bitnet/modeling_bitnet.py` is the modular-generated counterpart. |
| `convnextv2`     | `L4/convnextv2.py`           | "ConvNeXtV2 image classification model". HF `convnextv2/modeling_convnextv2.py` matches. |
| `deepseek_v3`    | `L4/deepseek.py`             | docstring: "Standalone DeepSeek V3.2 model implementation. Supports MLA, MoE, DSA". HF `deepseek_v3/modeling_deepseek_v3.py` is the V3 base. |
| `pi0`            | `L4/pi0.py`                  | "Pi0 vision-language-action model (L4 pipeline)". HF `pi0/modeling_pi0.py` matches. |
| `qwen2_5_omni`   | `L4/qwen2_5_omni.py`         | "Qwen2.5-Omni Thinker model". HF `qwen2_5_omni/modeling_qwen2_5_omni.py` is the modular-generated counterpart. |
| `qwen2_vl`       | `L4/qwen2_vl.py`             | "Qwen2-VL model: vision encoder + Qwen2 language model with M-RoPE". |
| `swinv2`         | `L4/swinv2.py`               | exists; HF folder `swinv2` matches. |

### Demote from `kb_nano_l4` to `composable` (1 folder)

| HF folder   | rationale |
|-------------|-----------|
| `sam2_video` | kb-nano L4 files `sam3_video.py`, `sam3_tracker.py`, `sam3.py` are explicitly SAM3-targeted (docstrings cite `sam3/model/sam3_*.py` reference). SAM2 is a different generation; the L4 doesn't cover it. Falls to `composable` via L1/L2 sam* primitives. |

### Agent overreaches I rejected (kept as `composable`)

These were L4-promotions claimed by individual shards but my source read shows
the kb-nano L4 doesn't actually cover the HF folder's full compute graph:

| HF folder         | shard | reason for rejection |
|-------------------|-------|----------------------|
| `qwen2_5_vl`      | 12    | kb-nano `L4/qwen25_vl_encoder.py` docstring: "Qwen2.5-VL text encoder for **HunyuanVideo-1.5**". It's encoder-only, not the full Qwen2.5-VL VL pipeline. |
| `falcon_mamba`    | 05    | HF `falcon_mamba/modeling_falcon_mamba.py:60-62, 267-269, 367-369` calls `rms_forward` on B/C/dt before the SSM scan. kb-nano `L4/mamba.py` doesn't do this extra normalization. Architecture differs. |
| `gemma4_assistant`| 05    | Couldn't verify equivalence to `gemma4` from header comments alone. Conservative: keep composable. |

---

## Genuine `unsupported` (31 folders)

Each requires a custom CUDA kernel, external library, or genuinely novel compute
that no kb-nano kernel implements AND no torch primitive captures cleanly.

### External CUDA kernel via `kernels-community/*` (5)

| folder    | HF file:line                                                | external dep |
|-----------|-------------------------------------------------------------|--------------|
| `mra`     | `mra/modeling_mra.py:57` `mra_cuda_kernel`                   | `kernels-community/mra` (`index_max`, `mm_to_sparse`, `sparse_dense_mm`) |
| `rwkv`    | `rwkv/modeling_rwkv.py:90-95` `kernels-community/rwkv`       | wkv_cuda forward/backward (v4-specific; kb-nano has only v7) |
| `yoso`    | `yoso/modeling_yoso.py:51-57` `kernels-community/yoso`       | `fast_hash`, `lsh_cumulation` |
| `xlstm`   | `xlstm/modeling_xlstm.py:525-589` mLSTM kernels              | `mlstm_chunkwise_kernel`, `mlstm_sequence_kernel`, `mlstm_step_kernel` (no kb-nano L1) |
| `dinat`   | `dinat/modeling_dinat.py:38-39` `from natten.functional`     | `natten2dav`, `natten2dqkrpb` (sliding+dilated 2D attention) |

### External library: `timm` / `detectron2` / HF AutoBackbone (10)

| folder                  | reason |
|-------------------------|--------|
| `timm_backbone`         | hard-imports `timm`; `timm.create_model(...)` |
| `timm_wrapper`          | hard-imports `timm` |
| `fast_vlm`              | vision tower is FastViT loaded via `timm_wrapper` (shard 05 confirmed) |
| `modernvbert`           | uses `AutoModel.from_config(backbone_config)` for both vision and text backbones |
| `chmv2`                 | `chmv2/modular_chmv2.py:23` imports `consolidate_backbone_kwargs_to_config, load_backbone`; `backbone_config: AutoConfig` |
| `conditional_detr`      | DETR-derived; relies on HF AutoBackbone for the visual feature extractor |
| `oneformer`             | deformable detr + AutoBackbone-style language backbone |
| `omdet_turbo`           | deformable detr + language backbone (`OmDetTurboLanguageBackbone`) |
| `lightglue`             | uses external matching; backbone abstraction |
| `layoutlmv2`            | visual feature extractor depends on detectron2 ResNet via META_ARCH_REGISTRY (already in original audit) |

### Genuinely novel compute primitives (16)

| folder                  | the missing primitive |
|-------------------------|-----------------------|
| `deepseek_v4`           | Heavily Compressed Attention + Compressed Sparse Attention + V4 indexer + HyperConnection routing + HashRouter + GroupedLinear + Laguna multi-rope (`modular_deepseek_v4.py:594` `DeepseekV4Attention`) |
| `evolla`                | `EvollaSequenceCompressorAttention` (Perceiver-style with concat-kv) + `EvollaSequenceAlignerCrossAttention` (gated multi-modality) — `modular_evolla.py:193, 340` |
| `exaone4_5`             | Qwen2.5-VL vision tower (`Exaone4_5_VisionAttention(Qwen2_5_VLVisionAttention)`) — kb-nano L4/qwen25_vl_encoder.py doesn't cover the VL spec |
| `florence2`             | DaViT vision: `Florence2VisionChannelAttention`, `Florence2VisionWindowAttention`, `Florence2VisionConvEmbed` (`modular_florence2.py:994, 1095`) — bespoke channel attention + window attention with depthwise conv pos enc |
| `focalnet`              | `FocalNetModulation` (`modeling_focalnet.py:276`) — multi-level depthwise Conv2d + GELU + gating + projection, replaces self-attention |
| `funnel`                | `FunnelRelMultiheadAttention` (`modeling_funnel.py:337`) — per-block stride-pooling on Q with phi/pi/psi/omega bias terms; `FunnelAttentionStructure` |
| `fuyu`                  | text backbone is Persimmon (parallel-attention with partial RoPE + LayerNorm); no kb-nano L4 for Persimmon family |
| `gemma3n`               | `Gemma3nTextAltUp` (parallel prediction routing), `Gemma3nTextLaurelBlock` (low-rank residual), `Gemma3nAudioConformerAttention`, `Gemma3nAudioCumulativeGroupNorm`, per-layer KV sharing (`modular_gemma3n.py:685, 1164, 1294`) |
| `phimoe`                | `sparsemixer` (`modeling_phimoe.py:151`) — two-pass masked-Gumbel + Heun's-third-order midpoint sampler with custom autograd Function `PhimoeMultiplier` |
| `phi4_multimodal`       | audio relative-attention-bias is a learned table + add to scores (could be partial actually; kept unsupported because of compounding gaps with depthwise Conv1d + audio path) |
| `glm_image`             | `GlmImageVisionAttention(Glm4vVisionAttention)` — depends on Glm4v vision spec which has unique compute |
| `glmasr`                | ASR-specific custom encoder layout (shard 06) |
| `got_ocr2`              | SAM ViT encoder uses Shaw-style decomposed rel-pos embeddings; kb-nano `L2/sam3_vit_attention.py` uses 2D RoPE — different attention math |
| `granite_speech`        | speech-specific bespoke encoder (shard 06) |
| `prophetnet`            | future-ngram attention (n-gram cross-attention pattern) |
| `nystromformer`         | iterative Moore-Penrose pseudo-inverse on softmax kernels (`modeling_nystromformer.py:141-160`). **NUANCE**: technically pure matmul + identity, so could be partial; agents disagreed (critical_reaudit_A says composable). Kept unsupported per shard 10's stricter read. |

---

## Non-trivial `partial` cases (118 total; documenting the load-bearing ones)

These need work but the L1 primitives all exist; they fall back to standard
PyTorch ops or compose from L1 in non-obvious ways.

### T5 / encoder-decoder cross-attention gap (~10 folders)

kb-nano `L2/t5_attention.py` only has `T5SelfAttention` (verified by grep). HF
T5 uses the same `T5Attention` class for self AND cross via `key_value_states is
not None` switch (`t5/modeling_t5.py:140-280`). Cross-attn with relative-position
bias has no kb-nano L2 wrapper.

Affected: `t5`, `t5gemma`, `t5gemma2`, `umt5`, `switch_transformers`,
`table_transformer`, `time_series_transformer`, `trocr`, `udop`, `pix2struct`,
`pop2piano`. (`prophetnet` keeps `unsupported` due to ngram-attn novelty
beyond just cross-attn.)

### Phi/Persimmon partial-RoPE + LayerNorm decoder (~3 folders)

kb-nano `L2/attention.py` hardcodes `RMSNorm` (line 26 import) and full-head
RoPE; no `partial_rotary_factor` parameter. HF `phi3/modeling_phi3.py:106-108`
explicitly slices `dim = int(head_dim * partial_rotary_factor)` for partial
RoPE. `persimmon/modeling_persimmon.py:97-99` does the same.

Affected: `persimmon`, `phi`, `phi3`, `moonshine` (partial_rotary_factor=0.9).

### ALiBi additive-bias attention (verified gap)

kb-nano L1 `flash_attn_*.py` and `dense_attention.py` have **no** `alibi_slopes`
parameter (verified by grep — zero matches across all L1 files). The
`attn_mask` parameter could carry ALiBi bias as a workaround, but this is a
non-native composition.

Affected: `bloom` (`modeling_bloom.py:45-86` `build_alibi_tensor`), `falcon`
(`modeling_falcon.py:216` `FalconAttention` legacy ALiBi path).

**Nuance**: shard 02 marked `bloom` composable (citing nonexistent
`alibi_slopes`). Shard 05 marked `falcon` unsupported. Both were inconsistent.
Final verdict: **partial** for both — ALiBi via attn_mask works but isn't
native.

### Conformer rel_shift / Transformer-XL-style attention (~7 folders)

These have `pos_bias_u` + `pos_bias_v` learnable biases and a `rel_shift`
operation that does index gather. The math is pure tensor ops but no kb-nano
L2 implements the conformer pattern.

| folder                 | HF location |
|------------------------|-------------|
| `wav2vec2_conformer`   | `Wav2Vec2ConformerSelfAttention` line 232 |
| `wav2vec2_bert`        | inherits Wav2Vec2 conformer |
| `wavlm`                | gated rel-pos bias via `nn.Embedding(num_buckets)` line 76 |
| `seamless_m4t`         | `SeamlessM4TConformerSelfAttention` line 444 |
| `seamless_m4t_v2`      | same |
| `fastspeech2_conformer`| `FastSpeech2ConformerAttention` line 362 |
| `sew_d`                | DeBERTa-v2 disentangled attn (c2p/p2c gather) |

**Nuance**: shards 16 and 05 called these unsupported; shard 13 called
seamless_m4t partial. I sided with shard 13 (partial) because the rel_shift is
pure index gather + bmm, decomposable from L1 ops.

### Two-stream rel-shift attention

| folder | HF location |
|--------|-------------|
| `xlnet` | `XLNetRelativeAttention` (`xlnet/modeling_xlnet.py:38, 68-91`) — two-stream attention with `rel_shift` reshape (pure einsum + index_select) |

### FFT-based mixing

| folder       | HF location |
|--------------|-------------|
| `autoformer` | `AutoformerAttention` (`autoformer/modeling_autoformer.py:404+`) — FFT-based autocorrelation (`torch.fft.rfft/irfft`) + top-k delay aggregation, replaces canonical attention |
| `fnet`       | `FNetBasicFourierTransform` (`fnet/modeling_fnet.py:142+`) — `torch.fft.fftn` token mixing replaces attention |
| `informer`   | Probsparse attention (Autoformer family, top-k + scatter) |

### Sliding-window / chunked attention (originally unsupported)

| folder       | HF location |
|--------------|-------------|
| `longformer` | `LongformerSelfAttention._sliding_chunks_query_key_matmul` + `_chunk` (`modeling_longformer.py:702-790`) — `as_strided` sliding-window with diagonal scoring |
| `longt5`     | longformer-derived |
| `led`        | longformer encoder + BART decoder |

**Nuance**: shard 08 called these unsupported. I downgraded to partial because
they decompose from L1 ops (`as_strided` + einsum + masking). The work is
non-trivial L2 wrapper authoring, but every primitive exists.

### Swin V1 vs V2 mismatch

kb-nano `L2/swinv2_window_attention.py` is the V2 variant (cosine attention +
CPB MLP). Swin V1 uses additive learned-bias-table windowed attention.

Affected: `maskformer`, `maskformer_swin`, `swin`, `donut` (Swin V1).

**Nuance**: shard 08 called these unsupported; shard 02 implicitly composable
via attn_mask. Final: partial. Swin V1 = encoder_attention + bias-as-attn_mask;
decomposable but not directly mapped.

### Custom activations / location-variable conv (decomposable)

| folder       | the missing primitive | decomposes via |
|--------------|-----------------------|-----------------|
| `apertus`    | xIELU activation       | xIELU has Python torch fallback path; CUDA is experimental |
| `dac`        | Snake1d (`x + sin²(αx)/α`) | pure elementwise |
| `pe_audio`   | Snake1d                 | same |
| `qwen2_5_omni` | (already L4 — Snake-Beta resolved) | |
| `univnet`    | location-variable conv via `einsum("bildsk,biokl->bolsd", ...)` | einsum + L1 conv |
| `mistral3`   | `Mistral3PatchMerger` uses `F.unfold` for spatial patch merging | torch.nn.functional |
| `vits`       | `weight_norm` parametrization, rational quadratic spline | parametrize on top of L1 conv |
| `mimi`       | RVQ + `weight_norm` parametrization + asymmetric padding conv | composable |
| `levit`      | BatchNorm1d (kb-nano has BatchNorm2d only) | trivial L1 wrapper to add |
| `slanet/slanext` | `nn.GRUCell` (kb-nano has lstm.py, no GRU) | small L1 wrapper |
| `layoutlmv3` | CogView softmax (`softmax((scores/α − max(scores/α)) * α)`) + rel_pos bucket | composable from softmax + bias add |

### per-dim / custom scaling

| folder              | nuance |
|---------------------|--------|
| `timesfm`/`timesfm2_5` | `F.softplus(self.scaling) * 1.4427/sqrt(d)` per-dim; kb-nano L2/attention only takes scalar scale. Pure tensor math = partial. |
| `olmo`/`olmoe`      | `clip_qkv.clamp_(min=-clip,max=clip)` — kb-nano L2/attention has no clip_qkv knob. Trivial wrapper to add = partial. |

### Subtle correctness-affecting differences (silent breakage)

| folder      | nuance |
|-------------|--------|
| `nanochat`  | uses `rotate_half = (x2, -x1)` while kb-nano L1/rotary_emb.py uses `(-x2, x1)`. Mathematically swappable BUT requires consistent treatment across q+k+cos+sin or outputs differ silently. Marked composable but flagged. |

### Vision encoder gaps (depending on multimodal model)

| folder                 | nuance |
|------------------------|--------|
| `videollama3`          | vision encoder inherits Siglip, NOT Qwen2-VL; uses separate Q/K/V (siglip-style) + 2D vision RoPE + cu_seqlens varlen. Map to `siglip_attention.py` + `vision_rotary_emb.py` + `flash_attn_varlen.py`. Easy classification trap — shard 15 caught it. |

---

## Cross-shard consistency disagreements (resolved here)

| topic | shard A position | shard B position | my resolution |
|-------|------------------|------------------|---------------|
| Bloom vs Falcon ALiBi | shard 02: bloom composable (cited nonexistent `alibi_slopes`) | shard 05: falcon unsupported | both **partial** (ALiBi via attn_mask works but isn't native) |
| Longformer/Longt5/Maskformer | shard 08: unsupported | (no other shard) | **partial** (chunked attn = `as_strided` + einsum, decomposable) |
| wav2vec2_conformer family | shard 16: unsupported | shard 13: seamless_m4t partial | all **partial** (conformer rel_shift = index gather + bmm) |
| Swin V1 (donut, maskformer_swin, swin) | mixed | mixed | **partial** (encoder_attention + attn_mask bias trick) |
| timesfm/timesfm2_5 | shard 14: unsupported | (no other) | **partial** (per-dim softplus scale = pure tensor math) |
| nystromformer | shard 10: unsupported | critical_reaudit_A: composable | **unsupported** (iterative inv loop is novel even if elementary) |
| qwen2_5_vl L4 | shard 12: yes L4 | (none) | **composable** (kb-nano L4 is encoder-only for HunyuanVideo) |
| falcon_mamba L4 | shard 05: yes L4 | (none) | **composable** (extra `rms_forward` on B/C/dt not in kb-nano L4/mamba.py) |

---

## Renderer bug rows (52 cases — separate from verdicts)

`md_to_tex.py` leaves textual sibling-class references in the mapping column
when a wiring class composes another wiring class. Examples:

- `ernie4_5_vl_moe`: `Ernie4_5_VLMoeMoeBlock → Ernie4_5_VLMoeSparseMoeBlock + L2/shared_expert_moe.py`
- `d_fine`: `DFineModel → DFineConvEncoder`
- `pi0`: `PI0ForConditionalGeneration → PI0Model`
- `dac`: `DacResidualUnit → Snake1d`

Fix: re-render after correcting the renderer's resolution pass to substitute
sibling classes with their kernel sets. Status verdicts are unaffected.

Files referenced in tex but missing in kb-nano (3):
- `L1/sparse_attn_indexer.py` and `L2/sparse_attn_indexer.py` (used in
  `glm_moe_dsa`)
- `L1/batch_norm1d.py` (used 3× in `phi4_multimodal` audio classes)
- `L2/rtdetrv2_encoder_layer.py` and `L3/rtdetrv2_encoder_layer.py` (used in
  `d_fine`, `ppdoclayoutv2`, `rt_detr`, `rt_detr_v2`)

---

## Coverage numbers — three honest framings

| framing                                              | %     | what it means |
|------------------------------------------------------|------:|---------------|
| Strict: maps to existing kb-nano L1/L2/L3 directly   | 64.9% | every compute class has an actual kb-nano kernel |
| With torch.nn fallback (decomposable from L1 + torch)| 92.7% | every primitive exists somewhere (kb-nano or torch); some need new L2 wrappers |
| **Paper claim (current)**                            | 96.2% | "every L1 op exists" — uses the loose `reclassify_A.md` definition |

The paper's 96.2% is defensible only under the loosest reading. A reviewer
counting `partial` rows in Table 9 will arithmetically derive the 92.7% number
and likely ask why the headline differs.

The biggest narrative risk is the unsupported count: paper says 7, reconciled
audit says 31. The 24 additional unsupported folders are mostly:
- 5 external CUDA: `dinat`, plus the original 4
- 8 external library / AutoBackbone-deps
- 16 genuinely novel compute (DaViT, FocalModulation, Funnel, sparsemixer,
  Perceiver+aligner, AltUp, etc.)

A subset of these (e.g., `nystromformer`, `phi4_multimodal`, `nanochat`-style
silent breakage) is judgment-call; a careful reviewer might accept partial.

---

# v4 RECONCILIATION (after cross-verification round)

After the 5 cross-verification agents reported, applied additional adjustments:

**Cross-verifier round 2 outcomes:**
- L4 (27 folders): all confirmed L4 ✓ (with notes that some L4 cover only the text/main path of multimodal models — gemma4, qwen2_5_omni, llama4, siglip2)
- Unsupported (31 folders): 21 downgraded to partial, 10 confirmed unsupported
- Partial-A (59 folders): 58 confirmed partial, 1 demoted to unsupported (`ibert` — IntLayerNorm/IntGELU/IntSoftmax custom integer arith)
- Partial-B (59 folders): all 59 confirmed partial
- Composable sample (63 folders): 58 confirmed, 4 downgraded to partial, 1 downgraded to unsupported (`diffllama` — differential attention dual-softmax + lambda subtraction)

**Final v4 counts (425 folders):**

| status         | count | %     |
|----------------|------:|------:|
| `kb_nano_l4`   |    27 |  6.4% |
| `composable`   |   244 | 57.4% |
| `partial`      |   142 | 33.4% |
| `unsupported`  |    12 |  2.8% |

**v4 coverage:**
- Strict (L4 + composable): **271/425 = 63.8%**
- Loose (+ partial / torch fallback): **413/425 = 97.2%**
- Unsupported: **12/425 = 2.8%**

**v4 unsupported list (12 folders):**
1. `mra` — kernels-community/mra CUDA (sparse attention)
2. `rwkv` — kernels-community/rwkv CUDA (v4 wkv)
3. `xlstm` — external xlstm library (mLSTM kernels)
4. `yoso` — kernels-community/yoso CUDA (LSH)
5. `dinat` — `natten` library (neighborhood attention CUDA)
6. `timm_backbone` — external `timm`
7. `timm_wrapper` — external `timm`
8. `fast_vlm` — external `timm` (FastViT)
9. `gemma3n` — external `timm` + custom audio Conformer + AltUp/Laurel
10. `layoutlmv2` — `detectron2` library
11. `ibert` — IntLayerNorm/IntGELU/IntSoftmax integer-arithmetic emulation
12. `diffllama` — differential attention (dual softmax + learnable lambda subtraction; novel attention formulation)

**Cross-verifier additions to the original 7 paper-claimed unsupported:**
- `dinat` (natten was overlooked)
- `fast_vlm` (timm dep)
- `gemma3n` (timm + Conformer + AltUp)
- `layoutlmv2` (detectron2 — was already in audit summary as partial-detectron2)
- `ibert` (integer arith emulation)
- `diffllama` (novel differential attention)

**Cross-verifier removed from paper's unsupported list:**
- `reformer` — cross-verifier (shard 12) reclassified to partial. LSH/Local self-attention with stable_argsort + chunk + custom autograd `ReverseSort` Function decomposes from torch primitives (matmul + sort + scatter + chunked attention). Bespoke and complex but no custom CUDA required.

## Three honest framings of coverage

| framing | % | what it means |
|---------|---|---------------|
| Strict: maps directly to existing kb-nano L1/L2/L3 | **63.8%** | every compute class has an actual kb-nano kernel |
| Loose: decomposable from L1 + standard PyTorch | **97.2%** | every primitive exists somewhere; some need new L2 wrappers |
| Paper claim (current) | 96.2% | uses the loose `reclassify_A.md` definition |

**The paper's 96.2% is between strict and loose.** It's defensible as a reasonable
midpoint reading. The unsupported number (paper: 7, reconciled: 12) is the
specific risk: 5 additional folders that the paper didn't flag (`dinat`,
`fast_vlm`, `gemma3n`, `ibert`, `diffllama`) genuinely need work that the
paper didn't acknowledge, with `dinat` (natten CUDA) being the most clearly
missed.

## Files

- `MENTOR_REVIEW_full_audit.tex` — full standalone tex (1927 rows, 425 folders)
- `hf_coverage_rows.tex` — paper-input version (3779 lines, no document wrappers)
- Markdown shards: `tools/manual_audit_shard_*.md` (regenerated from agent JSONs + cross-verifier corrections)
- Original shards backed up: `tools/_backup_20260507_153746/`
- Reclassify files moved aside (would have re-applied loose def): `tools/_backup_20260507_153746/reclassify_*.md`, `critical_reaudit_*.md`
- This doc: `REAUDIT_NOTES.md`
- JSON snapshots: `/tmp/reaudit_final_{v2,v3,v4}.json`, `/tmp/reaudit_results/shard_*.json`, `/tmp/xverify_results/*.json`

## Round-by-round summary

| round | what | result |
|-------|------|--------|
| Round 1 (16 shards) | First independent re-audit | 28 L4, 227 composable, 72 partial, 44 unsupported (some inconsistencies between shards on ALiBi, Conformer, Swin V1) |
| My personal verify (50+) | Source-read rigor | Reduced unsupported from agent's 44 to 31 (downgraded ALiBi/Conformer/chunked/etc. to partial) |
| Round 2 (5 cross-verifiers) | Independent re-verification | Reduced unsupported from 31 to 10 (over-strict in round 1 + my pass), added 2 new (ibert, diffllama) → final 12; added 4 partial promotions |
| Final v4 | Reconciled | 27 L4 / 244 composable / 142 partial / 12 unsupported |

---

# v7 RECONCILIATION (after Phase 2 full source-read of partial+unsupported)

After Phase 2 dispatched 6 verifiers covering all 154 partial+unsupported folders + 30 of 244 composable, plus my personal source-reads + cross-slice consistency analysis, applied v7 corrections.

## Cross-slice consistency disagreements (resolved)

The 6 Phase-2 verifiers each independently re-read source for their assigned slice. They disagreed on 4 recurring patterns. I personally read source for the disputed cases and applied a unified rule:

### 1. AutoBackbone (`load_backbone`) treatment
- Cross-verifier round 1: chmv2/conditional_detr → partial
- Slice 2: chmv2/conditional_detr/dab_detr/detr/deformable_detr → unsupported
- Slice 4: omdet_turbo/oneformer → unsupported
- Slice 6: tvp/zoedepth → unsupported

**My resolution: partial for all 9.** `load_backbone(config)` is HF runtime-routing infrastructure; the underlying compute (conv2d, batch_norm2d, relu) all exists in kb-nano L1. No new compute primitive needed; just a wiring abstraction.

### 2. Slice 3 promotions to composable
Slice 3 promoted 10 folders. Personal verification:
- accept (3): `kyutai_speech_to_text` (Moshi+Llama delegation), `mgp_str` (ViT-style), `mobilebert` (BERT-style + NoNorm composable)
- reject (7): `fuyu`, `glm`, `gpt_neox`, `glm_image` (partial_rotary+LayerNorm gap); `hubert`, `mimi` (weight_norm); `mistral3` (F.unfold)

### 3. apertus
- Slice 1: confirm_unsupported (xIELU custom)
- Slice 2: demote_to_unsupported (xIELU + Nemotron MLP)
- Slice 5: confirm_partial (Python torch fallback)

**My resolution: partial.** xIELU has a Python torch fallback in HF; CUDA wheel optional. Decomposable from elementwise torch ops.

### 4. deepseek_v4
- Cross-verifier round 1: downgrade_to_partial (decomposes from torch)
- Slice 2: demote_to_unsupported (multiple novel paths)

**My resolution: partial.** HCA/CSA/Indexer all use pure torch ops (matmul + ReLU + topk + softmax + linear). Novel WIRING, not novel primitives.

## v7 final counts (425 folders)

| status         | count | %     |
|----------------|------:|------:|
| `kb_nano_l4`   |    27 |  6.4% |
| `composable`   |   241 | 56.7% |
| `partial`      |   145 | 34.1% |
| `unsupported`  |    12 |  2.8% |

**Coverage:**
- Strict (L4 + composable): **268/425 = 63.1%**
- Loose (+partial / torch fallback): **413/425 = 97.2%**
- Unsupported: **12/425 = 2.8%**

The 12 unsupported (unchanged from v4): `diffllama`, `dinat`, `fast_vlm`, `gemma3n`, `ibert`, `layoutlmv2`, `mra`, `rwkv`, `timm_backbone`, `timm_wrapper`, `xlstm`, `yoso`.

## Evidence-claim hallucination check
Spot-sampled 10 random `hf_evidence` file:line refs from Phase 2 slices, ran `sed` on each cited file:line. **0/10 hallucinations** — Phase 2 verifier evidence is sound.

## Files

- `MENTOR_REVIEW_full_audit.tex` — full standalone tex (1927 rows, 425 folders)
- `hf_coverage_rows.tex` — paper-input version (3782 lines, 425 folders)
- Markdown shards regenerated: `tools/manual_audit_shard_*.md`
- Notes: `REAUDIT_NOTES.md` (this file)
- Verifier audit log: `VERIFIER_AUDIT.md`
- Per-folder evidence: `audit_evidence.csv` (425 rows × 12 columns)
- v7 status JSON: `_reaudit_final_v7.json`
- Backups: `tools/_backup_20260507_153746/`

## Coverage of audit work

- **First-pass agents (16 shards)**: covered all 425 folders.
- **Cross-verify round 1 (5 verifiers)**: 239 second-look folder-touches.
- **Phase 2 verifiers (6 slices)**: 162 deep-source-read folder-touches.
- **Personally source-read by me**: ~93 folders (cumulative across all batches).
- **Total folder-touches**: 425 + 239 + 162 + 93 = 919 (with significant overlap; many folders touched 2–4 times).
- **Folders with ≥2 independent agent touches**: 401 (everyone except those single-passed in shards 7/9 retries).

## Outstanding work

- Slice 7 (20 HF folders missing from original audit: mask2former, code_llama [tokenizer-only], lw_detr, mm_grounding_dino, etc.) — running in background
- Slice 8 (40 random composable folders sample-verify) — running in background

When those return, integrate, possibly v8 reconciliation, then final.


---

# v10 RECONCILIATION — final state with full transparency

After the user pushed back on whether I'd taken shortcuts, I went back and verified the interleaved-RoPE candidates that slice 8 had identified as a recurring pattern. Found 6 more that slice 8 didn't sample (gptj, cohere, codegen, glm4v, glm4, ernie4_5_vl_moe). Now corrected.

## v10 final counts (445 folders)

| status         | count | %     |
|----------------|------:|------:|
| `kb_nano_l4`   |    27 |  6.1% |
| `composable`   |   243 | 54.6% |
| `partial`      |   163 | 36.6% |
| `unsupported`  |    12 |  2.7% |

**Coverage:**
- Strict (L4 + composable): **270/445 = 60.7%**
- Loose (+partial / torch fallback): **433/445 = 97.3%**
- Unsupported: **12/445 = 2.7%**

## Honest self-assessment of the audit

### What I did personally (source-read both HF and kb-nano)

~99 unique folders. Mostly the 50+ I read in batches earlier, plus the ~20 spot-checks during slice audits, plus the 13 interleaved-RoPE follow-up reads.

The folders I source-read are listed in `audit_evidence.csv` column `i_personally_read = "yes"`. Anyone can grep that column to verify.

### What I delegated to subagents

The remaining ~346 folders were audited only by subagents:
- 16 first-pass shards: every folder touched once.
- 5 cross-verifiers (round 1): 239 second-touches.
- 6 phase-2 verifiers (slices 1-6): 162 third-touches.
- 1 missing-folders verifier (slice 7): 20 first-touches for previously-missed folders.
- 1 sample-verifier (slice 8): 40 fourth-touches.

Total: 425 first-pass + 239 cross-verify-r1 + 162 phase-2 + 20 slice-7 + 40 slice-8 + ~99 personal = **985 folder-touches, with significant overlap.** Most folders were touched by 2-3 independent agents.

### Where I took shortcuts and why

1. **I trusted agent verdicts on ~346 folders without re-reading source myself.** Justification: agents produced JSON with file:line evidence; I sample-verified 10 of those evidence claims (0/10 hallucinated); cross-verifier rounds caught major inconsistencies. But it's still delegation, not personal verification.

2. **I almost missed the interleaved-RoPE issue across the full corpus.** Slice 8 caught 7 in its sample of 40. Without the user's pushback, I would have stopped there, missing 6 more (`gptj`, `cohere`, `codegen`, `glm4v`, `glm4`, `ernie4_5_vl_moe`). After the pushback, I scanned all 244 composable folders for the pattern + read source for each candidate. **This is the kind of recurring-pattern follow-up that should be automatic when a verifier finds a systemic miss.**

3. **I made some methodology calls that the agents disagreed on.** Documented:
   - `load_backbone` → partial (not unsupported): cross-verifier-r1 said partial, slice 2/4/6 said unsupported. I picked partial for consistency. A reviewer could legitimately argue the other way.
   - `partial_rotary` → partial (not composable): slice 3 said composable, my v5 fix said partial. I picked partial. Same in spirit for interleaved RoPE.
   - `apertus` → partial (not unsupported): xIELU has Python torch fallback, kept partial.
   - `deepseek_v4` → partial (not unsupported): HCA/CSA decomposes from torch primitives.

4. **I built infrastructure shortcuts instead of always doing fresh reads.** I wrote regex helpers (`find_missing_class`), grep-based scans, and aggregation scripts. This is efficient but means my "verification" was sometimes "agent said X, my evidence-check passed, accept X."

### Are agents consistent across passes?

**Mostly, with two main inconsistency clusters:**

1. **`load_backbone` (AutoBackbone)**: cross-verifier-r1 said partial; slices 2/4/6/7 said unsupported. **9 folders affected.** I resolved with a single rule (partial), documented in VERIFIER_AUDIT.md.

2. **Strict vs loose composable definition**: round 1 used loose (any L1 op exists ⇒ composable); reclassify_A pass made it even looser; slice 3 partially reverted; my v5/v7/v9/v10 fixes apply strict ("must have L2 wrapper or non-trivial composition path"). The 78 `reclassify_A` original promotions are mostly invisible in v10 because we re-rendered from agent JSONs (which used strict).

3. **Hallucination check**: 0/10 random `hf_evidence` file:line refs were hallucinated. Agents are reading source, not making things up.

### Files (final state)

| file | purpose |
|---|---|
| `MENTOR_REVIEW_full_audit.tex` | full standalone tex (1994 rows, 445 folders) |
| `hf_coverage_rows.tex` | paper-input version (3947 lines) |
| `REAUDIT_NOTES.md` | this file (full reconciliation, methodology, self-assessment) |
| `VERIFIER_AUDIT.md` | per-slice agent audit + hallucination check + interleaved-RoPE follow-up |
| `CONSISTENCY_AUDIT.md` | Phase 1 pattern groups |
| `NUMBER_DRIFT_RECONCILIATION.md` | denominators (442 / 445 / 448) |
| `audit_evidence.csv` | per-folder evidence trail (425 rows pre-slice-7; updating to 445) |
| `_reaudit_final_v{2,5,6,7,8,9,10}.json` | per-version status snapshots for version control |
| Markdown shards: `tools/manual_audit_shard_{01..17}.md` | regenerated from agent JSONs + my overrides |
| Backups: `tools/_backup_20260507_153746/` | original (pre-reaudit) shards + reclassify files |

### Comparison vs paper claim

| metric | paper | v10 reaudit | delta |
|---|---|---|---|
| denominator | 421 (fictional) | 445 (filesystem-grounded) | +24 |
| L4 + composable (strict) | 96.2% | 60.7% | -35.5 pp |
| L4 + composable + partial (loose) | (paper dropped this) | 97.3% | — |
| unsupported count | 7 | 12 | +5 |

The 96.2% in the paper sits between my strict (60.7%) and loose (97.3%). Defensible only under the very loose `reclassify_A.md` definition. The 12-vs-7 unsupported gap is due to:
- adding `dinat` (natten library — paper missed)
- adding `fast_vlm`, `gemma3n` (timm dep — paper missed)
- adding `ibert` (integer arithmetic — paper missed)
- adding `diffllama` (differential attention — paper missed)
- removing `reformer` from unsupported (cross-verifier confirmed it decomposes from torch primitives)


---

# v11 RECONCILIATION — final final state

After the user prompted me to verify the 445 denominator, I found the math was off by 2 because the original audit treated `esm/` and `donut/` as single rows, ignoring their second modeling files (`esmfold` and `donut_swin`). Both added in v11.

## Stale-file cleanup
Moved 15 pre-reaudit files to `_stale_pre_reaudit/`:
- `coverage_summary.md`, `audit_methodology.md`, `MENTOR_REVIEW_audit_methodology.md` (old methodology docs with stale numbers)
- `AGENT_SPOT_CHECK.md`, `AUDIT_FINDINGS_FIRST_50.md`, `LESSONS_LEARNED.md` (early audit notes)
- `hf_architecture_operator_coverage.csv`, `hf_model_inventory.csv`, `kb_nano_operator_catalog.csv`, `unsupported_operator_summary.csv` (original CSVs from pre-reaudit; superseded by `audit_evidence.csv`)
- `PAPER_APPENDIX_TABLE.tex`, `appendix_hf_coverage_rows.tex`, `appendix_hf_coverage_rows_texttt.tex`, `MENTOR_REVIEW_10_rows.tex`, `MENTOR_REVIEW_preview_partial.tex` (intermediate tex experiments from earlier sessions)

## v11 final counts (447 folders) — superseded by v12 below

| status | count | %     |
|--------|------:|------:|
| `kb_nano_l4` | 27 | 6.0% |
| `composable` | 243 | 54.4% |
| `partial`    | 165 | 36.9% |
| `unsupported`| 12  | 2.7% |

**v11 coverage (superseded):**
- Strict (L4 + composable): 270/447 = 60.4%
- Loose (+partial): 435/447 = 97.3%
- Unsupported: 12/447 = 2.7%

**Use the v12 table at the bottom of this document for the canonical numbers.**

## Files (v12 canonical)

| file | purpose |
|---|---|
| `MENTOR_REVIEW_full_audit.tex` | full standalone tex (1999 rows, 447 folders) |
| `hf_coverage_rows.tex` | paper-input version (3962 lines) |
| `REAUDIT_NOTES.md` | this file (full reconciliation, methodology, self-assessment) |
| `VERIFIER_AUDIT.md` | per-slice agent audit |
| `CONSISTENCY_AUDIT.md` | Phase 1 pattern groups |
| `NUMBER_DRIFT_RECONCILIATION.md` | denominator (447) |
| `audit_evidence.csv` | per-folder evidence trail (447 rows × 12 cols) |
| `_reaudit_final_v{5,9,10,11}.json` | per-version snapshots |
| `_hf_coverage_rows_pre_reaudit_*.tex` | original (pre-reaudit) backup |
| `_stale_pre_reaudit/` | 15 stale files moved aside (not deleted) |
| `tools/manual_audit_shard_{01..17}.md` | regenerated markdown shards |
| `tools/_backup_20260507_153746/` | original pre-reaudit shards + reclassify files |

## High-importance model status (v11 spot-check)

Final status of 70+ critical/popular models:
- **L4 (already pipelines):** llama, llama4, qwen3_vl, gemma4, mixtral, deepseek_v3, dinov3_vit, sam3, whisper, qwen2_vl, qwen2_5_omni, mamba, mamba2, jamba, bitnet, convnextv2, swinv2 + 9 more
- **Composable (clean):** mistral, qwen2, qwen3, qwen3_moe, gemma, gemma2, gemma3, falcon_h1, bert, bart, clip, siglip, vit, blip_2, llava, llava_next, idefics, idefics3, qwen2_5_vl, smolvlm, paligemma, colpali, colqwen2, recurrent_gemma, dinov3_convnext, swin2sr, starcoder2, gpt2, dbrx, deepseek_v2, mobilebert, mgp_str + many more
- **Partial (decomposable, no L2 wrapper):** phi, phi3, phi4_multimodal, t5, falcon (ALiBi), wav2vec2, swin (V1), vits, glm/glm4 (interleaved RoPE), cohere/cohere2, olmo/olmoe, codegen, gptj, gpt_neox, mask2former, mm_grounding_dino, idefics2, sam (V2 dec rel-pos), zamba, deepseek_v4, donut_swin, esmfold, etc.
- **Unsupported (12):** diffllama, dinat, fast_vlm, gemma3n, ibert, layoutlmv2, mra, rwkv, timm_backbone, timm_wrapper, xlstm, yoso

## Audit work coverage (final)

| audit pass | folders touched |
|---|---:|
| First-pass (16 shards) | 425 |
| Cross-verify round 1 (5 verifiers) | 239 |
| Phase 2 verifiers (slices 1-6, 8) | 222 |
| Slice 7 (missing folders) | 20 |
| v11 additions (esmfold, donut_swin) | 2 |
| **Personally source-read by me (cumulative)** | **107** |

Total folder-touches: ~1010 (with significant overlap; many folders touched 2-4 times across rounds).


---

# v12 RECONCILIATION — partial-rotary consistency sweep

## What triggered v12

User asked for a comprehensive guideline-by-guideline re-audit. Going through
each of the 12 mandatory rules in the audit prompt I found:

- **Rule 3 (silu_and_mul)**: 4 SwiGLU classes mapped to bare `silu.py`; fixed to `silu_and_mul.py`. (dinov2.Dinov2SwiGLUFFN, dinov2_with_registers.Dinov2WithRegistersSwiGLUFFN, cohere2_vision.Cohere2VisionMultiModalProjector, ernie4_5_vl_moe.Ernie4_5_VLMoeVisionMLP). Status unchanged.
- **Rule 3 (gelu_and_mul)**: 2 GeGLU classes mapped to bare `gelu.py`; fixed to `gelu_and_mul.py`. (clvp.ClvpGatedLinearUnit, cpmant.CpmAntDenseGatedACT). Status unchanged.
- **Rule 5 (vision RoPE)**: 3 classes used `rotary_emb.py` for 2D vision RoPE; fixed to `vision_rotary_emb.py`. (edgetam_video x2, efficientloftr). Status unchanged.
- **Rule 5 (YaRN)**: longcat_flash.LongcatFlashRotaryEmbedding inherits DeepseekV3RotaryEmbedding (YaRN-scaled when `rope_type="yarn"`); fixed `rotary_emb.py` → `yarn_rotary_emb.py` and corrected the misleading "interleaved supported" rationale. Status unchanged (partial).
- **Rule 5 (partial-rotary consistency)** — the load-bearing v12 fix. The audit had marked phi/persimmon/glm/etc. as `partial` because of partial-rotary, but had left bamba/glm4v_moe/laguna/musicflamingo/recurrent_gemma/solar_open as `composable` despite each having `partial_rotary_factor < 1.0`. **kb-nano L1/rotary_emb.py rotates the full head_dim**; partial-rotary requires either (a) external q_rot/q_pass slicing in user code, or (b) the Gemma4-style `Gemma4ProportionalRotaryEmbedding` subclass. Neither is part of `L2/attention.py`'s default forward. Demoted these 6 folders to `partial` for consistency with the existing rule.
- **Rule 6 (MLA mapping)**: glm4_moe_lite.Glm4MoeLiteAttention is actually MLA (q_a_proj+q_b_proj LoRA, kv_a_proj_with_mqa, qk_rope_head_dim/qk_nope_head_dim) but was mapped to `L2/attention.py`. Fixed to `L2/deepseek_mla_attention.py`. Status unchanged (composable).
- **Rule 11 (sibling-attention)**: 4 bare `*Attention` classes that wrap `*SelfAttention` were tagged `[compute]`; re-tagged `[wiring]`. (bridgetower, roformer, tapas, xlm_roberta_xl). Render-identical bookkeeping.
- **Rules 4, 7, 8, 9, 10**: scanned, no real issues found. Norm variants (T5/BitNet/Gemma RMSNorm) are correctly partitioned. Linear variants (bitnet_linear, fp8_linear) are used where needed. MoE expert kernels are consistently mapped (mxfp4_moe → gpt_oss only; moe_grouped_gemm + topk_softmax across the standard MoE family). No vision MLPs were misrouted to llama_mlp.
- **Rule 12 (skipped classes)**: 1 minor case (xlstm.xLSTMPreTrainedModel listed as compute), in an already-unsupported folder. No effect.

## v12 final counts (447 folders)

| status | count | %     |
|--------|------:|------:|
| `kb_nano_l4` | 27 | 6.0% |
| `composable` | 237 | 53.0% |
| `partial`    | 171 | 38.3% |
| `unsupported`| 12  | 2.7% |

**Coverage:**
- Strict (L4 + composable): **264/447 = 59.1%**
- Loose (+partial): **435/447 = 97.3%**
- Unsupported: **12/447 = 2.7%**

## v11 → v12 deltas

| folder | v11 | v12 | reason |
|---|---|---|---|
| bamba | composable | partial | `partial_rotary_factor=0.5` hardcoded in config |
| glm4v_moe | composable | partial | `partial_rotary_factor=0.5` default; text RoPE is NeoX with q_rot/q_pass split (not interleaved as earlier audit said) |
| laguna | composable | partial | full_attention layers use `partial_rotary_factor=0.5` |
| musicflamingo | composable | partial | Qwen2 LM rope_parameters set `partial_rotary_factor=0.2` |
| recurrent_gemma | composable | partial | Griffin SDPA has q_rot/q_pass split with `partial_rotary_factor=0.5`; the earlier audit's "wrap that uses the same L1 rotary kernel" claim was incorrect — kb-nano L2/attention.py does not slice |
| solar_open | composable | partial | Inherits GLM4-MoE BC default `partial_rotary_factor=0.5` |

The 6 demotions move strict from 60.4% to 59.1%. Loose and unsupported are unchanged.

## What v12 does NOT change

- The 27 L4 promotions (verified — all map to existing `tasks/baseline/L4/<file>.py` files).
- The 12 unsupported list — all source-verified to need external libs (timm/natten/detectron2/xlstm) or kernels-community CUDA kernels (mra/rwkv/yoso) or bespoke compute (diffllama/ibert/gemma3n).
- The denominator (447) — every PyTorch `modeling_*.py` except `auto/`.
- The 9 file-mapping fixes from the v11→v12 sweep are reflected in the rendered tex; the kb-nano kernel column is now strictly correct per guidelines 3 and 5.

## Verification (post-v12)

- `_reaudit_final_v11.json` (file kept under v11 name; contains v12 final_status — 447 entries, includes `v12_demotions` list)
- `audit_evidence.csv` (447 rows; demoted rows annotated in `p2_rationale`)
- `hf_coverage_rows.tex` (re-rendered; 447 entries, 12 unsupported / 27 L4 / 237 cmark / 171 P)
- Triple cross-check: 0 mismatches across json/csv/tex.
