# Shard a-d notes

**Range:** folders `afmoe` through `dpt` (96 folders, 88 PyTorch modeling files + 9 NO_PT_MODELING).

**Output CSV:** `/home/olu/kb_nano/audits/hf_transformers_coverage/shards/shard_a-d_raw.csv`

## Row count and status breakdown

- Total rows produced: **97**
  - 88 modeling-file rows
  - 9 NO_PT_MODELING folder rows
- Status breakdown:
  - `kb_nano_l4`: **4** — `deepseek_v2`, `deepseek_v3`, `deepseek_v4`, `dinov3_vit`
  - `composable`: **64**
  - `partial`: **19**
  - `not_inference_required`: **10** (9 NO_PT_MODELING folders + `auto/modeling_auto.py` which is a registry-only module)
  - `unsupported`: **0**

The partial rate (19/87 modeling rows = 21.8%) is higher than the pilot rate (1/15 = 6.7%) but is concentrated in two well-bounded gaps: (1) `nn.AdaptiveAvgPool*` head poolers in vision/audio models, and (2) `nn.ConvTranspose*` upsamplers in segmentation/depth/audio-decoder heads. Both are CLAUDE.md-spirit gaps but are the project's de facto convention (kb-nano L4s like mobilenetv4, sam3 use these via torch.nn directly). Pilot decisions #2 and #3 explicitly cover this case as `partial`. The audit a-d range happens to be vision/depth-heavy (beit, depth_anything, depth_pro, dpt, clipseg, dinov3_convnext, donut, dinat, bit, align, deimv2) which inflates the partial count vs. the all-LLM pilot.

There are **no `unsupported` rows** in this shard. Every load-bearing op has either a kb-nano kernel, a torch.nn fallback per kb-nano convention, or a torch builtin.

## Every `partial` row (paragraph each)

**align/modeling_align.py — `adaptive_avg_pool_2d`.** ALIGN's vision tower is an EfficientNet whose squeeze-excitation block uses `nn.AdaptiveAvgPool2d(output_size=1)` at line 297. There is no kb-nano L1 adaptive-avg-pool kernel. kb-nano's existing L4s (mobilenetv4, yolov10) use `nn.AdaptiveAvgPool2d` directly via torch.nn — the project's convention. Partial via torch fallback. **Confidence: high.**

**aria/modeling_aria.py — `multihead_attention`.** `AriaCrossAttention` (line 116) uses `nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)` for the vision-language projector. kb-nano has no L1 wrapper for `nn.MultiheadAttention`; the rest of the model (Llama-style decoder + MoE) is composable. Partial via torch.nn fallback. **Confidence: high.**

**audioflamingo3/modeling_audioflamingo3.py — `avg_pool_1d`.** Audio projector uses `nn.AvgPool1d(2, stride=2)` at line 299 to downsample the audio encoder output before feeding the LLM. No L1 kernel for 1D avg-pool. **Confidence: high.**

**autoformer/modeling_autoformer.py — `avg_pool_1d`.** The series-decomposition block in Autoformer uses `nn.AvgPool1d` at line 369 to extract the trend component. No L1 kernel for 1D avg-pool. The auto-correlation attention (FFT-based) is implemented from torch.fft + torch builtins; not a missing primitive. **Confidence: high.**

**beit/modeling_beit.py — `adaptive_avg_pool_2d` + `conv_transpose2d`.** `BeitForSemanticSegmentation` uses `nn.AdaptiveAvgPool2d` at line 973 (UperHead pyramid pool) and 4x `nn.ConvTranspose2d` at lines 1183-1189 and 1335-1341 (FPN upsampler). `BeitModel`, `BeitForImageClassification`, `BeitForMaskedImageModeling`, `BeitBackbone` are all composable. Only the `ForSemanticSegmentation` variant triggers the partial flag. Per pilot precedent (data2vec_vision row), this is `partial` for the file overall with a sub-variant note. **Confidence: high.**

**bit/modeling_bit.py — `adaptive_avg_pool_2d`.** `BiTForImageClassification` uses `nn.AdaptiveAvgPool2d((1, 1))` at line 666 in the head pooler. The ResNet-style backbone with GroupNorm is composable. Partial via torch.nn fallback per convention. **Confidence: high.**

**bridgetower/modeling_bridgetower.py — `multihead_attention`.** `BridgeTowerLinkTower` uses `nn.MultiheadAttention(config.hidden_size, config.hidden_size // 64)` at line 110 for the cross-modal bridge. No L1 wrapper for `nn.MultiheadAttention`. Partial. **Confidence: high.**

**canine/modeling_canine.py — `max_pool_1d`.** CANINE downsamples the 2D character attention mask to a 2D molecule attention mask via `torch.nn.MaxPool1d(...)` at line 798. No L1 1D max-pool kernel in kb-nano. Partial via torch fallback. **Confidence: high.**

**chmv2/modeling_chmv2.py — `conv_transpose2d`.** CHMv2 uses `nn.ConvTranspose2d(channels, channels, kernel_size=factor, stride=factor, padding=0)` at line 50 in its upsampling resize block. Per kb-nano canonical map, `conv_transpose2d` is UNSUPPORTED on origin/experiments — but the project's convention (sam3_fpn_conv.py, sam3_mask_decoder.py, cosyvoice3_hifigan.py) uses `nn.ConvTranspose1d/2d` directly. Treated as `partial` per pilot decision #3. **Confidence: high.**

**clap/modeling_clap.py — `adaptive_avg_pool_2d` + `adaptive_avg_pool_1d`.** CLAP's audio Swin-style tower head uses `nn.AdaptiveAvgPool2d(1)` at line 228 (SE-block); the audio model overall pooler uses `nn.AdaptiveAvgPool1d(1)` at line 775. Two partial flags, both head-only. RoBERTa text tower is composable. **Confidence: high.**

**clipseg/modeling_clipseg.py — `conv_transpose2d`.** CLIPSegDecoder uses 3x `nn.ConvTranspose2d` at lines 522, 529, 534 for upsampling the CLIP visual features back to image resolution for the segmentation mask. CLIP backbone is composable. Partial via torch.nn fallback per convention. **Confidence: high.**

**dac/modeling_dac.py — `conv_transpose1d`.** Descript Audio Codec decoder uses `nn.ConvTranspose1d(...)` at line 243 for upsampling the latent codes back to waveform. Per canonical map, `conv_transpose1d` is UNSUPPORTED on origin/experiments. Encoder is conv1d-based and composable. Partial via torch.nn fallback per pilot decision #3. **Confidence: high.**

**deimv2/modeling_deimv2.py — `adaptive_avg_pool_2d`.** DEIM-v2 (RT-DETR-v2 family detection model) uses an HGNetv2 backbone whose SE-block uses `nn.AdaptiveAvgPool2d` (caught by the AST extractor). The deformable attention path itself maps to kb-nano's `MultiScaleDeformableAttentionV2` directly (helper at line 224, grid_sample composed inside at line 259). Partial only for the backbone SE-block pooling. **Confidence: high.**

**depth_anything/modeling_depth_anything.py — `conv_transpose2d`.** Depth-Anything (DINOv2 backbone + DPT depth head) uses `nn.ConvTranspose2d(channels, channels, kernel_size=factor, stride=factor, padding=0)` at line 38 in the reassemble block. Backbone composable; head partial via torch.nn fallback. **Confidence: high.**

**depth_pro/modeling_depth_pro.py — `conv_transpose2d`.** Apple Depth Pro uses 3+ `nn.ConvTranspose2d` calls in its multi-scale depth decoder (lines 471, 778, 973). Backbone composable; head partial. **Confidence: high.**

**dinat/modeling_dinat.py — `adaptive_avg_pool_1d`.** Dilated Neighborhood Attention Transformer's `DinatModel.pooler` is `nn.AdaptiveAvgPool1d(1)` at line 572 (when `add_pooling_layer=True`). NATTEN's neighborhood-attention itself is implemented in HF as a Python loop using gather/window-extraction (pure torch, no missing primitive). Partial only for the head pooler. **Confidence: high.**

**dinov3_convnext/modeling_dinov3_convnext.py — `adaptive_avg_pool_2d`.** DINOv3 ConvNeXt variant uses `nn.AdaptiveAvgPool2d(1)` at line 228 in its head pooler. Backbone composable; head partial. (Note: kb-nano L4 `dinov3.py` is the ViT variant — not this ConvNeXt one.) **Confidence: high.**

**donut/modeling_donut_swin.py — `adaptive_avg_pool_1d`.** Donut document-VLM uses a Swin-V1 vision encoder; `DonutSwinModel.pooler` is `nn.AdaptiveAvgPool1d(1)` at line 824 (when `add_pooling_layer=True`). Backbone composable; head partial. **Confidence: high.**

**dpt/modeling_dpt.py — `conv_transpose2d`.** DPT (Dense Prediction Transformer for monocular depth) uses `nn.ConvTranspose2d(channels, channels, kernel_size=factor, stride=factor, padding=0)` at line 576 in the reassemble block. Backbone composable; head partial. **Confidence: high.**

## NO_PT_MODELING folders (10 entries — 9 NO_PT_MODELING + 1 registry-only)

| folder | reason |
|---|---|
| `bartpho` | Tokenizer-only; reuses BART/mBART. |
| `bert_japanese` | Tokenizer-only; reuses BERT. |
| `bertweet` | Tokenizer-only; reuses RoBERTa. |
| `byt5` | Tokenizer-only; reuses T5. |
| `code_llama` | Tokenizer-only; reuses Llama. |
| `cpm` | Tokenizer-only; reuses GPT-2/XLNet. |
| `deprecated` | Wrapper folder containing deprecated submodules. |
| `dialogpt` | No separate modeling; uses GPT-2. |
| `dit` | Document Image Transformer; reuses BEiT (no separate modeling file). |
| `auto` | `modeling_auto.py` is the AutoModel registry — class-name lookup tables only. No nn.Module architectures, no compute primitives. Counted as `not_inference_required` per the methodology spirit. |

## Modular DSL files in this shard

The following folders have `modular_*.py` (the modular-DSL source) in addition to the generated `modeling_*.py`:

aimv2, apertus, arcee, audio_spectrogram_transformer, audioflamingo3, aya_vision, bamba, bitnet, chmv2, cohere2, cohere2_vision, cohere_asr, colmodernvbert, colpali, colqwen2, csm, cwm, deepseek_v3, deepseek_v4, deepseek_vl, deepseek_vl_hybrid, dinov3_vit, dots1.

Per methodology section 13, the audit reads the generated `modeling_*.py` (the runtime artifact); the modular-DSL caveat is noted in the row's `notes` field where relevant.

## `kb_nano_l4` rows (4 entries — verification)

| row | matched L4 file |
|---|---|
| `deepseek_v2` | `tasks/baseline/L4/deepseek.py` (covers MLA + MoE family) |
| `deepseek_v3` | `tasks/baseline/L4/deepseek.py` (V2/V3/V4 family) |
| `deepseek_v4` | `tasks/baseline/L4/deepseek.py` + L1 `sparse_attn_indexer.py`, `indexer_k_cache.py`, `fp8_mqa_logits.py` for the indexer |
| `dinov3_vit` | `tasks/baseline/L4/dinov3.py` + L1 `dinov3_rope.py` (DINOv3 ViT) |

## `composable` rows that need a coordinator spot-check

These are subtle classifications I would specifically flag for the coordinator's 10% spot-check pass:

1. **`bart`, `blenderbot`, `blenderbot_small`, `bigbird_pegasus`, `blip/modeling_blip_text.py`, `dia`** — all marked `composable` but rely on `EncoderDecoderCache` semantics. Per pilot decision #1 the encoder-decoder cache is composable from two stacked `StoreKVCache`s. None of these files trigger a `partial` flag because their other ops are all standard.
2. **`big_bird`, `bigbird_pegasus`** — sparse attention is implemented in HF as a Python loop over gather/index/masked_fill on top of standard SDPA. No missing kernel, but the wiring effort is significant. Marked `composable`.
3. **`d_fine`, `deimv2`** — RT-DETR-v2 family detection. Both define `multi_scale_deformable_attention_v2` as a local helper (d_fine line 150, deimv2 line 224); kb-nano L1 `MultiScaleDeformableAttentionV2` matches directly. The `grid_sample` call inside the helper (d_fine line 185, deimv2 line 259) is composed inside the L1 kernel. (`deimv2` is `partial` only because of the HGNetv2 SE-block AdaptiveAvgPool, not because of deformable attention.)
4. **`doge`** — registers a custom `doge_flex_attention` via `ALL_ATTENTION_FUNCTIONS["doge_flex_attention"] = ...` at line 256. The dispatcher (line 324) accepts this string from the config. kb-nano does NOT have a flex_attention wrapper. However `eager_attention_forward` is also defined (line 184) as the fallback when `_attn_implementation` is sdpa/flash/eager — so as long as the user picks any of those three, kb-nano covers it. Marked `composable` per pilot decision #6 ("if any of the runtime-selectable variants is supported, mark as direct"). If a downstream user forces `_attn_implementation="doge_flex_attention"`, this would be partial — but that's a config-dependent runtime choice, not a structural gap.
5. **`deberta`, `deberta_v2`** — disentangled self-attention is implemented as raw `torch.matmul` + `softmax` + extra relative-position matmuls; does NOT go through `ALL_ATTENTION_FUNCTIONS`. All ops are still kb-nano L1 (`Matmul`, `Softmax`, `Linear`); composable.
6. **`bros`** — BROS adds a 2D positional bias on top of BERT (encoder). Uses standard SDPA + extra bias-add (passthrough); composable.

## `new_canonical_name_needed` flags

None encountered in this shard. The canonical map covers all load-bearing ops I needed to map. Two new canonical names were considered but reduced to existing ones:

- `multi_scale_deformable_attention_v2` (d_fine, deimv2) → mapped to existing `deformable_attention` canonical (kb-nano L1 `MultiScaleDeformableAttentionV2`); no new name needed.
- `doge_flex_attention` → captured as an `attention_impl:` extractor token; not a separate canonical, since the dispatcher falls back to eager.

## Things I couldn't fully resolve / want coordinator to spot-check

1. **`audio_spectrogram_transformer`** — there is no kb-nano AST L4 explicitly named for AST; modeling is essentially a ViT over spectrograms. Marked `composable` based on ViT-equivalent primitives. Spot-check: does kb-nano consider AST as a near-match for an existing ViT L4 (e.g. dinov3, siglip2)? If so, this could be re-categorized as `kb_nano_l4`.
2. **`autoformer`** — uses `torch.fft.rfft/irfft` for auto-correlation (the FFT op is not in the canonical map; it's a torch builtin, so passthrough). No L1 FFT kernel exists in kb-nano. I treated this as composable since FFT is a torch builtin, but a strict reading might want to flag it as partial. **Coordinator: please rule.**
3. **`chmv2`** — relatively new model; `modular_chmv2.py` is the source. I read both files. Conclusion holds (partial due to ConvTranspose2d).
4. **`dac`** — partial classification rests on the fact that the canonical map says `conv_transpose1d` is UNSUPPORTED on origin/experiments, but kb-nano L2 `cosyvoice3_hifigan.py` uses `nn.ConvTranspose1d` via torch.nn. So I treated it as `partial` (torch fallback works) rather than `unsupported`. If the coordinator decides "no L1 kernel + canonical-map says UNSUPPORTED" should be `unsupported`, this row flips.

## Time taken

Approximately 2.5 hours: ~30 min reading the methodology, pilot, and canonical map; ~30 min building the AST analyzer + sweep scripts; ~60 min reading individual modeling files and verifying load-bearing ops; ~30 min building the row generator and verifying line numbers.
