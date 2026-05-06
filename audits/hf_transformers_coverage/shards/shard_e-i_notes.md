# Shard e-i notes

## Totals

- Total rows: **92** (covers 91 folders; `esm` has two modeling files: `modeling_esm.py` and `modeling_esmfold.py`).
- Status breakdown:
  - `kb_nano_l4`: **1** (`gpt_oss`)
  - `composable`: **68**
  - `partial`: **21**
  - `unsupported`: **0**
  - `not_inference_required`: **2** (`gpt_sw3`, `herbert`)

The partial rate (21 / 92 ≈ 23%) is higher than the pilot rate (1/15) because this shard contains a large concentration of audio codecs (encodec, higgs_audio_v2_tokenizer, fastspeech2_conformer), wav2vec2-style audio encoders (hubert, granite_speech, granite_speech_plus), segmentation models (edgetam, edgetam_video, eomt, eomt_dinov3), and CNN backbones (efficientnet, hgnet_v2, focalnet, hiera) — each of which uses the same well-known kb-nano gaps (`ConvTranspose1d/2d`, `BatchNorm1d`, `BatchNorm3d`, `AdaptiveAvgPool1d/2d`, `nn.MultiheadAttention`, `nn.LSTM`, standalone `grid_sample`).

No new canonical-name flags surfaced. All ops outside the canonical map (`einsum`, `elu`, `leaky_relu`, `unfold`, `upsample`, `multihead_attention`, `mamba_scan`, `batch_norm_1d`, `batch_norm_3d`, `lstm`) were classifiable as torch builtins (passthrough), already-flagged kb-nano gaps, or covered by the existing mamba L2 (vLLM-imported) — none required new entries.

## Per-row commentary on partial / unsupported

### edgetam — `partial` (conv_transpose2d)
EdgeTam (Meta SAM2-derivative) image segmentation. The mask decoder uses two `nn.ConvTranspose2d` layers at lines 729-730 of `edgetam/modeling_edgetam.py` for resolution upscaling, in the inference forward path. kb-nano's L1 catalog has no ConvTranspose kernel; the project convention (cf. `tasks/baseline/L2/sam3_fpn_conv.py`, `tasks/baseline/L3/sam3_mask_decoder.py`) is to use `torch.nn.ConvTranspose2d` directly. Confidence: high.

### edgetam_video — `partial` (conv_transpose2d)
Video extension of edgetam. Same mask decoder pattern at lines 1793-1794 of `edgetam_video/modeling_edgetam_video.py`. The added video components (memory-fuser CXBlock, vision rotary embedding, memory attention) are all standard primitives. Confidence: high.

### efficientnet — `partial` (adaptive_avg_pool_2d)
Classic EfficientNet CNN. The Squeeze-and-Excite block at line 197 of `efficientnet/modeling_efficientnet.py` uses `nn.AdaptiveAvgPool2d(output_size=1)`. kb-nano has no L1 adaptive pool; existing L4s (mobilenetv4, yolov10) fall back to torch.nn — same convention applies. Confidence: high.

### emu3 — `partial` (batch_norm_3d)
Emu3 multimodal model. The text decoder is Llama-style and fully composable. The Emu3 VQVAE image tokenizer uses `nn.BatchNorm3d` at lines 452, 459 of `emu3/modeling_emu3.py` in the temporal residual blocks. kb-nano L1 has only `BatchNorm2d`; 3D batch-norm has no kernel and would fall back to torch.nn. Confidence: high.

### encodec — `partial` (conv_transpose1d, lstm)
EnCodec audio codec. The decoder upsampling uses `nn.ConvTranspose1d` (line 192, wrapped in `EncodecConvTranspose1d`) and the encoder/decoder include `nn.LSTM` (line 243, wrapped in `EncodecLSTM`). Neither op has a kb-nano L1 kernel. Both are in the inference path. Confidence: high.

### eomt — `partial` (conv_transpose2d)
EOMT (encoder-only mask transformer) for universal segmentation. `EomtScaleLayer` at line 923 uses `nn.ConvTranspose2d` at line 927 in the inference forward (scale upsampling between blocks). The grid_sample reference at line 119 (`sample_point`) is only called inside `pair_wise_*_loss` and the matcher (lines 256, 259, 476, 478) — training-only. Confidence: high.

### eomt_dinov3 — `partial` (conv_transpose2d)
DINOv3-backbone variant of eomt. Same `EomtDinov3ScaleLayer.conv1 = nn.ConvTranspose2d` at line 1121. Same grid_sample-in-matcher caveat. Confidence: high.

### fastspeech2_conformer — `partial` (conv_transpose1d, batch_norm_1d)
FastSpeech 2 + HiFi-GAN vocoder. The HiFi-GAN decoder uses `nn.ConvTranspose1d` at line 1395 for upsampling (inference path). The conformer batch-norm-conv layer uses `nn.BatchNorm1d` (lines 219, 504). `F.leaky_relu` is a torch builtin (not a partial; passthrough). Confidence: high.

### focalnet — `partial` (adaptive_avg_pool_1d)
FocalNet vision encoder. Pooler uses `nn.AdaptiveAvgPool1d(1)` at line 615 of `focalnet/modeling_focalnet.py` (head-only). Confidence: high.

### glm4v — `partial` (grid_sample)
GLM-4V multimodal LLM. The vision tower interpolates positional embeddings using `F.grid_sample` directly at line 198 of `glm4v/modeling_glm4v.py` — this is **outside** of deformable attention, so kb-nano's `rtdetrv2_deformable_attention.py` (which composes grid_sample internally) does not cover it. There is no standalone L1 grid_sample kernel. Confidence: high.

### glm4v_moe — `partial` (grid_sample)
Same grid_sample pattern at line 612 of `glm4v_moe/modeling_glm4v_moe.py`. Text decoder is sigmoid-routed MoE (composable). Confidence: high.

### glm_image — `partial` (grid_sample)
Same grid_sample pattern at line 254 of `glm_image/modeling_glm_image.py`. Confidence: high.

### granite_speech — `partial` (batch_norm_1d)
Granite Speech (audio + Granite text decoder). Audio projector uses `nn.BatchNorm1d` at line 226 of `granite_speech/modeling_granite_speech.py`. Otherwise composable (Conformer-style + Llama). Confidence: high.

### granite_speech_plus — `partial` (batch_norm_1d)
Same architecture as granite_speech with `nn.BatchNorm1d` at line 228. Confidence: high.

### groupvit — `partial` (batch_norm_1d)
CLIP-style GroupViT. The text projection MLP uses `nn.BatchNorm1d` at lines 1133 and 1139 of `groupvit/modeling_groupvit.py`. Vision tower + text encoder otherwise composable. Confidence: high.

### hgnet_v2 — `partial` (adaptive_avg_pool_2d)
HGNet v2 (PaddleClas-derived CNN). Classifier head uses `nn.AdaptiveAvgPool2d((1, 1))` at line 426 of `hgnet_v2/modeling_hgnet_v2.py`. Confidence: high.

### hiera — `partial` (adaptive_avg_pool_1d)
Hiera hierarchical ViT (Meta). Pooler uses `nn.AdaptiveAvgPool1d(1)` at line 806 of `hiera/modeling_hiera.py`. Confidence: high.

### higgs_audio_v2_tokenizer — `partial` (conv_transpose1d)
Higgs DAC-style audio codec. Decoder uses `nn.ConvTranspose1d` at line 343 of `higgs_audio_v2_tokenizer/modeling_higgs_audio_v2_tokenizer.py`. Confidence: high.

### hubert — `partial` (batch_norm_1d)
HuBERT audio encoder (wav2vec2-derivative). Feature extractor uses `nn.BatchNorm1d` at line 58 inside `HubertGroupNormConvLayer`. CTC loss is training-only; `log_softmax` is a torch builtin. Confidence: high.

### idefics2 — `partial` (multihead_attention)
Idefics2 VLM. `Idefics2MultiheadAttentionPoolingHead` (line 306) uses `torch.nn.MultiheadAttention` directly at line 313 of `idefics2/modeling_idefics2.py`. kb-nano has no wrapper around `nn.MultiheadAttention`; the underlying Q/K/V/O+SDPA path is covered by L1 ops, but as the audit treats `nn.MultiheadAttention` consumers as falling back to torch.nn (per pilot convention for `aria`, etc.), this is `partial`. Vision tower + text decoder otherwise composable. Confidence: high.

### informer — `partial` (batch_norm_1d, max_pool_1d)
Informer time-series transformer. The downsampling block at line 615 of `informer/modeling_informer.py` uses both `nn.BatchNorm1d` (line 618) and `nn.MaxPool1d` (line 620). Both are in the inference path. ProbSparse attention is composable from standard ops. Confidence: high.

## Modular DSL caveats

Many files in this shard have a `modular_*.py` companion. Per the methodology (section 13), the audit reads the generated `modeling_*.py` (the runtime artifact). The modular companion is noted in the row-level `notes` for: `aimv2`-style cases — for this shard: `eurobert`, `ernie4_5`, `ernie4_5_moe`, `exaone4`, `exaone_moe`, `exaone4_5`, `fast_vlm`, `flex_olmo`, `gemma`, `gemma2`, `gemma4_assistant`, `glm`, `glm4`, `granite`, `helium`, `idefics3`, `ijepa`, `internvl`. For each of these the modular file is a generation source; the runtime behavior is captured by the audited `modeling_*.py`.

## NO_PT_MODELING / wrapper-only folders

- `gpt_sw3` — tokenization-only folder; uses GPT-2 architecture. Marked `not_inference_required`.
- `herbert` — Polish BERT tokenizer-only folder; uses BERT architecture. Marked `not_inference_required`.
- `encoder_decoder` — generic wrapper; coverage depends on the wrapped sub-models. Marked `composable` (only adds shift-tokens-right and a tied projection linear).
- `fast_vlm`, `fuyu`, `gemma4_assistant`, `glm46v` — thin VLM/assistant wrappers around AutoModel + small projectors. Marked `composable`.

## new_canonical_name_needed flags

None. Every op outside the canonical map (`einsum`, `elu`, `leaky_relu`, `unfold`, `upsample`, `multihead_attention`, `mamba_scan`, `batch_norm_1d`, `batch_norm_3d`, `lstm`) was classifiable using existing categories or the project's torch.nn-fallback convention.

## Unresolved / open ambiguities

- **Idefics2 multihead_attention vs aria multihead_attention**: in shard a-d, `aria` was marked `partial` for the same `nn.MultiheadAttention` use in a vision projector. I followed that precedent and marked `idefics2` `partial`. If the coordinator decides `nn.MultiheadAttention` should be treated as composable from existing Q/K/V/O Linear + SDPA, both rows would shift to `composable`.
- **Fastspeech2_conformer LSTM**: this audit only flagged `nn.ConvTranspose1d` and `nn.BatchNorm1d`; LSTM is not present in fastspeech2_conformer (verified via grep). Encodec and a couple of other audio codecs do use `nn.LSTM`, where I added a partial flag.
- **Funnel uses F.avg_pool2d/F.max_pool2d on 1D-shape tensors**: marked `composable` because the kb-nano L1 `avg_pool2d.py` / `max_pool2d.py` accept the same calls (the underlying op is identical regardless of whether the spatial dim is collapsed). If the coordinator wants to be strict about avg/max_pool_1d-with-singleton-dim, this should re-check.
- **glm4v / glm4v_moe / glm_image grid_sample**: all three GLM image-aware variants use `F.grid_sample` directly for vision pos-embed interpolation. None flow through deformable attention, so the kb-nano L1 `rtdetrv2_deformable_attention.py:MultiScaleDeformableAttentionV2` does NOT cover them. Marked `partial`.
- **Hubert-like wav2vec2 CTC heads**: the inference output is via `log_softmax` (torch builtin); CTC loss is training-only. Confirmed all such audio encoders are downgraded to `partial` only when there is a `nn.BatchNorm1d` (hubert, granite_speech, granite_speech_plus); otherwise they are `composable`.

## Spot-check guidance for the coordinator

Recommended `partial` rows to verify by hand (highest novelty / risk of misclassification):

1. `glm4v`, `glm4v_moe`, `glm_image` — all three flag `grid_sample`. The pilot established that `grid_sample` is composed inside the deformable-attention L1; these GLM variants use grid_sample **outside** of deformable attention, so the partial flag is correct, but worth a second read.
2. `idefics2` — `multihead_attention`-only partial; needs decision consistency with `aria` (shard a-d).
3. `emu3` — `batch_norm_3d` partial; this is the only file in the shard that needs a 3D batch-norm. Worth confirming the temporal residual blocks are reached at inference (they are — Emu3VQVAE is the image tokenizer, used to encode/decode visual tokens).
4. `informer` — the only file in the shard with `max_pool_1d`. Verify line 620 is in the encoder forward path (it is — `InformerConvLayer.forward` calls `self.maxPool(x)`).
