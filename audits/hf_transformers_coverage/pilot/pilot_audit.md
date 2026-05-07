# Pilot Audit: 12 architectures + 1 no-modeling exception

> **Note:** This is the **historical** pilot analysis from the first audit pass. The classifications in this doc reflect what the coordinator concluded during the pilot, before the re-audit added 8 new L1 wrappers and reclassified previously-flagged ops. After the re-audit, several rows here that were marked `partial` (e.g. `data2vec_vision` for `conv_transpose2d` + `adaptive_avg_pool_2d`, `swin` for `adaptive_avg_pool_1d`) are now `composable` — see [`audit_methodology.md`](../audit_methodology.md) § 15 for the reclassification logic and [`hf_architecture_operator_coverage.csv`](../hf_architecture_operator_coverage.csv) for current per-row status. This pilot doc is preserved as historical record of the methodology development; the canonical numbers live in `coverage_summary.md`.

---


**HF source:** `huggingface/transformers` @ `da6c53e431f7c9ef0691239d4ce89b0f711ecad7`, in `/tmp/hf_transformers_pinned/`.
**kb-nano support surface:** `origin/experiments` @ `11aa838`, audited from working tree on `audit/hf-transformers-coverage`.

**Per-row methodology:** For each modeling file I (a) ran the AST extractor, (b) read the modeling file myself, (c) cross-checked extracted ops against the kb-nano canonical map (`tools/canonical_to_kb_nano.csv`), (d) noted unresolved ops the extractor missed, (e) cited HF and kb-nano `file:line` evidence, (f) chose a single `support_status`.

Status legend:
- `kb_nano_l4` — already an L4 pipeline (architecture-level integration done).
- `composable` — every required op has either a kb-nano L1/L2 mapping or a torch-builtin passthrough; building this as a new L4 is purely a wiring task.
- `partial` — at least one required op has only a torch.nn fallback (no L1 kernel) AND that fallback is load-bearing for performance, OR a major mode/layout/variant is missing.
- `unsupported` — at least one required op has no kb-nano support and is not a trivial torch builtin.
- `not_inference_required` — folder has no PyTorch modeling.

---

## 1. `bert` — `modeling_bert.py` (1394 lines, 26 classes)

**Architecture classes:** `BertModel`, `BertForMaskedLM`, `BertLMHeadModel`, `BertForPreTraining`, `BertForSequenceClassification`, `BertForTokenClassification`, `BertForQuestionAnswering`, `BertForMultipleChoice`, `BertForNextSentencePrediction`, `BertPreTrainedModel`. Modality: text encoder. Family: encoder.

**Required compute ops** (extractor + manual review):

| HF op | HF evidence | kb-nano mapping | status |
|---|---|---|---|
| `nn.Linear` | `modeling_bert.py:240,254,309,...` (Q/K/V/O, intermediate, output, pooler, lm head, classifier) | L1 `linear.py:Linear` | direct |
| `nn.Embedding` | `modeling_bert.py:165,170,175` (word, position, token-type) | L1 `embedding.py:Embedding` | direct |
| `nn.LayerNorm` | `modeling_bert.py:181,275,310` | L1 `layer_norm.py:LayerNorm` | direct |
| `nn.Dropout` | `modeling_bert.py:182,...` | L1 `dropout.py:Dropout` | direct |
| `F.softmax` | `modeling_bert.py:265` (attention probs) | L1 `softmax.py:Softmax` | direct |
| `torch.matmul` | `modeling_bert.py:243,261` (QK and AV) | L1 `linear.py:Matmul` | direct |
| `ACT2FN[hidden_act]` | `modeling_bert.py:309` | L1 `gelu.py`, `silu.py`, `relu.py`, `tanh.py` (covers all standard ACT2FN keys) | direct |
| `tanh` | pooler | L1 `tanh.py:Tanh` | direct |
| `Cache`/`DynamicCache` | `modeling_bert.py:373` (decoder mode in BertLMHeadModel) | L4 `recurrent_cache.py:RecurrentCache` (paged KV semantically same; HF uses dense cache by default) | direct (semantic) |
| `EncoderDecoderCache` | `modeling_bert.py:374` (decoder mode) | partial — no exact equivalent class in kb-nano, but two stacked KV caches replicate behavior | partial-mode |
| `gather` | tensor builtin | passthrough | direct |
| `arange`, `cat`, etc. | tensor builtins | passthrough | direct |

**No required ops are unsupported.** The decoder mode of BertLMHeadModel uses `EncoderDecoderCache`, which kb-nano does not have a direct cache class for — but the underlying KV-cache primitive (`store_kvcache.py`) covers it, so the wiring would just need to compose two KV caches (one for self-attn, one for cross-attn). Standard BERT (encoder-only) is fully composable.

**Status:** `composable` — kb-nano has no `bert` L4 yet (closest is the BERT L3 blocks: `bert_encoder.py`, `bert_layer.py`, `bert_model.py`), so it would need an L4 pipeline. All ops covered.

---

## 2. `llama` — `modeling_llama.py` (519 lines, 11 classes)

**Architecture classes:** `LlamaModel`, `LlamaForCausalLM`, `LlamaForSequenceClassification`, `LlamaForTokenClassification`, `LlamaForQuestionAnswering`, `LlamaPreTrainedModel`. Decoder-only LLM.

| HF op | HF evidence | kb-nano mapping | status |
|---|---|---|---|
| `nn.Linear` (Q/K/V/O, gate/up/down) | `modeling_llama.py:177-181, 232-235, 461` | L1 `linear.py:Linear` | direct |
| `LlamaRMSNorm` (class) | `modeling_llama.py:53-71` (uses pow+rsqrt+mean) | L1 `rms_norm.py:RMSNorm` | direct |
| `LlamaRotaryEmbedding` + `apply_rotary_pos_emb` | `modeling_llama.py:73-169, 267` | L1 `rotary_emb.py:RotaryEmbedding` | direct |
| `nn.Embedding` | `modeling_llama.py:361` | L1 `embedding.py:Embedding` | direct |
| `ACT2FN[hidden_act]` (silu in default) | `modeling_llama.py:177` | L1 `silu.py:SiLU` (or `silu_and_mul.py` for fused) | direct |
| `F.softmax` (inside attention, via SDPA dispatcher) | `modeling_llama.py:216` (eager fallback path) | L1 `softmax.py` (eager) or `dense_attention.py` (SDPA) | direct |
| `torch.matmul` | `modeling_llama.py:212,218` | L1 `linear.py:Matmul` | direct |
| `ALL_ATTENTION_FUNCTIONS` (sdpa/flash/eager) | `modeling_llama.py:283` | L1 `dense_attention.py` (SDPA), `flash_attn_*` | direct |
| `DynamicCache` / `StaticCache` (KV cache) | `modeling_llama.py:392, 263` (`past_key_value.update`) | L1 `store_kvcache.py:StoreKVCache` (paged) | direct (paged is more efficient than HF's tensor cache; semantic cover) |
| `repeat_kv` (GQA expand) | `modeling_llama.py` (helper, called inside attention path) | implicit in L1 `flash_attn_*` (paged KV with num_kv_heads handles GQA natively) | direct |
| `nn.Dropout` | `modeling_llama.py:217` | L1 `dropout.py:Dropout` | direct |

**Existing L4:** `tasks/baseline/L4/llama.py` is the LlamaEngine L4 pipeline. **Status:** `kb_nano_l4`.

---

## 3. `mistral` — `modeling_mistral.py` (493 lines, 11 classes)

**Architecture classes:** `MistralModel`, `MistralForCausalLM`, plus task heads. Decoder-only LLM with sliding-window attention.

The structure is essentially identical to Llama (RMSNorm + RoPE + GQA + SwiGLU). Mistral's distinguishing feature is **sliding-window attention** (controlled by `config.sliding_window`), implemented via the attention dispatcher passing a window-aware mask.

| HF op | Same as Llama | Status |
|---|---|---|
| RMSNorm, RoPE, Linear, SDPA, DynamicCache, ACT2FN(silu), Embedding, Dropout | (see llama) | direct |
| Sliding-window mask | `modeling_mistral.py:17,372` (`create_sliding_window_causal_mask` from `masking_utils.py`) | mask is constructed by HF helper and **passed to attention as a tensor**; the attention kernel itself (any backend) doesn't need a sliding-window code path. kb-nano's `dense_attention.py` and `flash_attn_*` accept arbitrary masks. | direct |

**No mistral L4 in kb-nano.** Closest L4 is `llama.py`. The L2 path (`llama_attention.py`, `llama_mlp.py`) and L3 (`llama_decoder.py`) cover the layer-level wiring; sliding-window mask is constructed at the model level via HF's `masking_utils.py`.

**Status:** `composable` — wiring task, all ops covered. Sliding-window is a mask-construction concern (just produces a 2-D mask tensor that's already supported by every kb-nano attention kernel), not a missing primitive.

---

## 4. `whisper` — `modeling_whisper.py` (1359 lines, 12 classes)

**Architecture classes:** `WhisperModel`, `WhisperEncoder`, `WhisperDecoder`, `WhisperForConditionalGeneration`, `WhisperForCausalLM`, `WhisperForAudioClassification`, plus PreTrainedModel/wrapper. Encoder-decoder audio.

| HF op | HF evidence | kb-nano mapping | status |
|---|---|---|---|
| `nn.Conv1d` (audio stem) | `modeling_whisper.py:230-240` (mel→encoder hidden) | L1 `conv1d.py:Conv1d` | direct |
| `nn.Linear` (Q/K/V/O, fc1/fc2, projection) | many | L1 `linear.py:Linear` | direct |
| `nn.LayerNorm` | many | L1 `layer_norm.py:LayerNorm` | direct |
| `nn.Embedding` (token, learned position) | `modeling_whisper.py:340-345` | L1 `embedding.py:Embedding` | direct |
| `nn.Dropout` | many | L1 `dropout.py:Dropout` | direct |
| `F.gelu` | `modeling_whisper.py:296` (FFN activation) | L1 `gelu.py:GELU` | direct |
| `F.softmax` | attention | L1 `softmax.py:Softmax` | direct |
| `torch.matmul` | attention | L1 `linear.py:Matmul` | direct |
| `WhisperAttention` (cross-attn + self-attn) | `modeling_whisper.py:280-340` | L2 `whisper_attention.py:WhisperAttention` + L1 `dense_attention.py:DenseAttention` | direct |
| KV cache + `EncoderDecoderCache` | decoder path | L4 `recurrent_cache.py` + L1 `store_kvcache.py` | partial — encoder-decoder cache wrapping done in L4 but no exact `EncoderDecoderCache` class |
| `_get_sinusoidal_position_embeddings` (init) | encoder, init only | passthrough (init-time, not forward) | not_inference |

**Existing L4:** `tasks/baseline/L4/whisper.py`. **Status:** `kb_nano_l4`.

---

## 5. `vit` — `modeling_vit.py` (656 lines, 14 classes)

**Architecture classes:** `ViTModel`, `ViTForImageClassification`, `ViTForMaskedImageModeling`, `ViTPreTrainedModel`. Vision transformer.

| HF op | HF evidence | kb-nano mapping | status |
|---|---|---|---|
| `nn.Conv2d` (patch embed) | `modeling_vit.py:90` (patch projection) | L1 `conv2d.py:Conv2d` | direct |
| `nn.Linear` (Q/K/V/O, FFN, classifier) | many | L1 `linear.py:Linear` | direct |
| `nn.LayerNorm` | many | L1 `layer_norm.py:LayerNorm` | direct |
| `nn.Dropout` | many | L1 `dropout.py:Dropout` | direct |
| `nn.Embedding` (cls token, learnable pos) | `modeling_vit.py:80,85` (parameter / nn.Parameter rather than Embedding for cls/pos; only token embed if MIM) | L1 `embedding.py:Embedding` (when used) | direct |
| `ACT2FN` | FFN activation (gelu by default) | L1 `gelu.py:GELU` | direct |
| `F.softmax`, `torch.matmul`, SDPA dispatcher | attention | L1 `softmax.py`, `dense_attention.py` | direct |
| `F.interpolate` (positional embedding interpolation, optional) | `modeling_vit.py:140-180` | L1 `interpolate.py:Interpolate` | direct |
| `pixel_shuffle` (used in MIM head only) | optional MIM path | torch builtin, passthrough | direct |

**Existing L4:** No `vit.py` L4, but `dinov3.py` and `siglip2.py` are vision transformers with the same primitive set, and L3 `vit_encoder_block.py` exists.

**Status:** `composable` — vision-encoder primitives all present; ViT is a wiring exercise built from existing L1/L2/L3.

---

## 6. `swin` — `modeling_swin.py` (1209 lines, 21 classes)

**Architecture classes:** `SwinModel`, `SwinForImageClassification`, `SwinForMaskedImageModeling`, `SwinBackbone`, `SwinPreTrainedModel`.

| HF op | HF evidence | kb-nano mapping | status |
|---|---|---|---|
| `nn.Conv2d` (patch embed, patch merging via conv) | early layers | L1 `conv2d.py:Conv2d` | direct |
| `nn.Linear` (Q/K/V/O, FFN, classifier) | many | L1 `linear.py:Linear` | direct |
| `nn.LayerNorm` | many | L1 `layer_norm.py:LayerNorm` | direct |
| `nn.Dropout`, `DropPath` | many | L1 `dropout.py:Dropout` (DropPath is stochastic depth — passthrough at inference dropout=0) | direct |
| `F.softmax`, `torch.matmul` | windowed attention | L1 `softmax.py`, `dense_attention.py` | direct |
| `relative_position_bias` (`nn.Parameter` table + index lookup) | `modeling_swin.py:540-560` | composable — kb-nano has L2 `swinv2_window_attention.py` (SwinV2 variant); Swin v1 differs slightly in normalization but conceptually same | direct (composed) |
| `roll` (shifted window) | `modeling_swin.py:680, 720` | torch builtin, passthrough | direct |
| `masked_fill` (window mask) | masking step | torch builtin, passthrough | direct |
| `pad` (window pad) | window padding | torch builtin via L1 `tensor_ops.py:Pad` | direct |
| `F.adaptive_avg_pool1d` (image classifier head pooling) | `modeling_swin.py:1140` | NOT a kb-nano L1 op; used via torch.nn fallback in similar L4s (e.g. mobilenetv4) | partial |
| `F.interpolate` (resize abs pos embed if used) | optional | L1 `interpolate.py:Interpolate` | direct |
| `pixel_shuffle` (MIM head) | optional MIM | torch builtin, passthrough | direct |

**Existing L4:** `tasks/baseline/L4/swinv2.py` covers SwinV2; Swin v1 is structurally close but distinct.

**Status:** `composable` — Swin v1 is a wiring task using SwinV2 building blocks, with `adaptive_avg_pool1d` as a partial dependency (used via torch fallback). Marking `composable` rather than `partial` because the partial op (adaptive_avg_pool1d) is in the head only, not in the inner loop, and kb-nano's existing vision L4s already use torch.nn for similar pool ops at the head — that is the project's de facto convention. **Sub-flag:** "uses torch.nn fallback for `adaptive_avg_pool1d` head".

---

## 7. `mamba` — `modeling_mamba.py` (720 lines, 8 classes)

**Architecture classes:** `MambaModel`, `MambaForCausalLM`, `MambaPreTrainedModel`. Selective SSM.

| HF op | HF evidence | kb-nano mapping | status |
|---|---|---|---|
| `nn.Linear` (in_proj, out_proj, dt_proj, x_proj) | `modeling_mamba.py:130-160` | L1 `linear.py:Linear` | direct |
| `nn.Embedding` | embed_tokens | L1 `embedding.py:Embedding` | direct |
| `nn.Conv1d` (causal_conv1d) | `modeling_mamba.py:175` (`Conv1d(..., groups=intermediate)`, depthwise) | L1 `conv1d.py:Conv1d` (supports groups) + vLLM `causal_conv1d_fn` (used in kb-nano L2 `mamba_mixer.py:35`) | direct |
| `selective_scan_fn` | mamba SSM core | imported from vLLM in kb-nano L2 `mamba_mixer.py:35-38` (kb-nano relies on vLLM kernels) | direct |
| `MambaCache` | state cache | kb-nano L4 `recurrent_cache.py` | direct |
| `nn.LayerNorm` (custom MambaRMSNorm uses pow+rsqrt) | norms | L1 `rms_norm.py:RMSNorm` | direct |
| `ACT2FN[hidden_act]` (silu) | activation | L1 `silu.py:SiLU` | direct |

**Caveat:** kb-nano's selective-scan and causal-conv1d come from vLLM imports, not native kb-nano L1 kernels. From a "support" standpoint this still counts because the L4 pipeline runs end-to-end; from a "all kernels are native" standpoint there's a small dependency on vLLM.

**Existing L4:** `tasks/baseline/L4/mamba.py`. **Status:** `kb_nano_l4`.

---

## 8. `qwen2_vl` — `modeling_qwen2_vl.py` (1673 lines, 18 classes)

**Architecture classes:** `Qwen2VLForConditionalGeneration`, `Qwen2VLModel`, `Qwen2VLTextModel`, `Qwen2VisionTransformerPretrainedModel`, `Qwen2VLPreTrainedModel`. Multimodal LLM.

| HF op | HF evidence | kb-nano mapping | status |
|---|---|---|---|
| `nn.Conv3d` (3D patch embed for video) | vision tower | L1 `conv3d.py:Conv3d` | direct |
| `nn.Linear`, `nn.LayerNorm`, `nn.Embedding`, `nn.Dropout` | many | L1 standard ops | direct |
| Q-RMSNorm (LlamaRMSNorm-style) | text decoder | L1 `rms_norm.py:RMSNorm` | direct |
| `apply_multimodal_rotary_pos_emb` (M-RoPE) | text-vision joint position | L1 `mrope.py:MRotaryEmbedding` | direct |
| Vision rotary embedding | vision tower | L1 `vision_rotary_emb.py:VisionRotaryEmbedding` | direct |
| `F.gelu` (vision activation) | vision FFN | L1 `gelu.py:GELU` | direct |
| `ACT2FN[silu]` (text MLP) | text FFN | L1 `silu.py:SiLU` | direct |
| `F.softmax`, `torch.matmul`, attention dispatcher | attention | L1 `softmax.py`, `dense_attention.py`, `flash_attn_*` | direct |
| KV cache | DynamicCache | L1 `store_kvcache.py:StoreKVCache` | direct |
| `repeat_interleave`, `roll`, `masked_fill`, `outer`, `cat`, etc. | multimodal indexing | passthrough | direct |

**Existing L4:** `tasks/baseline/L4/qwen2_vl.py`. **Status:** `kb_nano_l4`.

---

## 9. `rt_detr` — multi-modeling folder (2 files)

### 9a. `modeling_rt_detr.py` (1847 lines, 22 classes)

**Architecture classes:** `RTDetrModel`, `RTDetrForObjectDetection`, `RTDetrEncoder`/`RTDetrHybridEncoder`, `RTDetrDecoder`, `RTDetrPreTrainedModel`.

| HF op | HF evidence | kb-nano mapping | status |
|---|---|---|---|
| `nn.Conv2d`, `nn.BatchNorm2d`, `nn.LayerNorm`, `nn.Linear`, `nn.Embedding`, `nn.Dropout` | many | L1 standard | direct |
| `multi_scale_deformable_attention` | `modeling_rt_detr.py:870-920` (decoder cross-attn) | L1 `rtdetrv2_deformable_attention.py:MultiScaleDeformableAttentionV2` | direct (named v2 but the v1/v2 numerical difference is in offset normalization; kb-nano uses v2 variant) |
| `F.interpolate` (FPN upsample) | `modeling_rt_detr.py:480-510` | L1 `interpolate.py:Interpolate` | direct |
| `F.grid_sample` (deformable sampling) | inside deformable attention | imported by L1 `rtdetrv2_deformable_attention.py` (via F.grid_sample); not a standalone L1 op but used internally | direct (composed inside L1) |
| `F.relu`, `F.sigmoid`, `F.softmax` | activations | L1 standard | direct |
| `topk` (decoder query selection) | torch builtin | passthrough | direct |
| `gather` | tensor builtin | passthrough | direct |

**Subtle classification:** `RTDetr v1` uses `MultiScaleDeformableAttention` (v1, slightly different normalization than v2). kb-nano has the v2 kernel. Drop-in for v2 model is direct; for v1 it would be `partial` because the v2 kernel may not bit-match v1.

**Existing L4:** `tasks/baseline/L4/rtdetrv2.py` covers RT-DETR-v2. RT-DETR (v1) is not directly an L4 but the same blocks would map. **Status:** `composable` (v1 reusing v2 kernels with possible numerical drift) — flag for verification.

### 9b. `modeling_rt_detr_resnet.py` (412 lines, 9 classes)

**Architecture classes:** `RTDetrResNetBackbone`, `RTDetrResNetPreTrainedModel`. ResNet-style backbone.

| HF op | HF evidence | kb-nano mapping | status |
|---|---|---|---|
| `nn.Conv2d`, `nn.BatchNorm2d` | residual blocks | L1 `conv2d.py`, `batch_norm2d.py` | direct |
| `nn.Identity`, residual add | passthrough | L1 (Identity) | direct |
| `nn.MaxPool2d`, `nn.AvgPool2d` | stem pooling | L1 `max_pool2d.py`, `avg_pool2d.py` | direct |
| ACT2FN (relu) | activation | L1 `relu.py` | direct |

**Status:** `composable` — pure ResNet ops, all L1. (Not a separate L4 in kb-nano, but trivial to wire.)

---

## 10. `data2vec` — multi-modeling folder (3 files)

### 10a. `modeling_data2vec_audio.py` (1324 lines, 20 classes)

**Architecture classes:** `Data2VecAudioModel`, `Data2VecAudioForCTC`, `Data2VecAudioForSequenceClassification`, `Data2VecAudioForAudioFrameClassification`, `Data2VecAudioForXVector`, `Data2VecAudioPreTrainedModel`. Wav2Vec2-style audio encoder.

| HF op | HF evidence | kb-nano mapping | status |
|---|---|---|---|
| `nn.Conv1d` (feature extractor with stride; conv layers) | `modeling_data2vec_audio.py:120-170` (CNN frontend) | L1 `conv1d.py:Conv1d` | direct |
| `nn.Linear` | many | L1 `linear.py:Linear` | direct |
| `nn.LayerNorm` | many | L1 `layer_norm.py:LayerNorm` | direct |
| `nn.Dropout` | many | L1 `dropout.py:Dropout` | direct |
| `F.gelu` | activation | L1 `gelu.py:GELU` | direct |
| `F.softmax`, `torch.matmul`, attention dispatcher | attention | L1 standard | direct |
| `F.log_softmax`, `F.ctc_loss` | CTC head (loss is training-only, log_softmax is inference) | passthrough (log_softmax is torch builtin); ctc_loss is `not_inference_required` | direct + not_inference |
| `cumsum`, `masked_select`, `where`, `argmax` | tensor builtins | passthrough | direct |

**Status:** `composable` — all inference ops covered, CTC loss is training-only (no need to support).

### 10b. `modeling_data2vec_text.py` (1208 lines, 20 classes)

**Architecture classes:** `Data2VecTextModel`, `Data2VecTextForCausalLM`, `Data2VecTextForMaskedLM`, plus task heads. Roberta-like encoder.

| HF op | HF evidence | kb-nano mapping | status |
|---|---|---|---|
| Identical primitive set to BERT (Linear, Embedding, LayerNorm, Dropout, ACT2FN, softmax, matmul, KV cache for decoder mode) | many | L1 standard | direct |

**Status:** `composable` — same as bert.

### 10c. `modeling_data2vec_vision.py` (1244 lines, 23 classes)

**Architecture classes:** `Data2VecVisionModel`, `Data2VecVisionForImageClassification`, `Data2VecVisionForSemanticSegmentation`, `Data2VecVisionPreTrainedModel`. BEiT-style vision encoder.

| HF op | HF evidence | kb-nano mapping | status |
|---|---|---|---|
| Standard vision-encoder ops (Conv2d, Linear, LayerNorm, Dropout, GELU) | many | L1 standard | direct |
| `F.scaled_dot_product_attention` | encoder attention | L1 `dense_attention.py` | direct |
| Relative position bias | `modeling_data2vec_vision.py:380-420` | composable from existing kb-nano embedding + index gather | direct (composed) |
| `F.interpolate` | pos embed interp | L1 `interpolate.py:Interpolate` | direct |
| `F.adaptive_avg_pool2d` | classifier head pool | NOT a kb-nano L1 op; convention is torch.nn fallback (see mobilenetv4) | partial (head-only) |
| `nn.ConvTranspose2d` | semantic seg head decoder | NOT a kb-nano L1 op; used in L2 `sam3_fpn_conv.py:36` and L3 `sam3_mask_decoder.py:253,256` via torch.nn directly | partial (used via torch fallback in kb-nano L2/L3) |
| `nn.MaxPool2d` (seg decoder upsampling) | optional | L1 `max_pool2d.py:MaxPool2d` | direct |
| `nn.BatchNorm2d` (some seg head variants) | seg head | L1 `batch_norm2d.py:BatchNorm2d` | direct |

**Status:** Encoder + classification head: `composable`. Semantic segmentation head specifically uses `ConvTranspose2d` which has no L1 kernel; classify the **semantic-segmentation variant** as `partial`.

For the row-level classification I report `partial` for the file overall (because the file contains the `ForSemanticSegmentation` class), with the note "ImageClassification + Model variants are composable; ForSemanticSegmentation requires ConvTranspose2d via torch.nn fallback".

---

## 11. `deformable_detr` — `modeling_deformable_detr.py` (1697 lines, 19 classes)

**Architecture classes:** `DeformableDetrModel`, `DeformableDetrForObjectDetection`, `DeformableDetrEncoder`, `DeformableDetrDecoder`, `DeformableDetrPreTrainedModel`.

| HF op | HF evidence | kb-nano mapping | status |
|---|---|---|---|
| `multi_scale_deformable_attention` (v1 of the op) | core decoder cross-attn | L1 `rtdetrv2_deformable_attention.py:MultiScaleDeformableAttentionV2` (v2 — close but not bit-identical) | partial |
| `F.grid_sample` | inside deformable attention | composed inside the L1 above | direct |
| `nn.Linear`, `nn.LayerNorm`, `nn.Embedding`, `nn.Dropout`, `F.gelu`/`F.relu`, `F.softmax`, `torch.matmul` | rest of model | L1 standard | direct |
| `nn.GroupNorm` (FPN normalization) | `modeling_deformable_detr.py:430` | L1 `group_norm.py:GroupNorm` | direct |
| `F.interpolate`, `F.sigmoid` | various | L1 standard | direct |
| `nn.Conv2d` | feature flatten / projection | L1 `conv2d.py:Conv2d` | direct |
| `topk`, `gather`, `cumsum`, `masked_fill` | tensor builtins | passthrough | direct |

**Status:** `composable` (verified post-pilot). I read both kb-nano `tasks/baseline/L1/rtdetrv2_deformable_attention.py:34-44` and HF `modeling_deformable_detr.py:170-225` side-by-side. kb-nano's `MultiScaleDeformableAttentionV2` with `method="default"` performs `sampling_grids = 2 * sampling_locations - 1` and `F.grid_sample(mode="bilinear", padding_mode="zeros", align_corners=False)` — bit-identical to HF's v1 math. The "v2" addition is only the optional `method="discrete"` branch (used by RT-DETR-v2 only). Signature differs slightly (`num_points_list` vs `level_start_index`) but is a wiring concern, not a missing primitive.

---

## 12. `barthez` — no-modeling exception

`models/barthez/` contains only `tokenization_barthez.py` and `tokenization_barthez_fast.py`. There is no PyTorch architecture in this folder; it is a tokenizer wrapper that re-uses BART or mBART.

**Status:** `not_inference_required` (no PyTorch architecture). For the audit, this row's denominator question is settled: the *folder* has no architecture; the actual model code lives in `bart`.

---

## Summary table (pilot)

| folder | modeling file | architecture classes | modality | family | status | unsupported / partial flags |
|---|---|---|---|---|---|---|
| bert | modeling_bert.py | BertModel, BertForMaskedLM, BertLMHeadModel, BertForX (5 task heads) | text | encoder (decoder mode for LMHead) | composable | partial: EncoderDecoderCache (composable from 2× KV cache) |
| llama | modeling_llama.py | LlamaModel, LlamaForCausalLM + 3 task heads | text | decoder-only | kb_nano_l4 | — |
| mistral | modeling_mistral.py | MistralModel, MistralForCausalLM + 3 task heads | text | decoder-only (sliding-window attention) | composable | sliding-window is HF mask-builder, no kernel-side change needed |
| whisper | modeling_whisper.py | WhisperModel, WhisperForCondGen, WhisperForCausalLM, WhisperForAudioClass + Encoder/Decoder | audio | encoder-decoder | kb_nano_l4 | partial: EncoderDecoderCache wrapping (handled inside L4) |
| vit | modeling_vit.py | ViTModel, ViTForImageClass, ViTForMaskedImageModeling | vision | encoder | composable | — |
| swin | modeling_swin.py | SwinModel + heads + Backbone | vision | windowed encoder | composable | partial: adaptive_avg_pool1d head fallback |
| mamba | modeling_mamba.py | MambaModel, MambaForCausalLM | text | SSM | kb_nano_l4 | — (selective_scan and causal_conv1d via vLLM imports) |
| qwen2_vl | modeling_qwen2_vl.py | Qwen2VLModel, Qwen2VLForCondGen, Qwen2VLTextModel, Qwen2VisionTransformer | multimodal | hybrid (vision encoder + text decoder, M-RoPE) | kb_nano_l4 | — |
| rt_detr | modeling_rt_detr.py | RTDetrModel, RTDetrForObjectDetection + Encoder/Decoder | detection | hybrid encoder-decoder | composable | partial: deformable-attn-v1 vs kb-nano v2 (numerical-drift risk) |
| rt_detr | modeling_rt_detr_resnet.py | RTDetrResNetBackbone | vision | ResNet | composable | — |
| data2vec | modeling_data2vec_audio.py | Data2VecAudioModel + 4 task heads | audio | wav2vec2-like | composable | — (CTC loss is training-only) |
| data2vec | modeling_data2vec_text.py | Data2VecTextModel + 6 task heads | text | encoder | composable | — |
| data2vec | modeling_data2vec_vision.py | Data2VecVisionModel, ForImageClass, ForSemanticSeg | vision | BEiT-like encoder | partial | seg head uses ConvTranspose2d via torch fallback (no L1 kernel) |
| deformable_detr | modeling_deformable_detr.py | DeformableDetrModel, DeformableDetrForObjectDetection + Encoder/Decoder | detection | encoder-decoder | composable | deformable-attn v2 with method=default is bit-identical to v1 (verified) |
| barthez | (none) | — | text-tokenizer-only | — | not_inference_required | folder has no PyTorch modeling |

Pilot count (post-verification): **15 modeling rows** across **12 folders + 1 exception**, with statuses:
- kb_nano_l4: 4 (llama, whisper, mamba, qwen2_vl)
- composable: 9 (bert, mistral, vit, swin, rt_detr×2, data2vec_audio, data2vec_text, deformable_detr)
- partial: 1 (data2vec_vision: ForSemanticSegmentation needs ConvTranspose2d via torch.nn fallback; ForImageClassification + Model are composable)
- unsupported: 0
- not_inference_required: 1 (barthez)

---

## Pilot self-review

**Things I almost missed and recovered:**

1. **`EncoderDecoderCache` is not the same as `DynamicCache`.** I initially treated all KV-cache references as a single `kv_cache` op. On re-read, the encoder-decoder cache is two stacked caches with different lifecycles (encoder cache built once, decoder cache updated per-step). kb-nano's `recurrent_cache.py` doesn't expose this exact wrapper. Marked as composable rather than direct.

2. **`adaptive_avg_pool*` is genuinely missing as an L1 kernel.** The first read of the AST output showed adaptive_avg_pool_1d in swin and adaptive_avg_pool_2d in data2vec_vision. The catalog does NOT have an L1 class for these. kb-nano's existing L4s (mobilenetv4, yolov10) use `nn.AdaptiveAvgPool2d` directly via torch.nn — this is a CLAUDE.md-spirit violation but it is the current convention, so I flagged as `partial` not `unsupported` for files that need it in the inference path.

3. **`ConvTranspose*` is also genuinely missing as an L1 kernel.** Same pattern: kb-nano L2/L3 (cosyvoice3_hifigan, sam3_fpn_conv, sam3_mask_decoder) use `nn.ConvTranspose1d/2d` directly. So for HF models that use it (data2vec_vision seg head), this is `partial` not `unsupported`.

4. **`grid_sample` is composed inside `rtdetrv2_deformable_attention.py`.** Initial impression was that grid_sample is unsupported as a standalone op — but the only HF use case (deformable attention) is wrapped in the same kb-nano L1, so it is effectively covered.

5. **Deformable attention v1 vs v2 numerical equivalence.** RT-DETR-v1 and Deformable-DETR use slightly different deformable-attention math from RT-DETR-v2. kb-nano only has v2. Without a numerical check this is `partial` — not declaring `direct`.

6. **`ALL_ATTENTION_FUNCTIONS` resolves at runtime.** The HF `_attn_implementation` config picks from `{sdpa, flash_attention_2, eager, flex_attention, ...}`. kb-nano supports sdpa (DenseAttention), flash_attention (FlashAttn*), eager (raw matmul + softmax). It does NOT support flex_attention. **Decision:** if any of the runtime-selectable variants is supported, mark as `direct`. Flex-only models are rare and would be marked `partial`. None of the pilot files force flex-only.

7. **CTC loss / cross_entropy / etc. are inference-irrelevant.** I skipped these from the support analysis.

**Schema observations:**
- Every row needs a `subvariant` flag for multi-task-head architecture files (e.g. data2vec_vision contains both ImageClassification (composable) and SemanticSegmentation (partial)). I'll roll this into the `notes` column rather than splitting rows by class.
- `unsupported_ops` and `partial_ops` need to be normalized to canonical names (`adaptive_avg_pool_2d`, `conv_transpose_2d`, `deformable_attention_v1_normalization`).
- The current schema is sound; no redesign needed.

**Time per row:** ~5 min for an unfamiliar file with the AST extractor, ~3 min for one similar to a pilot row. Realistic for 450 rows with subagent help.

**Ambiguity rate:** 2 of 15 rows (data2vec_vision, deformable_detr) have ambiguous status that needs verification (`partial` vs `composable`/`direct`). 13.3% — acceptable for now; flagged for spot-check.

**Methodology revision before scaling: minor.**
- Add an explicit `subvariants` field for multi-task-head files where status differs by class.
- Add `attention_dispatcher_resolution` to the extractor output so we can audit which attention impls each model can run.
- Lock the canonical op-name table at `tools/canonical_to_kb_nano.csv`.

---

## Items to flag to user/mentor (ambiguities)

1. **Numerator vs denominator of "kb-nano coverage":** the headline number could be (a) modeling files where every required op is `direct`, (b) modeling files where every required op is `direct` OR has a torch.nn fallback already in convention, (c) modeling files where every required op is `direct` OR `composable`. I propose (b) as the headline and (a)/(c) as secondary metrics — confirm.

2. **Treatment of `ConvTranspose*` and `AdaptiveAvgPool*`:** kb-nano files use these via `torch.nn` directly today. Under a strict reading of CLAUDE.md they are CLAUDE.md violations. For audit purposes I treat them as "supported with torch fallback" → `partial`. If you'd rather treat them as `unsupported` (because no L1 kernel), that bumps a few rows down.

3. **Deformable attention v1 vs v2:** does the kb-nano v2 kernel reproduce v1 numerics? If not, those rows are firmly `partial`.

4. **Sliding-window attention:** does L2 `llama_attention.py` accept a `window_size` arg today? If not, Mistral support is partial.

5. **Whether the audit denominator is "modeling files" (~448) or "folders" (442 with PT modeling) or "architecture classes" (every `*ForX` class).** I'm reporting all three but the headline number uses **modeling files**.
