# Manual audit shard 14 (swin → vibevoice_asr)

## swin
- **src**: modeling_swin.py
- **hidden_act**: gelu
- **status**: composable
- **classes**:
  - **`SwinEmbeddings`** [compute]: `L1/conv2d.py + L1/layer_norm.py` (wires `SwinPatchEmbeddings`; adds mask_token, optional pos embedding param, LayerNorm, dropout)
  - **`SwinPatchEmbeddings`** [compute]: `L1/conv2d.py` (Conv2d patch projection + flatten/transpose)
  - **`SwinPatchMerging`** [compute]: `L2/swinv2_patch_merging.py` (4-way 2x2 stride concat + nn.LayerNorm + nn.Linear; matches kb-nano patch-merging op for v1: norm-then-reduction)
  - **`SwinDropPath`** [compute]: stochastic depth (no kb-nano kernel; identity at inference)
  - **`SwinSelfAttention`** [compute]: `L1/linear.py + L1/dense_attention.py` (q/k/v Linear + manual matmul/softmax with relative position bias from table — no exact L2 match; kb-nano's `swinv2_window_attention.py` is V2 cosine variant only)
  - **`SwinSelfOutput`** [compute]: `L1/linear.py` (dense + dropout; LayerNorm/residual handled by SwinLayer not here)
  - **`SwinAttention`** [wiring]: wires `SwinSelfAttention`, `SwinSelfOutput`
  - **`SwinIntermediate`** [compute]: `L1/linear.py + L1/gelu.py` (dense + gelu)
  - **`SwinOutput`** [compute]: `L1/linear.py` (dense + dropout)
  - **`SwinLayer`** [wiring]: wires `SwinAttention`, `SwinIntermediate`, `SwinOutput`; direct `L1/layer_norm.py` (×2), window partition/shift logic
  - **`SwinStage`** [wiring]: wires `SwinLayer` (×depth), optional `SwinPatchMerging`
  - **`SwinEncoder`** [wiring]: wires `SwinStage` (×num_layers)
  - **`SwinModel`** [wiring]: wires `SwinEmbeddings`, `SwinEncoder`; direct `L1/layer_norm.py`, `L1/adaptive_avg_pool1d.py` (pooler)
- **task heads (3)**: ForMaskedImageModeling, ForImageClassification, Backbone — base + linear/conv2d (per-task)

## swin2sr
- **src**: modeling_swin2sr.py
- **hidden_act**: gelu
- **status**: composable
- **classes**:
  - **`Swin2SREmbeddings`** [compute]: `L1/conv2d.py` (wires `Swin2SRPatchEmbeddings`; adds optional pos embedding param, dropout)
  - **`Swin2SRPatchEmbeddings`** [compute]: `L1/conv2d.py + L1/layer_norm.py` (Conv2d projection + flatten/transpose + optional LayerNorm)
  - **`Swin2SRPatchUnEmbeddings`** [compute]: tensor reshape only (no kb-nano kernel — pure transpose/view)
  - **`Swin2SRPatchMerging`** [compute]: `L2/swinv2_patch_merging.py` (4-way concat + reduction + norm; matches v2-style merge with reduction-then-norm)
  - **`Swin2SRSelfAttention`** [compute]: `L2/swinv2_window_attention.py` (cosine attention with logit_scale + continuous position bias MLP — exact match for V2-style window attention)
  - **`Swin2SRSelfOutput`** [compute]: `L1/linear.py` (dense + dropout)
  - **`Swin2SRAttention`** [wiring]: wires `Swin2SRSelfAttention`, `Swin2SRSelfOutput`
  - **`Swin2SRIntermediate`** [compute]: `L1/linear.py + L1/gelu.py`
  - **`Swin2SROutput`** [compute]: `L1/linear.py` (dense + dropout)
  - **`Swin2SRLayer`** [wiring]: wires `Swin2SRAttention`, `Swin2SRIntermediate`, `Swin2SROutput`; direct `L1/layer_norm.py` (×2), window partition/shift
  - **`Swin2SRStage`** [wiring]: wires `Swin2SRLayer` (×depth), `Swin2SRPatchEmbeddings`, `Swin2SRPatchUnEmbeddings`; direct `L1/conv2d.py` (1conv or 3conv residual connection)
  - **`Swin2SREncoder`** [wiring]: wires `Swin2SRStage`
  - **`Swin2SRModel`** [wiring]: wires `Swin2SREmbeddings`, `Swin2SREncoder`, `Swin2SRPatchUnEmbeddings`; direct `L1/conv2d.py` (×2: first_conv, conv_after_body), `L1/layer_norm.py`
  - **`Upsample`** [compute]: `L1/conv2d.py` (Conv2d + PixelShuffle; PixelShuffle has no dedicated kb-nano kernel)
  - **`UpsampleOneStep`** [compute]: `L1/conv2d.py` (Conv2d + PixelShuffle)
  - **`PixelShuffleUpsampler`** [wiring]: wires `Upsample`; direct `L1/conv2d.py` (×2), LeakyReLU
  - **`NearestConvUpsampler`** [wiring]: direct `L1/conv2d.py` (×5), LeakyReLU, F.interpolate (no kb-nano kernel for interpolate/PixelShuffle)
  - **`PixelShuffleAuxUpsampler`** [wiring]: wires `Upsample`; direct `L1/conv2d.py` (×4), LeakyReLU
  - **`Swin2SRForImageSuperResolution`** [wiring]: wires `Swin2SRModel` and one of `PixelShuffleUpsampler`/`UpsampleOneStep`/`PixelShuffleAuxUpsampler`/`NearestConvUpsampler`; direct `L1/conv2d.py` (final_convolution fallback)
- **task heads (0)**: — (only For* head is ForImageSuperResolution which is the primary forward path; kept above as wiring)

## swinv2
- **src**: modeling_swinv2.py
- **hidden_act**: gelu
- **status**: composable
- **classes**:
  - **`Swinv2Embeddings`** [compute]: `L1/conv2d.py + L1/layer_norm.py` (wires `Swinv2PatchEmbeddings`; adds mask_token, optional pos embedding param, LayerNorm, dropout)
  - **`Swinv2PatchEmbeddings`** [compute]: `L1/conv2d.py`
  - **`Swinv2PatchMerging`** [compute]: `L2/swinv2_patch_merging.py` (4-way concat + reduction + norm; v2-style)
  - **`Swinv2DropPath`** [compute]: stochastic depth (identity at inference)
  - **`Swinv2SelfAttention`** [compute]: `L2/swinv2_window_attention.py` (cosine attention with logit_scale + continuous position bias MLP — direct match)
  - **`Swinv2SelfOutput`** [compute]: `L1/linear.py` (dense + dropout)
  - **`Swinv2Attention`** [wiring]: wires `Swinv2SelfAttention`, `Swinv2SelfOutput`
  - **`Swinv2Intermediate`** [compute]: `L1/linear.py + L1/gelu.py`
  - **`Swinv2Output`** [compute]: `L1/linear.py` (dense + dropout)
  - **`Swinv2Layer`** [wiring]: wires `Swinv2Attention`, `Swinv2Intermediate`, `Swinv2Output`; direct `L1/layer_norm.py` (×2 — post-norm style), window partition/shift
  - **`Swinv2Stage`** [wiring]: wires `Swinv2Layer` (×depth), optional `Swinv2PatchMerging`
  - **`Swinv2Encoder`** [wiring]: wires `Swinv2Stage` (×num_layers)
  - **`Swinv2Model`** [wiring]: wires `Swinv2Embeddings`, `Swinv2Encoder`; direct `L1/layer_norm.py`, `L1/adaptive_avg_pool1d.py` (pooler)
- **task heads (3)**: ForMaskedImageModeling, ForImageClassification, Backbone — base + linear/conv2d (per-task)

## switch_transformers
- **src**: modeling_switch_transformers.py
- **hidden_act**: relu (config.dense_act_fn default)
- **status**: composable
- **classes**:
  - **`SwitchTransformersTop1Router`** [compute]: `L1/linear.py + L1/softmax.py` (linear classifier + softmax + top1 + capacity mask; no exact L2 match)
  - **`SwitchTransformersLayerNorm`** [compute]: `L1/t5_layer_norm.py` (T5-style RMS norm, no centering, no bias)
  - **`SwitchTransformersDenseActDense`** [compute]: `L1/linear.py + L1/relu.py` (wi -> relu -> wo; matches T5DenseActDense pattern; closest L2 is `t5_dense.py` though that primarily targets gated variant)
  - **`SwitchTransformersExperts`** [compute]: `L1/linear.py` (per-expert dispatch loop calling DenseActDense; no exact L2 match — Switch uses token-choose with capacity, not standard fused MoE)
  - **`SwitchTransformersSparseMLP`** [wiring]: wires `SwitchTransformersTop1Router`, `SwitchTransformersExperts`
  - **`SwitchTransformersLayerFF`** [wiring]: wires `SwitchTransformersDenseActDense` or `SwitchTransformersSparseMLP`, `SwitchTransformersLayerNorm`; pre-norm + dropout + residual
  - **`SwitchTransformersAttention`** [compute]: `L2/t5_attention.py` (q/k/v/o Linear + relative position bias from nn.Embedding + manual SDPA — matches T5Attention; supports cross-attention via key_value_states)
  - **`SwitchTransformersLayerSelfAttention`** [wiring]: wires `SwitchTransformersAttention`, `SwitchTransformersLayerNorm`; pre-norm + residual
  - **`SwitchTransformersLayerCrossAttention`** [wiring]: wires `SwitchTransformersAttention` (cross), `SwitchTransformersLayerNorm`
  - **`SwitchTransformersBlock`** [wiring]: wires `SwitchTransformersLayerSelfAttention`, optional `SwitchTransformersLayerCrossAttention`, `SwitchTransformersLayerFF`
  - **`SwitchTransformersStack`** [wiring]: wires `SwitchTransformersBlock` (×layers); direct `L1/embedding.py`, `L1/t5_layer_norm.py`
  - **`SwitchTransformersModel`** [wiring]: wires `SwitchTransformersStack` (encoder + decoder); direct shared `L1/embedding.py`
  - **`SwitchTransformersForConditionalGeneration`** [wiring]: wires `SwitchTransformersStack` (encoder + decoder); direct `L1/embedding.py` (shared), `L1/linear.py` (lm_head)
  - **`SwitchTransformersEncoderModel`** [wiring]: wires encoder `SwitchTransformersStack`; direct `L1/embedding.py`
- **task heads (0)**: — (only For* head is ForConditionalGeneration which is the primary forward path)

## t5
- **src**: modeling_t5.py
- **hidden_act**: relu (feed_forward_proj default; "gated-gelu" sets gelu_new + gated path)
- **status**: composable
- **classes**:
  - **`T5LayerNorm`** [compute]: `L1/t5_layer_norm.py` (RMS-style, no bias, no centering)
  - **`T5DenseActDense`** [compute]: `L1/linear.py + L1/relu.py` (wi -> relu -> wo, no gating)
  - **`T5DenseGatedActDense`** [compute]: `L2/t5_dense.py` (wi_0/wi_1 gated FFN + wo; matches kb-nano T5DenseGatedActDense)
  - **`T5LayerFF`** [wiring]: wires `T5DenseActDense` or `T5DenseGatedActDense`, `T5LayerNorm`; pre-norm + dropout + residual
  - **`T5Attention`** [compute]: `L2/t5_attention.py` (q/k/v/o + relative position bias + manual SDPA; supports cross-attention via key_value_states and EncoderDecoderCache)
  - **`T5LayerSelfAttention`** [wiring]: wires `T5Attention`, `T5LayerNorm`
  - **`T5LayerCrossAttention`** [wiring]: wires `T5Attention` (cross), `T5LayerNorm`
  - **`T5Block`** [wiring]: wires `T5LayerSelfAttention`, optional `T5LayerCrossAttention`, `T5LayerFF`
  - **`T5ClassificationHead`** [compute]: `L1/linear.py + L1/tanh.py + L1/linear.py` (dense + tanh + out_proj — used by ForSequenceClassification)
  - **`T5Stack`** [wiring]: wires `T5Block` (×layers); direct `L1/embedding.py`, `L1/t5_layer_norm.py`
  - **`T5Model`** [wiring]: wires `T5Stack` (encoder + decoder); direct `L1/embedding.py` (shared)
  - **`T5ForConditionalGeneration`** [wiring]: wires `T5Stack` (encoder + decoder); direct `L1/embedding.py` (shared), `L1/linear.py` (lm_head)
  - **`T5EncoderModel`** [wiring]: wires `T5Stack` (encoder only); direct `L1/embedding.py`
- **task heads (3)**: ForSequenceClassification, ForTokenClassification, ForQuestionAnswering — base + linear (per-task)

## t5gemma
- **src**: modeling_t5gemma.py (and modular_t5gemma.py inheriting from gemma2)
- **hidden_act**: gelu_pytorch_tanh (hidden_activation default)
- **status**: composable
- **classes**:
  - **`T5GemmaRMSNorm`** [compute, inherits `Gemma2RMSNorm`]: `L1/gemma_rms_norm.py` (Gemma `(1+weight)` convention)
  - **`T5GemmaMLP`** [compute, inherits `Gemma2MLP`]: `L2/llama_mlp.py` (gate_proj + up_proj + down_proj SwiGLU pattern with gelu_pytorch_tanh)
  - **`T5GemmaRotaryEmbedding`** [compute, inherits `Gemma2RotaryEmbedding`]: `L1/rotary_emb.py` (standard Llama-style RoPE)
  - **`T5GemmaSelfAttention`** [compute, inherits `Gemma2Attention`]: `L2/attention.py` (q/k/v/o + RoPE + ALL_ATTENTION_FUNCTIONS dispatch + KV cache; supports sliding/full attention via layer_types and softcap)
  - **`T5GemmaCrossAttention`** [compute]: `L1/linear.py + L1/dense_attention.py` (q from hidden, k/v from encoder, with EncoderDecoderCache; no exact L2 match — kb-nano `whisper_attention.py` covers cross-attn but with different cache semantics)
  - **`T5GemmaEncoderLayer`** [wiring]: wires `T5GemmaSelfAttention`, `T5GemmaMLP`, `T5GemmaRMSNorm` (×4: pre/post for attn and FF)
  - **`T5GemmaDecoderLayer`** [wiring]: wires `T5GemmaSelfAttention`, `T5GemmaCrossAttention`, `T5GemmaMLP`, `T5GemmaRMSNorm` (×6: pre/post for self-attn, cross-attn, FF)
  - **`T5GemmaClassificationHead`** [compute]: `L1/linear.py` (dense out_proj with dropout)
  - **`T5GemmaLMHead`** [compute]: `L1/linear.py` (single Linear)
  - **`T5GemmaEncoder`** [wiring]: wires `T5GemmaEncoderLayer`, `T5GemmaRotaryEmbedding`; direct `L1/embedding.py`, `L1/gemma_rms_norm.py`
  - **`T5GemmaDecoder`** [wiring]: wires `T5GemmaDecoderLayer`, `T5GemmaRotaryEmbedding`; direct `L1/embedding.py`, `L1/gemma_rms_norm.py`
  - **`T5GemmaModel`** [wiring]: wires `T5GemmaEncoder`, `T5GemmaDecoder`
  - **`T5GemmaEncoderModel`** [wiring]: wires `T5GemmaEncoder`
  - **`T5GemmaForConditionalGeneration`** [wiring]: wires `T5GemmaModel`, `T5GemmaLMHead`
- **task heads (2)**: ForSequenceClassification, ForTokenClassification — base + classification head (per-task)

## t5gemma2
- **src**: modeling_t5gemma2.py (and modular_t5gemma2.py)
- **hidden_act**: gelu_pytorch_tanh (hidden_activation default)
- **status**: composable
- **classes**:
  - **`T5Gemma2RMSNorm`** [compute]: `L1/gemma_rms_norm.py` (Gemma `(1+weight)` convention)
  - **`T5Gemma2MLP`** [compute]: `L2/llama_mlp.py` (gate_proj + up_proj + down_proj SwiGLU with gelu_pytorch_tanh)
  - **`T5Gemma2RotaryEmbedding`** [compute]: `L1/rotary_emb.py` (per-layer-type RoPE buffers; standard rotary computation)
  - **`T5Gemma2SelfAttention`** [compute]: `L2/attention.py` (q/k/v/o + q_norm/k_norm + RoPE + ALL_ATTENTION_FUNCTIONS dispatch with sliding window option)
  - **`T5Gemma2MergedAttention`** [compute]: `L1/linear.py + L1/rms_norm.py + L1/dense_attention.py` (merged self+cross attention with concatenated KV; no exact L2 match — kb-nano lacks merged-attention variant)
  - **`T5Gemma2EncoderLayer`** [wiring]: wires `T5Gemma2SelfAttention`, `T5Gemma2MLP`, `T5Gemma2RMSNorm` (×4)
  - **`T5Gemma2DecoderLayer`** [wiring]: wires `T5Gemma2MergedAttention`, `T5Gemma2MLP`, `T5Gemma2RMSNorm` (×4)
  - **`T5Gemma2LMHead`** [compute]: `L1/linear.py`
  - **`T5Gemma2ClassificationHead`** [compute]: `L1/linear.py` (dropout + dense)
  - **`T5Gemma2MultiModalProjector`** [compute]: `L1/avg_pool2d.py + L1/gemma_rms_norm.py` (avg pool + RMSNorm + matmul; no exact L2 match)
  - **`T5Gemma2TextScaledWordEmbedding`** [compute, inherits `nn.Embedding`]: `L1/embedding.py` (Embedding + scale + eoi token override)
  - **`T5Gemma2TextEncoder`** [wiring]: wires `T5Gemma2EncoderLayer`, `T5Gemma2RotaryEmbedding`, `T5Gemma2TextScaledWordEmbedding`; direct `L1/gemma_rms_norm.py`
  - **`T5Gemma2Encoder`** [wiring]: wires `T5Gemma2TextEncoder`, optional `T5Gemma2MultiModalProjector` and Siglip vision components
  - **`T5Gemma2Decoder`** [wiring]: wires `T5Gemma2DecoderLayer`, `T5Gemma2RotaryEmbedding`, `T5Gemma2TextScaledWordEmbedding`; direct `L1/gemma_rms_norm.py`
  - **`T5Gemma2Model`** [wiring]: wires `T5Gemma2Encoder`, `T5Gemma2Decoder`
  - **`T5Gemma2ForConditionalGeneration`** [wiring]: wires `T5Gemma2Model`, `T5Gemma2LMHead`
- **task heads (2)**: ForSequenceClassification, ForTokenClassification — base + classification head (per-task)

## table_transformer
- **src**: modeling_table_transformer.py
- **hidden_act**: relu (activation_function default)
- **status**: composable
- **classes**:
  - **`TableTransformerFrozenBatchNorm2d`** [compute]: `L1/batch_norm2d.py` (frozen BN with rsqrt; no exact match — kb-nano BatchNorm2d doesn't freeze affine, but fundamentally same op)
  - **`TableTransformerConvEncoder`** [wiring]: wires backbone (loaded via load_backbone, e.g. timm/ResNet); replaces nn.BatchNorm2d with `TableTransformerFrozenBatchNorm2d`
  - **`TableTransformerConvModel`** [wiring]: wires `TableTransformerConvEncoder`, position_embedding (Sine or Learned)
  - **`TableTransformerSinePositionEmbedding`** [compute]: trig math on cumsum (no kb-nano kernel; pure tensor ops)
  - **`TableTransformerLearnedPositionEmbedding`** [compute]: `L1/embedding.py` (×2: row + column embeddings)
  - **`TableTransformerAttention`** [compute]: `L1/linear.py + L1/dense_attention.py` (k/v/q/out_proj + manual bmm-based attention with object_queries position embeddings; no exact L2 match — DETR-style with positional inject)
  - **`TableTransformerEncoderLayer`** [wiring]: wires `TableTransformerAttention`; direct `L1/layer_norm.py` (×2), `L1/linear.py` (fc1/fc2), `L1/relu.py`
  - **`TableTransformerDecoderLayer`** [wiring]: wires `TableTransformerAttention` (self + encoder_attn); direct `L1/layer_norm.py` (×3), `L1/linear.py` (fc1/fc2), `L1/relu.py`
  - **`TableTransformerEncoder`** [wiring]: wires `TableTransformerEncoderLayer`; direct `L1/layer_norm.py`
  - **`TableTransformerDecoder`** [wiring]: wires `TableTransformerDecoderLayer`; direct `L1/layer_norm.py`
  - **`TableTransformerModel`** [wiring]: wires `TableTransformerConvModel`, `TableTransformerEncoder`, `TableTransformerDecoder`; direct `L1/conv2d.py` (input_projection 1x1), `L1/embedding.py` (query_position_embeddings)
  - **`TableTransformerMLPPredictionHead`** [compute]: `L1/linear.py + L1/relu.py` (3-layer MLP for bbox regression)
  - **`TableTransformerForObjectDetection`** [wiring]: wires `TableTransformerModel`, `TableTransformerMLPPredictionHead`; direct `L1/linear.py` (class_labels_classifier)
- **task heads (1)**: ForObjectDetection — kept above as wiring (primary head); no other heads

## tapas
- **src**: modeling_tapas.py
- **hidden_act**: gelu
- **status**: composable
- **classes**:
  - **`TapasEmbeddings`** [compute]: `L1/embedding.py + L1/layer_norm.py` (word + position + multiple token_type embeddings + LayerNorm + dropout; multiple token-type variant of `L2/encoder_embeddings.py`)
  - **`TapasSelfAttention`** [compute]: `L2/encoder_attention.py` (q/k/v + manual SDPA; supports cross-attention via encoder_hidden_states and EncoderDecoderCache)
  - **`TapasSelfOutput`** [compute]: `L2/encoder_attention.py` (dense + LayerNorm + residual)
  - **`TapasAttention`** [wiring]: wires `TapasSelfAttention`, `TapasSelfOutput`
  - **`TapasIntermediate`** [compute]: `L1/linear.py + L1/gelu.py`
  - **`TapasOutput`** [compute]: `L2/encoder_attention.py` (dense + LayerNorm + residual; same pattern as SelfOutput)
  - **`TapasLayer`** [wiring]: wires `TapasAttention`, optional cross `TapasAttention`, `TapasIntermediate`, `TapasOutput`
  - **`TapasEncoder`** [wiring]: wires `TapasLayer` (×num_hidden_layers)
  - **`TapasPooler`** [compute]: `L1/linear.py + L1/tanh.py`
  - **`TapasPredictionHeadTransform`** [compute]: `L1/linear.py + L1/gelu.py + L1/layer_norm.py`
  - **`TapasLMPredictionHead`** [wiring]: wires `TapasPredictionHeadTransform`; direct `L1/linear.py`
  - **`TapasOnlyMLMHead`** [wiring]: wires `TapasLMPredictionHead`
  - **`TapasModel`** [wiring]: wires `TapasEmbeddings`, `TapasEncoder`, `TapasPooler`
  - **`TapasForMaskedLM`** [wiring]: wires `TapasModel`, `TapasOnlyMLMHead`
- **task heads (2)**: ForQuestionAnswering, ForSequenceClassification — base + linear/output (per-task)

## textnet
- **src**: modeling_textnet.py
- **hidden_act**: relu (per-layer ACT2CLS lookup, default for stem_act_func is "relu")
- **status**: composable
- **classes**:
  - **`TextNetConvLayer`** [compute]: `L1/conv2d.py + L1/batch_norm2d.py + L1/relu.py` (stem: Conv2d + BN + activation)
  - **`TextNetRepConvLayer`** [compute]: `L1/conv2d.py (×2-4) + L1/batch_norm2d.py (×2-4) + L1/relu.py` (multi-branch reparameterized conv: main + vertical + horizontal + identity branches; no exact L2 match — closest is `rtdetrv2_repvgg_block.py` semantically but different structure)
  - **`TextNetStage`** [wiring]: wires `TextNetRepConvLayer` (×depth)
  - **`TextNetEncoder`** [wiring]: wires `TextNetStage` (×num_stages)
  - **`TextNetModel`** [wiring]: wires `TextNetConvLayer` (stem), `TextNetEncoder`; direct `L1/adaptive_avg_pool2d.py` (pooler)
- **task heads (2)**: ForImageClassification, Backbone — base + linear (per-task)

## time_series_transformer
- **src**: modeling_time_series_transformer.py
- **hidden_act**: gelu (activation_function default)
- **status**: composable
- **classes**:
  - **`TimeSeriesFeatureEmbedder`** [compute]: `L1/embedding.py` (multiple nn.Embedding for static categorical features; concat)
  - **`TimeSeriesStdScaler`** [compute]: tensor mean/std normalize (no kb-nano kernel; pure tensor ops)
  - **`TimeSeriesMeanScaler`** [compute]: tensor mean normalize (no kb-nano kernel)
  - **`TimeSeriesNOPScaler`** [compute]: identity scaler (no kb-nano kernel)
  - **`TimeSeriesSinusoidalPositionalEmbedding`** [compute, inherits `nn.Embedding`]: `L1/embedding.py` (frozen sinusoidal weights)
  - **`TimeSeriesValueEmbedding`** [compute]: `L1/linear.py` (single Linear projection)
  - **`TimeSeriesTransformerAttention`** [compute]: `L1/linear.py + L1/dense_attention.py` (BART-style; q/k/v/out_proj + ALL_ATTENTION_FUNCTIONS dispatch; supports cross-attention with EncoderDecoderCache; no exact L2 match — kb-nano `whisper_attention.py` is closest semantically)
  - **`TimeSeriesTransformerEncoderLayer`** [wiring]: wires `TimeSeriesTransformerAttention`; direct `L1/layer_norm.py` (×2), `L1/linear.py` (fc1/fc2), `L1/gelu.py`
  - **`TimeSeriesTransformerDecoderLayer`** [wiring]: wires `TimeSeriesTransformerAttention` (self + encoder_attn); direct `L1/layer_norm.py` (×3), `L1/linear.py` (fc1/fc2), `L1/gelu.py`
  - **`TimeSeriesTransformerEncoder`** [wiring]: wires `TimeSeriesTransformerEncoderLayer`, `TimeSeriesValueEmbedding`, `TimeSeriesSinusoidalPositionalEmbedding`; direct `L1/layer_norm.py`
  - **`TimeSeriesTransformerDecoder`** [wiring]: wires `TimeSeriesTransformerDecoderLayer`, `TimeSeriesValueEmbedding`, `TimeSeriesSinusoidalPositionalEmbedding`; direct `L1/layer_norm.py`
  - **`TimeSeriesTransformerModel`** [wiring]: wires `TimeSeriesFeatureEmbedder`, scaler, `TimeSeriesTransformerEncoder`, `TimeSeriesTransformerDecoder`
  - **`TimeSeriesTransformerForPrediction`** [wiring]: wires `TimeSeriesTransformerModel`; direct distribution head (parameter projections — `L1/linear.py`)
- **task heads (0)**: — (only For* head is ForPrediction, kept as wiring)

## timesfm
- **src**: modeling_timesfm.py (and modular_timesfm.py)
- **hidden_act**: relu (hard-coded in TimesFmMLP — `F.relu(gate)`); ResidualBlock uses SiLU
- **status**: composable
- **classes**:
  - **`TimesFmMLP`** [compute]: `L1/linear.py + L1/layer_norm.py + L1/relu.py` (LN -> gate_proj -> relu -> down_proj + residual; no exact L2 match — 2-layer relu MLP with internal LN)
  - **`TimesFmResidualBlock`** [compute]: `L1/linear.py + L1/silu.py` (input + output + residual Linear with SiLU activation; no exact L2 match)
  - **`TimesFmRMSNorm`** [compute]: `L1/t5_layer_norm.py` (T5-style RMS, no centering, no bias — comment says "equivalent to T5LayerNorm" though weight is ones not zeros)
  - **`TimesFmPositionalEmbedding`** [compute]: sinusoidal computation with inv_timescales buffer (no kb-nano kernel; pure tensor ops)
  - **`TimesFmAttention`** [compute]: `L1/linear.py + L1/dense_attention.py` (q/k/v/o + per-dim learnable scaling on query + ALL_ATTENTION_FUNCTIONS dispatch; no exact L2 match — closest is `L2/attention.py` but uses softplus-based query scaling instead of fixed scaling)
  - **`TimesFmDecoderLayer`** [wiring]: wires `TimesFmAttention`, `TimesFmMLP`, `TimesFmRMSNorm`
  - **`TimesFmModel`** [wiring]: wires `TimesFmResidualBlock` (input_ff_layer), `TimesFmDecoderLayer` (×layers), `TimesFmPositionalEmbedding`; direct `L1/embedding.py` (freq_emb)
  - **`TimesFmModelForPrediction`** [wiring]: wires `TimesFmModel`; direct `TimesFmResidualBlock` (output projections)
- **task heads (0)**: — (ForPrediction is primary head)

## timesfm2_5
- **src**: modeling_timesfm2_5.py (and modular_timesfm2_5.py)
- **hidden_act**: swish (config.activation default "swish" → silu)
- **status**: composable
- **classes**:
  - **`TimesFm2_5MLP`** [compute]: `L1/linear.py + L1/silu.py` (fc1 -> swish -> fc2; 2-layer FFN; no exact L2 match — closest is a generic 2-layer MLP)
  - **`TimesFm2_5ResidualBlock`** [compute]: `L1/linear.py + L1/silu.py` (input/output/residual Linear with swish)
  - **`TimesFm2_5RMSNorm`** [compute]: `L1/t5_layer_norm.py` (T5-style RMS, equivalent comment in source)
  - **`TimesFm2_5RotaryEmbedding`** [compute]: `L1/rotary_emb.py` (standard RoPE inv_freq computation)
  - **`TimesFm2_5Attention`** [compute]: `L2/attention.py` (q/k/v/o + RoPE + q_norm/k_norm + per-dim learnable scaling + ALL_ATTENTION_FUNCTIONS dispatch + GQA via num_key_value_heads; close to Llama-style attention with extra query scaling)
  - **`TimesFm2_5DecoderLayer`** [wiring]: wires `TimesFm2_5Attention`, `TimesFm2_5MLP`, `TimesFm2_5RMSNorm` (×4)
  - **`TimesFm2_5PositionalEmbedding`** [compute]: sinusoidal pos embedding (no kb-nano kernel; tensor ops)
  - **`TimesFm2_5Model`** [wiring]: wires `TimesFm2_5ResidualBlock`, `TimesFm2_5DecoderLayer`, `TimesFm2_5RotaryEmbedding`, `TimesFm2_5PositionalEmbedding`; direct `L1/embedding.py` (freq_emb)
  - **`TimesFm2_5ModelForPrediction`** [wiring]: wires `TimesFm2_5Model`, `TimesFm2_5ResidualBlock` (output)
- **task heads (0)**: — (ForPrediction is primary head)

## timesformer
- **src**: modeling_timesformer.py
- **hidden_act**: gelu
- **status**: composable
- **classes**:
  - **`TimesformerPatchEmbeddings`** [compute]: `L1/conv2d.py` (patch projection)
  - **`TimesformerEmbeddings`** [compute]: `L1/conv2d.py` (wires `TimesformerPatchEmbeddings`; adds cls_token, position embeddings, optional time embeddings as nn.Parameter; dropout)
  - **`TimeSformerDropPath`** [compute]: stochastic depth (identity at inference)
  - **`TimesformerSelfAttention`** [compute]: `L1/linear.py + L1/dense_attention.py` (fused qkv Linear, manual SDPA via softmax; non-causal; no exact L2 match — kb-nano `siglip_attention.py` is non-causal but uses split q/k/v projections)
  - **`TimesformerSelfOutput`** [compute]: `L1/linear.py` (dense + dropout; no LN/residual here, handled by Layer)
  - **`TimeSformerAttention`** [wiring]: wires `TimesformerSelfAttention`, `TimesformerSelfOutput`
  - **`TimesformerIntermediate`** [compute]: `L1/linear.py + L1/gelu.py`
  - **`TimesformerOutput`** [compute]: `L1/linear.py` (dense + dropout)
  - **`TimesformerLayer`** [wiring]: wires `TimeSformerAttention` (spatial), optional `TimeSformerAttention` (temporal for divided_space_time), `TimesformerIntermediate`, `TimesformerOutput`; direct `L1/layer_norm.py` (×2 or ×3 with temporal_layernorm), optional `L1/linear.py` (temporal_dense)
  - **`TimesformerEncoder`** [wiring]: wires `TimesformerLayer` (×num_hidden_layers)
  - **`TimesformerModel`** [wiring]: wires `TimesformerEmbeddings`, `TimesformerEncoder`; direct `L1/layer_norm.py`
- **task heads (1)**: ForVideoClassification — base + linear (per-task)

## timm_backbone
- **src**: modeling_timm_backbone.py
- **hidden_act**: n/a (delegates to the wrapped timm model)
- **status**: unsupported (delegates to external `timm` library)
- **classes**:
  - **`TimmBackbone`** [wiring]: wraps a timm model created via `timm.create_model`; not a kb-nano composition (depends on external timm at runtime)
- **task heads (0)**: — (no For* heads; this is a backbone wrapper)

## timm_wrapper
- **src**: modeling_timm_wrapper.py
- **hidden_act**: n/a (delegates to the wrapped timm model)
- **status**: unsupported (delegates to external `timm` library)
- **classes**:
  - **`TimmWrapperModel`** [wiring]: wraps a timm model
  - **`TimmWrapperForImageClassification`** [wiring]: wraps timm classification model (not a kb-nano composition)
- **task heads (0)**: — (For* head ForImageClassification is direct timm wrapping)

## trocr
- **src**: modeling_trocr.py
- **hidden_act**: gelu (activation_function default)
- **status**: composable
- **classes**:
  - **`TrOCRLearnedPositionalEmbedding`** [compute, inherits `nn.Embedding`]: `L1/embedding.py` (offset-based positional embedding)
  - **`TrOCRScaledWordEmbedding`** [compute, inherits `nn.Embedding`]: `L1/embedding.py` (Embedding * embed_scale)
  - **`TrOCRSinusoidalPositionalEmbedding`** [compute]: sinusoidal weights with index_select (no kb-nano kernel; pure tensor ops)
  - **`TrOCRAttention`** [compute]: `L1/linear.py + L1/dense_attention.py` (BART-style; q/k/v/out_proj + manual bmm-based SDPA; supports cross-attention via key_value_states + EncoderDecoderCache; no exact L2 match — closest is `whisper_attention.py`)
  - **`TrOCRDecoderLayer`** [wiring]: wires `TrOCRAttention` (self), optional `TrOCRAttention` (encoder cross-attn); direct `L1/layer_norm.py` (×2 or ×3), `L1/linear.py` (fc1/fc2), `L1/gelu.py`
  - **`TrOCRDecoder`** [wiring]: wires `TrOCRDecoderLayer`, `TrOCRScaledWordEmbedding`, one of `TrOCRLearnedPositionalEmbedding`/`TrOCRSinusoidalPositionalEmbedding`; direct `L1/layer_norm.py`
  - **`TrOCRDecoderWrapper`** [wiring]: wires `TrOCRDecoder`
  - **`TrOCRForCausalLM`** [wiring]: wires `TrOCRDecoderWrapper`; direct `L1/linear.py` (output_projection / lm_head)
- **task heads (0)**: — (ForCausalLM is the primary head, kept above)

## tvp
- **src**: modeling_tvp.py
- **hidden_act**: gelu
- **status**: composable
- **classes**:
  - **`TvpLoss`** [compute]: loss class (skipped per audit rules — bipartite matching loss)
  - **`TvpVisionModel`** [wiring]: wires backbone (load_backbone); direct `L1/conv2d.py` (grid_encoder_conv 3x3), `L1/max_pool2d.py`, `L1/relu.py`
  - **`TvpVisualInputEmbedding`** [compute]: `L1/embedding.py + L1/layer_norm.py` (position + row + col + token_type embeddings; concat 2D position; LayerNorm + dropout; no exact L2 match)
  - **`TvpTextInputEmbeddings`** [compute]: `L2/encoder_embeddings.py` (BERT-style: word + position + token_type + LayerNorm + Dropout)
  - **`TvpAttention`** [compute]: `L2/encoder_attention.py` (q/k/v + manual SDPA + dense + LayerNorm + residual all-in-one; combines BERT SelfAttention + SelfOutput pattern)
  - **`TvpIntermediate`** [compute]: `L1/linear.py + L1/gelu.py`
  - **`TvpOutputLayer`** [compute]: `L2/encoder_attention.py` (dense + LayerNorm + residual; same shape as BertOutput)
  - **`TvpEncodeLayer`** [wiring]: wires `TvpAttention`, `TvpIntermediate`, `TvpOutputLayer`
  - **`TvpEncoder`** [wiring]: wires `TvpEncodeLayer` (×num_hidden_layers)
  - **`TvpPooler`** [compute]: `L1/linear.py + L1/tanh.py`
  - **`TvpFrameDownPadPrompter`** [compute]: nn.Parameter pad + multiply (no kb-nano kernel)
  - **`TvpFramePadPrompter`** [compute]: nn.Parameter pads + arithmetic (no kb-nano kernel)
  - **`TvpModel`** [wiring]: wires `TvpVisionModel`, `TvpVisualInputEmbedding`, `TvpTextInputEmbeddings`, `TvpEncoder`, `TvpPooler`, prompter (`TvpFrameDownPadPrompter`/`TvpFramePadPrompter`)
  - **`TvpVideoGroundingHead`** [compute]: `L1/linear.py + L1/relu.py` (regression head)
  - **`TvpForVideoGrounding`** [wiring]: wires `TvpModel`, `TvpVideoGroundingHead`
- **task heads (0)**: — (ForVideoGrounding is primary head)

## udop
- **src**: modeling_udop.py
- **hidden_act**: relu (feed_forward_proj default)
- **status**: composable
- **classes**:
  - **`UdopPatchEmbeddings`** [compute]: `L1/conv2d.py` (Conv2d patch projection + flatten/transpose)
  - **`UdopLayerNorm`** [compute]: `L1/t5_layer_norm.py` (T5-style RMS norm)
  - **`UdopDenseActDense`** [compute]: `L1/linear.py + L1/relu.py` (wi -> relu -> wo; no gating)
  - **`UdopDenseGatedActDense`** [compute]: `L2/t5_dense.py` (wi_0/wi_1 gated + wo)
  - **`UdopLayerFF`** [wiring]: wires `UdopDenseActDense` or `UdopDenseGatedActDense`, `UdopLayerNorm`
  - **`UdopAttention`** [compute]: `L2/t5_attention.py` (T5-style q/k/v/o + relative position bias)
  - **`UdopLayerSelfAttention`** [wiring]: wires `UdopAttention`, `UdopLayerNorm`
  - **`UdopLayerCrossAttention`** [wiring]: wires `UdopAttention` (cross), `UdopLayerNorm`
  - **`UdopBlock`** [wiring]: wires `UdopLayerSelfAttention`, optional `UdopLayerCrossAttention`, `UdopLayerFF`
  - **`UdopCellEmbeddings`** [compute]: `L1/embedding.py` (×2: x and y bbox cell embeddings)
  - **`RelativePositionBiasBase`** [compute, abstract]: `L1/embedding.py` (relative_attention_bias Embedding; subclasses for 1D/horizontal/vertical bias)
  - **`RelativePositionBias1D`** [compute, inherits `RelativePositionBiasBase`]: `L1/embedding.py`
  - **`RelativePositionBiasHorizontal`** [compute, inherits `RelativePositionBiasBase`]: `L1/embedding.py`
  - **`RelativePositionBiasVertical`** [compute, inherits `RelativePositionBiasBase`]: `L1/embedding.py`
  - **`RelativePositionBiasAggregated`** [wiring]: wires multiple `RelativePositionBiasBase` subclasses; sums their outputs
  - **`UdopStack`** [wiring]: wires `UdopBlock`, `UdopPatchEmbeddings`, `UdopCellEmbeddings`, `RelativePositionBiasAggregated`; direct `L1/embedding.py` (shared word embeddings), `L1/t5_layer_norm.py`
  - **`UdopModel`** [wiring]: wires `UdopStack` (encoder + decoder); direct `L1/embedding.py` (shared)
  - **`UdopForConditionalGeneration`** [wiring]: wires `UdopStack` (encoder + decoder); direct `L1/embedding.py` (shared), `L1/linear.py` (lm_head)
  - **`UdopEncoderModel`** [wiring]: wires encoder `UdopStack`
- **task heads (0)**: — (ForConditionalGeneration is primary head)

## umt5
- **src**: modeling_umt5.py
- **hidden_act**: gelu_new (gated-gelu default → dense_act_fn = "gelu_new")
- **status**: composable
- **classes**:
  - **`UMT5LayerNorm`** [compute]: `L1/t5_layer_norm.py` (T5-style RMS)
  - **`UMT5DenseActDense`** [compute]: `L1/linear.py + L1/gelu.py` (wi -> gelu_new -> wo, with default config)
  - **`UMT5DenseGatedActDense`** [compute]: `L2/t5_dense.py` (wi_0/wi_1 gated + wo)
  - **`UMT5LayerFF`** [wiring]: wires `UMT5DenseActDense` or `UMT5DenseGatedActDense`, `UMT5LayerNorm`
  - **`UMT5Attention`** [compute]: `L2/t5_attention.py` (T5-style q/k/v/o + relative position bias; UMT5 variant has per-layer rel-bias not shared)
  - **`UMT5LayerSelfAttention`** [wiring]: wires `UMT5Attention`, `UMT5LayerNorm`
  - **`UMT5LayerCrossAttention`** [wiring]: wires `UMT5Attention` (cross), `UMT5LayerNorm`
  - **`UMT5Block`** [wiring]: wires `UMT5LayerSelfAttention`, optional `UMT5LayerCrossAttention`, `UMT5LayerFF`
  - **`UMT5ClassificationHead`** [compute]: `L1/linear.py + L1/tanh.py + L1/linear.py` (dense + tanh + out_proj for ForSequenceClassification)
  - **`UMT5Stack`** [wiring]: wires `UMT5Block` (×layers); direct `L1/embedding.py`, `L1/t5_layer_norm.py`
  - **`UMT5Model`** [wiring]: wires `UMT5Stack` (encoder + decoder); direct `L1/embedding.py` (shared)
  - **`UMT5ForConditionalGeneration`** [wiring]: wires `UMT5Stack` (encoder + decoder); direct `L1/embedding.py` (shared), `L1/linear.py` (lm_head)
  - **`UMT5EncoderModel`** [wiring]: wires encoder `UMT5Stack`
- **task heads (3)**: ForSequenceClassification, ForTokenClassification, ForQuestionAnswering — base + linear/classification head (per-task)

## unispeech
- **src**: modeling_unispeech.py (and modular_unispeech.py)
- **hidden_act**: gelu (hidden_act and feat_extract_activation both default "gelu")
- **status**: composable
- **classes**:
  - **`UniSpeechSamePadLayer`** [compute]: tensor slicing only (no kb-nano kernel)
  - **`UniSpeechPositionalConvEmbedding`** [compute]: `L1/conv1d.py + L1/gelu.py` (Conv1d with weight_norm + same-pad + activation)
  - **`UniSpeechNoLayerNormConvLayer`** [compute]: `L1/conv1d.py + L1/gelu.py` (Conv1d + activation)
  - **`UniSpeechLayerNormConvLayer`** [compute]: `L1/conv1d.py + L1/layer_norm.py + L1/gelu.py`
  - **`UniSpeechGroupNormConvLayer`** [compute]: `L1/conv1d.py + L1/group_norm.py + L1/gelu.py`
  - **`UniSpeechFeatureEncoder`** [wiring]: wires `UniSpeechGroupNormConvLayer`/`UniSpeechNoLayerNormConvLayer` or `UniSpeechLayerNormConvLayer` (×N)
  - **`UniSpeechFeatureProjection`** [compute]: `L1/layer_norm.py + L1/linear.py`
  - **`UniSpeechAttention`** [compute]: `L1/linear.py + L1/dense_attention.py` (BART-style; q/k/v/out_proj + ALL_ATTENTION_FUNCTIONS dispatch; supports cross-attention; no exact L2 match)
  - **`UniSpeechFeedForward`** [compute]: `L1/linear.py + L1/gelu.py` (intermediate_dense -> act -> output_dense)
  - **`UniSpeechEncoderLayer`** [wiring]: wires `UniSpeechAttention`, `UniSpeechFeedForward`; direct `L1/layer_norm.py` (×2)
  - **`UniSpeechEncoder`** [wiring]: wires `UniSpeechPositionalConvEmbedding`, `UniSpeechEncoderLayer` (×layers); direct `L1/layer_norm.py`
  - **`UniSpeechAttnAdapterLayer`** [compute]: `L1/layer_norm.py + L1/linear.py + L1/relu.py`
  - **`UniSpeechEncoderLayerStableLayerNorm`** [wiring]: wires `UniSpeechAttention`, `UniSpeechFeedForward`, optional `UniSpeechAttnAdapterLayer`; direct `L1/layer_norm.py` (×2; pre-norm variant)
  - **`UniSpeechEncoderStableLayerNorm`** [wiring]: wires `UniSpeechPositionalConvEmbedding`, `UniSpeechEncoderLayerStableLayerNorm`
  - **`UniSpeechGumbelVectorQuantizer`** [compute]: `L1/linear.py` + Gumbel softmax (custom; no kb-nano kernel)
  - **`UniSpeechModel`** [wiring]: wires `UniSpeechFeatureEncoder`, `UniSpeechFeatureProjection`, `UniSpeechEncoder`/`UniSpeechEncoderStableLayerNorm`
  - **`UniSpeechForPreTraining`** [wiring]: wires `UniSpeechModel`, `UniSpeechGumbelVectorQuantizer`; direct `L1/linear.py`
- **task heads (3)**: ForCTC, ForSequenceClassification — base + linear (per-task)

## unispeech_sat
- **src**: modeling_unispeech_sat.py (and modular_unispeech_sat.py)
- **hidden_act**: gelu (hidden_act and feat_extract_activation both default "gelu")
- **status**: composable
- **classes**: (same structure as unispeech with `UniSpeechSat` prefix; copied modules)
  - **`UniSpeechSatSamePadLayer`** [compute]: slicing
  - **`UniSpeechSatPositionalConvEmbedding`** [compute]: `L1/conv1d.py + L1/gelu.py`
  - **`UniSpeechSatNoLayerNormConvLayer`** [compute]: `L1/conv1d.py + L1/gelu.py`
  - **`UniSpeechSatLayerNormConvLayer`** [compute]: `L1/conv1d.py + L1/layer_norm.py + L1/gelu.py`
  - **`UniSpeechSatGroupNormConvLayer`** [compute]: `L1/conv1d.py + L1/group_norm.py + L1/gelu.py`
  - **`UniSpeechSatFeatureEncoder`** [wiring]: wires conv layer variants
  - **`UniSpeechSatFeatureProjection`** [compute]: `L1/layer_norm.py + L1/linear.py`
  - **`UniSpeechSatAttention`** [compute]: `L1/linear.py + L1/dense_attention.py` (BART-style)
  - **`UniSpeechSatFeedForward`** [compute]: `L1/linear.py + L1/gelu.py`
  - **`UniSpeechSatEncoderLayer`** [wiring]: wires `UniSpeechSatAttention`, `UniSpeechSatFeedForward`; direct `L1/layer_norm.py` (×2)
  - **`UniSpeechSatEncoder`** [wiring]: wires `UniSpeechSatPositionalConvEmbedding`, `UniSpeechSatEncoderLayer`
  - **`UniSpeechSatAttnAdapterLayer`** [compute]: `L1/layer_norm.py + L1/linear.py + L1/relu.py`
  - **`UniSpeechSatEncoderLayerStableLayerNorm`** [wiring]: wires `UniSpeechSatAttention`, `UniSpeechSatFeedForward`, optional `UniSpeechSatAttnAdapterLayer`
  - **`UniSpeechSatEncoderStableLayerNorm`** [wiring]: wires `UniSpeechSatPositionalConvEmbedding`, `UniSpeechSatEncoderLayerStableLayerNorm`
  - **`UniSpeechSatGumbelVectorQuantizer`** [compute]: `L1/linear.py` + Gumbel softmax
  - **`UniSpeechSatModel`** [wiring]: wires `UniSpeechSatFeatureEncoder`, `UniSpeechSatFeatureProjection`, `UniSpeechSatEncoder`/`UniSpeechSatEncoderStableLayerNorm`
  - **`UniSpeechSatForPreTraining`** [wiring]: wires `UniSpeechSatModel`, `UniSpeechSatGumbelVectorQuantizer`; direct `L1/linear.py`
  - **`AMSoftmaxLoss`** [compute]: AM-softmax loss (skipped — loss class)
  - **`TDNNLayer`** [compute]: `L1/conv1d.py + L1/relu.py` (TDNN with kernel/dilation/stride; no exact L2 match — used by ForXVector)
  - **`UniSpeechSatForXVector`** [wiring]: wires `UniSpeechSatModel`, `TDNNLayer` (×N)
- **task heads (4)**: ForCTC, ForSequenceClassification, ForAudioFrameClassification, ForXVector — base + linear/TDNN (per-task)

## univnet
- **src**: modeling_univnet.py
- **hidden_act**: leaky_relu (hard-coded F.leaky_relu calls)
- **status**: composable
- **classes**:
  - **`UnivNetKernelPredictorResidualBlock`** [compute]: `L1/conv1d.py + L1/leaky_relu.py` (Conv1d + LeakyReLU residual; uses `F.leaky_relu` — kb-nano has no `leaky_relu.py` so this decomposes to L1 conv1d ops with custom activation handled in tensor ops)
  - **`UnivNetKernelPredictor`** [wiring]: wires `UnivNetKernelPredictorResidualBlock` (×num_blocks); direct `L1/conv1d.py` (input_conv, kernel_conv, bias_conv)
  - **`UnivNetLvcResidualBlock`** [compute]: `L1/conv1d.py + L1/sigmoid.py + L1/tanh.py` (location-variable conv with einsum + sigmoid/tanh gated activation; no exact L2 match — custom LVC operation)
  - **`UnivNetLvcBlock`** [wiring]: wires `UnivNetKernelPredictor`, `UnivNetLvcResidualBlock` (×num_blocks); direct `L1/conv_transpose1d.py` (convt_pre)
  - **`UnivNetModel`** [wiring]: wires `UnivNetLvcBlock` (×stages); direct `L1/conv1d.py` (conv_pre, conv_post)
- **task heads (0)**: — (UnivNetModel is the primary inference path)

## upernet
- **src**: modeling_upernet.py
- **hidden_act**: relu (hard-coded `nn.ReLU()` in `UperNetConvModule`)
- **status**: composable
- **classes**:
  - **`UperNetConvModule`** [compute]: `L1/conv2d.py + L1/batch_norm2d.py + L1/relu.py`
  - **`UperNetPyramidPoolingBlock`** [wiring]: wires `UperNetConvModule`; direct `L1/adaptive_avg_pool2d.py`
  - **`UperNetPyramidPoolingModule`** [wiring]: wires `UperNetPyramidPoolingBlock` (×len(pool_scales)); F.interpolate for upsampling
  - **`UperNetHead`** [wiring]: wires `UperNetPyramidPoolingModule`, `UperNetConvModule` (×N); direct `L1/conv2d.py` (classifier 1x1)
  - **`UperNetFCNHead`** [wiring]: wires `UperNetConvModule` (×num_convs); direct `L1/conv2d.py` (classifier)
  - **`UperNetForSemanticSegmentation`** [wiring]: wires backbone (load_backbone), `UperNetHead`, optional `UperNetFCNHead`
- **task heads (0)**: — (ForSemanticSegmentation is primary head)

## uvdoc
- **src**: modeling_uvdoc.py (and modular_uvdoc.py)
- **hidden_act**: prelu (hidden_act default "prelu")
- **status**: composable
- **classes**:
  - **`UVDocConvLayer`** [compute]: `L1/conv2d.py + L1/batch_norm2d.py` + activation (PReLU; no kb-nano kernel for PReLU, so decomposes to per-channel parameter scaling)
  - **`UVDocResidualBlock`** [wiring]: wires `UVDocConvLayer` (×3 with conv_down/conv_start/conv_final); residual + activation
  - **`UVDocResNetStage`** [wiring]: wires `UVDocResidualBlock` (×depth)
  - **`UVDocResNet`** [wiring]: wires `UVDocConvLayer` (head), `UVDocResNetStage` (×stages)
  - **`UVDocBridgeBlock`** [wiring]: wires `UVDocConvLayer` (×N with dilations)
  - **`UVDocPointPositions2D`** [wiring]: wires `UVDocConvLayer`; direct `L1/conv2d.py` (conv_up)
  - **`UVDocBridge`** [wiring]: wires `UVDocBridgeBlock` (×N)
  - **`UVDocBackbone`** [wiring]: wires `UVDocResNet`, `UVDocBridge`
  - **`UVDocHead`** [compute]: regression head — `L1/conv2d.py` + linear/activation (need to read)
  - **`UVDocModel`** [wiring]: wires `UVDocBackbone`, `UVDocBridge`, `UVDocHead`
- **task heads (0)**: — (UVDocModel is the primary inference path; `UVDocBackbone` is also a backbone wrapper)

## vaultgemma
- **src**: modeling_vaultgemma.py (and modular_vaultgemma.py)
- **hidden_act**: gelu_pytorch_tanh (hidden_activation default)
- **status**: composable
- **classes**:
  - **`VaultGemmaRMSNorm`** [compute]: `L1/gemma_rms_norm.py` (Gemma `(1+weight)` convention)
  - **`VaultGemmaMLP`** [compute]: `L2/llama_mlp.py` (gate_proj + up_proj + down_proj SwiGLU with gelu_pytorch_tanh)
  - **`VaultGemmaAttention`** [compute]: `L2/attention.py` (q/k/v/o + RoPE via apply_rotary_pos_emb + ALL_ATTENTION_FUNCTIONS dispatch + KV cache + sliding window option + softcap)
  - **`VaultGemmaDecoderLayer`** [wiring]: wires `VaultGemmaAttention`, `VaultGemmaMLP`, `VaultGemmaRMSNorm` (×2: input_layernorm + pre_feedforward_layernorm)
  - **`VaultGemmaRotaryEmbedding`** [compute]: `L1/rotary_emb.py`
  - **`VaultGemmaTextScaledWordEmbedding`** [compute, inherits `nn.Embedding`]: `L1/embedding.py` (Embedding + scale)
  - **`VaultGemmaModel`** [wiring]: wires `VaultGemmaTextScaledWordEmbedding`, `VaultGemmaDecoderLayer` (×layers), `VaultGemmaRotaryEmbedding`; direct `L1/gemma_rms_norm.py`
  - **`VaultGemmaForCausalLM`** [wiring]: wires `VaultGemmaModel`; direct `L1/linear.py` (lm_head)
- **task heads (0)**: — (ForCausalLM is the primary head)

## vibevoice_acoustic_tokenizer
- **src**: modeling_vibevoice_acoustic_tokenizer.py (and modular_vibevoice_acoustic_tokenizer.py)
- **hidden_act**: gelu (config.hidden_act default)
- **status**: composable
- **classes**:
  - **`VibeVoiceAcousticTokenizerRMSNorm`** [compute]: `L1/t5_layer_norm.py` (T5-style RMS — comment says equivalent)
  - **`VibeVoiceAcousticTokenizerFeedForward`** [compute]: `L1/linear.py + L1/gelu.py` (linear1 -> activation -> linear2)
  - **`VibeVoiceAcousticTokenizerConv1dCacheLayer`** / **`VibeVoiceAcousticTokenizerConv1dPaddingCache`** [compute]: streaming cache helpers (skipped — cache classes, no nn.Module forward in canonical sense)
  - **`VibeVoiceAcousticTokenizerCausalConv1d`** [compute]: `L1/causal_conv1d.py` (Conv1d with left causal padding + optional cache; matches kb-nano causal_conv1d semantics)
  - **`VibeVoiceAcousticTokenizerCausalConvTranspose1d`** [compute]: `L1/conv_transpose1d.py` (ConvTranspose1d with causal handling; no exact L2 match)
  - **`VibeVoiceAcousticTokenizerConvNext1dLayer`** [wiring]: wires `VibeVoiceAcousticTokenizerCausalConv1d` (mixer), `VibeVoiceAcousticTokenizerFeedForward`, `VibeVoiceAcousticTokenizerRMSNorm` (×2); direct nn.Parameter (gamma, ffn_gamma)
  - **`VibeVoiceAcousticTokenizerEncoderStem`** [wiring]: wires `VibeVoiceAcousticTokenizerCausalConv1d`, `VibeVoiceAcousticTokenizerConvNext1dLayer` (×depths[0])
  - **`VibeVoiceAcousticTokenizerEncoderLayer`** [wiring]: wires `VibeVoiceAcousticTokenizerCausalConv1d` (downsample), `VibeVoiceAcousticTokenizerConvNext1dLayer` (×depths[depth_idx])
  - **`VibeVoiceAcousticTokenizerEncoderModel`** [wiring]: wires `VibeVoiceAcousticTokenizerEncoderStem`, `VibeVoiceAcousticTokenizerEncoderLayer` (×stages), `VibeVoiceAcousticTokenizerCausalConv1d` (head)
  - **`VibeVoiceAcousticTokenizerDecoderStem`** [wiring]: wires `VibeVoiceAcousticTokenizerCausalConv1d`, `VibeVoiceAcousticTokenizerConvNext1dLayer` (×N)
  - **`VibeVoiceAcousticTokenizerDecoderLayer`** [wiring]: wires `VibeVoiceAcousticTokenizerCausalConvTranspose1d`, `VibeVoiceAcousticTokenizerConvNext1dLayer` (×N)
  - **`VibeVoiceAcousticTokenizerDecoderModel`** [wiring]: wires `VibeVoiceAcousticTokenizerDecoderStem`, `VibeVoiceAcousticTokenizerDecoderLayer` (×N), `VibeVoiceAcousticTokenizerCausalConv1d` (head)
  - **`VibeVoiceAcousticTokenizerModel`** [wiring]: wires `VibeVoiceAcousticTokenizerEncoderModel`, `VibeVoiceAcousticTokenizerDecoderModel` (VAE-style with optional sampling noise)
- **task heads (0)**: — (no For* heads; primary inference is encode/decode)

## vibevoice_asr
- **src**: modeling_vibevoice_asr.py (and modular_vibevoice_asr.py)
- **hidden_act**: gelu (config.hidden_act default in VibeVoiceAsrConfig)
- **status**: composable
- **classes**:
  - **`VibeVoiceAsrRMSNorm`** [compute]: `L1/t5_layer_norm.py` (T5-style RMS — comment says equivalent)
  - **`VibeVoiceAsrMultiModalProjector`** [compute]: `L1/linear.py + L1/t5_layer_norm.py + L1/linear.py` (acoustic and semantic paths each: linear -> norm -> linear; sum)
  - **`VibeVoiceAsrFeedForward`** [compute]: `L1/linear.py + L1/gelu.py`
  - **`VibeVoiceAsrConv1dCacheLayer`** / **`VibeVoiceAsrConv1dPaddingCache`** [compute]: streaming cache helpers (skipped — cache classes)
  - **`VibeVoiceAsrCausalConv1d`** [compute]: `L1/causal_conv1d.py`
  - **`VibeVoiceAsrConvNext1dLayer`** [wiring]: wires `VibeVoiceAsrCausalConv1d`, `VibeVoiceAsrFeedForward`, `VibeVoiceAsrRMSNorm` (×2); direct nn.Parameter (gamma, ffn_gamma)
  - **`VibeVoiceAsrForConditionalGeneration`** [wiring]: wires `language_model` (AutoModelForCausalLM), `VibeVoiceAsrMultiModalProjector`, `acoustic_tokenizer_encoder` (AutoModel — VibeVoiceAcousticTokenizer encoder), `semantic_tokenizer_encoder` (AutoModel — semantic tokenizer encoder)
- **task heads (0)**: — (ForConditionalGeneration is primary head)
