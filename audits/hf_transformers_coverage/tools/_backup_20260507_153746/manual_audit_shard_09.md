## mllama
- **src**: modeling_mllama.py
- **hidden_act**: gelu (vision config), silu (text config)
- **status**: composable
- **classes**:
  - **`MllamaPrecomputedAspectRatioEmbedding`** [compute]: `L1/embedding.py` (lookup + reshape + tanh-gated add)
  - **`MllamaPrecomputedPositionEmbedding`** [compute]: `L1/embedding.py + L1/tanh.py` (param + tile_embedding lookup, gated add)
  - **`MllamaVisionMLP`** [compute]: `L2/clip_mlp.py` (CLIP-style fc1 -> ACT2FN[gelu] -> fc2; copied from CLIPMLP)
  - **`MllamaVisionAttention`** [compute]: `L2/clip_attention.py` (q/k/v/o, non-causal, dispatch via ALL_ATTENTION_FUNCTIONS, no RoPE)
  - **`MllamaVisionEncoderLayer`** [wiring]: wires `MllamaVisionAttention`, `MllamaVisionMLP`, two `nn.LayerNorm`; optional gate_attn/gate_ffn parameters (residual + tanh-gated)
  - **`MllamaVisionEncoder`** [wiring]: wires `MllamaVisionEncoderLayer`
  - **`MllamaTextRMSNorm`** [compute]: `L1/rms_norm.py`
  - **`MllamaTextCrossAttention`** [compute]: `L1/linear.py + L1/rms_norm.py + L1/dense_attention.py + L1/store_kvcache.py` (q/k/v/o + q_norm/k_norm RMSNorm; cross-attn over vision states; no exact L2 match)
  - **`MllamaTextSelfAttention`** [compute]: `L2/attention.py` (Llama-style decoder causal q/k/v/o + RoPE + KV cache)
  - **`MllamaTextMLP`** [compute]: `L2/llama_mlp.py` (gate_proj/up_proj/down_proj SwiGLU, silu via ACT2FN[hidden_act])
  - **`MllamaSelfAttentionDecoderLayer`** [wiring]: wires `MllamaTextSelfAttention`, `MllamaTextMLP`, `MllamaTextRMSNorm` (x2)
  - **`MllamaCrossAttentionDecoderLayer`** [wiring]: wires `MllamaTextCrossAttention`, `MllamaTextMLP`, `MllamaTextRMSNorm` (x2); tanh-gated residuals (cross_attn_attn_gate, cross_attn_mlp_gate)
  - **`MllamaRotaryEmbedding`** [compute]: `L1/rotary_emb.py`
  - **`MllamaVisionModel`** [wiring]: wires `MllamaVisionEncoder` (x2: transformer + global_transformer), `MllamaPrecomputedAspectRatioEmbedding` (x2), `MllamaPrecomputedPositionEmbedding`; direct `L1/conv2d.py` (patch_embedding), `L1/layer_norm.py` (x2), class_embedding param
  - **`MllamaTextModel`** [wiring]: wires `MllamaSelfAttentionDecoderLayer`, `MllamaCrossAttentionDecoderLayer`, `MllamaTextRMSNorm`, `MllamaRotaryEmbedding`; direct `L1/embedding.py`
  - **`MllamaForCausalLM`** [wiring]: wires `MllamaTextModel`; direct `L1/linear.py` (lm_head)
  - **`MllamaModel`** [wiring]: wires `MllamaVisionModel`, `MllamaTextModel`; direct `L1/linear.py` (multi_modal_projector)
  - **`MllamaForConditionalGeneration`** [wiring]: wires `MllamaModel`; direct `L1/linear.py` (lm_head)

## mlp_mixer
- **src**: NOT FOUND in HF transformers (folder /tmp/hf_transformers_pinned/src/transformers/models/mlp_mixer does not exist)
- **status**: unsupported (folder absent in HF transformers v4.x; mlp_mixer is not a HF model)
- **classes**: n/a

## mobilebert
- **src**: modeling_mobilebert.py
- **hidden_act**: relu
- **status**: composable
- **classes**:
  - **`NoNorm`** [compute]: `L1/tensor_ops.py` (input * weight + bias; no actual norm; not a kb-nano kernel match — decompose to elementwise mul/add). NOTE: not a normal LayerNorm; toggled via config.normalization_type.
  - **`MobileBertEmbeddings`** [compute]: `L2/encoder_embeddings.py` (word + position + token_type + LayerNorm/NoNorm + dropout) + extra `L1/linear.py` (embedding_transformation for trigram_input)
  - **`MobileBertSelfAttention`** [compute]: `L2/encoder_attention.py` (q/k/v + dispatch via ALL_ATTENTION_FUNCTIONS, but takes separate query/key/value tensors due to bottleneck)
  - **`MobileBertSelfOutput`** [compute]: `L2/encoder_attention.py` (dense + LayerNorm/NoNorm + residual)
  - **`MobileBertAttention`** [wiring]: wires `MobileBertSelfAttention`, `MobileBertSelfOutput`
  - **`MobileBertIntermediate`** [compute]: `L1/linear.py + L1/relu.py` (no exact L2 match — encoder_mlp.py covers Intermediate+Output but BertIntermediate is just one half; relu activation)
  - **`OutputBottleneck`** [compute]: `L1/linear.py + L1/layer_norm.py` (dense + LayerNorm/NoNorm + residual + dropout)
  - **`MobileBertOutput`** [wiring]: wires optional `OutputBottleneck`; direct `L1/linear.py + L1/layer_norm.py` (intermediate down-projection + residual)
  - **`BottleneckLayer`** [compute]: `L1/linear.py + L1/layer_norm.py` (dense to intra_bottleneck_size + LayerNorm/NoNorm)
  - **`Bottleneck`** [wiring]: wires `BottleneckLayer` (input + optional attention)
  - **`FFNOutput`** [compute]: `L1/linear.py + L1/layer_norm.py` (dense + LayerNorm/NoNorm + residual)
  - **`FFNLayer`** [wiring]: wires `MobileBertIntermediate`, `FFNOutput`
  - **`MobileBertLayer`** [wiring]: wires `MobileBertAttention`, `MobileBertIntermediate`, `MobileBertOutput`, optional `Bottleneck`, optional `FFNLayer` (×N)
  - **`MobileBertEncoder`** [wiring]: wires `MobileBertLayer`
  - **`MobileBertPooler`** [compute]: `L1/linear.py + L1/tanh.py` (first-token + dense + tanh, optional)
  - **`MobileBertPredictionHeadTransform`** [compute]: `L1/linear.py + L1/relu.py + L1/layer_norm.py`
  - **`MobileBertLMPredictionHead`** [wiring]: wires `MobileBertPredictionHeadTransform`; direct `L1/linear.py` (decoder + dense matmul concat)
  - **`MobileBertOnlyMLMHead`** [wiring]: wires `MobileBertLMPredictionHead`
  - **`MobileBertPreTrainingHeads`** [wiring]: wires `MobileBertLMPredictionHead`; direct `L1/linear.py` (seq_relationship)
  - **`MobileBertModel`** [wiring]: wires `MobileBertEmbeddings`, `MobileBertEncoder`, optional `MobileBertPooler`
  - **`MobileBertForPreTraining`** [wiring]: wires `MobileBertModel`, `MobileBertPreTrainingHeads`
  - **`MobileBertForMaskedLM`** [wiring]: wires `MobileBertModel`, `MobileBertOnlyMLMHead`
  - **`MobileBertOnlyNSPHead`** [compute]: `L1/linear.py` (single dense to 2 logits)
- **task heads (5)**: ForNextSentencePrediction, ForSequenceClassification, ForQuestionAnswering, ForMultipleChoice, ForTokenClassification — base + linear (per-task)

## mobilenet_v1
- **src**: modeling_mobilenet_v1.py
- **hidden_act**: relu6
- **status**: composable
- **classes**:
  - **`MobileNetV1ConvLayer`** [compute]: `L1/conv2d.py + L1/batch_norm2d.py + L1/relu.py` (relu6 specifically, not in kb-nano L1 — decompose to L1/relu.py + clamp; nn.Conv2d + optional BatchNorm2d + ACT2FN[hidden_act])
  - **`MobileNetV1Model`** [wiring]: wires `MobileNetV1ConvLayer` (×27 in nn.ModuleList: depthwise + pointwise per stage); direct `nn.AdaptiveAvgPool2d` -> `L1/adaptive_avg_pool2d.py`
- **task heads (1)**: ForImageClassification — base + linear (per-task)

## mobilenet_v2
- **src**: modeling_mobilenet_v2.py
- **hidden_act**: relu6
- **status**: composable
- **classes**:
  - **`MobileNetV2ConvLayer`** [compute]: `L1/conv2d.py + L1/batch_norm2d.py + L1/relu.py` (relu6; same as v1)
  - **`MobileNetV2InvertedResidual`** [wiring]: wires `MobileNetV2ConvLayer` (expand_1x1, conv_3x3, reduce_1x1) + residual add (matches `L2/efficientnetv2_inverted_residual.py` pattern but specific to MobileNet)
  - **`MobileNetV2Stem`** [wiring]: wires `MobileNetV2ConvLayer` (first_conv, optional expand_1x1, conv_3x3, reduce_1x1)
  - **`MobileNetV2Model`** [wiring]: wires `MobileNetV2Stem`, `MobileNetV2InvertedResidual` (×16), `MobileNetV2ConvLayer` (conv_1x1); direct `L1/adaptive_avg_pool2d.py`
  - **`MobileNetV2DeepLabV3Plus`** [wiring]: wires `MobileNetV2ConvLayer` (×4); direct `L1/adaptive_avg_pool2d.py`, dropout, interpolate, cat
- **task heads (2)**: ForImageClassification, ForSemanticSegmentation — base + linear (per-task)

## mobilenetv4
- **src**: NOT FOUND in HF transformers (folder /tmp/hf_transformers_pinned/src/transformers/models/mobilenetv4 does not exist; only mobilenet_v1 and mobilenet_v2 exist). MobileNetV4 has its own kb-nano L4 pipeline (`L4/mobilenetv4.py`) but is not in HF Transformers.
- **status**: unsupported (HF transformers does not include mobilenetv4 in this pinned version)
- **classes**: n/a

## mobilevit
- **src**: modeling_mobilevit.py
- **hidden_act**: silu
- **status**: composable
- **classes**:
  - **`MobileViTConvLayer`** [compute]: `L1/conv2d.py + L1/batch_norm2d.py + L1/silu.py` (Conv2d + optional BatchNorm2d + ACT2FN[silu])
  - **`MobileViTInvertedResidual`** [wiring]: wires `MobileViTConvLayer` (expand_1x1, conv_3x3, reduce_1x1) + residual
  - **`MobileViTMobileNetLayer`** [wiring]: wires `MobileViTInvertedResidual` (×num_stages)
  - **`MobileViTSelfAttention`** [compute]: `L1/linear.py + L1/dense_attention.py` (q/k/v Linear, sdpa-style softmax(QK^T)V; no exact L2 match — non-causal vision; closest is `L2/vision_attention.py`)
  - **`MobileViTSelfOutput`** [compute]: `L1/linear.py` (dense + dropout)
  - **`MobileViTAttention`** [wiring]: wires `MobileViTSelfAttention`, `MobileViTSelfOutput`
  - **`MobileViTIntermediate`** [compute]: `L1/linear.py + L1/silu.py` (BertIntermediate-style fc1 + activation)
  - **`MobileViTOutput`** [compute]: `L1/linear.py` (dense + dropout + residual)
  - **`MobileViTTransformerLayer`** [wiring]: wires `MobileViTAttention`, `MobileViTIntermediate`, `MobileViTOutput`, two `nn.LayerNorm`
  - **`MobileViTTransformer`** [wiring]: wires `MobileViTTransformerLayer`
  - **`MobileViTLayer`** [wiring]: wires optional `MobileViTInvertedResidual` (downsampling), `MobileViTConvLayer` (×4: conv_kxk, conv_1x1, conv_projection, fusion), `MobileViTTransformer`, `nn.LayerNorm`; direct unfolding/folding (reshape ops)
  - **`MobileViTEncoder`** [wiring]: wires `MobileViTMobileNetLayer` (×2), `MobileViTLayer` (×3)
  - **`MobileViTModel`** [wiring]: wires `MobileViTConvLayer` (conv_stem), `MobileViTEncoder`, optional `MobileViTConvLayer` (conv_1x1_exp); direct global mean pool
  - **`MobileViTASPPPooling`** [wiring]: wires `MobileViTConvLayer`; direct `L1/adaptive_avg_pool2d.py`, interpolate
  - **`MobileViTASPP`** [wiring]: wires `MobileViTConvLayer` (×N), `MobileViTASPPPooling`
  - **`MobileViTDeepLabV3`** [wiring]: wires `MobileViTASPP`, `MobileViTConvLayer`
- **task heads (2)**: ForImageClassification, ForSemanticSegmentation — base + linear (per-task)

## mobilevitv2
- **src**: modeling_mobilevitv2.py
- **hidden_act**: swish (= silu)
- **status**: composable
- **classes**:
  - **`MobileViTV2ConvLayer`** [compute]: `L1/conv2d.py + L1/batch_norm2d.py + L1/silu.py` (copied from MobileViTConvLayer; swish == silu)
  - **`MobileViTV2InvertedResidual`** [wiring]: wires `MobileViTV2ConvLayer` (×3) + residual
  - **`MobileViTV2MobileNetLayer`** [wiring]: wires `MobileViTV2InvertedResidual`
  - **`MobileViTV2LinearSelfAttention`** [compute]: `L1/conv2d.py + L1/softmax.py + L1/relu.py` (linear-complexity attn: qkv_proj 1x1 conv, split q/k/v, softmax(q), context = sum(k * softmax(q)), out = relu(v) * context, then out_proj conv; no exact L2 match — unique linear attention)
  - **`MobileViTV2FFN`** [wiring]: wires `MobileViTV2ConvLayer` (×2 with 1x1 conv as fc); operates as conv-based FFN, not linear-based
  - **`MobileViTV2TransformerLayer`** [wiring]: wires `MobileViTV2LinearSelfAttention`, `MobileViTV2FFN`, two `nn.GroupNorm` (-> `L1/group_norm.py`)
  - **`MobileViTV2Transformer`** [wiring]: wires `MobileViTV2TransformerLayer`
  - **`MobileViTV2Layer`** [wiring]: wires optional `MobileViTV2InvertedResidual`, `MobileViTV2ConvLayer` (×3), `MobileViTV2Transformer`, `nn.GroupNorm`; direct unfold/fold
  - **`MobileViTV2Encoder`** [wiring]: wires `MobileViTV2MobileNetLayer` (×2), `MobileViTV2Layer` (×3)
  - **`MobileViTV2Model`** [wiring]: wires `MobileViTV2ConvLayer`, `MobileViTV2Encoder`; direct mean
  - **`MobileViTV2ASPPPooling`** [wiring]: wires `MobileViTV2ConvLayer`; direct `L1/adaptive_avg_pool2d.py`, interpolate
  - **`MobileViTV2ASPP`** [wiring]: wires `MobileViTV2ConvLayer`, `MobileViTV2ASPPPooling`
  - **`MobileViTV2DeepLabV3`** [wiring]: wires `MobileViTV2ASPP`, `MobileViTV2ConvLayer`
- **task heads (2)**: ForImageClassification, ForSemanticSegmentation — base + linear (per-task)

## modernbert
- **src**: modeling_modernbert.py (and modular_modernbert.py)
- **hidden_act**: gelu (config.hidden_activation = "gelu")
- **status**: composable
- **classes**:
  - **`ModernBertEmbeddings`** [compute]: `L1/embedding.py + L1/layer_norm.py` (tok_embeddings + LayerNorm + dropout; no token_type/position embedding; not full BERT-style)
  - **`ModernBertMLP`** [compute]: `L2/llama_mlp.py` variant — Wi (chunked into input/gate) -> ACT2FN[gelu] * gate -> Wo. This is a GeGLU pattern -> closest match: `L2/geglu.py` if exists, else `L1/linear.py + L1/gelu.py + L1/linear.py` (gated GLU with gelu activation; no exact L2 match)
  - **`ModernBertRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (per-layer-type RoPE for full_attention vs sliding_attention)
  - **`ModernBertAttention`** [compute]: `L2/encoder_attention.py` variant with RoPE + sliding_window (Wqkv combined, non-causal, dispatches via ALL_ATTENTION_FUNCTIONS; no exact L2 match — encoder attention with RoPE)
  - **`ModernBertEncoderLayer`** [wiring]: wires `ModernBertAttention`, `ModernBertMLP`, two `nn.LayerNorm` (or nn.Identity for layer 0)
  - **`ModernBertModel`** [wiring]: wires `ModernBertEmbeddings`, `ModernBertEncoderLayer`, `ModernBertRotaryEmbedding`; direct `L1/layer_norm.py` (final_norm)
  - **`ModernBertPredictionHead`** [compute]: `L1/linear.py + L1/gelu.py + L1/layer_norm.py` (dense + classifier_activation + norm)
  - **`ModernBertForMaskedLM`** [wiring]: wires `ModernBertModel`, `ModernBertPredictionHead`; direct `L1/linear.py` (decoder)
- **task heads (4)**: ForSequenceClassification, ForTokenClassification, ForQuestionAnswering, ForMultipleChoice — base + linear (per-task)

## modernbert_decoder
- **src**: modeling_modernbert_decoder.py (and modular_modernbert_decoder.py)
- **hidden_act**: gelu (config.hidden_activation = "gelu")
- **status**: composable
- **classes**:
  - **`ModernBertDecoderEmbeddings`** [compute]: `L1/embedding.py + L1/layer_norm.py` (same as ModernBertEmbeddings)
  - **`ModernBertDecoderMLP`** [compute]: GeGLU `L1/linear.py + L1/gelu.py + L1/linear.py` (Wi chunked + gelu*gate + Wo; same as ModernBertMLP)
  - **`ModernBertDecoderRotaryEmbedding`** [compute]: `L1/rotary_emb.py`
  - **`ModernBertDecoderAttention`** [compute]: `L2/attention.py` variant with sliding_window (separate q/k/v Linear, RoPE, KV cache via past_key_values.update, causal=True; no exact L2 match — closest is L2/attention.py)
  - **`ModernBertDecoderLayer`** [wiring]: wires `ModernBertDecoderAttention`, `ModernBertDecoderMLP`, two `nn.LayerNorm` (or nn.Identity layer 0)
  - **`ModernBertDecoderPredictionHead`** [compute]: `L1/linear.py + L1/gelu.py + L1/layer_norm.py`
  - **`ModernBertDecoderModel`** [wiring]: wires `ModernBertDecoderEmbeddings`, `ModernBertDecoderLayer`, `ModernBertDecoderRotaryEmbedding`; direct `L1/layer_norm.py`
  - **`ModernBertDecoderForCausalLM`** [wiring]: wires `ModernBertDecoderModel`, `ModernBertDecoderPredictionHead`; direct `L1/linear.py` (decoder)
- **task heads (1)**: ForSequenceClassification — base + linear (per-task)

## modernvbert
- **src**: modeling_modernvbert.py (and modular_modernvbert.py)
- **hidden_act**: inherits from text_config (modernbert: gelu) and vision_config (siglip: gelu_pytorch_tanh)
- **status**: composable (relies on AutoModel-loaded ModernBert text + SigLIP vision)
- **classes**:
  - **`ModernVBertConnector`** [compute]: `L1/linear.py` (pixel_shuffle reshape + Linear modality_projection)
  - **`ModernVBertModel`** [wiring]: wires `ModernVBertConnector`, AutoModel(vision_config) (typically SigLIP-style, `L4/siglip2.py`), AutoModel(text_config) (ModernBertModel)
  - **`ModernVBertPredictionHead`** [compute]: `L1/linear.py + L1/gelu.py + L1/layer_norm.py` (dense + classifier_activation + norm; same as ModernBertPredictionHead)
  - **`ModernVBertForMaskedLM`** [wiring]: wires `ModernVBertModel`, `ModernVBertPredictionHead`; direct `L1/linear.py` (lm_head)
- **task heads (2)**: ForSequenceClassification, ForTokenClassification — base + linear (per-task)

## moonshine
- **src**: modeling_moonshine.py (and modular_moonshine.py)
- **hidden_act**: encoder_hidden_act=gelu, decoder_hidden_act=silu
- **status**: composable
- **classes**:
  - **`MoonshineEncoderMLP`** [compute]: `L2/whisper_mlp.py` variant (fc1 -> ACT2FN[gelu] -> fc2; matches Whisper-style 2-layer MLP)
  - **`MoonshineDecoderMLP`** [compute]: `L2/llama_mlp.py` variant (fc1 chunked into hidden/gate, ACT2FN[silu](gate) * hidden, fc2; SwiGLU pattern with combined fc1)
  - **`MoonshineRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (with partial_rotary_factor support)
  - **`MoonshineAttention`** [compute]: `L2/whisper_attention.py` variant — multi-mode (self vs cross via key_value_states), supports RoPE for self-attn only, KV cache via EncoderDecoderCache. Closest match: encoder/decoder/cross variants of `L2/whisper_attention.py` but with RoPE.
  - **`MoonshineEncoderLayer`** [wiring]: wires `MoonshineAttention` (self_attn), `MoonshineEncoderMLP`, two `nn.LayerNorm`
  - **`MoonshineDecoderLayer`** [wiring]: wires `MoonshineAttention` (self_attn + encoder_attn), `MoonshineDecoderMLP`, three `nn.LayerNorm` (input, post_attention, final)
  - **`MoonshineEncoder`** [wiring]: wires `MoonshineEncoderLayer`, `MoonshineRotaryEmbedding`; direct `L1/conv1d.py` (×3), `L1/group_norm.py` (groupnorm), `L1/layer_norm.py`, tanh, gelu (via F.tanh, F.gelu in forward)
  - **`MoonshineDecoder`** [wiring]: wires `MoonshineDecoderLayer`, `MoonshineRotaryEmbedding`; direct `L1/embedding.py`, `L1/layer_norm.py`
  - **`MoonshineModel`** [wiring]: wires `MoonshineEncoder`, `MoonshineDecoder`
  - **`MoonshineForConditionalGeneration`** [wiring]: wires `MoonshineModel`; direct `L1/linear.py` (proj_out)

## moonshine_streaming
- **src**: modeling_moonshine_streaming.py (and modular_moonshine_streaming.py)
- **hidden_act**: encoder hidden_act=gelu, decoder hidden_act=silu
- **status**: composable
- **classes**:
  - **`MoonshineStreamingFrameCMVN`** [compute]: `L1/tensor_ops.py` (mean + center + RMS divide; no exact match — ad-hoc per-frame normalization)
  - **`MoonshineStreamingAsinhCompression`** [compute]: `L1/tensor_ops.py` (asinh(exp(log_k) * x); no kb-nano kernel — direct asinh op)
  - **`MoonshineStreamingCausalConv1d`** [compute]: `L1/conv1d.py` (subclass of nn.Conv1d with left padding for causal; mask handling via additional conv1d on mask)
  - **`MoonshineStreamingLayerNorm`** [compute]: `L1/layer_norm.py` (LayerNorm without affine + learned gamma; unit_offset variant)
  - **`MoonshineStreamingEncoderMLP`** [compute]: `L2/whisper_mlp.py` variant (fc1 + gelu + fc2)
  - **`MoonshineStreamingEncoderAttention`** [compute]: `L2/encoder_attention.py` variant (q/k/v/o, non-causal, dispatch via ALL_ATTENTION_FUNCTIONS; no RoPE in encoder)
  - **`MoonshineStreamingEncoderLayer`** [wiring]: wires `MoonshineStreamingEncoderAttention`, `MoonshineStreamingEncoderMLP`, two `MoonshineStreamingLayerNorm`
  - **`MoonshineStreamingEncoderEmbedder`** [wiring]: wires `MoonshineStreamingFrameCMVN`, `MoonshineStreamingAsinhCompression`, `MoonshineStreamingCausalConv1d` (×2); direct `L1/linear.py`, silu (F.silu)
  - **`MoonshineStreamingEncoder`** [wiring]: wires `MoonshineStreamingEncoderEmbedder`, `MoonshineStreamingEncoderLayer`, `MoonshineStreamingLayerNorm`
  - **`MoonshinMoonshineStreamingDecoderMLP`** [compute]: `L2/llama_mlp.py` (gate_proj/up_proj/down_proj SwiGLU with silu)
  - **`MoonshineStreamingDecoderMLP`** [compute]: `L2/llama_mlp.py` variant (fc1 chunked SwiGLU style; same as Moonshine MoonshineDecoderMLP)
  - **`MoonshineStreamingRotaryEmbedding`** [compute]: `L1/rotary_emb.py`
  - **`MoonshineStreamingAttention`** [compute]: `L2/whisper_attention.py` variant with RoPE for self-attn (same as MoonshineAttention)
  - **`MoonshineStreamingDecoderLayer`** [wiring]: wires `MoonshineStreamingAttention` (self + encoder), `MoonshineStreamingDecoderMLP`, three `nn.LayerNorm`
  - **`MoonshineStreamingDecoder`** [wiring]: wires `MoonshineStreamingDecoderLayer`, `MoonshineStreamingRotaryEmbedding`; direct `L1/embedding.py`, `L1/layer_norm.py`
  - **`MoonshineStreamingModel`** [wiring]: wires `MoonshineStreamingEncoder`, `MoonshineStreamingDecoder`
  - **`MoonshineStreamingForConditionalGeneration`** [wiring]: wires `MoonshineStreamingModel`; direct `L1/linear.py` (proj_out)

## moshi
- **src**: modeling_moshi.py
- **hidden_act**: silu (depth_decoder_config and main config both default to silu)
- **status**: composable (decoder LLM-style + Mimi audio submodule used at inference but not declared here)
- **classes**:
  - **`MoshiRMSNorm`** [compute]: `L1/gemma_rms_norm.py` (Gemma-style RMSNorm; weight initialized to ones; no +1 offset since not gemma_rms_norm)
  - **`MoshiFlexibleLinear`** [compute]: `L1/linear.py + L1/bmm.py` (per-codebook stacked weights; index_select + bmm; no exact match)
  - **`MoshiLinear`** [wiring]: wires either `nn.Linear` (-> `L1/linear.py`) or `MoshiFlexibleLinear` (toggle by use_flexible_linear)
  - **`MoshiRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (Llama-style)
  - **`MoshiGatingMLP`** [compute]: `L2/llama_mlp.py` variant (fc1 -> chunk into 2 -> ACT2FN[silu](first) * second -> fc2; SwiGLU with combined fc1; supports per-codebook flexible linear)
  - **`MoshiAttention`** [compute]: `L2/attention.py` variant (q/k/v/o via MoshiLinear, optional RoPE, repeat_kv for GQA, KV cache; causal)
  - **`MoshiFlashAttention2`** [compute, inherits MoshiAttention]: subclass for flash_attention_2 (overrides forward to call _flash_attention_forward); `L2/attention.py` (flash backend)
  - **`MoshiSdpaAttention`** [compute, inherits MoshiAttention]: subclass for SDPA (overrides forward to use F.scaled_dot_product_attention); `L2/attention.py` (sdpa backend)
  - **`MoshiDecoderLayer`** [wiring]: wires `MoshiAttention` (per attn_implementation dict), `MoshiGatingMLP`, two `MoshiRMSNorm`
  - **`MoshiDepthDecoder`** [wiring]: wires `MoshiDecoderLayer` (×depth_decoder_num_hidden_layers); direct `L1/embedding.py`, `L1/linear.py`, `MoshiFlexibleLinear`, `MoshiRMSNorm` etc — depth-axis decoder for codebooks
  - **`MoshiModel`** [wiring]: wires `MoshiDecoderLayer`, `MoshiRMSNorm`; direct `L1/embedding.py`
  - **`MoshiForCausalLM`** [wiring]: wires `MoshiModel`; direct `L1/linear.py` (lm_head)
  - **`MoshiForConditionalGeneration`** [wiring]: wires `MoshiModel`, `MoshiDepthDecoder`, MimiModel (audio); direct `L1/linear.py` (lm_head, audio_encoder_proj)

## mpnet
- **src**: modeling_mpnet.py
- **hidden_act**: gelu
- **status**: composable
- **classes**:
  - **`MPNetEmbeddings`** [compute]: `L1/embedding.py + L1/embedding.py + L1/layer_norm.py` (word_embeddings + position_embeddings + LayerNorm; NO token_type since MPNet uses just word + position; not exact match for `L2/encoder_embeddings.py` since it lacks token_type_embeddings)
  - **`MPNetSelfAttention`** [compute]: `L1/linear.py + L1/dense_attention.py` (q/k/v/o + position_bias added to scores; no ALL_ATTENTION_FUNCTIONS dispatch; closest is `L2/encoder_attention.py` but with t5-style relative position bias)
  - **`MPNetAttention`** [wiring]: wires `MPNetSelfAttention`; direct `L1/layer_norm.py` (LayerNorm) (mid-attention LayerNorm + dropout + residual)
  - **`MPNetIntermediate`** [compute]: `L1/linear.py + L1/gelu.py` (BertIntermediate-style)
  - **`MPNetOutput`** [compute]: `L1/linear.py + L1/layer_norm.py` (BertOutput-style; dense + dropout + LayerNorm + residual)
  - **`MPNetLayer`** [wiring]: wires `MPNetAttention`, `MPNetIntermediate`, `MPNetOutput`
  - **`MPNetEncoder`** [wiring]: wires `MPNetLayer`; direct `L1/embedding.py` (relative_attention_bias) + computes T5-style relative position bias
  - **`MPNetPooler`** [compute]: `L1/linear.py + L1/tanh.py` (BertPooler-style)
  - **`MPNetModel`** [wiring]: wires `MPNetEmbeddings`, `MPNetEncoder`, optional `MPNetPooler`
  - **`MPNetForMaskedLM`** [wiring]: wires `MPNetModel`, `MPNetLMHead`
  - **`MPNetLMHead`** [compute]: `L1/linear.py + L1/gelu.py + L1/layer_norm.py + L1/linear.py` (dense + gelu + layer_norm + decoder + bias)
  - **`MPNetClassificationHead`** [compute]: `L1/linear.py + L1/tanh.py + L1/linear.py`
- **task heads (4)**: ForSequenceClassification, ForMultipleChoice, ForTokenClassification, ForQuestionAnswering — base + linear/head (per-task)

## mpt
- **src**: modeling_mpt.py
- **hidden_act**: gelu (nn.GELU(approximate="none"); not config-driven)
- **status**: composable
- **classes**:
  - **`MptAttention`** [compute]: `L2/attention.py` variant — Wqkv (combined), softmax(QK^T)V with ALiBi position bias added to scores; no RoPE; KV cache via past_key_values.update; closest match: `L2/attention.py` but with position_bias instead of RoPE (no exact L2 match)
  - **`MptMLP`** [compute]: `L1/linear.py + L1/gelu.py + L1/linear.py` (up_proj 4*hidden -> nn.GELU -> down_proj + dropout + residual; 2-layer FF with gelu, NOT SwiGLU)
  - **`MptBlock`** [wiring]: wires `MptAttention`, `MptMLP`, two `nn.LayerNorm` (norm_1, norm_2 — bias forced to None)
  - **`MptModel`** [wiring]: wires `MptBlock`; direct `L1/embedding.py` (wte), `L1/layer_norm.py` (norm_f)
  - **`MptForCausalLM`** [wiring]: wires `MptModel`; direct `L1/linear.py` (lm_head)
- **task heads (3)**: ForSequenceClassification, ForTokenClassification, ForQuestionAnswering — base + linear (per-task)

## mra
- **src**: modeling_mra.py
- **hidden_act**: gelu
- **status**: partial (MRA uses custom CUDA mra2_attention kernel with no kb-nano equivalent)
- **classes**:
  - **`MraEmbeddings`** [compute]: `L2/encoder_embeddings.py` (word + position [+2 offset] + token_type + LayerNorm + dropout; BERT-style)
  - **`MraSelfAttention`** [compute]: custom `mra2_attention` CUDA kernel (multi-resolution analysis attention); no kb-nano equivalent — `L1/linear.py + (custom mra2_attention)`. **No L1/L2 match.**
  - **`MraSelfOutput`** [compute]: `L1/linear.py + L1/layer_norm.py` (BertSelfOutput-style, copied)
  - **`MraAttention`** [wiring]: wires `MraSelfAttention`, `MraSelfOutput`
  - **`MraIntermediate`** [compute]: `L1/linear.py + L1/gelu.py` (BertIntermediate)
  - **`MraOutput`** [compute]: `L1/linear.py + L1/layer_norm.py` (BertOutput)
  - **`MraLayer`** [wiring]: wires `MraAttention`, `MraIntermediate`, `MraOutput`
  - **`MraEncoder`** [wiring]: wires `MraLayer`
  - **`MraPredictionHeadTransform`** [compute]: `L1/linear.py + L1/gelu.py + L1/layer_norm.py`
  - **`MraLMPredictionHead`** [wiring]: wires `MraPredictionHeadTransform`; direct `L1/linear.py` (decoder)
  - **`MraOnlyMLMHead`** [wiring]: wires `MraLMPredictionHead`
  - **`MraModel`** [wiring]: wires `MraEmbeddings`, `MraEncoder`
  - **`MraForMaskedLM`** [wiring]: wires `MraModel`, `MraOnlyMLMHead`
  - **`MraClassificationHead`** [compute]: `L1/linear.py + L1/tanh.py + L1/linear.py`
- **task heads (4)**: ForSequenceClassification, ForMultipleChoice, ForTokenClassification, ForQuestionAnswering — base + linear/head (per-task)

## mt5
- **src**: modeling_mt5.py
- **hidden_act**: dense_act_fn = gelu_new (when feed_forward_proj="gated-gelu"; default), or relu (when "relu")
- **status**: composable
- **classes**:
  - **`MT5LayerNorm`** [compute]: `L1/t5_layer_norm.py` (no centering, RMS-style; copied from T5LayerNorm)
  - **`MT5DenseActDense`** [compute]: `L2/t5_dense.py` variant (wi -> ACT2FN[dense_act_fn] -> wo; non-gated; `L1/linear.py + L1/relu.py or L1/gelu.py + L1/linear.py`)
  - **`MT5DenseGatedActDense`** [compute]: `L2/t5_dense.py` (wi_0/wi_1 gated + wo with gelu_new; SwiGLU/GeGLU pattern; closest: `L2/t5_dense.py`)
  - **`MT5LayerFF`** [wiring]: wires `MT5DenseActDense` or `MT5DenseGatedActDense`, `MT5LayerNorm` + residual + dropout
  - **`MT5Attention`** [compute]: `L2/t5_attention.py` (q/k/v/o + relative_attention_bias + KV cache via Cache; matches T5Attention)
  - **`MT5LayerSelfAttention`** [wiring]: wires `MT5Attention`, `MT5LayerNorm` + residual
  - **`MT5LayerCrossAttention`** [wiring]: wires `MT5Attention`, `MT5LayerNorm` + residual
  - **`MT5Block`** [wiring]: wires `MT5LayerSelfAttention`, optional `MT5LayerCrossAttention` (decoder only), `MT5LayerFF`
  - **`MT5ClassificationHead`** [compute]: `L1/linear.py + L1/tanh.py + L1/linear.py`
  - **`MT5Stack`** [wiring]: wires `MT5Block` (×N), `MT5LayerNorm`; direct `L1/embedding.py`
  - **`MT5Model`** [wiring]: wires `MT5Stack` (×2: encoder + decoder); direct `L1/embedding.py` (shared)
  - **`MT5ForConditionalGeneration`** [wiring]: wires `MT5Stack` (×2); direct `L1/linear.py` (lm_head), `L1/embedding.py` (shared)
  - **`MT5EncoderModel`** [wiring]: wires `MT5Stack`; direct `L1/embedding.py`
- **task heads (3)**: ForSequenceClassification, ForTokenClassification, ForQuestionAnswering — base + linear/head (per-task)

## musicgen
- **src**: modeling_musicgen.py
- **hidden_act**: activation_function = gelu
- **status**: composable
- **classes**:
  - **`MusicgenSinusoidalPositionalEmbedding`** [compute]: `L1/sinusoidal_embed.py` (precomputed sinusoidal weights buffer + index_select)
  - **`MusicgenAttention`** [compute]: `L2/whisper_attention.py` variant (q/k/v/out, supports cross-attn via key_value_states, EncoderDecoderCache; matches Whisper-style multi-mode attention without RoPE)
  - **`MusicgenDecoderLayer`** [wiring]: wires `MusicgenAttention` (self + encoder), three `nn.LayerNorm`; direct `L1/linear.py` (fc1 + fc2), `L1/gelu.py`
  - **`MusicgenDecoder`** [wiring]: wires `MusicgenSinusoidalPositionalEmbedding`, `MusicgenDecoderLayer`; direct `L1/embedding.py` (×num_codebooks), `L1/layer_norm.py`
  - **`MusicgenModel`** [wiring]: wires `MusicgenDecoder`
  - **`MusicgenForCausalLM`** [wiring]: wires `MusicgenModel`; direct `L1/linear.py` (lm_heads, ×num_codebooks)
  - **`MusicgenForConditionalGeneration`** [wiring]: wires `MusicgenForCausalLM` (decoder), text_encoder (T5 typically), audio_encoder (Encodec); direct `L1/linear.py` (enc_to_dec_proj)

## musicgen_melody
- **src**: modeling_musicgen_melody.py
- **hidden_act**: activation_function = gelu
- **status**: composable
- **classes**:
  - **`MusicgenMelodySinusoidalPositionalEmbedding`** [compute]: `L1/sinusoidal_embed.py` (same as Musicgen)
  - **`MusicgenMelodyAttention`** [compute]: `L2/whisper_attention.py` variant (same multi-mode attention as MusicgenAttention; q/k/v/o, cross-attn via key_value_states, EncoderDecoderCache)
  - **`MusicgenMelodyDecoderLayer`** [wiring]: wires `MusicgenMelodyAttention` (self only — no separate cross-attn since melody is concatenated to embed; differs from musicgen here), two `nn.LayerNorm`; direct `L1/linear.py` (fc1, fc2), `L1/gelu.py`
  - **`MusicgenMelodyDecoder`** [wiring]: wires `MusicgenMelodySinusoidalPositionalEmbedding`, `MusicgenMelodyDecoderLayer`; direct `L1/embedding.py` (×num_codebooks), `L1/layer_norm.py`
  - **`MusicgenMelodyModel`** [wiring]: wires `MusicgenMelodyDecoder`
  - **`MusicgenMelodyForCausalLM`** [wiring]: wires `MusicgenMelodyModel`; direct `L1/linear.py` (lm_heads)
  - **`MusicgenMelodyForConditionalGeneration`** [wiring]: wires `MusicgenMelodyForCausalLM`, text_encoder (T5), audio_encoder (Encodec); direct `L1/linear.py` (enc_to_dec_proj, audio_enc_to_dec_proj)

## mvp
- **src**: modeling_mvp.py
- **hidden_act**: activation_function = gelu
- **status**: composable
- **classes**:
  - **`MvpLearnedPositionalEmbedding`** [compute, inherits nn.Embedding]: `L1/embedding.py` (offset by 2; learned positions)
  - **`MvpAttention`** [compute]: `L2/whisper_attention.py` variant (BART-style q/k/v/out_proj, supports prompts injected into K/V — `attn_prompt`, manual softmax(QK^T)V; closest: BART/Whisper attention)
  - **`MvpEncoderLayer`** [wiring]: wires `MvpAttention`, two `nn.LayerNorm`; direct `L1/linear.py` (fc1, fc2), `L1/gelu.py`
  - **`MvpDecoderLayer`** [wiring]: wires `MvpAttention` (self_attn + encoder_attn), three `nn.LayerNorm`; direct `L1/linear.py` (fc1, fc2), `L1/gelu.py`
  - **`MvpClassificationHead`** [compute]: `L1/linear.py + L1/tanh.py + L1/linear.py`
  - **`MvpPrompt`** [compute]: `L1/embedding.py + L1/linear.py + L1/gelu.py + L1/linear.py` (prompt_embedding + nn.Sequential[Linear, GELU, Linear])
  - **`MvpEncoder`** [wiring]: wires `MvpEncoderLayer`, `MvpLearnedPositionalEmbedding`, optional `MvpPrompt`; direct `L1/embedding.py`, `L1/layer_norm.py`
  - **`MvpDecoder`** [wiring]: wires `MvpDecoderLayer`, `MvpLearnedPositionalEmbedding`, optional `MvpPrompt` (×2); direct `L1/embedding.py`, `L1/layer_norm.py`
  - **`MvpModel`** [wiring]: wires `MvpEncoder`, `MvpDecoder`; direct `L1/embedding.py` (shared)
  - **`MvpForConditionalGeneration`** [wiring]: wires `MvpModel`; direct `L1/linear.py` (lm_head)
  - **`MvpDecoderWrapper`** [wiring]: wires `MvpDecoder`
  - **`MvpForCausalLM`** [wiring]: wires `MvpDecoderWrapper`; direct `L1/linear.py` (lm_head)
- **task heads (2)**: ForSequenceClassification, ForQuestionAnswering — base + classification head/linear (per-task)

## nanochat
- **src**: modeling_nanochat.py (and modular_nanochat.py)
- **hidden_act**: relu2 (squared relu)
- **status**: composable
- **classes**:
  - **`NanoChatRMSNorm`** [compute]: `L1/rms_norm.py` variant (no learnable weight; just normalizes; eps only). Note: differs from standard RMSNorm in that there's no weight. Closest: `L1/rms_norm.py` with weight=1 / `L1/l2_norm.py`.
  - **`NanoChatRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (Llama-style)
  - **`NanoChatAttention`** [compute]: `L2/attention.py` variant (q/k/v/o + RoPE applied BEFORE q_norm/k_norm; KV cache; causal). Uses `NanoChatRMSNorm` for q_norm/k_norm post-RoPE.
  - **`NanoChatMLP`** [compute]: `L1/linear.py + L1/squared_relu.py + L1/linear.py` (fc1 -> ACT2FN[relu2] -> fc2; 2-layer with squared relu, NOT SwiGLU)
  - **`NanoChatDecoderLayer`** [wiring]: wires `NanoChatAttention`, `NanoChatMLP`, two `NanoChatRMSNorm`
  - **`NanoChatModel`** [wiring]: wires `NanoChatDecoderLayer`, `NanoChatRotaryEmbedding`, `NanoChatRMSNorm`; direct `L1/embedding.py`
  - **`NanoChatForCausalLM`** [wiring]: wires `NanoChatModel`; direct `L1/linear.py` (lm_head)

## nemotron
- **src**: modeling_nemotron.py
- **hidden_act**: relu2 (squared relu)
- **status**: composable
- **classes**:
  - **`NemotronLayerNorm1P`** [compute, inherits nn.LayerNorm]: `L1/layer_norm.py` variant — applies F.layer_norm with weight+1 (offset). No exact kb-nano kernel; decompose to `L1/layer_norm.py` with adjusted weight.
  - **`NemotronRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (with partial_rotary_factor support)
  - **`NemotronMLP`** [compute]: `L1/linear.py + L1/squared_relu.py + L1/linear.py` (up_proj -> ACT2FN[relu2] -> down_proj; 2-layer with squared relu, NOT SwiGLU)
  - **`NemotronAttention`** [compute]: `L2/attention.py` (q/k/v/o + partial RoPE + KV cache; causal). Has internal RotaryEmbedding instance.
  - **`NemotronFlashAttention2`** [compute, inherits NemotronAttention]: subclass for flash backend; same kb-nano mapping
  - **`NemotronSdpaAttention`** [compute, inherits NemotronAttention]: subclass for SDPA backend; same kb-nano mapping
  - **`NemotronDecoderLayer`** [wiring]: wires `NemotronAttention` (per attn_implementation), `NemotronMLP`, two `NemotronLayerNorm1P`
  - **`NemotronModel`** [wiring]: wires `NemotronDecoderLayer`, `NemotronRotaryEmbedding`, `NemotronLayerNorm1P`; direct `L1/embedding.py`
  - **`NemotronForCausalLM`** [wiring]: wires `NemotronModel`; direct `L1/linear.py` (lm_head)
- **task heads (3)**: ForSequenceClassification, ForQuestionAnswering, ForTokenClassification — base + linear (per-task; uses Generic mixins)

## nemotron_h
- **src**: modeling_nemotron_h.py (and modular_nemotron_h.py)
- **hidden_act**: mlp_hidden_act = relu2; mamba_hidden_act = silu
- **status**: composable (Mamba2 + Attention + MoE hybrid)
- **classes**:
  - **`NemotronHMamba2Mixer`** [compute]: `L2/mamba2_mixer.py` (Mamba2 SSM with chunk scan, conv1d, in_proj/out_proj; uses external causal-conv1d and mamba-ssm kernels; closest: `L2/mamba2_mixer.py`)
  - **`NemotronHRMSNorm`** [compute]: `L1/rms_norm.py` (T5LayerNorm-equivalent)
  - **`NemotronHMLP`** [compute]: `L1/linear.py + L1/squared_relu.py + L1/linear.py` (up_proj -> ACT2FN[relu2] -> down_proj; non-gated 2-layer MLP)
  - **`NemotronHExperts`** [compute]: `L1/moe_grouped_gemm.py` variant — non-gated experts (only up_proj + down_proj; matches `L1/moe_grouped_gemm.py` for non-gated MoE)
  - **`NemotronHMoE`** [wiring]: wires `NemotronHExperts`, `NemotronHTopkRouter`, `NemotronHMLP` (shared_experts), optional latent projections (`L1/linear.py` x2)
  - **`NemotronHTopkRouter`** [compute]: `L1/linear.py` (single weight matmul + bias param)
  - **`NemotronHAttention`** [compute]: `L2/attention.py` (q/k/v/o, no RoPE in standalone attn; KV cache via past_key_values.update; causal)
  - **`NemotronHBlock`** [wiring]: wires `NemotronHRMSNorm` + one of {NemotronHMamba2Mixer, NemotronHAttention, NemotronHMoE, NemotronHMLP} per layers_block_type
  - **`NemotronHModel`** [wiring]: wires `NemotronHBlock`, `NemotronHRMSNorm`; direct `L1/embedding.py`
  - **`NemotronHForCausalLM`** [wiring]: wires `NemotronHModel`; direct `L1/linear.py` (lm_head)

## nllb_moe
- **src**: modeling_nllb_moe.py
- **hidden_act**: activation_function = relu
- **status**: composable
- **classes**:
  - **`NllbMoeScaledWordEmbedding`** [compute, inherits nn.Embedding]: `L1/embedding.py` (with embed_scale multiplier on output)
  - **`NllbMoeSinusoidalPositionalEmbedding`** [compute]: `L1/sinusoidal_embed.py` (sinusoidal weights, padded for padding_idx)
  - **`NllbMoeTop2Router`** [compute]: `L1/linear.py + L1/softmax.py + L1/sigmoid_topk.py` (classifier Linear + softmax + top-2 selection with capacity; no exact kb-nano kernel match — fairseq-style top-2)
  - **`NllbMoeDenseActDense`** [compute]: `L1/linear.py + L1/relu.py + L1/linear.py` (fc1 -> ACT2FN[relu] -> fc2; T5 DenseActDense-style; non-gated)
  - **`NllbMoeExperts`** [compute, inherits nn.ModuleDict]: `L1/moe_grouped_gemm.py` (collection of NllbMoeDenseActDense per expert; iterates over expert mask)
  - **`NllbMoeSparseMLP`** [wiring]: wires `NllbMoeTop2Router`, `NllbMoeExperts`
  - **`NllbMoeAttention`** [compute]: `L2/whisper_attention.py` variant (BART-style q/k/v/out_proj with optional cross-attn via key_value_states, EncoderDecoderCache; dispatches via ALL_ATTENTION_FUNCTIONS)
  - **`NllbMoeEncoderLayer`** [wiring]: wires `NllbMoeAttention`, optional `NllbMoeSparseMLP` or `NllbMoeDenseActDense`, two `nn.LayerNorm`
  - **`NllbMoeDecoderLayer`** [wiring]: wires `NllbMoeAttention` (self + cross), optional `NllbMoeSparseMLP` or `NllbMoeDenseActDense`, three `nn.LayerNorm`
  - **`NllbMoeEncoder`** [wiring]: wires `NllbMoeScaledWordEmbedding`, `NllbMoeSinusoidalPositionalEmbedding`, `NllbMoeEncoderLayer`; direct `L1/layer_norm.py`
  - **`NllbMoeDecoder`** [wiring]: wires `NllbMoeScaledWordEmbedding`, `NllbMoeSinusoidalPositionalEmbedding`, `NllbMoeDecoderLayer`; direct `L1/layer_norm.py`
  - **`NllbMoeModel`** [wiring]: wires `NllbMoeEncoder`, `NllbMoeDecoder`; direct `L1/embedding.py` (shared)
  - **`NllbMoeForConditionalGeneration`** [wiring]: wires `NllbMoeModel`; direct `L1/linear.py` (lm_head)

## nomic_bert
- **src**: modeling_nomic_bert.py (and modular_nomic_bert.py)
- **hidden_act**: silu
- **status**: composable
- **classes**:
  - **`NomicBertEmbeddings`** [compute]: `L1/embedding.py + L1/embedding.py + L1/layer_norm.py` (word + token_type + LayerNorm + dropout; no position_embeddings since uses RoPE; not exact `L2/encoder_embeddings.py`)
  - **`NomicBertRotaryEmbedding`** [compute]: `L1/rotary_emb.py`
  - **`NomicBertAttention`** [compute]: `L2/encoder_attention.py` variant with RoPE — q/k/v/o, non-causal, RoPE applied, dispatches via ALL_ATTENTION_FUNCTIONS (no exact L2 match — encoder attn with RoPE)
  - **`NomicBertMLP`** [compute]: `L2/llama_mlp.py` (gate_proj/up_proj/down_proj SwiGLU with silu; copied from LlamaMLP)
  - **`NomicBertLayer`** [wiring]: wires `NomicBertAttention`, `NomicBertMLP`, two `nn.LayerNorm` (post_attention_layernorm, post_mlp_layernorm); post-norm style
  - **`NomicBertLMPredictionHead`** [wiring]: wires `NomicBertPredictionHeadTransform`; direct `L1/linear.py` (decoder)
  - **`NomicBertPooler`** [compute]: `L1/linear.py + L1/tanh.py` (BertPooler-style)
  - **`NomicBertModel`** [wiring]: wires `NomicBertEmbeddings`, `NomicBertLayer`, `NomicBertRotaryEmbedding`, optional `NomicBertPooler`
  - **`NomicBertPredictionHeadTransform`** [compute]: `L1/linear.py + L1/silu.py + L1/layer_norm.py`
  - **`NomicBertOnlyMLMHead`** [wiring]: wires `NomicBertLMPredictionHead`
  - **`NomicBertForMaskedLM`** [wiring]: wires `NomicBertModel`, `NomicBertOnlyMLMHead`
- **task heads (2)**: ForSequenceClassification, ForTokenClassification — base + linear (per-task)

## nystromformer
- **src**: modeling_nystromformer.py
- **hidden_act**: gelu_new
- **status**: partial (uses iterative Moore-Penrose inverse on softmax outputs; no exact kb-nano kernel for Nystrom approximation)
- **classes**:
  - **`NystromformerEmbeddings`** [compute]: `L2/encoder_embeddings.py` (word + position [+2 offset] + token_type + LayerNorm + dropout; BERT-style)
  - **`NystromformerSelfAttention`** [compute]: `L1/linear.py + L1/softmax.py + L1/conv2d.py + custom iterative_inv` (q/k/v Linear; if num_landmarks==seq_len: standard softmax(QK^T)V; else Nystrom approximation: kernel_1 * iterative_inv(kernel_2) * kernel_3 * V; optional conv2d residual). **No L2 match — Nystrom-specific.**
  - **`NystromformerSelfOutput`** [compute]: `L1/linear.py + L1/layer_norm.py` (BertSelfOutput, copied)
  - **`NystromformerAttention`** [wiring]: wires `NystromformerSelfAttention`, `NystromformerSelfOutput`
  - **`NystromformerIntermediate`** [compute]: `L1/linear.py + L1/gelu.py` (BertIntermediate, copied; gelu_new)
  - **`NystromformerOutput`** [compute]: `L1/linear.py + L1/layer_norm.py` (BertOutput, copied)
  - **`NystromformerLayer`** [wiring]: wires `NystromformerAttention`, `NystromformerIntermediate`, `NystromformerOutput`
  - **`NystromformerEncoder`** [wiring]: wires `NystromformerLayer`
  - **`NystromformerPredictionHeadTransform`** [compute]: `L1/linear.py + L1/gelu.py + L1/layer_norm.py`
  - **`NystromformerLMPredictionHead`** [wiring]: wires `NystromformerPredictionHeadTransform`; direct `L1/linear.py`
  - **`NystromformerOnlyMLMHead`** [wiring]: wires `NystromformerLMPredictionHead`
  - **`NystromformerModel`** [wiring]: wires `NystromformerEmbeddings`, `NystromformerEncoder`
  - **`NystromformerForMaskedLM`** [wiring]: wires `NystromformerModel`, `NystromformerOnlyMLMHead`
  - **`NystromformerClassificationHead`** [compute]: `L1/linear.py + L1/gelu.py + L1/linear.py`
- **task heads (4)**: ForSequenceClassification, ForMultipleChoice, ForTokenClassification, ForQuestionAnswering — base + linear/head (per-task)

## olmo
- **src**: modeling_olmo.py (and modular_olmo.py)
- **hidden_act**: silu
- **status**: composable
- **classes**:
  - **`OlmoLayerNorm`** [compute]: `L1/layer_norm.py` (no learnable weight or bias; pure F.layer_norm with normalized_shape; closest match: `L1/layer_norm.py` with affine=False)
  - **`OlmoMLP`** [compute]: `L2/llama_mlp.py` (gate_proj/up_proj/down_proj SwiGLU with silu; matches LlamaMLP exactly)
  - **`OlmoRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (Llama-style)
  - **`OlmoAttention`** [compute]: `L2/attention.py` (q/k/v/o + RoPE + KV cache + causal; has clip_qkv option to clamp QKV before reshape)
  - **`OlmoDecoderLayer`** [wiring]: wires `OlmoAttention`, `OlmoMLP`, two `OlmoLayerNorm`
  - **`OlmoModel`** [wiring]: wires `OlmoDecoderLayer`, `OlmoRotaryEmbedding`, `OlmoLayerNorm`; direct `L1/embedding.py`
  - **`OlmoForCausalLM`** [wiring]: wires `OlmoModel`; direct `L1/linear.py` (lm_head)
- **task heads (1)**: ForSequenceClassification — base + linear (per-task; uses Generic mixin)
