# Manual audit shard 05 — falcon through glm4v_moe

## falcon
- **src**: modeling_falcon.py
- **hidden_act**: gelu (config `activation` default)
- **status**: partial
- **classes**:
  - **`FalconLinear`** [compute, inherits `nn.Linear`]: `L1/linear.py` (matmul + optional bias add; identical math to nn.Linear)
  - **`FalconRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (Llama-style RoPE; standard `apply_rotary_pos_emb`)
  - **`FalconAttention`** [compute]: `L1/linear.py + L1/rotary_emb.py + L1/store_kvcache.py + L1/dense_attention.py` (no exact L2 match — uses fused `query_key_value` single Linear with multi_query/new_decoder_architecture splits, plus optional alibi bias and broadcasting; L2/attention.py expects separate q/k/v projections)
  - **`FalconFlashAttention2`** [compute, inherits `FalconAttention`]: same fused-QKV + RoPE pattern but flash-attention path; same kb-nano decomposition (`L1/linear.py + L1/rotary_emb.py + L1/store_kvcache.py + L1/flash_attn_varlen.py`)
  - **`FalconMLP`** [compute]: `L1/linear.py + L1/gelu.py + L1/linear.py` (2-layer fc1->act->fc2; gelu activation; not SwiGLU so doesn't match L2/llama_mlp.py; encoder_mlp.py expects BertIntermediate/Output split)
  - **`FalconDecoderLayer`** [wiring, inherits `GradientCheckpointingLayer`]: wires `FalconAttention`/`FalconFlashAttention2`, `FalconMLP`; direct `L1/layer_norm.py` (×1 to ×2 LayerNorms depending on parallel_attn / new_decoder_architecture / num_ln_in_parallel_attn)
  - **`FalconModel`** [wiring]: wires `FalconDecoderLayer`, `FalconRotaryEmbedding`; direct `L1/embedding.py` (word_embeddings), `L1/layer_norm.py` (ln_f)
  - **`FalconForCausalLM`** [wiring]: wires `FalconModel`; direct `L1/linear.py` (lm_head)
- **task heads (3)**: ForSequenceClassification, ForTokenClassification, ForQuestionAnswering — base + linear (per-task)

## falcon_h1
- **src**: modeling_falcon_h1.py (and modular_falcon_h1.py)
- **hidden_act**: silu
- **status**: partial
- **classes**:
  - **`FalconH1RotaryEmbedding`** [compute]: `L1/rotary_emb.py` (Llama-style RoPE)
  - **`FalconH1Attention`** [compute]: `L2/attention.py` (q/k/v + RoPE + cache update + dispatch + o_proj; the only divergence is a per-layer `key_multiplier` scalar applied to `key_states` after k_proj — small numerical tweak, attention structure matches)
  - **`FalconH1RMSNormGated`** [compute]: `L1/rms_norm_gated.py` (RMSNorm with optional gated input — note `norm_before_gate` flag)
  - **`FalconH1Mixer`** [compute, adapted from `Mamba2Mixer`]: `L2/mamba2_mixer.py` (mamba2 SSM mixer with conv1d, in_proj, dt_bias, A_log, chunk-scan SSM; FalconH1 adds optional `mup_vector` and configurable `mamba_d_ssm`)
  - **`FalconH1MLP`** [compute]: `L2/llama_mlp.py` (gate*up SwiGLU pattern: `down_proj(up_proj(x) * silu(gate_proj(x) * gate_multiplier)) * down_multiplier`; multipliers are scalar tweaks on top of standard SwiGLU)
  - **`FalconH1RMSNorm`** [compute]: `L1/rms_norm.py` (standard Llama-style RMSNorm)
  - **`FalconH1DecoderLayer`** [wiring, inherits `GradientCheckpointingLayer`]: wires `FalconH1Mixer`, `FalconH1Attention`, `FalconH1MLP`, `FalconH1RMSNorm` (×2: input_layernorm, pre_ff_layernorm); hybrid mamba+attention in parallel
  - **`FalconH1Model`** [wiring]: wires `FalconH1DecoderLayer`, `FalconH1RotaryEmbedding`, `FalconH1RMSNorm` (final_layernorm); direct `L1/embedding.py`
  - **`FalconH1ForCausalLM`** [wiring]: wires `FalconH1Model`; direct `L1/linear.py` (lm_head)

## falcon_mamba
- **src**: modeling_falcon_mamba.py (and modular_falcon_mamba.py)
- **hidden_act**: silu
- **status**: composable
- **classes**:
  - **`FalconMambaMixer`** [compute]: `L2/mamba_mixer.py` (Mamba1 mixer: in_proj split into x and z, conv1d, x_proj for dt/B/C, dt_proj, A_log/D, selective scan; FalconMamba adds extra B/C RMSNorm layer per the paper but operator structure is Mamba1)
  - **`FalconMambaRMSNorm`** [compute]: `L1/rms_norm.py` (Llama-style RMSNorm; equivalent to T5LayerNorm per docstring)
  - **`FalconMambaBlock`** [wiring, inherits `GradientCheckpointingLayer`]: wires `FalconMambaRMSNorm`, `FalconMambaMixer`; residual add only
  - **`FalconMambaModel`** [wiring]: wires `FalconMambaBlock`, `FalconMambaRMSNorm` (final norm); direct `L1/embedding.py`
  - **`FalconMambaForCausalLM`** [wiring]: wires `FalconMambaModel`; direct `L1/linear.py` (lm_head)

## fast_vlm
- **src**: modeling_fast_vlm.py (and modular_fast_vlm.py)
- **hidden_act**: gelu (projector_hidden_act default)
- **status**: composable
- **classes**:
  - **`FastVlmMultiModalProjector`** [compute]: `L1/linear.py + L1/gelu.py + L1/linear.py` (linear -> act -> linear)
  - **`FastVlmModel`** [wiring]: wires `FastVlmMultiModalProjector`, vision tower (AutoModel from vision_config — typically MobileCLIP/FastViT), language model (AutoModel from text_config — typically a Llama-family CausalLM backbone)
  - **`FastVlmForConditionalGeneration`** [wiring]: wires `FastVlmModel`; direct `L1/linear.py` (lm_head)

## fastspeech2_conformer
- **src**: modeling_fastspeech2_conformer.py
- **hidden_act**: silu (default for ConvolutionModule when not given via module_config)
- **status**: unsupported
- **classes**:
  - **`FastSpeech2ConformerDurationPredictor`** [wiring]: wires `FastSpeech2ConformerPredictorLayer`; direct `L1/linear.py` (final regression linear); inference path adds `clamp(round(exp() - 1))` (no kb-nano kernel match for the inference postproc)
  - **`FastSpeech2ConformerBatchNormConvLayer`** [compute]: `L1/conv1d.py + L1/batch_norm2d.py + L1/tanh.py` (note: kb-nano has no `batch_norm1d.py` — partial match, `L1/batch_norm2d.py` is the closest analog but interface differs)
  - **`FastSpeech2ConformerSpeechDecoderPostnet`** [wiring]: wires `FastSpeech2ConformerBatchNormConvLayer`; direct `L1/linear.py` (feat_out)
  - **`FastSpeech2ConformerPredictorLayer`** [compute]: `L1/conv1d.py + L1/relu.py + L1/layer_norm.py` (conv1d -> relu -> layer_norm with transpose around it)
  - **`FastSpeech2ConformerVariancePredictor`** [wiring]: wires `FastSpeech2ConformerPredictorLayer`; direct `L1/linear.py`
  - **`FastSpeech2ConformerVarianceEmbedding`** [compute]: `L1/conv1d.py` (single Conv1d with transposes)
  - **`FastSpeech2ConformerAttention`** [compute]: `L1/linear.py + L1/dense_attention.py` (no exact L2 match — Conformer relative-position attention with `pos_bias_u`/`pos_bias_v` learned biases, separate q/k/v + linear_pos, manual matmul + relative shift; this is the rel-pos rel-pos-encoding attention pattern from Transformer-XL/Conformer that none of the kb-nano L2 attention files implement)
  - **`FastSpeech2ConformerConvolutionModule`** [compute]: `L1/conv1d.py + L1/batch_norm2d.py + L1/silu.py + L1/conv1d.py` (pointwise_conv1 -> GLU -> depthwise_conv -> batch_norm -> silu -> pointwise_conv2; uses `nn.functional.glu` which has no kb-nano kernel)
  - **`FastSpeech2ConformerEncoderLayer`** [wiring]: wires `FastSpeech2ConformerAttention`, `FastSpeech2ConformerMultiLayeredConv1d` (×2 for macaron), `FastSpeech2ConformerConvolutionModule`; direct `L1/layer_norm.py` (×4)
  - **`FastSpeech2ConformerMultiLayeredConv1d`** [compute]: `L1/conv1d.py + L1/relu.py + L1/conv1d.py` (replaces position-wise FFN with two conv1d)
  - **`FastSpeech2ConformerRelPositionalEncoding`** [compute]: no kb-nano kernel — sinusoidal relative positional encoding with positive/negative concat for shifting trick (custom math)
  - **`FastSpeech2ConformerEncoder`** [wiring]: wires `FastSpeech2ConformerEncoderLayer`, `FastSpeech2ConformerRelPositionalEncoding`; direct `L1/embedding.py` (optional)
  - **`FastSpeech2ConformerLoss`** [compute]: skip — training-only loss module
  - **`FastSpeech2ConformerModel`** [wiring]: wires `FastSpeech2ConformerEncoder` (encoder + decoder), `FastSpeech2ConformerDurationPredictor`, `FastSpeech2ConformerVariancePredictor` (×2), `FastSpeech2ConformerVarianceEmbedding` (×2), `FastSpeech2ConformerSpeechDecoderPostnet`, `FastSpeech2ConformerLoss`; direct `L1/embedding.py` (×2 optional speaker/lang), `L1/linear.py` (projection)
  - **`HifiGanResidualBlock`** [compute]: `L1/leaky_relu.py + L1/conv1d.py` (×N residual conv blocks with leaky_relu)
  - **`FastSpeech2ConformerHifiGan`** [wiring, inherits `PreTrainedModel`]: wires `HifiGanResidualBlock`; direct `L1/conv1d.py` (conv_pre, conv_post), `L1/conv_transpose1d.py` (upsamplers), `L1/leaky_relu.py`, `L1/tanh.py`
  - **`FastSpeech2ConformerWithHifiGan`** [wiring, inherits `PreTrainedModel`]: wires `FastSpeech2ConformerModel`, `FastSpeech2ConformerHifiGan`

## flaubert
- **src**: modeling_flaubert.py
- **hidden_act**: gelu (config `gelu_activation=True` default → gelu, else relu)
- **status**: composable
- **classes**:
  - **`MultiHeadAttention`** [compute]: `L1/linear.py + L1/dense_attention.py + L1/store_kvcache.py` (no exact L2 match — XLM/Flaubert non-causal encoder self-attention with EncoderDecoderCache for cross-attention; structure is q_lin/k_lin/v_lin + manual matmul + softmax + out_lin; closest is L2/encoder_attention.py but flaubert lacks a separate self_output residual+layernorm wrapper)
  - **`TransformerFFN`** [compute]: `L1/linear.py + L1/gelu.py + L1/linear.py` (lin1 -> gelu/relu -> lin2; 2-layer fc1->act->fc2 like encoder_mlp but Flaubert keeps it as a single class with no separate Intermediate/Output split, so encoder_mlp.py isn't an exact match)
  - **`FlaubertPredLayer`** [compute]: `L1/linear.py` (when asm=False, just a Linear projection to vocab; otherwise nn.AdaptiveLogSoftmaxWithLoss which has no kb-nano analog)
  - **`FlaubertPoolerStartLogits`** [compute]: `L1/linear.py` (single dense to 1)
  - **`FlaubertPoolerEndLogits`** [compute]: `L1/linear.py + L1/tanh.py + L1/layer_norm.py + L1/linear.py`
  - **`FlaubertPoolerAnswerClass`** [compute]: `L1/linear.py + L1/tanh.py + L1/linear.py` (cls projection)
  - **`FlaubertSQuADHead`** [wiring]: wires `FlaubertPoolerStartLogits`, `FlaubertPoolerEndLogits`, `FlaubertPoolerAnswerClass`
  - **`FlaubertSequenceSummary`** [compute]: `L1/linear.py + L1/tanh.py` (summary projection + optional activation)
  - **`FlaubertModel`** [wiring]: wires `MultiHeadAttention` (×n_layers), `TransformerFFN` (×n_layers); direct `L1/embedding.py` (×3: word, position, optional lang), `L1/layer_norm.py` (layer_norm_emb + per-layer layer_norm1 + layer_norm2)
  - **`FlaubertWithLMHeadModel`** [wiring]: wires `FlaubertModel`, `FlaubertPredLayer`
- **task heads (5)**: ForSequenceClassification, ForTokenClassification, ForQuestionAnsweringSimple, ForQuestionAnswering, ForMultipleChoice — base + linear (per-task)

## flava
- **src**: modeling_flava.py
- **hidden_act**: gelu
- **status**: composable
- **classes**:
  - **`FlavaImageEmbeddings`** [compute]: `L2/vision_patch_embed.py + L1/embedding.py` (cls_token + patch_embeddings + position_embeddings; uses bicubic interpolate for variable resolution; not a perfect L2 match — vision_patch_embed.py varies — fall back to `L1/conv2d.py + L1/embedding.py + L1/embedding.py`)
  - **`PatchEmbeddings`** [compute]: `L1/conv2d.py` (single Conv2d with patch_size kernel/stride; flatten + transpose)
  - **`FlavaTextEmbeddings`** [compute]: `L2/encoder_embeddings.py` (BERT-style word + position + token_type + LayerNorm + Dropout — matches `bert_embeddings.py`/`encoder_embeddings.py`)
  - **`FlavaSelfAttention`** [compute]: `L2/encoder_attention.py` (q/k/v + manual matmul/softmax + reshape; vision-style without separate output projection)
  - **`FlavaSelfOutput`** [compute]: `L1/linear.py` (just dense + dropout; ViT-style — residual is in FlavaLayer; partial encoder_attention match)
  - **`FlavaAttention`** [wiring]: wires `FlavaSelfAttention`, `FlavaSelfOutput`
  - **`FlavaIntermediate`** [compute]: `L1/linear.py + L1/gelu.py` (dense + activation)
  - **`FlavaOutput`** [compute]: `L1/linear.py` (dense + dropout + residual; residual added inside)
  - **`FlavaLayer`** [wiring, inherits `GradientCheckpointingLayer`]: wires `FlavaAttention`, `FlavaIntermediate`, `FlavaOutput`; direct `L1/layer_norm.py` (×2: layernorm_before, layernorm_after — pre-norm ViT pattern)
  - **`FlavaEncoder`** [wiring]: wires `FlavaLayer` (×n_hidden_layers)
  - **`FlavaPooler`** [compute]: `L1/linear.py + L1/tanh.py` (BERT-style CLS pool)
  - **`FlavaImageModel`** [wiring]: wires `FlavaImageEmbeddings`, `FlavaEncoder`, `FlavaPooler`; direct `L1/layer_norm.py`
  - **`FlavaTextModel`** [wiring]: wires `FlavaTextEmbeddings`, `FlavaEncoder`, `FlavaPooler`; direct `L1/layer_norm.py`
  - **`FlavaMultimodalModel`** [wiring]: wires `FlavaEncoder`, `FlavaPooler`; direct `L1/layer_norm.py`
  - **`FlavaModel`** [wiring]: wires `FlavaImageModel`, `FlavaTextModel`, `FlavaMultimodalModel`; direct `L1/linear.py` (×3 image/text/multimodal_projection)
  - **`FlavaImageCodebookResPath`** [compute]: `L1/relu.py + L1/conv2d.py` (×4 alternating — DALL-E style residual conv path)
  - **`FlavaImageCodebookBlock`** [wiring]: wires `FlavaImageCodebookResPath`; direct `L1/conv2d.py` (id_path 1x1 conv when in_size != out_size)
  - **`FlavaImageCodebookLayerGroup`** [wiring]: wires `FlavaImageCodebookBlock`; direct `L1/max_pool2d.py` (optional pool)
  - **`FlavaImageCodebook`** [wiring, inherits `FlavaPreTrainedModel`]: wires `FlavaImageCodebookLayerGroup` (×4 groups); direct `L1/conv2d.py` (×2 input + output), `L1/relu.py`
  - **`FlavaPredictionHeadTransform`** [compute]: `L1/linear.py + L1/gelu.py + L1/layer_norm.py`
  - **`FlavaMaskedPredictionHead`** [wiring]: wires `FlavaPredictionHeadTransform`; direct `L1/linear.py` (decoder)
  - **`FlavaITMHead`** [wiring]: wires `FlavaPooler`; direct `L1/linear.py` (seq_relationship to 2 classes)
  - **`FlavaGlobalContrastiveHead`** [compute]: matmul + temperature scaling + labels arange — no kb-nano kernel needed beyond `L1/linear.py` (none — just matmul); skip-able as it's training-only contrastive loss head
  - **`FlavaForPreTraining`** [wiring]: wires `FlavaModel`, `FlavaImageCodebook` (optional), `FlavaMaskedPredictionHead` (×4: mim/mlm/mmm_image/mmm_text), `FlavaITMHead`, `FlavaGlobalContrastiveHead`

## flex_olmo
- **src**: modeling_flex_olmo.py (and modular_flex_olmo.py)
- **hidden_act**: silu
- **status**: composable
- **classes**:
  - **`FlexOlmoRMSNorm`** [compute]: `L1/rms_norm.py`
  - **`FlexOlmoRotaryEmbedding`** [compute]: `L1/rotary_emb.py`
  - **`FlexOlmoMLP`** [compute]: `L2/llama_mlp.py` (standard SwiGLU: down_proj(silu(gate_proj(x)) * up_proj(x)))
  - **`FlexOlmoAttention`** [compute]: `L2/attention.py` (q/k/v + RoPE + cache + dispatch + o_proj; OLMo-style with full-hidden_size q_norm/k_norm RMSNorm BEFORE the view-to-heads — note: this is not per-head qk-norm; matches LlamaAttention with qk_norm=True modulo the norm being on the flat dim)
  - **`FlexOlmoTopKRouter`** [compute]: `L1/linear.py + L1/topk_softmax.py` (router weight + softmax + topk + optional norm)
  - **`FlexOlmoExperts`** [compute]: `L1/moe_grouped_gemm.py` or `L2/fused_experts.py` (per-expert gate_up + down with SwiGLU; standard fused MoE pattern)
  - **`FlexOlmoSparseMoeBlock`** [wiring]: wires `FlexOlmoTopKRouter`, `FlexOlmoExperts`; closest match `L2/qwen3_moe.py` (router + fused experts pattern)
  - **`FlexOlmoDecoderLayer`** [wiring, inherits `GradientCheckpointingLayer`]: wires `FlexOlmoAttention`, `FlexOlmoSparseMoeBlock`, `FlexOlmoRMSNorm` (×2: post_attention_layernorm, post_feedforward_layernorm — note: post-norm pattern, not pre-norm)
  - **`FlexOlmoModel`** [wiring]: wires `FlexOlmoDecoderLayer`, `FlexOlmoRotaryEmbedding`, `FlexOlmoRMSNorm` (final norm); direct `L1/embedding.py`
  - **`FlexOlmoForCausalLM`** [wiring]: wires `FlexOlmoModel`; direct `L1/linear.py` (lm_head)

## florence2
- **src**: modeling_florence2.py (and modular_florence2.py)
- **hidden_act**: gelu (vision_config.activation_function default)
- **status**: partial
- **classes**:
  - **`Florence2VisionDropPath`** [compute]: `L1/dropout.py` (stochastic depth via `drop_path` helper; partial — kb-nano doesn't have a dedicated drop-path)
  - **`Florence2VisionLearnedAbsolutePositionEmbedding2D`** [compute]: `L1/embedding.py + L1/embedding.py` (separate row + column embeddings concatenated)
  - **`Florence2VisionPositionalEmbeddingCosine1D`** [compute]: no kb-nano kernel — sinusoidal 1D positional encoding stored in a buffer (custom math)
  - **`Florence2VisionMLP`** [compute]: `L1/linear.py + L1/gelu.py + L1/linear.py` (fc1 -> activation -> fc2 — DaViT FFN)
  - **`Florence2VisionConvEmbed`** [compute]: `L1/conv2d.py + L1/layer_norm.py` (Conv2d patch embed + optional pre/post LayerNorm with permute)
  - **`Florence2VisionChannelAttention`** [compute]: `L1/linear.py + L1/dense_attention.py + L1/linear.py` (no L2 match — DaViT channel-group attention: qkv + grouped channel-to-channel attention with 1/sqrt(N) tokens scaling instead of 1/sqrt(d), then proj)
  - **`Florence2VisionChannelBlock`** [wiring]: wires `Florence2VisionChannelAttention`, `Florence2VisionMLP`, `Florence2VisionDropPath` (×2); direct `L1/conv2d.py` (×2 depthwise), `L1/layer_norm.py` (×2)
  - **`Florence2VisionWindowAttention`** [compute]: `L1/linear.py + L1/dense_attention.py + L1/linear.py` (no L2 match — DaViT-style local window attention with padding, partition, window-wise multi-head attention, then unmerge; closest is `L2/swinv2_window_attention.py` but Florence2's window-attn lacks Swin's relative-position bias)
  - **`Florence2VisionSpatialBlock`** [wiring]: wires `Florence2VisionWindowAttention`, `Florence2VisionMLP`, `Florence2VisionDropPath` (×2); direct `L1/conv2d.py` (×2 depthwise), `L1/layer_norm.py` (×2)
  - **`Florence2VisionBlock`** [wiring]: wires `Florence2VisionSpatialBlock`, `Florence2VisionChannelBlock`
  - **`Florence2VisionBackbone`** [wiring, inherits `Florence2VisionPreTrainedModel`]: wires `Florence2VisionConvEmbed` (×num_stages), `Florence2VisionBlock` (×depths per stage)
  - **`Florence2MultiModalProjector`** [wiring]: wires `Florence2VisionLearnedAbsolutePositionEmbedding2D`, `Florence2VisionPositionalEmbeddingCosine1D`; direct `L1/linear.py` (image_projection), `L1/layer_norm.py` (image_proj_norm)
  - **`Florence2Model`** [wiring]: wires `Florence2VisionBackbone`, `Florence2MultiModalProjector`, language_model (BART-style seq2seq AutoModel from text_config)
  - **`Florence2ForConditionalGeneration`** [wiring]: wires `Florence2Model`; direct `L1/linear.py` (lm_head)

## fnet
- **src**: modeling_fnet.py
- **hidden_act**: gelu_new
- **status**: partial
- **classes**:
  - **`FNetEmbeddings`** [compute]: `L2/encoder_embeddings.py + L1/linear.py` (BERT-style word + position + token_type + LayerNorm + projection + dropout — adds an extra projection vs vanilla BERT embeddings)
  - **`FNetBasicFourierTransform`** [compute]: no kb-nano kernel — applies `torch.fft.fftn` along (1,2) axes (or DFT matmul on TPU); FFT is not implemented in kb-nano
  - **`FNetBasicOutput`** [compute]: `L1/layer_norm.py` (LayerNorm with residual add)
  - **`FNetFourierTransform`** [wiring]: wires `FNetBasicFourierTransform`, `FNetBasicOutput`
  - **`FNetIntermediate`** [compute]: `L1/linear.py + L1/gelu.py` (BERT Intermediate; gelu_new resolves to gelu)
  - **`FNetOutput`** [compute]: `L1/linear.py + L1/layer_norm.py` (dense + dropout + LayerNorm with residual)
  - **`FNetLayer`** [wiring, inherits `GradientCheckpointingLayer`]: wires `FNetFourierTransform`, `FNetIntermediate`, `FNetOutput`
  - **`FNetEncoder`** [wiring]: wires `FNetLayer`
  - **`FNetPooler`** [compute]: `L1/linear.py + L1/tanh.py` (BERT-style CLS pooler)
  - **`FNetPredictionHeadTransform`** [compute]: `L1/linear.py + L1/gelu.py + L1/layer_norm.py`
  - **`FNetLMPredictionHead`** [wiring]: wires `FNetPredictionHeadTransform`; direct `L1/linear.py` (decoder)
  - **`FNetOnlyMLMHead`** [wiring]: wires `FNetLMPredictionHead`
  - **`FNetOnlyNSPHead`** [compute]: `L1/linear.py` (seq_relationship)
  - **`FNetPreTrainingHeads`** [wiring]: wires `FNetLMPredictionHead`; direct `L1/linear.py` (seq_relationship)
  - **`FNetModel`** [wiring]: wires `FNetEmbeddings`, `FNetEncoder`, `FNetPooler`
  - **`FNetForPreTraining`** [wiring]: wires `FNetModel`, `FNetPreTrainingHeads`
  - **`FNetForMaskedLM`** [wiring]: wires `FNetModel`, `FNetOnlyMLMHead`
  - **`FNetForNextSentencePrediction`** [wiring]: wires `FNetModel`, `FNetOnlyNSPHead`
- **task heads (5)**: ForSequenceClassification, ForMultipleChoice, ForTokenClassification, ForQuestionAnswering — base + linear (per-task) (NSP and PreTraining are kept as primary heads above)

## focalnet
- **src**: modeling_focalnet.py
- **hidden_act**: gelu
- **status**: partial
- **classes**:
  - **`FocalNetEmbeddings`** [wiring]: wires `FocalNetPatchEmbeddings`; direct `L1/layer_norm.py` (norm)
  - **`FocalNetPatchEmbeddings`** [compute]: `L1/conv2d.py + L1/layer_norm.py` (Conv2d patch projection + optional LayerNorm; with maybe_pad)
  - **`FocalNetDropPath`** [compute]: `L1/dropout.py` (stochastic depth — partial, no dedicated drop-path kernel)
  - **`FocalNetModulation`** [compute]: `L1/linear.py + L1/conv2d.py + L1/gelu.py + L1/linear.py` (no exact L2 match — Focal Modulation: projection_in -> split into q/ctx/gates -> per-level depthwise conv2d + gelu loop -> global pool + scale -> projection_context conv -> mul -> projection_out; this is FocalNet's signature op, no kb-nano analog)
  - **`FocalNetMlp`** [compute]: `L1/linear.py + L1/gelu.py + L1/linear.py` (fc1 -> activation -> fc2 with dropouts)
  - **`FocalNetLayer`** [wiring]: wires `FocalNetModulation`, `FocalNetMlp`, `FocalNetDropPath`; direct `L1/layer_norm.py` (×2)
  - **`FocalNetStage`** [wiring, inherits `GradientCheckpointingLayer`]: wires `FocalNetLayer` (×depth), `FocalNetPatchEmbeddings` (optional downsample)
  - **`FocalNetEncoder`** [wiring]: wires `FocalNetStage` (×num_stages)
  - **`FocalNetModel`** [wiring]: wires `FocalNetEmbeddings`, `FocalNetEncoder`; direct `L1/layer_norm.py` (final layernorm), `L1/adaptive_avg_pool2d.py` (pooler)
  - **`FocalNetForMaskedImageModeling`** [wiring]: wires `FocalNetModel`; direct `L1/conv2d.py` (decoder pixel-shuffle reconstruction)
  - **`FocalNetBackbone`** [wiring, inherits `BackboneMixin`, `FocalNetPreTrainedModel`]: wires `FocalNetEmbeddings`, `FocalNetEncoder`; direct `L1/layer_norm.py` (per-stage hidden_states_norms)
- **task heads (1)**: ForImageClassification — base + linear (per-task) (ForMaskedImageModeling is kept as primary above)

## fsmt
- **src**: modeling_fsmt.py
- **hidden_act**: relu
- **status**: composable
- **classes**:
  - **`EncoderLayer`** [wiring]: wires `Attention` (self-attn); direct `L1/layer_norm.py` (×2: self_attn_layer_norm, final_layer_norm), `L1/linear.py` (fc1, fc2), `L1/relu.py` (activation_fn)
  - **`FSMTEncoder`** [wiring]: wires `EncoderLayer`, `SinusoidalPositionalEmbedding`; direct `L1/embedding.py` (embed_tokens)
  - **`DecoderLayer`** [wiring]: wires `Attention` (×2: self_attn + encoder_attn cross-attn); direct `L1/layer_norm.py` (×3), `L1/linear.py` (fc1, fc2), `L1/relu.py`
  - **`FSMTDecoder`** [wiring]: wires `DecoderLayer`, `SinusoidalPositionalEmbedding`; direct `L1/embedding.py` (embed_tokens), `L1/linear.py` (output_projection)
  - **`Attention`** [compute]: `L1/linear.py + L1/dense_attention.py + L1/store_kvcache.py + L1/linear.py` (no exact L2 match — encoder/decoder bidirectional + cross-attention with EncoderDecoderCache, time-major (T,B,C) layout, separate q/k/v + bmm + softmax + bmm; closest is `L2/whisper_attention.py` for enc/dec/cross variants)
  - **`FSMTModel`** [wiring, inherits `PretrainedFSMTModel`]: wires `FSMTEncoder`, `FSMTDecoder`
  - **`FSMTForConditionalGeneration`** [wiring]: wires `FSMTModel`; output projection is in decoder, no separate lm_head linear
  - **`SinusoidalPositionalEmbedding`** [compute, inherits `nn.Embedding`]: `L1/embedding.py` (deterministic sinusoidal weights, lookup via underlying nn.Embedding)

## funnel
- **src**: modeling_funnel.py
- **hidden_act**: gelu_new
- **status**: unsupported
- **classes**:
  - **`FunnelEmbeddings`** [compute]: `L1/embedding.py + L1/layer_norm.py` (word embedding + layer_norm + dropout — no position/token_type embeddings here)
  - **`FunnelAttentionStructure`** [compute]: no kb-nano kernel — Funnel-Transformer's relative-position helper that produces sinusoidal position embeds (factorized or shifted variants), token_type_mat, cls_mask, and pre/post-attention pooling helpers; this is custom math
  - **`FunnelRelMultiheadAttention`** [compute]: no kb-nano kernel — Funnel relative multi-head attention with factorized position bias, content/position/segment-aware scoring (matrix C/D/E type composition like Transformer-XL), pooling on q only or q/k/v; no L2 match
  - **`FunnelPositionwiseFFN`** [compute]: `L1/linear.py + L1/gelu.py + L1/linear.py + L1/layer_norm.py` (linear_1 -> activation -> linear_2 -> add residual -> LayerNorm; post-norm FFN style)
  - **`FunnelLayer`** [wiring]: wires `FunnelRelMultiheadAttention`, `FunnelPositionwiseFFN`
  - **`FunnelEncoder`** [wiring]: wires `FunnelLayer` (per block per repeat), `FunnelAttentionStructure`; pooling logic per block
  - **`FunnelDecoder`** [wiring]: wires `FunnelLayer` (×num_decoder_layers), `FunnelAttentionStructure`; upsampling logic
  - **`FunnelDiscriminatorPredictions`** [compute]: `L1/linear.py + L1/gelu.py + L1/linear.py` (dense + activation + dense_prediction)
  - **`FunnelClassificationHead`** [compute]: `L1/linear.py + L1/tanh.py + L1/linear.py`
  - **`FunnelBaseModel`** [wiring]: wires `FunnelEmbeddings`, `FunnelEncoder`
  - **`FunnelModel`** [wiring]: wires `FunnelEmbeddings`, `FunnelEncoder`, `FunnelDecoder`
  - **`FunnelForPreTraining`** [wiring]: wires `FunnelModel`, `FunnelDiscriminatorPredictions`
  - **`FunnelForMaskedLM`** [wiring]: wires `FunnelModel`; direct `L1/linear.py` (lm_head)
- **task heads (4)**: ForSequenceClassification, ForMultipleChoice, ForTokenClassification, ForQuestionAnswering — base + linear (per-task)

## fuyu
- **src**: modeling_fuyu.py
- **hidden_act**: relu2 (squared_relu) — text_config sub-config; persimmon-based language model
- **status**: composable
- **classes**:
  - **`FuyuModel`** [wiring]: wires language_model (Persimmon-style AutoModel from text_config); direct `L1/linear.py` (vision_embed_tokens — single linear that flattens patches and projects to hidden_size; no separate vision tower)
  - **`FuyuForCausalLM`** [wiring]: wires `FuyuModel`; direct `L1/linear.py` (lm_head)

## gemma
- **src**: modeling_gemma.py (and modular_gemma.py)
- **hidden_act**: gelu_pytorch_tanh
- **status**: composable
- **classes**:
  - **`GemmaTextScaledWordEmbedding`** [compute, inherits `nn.Embedding`]: `L1/embedding.py` (embedding + scalar multiply by sqrt(hidden_size))
  - **`GemmaRMSNorm`** [compute]: `L1/gemma_rms_norm.py` (Gemma uses `(1 + weight)` scaling and fp32 norm — different from Llama RMSNorm)
  - **`GemmaMLP`** [compute]: `L2/llama_mlp.py` (standard SwiGLU pattern with gelu_pytorch_tanh activation; technically GeGLU since activation is gelu, but kb-nano L2/llama_mlp.py supports configurable activation)
  - **`GemmaRotaryEmbedding`** [compute]: `L1/rotary_emb.py`
  - **`GemmaAttention`** [compute]: `L2/attention.py` (q/k/v + RoPE + cache + dispatch + o_proj — standard Llama-family attention)
  - **`GemmaDecoderLayer`** [wiring, inherits `GradientCheckpointingLayer`]: wires `GemmaAttention`, `GemmaMLP`, `GemmaRMSNorm` (×2: input_layernorm, post_attention_layernorm)
  - **`GemmaModel`** [wiring]: wires `GemmaDecoderLayer`, `GemmaRotaryEmbedding`, `GemmaTextScaledWordEmbedding`, `GemmaRMSNorm` (final norm)
  - **`GemmaForCausalLM`** [wiring]: wires `GemmaModel`; direct `L1/linear.py` (lm_head)
- **task heads (2)**: ForSequenceClassification, ForTokenClassification — base + linear (per-task) (gemma)

## gemma2
- **src**: modeling_gemma2.py (and modular_gemma2.py)
- **hidden_act**: gelu_pytorch_tanh (config field is `hidden_activation`)
- **status**: composable
- **classes**:
  - **`Gemma2RMSNorm`** [compute]: `L1/gemma_rms_norm.py` (Gemma-style `(1+w)` scaling, fp32 norm)
  - **`Gemma2MLP`** [compute]: `L2/llama_mlp.py` (SwiGLU/GeGLU pattern with gelu_pytorch_tanh activation)
  - **`Gemma2RotaryEmbedding`** [compute]: `L1/rotary_emb.py`
  - **`Gemma2Attention`** [compute]: `L2/attention.py` (q/k/v + RoPE + cache + dispatch + o_proj; adds `attn_logit_softcapping` and per-layer `sliding_window` for "sliding_attention" layers — these are runtime kwargs to the attention dispatcher)
  - **`Gemma2DecoderLayer`** [wiring, inherits `GradientCheckpointingLayer`]: wires `Gemma2Attention`, `Gemma2MLP`, `Gemma2RMSNorm` (×4: input, post_attention, pre_feedforward, post_feedforward — Gemma2 uses double-norm sandwich pattern)
  - **`Gemma2TextScaledWordEmbedding`** [compute, inherits `nn.Embedding`]: `L1/embedding.py` (× sqrt(hidden_size))
  - **`Gemma2Model`** [wiring]: wires `Gemma2DecoderLayer`, `Gemma2RotaryEmbedding`, `Gemma2TextScaledWordEmbedding`, `Gemma2RMSNorm` (final norm)
  - **`Gemma2ForCausalLM`** [wiring]: wires `Gemma2Model`; direct `L1/linear.py` (lm_head); applies `final_logit_softcapping` post-hoc
- **task heads (2)**: ForSequenceClassification, ForTokenClassification — base + linear (per-task) (gemma2)

## gemma3
- **src**: modeling_gemma3.py (and modular_gemma3.py)
- **hidden_act**: gelu_pytorch_tanh (text_config.hidden_activation default)
- **status**: composable
- **classes**:
  - **`Gemma3TextScaledWordEmbedding`** [compute, inherits `nn.Embedding`]: `L1/embedding.py` (× sqrt(hidden_size))
  - **`Gemma3MLP`** [compute]: `L2/llama_mlp.py` (gate_proj/up_proj/down_proj SwiGLU with gelu_pytorch_tanh)
  - **`Gemma3RMSNorm`** [compute]: `L1/gemma_rms_norm.py` (Gemma `(1+w)` style)
  - **`Gemma3RotaryEmbedding`** [compute]: `L1/rotary_emb.py` (with separate global vs local-sliding rope_local for sliding-window layers)
  - **`Gemma3Attention`** [compute]: `L2/attention.py` (q/k/v + per-head q_norm/k_norm RMSNorm AFTER head-view AND BEFORE rope (per-head qk-norm) + RoPE + cache + dispatch + o_proj; supports sliding window per layer; attn_logit_softcapping)
  - **`Gemma3DecoderLayer`** [wiring, inherits `GradientCheckpointingLayer`]: wires `Gemma3Attention`, `Gemma3MLP`, `Gemma3RMSNorm` (×4: input, post_attention, pre_feedforward, post_feedforward — Gemma2 sandwich pattern)
  - **`Gemma3TextModel`** [wiring]: wires `Gemma3DecoderLayer`, `Gemma3RotaryEmbedding` (×2 for global + rope_local), `Gemma3TextScaledWordEmbedding`, `Gemma3RMSNorm` (final)
  - **`Gemma3ForCausalLM`** [wiring]: wires `Gemma3TextModel`; direct `L1/linear.py` (lm_head)
  - **`Gemma3MultiModalProjector`** [compute]: `L1/avg_pool2d.py + L1/gemma_rms_norm.py + L1/linear.py` (avg-pool patches to reduce token count + soft_emb_norm + matmul with mm_input_projection_weight)
  - **`Gemma3Model`** [wiring]: wires vision_tower (SigLIP-style AutoModel), `Gemma3MultiModalProjector`, language_model (Gemma3TextModel via AutoModel)
  - **`Gemma3ForConditionalGeneration`** [wiring]: wires `Gemma3Model`; direct `L1/linear.py` (lm_head)
- **task heads (2)**: ForSequenceClassification (multimodal), TextForSequenceClassification — base + linear (per-task)

## gemma3n
- **src**: modeling_gemma3n.py (and modular_gemma3n.py)
- **hidden_act**: gelu_pytorch_tanh (text_config.hidden_activation default)
- **status**: partial
- **classes**:
  - **`Gemma3nRMSNorm`** [compute]: `L1/gemma_rms_norm.py` (Gemma `(1+w)` style; supports with_scale=False for unscaled variant)
  - **`Gemma3nAudioRelativePositionEmbedding`** [compute]: no kb-nano kernel — Conformer relative position embedding for audio (factorized rel-pos like Transformer-XL/Conformer)
  - **`Gemma3nAudioAttention`** [compute]: `L1/linear.py + L1/dense_attention.py` (no L2 match — block-chunked local attention with relative-pos embed, per-dim softplus scaling, attention logit softcap, custom local_causal_valid_mask)
  - **`Gemma3nAudioCumulativeGroupNorm`** [compute]: no kb-nano kernel — cumulative GroupNorm over time dim; closest is `L1/group_norm.py` but cumulative variant differs
  - **`Gemma3nAudioSSCPConvBlock`** [compute]: `L1/conv2d.py + L1/relu.py` (Conv2d + cumulative group norm + ReLU — note norm has no kb-nano analog)
  - **`Gemma3nAudioSubSampleConvProjection`** [wiring]: wires `Gemma3nAudioSSCPConvBlock` (×2); direct `L1/linear.py` (input_proj_linear)
  - **`Gemma3nAudioConformerAttention`** [wiring]: wires `Gemma3nAudioAttention`, `Gemma3nRMSNorm` (pre/post); direct `L1/linear.py` (post)
  - **`Gemma3nAudioConformerFeedForward`** [compute]: `L1/gemma_rms_norm.py + L1/linear.py + L1/silu.py + L1/linear.py + L1/gemma_rms_norm.py` (pre_norm -> ffw1 -> silu -> ffw2 -> post_norm + residual scaled by post_layer_scale)
  - **`Gemma3nAudioConformerLightConv1d`** [compute]: `L1/gemma_rms_norm.py + L1/linear.py + L1/conv1d.py + L1/silu.py + L1/linear.py` (norm -> linear_start -> GLU -> depthwise causal Conv1d -> conv_norm -> silu -> linear_end + residual)
  - **`Gemma3nAudioConformerBlock`** [wiring]: wires `Gemma3nAudioConformerFeedForward` (×2: start, end), `Gemma3nAudioConformerAttention`, `Gemma3nAudioConformerLightConv1d`, `Gemma3nRMSNorm`
  - **`Gemma3nTextScaledWordEmbedding`** [compute, inherits `nn.Embedding`]: `L1/embedding.py` (× scale)
  - **`Gemma3nTextLaurelBlock`** [compute]: `L1/linear.py + L1/linear.py + L1/gemma_rms_norm.py` (low-rank residual: linear_left → linear_right → norm → add to input; Learned Augmented Residual Layer)
  - **`Gemma3nTextMLP`** [compute]: `L2/llama_mlp.py` (SwiGLU/GeGLU with optional gaussian topk activation sparsity — closest L2 match; sparsity is a runtime tweak)
  - **`Gemma3nTextAltUp`** [compute]: no kb-nano kernel — Alternating Updates module: predict/correct steps with prediction_coefs, correction_coefs, modality_router, custom AltUp algorithm; closest is `L1/linear.py` decomposition for the projections
  - **`Gemma3nTextAttention`** [compute]: `L2/attention.py` (q/k/v + per-head q_norm/k_norm/v_norm RMSNorm + RoPE + cache + dispatch + o_proj; supports KV-sharing layers, sliding window per layer; v_norm without scale)
  - **`Gemma3nTextDecoderLayer`** [wiring, inherits `GradientCheckpointingLayer`]: wires `Gemma3nTextAttention`, `Gemma3nTextMLP`, `Gemma3nRMSNorm` (×4 + post_per_layer_input_norm), `Gemma3nTextAltUp`, `Gemma3nTextLaurelBlock`; direct `L1/linear.py` (per_layer_input_gate, per_layer_projection)
  - **`Gemma3nAudioEncoder`** [wiring, inherits `Gemma3nPreTrainedModel`]: wires `Gemma3nAudioSubSampleConvProjection`, `Gemma3nAudioConformerBlock` (×conf_num_hidden_layers)
  - **`Gemma3nRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (with rope_local for sliding-window layers)
  - **`Gemma3nTextModel`** [wiring]: wires `Gemma3nTextDecoderLayer`, `Gemma3nRotaryEmbedding` (×2 global+local), `Gemma3nTextScaledWordEmbedding` (×2: embed_tokens + embed_tokens_per_layer), `Gemma3nRMSNorm`
  - **`Gemma3nForCausalLM`** [wiring]: wires `Gemma3nTextModel`; direct `L1/linear.py` (lm_head)
  - **`Gemma3nMultimodalEmbedder`** [compute]: `L1/embedding.py + L1/gemma_rms_norm.py + L1/linear.py + L1/gemma_rms_norm.py` (embedding lookup or soft embed -> norm -> projection -> post-projection norm without scale)
  - **`Gemma3nModel`** [wiring]: wires vision_tower (AutoModel), audio_tower (Gemma3nAudioEncoder), `Gemma3nMultimodalEmbedder` (×2: vision/audio), language_model (Gemma3nTextModel)
  - **`Gemma3nForConditionalGeneration`** [wiring]: wires `Gemma3nModel`; direct `L1/linear.py` (lm_head)

## gemma4
- **src**: modeling_gemma4.py (and modular_gemma4.py)
- **hidden_act**: silu (top-level Gemma4Config), gelu_pytorch_tanh (text/vision sub-configs hidden_activation)
- **status**: kb_nano_l4 (text-only path implemented in `L4/gemma4.py`)
- **classes**:
  - **`Gemma4ClippableLinear`** [compute]: `L1/linear.py` (Linear with optional output clipping; clipping is a runtime tweak)
  - **`Gemma4RMSNorm`** [compute]: `L1/gemma_rms_norm.py` (Gemma `(1+w)` style; with_scale=False variant exists)
  - **`Gemma4AudioRelPositionalEncoding`** [compute]: no kb-nano kernel — Conformer relative-pos encoding for audio
  - **`Gemma4AudioAttention`** [compute]: `L1/linear.py + L1/dense_attention.py` (no L2 match — block-chunked local attention with rel-pos)
  - **`Gemma4AudioSubSampleConvProjectionLayer`** [compute]: `L1/conv2d.py + L1/relu.py` (with cumulative GroupNorm — no kb-nano analog)
  - **`Gemma4AudioSubSampleConvProjection`** [wiring]: wires `Gemma4AudioSubSampleConvProjectionLayer` (×2); direct `L1/linear.py`
  - **`Gemma4AudioFeedForward`** [compute]: `L1/gemma_rms_norm.py + L1/linear.py + L1/silu.py + L1/linear.py + L1/gemma_rms_norm.py` (residual scaled by post_layer_scale)
  - **`Gemma4AudioCausalConv1d`** [compute, inherits `nn.Conv1d`]: `L1/causal_conv1d.py` or `L1/conv1d.py` (causal Conv1d wrapper)
  - **`Gemma4AudioLightConv1d`** [compute]: `L1/gemma_rms_norm.py + L1/linear.py + L1/causal_conv1d.py + L1/silu.py + L1/linear.py` (norm -> linear_start -> GLU -> causal conv1d -> norm -> silu -> linear_end)
  - **`Gemma4AudioLayer`** [wiring]: wires `Gemma4AudioFeedForward` (×2), `Gemma4AudioAttention`, `Gemma4AudioLightConv1d`, `Gemma4RMSNorm`
  - **`Gemma4VisionPatchEmbedder`** [compute]: `L1/linear.py` (input_proj on flattened patch + custom 2D one-hot position embedding via factorized table, no kb-nano analog for the position-embedding sum)
  - **`Gemma4VisionPooler`** [compute]: no kb-nano kernel — 2D average-pool by patch positions via one-hot weights matmul
  - **`Gemma4VisionMLP`** [compute]: `L2/llama_mlp.py` (SwiGLU/GeGLU pattern; uses Gemma4ClippableLinear)
  - **`Gemma4VisionRotaryEmbedding`** [compute]: `L1/rotary_emb.py`
  - **`Gemma4VisionAttention`** [compute]: `L2/attention.py` (q/k/v + RoPE + dispatch + o_proj — vision encoder attention)
  - **`Gemma4VisionEncoderLayer`** [wiring, inherits `GradientCheckpointingLayer`]: wires `Gemma4VisionAttention`, `Gemma4VisionMLP`, `Gemma4RMSNorm` (×4)
  - **`Gemma4VisionEncoder`** [wiring]: wires `Gemma4VisionEncoderLayer`, `Gemma4VisionRotaryEmbedding`
  - **`Gemma4TextMLP`** [compute]: `L2/llama_mlp.py` (SwiGLU/GeGLU; supports double-wide MLP for kv-shared layers)
  - **`Gemma4TextRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (per-layer-type: full_attention with proportional rope, sliding_attention with default rope; matches kb-nano `Gemma4ProportionalRotaryEmbedding` in `L1/rotary_emb.py`)
  - **`Gemma4TextAttention`** [compute]: `L2/gemma4_attention.py` (q/k/v with q_norm/k_norm/v_norm per-head, RoPE, KV-sharing layers, sliding window, attention_k_eq_v alternative attention, store_full_length_kv)
  - **`Gemma4TextExperts`** [compute]: `L1/moe_grouped_gemm.py` or `L2/fused_experts.py` (per-expert gate_up + down with SwiGLU)
  - **`Gemma4TextRouter`** [compute]: `L1/gemma4_routing.py` (RMS-norm without scale + scale * scalar_root_size + linear projection + softmax + topk + per_expert_scale)
  - **`Gemma4TextDecoderLayer`** [wiring, inherits `GradientCheckpointingLayer`]: wires `Gemma4TextAttention`, `Gemma4TextMLP`, `Gemma4RMSNorm` (×4 base + ×3 if MoE: post_feedforward_layernorm_1/_2 and pre_feedforward_layernorm_2 + post_per_layer_input_norm if hidden_size_per_layer_input), `Gemma4TextRouter` and `Gemma4TextExperts` (optional MoE branch); direct `L1/linear.py` (per_layer_input_gate, per_layer_projection); kb-nano L3 file: `L3/gemma4_decoder.py`
  - **`Gemma4TextScaledWordEmbedding`** [compute, inherits `nn.Embedding`]: `L1/embedding.py` (× scale)
  - **`Gemma4TextModel`** [wiring]: wires `Gemma4TextDecoderLayer`, `Gemma4TextRotaryEmbedding`, `Gemma4TextScaledWordEmbedding` (×2 if hidden_size_per_layer_input), `Gemma4RMSNorm`; kb-nano L4 file: `L4/gemma4.py`
  - **`Gemma4ForCausalLM`** [wiring]: wires `Gemma4TextModel`; direct `L1/linear.py` (lm_head)
  - **`Gemma4AudioModel`** [wiring]: wires `Gemma4AudioSubSampleConvProjection`, `Gemma4AudioLayer`, `Gemma4AudioRelPositionalEncoding`
  - **`Gemma4VisionModel`** [wiring]: wires `Gemma4VisionPatchEmbedder`, `Gemma4VisionEncoder`, `Gemma4VisionPooler`
  - **`Gemma4MultimodalEmbedder`** [compute]: `L1/embedding.py + L1/gemma_rms_norm.py + L1/linear.py + L1/gemma_rms_norm.py` (norm -> projection -> post-norm)
  - **`Gemma4Model`** [wiring]: wires `Gemma4VisionModel`, `Gemma4AudioModel`, language_model (Gemma4TextModel via AutoModel), `Gemma4MultimodalEmbedder` (×2: vision/audio)
  - **`Gemma4ForConditionalGeneration`** [wiring]: wires `Gemma4Model`; direct `L1/linear.py` (lm_head)

## gemma4_assistant
- **src**: modeling_gemma4_assistant.py
- **hidden_act**: inherited from text_config (gelu_pytorch_tanh)
- **status**: partial
- **classes**:
  - **`Gemma4AssistantMaskedEmbedder`** [compute]: `L1/linear.py + L1/topk_softmax.py` (centroid-based vocab clustering: linear -> topk -> gather embeddings from lm_head -> dot product -> scatter to canonical positions; multi-token-prediction assisted-decoding head, no exact kb-nano kernel)
  - **`Gemma4AssistantForCausalLM`** [wiring]: wires inner Gemma4-text model (AutoModel from text_config), `Gemma4AssistantMaskedEmbedder`; direct `L1/linear.py` (lm_head, pre_projection, post_projection)

## git
- **src**: modeling_git.py
- **hidden_act**: gelu (text) / quick_gelu (vision)
- **status**: composable
- **classes**:
  - **`GitEmbeddings`** [compute]: `L1/embedding.py + L1/embedding.py + L1/layer_norm.py` (word + position; no token_type — slightly simpler than full BERT-style encoder_embeddings.py)
  - **`GitSelfAttention`** [compute]: `L2/encoder_attention.py` (q/k/v + manual matmul/softmax + KV cache support; vision-style with extra image_patch_tokens tracking)
  - **`GitSelfOutput`** [compute]: `L2/encoder_attention.py` (BERT SelfOutput: dense + LayerNorm + residual)
  - **`GitAttention`** [wiring]: wires `GitSelfAttention`, `GitSelfOutput`
  - **`GitIntermediate`** [compute]: `L1/linear.py + L1/gelu.py` (BERT-style)
  - **`GitOutput`** [compute]: `L1/linear.py + L1/layer_norm.py` (BERT-style)
  - **`GitLayer`** [wiring, inherits `GradientCheckpointingLayer`]: wires `GitAttention`, `GitIntermediate`, `GitOutput`
  - **`GitEncoder`** [wiring]: wires `GitLayer`
  - **`GitVisionEmbeddings`** [compute]: `L1/conv2d.py + L1/embedding.py` (CLIP-style: class_embedding + patch_embedding + position_embedding)
  - **`GitVisionMLP`** [compute]: `L2/clip_mlp.py` (fc1 -> quickgelu -> fc2)
  - **`GitVisionAttention`** [compute]: `L2/clip_attention.py` (q/k/v + softmax + out_proj; non-causal CLIP-style)
  - **`GitVisionEncoderLayer`** [wiring, inherits `GradientCheckpointingLayer`]: wires `GitVisionAttention`, `GitVisionMLP`; direct `L1/layer_norm.py` (×2, pre-norm)
  - **`GitVisionEncoder`** [wiring]: wires `GitVisionEncoderLayer`
  - **`GitVisionTransformer`** [wiring]: wires `GitVisionEmbeddings`, `GitVisionEncoder`; direct `L1/layer_norm.py` (×2: pre_layrnorm, post_layernorm)
  - **`GitVisionModel`** [wiring, inherits `GitPreTrainedModel`]: wires `GitVisionTransformer`
  - **`GitProjection`** [compute]: `L1/linear.py + L1/layer_norm.py` (visual_projection: linear + layernorm)
  - **`GitModel`** [wiring]: wires `GitEmbeddings`, `GitVisionModel`, `GitEncoder`, `GitProjection`; optional temporal embedding parameters
  - **`GitForCausalLM`** [wiring]: wires `GitModel`; direct `L1/linear.py` (output)

## glm
- **src**: modeling_glm.py (and modular_glm.py)
- **hidden_act**: silu
- **status**: composable
- **classes**:
  - **`GlmMLP`** [compute]: `L2/llama_mlp.py` (fused gate_up_proj single linear that's chunked into gate/up, then `up * silu(gate)` -> down_proj — semantically equivalent to SwiGLU but with fused gate+up linear)
  - **`GlmRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (supports `partial_rotary_factor` — partial RoPE on first dim*factor of head dim)
  - **`GlmAttention`** [compute]: `L2/attention.py` (q/k/v + RoPE + cache + dispatch + o_proj; o_proj has no bias regardless)
  - **`GlmRMSNorm`** [compute]: `L1/rms_norm.py` (Llama-style RMSNorm)
  - **`GlmDecoderLayer`** [wiring, inherits `GradientCheckpointingLayer`]: wires `GlmAttention`, `GlmMLP`, `GlmRMSNorm` (×2: input_layernorm, post_attention_layernorm)
  - **`GlmModel`** [wiring]: wires `GlmDecoderLayer`, `GlmRotaryEmbedding`, `GlmRMSNorm` (final); direct `L1/embedding.py`
  - **`GlmForCausalLM`** [wiring]: wires `GlmModel`; direct `L1/linear.py` (lm_head)
- **task heads (2)**: ForSequenceClassification, ForTokenClassification — base + linear (per-task) (glm)

## glm4
- **src**: modeling_glm4.py (and modular_glm4.py)
- **hidden_act**: silu
- **status**: composable
- **classes**:
  - **`Glm4MLP`** [compute]: `L2/llama_mlp.py` (fused gate_up_proj + chunk + `up * silu(gate)` + down_proj — same as GlmMLP)
  - **`Glm4DecoderLayer`** [wiring, inherits `GradientCheckpointingLayer`]: wires `Glm4Attention`, `Glm4MLP`, `Glm4RMSNorm` (×4: input_layernorm, post_self_attn_layernorm, post_attention_layernorm, post_mlp_layernorm — sandwich norms like Gemma2)
  - **`Glm4Attention`** [compute]: `L2/attention.py` (q/k/v + RoPE with interleaved-rotary variant + cache + dispatch + o_proj; Glm4 uses interleaved cos/sin pattern via `rotate_half(x[...,0::2], x[...,1::2])` — different from Llama RoPE)
  - **`Glm4RotaryEmbedding`** [compute]: `L1/rotary_emb.py` (with partial_rotary_factor support)
  - **`Glm4RMSNorm`** [compute]: `L1/rms_norm.py`
  - **`Glm4Model`** [wiring]: wires `Glm4DecoderLayer`, `Glm4RotaryEmbedding`, `Glm4RMSNorm`; direct `L1/embedding.py`
  - **`Glm4ForCausalLM`** [wiring]: wires `Glm4Model`; direct `L1/linear.py` (lm_head)
- **task heads (2)**: ForSequenceClassification, ForTokenClassification — base + linear (per-task) (glm4)

## glm46v
- **src**: modeling_glm46v.py (and modular_glm46v.py)
- **hidden_act**: silu (text_config); follows Glm4MoE-style
- **status**: composable
- **classes**:
  - **`Glm46VModel`** [wiring]: wires visual tower (AutoModel from vision_config — Glm4v-style with M-RoPE), language_model (AutoModel from text_config — Glm4MoE-style); custom `get_rope_index`/`get_vision_position_ids` for 3D M-RoPE indices
  - **`Glm46VForConditionalGeneration`** [wiring]: wires `Glm46VModel`; direct `L1/linear.py` (lm_head)

## glm4_moe
- **src**: modeling_glm4_moe.py (and modular_glm4_moe.py)
- **hidden_act**: silu
- **status**: composable
- **classes**:
  - **`Glm4MoeRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (with partial_rotary_factor)
  - **`Glm4MoeAttention`** [compute]: `L2/attention.py` (q/k/v + optional per-head q_norm/k_norm BEFORE transpose + RoPE + cache + dispatch + o_proj)
  - **`Glm4MoeMLP`** [compute]: `L2/llama_mlp.py` (standard SwiGLU)
  - **`Glm4MoeTopkRouter`** [compute]: `L1/linear.py` (group-bounded MoE router weights; route logic in MoE block)
  - **`Glm4MoeRMSNorm`** [compute]: `L1/rms_norm.py`
  - **`Glm4MoeNaiveMoe`** [compute]: `L1/moe_grouped_gemm.py` or `L2/fused_experts.py` (per-expert gate_up + down with SwiGLU; loop over expert_hit indices)
  - **`Glm4MoeMoE`** [wiring]: wires `Glm4MoeNaiveMoe`, `Glm4MoeTopkRouter`, `Glm4MoeMLP` (shared_experts); closest L2 match: `L2/shared_expert_moe.py` (DeepSeek-style group-routed MoE with shared experts and e_score_correction_bias)
  - **`Glm4MoeDecoderLayer`** [wiring, inherits `GradientCheckpointingLayer`]: wires `Glm4MoeAttention`, `Glm4MoeMoE` or `Glm4MoeMLP` (per `first_k_dense_replace`), `Glm4MoeRMSNorm` (×2: input_layernorm, post_attention_layernorm)
  - **`Glm4MoeModel`** [wiring]: wires `Glm4MoeDecoderLayer`, `Glm4MoeRotaryEmbedding`, `Glm4MoeRMSNorm`; direct `L1/embedding.py`
  - **`Glm4MoeForCausalLM`** [wiring]: wires `Glm4MoeModel`; direct `L1/linear.py` (lm_head)

## glm4_moe_lite
- **src**: modeling_glm4_moe_lite.py (and modular_glm4_moe_lite.py)
- **hidden_act**: silu
- **status**: composable
- **classes**:
  - **`Glm4MoeLiteRotaryEmbedding`** [compute]: `L1/yarn_rotary_emb.py` (with YARN scaling for DeepSeek MLA)
  - **`Glm4MoeLiteAttention`** [compute]: `L2/deepseek_mla_attention.py` (DeepSeek MLA: q LoRA path (q_a_proj/q_a_layernorm/q_b_proj) or direct q_proj, kv_a_proj_with_mqa with qk_rope_head_dim split, kv_a_layernorm, kv_b_proj, NoPE/RoPE split, optional rope_interleave; full DeepSeek MLA structure)
  - **`Glm4MoeLiteMLP`** [compute]: `L2/llama_mlp.py` (standard SwiGLU)
  - **`Glm4MoeLiteTopkRouter`** [compute]: `L1/linear.py` (router, group-routed)
  - **`Glm4MoeLiteRMSNorm`** [compute]: `L1/rms_norm.py`
  - **`Glm4MoeLiteNaiveMoe`** [compute]: `L1/moe_grouped_gemm.py` or `L2/fused_experts.py` (fused experts)
  - **`Glm4MoeLiteMoE`** [wiring]: wires `Glm4MoeLiteNaiveMoe`, `Glm4MoeLiteTopkRouter`, `Glm4MoeLiteMLP` (shared_experts); closest L2: `L2/shared_expert_moe.py` (DeepSeek-style group-routed MoE)
  - **`Glm4MoeLiteDecoderLayer`** [wiring, inherits `GradientCheckpointingLayer`]: wires `Glm4MoeLiteAttention`, `Glm4MoeLiteMoE`/`Glm4MoeLiteMLP`, `Glm4MoeLiteRMSNorm` (×2)
  - **`Glm4MoeLiteModel`** [wiring]: wires `Glm4MoeLiteDecoderLayer`, `Glm4MoeLiteRotaryEmbedding`, `Glm4MoeLiteRMSNorm`; direct `L1/embedding.py`
  - **`Glm4MoeLiteForCausalLM`** [wiring]: wires `Glm4MoeLiteModel`; direct `L1/linear.py` (lm_head)

## glm4v
- **src**: modeling_glm4v.py (and modular_glm4v.py)
- **hidden_act**: silu
- **status**: composable
- **classes**:
  - **`Glm4vRMSNorm`** [compute]: `L1/rms_norm.py`
  - **`Glm4VisionMlp`** [compute]: `L2/llama_mlp.py` (gate_proj/up_proj/down_proj SwiGLU)
  - **`Glm4vVisionPatchEmbed`** [compute]: `L1/conv3d.py` (Conv3d temporal+spatial patch projection)
  - **`Glm4vVisionRotaryEmbedding`** [compute]: `L1/vision_rotary_emb.py` (or simple rotary_emb on 1D seq)
  - **`Glm4vVisionPatchMerger`** [compute]: `L1/linear.py + L1/layer_norm.py + L1/gelu.py + L2/llama_mlp.py` (proj -> norm -> gelu -> SwiGLU)
  - **`Glm4vVisionEmbeddings`** [compute]: `L1/embedding.py + L1/grid_sample.py` (learned 2D position embeddings + bicubic interpolation via grid_sample for variable image shapes)
  - **`Glm4vVisionAttention`** [compute]: `L1/linear.py + L1/flash_attn_varlen.py` or `L1/dense_attention.py` (no exact L2 match — fused QKV + variable-length attention with cu_seqlens for packed images, vision-style with vision rotary, non-causal)
  - **`Glm4vVisionBlock`** [wiring, inherits `GradientCheckpointingLayer`]: wires `Glm4vVisionAttention`, `Glm4VisionMlp`, `Glm4vRMSNorm` (×2)
  - **`Glm4vTextRotaryEmbedding`** [compute]: `L1/mrope.py` (M-RoPE: 3D rotary with mrope_section grouping for temporal/height/width)
  - **`Glm4vTextAttention`** [compute]: `L2/attention.py` (q/k/v + RoPE + cache + dispatch + o_proj; q/k/v use bias=True, o_proj bias=False)
  - **`Glm4vTextMLP`** [compute]: `L2/llama_mlp.py` (fused gate_up_proj + chunk + `up * silu(gate)` + down_proj)
  - **`Glm4vTextDecoderLayer`** [wiring, inherits `GradientCheckpointingLayer`]: wires `Glm4vTextAttention`, `Glm4vTextMLP`, `Glm4vRMSNorm` (×4: input, post_self_attn, post_attention, post_mlp — sandwich)
  - **`Glm4vVisionModel`** [wiring, inherits `Glm4vPreTrainedModel`]: wires `Glm4vVisionPatchEmbed`, `Glm4vVisionEmbeddings`, `Glm4vVisionRotaryEmbedding`, `Glm4vVisionBlock`, `Glm4vVisionPatchMerger`; direct `L1/rms_norm.py`
  - **`Glm4vTextModel`** [wiring]: wires `Glm4vTextDecoderLayer`, `Glm4vTextRotaryEmbedding`, `Glm4vRMSNorm`; direct `L1/embedding.py`
  - **`Glm4vModel`** [wiring]: wires `Glm4vVisionModel`, `Glm4vTextModel`
  - **`Glm4vForConditionalGeneration`** [wiring]: wires `Glm4vModel`; direct `L1/linear.py` (lm_head)

## glm4v_moe
- **src**: modeling_glm4v_moe.py (and modular_glm4v_moe.py)
- **hidden_act**: silu (text and vision)
- **status**: composable
- **classes**:
  - **`Glm4vMoeTextAttention`** [compute]: `L2/attention.py` (q/k/v + M-RoPE + cache + dispatch + o_proj — same as Glm4vTextAttention)
  - **`Glm4vMoeTextTopkRouter`** [compute]: `L1/linear.py` (group-routed router with e_score_correction_bias)
  - **`Glm4vMoeTextNaiveMoe`** [compute]: `L1/moe_grouped_gemm.py` or `L2/fused_experts.py` (per-expert SwiGLU)
  - **`Glm4vMoeTextMoE`** [wiring]: wires `Glm4vMoeTextNaiveMoe`, `Glm4vMoeTextTopkRouter`, `Glm4vMoeTextMLP` (shared_experts); closest L2: `L2/shared_expert_moe.py`
  - **`Glm4vMoeTextMLP`** [compute]: `L2/llama_mlp.py` (standard SwiGLU)
  - **`Glm4vMoeTextRMSNorm`** [compute]: `L1/rms_norm.py`
  - **`Glm4vMoeTextDecoderLayer`** [wiring, inherits `GradientCheckpointingLayer`]: wires `Glm4vMoeTextAttention`, `Glm4vMoeTextMoE`/`Glm4vMoeTextMLP` (per first_k_dense_replace), `Glm4vMoeTextRMSNorm` (×2)
  - **`Glm4vMoeVisionRotaryEmbedding`** [compute]: `L1/vision_rotary_emb.py`
  - **`Glm4vMoeRMSNorm`** [compute]: `L1/rms_norm.py`
  - **`Glm4vMoeisionMlp`** [compute]: `L2/llama_mlp.py` (vision SwiGLU)
  - **`Glm4vMoeVisionPatchEmbed`** [compute]: `L1/conv3d.py`
  - **`Glm4vMoeVisionPatchMerger`** [compute]: `L1/linear.py + L1/layer_norm.py + L1/gelu.py + L2/llama_mlp.py`
  - **`Glm4vMoeVisionEmbeddings`** [compute]: `L1/embedding.py + L1/grid_sample.py` (2D pos embed + bicubic via grid_sample)
  - **`Glm4vMoeVisionAttention`** [compute]: `L1/linear.py + L1/flash_attn_varlen.py` (fused QKV + varlen, vision rotary, non-causal)
  - **`Glm4vMoeVisionBlock`** [wiring, inherits `GradientCheckpointingLayer`]: wires `Glm4vMoeVisionAttention`, `Glm4vMoeisionMlp`, `Glm4vMoeRMSNorm` (×2)
  - **`Glm4vMoeVisionModel`** [wiring, inherits `Glm4vMoePreTrainedModel`]: wires `Glm4vMoeVisionPatchEmbed`, `Glm4vMoeVisionEmbeddings`, `Glm4vMoeVisionRotaryEmbedding`, `Glm4vMoeVisionBlock`, `Glm4vMoeVisionPatchMerger`; direct `L1/rms_norm.py`
  - **`Glm4vMoeTextRotaryEmbedding`** [compute]: `L1/mrope.py` (M-RoPE 3D)
  - **`Glm4vMoeTextModel`** [wiring]: wires `Glm4vMoeTextDecoderLayer`, `Glm4vMoeTextRotaryEmbedding`, `Glm4vMoeTextRMSNorm`; direct `L1/embedding.py`
  - **`Glm4vMoeModel`** [wiring]: wires `Glm4vMoeVisionModel`, `Glm4vMoeTextModel`
  - **`Glm4vMoeForConditionalGeneration`** [wiring]: wires `Glm4vMoeModel`; direct `L1/linear.py` (lm_head)



