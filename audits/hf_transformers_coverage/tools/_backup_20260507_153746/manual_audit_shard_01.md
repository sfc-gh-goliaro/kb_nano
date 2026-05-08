## afmoe
- **src**: modeling_afmoe.py, modular_afmoe.py
- **hidden_act**: silu
- **status**: composable
- **classes**:
  - **`AfmoeRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (standard Llama-style RoPE; supports default and ROPE_INIT_FUNCTIONS scalings)
  - **`AfmoeRMSNorm`** [compute]: `L1/rms_norm.py`
  - **`AfmoeMLP`** [compute]: `L2/llama_mlp.py` (SwiGLU: gate_proj * silu * up_proj -> down_proj)
  - **`AfmoeTokenChoiceRouter`** [compute]: `L1/linear.py + L1/sigmoid.py + L1/sigmoid_topk.py` (sigmoid scoring + topk + bias correction; no exact L2 match — pattern lives inside `L2/shared_expert_moe.py`)
  - **`AfmoeExperts`** [compute]: `L1/moe_grouped_gemm.py` (chunked SwiGLU experts; matches the routed-expert path used by `L2/shared_expert_moe.py`)
  - **`AfmoeSparseMoeBlock`** [compute]: `L2/shared_expert_moe.py` (router + shared SwiGLU expert + routed experts, sigmoid routing — matches Kimi-Linear flavor)
  - **`AfmoeAttention`** [compute]: `L2/attention.py` (Llama-style q/k/v + RoPE + KV cache; adds q_norm/k_norm + sigmoid output gate — gating is not a stock kb-nano op, so partial: `L2/attention.py + L1/rms_norm.py (q/k norms) + L1/linear.py (gate_proj) + L1/sigmoid.py`)
  - **`AfmoeDecoderLayer`** [wiring]: wires `AfmoeAttention`, `AfmoeMLP` or `AfmoeSparseMoeBlock`, `AfmoeRMSNorm` (x4: input/post_attention/pre_mlp/post_mlp)
  - **`AfmoeModel`** [wiring]: wires `AfmoeDecoderLayer`, `AfmoeRMSNorm` (final), `AfmoeRotaryEmbedding`; direct `L1/embedding.py`
  - **`AfmoeForCausalLM`** [wiring]: wires `AfmoeModel`; direct `L1/linear.py` (lm_head)

## aimv2
- **src**: modeling_aimv2.py, modular_aimv2.py
- **hidden_act**: silu (vision and text configs both)
- **status**: composable
- **classes**:
  - **`Aimv2RMSNorm`** [compute]: `L1/rms_norm.py`
  - **`Aimv2MLP`** [compute]: `L2/llama_mlp.py` (SwiGLU: gate_proj * silu * up_proj -> down_proj)
  - **`Aimv2VisionEmbeddings`** [compute]: `L1/conv2d.py + L1/rms_norm.py + L1/embedding.py` (patch_embed Conv2d + rms_norm + learned position embedding; sincos branch is buffer-only)
  - **`Aimv2TextEmbeddings`** [compute]: `L1/embedding.py + L1/embedding.py` (token_embedding + position_embedding, sum)
  - **`Aimv2Attention`** [compute]: `L2/siglip_attention.py` (non-causal q/k/v + dispatch via ALL_ATTENTION_FUNCTIONS, no RoPE, no KV cache)
  - **`Aimv2EncoderLayer`** [wiring]: wires `Aimv2Attention`, `Aimv2MLP`, `Aimv2RMSNorm` (x2: rms_norm1/rms_norm2)
  - **`Aimv2Encoder`** [wiring]: wires `Aimv2EncoderLayer`
  - **`Aimv2AttentionPoolingHead`** [compute]: `L1/linear.py + L1/dense_attention.py` (k_proj/v_proj + cls_token query, F.scaled_dot_product_attention, then mean + output_proj)
  - **`Aimv2VisionModel`** [wiring]: wires `Aimv2VisionEmbeddings`, `Aimv2Encoder`, `Aimv2RMSNorm`, optional `Aimv2AttentionPoolingHead`
  - **`Aimv2TextModel`** [wiring]: wires `Aimv2TextEmbeddings`, `Aimv2Encoder`, `Aimv2RMSNorm` (eos pooling is index op)
  - **`Aimv2Model`** [wiring]: wires `Aimv2VisionModel`, `Aimv2TextModel`; direct `L1/linear.py` (visual_projection, text_projection)

## albert
- **src**: modeling_albert.py
- **hidden_act**: gelu_new
- **status**: composable
- **classes**:
  - **`AlbertEmbeddings`** [compute]: `L2/encoder_embeddings.py` (word + position + token_type + LayerNorm + Dropout)
  - **`AlbertAttention`** [compute]: `L2/encoder_attention.py` (q/k/v + dispatch via ALL_ATTENTION_FUNCTIONS, then dense + dropout + LayerNorm + residual — full BERT-style attn block fused into one class)
  - **`AlbertLayer`** [wiring]: wires `AlbertAttention`; direct `L1/linear.py` (ffn, ffn_output) + `L1/gelu.py` (gelu_new) + `L1/layer_norm.py` (full_layer_layer_norm) — Albert FFN is a 2-layer linear+act with post-LN; close to encoder_mlp but inlined
  - **`AlbertLayerGroup`** [wiring]: wires `AlbertLayer`
  - **`AlbertTransformer`** [wiring]: wires `AlbertLayerGroup`; direct `L1/linear.py` (embedding_hidden_mapping_in)
  - **`AlbertModel`** [wiring]: wires `AlbertEmbeddings`, `AlbertTransformer`; direct `L1/linear.py + L1/tanh.py` (pooler)
  - **`AlbertMLMHead`** [compute]: `L1/linear.py + L1/gelu.py + L1/layer_norm.py + L1/linear.py` (dense + act + LayerNorm + decoder)
  - **`AlbertForPreTraining`** [wiring]: wires `AlbertModel`, `AlbertMLMHead`, `AlbertSOPHead`
  - **`AlbertForMaskedLM`** [wiring]: wires `AlbertModel`, `AlbertMLMHead`
- **task heads (4)**: ForMultipleChoice, ForQuestionAnswering, ForSequenceClassification, ForTokenClassification — base + linear (per-task)

## align
- **src**: modeling_align.py
- **hidden_act**: text=gelu, vision=swish (silu)
- **status**: composable
- **classes**:
  - **`AlignVisionEmbeddings`** [compute]: `L1/conv2d.py + L1/batch_norm2d.py + L1/silu.py` (ZeroPad2d + 3x3 stem conv + BN + swish)
  - **`AlignVisionDepthwiseConv2d`** [compute]: `L1/conv2d.py` (depthwise — Conv2d with groups=in_channels)
  - **`AlignVisionExpansionLayer`** [compute]: `L1/conv2d.py + L1/batch_norm2d.py + L1/silu.py` (1x1 conv + BN + swish)
  - **`AlignVisionDepthwiseLayer`** [compute]: `L1/conv2d.py + L1/batch_norm2d.py + L1/silu.py` (depthwise conv + BN + swish)
  - **`AlignVisionSqueezeExciteLayer`** [compute]: `L1/adaptive_avg_pool2d.py + L1/conv2d.py + L1/silu.py + L1/conv2d.py + L1/sigmoid.py` (SE block: pool -> reduce conv -> swish -> expand conv -> sigmoid -> mul) — close to `L2/efficientnetv2_squeeze_excite.py`
  - **`AlignVisionFinalBlockLayer`** [compute]: `L1/conv2d.py + L1/batch_norm2d.py + L1/dropout.py` (1x1 project conv + BN + optional residual)
  - **`AlignVisionBlock`** [wiring]: wires `AlignVisionExpansionLayer` (optional), `AlignVisionDepthwiseLayer`, `AlignVisionSqueezeExciteLayer`, `AlignVisionFinalBlockLayer` — close to `L2/efficientnetv2_inverted_residual.py` pattern but distinct
  - **`AlignVisionEncoder`** [wiring]: wires `AlignVisionBlock`
  - **`AlignTextEmbeddings`** [compute]: `L2/encoder_embeddings.py` (word + position + token_type + LayerNorm + Dropout)
  - **`AlignTextSelfAttention`** [compute]: `L2/encoder_attention.py` (BERT-style q/k/v + dispatch via ALL_ATTENTION_FUNCTIONS)
  - **`AlignTextSelfOutput`** [compute]: `L2/encoder_attention.py` (dense + dropout + LayerNorm + residual)
  - **`AlignTextAttention`** [compute]: `L2/encoder_attention.py` (wrapper: self.self + self.output)
  - **`AlignTextIntermediate`** [compute]: `L1/linear.py + L1/gelu.py` (no exact L2 match — encoder_mlp.py covers Intermediate+Output)
  - **`AlignTextOutput`** [compute]: `L1/linear.py + L1/dropout.py + L1/layer_norm.py` (dense + dropout + LayerNorm + residual)
  - **`AlignTextLayer`** [wiring]: wires `AlignTextAttention`, `AlignTextIntermediate`, `AlignTextOutput`
  - **`AlignTextEncoder`** [wiring]: wires `AlignTextLayer`
  - **`AlignTextPooler`** [compute]: `L1/linear.py + L1/tanh.py`
  - **`AlignTextModel`** [wiring]: wires `AlignTextEmbeddings`, `AlignTextEncoder`, `AlignTextPooler`
  - **`AlignVisionModel`** [wiring]: wires `AlignVisionEmbeddings`, `AlignVisionEncoder`; direct `L1/avg_pool2d.py` or `L1/max_pool2d.py` (pooler)
  - **`AlignModel`** [wiring]: wires `AlignTextModel`, `AlignVisionModel`; direct `L1/linear.py` (text_projection)

## altclip
- **src**: modeling_altclip.py, modular_altclip.py
- **hidden_act**: text=gelu (Roberta-style), vision=quick_gelu
- **status**: composable
- **classes**:
  - **`AltRobertaEmbeddings`** [compute]: `L2/encoder_embeddings.py` (word + position + token_type + LayerNorm + Dropout — Roberta-style with padding_idx-based position id creation)
  - **`AltRobertaSelfAttention`** [compute]: `L2/encoder_attention.py` (BERT-style q/k/v + dispatch via ALL_ATTENTION_FUNCTIONS)
  - **`AltRobertaSelfOutput`** [compute]: `L2/encoder_attention.py` (dense + dropout + LayerNorm + residual)
  - **`AltRobertaAttention`** [compute]: `L2/encoder_attention.py` (wrapper: self.self + self.output)
  - **`AltRobertaIntermediate`** [compute]: `L1/linear.py + L1/gelu.py`
  - **`AltRobertaOutput`** [compute]: `L1/linear.py + L1/dropout.py + L1/layer_norm.py` (dense + dropout + LayerNorm + residual)
  - **`AltRobertaLayer`** [wiring]: wires `AltRobertaAttention`, `AltRobertaIntermediate`, `AltRobertaOutput`
  - **`AltRobertaEncoder`** [wiring]: wires `AltRobertaLayer`
  - **`AltRobertaPooler`** [compute]: `L1/linear.py + L1/tanh.py`
  - **`AltCLIPAttention`** [compute]: `L2/clip_attention.py` (CLIP-style q/k/v + out_proj, non-causal here)
  - **`AltCLIPMLP`** [compute]: `L2/clip_mlp.py` (fc1 + quickgelu + fc2)
  - **`AltCLIPEncoderLayer`** [wiring]: wires `AltCLIPAttention`, `AltCLIPMLP`; direct `L1/layer_norm.py` (x2: layer_norm1/layer_norm2)
  - **`AltCLIPEncoder`** [wiring]: wires `AltCLIPEncoderLayer`
  - **`AltCLIPVisionEmbeddings`** [compute]: `L1/conv2d.py + L1/embedding.py` (CLS token + patch_embedding Conv2d + position_embedding; supports interpolate)
  - **`AltCLIPVisionModel`** [wiring]: wires `AltCLIPVisionEmbeddings`, `AltCLIPEncoder`; direct `L1/layer_norm.py` (x2: pre_layrnorm, post_layernorm)
  - **`AltRobertaModel`** [wiring]: wires `AltRobertaEmbeddings`, `AltRobertaEncoder`, `AltRobertaPooler`
  - **`AltCLIPTextModel`** [wiring]: wires `AltRobertaModel`; direct `L1/linear.py` (transformation), `L1/layer_norm.py` (pre_LN)
  - **`AltCLIPModel`** [wiring]: wires `AltCLIPTextModel`, `AltCLIPVisionModel`; direct `L1/linear.py` (visual_projection, text_projection)

## apertus
- **src**: modeling_apertus.py, modular_apertus.py
- **hidden_act**: xielu (custom activation, not in ACT2FN)
- **status**: partial (xielu activation has no kb-nano kernel; non-SwiGLU MLP shape)
- **classes**:
  - **`ApertusMLP`** [compute]: `L1/linear.py + L1/linear.py` (no exact L2 match — 2-layer up_proj + xielu + down_proj; xielu has no kb-nano kernel)
  - **`ApertusRMSNorm`** [compute]: `L1/rms_norm.py`
  - **`ApertusRotaryEmbedding`** [compute]: `L1/rotary_emb.py`
  - **`ApertusAttention`** [compute]: `L2/attention.py + L1/rms_norm.py (q_norm, k_norm)` (Llama-style q/k/v + RoPE + KV cache, with extra QK norms)
  - **`ApertusDecoderLayer`** [wiring]: wires `ApertusAttention`, `ApertusMLP`, `ApertusRMSNorm` (x2: attention_layernorm/feedforward_layernorm)
  - **`ApertusModel`** [wiring]: wires `ApertusDecoderLayer`, `ApertusRMSNorm` (final), `ApertusRotaryEmbedding`; direct `L1/embedding.py`
  - **`ApertusForCausalLM`** [wiring]: wires `ApertusModel`; direct `L1/linear.py` (lm_head)
- **task heads (1)**: ForTokenClassification — base + linear (per-task)

## arcee
- **src**: modeling_arcee.py, modular_arcee.py
- **hidden_act**: relu2 (squared relu)
- **status**: composable
- **classes**:
  - **`ArceeMLP`** [compute, inherits `NemotronMLP`]: `L1/linear.py + L1/squared_relu.py + L1/linear.py` (no exact L2 match — 2-layer up_proj + relu^2 + down_proj, no gate)
  - **`ArceeRMSNorm`** [compute]: `L1/rms_norm.py`
  - **`ArceeRotaryEmbedding`** [compute]: `L1/rotary_emb.py`
  - **`ArceeAttention`** [compute]: `L2/attention.py` (Llama-style q/k/v + RoPE + KV cache)
  - **`ArceeDecoderLayer`** [wiring]: wires `ArceeAttention`, `ArceeMLP`, `ArceeRMSNorm` (x2: input_layernorm/post_attention_layernorm)
  - **`ArceeModel`** [wiring]: wires `ArceeDecoderLayer`, `ArceeRMSNorm` (final), `ArceeRotaryEmbedding`; direct `L1/embedding.py`
  - **`ArceeForCausalLM`** [wiring, inherits `LlamaForCausalLM`]: wires `ArceeModel`; direct `L1/linear.py` (lm_head)
- **task heads (3)**: ForQuestionAnswering, ForSequenceClassification, ForTokenClassification — base + linear (per-task)

## aria
- **src**: modeling_aria.py, modular_aria.py
- **hidden_act**: silu (text); projector uses gelu_new
- **status**: composable
- **classes**:
  - **`AriaTextRMSNorm`** [compute]: `L1/rms_norm.py`
  - **`AriaProjectorMLP`** [compute]: `L1/linear.py + L1/gelu.py + L1/linear.py` (linear_in + gelu_new + linear_out)
  - **`AriaCrossAttention`** [compute]: `L1/linear.py + L1/layer_norm.py (x2) + L1/dense_attention.py + L1/linear.py + L1/dropout.py` (uses nn.MultiheadAttention internally; q/k/v projections + LayerNorms + cross-attn + linear + dropout) — no exact L2 match (nn.MultiheadAttention is the compute primitive)
  - **`AriaProjector`** [wiring]: wires `AriaCrossAttention`, `AriaProjectorMLP`; direct `L1/layer_norm.py` (layer_norm)
  - **`AriaSharedExpertsMLP`** [compute]: `L2/llama_mlp.py` (SwiGLU with intermediate_size scaled by num_shared_experts)
  - **`AriaGroupedExpertsGemm`** [compute]: `L1/moe_grouped_gemm.py` (sequential GEMM fallback over expert weights)
  - **`AriaExperts`** [compute]: `L1/moe_grouped_gemm.py + L1/silu_and_mul.py` (top-k softmax routing + permute + grouped GEMM with SwiGLU + unpermute)
  - **`AriaTextMoELayer`** [wiring]: wires `AriaExperts`, `AriaSharedExpertsMLP`; direct `L1/linear.py` (router) — overall pattern is similar to `L2/shared_expert_moe.py` (softmax routing) but custom sequential-GEMM expert path
  - **`AriaTextAttention`** [compute]: `L2/attention.py` (Llama-style q/k/v + RoPE + KV cache)
  - **`AriaTextDecoderLayer`** [wiring]: wires `AriaTextAttention`, `AriaTextMoELayer`, `AriaTextRMSNorm` (x2: input_layernorm/post_attention_layernorm)
  - **`AriaTextRotaryEmbedding`** [compute]: `L1/rotary_emb.py`
  - **`AriaTextModel`** [wiring]: wires `AriaTextDecoderLayer`, `AriaTextRMSNorm` (final), `AriaTextRotaryEmbedding`; direct `L1/embedding.py`
  - **`AriaTextForCausalLM`** [wiring]: wires `AriaTextModel`; direct `L1/linear.py` (lm_head)
  - **`AriaModel`** [wiring]: wires vision_tower (AutoModel), `AriaProjector`, language_model (AutoModel = AriaTextModel)
  - **`AriaForConditionalGeneration`** [wiring]: wires `AriaModel`; direct `L1/linear.py` (lm_head)

## audio_spectrogram_transformer
- **src**: modeling_audio_spectrogram_transformer.py
- **hidden_act**: gelu
- **status**: composable
- **classes**:
  - **`ASTPatchEmbeddings`** [compute]: `L1/conv2d.py` (Conv2d with rectangular kernel/stride for spectrogram patches)
  - **`ASTEmbeddings`** [compute]: `L1/conv2d.py + L1/dropout.py` (cls + distillation tokens + patch_embeddings + position embedding parameter + dropout); wires `ASTPatchEmbeddings`
  - **`ASTSelfAttention`** [compute]: `L2/encoder_attention.py` (q/k/v + dispatch via ALL_ATTENTION_FUNCTIONS, ViT-style)
  - **`ASTSelfOutput`** [compute]: `L1/linear.py + L1/dropout.py` (dense + dropout, no LayerNorm — ViT pre-norm style)
  - **`ASTAttention`** [compute]: `L2/encoder_attention.py` (wrapper: self.attention + self.output) — note: ASTSelfOutput omits LayerNorm and residual (handled in ASTLayer)
  - **`ASTIntermediate`** [compute]: `L1/linear.py + L1/gelu.py`
  - **`ASTOutput`** [compute]: `L1/linear.py + L1/dropout.py` (dense + dropout + residual; no LayerNorm)
  - **`ASTLayer`** [wiring]: wires `ASTAttention`, `ASTIntermediate`, `ASTOutput`; direct `L1/layer_norm.py` (x2: layernorm_before/layernorm_after) — ViT pre-norm pattern
  - **`ASTEncoder`** [wiring]: wires `ASTLayer`
  - **`ASTModel`** [wiring]: wires `ASTEmbeddings`, `ASTEncoder`; direct `L1/layer_norm.py` (final layernorm); pooled output is mean of cls/distillation tokens
  - **`ASTMLPHead`** [compute]: `L1/layer_norm.py + L1/linear.py`
- **task heads (1)**: ForAudioClassification — base + linear (per-task)

## audioflamingo3
- **src**: modeling_audioflamingo3.py, modular_audioflamingo3.py
- **hidden_act**: encoder activation_function=gelu (Whisper-style); projector_hidden_act=gelu
- **status**: composable
- **classes**:
  - **`AudioFlamingo3Attention`** [compute]: `L2/whisper_attention.py` (Whisper-style with optional cross-attention via key_value_states + EncoderDecoderCache; here used non-causal as encoder)
  - **`AudioFlamingo3EncoderLayer`** [wiring]: wires `AudioFlamingo3Attention`; direct `L1/layer_norm.py` (x2: self_attn_layer_norm, final_layer_norm), `L1/linear.py` (fc1, fc2), `L1/gelu.py`, `L1/dropout.py` — Whisper encoder layer pattern
  - **`AudioFlamingo3Encoder`** [wiring]: wires `AudioFlamingo3EncoderLayer`; direct `L1/conv1d.py` (x2: conv1, conv2 with stride 2), `L1/embedding.py` (embed_positions), `L1/gelu.py`, `L1/layer_norm.py` (final), `L1/avg_pool1d.py` (time/2 downsample) — extends Whisper encoder with avg-pool
  - **`AudioFlamingo3MultiModalProjector`** [compute]: `L1/linear.py + L1/gelu.py + L1/linear.py`
  - **`AudioFlamingo3ForConditionalGeneration`** [wiring]: wires audio_tower (AutoModel = `AudioFlamingo3Encoder`), `AudioFlamingo3MultiModalProjector`, language_model (AutoModelForCausalLM = Qwen2)

## autoformer
- **src**: modeling_autoformer.py
- **hidden_act**: activation_function=gelu
- **status**: unsupported (FFT-based AutoCorrelation attention has no kb-nano kernel; specialized time-series ops)
- **classes**:
  - **`AutoformerFeatureEmbedder`** [compute]: `L1/embedding.py` (multiple embedding tables for categorical features)
  - **`AutoformerStdScaler`** [compute]: pure tensor ops (mean/var scaling) — no specific kb-nano kernel
  - **`AutoformerMeanScaler`** [compute]: pure tensor ops — no specific kb-nano kernel
  - **`AutoformerNOPScaler`** [compute]: identity scaler — no kb-nano kernel needed
  - **`AutoformerSinusoidalPositionalEmbedding`** [compute, inherits `nn.Embedding`]: `L1/embedding.py + L1/sinusoidal_embed.py`
  - **`AutoformerValueEmbedding`** [compute]: `L1/linear.py`
  - **`AutoformerSeriesDecompositionLayer`** [compute]: `L1/avg_pool1d.py` (moving-average trend extractor; pads via repeat then AvgPool1d)
  - **`AutoformerLayernorm`** [compute]: `L1/layer_norm.py` (LayerNorm with seasonal mean subtraction — no exact kb-nano variant)
  - **`AutoformerAttention`** [compute]: no kb-nano kernel — FFT-based autocorrelation (rfft/irfft) + top-k delay aggregation; replaces canonical attention
  - **`AutoformerEncoderLayer`** [wiring]: wires `AutoformerAttention`, `AutoformerSeriesDecompositionLayer` (x2), `AutoformerLayernorm`; direct `L1/linear.py` (fc1/fc2), `L1/gelu.py`, `L1/layer_norm.py` (self_attn_layer_norm), `L1/dropout.py`
  - **`AutoformerDecoderLayer`** [wiring]: wires `AutoformerAttention` (x2: self + cross), `AutoformerSeriesDecompositionLayer` (x3), `AutoformerLayernorm`; direct `L1/linear.py` (fc1/fc2), `L1/conv1d.py` (trend_projection), `L1/layer_norm.py` (self_attn/encoder_attn LNs), `L1/gelu.py`, `L1/dropout.py`
  - **`AutoformerEncoder`** [wiring]: wires `AutoformerEncoderLayer`, `AutoformerSinusoidalPositionalEmbedding`, `AutoformerValueEmbedding`; direct `L1/layer_norm.py`
  - **`AutoformerDecoder`** [wiring]: wires `AutoformerDecoderLayer`, `AutoformerSinusoidalPositionalEmbedding`, `AutoformerValueEmbedding`; direct `L1/layer_norm.py`
  - **`AutoformerModel`** [wiring]: wires `AutoformerEncoder`, `AutoformerDecoder`, scaler (Std/Mean/NOP), `AutoformerFeatureEmbedder`, `AutoformerSeriesDecompositionLayer`
  - **`AutoformerForPrediction`** [wiring]: wires `AutoformerModel`; distribution head (no LM head)

## aya_vision
- **src**: modeling_aya_vision.py, modular_aya_vision.py
- **hidden_act**: silu (projector hardcoded to silu for SwiGLU)
- **status**: composable
- **classes**:
  - **`AyaVisionMultiModalProjector`** [compute]: `L1/layer_norm.py + L1/linear.py + L1/silu_and_mul.py + L1/linear.py` (pixel_shuffle reshape + LayerNorm + linear_1 + chunk-then-SwiGLU + linear_2)
  - **`AyaVisionModel`** [wiring]: wires vision_tower (AutoModel = SigLIP), `AyaVisionMultiModalProjector`, language_model (AutoModel = Cohere2)
  - **`AyaVisionForConditionalGeneration`** [wiring]: wires `AyaVisionModel`; direct `L1/linear.py` (lm_head)

## bamba
- **src**: modeling_bamba.py, modular_bamba.py
- **hidden_act**: silu
- **status**: composable
- **classes**:
  - **`BambaRotaryEmbedding`** [compute]: `L1/rotary_emb.py`
  - **`BambaAttention`** [compute]: `L2/attention.py` (Llama-style q/k/v + RoPE + KV cache)
  - **`BambaRMSNormGated`** [compute]: `L1/rms_norm_gated.py` (RMSNorm with optional silu(gate) multiplied in)
  - **`BambaMixer`** [compute]: `L2/mamba2_mixer.py` (Mamba2-style SSM mixer with conv1d, in_proj, dt_bias, A_log/D, chunked scan; uses `L1/causal_conv1d.py` and Mamba2 ops)
  - **`BambaMLP`** [compute]: `L2/llama_mlp.py` (SwiGLU)
  - **`BambaRMSNorm`** [compute]: `L1/rms_norm.py`
  - **`BambaDecoderLayer`** [wiring]: wires `BambaAttention` (when layer_type='attention') or `BambaMixer` (when layer_type='mamba'), `BambaMLP`, `BambaRMSNorm` (x2: input_layernorm, pre_ff_layernorm)
  - **`BambaModel`** [wiring]: wires `BambaDecoderLayer` (mixed mamba/attention), `BambaRMSNorm` (final), `BambaRotaryEmbedding`; direct `L1/embedding.py`
  - **`BambaForCausalLM`** [wiring]: wires `BambaModel`; direct `L1/linear.py` (lm_head)

## bark
- **src**: modeling_bark.py
- **hidden_act**: hardcoded gelu (`nn.GELU()`); no hidden_act in config
- **status**: composable
- **classes**:
  - **`BarkSelfAttention`** [compute]: `L1/linear.py + L1/dense_attention.py + L1/store_kvcache.py + L1/linear.py + L1/dropout.py` (fused QKV via single att_proj split into q/k/v; manual masked softmax with optional causal mask; out_proj + resid dropout) — no exact L2 match (it's GPT-2-style with combined att_proj rather than separate q/k/v)
  - **`BarkSelfFlashAttention2`** [compute, inherits `BarkSelfAttention`]: same as above but with flash_attention_2 path — kb-nano equivalent: `L1/flash_attn_prefill.py` / `L1/flash_attn_decode.py`
  - **`BarkMLP`** [compute]: `L1/linear.py + L1/gelu.py + L1/linear.py + L1/dropout.py` (in_proj 4x + GELU + out_proj + dropout — GPT2-style 2-layer MLP)
  - **`BarkBlock`** [wiring]: wires `BarkSelfAttention` (or FlashAttention2 variant), `BarkMLP`; direct `L1/layer_norm.py` (x2: layernorm_1/layernorm_2)
  - **`BarkCausalModel`** [wiring]: wires `BarkBlock`; direct `L1/embedding.py` (x2: input_embeds_layer + position_embeds_layer), `L1/layer_norm.py` (final), `L1/linear.py` (lm_head), `L1/dropout.py`
  - **`BarkSemanticModel`** [wiring, inherits `BarkCausalModel`]: same as `BarkCausalModel`
  - **`BarkCoarseModel`** [wiring, inherits `BarkCausalModel`]: same as `BarkCausalModel`
  - **`BarkFineModel`** [wiring]: wires `BarkBlock` (non-causal); direct `L1/embedding.py` (n_codes_total embedding tables), `L1/embedding.py` (position), `L1/layer_norm.py` (final), `L1/linear.py` (multiple lm_heads), `L1/dropout.py`
  - **`BarkModel`** [wiring]: wires `BarkSemanticModel`, `BarkCoarseModel`, `BarkFineModel`, plus an Encodec codec model (external)

## bart
- **src**: modeling_bart.py
- **hidden_act**: activation_function=gelu
- **status**: composable
- **classes**:
  - **`BartLearnedPositionalEmbedding`** [compute, inherits `nn.Embedding`]: `L1/embedding.py` (with offset=2)
  - **`BartScaledWordEmbedding`** [compute, inherits `nn.Embedding`]: `L1/embedding.py` (with embed_scale multiplier)
  - **`BartAttention`** [compute]: `L2/whisper_attention.py` (encoder/decoder/cross variants via is_decoder + key_value_states + EncoderDecoderCache; same Whisper-family pattern)
  - **`BartEncoderLayer`** [wiring]: wires `BartAttention`; direct `L1/layer_norm.py` (x2: self_attn_layer_norm, final_layer_norm), `L1/linear.py` (fc1, fc2), `L1/gelu.py`, `L1/dropout.py`
  - **`BartDecoderLayer`** [wiring]: wires `BartAttention` (x2: self_attn + encoder_attn cross); direct `L1/layer_norm.py` (x3), `L1/linear.py` (fc1, fc2), `L1/gelu.py`, `L1/dropout.py`
  - **`BartClassificationHead`** [compute]: `L1/linear.py + L1/tanh.py + L1/linear.py + L1/dropout.py`
  - **`BartEncoder`** [wiring]: wires `BartEncoderLayer`, `BartScaledWordEmbedding`, `BartLearnedPositionalEmbedding`; direct `L1/layer_norm.py` (layernorm_embedding), `L1/dropout.py`
  - **`BartDecoder`** [wiring]: wires `BartDecoderLayer`, `BartScaledWordEmbedding`, `BartLearnedPositionalEmbedding`; direct `L1/layer_norm.py` (layernorm_embedding), `L1/dropout.py`
  - **`BartModel`** [wiring]: wires `BartEncoder`, `BartDecoder`, `BartScaledWordEmbedding` (shared)
  - **`BartForConditionalGeneration`** [wiring]: wires `BartModel`; direct `L1/linear.py` (lm_head); also has final_logits_bias buffer
  - **`BartDecoderWrapper`** [wiring]: wires `BartDecoder`
  - **`BartForCausalLM`** [wiring]: wires `BartDecoderWrapper`; direct `L1/linear.py` (lm_head)
- **task heads (2)**: ForQuestionAnswering, ForSequenceClassification — base + linear (per-task)

## beit
- **src**: modeling_beit.py
- **hidden_act**: gelu
- **status**: composable
- **classes**:
  - **`BeitDropPath`** [compute]: stochastic depth (`L1/dropout.py` semantically)
  - **`BeitPatchEmbeddings`** [compute]: `L1/conv2d.py` (Conv2d patch projection)
  - **`BeitEmbeddings`** [compute]: wires `BeitPatchEmbeddings`; direct CLS token, optional mask token, learned position embedding parameter, dropout — supports `interpolate_pos_encoding`
  - **`BeitSelfAttention`** [compute]: `L1/linear.py + L1/dense_attention.py` (q/k/v + manual softmax, with optional learned relative position bias added to attention scores) — close to `L2/encoder_attention.py` but with relative-position-bias addition; key bias is False
  - **`BeitSdpaSelfAttention`** [compute, inherits `BeitSelfAttention`]: same with `L1/dense_attention.py` (F.scaled_dot_product_attention)
  - **`BeitSelfOutput`** [compute]: `L1/linear.py + L1/dropout.py` (no LayerNorm — pre-norm style; residual handled in BeitLayer)
  - **`BeitAttention`** [compute]: wraps self.attention + self.output (BERT-style wrapper) — close to `L2/encoder_attention.py` but with relative position bias
  - **`BeitIntermediate`** [compute]: `L1/linear.py + L1/gelu.py`
  - **`BeitOutput`** [compute]: `L1/linear.py + L1/dropout.py` (no LayerNorm or residual; handled in BeitLayer)
  - **`BeitLayer`** [wiring]: wires `BeitAttention`, `BeitIntermediate`, `BeitOutput`; direct `L1/layer_norm.py` (x2: layernorm_before/after), optional `BeitDropPath`, optional `lambda_1`/`lambda_2` LayerScale parameters
  - **`BeitRelativePositionBias`** [compute]: learned relative-position-bias table indexed by relative coords; pure tensor ops
  - **`BeitEncoder`** [wiring]: wires `BeitLayer` (with optional shared `BeitRelativePositionBias`)
  - **`BeitModel`** [wiring]: wires `BeitEmbeddings`, `BeitEncoder`; direct `L1/layer_norm.py` (final), optional `BeitPooler`
  - **`BeitPooler`** [compute]: `L1/layer_norm.py` (mean-pool patch tokens then LayerNorm)
  - **`BeitConvModule`** [compute]: `L1/conv2d.py + L1/batch_norm2d.py + L1/relu.py` (segmentation head conv block)
  - **`BeitPyramidPoolingBlock`** [compute]: `L1/adaptive_avg_pool2d.py + L1/conv2d.py + L1/batch_norm2d.py + L1/relu.py`
  - **`BeitPyramidPoolingModule`** [wiring]: wires `BeitPyramidPoolingBlock` (multi-scale)
  - **`BeitUperHead`** [wiring]: wires `BeitConvModule`, `BeitPyramidPoolingModule` (UperNet decode head for segmentation)
  - **`BeitFCNHead`** [wiring]: wires `BeitConvModule` (FCN auxiliary head)
  - **`BeitBackbone`** [wiring, inherits `BackboneMixin`]: wires `BeitEmbeddings`, `BeitEncoder`
- **task heads (3)**: ForImageClassification, ForMaskedImageModeling, ForSemanticSegmentation — base + linear (per-task)

## bert
- **src**: modeling_bert.py
- **hidden_act**: gelu
- **status**: composable
- **classes**:
  - **`BertEmbeddings`** [compute]: `L2/encoder_embeddings.py` (word + position + token_type + LayerNorm + Dropout)
  - **`BertSelfAttention`** [compute]: `L2/encoder_attention.py` (q/k/v + dispatch via ALL_ATTENTION_FUNCTIONS)
  - **`BertCrossAttention`** [compute]: `L2/encoder_attention.py` (q/k/v with key_value_states for cross-attention)
  - **`BertSelfOutput`** [compute]: `L2/encoder_attention.py` (dense + LayerNorm + residual)
  - **`BertAttention`** [compute]: `L2/encoder_attention.py` (wrapper: self.self + self.output)
  - **`BertIntermediate`** [compute]: `L1/linear.py + L1/gelu.py` (no exact L2 match — encoder_mlp.py covers Intermediate+Output; BertIntermediate is just one half)
  - **`BertOutput`** [compute]: `L1/linear.py + L1/dropout.py + L1/layer_norm.py` (dense + dropout + LayerNorm + residual)
  - **`BertLayer`** [wiring]: wires `BertAttention`, `BertIntermediate`, `BertOutput`, optional `BertCrossAttention`
  - **`BertEncoder`** [wiring]: wires `BertLayer`
  - **`BertPooler`** [compute]: `L1/linear.py + L1/tanh.py`
  - **`BertPredictionHeadTransform`** [compute]: `L1/linear.py + L1/gelu.py + L1/layer_norm.py`
  - **`BertLMPredictionHead`** [wiring]: wires `BertPredictionHeadTransform`; direct `L1/linear.py` (decoder)
  - **`BertOnlyMLMHead`** [wiring]: wires `BertLMPredictionHead`
  - **`BertOnlyNSPHead`** [compute]: `L1/linear.py` (binary classification)
  - **`BertPreTrainingHeads`** [wiring]: wires `BertLMPredictionHead`; direct `L1/linear.py` (seq_relationship)
  - **`BertModel`** [wiring]: wires `BertEmbeddings`, `BertEncoder`, optional `BertPooler`
  - **`BertForPreTraining`** [wiring]: wires `BertModel`, `BertPreTrainingHeads`
  - **`BertLMHeadModel`** [wiring]: wires `BertModel`, `BertOnlyMLMHead`
  - **`BertForMaskedLM`** [wiring]: wires `BertModel`, `BertOnlyMLMHead`
- **task heads (5)**: ForMultipleChoice, ForNextSentencePrediction, ForQuestionAnswering, ForSequenceClassification, ForTokenClassification — base + linear (per-task)

## bert_generation
- **src**: modeling_bert_generation.py
- **hidden_act**: gelu
- **status**: composable
- **classes**:
  - **`BertGenerationSelfOutput`** [compute]: `L2/encoder_attention.py` (dense + LayerNorm + residual)
  - **`BertGenerationSelfAttention`** [compute]: `L2/encoder_attention.py` (q/k/v + dispatch via ALL_ATTENTION_FUNCTIONS)
  - **`BertGenerationCrossAttention`** [compute]: `L2/encoder_attention.py` (cross-attention with key_value_states)
  - **`BertGenerationAttention`** [compute]: `L2/encoder_attention.py` (wrapper: self.self + self.output)
  - **`BertGenerationIntermediate`** [compute]: `L1/linear.py + L1/gelu.py`
  - **`BertGenerationOutput`** [compute]: `L1/linear.py + L1/dropout.py + L1/layer_norm.py` (dense + dropout + LayerNorm + residual)
  - **`BertGenerationLayer`** [wiring]: wires `BertGenerationAttention`, `BertGenerationIntermediate`, `BertGenerationOutput`, optional `BertGenerationCrossAttention`
  - **`BertEncoder`** [wiring]: wires `BertGenerationLayer`
  - **`BertGenerationEmbeddings`** [compute]: `L1/embedding.py + L1/embedding.py + L1/layer_norm.py + L1/dropout.py` (word + position only — no token_type — close to but not exactly `L2/encoder_embeddings.py`)
  - **`BertGenerationEncoder`** [wiring]: wires `BertGenerationEmbeddings`, `BertEncoder`
  - **`BertGenerationOnlyLMHead`** [compute]: `L1/linear.py` (LM head with bias parameter)
  - **`BertGenerationDecoder`** [wiring]: wires `BertGenerationEncoder`, `BertGenerationOnlyLMHead`

## big_bird
- **src**: modeling_big_bird.py
- **hidden_act**: gelu_new
- **status**: partial (block-sparse attention has no kb-nano kernel — falls back to dense path; full-attention path is composable)
- **classes**:
  - **`BigBirdEmbeddings`** [compute]: `L2/encoder_embeddings.py` (word + position + token_type + LayerNorm + Dropout, with optional rescale)
  - **`BigBirdSelfAttention`** [compute]: `L2/encoder_attention.py` (q/k/v + manual softmax with optional cross-attention; close to BertSelfAttention but with cross-attn KV cache support)
  - **`BigBirdBlockSparseAttention`** [compute]: no kb-nano kernel — block-sparse pattern with band/random/global blocks; custom `bigbird_block_sparse_attention` math
  - **`BigBirdSelfOutput`** [compute]: `L2/encoder_attention.py` (dense + LayerNorm + residual)
  - **`BigBirdAttention`** [compute]: `L2/encoder_attention.py` (wrapper: dispatches to either `original_full` self-attn or block-sparse + self.output)
  - **`BigBirdIntermediate`** [compute]: `L1/linear.py + L1/gelu.py` (gelu_new)
  - **`BigBirdOutput`** [compute]: `L1/linear.py + L1/dropout.py + L1/layer_norm.py`
  - **`BigBirdLayer`** [wiring]: wires `BigBirdAttention`, `BigBirdIntermediate`, `BigBirdOutput`, optional cross-attn
  - **`BigBirdEncoder`** [wiring]: wires `BigBirdLayer`
  - **`BigBirdPredictionHeadTransform`** [compute]: `L1/linear.py + L1/gelu.py + L1/layer_norm.py`
  - **`BigBirdLMPredictionHead`** [wiring]: wires `BigBirdPredictionHeadTransform`; direct `L1/linear.py` (decoder)
  - **`BigBirdOnlyMLMHead`** [wiring]: wires `BigBirdLMPredictionHead`
  - **`BigBirdOnlyNSPHead`** [compute]: `L1/linear.py`
  - **`BigBirdPreTrainingHeads`** [wiring]: wires `BigBirdLMPredictionHead`; direct `L1/linear.py`
  - **`BigBirdClassificationHead`** [compute]: `L1/linear.py + L1/tanh.py + L1/dropout.py + L1/linear.py`
  - **`BigBirdForQuestionAnsweringHead`** [compute]: `L1/dropout.py + L1/linear.py` (intermediate transform)
  - **`BigBirdModel`** [wiring]: wires `BigBirdEmbeddings`, `BigBirdEncoder`, optional pooler (Linear + Tanh)
  - **`BigBirdForPreTraining`** [wiring]: wires `BigBirdModel`, `BigBirdPreTrainingHeads`
  - **`BigBirdForMaskedLM`** [wiring]: wires `BigBirdModel`, `BigBirdOnlyMLMHead`
  - **`BigBirdForCausalLM`** [wiring]: wires `BigBirdModel`, `BigBirdOnlyMLMHead`
- **task heads (5)**: ForMultipleChoice, ForQuestionAnswering, ForSequenceClassification, ForTokenClassification — base + linear (per-task)

## bigbird_pegasus
- **src**: modeling_bigbird_pegasus.py
- **hidden_act**: activation_function=gelu_new
- **status**: partial (block-sparse attention has no kb-nano kernel; full-attention path is composable)
- **classes**:
  - **`BigBirdPegasusLearnedPositionalEmbedding`** [compute, inherits `nn.Embedding`]: `L1/embedding.py`
  - **`BigBirdPegasusScaledWordEmbedding`** [compute, inherits `nn.Embedding`]: `L1/embedding.py` (with embed_scale)
  - **`BigBirdPegasusSelfAttention`** [compute]: `L2/encoder_attention.py` (q/k/v + manual softmax — full self-attention path used for encoder when not using block-sparse)
  - **`BigBirdPegasusBlockSparseAttention`** [compute]: no kb-nano kernel — same block-sparse pattern as BigBird
  - **`BigBirdPegasusEncoderAttention`** [compute]: `L2/encoder_attention.py` (wrapper that dispatches to either full or block-sparse self-attention)
  - **`BigBirdPegasusDecoderAttention`** [compute]: `L2/whisper_attention.py` (Pegasus-style decoder/cross attention with EncoderDecoderCache)
  - **`BigBirdPegasusEncoderLayer`** [wiring]: wires `BigBirdPegasusEncoderAttention`; direct `L1/layer_norm.py` (x2), `L1/linear.py` (fc1, fc2), `L1/gelu.py` (gelu_new), `L1/dropout.py` — Pegasus pre-norm pattern
  - **`BigBirdPegasusDecoderLayer`** [wiring]: wires `BigBirdPegasusDecoderAttention` (x2: self + encoder cross-attn); direct `L1/layer_norm.py` (x3), `L1/linear.py` (fc1/fc2), `L1/gelu.py`, `L1/dropout.py`
  - **`BigBirdPegasusClassificationHead`** [compute]: `L1/linear.py + L1/tanh.py + L1/dropout.py + L1/linear.py`
  - **`BigBirdPegasusEncoder`** [wiring]: wires `BigBirdPegasusEncoderLayer`, `BigBirdPegasusScaledWordEmbedding`, `BigBirdPegasusLearnedPositionalEmbedding`; direct `L1/layer_norm.py` (final), `L1/dropout.py`
  - **`BigBirdPegasusDecoder`** [wiring]: wires `BigBirdPegasusDecoderLayer`, `BigBirdPegasusScaledWordEmbedding`, `BigBirdPegasusLearnedPositionalEmbedding`; direct `L1/layer_norm.py` (final), `L1/dropout.py`
  - **`BigBirdPegasusModel`** [wiring]: wires `BigBirdPegasusEncoder`, `BigBirdPegasusDecoder`, `BigBirdPegasusScaledWordEmbedding` (shared)
  - **`BigBirdPegasusForConditionalGeneration`** [wiring]: wires `BigBirdPegasusModel`; direct `L1/linear.py` (lm_head); has final_logits_bias
  - **`BigBirdPegasusDecoderWrapper`** [wiring]: wires `BigBirdPegasusDecoder`
  - **`BigBirdPegasusForCausalLM`** [wiring]: wires `BigBirdPegasusDecoderWrapper`; direct `L1/linear.py` (lm_head)
- **task heads (2)**: ForQuestionAnswering, ForSequenceClassification — base + linear (per-task)

## biogpt
- **src**: modeling_biogpt.py
- **hidden_act**: gelu
- **status**: composable
- **classes**:
  - **`BioGptLearnedPositionalEmbedding`** [compute, inherits `nn.Embedding`]: `L1/embedding.py` (with offset=2 like BART)
  - **`BioGptScaledWordEmbedding`** [compute, inherits `nn.Embedding`]: `L1/embedding.py` (with embed_scale)
  - **`BioGptAttention`** [compute]: `L2/whisper_attention.py` (decoder-style with KV cache, no cross-attention used here since BioGpt is decoder-only)
  - **`BioGptDecoderLayer`** [wiring]: wires `BioGptAttention`; direct `L1/layer_norm.py` (x2: self_attn_layer_norm, final_layer_norm), `L1/linear.py` (fc1, fc2), `L1/gelu.py`, `L1/dropout.py` — pre-norm pattern
  - **`BioGptModel`** [wiring]: wires `BioGptDecoderLayer`, `BioGptScaledWordEmbedding`, `BioGptLearnedPositionalEmbedding`; direct `L1/layer_norm.py` (final), `L1/dropout.py`
  - **`BioGptForCausalLM`** [wiring]: wires `BioGptModel`; direct `L1/linear.py` (output_projection)
- **task heads (2)**: ForSequenceClassification, ForTokenClassification — base + linear (per-task)

## bit
- **src**: modeling_bit.py
- **hidden_act**: relu
- **status**: composable
- **classes**:
  - **`WeightStandardizedConv2d`** [compute, inherits `nn.Conv2d`]: `L1/conv2d.py + L1/batch_norm2d.py` (Conv2d with weight standardization via batch_norm on weights — no exact kb-nano kernel; weight-norm trick is at-call-time)
  - **`BitGroupNormActivation`** [compute, inherits `nn.GroupNorm`]: `L1/group_norm.py + L1/relu.py` (GroupNorm + activation)
  - **`DynamicPad2d`** [compute]: pure tensor pad (no specific kb-nano kernel)
  - **`BitMaxPool2d`** [compute, inherits `nn.MaxPool2d`]: `L1/max_pool2d.py` (with optional dynamic padding)
  - **`BitEmbeddings`** [wiring]: wires `WeightStandardizedConv2d`, `BitMaxPool2d`, `BitGroupNormActivation`
  - **`BitDropPath`** [compute]: stochastic depth (no specific kb-nano kernel — use `L1/dropout.py` semantically)
  - **`BitPreActivationBottleneckLayer`** [wiring]: wires `WeightStandardizedConv2d` (x3), `BitGroupNormActivation` (x3), optional `BitDownsampleConv`, `BitDropPath` — pre-act ResNet bottleneck
  - **`BitBottleneckLayer`** [wiring]: wires `WeightStandardizedConv2d` (x3), `BitGroupNormActivation` (x3), optional `BitDownsampleConv`, `BitDropPath`; direct `L1/relu.py` (final activation)
  - **`BitDownsampleConv`** [wiring]: wires `WeightStandardizedConv2d`, optional `BitGroupNormActivation`
  - **`BitStage`** [wiring]: wires `BitPreActivationBottleneckLayer` or `BitBottleneckLayer`
  - **`BitEncoder`** [wiring]: wires `BitStage`
  - **`BitModel`** [wiring]: wires `BitEmbeddings`, `BitEncoder`, optional final `BitGroupNormActivation`; direct `L1/adaptive_avg_pool2d.py` (pooler)
  - **`BitBackbone`** [wiring, inherits `BackboneMixin`]: wires `BitEmbeddings`, `BitEncoder`
- **task heads (1)**: ForImageClassification — base + linear (per-task)

## bitnet
- **src**: modeling_bitnet.py, modular_bitnet.py
- **hidden_act**: relu2 (squared relu)
- **status**: composable
- **classes**:
  - **`BitNetRMSNorm`** [compute]: `L1/rms_norm.py` (note: kb-nano also has `L1/bitnet_rms_norm.py` but BitNetRMSNorm in HF source is identical to standard RMSNorm — bitnet_rms_norm in kb-nano is for the full BitNet quantized linear path)
  - **`BitNetMLP`** [compute]: `L2/bitnet_mlp.py` (SwiGLU with relu2 activation + sub-norm before down_proj — matches kb-nano BitNet MLP)
  - **`BitNetAttention`** [compute]: `L2/bitnet_attention.py` (Llama-style q/k/v + RoPE + KV cache + attn_sub_norm before o_proj — matches kb-nano BitNet attention)
  - **`BitNetDecoderLayer`** [wiring]: wires `BitNetAttention`, `BitNetMLP`, `BitNetRMSNorm` (x2: input_layernorm/post_attention_layernorm)
  - **`BitNetRotaryEmbedding`** [compute]: `L1/rotary_emb.py`
  - **`BitNetModel`** [wiring]: wires `BitNetDecoderLayer`, `BitNetRMSNorm` (final), `BitNetRotaryEmbedding`; direct `L1/embedding.py`
  - **`BitNetForCausalLM`** [wiring]: wires `BitNetModel`; direct `L1/linear.py` (lm_head)

## blenderbot
- **src**: modeling_blenderbot.py
- **hidden_act**: activation_function=gelu
- **status**: composable
- **classes**:
  - **`BlenderbotLearnedPositionalEmbedding`** [compute, inherits `nn.Embedding`]: `L1/embedding.py`
  - **`BlenderbotScaledWordEmbedding`** [compute, inherits `nn.Embedding`]: `L1/embedding.py` (with embed_scale)
  - **`BlenderbotAttention`** [compute]: `L2/whisper_attention.py` (encoder/decoder/cross variants via is_decoder + key_value_states + EncoderDecoderCache)
  - **`BlenderbotEncoderLayer`** [wiring]: wires `BlenderbotAttention`; direct `L1/layer_norm.py` (x2), `L1/linear.py` (fc1, fc2), `L1/gelu.py`, `L1/dropout.py` — pre-norm BART-style
  - **`BlenderbotDecoderLayer`** [wiring]: wires `BlenderbotAttention` (x2: self + cross); direct `L1/layer_norm.py` (x3), `L1/linear.py` (fc1/fc2), `L1/gelu.py`, `L1/dropout.py`
  - **`BlenderbotEncoder`** [wiring]: wires `BlenderbotEncoderLayer`, `BlenderbotScaledWordEmbedding`, `BlenderbotLearnedPositionalEmbedding`; direct `L1/layer_norm.py` (final), `L1/dropout.py`
  - **`BlenderbotDecoder`** [wiring]: wires `BlenderbotDecoderLayer`, `BlenderbotScaledWordEmbedding`, `BlenderbotLearnedPositionalEmbedding`; direct `L1/layer_norm.py` (final), `L1/dropout.py`
  - **`BlenderbotModel`** [wiring]: wires `BlenderbotEncoder`, `BlenderbotDecoder`, `BlenderbotScaledWordEmbedding` (shared)
  - **`BlenderbotForConditionalGeneration`** [wiring]: wires `BlenderbotModel`; direct `L1/linear.py` (lm_head); has final_logits_bias
  - **`BlenderbotDecoderWrapper`** [wiring]: wires `BlenderbotDecoder`
  - **`BlenderbotForCausalLM`** [wiring]: wires `BlenderbotDecoderWrapper`; direct `L1/linear.py` (lm_head)

## blenderbot_small
- **src**: modeling_blenderbot_small.py
- **hidden_act**: activation_function=gelu
- **status**: composable
- **classes**:
  - **`BlenderbotSmallLearnedPositionalEmbedding`** [compute, inherits `nn.Embedding`]: `L1/embedding.py`
  - **`BlenderbotSmallAttention`** [compute]: `L2/whisper_attention.py` (BART-style with EncoderDecoderCache)
  - **`BlenderbotSmallEncoderLayer`** [wiring]: wires `BlenderbotSmallAttention`; direct `L1/layer_norm.py` (x2: post-norm, BART-style), `L1/linear.py` (fc1, fc2), `L1/gelu.py`, `L1/dropout.py`
  - **`BlenderbotSmallDecoderLayer`** [wiring]: wires `BlenderbotSmallAttention` (x2: self + encoder cross); direct `L1/layer_norm.py` (x3), `L1/linear.py` (fc1/fc2), `L1/gelu.py`, `L1/dropout.py`
  - **`BlenderbotSmallEncoder`** [wiring]: wires `BlenderbotSmallEncoderLayer`, `BlenderbotSmallLearnedPositionalEmbedding`; direct `L1/embedding.py` (embed_tokens), `L1/layer_norm.py` (layernorm_embedding), `L1/dropout.py`
  - **`BlenderbotSmallDecoder`** [wiring]: wires `BlenderbotSmallDecoderLayer`, `BlenderbotSmallLearnedPositionalEmbedding`; direct `L1/embedding.py`, `L1/layer_norm.py`, `L1/dropout.py`
  - **`BlenderbotSmallModel`** [wiring]: wires `BlenderbotSmallEncoder`, `BlenderbotSmallDecoder`; direct `L1/embedding.py` (shared)
  - **`BlenderbotSmallForConditionalGeneration`** [wiring]: wires `BlenderbotSmallModel`; direct `L1/linear.py` (lm_head); has final_logits_bias
  - **`BlenderbotSmallDecoderWrapper`** [wiring]: wires `BlenderbotSmallDecoder`
  - **`BlenderbotSmallForCausalLM`** [wiring]: wires `BlenderbotSmallDecoderWrapper`; direct `L1/linear.py` (lm_head)

## blip/blip
- **src**: modeling_blip.py
- **hidden_act**: gelu (both vision and text configs)
- **status**: composable
- **classes**:
  - **`BlipVisionEmbeddings`** [compute]: `L1/conv2d.py` (CLS token + patch_embedding Conv2d + learned position embedding parameter; supports `interpolate_pos_encoding`)
  - **`BlipTextEmbeddings`** [compute]: `L1/embedding.py + L1/embedding.py` (token_embedding + position_embedding sum)
  - **`BlipAttention`** [compute]: `L2/clip_attention.py` (fused qkv into a single Linear; non-causal vision attention with manual softmax + projection) — note: uses fused qkv rather than separate q/k/v Linears like CLIP
  - **`BlipMLP`** [compute]: `L2/clip_mlp.py` (fc1 + gelu + fc2 — same as CLIP MLP but gelu instead of quickgelu)
  - **`BlipEncoderLayer`** [wiring]: wires `BlipAttention`, `BlipMLP`; direct `L1/layer_norm.py` (x2: layer_norm1/layer_norm2)
  - **`BlipEncoder`** [wiring]: wires `BlipEncoderLayer`
  - **`BlipVisionModel`** [wiring]: wires `BlipVisionEmbeddings`, `BlipEncoder`; direct `L1/layer_norm.py` (x2: pre_layernorm + post_layernorm)
  - **`BlipModel`** [wiring]: wires `BlipVisionModel`, BlipTextModel (from modeling_blip_text); direct `L1/linear.py` (visual_projection, text_projection)
  - **`BlipForConditionalGeneration`** [wiring]: wires `BlipVisionModel`, BlipTextLMHeadModel (from modeling_blip_text)
  - **`BlipForQuestionAnswering`** [wiring]: wires `BlipVisionModel`, BlipTextModel (encoder), BlipTextLMHeadModel (decoder)
  - **`BlipForImageTextRetrieval`** [wiring]: wires `BlipVisionModel`, BlipTextModel; direct `L1/linear.py` (vision_proj, text_proj, itm_head)

## blip/blip_text
- **src**: modeling_blip_text.py
- **hidden_act**: gelu
- **status**: composable
- **classes**:
  - **`BlipTextEmbeddings`** [compute]: `L2/encoder_embeddings.py` (word + position + token_type + LayerNorm + Dropout — BERT-style)
  - **`BlipTextSelfAttention`** [compute]: `L2/encoder_attention.py` (BERT-style q/k/v with optional cross-attention via encoder_hidden_states; manual softmax + KV cache)
  - **`BlipTextSelfOutput`** [compute]: `L2/encoder_attention.py` (dense + LayerNorm + residual)
  - **`BlipTextAttention`** [compute]: `L2/encoder_attention.py` (wrapper: self.self + self.output, optionally with cross-attention)
  - **`BlipTextIntermediate`** [compute]: `L1/linear.py + L1/gelu.py`
  - **`BlipTextOutput`** [compute]: `L1/linear.py + L1/dropout.py + L1/layer_norm.py`
  - **`BlipTextLayer`** [wiring]: wires `BlipTextAttention` (self), optional `BlipTextAttention` (crossattention), `BlipTextIntermediate`, `BlipTextOutput`
  - **`BlipTextEncoder`** [wiring]: wires `BlipTextLayer`
  - **`BlipTextPooler`** [compute]: `L1/linear.py + L1/tanh.py`
  - **`BlipTextPredictionHeadTransform`** [compute]: `L1/linear.py + L1/gelu.py + L1/layer_norm.py`
  - **`BlipTextLMPredictionHead`** [wiring]: wires `BlipTextPredictionHeadTransform`; direct `L1/linear.py` (decoder)
  - **`BlipTextOnlyMLMHead`** [wiring]: wires `BlipTextLMPredictionHead`
  - **`BlipTextModel`** [wiring]: wires `BlipTextEmbeddings`, `BlipTextEncoder`, optional `BlipTextPooler` (BERT-style with optional cross-attention)
  - **`BlipTextLMHeadModel`** [wiring]: wires `BlipTextModel`, `BlipTextOnlyMLMHead`

## blip_2
- **src**: modeling_blip_2.py
- **hidden_act**: gelu (vision and qformer); language model uses its own (e.g. OPT/T5)
- **status**: composable
- **classes**:
  - **`Blip2VisionEmbeddings`** [compute]: `L1/conv2d.py` (CLS token + patch_embedding Conv2d + learned position embedding parameter; supports interpolate)
  - **`Blip2Attention`** [compute]: `L2/clip_attention.py` (fused qkv with optional q_bias/v_bias; non-causal vision attention via ALL_ATTENTION_FUNCTIONS)
  - **`Blip2MLP`** [compute]: `L2/clip_mlp.py` (fc1 + gelu + fc2)
  - **`Blip2EncoderLayer`** [wiring]: wires `Blip2Attention`, `Blip2MLP`; direct `L1/layer_norm.py` (x2)
  - **`Blip2Encoder`** [wiring]: wires `Blip2EncoderLayer`
  - **`Blip2VisionModel`** [wiring]: wires `Blip2VisionEmbeddings`, `Blip2Encoder`; direct `L1/layer_norm.py` (post_layernorm)
  - **`Blip2QFormerMultiHeadAttention`** [compute]: `L2/encoder_attention.py` (BERT-style q/k/v with optional cross-attention via encoder_hidden_states; manual softmax)
  - **`Blip2QFormerSelfOutput`** [compute]: `L2/encoder_attention.py` (dense + LayerNorm + residual)
  - **`Blip2QFormerAttention`** [compute]: `L2/encoder_attention.py` (wrapper: self.attention + self.output)
  - **`Blip2QFormerIntermediate`** [compute]: `L1/linear.py + L1/gelu.py`
  - **`Blip2QFormerOutput`** [compute]: `L1/linear.py + L1/dropout.py + L1/layer_norm.py`
  - **`Blip2QFormerLayer`** [wiring]: wires `Blip2QFormerAttention` (self), optional `Blip2QFormerAttention` (crossattention), `Blip2QFormerIntermediate`, `Blip2QFormerOutput`
  - **`Blip2QFormerEncoder`** [wiring]: wires `Blip2QFormerLayer`
  - **`Blip2TextEmbeddings`** [compute]: `L1/embedding.py + L1/embedding.py + L1/layer_norm.py + L1/dropout.py` (word + position with LayerNorm)
  - **`Blip2QFormerModel`** [wiring]: wires `Blip2TextEmbeddings`, `Blip2QFormerEncoder`; direct `L1/layer_norm.py`
  - **`Blip2Model`** [wiring]: wires `Blip2VisionModel`, `Blip2QFormerModel`, language_model (AutoModelForCausalLM/Seq2SeqLM); direct `L1/linear.py` (language_projection)
  - **`Blip2TextModelWithProjection`** [wiring]: wires `Blip2QFormerModel`; direct `L1/linear.py` (text_projection)
  - **`Blip2VisionModelWithProjection`** [wiring]: wires `Blip2VisionModel`, `Blip2QFormerModel`; direct `L1/linear.py` (vision_projection)
  - **`Blip2ForConditionalGeneration`** [wiring]: wires `Blip2VisionModel`, `Blip2QFormerModel`, language_model; direct `L1/linear.py` (language_projection)
  - **`Blip2ForImageTextRetrieval`** [wiring]: wires `Blip2VisionModel`, `Blip2QFormerModel`; direct `L1/linear.py` (vision_projection, text_projection, itm_head)

