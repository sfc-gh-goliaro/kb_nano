## cvt
- **src**: modeling_cvt.py
- **hidden_act**: n/a (uses `nn.GELU()` directly in `CvtIntermediate`)
- **status**: composable
- **classes**:
  - **`CvtDropPath`** [compute]: stochastic depth (no kb-nano kernel; identity at inference)
  - **`CvtEmbeddings`** [wiring]: wires `CvtConvEmbeddings`; skips `nn.Dropout`
  - **`CvtConvEmbeddings`** [compute]: `L1/conv2d.py + L1/layer_norm.py`
  - **`CvtSelfAttentionConvProjection`** [compute]: `L1/conv2d.py + L1/batch_norm2d.py` (depthwise conv + bn)
  - **`CvtSelfAttentionLinearProjection`** [compute]: pure reshape (no kb-nano kernel; just permute/view)
  - **`CvtSelfAttentionProjection`** [wiring]: wires `CvtSelfAttentionConvProjection`, `CvtSelfAttentionLinearProjection`
  - **`CvtSelfAttention`** [compute]: `L1/linear.py (×3 q/k/v) + L1/dense_attention.py + CvtSelfAttentionProjection (×3)` (no exact L2 match — qkv from conv-proj then linear-proj, manual softmax via einsum)
  - **`CvtSelfOutput`** [compute]: `L1/linear.py` (just dense; residual is in CvtLayer)
  - **`CvtAttention`** [wiring]: wires `CvtSelfAttention`, `CvtSelfOutput`
  - **`CvtIntermediate`** [compute]: `L1/linear.py + L1/gelu.py`
  - **`CvtOutput`** [compute]: `L1/linear.py` + residual (no exact L2 match)
  - **`CvtLayer`** [wiring]: wires `CvtAttention`, `CvtIntermediate`, `CvtOutput`, `CvtDropPath`; direct `L1/layer_norm.py (×2 layernorm_before/after)`
  - **`CvtStage`** [wiring]: wires `CvtEmbeddings`, `CvtLayer (×depth)`; direct `L1/embedding.py`-style `cls_token` Parameter (not a real nn.Embedding)
  - **`CvtEncoder`** [wiring]: wires `CvtStage` (×len(config.depth))
  - **`CvtModel`** [wiring]: wires `CvtEncoder`
- **task heads (1)**: ForImageClassification — base + linear (per-task)

## cwm
- **src**: modeling_cwm.py (and modular_cwm.py — inherits from Llama/Qwen2)
- **hidden_act**: silu
- **status**: composable
- **classes**:
  - **`CwmRotaryEmbedding`** [compute, inherits `Qwen2RotaryEmbedding`]: `L1/rotary_emb.py`
  - **`CwmAttention`** [compute, inherits `Qwen2Attention`]: `L2/attention.py` (decoder causal w/ RoPE + KV cache + sliding-window for sliding_attention layers)
  - **`CwmRMSNorm`** [compute]: `L1/rms_norm.py`
  - **`CwmMLP`** [compute]: `L2/llama_mlp.py` (gate_proj/up_proj/down_proj + silu)
  - **`CwmDecoderLayer`** [wiring, inherits `LlamaDecoderLayer`]: wires `CwmAttention`, `CwmMLP`, `CwmRMSNorm` (×2)
  - **`CwmModel`** [wiring, inherits `LlamaModel`]: wires `CwmDecoderLayer (×N)`, `CwmRMSNorm`, `CwmRotaryEmbedding`; direct `L1/embedding.py`
  - **`CwmForCausalLM`** [wiring, inherits `LlamaForCausalLM`]: wires `CwmModel`; direct `L1/linear.py` (lm_head)

## d_fine
- **src**: modeling_d_fine.py (and modular_d_fine.py — inherits from RT-DETR)
- **hidden_act**: encoder_activation_function=gelu, activation_function=silu, decoder_activation_function=relu
- **status**: composable
- **classes**:
  - **`DFineMLP`** [compute]: `L1/linear.py (×num_layers)` + activation per `act` kwarg (no exact L2 match — DETR-style FFN with non-standard num_layers)
  - **`DFineGate`** [compute]: `L1/linear.py + L1/sigmoid.py + L1/layer_norm.py` (gate fusion)
  - **`DFineFrozenBatchNorm2d`** [compute]: `L1/frozen_batch_norm2d.py`
  - **`DFineMultiscaleDeformableAttention`** [compute]: `L1/rtdetrv2_deformable_attention.py` (multi-scale deformable attn; v2 variant with grid_sample)
  - **`DFineConvNormLayer`** [compute, inherits `RTDetrConvNormLayer`]: `L2/rtdetrv2_conv_norm.py` (conv2d + bn + activation)
  - **`DFineRepVggBlock`** [compute, inherits `RTDetrRepVggBlock`]: `L2/rtdetrv2_repvgg_block.py` (×2 ConvNorm + activation)
  - **`DFineCSPRepLayer`** [compute]: `L2/rtdetrv2_csp_rep_layer.py` (cross-stage partial w/ RepVGG bottlenecks)
  - **`DFineRepNCSPELAN4`** [compute]: composition of `DFineConvNormLayer` and `DFineCSPRepLayer` (no exact L2 match)
  - **`DFineSCDown`** [wiring]: wires 2× `DFineConvNormLayer`
  - **`DFineSelfAttention`** [compute]: `L1/linear.py (×4 q/k/v/o) + L1/dense_attention.py` (DETR-style q/k get pos-embed added; non-causal)
  - **`DFineEncoderLayer`** [wiring, inherits `RTDetrEncoderLayer`]: wires `DFineSelfAttention`, `DFineMLP`; direct `L1/layer_norm.py (×2)` (`L2/rtdetrv2_encoder_layer.py`)
  - **`DFineSinePositionEmbedding`** [compute]: 2D sin/cos positions (no kb-nano L1 — math.arange/sin/cos)
  - **`DFineAIFILayer`** [wiring, inherits `RTDetrAIFILayer`]: wires `DFineSinePositionEmbedding`, `DFineEncoderLayer (×N)`
  - **`DFineIntegral`** [compute]: `L1/softmax.py + L1/linear.py` (integral over discrete bins)
  - **`DFineLQE`** [wiring]: wires `DFineMLP`; direct `L1/softmax.py` + topk
  - **`DFineDecoderLayer`** [wiring, inherits `RTDetrDecoderLayer`]: wires `DFineSelfAttention`, `DFineMultiscaleDeformableAttention`, `DFineMLP`, `DFineGate`; direct `L1/layer_norm.py (×2)`
  - **`DFineMLPPredictionHead`** [compute, inherits `RTDetrMLPPredictionHead`]: `L2/rtdetrv2_mlp_head.py` (linear + relu stack)
  - **`DFineHybridEncoder`** [wiring, inherits `RTDetrHybridEncoder`]: hybrid encoder mixing AIFI + CSP layers
  - **`DFineDecoder`** [wiring, inherits `RTDetrDecoder`]: wires `DFineDecoderLayer (×N)`, `DFineMLP` heads
  - **`DFineConvEncoder`** [wiring]: wires `load_backbone(config)` + replaces BN with frozen BN
  - **`DFineModel`** [wiring, inherits `RTDetrModel`]: wires `DFineConvEncoder`, `DFineHybridEncoder`, `DFineDecoder`; direct `L1/embedding.py + L1/linear.py + L1/layer_norm.py`
- **task heads (1)**: ForObjectDetection — base + linear (per-task)

## dab_detr
- **src**: modeling_dab_detr.py
- **hidden_act**: prelu (activation_function default)
- **status**: composable
- **classes**:
  - **`DabDetrFrozenBatchNorm2d`** [compute]: `L1/frozen_batch_norm2d.py`
  - **`DabDetrConvEncoder`** [wiring]: wires `load_backbone(config)` + replaces BN with frozen BN
  - **`DabDetrConvModel`** [wiring]: wires `DabDetrConvEncoder`, `DabDetrSinePositionEmbedding`
  - **`DabDetrSinePositionEmbedding`** [compute]: 2D sine positions on cumsum(pixel_mask) (no kb-nano L1 — pure math)
  - **`DetrAttention`** [compute]: `L1/linear.py (×4) + L1/dense_attention.py` (DETR self-attn with pos added to q/k; non-causal)
  - **`DabDetrAttention`** [compute]: `L1/dense_attention.py + L1/linear.py` (cross-attn with externally-provided q/k/v projections; head_dim differs for v)
  - **`DabDetrDecoderLayerSelfAttention`** [compute]: `L1/linear.py (×5 q-content/q-pos/k-content/k-pos/v) + DabDetrAttention + L1/layer_norm.py + residual`
  - **`DabDetrDecoderLayerCrossAttention`** [compute]: `L1/linear.py (×6) + DabDetrAttention + L1/layer_norm.py + residual` (DAB-DETR conditional cross-attn with sine query embed concat)
  - **`DabDetrDecoderLayerFFN`** [compute]: `L1/linear.py + ACT2FN[prelu] + L1/linear.py + L1/layer_norm.py + residual` (no exact L2 match; activation=PReLU)
  - **`DabDetrEncoderLayer`** [compute]: `DetrAttention + L1/linear.py (×2) + L1/layer_norm.py (×2) + ACT2FN[prelu]` (no exact L2 match)
  - **`DabDetrDecoderLayer`** [wiring]: wires `DabDetrDecoderLayerSelfAttention`, `DabDetrDecoderLayerCrossAttention`, `DabDetrDecoderLayerFFN`
  - **`DabDetrMLP`** [compute]: `L1/linear.py (×num_layers) + L1/relu.py` (DETR-style box-coord head)
  - **`DabDetrEncoder`** [wiring]: wires `DabDetrMLP` (query_scale), `DabDetrEncoderLayer (×N)`; optional `L1/layer_norm.py`
  - **`DabDetrDecoder`** [wiring]: wires `DabDetrDecoderLayer (×N)`, `DabDetrMLP` (query_scale, ref_anchor_head, bbox_embed)
  - **`DabDetrModel`** [wiring]: wires `DabDetrConvEncoder`, `DabDetrSinePositionEmbedding`, `DabDetrEncoder`, `DabDetrDecoder`; direct `L1/conv2d.py + L1/embedding.py`
  - **`DabDetrMHAttentionMap`** [compute]: `L1/linear.py + L1/conv2d.py + L1/softmax.py` (1×1 conv used as linear on 2D map; segmentation aux head — unused at inference for object-detection model)
- **task heads (1)**: ForObjectDetection — base + linear (per-task)

## dac
- **src**: modeling_dac.py
- **hidden_act**: n/a (uses `Snake1d` activation)
- **status**: partial (Snake1d activation has no L1 match — must decompose)
- **classes**:
  - **`Snake1d`** [compute]: `x + (1/(alpha+eps)) * sin(alpha*x)^2` — no kb-nano kernel for snake; decomposes to elementwise sin/mul/add (no L1 match)
  - **`DacVectorQuantize`** [compute]: `L1/conv1d.py (in_proj 1×1) + L1/embedding.py (codebook) + L1/conv1d.py (out_proj 1×1) + L2-normalize + euclidean dist + STE` (no exact L2 match — VQ-VAE quantizer)
  - **`DacResidualUnit`** [wiring]: wires `Snake1d (×2)`, `nn.Conv1d (×2)` with dilation; residual + center crop
  - **`DacEncoderBlock`** [wiring]: wires `DacResidualUnit (×3)`, `Snake1d`; direct `L1/conv1d.py` (downsample)
  - **`DacDecoderBlock`** [wiring]: wires `DacResidualUnit (×3)`, `Snake1d`; direct `L1/conv_transpose1d.py` (upsample)
  - **`DacResidualVectorQuantizer`** [wiring]: wires `DacVectorQuantize (×n_codebooks)` (residual loop)
  - **`DacDecoder`** [wiring]: wires `DacDecoderBlock (×len(strides))`, `Snake1d`; direct `L1/conv1d.py (×2) + L1/tanh.py`
  - **`DacEncoder`** [wiring]: wires `DacEncoderBlock (×len(strides))`, `Snake1d`; direct `L1/conv1d.py (×2)`
  - **`DacModel`** [wiring]: wires `DacEncoder`, `DacDecoder`, `DacResidualVectorQuantizer`
- **task heads (0)**: — (DAC is an audio codec; only the base `DacModel` and a `DacForX`-free encode/decode forward path)


## data2vec_audio
- **src**: modeling_data2vec_audio.py (and modular_data2vec_audio.py — inherits from Wav2Vec2)
- **hidden_act**: gelu (also `feat_extract_activation`=gelu)
- **status**: composable
- **classes**:
  - **`Data2VecAudioConvLayer`** [compute]: `L1/conv1d.py + L1/layer_norm.py + L1/gelu.py`
  - **`Data2VecAudioPadLayer`** [compute, inherits `Wav2Vec2SamePadLayer`]: trim 1 sample on right (no kb-nano kernel)
  - **`Data2VecAudioPositionalConvLayer`** [compute]: `L1/conv1d.py (depthwise) + Data2VecAudioPadLayer + L1/layer_norm.py (no affine) + L1/gelu.py`
  - **`Data2VecAudioPositionalConvEmbedding`** [wiring]: wires `Data2VecAudioPositionalConvLayer (×num_conv_pos_embeddings)`
  - **`Data2VecAudioFeatureEncoder`** [wiring, inherits `Wav2Vec2FeatureEncoder`]: wires `Data2VecAudioConvLayer (×num_feat_extract_layers)`
  - **`Data2VecAudioFeatureProjection`** [compute, inherits `Wav2Vec2FeatureProjection`]: `L1/layer_norm.py + L1/linear.py`
  - **`Data2VecAudioAttention`** [compute]: `L1/linear.py (×4 q/k/v/out) + L1/dense_attention.py` (encoder bidirectional self-attention)
  - **`Data2VecAudioFeedForward`** [compute]: `L1/linear.py + L1/gelu.py + L1/linear.py` (no exact L2 match — Wav2Vec2-style FFN)
  - **`Data2VecAudioEncoderLayer`** [wiring]: wires `Data2VecAudioAttention`, `Data2VecAudioFeedForward`; direct `L1/layer_norm.py (×2)`
  - **`Data2VecAudioEncoder`** [wiring, inherits `Wav2Vec2Encoder`]: wires `Data2VecAudioPositionalConvEmbedding`, `Data2VecAudioEncoderLayer (×N)`; direct `L1/layer_norm.py`
  - **`Data2VecAudioAdapterLayer`** [compute]: `L1/conv1d.py + GLU` (GLU has no direct L1 — uses `nn.functional.glu`)
  - **`Data2VecAudioAdapter`** [wiring, inherits `Wav2Vec2Adapter`]: wires `Data2VecAudioAdapterLayer (×N)`; optional `L1/linear.py + L1/layer_norm.py`
  - **`Data2VecAudioModel`** [wiring, inherits `Wav2Vec2Model`]: wires `Data2VecAudioFeatureEncoder`, `Data2VecAudioFeatureProjection`, `Data2VecAudioEncoder`, optional `Data2VecAudioAdapter`
  - **`AMSoftmaxLoss`** [compute]: training-only (loss head; not part of inference)
  - **`TDNNLayer`** [compute]: `L1/linear.py + L1/relu.py` (uses F.conv1d trick on Linear weight at runtime)
- **task heads (4)**: ForCTC, ForSequenceClassification, ForAudioFrameClassification, ForXVector — base + linear (per-task)

## data2vec_text
- **src**: modeling_data2vec_text.py (and modular_data2vec_text.py — inherits from Roberta)
- **hidden_act**: gelu
- **status**: composable
- **classes**:
  - **`Data2VecTextEmbeddings`** [compute, inherits `RobertaEmbeddings`]: `L2/encoder_embeddings.py` (word + position + token_type + LayerNorm)
  - **`Data2VecTextSelfAttention`** [compute, inherits `RobertaSelfAttention`]: `L2/encoder_attention.py` (q/k/v + dispatch via ALL_ATTENTION_FUNCTIONS)
  - **`Data2VecTextCrossAttention`** [compute]: `L2/encoder_attention.py` (cross-attn variant; q from hidden, k/v from encoder_hidden_states)
  - **`Data2VecTextSelfOutput`** [compute]: `L2/encoder_attention.py` (dense + LayerNorm + residual)
  - **`Data2VecTextAttention`** [wiring]: wires `Data2VecTextSelfAttention` (or `Data2VecTextCrossAttention`), `Data2VecTextSelfOutput`
  - **`Data2VecTextIntermediate`** [compute]: `L1/linear.py + L1/gelu.py`
  - **`Data2VecTextOutput`** [compute]: `L1/linear.py + L1/layer_norm.py` + residual
  - **`Data2VecTextLayer`** [wiring, inherits `RobertaLayer`]: wires `Data2VecTextAttention`, optional `Data2VecTextAttention(is_cross_attention=True)`, `Data2VecTextIntermediate`, `Data2VecTextOutput`
  - **`Data2VecTextEncoder`** [wiring]: wires `Data2VecTextLayer (×N)`
  - **`Data2VecTextPooler`** [compute]: `L1/linear.py + L1/tanh.py`
  - **`Data2VecTextModel`** [wiring, inherits `RobertaModel`]: wires `Data2VecTextEmbeddings`, `Data2VecTextEncoder`, optional `Data2VecTextPooler`
  - **`Data2VecTextLMHead`** [compute, inherits `RobertaLMHead`]: `L1/linear.py + L1/gelu.py + L1/layer_norm.py + L1/linear.py` (decoder)
  - **`Data2VecTextClassificationHead`** [compute, inherits `RobertaClassificationHead`]: `L1/linear.py + L1/tanh.py + L1/linear.py` (training-time head; skipped per task-head rule)
  - **`Data2VecTextForCausalLM`** [wiring]: wires `Data2VecTextModel`, `Data2VecTextLMHead`
  - **`Data2VecTextForMaskedLM`** [wiring]: wires `Data2VecTextModel`, `Data2VecTextLMHead`
- **task heads (5)**: ForSequenceClassification, ForMultipleChoice, ForTokenClassification, ForQuestionAnswering, ForCausalLM/ForMaskedLM kept above — base + linear (per-task)

## data2vec_vision
- **src**: modeling_data2vec_vision.py (no modular file)
- **hidden_act**: gelu
- **status**: composable (BEiT-derived ViT with relative-position bias)
- **classes**:
  - **`Data2VecVisionDropPath`** [compute]: stochastic depth (no kb-nano kernel; identity at inference)
  - **`Data2VecVisionEmbeddings`** [compute]: `L1/conv2d.py (via Data2VecVisionPatchEmbeddings) + L1/embedding.py` (cls_token, mask_token, position_embeddings as Parameters); optional bicubic interpolation of pos enc
  - **`Data2VecVisionPatchEmbeddings`** [compute]: `L1/conv2d.py` (patch_size kernel, no LayerNorm)
  - **`Data2VecVisionSelfAttention`** [compute]: `L1/linear.py (×3 q/k/v) + L1/dense_attention.py + Data2VecVisionRelativePositionBias` (eager fallback; key_proj has bias=False)
  - **`Data2VecVisionSdpaSelfAttention`** [compute, inherits `Data2VecVisionSelfAttention`]: `L1/linear.py (×3) + L1/sdpa.py` (calls `F.scaled_dot_product_attention`)
  - **`Data2VecVisionSelfOutput`** [compute]: `L1/linear.py` (residual is in Layer)
  - **`Data2VecVisionAttention`** [wiring]: wires `Data2VecVisionSelfAttention` or `Data2VecVisionSdpaSelfAttention`, `Data2VecVisionSelfOutput`
  - **`Data2VecVisionIntermediate`** [compute]: `L1/linear.py + L1/gelu.py`
  - **`Data2VecVisionOutput`** [compute]: `L1/linear.py`
  - **`Data2VecVisionLayer`** [wiring]: wires `Data2VecVisionAttention`, `Data2VecVisionIntermediate`, `Data2VecVisionOutput`, `Data2VecVisionDropPath`; direct `L1/layer_norm.py (×2)` + lambda_1/lambda_2 layer-scale Parameters
  - **`Data2VecVisionRelativePositionBias`** [compute]: relative-position bias table (Parameter) + index gather + bilinear interpolation (no kb-nano L1 — pure index/lookup; closest is custom)
  - **`Data2VecVisionEncoder`** [wiring]: wires `Data2VecVisionLayer (×N)`, optional `Data2VecVisionRelativePositionBias`
  - **`Data2VecVisionModel`** [wiring]: wires `Data2VecVisionEmbeddings`, `Data2VecVisionEncoder`, `Data2VecVisionPooler` (optional); direct `L1/layer_norm.py` (or Identity)
  - **`Data2VecVisionPooler`** [compute]: optional `L1/layer_norm.py` over patch-token mean (or [CLS] only)
  - **`Data2VecVisionConvModule`** [compute]: `L1/conv2d.py + L1/batch_norm2d.py + L1/relu.py` (used in segmentation heads)
  - **`Data2VecVisionPyramidPoolingBlock`** [wiring]: wires `nn.AdaptiveAvgPool2d`, `Data2VecVisionConvModule`; uses `L1/adaptive_avg_pool2d.py`
  - **`Data2VecVisionPyramidPoolingModule`** [wiring]: wires `Data2VecVisionPyramidPoolingBlock (×len(pool_scales))` + bilinear interpolate
  - **`Data2VecVisionUperHead`** [wiring]: wires `Data2VecVisionPyramidPoolingModule`, `Data2VecVisionConvModule (×many)`; direct `L1/conv2d.py` (classifier 1×1)
  - **`Data2VecVisionFCNHead`** [wiring]: wires `Data2VecVisionConvModule (×num_convs)`; direct `L1/conv2d.py` (classifier 1×1)
- **task heads (2)**: ForImageClassification, ForSemanticSegmentation — base + linear (per-task)

## dbrx
- **src**: modeling_dbrx.py (no modular file)
- **hidden_act**: silu (ffn_act_fn defaults to {"name": "silu"})
- **status**: composable
- **classes**:
  - **`DbrxRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (default Llama-style RoPE)
  - **`DbrxAttention`** [compute]: `L2/attention.py` (decoder causal w/ RoPE + KV cache + GQA; uses fused `Wqkv` Linear + qkv-clamp + dispatch via ALL_ATTENTION_FUNCTIONS) — note: kb-nano `L2/attention.py` typically uses split q/k/v projections, not fused; mapping is structurally equivalent
  - **`DbrxExpertGLU`** [compute]: `L1/silu.py + L1/linear.py` (gated GLU expert via raw matmul on flat weight Parameters; non-standard)
  - **`DbrxExperts`** [compute]: per-token routed dispatch over experts (no exact L2 — uses `expert_mask` permute + `index_add_`); closest analogue is custom MoE; not `L1/moe_grouped_gemm.py` (no grouped GEMM)
  - **`DbrxRouter`** [compute]: `L1/linear.py` + softmax (training-time jitter)
  - **`DbrxFFN`** [wiring]: wires `DbrxRouter`, `DbrxExperts`; direct `L1/softmax.py` + `L1/top_k_per_row.py` (or topk)
  - **`DbrxNormAttentionNorm`** [wiring]: wires `DbrxAttention`; direct `L1/layer_norm.py (×2 norm_1/norm_2, no bias)`
  - **`DbrxBlock`** [wiring]: wires `DbrxNormAttentionNorm`, `DbrxFFN`
  - **`DbrxModel`** [wiring]: wires `DbrxRotaryEmbedding`, `DbrxBlock (×n_layers)`; direct `L1/embedding.py + L1/layer_norm.py (no bias)`
  - **`DbrxForCausalLM`** [wiring]: wires `DbrxModel`; direct `L1/linear.py` (lm_head)

## deberta
- **src**: modeling_deberta.py (no modular file)
- **hidden_act**: gelu
- **status**: partial (DisentangledSelfAttention has no kb-nano kernel — relative-position bias with c2p/p2c gathers)
- **classes**:
  - **`DebertaLayerNorm`** [compute]: custom LayerNorm (eps inside sqrt) — closest match `L1/layer_norm.py` but math differs slightly (mean computed in float32)
  - **`DebertaSelfOutput`** [compute]: `L1/linear.py + DebertaLayerNorm` (residual added before LayerNorm)
  - **`DisentangledSelfAttention`** [compute]: fused `in_proj` (×3 q/k/v) + q/v bias Parameters + relative-position c2p/p2c gather + softmax + matmul (no exact L2 match — DeBERTa-specific disentangled attention)
  - **`DebertaEmbeddings`** [compute]: `L1/embedding.py (×3 word/position/token_type) + DebertaLayerNorm` + optional `L1/linear.py` (embed_proj)
  - **`DebertaAttention`** [wiring]: wires `DisentangledSelfAttention`, `DebertaSelfOutput`
  - **`DebertaIntermediate`** [compute]: `L1/linear.py + L1/gelu.py`
  - **`DebertaOutput`** [compute]: `L1/linear.py + DebertaLayerNorm` (residual added before LayerNorm)
  - **`DebertaLayer`** [wiring]: wires `DebertaAttention`, `DebertaIntermediate`, `DebertaOutput`
  - **`DebertaEncoder`** [wiring]: wires `DebertaLayer (×N)`; optional `L1/embedding.py` (rel_embeddings)
  - **`DebertaModel`** [wiring]: wires `DebertaEmbeddings`, `DebertaEncoder`
  - **`LegacyDebertaPredictionHeadTransform`** [compute]: `L1/linear.py + L1/gelu.py + L1/layer_norm.py`
  - **`LegacyDebertaLMPredictionHead`** [wiring]: wires `LegacyDebertaPredictionHeadTransform`; direct `L1/linear.py` (decoder)
  - **`LegacyDebertaOnlyMLMHead`** [wiring]: wires `LegacyDebertaLMPredictionHead`
  - **`DebertaLMPredictionHead`** [compute]: `L1/linear.py + L1/gelu.py + L1/layer_norm.py + matmul(word_emb.T)+bias` (tied to embedding)
  - **`DebertaOnlyMLMHead`** [wiring]: wires `DebertaLMPredictionHead`
  - **`DebertaForMaskedLM`** [wiring]: wires `DebertaModel`, (Legacy)`DebertaOnlyMLMHead`
  - **`ContextPooler`** [compute]: `L1/linear.py + ACT2FN[pooler_hidden_act]` (gelu)
- **task heads (3)**: ForSequenceClassification, ForTokenClassification, ForQuestionAnswering — base + linear (per-task)

## deberta_v2
- **src**: modeling_deberta_v2.py (no modular file)
- **hidden_act**: gelu
- **status**: partial (DisentangledSelfAttention with bucketed relative position has no kb-nano kernel)
- **classes**:
  - **`DebertaV2SelfOutput`** [compute]: `L1/linear.py + L1/layer_norm.py` (residual then LayerNorm; uses standard `nn.LayerNorm`)
  - **`DisentangledSelfAttention`** [compute]: separate q_proj/k_proj/v_proj + bucketed log-position c2p/p2c bias (no exact L2 match — DeBERTa v2 disentangled attention)
  - **`DebertaV2Attention`** [wiring]: wires `DisentangledSelfAttention`, `DebertaV2SelfOutput`
  - **`DebertaV2Intermediate`** [compute]: `L1/linear.py + L1/gelu.py`
  - **`DebertaV2Output`** [compute]: `L1/linear.py + L1/layer_norm.py` (residual then LayerNorm)
  - **`DebertaV2Layer`** [wiring]: wires `DebertaV2Attention`, `DebertaV2Intermediate`, `DebertaV2Output`
  - **`ConvLayer`** [compute]: `L1/conv1d.py + L1/layer_norm.py + ACT2FN[conv_act] (default tanh)` (optional 1D conv mixing layer)
  - **`DebertaV2Embeddings`** [compute]: `L1/embedding.py (×3 word/position/token_type) + L1/layer_norm.py` + optional `L1/linear.py` (embed_proj)
  - **`DebertaV2Encoder`** [wiring]: wires `DebertaV2Layer (×N)`, optional `ConvLayer`; direct `L1/embedding.py` (rel_embeddings) and optional `L1/layer_norm.py`
  - **`DebertaV2Model`** [wiring]: wires `DebertaV2Embeddings`, `DebertaV2Encoder`
  - **`LegacyDebertaV2PredictionHeadTransform`** [compute]: `L1/linear.py + L1/gelu.py + L1/layer_norm.py`
  - **`LegacyDebertaV2LMPredictionHead`** [wiring]: wires `LegacyDebertaV2PredictionHeadTransform`; direct `L1/linear.py`
  - **`LegacyDebertaV2OnlyMLMHead`** [wiring]: wires `LegacyDebertaV2LMPredictionHead`
  - **`DebertaV2LMPredictionHead`** [compute]: `L1/linear.py + L1/gelu.py + L1/layer_norm.py + matmul(word_emb.T)+bias`
  - **`DebertaV2OnlyMLMHead`** [wiring]: wires `DebertaV2LMPredictionHead`
  - **`DebertaV2ForMaskedLM`** [wiring]: wires `DebertaV2Model`, (Legacy)`DebertaV2OnlyMLMHead`
  - **`ContextPooler`** [compute]: `L1/linear.py + ACT2FN[pooler_hidden_act]` (gelu)
- **task heads (4)**: ForSequenceClassification, ForTokenClassification, ForQuestionAnswering, ForMultipleChoice — base + linear (per-task)

## decision_transformer
- **src**: modeling_decision_transformer.py (no modular file)
- **hidden_act**: relu (activation_function default in DecisionTransformerConfig)
- **status**: composable (GPT-2 backbone + linear input/output projections)
- **classes**:
  - **`DecisionTransformerGPT2Attention`** [compute]: GPT-2 attention with `Conv1D` (which is a fused linear) for c_attn/c_proj/q_attn; supports causal + cross-attn — closest L2 is `L2/attention.py` (decoder causal); no exact match for Conv1D-style fused projection
  - **`DecisionTransformerGPT2MLP`** [compute]: `Conv1D + L1/relu.py + Conv1D` (Conv1D = transposed Linear) — no exact L2 match; activation=relu
  - **`DecisionTransformerGPT2Block`** [wiring]: wires `DecisionTransformerGPT2Attention`, `DecisionTransformerGPT2MLP`, optional cross-attention; direct `L1/layer_norm.py (×2-3)`
  - **`DecisionTransformerGPT2Model`** [wiring]: wires `DecisionTransformerGPT2Block (×N)`; direct `L1/embedding.py (×2 wte/wpe) + L1/layer_norm.py` (ln_f)
  - **`DecisionTransformerModel`** [wiring]: wires `DecisionTransformerGPT2Model`; direct `L1/embedding.py` (timestep), `L1/linear.py` (×4 embed_return/state/action, predict_state, predict_return), `L1/linear.py + L1/tanh.py` (predict_action sequential), `L1/layer_norm.py` (embed_ln)
- **task heads (0)**: — (no separate `For*` heads; `DecisionTransformerModel` itself is the inference target with state/action/return prediction heads built in)

## deepseek_v2
- **src**: modeling_deepseek_v2.py (and modular_deepseek_v2.py — inherits from Llama/Qwen2-MoE)
- **hidden_act**: silu
- **status**: composable (MLA attention + DeepseekMoE shared-expert)
- **classes**:
  - **`DeepseekV2Experts`** [compute, inherits `Qwen2MoeExperts`]: per-token routed dispatch with stacked 3D weight Parameters (no exact L2 — closest is `L1/moe_grouped_gemm.py` family but uses python loop over expert hits)
  - **`DeepseekV2Moe`** [wiring]: wires `DeepseekV2Experts`, `DeepseekV2MLP` (shared), `nn.Linear` gate; closest L2 is `L2/shared_expert_moe.py` + `L2/deepseek_moe.py`
  - **`DeepseekV2MLP`** [compute, inherits `LlamaMLP`]: `L2/llama_mlp.py` (gate/up/down + silu)
  - **`DeepseekV2RMSNorm`** [compute, inherits `LlamaRMSNorm`]: `L1/rms_norm.py`
  - **`DeepseekV2RotaryEmbedding`** [compute, inherits `LlamaRotaryEmbedding`]: `L1/rotary_emb.py` (returns complex `freqs_cis`)
  - **`DeepseekV2Attention`** [compute]: `L2/deepseek_mla_attention.py` (MLA with q-LoRA, kv-LoRA, separate qk_nope/qk_rope/v head dims)
  - **`DeepseekV2DecoderLayer`** [wiring, inherits `LlamaDecoderLayer`]: wires `DeepseekV2Attention`, `DeepseekV2Moe` (or `DeepseekV2MLP` for first_k_dense), `DeepseekV2RMSNorm` (×2)
  - **`DeepseekV2Model`** [wiring, inherits `LlamaModel`]: wires `DeepseekV2DecoderLayer (×N)`, `DeepseekV2RMSNorm`, `DeepseekV2RotaryEmbedding`; direct `L1/embedding.py`
  - **`DeepseekV2ForCausalLM`** [wiring, inherits `LlamaForCausalLM`]: wires `DeepseekV2Model`; direct `L1/linear.py` (lm_head)
- **task heads (1)**: ForSequenceClassification — base + linear (per-task)

## deepseek_v3
- **src**: modeling_deepseek_v3.py (and modular_deepseek_v3.py — inherits from Llama / Qwen3-MoE)
- **hidden_act**: silu
- **status**: composable (MLA + sigmoid-routed grouped MoE with shared experts)
- **classes**:
  - **`DeepseekV3RMSNorm`** [compute]: `L1/rms_norm.py`
  - **`DeepseekV3RotaryEmbedding`** [compute]: `L1/rotary_emb.py` (interleaved variant possible via `apply_rotary_pos_emb_interleave`)
  - **`DeepseekV3MLP`** [compute]: `L2/llama_mlp.py` (gate/up/down + silu)
  - **`DeepseekV3TopkRouter`** [compute]: `L1/linear.py` (router via raw F.linear on Parameter weight; sigmoid scoring + e_score_correction_bias)
  - **`DeepseekV3NaiveMoe`** [compute]: per-token routed dispatch with stacked 3D weight Parameters (no exact L2 — closest is `L1/moe_grouped_gemm.py` family)
  - **`DeepseekV3MoE`** [wiring]: wires `DeepseekV3NaiveMoe`, `DeepseekV3TopkRouter`, `DeepseekV3MLP` (shared); closest L2 is `L2/shared_expert_moe.py` + `L2/deepseek_moe.py`
  - **`DeepseekV3Attention`** [compute]: `L2/deepseek_mla_attention.py` (same MLA as v2)
  - **`DeepseekV3DecoderLayer`** [wiring]: wires `DeepseekV3Attention`, `DeepseekV3MoE` (or `DeepseekV3MLP` first_k_dense), `DeepseekV3RMSNorm` (×2)
  - **`DeepseekV3Model`** [wiring]: wires `DeepseekV3DecoderLayer (×N)`, `DeepseekV3RMSNorm`, `DeepseekV3RotaryEmbedding`; direct `L1/embedding.py`
  - **`DeepseekV3ForCausalLM`** [wiring]: wires `DeepseekV3Model`; direct `L1/linear.py` (lm_head)
- **task heads (2)**: ForSequenceClassification, ForTokenClassification — base + linear (per-task)

## deepseek_v4
- **src**: modeling_deepseek_v4.py (and modular_deepseek_v4.py)
- **hidden_act**: silu
- **status**: partial (V4 introduces HCA/CSA compressed attention, hyper-connections, hash-routed MoE; many pieces have no kb-nano kernel)
- **classes**:
  - **`DeepseekV4RMSNorm`** [compute]: `L1/rms_norm.py`
  - **`DeepseekV4UnweightedRMSNorm`** [compute]: `L1/rms_norm.py` (no weight Parameter — pure normalization)
  - **`DeepseekV4RotaryEmbedding`** [compute]: per-layer-type partial RoPE with interleaved pairing — closest `L1/yarn_rotary_emb.py` family or custom; `L1/rotary_emb.py` doesn't cover partial
  - **`DeepseekV4HCACache`** / **`DeepseekV4CSACache`**: cache classes (skipped per `*Cache` rule)
  - **`DeepseekV4GroupedLinear`** [compute]: block-diagonal grouped linear via bmm (no exact L1 match — closest is `L1/bmm.py`)
  - **`DeepseekV4HCACompressor`** [compute]: long-range KV compressor (paper §2.3.2; softmax-aggregated windows). No kb-nano L2 match
  - **`DeepseekV4Indexer`** [compute]: per-window index head for CSA. No kb-nano kernel
  - **`DeepseekV4CSACompressor`** [compute]: short-range overlapping CSA compressor (paper §2.3.1). No kb-nano L2 match
  - **`DeepseekV4Attention`** [compute]: V4 shared-KV MQA + partial-RoPE + per-head sinks + grouped output projection + optional HCA/CSA compressor concat (no exact L2 match — closest L2 is `L2/deepseek_mla_attention.py` but V4 differs; eager-only)
  - **`DeepseekV4HyperConnection`** [compute]: manifold-constrained hyper-connections via Sinkhorn-Knopp projection (paper §2.2). No kb-nano kernel
  - **`DeepseekV4HyperHead`** [compute]: final HC-stream collapse. No kb-nano kernel
  - **`DeepseekV4MLP`** [compute]: `L2/llama_mlp.py` (gate/up/down + silu)
  - **`DeepseekV4Experts`** [compute]: routed experts with SwiGLU clamping + stacked 3D weight Parameters (no exact L2 match — closest is `L1/moe_grouped_gemm.py` family)
  - **`DeepseekV4TopKRouter`** [compute]: `L1/linear.py` + score_fn + topk
  - **`DeepseekV4HashRouter`** [compute]: `L1/linear.py` + frozen `tid2eid` lookup table (no kb-nano kernel for hash routing)
  - **`DeepseekV4SparseMoeBlock`** [wiring]: wires `DeepseekV4TopKRouter` or `DeepseekV4HashRouter`, `DeepseekV4Experts`, `DeepseekV4MLP` (shared)
  - **`DeepseekV4DecoderLayer`** [wiring]: wires `DeepseekV4Attention`, `DeepseekV4SparseMoeBlock`, `DeepseekV4HyperConnection (×2)`, `DeepseekV4RMSNorm` (×2)
  - **`DeepseekV4Model`** [wiring]: wires `DeepseekV4DecoderLayer (×N)`, `DeepseekV4RMSNorm`, `DeepseekV4RotaryEmbedding`, `DeepseekV4HyperHead`; direct `L1/embedding.py`
  - **`DeepseekV4ForCausalLM`** [wiring]: wires `DeepseekV4Model`; direct `L1/linear.py` (lm_head)
- **task heads (0)**: — (no ForX heads beyond ForCausalLM kept above)

## deepseek_vl
- **src**: modeling_deepseek_vl.py (and modular_deepseek_vl.py)
- **hidden_act**: gelu (in DeepseekVLAligner, hardcoded `nn.GELU()`); text_config inherits hidden_act (silu for Llama backbone); vision_config gelu
- **status**: composable (vision-language model wiring SigLIP-style vision + Llama text)
- **classes**:
  - **`DeepseekVLAligner`** [compute]: `L1/linear.py + L1/gelu.py + L1/linear.py` (no exact L2 match — small 2-layer MLP aligner)
  - **`DeepseekVLModel`** [wiring]: wires `vision_model` (AutoModel via vision_config), `DeepseekVLAligner`, `language_model` (AutoModel via text_config)
  - **`DeepseekVLForConditionalGeneration`** [wiring]: wires `DeepseekVLModel`; direct `L1/linear.py` (lm_head)

## deepseek_vl_hybrid
- **src**: modeling_deepseek_vl_hybrid.py (and modular_deepseek_vl_hybrid.py)
- **hidden_act**: gelu (in DeepseekVLHybridAligner, hardcoded `nn.GELU()`); text_config silu, vision_config(s) gelu
- **status**: composable (dual SigLIP + SAM vision + Llama text)
- **classes**:
  - **`DeepseekVLHybridLayerNorm`** [compute, inherits `nn.LayerNorm`]: `L1/layer_norm.py` with channels_first/last permute
  - **`DeepseekVLSamVisionNeck`** [compute]: `L1/conv2d.py + DeepseekVLHybridLayerNorm + L1/conv2d.py + DeepseekVLHybridLayerNorm`
  - **`DeepseekVLSamVisionProj`** [compute]: bilinear interpolate + `L1/conv2d.py (×2 stride-2)`
  - **`DeepseekVLHybridAligner`** [compute]: `L1/linear.py (×3 vision_proj/high_res_vision_proj/proj) + L1/gelu.py + concat`
  - **`DeepseekVLHybridModel`** [wiring]: wires `vision_model`, `high_res_vision_model` (SAM-style), `DeepseekVLSamVisionNeck`, `DeepseekVLSamVisionProj`, `DeepseekVLHybridAligner`, `language_model`; direct `nn.Parameter` (`high_res_vision_alpha`)
  - **`DeepseekVLHybridForConditionalGeneration`** [wiring]: wires `DeepseekVLHybridModel`; direct `L1/linear.py` (lm_head)

## deformable_detr
- **src**: modeling_deformable_detr.py (and modular_deformable_detr.py)
- **hidden_act**: relu (activation_function default)
- **status**: composable
- **classes**:
  - **`MultiScaleDeformableAttention`** [compute]: low-level CUDA-kernel-backed multi-scale deformable attention (closest `L1/rtdetrv2_deformable_attention.py`)
  - **`DeformableDetrFrozenBatchNorm2d`** [compute]: `L1/frozen_batch_norm2d.py`
  - **`DeformableDetrConvEncoder`** [wiring]: wires `load_backbone(config)` + replaces BN with frozen BN
  - **`DeformableDetrSinePositionEmbedding`** [compute]: 2D sine positions (no kb-nano L1 — pure math)
  - **`DeformableDetrLearnedPositionEmbedding`** [compute]: `L1/embedding.py (×2 row/col)` + concat
  - **`DeformableDetrSelfAttention`** [compute]: `L1/linear.py (×4 q/k/v/o) + L1/dense_attention.py` (DETR-style: pos added to q/k)
  - **`DeformableDetrMultiscaleDeformableAttention`** [compute]: `L1/rtdetrv2_deformable_attention.py` (closest match — multi-scale deformable attn with `MultiScaleDeformableAttention` core)
  - **`DeformableDetrMLP`** [compute]: `L1/linear.py + L1/relu.py + L1/linear.py` (no exact L2 match)
  - **`DeformableDetrEncoderLayer`** [wiring]: wires `DeformableDetrMultiscaleDeformableAttention`, `DeformableDetrMLP`; direct `L1/layer_norm.py (×2)`
  - **`DeformableDetrDecoderLayer`** [wiring]: wires `DeformableDetrSelfAttention`, `DeformableDetrMultiscaleDeformableAttention`, `DeformableDetrMLP`; direct `L1/layer_norm.py (×3)`
  - **`DeformableDetrEncoder`** [wiring]: wires `DeformableDetrEncoderLayer (×N)`
  - **`DeformableDetrDecoder`** [wiring]: wires `DeformableDetrDecoderLayer (×N)`; bbox_embed/class_embed heads
  - **`DeformableDetrModel`** [wiring]: wires `DeformableDetrConvEncoder`, `DeformableDetrSinePositionEmbedding`, `DeformableDetrEncoder`, `DeformableDetrDecoder`; direct `L1/conv2d.py + L1/linear.py + L1/layer_norm.py + L1/embedding.py`
  - **`DeformableDetrMLPPredictionHead`** [compute]: `L1/linear.py (×num_layers) + L1/relu.py`
- **task heads (1)**: ForObjectDetection — base + linear (per-task)

## deimv2
- **src**: modeling_deimv2.py (and modular_deimv2.py — heavy inheritance from D-Fine + Llama)
- **hidden_act**: encoder_activation_function=gelu, activation_function=silu, decoder_activation_function=relu
- **status**: composable
- **classes**:
  - **`Deimv2RMSNorm`** [compute, inherits `LlamaRMSNorm`]: `L1/rms_norm.py`
  - **`Deimv2SwiGLUFFN`** [compute, inherits `LlamaMLP`]: `L2/llama_mlp.py`
  - **`Deimv2Gate`** [compute, inherits `DFineGate`]: `L1/linear.py + L1/sigmoid.py + L1/layer_norm.py` (gate fusion)
  - **`Deimv2MLP`** [compute, inherits `DFineMLP`]: same as `DFineMLP`
  - **`Deimv2MultiscaleDeformableAttention`** [compute, inherits `DFineMultiscaleDeformableAttention`]: `L1/rtdetrv2_deformable_attention.py`
  - **`Deimv2ConvNormLayer`** [compute, inherits `DFineConvNormLayer`]: `L2/rtdetrv2_conv_norm.py`
  - **`Deimv2RepVggBlock`** [compute, inherits `DFineRepVggBlock`]: `L2/rtdetrv2_repvgg_block.py`
  - **`Deimv2CSPRepLayer`** [compute]: `L2/rtdetrv2_csp_rep_layer.py`
  - **`Deimv2RepNCSPELAN5`** [compute]: composition of ConvNorm + CSPRep (no exact L2)
  - **`Deimv2SCDown`** [wiring, inherits `DFineSCDown`]: wires 2× ConvNormLayer
  - **`Deimv2SelfAttention`** [compute]: `L1/linear.py (×4) + L1/dense_attention.py`
  - **`Deimv2EncoderLayer`** [wiring, inherits `DFineEncoderLayer`]: same as DFineEncoderLayer
  - **`Deimv2SinePositionEmbedding`** [compute]: 2D sin/cos
  - **`Deimv2AIFILayer`** [wiring, inherits `DFineAIFILayer`]: same as DFineAIFILayer
  - **`Deimv2SpatialTuningAdapter`** [compute]: small Conv2d/Linear adapter (composition of L1)
  - **`Deimv2FrozenBatchNorm2d`** [compute]: `L1/frozen_batch_norm2d.py`
  - **`Deimv2ConvEncoder`** [wiring, inherits `DFineConvEncoder`]: load_backbone + frozen BN
  - **`Deimv2DINOv3ConvEncoder`** [wiring]: DINOv3 backbone wrapper
  - **`Deimv2Integral`** [compute, inherits `DFineIntegral`]: `L1/softmax.py + L1/linear.py`
  - **`Deimv2LQE`** [wiring, inherits `DFineLQE`]: wires `Deimv2MLP`
  - **`Deimv2DecoderLayer`** [wiring, inherits `DFineDecoderLayer`]: same as DFineDecoderLayer
  - **`Deimv2LiteEncoder`** [wiring]: hybrid encoder (lite variant)
  - **`Deimv2HybridEncoder`** [wiring, inherits `DFineHybridEncoder`]: same as DFineHybridEncoder
  - **`Deimv2Decoder`** [wiring, inherits `DFineDecoder`]: same as DFineDecoder
  - **`Deimv2Model`** [wiring, inherits `DFineModel`]: wires `Deimv2ConvEncoder` (or DINOv3), `Deimv2HybridEncoder` (or LiteEncoder), `Deimv2Decoder`
- **task heads (1)**: ForObjectDetection — base + linear (per-task)

## deit
- **src**: modeling_deit.py (no modular file)
- **hidden_act**: gelu
- **status**: composable
- **classes**:
  - **`DeiTEmbeddings`** [compute]: `L1/conv2d.py (via DeiTPatchEmbeddings) + L1/embedding.py` (cls_token, distillation_token, position_embeddings as Parameters); optional bicubic interpolation
  - **`DeiTPatchEmbeddings`** [compute]: `L1/conv2d.py` (patch_size kernel)
  - **`DeiTSelfAttention`** [compute]: `L1/linear.py (×3 q/k/v) + L1/dense_attention.py` (ViT-style; no relative position bias)
  - **`DeiTSelfOutput`** [compute]: `L1/linear.py` (residual is in Layer)
  - **`DeiTAttention`** [wiring]: wires `DeiTSelfAttention`, `DeiTSelfOutput`
  - **`DeiTIntermediate`** [compute]: `L1/linear.py + L1/gelu.py`
  - **`DeiTOutput`** [compute]: `L1/linear.py` (residual is in Layer)
  - **`DeiTLayer`** [wiring]: wires `DeiTAttention`, `DeiTIntermediate`, `DeiTOutput`; direct `L1/layer_norm.py (×2)` — closest L3 is `L3/vit_encoder_block.py`
  - **`DeiTEncoder`** [wiring]: wires `DeiTLayer (×N)`
  - **`DeiTPreTrainedModel`**: skipped per rule
  - **`DeiTModel`** [wiring]: wires `DeiTEmbeddings`, `DeiTEncoder`; direct `L1/layer_norm.py` + optional `DeiTPooler`
  - **`DeiTPooler`** [compute]: `L1/linear.py + L1/tanh.py`
- **task heads (3)**: ForMaskedImageModeling, ForImageClassification, ForImageClassificationWithTeacher — base + linear (per-task)

## depth_anything
- **src**: modeling_depth_anything.py (no modular file)
- **hidden_act**: n/a (uses ReLU/Identity directly in heads; backbone determines hidden_act)
- **status**: composable (DPT-style decoder on top of DINOv2-style backbone)
- **classes**:
  - **`DepthAnythingReassembleLayer`** [compute]: `L1/conv2d.py (1×1) + L1/conv_transpose2d.py (or L1/conv2d.py stride>1)` (resize + project)
  - **`DepthAnythingReassembleStage`** [wiring]: wires `DepthAnythingReassembleLayer (×len(neck_hidden_sizes))`
  - **`DepthAnythingPreActResidualLayer`** [compute]: `L1/relu.py + L1/conv2d.py + L1/relu.py + L1/conv2d.py` + residual (no exact L2 match)
  - **`DepthAnythingFeatureFusionLayer`** [wiring]: wires `DepthAnythingPreActResidualLayer (×2)`; direct `L1/conv2d.py` (1×1 projection) + bilinear interpolate
  - **`DepthAnythingFeatureFusionStage`** [wiring]: wires `DepthAnythingFeatureFusionLayer (×N)`
  - **`DepthAnythingNeck`** [wiring]: wires `DepthAnythingReassembleStage`, `DepthAnythingFeatureFusionStage`; direct `L1/conv2d.py` (×len(neck_hidden_sizes))
  - **`DepthAnythingDepthEstimationHead`** [compute]: `L1/conv2d.py (×3) + L1/relu.py + bilinear interpolate + L1/relu.py` (final dense depth map)
- **task heads (1)**: ForDepthEstimation — base + linear/conv (per-task)

## depth_pro
- **src**: modeling_depth_pro.py (no modular file)
- **hidden_act**: n/a (uses ReLU/GELU directly in heads; backbone hidden_act inherited from vision config)
- **status**: composable (multi-resolution DPT-style decoder + FOV head; uses DINOv2-style backbone)
- **classes**:
  - **`DepthProPatchEncoder`** [wiring]: wires backbone vision_model `AutoModel.from_config`; iterates patches at multiple scales
  - **`DepthProImageEncoder`** [wiring]: wires backbone image_model `AutoModel.from_config` (full-image branch)
  - **`DepthProEncoder`** [wiring]: wires `DepthProPatchEncoder`, `DepthProImageEncoder`
  - **`DepthProFeatureUpsampleBlock`** [compute]: `L1/conv2d.py + L1/conv_transpose2d.py (×N)` (upsample stack)
  - **`DepthProFeatureUpsample`** [wiring]: wires `DepthProFeatureUpsampleBlock (×many)` (per scale)
  - **`DepthProFeatureProjection`** [compute]: `L1/conv2d.py (×N 1×1 projections)` (per scale)
  - **`DepthProNeck`** [wiring]: wires `DepthProFeatureProjection`, optional `DepthProFeatureUpsample`
  - **`DepthProModel`** [wiring]: wires `DepthProEncoder`, `DepthProNeck`
  - **`DepthProPreActResidualLayer`** [compute]: same as DepthAnythingPreActResidualLayer
  - **`DepthProFeatureFusionLayer`** [wiring]: wires `DepthProPreActResidualLayer (×2)`; direct `L1/conv2d.py + L1/conv_transpose2d.py`
  - **`DepthProFeatureFusionStage`** [wiring]: wires `DepthProFeatureFusionLayer (×N)`
  - **`DepthProFovEncoder`** [wiring]: wires backbone vision_model
  - **`DepthProFovHead`** [compute]: `L1/conv2d.py (×N) + L1/relu.py + L1/linear.py` (FOV regression)
  - **`DepthProFovModel`** [wiring]: wires `DepthProFovEncoder`, `DepthProFovHead`
  - **`DepthProDepthEstimationHead`** [compute]: `L1/conv2d.py (×N) + L1/relu.py + L1/conv_transpose2d.py + L1/relu.py` (final dense depth map)
- **task heads (1)**: ForDepthEstimation — base + linear/conv (per-task)

## detr
- **src**: modeling_detr.py (no modular file)
- **hidden_act**: relu (activation_function default)
- **status**: composable
- **classes**:
  - **`DetrFrozenBatchNorm2d`** [compute]: `L1/frozen_batch_norm2d.py`
  - **`DetrConvEncoder`** [wiring]: wires `load_backbone(config)` + replaces BN with frozen BN
  - **`DetrSinePositionEmbedding`** [compute]: 2D sine positions (no kb-nano L1 — pure math)
  - **`DetrLearnedPositionEmbedding`** [compute]: `L1/embedding.py (×2 row/col) + concat`
  - **`DetrSelfAttention`** [compute]: `L1/linear.py (×4 q/k/v/o) + L1/dense_attention.py` (DETR-style; pos added to q/k)
  - **`DetrCrossAttention`** [compute]: `L1/linear.py (×4) + L1/dense_attention.py` (cross-attn with pos added to query and to key from encoder)
  - **`DetrMLP`** [compute]: `L1/linear.py + L1/relu.py + L1/linear.py` (no exact L2 match — DETR-style FFN)
  - **`DetrEncoderLayer`** [wiring]: wires `DetrSelfAttention`, `DetrMLP`; direct `L1/layer_norm.py (×2)`
  - **`DetrDecoderLayer`** [wiring]: wires `DetrSelfAttention`, `DetrCrossAttention`, `DetrMLP`; direct `L1/layer_norm.py (×3)`
  - **`DetrConvBlock`** [compute]: `L1/conv2d.py + L1/group_norm.py + L1/relu.py` (segmentation head residual)
  - **`DetrFPNFusionStage`** [wiring]: wires `DetrConvBlock (×N)`
  - **`DetrMaskHeadSmallConv`** [compute]: small Conv2d stack for segmentation mask head (composition of `L1/conv2d.py` + `L1/group_norm.py` + `L1/relu.py`)
  - **`DetrMHAttentionMap`** [compute]: `L1/linear.py + L1/conv2d.py + L1/softmax.py` (segmentation aux attention map)
  - **`DetrEncoder`** [wiring]: wires `DetrEncoderLayer (×N)`
  - **`DetrDecoder`** [wiring]: wires `DetrDecoderLayer (×N)`; direct `L1/layer_norm.py`
  - **`DetrModel`** [wiring]: wires `DetrConvEncoder`, `DetrSinePositionEmbedding` or `DetrLearnedPositionEmbedding`, `DetrEncoder`, `DetrDecoder`; direct `L1/conv2d.py + L1/embedding.py`
  - **`DetrMLPPredictionHead`** [compute]: `L1/linear.py (×num_layers) + L1/relu.py`
- **task heads (2)**: ForObjectDetection, ForSegmentation — base + linear/conv (per-task)

## dia
- **src**: modeling_dia.py (and modular_dia.py)
- **hidden_act**: silu
- **status**: composable (text-to-audio seq2seq with separate encoder/decoder configs)
- **classes**:
  - **`DiaMultiChannelEmbedding`** [compute]: `L1/embedding.py` + offset trick (multiple channels via single embedding lookup)
  - **`DiaMLP`** [compute]: `L2/llama_mlp.py`-style with fused `gate_up_proj` (one linear → chunk(2) → silu(gate) * up → down) — closest L2 is `L2/llama_mlp.py` (variant with fused gate+up)
  - **`DiaRMSNorm`** [compute]: `L1/rms_norm.py`
  - **`DiaRotaryEmbedding`** [compute]: `L1/rotary_emb.py`
  - **`DiaSelfAttention`** [compute]: `L2/attention.py` (decoder causal w/ RoPE + KV cache + GQA when num_kv_heads<num_heads); also used in encoder with `is_causal=False`
  - **`DiaCrossAttention`** [compute]: `L1/linear.py (×4 q/k/v/o) + L1/dense_attention.py` (cross-attn with EncoderDecoderCache); closest L2 is `L2/whisper_attention.py` (cross variant)
  - **`DiaEncoderLayer`** [wiring]: wires `DiaSelfAttention`, `DiaMLP`; direct `DiaRMSNorm (×2)`
  - **`DiaEncoder`** [wiring]: wires `DiaEncoderLayer (×N)`, `DiaRotaryEmbedding`, `DiaRMSNorm`; direct `L1/embedding.py`
  - **`DiaDecoderLayer`** [wiring]: wires `DiaSelfAttention`, `DiaCrossAttention`, `DiaMLP`; direct `DiaRMSNorm (×3)`
  - **`DiaDecoder`** [wiring]: wires `DiaDecoderLayer (×N)`, `DiaRotaryEmbedding`, `DiaMultiChannelEmbedding`, `DiaRMSNorm`
  - **`DiaModel`** [wiring]: wires `DiaEncoder`, `DiaDecoder`
  - **`DiaForConditionalGeneration`** [wiring]: wires `DiaModel`; direct `L1/linear.py` (output channel logits per codebook)

## diffllama
- **src**: modeling_diffllama.py (and modular_diffllama.py)
- **hidden_act**: silu
- **status**: composable (Differential Transformer Llama variant)
- **classes**:
  - **`DiffLlamaMLP`** [compute]: `L2/llama_mlp.py`
  - **`DiffLlamaRotaryEmbedding`** [compute]: `L1/rotary_emb.py`
  - **`DiffLlamaAttention`** [compute]: differential attention (two attention computations subtracted with learnable lambda) — no exact L2 match; closest is `L2/attention.py` but with 2× q/k splits + lambda combination
  - **`DiffLlamaFlashAttention2`** [compute, inherits `DiffLlamaAttention`]: same as DiffLlamaAttention but using FlashAttention backend
  - **`DiffLlamaSdpaAttention`** [compute, inherits `DiffLlamaAttention`]: same as DiffLlamaAttention but using SDPA backend
  - **`DiffLlamaRMSNorm`** [compute]: `L1/rms_norm.py`
  - **`DiffLlamaDecoderLayer`** [wiring]: wires `DiffLlamaAttention` (or variants), `DiffLlamaMLP`, `DiffLlamaRMSNorm` (×2)
  - **`DiffLlamaModel`** [wiring]: wires `DiffLlamaDecoderLayer (×N)`, `DiffLlamaRMSNorm`, `DiffLlamaRotaryEmbedding`; direct `L1/embedding.py`
  - **`DiffLlamaForCausalLM`** [wiring]: wires `DiffLlamaModel`; direct `L1/linear.py` (lm_head)
- **task heads (3)**: ForSequenceClassification, ForQuestionAnswering, ForTokenClassification — base + linear (per-task)

## dinat
- **src**: modeling_dinat.py (no modular file)
- **hidden_act**: gelu
- **status**: partial (NeighborhoodAttention requires NATTEN custom kernel; no kb-nano L1)
- **classes**:
  - **`DinatEmbeddings`** [wiring]: wires `DinatPatchEmbeddings`; direct `L1/layer_norm.py` + dropout
  - **`DinatPatchEmbeddings`** [compute]: `L1/conv2d.py (×2 stride-2 stack)` (patch embed via 2 strided conv)
  - **`DinatDownsampler`** [compute]: `L1/conv2d.py + L1/layer_norm.py` (stride-2 conv with norm)
  - **`DinatDropPath`** [compute]: stochastic depth (no kernel at inference)
  - **`NeighborhoodAttention`** [compute]: NATTEN-based local attention with relative position bias — no kb-nano kernel; requires natten library
  - **`NeighborhoodAttentionOutput`** [compute]: `L1/linear.py` (residual is in Layer)
  - **`NeighborhoodAttentionModule`** [wiring]: wires `NeighborhoodAttention`, `NeighborhoodAttentionOutput`
  - **`DinatIntermediate`** [compute]: `L1/linear.py + L1/gelu.py`
  - **`DinatOutput`** [compute]: `L1/linear.py`
  - **`DinatLayer`** [wiring]: wires `NeighborhoodAttentionModule`, `DinatIntermediate`, `DinatOutput`, `DinatDropPath`; direct `L1/layer_norm.py (×2)`
  - **`DinatStage`** [wiring]: wires `DinatLayer (×depth)`, optional `DinatDownsampler`
  - **`DinatEncoder`** [wiring]: wires `DinatStage (×len(depths))`
  - **`DinatModel`** [wiring]: wires `DinatEmbeddings`, `DinatEncoder`; direct `L1/layer_norm.py + L1/adaptive_avg_pool1d.py` (pooler)
- **task heads (1)**: ForImageClassification — base + linear (per-task)

## dinov2
- **src**: modeling_dinov2.py (no modular file)
- **hidden_act**: gelu
- **status**: composable
- **classes**:
  - **`Dinov2Embeddings`** [compute]: `L1/conv2d.py (via Dinov2PatchEmbeddings) + L1/embedding.py` (cls_token, mask_token, position_embeddings as Parameters); bicubic interpolation
  - **`Dinov2PatchEmbeddings`** [compute]: `L1/conv2d.py` (patch_size kernel/stride)
  - **`Dinov2SelfAttention`** [compute, inherits `ViTSelfAttention`]: `L1/linear.py (×3) + L1/dense_attention.py` (ViT-style)
  - **`Dinov2SelfOutput`** [compute, inherits `ViTSelfOutput`]: `L1/linear.py`
  - **`Dinov2Attention`** [wiring]: wires `Dinov2SelfAttention`, `Dinov2SelfOutput`
  - **`Dinov2LayerScale`** [compute]: `lambda1 * x` (single learnable scale; no L1 match — pure mul)
  - **`Dinov2DropPath`** [compute]: stochastic depth (identity at inference)
  - **`Dinov2MLP`** [compute]: `L1/linear.py + L1/gelu.py + L1/linear.py` (2-layer FFN)
  - **`Dinov2SwiGLUFFN`** [compute]: `L1/linear.py + L1/silu.py + chunk-mul + L1/linear.py` (SwiGLU variant when use_swiglu_ffn=True) — closest L2 is `L2/llama_mlp.py` (fused gate+up)
  - **`Dinov2Layer`** [wiring]: wires `Dinov2Attention`, `Dinov2MLP` (or `Dinov2SwiGLUFFN`), `Dinov2LayerScale (×2)`, `Dinov2DropPath`; direct `L1/layer_norm.py (×2)` — closest L3 is `L3/vit_encoder_block.py`
  - **`Dinov2Encoder`** [wiring]: wires `Dinov2Layer (×N)`
  - **`Dinov2Model`** [wiring]: wires `Dinov2Embeddings`, `Dinov2Encoder`; direct `L1/layer_norm.py`
  - **`Dinov2Backbone`** [wiring]: wires `Dinov2Embeddings`, `Dinov2Encoder`; direct `L1/layer_norm.py`
- **task heads (1)**: ForImageClassification — base + linear (per-task)

## dinov2_with_registers
- **src**: modeling_dinov2_with_registers.py (and modular_dinov2_with_registers.py — inherits from Dinov2)
- **hidden_act**: gelu
- **status**: composable
- **classes**:
  - **`Dinov2WithRegistersPatchEmbeddings`** [compute, inherits `Dinov2PatchEmbeddings`]: `L1/conv2d.py`
  - **`Dinov2WithRegistersEmbeddings`** [compute]: `L1/conv2d.py + L1/embedding.py` + `nn.Parameter` register tokens (extra learnable tokens prepended to sequence)
  - **`Dinov2WithRegistersSelfAttention`** [compute, inherits `Dinov2SelfAttention`]: `L1/linear.py (×3) + L1/dense_attention.py`
  - **`Dinov2WithRegistersSelfOutput`** [compute, inherits `Dinov2SelfOutput`]: `L1/linear.py`
  - **`Dinov2WithRegistersAttention`** [wiring, inherits `Dinov2Attention`]: wires self+output
  - **`Dinov2WithRegistersLayerScale`** [compute, inherits `Dinov2LayerScale`]: lambda1 * x
  - **`Dinov2WithRegistersDropPath`** [compute, inherits `Dinov2DropPath`]: identity at inference
  - **`Dinov2WithRegistersMLP`** [compute, inherits `Dinov2MLP`]: `L1/linear.py + L1/gelu.py + L1/linear.py`
  - **`Dinov2WithRegistersSwiGLUFFN`** [compute, inherits `Dinov2SwiGLUFFN`]: SwiGLU variant
  - **`Dinov2WithRegistersLayer`** [wiring, inherits `Dinov2Layer`]: wires attention, MLP, LayerScale (×2), DropPath; direct `L1/layer_norm.py (×2)`
  - **`Dinov2WithRegistersEncoder`** [wiring, inherits `Dinov2Encoder`]: wires `Dinov2WithRegistersLayer (×N)`
  - **`Dinov2WithRegistersModel`** [wiring, inherits `Dinov2Model`]: wires Embeddings, Encoder; direct `L1/layer_norm.py`
  - **`Dinov2WithRegistersBackbone`** [wiring, inherits `Dinov2Backbone`]: same as Dinov2Backbone
- **task heads (1)**: ForImageClassification — base + linear (per-task)
