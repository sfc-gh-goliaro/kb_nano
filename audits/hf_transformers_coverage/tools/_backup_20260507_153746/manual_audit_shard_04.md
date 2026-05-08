# Manual audit — shard 04 (dinov3_convnext through exaone_moe)

## dinov3_convnext
- **src**: modeling_dinov3_convnext.py
- **hidden_act**: gelu
- **status**: composable
- **classes**:
  - **`DINOv3ConvNextDropPath`** [compute]: stochastic-depth wrapper, no kb-nano kernel needed (training-only; inference no-op). `nn.Identity` at inference.
  - **`DINOv3ConvNextLayerNorm`** [compute, inherits `nn.LayerNorm`]: `L1/layer_norm.py` (with permute for channels_first/last)
  - **`DINOv3ConvNextLayer`** [compute]: `L1/conv2d.py` (depthwise_conv) + `L1/layer_norm.py` + `L1/linear.py` (pointwise_conv1) + `L1/gelu.py` + `L1/linear.py` (pointwise_conv2) + residual scale `gamma` (no exact L2 match; ConvNeXt block)
  - **`DINOv3ConvNextStage`** [wiring]: wires `DINOv3ConvNextLayerNorm`, `DINOv3ConvNextLayer`; direct `L1/conv2d.py` (downsample)
  - **`DINOv3ConvNextEncoder`** [wiring, inherits `DINOv3ConvNextPreTrainedModel`]: wires `DINOv3ConvNextStage`
  - **`DINOv3ConvNextModel`** [wiring]: wires `DINOv3ConvNextEncoder`; direct `L1/layer_norm.py` (final norm) + `L1/adaptive_avg_pool2d.py` (pool)
  - **`DINOv3ConvNextBackbone`** [wiring]: wires `DINOv3ConvNextEncoder`

## dinov3_vit
- **src**: modeling_dinov3_vit.py (and modular_dinov3_vit.py)
- **hidden_act**: gelu
- **status**: kb_nano_l4 (kb-nano `L4/dinov3.py` covers the SwiGLU-MLP variant; the un-gated `DINOv3ViTMLP` and the BERT-style attention parts also map to L1/L2)
- **classes**:
  - **`DINOv3ViTBackboneOutput`** [skip] (Output dataclass)
  - **`DINOv3ViTEmbeddings`** [compute]: `L1/conv2d.py` (patch_embeddings) + cls/mask/register tokens (params, no kernel) — close to `L4/dinov3.py` patch_embed
  - **`DINOv3ViTRopePositionEmbedding`** [compute]: `L1/dinov3_rope.py` (DINOv3 2D RoPE)
  - **`DINOv3ViTAttention`** [compute]: `L1/linear.py` (q/k/v/o_proj) + `L1/dense_attention.py` (SDPA, non-causal) + apply_rotary_pos_emb on patch tokens only (no exact L2 match — closest is `L2/eva_attention.py` used inside `L3/eva_block.py` from `L4/dinov3.py`)
  - **`DINOv3ViTLayerScale`** [compute]: scalar mul; no dedicated kb-nano kernel (inline `*` op)
  - **`DINOv3ViTDropPath`** [compute]: stochastic-depth, inference no-op
  - **`DINOv3ViTMLP`** [compute]: `L1/linear.py + L1/gelu.py + L1/linear.py` (un-gated 2-layer MLP, no exact L2 match)
  - **`DINOv3ViTGatedMLP`** [compute]: `L2/llama_mlp.py` (SwiGLU pattern: gate_proj + up_proj * silu/gelu, down_proj) — note default uses gelu; matches `L3/eva_block.py` swiglu used in `L4/dinov3.py`
  - **`DINOv3ViTLayer`** [wiring, inherits `GradientCheckpointingLayer`]: wires `DINOv3ViTAttention`, `DINOv3ViTLayerScale` (×2), `DINOv3ViTGatedMLP` or `DINOv3ViTMLP`, `DINOv3ViTDropPath`; direct `L1/layer_norm.py` (norm1, norm2)
  - **`DINOv3ViTEncoder`** [wiring]: wires `DINOv3ViTLayer`
  - **`DINOv3ViTModel`** [wiring]: wires `DINOv3ViTEmbeddings`, `DINOv3ViTRopePositionEmbedding`, `DINOv3ViTEncoder`; direct `L1/layer_norm.py` (final norm)
  - **`DINOv3ViTBackbone`** [wiring]: wires `DINOv3ViTEmbeddings`, `DINOv3ViTRopePositionEmbedding`, `DINOv3ViTEncoder`; direct `L1/layer_norm.py`

## distilbert
- **src**: modeling_distilbert.py
- **hidden_act**: gelu (config field `activation`)
- **status**: composable
- **classes**:
  - **`Embeddings`** [compute]: `L1/embedding.py` (word_embeddings) + `L1/embedding.py` (position_embeddings) + `L1/layer_norm.py` (no token_type → not full BERT pattern; close to but slimmer than `L2/encoder_embeddings.py`)
  - **`DistilBertSelfAttention`** [compute]: `L2/encoder_attention.py` (q_lin/k_lin/v_lin + dispatch via ALL_ATTENTION_FUNCTIONS + out_lin). Note: includes `out_lin` in the same class (BERT-style splits into SelfAttention+SelfOutput; DistilBert merges them).
  - **`FFN`** [compute]: `L1/linear.py + L1/gelu.py + L1/linear.py` (lin1 → activation → lin2; no exact L2 match — `L2/encoder_mlp.py` requires the BERT Intermediate+Output split with LayerNorm/residual, which DistilBert does outside FFN)
  - **`TransformerBlock`** [wiring, inherits `GradientCheckpointingLayer`]: wires `DistilBertSelfAttention`, `FFN`; direct `L1/layer_norm.py` (sa_layer_norm, output_layer_norm) + residual adds
  - **`Transformer`** [wiring]: wires `TransformerBlock`
  - **`DistilBertModel`** [wiring]: wires `Embeddings`, `Transformer`
  - **`DistilBertForMaskedLM`** [wiring]: wires `DistilBertModel`; direct `L1/linear.py` (vocab_transform) + `L1/gelu.py` + `L1/layer_norm.py` (vocab_layer_norm) + `L1/linear.py` (vocab_projector)
- **task heads (4)**: ForSequenceClassification, ForQuestionAnswering, ForTokenClassification, ForMultipleChoice — base + linear (per-task)

## doge
- **src**: modeling_doge.py (and modular_doge.py)
- **hidden_act**: silu
- **status**: partial (DogeAttention uses dynamic-mask attention with extra dt_proj/A params + flex_attention; DogeCDMoE is a unique cross-domain MoE pattern — neither has an exact kb-nano L2)
- **classes**:
  - **`DogeRMSNorm`** [compute]: `L1/rms_norm.py`
  - **`DogeRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (standard Llama RoPE)
  - **`DogeAttention`** [compute]: `L1/linear.py` (q/k/v/o/dt_proj) + `L1/rms_norm.py` (q_norm/k_norm) + apply_rotary_pos_emb + dynamic-mask softplus + `L1/dense_attention.py` (SDPA backend) (no exact L2 match — non-standard dynamic mask attention with `A`, `dt_proj`, `prepare_dynamic_mask`)
  - **`DogeMLP`** [compute]: `L2/llama_mlp.py` (SwiGLU: gate_proj + up_proj * silu, down_proj)
  - **`DogeCDMoE`** [compute]: cross-domain MoE with router_gate + down_embed/up_embed + shared expert (gate/up/down). No exact kb-nano L2 match — closest pattern is shared expert MoE but with embedding-based experts. `L1/linear.py + L1/embedding.py + L1/silu.py + L2/llama_mlp.py` (shared expert) (no exact L2 match)
  - **`DogeDecoderLayer`** [wiring, inherits `GradientCheckpointingLayer`]: wires `DogeRMSNorm` (×2), `DogeAttention`, `DogeMLP` or `DogeCDMoE`; learnable residual scales (`input_residual`, `post_attention_residual`)
  - **`DogeModel`** [wiring]: wires `DogeDecoderLayer`, `DogeRMSNorm` (final), `DogeRotaryEmbedding`; direct `L1/embedding.py`
  - **`DogeForCausalLM`** [wiring]: wires `DogeModel`; direct `L1/linear.py` (lm_head)
- **task heads (1)**: ForSequenceClassification — base + linear

## donut
- **src**: modeling_donut_swin.py (Swin-style image encoder)
- **hidden_act**: gelu
- **status**: partial (Swin window attention with relative position bias is not in kb-nano; DonutSwin* parallels DonutSwin; closest is generic `L2/encoder_attention.py` but the relative_position_bias_table makes it a Swin-specific compute)
- **classes**:
  - **`DonutSwinEncoderOutput`** / **`DonutSwinModelOutput`** / **`DonutSwinImageClassifierOutput`** [skip] (Output dataclasses)
  - **`DonutSwinEmbeddings`** [compute]: `L1/conv2d.py` (patch_embeddings.projection) + `L1/layer_norm.py` + optional position_embeddings parameter (no exact L2 match)
  - **`DonutSwinPatchEmbeddings`** [compute]: `L1/conv2d.py` (projection) + flatten/transpose
  - **`DonutSwinPatchMerging`** [compute]: window-merge slicing + `L1/layer_norm.py` + `L1/linear.py` (reduction). No exact L2 match — `L2/swinv2_patch_merging.py` is for swinv2 (slightly different).
  - **`DonutSwinDropPath`** [compute]: stochastic-depth, inference no-op
  - **`DonutSwinSelfAttention`** [compute]: q/k/v `L1/linear.py` + relative_position_bias + `L1/dense_attention.py` (no exact L2 match — has Swin relative_position_bias_table)
  - **`DonutSwinSelfOutput`** [compute]: `L1/linear.py` (dense) — no LayerNorm here (different from BERT); no residual in this class
  - **`DonutSwinAttention`** [wiring]: wires `DonutSwinSelfAttention`, `DonutSwinSelfOutput`
  - **`DonutSwinIntermediate`** [compute]: `L1/linear.py + L1/gelu.py`
  - **`DonutSwinOutput`** [compute]: `L1/linear.py` (dense) (no LayerNorm/residual here; layer-level handles it)
  - **`DonutSwinLayer`** [wiring]: wires `DonutSwinAttention`, `DonutSwinIntermediate`, `DonutSwinOutput`, `DonutSwinDropPath`; direct `L1/layer_norm.py` (layernorm_before, layernorm_after); window partition + cyclic shift (Swin)
  - **`DonutSwinStage`** [wiring, inherits `GradientCheckpointingLayer`]: wires `DonutSwinLayer`, optional `DonutSwinPatchMerging` (downsample)
  - **`DonutSwinEncoder`** [wiring]: wires `DonutSwinStage`
  - **`DonutSwinModel`** [wiring]: wires `DonutSwinEmbeddings`, `DonutSwinEncoder`; direct `L1/layer_norm.py` + `L1/adaptive_avg_pool2d.py` (optional pooler)
- **task heads (1)**: ForImageClassification — base + linear

## dots1
- **src**: modeling_dots1.py (and modular_dots1.py — inherits Qwen3 / DeepSeek-V3 lineage)
- **hidden_act**: silu
- **status**: composable
- **classes**:
  - **`Dots1RMSNorm`** [compute, inherits `Qwen3RMSNorm`]: `L1/rms_norm.py`
  - **`Dots1RotaryEmbedding`** [compute, inherits `Qwen3RotaryEmbedding`]: `L1/rotary_emb.py`
  - **`Dots1Attention`** [compute, inherits `Qwen3Attention`]: `L2/attention.py` (Qwen3-style: GQA + qk_norm + RoPE + sliding_window per layer_type)
  - **`Dots1MLP`** [compute, inherits `DeepseekV3MLP`]: `L2/llama_mlp.py` (SwiGLU)
  - **`Dots1TopkRouter`** [compute, inherits `DeepseekV3TopkRouter`]: `L1/linear.py` (no bias) — float32 router with `e_score_correction_bias`
  - **`Dots1NaiveMoe`** [compute]: grouped expert weights (gate_up_proj + down_proj as 3D Parameter); per-token loop. Closest kb-nano: `L1/moe_grouped_gemm.py`
  - **`Dots1MoE`** [compute, inherits `DeepseekV3MoE`]: `L2/shared_expert_moe.py` (group-limited topk + shared expert; DeepSeek-V3 pattern)
  - **`Dots1DecoderLayer`** [wiring, inherits `DeepseekV3DecoderLayer`]: wires `Dots1Attention`, `Dots1MoE` or `Dots1MLP` (per `first_k_dense_replace`), `Dots1RMSNorm` (×2)
  - **`Dots1Model`** [wiring, inherits `Qwen3Model`]: wires `Dots1DecoderLayer`, `Dots1RMSNorm`, `Dots1RotaryEmbedding`; direct `L1/embedding.py`
  - **`Dots1ForCausalLM`** [wiring, inherits `Qwen3ForCausalLM`]: wires `Dots1Model`; direct `L1/linear.py` (lm_head)

## dpr
- **src**: modeling_dpr.py
- **hidden_act**: gelu (delegates to BertModel)
- **status**: composable (delegates entirely to BERT)
- **classes**:
  - **`DPRContextEncoderOutput`** / **`DPRQuestionEncoderOutput`** / **`DPRReaderOutput`** [skip] (Output dataclasses)
  - **`DPREncoder`** [wiring]: wires `BertModel` (from bert); direct optional `L1/linear.py` (encode_proj when projection_dim>0)
  - **`DPRSpanPredictor`** [wiring]: wires `DPREncoder`; direct `L1/linear.py` (qa_outputs, qa_classifier)
  - **`DPRPretrainedContextEncoder`** / **`DPRPretrainedQuestionEncoder`** / **`DPRPretrainedReader`** [skip] (PreTrainedModel base classes)
  - **`DPRContextEncoder`** [wiring]: wires `DPREncoder`
  - **`DPRQuestionEncoder`** [wiring]: wires `DPREncoder`
  - **`DPRReader`** [wiring]: wires `DPRSpanPredictor`

## dpt
- **src**: modeling_dpt.py
- **hidden_act**: gelu
- **status**: partial (depth-estimation specific reassemble/fusion stages have no L2 matches)
- **classes**:
  - **`BaseModelOutputWithIntermediateActivations`** / **`BaseModelOutputWithPoolingAndIntermediateActivations`** [skip] (Output dataclasses)
  - **`DPTViTHybridEmbeddings`** [compute]: `load_backbone(config)` + `L1/conv2d.py` (projection) + cls_token/position_embeddings params (no exact L2 match)
  - **`DPTViTEmbeddings`** [compute]: cls_token/position_embeddings + `DPTViTPatchEmbeddings` (no exact L2 match — pure ViT patch + position; no LayerNorm in __init__)
  - **`DPTViTPatchEmbeddings`** [compute]: `L1/conv2d.py` (projection) + flatten/transpose
  - **`DPTSelfAttention`** [compute, copied from ViTSelfAttention]: `L2/encoder_attention.py` self-attention (q/k/v + dispatch; no out_lin in this class — handled by DPTViTSelfOutput)
  - **`DPTViTSelfOutput`** [compute]: `L1/linear.py` (dense) — no residual/LayerNorm here (handled at layer level)
  - **`DPTViTAttention`** [wiring]: wires `DPTSelfAttention`, `DPTViTSelfOutput`
  - **`DPTViTIntermediate`** [compute]: `L1/linear.py + L1/gelu.py`
  - **`DPTViTOutput`** [compute]: `L1/linear.py` (dense) + residual add (no LayerNorm in this class)
  - **`DPTViTLayer`** [wiring, inherits `GradientCheckpointingLayer`]: wires `DPTViTAttention`, `DPTViTIntermediate`, `DPTViTOutput`; direct `L1/layer_norm.py` (layernorm_before, layernorm_after); residual adds
  - **`DPTReassembleStage`** [compute]: cls-token readout projection + `DPTReassembleLayer` per-stage. `L1/linear.py + L1/gelu.py` for readout_projects (no exact L2 match)
  - **`DPTReassembleLayer`** [compute]: `L1/conv2d.py` (1x1 projection) + `L1/conv_transpose2d.py` or `L1/conv2d.py` (resize)
  - **`DPTFeatureFusionStage`** [wiring]: wires `DPTFeatureFusionLayer`
  - **`DPTPreActResidualLayer`** [compute]: `L1/relu.py + L1/conv2d.py + L1/relu.py + L1/conv2d.py` + optional `L1/batch_norm2d.py`
  - **`DPTFeatureFusionLayer`** [wiring]: wires `DPTPreActResidualLayer` (×2); direct `L1/conv2d.py` (1x1 projection); bilinear upsample (no kernel)
  - **`DPTViTEncoder`** [wiring]: wires `DPTViTLayer`
  - **`DPTModel`** [wiring]: wires `DPTViTEmbeddings` or `DPTViTHybridEmbeddings`, `DPTViTEncoder`, optional `DPTViTPooler`; direct `L1/layer_norm.py`
  - **`DPTViTPooler`** [compute]: `L1/linear.py + L1/tanh.py` (or other ACT2FN)
  - **`DPTNeck`** [wiring]: wires optional `DPTReassembleStage`, `DPTFeatureFusionStage`; direct `L1/conv2d.py` × len(neck_hidden_sizes)
  - **`DPTDepthEstimationHead`** [compute]: `L1/conv2d.py` (×3-4) + interpolate + `L1/relu.py` (no exact L2 match)
  - **`DPTSemanticSegmentationHead`** [compute]: `L1/conv2d.py + L1/batch_norm2d.py + L1/relu.py + L1/conv2d.py` + interpolate
  - **`DPTAuxiliaryHead`** [compute]: `L1/conv2d.py + L1/batch_norm2d.py + L1/relu.py + L1/conv2d.py`
  - **`DPTForDepthEstimation`** [wiring]: wires `DPTModel` (or load_backbone), `DPTNeck`, `DPTDepthEstimationHead`
  - **`DPTForSemanticSegmentation`** [wiring]: wires `DPTModel`, `DPTNeck`, `DPTSemanticSegmentationHead`, optional `DPTAuxiliaryHead`

## edgetam
- **src**: modeling_edgetam.py (and modular_edgetam.py)
- **hidden_act**: gelu
- **status**: partial (SAM-style mask decoder + two-way transformer; closest L2/L3 are sam3_*; EdgeTAM is similar but distinct)
- **classes**:
  - **`EdgeTamLayerNorm`** [compute, inherits `nn.LayerNorm`]: `L1/layer_norm.py` (with channels_first/last variant)
  - **`EdgeTamVisionEncoderOutput`** [skip] (Output dataclass)
  - **`EdgeTamAttention`** [compute]: `L1/linear.py` (q/k/v/o_proj with downsample) + `L1/dense_attention.py` dispatch (eager/SDPA/FA), allows attention_similarity bias (no exact L2 match — closest is `L2/sam3_cross_attention.py`)
  - **`EdgeTamTwoWayAttentionBlock`** [wiring, inherits `GradientCheckpointingLayer`]: wires `EdgeTamAttention` (×3: self_attn, cross_attn_token_to_image, cross_attn_image_to_token), `EdgeTamFeedForward`, `nn.LayerNorm` × 4 (residual adds + layer_norms 1-4)
  - **`EdgeTamFeedForward`** [compute]: `L1/linear.py` (proj_in) + `L1/relu.py` + `L1/linear.py` × N (intermediate layers + activation) + `L1/linear.py` (proj_out) + optional `L1/sigmoid.py` (no exact L2 match — multi-layer MLP with relu)
  - **`EdgeTamSinePositionEmbedding`** [compute]: sine/cosine position encoding (no exact kb-nano kernel; closest: `L1/sinusoidal_embed.py`)
  - **`EdgeTamVisionNeck`** [compute]: `L1/conv2d.py` for FPN + interpolate (no exact L2 match)
  - **`EdgeTamVisionModel`** [wiring]: wires vision backbone + `EdgeTamVisionNeck` + `EdgeTamSinePositionEmbedding`
  - **`EdgeTamImageSegmentationOutput`** [skip] (Output dataclass)
  - **`EdgeTamPositionalEmbedding`** [compute]: positional embedding params (no exact L2 match)
  - **`EdgeTamMaskEmbedding`** [compute]: `L1/conv2d.py` × N + activation + `L1/layer_norm.py`
  - **`EdgeTamPromptEncoder`** [wiring]: wires `EdgeTamMaskEmbedding`, `EdgeTamPositionalEmbedding`; direct `L1/embedding.py` (point/no-mask)
  - **`EdgeTamTwoWayTransformer`** [wiring]: wires `EdgeTamTwoWayAttentionBlock` (×N), `EdgeTamAttention` (final cross-attn); direct `L1/layer_norm.py` (final layernorm)
  - **`EdgeTamMaskDecoder`** [wiring]: wires `EdgeTamTwoWayTransformer`; direct `L1/conv_transpose2d.py` (upscale convs) + `L1/linear.py` (output_hypernetworks_mlps)
  - **`EdgeTamModel`** [wiring]: wires `EdgeTamVisionModel`, `EdgeTamPromptEncoder`, `EdgeTamMaskDecoder`

## edgetam_video
- **src**: modeling_edgetam_video.py (and modular_edgetam_video.py)
- **hidden_act**: gelu (memory_attention_mlp_hidden_act default `relu`)
- **status**: partial (large SAM-style video model — many bespoke modules; closest match is `L4/sam3_video.py`)
- **classes**:
  - **`EdgeTamVideoLayerNorm`** [compute, inherits `nn.LayerNorm`]: `L1/layer_norm.py`
  - **`EdgeTamVideoMemoryFuserCXBlock`** [compute, inherits `GradientCheckpointingLayer`]: `L1/conv2d.py` (depthwise) + `L1/layer_norm.py` + `L1/linear.py` × 2 + `L1/gelu.py` (ConvNext-like block)
  - **`EdgeTamVideoVisionEncoderOutput`** [skip] (Output dataclass)
  - **`EdgeTamVideoVisionRotaryEmbedding`** [compute]: `L1/vision_rotary_emb.py` (2D vision RoPE; closest match)
  - **`EdgeTamVideoAttention`** [compute]: `L1/linear.py` (q/k/v/o) + `L1/dense_attention.py` (no exact L2 match)
  - **`EdgeTamVideoRoPESelfAttention`** [compute]: `L1/linear.py` + RoPE + `L1/dense_attention.py` (no exact L2 match)
  - **`EdgeTamVideoRoPECrossAttention`** [compute]: `L1/linear.py` + RoPE + cross-attention `L1/dense_attention.py` (no exact L2 match)
  - **`EdgeTamVideoTwoWayAttentionBlock`** [wiring, inherits `GradientCheckpointingLayer`]: wires self/cross attentions + feedforward + `L1/layer_norm.py` × 4
  - **`EdgeTamVideoPositionEmbeddingSine`** [compute]: sine/cosine encoding (closest: `L1/sinusoidal_embed.py`)
  - **`EdgeTamVideoMemoryFuser`** [wiring]: wires `EdgeTamVideoMemoryFuserCXBlock`
  - **`EdgeTamVideoMaskDownSamplerLayer`** [compute]: `L1/conv2d.py + L1/layer_norm.py + L1/gelu.py`
  - **`EdgeTamVideoMaskDownSampler`** [wiring]: wires `EdgeTamVideoMaskDownSamplerLayer`; direct `L1/conv2d.py` (final)
  - **`EdgeTamVideoMemoryEncoder`** [wiring]: wires `EdgeTamVideoMaskDownSampler`, `EdgeTamVideoMemoryFuser`, `EdgeTamVideoPositionEmbeddingSine`; direct `L1/conv2d.py`
  - **`EdgeTamVideoFeedForward`** [compute]: `L1/linear.py` + activation (relu) + `L1/linear.py` × N + optional `L1/sigmoid.py`
  - **`EdgeTamVideoPositionalEmbedding`** [compute]: positional embedding params (no exact L2 match)
  - **`EdgeTamVideoInferenceCache`** / **`EdgeTamVideoInferenceSession`** [skip] (Cache/Session dataclasses)
  - **`EdgeTamVideoMemoryAttentionMLP`** [compute]: `L1/linear.py + L1/relu.py + L1/linear.py`
  - **`EdgeTamVideoMemoryAttentionLayer`** [wiring]: wires self/cross attention + MLP + LayerNorms
  - **`EdgeTamVideoMemoryAttention`** [wiring]: wires `EdgeTamVideoMemoryAttentionLayer`; direct `L1/layer_norm.py`
  - **`EdgeTamVideoPerceiverMLP`** [compute]: `L1/linear.py + L1/gelu.py + L1/linear.py`
  - **`EdgeTamVideoPerceiverAttention`** [compute]: `L1/linear.py` (q/k/v/o) + cross-attention + `L1/dense_attention.py` (no exact L2 match)
  - **`EdgeTamVideoPerceiverEncoderLayer`** [wiring]: wires `EdgeTamVideoPerceiverAttention`, `EdgeTamVideoPerceiverMLP`, `L1/layer_norm.py` × 2
  - **`EdgeTamVideoPerceiverResampler`** [wiring]: wires `EdgeTamVideoPerceiverEncoderLayer`; direct `L1/layer_norm.py`
  - **`EdgeTamVideoImageSegmentationOutput`** / **`EdgeTamVideoSegmentationOutput`** [skip] (Output dataclasses)
  - **`EdgeTamVideoMaskEmbedding`** [compute]: `L1/conv2d.py` × N + activation + `L1/layer_norm.py`
  - **`EdgeTamVideoPromptEncoder`** [wiring]: wires `EdgeTamVideoMaskEmbedding`, `EdgeTamVideoPositionalEmbedding`; direct `L1/embedding.py`
  - **`EdgeTamVideoTwoWayTransformer`** [wiring]: wires `EdgeTamVideoTwoWayAttentionBlock`; direct `L1/layer_norm.py`
  - **`EdgeTamVideoMaskDecoder`** [wiring]: wires `EdgeTamVideoTwoWayTransformer`; direct `L1/conv_transpose2d.py + L1/linear.py`
  - **`EdgeTamVideoModel`** [wiring]: wires VisionModel, MemoryEncoder, MemoryAttention, PerceiverResampler, PromptEncoder, MaskDecoder

## efficientloftr
- **src**: modeling_efficientloftr.py (and modular_efficientloftr.py)
- **hidden_act**: gelu (default — config field absent → gelu via ACT2FN; uses LeakyReLU in some convs)
- **status**: unsupported (LoFTR-style keypoint matcher; no kb-nano L4 or L2 specialised for it)
- **classes**:
  - **`EfficientLoFTRKeypointMatchingOutput`** [skip] (Output dataclass)
  - **`EfficientLoFTRRotaryEmbedding`** [compute]: 2D RoPE for keypoint matching → closest `L1/vision_rotary_emb.py` (no exact match)
  - **`EfficientLoFTRConvNormLayer`** [compute]: `L1/conv2d.py + L1/batch_norm2d.py + L1/leaky_relu.py` (no exact L2 match)
  - **`EfficientLoFTRRepVGGBlock`** [compute, inherits `GradientCheckpointingLayer`]: `L1/conv2d.py` (3x3) + `L1/conv2d.py` (1x1) + `L1/batch_norm2d.py` + `L1/leaky_relu.py` (no exact L2 match — closest `L2/rtdetrv2_repvgg_block.py` but different)
  - **`EfficientLoFTRRepVGGStage`** [wiring]: wires `EfficientLoFTRRepVGGBlock`
  - **`EfficientLoFTRepVGG`** [wiring]: wires `EfficientLoFTRRepVGGStage`
  - **`EfficientLoFTRAggregationLayer`** [compute]: `L1/conv2d.py + L1/layer_norm.py + L1/gelu.py` (aggregation layer; no exact L2 match)
  - **`EfficientLoFTRAttention`** [compute]: `L1/linear.py` (q/k/v/o) + RoPE + `L1/dense_attention.py` (no exact L2 match)
  - **`EfficientLoFTRMLP`** [compute]: `L1/linear.py + L1/gelu.py + L1/linear.py` (no exact L2 match — fc1→act→fc2 without LayerNorm in this class)
  - **`EfficientLoFTRAggregatedAttention`** [compute]: aggregation + attention combined; `L1/linear.py + L1/conv2d.py + L1/dense_attention.py` (no exact L2 match)
  - **`EfficientLoFTRLocalFeatureTransformerLayer`** [wiring, inherits `GradientCheckpointingLayer`]: wires `EfficientLoFTRAggregatedAttention`, `EfficientLoFTRMLP`, `L1/layer_norm.py` × 2
  - **`EfficientLoFTRLocalFeatureTransformer`** [wiring]: wires `EfficientLoFTRLocalFeatureTransformerLayer`
  - **`EfficientLoFTROutConvBlock`** [compute]: `L1/conv2d.py + L1/batch_norm2d.py + L1/leaky_relu.py + L1/conv2d.py`
  - **`EfficientLoFTRFineFusionLayer`** [compute]: `L1/conv2d.py` × N + `L1/leaky_relu.py` + interpolate (no exact L2 match)
  - **`EfficientLoFTRModel`** [wiring]: wires `EfficientLoFTRepVGG`, `EfficientLoFTRLocalFeatureTransformer`, `EfficientLoFTRFineFusionLayer`, `EfficientLoFTRRotaryEmbedding`
  - **`EfficientLoFTRForKeypointMatching`** [wiring]: wires `EfficientLoFTRModel`; direct `L1/conv2d.py` (output projections)

## efficientnet
- **src**: modeling_efficientnet.py
- **hidden_act**: swish (silu)
- **status**: partial (EfficientNet v1; closest is `L4/efficientnetv2.py` for v2)
- **classes**:
  - **`EfficientNetEmbeddings`** [compute]: `L1/conv2d.py` (with manual padding) + `L1/batch_norm2d.py` + `L1/silu.py` (swish)
  - **`EfficientNetDepthwiseConv2d`** [compute, inherits `nn.Conv2d`]: `L1/conv2d.py` (depthwise variant)
  - **`EfficientNetExpansionLayer`** [compute]: `L1/conv2d.py` (1x1) + `L1/batch_norm2d.py` + `L1/silu.py`
  - **`EfficientNetDepthwiseLayer`** [compute]: `EfficientNetDepthwiseConv2d` + `L1/batch_norm2d.py` + `L1/silu.py` (depthwise + BN + activation)
  - **`EfficientNetSqueezeExciteLayer`** [compute]: `L1/adaptive_avg_pool2d.py` + `L1/conv2d.py` (×2) + `L1/silu.py` + `L1/sigmoid.py` (closest L2: `L2/efficientnetv2_squeeze_excite.py`)
  - **`EfficientNetFinalBlockLayer`** [compute]: `L1/conv2d.py` (1x1) + `L1/batch_norm2d.py` + skip + dropout
  - **`EfficientNetBlock`** [wiring]: wires `EfficientNetExpansionLayer`, `EfficientNetDepthwiseLayer`, `EfficientNetSqueezeExciteLayer`, `EfficientNetFinalBlockLayer` (closest L2: `L2/efficientnetv2_inverted_residual.py`)
  - **`EfficientNetEncoder`** [wiring]: wires `EfficientNetBlock`; direct `L1/conv2d.py + L1/batch_norm2d.py + L1/silu.py` (top conv)
  - **`EfficientNetModel`** [wiring]: wires `EfficientNetEmbeddings`, `EfficientNetEncoder`; direct `L1/avg_pool2d.py` (pooling)
- **task heads (1)**: ForImageClassification — base + linear

## electra
- **src**: modeling_electra.py
- **hidden_act**: gelu
- **status**: composable (BERT-clone)
- **classes**:
  - **`ElectraEmbeddings`** [compute]: `L2/encoder_embeddings.py` (word + position + token_type + LayerNorm + Dropout, BERT pattern)
  - **`ElectraSelfAttention`** [compute, copied from BertSelfAttention]: `L2/encoder_attention.py` (q/k/v + dispatch + KV cache when decoder)
  - **`ElectraCrossAttention`** [compute]: `L2/encoder_attention.py` (cross-attention variant)
  - **`ElectraSelfOutput`** [compute]: `L2/encoder_attention.py` (dense + LayerNorm + residual)
  - **`ElectraAttention`** [wiring]: wires `ElectraSelfAttention` (or `ElectraCrossAttention`), `ElectraSelfOutput`
  - **`ElectraIntermediate`** [compute]: `L1/linear.py + L1/gelu.py`
  - **`ElectraOutput`** [compute]: `L2/encoder_attention.py` shape (dense + LayerNorm + residual)
  - **`ElectraLayer`** [wiring, inherits `GradientCheckpointingLayer`]: wires `ElectraAttention`, `ElectraIntermediate`, `ElectraOutput`; optional `ElectraAttention` (cross-attention when add_cross_attention)
  - **`ElectraEncoder`** [wiring]: wires `ElectraLayer`
  - **`ElectraDiscriminatorPredictions`** [compute]: `L1/linear.py + L1/gelu.py + L1/linear.py`
  - **`ElectraGeneratorPredictions`** [compute]: `L1/linear.py + L1/gelu.py + L1/layer_norm.py`
  - **`ElectraForPreTrainingOutput`** [skip] (Output dataclass)
  - **`ElectraModel`** [wiring]: wires `ElectraEmbeddings`, `ElectraEncoder`; direct optional `L1/linear.py` (embeddings_project)
  - **`ElectraClassificationHead`** [compute]: `L1/linear.py + L1/gelu.py + L1/linear.py`
  - **`ElectraSequenceSummary`** [compute]: `L1/linear.py` + activation (configurable)
  - **`ElectraForCausalLM`** [wiring]: wires `ElectraModel`, `ElectraGeneratorPredictions`; direct `L1/linear.py` (generator_lm_head)
  - **`ElectraForMaskedLM`** [wiring]: wires `ElectraModel`, `ElectraGeneratorPredictions`; direct `L1/linear.py` (generator_lm_head)
- **task heads (5)**: ForSequenceClassification, ForPreTraining, ForTokenClassification, ForQuestionAnswering, ForMultipleChoice — base + linear (per-task)

## emu3
- **src**: modeling_emu3.py (and modular_emu3.py)
- **hidden_act**: silu
- **status**: partial (text part is Llama-style → composable; VQVAE part has 3D conv and bespoke blocks → no exact L2 matches)
- **classes**:
  - **`Emu3VQVAEModelOutput`** [skip] (Output dataclass)
  - **`Emu3Attention`** [compute]: `L2/attention.py` (Llama-style: GQA + RoPE + KV cache)
  - **`Emu3RMSNorm`** [compute]: `L1/rms_norm.py`
  - **`Emu3MLP`** [compute]: `L2/llama_mlp.py` (SwiGLU)
  - **`Emu3DecoderLayer`** [wiring, inherits `GradientCheckpointingLayer`]: wires `Emu3RMSNorm` (×2), `Emu3Attention`, `Emu3MLP`
  - **`Emu3VQVAEVectorQuantizer`** [compute]: `L1/embedding.py` (codebook) + L2-norm distance + index lookup (no exact L2 match)
  - **`Emu3VQVAEEncoderConvDownsample`** [compute]: `L1/conv2d.py` (stride=2)
  - **`Emu3VQVAEEncoderConvUpsample`** [compute]: `L1/conv2d.py` + interpolate
  - **`Emu3VQVAEConv3d`** [compute]: `L1/conv3d.py` (with manual padding)
  - **`Emu3VQVAESpatialNorm`** [compute]: `L1/group_norm.py` + `L1/conv2d.py` × 2 (conditional spatial norm; no exact L2 match)
  - **`Emu3VQVAETemporalUpsample`** [compute]: interpolate + `L1/conv3d.py`
  - **`Emu3VQVAETemporalDownsample`** [compute]: `L1/conv3d.py` (stride=2 in time dim)
  - **`Emu3VQVAETemporalResnetBlock`** [compute]: `Emu3VQVAEConv3d` + `L1/group_norm.py` + `L1/silu.py` (no exact L2 match)
  - **`Emu3VQVAEResnetBlock`** [compute]: `L1/conv2d.py + L1/group_norm.py + L1/silu.py` (or SpatialNorm variant) (no exact L2 match)
  - **`Emu3VQVAEAttentionBlock`** [compute]: `L1/group_norm.py` + q/k/v `L1/linear.py` + `L1/dense_attention.py` + out `L1/linear.py` (no exact L2 match)
  - **`Emu3VQVAEGroupNorm`** [compute, inherits `nn.GroupNorm`]: `L1/group_norm.py`
  - **`Emu3VQVAEMiddleBlock`** [wiring]: wires `Emu3VQVAEResnetBlock`, `Emu3VQVAEAttentionBlock`
  - **`Emu3VQVAEDownBlock`** [wiring]: wires `Emu3VQVAEResnetBlock`, optional `Emu3VQVAEEncoderConvDownsample`, optional `Emu3VQVAEAttentionBlock`
  - **`Emu3VQVAEUpBlock`** [wiring]: wires `Emu3VQVAEResnetBlock`, optional `Emu3VQVAEEncoderConvUpsample`, optional `Emu3VQVAEAttentionBlock`
  - **`Emu3VQVAEEncoder`** [wiring]: wires `Emu3VQVAEDownBlock`, `Emu3VQVAEMiddleBlock`, `Emu3VQVAETemporalDownsample`; direct `L1/conv2d.py + L1/group_norm.py + L1/silu.py + L1/conv3d.py`
  - **`Emu3VQVAEDecoder`** [wiring]: wires `Emu3VQVAEMiddleBlock`, `Emu3VQVAEUpBlock`, `Emu3VQVAETemporalUpsample`; direct `L1/conv2d.py + L1/group_norm.py + L1/silu.py`
  - **`Emu3VQVAE`** [wiring]: wires `Emu3VQVAEEncoder`, `Emu3VQVAEDecoder`, `Emu3VQVAEVectorQuantizer`; direct `L1/conv3d.py` × 2 (quant_conv, post_quant_conv)
  - **`Emu3ImageVocabularyMapping`** [skip] (utility class, not nn.Module)
  - **`Emu3RotaryEmbedding`** [compute]: `L1/rotary_emb.py` (Llama RoPE)
  - **`Emu3TextModel`** [wiring]: wires `Emu3DecoderLayer`, `Emu3RMSNorm`, `Emu3RotaryEmbedding`; direct `L1/embedding.py`
  - **`Emu3ForCausalLM`** [wiring]: wires `Emu3TextModel`; direct `L1/linear.py` (lm_head)
  - **`Emu3Model`** [wiring]: wires `Emu3VQVAE`, `Emu3TextModel`
  - **`Emu3ForConditionalGeneration`** [wiring]: wires `Emu3Model`; direct `L1/linear.py` (lm_head)

## encodec
- **src**: modeling_encodec.py
- **hidden_act**: elu (encodec-specific config field; ELU activation in convs)
- **status**: unsupported (audio codec; no kb-nano L2/L4 for Encodec — closest unrelated)
- **classes**:
  - **`EncodecOutput`** / **`EncodecEncoderOutput`** / **`EncodecDecoderOutput`** [skip] (Output dataclasses)
  - **`EncodecConv1d`** [compute]: `L1/conv1d.py` with manual reflection-padding + optional weight-norm (no exact L2 match)
  - **`EncodecConvTranspose1d`** [compute]: `L1/conv_transpose1d.py` with optional weight-norm
  - **`EncodecLSTM`** [compute]: `L1/lstm.py`
  - **`EncodecResnetBlock`** [compute]: `L1/elu.py + EncodecConv1d` × N (residual block)
  - **`EncodecEncoder`** [wiring]: wires `EncodecConv1d`, `EncodecResnetBlock`, `EncodecLSTM`; direct `L1/elu.py`
  - **`EncodecDecoder`** [wiring]: wires `EncodecConvTranspose1d`, `EncodecResnetBlock`, `EncodecLSTM`; direct `L1/elu.py`
  - **`EncodecEuclideanCodebook`** [compute]: nearest-neighbor lookup over codebook embedding (no exact kb-nano kernel)
  - **`EncodecVectorQuantization`** [wiring]: wires `EncodecEuclideanCodebook`
  - **`EncodecResidualVectorQuantizer`** [wiring]: wires `EncodecVectorQuantization` × N
  - **`EncodecModel`** [wiring]: wires `EncodecEncoder`, `EncodecDecoder`, `EncodecResidualVectorQuantizer`

## encoder_decoder
- **src**: modeling_encoder_decoder.py
- **hidden_act**: N/A (delegates to encoder/decoder configs)
- **status**: composable (pure wiring of two pretrained models)
- **classes**:
  - **`EncoderDecoderModel`** [wiring]: wires arbitrary encoder PreTrainedModel + arbitrary decoder PreTrainedModel; direct optional `L1/linear.py` (enc_to_dec_proj for dim adapter)

## eomt
- **src**: modeling_eomt.py (and modular_eomt.py — inherits from `dinov2`, `siglip`, `mask2former`, `vit`)
- **hidden_act**: gelu
- **status**: partial (universal segmentation; depends on Dinov2 backbone + SigLIP attention + Mask2Former loss)
- **classes**:
  - **`EomtForUniversalSegmentationOutput`** [skip] (Output dataclass)
  - **`EomtHungarianMatcher`** [compute]: bipartite matching helper; CPU-side; no kb-nano kernel needed
  - **`EomtLoss`** [compute, inherits `Mask2FormerLoss`]: training-only loss; no inference kernel needed
  - **`EomtPatchEmbeddings`** [compute, inherits `Dinov2PatchEmbeddings`]: `L1/conv2d.py` (projection)
  - **`EomtEmbeddings`** [compute, inherits `Dinov2Embeddings`]: cls_token + register_tokens + patch_embed + position_embeddings (no exact L2 match)
  - **`EomtAttention`** [compute, inherits `SiglipAttention`]: `L2/siglip_attention.py` (non-causal multi-head)
  - **`EomtLayerScale`** [compute, inherits `Dinov2LayerScale`]: scalar mul (no kernel)
  - **`EomtDropPath`** [compute]: stochastic-depth, inference no-op
  - **`EomtMLP`** [compute]: `L1/linear.py + L1/gelu.py + L1/linear.py` (un-gated 2-layer)
  - **`EomtSwiGLUFFN`** [compute]: `L2/llama_mlp.py` (SwiGLU)
  - **`EomtLayer`** [wiring, inherits `Dinov2Layer`/`GradientCheckpointingLayer`]: wires `EomtAttention`, `EomtMLP` or `EomtSwiGLUFFN`, `EomtLayerScale` (×2), `EomtDropPath`; direct `L1/layer_norm.py` × 2
  - **`EomtLayerNorm2d`** [compute, inherits `nn.LayerNorm`]: `L1/layer_norm.py` (with permute)
  - **`EomtScaleLayer`** [compute]: `L1/conv2d.py + L1/layer_norm.py + L1/gelu.py` + scale param (no exact L2 match)
  - **`EomtScaleBlock`** [wiring]: wires `EomtScaleLayer`
  - **`EomtMaskHead`** [compute]: `L1/linear.py + L1/gelu.py + L1/linear.py + L1/gelu.py + L1/linear.py` (3-layer MLP with gelu)
  - **`EomtForUniversalSegmentation`** [wiring]: wires `EomtEmbeddings`, `EomtLayer` × N, `EomtScaleBlock`, `EomtMaskHead`; direct `L1/embedding.py` (queries) + `L1/layer_norm.py` (final)

## eomt_dinov3
- **src**: modeling_eomt_dinov3.py (and modular_eomt_dinov3.py)
- **hidden_act**: gelu
- **status**: partial (DINOv3 ViT backbone + EoMT segmentation; mostly inherits from eomt and dinov3_vit modular lineage)
- **classes**:
  - **`EomtDinov3Attention`** [compute]: `L1/linear.py` (q/k/v/o) + `L1/dense_attention.py` + RoPE on patch tokens (similar to DINOv3ViTAttention; no exact L2 match)
  - **`EomtDinov3Embeddings`** [compute]: cls + register tokens + `L1/conv2d.py` patch embed (similar to DINOv3ViTEmbeddings)
  - **`EomtDinov3DropPath`** [compute]: stochastic-depth
  - **`EomtDinov3MLP`** [compute]: `L1/linear.py + L1/gelu.py + L1/linear.py` (2-layer MLP)
  - **`EomtDinov3GatedMLP`** [compute]: `L2/llama_mlp.py` (SwiGLU)
  - **`EomtDinov3Layer`** [wiring, inherits `GradientCheckpointingLayer`]: wires `EomtDinov3Attention`, `EomtDinov3MLP` or `EomtDinov3GatedMLP`, `EomtDinov3LayerScale` (×2), `EomtDinov3DropPath`; direct `L1/layer_norm.py` × 2
  - **`EomtDinov3LayerScale`** [compute]: scalar mul
  - **`EomtDinov3RotaryEmbedding`** [compute]: `L1/dinov3_rope.py` (DINOv3 2D RoPE)
  - **`EomtDinov3HungarianMatcher`** [compute]: bipartite matching (training)
  - **`EomtDinov3Loss`** [compute]: training-only loss
  - **`EomtDinov3ForUniversalSegmentationOutput`** [skip] (Output dataclass)
  - **`EomtDinov3LayerNorm2d`** [compute, inherits `nn.LayerNorm`]: `L1/layer_norm.py`
  - **`EomtDinov3ScaleLayer`** [compute]: `L1/conv2d.py + L1/layer_norm.py + L1/gelu.py`
  - **`EomtDinov3ScaleBlock`** [wiring]: wires `EomtDinov3ScaleLayer`
  - **`EomtDinov3MaskHead`** [compute]: `L1/linear.py + L1/gelu.py + L1/linear.py + L1/gelu.py + L1/linear.py`
  - **`EomtDinov3ForUniversalSegmentation`** [wiring]: wires `EomtDinov3Embeddings`, `EomtDinov3Layer` × N, `EomtDinov3RotaryEmbedding`, `EomtDinov3ScaleBlock`, `EomtDinov3MaskHead`; direct `L1/embedding.py` (queries) + `L1/layer_norm.py`

## ernie
- **src**: modeling_ernie.py (and modular_ernie.py — inherits BERT)
- **hidden_act**: gelu
- **status**: composable (BERT-clone with optional task_type_ids)
- **classes**:
  - **`ErnieEmbeddings`** [compute]: `L2/encoder_embeddings.py` (word + position + token_type + LayerNorm + Dropout) + optional task_type embedding (`L1/embedding.py`)
  - **`ErnieSelfAttention`** [compute]: `L2/encoder_attention.py` (q/k/v + dispatch)
  - **`ErnieCrossAttention`** [compute]: `L2/encoder_attention.py` (cross-attn variant)
  - **`ErnieSelfOutput`** [compute]: `L2/encoder_attention.py` (dense + LayerNorm + residual)
  - **`ErnieAttention`** [wiring]: wires `ErnieSelfAttention` (or `ErnieCrossAttention`), `ErnieSelfOutput`
  - **`ErnieIntermediate`** [compute]: `L1/linear.py + L1/gelu.py`
  - **`ErnieOutput`** [compute]: `L2/encoder_attention.py` shape (dense + LayerNorm + residual)
  - **`ErnieLayer`** [wiring, inherits `GradientCheckpointingLayer`]: wires `ErnieAttention`, `ErnieIntermediate`, `ErnieOutput`, optional cross-attention `ErnieAttention`
  - **`ErniePooler`** [compute]: `L1/linear.py + L1/tanh.py`
  - **`ErniePredictionHeadTransform`** [compute]: `L1/linear.py + L1/gelu.py + L1/layer_norm.py`
  - **`ErnieLMPredictionHead`** [wiring]: wires `ErniePredictionHeadTransform`; direct `L1/linear.py` (decoder)
  - **`ErnieEncoder`** [wiring]: wires `ErnieLayer`
  - **`ErnieModel`** [wiring]: wires `ErnieEmbeddings`, `ErnieEncoder`, optional `ErniePooler`
  - **`ErnieForPreTrainingOutput`** [skip] (Output dataclass)
  - **`ErniePreTrainingHeads`** [wiring]: wires `ErnieLMPredictionHead`; direct `L1/linear.py` (seq_relationship)
  - **`ErnieForPreTraining`** [wiring]: wires `ErnieModel`, `ErniePreTrainingHeads`
  - **`ErnieOnlyMLMHead`** [wiring]: wires `ErnieLMPredictionHead`
  - **`ErnieForCausalLM`** [wiring]: wires `ErnieModel`, `ErnieOnlyMLMHead`
  - **`ErnieForMaskedLM`** [wiring]: wires `ErnieModel`, `ErnieOnlyMLMHead`
  - **`ErnieOnlyNSPHead`** [wiring]: direct `L1/linear.py` (seq_relationship)
- **task heads (5)**: ForNextSentencePrediction, ForSequenceClassification, ForMultipleChoice, ForTokenClassification, ForQuestionAnswering — base + linear (per-task)

## ernie4_5
- **src**: modeling_ernie4_5.py (and modular_ernie4_5.py — inherits Llama / Olmo)
- **hidden_act**: silu
- **status**: composable (Llama-clone)
- **classes**:
  - **`Ernie4_5RotaryEmbedding`** [compute, inherits `OlmoRotaryEmbedding`]: `L1/rotary_emb.py`
  - **`Ernie4_5MLP`** [compute, inherits `LlamaMLP`]: `L2/llama_mlp.py` (SwiGLU)
  - **`Ernie4_5Attention`** [compute, inherits `LlamaAttention`]: `L2/attention.py` (Llama-style: GQA + RoPE + KV cache)
  - **`Ernie4_5RMSNorm`** [compute]: `L1/rms_norm.py`
  - **`Ernie4_5DecoderLayer`** [wiring, inherits `GradientCheckpointingLayer`]: wires `Ernie4_5RMSNorm` (×2), `Ernie4_5Attention`, `Ernie4_5MLP`
  - **`Ernie4_5Model`** [wiring]: wires `Ernie4_5DecoderLayer`, `Ernie4_5RMSNorm`, `Ernie4_5RotaryEmbedding`; direct `L1/embedding.py`
  - **`Ernie4_5ForCausalLM`** [wiring, inherits `LlamaForCausalLM`]: wires `Ernie4_5Model`; direct `L1/linear.py` (lm_head)

## ernie4_5_moe
- **src**: modeling_ernie4_5_moe.py (and modular_ernie4_5_moe.py — inherits LlamaAttention, Qwen3MoeMLP, MixtralExperts/PreTrainedModel)
- **hidden_act**: silu
- **status**: composable (Mixtral/Qwen3-MoE clone with custom router)
- **classes**:
  - **`Ernie4_5_MoeRMSNorm`** [compute, inherits `LlamaRMSNorm`]: `L1/rms_norm.py`
  - **`Ernie4_5_MoeMLP`** [compute, inherits `Qwen3MoeMLP`]: `L2/llama_mlp.py` (SwiGLU)
  - **`Ernie4_5_MoeRotaryEmbedding`** [compute, inherits `Ernie4_5RotaryEmbedding`]: `L1/rotary_emb.py`
  - **`Ernie4_5_MoeAttention`** [compute, inherits `LlamaAttention`]: `L2/attention.py`
  - **`Ernie4_5_MoeStatics`** [compute]: stat-tracking module (router stats); training-side; no inference kernel
  - **`Ernie4_5_MoeExperts`** [compute, inherits `MixtralExperts`]: `L1/moe_grouped_gemm.py`
  - **`Ernie4_5_MoeTopKRouter`** [compute]: `L1/linear.py` (router) + topk softmax (no exact L2 match — closest is similar router patterns in `L2/mixtral_moe.py`)
  - **`Ernie4_5_MoeSparseMoeBlock`** [wiring]: wires `Ernie4_5_MoeTopKRouter`, `Ernie4_5_MoeExperts`, optional shared expert (closest `L2/mixtral_moe.py` or `L2/shared_expert_moe.py`)
  - **`Ernie4_5_MoeDecoderLayer`** [wiring, inherits `Qwen3MoeDecoderLayer`]: wires `Ernie4_5_MoeAttention`, `Ernie4_5_MoeSparseMoeBlock` or `Ernie4_5_MoeMLP`, `Ernie4_5_MoeRMSNorm` × 2
  - **`Ernie4_5_MoeModel`** [wiring]: wires `Ernie4_5_MoeDecoderLayer`, `Ernie4_5_MoeRMSNorm`, `Ernie4_5_MoeRotaryEmbedding`; direct `L1/embedding.py`
  - **`Ernie4_5_MoeForCausalLM`** [wiring, inherits `MixtralForCausalLM`]: wires `Ernie4_5_MoeModel`; direct `L1/linear.py` (lm_head)

## ernie4_5_vl_moe
- **src**: modeling_ernie4_5_vl_moe.py (and modular_ernie4_5_vl_moe.py)
- **hidden_act**: quick_gelu (vision) / silu (text)
- **status**: partial (text MoE part composable; vision encoder uses 2D RoPE patch transformer with quick_gelu)
- **classes**:
  - **`Ernie4_5_VLMoeTextRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (Llama RoPE)
  - **`Ernie4_5_VLMoeTextAttention`** [compute]: `L2/attention.py`
  - **`Ernie4_5_VLMoeRMSNorm`** [compute]: `L1/rms_norm.py`
  - **`Ernie4_5_VLMoeMLP`** [compute]: `L2/llama_mlp.py` (SwiGLU)
  - **`Ernie4_5_VLMoeMoeStatics`** [compute]: training stats helper
  - **`Ernie4_5_VLMoeMoeTopKRouter`** [compute]: `L1/linear.py` + topk softmax
  - **`Ernie4_5_VLMoeMoeExperts`** [compute]: `L1/moe_grouped_gemm.py` (grouped-experts pattern)
  - **`Ernie4_5_VLMoeSparseMoeBlock`** [wiring]: wires router + experts (similar to Mixtral)
  - **`Ernie4_5_VLMoeMoeBlock`** [wiring]: wires `Ernie4_5_VLMoeSparseMoeBlock` × N (text + vision experts) + shared expert (closest `L2/shared_expert_moe.py`)
  - **`Ernie4_5_VLMoeDecoderLayer`** [wiring, inherits `GradientCheckpointingLayer`]: wires `Ernie4_5_VLMoeTextAttention`, `Ernie4_5_VLMoeMoeBlock` or `Ernie4_5_VLMoeMLP`, `Ernie4_5_VLMoeRMSNorm` × 2
  - **`Ernie4_5_VLMoeVisionAttention`** [compute]: `L1/linear.py` + 2D-RoPE + `L1/dense_attention.py` (no exact L2 match — like vision_attention)
  - **`Ernie4_5_VLMoeVisionBlock`** [wiring, inherits `GradientCheckpointingLayer`]: wires `Ernie4_5_VLMoeVisionAttention`, `Ernie4_5VLVisionMLP`, `L1/layer_norm.py` × 2
  - **`Ernie4_5_VLMoeTextModel`** [wiring]: wires `Ernie4_5_VLMoeDecoderLayer`, `Ernie4_5_VLMoeRMSNorm`, `Ernie4_5_VLMoeTextRotaryEmbedding`; direct `L1/embedding.py`
  - **`Ernie4_5VLVisionMLP`** [compute]: `L1/linear.py + L1/quickgelu.py + L1/linear.py` (vision MLP with quick_gelu)
  - **`Ernie4_5_VLMoePatchEmbed`** [compute]: `L1/conv3d.py` or `L1/conv2d.py` (patch embed)
  - **`Ernie4_5_VLMoeVisionRotaryEmbedding`** [compute]: `L1/vision_rotary_emb.py`
  - **`Ernie4_5_VLMoeVisionTransformerPretrainedModel`** [wiring]: wires `Ernie4_5_VLMoePatchEmbed`, `Ernie4_5_VLMoeVisionBlock`, `Ernie4_5_VLMoeVisionRotaryEmbedding`
  - **`Ernie4_5_VLMoeVisionMLP`** [compute]: `L2/llama_mlp.py` (SwiGLU; differs from `Ernie4_5VLVisionMLP`)
  - **`Ernie4_5_VLMoeVariableResolutionResamplerModel`** [wiring]: wires resampling layers (cross-attention to learned queries)
  - **`Ernie4_5_VLMoeModel`** [wiring]: wires `Ernie4_5_VLMoeTextModel`, `Ernie4_5_VLMoeVisionTransformerPretrainedModel`, `Ernie4_5_VLMoeVariableResolutionResamplerModel`
  - **`Ernie4_5_VLMoeForConditionalGeneration`** [wiring]: wires `Ernie4_5_VLMoeModel`; direct `L1/linear.py` (lm_head)
  - **`Ernie4_5_VL_MoeForConditionalGeneration`** / **`Ernie4_5_VL_MoePreTrainedModel`** / **`Ernie4_5_VL_MoeModel`** / **`Ernie4_5_VL_MoeTextModel`** / **`Ernie4_5_VL_MoeVisionTransformerPretrainedModel`** / **`Ernie4_5_VL_MoeVariableResolutionResamplerModel`** [wiring, simple aliases]: same kernels as their non-underscored siblings

## esm
- **src**: modeling_esm.py
- **hidden_act**: gelu
- **status**: composable (BERT-style protein language model with rotary embedding)
- **classes**:
  - **`EsmRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (Llama-style RoPE)
  - **`EsmContactPredictionHead`** [compute]: `L1/linear.py + L1/sigmoid.py` (contact prediction; symmetric)
  - **`EsmEmbeddings`** [compute]: `L1/embedding.py` (word + position) + optional `L1/layer_norm.py` (no token_type by default — slimmer than `L2/encoder_embeddings.py`)
  - **`EsmSelfAttention`** [compute]: `L2/encoder_attention.py` (q/k/v + dispatch with optional rotary on q,k)
  - **`EsmSelfOutput`** [compute]: `L1/linear.py` (dense) + dropout (no LayerNorm in this class)
  - **`EsmAttention`** [wiring]: wires `EsmSelfAttention`, `EsmSelfOutput`; direct `L1/layer_norm.py` (LayerNorm here, pre-attn)
  - **`EsmIntermediate`** [compute]: `L1/linear.py + L1/gelu.py`
  - **`EsmOutput`** [compute]: `L1/linear.py` (dense) + residual (no LayerNorm in this class)
  - **`EsmLayer`** [wiring, inherits `GradientCheckpointingLayer`]: wires `EsmAttention`, `EsmIntermediate`, `EsmOutput`; direct `L1/layer_norm.py` (LayerNorm pre-MLP)
  - **`EsmEncoder`** [wiring]: wires `EsmLayer`; direct `L1/layer_norm.py` (final emb_layer_norm_after)
  - **`EsmPooler`** [compute]: `L1/linear.py + L1/tanh.py`
  - **`EsmModel`** [wiring]: wires `EsmEmbeddings`, `EsmEncoder`, optional `EsmPooler`, `EsmContactPredictionHead`
  - **`EsmForMaskedLM`** [wiring]: wires `EsmModel`, `EsmLMHead`
  - **`EsmLMHead`** [compute]: `L1/linear.py + L1/gelu.py + L1/layer_norm.py + L1/linear.py` (decoder)
  - **`EsmClassificationHead`** [compute]: `L1/linear.py + L1/tanh.py + L1/linear.py`
- **task heads (2)**: ForSequenceClassification, ForTokenClassification — base + classification head

## eurobert
- **src**: modeling_eurobert.py (and modular_eurobert.py — inherits Llama)
- **hidden_act**: silu
- **status**: composable (Llama-decoder used as bidirectional encoder)
- **classes**:
  - **`EuroBertRMSNorm`** [compute, inherits `LlamaRMSNorm`]: `L1/rms_norm.py`
  - **`EuroBertAttention`** [compute, inherits `LlamaAttention`]: `L2/attention.py` (Llama-style)
  - **`EuroBertMLP`** [compute]: `L2/llama_mlp.py` (SwiGLU)
  - **`EuroBertDecoderLayer`** [wiring, inherits `GradientCheckpointingLayer`]: wires `EuroBertRMSNorm` (×2), `EuroBertAttention`, `EuroBertMLP`
  - **`EuroBertRotaryEmbedding`** [compute]: `L1/rotary_emb.py`
  - **`EuroBertModel`** [wiring, inherits `LlamaModel`]: wires `EuroBertDecoderLayer`, `EuroBertRMSNorm`, `EuroBertRotaryEmbedding`; direct `L1/embedding.py`
  - **`EuroBertForMaskedLM`** [wiring]: wires `EuroBertModel`; direct `L1/linear.py + L1/silu.py + L1/linear.py` (lm_head with intermediate projection)
- **task heads (2)**: ForSequenceClassification, ForTokenClassification — base + classification head

## evolla
- **src**: modeling_evolla.py (and modular_evolla.py)
- **hidden_act**: silu (Llama-like main model); relu/gelu in protein encoder
- **status**: partial (text decoder is Llama-clone → composable; SaProt protein encoder is ESM-style → composable; cross-attention adapter is bespoke)
- **classes**:
  - **`EvollaSaProtEmbeddings`** [compute]: `L1/embedding.py` (word + token_type) + position encoding (no exact L2 match — ESM-style with extra token_type)
  - **`EvollaSaProtRotaryEmbedding`** [compute]: `L1/rotary_emb.py`
  - **`EvollaSaProtSelfAttention`** [compute]: `L2/encoder_attention.py` (q/k/v with rotary on q,k)
  - **`EvollaSaProtSelfOutput`** [compute]: `L1/linear.py` (dense) + dropout (no LayerNorm here)
  - **`EvollaSaProtAttention`** [wiring]: wires `EvollaSaProtSelfAttention`, `EvollaSaProtSelfOutput`; direct `L1/layer_norm.py` (pre-attn)
  - **`EvollaSaProtIntermediate`** [compute]: `L1/linear.py + L1/gelu.py`
  - **`EvollaSaProtOutput`** [compute]: `L1/linear.py` (dense) + residual
  - **`EvollaSaProtLayer`** [wiring, inherits `GradientCheckpointingLayer`]: wires `EvollaSaProtAttention`, `EvollaSaProtIntermediate`, `EvollaSaProtOutput`; direct `L1/layer_norm.py` (pre-MLP)
  - **`EvollaSaProtEncoder`** [wiring]: wires `EvollaSaProtLayer`; direct `L1/layer_norm.py`
  - **`EvollaSaProtPooler`** [compute]: `L1/linear.py + L1/tanh.py`
  - **`EvollaSaProtProteinEncoder`** [wiring]: wires `EvollaSaProtEmbeddings`, `EvollaSaProtEncoder`, optional `EvollaSaProtPooler`
  - **`EvollaSequenceCompressorAttention`** [compute]: cross-attention `L1/linear.py` (q/k/v/o) + `L1/dense_attention.py` (no exact L2 match)
  - **`EvollaFeedForward`** [compute]: `L1/linear.py + L1/gelu.py + L1/linear.py`
  - **`EvollaSequenceCompressorResampler`** [wiring]: wires `EvollaSequenceCompressorAttention`, `EvollaFeedForward`; direct `L1/layer_norm.py` × 2
  - **`EvollaProteinEncoderModelOutput`** [skip] (Output dataclass)
  - **`EvollaProteinEncoder`** [wiring]: wires `EvollaSaProtProteinEncoder`, `EvollaSequenceCompressorResampler`
  - **`EvollaSequenceAlignerCrossAttention`** [compute]: cross-attn adapter inserted into the LLM `L1/linear.py` (q/k/v/o) + gate; `L1/rms_norm.py` (no exact L2 match)
  - **`EvollaRMSNorm`** [compute]: `L1/rms_norm.py`
  - **`EvollaRotaryEmbedding`** [compute]: `L1/rotary_emb.py`
  - **`EvollaMLP`** [compute]: `L2/llama_mlp.py` (SwiGLU)
  - **`EvollaAttention`** [compute]: `L2/attention.py` (Llama-style: GQA + RoPE + KV cache)
  - **`EvollaDecoderLayer`** [wiring, inherits `GradientCheckpointingLayer`]: wires `EvollaAttention`, `EvollaMLP`, `EvollaRMSNorm` × 2, optional `EvollaSequenceAlignerCrossAttention` (every-N layers)
  - **`EvollaModel`** [wiring]: wires `EvollaProteinEncoder`, `EvollaDecoderLayer`, `EvollaRMSNorm`, `EvollaRotaryEmbedding`; direct `L1/embedding.py`
  - **`EvollaForProteinText2Text`** [wiring]: wires `EvollaModel`; direct `L1/linear.py` (lm_head)

## exaone4
- **src**: modeling_exaone4.py (and modular_exaone4.py — inherits Llama / Olmo2 / Gemma2)
- **hidden_act**: silu
- **status**: composable (Llama-style with sliding-window-by-layer + Gemma2 RoPE)
- **classes**:
  - **`Exaone4RMSNorm`** [compute, inherits `LlamaRMSNorm`]: `L1/rms_norm.py`
  - **`Exaone4RotaryEmbedding`** [compute, inherits `Gemma2RotaryEmbedding`]: `L1/rotary_emb.py` (Gemma2's RoPE matches default Llama RoPE here; not Gemma2-specific RoPE kernel)
  - **`Exaone4Attention`** [compute]: `L2/attention.py` (with qk_norm + sliding_window per layer_type, GQA + RoPE + KV cache)
  - **`Exaone4MLP`** [compute, inherits `Olmo2MLP`]: `L2/llama_mlp.py` (SwiGLU)
  - **`Exaone4DecoderLayer`** [wiring, inherits `Olmo2DecoderLayer`/`GradientCheckpointingLayer`]: wires `Exaone4Attention`, `Exaone4MLP`, `Exaone4RMSNorm` × 2
  - **`Exaone4Model`** [wiring, inherits `LlamaModel`]: wires `Exaone4DecoderLayer`, `Exaone4RMSNorm`, `Exaone4RotaryEmbedding`; direct `L1/embedding.py`
  - **`Exaone4ForCausalLM`** [wiring, inherits `LlamaForCausalLM`]: wires `Exaone4Model`; direct `L1/linear.py` (lm_head)
- **task heads (3)**: ForSequenceClassification, ForTokenClassification, ForQuestionAnswering — base + linear (per-task)

## exaone4_5
- **src**: modeling_exaone4_5.py (and modular_exaone4_5.py — inherits Qwen2.5-VL family + Exaone4 text)
- **hidden_act**: silu
- **status**: partial (multimodal VL model; vision is Qwen2.5-VL-clone, text is Exaone4-clone — both composable but composed)
- **classes**:
  - **`Exaone4_5_PatchEmbed`** [compute, inherits `Qwen2_5_VisionPatchEmbed`]: `L1/conv3d.py` (or 2D variant)
  - **`Exaone4_5_VisionRotaryEmbedding`** [compute, inherits `Qwen2_5_VisionRotaryEmbedding`]: `L1/vision_rotary_emb.py`
  - **`Exaone4_5_RMSNorm`** [compute]: `L1/rms_norm.py`
  - **`Exaone4_5_PatchMerger`** [compute, inherits `Qwen2_5_VLPatchMerger`]: `L1/linear.py + L1/gelu.py + L1/linear.py + L1/layer_norm.py` (closest L2: `L2/vision_patch_merger.py`)
  - **`Exaone4_5_VisionAttention`** [compute, inherits `Qwen2_5_VLVisionAttention`]: `L1/linear.py` (q/k/v/o) + `L1/vision_rotary_emb.py` + `L1/dense_attention.py` (no exact L2 match — closest `L2/vision_attention.py`)
  - **`Exaone4_5_MLP`** [compute, inherits `Qwen2_5_VLMLP`]: `L2/llama_mlp.py` (SwiGLU)
  - **`Exaone4_5_VisionBlock`** [wiring, inherits `Qwen2_5_VLVisionBlock`/`GradientCheckpointingLayer`]: wires `Exaone4_5_VisionAttention`, `Exaone4_5_MLP`, `L1/rms_norm.py` × 2
  - **`Exaone4_5_Attention`** [compute]: `L2/attention.py` (Llama-style)
  - **`Exaone4_5_DecoderLayer`** [wiring, inherits `GradientCheckpointingLayer`]: wires `Exaone4_5_Attention`, `Exaone4_5_MLP`, `Exaone4_5_RMSNorm` × 2
  - **`Exaone4_5_VisionModel`** [wiring, inherits `Qwen2_5_VisionTransformerPretrainedModel`]: wires `Exaone4_5_PatchEmbed`, `Exaone4_5_VisionBlock`, `Exaone4_5_VisionRotaryEmbedding`, `Exaone4_5_PatchMerger`
  - **`Exaone4_5_Model`** [wiring, inherits `Qwen2_5_VLModel`]: wires `Exaone4_5_VisionModel`, `Exaone4_5_DecoderLayer`, `Exaone4_5_RMSNorm`; direct `L1/embedding.py`
  - **`Exaone4_5_ForConditionalGeneration`** [wiring]: wires `Exaone4_5_Model`; direct `L1/linear.py` (lm_head)

## exaone_moe
- **src**: modeling_exaone_moe.py (and modular_exaone_moe.py — inherits Exaone4Attention, Qwen2MoeMLP, DeepseekV3 router/MoE, OlmoeDecoderLayer)
- **hidden_act**: silu
- **status**: composable (Exaone4 + DeepSeek-V3 style MoE)
- **classes**:
  - **`ExaoneMoeRMSNorm`** [compute]: `L1/rms_norm.py`
  - **`ExaoneMoeAttention`** [compute, inherits `Exaone4Attention`]: `L2/attention.py` (with qk_norm + sliding_window per layer)
  - **`ExaoneMoeMLP`** [compute, inherits `Qwen2MoeMLP`]: `L2/llama_mlp.py` (SwiGLU)
  - **`ExaoneMoeTopkRouter`** [compute, inherits `DeepseekV3TopkRouter`]: `L1/linear.py` (router) + group-limited topk
  - **`ExaoneMoeExperts`** [compute, inherits `DeepseekV3NaiveMoe`]: grouped-experts (`L1/moe_grouped_gemm.py` is the closest, but DeepSeek-V3 naive is per-expert loop)
  - **`ExaoneMoeSparseMoEBlock`** [wiring, inherits `DeepseekV3MoE`]: `L2/shared_expert_moe.py` (DeepSeek-V3 pattern: routed experts + shared expert)
  - **`ExaoneMoeDecoderLayer`** [wiring, inherits `OlmoeDecoderLayer`/`GradientCheckpointingLayer`]: wires `ExaoneMoeAttention`, `ExaoneMoeSparseMoEBlock`, `ExaoneMoeRMSNorm` × 2
  - **`ExaoneMoeRotaryEmbedding`** [compute]: `L1/rotary_emb.py`
  - **`ExaoneMoeModel`** [wiring, inherits `Exaone4Model`]: wires `ExaoneMoeDecoderLayer`, `ExaoneMoeRMSNorm`, `ExaoneMoeRotaryEmbedding`; direct `L1/embedding.py`
  - **`ExaoneMoeForCausalLM`** [wiring, inherits `Exaone4ForCausalLM`]: wires `ExaoneMoeModel`; direct `L1/linear.py` (lm_head)
