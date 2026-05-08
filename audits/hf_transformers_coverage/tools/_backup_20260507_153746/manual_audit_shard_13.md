# Manual audit shard 13 (sam3_tracker_video … swiftformer)

## sam3_tracker_video
- **src**: modeling_sam3_tracker_video.py, modular_sam3_tracker_video.py (most classes inherit from sam2_video equivalents via modular)
- **hidden_act**: gelu (general); memory_attention_feed_forward_hidden_act=relu; mask_downsampler_hidden_act=gelu; memory_fuser_hidden_act=gelu
- **status**: kb_nano_l4 (L4/sam3_tracker.py wraps the SAM3 tracker pipeline; nearly identical to SAM2 tracker which this folder inherits)
- **classes**:
  - **`Sam3TrackerVideoLayerNorm`** [compute, inherits `Sam2VideoLayerNorm`/`nn.LayerNorm`]: `L1/layer_norm.py` (channels-first/-last LayerNorm wrapper)
  - **`Sam3TrackerVideoPositionEmbeddingSine`** [compute]: matches `L2/sam3_memory_encoder.py::PositionEmbeddingSine` (sine positional encoding, no kernel beyond arithmetic)
  - **`Sam3TrackerVideoAttention`** [compute]: q/k/v/o `L1/linear.py` + `L1/dense_attention.py` (eager mha with scaling, no RoPE) — closest match `L2/sam3_cross_attention.py`
  - **`Sam3TrackerVideoTwoWayAttentionBlock`** [wiring, inherits `Sam2VideoTwoWayAttentionBlock`]: wires `Sam3TrackerVideoAttention` (×3 self+2 cross), `Sam3TrackerVideoFeedForward`, 4× `nn.LayerNorm` (`L1/layer_norm.py`); covered by `L3/sam3_mask_decoder.py::TwoWayAttentionBlock`
  - **`Sam3TrackerVideoFeedForward`** [compute]: `L1/linear.py` + activation (`L1/relu.py` default; configurable) + optional `L1/sigmoid.py` — matches `L3/sam3_mask_decoder.py::MLP`
  - **`Sam3TrackerVideoVisionRotaryEmbedding`** [compute]: 2D axial RoPE precomputed cos/sin buffer — matches `L1/sam3_rope.py` / `L1/vision_rotary_emb.py`
  - **`Sam3TrackerVideoRoPEAttention`** [compute]: q/k/v/o `L1/linear.py` + 2D pairwise RoPE + `L1/dense_attention.py` — matches `L1/sam3_rope_attention.py`
  - **`Sam3TrackerVideoMemoryAttentionLayer`** [wiring]: wires `Sam3TrackerVideoRoPEAttention` (self + cross), `nn.Linear` ×2, `nn.LayerNorm` ×3, `relu` activation; covered by `L3/sam3_memory_attention.py::Sam3MemoryAttentionLayer`
  - **`Sam3TrackerVideoMemoryAttention`** [wiring]: wires `Sam3TrackerVideoMemoryAttentionLayer` ×N, `nn.LayerNorm`, `Sam3TrackerVideoVisionRotaryEmbedding`; covered by `L3/sam3_memory_attention.py::Sam3MemoryAttention`
  - **`Sam3TrackerVideoMemoryFuserCXBlock`** [compute]: depthwise `L1/conv2d.py` + `Sam3TrackerVideoLayerNorm` + `L1/linear.py` ×2 + `L1/gelu.py` + scale param; matches `L2/sam3_memory_encoder.py::CXBlock`
  - **`Sam3TrackerVideoMemoryFuser`** [wiring]: wires `Sam3TrackerVideoMemoryFuserCXBlock` ×N; matches `L2/sam3_memory_encoder.py::SimpleFuser`
  - **`Sam3TrackerVideoMaskDownSamplerLayer`** [compute]: `L1/conv2d.py` + `Sam3TrackerVideoLayerNorm` + `L1/gelu.py`
  - **`Sam3TrackerVideoMaskDownSampler`** [wiring]: wires `Sam3TrackerVideoMaskDownSamplerLayer` ×N + final `L1/conv2d.py`; matches `L2/sam3_memory_encoder.py::SimpleMaskDownSampler`
  - **`Sam3TrackerVideoMemoryEncoder`** [wiring]: wires `Sam3TrackerVideoMaskDownSampler`, `Sam3TrackerVideoMemoryFuser`, `Sam3TrackerVideoPositionEmbeddingSine`; direct `L1/conv2d.py` ×2; matches `L2/sam3_memory_encoder.py::Sam3MemoryEncoder`
  - **`Sam3TrackerVideoPositionalEmbedding`** [compute]: `register_buffer` + sin/cos projection (no kernel match; arithmetic over learned 2-D matrix)
  - **`Sam3TrackerVideoMaskEmbedding`** [compute]: `L1/conv2d.py` ×3 + `L1/layer_norm.py` ×2 + `L1/gelu.py` ×2
  - **`Sam3TrackerVideoPromptEncoder`** [wiring]: wires `Sam3TrackerVideoPositionalEmbedding`, `Sam3TrackerVideoMaskEmbedding`; direct `L1/embedding.py` ×3; matches `L2/sam3_prompt_encoder.py::Sam3PromptEncoder`
  - **`Sam3TrackerVideoTwoWayTransformer`** [wiring]: wires `Sam3TrackerVideoTwoWayAttentionBlock` ×N, `Sam3TrackerVideoAttention`, `nn.LayerNorm`; covered by `L3/sam3_mask_decoder.py::TwoWayTransformer`
  - **`Sam3TrackerVideoMaskDecoder`** [wiring]: wires `Sam3TrackerVideoTwoWayTransformer`, `Sam3TrackerVideoFeedForward` ×N; direct `L1/embedding.py` ×3, `L1/conv_transpose2d.py` ×2, `L1/conv2d.py` ×2, `Sam3TrackerVideoLayerNorm`, `L1/gelu.py`; matches `L3/sam3_mask_decoder.py::Sam3MaskDecoder`
  - **`Sam3TrackerVideoModel`** [wiring]: top-level pipeline; wires `Sam3TrackerVideoMemoryAttention`, `Sam3TrackerVideoMemoryEncoder`, `Sam3TrackerVideoPromptEncoder`, `Sam3TrackerVideoMaskDecoder`, vision encoder via `AutoModel`; covered by `L4/sam3_tracker.py::Sam3TrackerBase`/`Sam3TrackerPredictor`

## sam3_video
- **src**: modeling_sam3_video.py (no modular)
- **hidden_act**: n/a (composition only — sub-models bring their own activations)
- **status**: kb_nano_l4 (covered by `L4/sam3_video.py::Sam3VideoModel`)
- **classes**:
  - **`Sam3VideoModel`** [wiring]: top-level video pipeline; wires `detector_model` (`AutoModel` → SAM3 detector), `tracker_model` (`AutoModel` → Sam3TrackerVideoModel), and `Sam3VisionNeck` (imported from sam3 modeling); covered by `L4/sam3_video.py::Sam3VideoModel` plus `L4/sam3_tracker.py` and `L3/sam3_neck.py`

## sam_hq
- **src**: modeling_sam_hq.py, modular_sam_hq.py (much inherits from `..sam.modeling_sam` SAM)
- **hidden_act**: vision=gelu, prompt_encoder=gelu, mask_decoder=relu
- **status**: partial (uses windowed ViT-Det attention with relative-position bias and SAM mask decoder; closest kb-nano analog is the SAM3 stack — `L4/sam3_tracker.py` and `L2/sam3_*` — but no exact SAM-HQ pipeline)
- **classes**:
  - **`SamHQVisionAttention`** [compute, inherits `SamVisionAttention`]: `L1/linear.py` (qkv, proj) + decomposed rel-pos bias + `L1/dense_attention.py` (manual softmax+matmul) — closest match `L2/sam3_vit_attention.py` (different rel-pos handling)
  - **`SamHQMLPBlock`** [compute]: `L1/linear.py` ×2 + `L1/gelu.py` (fc1→gelu→fc2) — matches `L2/sam3_vit_mlp.py`
  - **`SamHQVisionSdpaAttention`** [compute, inherits `SamHQVisionAttention`]: `L1/linear.py` ×2 + decomposed rel-pos bias + `L1/sdpa.py`/`L1/dense_attention.py` (calls `F.scaled_dot_product_attention`)
  - **`SamHQVisionLayer`** [wiring, inherits `SamVisionLayer`]: wires `nn.LayerNorm` ×2 (`L1/layer_norm.py`), `SamHQVisionAttention`/`SamHQVisionSdpaAttention`, `SamHQMLPBlock`; window partition is reshape arithmetic (no kernel)
  - **`SamHQPositionalEmbedding`** [compute]: scaled random projection; `register_buffer`; arithmetic-only sin/cos (no kernel match)
  - **`SamHQPatchEmbeddings`** [compute]: `L1/conv2d.py` (patch projection) — matches `L2/vision_patch_embed.py`
  - **`SamHQVisionNeck`** [compute]: `L1/conv2d.py` ×2 + `SamHQLayerNorm` ×2 — covered by `L3/sam3_neck.py::Sam3VisionNeck`
  - **`SamHQVisionEncoder`** [wiring, inherits `SamVisionEncoder`+`SamHQPreTrainedModel`]: wires `SamHQPatchEmbeddings`, `SamHQVisionLayer` ×N, `SamHQVisionNeck`; direct `nn.Parameter` pos_embed
  - **`SamHQLayerNorm`** [compute, inherits `SamLayerNorm`/`nn.LayerNorm`]: `L1/layer_norm.py` (channels-first/-last LayerNorm wrapper)
  - **`SamHQAttention`** [compute]: q/k/v/o `L1/linear.py` + `L1/dense_attention.py` (eager softmax MHA) — matches `L2/sam3_cross_attention.py`
  - **`SamHQTwoWayAttentionBlock`** [wiring]: wires `SamHQAttention` ×3, `SamHQMLPBlock`, `nn.LayerNorm` ×4; covered by `L3/sam3_mask_decoder.py::TwoWayAttentionBlock`
  - **`SamHQTwoWayTransformer`** [wiring, inherits `SamTwoWayTransformer`]: wires `SamHQTwoWayAttentionBlock` ×N, `SamHQAttention`, `nn.LayerNorm`
  - **`SamHQFeedForward`** [compute, inherits `SamFeedForward`]: `L1/linear.py` ×N + `L1/relu.py` + optional `L1/sigmoid.py` — matches `L3/sam3_mask_decoder.py::MLP`
  - **`SamHQMaskDecoder`** [wiring]: HQ extension of SAM mask decoder; wires `SamHQTwoWayTransformer`, `SamHQFeedForward` ×(num_mask_tokens), HQ token MLP, `nn.Embedding` ×3, `nn.ConvTranspose2d` ×4 (`L1/conv_transpose2d.py`), `nn.Conv2d` ×2 (`L1/conv2d.py`), `SamHQLayerNorm`, `nn.GELU`; extends `L3/sam3_mask_decoder.py::Sam3MaskDecoder`
  - **`SamHQVisionModel`** [wiring, inherits `SamVisionModel`]: wires `SamHQVisionEncoder`
  - **`SamHQMaskEmbedding`** [compute]: `L1/conv2d.py` ×3 + `SamHQLayerNorm` ×2 + `L1/gelu.py` ×2
  - **`SamHQPromptEncoder`** [wiring]: wires `SamHQPositionalEmbedding`, `SamHQMaskEmbedding`; direct `nn.Embedding` (`L1/embedding.py`) ×N (point_embed list + no_mask_embed + not_a_point_embed)
  - **`SamHQModel`** [wiring, inherits `SamModel`]: top-level; wires `SamHQPositionalEmbedding`, `SamHQVisionEncoder`, `SamHQPromptEncoder`, `SamHQMaskDecoder`

## seamless_m4t
- **src**: modeling_seamless_m4t.py (no modular)
- **hidden_act**: speech_encoder_hidden_act=swish (silu); activation_function=relu (text encoder/decoder); intermediate_ffn act_fn=relu; adapter ffn act_fn=relu; variance predictor=ReLU; HiFi-GAN=leaky_relu
- **status**: unsupported (multilingual S2S/T2T translation with conformer speech encoder + Transformer text enc/dec + T2U + HiFi-GAN; no kb-nano pipeline)
- **classes**:
  - **`SeamlessM4TConformerPositionalConvEmbedding`** [compute]: `L1/conv1d.py` (with weight_norm) + `SamePadLayer` slice + `L1/silu.py` activation
  - **`SeamlessM4TConformerRotaryPositionalEmbedding`** [compute]: cached cos/sin RoPE buffer (different layout vs Llama; no exact L2 match) — closest `L1/rotary_emb.py`
  - **`SeamlessM4TConformerRelPositionalEmbedding`** [compute]: relative-position sinusoid PE buffer (Transformer-XL style); arithmetic-only
  - **`SeamlessM4TConformerSamePadLayer`** [compute]: tensor slicing only (no kernel)
  - **`SeamlessM4TConformerFeatureProjection`** [compute]: `L1/layer_norm.py` + `L1/linear.py` (+ dropout)
  - **`SeamlessM4TConformerFeedForward`** [compute]: `L1/linear.py` ×2 + `L1/silu.py` (or other ACT2FN) — fc1→act→fc2 — no exact L2 match (closest `L2/encoder_mlp.py`)
  - **`SeamlessM4TConformerConvolutionModule`** [compute]: `L1/layer_norm.py` + `L1/conv1d.py` (pointwise ×2) + GLU + depthwise `L1/conv1d.py` + nn.BatchNorm1d + `L1/silu.py` (no exact L2 match)
  - **`SeamlessM4TConformerSelfAttention`** [compute]: `L1/linear.py` ×4 + manual softmax + matmul attention with optional rotary or relative-position handling — no exact kb-nano match
  - **`SeamlessM4TConformerEncoderLayer`** [wiring]: wires `SeamlessM4TConformerFeedForward` ×2, `SeamlessM4TConformerSelfAttention`, `SeamlessM4TConformerConvolutionModule`, `nn.LayerNorm` ×4 (×0.5 macaron-FFN sandwich)
  - **`SeamlessM4TConformerEncoder`** [wiring]: wires positional emb, `SeamlessM4TConformerEncoderLayer` ×N, `nn.LayerNorm`
  - **`SeamlessM4TConformerAdapterLayer`** [wiring]: wires `nn.Conv1d` ×2 (residual + self_attn_conv) + `nn.GLU` + `SeamlessM4TConformerSelfAttention` + `SeamlessM4TConformerFeedForward`; `nn.LayerNorm` ×3
  - **`SeamlessM4TConformerAdapter`** [wiring]: wires `SeamlessM4TConformerAdapterLayer` ×N
  - **`SeamlessM4TScaledWordEmbedding`** [compute]: `L1/embedding.py` × scalar scale
  - **`SeamlessM4TSinusoidalPositionalEmbedding`** [compute]: precomputed sinusoid weights + index_select; arithmetic-only
  - **`SeamlessM4TAttention`** [compute]: BART-style q/k/v/o `L1/linear.py` + manual bmm-softmax-bmm (eager only); used as self/cross attention with `EncoderDecoderCache` — no exact kb-nano match (closest `L2/whisper_attention.py`)
  - **`SeamlessM4TFeedForwardNetwork`** [compute]: `L1/linear.py` ×2 + `L1/relu.py` (fc1→act→fc2; no exact L2 match)
  - **`SeamlessM4TEncoderLayer`** [wiring]: wires `SeamlessM4TAttention` (self), `SeamlessM4TFeedForwardNetwork`, `nn.LayerNorm` ×2
  - **`SeamlessM4TDecoderLayer`** [wiring]: wires `SeamlessM4TAttention` ×2 (self+cross), `SeamlessM4TFeedForwardNetwork`, `nn.LayerNorm` ×3
  - **`SeamlessM4TSpeechEncoder`** [wiring]: wires `SeamlessM4TConformerFeatureProjection`, `SeamlessM4TConformerEncoder`, `SeamlessM4TConformerFeedForward` (intermediate_ffn), `SeamlessM4TConformerAdapter` (optional), `nn.LayerNorm`
  - **`SeamlessM4TEncoder`** [wiring]: wires `SeamlessM4TScaledWordEmbedding`, `SeamlessM4TSinusoidalPositionalEmbedding`, `SeamlessM4TEncoderLayer` ×N, `nn.LayerNorm`
  - **`SeamlessM4TDecoder`** [wiring]: wires `SeamlessM4TScaledWordEmbedding`, `SeamlessM4TSinusoidalPositionalEmbedding`, `SeamlessM4TDecoderLayer` ×N, `nn.LayerNorm`
  - **`SeamlessM4TTextToUnitModel`** [wiring]: wires `SeamlessM4TEncoder` (t2u_encoder=True), `SeamlessM4TDecoder`
  - **`SeamlessM4TTextToUnitForConditionalGeneration`** [wiring]: wires `SeamlessM4TTextToUnitModel`; direct `L1/linear.py` (lm_head)
  - **`HifiGanResidualBlock`** [compute]: dilated `L1/conv1d.py` ×2 stacks + `L1/leaky_relu.py` + residual
  - **`SeamlessM4TVariancePredictor`** [compute]: `L1/conv1d.py` ×2 + `L1/relu.py` ×2 + `L1/layer_norm.py` ×2 + `L1/linear.py` (proj)
  - **`SeamlessM4THifiGan`** [wiring]: wires `nn.Conv1d` (conv_pre, conv_post), `nn.ConvTranspose1d` ×N upsampler (`L1/conv_transpose1d.py`), `HifiGanResidualBlock` ×N, `L1/leaky_relu.py`, `L1/tanh.py`
  - **`SeamlessM4TCodeHifiGan`** [wiring]: wires `SeamlessM4TVariancePredictor`, `nn.Embedding` ×3 (unit/speaker/language), `SeamlessM4THifiGan`
  - **`SeamlessM4TForTextToText`** [wiring]: wires `SeamlessM4TEncoder`, `SeamlessM4TDecoder`; direct `L1/linear.py` (lm_head); shared `SeamlessM4TScaledWordEmbedding`
  - **`SeamlessM4TForSpeechToText`** [wiring]: wires `SeamlessM4TSpeechEncoder`, `SeamlessM4TDecoder`; direct `L1/linear.py` (lm_head)
  - **`SeamlessM4TForTextToSpeech`** [wiring]: wires `SeamlessM4TEncoder`, `SeamlessM4TDecoder`, `SeamlessM4TTextToUnitForConditionalGeneration`, `SeamlessM4TCodeHifiGan`; direct `L1/linear.py` (lm_head)
  - **`SeamlessM4TForSpeechToSpeech`** [wiring]: wires `SeamlessM4TSpeechEncoder`, `SeamlessM4TDecoder`, `SeamlessM4TTextToUnitForConditionalGeneration`, `SeamlessM4TCodeHifiGan`; direct `L1/linear.py` (lm_head)
  - **`SeamlessM4TModel`** [wiring]: wires `SeamlessM4TSpeechEncoder`, `SeamlessM4TEncoder`, `SeamlessM4TDecoder`, `SeamlessM4TTextToUnitForConditionalGeneration`, `SeamlessM4TCodeHifiGan`; direct `L1/linear.py` (lm_head)

## seamless_m4t_v2
- **src**: modeling_seamless_m4t_v2.py (no modular)
- **hidden_act**: speech_encoder_hidden_act=swish (silu); activation_function=relu
- **status**: unsupported (extends seamless_m4t with new T2U decoder using conv blocks; no kb-nano pipeline)
- **classes**:
  - **`SeamlessM4Tv2ConformerFeatureProjection`** [compute]: `L1/layer_norm.py` + `L1/linear.py`
  - **`SeamlessM4Tv2ConformerFeedForward`** [compute]: `L1/linear.py` ×2 + `L1/silu.py` (no exact L2 match)
  - **`SeamlessM4Tv2ConformerConvolutionModule`** [compute]: `L1/layer_norm.py` + `L1/conv1d.py` ×3 (pointwise + depthwise + pointwise) + GLU + nn.BatchNorm1d + `L1/silu.py`
  - **`SeamlessM4Tv2ConformerSelfAttention`** [compute]: `L1/linear.py` ×4 (q/k/v/o) + manual softmax + matmul attention with optional shaw-style relative-position
  - **`SeamlessM4Tv2ConformerEncoderLayer`** [wiring]: wires `SeamlessM4Tv2ConformerFeedForward` ×2, `SeamlessM4Tv2ConformerSelfAttention`, `SeamlessM4Tv2ConformerConvolutionModule`, `nn.LayerNorm` ×4 (macaron sandwich)
  - **`SeamlessM4Tv2ConformerEncoder`** [wiring]: wires `SeamlessM4Tv2ConformerEncoderLayer` ×N, `nn.LayerNorm`
  - **`SeamlessM4Tv2ConformerAdapterLayer`** [wiring]: wires `nn.Conv1d` ×2 (residual+self_attn_conv), `nn.GLU` ×2, `SeamlessM4Tv2ConformerSelfAttention`, `SeamlessM4Tv2ConformerFeedForward`, `nn.LayerNorm` ×3
  - **`SeamlessM4Tv2ConformerAdapter`** [wiring]: wires `SeamlessM4Tv2ConformerAdapterLayer` ×N
  - **`SeamlessM4Tv2ScaledWordEmbedding`** [compute]: `L1/embedding.py` × scalar scale
  - **`SeamlessM4Tv2SinusoidalPositionalEmbedding`** [compute]: precomputed sinusoid + index_select
  - **`SeamlessM4Tv2Attention`** [compute]: BART-style `L1/linear.py` ×4 + manual bmm-softmax-bmm with `EncoderDecoderCache`
  - **`SeamlessM4Tv2FeedForwardNetwork`** [compute]: `L1/linear.py` ×2 + `L1/relu.py`
  - **`SeamlessM4Tv2EncoderLayer`** [wiring]: wires `SeamlessM4Tv2Attention`, `SeamlessM4Tv2FeedForwardNetwork`, `nn.LayerNorm` ×2
  - **`SeamlessM4Tv2DecoderLayer`** [wiring]: wires `SeamlessM4Tv2Attention` ×2 (self+cross), `SeamlessM4Tv2FeedForwardNetwork`, `nn.LayerNorm` ×3
  - **`SeamlessM4Tv2TextToUnitDecoderLayer`** [wiring]: wires `SeamlessM4Tv2Attention` (self only), `nn.Conv1d` ×2 (`L1/conv1d.py`), `L1/relu.py`, `nn.LayerNorm` ×2 (post-norm style)
  - **`SeamlessM4Tv2SpeechEncoder`** [wiring]: wires `SeamlessM4Tv2ConformerFeatureProjection`, `SeamlessM4Tv2ConformerEncoder`, `SeamlessM4Tv2ConformerFeedForward` (intermediate), `SeamlessM4Tv2ConformerAdapter`, `nn.LayerNorm`
  - **`SeamlessM4Tv2Encoder`** [wiring]: wires `SeamlessM4Tv2ScaledWordEmbedding`, `SeamlessM4Tv2SinusoidalPositionalEmbedding`, `SeamlessM4Tv2EncoderLayer` ×N, `nn.LayerNorm`
  - **`SeamlessM4Tv2Decoder`** [wiring]: wires `SeamlessM4Tv2ScaledWordEmbedding`, `SeamlessM4Tv2SinusoidalPositionalEmbedding`, `SeamlessM4Tv2DecoderLayer` ×N, `nn.LayerNorm`
  - **`SeamlessM4Tv2TextToUnitDecoder`** [wiring]: wires `SeamlessM4Tv2ScaledWordEmbedding`, `SeamlessM4Tv2SinusoidalPositionalEmbedding`, `SeamlessM4Tv2TextToUnitDecoderLayer` ×N, `nn.LayerNorm`
  - **`SeamlessM4Tv2TextToUnitModel`** [wiring]: wires `SeamlessM4Tv2Encoder` (t2u), `SeamlessM4Tv2TextToUnitDecoder`, `SeamlessM4Tv2VariancePredictor`
  - **`SeamlessM4Tv2TextToUnitForConditionalGeneration`** [wiring]: wires `SeamlessM4Tv2TextToUnitModel`; direct `L1/linear.py` (lm_head)
  - **`HifiGanResidualBlock`** [compute]: same as v1 — dilated `L1/conv1d.py` ×N + `L1/leaky_relu.py` + residual
  - **`SeamlessM4Tv2VariancePredictor`** [compute]: `L1/conv1d.py` ×2 + `L1/relu.py` ×2 + `L1/layer_norm.py` ×2 + `L1/linear.py`
  - **`SeamlessM4Tv2HifiGan`** [wiring]: wires `nn.Conv1d` (pre/post), `nn.ConvTranspose1d` ×N, `HifiGanResidualBlock` ×N, `L1/leaky_relu.py`, `L1/tanh.py`
  - **`SeamlessM4Tv2CodeHifiGan`** [wiring]: wires `SeamlessM4Tv2VariancePredictor`, `nn.Embedding` ×3, `SeamlessM4Tv2HifiGan`
  - **`SeamlessM4Tv2ForTextToText`** [wiring]: wires `SeamlessM4Tv2Encoder`, `SeamlessM4Tv2Decoder`; direct `L1/linear.py` (lm_head)
  - **`SeamlessM4Tv2ForSpeechToText`** [wiring]: wires `SeamlessM4Tv2SpeechEncoder`, `SeamlessM4Tv2Decoder`; direct `L1/linear.py` (lm_head)
  - **`SeamlessM4Tv2ForTextToSpeech`** [wiring]: wires `SeamlessM4Tv2Encoder`, `SeamlessM4Tv2Decoder`, `SeamlessM4Tv2TextToUnitForConditionalGeneration`, `SeamlessM4Tv2CodeHifiGan`; direct `L1/linear.py`
  - **`SeamlessM4Tv2ForSpeechToSpeech`** [wiring]: wires `SeamlessM4Tv2SpeechEncoder`, `SeamlessM4Tv2Decoder`, `SeamlessM4Tv2TextToUnitForConditionalGeneration`, `SeamlessM4Tv2CodeHifiGan`; direct `L1/linear.py`
  - **`SeamlessM4Tv2Model`** [wiring]: wires all of the above

## seed_oss
- **src**: modeling_seed_oss.py, modular_seed_oss.py (inherits from llama)
- **hidden_act**: silu
- **status**: composable (Llama-family with GQA + RoPE; covered by `L4/llama.py` engine)
- **classes**:
  - **`SeedOssRMSNorm`** [compute, inherits `LlamaRMSNorm`]: `L1/rms_norm.py`
  - **`SeedOssMLP`** [compute]: SwiGLU (gate_proj + up_proj + down_proj + silu) — matches `L2/llama_mlp.py`
  - **`SeedOssAttention`** [compute]: q/k/v/o `L1/linear.py` + `L1/rotary_emb.py` + KV cache update + GQA attention — matches `L2/attention.py`
  - **`SeedOssDecoderLayer`** [wiring, inherits `LlamaDecoderLayer`]: wires `SeedOssAttention`, `SeedOssMLP`, `SeedOssRMSNorm` ×2
  - **`SeedOssRotaryEmbedding`** [compute]: standard Llama RoPE (default rope type) → `L1/rotary_emb.py`
  - **`SeedOssModel`** [wiring, inherits `LlamaModel`]: wires `nn.Embedding`, `SeedOssDecoderLayer` ×N, `SeedOssRMSNorm`, `SeedOssRotaryEmbedding`
  - **`SeedOssForCausalLM`** [wiring, inherits `LlamaForCausalLM`]: wires `SeedOssModel`; direct `L1/linear.py` (lm_head)
- **task heads (3)**: ForSequenceClassification, ForTokenClassification, ForQuestionAnswering — base + linear (per-task)

## segformer
- **src**: modeling_segformer.py, modular_segformer.py
- **hidden_act**: gelu
- **status**: partial (hierarchical efficient-self-attention vision transformer with mix-FFN; no exact kb-nano pipeline; primitives all available)
- **classes**:
  - **`SegformerDropPath`** [compute]: stochastic-depth drop_path (no kernel match; arithmetic+bernoulli)
  - **`SegformerOverlapPatchEmbeddings`** [compute]: `L1/conv2d.py` + `L1/layer_norm.py` (overlap stride conv + LN)
  - **`SegformerEfficientSelfAttention`** [compute]: q/k/v `L1/linear.py` + sequence-reduction via `L1/conv2d.py` + `L1/layer_norm.py` + manual softmax matmul attention (no exact kb-nano L2 match; PvT-style sequence-reduction)
  - **`SegformerSelfOutput`** [compute]: `L1/linear.py` (+ dropout) — half of an encoder output (no LayerNorm here, residual is in SegformerLayer)
  - **`SegformerAttention`** [wiring]: wires `SegformerEfficientSelfAttention` (self.self), `SegformerSelfOutput` (self.output) — BERT-style wrapper but residual handled in caller
  - **`SegformerDWConv`** [compute]: depthwise `L1/conv2d.py` (groups=dim)
  - **`SegformerMixFFN`** [compute]: `L1/linear.py` + `SegformerDWConv` + `L1/gelu.py` + `L1/linear.py` (Mix-FFN; no exact L2 match)
  - **`SegformerLayer`** [wiring]: wires `nn.LayerNorm` ×2 (`L1/layer_norm.py`), `SegformerAttention`, `SegformerMixFFN`, optional `SegformerDropPath`
  - **`SegformerEncoder`** [wiring]: wires `SegformerOverlapPatchEmbeddings` ×N, `SegformerLayer` ×N (per stage), `nn.LayerNorm` ×N
  - **`SegformerModel`** [wiring]: wires `SegformerEncoder`
  - **`SegformerMLP`** [compute]: `L1/linear.py` only (decode-head linear projection)
  - **`SegformerDecodeHead`** [wiring]: wires `SegformerMLP` ×N; direct `L1/conv2d.py` ×2 (linear_fuse + classifier), `L1/batch_norm2d.py`, `L1/relu.py`, `nn.functional.interpolate` (`L1/interpolate.py`)
  - **`SegformerForSemanticSegmentation`** [wiring]: wires `SegformerModel`, `SegformerDecodeHead`
- **task heads (1)**: ForImageClassification — base + linear (per-task)

## seggpt
- **src**: modeling_seggpt.py (no modular)
- **hidden_act**: gelu
- **status**: partial (ViT-style with relative-position bias and prompt-image segmentation; primitives available; no exact pipeline)
- **classes**:
  - **`SegGptPatchEmbeddings`** [compute]: `L1/conv2d.py` — matches `L2/vision_patch_embed.py`
  - **`SegGptEmbeddings`** [wiring]: wires `SegGptPatchEmbeddings`; direct `nn.Parameter` (mask_token, segment_token_input/prompt, type_token_semantic/instance, position_embeddings); arithmetic addition; `nn.functional.interpolate` (`L1/interpolate.py`) for pos
  - **`SegGptAttention`** [compute]: fused `L1/linear.py` (qkv) + decomposed rel-pos bias + manual softmax matmul attention + `L1/linear.py` (proj) — closest match `L2/sam3_vit_attention.py`
  - **`SegGptMlp`** [compute, copied from `SamMLPBlock`]: `L1/linear.py` ×2 + `L1/gelu.py` (fc1→gelu→fc2) — matches `L2/sam3_vit_mlp.py`
  - **`SegGptDropPath`** [compute]: stochastic-depth drop_path
  - **`SegGptLayer`** [wiring]: wires `SegGptAttention`, `SegGptMlp`, `nn.LayerNorm` ×2 (`L1/layer_norm.py`), optional `SegGptDropPath`
  - **`SegGptEncoder`** [wiring]: wires `SegGptLayer` ×N, `nn.LayerNorm`
  - **`SegGptLayerNorm`** [compute, copied from ConvNext]: `L1/layer_norm.py` (channels-first/-last wrapper)
  - **`SegGptDecoderHead`** [compute]: `L1/conv2d.py` ×2 + `SegGptLayerNorm` + `L1/gelu.py`
  - **`SegGptDecoder`** [wiring]: wires `SegGptDecoderHead`; direct `L1/linear.py` (decoder_embed)
  - **`SegGptModel`** [wiring]: wires `SegGptEmbeddings`, `SegGptEncoder`
  - **`SegGptForImageSegmentation`** [wiring]: wires `SegGptModel`, `SegGptDecoder`, `SegGptLoss` (skipped — Loss class)
- **task heads (0)**: ForImageSegmentation captured above (custom segmentation head, not generic For*)

## sew
- **src**: modeling_sew.py, modular_sew.py (inherits heavily from wav2vec2)
- **hidden_act**: gelu
- **status**: unsupported (Wav2Vec2-style speech encoder with squeeze-and-excite features; no kb-nano speech pipeline beyond Whisper)
- **classes**:
  - **`SEWNoLayerNormConvLayer`** [compute, inherits `Wav2Vec2NoLayerNormConvLayer`]: `L1/conv1d.py` + `L1/gelu.py`
  - **`SEWLayerNormConvLayer`** [compute, inherits `Wav2Vec2LayerNormConvLayer`]: `L1/conv1d.py` + `L1/layer_norm.py` + `L1/gelu.py`
  - **`SEWGroupNormConvLayer`** [compute, inherits `Wav2Vec2GroupNormConvLayer`]: `L1/conv1d.py` + `L1/group_norm.py` + `L1/gelu.py`
  - **`SEWPositionalConvEmbedding`** [compute]: `L1/conv1d.py` (with weight_norm) + `SEWSamePadLayer` + `L1/gelu.py`
  - **`SEWSamePadLayer`** [compute, inherits `Wav2Vec2SamePadLayer`]: tensor slicing only (no kernel)
  - **`SEWUpsampling`** [compute]: `L1/linear.py` + `L1/gelu.py` + reshape (custom upsample for SEW squeeze)
  - **`SEWFeatureEncoder`** [wiring, inherits `Wav2Vec2FeatureEncoder`]: wires `SEWGroupNormConvLayer`/`SEWLayerNormConvLayer`/`SEWNoLayerNormConvLayer` ×N
  - **`SEWAttention`** [compute, inherits `Wav2Vec2Attention`]: BART-style q/k/v/o `L1/linear.py` + manual bmm-softmax attention — no exact kb-nano match
  - **`SEWFeedForward`** [compute, inherits `Wav2Vec2FeedForward`]: `L1/linear.py` ×2 + `L1/gelu.py` (intermediate→activation→output, with dropouts)
  - **`SEWEncoderLayer`** [wiring, inherits `Wav2Vec2EncoderLayer`]: wires `SEWAttention`, `SEWFeedForward`, `nn.LayerNorm` ×2
  - **`SEWEncoder`** [wiring]: wires `SEWPositionalConvEmbedding`, `SEWUpsampling`, `nn.AvgPool1d` (`L1/avg_pool1d.py`), `SEWEncoderLayer` ×N, `nn.LayerNorm`
  - **`SEWModel`** [wiring]: wires `SEWFeatureEncoder`, `nn.LayerNorm`, `nn.Linear` (feature_projection), `SEWEncoder`
- **task heads (2)**: ForCTC, ForSequenceClassification — base + linear (per-task)

## sew_d
- **src**: modeling_sew_d.py (no modular)
- **hidden_act**: gelu_python
- **status**: unsupported (DeBERTa-disentangled-attention SEW variant; no kb-nano support for disentangled attention)
- **classes**:
  - **`SEWDNoLayerNormConvLayer`** [compute]: `L1/conv1d.py` + `L1/gelu.py`
  - **`SEWDLayerNormConvLayer`** [compute]: `L1/conv1d.py` + `L1/layer_norm.py` + `L1/gelu.py`
  - **`SEWDGroupNormConvLayer`** [compute]: `L1/conv1d.py` + `L1/group_norm.py` + `L1/gelu.py`
  - **`SEWDPositionalConvEmbedding`** [compute]: weight-normed `L1/conv1d.py` + `SEWDSamePadLayer` + `L1/gelu.py`
  - **`SEWDSamePadLayer`** [compute]: tensor slice (no kernel)
  - **`SEWDUpsampling`** [compute]: `L1/linear.py` + `L1/gelu.py` + reshape upsample
  - **`SEWDFeatureEncoder`** [wiring]: wires SEWDGroupNorm/LayerNorm/NoLayerNormConvLayer ×N
  - **`ContextPooler`** [compute]: `L1/linear.py` + StableDropout
  - **`StableDropout`** [compute]: dropout with optional context
  - **`SEWDSelfOutput`** [compute]: `L1/linear.py` + StableDropout + `L1/layer_norm.py` (+ residual) — half of an encoder output
  - **`DisentangledSelfAttention`** [compute]: `L1/linear.py` ×2 (in_proj + pos_proj_q/k via parameters), manual relative-position-attention bias matmul-softmax-matmul; XSoftmax (autograd) — no kb-nano match (DeBERTa disentangled attention)
  - **`SEWDAttention`** [wiring]: wires `DisentangledSelfAttention`, `SEWDSelfOutput`
  - **`SEWDIntermediate`** [compute]: `L1/linear.py` + `L1/gelu.py` (encoder intermediate half)
  - **`SEWDOutput`** [compute]: `L1/linear.py` + `L1/layer_norm.py` (+ residual)
  - **`SEWDLayer`** [wiring]: wires `SEWDAttention`, `SEWDIntermediate`, `SEWDOutput`
  - **`ConvLayer`** [compute]: `L1/conv1d.py` + `L1/tanh.py` + `L1/layer_norm.py` (+residual)
  - **`SEWDTransformerEncoder`** [wiring]: wires `SEWDLayer` ×N, optional `ConvLayer`, `nn.Embedding` (rel_embeddings), `LayerNorm`
  - **`SEWDEncoder`** [wiring]: wires `SEWDPositionalConvEmbedding`, `SEWDUpsampling`, `nn.AvgPool1d` (`L1/avg_pool1d.py`), `SEWDTransformerEncoder`, `nn.LayerNorm`
  - **`SEWDModel`** [wiring]: wires `SEWDFeatureEncoder`, `nn.LayerNorm`, `nn.Linear` (feature_projection), `SEWDEncoder`
- **task heads (2)**: ForCTC, ForSequenceClassification — base + linear (per-task)

## shieldgemma2
- **src**: modeling_shieldgemma2.py (no modular)
- **hidden_act**: n/a (wraps an `AutoModelForImageTextToText` — typically Gemma2-VL)
- **status**: composable (delegates to underlying VLM; if the inner model is Gemma3 or PaliGemma2, those map to existing kb-nano L4 pipelines via Gemma)
- **classes**:
  - **`ShieldGemma2ForImageClassification`** [wiring]: top-level safety classifier; wires inner `AutoModelForImageTextToText` (typically Gemma3/PaliGemma2); reads logits at `yes_token_index`/`no_token_index` to produce 2-way classification

## siglip
- **src**: modeling_siglip.py (no modular)
- **hidden_act**: gelu_pytorch_tanh
- **status**: composable (text/vision encoders via L2/siglip_attention.py + L2/siglip_mlp.py); no L4 yet but full primitives available
- **classes**:
  - **`SiglipVisionEmbeddings`** [compute]: `L1/conv2d.py` (patch_embedding) + `L1/embedding.py` (position_embedding) — matches `L2/vision_patch_embed.py`
  - **`SiglipTextEmbeddings`** [compute, copied from CLIP]: `L1/embedding.py` ×2 (token + position)
  - **`SiglipAttention`** [compute]: q/k/v/o `L1/linear.py` + `L1/dense_attention.py` (non-causal MHA) — matches `L2/siglip_attention.py`
  - **`SiglipMLP`** [compute, copied from CLIP]: `L1/linear.py` ×2 + `L1/gelu.py` (gelu_pytorch_tanh) — matches `L2/siglip_mlp.py`
  - **`SiglipEncoderLayer`** [wiring]: wires `nn.LayerNorm` ×2 (`L1/layer_norm.py`), `SiglipAttention`, `SiglipMLP`
  - **`SiglipEncoder`** [wiring]: wires `SiglipEncoderLayer` ×N
  - **`SiglipTextModel`** [wiring]: wires `SiglipTextEmbeddings`, `SiglipEncoder`, `nn.LayerNorm`; direct `L1/linear.py` (head)
  - **`SiglipVisionModel`** [wiring]: wires `SiglipVisionEmbeddings`, `SiglipEncoder`, `nn.LayerNorm`, optional `SiglipMultiheadAttentionPoolingHead`
  - **`SiglipMultiheadAttentionPoolingHead`** [compute]: `nn.MultiheadAttention` (closest `L2/siglip_attention.py` w/ probe query) + `nn.LayerNorm` + `SiglipMLP` (with probe parameter)
  - **`SiglipModel`** [wiring]: wires `SiglipTextModel`, `SiglipVisionModel`; direct `nn.Parameter` (logit_scale, logit_bias)
- **task heads (1)**: ForImageClassification — base + linear (per-task)

## siglip2
- **src**: modeling_siglip2.py, modular_siglip2.py (mostly inherits siglip; new `Siglip2VisionEmbeddings`)
- **hidden_act**: gelu_pytorch_tanh
- **status**: kb_nano_l4 (`L4/siglip2.py` exists)
- **classes**:
  - **`Siglip2VisionEmbeddings`** [compute]: `L1/linear.py` (patch_embedding — Linear over already-patchified input) + `L1/embedding.py` (position_embedding) + `nn.functional.interpolate` (`L1/interpolate.py`) for resizing
  - **`Siglip2TextEmbeddings`** [compute]: `L1/embedding.py` ×2 (token + position)
  - **`Siglip2Attention`** [compute]: q/k/v/o `L1/linear.py` + `L1/dense_attention.py` (non-causal MHA) — matches `L2/siglip_attention.py`
  - **`Siglip2MLP`** [compute]: `L1/linear.py` ×2 + `L1/gelu.py` (gelu_pytorch_tanh) — matches `L2/siglip_mlp.py`
  - **`Siglip2EncoderLayer`** [wiring]: wires `nn.LayerNorm` ×2, `Siglip2Attention`, `Siglip2MLP`
  - **`Siglip2Encoder`** [wiring]: wires `Siglip2EncoderLayer` ×N
  - **`Siglip2VisionModel`** [wiring, inherits `SiglipVisionModel`]: wires `Siglip2VisionEmbeddings`, `Siglip2Encoder`, `nn.LayerNorm`, optional pooling head — covered by `L4/siglip2.py`
  - **`Siglip2TextModel`** [wiring, inherits `SiglipTextModel`]: wires `Siglip2TextEmbeddings`, `Siglip2Encoder`, `nn.LayerNorm`; direct `L1/linear.py` (head)
  - **`Siglip2MultiheadAttentionPoolingHead`** [compute, inherits `SiglipMultiheadAttentionPoolingHead`]: `nn.MultiheadAttention` + `nn.LayerNorm` + `Siglip2MLP` w/ probe parameter
  - **`Siglip2Model`** [wiring, inherits `SiglipModel`]: wires `Siglip2TextModel`, `Siglip2VisionModel`; direct `nn.Parameter` (logit_scale, logit_bias)
- **task heads (1)**: ForImageClassification — base + linear (per-task)

## slanet
- **src**: modeling_slanet.py, modular_slanet.py (inherits from slanext + ppLCNet)
- **hidden_act**: hardswish
- **status**: unsupported (PP-LCNet+CSP-PAN backbone for table recognition with attention-GRU head; no kb-nano OCR/table pipeline)
- **classes**:
  - **`SLANetAttentionGRUCell`** [compute]: `L1/linear.py` ×3 + `L1/tanh.py` + softmax + matmul + `nn.GRUCell` (no L1/L2 GRU; closest `L1/lstm.py`)
  - **`SLANetMLP`** [compute]: `L1/linear.py` ×2 + optional ACT2CLS activation
  - **`SLANetSLAHead`** [wiring]: wires `SLANetAttentionGRUCell`, `SLANetMLP`; iterative loop with one-hot embeddings
  - **`SLANetConvLayer`** [compute]: `L1/conv2d.py` + `L1/batch_norm2d.py` + `L1/hardswish.py`
  - **`SLANetDepthwiseSeparableConvLayer`** [wiring, inherits `PPLCNetDepthwiseSeparableConvLayer`]: wires `SLANetConvLayer` ×2 (depthwise + pointwise) + Identity SE
  - **`SLANetBottleneck`** [wiring]: wires `SLANetConvLayer` (1×1) + `SLANetDepthwiseSeparableConvLayer`
  - **`SLANetCSPLayer`** [wiring]: wires `SLANetConvLayer` ×3 + `SLANetBottleneck` ×N (CSP split-merge)
  - **`SLANetCSPPAN`** [wiring]: wires `SLANetConvLayer` ×N (channel projectors), `SLANetCSPLayer` ×N (top-down + bottom-up), `SLANetDepthwiseSeparableConvLayer` ×N, `nn.Upsample` (`L1/interpolate.py`); CSP-Path-Aggregation Network
  - **`SLANetBackbone`** [wiring]: wires `load_backbone(config)` (PP-LCNet typically) + `SLANetCSPPAN`
  - **`SLANetForTableRecognition`** [wiring, inherits `SLANeXtForTableRecognition`]: wires `SLANetBackbone`, `SLANetSLAHead`

## slanext
- **src**: modeling_slanext.py, modular_slanext.py (mostly inherits from GotOcr2 SAM-style ViT for vision; new SLA head)
- **hidden_act**: gelu
- **status**: unsupported (SAM-style ViT vision encoder + attention-GRU SLA head for table recognition; no kb-nano OCR pipeline)
- **classes**:
  - **`SLANeXtVisionAttention`** [compute, inherits `GotOcr2VisionAttention`]: `L1/linear.py` (qkv, proj) + decomposed rel-pos bias + manual softmax matmul (windowed/global ViT-Det style) — closest match `L2/sam3_vit_attention.py`
  - **`SLANeXtAttentionGRUCell`** [compute]: `L1/linear.py` ×3 + `L1/tanh.py` + softmax + matmul + `nn.GRUCell`
  - **`SLANeXtMLP`** [compute]: `L1/linear.py` ×2 + optional ACT2CLS activation
  - **`SLANeXtMLPBlock`** [compute]: `L1/linear.py` ×2 + `L1/gelu.py` (fc1→gelu→fc2)
  - **`SLANeXtVisionLayer`** [wiring]: wires `nn.LayerNorm` ×2, `SLANeXtVisionAttention`, `SLANeXtMLPBlock`, window partition (reshape arithmetic)
  - **`SLANeXtPatchEmbeddings`** [compute]: `L1/conv2d.py` (patch projection) — matches `L2/vision_patch_embed.py`
  - **`SLANeXtLayerNorm`** [compute]: `L1/layer_norm.py` (channels-first/-last wrapper)
  - **`SLANeXtVisionNeck`** [compute]: `L1/conv2d.py` ×2 + `SLANeXtLayerNorm` ×2 — matches `L3/sam3_neck.py::Sam3VisionNeck`
  - **`SLANeXtVisionEncoder`** [wiring, inherits `GotOcr2VisionEncoder`]: wires `SLANeXtPatchEmbeddings`, `SLANeXtVisionLayer` ×N, `SLANeXtVisionNeck`
  - **`SLANeXtBackbone`** [wiring]: wires `SLANeXtVisionEncoder`
  - **`SLANeXtSLAHead`** [wiring]: wires `SLANeXtAttentionGRUCell`, `SLANeXtMLP` ×2 (structure_generator + loc_generator) — same loop as SLANet
  - **`SLANeXtForTableRecognition`** [wiring]: wires `SLANeXtBackbone`, `SLANeXtSLAHead`

## smollm3
- **src**: modeling_smollm3.py, modular_smollm3.py (inherits llama/qwen2)
- **hidden_act**: silu
- **status**: composable (Llama-family with GQA + RoPE; covered by `L4/llama.py`)
- **classes**:
  - **`SmolLM3RotaryEmbedding`** [compute, inherits `Qwen2RotaryEmbedding`]: `L1/rotary_emb.py`
  - **`SmolLM3Attention`** [compute, inherits `LlamaAttention`]: q/k/v/o `L1/linear.py` + `L1/rotary_emb.py` + KV cache + GQA — matches `L2/attention.py`
  - **`SmolLM3RMSNorm`** [compute]: `L1/rms_norm.py`
  - **`SmolLM3MLP`** [compute]: SwiGLU (gate_proj + up_proj + down_proj + silu) — matches `L2/llama_mlp.py`
  - **`SmolLM3DecoderLayer`** [wiring, inherits `LlamaDecoderLayer`]: wires `SmolLM3Attention`, `SmolLM3MLP`, `SmolLM3RMSNorm` ×2
  - **`SmolLM3Model`** [wiring, inherits `Qwen2Model`]: wires `nn.Embedding`, `SmolLM3DecoderLayer` ×N, `SmolLM3RMSNorm`, `SmolLM3RotaryEmbedding`
  - **`SmolLM3ForCausalLM`** [wiring, inherits `LlamaForCausalLM`]: wires `SmolLM3Model`; direct `L1/linear.py` (lm_head)
- **task heads (3)**: ForSequenceClassification, ForTokenClassification, ForQuestionAnswering — base + linear (per-task)

## smolvlm
- **src**: modeling_smolvlm.py, modular_smolvlm.py (inherits from idefics3 + siglip vision)
- **hidden_act**: gelu_pytorch_tanh (vision); LM uses Llama3-style silu
- **status**: partial (SmolLM3 LM via `L4/llama.py` plus SigLIP-style vision via `L2/siglip_*`; no integrated VLM L4)
- **classes**:
  - **`SmolVLMVisionEmbeddings`** [compute]: `L1/conv2d.py` (patch_embedding) + `L1/embedding.py` (position_embedding) + bucketize-based fractional position ids — closest match `L2/vision_patch_embed.py`
  - **`SmolVLMVisionAttention`** [compute]: q/k/v/o `L1/linear.py` + `L1/dense_attention.py` (non-causal) — matches `L2/siglip_attention.py`
  - **`SmolVLMVisionMLP`** [compute]: `L1/linear.py` ×2 + `L1/gelu.py` (gelu_pytorch_tanh) — matches `L2/siglip_mlp.py`
  - **`SmolVLMEncoderLayer`** [wiring]: wires `SmolVLMVisionAttention`, `SmolVLMVisionMLP`, `nn.LayerNorm` ×2
  - **`SmolVLMEncoder`** [wiring]: wires `SmolVLMEncoderLayer` ×N
  - **`SmolVLMVisionTransformer`** [wiring, inherits `Idefics3VisionTransformer`]: wires `SmolVLMVisionEmbeddings`, `SmolVLMEncoder`, `nn.LayerNorm`
  - **`SmolVLMSimpleMLP`** [compute]: `L1/linear.py` (single proj, no bias)
  - **`SmolVLMConnector`** [wiring]: wires `SmolVLMSimpleMLP`; pixel_shuffle reshape arithmetic
  - **`SmolVLMModel`** [wiring, inherits `Idefics3Model`]: wires `SmolVLMVisionTransformer`, `SmolVLMConnector`, text `AutoModel` (typically SmolLM3)
  - **`SmolVLMForConditionalGeneration`** [wiring, inherits `Idefics3ForConditionalGeneration`]: wires `SmolVLMModel`; direct `L1/linear.py` (lm_head)

## solar_open
- **src**: modeling_solar_open.py, modular_solar_open.py (inherits from glm4_moe + llama)
- **hidden_act**: silu
- **status**: composable (Llama-family with shared-expert MoE; covered by `L2/shared_expert_moe.py` + `L2/attention.py` + `L4/llama.py` engine)
- **classes**:
  - **`SolarOpenMLP`** [compute]: SwiGLU (gate_proj + up_proj + down_proj + silu) — matches `L2/llama_mlp.py`
  - **`SolarOpenTopkRouter`** [compute]: linear logits with score-correction bias buffer (`F.linear`) — closest match `L1/topk_softmax.py` plus group-topk bias logic
  - **`SolarOpenNaiveMoe`** [compute, `@use_experts_implementation`]: stacked-expert SwiGLU with top-k dispatch and index_add — matches `L1/moe_grouped_gemm.py` (or `L1/fp8_moe_grouped_gemm.py`)
  - **`SolarOpenMoE`** [wiring, inherits `Glm4MoeMoE`]: wires `SolarOpenNaiveMoe`, `SolarOpenTopkRouter`, `SolarOpenMLP` (shared experts) — matches `L2/shared_expert_moe.py`
  - **`SolarOpenAttention`** [compute, inherits `LlamaAttention`]: q/k/v/o `L1/linear.py` + `L1/rotary_emb.py` + KV cache + GQA — matches `L2/attention.py`
  - **`SolarOpenRMSNorm`** [compute, inherits `Glm4MoeRMSNorm`]: `L1/rms_norm.py`
  - **`SolarOpenDecoderLayer`** [wiring, inherits `LlamaDecoderLayer`]: wires `SolarOpenAttention`, `SolarOpenMoE` (or `SolarOpenMLP` for early layers), `SolarOpenRMSNorm` ×2
  - **`SolarOpenRotaryEmbedding`** [compute]: standard Llama RoPE → `L1/rotary_emb.py`
  - **`SolarOpenModel`** [wiring, inherits `Glm4MoeModel`]: wires `nn.Embedding`, `SolarOpenDecoderLayer` ×N, `SolarOpenRMSNorm`, `SolarOpenRotaryEmbedding`
  - **`SolarOpenForCausalLM`** [wiring, inherits `Glm4MoeForCausalLM`]: wires `SolarOpenModel`; direct `L1/linear.py` (lm_head)

## speech_encoder_decoder
- **src**: modeling_speech_encoder_decoder.py (no modular)
- **hidden_act**: n/a (composition of arbitrary speech encoder + text decoder)
- **status**: unsupported (generic encoder-decoder wrapper combining a speech encoder + text decoder via cross-attention; no kb-nano integrated speech-translation pipeline)
- **classes**:
  - **`SpeechEncoderDecoderModel`** [wiring]: top-level wrapper; wires `encoder` (`AutoModel`, e.g. wav2vec2/whisper-encoder) and `decoder` (`AutoModelForCausalLM`); optional `enc_to_dec_proj` `nn.Linear` (`L1/linear.py`)

## speech_to_text
- **src**: modeling_speech_to_text.py (no modular)
- **hidden_act**: activation_function=relu
- **status**: unsupported (CNN subsampler + Transformer encoder-decoder; no kb-nano speech translation pipeline; closest is `L4/whisper.py`)
- **classes**:
  - **`Conv1dSubsampler`** [compute]: stack of `L1/conv1d.py` + GLU (no exact L2 match)
  - **`Speech2TextSinusoidalPositionalEmbedding`** [compute]: precomputed sinusoid + index_select
  - **`Speech2TextAttention`** [compute]: BART-style q/k/v/o `L1/linear.py` + manual bmm-softmax-bmm with `EncoderDecoderCache` — closest match `L2/whisper_attention.py`
  - **`Speech2TextEncoderLayer`** [wiring]: wires `Speech2TextAttention` (self), `nn.Linear` ×2 (fc1/fc2 with `L1/relu.py`), `nn.LayerNorm` ×2
  - **`Speech2TextDecoderLayer`** [wiring]: wires `Speech2TextAttention` ×2 (self+cross), `nn.Linear` ×2, `nn.LayerNorm` ×3
  - **`Speech2TextEncoder`** [wiring]: wires `Conv1dSubsampler`, `Speech2TextSinusoidalPositionalEmbedding`, `Speech2TextEncoderLayer` ×N, `nn.LayerNorm`
  - **`Speech2TextDecoder`** [wiring]: wires `nn.Embedding` (`L1/embedding.py`), `Speech2TextSinusoidalPositionalEmbedding`, `Speech2TextDecoderLayer` ×N, `nn.LayerNorm`
  - **`Speech2TextModel`** [wiring]: wires `Speech2TextEncoder`, `Speech2TextDecoder`
  - **`Speech2TextForConditionalGeneration`** [wiring]: wires `Speech2TextModel`; direct `L1/linear.py` (lm_head)

## speecht5
- **src**: modeling_speecht5.py (no modular)
- **hidden_act**: gelu
- **status**: unsupported (unified speech/text encoder-decoder with speech prenets/postnets and HiFi-GAN vocoder; no kb-nano speech-T5 pipeline)
- **classes**:
  - **`SpeechT5NoLayerNormConvLayer`** [compute]: `L1/conv1d.py` + `L1/gelu.py`
  - **`SpeechT5LayerNormConvLayer`** [compute]: `L1/conv1d.py` + `L1/layer_norm.py` + `L1/gelu.py`
  - **`SpeechT5GroupNormConvLayer`** [compute]: `L1/conv1d.py` + `L1/group_norm.py` + `L1/gelu.py`
  - **`SpeechT5SinusoidalPositionalEmbedding`** [compute]: precomputed sinusoid + index_select
  - **`SpeechT5PositionalConvEmbedding`** [compute]: weight-normed `L1/conv1d.py` + `L1/gelu.py`
  - **`SpeechT5ScaledPositionalEncoding`** [compute]: sinusoid + `nn.Parameter` scale
  - **`SpeechT5RelativePositionalEncoding`** [compute]: relative position embedding lookup (`L1/embedding.py`)
  - **`SpeechT5SamePadLayer`** [compute]: tensor slice
  - **`SpeechT5FeatureEncoder`** [wiring]: wires SpeechT5GroupNorm/LayerNorm/NoLayerNormConvLayer ×N
  - **`SpeechT5FeatureProjection`** [compute]: `L1/layer_norm.py` + `L1/linear.py`
  - **`SpeechT5SpeechEncoderPrenet`** [wiring]: wires `SpeechT5FeatureEncoder`, `SpeechT5FeatureProjection`, `SpeechT5PositionalConvEmbedding`
  - **`SpeechT5SpeechDecoderPrenet`** [compute]: `L1/linear.py` ×N + `L1/relu.py` + speaker-embedding linear; `SpeechT5ScaledPositionalEncoding`
  - **`SpeechT5BatchNormConvLayer`** [compute]: `L1/conv1d.py` + nn.BatchNorm1d + `L1/tanh.py`
  - **`SpeechT5SpeechDecoderPostnet`** [wiring]: wires `SpeechT5BatchNormConvLayer` ×N; direct `L1/linear.py` (mel_proj, prob_proj)
  - **`SpeechT5TextEncoderPrenet`** [compute]: `L1/embedding.py` (token) + `SpeechT5ScaledPositionalEncoding`
  - **`SpeechT5TextDecoderPrenet`** [compute]: `L1/embedding.py` (token) + `SpeechT5ScaledPositionalEncoding`
  - **`SpeechT5TextDecoderPostnet`** [compute]: `L1/linear.py` (lm_head)
  - **`SpeechT5Attention`** [compute]: BART-style q/k/v/o `L1/linear.py` + manual bmm-softmax-bmm + optional relative-position bias — closest match `L2/whisper_attention.py`
  - **`SpeechT5FeedForward`** [compute]: `L1/linear.py` ×2 + `L1/gelu.py`
  - **`SpeechT5EncoderLayer`** [wiring]: wires `SpeechT5Attention` (self), `SpeechT5FeedForward`, `nn.LayerNorm` ×2
  - **`SpeechT5DecoderLayer`** [wiring]: wires `SpeechT5Attention` ×2 (self+cross), `SpeechT5FeedForward`, `nn.LayerNorm` ×3
  - **`SpeechT5Encoder`** [wiring]: wires `SpeechT5EncoderLayer` ×N, `nn.LayerNorm`, `SpeechT5RelativePositionalEncoding` (optional)
  - **`SpeechT5EncoderWithSpeechPrenet`** [wiring]: wires `SpeechT5SpeechEncoderPrenet`, `SpeechT5Encoder`
  - **`SpeechT5EncoderWithTextPrenet`** [wiring]: wires `SpeechT5TextEncoderPrenet`, `SpeechT5Encoder`
  - **`SpeechT5EncoderWithoutPrenet`** [wiring]: wires `SpeechT5Encoder`
  - **`SpeechT5Decoder`** [wiring]: wires `SpeechT5DecoderLayer` ×N, `nn.LayerNorm`
  - **`SpeechT5DecoderWithSpeechPrenet`** [wiring]: wires `SpeechT5SpeechDecoderPrenet`, `SpeechT5Decoder`
  - **`SpeechT5DecoderWithTextPrenet`** [wiring]: wires `SpeechT5TextDecoderPrenet`, `SpeechT5Decoder`
  - **`SpeechT5DecoderWithoutPrenet`** [wiring]: wires `SpeechT5Decoder`
  - **`SpeechT5Model`** [wiring]: wires encoder + decoder (any prenet variant)
  - **`SpeechT5ForSpeechToText`** [wiring]: wires `SpeechT5EncoderWithSpeechPrenet`, `SpeechT5DecoderWithTextPrenet`, `SpeechT5TextDecoderPostnet`
  - **`SpeechT5ForTextToSpeech`** [wiring]: wires `SpeechT5EncoderWithTextPrenet`, `SpeechT5DecoderWithSpeechPrenet`, `SpeechT5SpeechDecoderPostnet`
  - **`SpeechT5ForSpeechToSpeech`** [wiring]: wires `SpeechT5EncoderWithSpeechPrenet`, `SpeechT5DecoderWithSpeechPrenet`, `SpeechT5SpeechDecoderPostnet`
  - **`HifiGanResidualBlock`** [compute]: dilated `L1/conv1d.py` ×N + `L1/leaky_relu.py` + residual
  - **`SpeechT5HifiGan`** [wiring]: wires `nn.Conv1d` (pre/post), `nn.ConvTranspose1d` ×N (`L1/conv_transpose1d.py`), `HifiGanResidualBlock` ×N, `L1/leaky_relu.py`, `L1/tanh.py`

## splinter
- **src**: modeling_splinter.py (no modular)
- **hidden_act**: gelu
- **status**: composable (BERT-style encoder for QA span selection; covered by encoder L2 ops)
- **classes**:
  - **`SplinterEmbeddings`** [compute]: BERT-style word + position + token_type + LayerNorm + Dropout — matches `L2/encoder_embeddings.py`
  - **`SplinterSelfAttention`** [compute]: q/k/v `L1/linear.py` + dispatch via `ALL_ATTENTION_FUNCTIONS` — matches `L2/encoder_attention.py`
  - **`SplinterSelfOutput`** [compute]: `L1/linear.py` + `L1/layer_norm.py` + residual — matches `L2/encoder_attention.py`
  - **`SplinterAttention`** [wiring]: wires `SplinterSelfAttention`, `SplinterSelfOutput` — matches `L2/encoder_attention.py`
  - **`SplinterIntermediate`** [compute]: `L1/linear.py` + `L1/gelu.py`
  - **`SplinterOutput`** [compute]: `L1/linear.py` + `L1/layer_norm.py` + residual
  - **`SplinterLayer`** [wiring]: wires `SplinterAttention`, `SplinterIntermediate`, `SplinterOutput`
  - **`SplinterEncoder`** [wiring]: wires `SplinterLayer` ×N
  - **`SplinterModel`** [wiring]: wires `SplinterEmbeddings`, `SplinterEncoder`
  - **`SplinterFullyConnectedLayer`** [compute]: `L1/linear.py` + `L1/gelu.py` + `L1/layer_norm.py`
  - **`QuestionAwareSpanSelectionHead`** [wiring]: wires `SplinterFullyConnectedLayer` ×4; direct `L1/linear.py` ×2 (start_logits, end_logits)
  - **`SplinterForQuestionAnswering`** [wiring]: wires `SplinterModel`, `QuestionAwareSpanSelectionHead`
  - **`SplinterForPreTraining`** [wiring]: wires `SplinterModel`, `QuestionAwareSpanSelectionHead`

## squeezebert
- **src**: modeling_squeezebert.py (no modular)
- **hidden_act**: gelu
- **status**: partial (BERT variant using grouped Conv1d in place of Linear; primitives via L1/conv1d.py + L1/layer_norm.py + L1/gelu.py)
- **classes**:
  - **`SqueezeBertEmbeddings`** [compute]: BERT-style word + position + token_type + LayerNorm + Dropout — matches `L2/encoder_embeddings.py`
  - **`MatMulWrapper`** [compute]: `torch.matmul` wrapper (no kernel)
  - **`SqueezeBertLayerNorm`** [compute]: `L1/layer_norm.py` over channels-first 1D layout
  - **`ConvDropoutLayerNorm`** [compute]: grouped `L1/conv1d.py` (1×1) + dropout + residual + `SqueezeBertLayerNorm`
  - **`ConvActivation`** [compute]: grouped `L1/conv1d.py` (1×1) + `L1/gelu.py`
  - **`SqueezeBertSelfAttention`** [compute]: q/k/v as grouped `L1/conv1d.py` (1×1) + manual softmax matmul attention — analogous to `L2/encoder_attention.py` but with Conv1d
  - **`SqueezeBertModule`** [wiring]: wires `SqueezeBertSelfAttention`, `ConvDropoutLayerNorm` (post_attention), `ConvActivation` (intermediate), `ConvDropoutLayerNorm` (output)
  - **`SqueezeBertEncoder`** [wiring]: wires `SqueezeBertModule` ×N
  - **`SqueezeBertPooler`** [compute]: `L1/linear.py` + `L1/tanh.py`
  - **`SqueezeBertPredictionHeadTransform`** [compute]: `L1/linear.py` + `L1/gelu.py` + `L1/layer_norm.py`
  - **`SqueezeBertLMPredictionHead`** [wiring]: wires `SqueezeBertPredictionHeadTransform`; direct `L1/linear.py` (decoder)
  - **`SqueezeBertOnlyMLMHead`** [wiring]: wires `SqueezeBertLMPredictionHead`
  - **`SqueezeBertModel`** [wiring]: wires `SqueezeBertEmbeddings`, `SqueezeBertEncoder`, `SqueezeBertPooler`
  - **`SqueezeBertForMaskedLM`** [wiring]: wires `SqueezeBertModel`, `SqueezeBertOnlyMLMHead`
- **task heads (4)**: ForSequenceClassification, ForMultipleChoice, ForTokenClassification, ForQuestionAnswering — base + linear (per-task)

## stablelm
- **src**: modeling_stablelm.py (no modular)
- **hidden_act**: silu
- **status**: composable (Llama-family with partial RoPE + LayerNorm + optional QK-LayerNorm-per-head; covered by `L4/llama.py` engine, attention is a slight variant)
- **classes**:
  - **`StableLmRotaryEmbedding`** [compute]: standard Llama RoPE → `L1/rotary_emb.py`
  - **`StableLmMLP`** [compute]: SwiGLU (gate_proj + up_proj + down_proj + silu) — matches `L2/llama_mlp.py`
  - **`StableLmLayerNormPerHead`** [compute]: per-head `nn.LayerNorm` ×num_heads (no kb-nano match; closest is `L1/layer_norm.py` looped)
  - **`StableLmAttention`** [compute]: q/k/v/o `L1/linear.py` + partial-rotary `L1/rotary_emb.py` (only rotary_ndims) + optional `StableLmLayerNormPerHead` (qk_layernorm) + KV cache + GQA — closest match `L2/attention.py` (with partial RoPE)
  - **`StableLmDecoderLayer`** [wiring]: wires `StableLmAttention`, `StableLmMLP`, `nn.LayerNorm` ×1-2 (parallel-residual variant skips post_attention_layernorm)
  - **`StableLmModel`** [wiring]: wires `nn.Embedding`, `StableLmDecoderLayer` ×N, `nn.LayerNorm`, `StableLmRotaryEmbedding`
  - **`StableLmForCausalLM`** [wiring]: wires `StableLmModel`; direct `L1/linear.py` (lm_head)
- **task heads (2)**: ForSequenceClassification, ForTokenClassification — base + linear (per-task)

## starcoder2
- **src**: modeling_starcoder2.py, modular_starcoder2.py (inherits from mistral)
- **hidden_act**: gelu_pytorch_tanh
- **status**: composable (Llama-family with GQA + RoPE; standard 2-layer (fc1→gelu→fc2) MLP rather than SwiGLU; covered by `L4/llama.py` engine)
- **classes**:
  - **`Starcoder2MLP`** [compute]: `L1/linear.py` ×2 (c_fc, c_proj) + `L1/gelu.py` (gelu_pytorch_tanh) — closest match `L2/encoder_mlp.py` (no SwiGLU)
  - **`Starcoder2Attention`** [compute, inherits `MistralAttention`]: q/k/v/o `L1/linear.py` + `L1/rotary_emb.py` + KV cache + GQA — matches `L2/attention.py`
  - **`Starcoder2DecoderLayer`** [wiring, inherits `MistralDecoderLayer`]: wires `Starcoder2Attention`, `Starcoder2MLP`, `nn.LayerNorm` ×2 (`L1/layer_norm.py`)
  - **`Starcoder2RotaryEmbedding`** [compute]: standard Llama RoPE → `L1/rotary_emb.py`
  - **`Starcoder2Model`** [wiring, inherits `MistralModel`]: wires `nn.Embedding`, `Starcoder2DecoderLayer` ×N, `nn.LayerNorm`, `Starcoder2RotaryEmbedding`
  - **`Starcoder2ForCausalLM`** [wiring, inherits `MistralForCausalLM`]: wires `Starcoder2Model`; direct `L1/linear.py` (lm_head)
- **task heads (2)**: ForSequenceClassification, ForTokenClassification — base + linear (per-task)

## superglue
- **src**: modeling_superglue.py (no modular)
- **hidden_act**: n/a (uses `nn.ReLU` directly)
- **status**: unsupported (Graph Neural Network for keypoint matching with self/cross attention; no kb-nano keypoint matching pipeline)
- **classes**:
  - **`SuperGlueMultiLayerPerceptron`** [compute]: `L1/linear.py` + nn.BatchNorm1d + `L1/relu.py` (transposed for batchnorm over channel)
  - **`SuperGlueKeypointEncoder`** [wiring]: wires `SuperGlueMultiLayerPerceptron` ×N + final `nn.Linear` (`L1/linear.py`)
  - **`SuperGlueSelfAttention`** [compute]: q/k/v `L1/linear.py` + manual softmax matmul attention (eager only) — closest match `L2/encoder_attention.py` but supports cross-attention via encoder_hidden_states
  - **`SuperGlueSelfOutput`** [compute]: `L1/linear.py` (no LayerNorm/residual — caller adds residual)
  - **`SuperGlueAttention`** [wiring]: wires `SuperGlueSelfAttention`, `SuperGlueSelfOutput`
  - **`SuperGlueAttentionalPropagation`** [wiring]: wires `SuperGlueAttention`, `SuperGlueMultiLayerPerceptron` ×N + final `nn.Linear`
  - **`SuperGlueAttentionalGNN`** [wiring]: wires `SuperGlueAttentionalPropagation` ×N (alternating self/cross types)
  - **`SuperGlueFinalProjection`** [compute]: `L1/linear.py` (final_proj)
  - **`SuperGlueForKeypointMatching`** [wiring]: wires `SuperGlueKeypointEncoder`, `SuperGlueAttentionalGNN`, `SuperGlueFinalProjection`; Sinkhorn iteration in forward (no kernel)

## superpoint
- **src**: modeling_superpoint.py (no modular)
- **hidden_act**: n/a (uses `nn.ReLU` directly)
- **status**: unsupported (VGG-style CNN for keypoint detection + descriptor extraction; no kb-nano keypoint detection pipeline)
- **classes**:
  - **`SuperPointConvBlock`** [compute]: `L1/conv2d.py` ×2 + `L1/relu.py` ×2 + optional `L1/max_pool2d.py`
  - **`SuperPointEncoder`** [wiring]: wires `SuperPointConvBlock` ×4
  - **`SuperPointInterestPointDecoder`** [compute]: `L1/conv2d.py` ×2 (conv_score_a/b) + `L1/relu.py` + softmax + simple_nms (post-processing)
  - **`SuperPointDescriptorDecoder`** [compute]: `L1/conv2d.py` ×2 (conv_descriptor_a/b) + `L1/relu.py` + L2-normalize + `nn.functional.grid_sample` (`L1/grid_sample.py`)
  - **`SuperPointForKeypointDetection`** [wiring]: wires `SuperPointEncoder`, `SuperPointInterestPointDecoder`, `SuperPointDescriptorDecoder`

## swiftformer
- **src**: modeling_swiftformer.py (no modular)
- **hidden_act**: gelu
- **status**: unsupported (efficient hybrid CNN-Transformer with additive attention; no kb-nano pipeline)
- **classes**:
  - **`SwiftFormerPatchEmbedding`** [compute]: `nn.Sequential` of `L1/conv2d.py` ×2 + `L1/batch_norm2d.py` ×2 + `L1/relu.py` ×2 (stem)
  - **`SwiftFormerDropPath`** [compute]: stochastic-depth drop_path
  - **`SwiftFormerEmbeddings`** [compute]: `L1/conv2d.py` (downsample patch) + `L1/batch_norm2d.py`
  - **`SwiftFormerConvEncoder`** [compute]: depthwise `L1/conv2d.py` + `L1/batch_norm2d.py` + `L1/conv2d.py` (1×1) + `L1/gelu.py` + `L1/conv2d.py` (1×1) + dropout + scale param + residual (ConvNeXt-like block)
  - **`SwiftFormerMlp`** [compute]: `L1/batch_norm2d.py` + `L1/conv2d.py` (1×1) ×2 + `L1/gelu.py` (operates on 4-D tensor as Conv2d 1×1)
  - **`SwiftFormerEfficientAdditiveAttention`** [compute]: `L1/linear.py` ×4 (to_query, to_key, proj, final) + `nn.functional.normalize` + softmax over additive learned weights `w_g` (no exact kb-nano match)
  - **`SwiftFormerLocalRepresentation`** [compute]: depthwise `L1/conv2d.py` + `L1/batch_norm2d.py` + `L1/conv2d.py` (1×1) + `L1/gelu.py` + `L1/conv2d.py` (1×1) + scale param
  - **`SwiftFormerEncoderBlock`** [wiring]: wires `SwiftFormerLocalRepresentation`, `SwiftFormerEfficientAdditiveAttention`, `SwiftFormerMlp`, optional `SwiftFormerDropPath`
  - **`SwiftFormerStage`** [wiring]: wires mostly `SwiftFormerConvEncoder` (depth-1) + `SwiftFormerEncoderBlock` (last)
  - **`SwiftFormerEncoder`** [wiring]: wires `SwiftFormerStage` ×N + `SwiftFormerEmbeddings` between stages (downsample)
  - **`SwiftFormerModel`** [wiring]: wires `SwiftFormerPatchEmbedding`, `SwiftFormerEncoder`
- **task heads (1)**: ForImageClassification — base + linear (per-task)

