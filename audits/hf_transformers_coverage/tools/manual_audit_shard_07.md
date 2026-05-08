## hiera
- **src**: modeling_hiera.py
- **status**: composable
- **rationale**: Hierarchical ViT with mask-unit windowed attention; all primitives (LayerNorm, Linear, Conv2d, max-pool, manual SDPA via window reshape) map to existing kb-nano L1 ops. The mask-unit attention can be expressed via reshape + L1/sdpa.py per-window.
- **classes**:
  - **`HieraPatchEmbeddings`** [compute]: `L1/conv2d.py`, `L1/interpolate.py` (Conv2d patch projection with optional masked bilinear interpolate before conv.)
  - **`HieraEmbeddings`** [wiring]: Composes patch + position embeddings (with optional bicubic interpolate for resolution change).
  - **`HieraMaskUnitAttention`** [compute]: `L1/linear.py`, `L1/max_pool2d.py`, `L1/sdpa.py` (Mask-unit window attention with optional max-pool query (q-pool); fused QKV linear, manual softmax via SDPA, no kb-nano fused windowed-attn kernel exists but compute is composable.)
  - **`HieraMlp`** [compute]: `L2/encoder_mlp.py` (fc1 -> act(GELU) -> fc2 (not gated); matches encoder_mlp.py pattern.)
  - **`HieraLayer`** [wiring]: Wires LayerNorm -> attn (with optional q-pool projection) -> LayerNorm -> MLP.
  - **`HieraStage`** [wiring]: Stack of HieraLayers.
  - **`HieraEncoder`** [wiring]: Stack of HieraStages with reroll schedule.
  - **`HieraPooler`** [wiring]: Mean pool + LayerNorm.
  - **`HieraDecoder`** [wiring]: MAE decoder wiring.
  - **`HieraMultiScaleHead`** [wiring]: Multi-scale pooling head wiring.

## hubert
- **src**: modeling_hubert.py
- **status**: partial
- **partial_reason**: BatchNorm1d (used in HubertPositionalConvEmbedding when conv_pos_batch_norm=True) and Conv1d weight_norm parametrization have no kb-nano L1 equivalent. GroupNorm is available, BatchNorm2d is, but BatchNorm1d is missing.
- **rationale**: Hubert audio feature encoder uses BatchNorm1d on positional Conv1d and weight_norm parametrization on Conv1d; kb-nano lacks BatchNorm1d L1 op. Transformer encoder layers themselves are BART-style and composable.
- **classes**:
  - **`HubertFeatureEncoder`** [compute]: BatchNorm1d (used in HubertPositionalConvEmbedding when conv_pos_batch_norm=True) and Conv1d weight_norm parametrization have no kb-nano L1 equivalent. GroupNorm is available, BatchNorm2d is, but Batc
  - **`HubertPositionalConvEmbedding`** [compute]: `L1/conv1d.py` (Grouped Conv1d with weight_norm parametrization or BatchNorm1d alternative; weight_norm not directly supported, BatchNorm1d missing in kb-nano.)
  - **`HubertNoLayerNormConvLayer`** [compute]: `L1/conv1d.py`, `L1/gelu.py` (Conv1d + activation.)
  - **`HubertLayerNormConvLayer`** [compute]: `L1/conv1d.py`, `L1/layer_norm.py` (Conv1d + LayerNorm + activation.)
  - **`HubertGroupNormConvLayer`** [compute]: `L1/conv1d.py`, `L1/group_norm.py` (Conv1d + GroupNorm + activation.)
  - **`HubertFeatureProjection`** [compute]: `L1/layer_norm.py`, `L1/linear.py` (LayerNorm + Linear projection.)
  - **`HubertAttention`** [compute]: `L2/whisper_attention.py` (BART-style attention with separate q/k/v/out_proj and bias; matches WhisperEncoderSelfAttention pattern.)
  - **`HubertFeedForward`** [compute]: `L2/whisper_mlp.py` (fc1 -> act -> fc2 (not gated).)
  - **`HubertEncoderLayer`** [wiring]: Self-attn + LN + FFN + LN.
  - **`HubertEncoder`** [wiring]: Stack of encoder layers.
  - **`HubertEncoderLayerStableLayerNorm`** [wiring]: Pre-LN variant of encoder layer.
  - **`HubertEncoderStableLayerNorm`** [wiring]: Stack of pre-LN encoder layers.
  - **`HubertAttnAdapterLayer`** [compute]: `L1/linear.py`, `L1/layer_norm.py`, `L1/relu.py` (LN + down_proj + ReLU + up_proj adapter.)

## hunyuan_v1_dense
- **src**: modular_hunyuan_v1_dense.py
- **status**: composable
- **rationale**: Llama-family decoder with QK RMSNorm applied AFTER RoPE (rather than before like Qwen3); SwiGLU MLP. All L1 ops (rms_norm, rotary_emb, silu_and_mul, linear) exist; only the QK-norm-after-RoPE ordering differs from existing L2/attention.py wiring.
- **classes**:
  - **`HunYuanDenseV1RMSNorm`** [compute]: `L1/rms_norm.py` (Standard Llama RMSNorm.)
  - **`HunYuanDenseV1MLP`** [compute]: `L2/llama_mlp.py` (SwiGLU MLP (gate/up/down projections).)
  - **`HunYuanDenseV1Attention`** [compute]: `L2/attention.py` (GQA attention with QK-RMSNorm AFTER RoPE; L1 ops exist but L2/attention.py currently wires QK-norm before RoPE (Qwen3 ordering). Compute primitives match.)
  - **`HunYuanDenseV1RotaryEmbedding`** [compute]: `L1/rotary_emb.py` (DynamicNTKAlpha RoPE variant; standard inv_freq scheme.)

## hunyuan_v1_moe
- **src**: modular_hunyuan_v1_moe.py
- **status**: composable
- **rationale**: Llama-style attention (with QK-norm after RoPE like dense variant) + Mixtral-style MoE with shared MLP + routed experts using softmax routing. Maps to L2/shared_expert_moe.py (softmax routing) + L2/attention.py.
- **classes**:
  - **`HunYuanMoEV1RMSNorm`** [compute]: `L1/rms_norm.py` (Llama RMSNorm.)
  - **`HunYuanMoEV1MLP`** [compute]: `L2/llama_mlp.py` (SwiGLU MLP.)
  - **`HunYuanMoEV1Attention`** [compute]: `L2/attention.py` (GQA + QK-RMSNorm after RoPE.)
  - **`HunYuanMoEV1Gate`** [compute]: `L1/linear.py` (fp32 router linear.)
  - **`HunYuanMoEV1Experts`** [compute]: `L1/moe_grouped_gemm.py` (Standard fused-MoE expert tensor layout.)
  - **`HunYuanMoEV1Moe`** [compute]: `L2/shared_expert_moe.py` (Shared MLP + softmax routing + topk + experts; matches shared_expert_moe.py with routing='softmax'.)

## hy_v3
- **src**: modular_hy_v3.py
- **status**: composable
- **rationale**: Apertus-style attention (QK-RMSNorm BEFORE RoPE) + LlamaMLP + Mixtral/MiniMaxM2-style MoE with sigmoid routing + e_score_correction_bias + shared experts. Maps to L2/attention.py (qk_norm) and L2/shared_expert_moe.py (routing='sigmoid').
- **classes**:
  - **`HYV3RMSNorm`** [compute]: `L1/rms_norm.py` (Llama RMSNorm.)
  - **`HYV3RotaryEmbedding`** [compute]: `L1/rotary_emb.py` (Standard NeoX RoPE.)
  - **`HYV3MLP`** [compute]: `L2/llama_mlp.py` (SwiGLU MLP, configurable intermediate size for shared experts.)
  - **`HYV3Attention`** [compute]: `L2/attention.py` (GQA + QK-RMSNorm BEFORE RoPE (Qwen3-style); matches attention.py qk_norm flag.)
  - **`HYV3TopKRouter`** [compute]: `L1/linear.py`, `L1/sigmoid_topk.py` (Sigmoid routing + e_score_correction_bias + topk + extra router_scaling_factor.)
  - **`HYV3Experts`** [compute]: `L1/moe_grouped_gemm.py` (Standard fused MoE experts.)
  - **`HYV3MoE`** [compute]: `L2/shared_expert_moe.py` (Sigmoid routing + bias correction + shared experts; matches shared_expert_moe with routing='sigmoid'.)
  - **`HYV3DecoderLayer`** [wiring]: Wiring: norm + self_attn + norm + (MoE or dense MLP per layer schedule).

## ibert
- **src**: modeling_ibert.py
- **status**: unsupported
- **unsupported_reason**: IBert's IntLayerNorm/IntGELU/IntSoftmax/QuantLinear are integer-arithmetic emulations of normal float ops with explicit per-channel INT8/INT32 scaling factors propagated through every op. No kb-nano L1 kernel implements this scheme; would require custom CUDA kernels for integer-only inference.
- **rationale**: Quantized BERT with bespoke INT8/INT16/INT32 integer-only quantization scheme: QuantLinear, QuantEmbedding, QuantAct, IntLayerNorm, IntGELU, IntSoftmax. These integer-arithmetic kernels have no kb-nano equivalent (kb-nano supports BitNet INT8xINT2 and FP8 but not IBert's per-channel symmetric INT8/INT32 scheme).
- **classes**:
  - **`IBertSelfAttention`** [compute]: IBert's IntLayerNorm/IntGELU/IntSoftmax/QuantLinear are integer-arithmetic emulations of normal float ops with explicit per-channel INT8/INT32 scaling factors propagated through every op. No kb-nano L
  - **`IBertEmbeddings`** [wiring]: QuantEmbedding + IntLayerNorm + QuantAct; integer-quantized embeddings have no kb-nano equivalent.
  - **`IBertSelfOutput`** [wiring]: QuantLinear + IntLayerNorm; missing.
  - **`IBertIntermediate`** [wiring]: QuantLinear + IntGELU.
  - **`IBertOutput`** [wiring]: QuantLinear + IntLayerNorm.
  - **`IBertLayer`** [wiring]: Wiring of integer-quantized BERT layer.
  - **`IBertEncoder`** [wiring]: Stack of layers.
  - **`IBertPooler`** [wiring]: Pooling head.

## idefics
- **src**: modeling_idefics.py
- **status**: composable
- **rationale**: Llama-style decoder + IdeficsGatedCrossAttentionLayer (cross-attention with learnable alpha/tanh gate) + CLIP/timm-style vision encoder + Perceiver resampler. All compute primitives (linear, RMSNorm, RoPE, SwiGLU, tanh, masked_fill) exist in kb-nano L1; the gated cross-attn wrapper is novel wiring on top of cross-attention but composes existing ops.
- **classes**:
  - **`IdeficsRMSNorm`** [compute]: `L1/rms_norm.py` (Llama RMSNorm.)
  - **`IdeficsEmbedding`** [compute]: `L1/rotary_emb.py` (Llama RoPE.)
  - **`IdeficsMLP`** [compute]: `L2/llama_mlp.py` (SwiGLU MLP (gate/up/down).)
  - **`IdeficsAttention`** [compute]: `L2/attention.py` (Llama-style attention with optional QK-LayerNorm, supports cross-attention via key_value_states.)
  - **`IdeficsDecoderLayer`** [wiring]: Standard Llama decoder layer wiring.
  - **`IdeficsGatedCrossAttentionLayer`** [compute]: `L2/whisper_attention.py`, `L1/tanh.py` (Cross-attn (BART-style) + masked_fill + tanh-gated alpha residual + MLP with tanh-gated alpha. No kb-nano L2 for gated cross-attn; primitives exist.)
  - **`IdeficsDecoupledEmbedding`** [compute]: `L1/embedding.py` (Standard embedding with extra additional_embedding partition; both are nn.Embedding.)
  - **`IdeficsDecoupledLinear`** [compute]: `L1/linear.py` (Standard linear with extra additional_fc; both Linear.)
  - **`IdeficsPerceiverResampler`** [wiring]: Perceiver resampler with learned latents and stacked cross-attn (defined in perceiver.py); all primitives exist.
  - **`IdeficsPerceiverAttention`** [compute]: `L1/sdpa.py`, `L1/layer_norm.py` (Cross-attn between latents (queries) and context (keys/values from concat([context, latents])); standard SDPA.)

## idefics2
- **src**: modeling_idefics2.py
- **status**: partial
- **rationale**: SigLIP-style vision encoder + Perceiver resampler with cross-attn + AutoModel text decoder + Idefics2Connector (linear projection). MultiheadAttentionPoolingHead uses torch.nn.MultiheadAttention which is plain SDPA. All primitives present.
- **classes**:
  - **`Idefics2EncoderLayer`** [compute]: no kb-nano kernel — SigLIP-style vision encoder + Perceiver resampler with cross-attn + AutoModel text decoder + Idefics2Connector (linear projection). MultiheadAttentionPoolingHead uses torch.nn.MultiheadAttention which
  - **`Idefics2VisionEmbeddings`** [compute]: `L1/conv2d.py`, `L1/embedding.py` (Conv2d patch embed + learned position embedding.)
  - **`Idefics2VisionAttention`** [compute]: `L2/siglip_attention.py` (Separate q/k/v/out_proj with bias, non-causal SDPA; matches SigLIP attention.)
  - **`Idefics2VisionMLP`** [compute]: `L2/siglip_mlp.py` (fc1 -> act -> fc2 (not gated); matches SigLIP MLP.)
  - **`Idefics2MLP`** [compute]: `L2/llama_mlp.py` (SwiGLU MLP (gate/up/down).)
  - **`Idefics2MultiheadAttentionPoolingHead`** [compute]: `L1/sdpa.py`, `L1/layer_norm.py` (Learned probe + nn.MultiheadAttention (cross-attn from probe to hidden) + LN + MLP; SDPA suffices.)
  - **`Idefics2Encoder`** [wiring]: Stack of encoder layers.
  - **`Idefics2VisionTransformer`** [wiring]: Vision encoder wiring.
  - **`Idefics2RMSNorm`** [compute]: `L1/rms_norm.py` (Llama RMSNorm.)
  - **`Idefics2PerceiverAttention`** [compute]: `L1/sdpa.py`, `L1/linear.py` (Cross-attention between perceiver latents and context.)
  - **`Idefics2PerceiverLayer`** [wiring]: Perceiver layer wiring.
  - **`Idefics2PerceiverResampler`** [wiring]: Stack of perceiver layers + final norm.
  - **`Idefics2Connector`** [compute]: `L1/linear.py` (Linear projection vision -> text dim.)

## idefics3
- **src**: modeling_idefics3.py
- **status**: composable
- **rationale**: SigLIP-style vision encoder + Idefics3Connector (pixel shuffle + linear projection) + AutoModel text decoder. No perceiver resampler in Idefics3.
- **classes**:
  - **`Idefics3VisionEmbeddings`** [compute]: `L1/conv2d.py`, `L1/embedding.py` (Conv2d patch embed + position embedding.)
  - **`Idefics3VisionAttention`** [compute]: `L2/siglip_attention.py` (SigLIP-style separate q/k/v/out_proj attention.)
  - **`Idefics3VisionMLP`** [compute]: `L2/siglip_mlp.py` (fc1 + act + fc2 (not gated).)
  - **`Idefics3SimpleMLP`** [compute]: `L1/linear.py` (Single Linear projection.)
  - **`Idefics3EncoderLayer`** [wiring]: Pre-LN attn + pre-LN MLP.
  - **`Idefics3Encoder`** [wiring]: Stack of layers.
  - **`Idefics3RMSNorm`** [compute]: `L1/rms_norm.py` (Llama RMSNorm (unused in vision but present).)
  - **`Idefics3Connector`** [wiring]: Pixel-shuffle reshape + Idefics3SimpleMLP projection.
  - **`Idefics3VisionTransformer`** [wiring]: Vision encoder wiring.

## ijepa
- **src**: modeling_ijepa.py
- **status**: composable
- **rationale**: ViT clone with separate query/key/value Linear projections, GELU MLP (fc1+act+fc2), and LayerNorm. Maps to L2/encoder_attention.py + L2/encoder_mlp.py.
- **classes**:
  - **`IJepaPatchEmbeddings`** [compute]: `L1/conv2d.py` (Conv2d patch embed.)
  - **`IJepaEmbeddings`** [wiring]: Patch embed + position embedding (no CLS token in IJEPA).
  - **`IJepaSelfAttention`** [compute]: `L2/encoder_attention.py` (Separate q/k/v Linear + manual SDPA, non-causal. Matches encoder_attention.py.)
  - **`IJepaSelfOutput`** [compute]: `L1/linear.py` (Output projection (no residual added here).)
  - **`IJepaAttention`** [wiring]: Wires SelfAttention + SelfOutput.
  - **`IJepaIntermediate`** [compute]: `L1/linear.py`, `L1/gelu.py` (Linear + GELU.)
  - **`IJepaOutput`** [compute]: `L1/linear.py` (Linear + residual.)
  - **`IJepaLayer`** [wiring]: ViT block wiring (LN -> attn -> LN -> MLP).
  - **`IJepaEncoder`** [wiring]: Stack of layers.
  - **`IJepaPooler`** [wiring]: Mean pool over patch tokens.

## imagegpt
- **src**: modeling_imagegpt.py
- **status**: composable
- **rationale**: GPT-2 style causal LM with custom T5-like LayerNorm (variance-only, no centering, no bias) + Conv1D-as-Linear + fc1+act+fc2 MLP. T5LayerNorm maps to L1/t5_layer_norm.py; the Conv1D wrapper from transformers.modeling_utils is just a Linear with a transposed weight layout.
- **classes**:
  - **`ImageGPTLayerNorm`** [compute]: `L1/t5_layer_norm.py` (Variance-only LayerNorm (no centering, no bias) - same as T5 LayerNorm.)
  - **`ImageGPTAttention`** [compute]: `L2/attention.py` (Causal self-attn with optional cross-attn; fused c_attn (Conv1D) for QKV. Layer-wise attn scaling and reorder/upcast options are pure-torch wiring on top of standard attention.)
  - **`ImageGPTMLP`** [compute]: `L2/encoder_mlp.py` (Conv1D (=Linear) c_fc -> act -> Conv1D c_proj. Not gated.)
  - **`ImageGPTBlock`** [wiring]: GPT-2 block wiring (LN -> attn -> residual -> LN -> MLP).

## informer
- **src**: modeling_informer.py
- **status**: partial
- **rationale**: Informer's defining contribution is InformerProbSparseAttention: random key sampling, sparsity measurement (max - mean) on Q-K_sample, top-u query selection, sparse attention only on top-u queries with cumsum-based context for the rest. This algorithm has no kb-nano kernel and cannot be expressed via standard SDPA; would require a bespoke CUDA kernel.
- **classes**:
  - **`InformerProbSparseAttention`** [compute]: no kb-nano kernel — Informer's defining contribution is InformerProbSparseAttention: random key sampling, sparsity measurement (max - mean) on Q-K_sample, top-u query selection, sparse attention only on top-u queries wit
  - **`InformerFeatureEmbedder`** [compute]: `L1/embedding.py` (Categorical feature embedder (multiple Embeddings).)
  - **`InformerStdScaler`** [wiring]: Standardization scaler (pure torch ops).
  - **`InformerSinusoidalPositionalEmbedding`** [compute]: `L1/sinusoidal_embed.py` (Sinusoidal positional embedding.)
  - **`InformerValueEmbedding`** [compute]: `L1/linear.py` (Linear value projection.)
  - **`InformerAttention`** [compute]: `L2/whisper_attention.py` (Standard BART-style attention (separate q/k/v/out_proj).)
  - **`InformerConvLayer`** [wiring]: Conv1d (circular padding) + BatchNorm1d + ELU + MaxPool1d. BatchNorm1d and circular-padding Conv1d not in kb-nano.
  - **`InformerEncoderLayer`** [wiring]: Wires sparse-attn + LN + FFN + LN.
  - **`InformerDecoderLayer`** [wiring]: Self-sparse-attn + cross-attn + FFN.

## instructblip
- **src**: modeling_instructblip.py
- **status**: composable
- **rationale**: BLIP CLIP-style vision encoder (fused QKV with optional q/v bias only, projection, fc1+gelu+fc2 MLP) + Q-Former (BERT-like with cross-attention) + AutoModelForCausalLM/Seq2SeqLM language model. All primitives (LayerNorm, Linear, GELU, SDPA, cross-attn) exist in kb-nano L1/L2.
- **classes**:
  - **`InstructBlipVisionEmbeddings`** [compute]: `L1/conv2d.py`, `L1/embedding.py` (CLS + Conv2d patch + position embeddings (with bicubic interpolate for resolution change).)
  - **`InstructBlipAttention`** [compute]: `L2/encoder_attention.py` (Fused QKV with concatenated q_bias/zero/v_bias, projection. Same compute as encoder_attention with bias variant.)
  - **`InstructBlipMLP`** [compute]: `L2/clip_mlp.py` (fc1 -> act (GELU) -> fc2. Matches BLIP/CLIP MLP.)
  - **`InstructBlipEncoderLayer`** [wiring]: Pre-LN attn + Pre-LN MLP.
  - **`InstructBlipEncoder`** [wiring]: Stack of layers.
  - **`InstructBlipQFormerMultiHeadAttention`** [compute]: `L2/encoder_attention.py` (BERT-style separate q/k/v Linear + cross-attn variant.)
  - **`InstructBlipQFormerSelfOutput`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (Linear + LN + residual.)
  - **`InstructBlipQFormerAttention`** [wiring]: Wires QFormerMultiHeadAttention + QFormerSelfOutput.
  - **`InstructBlipQFormerIntermediate`** [compute]: `L1/linear.py`, `L1/gelu.py` (BERT-style intermediate (Linear + GELU).)
  - **`InstructBlipQFormerOutput`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (Linear + LN + residual.)
  - **`InstructBlipQFormerLayer`** [wiring]: Self-attn + cross-attn + FFN wiring.
  - **`InstructBlipQFormerEncoder`** [wiring]: Stack of QFormer layers.
  - **`InstructBlipQFormerEmbeddings`** [compute]: `L1/embedding.py`, `L1/layer_norm.py` (Word + position embeddings + LN.)

## instructblipvideo
- **src**: modeling_instructblipvideo.py
- **status**: composable
- **rationale**: Same compute pattern as instructblip — BLIP vision encoder + Q-Former + language model. The 'Video' variant adds frame-level temporal handling (multiple frames concatenated as spatial tokens) but the per-frame compute is identical to InstructBlip.
- **classes**:
  - **`InstructBlipVideoVisionEmbeddings`** [compute]: `L1/conv2d.py`, `L1/embedding.py` (CLS + Conv2d patch + position embeddings.)
  - **`InstructBlipVideoAttention`** [compute]: `L2/encoder_attention.py` (Fused QKV with concatenated bias.)
  - **`InstructBlipVideoMLP`** [compute]: `L2/clip_mlp.py` (fc1 + GELU + fc2.)
  - **`InstructBlipVideoEncoderLayer`** [wiring]: Pre-LN attn + Pre-LN MLP.
  - **`InstructBlipVideoEncoder`** [wiring]: Stack of layers.
  - **`InstructBlipVideoQFormerMultiHeadAttention`** [compute]: `L2/encoder_attention.py` (BERT-style separate q/k/v + cross-attn.)
  - **`InstructBlipVideoQFormerSelfOutput`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (Linear + LN + residual.)
  - **`InstructBlipVideoQFormerAttention`** [wiring]: Composes self-attn + output.
  - **`InstructBlipVideoQFormerIntermediate`** [compute]: `L1/linear.py`, `L1/gelu.py` (Linear + GELU.)
  - **`InstructBlipVideoQFormerOutput`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (Linear + LN + residual.)
  - **`InstructBlipVideoQFormerLayer`** [wiring]: QFormer layer.
  - **`InstructBlipVideoQFormerEncoder`** [wiring]: Stack of QFormer layers.
  - **`InstructBlipVideoQFormerEmbeddings`** [compute]: `L1/embedding.py`, `L1/layer_norm.py` (Embedding + LN.)

## internvl
- **src**: modular_internvl.py
- **status**: composable
- **rationale**: BEiT-style vision encoder (with optional QK-RMSNorm using full-dim norm) inheriting from JanusVisionAttention + CLIPMLP (fc1+gelu+fc2) + LlavaModel for the multimodal/text side. All primitives map.
- **classes**:
  - **`InternVLVisionRMSNorm`** [compute]: `L1/rms_norm.py` (Llama RMSNorm.)
  - **`InternVLVisionAttention`** [compute]: `L2/encoder_attention.py`, `L1/rms_norm.py` (Separate q/k/v Linear + optional QK-RMSNorm at full embed_dim (not per-head); compute is encoder-attention with extra norm step.)
  - **`InternVLVisionPatchEmbeddings`** [compute]: `L1/conv2d.py` (Conv2d patch projection.)
  - **`InternVLVisionEmbeddings`** [wiring]: CLS + patch + optional absolute pos embed (with bicubic interpolate).
  - **`InternVLVisionMLP`** [compute]: `L2/clip_mlp.py` (CLIP-style fc1+act+fc2.)
  - **`InternVLVisionLayer`** [wiring]: Pre-LN attn + Pre-LN MLP.
  - **`InternVLVisionEncoder`** [wiring]: Stack of layers.
  - **`InternVLMultiModalProjector`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (Pixel-shuffle + LN + Linear projector vision -> text dim.)

## jamba
- **src**: modeling_jamba.py
- **status**: kb_nano_l4
- **rationale**: Jamba (AI21Labs hybrid: Transformer attention + Mamba SSM + MoE) is a kb-nano L4 pipeline.
- **classes**:
  - **`JambaRMSNorm`** [compute]: `L1/rms_norm.py` (Llama RMSNorm.)
  - **`JambaAttention`** [compute]: `L2/jamba_attention.py` (Llama-style attention adapted for Jamba.)
  - **`JambaMambaMixer`** [compute]: `L2/jamba_mamba_mixer.py` (Mamba v1 selective SSM.)
  - **`JambaMLP`** [compute]: `L2/jamba_mlp.py` (SwiGLU MLP for non-MoE layers.)
  - **`JambaExperts`** [compute]: `L2/jamba_moe.py` (Fused experts.)
  - **`JambaSparseMoeBlock`** [compute]: `L2/jamba_moe.py` (Top-k MoE routing.)
  - **`JambaAttentionDecoderLayer`** [compute]: `L3/jamba_decoder.py` (Attention layer wiring.)
  - **`JambaMambaDecoderLayer`** [compute]: `L3/jamba_decoder.py` (Mamba layer wiring.)

## janus
- **src**: modeling_janus.py
- **status**: composable
- **rationale**: Vision encoder (CLIP-style with QK-norm) + Llama-style language decoder + VQ-VAE encoder/decoder built from Conv2d + GroupNorm + sigmoid + 1x1-conv attention + VectorQuantizer (codebook lookup). All primitives in kb-nano (Conv2d, GroupNorm, sigmoid, embedding, interpolate). VectorQuantizer is pure-torch argmin codebook lookup composable from existing ops.
- **classes**:
  - **`JanusVisionEmbeddings`** [compute]: `L1/conv2d.py`, `L1/embedding.py` (Conv2d patch + CLS + position embedding.)
  - **`JanusVisionAttention`** [compute]: `L2/encoder_attention.py` (Separate q/k/v + optional QK-norm (full embed_dim) + projection.)
  - **`JanusVisionMLP`** [compute]: `L2/clip_mlp.py` (fc1 + act + fc2.)
  - **`JanusVisionEncoderLayer`** [wiring]: Pre-LN attn + Pre-LN MLP.
  - **`JanusVisionEncoder`** [wiring]: Stack of layers.
  - **`JanusVisionAlignerMLP`** [compute]: `L1/linear.py`, `L1/gelu.py` (Linear + GELU + Linear vision -> text alignment.)
  - **`JanusVQVAEVectorQuantizer`** [compute]: `L1/embedding.py` (Codebook lookup via argmin distance + embedding lookup; pure torch composable.)
  - **`JanusVQVAEResnetBlock`** [compute]: `L1/conv2d.py`, `L1/group_norm.py`, `L1/sigmoid.py` (GroupNorm + Conv2d + Swish (x*sigmoid).)
  - **`JanusVQVAEAttnBlock`** [compute]: `L1/conv2d.py`, `L1/group_norm.py`, `L1/sdpa.py` (1x1-conv q/k/v + manual SDPA on flattened spatial tokens.)
  - **`JanusVQVAEConvDownsample`** [compute]: `L1/conv2d.py` (Asymmetric-pad + Conv2d stride-2.)
  - **`JanusVQVAEConvUpsample`** [compute]: `L1/interpolate.py`, `L1/conv2d.py` (Nearest interpolate 2x + Conv2d.)
  - **`JanusVQVAEMidBlock`** [wiring]: ResnetBlock + AttnBlock + ResnetBlock.
  - **`JanusVQVAEEncoder`** [wiring]: Stack of resnet+attn blocks with downsampling.
  - **`JanusVQVAEDecoder`** [wiring]: Symmetric VQ-VAE decoder.
  - **`JanusVQVAEAlignerMLP`** [compute]: `L1/linear.py`, `L1/gelu.py` (Linear + GELU + Linear.)
  - **`JanusVQVAEHead`** [compute]: `L1/linear.py`, `L1/gelu.py` (Linear + GELU + Linear -> codebook logits.)

## jetmoe
- **src**: modeling_jetmoe.py
- **status**: partial
- **partial_reason**: JetMoeMoA wraps the attention computation with input-side routed q-projection and output-side routed combine, with per-expert weights stored in JetMoeParallelExperts (looped F.linear per expert). kb-nano has no equivalent for routing query projection through experts; the closest is moe_grouped_gemm but that is for MLPs.
- **rationale**: JetMoeMoE (SwiGLU MoE with index_add scatter) is composable, but JetMoeMoA (Mixture-of-Attention with routed per-expert q-projection + output-projection wrapping standard attention) is a bespoke routed-attention pattern with no kb-nano kernel; the routing topology is also custom (sort + index_sorted_experts).
- **classes**:
  - **`JetMoeAttention`** [compute]: no kb-nano kernel — JetMoeMoA wraps the attention computation with input-side routed q-projection and output-side routed combine, with per-expert weights stored in JetMoeParallelExperts (looped F.linear per expert). kb-n
  - **`JetMoeRMSNorm`** [compute]: `L1/rms_norm.py` (Llama RMSNorm.)
  - **`JetMoeRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (Standard NeoX RoPE.)
  - **`JetMoeParallelExperts`** [wiring]: Per-expert F.linear in a Python loop. Could use moe_grouped_gemm but routing topology differs.
  - **`JetMoeTopKGating`** [compute]: `L1/linear.py`, `L1/topk_softmax.py` (Linear router + topk + softmax + sort-based topology.)
  - **`JetMoeMoE`** [compute]: `L1/silu_and_mul.py` (Routed SwiGLU experts (input_linear -> chunk -> silu_and_mul -> output_linear) + index_add scatter. Composable with custom routing.)
  - **`JetMoeMoA`** [wiring]: Routed q-projection (map) + standard attention with k/v repeated top_k times + routed o-projection (reduce). No kb-nano routed-attention kernel.
  - **`JetMoeDecoderLayer`** [wiring]: Norm + self-attn + norm + MoE.

## kosmos2
- **src**: modeling_kosmos2.py
- **status**: composable
- **rationale**: CLIP-style vision encoder (separate q/k/v with bias, fc1+gelu+fc2 MLP) + BART-style text decoder (separate q/k/v/out_proj) with optional inner_attn_ln (extra LayerNorm inside attention before out_proj) + sinusoidal positional embedding + Kosmos2TextFFN with extra ffn_layernorm (LayerNorm between act and fc2). The extra LayerNorms are wiring on top of existing primitives.
- **classes**:
  - **`Kosmos2VisionEmbeddings`** [compute]: `L1/conv2d.py`, `L1/embedding.py` (CLIP-style CLS + Conv2d patch + position embeddings.)
  - **`Kosmos2VisionAttention`** [compute]: `L2/clip_attention.py` (Separate q/k/v/out_proj with bias; matches CLIP.)
  - **`Kosmos2VisionMLP`** [compute]: `L2/clip_mlp.py` (fc1 + act + fc2.)
  - **`Kosmos2VisionEncoderLayer`** [wiring]: Pre-LN attn + Pre-LN MLP.
  - **`Kosmos2VisionEncoder`** [wiring]: Stack of layers.
  - **`Kosmos2VisionTransformer`** [wiring]: Vision wiring.
  - **`Kosmos2TextSinusoidalPositionalEmbedding`** [compute]: `L1/sinusoidal_embed.py` (Sinusoidal position embedding.)
  - **`KosmosTextAttention`** [compute]: `L2/whisper_attention.py`, `L1/layer_norm.py` (BART-style separate q/k/v/out_proj + optional inner_attn_ln (LayerNorm between SDPA and out_proj). LayerNorm is L1; wrapper needs extra wiring.)
  - **`Kosmos2TextFFN`** [compute]: `L1/linear.py`, `L1/gelu.py`, `L1/layer_norm.py` (fc1 -> act -> ffn_layernorm -> fc2 (extra LN between act and fc2).)
  - **`Kosmos2TextBlock`** [wiring]: Self-attn + optional cross-attn + FFN.
  - **`Kosmos2ImageToTextProjection`** [compute]: `L1/linear.py` (Linear projection vision -> text dim.)

## kosmos2_5
- **src**: modeling_kosmos2_5.py
- **status**: composable
- **rationale**: Kosmos2.5 is a Pix2Struct-derived OCR model. Vision uses Pix2Struct-style Linear patch projection (no Conv) + row/col embedding + T5LayerNorm + T5DenseGatedActDense (gated GELU MLP) + non-causal attention (separate q/k/v Linear). Text decoder is BART-style with extra inner_attn_ln + ffn_layernorm. All primitives present.
- **classes**:
  - **`Kosmos2_5LayerNorm`** [compute]: `L1/t5_layer_norm.py` (T5-style variance-only LayerNorm (no centering, no bias).)
  - **`Kosmos2_5VisionEmbeddings`** [compute]: `L1/linear.py`, `L1/embedding.py` (Linear patch projection + row/column embeddings (Pix2Struct style).)
  - **`Kosmos2_5VisionMlp`** [compute]: `L2/t5_dense.py` (T5DenseGatedActDense: wi_0 + act -> * wi_1 -> wo. Matches T5 gated MLP.)
  - **`Kosmos2_5VisionAttention`** [compute]: `L2/encoder_attention.py` (Separate q/k/v Linear (no bias) + non-causal SDPA.)
  - **`Kosmos2_5VisionLayer`** [wiring]: T5-LN before attn + T5-LN before MLP.
  - **`Kosmos2_5VisionEncoder`** [wiring]: Stack of vision layers.
  - **`Kosmos2_5TextSinusoidalPositionalEmbedding`** [compute]: `L1/sinusoidal_embed.py` (Sinusoidal position embedding.)
  - **`Kosmos2_5TextFFN`** [compute]: `L1/linear.py`, `L1/gelu.py`, `L1/layer_norm.py` (fc1 -> act -> LN -> fc2.)
  - **`Kosmos2_5TextAttention`** [compute]: `L2/whisper_attention.py`, `L1/layer_norm.py` (BART-style with optional inner_attn_ln between SDPA and out_proj.)
  - **`Kosmos2_5TextBlock`** [wiring]: Self-attn + (optional cross-attn) + FFN.
  - **`Kosmos2_5ImageToTextProjection`** [compute]: `L1/linear.py` (Linear vision -> text projection + learned latent queries.)

## kyutai_speech_to_text
- **src**: modular_kyutai_speech_to_text.py
- **status**: partial
- **partial_reason**: The codec_model (Mimi) uses MimiConv1d with streaming padding cache and weight-normalized Conv1d/ConvTranspose1d (audio codec primitives). kb-nano has Conv1d but no weight_norm parametrization or streaming padding cache. The Llama text decoder portion is composable; only the codec audio encoder has missing primitives.
- **rationale**: Inherits Llama text decoder (composable) but wraps a Mimi audio codec model. The Mimi codec uses streaming Conv1d with weight_norm parametrization, transposed Conv1d, padding cache for streaming inference, plus Vector Quantizer codebooks. Streaming Conv1d with weight_norm has no kb-nano L1 op.
- **classes**:
  - **`KyutaiSpeechToTextConv1dPaddingCache`** [compute]: no kb-nano kernel — The codec_model (Mimi) uses MimiConv1d with streaming padding cache and weight-normalized Conv1d/ConvTranspose1d (audio codec primitives). kb-nano has Conv1d but no weight_norm parametrization or stre
  - **`KyutaiSpeechToTextEmbeddings`** [compute]: `L1/embedding.py` (Embedding table over (text vocab + audio codebook tokens) with offset table; sums per-codebook embeddings.)
  - **`KyutaiSpeechToTextModel`** [wiring]: Inherits Moshi (Llama-style) text model with custom embeddings.
  - **`KyutaiSpeechToTextForConditionalGeneration`** [wiring]: Wraps text causal LM + Mimi codec model via AutoModel.from_config.

## lasr
- **src**: modular_lasr.py
- **status**: partial
- **partial_reason**: Conformer convolution module (inherited from Parakeet) uses nn.BatchNorm1d as the in-conv normalization. kb-nano has BatchNorm2d, GroupNorm, FrozenBatchNorm2d but no BatchNorm1d. ReLU + nn.Conv1d (subsampling) are present.
- **rationale**: Conformer-style ASR encoder using Llama-style attention (composable) + ParakeetEncoderConvolutionModule which uses BatchNorm1d (kb-nano lacks BatchNorm1d L1 op).
- **classes**:
  - **`LasrEncoderBlock`** [compute]: no kb-nano kernel — Conformer convolution module (inherited from Parakeet) uses nn.BatchNorm1d as the in-conv normalization. kb-nano has BatchNorm2d, GroupNorm, FrozenBatchNorm2d but no BatchNorm1d. ReLU + nn.Conv1d (sub
  - **`LasrEncoderSubsampling`** [compute]: `L1/linear.py`, `L1/conv1d.py`, `L1/relu.py` (Linear -> Conv1d (stride 2) -> Conv1d (stride 2) -> Linear.)
  - **`LasrEncoderRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (Llama RoPE.)
  - **`LasrEncoderAttention`** [compute]: `L2/attention.py` (Llama-style GQA non-causal attention.)
  - **`LasrEncoderConvolutionModule`** [wiring]: Conformer conv module: pointwise_conv1 -> GLU -> depthwise_conv -> BatchNorm1d -> activation -> pointwise_conv2. BatchNorm1d missing in kb-nano.
  - **`LasrEncoder`** [wiring]: Subsampling + stack of Conformer blocks.

## layoutlm
- **src**: modeling_layoutlm.py
- **status**: composable
- **rationale**: BERT clone with extra 2D position embeddings (x/y/h/w from bounding boxes summed with token+pos+type embeds). Standard BERT attention (separate q/k/v Linear, no rel pos). Maps to L2/encoder_attention.py + L2/encoder_mlp.py.
- **classes**:
  - **`LayoutLMEmbeddings`** [compute]: `L1/embedding.py`, `L1/layer_norm.py` (word + position + token_type + x_position + y_position + h_position + w_position embeddings; LayerNorm + dropout.)
  - **`LayoutLMSelfAttention`** [compute]: `L2/encoder_attention.py` (Separate q/k/v Linear, no relative bias.)
  - **`LayoutLMSelfOutput`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (Linear + LN + residual.)
  - **`LayoutLMAttention`** [wiring]: Composes SelfAttention + SelfOutput.
  - **`LayoutLMIntermediate`** [compute]: `L1/linear.py`, `L1/gelu.py` (Linear + GELU.)
  - **`LayoutLMOutput`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (Linear + LN + residual.)
  - **`LayoutLMLayer`** [wiring]: BERT layer wiring.
  - **`LayoutLMEncoder`** [wiring]: Stack of layers.
  - **`LayoutLMPooler`** [compute]: `L1/linear.py`, `L1/tanh.py` (Linear + Tanh on CLS.)

## layoutlmv2
- **src**: modeling_layoutlmv2.py
- **status**: unsupported
- **unsupported_reason**: LayoutLMv2VisualBackbone uses detectron2.modeling.backbone.FPN with META_ARCH_REGISTRY (external dependency). Also LayoutLMv2SelfAttention adds rel_pos and rel_2d_pos as additive attention bias which kb-nano L1 flash_attn_* and dense_attention do not support (no alibi_slopes-style additive bias parameter).
- **rationale**: LayoutLMv2 pulls a detectron2 visual backbone (FPN-based) and uses additive 1D + 2D relative position bias in self-attention. The detectron2 backbone is an external library with no kb-nano equivalent; additive attention bias is also not supported by kb-nano flash kernels.
- **classes**:
  - **`LayoutLMv2SelfAttention`** [compute]: LayoutLMv2VisualBackbone uses detectron2.modeling.backbone.FPN with META_ARCH_REGISTRY (external dependency). Also LayoutLMv2SelfAttention adds rel_pos and rel_2d_pos as additive attention bias which 
  - **`LayoutLMv2Embeddings`** [compute]: `L1/embedding.py`, `L1/layer_norm.py` (Word + 2D position embeddings + LN.)
  - **`LayoutLMv2VisualBackbone`** [wiring]: Wraps detectron2 FPN backbone + AdaptiveAvgPool2d. External dependency, no kb-nano equivalent.
  - **`LayoutLMv2Encoder`** [wiring]: Stack with relative_position_bucket / relative_position_bias_table computation; integrates with attention.
  - **`LayoutLMv2Layer`** [wiring]: BERT layer wiring.
  - **`LayoutLMv2Pooler`** [compute]: `L1/linear.py`, `L1/tanh.py` (Linear + Tanh.)

## layoutlmv3
- **src**: modeling_layoutlmv3.py
- **status**: partial
- **partial_reason**: LayoutLMv3SelfAttention adds rel_pos + rel_2d_pos to attention scores before softmax (additive bias not supported by kb-nano flash kernels) and uses the CogView numerical-stability softmax variant (not in kb-nano L1/softmax.py). Otherwise the rest (BERT-style q/k/v Linear, intermediate, output) is composable.
- **rationale**: LayoutLMv3 uses additive rel_pos + rel_2d_pos attention bias and a CogView softmax variant (subtract max scaled by alpha then re-scale before softmax) for numerical stability. Additive attention bias is not supported by kb-nano flash kernels; the CogView softmax variant is not the standard softmax in kb-nano L1.
- **classes**:
  - **`LayoutLMv3SelfAttention`** [compute]: no kb-nano kernel — LayoutLMv3SelfAttention adds rel_pos + rel_2d_pos to attention scores before softmax (additive bias not supported by kb-nano flash kernels) and uses the CogView numerical-stability softmax variant (no
  - **`LayoutLMv3PatchEmbeddings`** [compute]: `L1/conv2d.py` (Conv2d patch embed (no detectron2 in v3, unlike v2).)
  - **`LayoutLMv3TextEmbeddings`** [compute]: `L1/embedding.py`, `L1/layer_norm.py` (Word + position + token_type + 2D position embeddings + LN.)
  - **`LayoutLMv3SelfOutput`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (Linear + LN + residual.)
  - **`LayoutLMv3Attention`** [wiring]: Composes SelfAttention + SelfOutput.
  - **`LayoutLMv3Layer`** [wiring]: BERT layer wiring with rel_pos passed through.
  - **`LayoutLMv3Encoder`** [wiring]: Stack with rel_pos / rel_2d_pos bucket computation.
  - **`LayoutLMv3Intermediate`** [compute]: `L1/linear.py`, `L1/gelu.py` (Linear + GELU.)
  - **`LayoutLMv3Output`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (Linear + LN + residual.)
  - **`LayoutLMv3ClassificationHead`** [compute]: `L1/linear.py`, `L1/tanh.py` (Linear + Tanh + Linear classification.)

## led
- **src**: modeling_led.py
- **status**: partial
- **rationale**: LED uses Longformer-style sliding-window self-attention (O(N*W)) with global attention on a subset of tokens. The chunked sliding-window matmul algorithm (_sliding_chunks_query_key_matmul, _sliding_chunks_matmul_attn_probs_value) and global-token attention combination are bespoke; kb-nano has no Longformer/sliding-window-attention kernel.
- **classes**:
  - **`LEDEncoderSelfAttention`** [compute]: no kb-nano kernel — LED uses Longformer-style sliding-window self-attention (O(N*W)) with global attention on a subset of tokens. The chunked sliding-window matmul algorithm (_sliding_chunks_query_key_matmul, _sliding_ch
  - **`LEDLearnedPositionalEmbedding`** [compute]: `L1/embedding.py` (Learned position embedding.)
  - **`LEDEncoderAttention`** [wiring]: Wraps LEDEncoderSelfAttention + output projection.
  - **`LEDDecoderAttention`** [compute]: `L2/whisper_attention.py` (Standard BART-style decoder attention; this part is composable but the encoder is not.)
  - **`LEDEncoderLayer`** [wiring]: Wraps LEDEncoderAttention + LN + FFN.
  - **`LEDDecoderLayer`** [wiring]: Self-attn + cross-attn + FFN.
  - **`LEDClassificationHead`** [compute]: `L1/linear.py`, `L1/tanh.py` (MLP classification head.)

## levit
- **src**: modeling_levit.py
- **status**: partial
- **partial_reason**: MLPLayerWithBN applies nn.BatchNorm1d after every Linear; kb-nano has BatchNorm2d but no BatchNorm1d. LevitAttention/LevitAttentionSubsample also add a learned 2D positional attention_biases tensor to attention scores before softmax — kb-nano flash kernels do not support additive attention bias.
- **rationale**: LeViT uses BatchNorm2d after every Conv2d (LevitConvEmbeddings) and BatchNorm1d after every Linear (MLPLayerWithBN), Hardswish activation, and a learned 2D positional attention bias added to attention scores. BatchNorm1d is missing in kb-nano, and additive attention bias is not supported by kb-nano flash kernels.
- **classes**:
  - **`LevitAttention`** [compute]: no kb-nano kernel — MLPLayerWithBN applies nn.BatchNorm1d after every Linear; kb-nano has BatchNorm2d but no BatchNorm1d. LevitAttention/LevitAttentionSubsample also add a learned 2D positional attention_biases tensor to
  - **`LevitConvEmbeddings`** [compute]: `L1/conv2d.py`, `L1/batch_norm2d.py` (Conv2d + BatchNorm2d.)
  - **`LevitPatchEmbeddings`** [compute]: `L1/hardswish.py` (Stack of LevitConvEmbeddings + Hardswish; downsamples 4 stages of conv+BN.)
  - **`MLPLayerWithBN`** [compute]: `L1/linear.py` (Linear + BatchNorm1d. BatchNorm1d missing in kb-nano.)
  - **`LevitSubsample`** [wiring]: Strided slicing on spatial grid (pure torch view ops).
  - **`LevitAttentionSubsample`** [wiring]: Subsampled queries + KV proj + manual SDPA with 2D positional bias.
  - **`LevitMLPLayer`** [compute]: `L1/hardswish.py` (MLPLayerWithBN -> Hardswish -> MLPLayerWithBN.)
  - **`LevitResidualLayer`** [wiring]: Residual wrapper with stochastic depth.
  - **`LevitStage`** [wiring]: Stack of attn + MLP residual blocks per stage.
  - **`LevitEncoder`** [wiring]: Stack of stages.
  - **`LevitClassificationLayer`** [compute]: `L1/batch_norm2d.py`, `L1/linear.py` (BatchNorm + Linear classifier head; uses BatchNorm1d-on-features.)
