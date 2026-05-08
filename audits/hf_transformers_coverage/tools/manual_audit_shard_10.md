## nomic_bert
- **src**: modular_nomic_bert.py
- **status**: composable
- **rationale**: BERT-style encoder + RoPE + SwiGLU MLP (GemmaMLP) inheriting JinaEmbeddingsV3 layer; all primitives map to encoder_attention + llama_mlp + rotary_emb + layer_norm.
- **classes**:
  - **`NomicBertEmbeddings`** [compute]: `L1/embedding.py`, `L1/layer_norm.py` (Token + token_type sum + LayerNorm; standard embedding wiring.)
  - **`NomicBertRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (Standard NeoX/Llama RoPE.)
  - **`NomicBertAttention`** [compute]: `L2/encoder_attention.py`, `L1/sdpa.py`, `L1/linear.py`, `L1/rotary_emb.py` (Bidirectional multi-head attention with RoPE; bias=False overrides parent's bias=True. Maps to EncoderSelfAttention pattern with RoPE applied before SDPA.)
  - **`NomicBertMLP`** [compute]: `L2/llama_mlp.py`, `L1/silu_and_mul.py` (SwiGLU: gate * silu * up -> down (GemmaMLP); maps to LlamaMLP/SiluAndMul.)
  - **`NomicBertLayer`** [wiring]: Wiring: post-attn LN + post-MLP LN; composes self_attn + mlp.
  - **`NomicBertModel`** [wiring]: Top-level wiring (embeddings + layers + pooler).
  - **`NomicBertPredictionHeadTransform`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (Dense + activation + LayerNorm; standard primitives.)
  - **`NomicBertOnlyMLMHead`** [wiring]: Wiring around prediction head.
  - **`NomicBertForMaskedLM`** [wiring]: Wiring class composing model + cls head.
  - **`NomicBertForSequenceClassification`** [wiring]: Wiring class around model + classification head.
  - **`NomicBertForTokenClassification`** [wiring]: Wiring class around model + token-level head.

## nystromformer
- **src**: modeling_nystromformer.py
- **status**: partial
- **rationale**: Nystromformer's self-attention is a custom Nystrom approximation requiring iterative pseudo-inverse computation and depthwise conv2d residual — no kb-nano kernel approximates this.
- **classes**:
  - **`NystromformerSelfAttention`** [compute]: Nystromformer's self-attention is a custom Nystrom approximation requiring iterative pseudo-inverse computation and depthwise conv2d residual — no kb-nano kernel approximates this.
  - **`NystromformerEmbeddings`** [compute]: `L1/embedding.py`, `L1/layer_norm.py` (Token + position + token_type embeddings + LayerNorm + dropout.)
  - **`NystromformerSelfOutput`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (Dense + dropout + LayerNorm(residual); standard ops.)
  - **`NystromformerAttention`** [wiring]: Wiring around SelfAttention + SelfOutput.
  - **`NystromformerIntermediate`** [compute]: `L2/encoder_mlp.py`, `L1/linear.py`, `L1/gelu.py` (Two-layer feedforward (fc1 + activation); standard encoder MLP.)
  - **`NystromformerOutput`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (Dense + dropout + LayerNorm(residual).)
  - **`NystromformerLayer`** [wiring]: Wiring: attention + intermediate + output.
  - **`NystromformerEncoder`** [wiring]: Stack of layers.
  - **`NystromformerModel`** [wiring]: Wiring: embeddings + encoder.
  - **`NystromformerForMaskedLM`** [wiring]: Wiring: model + MLM head.
  - **`NystromformerForSequenceClassification`** [wiring]: Wiring: model + classification head.
  - **`NystromformerForMultipleChoice`** [wiring]: Wiring.
  - **`NystromformerForTokenClassification`** [wiring]: Wiring.
  - **`NystromformerForQuestionAnswering`** [wiring]: Wiring.

## olmo
- **src**: modular_olmo.py
- **status**: partial
- **partial_reason**: OlmoAttention applies torch.Tensor.clamp_ on Q/K/V if config.clip_qkv is set; kb-nano L2/attention.py:LlamaAttention has no clip_qkv option. clamp_ is a torch primitive but isn't exposed through the kb-nano attention class.
- **rationale**: OLMo is Llama-style (LlamaAttention + LlamaMLP + RoPE) but with parameterless LayerNorm and per-projection clip_qkv (clamp Q/K/V before reshape). kb-nano has the L2/attention.py + L1/layer_norm.py with elementwise_affine=False, but no clip_qkv knob.
- **classes**:
  - **`OlmoDecoderLayer`** [compute]: OlmoAttention applies torch.Tensor.clamp_ on Q/K/V if config.clip_qkv is set; kb-nano L2/attention.py:LlamaAttention has no clip_qkv option. clamp_ is a torch primitive but isn't exposed through the k
  - **`OlmoLayerNorm`** [compute]: `L1/layer_norm.py` (Parameterless LayerNorm (no weight/bias). kb-nano L1/layer_norm.py supports elementwise_affine=False.)
  - **`OlmoMLP`** [compute]: `L2/llama_mlp.py`, `L1/silu_and_mul.py` (SwiGLU: act(gate) * up -> down.)
  - **`OlmoRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (Standard NeoX RoPE; only difference is fp32-cast cos/sin output.)
  - **`OlmoAttention`** [compute]: `L2/attention.py`, `L1/rotary_emb.py`, `L1/sdpa.py` (Llama-style GQA + RoPE; clip_qkv clamp on Q/K/V is the partial gap (no kb-nano knob).)
  - **`OlmoModel`** [wiring]: Wiring class composing decoder layers.
  - **`OlmoForCausalLM`** [wiring]: Wiring: model + lm_head.
  - **`OlmoForSequenceClassification`** [wiring]: Wiring.

## olmo2
- **src**: modular_olmo2.py
- **status**: composable
- **rationale**: OLMo2 = OLMo + RMSNorm everywhere + full-width q_norm/k_norm + post-norm placement. No clip_qkv. All primitives (RMSNorm, LlamaAttention with q_norm wiring, RoPE, SwiGLU MLP) exist in kb-nano.
- **classes**:
  - **`Olmo2RMSNorm`** [compute]: `L1/rms_norm.py` (Llama-style RMSNorm with weight*hidden multiplied before dtype cast back.)
  - **`Olmo2RotaryEmbedding`** [compute]: `L1/rotary_emb.py` (NeoX RoPE.)
  - **`Olmo2Attention`** [compute]: `L2/attention.py`, `L1/rms_norm.py`, `L1/rotary_emb.py`, `L1/sdpa.py` (qk_norm applied with full-width RMSNorm(num_heads*head_dim) on Q/K projection output, then reshape; mathematically a per-token full-width RMSNorm. Maps to L1/rms_norm + L2/attention wiring.)
  - **`Olmo2DecoderLayer`** [wiring]: Wiring: post-attention LN + post-feedforward LN (post-norm placement).
  - **`Olmo2Model`** [wiring]: Wiring composing decoder layers + final RMSNorm.
  - **`Olmo2ForCausalLM`** [wiring]: Wiring.
  - **`Olmo2ForSequenceClassification`** [wiring]: Wiring.

## olmo3
- **src**: modular_olmo3.py
- **status**: composable
- **rationale**: OLMo3 = OLMo2 + sliding window attention (every 4th layer is full, others sliding); kb-nano L2/attention.py supports sliding_window argument.
- **classes**:
  - **`Olmo3RMSNorm`** [compute]: `L1/rms_norm.py` (RMSNorm.)
  - **`Olmo3Attention`** [compute]: `L2/attention.py`, `L1/rms_norm.py`, `L1/rotary_emb.py`, `L1/sdpa.py` (Adds sliding_window kwarg to attention call; kb-nano LlamaAttention supports sliding_window.)
  - **`Olmo3DecoderLayer`** [wiring]: Wiring.
  - **`Olmo3RotaryEmbedding`** [compute]: `L1/rotary_emb.py` (Standard NeoX RoPE.)
  - **`Olmo3Model`** [wiring]: Wiring with mixed attention masks (full vs sliding) per layer.
  - **`Olmo3ForCausalLM`** [wiring]: Wiring.
  - **`Olmo3ForSequenceClassification`** [wiring]: Wiring.

## olmo_hybrid
- **src**: modular_olmo_hybrid.py
- **status**: composable
- **rationale**: Hybrid Llama + GatedDeltaNet linear attention (Qwen3-Next family) with optional NoPE. All compute kernels exist (CausalConv1d, GDN chunk/recurrent, RMSNormGated, L2 norm). Weight layout differs from kb-nano qwen3_next (separate q/k/v/a/b/g projections with per-projection conv1d) but the L1 ops are reusable.
- **classes**:
  - **`OlmoHybridRMSNormGated`** [compute]: `L1/rms_norm_gated.py` (Gated RMSNorm used at GDN output.)
  - **`OlmoHybridRMSNorm`** [compute]: `L1/rms_norm.py` (Standard Llama RMSNorm.)
  - **`OlmoHybridShortConvolution`** [compute]: `L1/causal_conv1d.py` (Depthwise causal conv1d with cached state; SiLU activation. Maps to kb-nano CausalConv1d.)
  - **`OlmoHybridAttention`** [compute]: `L2/attention.py`, `L1/rms_norm.py`, `L1/rotary_emb.py`, `L1/sdpa.py` (Olmo3-style attention with optional NoPE (skip RoPE if position_embeddings is None).)
  - **`OlmoHybridRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (RoPE returning fp32 cos/sin.)
  - **`OlmoHybridGatedDeltaNet`** [compute]: `L1/causal_conv1d.py`, `L1/chunk_gated_delta_rule.py`, `L1/gdn_recurrence.py`, `L1/rms_norm_gated.py`, `L1/l2norm_kernel.py`, `L1/softplus.py` (GDN with separate q/k/v/a/b/g projections (vs Qwen3-Next fused qkvz+ba) and per-projection conv1d. Compute kernels exist; new L2 wiring required but no missing primitive.)
  - **`OlmoHybridMLP`** [compute]: `L2/llama_mlp.py`, `L1/silu_and_mul.py` (SwiGLU MLP.)
  - **`OlmoHybridAttentionDecoderLayer`** [wiring]: Wiring for full-attention layer.
  - **`OlmoHybridLinearAttentionDecoderLayer`** [wiring]: Wiring for linear-attention (GDN) layer.
  - **`OlmoHybridModel`** [wiring]: Top-level wiring: alternates linear/full attention layers per layer_types.
  - **`OlmoHybridForCausalLM`** [wiring]: Wiring.

## olmoe
- **src**: modular_olmoe.py
- **status**: partial
- **partial_reason**: OlmoeAttention applies torch.Tensor.clamp_ on Q/K/V if config.clip_qkv is set; same gap as olmo. kb-nano L2/attention.py:LlamaAttention has no clip_qkv knob.
- **rationale**: OLMoE = Mixtral MoE + Llama RoPE + GemmaMLP + qk_norm (full-width) + clip_qkv. clip_qkv clamp on Q/K/V is not exposed via kb-nano L2/attention.py.
- **classes**:
  - **`OlmoeSparseMoeBlock`** [compute]: OlmoeAttention applies torch.Tensor.clamp_ on Q/K/V if config.clip_qkv is set; same gap as olmo. kb-nano L2/attention.py:LlamaAttention has no clip_qkv knob.
  - **`OlmoeRMSNorm`** [compute]: `L1/rms_norm.py` (Standard Llama RMSNorm.)
  - **`OlmoeRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (NeoX RoPE.)
  - **`OlmoeMLP`** [compute]: `L2/llama_mlp.py`, `L1/silu_and_mul.py` (SwiGLU.)
  - **`OlmoeAttention`** [compute]: `L2/attention.py`, `L1/rms_norm.py`, `L1/rotary_emb.py`, `L1/sdpa.py` (qk_norm (full-width) + clip_qkv + sliding_window. clip_qkv is the partial gap.)
  - **`OlmoeExperts`** [compute]: `L2/mixtral_moe.py`, `L1/moe_grouped_gemm.py` (Mixtral-style fused MoE experts; gate_up_proj + down_proj with grouped GEMM.)
  - **`OlmoeTopKRouter`** [compute]: `L1/topk_softmax.py`, `L1/linear.py` (Top-k softmax router (Qwen2MoE-style).)
  - **`OlmoeDecoderLayer`** [wiring]: Wiring: self_attn + sparse moe.
  - **`OlmoeModel`** [wiring]: Top-level wiring.
  - **`OlmoeForCausalLM`** [wiring]: Wiring: model + lm_head.

## omdet_turbo
- **src**: modeling_omdet_turbo.py
- **status**: partial
- **rationale**: Open-vocab detection model: depends on AutoBackbone (timm) for vision tower, AutoModel for text, plus multi-scale deformable attention (MSDA v1, kernels-community kernel) and a custom hybrid encoder/decoder. No L4 pipeline; kb-nano deformable attention is RT-DETR v2-specific.
- **classes**:
  - **`OmDetTurboEncoderLayer`** [compute]: no kb-nano kernel — Open-vocab detection model: depends on AutoBackbone (timm) for vision tower, AutoModel for text, plus multi-scale deformable attention (MSDA v1, kernels-community kernel) and a custom hybrid encoder/d
  - **`MultiScaleDeformableAttention`** [compute]: `L1/rtdetrv2_deformable_attention.py` (Bilinear sampling + weighted aggregation; kb-nano has the V2 sampling primitive (compatible).)
  - **`OmDetTurboLanguageBackbone`** [wiring]: Wraps AutoModel for text encoding; AutoModel coupling.
  - **`OmDetTurboVisionBackbone`** [wiring]: Wraps AutoBackbone with timm_kwargs; external library dependency.
  - **`OmDetTurboMultiscaleDeformableAttention`** [compute]: `L2/rtdetrv2_deformable_attention.py`, `L1/rtdetrv2_deformable_attention.py` (Deformable-DETR style MSDA (V1). kb-nano has the V2 variant; close but not bit-identical.)
  - **`OmDetTurboConvNormLayer`** [compute]: `L2/rtdetrv2_conv_norm.py`, `L1/conv2d.py`, `L1/batch_norm2d.py` (Conv2d + BatchNorm2d + activation.)
  - **`OmDetTurboRepVggBlock`** [compute]: `L2/rtdetrv2_repvgg_block.py` (RepVGG block (3x3 + 1x1 conv branches).)
  - **`OmDetTurboCSPRepLayer`** [compute]: `L2/rtdetrv2_csp_rep_layer.py` (CSP repvgg layer.)
  - **`OmDetTurboMultiheadAttention`** [compute]: `L2/rtdetrv2_multihead_attention.py` (Standard multi-head attention with manual scaling.)
  - **`OmDetTurboEncoder`** [wiring]: Stack.
  - **`OmDetTurboHybridEncoder`** [wiring]: Wiring with FPN.
  - **`OmDetTurboMLPWithDropout`** [compute]: `L1/linear.py`, `L1/relu.py` (MLP.)
  - **`OmDetTurboMLP`** [compute]: `L1/linear.py` (MLP.)
  - **`OmDetTurboResidualLayer`** [wiring]: Wiring.
  - **`OmDetTurboTaskEncoder`** [wiring]: Wiring.
  - **`OmDetTurboDeformableTransformerDecoderLayer`** [wiring]: Wiring around MSDA + cross-attention + FFN.
  - **`OmDetTurboDecoder`** [wiring]: Stack of decoder layers + heads.
  - **`OmDetTurboForObjectDetection`** [wiring]: Top-level wiring.

## oneformer
- **src**: modeling_oneformer.py
- **status**: partial
- **rationale**: OneFormer universal segmentation depends on multi-scale deformable attention, bare nn.MultiheadAttention (Transformer decoder cross-attn), Hungarian matcher (scipy), and AutoBackbone (timm/swin/dinat). No kb-nano equivalent for the cross-attention decoder + matching pipeline.
- **classes**:
  - **`OneFormerPixelDecoderEncoderLayer`** [compute]: no kb-nano kernel — OneFormer universal segmentation depends on multi-scale deformable attention, bare nn.MultiheadAttention (Transformer decoder cross-attn), Hungarian matcher (scipy), and AutoBackbone (timm/swin/dinat)
  - **`OneFormerHungarianMatcher`** [wiring]: scipy linear_sum_assignment; loss-time only.
  - **`OneFormerLoss`** [wiring]: Loss-time computations including focal/dice.
  - **`OneFormerPixelDecoderEncoderMultiscaleDeformableAttention`** [compute]: `L2/rtdetrv2_deformable_attention.py`, `L1/rtdetrv2_deformable_attention.py` (Deformable-DETR MSDA; kb-nano has V2 sampling.)
  - **`OneFormerPixelDecoderEncoderOnly`** [wiring]: Stack.
  - **`OneFormerPixelDecoder`** [wiring]: Wiring.
  - **`OneFormerPixelLevelModule`** [wiring]: Wraps backbone + pixel decoder.
  - **`OneFormerAttention`** [compute]: `L1/linear.py`, `L1/sdpa.py` (Standard MHA with optional positional bias on Q/K (DETR-style); composable in principle.)
  - **`OneFormerTransformerDecoderSelfAttentionLayer`** [wiring]: Wiring.
  - **`OneFormerTransformerDecoderCrossAttentionLayer`** [wiring]: Uses nn.MultiheadAttention directly.
  - **`OneFormerTransformerDecoderFFNLayer`** [compute]: `L1/linear.py` (FFN.)
  - **`OneFormerMLPPredictionHead`** [compute]: `L1/linear.py` (MLP head.)
  - **`OneFormerTransformerDecoderLayer`** [wiring]: Wiring.
  - **`OneFormerTransformerDecoderQueryTransformerDecoder`** [wiring]: Wiring.
  - **`OneFormerTransformerDecoderQueryTransformerDecoderLayer`** [wiring]: Wiring.
  - **`OneFormerTransformerDecoderQueryTransformer`** [wiring]: Wiring.
  - **`OneFormerTransformerDecoder`** [wiring]: Wiring.
  - **`OneFormerTransformerModule`** [wiring]: Wiring.
  - **`OneFormerSinePositionEmbedding`** [compute]: `L1/sinusoidal_embed.py` (Sinusoidal pos embeddings.)
  - **`PredictionBlock`** [compute]: `L1/linear.py` (Linear + activation.)
  - **`OneFormerTextMapperAttention`** [compute]: `L1/linear.py`, `L1/sdpa.py` (MHA.)
  - **`OneFormerTextTransformerDecoderLayer`** [wiring]: Wiring.
  - **`OneFormerTextContextDecoder`** [wiring]: Wiring.
  - **`OneFormerTextMLP`** [compute]: `L2/clip_mlp.py` (CLIP-style MLP (fc1 + activation + fc2).)
  - **`OneFormerTextTransformerLayer`** [wiring]: Wiring.
  - **`OneFormerTextTransformer`** [wiring]: Stack.
  - **`OneFormerTextEncoder`** [wiring]: Wiring.
  - **`OneFormerTextMapper`** [wiring]: Wiring.
  - **`OneFormerTaskModel`** [wiring]: Wiring.
  - **`OneFormerModel`** [wiring]: Top-level wiring.
  - **`OneFormerForUniversalSegmentation`** [wiring]: Top-level wiring with loss + Hungarian matching.

## openai
- **src**: modeling_openai.py
- **status**: composable
- **rationale**: OpenAI GPT-1: causal MHA via Conv1D-style fused QKV projection (a torch.nn.Linear under the hood), GELU MLP, LayerNorm, learned positional embeddings. All primitives map to kb-nano L1 ops.
- **classes**:
  - **`Attention`** [compute]: `L1/linear.py`, `L1/sdpa.py`, `L1/softmax.py` (Conv1D-fused QKV (= Linear), causal mask via lower-triangular bias, scaled dot-product attention. Maps to standard linear + sdpa.)
  - **`MLP`** [compute]: `L1/linear.py`, `L1/gelu.py` (Two-layer Conv1D MLP with GELU/swish; maps to Linear + GELU + Linear.)
  - **`Block`** [wiring]: Wiring: attn + LN + MLP + LN.
  - **`OpenAIGPTSequenceSummary`** [compute]: `L1/linear.py`, `L1/tanh.py` (Optional projection + activation for pooled summary.)
  - **`OpenAIGPTModel`** [wiring]: Wiring.
  - **`OpenAIGPTLMHeadModel`** [wiring]: Wiring.
  - **`OpenAIGPTDoubleHeadsModel`** [wiring]: Wiring.
  - **`OpenAIGPTForSequenceClassification`** [wiring]: Wiring.

## openai_privacy_filter
- **src**: modular_openai_privacy_filter.py
- **status**: composable
- **rationale**: Bidirectional GPT-OSS variant: encoder-style attention with sinks, sliding window, interleaved-RoPE; MoE experts with chunk-split clamped GLU. All compute primitives exist (sinks, swiglu_oai, RoPE, sliding-window SDPA, fused experts).
- **classes**:
  - **`OpenAIPrivacyFilterRMSNorm`** [compute]: `L1/rms_norm.py` (Standard RMSNorm.)
  - **`OpenAIPrivacyFilterRotaryEmbedding`** [compute]: `L1/yarn_rotary_emb.py` (GPT-OSS YaRN-style RoPE.)
  - **`OpenAIPrivacyFilterAttention`** [compute]: `L2/gpt_oss_attention.py`, `L1/yarn_rotary_emb.py`, `L1/sdpa.py` (Bidirectional (is_causal=False) GPT-OSS attention with sinks + sliding window. kb-nano L2/gpt_oss_attention covers sinks/SWA; bidirectional flag is a configuration of the same compute.)
  - **`OpenAIPrivacyFilterExperts`** [compute]: `L1/swiglu_oai.py`, `L1/linear.py` (BF16 MoE experts (not MXFP4) with chunk-split clamped GLU. Compute uses linear + sigmoid + clamp; kb-nano L1/swiglu_oai.py implements the (up+1)*gate*sigmoid(alpha*gate) formula. Uses non-quantized fused-experts pattern.)
  - **`OpenAIPrivacyFilterTopKRouter`** [compute]: `L1/linear.py`, `L1/topk_softmax.py` (Linear + topk + softmax router.)
  - **`OpenAIPrivacyFilterMLP`** [wiring]: Wiring around router + experts.
  - **`OpenAIPrivacyFilterEncoderLayer`** [wiring]: Wiring.
  - **`OpenAIPrivacyFilterModel`** [wiring]: Wiring with bidirectional+SWA mask.
  - **`OpenAIPrivacyFilterForTokenClassification`** [wiring]: Wiring.

## opt
- **src**: modeling_opt.py
- **status**: composable
- **rationale**: OPT decoder: standard causal multi-head attention with bias, two-layer MLP (fc1+activation+fc2), LayerNorm, learned positional embeddings. All primitives exist in kb-nano.
- **classes**:
  - **`OPTLearnedPositionalEmbedding`** [compute]: `L1/embedding.py` (Learned positional embedding with offset of 2.)
  - **`OPTAttention`** [compute]: `L1/linear.py`, `L1/sdpa.py` (Standard causal MHA with optional bias; maps to linear + sdpa.)
  - **`OPTDecoderLayer`** [wiring]: Wiring: pre/post LN + attention + fc1 + activation + fc2.
  - **`OPTDecoder`** [wiring]: Stack with optional embed projection.
  - **`OPTModel`** [wiring]: Wiring.
  - **`OPTForCausalLM`** [wiring]: Wiring.
  - **`OPTForSequenceClassification`** [wiring]: Wiring.
  - **`OPTForQuestionAnswering`** [wiring]: Wiring.

## ovis2
- **src**: modular_ovis2.py
- **status**: partial
- **partial_reason**: Ovis2VisionModel.forward calls nn.functional.gumbel_softmax (when tokenize_function='gumbel_argmax') which is not in kb-nano L1; it's a torch primitive but no fused kernel.
- **rationale**: VLM combining AIMV2/SigLIP-style vision encoder with AutoModel-loaded LLM. Vision tokenization uses gumbel_softmax / hard_softmax / softmax with straight-through estimator — gumbel_softmax has no kb-nano kernel.
- **classes**:
  - **`Ovis2VisionEncoderLayer`** [compute]: Ovis2VisionModel.forward calls nn.functional.gumbel_softmax (when tokenize_function='gumbel_argmax') which is not in kb-nano L1; it's a torch primitive but no fused kernel.
  - **`Ovis2RMSNorm`** [compute]: `L1/rms_norm.py` (Standard RMSNorm.)
  - **`Ovis2VisionMLP`** [compute]: `L2/llama_mlp.py`, `L1/silu_and_mul.py` (SwiGLU MLP.)
  - **`Ovis2VisionEmbeddings`** [compute]: `L1/conv2d.py`, `L1/embedding.py`, `L1/rms_norm.py` (Patch embed (Conv2d) + RMSNorm + position embedding.)
  - **`Ovis2VisionAttention`** [compute]: `L2/siglip_attention.py`, `L1/sdpa.py`, `L1/linear.py` (AIMV2/SigLIP-style multi-head attention.)
  - **`Ovis2VisionEncoder`** [wiring]: Stack.
  - **`Ovis2VisionTransformer`** [wiring]: Wiring.
  - **`Ovis2VisualEmbeddingTable`** [compute]: `L1/embedding.py` (Embedding lookup or matmul depending on dtype.)
  - **`Ovis2VisionModel`** [wiring]: Top vision tower; uses gumbel_softmax (partial gap).
  - **`Ovis2Model`** [wiring]: Wiring around vision tower + AutoModel-loaded LLM.
  - **`Ovis2ForConditionalGeneration`** [wiring]: Top-level wiring.

## owlv2
- **src**: modeling_owlv2.py
- **status**: composable
- **rationale**: OWL-ViT v2 = CLIP-derived dual-encoder (CLIP attention + CLIP MLP) for open-vocab object detection. Vision and text towers are composable using kb-nano CLIP L2 modules; the box/class prediction heads are simple MLPs.
- **classes**:
  - **`Owlv2VisionEmbeddings`** [compute]: `L1/conv2d.py`, `L1/embedding.py` (Patch embed + class token + position embedding.)
  - **`Owlv2TextEmbeddings`** [compute]: `L1/embedding.py` (Token + position embedding.)
  - **`Owlv2Attention`** [compute]: `L2/clip_attention.py`, `L1/sdpa.py`, `L1/linear.py` (CLIP-style bidirectional MHA with q/k/v projections and out_proj.)
  - **`Owlv2MLP`** [compute]: `L2/clip_mlp.py`, `L1/linear.py` (CLIP MLP (fc1 + activation + fc2).)
  - **`Owlv2EncoderLayer`** [wiring]: Wiring: LN + self_attn + LN + mlp.
  - **`Owlv2Encoder`** [wiring]: Stack.
  - **`Owlv2TextTransformer`** [wiring]: Wiring.
  - **`Owlv2TextModel`** [wiring]: Wiring.
  - **`Owlv2VisionTransformer`** [wiring]: Wiring.
  - **`Owlv2VisionModel`** [wiring]: Wiring.
  - **`Owlv2Model`** [wiring]: Top-level wiring with text + vision projection heads.
  - **`Owlv2BoxPredictionHead`** [compute]: `L1/linear.py`, `L1/gelu.py` (MLP for box prediction.)
  - **`Owlv2ClassPredictionHead`** [compute]: `L1/linear.py` (Linear projection.)
  - **`Owlv2ForObjectDetection`** [wiring]: Top-level wiring with detection heads.

## owlvit
- **src**: modeling_owlvit.py
- **status**: composable
- **rationale**: OWL-ViT v1 = CLIP-derived dual-encoder for open-vocab object detection (same structure as Owlv2). All compute maps to kb-nano CLIP L2.
- **classes**:
  - **`OwlViTVisionEmbeddings`** [compute]: `L1/conv2d.py`, `L1/embedding.py` (Patch embed + class token + position embedding.)
  - **`OwlViTTextEmbeddings`** [compute]: `L1/embedding.py` (Token + position embedding.)
  - **`OwlViTAttention`** [compute]: `L2/clip_attention.py`, `L1/sdpa.py`, `L1/linear.py` (CLIP-style multi-head attention.)
  - **`OwlViTMLP`** [compute]: `L2/clip_mlp.py`, `L1/linear.py` (CLIP MLP.)
  - **`OwlViTEncoderLayer`** [wiring]: Wiring.
  - **`OwlViTEncoder`** [wiring]: Stack.
  - **`OwlViTTextTransformer`** [wiring]: Wiring.
  - **`OwlViTTextModel`** [wiring]: Wiring.
  - **`OwlViTVisionTransformer`** [wiring]: Wiring.
  - **`OwlViTVisionModel`** [wiring]: Wiring.
  - **`OwlViTModel`** [wiring]: Top-level wiring.
  - **`OwlViTBoxPredictionHead`** [compute]: `L1/linear.py`, `L1/gelu.py` (MLP.)
  - **`OwlViTClassPredictionHead`** [compute]: `L1/linear.py` (Linear projection.)
  - **`OwlViTForObjectDetection`** [wiring]: Top-level wiring.

## paddleocr_vl
- **src**: modular_paddleocr_vl.py
- **status**: composable
- **rationale**: OCR-focused VLM combining Ernie4.5 (Llama-style) LLM + Qwen2.5-Omni attention + Qwen2-VL RoPE + SigLIP vision MLP + VideoLlama3 vision attention. All architectural pieces map to existing kb-nano L2/L3 (encoder/llama-style attention, SwiGLU MLP, vision attention, RMSNorm).
- **classes**:
  - **`PaddleOCRDecoderLayer`** [wiring]: Decoder layer wiring; compute lives on PaddleOCRAttention + PaddleOCRMLP + PaddleOCRRMSNorm.
  - **`PaddleOCRProjector`** [compute]: `L1/linear.py`, `L1/gelu.py`, `L1/layer_norm.py` (Vision -> text projector.)
  - **`PaddleOCRVisionRotaryEmbedding`** [compute]: `L1/vision_rotary_emb.py` (Qwen2-VL 2D vision RoPE.)
  - **`PaddleOCRRotaryEmbedding`** [compute]: `L1/mrope.py` (M-RoPE for Qwen2-VL.)
  - **`PaddleOCRMLP`** [compute]: `L2/llama_mlp.py`, `L1/silu_and_mul.py` (Llama-style SwiGLU.)
  - **`PaddleOCRAttention`** [compute]: `L2/attention.py`, `L1/rms_norm.py`, `L1/mrope.py`, `L1/sdpa.py` (Qwen2.5 Omni attention with M-RoPE; maps to LlamaAttention pattern.)
  - **`PaddleOCRRMSNorm`** [compute]: `L1/rms_norm.py` (Standard RMSNorm.)
  - **`PaddleOCRTextModel`** [wiring]: Top-level LLM wiring.
  - **`PaddleOCRVisionEmbeddings`** [compute]: `L1/conv2d.py`, `L1/embedding.py` (Patch + position embeddings.)
  - **`PaddleOCRVisionAttention`** [compute]: `L2/vision_attention.py`, `L1/vision_rotary_emb.py`, `L1/sdpa.py` (Bidirectional vision attention.)
  - **`PaddleOCRVisionMLP`** [compute]: `L2/siglip_mlp.py`, `L1/linear.py` (SigLIP MLP (fc1 + activation + fc2).)
  - **`PaddleOCRVisionEncoderLayer`** [wiring]: Wiring.
  - **`PaddleOCRVisionEncoder`** [wiring]: Stack.
  - **`PaddleOCRVisionTransformer`** [wiring]: Wiring.
  - **`PaddleOCRVisionModel`** [wiring]: Wiring.
  - **`PaddleOCRVLModel`** [wiring]: Top-level wiring.
  - **`PaddleOCRVLForConditionalGeneration`** [wiring]: Top-level wiring with LM head.

## paligemma
- **src**: modeling_paligemma.py
- **status**: composable
- **rationale**: PaliGemma is a thin VLM wrapper: SigLIP vision tower + linear projector + Gemma LLM via AutoModel. Both component architectures (SigLIP, Gemma) are covered by kb-nano L4 pipelines (siglip2, gemma4).
- **classes**:
  - **`PaliGemmaMultiModalProjector`** [compute]: `L1/linear.py` (Single linear projection from vision dim to text dim.)
  - **`PaliGemmaModel`** [wiring]: Wiring around AutoModel-loaded vision (SigLIP) + text (Gemma) + projector.
  - **`PaliGemmaForConditionalGeneration`** [wiring]: Top-level wiring with LM head.

## parakeet
- **src**: modular_parakeet.py
- **status**: partial
- **partial_reason**: ParakeetEncoderAttention adds learnable bias_u/bias_v (Transformer-XL style) and applies _rel_shift on relative positional logits before SDPA; this fused (matrix_ac + matrix_bd) addition pattern is not exposed as a kb-nano kernel. FastSpeech2ConformerConvolutionModule (depthwise conv + GLU + LN) also has no direct kb-nano L2.
- **rationale**: Parakeet (Conformer) ASR encoder uses Transformer-XL/Shaw-style relative-position attention with custom _rel_shift, plus FastSpeech2 conformer convolution module. The relative-position-bias pattern is not in kb-nano.
- **classes**:
  - **`ParakeetEncoderAttention`** [compute]: no kb-nano kernel — ParakeetEncoderAttention adds learnable bias_u/bias_v (Transformer-XL style) and applies _rel_shift on relative positional logits before SDPA; this fused (matrix_ac + matrix_bd) addition pattern is no
  - **`ParakeetEncoderRelPositionalEncoding`** [compute]: `L1/sinusoidal_embed.py` (Sinusoidal positional encoding (relative).)
  - **`ParakeetEncoderFeedForward`** [compute]: `L2/encoder_mlp.py`, `L1/linear.py` (Two-layer FFN (linear + activation + linear).)
  - **`ParakeetEncoderConvolutionModule`** [compute]: `L1/conv1d.py` (Conformer conv module (pointwise + depthwise + GLU + LN); composable from L1 conv1d but no fused L2 wrapper.)
  - **`ParakeetEncoderSubsamplingConv2D`** [compute]: `L1/conv2d.py`, `L1/relu.py`, `L1/linear.py` (Conv2d stack with depthwise + pointwise + ReLU.)
  - **`ParakeetEncoderBlock`** [wiring]: Wiring: FFN + self_attn + conv + FFN with LayerNorms.
  - **`ParakeetEncoder`** [wiring]: Stack of ParakeetEncoderBlock.
  - **`ParakeetForCTC`** [wiring]: Wiring: encoder + CTC head.

## patchtsmixer
- **src**: modeling_patchtsmixer.py
- **status**: partial
- **partial_reason**: PatchTSMixerBatchNorm uses nn.BatchNorm1d. kb-nano has L1/batch_norm2d.py but no batch_norm1d.py; would need an additional L1 op (trivial wrapper around F.batch_norm).
- **rationale**: PatchTSMixer is MLP-Mixer + gated attention for time series; uses BatchNorm1d (not in kb-nano L1, only batch_norm2d).
- **classes**:
  - **`PatchTSMixerBatchNorm`** [compute]: no kb-nano kernel — PatchTSMixerBatchNorm uses nn.BatchNorm1d. kb-nano has L1/batch_norm2d.py but no batch_norm1d.py; would need an additional L1 op (trivial wrapper around F.batch_norm).
  - **`PatchTSMixerGatedAttention`** [compute]: `L1/linear.py`, `L1/sigmoid.py` (Linear + sigmoid gated attention; standard primitives.)
  - **`PatchTSMixerPositionalEncoding`** [compute]: `L1/sinusoidal_embed.py`, `L1/embedding.py` (Sinusoidal or learned positional encoding.)
  - **`PatchTSMixerNormLayer`** [compute]: `L1/layer_norm.py` (LayerNorm or BatchNorm dispatch.)
  - **`PatchTSMixerMLP`** [compute]: `L1/linear.py`, `L1/gelu.py` (Standard MLP.)
  - **`PatchTSMixerChannelFeatureMixerBlock`** [wiring]: Wiring.
  - **`PatchTSMixerAttention`** [compute]: `L1/linear.py`, `L1/sdpa.py` (Standard MHA.)
  - **`PatchMixerBlock`** [wiring]: Wiring.
  - **`FeatureMixerBlock`** [wiring]: Wiring.
  - **`PatchTSMixerLayer`** [wiring]: Wiring.
  - **`PatchTSMixerBlock`** [wiring]: Stack.
  - **`PatchTSMixerEncoder`** [wiring]: Wiring.
  - **`PatchTSMixerModel`** [wiring]: Wiring.

## patchtst
- **src**: modeling_patchtst.py
- **status**: partial
- **partial_reason**: PatchTSTBatchNorm uses nn.BatchNorm1d (L1 only has batch_norm2d.py). Needs new L1 op or torch fallback.
- **rationale**: PatchTST is a time-series transformer encoder over patches; uses BatchNorm1d which is not in kb-nano L1.
- **classes**:
  - **`PatchTSTBatchNorm`** [compute]: no kb-nano kernel — PatchTSTBatchNorm uses nn.BatchNorm1d (L1 only has batch_norm2d.py). Needs new L1 op or torch fallback.
  - **`PatchTSTAttention`** [compute]: `L1/linear.py`, `L1/sdpa.py` (Standard MHA with optional cross-attention.)
  - **`PatchTSTPatchify`** [wiring]: Reshape/unfold operation.
  - **`PatchTSTMasking`** [wiring]: Masking utility.
  - **`PatchTSTEncoderLayer`** [wiring]: Wiring: attn + MLP + (BatchNorm or LayerNorm).
  - **`PatchTSTEmbedding`** [compute]: `L1/linear.py` (Linear embedding of patches.)
  - **`PatchTSTPositionalEncoding`** [compute]: `L1/sinusoidal_embed.py` (Sinusoidal/learned positional encoding.)
  - **`PatchTSTEncoder`** [wiring]: Stack.
  - **`PatchTSTModel`** [wiring]: Top-level wiring.

## pe_audio
- **src**: modular_pe_audio.py
- **status**: partial
- **partial_reason**: DacEncoder uses Snake1d activation: x + (1/alpha) * sin^2(alpha * x). Not in kb-nano L1; pure torch primitives but no fused kernel.
- **rationale**: Audio embedding model wrapping DAC encoder (with Snake1d activation) + Qwen3-style transformer encoder + AutoModel for text. Snake1d (alpha-parameterized periodic activation) has no kb-nano kernel.
- **classes**:
  - **`PeAudioDacEncoder`** [compute]: DacEncoder uses Snake1d activation: x + (1/alpha) * sin^2(alpha * x). Not in kb-nano L1; pure torch primitives but no fused kernel.
  - **`PeAudioDacEncoderBlock`** [compute]: `L1/conv1d.py`, `L1/conv_transpose1d.py` (DAC residual conv block with Snake1d (partial gap).)
  - **`PeAudioEncoderEmbedder`** [compute]: `L1/conv1d.py`, `L1/linear.py` (DAC encoder + bottleneck conv + projection.)
  - **`PeAudioContrastiveHead`** [compute]: `L1/layer_norm.py`, `L1/linear.py` (LayerNorm + linear projection.)
  - **`PeAudioEncoder`** [wiring]: Wiring composing embedder + transformer encoder.
  - **`PeAudioModel`** [wiring]: Wiring around audio encoder + AutoModel text encoder + contrastive heads.
  - **`PeAudioFrameLevelModel`** [wiring]: Wiring.

## pe_audio_video
- **src**: modular_pe_audio_video.py
- **status**: partial
- **partial_reason**: PeAudioVideoMaskedGroupNorm uses torch.masked.mean/var for padding-aware GroupNorm; kb-nano has L1/group_norm.py but no masked variant. Plus AutoModel coupling for sub-encoders.
- **rationale**: Audio-video contrastive model with masked group norm, ResNet conv blocks, Qwen3-style transformer encoder, and AutoModel-loaded sub-encoders. MaskedGroupNorm uses torch.masked.mean/var which is missing in kb-nano.
- **classes**:
  - **`PeAudioVideoMaskedGroupNorm`** [compute]: no kb-nano kernel — PeAudioVideoMaskedGroupNorm uses torch.masked.mean/var for padding-aware GroupNorm; kb-nano has L1/group_norm.py but no masked variant. Plus AutoModel coupling for sub-encoders.
  - **`PeAudioVideoConvBlock1d`** [compute]: `L1/conv1d.py`, `L1/silu.py` (GroupNorm + SiLU + Conv1d.)
  - **`PeAudioVideoResnetBlock1d`** [wiring]: Wiring: two conv blocks with residual.
  - **`PeAudioVideoEncoderPatchEmbedder`** [wiring]: Wiring: prepend class token + ResNet block.
  - **`PeAudioVideoContrastiveHead`** [compute]: `L1/layer_norm.py`, `L1/linear.py` (LayerNorm + projection.)
  - **`PeAudioVideoEncoderEmbedder`** [compute]: `L1/conv1d.py`, `L1/linear.py`, `L1/layer_norm.py` (Audio + video sub-encoders + projection (uses AutoModel).)
  - **`PeAudioVideoEncoderAttention`** [compute]: `L2/attention.py`, `L1/rms_norm.py`, `L1/rotary_emb.py`, `L1/sdpa.py` (Qwen3-style attention with q_norm/k_norm.)
  - **`PeAudioVideoEncoderLayer`** [wiring]: Wiring.
  - **`PeAudioVideoEncoderRMSNorm`** [compute]: `L1/rms_norm.py` (RMSNorm.)
  - **`PeAudioVideoEncoderRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (RoPE.)
  - **`PeAudioVideoEncoder`** [wiring]: Wiring.
  - **`PeAudioVideoModel`** [wiring]: Top-level wiring with contrastive heads.

## pe_video
- **src**: modular_pe_video.py
- **status**: partial
- **partial_reason**: Inherits PeAudioVideoMaskedGroupNorm using torch.masked.mean/var; missing in kb-nano. Plus AutoModel coupling for video sub-encoder.
- **rationale**: Video-only sibling of pe_audio_video; inherits MaskedGroupNorm and same Qwen3-encoder pieces. Same partial gap.
- **classes**:
  - **`PeVideoEncoder`** [compute]: Inherits PeAudioVideoMaskedGroupNorm using torch.masked.mean/var; missing in kb-nano. Plus AutoModel coupling for video sub-encoder.
  - **`PeVideoContrastiveHead`** [compute]: `L1/layer_norm.py`, `L1/linear.py` (LayerNorm + projection.)
  - **`PeVideoEncoderPatchEmbedder`** [wiring]: Wiring around ResNet block + class token (with masked GroupNorm).
  - **`PeVideoEncoderEmbedder`** [compute]: `L1/linear.py` (Wiring around AutoModel video sub-encoder.)
  - **`PeVideoModel`** [wiring]: Top-level wiring.

## pegasus
- **src**: modeling_pegasus.py
- **status**: composable
- **rationale**: BART-style encoder-decoder with sinusoidal positional embeddings. The PegasusAttention class supports self/cross-attention with bias; maps to kb-nano L2/whisper_attention.py (which covers BART/Whisper enc-dec attention).
- **classes**:
  - **`PegasusSinusoidalPositionalEmbedding`** [compute]: `L1/sinusoidal_embed.py`, `L1/embedding.py` (Fixed sinusoidal positional embedding (non-learned).)
  - **`PegasusAttention`** [compute]: `L2/whisper_attention.py`, `L1/linear.py`, `L1/sdpa.py` (Encoder/decoder/cross-attention with bias; same shape as Whisper attention family.)
  - **`PegasusEncoderLayer`** [wiring]: Wiring: self_attn + LN + fc1 + activation + fc2 + LN.
  - **`PegasusDecoderLayer`** [wiring]: Wiring: self_attn + cross_attn + FFN.
  - **`PegasusEncoder`** [wiring]: Stack.
  - **`PegasusDecoder`** [wiring]: Stack.
  - **`PegasusModel`** [wiring]: Top-level enc-dec wiring.
  - **`PegasusForConditionalGeneration`** [wiring]: Wiring + LM head.
  - **`PegasusForCausalLM`** [wiring]: Wiring with decoder-only mode.

## pegasus_x
- **src**: modeling_pegasus_x.py
- **status**: partial
- **partial_reason**: PegasusXGlobalLocalAttention implements custom block-wise local attention with cross-attention to global tokens via einsum (BHGF/BHXF, BHGX/BHXF). The block-local + global-token pattern has no kb-nano kernel; would need a custom L1/L2 or torch fallback.
- **rationale**: Pegasus-X = Pegasus + custom block-local + global-token attention pattern. Encoder uses PegasusXGlobalLocalAttention which performs blocked sliding-window attention + global tokens via einsum reshapes; not in kb-nano.
- **classes**:
  - **`PegasusXGlobalLocalAttention`** [compute]: PegasusXGlobalLocalAttention implements custom block-wise local attention with cross-attention to global tokens via einsum (BHGF/BHXF, BHGX/BHXF). The block-local + global-token pattern has no kb-nano
  - **`PegasusXScaledWordEmbedding`** [compute]: `L1/embedding.py` (Embedding with learned scale.)
  - **`PegasusXSinusoidalPositionalEmbedding`** [compute]: `L1/sinusoidal_embed.py` (Sinusoidal pos embed.)
  - **`PegasusXAttention`** [compute]: `L2/whisper_attention.py`, `L1/linear.py`, `L1/sdpa.py` (Standard MHA (used in decoder).)
  - **`PegasusXEncoderLayer`** [wiring]: Wiring around global-local attention.
  - **`PegasusXDecoderLayer`** [wiring]: Wiring.
  - **`PegasusXEncoder`** [wiring]: Stack.
  - **`PegasusXDecoder`** [wiring]: Stack.
  - **`PegasusXModel`** [wiring]: Top-level enc-dec wiring.
  - **`PegasusXForConditionalGeneration`** [wiring]: Wiring + LM head.

## perceiver
- **src**: modeling_perceiver.py
- **status**: partial
- **partial_reason**: Fourier-based position encoding (PerceiverFourierPositionEncoding) builds frequency basis with linspace + cos/sin; not in kb-nano. The cross-attention pattern (latent_array attend to inputs) uses generic attention but kb-nano doesn't have a Perceiver-specific cross-attention L2.
- **rationale**: Perceiver IO uses cross-attention from learned latents to inputs, with multiple position-encoding flavors (Fourier features, learned, conv-processing) and bespoke decoders (optical flow, multimodal autoencoding). Fourier feature position encoding requires sin/cos of frequency-mapped positions which kb-nano doesn't have a fused kernel for; otherwise the underlying attention is standard.
- **classes**:
  - **`PerceiverAttention`** [compute]: no kb-nano kernel — Fourier-based position encoding (PerceiverFourierPositionEncoding) builds frequency basis with linspace + cos/sin; not in kb-nano. The cross-attention pattern (latent_array attend to inputs) uses gene
  - **`PerceiverEmbeddings`** [wiring]: Learned latent array initialization.
  - **`PerceiverSelfAttention`** [compute]: `L1/linear.py`, `L1/sdpa.py` (MHA with optional cross-attention; uses standard primitives.)
  - **`PerceiverSelfOutput`** [compute]: `L1/linear.py` (Linear + dropout.)
  - **`PerceiverMLP`** [compute]: `L1/linear.py`, `L1/gelu.py` (Standard MLP.)
  - **`PerceiverLayer`** [wiring]: Wiring.
  - **`PerceiverEncoder`** [wiring]: Stack of cross-attn + self-attn layers.
  - **`PerceiverModel`** [wiring]: Top-level wiring with various input/output preprocessors.
  - **`PerceiverForMaskedLM`** [wiring]: Wiring.
  - **`PerceiverForSequenceClassification`** [wiring]: Wiring.
  - **`PerceiverForImageClassificationLearned`** [wiring]: Wiring.
  - **`PerceiverForImageClassificationFourier`** [wiring]: Wiring; relies on Fourier position encoding (gap).
  - **`PerceiverForImageClassificationConvProcessing`** [wiring]: Wiring.
  - **`PerceiverForOpticalFlow`** [wiring]: Wiring with custom optical flow decoder.
  - **`PerceiverForMultimodalAutoencoding`** [wiring]: Wiring.
  - **`PerceiverProjectionDecoder`** [compute]: `L1/linear.py` (Linear decoder.)
  - **`PerceiverBasicDecoder`** [wiring]: Wiring with cross-attention to latents.
  - **`PerceiverClassificationDecoder`** [wiring]: Wiring.
  - **`PerceiverOpticalFlowDecoder`** [wiring]: Wiring.
  - **`PerceiverBasicVideoAutoencodingDecoder`** [wiring]: Wiring.
  - **`PerceiverMultimodalDecoder`** [wiring]: Wiring.
  - **`Conv2dSamePadding`** [compute]: `L1/conv2d.py` (Conv2d with custom padding.)
  - **`Conv2DDownsample`** [compute]: `L1/conv2d.py` (Conv downsampling.)

## perception_lm
- **src**: modular_perception_lm.py
- **status**: partial
- **partial_reason**: Vision tower is loaded via AutoModel.from_config with model_args['embed_dim'] suggesting a custom timm-like perception encoder; no fixed kb-nano vision pipeline corresponds. Adaptive avg pool 2d exists in kb-nano (L1/adaptive_avg_pool2d.py).
- **rationale**: PerceptionLM is a Llava-style VLM with adaptive avg pool 2D + 2-layer GELU projector + AutoModel-loaded vision and text. Vision tower is loaded from a non-strict timm-like AutoModel config (model_args['embed_dim']) — not bound to a specific kb-nano vision pipeline.
- **classes**:
  - **`PerceptionLMModel`** [compute]: no kb-nano kernel — Vision tower is loaded via AutoModel.from_config with model_args['embed_dim'] suggesting a custom timm-like perception encoder; no fixed kb-nano vision pipeline corresponds. Adaptive avg pool 2d exist
  - **`PerceptionLMAdaptiveAvgPooling`** [compute]: `L1/adaptive_avg_pool2d.py` (F.adaptive_avg_pool2d wrapper.)
  - **`PerceptionLMMultiModalProjector`** [compute]: `L1/linear.py`, `L1/gelu.py` (2-layer linear + GELU + optional pooling.)
  - **`PerceptionLMForConditionalGeneration`** [wiring]: Top-level wiring with LM head.
