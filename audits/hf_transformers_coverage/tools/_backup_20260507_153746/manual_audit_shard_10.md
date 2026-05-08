# Manual audit shard 10 — olmo2 .. phi4_multimodal

## olmo2
- **src**: modeling_olmo2.py (and modular_olmo2.py)
- **hidden_act**: silu
- **status**: composable
- **classes**:
  - **`Olmo2RMSNorm`** [compute, inherits `LlamaRMSNorm`]: `L1/rms_norm.py`
  - **`Olmo2RotaryEmbedding`** [compute, inherits `OlmoRotaryEmbedding`]: `L1/rotary_emb.py`
  - **`Olmo2Attention`** [compute, inherits `OlmoAttention`]: `L2/attention.py` (q/k/v + per-tensor q_norm/k_norm RMSNorm + RoPE + KV cache + dispatch via ALL_ATTENTION_FUNCTIONS; note q_norm/k_norm applied to full Q/K not per-head)
  - **`Olmo2MLP`** [compute]: `L2/llama_mlp.py` (SwiGLU: gate_proj + up_proj + down_proj, silu)
  - **`Olmo2DecoderLayer`** [wiring, inherits `OlmoDecoderLayer`]: wires `Olmo2Attention`, `Olmo2MLP`, `Olmo2RMSNorm` (post_attention_layernorm and post_feedforward_layernorm — post-norm pattern, no input_layernorm)
  - **`Olmo2Model`** [wiring]: wires `Olmo2DecoderLayer`, `Olmo2RMSNorm`, `Olmo2RotaryEmbedding`; direct `L1/embedding.py` (embed_tokens)
  - **`Olmo2ForCausalLM`** [wiring]: wires `Olmo2Model`; direct `L1/linear.py` (lm_head)
- **task heads (1)**: ForSequenceClassification — base + linear (per-task)

## olmo3
- **src**: modeling_olmo3.py (and modular_olmo3.py)
- **hidden_act**: silu
- **status**: composable
- **classes**:
  - **`Olmo3RMSNorm`** [compute, inherits `Olmo2RMSNorm`]: `L1/rms_norm.py`
  - **`Olmo3Attention`** [compute, inherits `Olmo2Attention`]: `L2/attention.py` (q_norm/k_norm + RoPE + sliding-window via layer_types)
  - **`Olmo3MLP`** [compute]: `L2/llama_mlp.py` (SwiGLU)
  - **`Olmo3DecoderLayer`** [wiring, inherits `Olmo2DecoderLayer`]: wires `Olmo3Attention`, `Olmo3MLP`, `Olmo3RMSNorm` (post-norm pattern)
  - **`Olmo3RotaryEmbedding`** [compute, inherits `Gemma2RotaryEmbedding`]: `L1/rotary_emb.py` (default RoPE, dtype cast at end)
  - **`Olmo3Model`** [wiring]: wires `Olmo3DecoderLayer`, `Olmo3RMSNorm`, `Olmo3RotaryEmbedding`; direct `L1/embedding.py`
  - **`Olmo3ForCausalLM`** [wiring]: wires `Olmo3Model`; direct `L1/linear.py`
- **task heads (1)**: ForSequenceClassification — base + linear (per-task)

## olmo_hybrid
- **src**: modeling_olmo_hybrid.py (and modular_olmo_hybrid.py)
- **hidden_act**: silu
- **status**: partial (linear-attention GatedDeltaNet branch shares an L1 op with Qwen3-Next; no L4 OlmoHybrid pipeline)
- **classes**:
  - **`OlmoHybridRMSNormGated`** [compute, inherits `Qwen3NextRMSNormGated`]: `L1/rms_norm_gated.py` (RMSNorm with silu(gate) gating)
  - **`OlmoHybridRMSNorm`** [compute, inherits `Olmo3RMSNorm`]: `L1/rms_norm.py`
  - **`OlmoHybridShortConvolution`** [compute, inherits `nn.Conv1d`]: `L1/causal_conv1d.py + L1/silu.py` (depthwise causal conv1d + silu, with state for decode)
  - **`OlmoHybridAttention`** [compute, inherits `Olmo3Attention`]: `L2/attention.py` (q_norm/k_norm; supports NoPE via optional position_embeddings)
  - **`OlmoHybridRotaryEmbedding`** [compute, inherits `Olmo3RotaryEmbedding`]: `L1/rotary_emb.py`
  - **`OlmoHybridGatedDeltaNet`** [compute]: `L1/linear.py + L1/causal_conv1d.py + L1/silu.py + L1/chunk_gated_delta_rule.py + L1/gdn_recurrence.py + L1/rms_norm_gated.py` (separate q/k/v/a/b/g projections, per-projection conv1d, chunk_gated_delta_rule for prefill / fused_recurrent for decode, RMSNorm-gated output; closest L2 is `L2/qwen3_next_gdn_attention.py` but interface differs — Qwen3-Next fuses qkvz)
  - **`OlmoHybridMLP`** [compute, inherits `Olmo3MLP`]: `L2/llama_mlp.py`
  - **`OlmoHybridAttentionDecoderLayer`** [wiring, inherits `Olmo3DecoderLayer`]: wires `OlmoHybridAttention`, `OlmoHybridMLP`, `OlmoHybridRMSNorm` (post-norm)
  - **`OlmoHybridLinearAttentionDecoderLayer`** [wiring, inherits `LlamaDecoderLayer`]: wires `OlmoHybridGatedDeltaNet`, `OlmoHybridMLP`, `OlmoHybridRMSNorm` (input_layernorm + post_attention_layernorm — pre-norm)
  - **`OlmoHybridModel`** [wiring, inherits `Qwen3NextModel`]: wires `OlmoHybridAttentionDecoderLayer`/`OlmoHybridLinearAttentionDecoderLayer` per layer_types, `OlmoHybridRMSNorm`, optional `OlmoHybridRotaryEmbedding`; direct `L1/embedding.py`
  - **`OlmoHybridForCausalLM`** [wiring, inherits `Olmo3ForCausalLM`]: wires `OlmoHybridModel`; direct `L1/linear.py`

## olmoe
- **src**: modeling_olmoe.py (and modular_olmoe.py)
- **hidden_act**: silu
- **status**: composable
- **classes**:
  - **`OlmoeRMSNorm`** [compute, inherits `LlamaRMSNorm`]: `L1/rms_norm.py`
  - **`OlmoeRotaryEmbedding`** [compute, inherits `LlamaRotaryEmbedding`]: `L1/rotary_emb.py`
  - **`OlmoeMLP`** [compute, inherits `GemmaMLP`]: `L2/llama_mlp.py` (SwiGLU)
  - **`OlmoeAttention`** [compute, inherits `LlamaAttention`]: `L2/attention.py` (q_norm/k_norm RMSNorm + optional clip_qkv clamp + RoPE + dispatch; sliding_window optional)
  - **`OlmoeExperts`** [compute, inherits `MixtralExperts`]: `L2/mixtral_moe.py` (3D parameter experts with fused gate_up + chunked gate*act + down; matches Mixtral fused-MoE pattern, dispatched to `L1/moe_grouped_gemm.py`)
  - **`OlmoeTopKRouter`** [compute, inherits `Qwen2MoeTopKRouter`]: `L1/linear.py + L1/topk_softmax.py` (linear gate + softmax + topk + optional norm)
  - **`OlmoeSparseMoeBlock`** [wiring]: wires `OlmoeTopKRouter`, `OlmoeExperts`
  - **`OlmoeDecoderLayer`** [wiring, inherits `LlamaDecoderLayer`]: wires `OlmoeAttention`, `OlmoeSparseMoeBlock`, `OlmoeRMSNorm` (input_layernorm, post_attention_layernorm)
  - **`OlmoeModel`** [wiring, inherits `MixtralModel`]: wires `OlmoeDecoderLayer`, `OlmoeRMSNorm`, `OlmoeRotaryEmbedding`; direct `L1/embedding.py`
  - **`OlmoeForCausalLM`** [wiring, inherits `MixtralForCausalLM`]: wires `OlmoeModel`; direct `L1/linear.py`


## omdet_turbo
- **src**: modeling_omdet_turbo.py
- **hidden_act**: csp_activation=silu, conv_norm_activation=gelu, encoder_feedforward_activation=relu, decoder_activation=relu
- **status**: unsupported (open-vocabulary object detection — no kb-nano OmDet-Turbo L4; closest analogue is `L4/rtdetrv2.py` for the encoder/decoder backbone but doesn't cover language backbone or task encoder)
- **classes**:
  - **`MultiScaleDeformableAttention`** [compute]: `L1/rtdetrv2_deformable_attention.py` (grid_sample-based multi-scale deformable; same op shared with RT-DETR-V2)
  - **`OmDetTurboLanguageBackbone`** [wiring]: wires AutoModel (text encoder, e.g. CLIPText); direct text_projection parameter matmul
  - **`OmDetTurboVisionBackbone`** [wiring]: wires AutoBackbone, `nn.LayerNorm` (×N); direct `L1/layer_norm.py`
  - **`OmDetTurboMultiscaleDeformableAttention`** [compute]: `L2/rtdetrv2_deformable_attention.py` (sampling_offsets/attention_weights linear + softmax + `L1/rtdetrv2_deformable_attention.py` op + output_proj)
  - **`OmDetTurboConvNormLayer`** [compute]: `L1/conv2d.py + L1/batch_norm2d.py + L1/relu.py` (conv + bn + activation; activation varies per call, gelu for encoder feature maps)
  - **`OmDetTurboRepVggBlock`** [compute]: `L2/rtdetrv2_repvgg_block.py` (two ConvNormLayers + sum + activation)
  - **`OmDetTurboCSPRepLayer`** [compute]: `L2/rtdetrv2_csp_rep_layer.py` (1x1 + bottlenecks + concat + 1x1)
  - **`OmDetTurboMultiheadAttention`** [compute]: `L1/linear.py + L1/dense_attention.py` (q/k/v Linear + manual attention scoring + softmax + matmul + out_proj; equivalent to nn.MultiheadAttention with batch_first)
  - **`OmDetTurboEncoderLayer`** [wiring]: wires `OmDetTurboMultiheadAttention`, `nn.LayerNorm` (×2); direct `L1/linear.py` (fc1, fc2), `L1/relu.py`, `L1/layer_norm.py`
  - **`OmDetTurboEncoder`** [wiring]: wires `OmDetTurboEncoderLayer`
  - **`OmDetTurboHybridEncoder`** [wiring]: wires `OmDetTurboEncoder` (×N), `OmDetTurboConvNormLayer` (lateral/downsample), `OmDetTurboCSPRepLayer` (FPN/PAN); direct `L1/conv2d.py`, `L1/batch_norm2d.py`, `L1/interpolate.py`
  - **`OmDetTurboMLPWithDropout`** [compute]: `L1/linear.py + L1/relu.py + L1/linear.py` (2-layer MLP with relu; dropout)
  - **`OmDetTurboMLP`** [compute]: `L1/linear.py + L1/relu.py` (multi-layer FFN)
  - **`OmDetTurboResidualLayer`** [compute]: `L1/layer_norm.py` (residual add + layer norm)
  - **`OmDetTurboTaskEncoder`** [wiring]: wires `OmDetTurboMLPWithDropout`, `OmDetTurboResidualLayer`
  - **`OmDetTurboDeformableTransformerDecoderLayer`** [wiring]: wires `OmDetTurboMultiheadAttention` (self-attn), `OmDetTurboMultiscaleDeformableAttention` (cross-attn), `nn.LayerNorm` (×3); direct `L1/linear.py` (linear1, linear2), `L1/relu.py`
  - **`OmDetTurboDecoder`** [wiring]: wires `OmDetTurboDeformableTransformerDecoderLayer` (×N), `OmDetTurboMLP` (bbox heads, encoder_class_head, query_position_head); direct `L1/embedding.py`, `L1/linear.py`
  - **`OmDetTurboForObjectDetection`** [wiring]: wires `OmDetTurboLanguageBackbone`, `OmDetTurboVisionBackbone`, `OmDetTurboHybridEncoder`, `OmDetTurboTaskEncoder`, `OmDetTurboDecoder`

## oneformer
- **src**: modeling_oneformer.py
- **hidden_act**: activation_function="relu" for transformer decoder; quick_gelu for OneFormerTextMLP; gelu in OneFormerTextTransformerDecoderLayer text MLP
- **status**: unsupported (universal segmentation — no kb-nano OneFormer L4)
- **classes**:
  - **`OneFormerHungarianMatcher`** [compute]: not a forward op (matching loss); skipped from kernel mapping
  - **`OneFormerLoss`** [compute]: training loss; skipped from kernel mapping
  - **`OneFormerPixelDecoderEncoderMultiscaleDeformableAttention`** [compute]: `L2/rtdetrv2_deformable_attention.py` (sampling_offsets + grid_sample-based deformable attention; structurally same as RT-DETR-V2 deformable)
  - **`OneFormerPixelDecoderEncoderLayer`** [wiring]: wires `OneFormerPixelDecoderEncoderMultiscaleDeformableAttention`, `nn.LayerNorm` (×2); direct `L1/linear.py` (fc1, fc2), activation
  - **`OneFormerPixelDecoderEncoderOnly`** [wiring]: wires `OneFormerPixelDecoderEncoderLayer` (×N)
  - **`OneFormerPixelDecoder`** [wiring]: wires `OneFormerPixelDecoderEncoderOnly`; direct `L1/conv2d.py`, `L1/group_norm.py`, position embedder
  - **`OneFormerPixelLevelModule`** [wiring]: wires AutoBackbone, `OneFormerPixelDecoder`
  - **`OneFormerAttention`** [compute]: `L1/linear.py + L1/dense_attention.py` (DETR-style q/k/v with optional position embed, manual bmm + softmax; supports cross-attention)
  - **`OneFormerTransformerDecoderSelfAttentionLayer`** [wiring]: wires `OneFormerAttention`, `nn.LayerNorm`
  - **`OneFormerTransformerDecoderCrossAttentionLayer`** [wiring]: wires `nn.MultiheadAttention`, `nn.LayerNorm`; direct `L1/dense_attention.py`
  - **`OneFormerTransformerDecoderFFNLayer`** [compute]: `L1/linear.py + L1/relu.py + L1/linear.py + L1/layer_norm.py` (2-layer FFN with residual + LN)
  - **`OneFormerMLPPredictionHead`** [wiring]: wires `PredictionBlock` (×num_layers)
  - **`OneFormerTransformerDecoderLayer`** [wiring]: wires `OneFormerTransformerDecoderCrossAttentionLayer`, `OneFormerTransformerDecoderSelfAttentionLayer`, `OneFormerTransformerDecoderFFNLayer`
  - **`OneFormerTransformerDecoderQueryTransformerDecoder`** [wiring]: wires `OneFormerTransformerDecoderQueryTransformerDecoderLayer` (×N)
  - **`OneFormerTransformerDecoderQueryTransformerDecoderLayer`** [wiring]: wires `nn.MultiheadAttention` (×2), `nn.LayerNorm` (×3); direct `L1/linear.py`, `L1/relu.py`, `L1/dense_attention.py`
  - **`OneFormerTransformerDecoderQueryTransformer`** [wiring]: wires `OneFormerTransformerDecoderQueryTransformerDecoder`, `nn.LayerNorm`
  - **`OneFormerTransformerDecoder`** [wiring]: wires `OneFormerTransformerDecoderQueryTransformer`, `OneFormerTransformerDecoderLayer` (×N), `nn.LayerNorm`, `OneFormerMLPPredictionHead`; direct `L1/conv2d.py`, `L1/linear.py`, `L1/interpolate.py`
  - **`OneFormerTransformerModule`** [wiring]: wires `OneFormerSinePositionEmbedding`, `nn.Embedding` (×2), `OneFormerTransformerDecoder`; direct `L1/conv2d.py`
  - **`OneFormerSinePositionEmbedding`** [compute]: cumsum/sin/cos position encoding; no exact L1 match (`L1/sinusoidal_embed.py` is closest but for 1D positional)
  - **`PredictionBlock`** [compute]: `L1/linear.py` + activation (relu or identity)
  - **`OneFormerTextMapperAttention`** [compute]: `L1/linear.py + L1/dense_attention.py` (q/k/v Linear + einsum-based attention + softmax + proj)
  - **`OneFormerTextTransformerDecoderLayer`** [wiring]: wires `OneFormerTextMapperAttention` (×2), `nn.LayerNorm` (×3); direct `L1/linear.py + L1/gelu.py + L1/linear.py` (mlp Sequential)
  - **`OneFormerTextContextDecoder`** [wiring]: wires `OneFormerTextTransformerDecoderLayer` (×N), `nn.LayerNorm`, `nn.Linear`; direct `L1/layer_norm.py`, `L1/linear.py`
  - **`OneFormerTextMLP`** [compute]: `L1/linear.py + L1/quickgelu.py + L1/linear.py` (matches `L2/clip_mlp.py` pattern)
  - **`OneFormerTextTransformerLayer`** [wiring]: wires `nn.MultiheadAttention`, `nn.LayerNorm` (×2), `OneFormerTextMLP`; direct `L1/dense_attention.py`
  - **`OneFormerTextTransformer`** [wiring]: wires `OneFormerTextTransformerLayer` (×N)
  - **`OneFormerTextEncoder`** [wiring]: wires `OneFormerTextTransformer`, `nn.LayerNorm`, `nn.Embedding`; direct positional_embedding parameter add
  - **`OneFormerTextMapper`** [wiring]: wires `OneFormerTextEncoder`, `OneFormerMLPPredictionHead`, optional `nn.Embedding`
  - **`OneFormerTaskModel`** [wiring]: wires `OneFormerTextEncoder` (and small projection head)
  - **`OneFormerModel`** [wiring]: wires `OneFormerPixelLevelModule`, `OneFormerTransformerModule`, `OneFormerTaskModel`, optional `OneFormerTextMapper`
- **task heads (1)**: ForUniversalSegmentation — base + linear (per-task)

## openai
- **src**: modeling_openai.py
- **hidden_act**: afn=gelu (uses gelu_new from utils via ACT_FNS)
- **status**: unsupported (legacy GPT-1 with Conv1D-based projections; no kb-nano OpenAIGPT pipeline)
- **classes**:
  - **`Attention`** [compute]: `L1/linear.py + L1/dense_attention.py` (Conv1D-based c_attn QKV split + manual softmax + causal triangular mask; Conv1D = `nn.Linear`-equivalent; no KV cache, no RoPE)
  - **`MLP`** [compute]: `L1/linear.py + L1/gelu.py + L1/linear.py` (Conv1D fc + gelu_new + Conv1D proj; matches `L2/encoder_mlp.py`-like 2-layer pattern; gelu_new resolves to `L1/gelu.py`)
  - **`Block`** [wiring]: wires `Attention`, `MLP`, `nn.LayerNorm` (×2)
  - **`OpenAIGPTSequenceSummary`** [compute]: `L1/linear.py + L1/tanh.py` (proj + activation + dropout, optional)
  - **`OpenAIGPTModel`** [wiring]: wires `Block` (×N); direct `L1/embedding.py` (tokens_embed, positions_embed)
  - **`OpenAIGPTLMHeadModel`** [wiring]: wires `OpenAIGPTModel`; direct `L1/linear.py` (lm_head, weight-tied)
- **task heads (2)**: ForSequenceClassification, DoubleHeadsModel — base + linear (per-task)

## openai_privacy_filter
- **src**: modeling_openai_privacy_filter.py (and modular_openai_privacy_filter.py)
- **hidden_act**: silu (inherits from GptOssConfig); GPT-OSS uses swiglu_oai gating with sigmoid
- **status**: composable (mirrors GPT-OSS structure; inherits compute from gpt_oss classes)
- **classes**:
  - **`OpenAIPrivacyFilterRMSNorm`** [compute, inherits `GptOssRMSNorm`]: `L1/rms_norm.py`
  - **`OpenAIPrivacyFilterRotaryEmbedding`** [compute, inherits `GptOssRotaryEmbedding`]: `L1/yarn_rotary_emb.py` (YaRN-NeoX, same as GPT-OSS)
  - **`OpenAIPrivacyFilterAttention`** [compute, inherits `GptOssAttention`]: `L2/gpt_oss_attention.py` (q/k/v + RoPE + attention sinks `s_aux` + sliding_window + per-tensor pre-scaling — `is_causal=False`)
  - **`OpenAIPrivacyFilterExperts`** [compute, inherits `GptOssExperts`]: `L2/gpt_oss_moe.py` (3D parameter experts with bias, swiglu_oai gating with clamp, fp32 accumulation)
  - **`OpenAIPrivacyFilterTopKRouter`** [compute, inherits `GptOssTopKRouter`]: `L1/linear.py + L1/topk_softmax.py` (linear (fp32) + topk + softmax of selected scores + scale by 1/top_k)
  - **`OpenAIPrivacyFilterMLP`** [wiring]: wires `OpenAIPrivacyFilterTopKRouter`, `OpenAIPrivacyFilterExperts` (×num_experts scale)
  - **`OpenAIPrivacyFilterEncoderLayer`** [wiring, inherits `GptOssDecoderLayer`]: wires `OpenAIPrivacyFilterAttention`, `OpenAIPrivacyFilterMLP`, `OpenAIPrivacyFilterRMSNorm` (×2 — pre-norm pattern)
  - **`OpenAIPrivacyFilterModel`** [wiring, inherits `GptOssModel`]: wires `OpenAIPrivacyFilterEncoderLayer` (×N), `OpenAIPrivacyFilterRMSNorm`, `OpenAIPrivacyFilterRotaryEmbedding`; direct `L1/embedding.py`
- **task heads (1)**: ForTokenClassification — base + linear (per-task)

## opt
- **src**: modeling_opt.py
- **hidden_act**: relu (default; activation_function="relu")
- **status**: unsupported (no kb-nano OPT L4 — OPT uses learned positional embeddings + LayerNorm, distinct from Llama-family)
- **classes**:
  - **`OPTLearnedPositionalEmbedding`** [compute, inherits `nn.Embedding`]: `L1/embedding.py` (learned positional embedding with offset=2 hack)
  - **`OPTAttention`** [compute]: `L1/linear.py + L1/dense_attention.py + L1/store_kvcache.py` (k/v/q/out_proj — no merged QKV, no RoPE, with KV cache update; closest L2 is `L2/encoder_attention.py`-style but with causal mask and KV cache)
  - **`OPTDecoderLayer`** [wiring]: wires `OPTAttention`, `nn.LayerNorm` (×2 — self_attn_layer_norm and final_layer_norm); direct `L1/linear.py` (fc1, fc2), `L1/relu.py` (activation_fn), `L1/dropout.py` (do_layer_norm_before toggles pre/post-norm)
  - **`OPTDecoder`** [wiring]: wires `OPTLearnedPositionalEmbedding`, `OPTDecoderLayer` (×N), optional `nn.LayerNorm` (final), optional `nn.Linear` (project_in/out); direct `L1/embedding.py`
  - **`OPTModel`** [wiring]: wires `OPTDecoder`
  - **`OPTForCausalLM`** [wiring]: wires `OPTModel`; direct `L1/linear.py` (lm_head)
- **task heads (2)**: ForSequenceClassification, ForQuestionAnswering — base + linear (per-task)

## ovis2
- **src**: modeling_ovis2.py (and modular_ovis2.py)
- **hidden_act**: silu
- **status**: partial (vision tower is composable from existing kb-nano kernels; uses AutoModel for the LM tower → composable when LM is e.g. Qwen2/Llama)
- **classes**:
  - **`Ovis2RMSNorm`** [compute, inherits `LlamaRMSNorm`]: `L1/rms_norm.py`
  - **`Ovis2VisionMLP`** [compute, inherits `LlamaMLP`]: `L2/llama_mlp.py` (SwiGLU with optional bias)
  - **`Ovis2VisionEmbeddings`** [compute, inherits `SiglipVisionEmbeddings`]: `L1/conv2d.py + L1/rms_norm.py + L1/embedding.py` (patch_embedding Conv2d + RMSNorm + position embedding)
  - **`Ovis2VisionAttention`** [compute, inherits `Aimv2Attention`]: `L2/siglip_attention.py` (q/k/v/out_proj + ALL_ATTENTION_FUNCTIONS dispatch; non-causal; matches SigLIP/CLIP attention shape)
  - **`Ovis2MLP`** [compute]: `L2/llama_mlp.py` (SwiGLU; same as Ovis2VisionMLP — duplicate class)
  - **`Ovis2VisionEncoderLayer`** [wiring, inherits `Aimv2EncoderLayer`]: wires `Ovis2VisionAttention`, `Ovis2MLP`, `Ovis2RMSNorm` (×2)
  - **`Ovis2VisionEncoder`** [wiring, inherits `SiglipEncoder`]: wires `Ovis2VisionEncoderLayer` (×N)
  - **`Ovis2VisionTransformer`** [wiring]: wires `Ovis2VisionEmbeddings`, `Ovis2VisionEncoder`, `Ovis2RMSNorm` (final)
  - **`Ovis2VisualEmbeddingTable`** [compute, inherits `nn.Embedding`]: `L1/embedding.py` (with optional matmul fallback for non-int dtypes)
  - **`Ovis2VisionModel`** [wiring]: wires `Ovis2VisionTransformer`, `nn.LayerNorm` (head_norm); direct `L1/linear.py` (head_linear)
  - **`Ovis2Model`** [wiring, inherits `LlavaModel`]: wires `Ovis2VisionModel`, AutoModel (language_model), `Ovis2VisualEmbeddingTable`
  - **`Ovis2ForConditionalGeneration`** [wiring, inherits `LlavaForConditionalGeneration`]: wires `Ovis2Model`; direct `L1/linear.py` (lm_head)

## owlv2
- **src**: modeling_owlv2.py (modular_owlv2.py only redefines image processor classes — not modeling)
- **hidden_act**: quick_gelu
- **status**: unsupported (open-vocabulary detection via OWL-ViT; no kb-nano L4)
- **classes**:
  - **`Owlv2VisionEmbeddings`** [compute]: `L1/conv2d.py + L1/embedding.py` (Conv2d patch_embedding + class token + position_embedding; with optional bicubic interpolation)
  - **`Owlv2TextEmbeddings`** [compute]: `L1/embedding.py` (token + position embedding sum)
  - **`Owlv2Attention`** [compute]: `L2/clip_attention.py` (q/k/v/out_proj + ALL_ATTENTION_FUNCTIONS dispatch; non-causal vision, optional causal text)
  - **`Owlv2MLP`** [compute]: `L2/clip_mlp.py` (fc1 → quick_gelu → fc2)
  - **`Owlv2EncoderLayer`** [wiring]: wires `Owlv2Attention`, `Owlv2MLP`, `nn.LayerNorm` (×2)
  - **`Owlv2Encoder`** [wiring]: wires `Owlv2EncoderLayer` (×N)
  - **`Owlv2TextTransformer`** [wiring]: wires `Owlv2TextEmbeddings`, `Owlv2Encoder`, `nn.LayerNorm` (final_layer_norm)
  - **`Owlv2TextModel`** [wiring]: wires `Owlv2TextTransformer`
  - **`Owlv2VisionTransformer`** [wiring]: wires `Owlv2VisionEmbeddings`, `nn.LayerNorm` (pre_layrnorm), `Owlv2Encoder`, `nn.LayerNorm` (post_layernorm)
  - **`Owlv2VisionModel`** [wiring]: wires `Owlv2VisionTransformer`
  - **`Owlv2Model`** [wiring]: wires `Owlv2TextModel`, `Owlv2VisionModel`; direct `L1/linear.py` (visual_projection, text_projection), logit_scale parameter
  - **`Owlv2BoxPredictionHead`** [compute]: `L1/linear.py + L1/gelu.py + L1/linear.py + L1/gelu.py + L1/linear.py` (3-layer MLP with GELU)
  - **`Owlv2ClassPredictionHead`** [compute]: `L1/linear.py + L1/gelu.py + L1/linear.py + L1/sigmoid.py` (similarity-based class prediction with logit shift+scale via ELU/sigmoid)
  - **`Owlv2ForObjectDetection`** [wiring]: wires `Owlv2Model`, `Owlv2BoxPredictionHead`, `Owlv2ClassPredictionHead`, `nn.LayerNorm`

## owlvit
- **src**: modeling_owlvit.py
- **hidden_act**: quick_gelu
- **status**: unsupported (open-vocabulary detection — no kb-nano L4)
- **classes**:
  - **`OwlViTVisionEmbeddings`** [compute]: `L1/conv2d.py + L1/embedding.py` (Conv2d patch + class token + position embedding)
  - **`OwlViTTextEmbeddings`** [compute]: `L1/embedding.py` (token + position embedding sum)
  - **`OwlViTAttention`** [compute]: `L2/clip_attention.py` (q/k/v/out_proj + ALL_ATTENTION_FUNCTIONS dispatch)
  - **`OwlViTMLP`** [compute]: `L2/clip_mlp.py` (fc1 → quick_gelu → fc2)
  - **`OwlViTEncoderLayer`** [wiring]: wires `OwlViTAttention`, `OwlViTMLP`, `nn.LayerNorm` (×2)
  - **`OwlViTEncoder`** [wiring]: wires `OwlViTEncoderLayer` (×N)
  - **`OwlViTTextTransformer`** [wiring]: wires `OwlViTTextEmbeddings`, `OwlViTEncoder`, `nn.LayerNorm`
  - **`OwlViTTextModel`** [wiring]: wires `OwlViTTextTransformer`
  - **`OwlViTVisionTransformer`** [wiring]: wires `OwlViTVisionEmbeddings`, `nn.LayerNorm` (pre_layernorm), `OwlViTEncoder`, `nn.LayerNorm` (post_layernorm)
  - **`OwlViTVisionModel`** [wiring]: wires `OwlViTVisionTransformer`
  - **`OwlViTModel`** [wiring]: wires `OwlViTTextModel`, `OwlViTVisionModel`; direct `L1/linear.py` (text_projection, visual_projection), logit_scale parameter
  - **`OwlViTBoxPredictionHead`** [compute]: `L1/linear.py + L1/gelu.py + L1/linear.py + L1/gelu.py + L1/linear.py`
  - **`OwlViTClassPredictionHead`** [compute]: `L1/linear.py + L1/gelu.py + L1/linear.py + L1/sigmoid.py`
  - **`OwlViTForObjectDetection`** [wiring]: wires `OwlViTModel`, `OwlViTBoxPredictionHead`, `OwlViTClassPredictionHead`, `nn.LayerNorm`

## paddleocr_vl
- **src**: modeling_paddleocr_vl.py (and modular_paddleocr_vl.py)
- **hidden_act**: vision=gelu_pytorch_tanh, text=silu
- **status**: composable (text branch is Ernie-4.5/Qwen2-VL-style; vision is SigLIP-style with M-RoPE)
- **classes**:
  - **`PaddleOCRProjector`** [compute]: `L1/layer_norm.py + L1/linear.py + L1/gelu.py + L1/linear.py` (LayerNorm + 2-layer MLP with GELU; spatial-merge reshape)
  - **`PaddleOCRVisionRotaryEmbedding`** [compute, inherits `VisionRotaryEmbedding`]: `L1/vision_rotary_emb.py` (2D vision RoPE)
  - **`PaddleOCRRotaryEmbedding`** [compute, inherits `Qwen2VLRotaryEmbedding`]: `L1/mrope.py` (M-RoPE for Qwen2-VL with 3D position grid)
  - **`PaddleOCRMLP`** [compute, inherits `Ernie4_5MLP`]: `L2/llama_mlp.py` (SwiGLU with optional bias)
  - **`PaddleOCRAttention`** [compute, inherits `Qwen2_5OmniAttention`]: `L2/attention.py` (q/k/v/o + M-RoPE + KV cache + sliding_window)
  - **`PaddleOCRRMSNorm`** [compute, inherits `Ernie4_5RMSNorm`]: `L1/rms_norm.py`
  - **`PaddleOCRDecoderLayer`** [wiring, inherits `Ernie4_5DecoderLayer`]: wires `PaddleOCRAttention`, `PaddleOCRMLP`, `PaddleOCRRMSNorm` (×2)
  - **`PaddleOCRTextModel`** [wiring]: wires `PaddleOCRDecoderLayer`, `PaddleOCRRMSNorm`, `PaddleOCRRotaryEmbedding`; direct `L1/embedding.py`
  - **`PaddleOCRVisionEmbeddings`** [compute, inherits `SiglipVisionEmbeddings`]: `L1/conv2d.py + L1/embedding.py + L1/interpolate.py` (Conv2d patch_embedding + bilinear interpolated position embedding for variable-resolution images)
  - **`PaddleOCRVisionAttention`** [compute, inherits `VideoLlama3VisionAttention`]: `L2/siglip_attention.py` (q/k/v/out_proj with 2D vision RoPE + cu_seqlens varlen attention)
  - **`PaddleOCRVisionMLP`** [compute, inherits `SiglipMLP`]: `L2/siglip_mlp.py` (fc1 → gelu_pytorch_tanh → fc2)
  - **`PaddleOCRVisionEncoderLayer`** [wiring, inherits `VideoLlama3VisionEncoderLayer`]: wires `PaddleOCRVisionAttention`, `PaddleOCRVisionMLP`, `nn.LayerNorm` (×2)
  - **`PaddleOCRVisionEncoder`** [wiring, inherits `VideoLlama3VisionEncoder`]: wires `PaddleOCRVisionEncoderLayer` (×N), `PaddleOCRVisionRotaryEmbedding`
  - **`PaddleOCRVisionTransformer`** [wiring]: wires `PaddleOCRVisionEmbeddings`, `PaddleOCRVisionEncoder`, `nn.LayerNorm` (post_layernorm)
  - **`PaddleOCRVisionModel`** [wiring]: wires `PaddleOCRVisionTransformer`
  - **`PaddleOCRVLModel`** [wiring, inherits `Qwen2VLModel`]: wires `PaddleOCRVisionModel`, `PaddleOCRTextModel`, `PaddleOCRProjector`
  - **`PaddleOCRVLForConditionalGeneration`** [wiring, inherits `Qwen2VLForConditionalGeneration`]: wires `PaddleOCRVLModel`; direct `L1/linear.py` (lm_head)

## paligemma
- **src**: modeling_paligemma.py
- **hidden_act**: defers to vision_config (SigLIP gelu_pytorch_tanh) and text_config (Gemma gelu_pytorch_tanh / silu via Gemma2/3 variants)
- **status**: composable (uses AutoModel for vision tower (SigLIP) and language tower (Gemma/Gemma2/Gemma3); only adds a linear projector)
- **classes**:
  - **`PaliGemmaMultiModalProjector`** [compute]: `L1/linear.py` (single Linear: vision hidden_size → projection_dim)
  - **`PaliGemmaModel`** [wiring]: wires AutoModel (vision_tower — SigLIP), AutoModel (language_model — Gemma/Gemma2/Gemma3), `PaliGemmaMultiModalProjector`
  - **`PaliGemmaForConditionalGeneration`** [wiring]: wires `PaliGemmaModel`; direct `L1/linear.py` (lm_head, weight-tied)

## parakeet
- **src**: modeling_parakeet.py (and modular_parakeet.py)
- **hidden_act**: silu
- **status**: unsupported (Conformer-style ASR encoder + CTC head; no kb-nano Parakeet/Conformer L4)
- **classes**:
  - **`ParakeetEncoderRelPositionalEncoding`** [compute]: position encoding generating sin/cos from inv_freq with relative positions; not a standard kb-nano kernel — closest is `L1/sinusoidal_embed.py` for 1D, but signature differs; (no exact match)
  - **`ParakeetEncoderFeedForward`** [compute]: `L1/linear.py + L1/silu.py + L1/linear.py` (2-layer FFN with silu + dropout)
  - **`ParakeetEncoderConvolutionModule`** [compute, inherits `FastSpeech2ConformerConvolutionModule`]: `L1/conv1d.py + L1/silu.py` (pointwise + GLU + depthwise conv1d + BatchNorm1d + activation + pointwise; no exact L2 match — Conformer-style conv module)
  - **`ParakeetEncoderAttention`** [compute, inherits `LlamaAttention`]: `L1/linear.py + L1/dense_attention.py` (q/k/v/o + relative_k_proj + bias_u/bias_v + relative position bias matrix; non-causal; structurally similar to T5 rel-pos attention but with explicit Shaw-style _rel_shift; no exact L2 match — `L2/t5_attention.py` differs in interface)
  - **`ParakeetEncoderSubsamplingConv2D`** [compute]: `L1/conv2d.py + L1/relu.py + L1/linear.py` (Conv2d subsampling + ReLU stack + linear projection)
  - **`ParakeetEncoderBlock`** [wiring]: wires `ParakeetEncoderFeedForward` (×2 with 0.5 scaling — Macaron pattern), `ParakeetEncoderAttention`, `ParakeetEncoderConvolutionModule`, `nn.LayerNorm` (×5)
  - **`ParakeetEncoder`** [wiring]: wires `ParakeetEncoderSubsamplingConv2D`, `ParakeetEncoderRelPositionalEncoding`, `ParakeetEncoderBlock` (×N)
  - **`ParakeetForCTC`** [wiring]: wires `ParakeetEncoder`; direct `L1/linear.py` (lm_head — CTC blank prediction)

## patchtsmixer
- **src**: modeling_patchtsmixer.py
- **hidden_act**: gelu (used in PatchTSMixerMLP via `nn.functional.gelu` directly)
- **status**: unsupported (time-series MLP-mixer; no kb-nano L4)
- **classes**:
  - **`PatchTSMixerGatedAttention`** [compute]: `L1/linear.py + L1/softmax.py` (Linear projection + softmax-as-attention-weight + element-wise mul)
  - **`PatchTSMixerBatchNorm`** [compute]: `L1/batch_norm2d.py` (BatchNorm1d on transposed dim; closest match is batch_norm2d but for 1D)
  - **`PatchTSMixerPositionalEncoding`** [compute]: parameter-only sin/cos or random positional encoding (no exact L1 match; `L1/sinusoidal_embed.py` differs)
  - **`PatchTSMixerNormLayer`** [wiring]: wires `PatchTSMixerBatchNorm` or `nn.LayerNorm` based on config
  - **`PatchTSMixerMLP`** [compute]: `L1/linear.py + L1/gelu.py + L1/linear.py` (2-layer fc with GELU + dropout)
  - **`PatchTSMixerChannelFeatureMixerBlock`** [wiring]: wires `PatchTSMixerNormLayer`, `PatchTSMixerMLP`, optional `PatchTSMixerGatedAttention`
  - **`PatchTSMixerAttention`** [compute]: `L1/linear.py + L1/dense_attention.py` (Wav2Vec2-style q/k/v/out_proj attention; non-causal)
  - **`PatchMixerBlock`** [wiring]: wires `PatchTSMixerNormLayer`, `PatchTSMixerMLP`, optional `PatchTSMixerGatedAttention`, optional `PatchTSMixerAttention`
  - **`FeatureMixerBlock`** [wiring]: wires `PatchTSMixerNormLayer`, `PatchTSMixerMLP`, optional `PatchTSMixerGatedAttention`
  - **`PatchTSMixerLayer`** [wiring]: wires `PatchMixerBlock`, `FeatureMixerBlock`, optional `PatchTSMixerChannelFeatureMixerBlock`
  - **`PatchTSMixerBlock`** [wiring]: wires `PatchTSMixerLayer` (×N)
  - **`PatchTSMixerForPredictionHead`** [compute]: `L1/linear.py` (single linear forecast head with flatten + dropout)
  - **`PatchTSMixerLinearHead`** [compute]: `L1/linear.py` (linear projection head)
  - **`PatchTSMixerPretrainHead`** [compute]: `L1/linear.py` (pretrain reconstruction head)
  - **`PatchTSMixerPatchify`** [compute]: tensor unfold + transpose; no kernel
  - **`PatchTSMixerMasking`** [compute]: random/forecast masking; no kernel
  - **`PatchTSMixerStdScaler/MeanScaler/NOPScaler`** [compute]: input scaling; no kernel
  - **`PatchTSMixerEncoder`** [wiring]: wires `PatchTSMixerPositionalEncoding`, `PatchTSMixerBlock`; direct `L1/linear.py` (input projection)
  - **`PatchTSMixerModel`** [wiring]: wires `PatchTSMixerEncoder`, `PatchTSMixerPatchify`, `PatchTSMixerMasking`, scaler
  - **`PatchTSMixerForPretraining`** [wiring]: wires `PatchTSMixerModel`, `PatchTSMixerPretrainHead`
  - **`PatchTSMixerForPrediction`** [wiring]: wires `PatchTSMixerModel`, `PatchTSMixerForPredictionHead`
  - **`InjectScalerStatistics4D`** [compute]: `L1/linear.py` (concat + linear)
- **task heads (2)**: ForTimeSeriesClassification, ForRegression — base + linear (per-task)

## patchtst
- **src**: modeling_patchtst.py
- **hidden_act**: activation_function=gelu
- **status**: unsupported (time-series transformer; no kb-nano L4)
- **classes**:
  - **`PatchTSTAttention`** [compute]: `L1/linear.py + L1/dense_attention.py` (Wav2Vec2-style q/k/v/out_proj; ALL_ATTENTION_FUNCTIONS dispatch; non-causal)
  - **`PatchTSTBatchNorm`** [compute]: `L1/batch_norm2d.py` (BatchNorm1d on transposed dim)
  - **`PatchTSTPatchify`** [compute]: tensor unfold; no kernel
  - **`PatchTSTMasking`** [compute]: random/forecast masking; no kernel
  - **`PatchTSTEncoderLayer`** [wiring]: wires `PatchTSTAttention`, `PatchTSTBatchNorm`/`nn.LayerNorm` (sublayer norms ×3), optional second attention for channel; direct `L1/linear.py + L1/gelu.py + L1/linear.py` (FF Sequential)
  - **`PatchTSTEmbedding`** [compute]: `L1/linear.py` (per-channel or shared input projection)
  - **`PatchTSTPositionalEncoding`** [compute]: sin/cos or random parameter; no exact L1 match
  - **`PatchTSTEncoder`** [wiring]: wires `PatchTSTEmbedding`, `PatchTSTPositionalEncoding`, `PatchTSTEncoderLayer` (×N)
  - **`PatchTSTStdScaler/MeanScaler/NOPScaler/Scaler`** [compute]: input scaling; no kernel
  - **`PatchTSTModel`** [wiring]: wires `PatchTSTEncoder`, `PatchTSTPatchify`, `PatchTSTMasking`, `PatchTSTScaler`
  - **`PatchTSTMaskPretrainHead`** [compute]: `L1/linear.py` (single linear)
  - **`PatchTSTForPretraining`** [wiring]: wires `PatchTSTModel`, `PatchTSTMaskPretrainHead`
  - **`PatchTSTClassificationHead`** [compute]: `L1/linear.py` (flatten + linear)
  - **`PatchTSTPredictionHead`** [compute]: `L1/linear.py` (per-channel forecast head)
  - **`PatchTSTForPrediction`** [wiring]: wires `PatchTSTModel`, `PatchTSTPredictionHead`
  - **`PatchTSTRegressionHead`** [compute]: `L1/linear.py` (flatten + linear)
- **task heads (3)**: ForClassification, ForPretraining, ForRegression — base + linear (per-task)

## pe_audio
- **src**: modeling_pe_audio.py (and modular_pe_audio.py)
- **hidden_act**: silu
- **status**: partial (transformer encoder is composable from kb-nano L1/L2; DAC encoder + Snake1d activations are unique audio codec components without exact kb-nano analogues)
- **classes**:
  - **`Snake1d`** [compute]: hidden + (alpha+eps).reciprocal * sin(alpha*hidden)^2; no L1 match (Snake activation)
  - **`PeAudioDacResidualUnit`** [compute]: `L1/conv1d.py + Snake1d + L1/conv1d.py` (residual + 2 conv1d with dilation; weight-normalized)
  - **`PeAudioDacEncoderBlock`** [wiring]: wires `PeAudioDacResidualUnit` (×3), `Snake1d`, `nn.Conv1d` (downsampling)
  - **`PeAudioDacEncoder`** [wiring]: wires `PeAudioDacEncoderBlock` (×N), `Snake1d`, `nn.Conv1d`
  - **`PeAudioEncoderEmbedder`** [wiring, inherits `nn.Module`]: wires `PeAudioDacEncoder`; direct `L1/conv1d.py` (bottleneck), `L1/linear.py` (data_proj)
  - **`PeAudioContrastiveHead`** [compute, inherits `PeAudioVideoContrastiveHead`]: `L1/layer_norm.py + L1/linear.py`
  - **`PeAudioMaskedGroupNorm`** [compute, inherits `nn.GroupNorm`]: `L1/group_norm.py` (with masked mean/var)
  - **`PeAudioConvBlock1d`** [compute]: `L1/group_norm.py + L1/silu.py + L1/conv1d.py` (group_norm + silu + pointwise conv1d)
  - **`PeAudioResnetBlock1d`** [wiring]: wires `PeAudioConvBlock1d` (×2)
  - **`PeAudioEncoderPatchEmbedder`** [wiring]: wires `PeAudioResnetBlock1d`; direct class_embedding parameter
  - **`PeAudioEncoderRMSNorm`** [compute]: `L1/rms_norm.py`
  - **`PeAudioEncoderAttention`** [compute]: `L2/attention.py` (q/k/v/o + non-standard rotary stack_freqs RoPE + ALL_ATTENTION_FUNCTIONS dispatch; non-causal — note: uses 2x2 freqs_cis stacking, not standard rotate_half)
  - **`PeAudioEncoderMLP`** [compute]: `L2/llama_mlp.py` (SwiGLU)
  - **`PeAudioEncoderLayer`** [wiring]: wires `PeAudioEncoderAttention`, `PeAudioEncoderMLP`, `PeAudioEncoderRMSNorm` (×2)
  - **`PeAudioEncoderRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (default RoPE init)
  - **`PeAudioEncoder`** [wiring, inherits `PeAudioVideoEncoder`]: wires `PeAudioEncoderEmbedder`, `PeAudioEncoderPatchEmbedder`, `PeAudioEncoderRotaryEmbedding`, `PeAudioEncoderLayer` (×N), `nn.LayerNorm`, `PeAudioContrastiveHead`
  - **`PeAudioModel`** [wiring]: wires `PeAudioEncoder`
  - **`PeAudioFrameLevelModel`** [wiring]: wires `PeAudioModel` (frame-level outputs)

## pe_audio_video
- **src**: modeling_pe_audio_video.py (and modular_pe_audio_video.py)
- **hidden_act**: silu
- **status**: partial (transformer encoder is composable; uses AutoModel for audio + video sub-encoders)
- **classes**:
  - **`PeAudioVideoMaskedGroupNorm`** [compute]: `L1/group_norm.py` (with masked mean/var)
  - **`PeAudioVideoConvBlock1d`** [compute]: `L1/group_norm.py + L1/silu.py + L1/conv1d.py`
  - **`PeAudioVideoResnetBlock1d`** [wiring]: wires `PeAudioVideoConvBlock1d` (×2)
  - **`PeAudioVideoEncoderPatchEmbedder`** [wiring]: wires `PeAudioVideoResnetBlock1d`; direct class_embedding parameter
  - **`PeAudioVideoContrastiveHead`** [compute]: `L1/layer_norm.py + L1/linear.py`
  - **`PeAudioVideoEncoderEmbedder`** [wiring]: wires AutoModel (audio_encoder), AutoModel (video_encoder); direct `L1/conv1d.py` (video_proj), `L1/layer_norm.py`, `L1/linear.py` (concat_modality_proj, data_proj), `L1/interpolate.py` (alignment)
  - **`PeAudioVideoEncoderAttention`** [compute, inherits `Qwen3Attention`]: `L2/attention.py` (q_norm/k_norm RMSNorm per head_dim + custom stack_freqs RoPE + ALL_ATTENTION_FUNCTIONS dispatch; non-causal)
  - **`PeAudioVideoEncoderMLP`** [compute]: `L2/llama_mlp.py` (SwiGLU)
  - **`PeAudioVideoEncoderLayer`** [wiring, inherits `Qwen3DecoderLayer`]: wires `PeAudioVideoEncoderAttention`, `PeAudioVideoEncoderMLP`, `PeAudioVideoEncoderRMSNorm` (×2)
  - **`PeAudioVideoEncoderRMSNorm`** [compute, inherits `Qwen3RMSNorm`]: `L1/rms_norm.py`
  - **`PeAudioVideoEncoderRotaryEmbedding`** [compute, inherits `Qwen3RotaryEmbedding`]: `L1/rotary_emb.py`
  - **`PeAudioVideoEncoder`** [wiring]: wires `PeAudioVideoEncoderEmbedder`, `PeAudioVideoEncoderPatchEmbedder`, `PeAudioVideoEncoderRotaryEmbedding`, `PeAudioVideoEncoderLayer` (×N), `nn.LayerNorm`, `PeAudioVideoContrastiveHead`
  - **`PeAudioVideoModel`** [wiring]: wires `PeAudioVideoEncoder`

## pe_video
- **src**: modeling_pe_video.py (and modular_pe_video.py)
- **hidden_act**: silu
- **status**: partial (transformer encoder is composable; uses AutoModelForImageClassification for vision sub-encoder)
- **classes**:
  - **`PeVideoContrastiveHead`** [compute, inherits `PeAudioVideoContrastiveHead`]: `L1/layer_norm.py + L1/linear.py`
  - **`PeVideoMaskedGroupNorm`** [compute]: `L1/group_norm.py`
  - **`PeVideoConvBlock1d`** [compute]: `L1/group_norm.py + L1/silu.py + L1/conv1d.py`
  - **`PeVideoResnetBlock1d`** [wiring]: wires `PeVideoConvBlock1d` (×2)
  - **`PeVideoEncoderPatchEmbedder`** [wiring, inherits `PeAudioVideoEncoderPatchEmbedder`]: wires `PeVideoResnetBlock1d`
  - **`PeVideoEncoderEmbedder`** [wiring]: wires AutoModelForImageClassification; direct `L1/linear.py` (proj, data_proj)
  - **`PeVideoEncoderRMSNorm`** [compute]: `L1/rms_norm.py`
  - **`PeVideoEncoderAttention`** [compute]: `L2/attention.py` (q_norm/k_norm per head_dim + stack_freqs RoPE)
  - **`PeVideoEncoderMLP`** [compute]: `L2/llama_mlp.py` (SwiGLU)
  - **`PeVideoEncoderLayer`** [wiring]: wires `PeVideoEncoderAttention`, `PeVideoEncoderMLP`, `PeVideoEncoderRMSNorm` (×2)
  - **`PeVideoEncoderRotaryEmbedding`** [compute]: `L1/rotary_emb.py`
  - **`PeVideoEncoder`** [wiring, inherits `PeAudioVideoEncoder`]: wires `PeVideoEncoderEmbedder`, `PeVideoEncoderPatchEmbedder`, `PeVideoEncoderRotaryEmbedding`, `PeVideoEncoderLayer` (×N), `nn.LayerNorm`, `PeVideoContrastiveHead`
  - **`PeVideoModel`** [wiring]: wires `PeVideoEncoder`

## pegasus
- **src**: modeling_pegasus.py
- **hidden_act**: activation_function=gelu
- **status**: unsupported (encoder-decoder summarization; no kb-nano Pegasus L4 — closest is `L4/whisper.py` for enc-dec)
- **classes**:
  - **`PegasusSinusoidalPositionalEmbedding`** [compute, inherits `nn.Embedding`]: fixed sinusoidal positional embedding (no exact L1 match — `L1/sinusoidal_embed.py` is similar but interface differs)
  - **`PegasusAttention`** [compute]: `L1/linear.py + L1/dense_attention.py + L1/store_kvcache.py` (Bart-style q/k/v/out_proj; supports self/cross attention with KV cache and EncoderDecoderCache; closest L2 is `L2/whisper_attention.py`)
  - **`PegasusEncoderLayer`** [wiring]: wires `PegasusAttention`, `nn.LayerNorm` (×2); direct `L1/linear.py` (fc1, fc2), `L1/gelu.py`
  - **`PegasusDecoderLayer`** [wiring]: wires `PegasusAttention` (self), `PegasusAttention` (cross), `nn.LayerNorm` (×3); direct `L1/linear.py`, `L1/gelu.py`
  - **`PegasusEncoder`** [wiring]: wires `PegasusSinusoidalPositionalEmbedding`, `PegasusEncoderLayer` (×N), `nn.LayerNorm`; direct `L1/embedding.py`
  - **`PegasusDecoder`** [wiring]: wires `PegasusSinusoidalPositionalEmbedding`, `PegasusDecoderLayer` (×N), `nn.LayerNorm`; direct `L1/embedding.py`
  - **`PegasusModel`** [wiring]: wires `PegasusEncoder`, `PegasusDecoder`
  - **`PegasusForConditionalGeneration`** [wiring]: wires `PegasusModel`; direct `L1/linear.py` (lm_head)
  - **`PegasusDecoderWrapper`** [wiring]: wires `PegasusDecoder`
  - **`PegasusForCausalLM`** [wiring]: wires `PegasusDecoderWrapper`; direct `L1/linear.py` (lm_head)

## pegasus_x
- **src**: modeling_pegasus_x.py
- **hidden_act**: activation_function=gelu
- **status**: unsupported (long-context summarization with global+local attention blocks; no kb-nano L4)
- **classes**:
  - **`PegasusXScaledWordEmbedding`** [compute, inherits `nn.Embedding`]: `L1/embedding.py` (with embed_scale multiplier)
  - **`PegasusXSinusoidalPositionalEmbedding`** [compute]: sinusoidal positional embedding parameter; no exact L1 match
  - **`PegasusXAttention`** [compute]: `L1/linear.py + L1/dense_attention.py + L1/store_kvcache.py` (Bart-style decoder attention)
  - **`PegasusXGlobalLocalAttention`** [compute]: einsum-based global+blocked-local attention with shared q/k/v/out_proj; no L2 match (custom long-context pattern, akin to BigBird/LongFormer)
  - **`PegasusXEncoderLayer`** [wiring]: wires `PegasusXGlobalLocalAttention`, `nn.LayerNorm` (×2); direct `L1/linear.py + L1/gelu.py + L1/linear.py`
  - **`PegasusXDecoderLayer`** [wiring]: wires `PegasusXAttention` (self+cross), `nn.LayerNorm` (×3); direct `L1/linear.py + L1/gelu.py + L1/linear.py`
  - **`PegasusXEncoder`** [wiring]: wires `PegasusXScaledWordEmbedding`, `PegasusXSinusoidalPositionalEmbedding`, `PegasusXEncoderLayer` (×N), `nn.LayerNorm`
  - **`PegasusXDecoder`** [wiring]: wires `PegasusXScaledWordEmbedding`, `PegasusXSinusoidalPositionalEmbedding`, `PegasusXDecoderLayer` (×N), `nn.LayerNorm`
  - **`PegasusXModel`** [wiring]: wires `PegasusXEncoder`, `PegasusXDecoder`
  - **`PegasusXForConditionalGeneration`** [wiring]: wires `PegasusXModel`; direct `L1/linear.py` (lm_head)
  - **`PegasusXDecoderWrapper`** [wiring]: wires `PegasusXDecoder`

## perceiver
- **src**: modeling_perceiver.py
- **hidden_act**: gelu
- **status**: unsupported (Perceiver IO with cross-attention + latent self-attention + multimodal preprocessors/postprocessors; no kb-nano Perceiver L4)
- **classes**:
  - **`PerceiverEmbeddings`** [compute]: latents `nn.Parameter` with batch expand; no L1 match (just learnable parameter)
  - **`PerceiverSelfAttention`** [compute]: `L1/layer_norm.py + L1/linear.py + L1/dense_attention.py` (q/k/v Linear + manual matmul/softmax; supports cross-attention with separate kv layernorm; non-causal)
  - **`PerceiverSelfOutput`** [compute]: `L1/linear.py` (single dense projection)
  - **`PerceiverAttention`** [wiring]: wires `PerceiverSelfAttention`, `PerceiverSelfOutput` (with optional query residual)
  - **`PerceiverMLP`** [compute]: `L1/linear.py + L1/gelu.py + L1/linear.py` (2-layer FFN with GELU)
  - **`PerceiverLayer`** [wiring]: wires `PerceiverAttention`, `nn.LayerNorm`, `PerceiverMLP`
  - **`PerceiverEncoder`** [wiring]: wires `PerceiverLayer` (cross-attention + N self-attention layers, repeated num_blocks times)
  - **`PerceiverModel`** [wiring]: wires `PerceiverEmbeddings`, `PerceiverEncoder`, optional input_preprocessor (one of: Text/Image/Audio/MultimodalPreprocessor), optional decoder, optional output_postprocessor
  - **`PerceiverProjectionDecoder/PerceiverBasicDecoder/PerceiverClassificationDecoder/PerceiverOpticalFlowDecoder/PerceiverBasicVideoAutoencodingDecoder/PerceiverMultimodalDecoder`** [wiring/compute]: decoder variants — each wires `PerceiverLayer` (single cross-attn) + position encoding + `L1/linear.py` (final_layer)
  - **`Conv2dSamePadding`** [compute, inherits `nn.Conv2d`]: `L1/conv2d.py` (with TF-style same padding)
  - **`Conv2DDownsample`** [compute]: `L1/conv2d.py + L1/relu.py + L1/max_pool2d.py + L1/batch_norm2d.py` (downsampling block)
  - **`PerceiverTrainablePositionEncoding`** [compute]: `L1/embedding.py` (learned per-position embedding)
  - **`PerceiverFourierPositionEncoding`** [compute]: Fourier features (sin/cos); no exact L1 match
  - **`PerceiverTextPreprocessor`** [compute]: `L1/embedding.py + L1/embedding.py` (token + position embedding sum)
  - **`PerceiverImagePreprocessor/PerceiverAudioPreprocessor/PerceiverOneHotPreprocessor/PerceiverMultimodalPreprocessor`** [wiring/compute]: variant-specific input preprocessors with various conv2d/conv1d + positional encoding combinations
  - **`PerceiverEmbeddingDecoder`** [compute]: `L1/linear.py` (project to vocab via tied embedding matmul)
  - **`PerceiverMultimodalPostprocessor/PerceiverClassificationPostprocessor/PerceiverAudioPostprocessor/PerceiverProjectionPostprocessor`** [wiring/compute]: variant postprocessors
  - **`PerceiverForMaskedLM`** [wiring]: wires `PerceiverModel` with text preprocessor, basic decoder, embedding decoder
- **task heads (8)**: ForSequenceClassification, ForImageClassificationLearned, ForImageClassificationFourier, ForImageClassificationConvProcessing, ForOpticalFlow, ForMultimodalAutoencoding — base + variant decoders/postprocessors (per-task)

## perception_lm
- **src**: modeling_perception_lm.py (and modular_perception_lm.py)
- **hidden_act**: defers to vision_config (timm-based) and text_config (Llama-family silu)
- **status**: composable (uses AutoModel for vision and language sub-models — Llava-style)
- **classes**:
  - **`PerceptionLMAdaptiveAvgPooling`** [compute]: `L1/adaptive_avg_pool2d.py` (reshape + adaptive_avg_pool2d for spatial token reduction)
  - **`PerceptionLMMultiModalProjector`** [compute]: `L1/linear.py + L1/gelu.py + L1/linear.py + PerceptionLMAdaptiveAvgPooling` (2-layer MLP + optional pooling)
  - **`PerceptionLMModel`** [wiring, inherits `LlavaModel`]: wires AutoModel (vision_tower), AutoModel (language_model), `PerceptionLMMultiModalProjector`
  - **`PerceptionLMForConditionalGeneration`** [wiring, inherits `LlavaForConditionalGeneration`]: wires `PerceptionLMModel`; direct `L1/linear.py` (lm_head)

## persimmon
- **src**: modeling_persimmon.py
- **hidden_act**: relu2 (squared_relu)
- **status**: composable (Persimmon uses fused QKV with partial RoPE + squared_relu activation in 2-layer MLP — no exact L4 but composable)
- **classes**:
  - **`PersimmonRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (default RoPE with partial_rotary_factor)
  - **`PersimmonMLP`** [compute]: `L1/linear.py + L1/squared_relu.py + L1/linear.py` (2-layer FFN: dense_h_to_4h + relu2 + dense_4h_to_h; matches GPTNeoX-style pattern)
  - **`PersimmonAttention`** [compute]: `L1/linear.py + L1/dense_attention.py + L1/store_kvcache.py` (fused QKV split + per-head q_layernorm/k_layernorm (LayerNorm, not RMSNorm) + partial RoPE; closest L2 `L2/attention.py` doesn't match exactly due to fused QKV linear + partial RoPE + LayerNorm-not-RMSNorm qk_norm)
  - **`PersimmonDecoderLayer`** [wiring]: wires `PersimmonAttention`, `PersimmonMLP`, `nn.LayerNorm` (×2 — input_layernorm, post_attention_layernorm)
  - **`PersimmonModel`** [wiring]: wires `PersimmonDecoderLayer`, `nn.LayerNorm` (final_layernorm), `PersimmonRotaryEmbedding`; direct `L1/embedding.py`
  - **`PersimmonForCausalLM`** [wiring]: wires `PersimmonModel`; direct `L1/linear.py` (lm_head)
- **task heads (2)**: ForSequenceClassification, ForTokenClassification — base + linear (per-task)

## phi
- **src**: modeling_phi.py (and modular_phi.py)
- **hidden_act**: gelu_new
- **status**: composable (Phi-1/1.5/2 — partial RoPE + parallel attn+MLP residual)
- **classes**:
  - **`PhiRotaryEmbedding`** [compute, inherits `LlamaRotaryEmbedding`]: `L1/rotary_emb.py` (with partial_rotary_factor)
  - **`PhiAttention`** [compute, inherits `LlamaAttention`]: `L1/linear.py + L1/dense_attention.py + L1/store_kvcache.py` (q/k/v/dense Linear with bias + optional q_layernorm/k_layernorm (LayerNorm) + partial RoPE; not a clean fit for `L2/attention.py` due to LayerNorm-qk_norm + dense (not o_proj) + partial RoPE)
  - **`PhiMLP`** [compute, inherits `CLIPMLP`]: `L2/clip_mlp.py` (fc1 → gelu_new → fc2; gelu_new resolves to `L1/gelu.py`)
  - **`PhiDecoderLayer`** [wiring]: wires `PhiAttention`, `PhiMLP`, `nn.LayerNorm` (input_layernorm) — note: parallel attn+MLP residual (attn_outputs + feed_forward_hidden_states + residual, single LN)
  - **`PhiModel`** [wiring, inherits `LlamaModel`]: wires `PhiDecoderLayer`, `nn.LayerNorm` (final_layernorm), `PhiRotaryEmbedding`; direct `L1/embedding.py`
  - **`PhiForCausalLM`** [wiring, inherits `LlamaForCausalLM`]: wires `PhiModel`; direct `L1/linear.py` (lm_head with bias)
- **task heads (2)**: ForSequenceClassification, ForTokenClassification — base + linear (per-task)

## phi3
- **src**: modeling_phi3.py (and modular_phi3.py)
- **hidden_act**: silu
- **status**: composable (Phi-3 — fused gate_up_proj SwiGLU + partial RoPE + sliding window)
- **classes**:
  - **`Phi3MLP`** [compute]: `L2/llama_mlp.py` (fused gate_up_proj single Linear with chunk(2) → silu(gate)*up → down_proj — same MergedColumnParallelLinear pattern as `LlamaMLP`)
  - **`Phi3RotaryEmbedding`** [compute, inherits `PhiRotaryEmbedding`]: `L1/rotary_emb.py` (with partial_rotary_factor; supports yarn/longrope via ROPE_INIT_FUNCTIONS)
  - **`Phi3Attention`** [compute]: `L2/attention.py` (fused qkv_proj single Linear + slice into Q/K/V + partial RoPE + KV cache + sliding_window via ALL_ATTENTION_FUNCTIONS dispatch — note: Phi3 fuses QKV into one linear like Mistral; partial RoPE applies only to first rotary_dim channels)
  - **`Phi3RMSNorm`** [compute]: `L1/rms_norm.py`
  - **`Phi3DecoderLayer`** [wiring, inherits `MistralDecoderLayer`]: wires `Phi3Attention`, `Phi3MLP`, `Phi3RMSNorm` (×2 — input_layernorm + post_attention_layernorm)
  - **`Phi3Model`** [wiring]: wires `Phi3DecoderLayer`, `Phi3RMSNorm` (final), `Phi3RotaryEmbedding`; direct `L1/embedding.py`
  - **`Phi3ForCausalLM`** [wiring, inherits `MistralForCausalLM`]: wires `Phi3Model`; direct `L1/linear.py` (lm_head)
- **task heads (2)**: ForSequenceClassification, ForTokenClassification — base + linear (per-task)

## phi4_multimodal
- **src**: modeling_phi4_multimodal.py (and modular_phi4_multimodal.py)
- **hidden_act**: vision=gelu_pytorch_tanh, text=silu, audio=swish (silu)
- **status**: partial (text branch is Phi3-style → composable; vision branch is SigLIP-style → composable; audio branch is Conformer-style with relative attention bias and depthwise conv → no exact L4)
- **classes**:
  - **`Phi4MultimodalVisionMLP`** [compute, inherits `SiglipMLP`]: `L2/siglip_mlp.py` (fc1 → gelu_pytorch_tanh → fc2)
  - **`Phi4MultimodalVisionAttention`** [compute]: `L2/siglip_attention.py` (q/k/v/out_proj with ALL_ATTENTION_FUNCTIONS dispatch; non-causal)
  - **`Phi4MultimodalVisionEncoderLayer`** [wiring, inherits `SiglipEncoderLayer`]: wires `Phi4MultimodalVisionAttention`, `Phi4MultimodalVisionMLP`, `nn.LayerNorm` (×2)
  - **`Phi4MultimodalVisionEncoder`** [wiring, inherits `SiglipEncoder`]: wires `Phi4MultimodalVisionEncoderLayer` (×N)
  - **`Phi4MultimodalVisionEmbeddings`** [compute, inherits `SiglipVisionEmbeddings`]: `L1/conv2d.py + L1/embedding.py` (Conv2d patch_embedding + position_embedding)
  - **`Phi4MultimodalVisionMultiheadAttentionPoolingHead`** [compute, inherits `SiglipMultiheadAttentionPoolingHead`]: `L1/dense_attention.py + L1/linear.py` (single learnable query + multi-head attention pooling)
  - **`Phi4MultimodalVisionModel`** [wiring]: wires `Phi4MultimodalVisionEmbeddings`, `Phi4MultimodalVisionEncoder`, `nn.LayerNorm`, `Phi4MultimodalVisionMultiheadAttentionPoolingHead`
  - **`Phi4MultimodalImageEmbedding`** [wiring]: wires `Phi4MultimodalVisionModel`; direct `L1/linear.py` (image projection)
  - **`Phi4MultimodalAudioMLP`** [compute]: `L1/linear.py + L1/silu.py + L1/linear.py` (2-layer FFN with silu)
  - **`Phi4MultimodalAudioAttention`** [compute]: `L1/linear.py + L1/dense_attention.py` (q/k/v/o + relative attention bias; non-causal)
  - **`Phi4MultimodalAudioDepthWiseSeparableConv1d`** [compute]: `L1/conv1d.py + L1/conv1d.py` (pointwise + depthwise)
  - **`Phi4MultimodalAudioGluPointWiseConv`** [compute]: `L1/conv1d.py + L1/sigmoid.py` (GLU mechanism)
  - **`Phi4MultimodalAudioConvModule`** [wiring]: wires `Phi4MultimodalAudioGluPointWiseConv`, `Phi4MultimodalAudioDepthWiseSeparableConv1d`, `nn.LayerNorm`, `nn.BatchNorm1d`; direct `L1/silu.py`
  - **`Phi4MultimodalAudioConformerEncoderLayer`** [wiring]: wires `Phi4MultimodalAudioMLP` (×2 with 0.5 scaling — Macaron pattern), `Phi4MultimodalAudioAttention`, `Phi4MultimodalAudioConvModule`, `nn.LayerNorm` (×N)
  - **`Phi4MultimodalAudioNemoConvSubsampling`** [compute]: `L1/conv2d.py + L1/relu.py + L1/linear.py` (Conv2d-based subsampling stack)
  - **`Phi4MultimodalAudioRelativeAttentionBias`** [compute]: `L1/embedding.py` (relative bias as learned embedding)
  - **`Phi4MultimodalAudioMeanVarianceNormLayer`** [compute]: simple mean/variance normalization; no kernel
  - **`Phi4MultimodalAudioModel`** [wiring]: wires `Phi4MultimodalAudioMeanVarianceNormLayer`, `Phi4MultimodalAudioNemoConvSubsampling`, `Phi4MultimodalAudioRelativeAttentionBias`, `Phi4MultimodalAudioConformerEncoderLayer` (×N), `nn.LayerNorm`
  - **`Phi4MultimodalAudioEmbedding`** [wiring]: wires `Phi4MultimodalAudioModel`; direct `L1/linear.py` (audio projection)
  - **`Phi4MultimodalRMSNorm`** [compute, inherits `Phi3RMSNorm`]: `L1/rms_norm.py`
  - **`Phi4MultimodalMLP`** [compute]: `L2/llama_mlp.py` (fused gate_up_proj SwiGLU — same as Phi3MLP)
  - **`Phi4MultimodalAttention`** [compute]: `L2/attention.py` (fused qkv_proj + partial RoPE + KV cache + sliding_window — same as Phi3Attention)
  - **`Phi4MultimodalDecoderLayer`** [wiring]: wires `Phi4MultimodalAttention`, `Phi4MultimodalMLP`, `Phi4MultimodalRMSNorm` (×2)
  - **`Phi4MultimodalFeatureEmbedding`** [wiring]: wires `Phi4MultimodalImageEmbedding`, `Phi4MultimodalAudioEmbedding`; direct `L1/embedding.py` (text token embedding)
  - **`Phi4MultimodalRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (with partial_rotary_factor)
  - **`Phi4MultimodalModel`** [wiring]: wires `Phi4MultimodalFeatureEmbedding`, `Phi4MultimodalDecoderLayer` (×N), `Phi4MultimodalRMSNorm`, `Phi4MultimodalRotaryEmbedding`
  - **`Phi4MultimodalForCausalLM`** [wiring]: wires `Phi4MultimodalModel`; direct `L1/linear.py` (lm_head)
