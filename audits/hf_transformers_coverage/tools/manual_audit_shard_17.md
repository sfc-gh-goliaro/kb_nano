## granite4_vision
- **src**: modular_granite4_vision.py
- **status**: composable
- **rationale**: LlavaNext-derived multimodal wrapper: Granite (Llama-family with scalar multipliers) text backbone + AutoModel-loaded vision encoder (typically SigLIP) + Window QFormer downsampler (Blip2QFormer via AutoModel) + deepstack feature injection. Vision/text components are composable; QFormer is BERT-style cross-attention which decomposes to encoder_attention + linear primitives.
- **classes**:
  - **`Granite4VisionWindowQFormerDownsampler`** [wiring]: QFormer-based windowed downsampler that pairs learnable queries with windowed image patches via BERT cross-attention. AutoModel.from_config(qformer_config) loads Blip2QFormer (encoder_attention + cross-attention + GELU MLP); primitives exist but no L2 wrapper specifically for QFormer.
  - **`Granite4VisionTextRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (GraniteRotaryEmbedding = standard NeoX RoPE.)
  - **`Granite4VisionTextAttention`** [compute]: `L2/attention.py` (GraniteAttention — Llama GQA with attention_multiplier as scaling. attention.py supports scalar scale override.)
  - **`Granite4VisionTextDecoderLayer`** [wiring]: GraniteDecoderLayer wiring + residual_multiplier scalar mul.
  - **`Granite4VisionTextModel`** [wiring]: Wiring + deepstack feature injection at marked layers (masked_scatter).
  - **`Granite4VisionModel`** [wiring]: Wiring: vision encoder via AutoModel.from_config + deepstack/spatial Window QFormer projectors + Granite text model.
  - **`Granite4VisionForConditionalGeneration`** [wiring]: Wiring (inherits LlavaNextForConditionalGeneration).

## granite_speech_plus
- **src**: modular_granite_speech_plus.py
- **status**: partial
- **partial_reason**: GraniteSpeechConformerConvModule uses nn.BatchNorm1d (granite_speech/modeling_granite_speech.py:226) — kb-nano only has L1/batch_norm2d.py; BatchNorm1d would fall back to torch.nn.BatchNorm1d.
- **rationale**: Conformer audio encoder relies on nn.BatchNorm1d in its conv module, and kb-nano has only batch_norm2d (no batch_norm1d). Other Conformer ops (depthwise conv1d, GLU, Shaw rel-pos via SDPA attn_mask, LayerNorm/SiLU, linear FFN) are composable from existing primitives. Granite LM + Blip2QFormer projector + LoRA adapter are otherwise composable.
- **classes**:
  - **`GraniteSpeechPlusCTCEncoder`** [compute]: no kb-nano kernel — GraniteSpeechConformerConvModule uses nn.BatchNorm1d (granite_speech/modeling_granite_speech.py:226) — kb-nano only has L1/batch_norm2d.py; BatchNorm1d would fall back to torch.nn.BatchNorm1d.
  - **`GraniteSpeechConformerFeedForward`** [compute]: `L1/linear.py`, `L1/silu.py`, `L1/layer_norm.py` (LayerNorm -> Linear -> SiLU -> dropout -> Linear -> dropout (granite_speech/modeling_granite_speech.py:107).)
  - **`GraniteSpeechConformerAttention`** [compute]: `L1/sdpa.py`, `L1/embedding.py` (Block-windowed self-attention with Shaw relative position bias (rel_pos_emb from Embedding + einsum-based pos_attn) used as attn_mask in scaled_dot_product_attention. Composable from sdpa + embedding + linear (no kb-nano L2 wrapper for this exact pattern).)
  - **`GraniteSpeechConformerDepthWiseConv1d`** [compute]: `L1/conv1d.py` (F.pad + Conv1d with groups=in_channels.)
  - **`GraniteSpeechConformerConvModule`** [wiring]: LayerNorm + pointwise Conv1d + GLU + DepthwiseConv1d + nn.BatchNorm1d + SiLU + Conv1d. The BatchNorm1d has no kb-nano kernel; everything else composes from L1/conv1d.py + L1/sigmoid.py (for GLU) + L1/silu.py + L1/layer_norm.py.
  - **`GraniteSpeechConformerBlock`** [wiring]: Wiring: 0.5*FF1 + Attn + ConvModule + 0.5*FF2 + LayerNorm (Conformer macaron-style).
  - **`GraniteSpeechPlusForConditionalGeneration`** [wiring]: Wiring: CTC encoder + projector (Blip2QFormer-derived) + Granite LM + optional LoRA adapter.

## higgs_audio_v2
- **src**: modular_higgs_audio_v2.py
- **status**: composable
- **rationale**: Llama backbone with dual-pathway decoder layers: parallel `audio_mlp + audio_input_layernorm + audio_post_attention_layernorm` for audio tokens vs. standard text MLP/norms. Routing is via masked_scatter on a boolean audio_token_mask. Plus multi-codebook audio token embeddings (sum over codebooks). All compute decomposes into existing primitives.
- **classes**:
  - **`HiggsAudioV2MLP`** [compute]: `L2/llama_mlp.py` (LlamaMLP — SwiGLU.)
  - **`HiggsAudioV2RMSNorm`** [compute]: `L1/rms_norm.py` (Standard RMSNorm.)
  - **`HiggsAudioV2DecoderLayer`** [wiring]: Wiring: shared self-attn; per-token-type masked_scatter to apply audio_input_layernorm + audio_mlp on audio tokens vs input_layernorm + mlp on text tokens. Pure torch composition over RMSNorm + llama_mlp + attention.py.
  - **`HiggsAudioV2Embeddings`** [compute]: `L1/embedding.py` (embed_audio_tokens(input_ids + audio_tokens_offsets).sum(dim=-2) — multi-codebook audio token embedding; sum over codebooks. Standard embedding + tensor sum.)
  - **`HiggsAudioV2Model`** [wiring]: Wiring: replaces audio placeholder positions with audio embeddings then runs Llama decoder.
  - **`HiggsAudioV2ForConditionalGeneration`** [wiring]: Wiring.

## jais2
- **src**: modular_jais2.py
- **status**: composable
- **rationale**: Llama-family + LayerNorm (instead of RMSNorm) + 2-layer up/down MLP with squared-relu activation (Nemotron-style) + bias on attention/MLP. All primitives present: layer_norm, squared_relu, linear, attention.py (bias=True supported), embedding.
- **classes**:
  - **`Jais2MLP`** [compute]: `L1/linear.py`, `L1/squared_relu.py` (NemotronMLP: up_proj -> squared_relu -> down_proj (with bias). Two-layer fc1+act+fc2; squared_relu activation matches L1/squared_relu.py.)
  - **`Jais2DecoderLayer`** [wiring]: Wiring; uses nn.LayerNorm instead of RMSNorm.
  - **`Jais2Model`** [wiring]: Wiring; final norm = nn.LayerNorm.
  - **`Jais2ForCausalLM`** [wiring]: Wiring.

## jina_embeddings_v3
- **src**: modular_jina_embeddings_v3.py
- **status**: composable
- **rationale**: XLM-Roberta-style bidirectional encoder + RoPE on Q/K + LayerNorm + GELU MLP + bias. All primitives present (rotary_emb, dense_attention/sdpa, linear with bias, layer_norm, embedding, GELU MLP). The pre-MLP+post-MLP LayerNorm pattern from GPTNeoXLayer is wiring.
- **classes**:
  - **`JinaEmbeddingsV3Embeddings`** [compute]: `L2/encoder_embeddings.py`, `L1/embedding.py`, `L1/layer_norm.py` (word_emb + token_type_emb + LayerNorm + dropout (no learnable position embedding — RoPE applied later).)
  - **`JinaEmbeddingsV3RotaryEmbedding`** [compute]: `L1/rotary_emb.py` (Inherits LlamaRotaryEmbedding.)
  - **`JinaEmbeddingsV3Attention`** [compute]: `L1/sdpa.py`, `L1/linear.py`, `L1/rotary_emb.py` (LlamaAttention-derived but bidirectional (is_causal=False), num_kv_heads = num_attention_heads (no GQA), bias=True on Q/K/V/O. Compute is RoPE -> sdpa with bidirectional mask.)
  - **`JinaEmbeddingsV3MLP`** [compute]: `L2/clip_mlp.py` (CLIPMLP — fc1+act+fc2 with bias.)
  - **`JinaEmbeddingsV3Layer`** [wiring]: Wiring (residual+post_attn LayerNorm; residual+post_mlp LayerNorm).
  - **`JinaEmbeddingsV3Pooler`** [wiring]: Wiring (linear + tanh on first token).
  - **`JinaEmbeddingsV3Model`** [wiring]: Wiring; bidirectional mask via create_bidirectional_mask.
  - **`JinaEmbeddingsV3LMHead`** [wiring]: Wiring.
  - **`JinaEmbeddingsV3ForMaskedLM`** [wiring]: Wiring.

## laguna
- **src**: modular_laguna.py
- **status**: partial
- **partial_reason**: configuration_laguna.py sets `full_attention.partial_rotary_factor = 0.5` (sliding_attention uses 1.0). kb-nano L1/rotary_emb.py rotates the full head_dim; partial-rotary on the full-attention layers requires either Gemma4-style proportional embedding or external q_rot/q_pass slicing. Same gap as phi/persimmon/glm. Audit's earlier "partial RoPE supported by L1/rotary_emb.py via rotary_dim < head_dim" was incorrect — that path only exists in Gemma4ProportionalRotaryEmbedding, not the standard NeoX RoPE.
- **rationale**: Per-layer head-count Llama-family attention with QK norm + softplus per-head gate (g_proj) + partial RoPE on full-attn layers; sigmoid+correction-bias router with optional tanh logit softcapping; Qwen3-MoE-style sparse block with shared experts. MoE/attention/RMSNorm/SwiGLU primitives all map to existing L2 wrappers; partial-rotary path on full-attn layers needs external slicing (decomposable from L1 + standard PyTorch).
- **classes**:
  - **`LagunaRMSNorm`** [compute]: `L1/rms_norm.py` (Qwen2MoeRMSNorm = standard RMSNorm.)
  - **`LagunaRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (Per-layer-type rope_parameters (rope_theta, partial_rotary_factor). Standard L1/rotary_emb.RotaryEmbedding rotates the full head_dim; partial-rotary on full_attention layers (factor 0.5) needs external slicing or Gemma4-style proportional embedding.)
  - **`LagunaMLP`** [compute]: `L2/llama_mlp.py` (Qwen2MoeMLP — SwiGLU.)
  - **`LagunaTopKRouter`** [compute]: `L1/sigmoid_topk.py` (tanh logit softcapping (torch op) -> sigmoid -> +e_score_correction_bias -> topk -> renormalize. sigmoid_topk + small torch glue covers it.)
  - **`LagunaExperts`** [compute]: `L1/moe_grouped_gemm.py` (Qwen3MoeExperts — gate_up + down expert kernel.)
  - **`LagunaSparseMoeBlock`** [compute]: `L2/shared_expert_moe.py` (Routed experts + shared expert + routed_scaling_factor scalar mul before adding shared output. shared_expert_moe.py is the canonical kb-nano shared-expert pattern.)
  - **`LagunaAttention`** [compute]: `L2/attention.py` (AfmoeAttention-derived: q/k norm, partial RoPE, sliding-window option, plus per-head softplus gate `g = softplus(g_proj(x))` multiplied per-head onto attn output. Base attention is L2/attention.py with QK-norm; the gate is `softplus(linear) * reshape` torch glue.)
  - **`LagunaDecoderLayer`** [wiring]: Wiring; per-layer dense vs sparse MLP.
  - **`LagunaModel`** [wiring]: Wiring; per-layer-type causal mask + position embeddings.
  - **`LagunaForCausalLM`** [wiring]: Wiring.

## lighton_ocr
- **src**: modular_lighton_ocr.py
- **status**: composable
- **rationale**: Mistral3-style multimodal wrapper: Pixtral vision encoder (default vision_config), Qwen3 text LM (default text_config), and a 2-layer MultiModalProjector with GELU. All sub-components are composable (Pixtral via vision_attention/2D RoPE; Qwen3 via attention.py/llama_mlp/rms_norm).
- **classes**:
  - **`LightOnOcrMultiModalProjector`** [compute]: `L1/linear.py`, `L1/gelu.py` (linear_1 -> GELU -> linear_2 (no bias).)
  - **`LightOnOcrModel`** [wiring]: Wiring: vision_encoder (Pixtral via AutoModel.from_config) + vision_projection + language_model (Qwen3 via AutoModel.from_config) + masked_scatter to inject image embeddings at image_token_id positions.
  - **`LightOnOcrForConditionalGeneration`** [wiring]: Wiring (inherits Mistral3ForConditionalGeneration).

## longcat_flash
- **src**: modular_longcat_flash.py
- **status**: partial
- **partial_reason**: LongcatFlashExperts has zero-compute (Identity) experts (modular_longcat_flash.py:97, 134-135) — kb-nano L1/moe_grouped_gemm.py and L2/deepseek_moe.py have no pass-through identity-expert path; the routing+identity branch falls back to torch index_add_/torch.where logic.
- **rationale**: DeepseekV3-style MLA attention with extra LoRA scaling factors + sigmoid+correction-bias topk routing + custom MoE that contains zero-compute (identity) experts. The identity-expert pass-through is not supported by kb-nano's fused MoE kernels; would require torch fallback.
- **classes**:
  - **`LongcatFlashDecoderLayer`** [compute]: LongcatFlashExperts has zero-compute (Identity) experts (modular_longcat_flash.py:97, 134-135) — kb-nano L1/moe_grouped_gemm.py and L2/deepseek_moe.py have no pass-through identity-expert path; the ro
  - **`LongcatFlashRMSNorm`** [compute]: `L1/rms_norm.py` (DeepseekV3RMSNorm = standard RMSNorm.)
  - **`LongcatFlashRotaryEmbedding`** [compute]: `L1/yarn_rotary_emb.py` (Inherits DeepseekV3RotaryEmbedding; supports rope_type="yarn" via yarn_get_mscale + yarn-init. Per guideline 5 YaRN RoPE maps to yarn_rotary_emb, not bare rotary_emb.)
  - **`LongcatFlashMLP`** [compute]: `L2/llama_mlp.py` (DeepseekV3MLP — SwiGLU gate*up.)
  - **`LongcatFlashTopkRouter`** [compute]: `L1/sigmoid_topk.py` (softmax(router_logits) -> +e_score_correction_bias -> topk -> gather -> scale; analogous to deepseek noaux_tc routing without group-restriction. kb-nano sigmoid_topk supports the analogous sigmoid+bias+topk; here HF uses softmax+bias+topk (small departure).)
  - **`LongcatFlashExperts`** [wiring]: Per-expert gate_up_proj and down_proj with optional zero-compute Identity experts (gate_up_proj/down_proj=None). The identity branch is bespoke; no kb-nano fused kernel supports it.
  - **`LongcatFlashMoE`** [wiring]: Wiring: router + experts (with identity pass-through).
  - **`LongcatFlashMLA`** [compute]: `L2/deepseek_mla_attention.py` (DeepseekV3 MLA + extra LoRA scaling factors mla_scale_q_lora and mla_scale_kv_lora applied as elementwise mul on q_pass/q_rot/k_pass; primitives in deepseek_mla_attention cover everything except the extra scalar mul, which is trivial torch.)
  - **`LongcatFlashModel`** [wiring]: Wiring.
  - **`LongcatFlashForCausalLM`** [wiring]: Wiring.

## lw_detr
- **src**: modular_lw_detr.py
- **status**: composable
- **rationale**: Self-contained ViTDet windowed-attention backbone + multiscale deformable decoder; no AutoBackbone. Vision attention, deformable attention, conv2d/convtranspose2d, layer_norm/layer_norm2d, GELU MLP, BatchNorm2d, RTDetr-style RepVgg/CSP — all primitives available in kb-nano.
- **classes**:
  - **`LwDetrViTSelfAttention`** [compute]: `L1/sdpa.py`, `L1/linear.py` (Standard ViT self-attention (Q/K/V projections + SDPA, no RoPE/no causal). k_proj has bias=False, q/v have bias=True. Maps to bidirectional sdpa primitive.)
  - **`LwDetrViTAttention`** [wiring]: Wiring around LwDetrViTSelfAttention + output projection.
  - **`LwDetrViTMlp`** [compute]: `L2/vit_encoder_mlp.py` (fc1 -> GELU -> dropout -> fc2 -> dropout (modular_lw_detr.py:287, VitDetMlp). Two-layer MLP with bias; matches ViT-style MLP.)
  - **`LwDetrViTLayer`** [wiring]: Wiring with optional window reshape, learned per-channel gamma_1/gamma_2 scaling, LayerNorm, attention, MLP.
  - **`LwDetrViTBackbone`** [wiring]: Wiring; ViTDetBackbone composing patch embed + ViT layers + output indices.
  - **`LwDetrConvNormLayer`** [compute]: `L2/rtdetrv2_conv_norm.py` (Conv2d + BatchNorm2d + activation; same as RTDetrConvNormLayer.)
  - **`LwDetrRepVggBlock`** [compute]: `L2/rtdetrv2_repvgg_block.py` (Two stacked Conv-BN-Act blocks; matches RTDetr RepVgg pattern.)
  - **`LwDetrC2FLayer`** [compute]: `L2/rtdetrv2_csp_rep_layer.py` (CSP/C2F-style branching with bottlenecks; closely matches RTDetr CSPRepLayer.)
  - **`LwDetrLayerNorm`** [compute]: `L1/layer_norm2d.py` (ConvNeXt-style channels-first LayerNorm; layer_norm2d.py covers permute+layer_norm+permute.)
  - **`LwDetrSamplingLayer`** [compute]: `L1/conv2d.py`, `L1/conv_transpose2d.py` (Up/downsampling via Conv2d / ConvTranspose2d.)
  - **`LwDetrScaleProjector`** [wiring]: Wiring: sampling layers + C2F + LayerNorm.
  - **`LwDetrMultiScaleProjector`** [wiring]: Wiring: stack of LwDetrScaleProjectors.
  - **`LwDetrConvEncoder`** [wiring]: Wiring: ViT backbone + projector.
  - **`LwDetrAttention`** [compute]: `L2/encoder_attention.py` (Self-attention with optional group-detr split during training (just torch reshape); bidirectional, bias-aware Q/K/V projections.)
  - **`LwDetrMultiscaleDeformableAttention`** [compute]: `L1/rtdetrv2_deformable_attention.py`, `L2/rtdetrv2_deformable_attention.py` (Inherits DeformableDetrMultiscaleDeformableAttention; same sampling-grid math as kb-nano rtdetrv2 deformable attention.)
  - **`LwDetrMLP`** [compute]: `L2/vit_encoder_mlp.py` (fc1 -> ReLU -> dropout -> fc2 -> dropout + residual; two-layer MLP.)
  - **`LwDetrDecoderLayer`** [wiring]: Wiring: self-attn + deformable cross-attn + MLP + LayerNorms.
  - **`LwDetrDecoder`** [wiring]: Wiring; iterative bbox refinement loop.
  - **`LwDetrModel`** [wiring]: Wiring.
  - **`LwDetrForObjectDetection`** [wiring]: Wiring + classification/bbox heads.

## mask2former
- **src**: modeling_mask2former.py
- **status**: partial
- **rationale**: Pixel encoder is loaded via AutoBackbone (load_backbone) — kb-nano has no AutoBackbone/timm equivalent. Decoder also uses nn.MultiheadAttention for cross-attn.
- **classes**:
  - **`Mask2FormerAttention`** [compute]: no kb-nano kernel — Pixel encoder is loaded via AutoBackbone (load_backbone) — kb-nano has no AutoBackbone/timm equivalent. Decoder also uses nn.MultiheadAttention for cross-attn.
  - **`Mask2FormerSinePositionEmbedding`** [wiring]: Sinusoidal position embedding from coordinates; pure torch math (sin/cos).
  - **`Mask2FormerPixelDecoderEncoderMultiscaleDeformableAttention`** [compute]: `L1/rtdetrv2_deformable_attention.py`, `L2/rtdetrv2_deformable_attention.py` (Multi-scale deformable attention from Deformable DETR (modeling_mask2former.py:888); same sampling math as kb-nano L2/rtdetrv2_deformable_attention.py.)
  - **`Mask2FormerPixelLevelModule`** [wiring]: Loads backbone via load_backbone(config) (modeling_mask2former.py:1402) — AutoBackbone path. No kb-nano equivalent.
  - **`Mask2FormerMaskedAttentionDecoderLayer`** [wiring]: Uses nn.MultiheadAttention for cross-attn (modeling_mask2former.py:1585) — would need torch.nn fallback.
  - **`Mask2FormerMaskPredictor`** [wiring]: MLP head (linear + GELU) wiring; primitives exist.

## metaclip_2
- **src**: modular_metaclip_2.py
- **status**: composable
- **rationale**: Pure CLIP inheritance (CLIPAttention / CLIPMLP / CLIPVisionEmbeddings / CLIPTextEmbeddings / CLIPModel / CLIP*Projection). No kernel changes; quickgelu activation supported in kb-nano via L1/quickgelu.py.
- **classes**:
  - **`MetaClip2TextEmbeddings`** [compute]: `L2/encoder_embeddings.py` (CLIPTextEmbeddings: token_emb + position_emb.)
  - **`MetaClip2VisionEmbeddings`** [compute]: `L1/conv2d.py`, `L1/embedding.py` (patch conv + class_embedding + learned position_embedding.)
  - **`MetaClip2Attention`** [compute]: `L2/clip_attention.py` (CLIPAttention — Q/K/V/out projections, no RoPE.)
  - **`MetaClip2MLP`** [compute]: `L2/clip_mlp.py` (CLIPMLP: fc1 -> activation -> fc2 (with bias). Quickgelu supported (L1/quickgelu.py).)
  - **`MetaClip2TextModel`** [compute]: `L4/clip_text_model.py` (Inherits CLIPTextModel; same pre/post LayerNorm + EOS-token pooling.)
  - **`MetaClip2VisionModel`** [wiring]: Wiring (inherits CLIPVisionModel).
  - **`MetaClip2Model`** [wiring]: Wiring + text/visual projection + logit_scale.
  - **`MetaClip2TextModelWithProjection`** [wiring]: Wiring.
  - **`MetaClip2VisionModelWithProjection`** [wiring]: Wiring.
  - **`MetaClip2ForImageClassification`** [wiring]: Wiring + classifier head.

## minimax_m2
- **src**: modular_minimax_m2.py
- **status**: composable
- **rationale**: FlexOlmo (Llama-style) attention without bias + Mixtral-derived MoE with sigmoid + e_score_correction_bias topk routing. The deepseek_moe-with-n_group=1 path covers sigmoid+bias topk; mixtral_moe covers the underlying SparseMoeBlock structure. Standard rotary + RMSNorm.
- **classes**:
  - **`MiniMaxM2TopKRouter`** [compute]: `L1/sigmoid_topk.py` (sigmoid(router_logits) + e_score_correction_bias -> topk -> renormalize. Same shape as kb-nano sigmoid_topk + bias addition.)
  - **`MiniMaxM2Experts`** [compute]: `L1/moe_grouped_gemm.py` (MixtralExperts: per-token gather, gate_up + down (SwiGLU).)
  - **`MiniMaxM2SparseMoeBlock`** [compute]: `L2/mixtral_moe.py`, `L2/deepseek_moe.py` (Mixtral-style block with e_score_correction_bias added at routing — combines patterns from mixtral_moe (block layout) and deepseek_moe (sigmoid+bias topk).)
  - **`MiniMaxM2RMSNorm`** [compute]: `L1/rms_norm.py` (MixtralRMSNorm = standard RMSNorm.)
  - **`MiniMaxM2RotaryEmbedding`** [compute]: `L1/rotary_emb.py` (Glm4MoeRotaryEmbedding = standard NeoX RoPE.)
  - **`MiniMaxM2Attention`** [compute]: `L2/attention.py` (FlexOlmoAttention (Llama-style GQA) with bias=False on Q/K/V/O. attention.py covers this directly.)
  - **`MiniMaxM2Model`** [wiring]: Wiring; full attention only (no sliding window).
  - **`MiniMaxM2ForCausalLM`** [wiring]: Wiring (inherits MixtralForCausalLM).

## ministral
- **src**: modular_ministral.py
- **status**: composable
- **rationale**: Qwen2-derived (which is Llama-family) with Mistral-style bias=False on Q/K/V/O and per-layer sliding/full attention. All primitives in kb-nano: rotary_emb, rms_norm, llama_mlp (SwiGLU), attention.py (sliding window supported via softmax mask).
- **classes**:
  - **`MinistralMLP`** [compute]: `L2/llama_mlp.py` (Qwen2MLP — gate*up SwiGLU with SiLU; matches L2/llama_mlp.py + L1/silu_and_mul.py.)
  - **`MinistralAttention`** [compute]: `L2/attention.py` (Llama-family GQA with optional sliding window per layer; q/k/v/o bias=False. attention.py supports per-layer sliding via attention mask.)
  - **`MinistralRMSNorm`** [compute]: `L1/rms_norm.py` (Standard RMSNorm.)
  - **`MinistralRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (Standard NeoX RoPE.)
  - **`MinistralDecoderLayer`** [wiring]: Wiring (inherits Qwen2DecoderLayer).
  - **`MinistralModel`** [wiring]: Wiring; per-layer mask routing on layer_types.
  - **`MinistralForCausalLM`** [wiring]: Wiring.

## ministral3
- **src**: modular_ministral3.py
- **status**: composable
- **rationale**: Mistral attention augmented with Llama-4-style per-position attention temperature scaling on Q. kb-nano L2/llama4_attention.py already implements that exact `1 + beta*log(1 + floor(pos/max))` Q-scaling shape.
- **classes**:
  - **`Ministral3Attention`** [compute]: `L2/llama4_attention.py` (RoPE -> Q-scale by 1 + beta*log(1 + floor(pos / orig_max_pos)) -> attention. Maps to Llama4Attention with attn_temperature_tuning=True (kb-nano Llama4Attention._get_attn_scale uses log(floor+1)*scale + 1).)
  - **`Ministral3DecoderLayer`** [wiring]: Wiring (inherits MistralDecoderLayer).
  - **`Ministral3Model`** [wiring]: Wiring (inherits MistralModel).
  - **`Ministral3ForCausalLM`** [wiring]: Wiring.

## mistral4
- **src**: modular_mistral4.py
- **status**: composable
- **rationale**: DeepseekV3-style MLA attention (composes to L2/deepseek_mla_attention.py) + DeepseekV3-style grouped-topk MoE (deepseek_moe.py) + a per-position llama_4_scaling on Q (elementwise mul, primitive available).
- **classes**:
  - **`Mistral4RMSNorm`** [compute]: `L1/rms_norm.py` (Inherits LlamaRMSNorm; standard RMSNorm.)
  - **`Mistral4RotaryEmbedding`** [compute]: `L1/rotary_emb.py` (Inherits LlamaRotaryEmbedding (interleaved variant supported).)
  - **`Mistral4MLP`** [compute]: `L2/llama_mlp.py` (Inherits Qwen2MoeMLP (gate_proj/up_proj/down_proj with SiLU) — SwiGLU gate*up pattern.)
  - **`Mistral4TopkRouter`** [compute]: `L1/grouped_topk.py` (Linear classifier producing router logits — used downstream by grouped-topk in MoE.)
  - **`Mistral4NaiveMoe`** [wiring]: Inherits DeepseekV3NaiveMoe; reference path.
  - **`Mistral4MoE`** [compute]: `L2/deepseek_moe.py` (Group-topk routing with softmax + n_group/topk_group + routed_scaling_factor. Same algorithm as DeepseekV3 grouped-topk; deepseek_moe.py implements it.)
  - **`Mistral4Attention`** [compute]: `L2/deepseek_mla_attention.py` (MLA with q-LoRA + kv-LoRA + interleaved RoPE on rotary half + Llama4-style attention temperature scaling on Q (elementwise mul). All primitives present; the temp scaling is a torch elementwise mul on top of MLA.)
  - **`Mistral4DecoderLayer`** [wiring]: Wiring: dense MLP for first layers, MoE thereafter; RMSNorm pre-attn / pre-MLP.
  - **`Mistral4Model`** [wiring]: Wiring (inherits LlamaModel).
  - **`Mistral4ForCausalLM`** [wiring]: Wiring (inherits LlamaForCausalLM).

## mlcd
- **src**: modular_mlcd.py
- **status**: composable
- **rationale**: CLIP-Vision encoder with 2D vision RoPE (Qwen2-VL-style apply_rotary_pos_emb_vision) replacing learnable position embeddings. Bidirectional CLIP attention + LayerNorm + GELU MLP. All primitives present.
- **classes**:
  - **`MLCDMLP`** [compute]: `L2/clip_mlp.py` (CLIPMLP: fc1 -> activation -> fc2 with bias.)
  - **`MLCDRotaryEmbedding`** [compute]: `L1/vision_rotary_emb.py` (Inherits VisionRotaryEmbedding from Qwen2-VL; computes 2D (h,w) RoPE via outer product. vision_rotary_emb.py covers the 2D NeoX-style RoPE.)
  - **`MLCDVisionEmbeddings`** [compute]: `L1/conv2d.py` (Patch conv + class token; learnable position embedding deleted (RoPE used instead).)
  - **`MLCDAttention`** [compute]: `L2/clip_attention.py` (CLIPAttention with vision RoPE applied to Q/K. Bidirectional, uses sdpa fallback. clip_attention.py covers the projection layout; RoPE step uses vision_rotary_emb.py.)
  - **`MLCDEncoderLayer`** [wiring]: Wiring: pre-norm attention + pre-norm MLP.
  - **`MLCDEncoder`** [wiring]: Wiring: stack of encoder layers.
  - **`MLCDVisionModel`** [wiring]: Wiring + class_pos_emb prepended to per-patch RoPE freqs + pre/post LayerNorm + cls-token pooling.

## mm_grounding_dino
- **src**: modular_mm_grounding_dino.py
- **status**: partial
- **rationale**: Inherits Grounding DINO conv encoder which calls load_backbone(config) for the (Swin) image backbone, plus loads BERT text backbone via AutoModel.from_config. AutoBackbone has no kb-nano equivalent.
- **classes**:
  - **`MMGroundingDinoConvEncoder`** [compute]: Inherits Grounding DINO conv encoder which calls load_backbone(config) for the (Swin) image backbone, plus loads BERT text backbone via AutoModel.from_config. AutoBackbone has no kb-nano equivalent.
  - **`MMGroundingDinoContrastiveEmbedding`** [wiring]: Vision-text dot-product + bias + masking + padding to max_text_len; pure torch ops.
  - **`MMGroundingDinoConvModel`** [wiring]: Wiring.
  - **`MMGroundingDinoEncoder`** [wiring]: Inherits GroundingDinoEncoder (deformable attention + text enhancer + cross-modal fusion).
  - **`MMGroundingDinoDecoder`** [wiring]: Inherits GroundingDinoDecoder.
  - **`MMGroundingDinoModel`** [wiring]: Wiring; loads text backbone via AutoModel.from_config (modular_mm_grounding_dino.py:239) and image backbone via load_backbone.
  - **`MMGroundingDinoForObjectDetection`** [wiring]: Wiring + class/bbox heads.

## musicflamingo
- **src**: modular_musicflamingo.py
- **status**: partial
- **partial_reason**: configuration_musicflamingo.py sets `partial_rotary_factor = 0.2` for the Qwen2 LM rope_parameters; standard L1/rotary_emb rotates the full head_dim, so the LM path needs external q_rot/q_pass slicing or Gemma4-style proportional rotary. Custom MusicFlamingoRotaryEmbedding (axial 2D over window+time) is itself wiring (pure torch composition), but the LM partial-rotary gap matches phi/persimmon/glm.
- **rationale**: AudioFlamingo3 wrapper (Whisper-derived audio encoder + multi-modal projector + Qwen2 LM with partial-rotary 0.2) with a custom MusicFlamingoRotaryEmbedding (axial 2D rotary over window+time, time-modulated by absolute timestamps) applied to audio encoder output. Audio encoder + projector + LM compute primitives map to existing kernels; LM partial-rotary needs external slicing.
- **classes**:
  - **`MusicFlamingoRotaryEmbedding`** [wiring]: Bespoke 2D rotary time embedding (axial window x time freqs modulated by absolute timestamps). Pure torch composition (arange / outer / cat / repeat_interleave / cos / sin); no kb-nano L1 kernel needed beyond elementwise cos/sin.
  - **`MusicFlamingoForConditionalGeneration`** [wiring]: Wiring: builds audio_timestamps; calls audio_tower (AudioFlamingo3 = Whisper-derived encoder); applies apply_rotary_time_emb; multi_modal_projector; replaces audio token placeholders in inputs_embeds; calls language_model (Qwen2). Underlying engines: L2/whisper_attention.py + L2/whisper_mlp.py for audio encoder; L2/attention.py + L2/llama_mlp.py + L1/rotary_emb.py + L1/rms_norm.py for Qwen2 LM.

## rag
- **src**: modeling_rag.py
- **status**: composable
- **rationale**: Pure wrapper that delegates to AutoModel for question encoder (DPR/BERT-family) and AutoModelForSeq2SeqLM for the generator (BART/T5-family). RagRetriever is a non-NN component (FAISS/index lookup). All compute is in delegated models; both DPR/BERT (encoder_attention) and BART/T5 (whisper_attention/t5_attention) are composable in kb-nano.
- **classes**:
  - **`RagModel`** [wiring]: Wiring: question_encoder (AutoModel) + generator (AutoModelForSeq2SeqLM) + retriever; combines retrieved docs with input via simple torch concat/repeat.
  - **`RagSequenceForGeneration`** [wiring]: Wiring: marginalisation over docs (logsumexp), generation.
  - **`RagTokenForGeneration`** [wiring]: Wiring: token-level marginalisation.

## shieldgemma2
- **src**: modeling_shieldgemma2.py
- **status**: composable
- **rationale**: Pure wrapper class around AutoModelForImageTextToText (a Gemma3-derived image-text-to-text model). All compute is delegated; on top, only torch ops (slice last position, gather Yes/No logit indices, softmax). Underlying Gemma3 stack is composable in kb-nano (gemma_rms_norm, attention.py with sliding window, llama_mlp, rotary).
- **classes**:
  - **`ShieldGemma2ForImageClassification`** [wiring]: Wraps AutoModelForImageTextToText.from_config(config); takes last position logits and the [yes_token_index, no_token_index] columns, then softmax. Pure delegation + torch slicing/softmax.

## donut_swin
- **src**: modeling_donut_swin.py
- **status**: partial
- **partial_reason**: Swin V1 windowed attention uses relative_position_bias_table lookup (additive bias on attention scores). kb-nano L2/swinv2_window_attention.py is V2-only (cosine attention + CPB MLP) — different math.
- **rationale**: Donut visual encoder reuses Swin V1 attention (window_partition + relative_position_bias_table). kb-nano lacks a Swin V1 wrapper but the underlying primitives (linear, softmax, attn-mask trick) all exist in L1.
- **classes**:
  - **`DonutSwinSelfAttention`** [compute]: no kb-nano kernel — Swin V1 relative_position_bias_table not in kb-nano L2/swinv2_window_attention.py (which is V2 only)
  - **`DonutSwinAttention`** [wiring]: wires self + output
  - **`DonutSwinIntermediate`** [compute]: `L1/linear.py`, `L1/gelu.py`
  - **`DonutSwinOutput`** [compute]: `L1/linear.py`, `L1/dropout.py`
  - **`DonutSwinLayer`** [wiring]: wires above
  - **`DonutSwinPatchEmbeddings`** [compute]: `L1/conv2d.py`
  - **`DonutSwinModel`** [wiring]: full encoder

## esmfold
- **src**: modeling_esmfold.py
- **status**: partial
- **partial_reason**: EsmFoldTriangleAttention is AlphaFold-style triangular attention (no kb-nano L2 wrapper). The compute decomposes from torch primitives (matmul + softmax + tensor reshape) but no kb-nano kernel implements the triangular pattern.
- **rationale**: ESMFold is the protein-folding head on top of ESM-2. Uses triangle multiplication and triangle attention (AlphaFold-style). kb-nano has standard SDPA but no triangular attention or triangle multiplication primitives.
- **classes**:
  - **`EsmFoldTriangleAttention`** [compute]: no kb-nano kernel — triangular attention pattern (q_x_attended = einsum) decomposes from primitives but no L2 wrapper
  - **`EsmFoldAttention`** [compute]: `L1/linear.py`, `L1/dense_attention.py` (standard MHA pattern)
  - **`EsmFoldSelfAttention`** [compute]: `L1/linear.py`, `L1/dense_attention.py`
  - **`EsmFoldStructureModule`** [wiring]: full IPA + frame update; no kb-nano IPA kernel
  - **`EsmFoldTriangularMultiplicativeUpdate`** [compute]: no kb-nano kernel — triangle multiplication (a × b × c) for protein structure
