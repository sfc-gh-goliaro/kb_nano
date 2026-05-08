## phimoe
- **src**: modeling_phimoe.py, modular_phimoe.py
- **hidden_act**: silu
- **status**: composable
- **classes**:
  - **`PhimoeRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (default RoPE plus scaling factor mscale; standard cos/sin builder, ROPE_INIT_FUNCTIONS used for non-default rope_type)
  - **`PhimoeAttention`** [compute]: `L2/attention.py` (q/k/v/o Linear with optional bias, RoPE, KV cache update, ALL_ATTENTION_FUNCTIONS dispatch — Llama-family pattern)
  - **`PhimoeExperts`** [compute]: `L1/moe_grouped_gemm.py` (3D-tensor experts: gate_up_proj split into gate+up, silu(gate)*up, down_proj, scatter weighted)
  - **`PhimoeTopKRouter`** [compute, inherits `nn.Linear`]: `L1/linear.py + L1/sigmoid_topk.py` (sparsemixer with two top-k sub-rounds and softmax-gated multipliers — no exact L2 match; sparsemixer not implemented in kb-nano)
  - **`PhimoeSparseMoeBlock`** [wiring]: wires `PhimoeTopKRouter`, `PhimoeExperts` — composite MoE (closest match `L2/mixtral_moe.py` but uses sparsemixer routing; no exact L2 match)
  - **`PhimoeDecoderLayer`** [wiring]: wires `PhimoeAttention`, `PhimoeSparseMoeBlock`; direct `L1/layer_norm.py` (input_layernorm and post_attention_layernorm are nn.LayerNorm, not RMSNorm — note: rms_norm_eps used as eps but the module is LayerNorm)
  - **`PhimoeModel`** [wiring]: wires `PhimoeDecoderLayer`, `PhimoeRotaryEmbedding`; direct `L1/embedding.py` (embed_tokens), `L1/layer_norm.py` (final norm)
  - **`PhimoeForCausalLM`** [wiring]: wires `PhimoeModel`; direct `L1/linear.py` (lm_head)
- **task heads (1)**: ForSequenceClassification — base + linear (per-task)

## pi0
- **src**: modeling_pi0.py, modular_pi0.py
- **hidden_act**: silu (uses F.silu inline; vlm and dit have own configs)
- **status**: composable
- **classes**:
  - **`PI0TimestepEmbeddings`** [compute]: `L1/sinusoidal_embed.py` (computes sinusoidal time embedding from min/max period; cos/sin concat)
  - **`PI0ActionTimeEmbedding`** [compute]: `L2/pi0_action_embed.py` (state_proj + action_in_proj + sinusoid timestep + action_time_mlp_in/out with SiLU — direct match)
  - **`PI0Model`** [wiring]: wires `AutoModel.from_config(dit_config)` and `AutoModel.from_config(vlm_config)` (DiT + VLM submodels; configurable, typically Gemma2/Pi0DiT)
  - **`PI0ForConditionalGeneration`** [wiring]: wires `PI0Model`, `PI0ActionTimeEmbedding`; direct `L1/linear.py` (action_out_proj). Flow-matching loop in sample_actions uses `L4/pi0.py` engine.

## pix2struct
- **src**: modeling_pix2struct.py
- **hidden_act**: gelu_new (dense_act_fn for both vision and text)
- **status**: composable
- **classes**:
  - **`Pix2StructLayerNorm`** [compute]: `L1/t5_layer_norm.py` (T5-style RMSNorm: variance over hidden, scale only, no bias)
  - **`Pix2StructVisionEmbeddings`** [compute]: `L1/linear.py + L1/embedding.py + L1/embedding.py` (patch_projection Linear + row_embedder Embedding + column_embedder Embedding; no exact L2 match)
  - **`Pix2StructVisionAttention`** [compute]: `L2/t5_attention.py` (QKV/output Linears no bias, position_bias added pre-softmax — T5-style attention without rel-pos-bias as init param; non-causal here)
  - **`Pix2StructVisionMlp`** [compute]: `L2/t5_dense.py` (T5DenseGatedActDense pattern: wi_0/wi_1 + gated act * up + wo, with gelu_new)
  - **`Pix2StructVisionLayer`** [wiring]: wires `Pix2StructVisionAttention`, `Pix2StructVisionMlp`, `Pix2StructLayerNorm` (×2: pre_attention and pre_mlp)
  - **`Pix2StructVisionEncoder`** [wiring]: wires `Pix2StructVisionLayer`
  - **`Pix2StructVisionModel`** [wiring]: wires `Pix2StructVisionEmbeddings`, `Pix2StructVisionEncoder`, `Pix2StructLayerNorm`
  - **`Pix2StructTextDenseGatedActDense`** [compute]: `L2/t5_dense.py` (T5 gated dense, identical to vision MLP)
  - **`Pix2StructTextLayerFF`** [wiring]: wires `Pix2StructTextDenseGatedActDense`, `Pix2StructLayerNorm`
  - **`Pix2StructTextAttention`** [compute]: `L2/t5_attention.py` (T5 attention with relative_attention_bias on first layer; supports cross-attention via key_value_states, causal in decoder)
  - **`Pix2StructTextLayerSelfAttention`** [wiring]: wires `Pix2StructTextAttention`, `Pix2StructLayerNorm`
  - **`Pix2StructTextLayerCrossAttention`** [wiring]: wires `Pix2StructTextAttention` (no rel-pos-bias), `Pix2StructLayerNorm`
  - **`Pix2StructTextBlock`** [wiring]: wires `Pix2StructTextLayerSelfAttention`, `Pix2StructTextLayerCrossAttention`, `Pix2StructTextLayerFF`
  - **`Pix2StructTextModel`** [wiring]: wires `Pix2StructTextBlock`, `Pix2StructLayerNorm`; direct `L1/embedding.py` (embed_tokens), `L1/linear.py` (lm_head)
  - **`Pix2StructForConditionalGeneration`** [wiring]: wires `Pix2StructVisionModel`, `Pix2StructTextModel`

## pixio
- **src**: modeling_pixio.py, modular_pixio.py (inherits from Dinov2/ViT)
- **hidden_act**: gelu
- **status**: composable
- **classes**:
  - **`PixioPatchEmbeddings`** [compute, inherits `ViTPatchEmbeddings`]: `L1/conv2d.py` (single Conv2d projection, flatten to seq)
  - **`PixioEmbeddings`** [compute]: wires `PixioPatchEmbeddings`; direct cls token + position embedding parameters with bicubic interpolation (no exact L2 match — closest is `L2/vision_patch_embed.py`)
  - **`PixioSelfAttention`** [compute, inherits `ViTSelfAttention`]: `L1/linear.py + L1/dense_attention.py` (q/k/v Linear + ALL_ATTENTION_FUNCTIONS dispatch, non-causal, no RoPE/KV cache — no exact L2 match)
  - **`PixioSelfOutput`** [compute]: `L1/linear.py` (dense + dropout, residual external — no exact L2 match)
  - **`PixioAttention`** [wiring, inherits `ViTAttention`]: wires `PixioSelfAttention`, `PixioSelfOutput`
  - **`PixioDropPath`** [compute, inherits `Dinov2DropPath`]: `L1/dropout.py` (stochastic depth)
  - **`PixioMLP`** [compute, inherits `Dinov2MLP`]: `L1/linear.py + L1/gelu.py + L1/linear.py` (fc1 + gelu + fc2; closest L2 is `L2/clip_mlp.py` but activation is gelu not quickgelu — closer: `L2/whisper_mlp.py` or new vit-style mlp)
  - **`PixioLayer`** [wiring]: wires `PixioAttention`, `PixioMLP`, `PixioDropPath`; direct `L1/layer_norm.py` (norm1, norm2)
  - **`PixioEncoder`** [wiring]: wires `PixioLayer`
  - **`PixioModel`** [wiring]: wires `PixioEmbeddings`, `PixioEncoder`; direct `L1/layer_norm.py`
  - **`PixioBackbone`** [wiring, inherits `Dinov2Backbone`]: wires `PixioEmbeddings`, `PixioEncoder`; direct `L1/layer_norm.py`

## pixtral
- **src**: modeling_pixtral.py
- **hidden_act**: gelu (vision; default)
- **status**: composable
- **classes**:
  - **`PixtralRotaryEmbedding`** [compute]: `L1/vision_rotary_emb.py` (2D vision RoPE: outer product of h-freqs and w-freqs over patch grid — non-default rope_type rejected)
  - **`PixtralAttention`** [compute]: `L1/linear.py + L1/dense_attention.py` (q/k/v/o Linear no bias, RoPE applied to q/k, ALL_ATTENTION_FUNCTIONS — non-causal vision attention, no KV cache; closest L2 is `L2/siglip_attention.py` but with RoPE — no exact L2 match)
  - **`PixtralMLP`** [compute]: `L2/llama_mlp.py` (SwiGLU: gate_proj + up_proj + act_fn(gate)*up + down_proj — uses gelu via ACT2FN[hidden_act])
  - **`PixtralRMSNorm`** [compute]: `L1/rms_norm.py` (Llama-style RMSNorm)
  - **`PixtralAttentionLayer`** [wiring]: wires `PixtralRMSNorm` (×2), `PixtralAttention`, `PixtralMLP`
  - **`PixtralTransformer`** [wiring]: wires `PixtralAttentionLayer`
  - **`PixtralVisionModel`** [wiring]: wires `PixtralRotaryEmbedding`, `PixtralRMSNorm` (ln_pre), `PixtralTransformer`; direct `L1/conv2d.py` (patch_conv)

## plbart
- **src**: modeling_plbart.py, modular_plbart.py
- **hidden_act**: gelu (activation_function default)
- **status**: composable
- **classes**:
  - **`PLBartScaledWordEmbedding`** [compute, inherits `nn.Embedding`]: `L1/embedding.py` (embedding then scale by sqrt(d_model))
  - **`PLBartLearnedPositionalEmbedding`** [compute, inherits `nn.Embedding`]: `L1/embedding.py` (offset positional embedding)
  - **`PLBartAttention`** [compute]: `L1/linear.py + L1/dense_attention.py + L1/store_kvcache.py` (Bart-style attention with self/cross/causal modes; uses ALL_ATTENTION_FUNCTIONS; closest L2 is `L2/whisper_attention.py` — no exact L2 match for plbart)
  - **`PLBartEncoderLayer`** [wiring]: wires `PLBartAttention`; direct `L1/linear.py` (fc1, fc2), `L1/gelu.py` (activation_fn), `L1/layer_norm.py` (self_attn_layer_norm, final_layer_norm)
  - **`PLBartEncoder`** [wiring]: wires `PLBartScaledWordEmbedding`, `PLBartLearnedPositionalEmbedding`, `PLBartEncoderLayer`; direct `L1/layer_norm.py` (layernorm_embedding)
  - **`PLBartDecoderLayer`** [wiring]: wires `PLBartAttention` (×2: self_attn causal, encoder_attn cross); direct `L1/linear.py` (fc1, fc2), `L1/gelu.py`, `L1/layer_norm.py` (×3)
  - **`PLBartDecoder`** [wiring]: wires `PLBartScaledWordEmbedding`, `PLBartLearnedPositionalEmbedding`, `PLBartDecoderLayer`; direct `L1/layer_norm.py`
  - **`PLBartModel`** [wiring]: wires `PLBartScaledWordEmbedding` (shared), `PLBartEncoder`, `PLBartDecoder`
  - **`PLBartForConditionalGeneration`** [wiring]: wires `PLBartModel`; direct `L1/linear.py` (lm_head)
  - **`PLBartClassificationHead`** [compute]: `L1/linear.py + L1/tanh.py + L1/linear.py` (dense + tanh + out_proj)
  - **`PLBartDecoderWrapper`** [wiring]: wires `PLBartDecoder`
  - **`PLBartForCausalLM`** [wiring]: wires `PLBartDecoderWrapper`; direct `L1/linear.py` (lm_head)
- **task heads (1)**: ForSequenceClassification — base + linear (per-task)

## poolformer
- **src**: modeling_poolformer.py
- **hidden_act**: gelu
- **status**: composable
- **classes**:
  - **`PoolFormerDropPath`** [compute]: `L1/dropout.py` (stochastic depth)
  - **`PoolFormerEmbeddings`** [compute]: `L1/conv2d.py` (Conv2d projection; optional norm — usually nn.Identity)
  - **`PoolFormerGroupNorm`** [compute, inherits `nn.GroupNorm`]: `L1/group_norm.py` (single-group GN)
  - **`PoolFormerPooling`** [compute]: `L1/avg_pool2d.py` (AvgPool2d - identity, the "MetaFormer" token mixer; no exact L2 match — fits primitive composition)
  - **`PoolFormerOutput`** [compute]: `L1/conv2d.py + L1/gelu.py + L1/conv2d.py` (1×1 conv + act + 1×1 conv with drop path; channel-first MLP-via-conv variant)
  - **`PoolFormerLayer`** [wiring]: wires `PoolFormerPooling`, `PoolFormerOutput`, `PoolFormerGroupNorm` (×2: before_norm, after_norm), `PoolFormerDropPath`; direct `nn.Parameter` layer_scale_1/2 (channel-wise scaling, no kernel needed)
  - **`PoolFormerEncoder`** [wiring]: wires `PoolFormerEmbeddings`, `PoolFormerLayer`
  - **`PoolFormerFinalPooler`** [compute]: `L1/linear.py` (single dense)
  - **`PoolFormerModel`** [wiring]: wires `PoolFormerEncoder`
- **task heads (1)**: ForImageClassification — base + linear (per-task)

## pop2piano
- **src**: modeling_pop2piano.py
- **hidden_act**: relu (dense_act_fn) — gated path uses relu via ACT2FN; feed_forward_proj defaults to "gated-gelu" but ACT2FN uses dense_act_fn
- **status**: composable
- **classes**:
  - **`Pop2PianoLayerNorm`** [compute]: `L1/t5_layer_norm.py` (T5-style RMSNorm)
  - **`Pop2PianoDenseActDense`** [compute]: `L1/linear.py + L1/relu.py + L1/linear.py` (non-gated: wi + relu + wo; closest L2 is `L2/encoder_mlp.py` pattern but no LayerNorm)
  - **`Pop2PianoDenseGatedActDense`** [compute]: `L2/t5_dense.py` (T5 gated dense: wi_0/wi_1 + gated_act * linear + wo)
  - **`Pop2PianoLayerFF`** [wiring]: wires `Pop2PianoDenseActDense` or `Pop2PianoDenseGatedActDense`, `Pop2PianoLayerNorm`
  - **`Pop2PianoAttention`** [compute]: `L2/t5_attention.py` (T5 attention with relative position bias, supports cross-attention)
  - **`Pop2PianoLayerSelfAttention`** [wiring]: wires `Pop2PianoAttention`, `Pop2PianoLayerNorm`
  - **`Pop2PianoLayerCrossAttention`** [wiring]: wires `Pop2PianoAttention` (no rel-pos-bias), `Pop2PianoLayerNorm`
  - **`Pop2PianoBlock`** [wiring]: wires `Pop2PianoLayerSelfAttention`, optional `Pop2PianoLayerCrossAttention` (decoder), `Pop2PianoLayerFF`
  - **`Pop2PianoStack`** [wiring]: wires `Pop2PianoBlock`, `Pop2PianoLayerNorm`; direct `L1/embedding.py` (embed_tokens)
  - **`Pop2PianoConcatEmbeddingToMel`** [compute]: `L1/embedding.py` (composer-id embedding then concat to feature)
  - **`Pop2PianoForConditionalGeneration`** [wiring]: wires `Pop2PianoStack` (encoder & decoder), `Pop2PianoConcatEmbeddingToMel`; direct `L1/embedding.py` (shared), `L1/linear.py` (lm_head)

## pp_doclayout_v2
- **src**: modeling_pp_doclayout_v2.py, modular_pp_doclayout_v2.py (inherits LayoutLMv3 + RT-DETR)
- **hidden_act**: gelu
- **status**: partial (combines LayoutLMv3 reading-order encoder with RT-DETR detector; kb-nano has rtdetrv2 building blocks but not LayoutLMv3 encoder ops or the GlobalPointer/PositionRelationEmbedding pieces)
- **classes**:
  - **`PPDocLayoutV2GlobalPointer`** [compute]: `L1/linear.py` (custom span head with rotary-style position-aware logits — no exact L2 match)
  - **`PPDocLayoutV2PositionRelationEmbedding`** [compute]: `L1/linear.py + L1/embedding.py` (pairwise relation features; no exact L2 match)
  - **`PPDocLayoutV2ReadingOrderSelfAttention`** [compute, inherits `LayoutLMv3SelfAttention`]: `L2/encoder_attention.py` (BERT-style q/k/v with extra spatial/relative bias — no exact L2 match for the bias variants)
  - **`PPDocLayoutV2ReadingOrderSelfOutput`** [compute, inherits `LayoutLMv3SelfOutput`]: `L2/encoder_attention.py` (dense + LayerNorm + residual portion)
  - **`PPDocLayoutV2ReadingOrderIntermediate`** [compute, inherits `LayoutLMv3Intermediate`]: `L1/linear.py + L1/gelu.py`
  - **`PPDocLayoutV2ReadingOrderOutput`** [compute, inherits `LayoutLMv3Output`]: `L1/linear.py + L1/layer_norm.py`
  - **`PPDocLayoutV2ReadingOrderAttention`** [wiring, inherits `LayoutLMv3Attention`]: wires `PPDocLayoutV2ReadingOrderSelfAttention`, `PPDocLayoutV2ReadingOrderSelfOutput`
  - **`PPDocLayoutV2ReadingOrderLayer`** [wiring, inherits `LayoutLMv3Layer`]: wires `PPDocLayoutV2ReadingOrderAttention`, `PPDocLayoutV2ReadingOrderIntermediate`, `PPDocLayoutV2ReadingOrderOutput`
  - **`PPDocLayoutV2ReadingOrderEncoder`** [wiring, inherits `LayoutLMv3Encoder`]: wires `PPDocLayoutV2ReadingOrderLayer`
  - **`PPDocLayoutV2TextEmbeddings`** [compute, inherits `LayoutLMv3TextEmbeddings`]: `L2/encoder_embeddings.py` (word + position + token_type + 2D bbox embeddings)
  - **`MultiScaleDeformableAttention`** [compute]: `L1/rtdetrv2_deformable_attention.py` (multi-scale deformable attention primitive)
  - **`PPDocLayoutV2MultiscaleDeformableAttention`** [compute]: `L2/rtdetrv2_deformable_attention.py` (wrapper class with sampling offsets, attention weights)
  - **`PPDocLayoutV2ReadingOrder`** [wiring]: wires reading-order encoder + global pointer head
  - **`PPDocLayoutV2MLPPredictionHead`** [compute, inherits `RTDetrMLPPredictionHead`]: `L2/rtdetrv2_mlp_head.py`
  - **`PPDocLayoutV2MLP`** [compute]: `L1/linear.py + L1/gelu.py + L1/linear.py` (2-layer)
  - **`PPDocLayoutV2FrozenBatchNorm2d`** [compute]: `L1/frozen_batch_norm2d.py`
  - **`PPDocLayoutV2SelfAttention`** [compute]: `L1/linear.py + L1/dense_attention.py` (multi-head attention block in encoder/decoder)
  - **`PPDocLayoutV2ConvEncoder`** [wiring]: wires `PPDocLayoutV2FrozenBatchNorm2d`, backbone via `consolidate_backbone_kwargs_to_config`
  - **`PPDocLayoutV2ConvNormLayer`** [compute]: `L2/rtdetrv2_conv_norm.py`
  - **`PPDocLayoutV2EncoderLayer`** [wiring, inherits patterns from RTDetr]: wires `PPDocLayoutV2SelfAttention`, MLP — close to `L2/rtdetrv2_encoder_layer.py`
  - **`PPDocLayoutV2RepVggBlock`** [compute]: `L2/rtdetrv2_repvgg_block.py`
  - **`PPDocLayoutV2CSPRepLayer`** [compute]: `L2/rtdetrv2_csp_rep_layer.py`
  - **`PPDocLayoutV2DecoderLayer`** [wiring]: wires `PPDocLayoutV2SelfAttention`, `PPDocLayoutV2MultiscaleDeformableAttention`, MLP — RT-DETR style
  - **`PPDocLayoutV2SinePositionEmbedding`** [compute]: `L1/sinusoidal_embed.py`
  - **`PPDocLayoutV2AIFILayer`** [wiring]: wires self-attention + ffn (encoder layer in hybrid encoder)
  - **`PPDocLayoutV2HybridEncoder`** [wiring, inherits RTDetr pattern]: wires AIFI + CSPRep stages
  - **`PPDocLayoutV2Decoder`** [wiring]: wires `PPDocLayoutV2DecoderLayer`
  - **`PPDocLayoutV2Model`** [wiring]: wires conv encoder + hybrid encoder + decoder + reading order head
- **task heads (1)**: ForObjectDetection — base + linear/MLP heads

## pp_doclayout_v3
- **src**: modeling_pp_doclayout_v3.py, modular_pp_doclayout_v3.py
- **hidden_act**: (no top-level hidden_act in main config; uses silu/relu in subcomponents — verify per submodule)
- **status**: partial (RT-DETR-style detector with mask FPN + mask head extensions; like v2 minus reading-order)
- **classes**:
  - **`PPDocLayoutV3GlobalPointer`** [compute]: `L1/linear.py` (no exact L2 match — same head as v2)
  - **`MultiScaleDeformableAttention`** [compute]: `L1/rtdetrv2_deformable_attention.py`
  - **`PPDocLayoutV3MultiscaleDeformableAttention`** [compute]: `L2/rtdetrv2_deformable_attention.py`
  - **`PPDocLayoutV3MLPPredictionHead`** [compute]: `L2/rtdetrv2_mlp_head.py`
  - **`PPDocLayoutV3ConvLayer`** [compute]: `L1/conv2d.py + L1/batch_norm2d.py + activation` (varies)
  - **`PPDocLayoutV3ScaleHead`** [compute]: `L1/conv2d.py` (small scale prediction head; no exact L2 match)
  - **`PPDocLayoutV3MaskFeatFPN`** [wiring]: wires conv layers and upsample (FPN-style mask features)
  - **`PPDocLayoutV3EncoderMaskOutput`** [compute]: `L1/conv2d.py` (mask logits via conv)
  - **`PPDocLayoutV3MLP`** [compute]: `L1/linear.py + L1/relu.py + L1/linear.py` (2-layer)
  - **`PPDocLayoutV3SelfAttention`** [compute]: `L1/linear.py + L1/dense_attention.py`
  - **`PPDocLayoutV3ConvNormLayer`** [compute]: `L2/rtdetrv2_conv_norm.py`
  - **`PPDocLayoutV3EncoderLayer`** [wiring]: wires self-attn + ffn
  - **`PPDocLayoutV3RepVggBlock`** [compute]: `L2/rtdetrv2_repvgg_block.py`
  - **`PPDocLayoutV3CSPRepLayer`** [compute]: `L2/rtdetrv2_csp_rep_layer.py`
  - **`PPDocLayoutV3SinePositionEmbedding`** [compute]: `L1/sinusoidal_embed.py`
  - **`PPDocLayoutV3AIFILayer`** [wiring]: wires self-attn + ffn
  - **`PPDocLayoutV3HybridEncoder`** [wiring]: wires AIFI + CSPRep stages
  - **`PPDocLayoutV3DecoderLayer`** [wiring]: wires self-attn + cross-deformable + MLP
  - **`PPDocLayoutV3Decoder`** [wiring]: wires `PPDocLayoutV3DecoderLayer`
  - **`PPDocLayoutV3FrozenBatchNorm2d`** [compute]: `L1/frozen_batch_norm2d.py`
  - **`PPDocLayoutV3ConvEncoder`** [wiring]: wires backbone via consolidate_backbone_kwargs
  - **`PPDocLayoutV3Model`** [wiring]: wires conv encoder + hybrid encoder + decoder + mask FPN
- **task heads (1)**: ForObjectDetection — base + linear/MLP heads

## pp_formulanet
- **src**: modeling_pp_formulanet.py, modular_pp_formulanet.py
- **hidden_act**: gelu
- **status**: composable
- **classes**:
  - **`PPFormulaNetVisionEncoderOutput`** [skip — output dataclass]
  - **`PPFormulaNetVisionAttention`** [compute]: `L1/linear.py + L1/dense_attention.py` (multi-head non-causal attention with separate qkv projections — closest L2: `L2/siglip_attention.py` if there's no causality)
  - **`PPFormulaNetMultiModalProjector`** [compute]: `L1/linear.py + L1/gelu.py + L1/linear.py` (2-layer projector)
  - **`PPFormulaNetMLPBlock`** [compute]: `L1/linear.py + L1/gelu.py + L1/linear.py` (encoder-style MLP — closest L2: `L2/whisper_mlp.py`)
  - **`PPFormulaNetVisionLayer`** [wiring]: wires `PPFormulaNetVisionAttention`, `PPFormulaNetMLPBlock`, `PPFormulaNetLayerNorm`
  - **`PPFormulaNetPatchEmbeddings`** [compute]: `L1/conv2d.py` (patch projection)
  - **`PPFormulaNetLayerNorm`** [compute, inherits `nn.LayerNorm`]: `L1/layer_norm.py`
  - **`PPFormulaNetVisionNeck`** [compute]: `L1/linear.py` (dimension projection neck)
  - **`PPFormulaNetVisionModel`** [wiring]: wires `PPFormulaNetPatchEmbeddings`, `PPFormulaNetVisionLayer`, `PPFormulaNetVisionNeck`
  - **`PPFormulaNetLearnedPositionalEmbedding`** [compute, inherits `nn.Embedding`]: `L1/embedding.py` (offset positional embedding, BART-style)
  - **`PPFormulaNetScaledWordEmbedding`** [compute, inherits `nn.Embedding`]: `L1/embedding.py` (scaled word embedding)
  - **`PPFormulaNetAttention`** [compute]: `L1/linear.py + L1/dense_attention.py + L1/store_kvcache.py` (Bart/decoder-style attention with self/cross modes)
  - **`PPFormulaNetDecoderLayer`** [wiring]: wires `PPFormulaNetAttention` (×2: self + cross); direct `L1/linear.py`, `L1/gelu.py`, `L1/layer_norm.py` (×3)
  - **`PPFormulaNetTextModel`** [wiring]: wires `PPFormulaNetScaledWordEmbedding`, `PPFormulaNetLearnedPositionalEmbedding`, `PPFormulaNetDecoderLayer`; direct `L1/layer_norm.py`
  - **`PPFormulaNetModel`** [wiring]: wires `PPFormulaNetVisionModel`, `PPFormulaNetTextModel`, `PPFormulaNetMultiModalProjector`
  - **`PPFormulaNetForConditionalGeneration`** [wiring]: wires `PPFormulaNetModel`; direct `L1/linear.py` (lm_head)

## pp_lcnet
- **src**: modeling_pp_lcnet.py, modular_pp_lcnet.py
- **hidden_act**: hardswish
- **status**: composable
- **classes**:
  - **`PPLCNetConvLayer`** [compute]: `L1/conv2d.py + L1/batch_norm2d.py + L1/hardswish.py` (conv + bn + activation)
  - **`PPLCNetDepthwiseSeparableConvLayer`** [wiring]: wires `PPLCNetConvLayer` (depthwise + pointwise) and optional `PPLCNetSqueezeExcitationModule`
  - **`PPLCNetSqueezeExcitationModule`** [compute]: `L1/global_avg_pool2d.py + L1/conv2d.py + L1/relu.py + L1/conv2d.py + L1/hardsigmoid.py` (SE block; closest L2: `L2/efficientnetv2_squeeze_excite.py`)
  - **`PPLCNetBlock`** [wiring]: wires `PPLCNetDepthwiseSeparableConvLayer`
  - **`PPLCNetEncoder`** [wiring]: wires `PPLCNetBlock`, `PPLCNetConvLayer` (stem)
  - **`PPLCNetBackbone`** [wiring]: wires `PPLCNetEncoder`
- **task heads (1)**: ForImageClassification — base + linear (per-task)

## pp_lcnet_v3
- **src**: modeling_pp_lcnet_v3.py, modular_pp_lcnet_v3.py
- **hidden_act**: hardswish
- **status**: composable
- **classes**:
  - **`PPLCNetV3ConvLayer`** [compute]: `L1/conv2d.py + L1/batch_norm2d.py + L1/hardswish.py`
  - **`PPLCNetV3LearnableAffineBlock`** [compute]: `L1/linear.py` (1×1 affine — scale + shift, parametric)
  - **`PPLCNetV3ActLearnableAffineBlock`** [compute]: `L1/hardswish.py` + `LearnableAffineBlock` composition (no exact L2 match)
  - **`PPLCNetV3LearnableRepLayer`** [compute]: multi-branch conv + add (RepVGG-style with learnable affine; closest L2: `L2/rtdetrv2_repvgg_block.py`)
  - **`PPLCNetV3SqueezeExcitationModule`** [compute]: `L1/global_avg_pool2d.py + L1/conv2d.py + L1/relu.py + L1/conv2d.py + L1/hardsigmoid.py`
  - **`PPLCNetV3DepthwiseSeparableConvLayer`** [wiring]: wires conv layers and SE
  - **`PPLCNetV3Block`** [wiring]: wires `PPLCNetV3DepthwiseSeparableConvLayer`
  - **`PPLCNetV3Backbone`** [wiring]: wires conv stem + blocks
  - **`PPLCNetV3Encoder`** [wiring]: wires backbone

## pp_ocrv5_mobile_det
- **src**: modeling_pp_ocrv5_mobile_det.py, modular_pp_ocrv5_mobile_det.py
- **hidden_act**: (no top-level; uses ReLU/HardSwish in components)
- **status**: composable
- **classes**:
  - **`PPOCRV5MobileDetSqueezeExcitationModule`** [compute]: `L1/global_avg_pool2d.py + L1/conv2d.py + L1/relu.py + L1/conv2d.py + L1/hardsigmoid.py`
  - **`PPOCRV5MobileDetResidualSqueezeExcitationLayer`** [wiring]: wires SE module + residual add
  - **`PPOCRV5MobileDetNeck`** [wiring]: wires conv layers (FPN-like neck)
  - **`PPOCRV5MobileDetConvBatchnormLayer`** [compute]: `L1/conv2d.py + L1/batch_norm2d.py + activation`
  - **`PPOCRV5MobileDetHead`** [compute]: `L1/conv2d.py + L1/conv_transpose2d.py` (segmentation head with deconv upsampling)
  - **`PPOCRV5MobileDetModel`** [wiring]: wires backbone (lcnet) + neck
- **task heads (1)**: ForObjectDetection — base + segmentation head (per-task)

## pp_ocrv5_mobile_rec
- **src**: modeling_pp_ocrv5_mobile_rec.py, modular_pp_ocrv5_mobile_rec.py
- **hidden_act**: silu
- **status**: composable
- **classes**:
  - **`PPOCRV5MobileRecAttention`** [compute]: `L1/linear.py + L1/dense_attention.py` (vision-style self-attention — closest L2: `L2/siglip_attention.py`)
  - **`PPOCRV5MobileRecMLP`** [compute]: `L1/linear.py + L1/silu.py + L1/linear.py` (2-layer with silu)
  - **`PPOCRV5MobileRecBlock`** [wiring]: wires `PPOCRV5MobileRecAttention`, `PPOCRV5MobileRecMLP`, `nn.LayerNorm`
  - **`PPOCRV5MobileRecConvLayer`** [compute]: `L1/conv2d.py + L1/batch_norm2d.py + L1/silu.py`
  - **`PPOCRV5MobileRecEncoderWithSVTR`** [wiring]: wires conv layers and SVTR transformer blocks
  - **`PPOCRV5MobileRecModel`** [wiring]: wires backbone (lcnet) + encoder
  - **`PPOCRV5MobileRecHead`** [compute]: `L1/linear.py` (text recognition head)
- **task heads (1)**: ForTextRecognition — base + linear (per-task)

## pp_ocrv5_server_det
- **src**: modeling_pp_ocrv5_server_det.py, modular_pp_ocrv5_server_det.py
- **hidden_act**: relu
- **status**: composable
- **classes**:
  - **`PPOCRV5ServerDetIntraclassBlock`** [wiring]: wires conv + bn + relu blocks (intraclass aggregation)
  - **`PPOCRV5ServerDetNeck`** [wiring]: wires conv blocks
  - **`PPOCRV5ServerDetConvBatchnormLayer`** [compute]: `L1/conv2d.py + L1/batch_norm2d.py + L1/relu.py`
  - **`PPOCRV5ServerDetSegmentationHead`** [compute]: `L1/conv2d.py + L1/conv_transpose2d.py` (segmentation head)
  - **`PPOCRV5ServerDetLocalModule`** [compute]: `L1/conv2d.py + L1/batch_norm2d.py` (local context)
  - **`PPOCRV5ServerDetHead`** [wiring]: wires segmentation head + local module
  - **`PPOCRV5ServerDetModel`** [wiring]: wires backbone + neck + head
- **task heads (1)**: ForObjectDetection — base + segmentation head (per-task)

## pp_ocrv5_server_rec
- **src**: modeling_pp_ocrv5_server_rec.py, modular_pp_ocrv5_server_rec.py
- **hidden_act**: silu
- **status**: composable
- **classes**:
  - **`PPOCRV5ServerRecBlock`** [wiring]: wires `PPOCRV5ServerRecAttention`, `PPOCRV5ServerRecMLP`, `nn.LayerNorm`
  - **`PPOCRV5ServerRecAttention`** [compute]: `L1/linear.py + L1/dense_attention.py` (closest L2: `L2/siglip_attention.py`)
  - **`PPOCRV5ServerRecConvLayer`** [compute]: `L1/conv2d.py + L1/batch_norm2d.py + L1/silu.py`
  - **`PPOCRV5ServerRecHead`** [compute]: `L1/linear.py`
  - **`PPOCRV5ServerRecMLP`** [compute]: `L1/linear.py + L1/silu.py + L1/linear.py`
  - **`PPOCRV5ServerRecEncoderWithSVTR`** [wiring]: wires conv layers + SVTR blocks
  - **`PPOCRV5ServerRecModel`** [wiring]: wires backbone + encoder
- **task heads (1)**: ForTextRecognition — base + linear (per-task)

## prompt_depth_anything
- **src**: modeling_prompt_depth_anything.py, modular_prompt_depth_anything.py
- **hidden_act**: relu (typically; uses ReLU and Conv layers)
- **status**: partial (DPT-like depth estimation; kb-nano has no DPT/depth-estimation head)
- **classes**:
  - **`PromptDepthAnythingLayer`** [compute]: `L1/conv2d.py + L1/relu.py + L1/conv2d.py` (residual conv block)
  - **`PromptDepthAnythingPreActResidualLayer`** [compute]: `L1/relu.py + L1/conv2d.py + L1/relu.py + L1/conv2d.py` (pre-activation residual)
  - **`PromptDepthAnythingFeatureFusionLayer`** [wiring]: wires `PromptDepthAnythingPreActResidualLayer` + interpolate
  - **`PromptDepthAnythingFeatureFusionStage`** [wiring]: wires `PromptDepthAnythingFeatureFusionLayer`
  - **`PromptDepthAnythingDepthEstimationHead`** [compute]: `L1/conv2d.py + L1/relu.py + L1/conv2d.py + L1/sigmoid.py` (final depth head — no exact L2 match)
  - **`PromptDepthAnythingReassembleLayer`** [compute]: `L1/conv_transpose2d.py + L1/conv2d.py` (resampling for multi-scale)
  - **`PromptDepthAnythingReassembleStage`** [wiring]: wires `PromptDepthAnythingReassembleLayer`
  - **`PromptDepthAnythingNeck`** [wiring]: wires reassemble + fusion + projection convs
- **task heads (1)**: ForDepthEstimation — base + conv head (per-task)

## prophetnet
- **src**: modeling_prophetnet.py
- **hidden_act**: gelu (activation_function)
- **status**: composable
- **classes**:
  - **`ProphetNetPositionalEmbeddings`** [compute, inherits `nn.Embedding`]: `L1/embedding.py` (with padding_idx offset)
  - **`ProphetNetAttention`** [compute]: `L1/linear.py + L1/dense_attention.py + L1/store_kvcache.py` (Bart-style attention with self/cross modes — closest L2: `L2/whisper_attention.py`; no exact L2)
  - **`ProphetNetFeedForward`** [compute]: `L1/linear.py + L1/gelu.py + L1/linear.py` (2-layer FFN — closest L2: `L2/whisper_mlp.py`)
  - **`ProphetNetNgramSelfAttention`** [compute]: `L1/linear.py + L1/dense_attention.py + L1/embedding.py` (n-gram attention with predict-next-tokens stream and relative position embeddings — no exact L2 match; bespoke)
  - **`ProphetNetEncoderLayer`** [wiring]: wires `ProphetNetAttention`, `ProphetNetFeedForward`; direct `L1/layer_norm.py` (×2)
  - **`ProphetNetDecoderLayer`** [wiring]: wires `ProphetNetNgramSelfAttention`, optional `ProphetNetAttention` (cross-attn), `ProphetNetFeedForward`; direct `L1/layer_norm.py` (×3)
  - **`ProphetNetEncoder`** [wiring]: wires `ProphetNetPositionalEmbeddings`, `ProphetNetEncoderLayer`; direct `L1/embedding.py` (word_embeddings), `L1/layer_norm.py` (embeddings_layer_norm)
  - **`ProphetNetDecoder`** [wiring]: wires `ProphetNetPositionalEmbeddings`, `ProphetNetDecoderLayer`; direct `L1/embedding.py` (word_embeddings, ngram_embeddings), `L1/layer_norm.py`
  - **`ProphetNetModel`** [wiring]: wires `ProphetNetEncoder`, `ProphetNetDecoder`; direct `L1/embedding.py` (shared word_embeddings)
  - **`ProphetNetForConditionalGeneration`** [wiring]: wires `ProphetNetModel`; direct `L1/linear.py` (lm_head), `L1/embedding.py` (padding_idx_embed)
  - **`ProphetNetDecoderWrapper`** [wiring]: wires `ProphetNetDecoder`
  - **`ProphetNetForCausalLM`** [wiring]: wires `ProphetNetDecoderWrapper`; direct `L1/linear.py` (lm_head)

## pvt
- **src**: modeling_pvt.py
- **hidden_act**: gelu
- **status**: composable
- **classes**:
  - **`PvtDropPath`** [compute]: `L1/dropout.py`
  - **`PvtPatchEmbeddings`** [compute]: `L1/conv2d.py + L1/layer_norm.py` (Conv2d patch + LayerNorm + position embedding parameter)
  - **`PvtSelfOutput`** [compute]: `L1/linear.py` (dense + dropout — residual external)
  - **`PvtEfficientSelfAttention`** [compute]: `L1/linear.py + L1/dense_attention.py + L1/conv2d.py + L1/layer_norm.py` (spatial reduction attention: q from full seq, k/v from spatially reduced seq via Conv2d + LN)
  - **`PvtAttention`** [wiring]: wires `PvtEfficientSelfAttention`, `PvtSelfOutput`
  - **`PvtFFN`** [compute]: `L1/linear.py + L1/gelu.py + L1/linear.py` (2-layer FFN)
  - **`PvtLayer`** [wiring]: wires `PvtAttention`, `PvtFFN`, `PvtDropPath`; direct `L1/layer_norm.py` (×2)
  - **`PvtEncoder`** [wiring]: wires `PvtPatchEmbeddings`, `PvtLayer`
  - **`PvtModel`** [wiring]: wires `PvtEncoder`; direct `L1/layer_norm.py`
- **task heads (1)**: ForImageClassification — base + linear (per-task)

## pvt_v2
- **src**: modeling_pvt_v2.py
- **hidden_act**: gelu
- **status**: composable
- **classes**:
  - **`PvtV2DropPath`** [compute]: `L1/dropout.py`
  - **`PvtV2OverlapPatchEmbeddings`** [compute]: `L1/conv2d.py + L1/layer_norm.py` (overlapping conv patch embed)
  - **`PvtV2DepthWiseConv`** [compute]: `L1/conv2d.py` (depthwise 3×3 conv used inside FFN)
  - **`PvtV2SelfAttention`** [compute]: `L1/linear.py + L1/dense_attention.py + L1/conv2d.py + L1/layer_norm.py` (linear/SR attention: optional adaptive avgpool + Conv2d + LayerNorm reduction of K/V)
  - **`PvtV2ConvFeedForwardNetwork`** [compute]: `L1/linear.py + L1/conv2d.py + L1/gelu.py + L1/linear.py` (FFN with intermediate depthwise conv)
  - **`PvtV2BlockLayer`** [wiring]: wires `PvtV2SelfAttention`, `PvtV2ConvFeedForwardNetwork`, `PvtV2DropPath`; direct `L1/layer_norm.py` (×2)
  - **`PvtV2EncoderLayer`** [wiring]: wires `PvtV2OverlapPatchEmbeddings`, `PvtV2BlockLayer`; direct `L1/layer_norm.py`
  - **`PvtV2Encoder`** [wiring]: wires `PvtV2EncoderLayer`
  - **`PvtV2Model`** [wiring]: wires `PvtV2Encoder`
  - **`PvtV2Backbone`** [wiring]: wires `PvtV2Model`
- **task heads (1)**: ForImageClassification — base + linear (per-task)

## qianfan_ocr
- **src**: modeling_qianfan_ocr.py, modular_qianfan_ocr.py (inherits InternVL/Beit)
- **hidden_act**: gelu
- **status**: composable
- **classes**:
  - **`QianfanOCRDropPath`** [compute, inherits `BeitDropPath`]: `L1/dropout.py`
  - **`QianfanOCRVisionRMSNorm`** [compute]: `L1/rms_norm.py` (Llama-style RMSNorm)
  - **`QianfanOCRVisionAttention`** [compute, inherits `InternVLVisionAttention`]: `L1/linear.py + L1/dense_attention.py` (qkv projection + non-causal attention; closest L2 `L2/siglip_attention.py`)
  - **`QianfanOCRVisionMLP`** [compute, inherits `InternVLVisionMLP`]: `L1/linear.py + L1/gelu.py + L1/linear.py`
  - **`QianfanOCRVisionLayer`** [wiring, inherits `InternVLVisionLayer`]: wires `QianfanOCRVisionAttention`, `QianfanOCRVisionMLP`, `QianfanOCRVisionRMSNorm` (×2), `QianfanOCRDropPath`
  - **`QianfanOCRVisionPatchEmbeddings`** [compute]: `L1/conv2d.py` (patch projection)
  - **`QianfanOCRVisionEmbeddings`** [compute, inherits `InternVLVisionEmbeddings`]: wires `QianfanOCRVisionPatchEmbeddings`; direct cls token + interpolatable position embedding
  - **`QianfanOCRVisionPreTrainedModel`** [skip — base class]
  - **`QianfanOCRVisionModel`** [wiring, inherits `InternVLVisionModel`]: wires `QianfanOCRVisionEmbeddings`, `QianfanOCRVisionLayer`
  - **`QianfanOCRMultiModalProjector`** [compute, inherits `InternVLMultiModalProjector`]: `L1/linear.py + L1/gelu.py + L1/linear.py + L1/layer_norm.py` (typically: norm + linear + gelu + linear)
  - **`QianfanOCRModel`** [wiring, inherits `InternVLModel`]: wires `QianfanOCRVisionModel`, language_model (typically Qwen2/Llama), `QianfanOCRMultiModalProjector`
  - **`QianfanOCRForConditionalGeneration`** [wiring, inherits `InternVLForConditionalGeneration`]: wires `QianfanOCRModel`; direct `L1/linear.py` (lm_head)

## qwen2
- **src**: modeling_qwen2.py, modular_qwen2.py (inherits Llama/Mistral/Gemma2)
- **hidden_act**: silu
- **status**: composable
- **classes**:
  - **`Qwen2MLP`** [compute, inherits `LlamaMLP`]: `L2/llama_mlp.py` (SwiGLU: gate_proj + up_proj + silu(gate)*up + down_proj)
  - **`Qwen2RotaryEmbedding`** [compute, inherits `Gemma2RotaryEmbedding`]: `L1/rotary_emb.py` (default RoPE, supports rope_scaling via ROPE_INIT_FUNCTIONS)
  - **`Qwen2Attention`** [compute, inherits `LlamaAttention`]: `L2/attention.py` (q/k/v with bias=True, o_proj no bias, RoPE, KV cache, sliding-window for sliding_attention layers)
  - **`Qwen2RMSNorm`** [compute]: `L1/rms_norm.py`
  - **`Qwen2DecoderLayer`** [wiring, inherits `LlamaDecoderLayer`]: wires `Qwen2Attention`, `Qwen2MLP`, `Qwen2RMSNorm` (×2)
  - **`Qwen2Model`** [wiring, inherits `MistralModel`]: wires `Qwen2DecoderLayer`, `Qwen2RotaryEmbedding`, `Qwen2RMSNorm`; direct `L1/embedding.py` (embed_tokens)
  - **`Qwen2ForCausalLM`** [wiring, inherits `LlamaForCausalLM`]: wires `Qwen2Model`; direct `L1/linear.py` (lm_head)
- **task heads (3)**: ForSequenceClassification, ForTokenClassification, ForQuestionAnswering — base + linear (per-task)

## qwen2_5_omni
- **src**: modeling_qwen2_5_omni.py, modular_qwen2_5_omni.py
- **hidden_act**: silu (multiple sub-configs all silu)
- **status**: partial (massive multi-modal: thinker text/audio/vision + talker DiT + token2wav BigVGAN; kb-nano has Qwen2-VL but not the audio encoder, talker DiT, or vocoder)
- **classes**:
  - **`Qwen2_5OmniAudioAttention`** [compute]: `L1/linear.py + L1/dense_attention.py` (Whisper-style audio self-attention; closest L2 `L2/whisper_attention.py`)
  - **`Qwen2_5OmniAudioEncoderLayer`** [wiring]: wires `Qwen2_5OmniAudioAttention`, MLP, LayerNorm
  - **`SinusoidsPositionEmbedding`** [compute]: `L1/sinusoidal_embed.py`
  - **`Qwen2_5OmniAudioEncoder`** [wiring]: wires `Qwen2_5OmniAudioEncoderLayer`, `SinusoidsPositionEmbedding`; direct `L1/conv1d.py` (conv1, conv2)
  - **`Qwen2_5OmniVisionAttention`** [compute]: `L1/linear.py + L1/dense_attention.py` (M-RoPE / windowed; closest L2 has no direct match — vision-style)
  - **`Qwen2_5OmniRMSNorm`** [compute]: `L1/rms_norm.py`
  - **`Qwen2_5OmniMLP`** [compute]: `L2/llama_mlp.py` (SwiGLU pattern)
  - **`Qwen2_5OmniVisionBlock`** [wiring]: wires vision attention + MLP + RMSNorm
  - **`Qwen2_5_VisionRotaryEmbedding`** [compute]: `L1/vision_rotary_emb.py`
  - **`Qwen2_5_VisionPatchEmbed`** [compute]: `L1/conv3d.py` (3D patch embed for video; uses Conv3d)
  - **`Qwen2_5OmniPatchMerger`** [compute]: `L1/linear.py + L1/gelu.py + L1/linear.py` (patch merger MLP)
  - **`Qwen2_5OmniVisionEncoder`** [wiring]: wires patch_embed, rotary_emb, vision blocks, patch_merger
  - **`Qwen2_5OmniRotaryEmbedding`** [compute]: `L1/mrope.py` (M-RoPE)
  - **`Qwen2_5OmniAttention`** [compute]: `L2/attention.py` (Llama-family with M-RoPE)
  - **`Qwen2MLP`** [compute]: `L2/llama_mlp.py`
  - **`Qwen2_5OmniDecoderLayer`** [wiring]: wires Qwen2_5OmniAttention, Qwen2MLP, Qwen2_5OmniRMSNorm (×2)
  - **`Qwen2_5OmniThinkerTextModel`** [wiring]: wires decoder layers + RotaryEmbedding + RMSNorm
  - **`Qwen2_5OmniThinkerForConditionalGeneration`** [wiring]: wires audio encoder + vision encoder + text model; direct `L1/linear.py` (lm_head)
  - **`Qwen2_5OmniTalkerModel`** [wiring]: wires Qwen2-style decoder for talker (text-to-codec)
  - **`Qwen2_5OmniTalkerForConditionalGeneration`** [wiring]: wires Qwen2_5OmniTalkerModel; direct `L1/linear.py` (codec head)
  - **`Qwen2_5OmniDiTRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (DiT rotary)
  - **`TimeDelayNetBlock`** [compute]: `L1/conv1d.py + L1/relu.py + L1/batch_norm2d.py` (TDNN block — speaker encoder; no exact L2 match)
  - **`Res2NetBlock`** [compute]: `L1/conv1d.py + L1/relu.py + L1/batch_norm2d.py` (multi-scale Res2Net; no exact L2 match)
  - **`SqueezeExcitationBlock`** [compute]: `L1/linear.py + L1/relu.py + L1/sigmoid.py` (1D SE; no exact L2 match)
  - **`AttentiveStatisticsPooling`** [compute]: `L1/linear.py + L1/conv1d.py + L1/tanh.py + L1/softmax.py` (attentive pooling)
  - **`SqueezeExcitationRes2NetBlock`** [wiring]: wires Res2Net + SE
  - **`ECAPA_TimeDelayNet`** [wiring]: wires TDNN + Res2Net stacks + AttentivePooling (speaker-encoder backbone)
  - **`DiTInputEmbedding`** [compute]: `L1/linear.py + L1/silu.py` (input projection for DiT)
  - **`DiTCodecEmbedding`** [compute]: `L1/embedding.py` (codec token embed)
  - **`Qwen2_5_OmniAdaLayerNormZero`** [compute]: `L1/silu.py + L1/linear.py + L1/layer_norm.py` (adaLN-zero modulation; closest L2: `L2/ada_layer_norm.py` — verify)
  - **`Qwen2_5_OmniAdaLayerNormZero_Final`** [compute]: same pattern as above
  - **`DiTMLP`** [compute]: `L1/linear.py + L1/gelu.py + L1/linear.py` (2-layer MLP)
  - **`DiTAttention`** [compute]: `L1/linear.py + L1/dense_attention.py` (DiT-style self-attention)
  - **`SinusPositionEmbedding`** [compute]: `L1/sinusoidal_embed.py`
  - **`DiTTimestepEmbedding`** [compute]: `L1/linear.py + L1/silu.py + L1/linear.py` (timestep embed)
  - **`DiTDecoderLayer`** [wiring]: wires DiTAttention, DiTMLP, AdaLN-zero
  - **`SnakeBeta`** [compute]: custom snake activation (no L1 match — composable from L1/sigmoid + math)
  - **`UpSample1d`** [compute]: `L1/conv1d.py` (upsampling via conv transpose / interp)
  - **`DownSample1d`** [compute]: `L1/conv1d.py` (downsampling)
  - **`TorchActivation1d`** [wiring]: wires UpSample1d + Activation + DownSample1d (snake-aliasing)
  - **`AMPBlock`** [compute]: `L1/conv1d.py` (multi-scale conv block in BigVGAN)
  - **`Qwen2_5OmniToken2WavBigVGANModel`** [wiring]: wires AMPBlock + UpSample1d (vocoder)
  - **`RungeKutta4ODESolver`** [skip — solver class, not nn.Module]
  - **`Qwen2_5OmniToken2WavDiTModel`** [wiring]: wires DiT layers + ECAPA + flow ODE solver
  - **`Qwen2_5OmniToken2WavModel`** [wiring]: wires DiT + BigVGAN (codec→audio)
  - **`Qwen2_5OmniForConditionalGeneration`** [wiring]: wires Thinker + Talker + Token2Wav

## qwen2_5_vl
- **src**: modeling_qwen2_5_vl.py, modular_qwen2_5_vl.py (inherits Qwen2VL)
- **hidden_act**: silu
- **status**: composable (kb-nano has `L4/qwen25_vl_encoder.py`, `L4/qwen2_vl.py`)
- **classes**:
  - **`Qwen2_5_VLRMSNorm`** [compute, inherits `LlamaRMSNorm`]: `L1/rms_norm.py`
  - **`Qwen2_5_VLMLP`** [compute]: `L2/llama_mlp.py` (SwiGLU)
  - **`Qwen2_5_VisionPatchEmbed`** [compute, inherits `PatchEmbed`]: `L1/conv3d.py` (Conv3d for spatio-temporal patches)
  - **`Qwen2_5_VisionRotaryEmbedding`** [compute, inherits `VisionRotaryEmbedding`]: `L1/vision_rotary_emb.py`
  - **`Qwen2_5_VLPatchMerger`** [compute, inherits `PatchMerger`]: `L1/linear.py + L1/gelu.py + L1/linear.py` (with RMSNorm) — `L2/vision_patch_merger.py`
  - **`Qwen2_5_VLVisionAttention`** [compute, inherits `VisionAttention`]: `L1/linear.py + L1/dense_attention.py` (closest L2: `L2/vision_attention.py`)
  - **`Qwen2_5_VLVisionBlock`** [wiring]: wires `Qwen2_5_VLVisionAttention`, `Qwen2_5_VLMLP`, `Qwen2_5_VLRMSNorm` (×2)
  - **`Qwen2_5_VisionTransformerPretrainedModel`** [wiring]: wires patch_embed, rotary_emb, blocks, patch_merger
  - **`Qwen2_5_VLRotaryEmbedding`** [compute]: `L1/mrope.py` (M-RoPE for text decoder)
  - **`Qwen2MLP`** [compute]: `L2/llama_mlp.py`
  - **`Qwen2_5_VLAttention`** [compute]: `L2/attention.py` (Qwen2 with M-RoPE)
  - **`Qwen2_5_VLDecoderLayer`** [wiring]: wires `Qwen2_5_VLAttention`, `Qwen2MLP`, `Qwen2_5_VLRMSNorm` (×2)
  - **`Qwen2_5_VLTextModel`** [wiring]: wires `Qwen2_5_VLDecoderLayer`, `Qwen2_5_VLRotaryEmbedding`, `Qwen2_5_VLRMSNorm`; direct `L1/embedding.py`
  - **`Qwen2_5_VLModel`** [wiring, inherits `Qwen2VLModel`]: wires `Qwen2_5_VisionTransformerPretrainedModel`, `Qwen2_5_VLTextModel`
  - **`Qwen2_5_VLForConditionalGeneration`** [wiring, inherits `Qwen2VLForConditionalGeneration`]: wires `Qwen2_5_VLModel`; direct `L1/linear.py` (lm_head)

## qwen2_audio
- **src**: modeling_qwen2_audio.py
- **hidden_act**: gelu (activation_function)
- **status**: composable
- **classes**:
  - **`Qwen2AudioAttention`** [compute]: `L1/linear.py + L1/dense_attention.py` (Whisper-style audio attention with q/k/v/out projections; closest L2: `L2/whisper_attention.py`)
  - **`Qwen2AudioEncoderLayer`** [wiring]: wires `Qwen2AudioAttention`; direct `L1/linear.py` (fc1, fc2), `L1/gelu.py`, `L1/layer_norm.py` (×2)
  - **`Qwen2AudioEncoder`** [wiring]: wires `Qwen2AudioEncoderLayer`; direct `L1/conv1d.py` (conv1, conv2), `L1/embedding.py` (sinusoidal positional), `L1/avg_pool1d.py` (avg_pooler), `L1/layer_norm.py`
  - **`Qwen2AudioMultiModalProjector`** [compute]: `L1/linear.py` (single linear projection)
  - **`Qwen2AudioForConditionalGeneration`** [wiring]: wires `Qwen2AudioEncoder`, `Qwen2AudioMultiModalProjector`, language_model (Qwen2); direct `L1/linear.py` (lm_head if not tied)

## qwen2_moe
- **src**: modeling_qwen2_moe.py, modular_qwen2_moe.py (inherits Llama/Mixtral/Gemma2)
- **hidden_act**: silu
- **status**: composable (kb-nano has shared_expert_moe pattern)
- **classes**:
  - **`Qwen2MoeRMSNorm`** [compute, inherits `LlamaRMSNorm`]: `L1/rms_norm.py`
  - **`Qwen2MoeRotaryEmbedding`** [compute, inherits `Gemma2RotaryEmbedding`]: `L1/rotary_emb.py`
  - **`Qwen2MoeMLP`** [compute, inherits `GemmaMLP`]: `L2/llama_mlp.py` (SwiGLU; used as both regular MLP and shared expert when intermediate_size differs)
  - **`Qwen2MoeAttention`** [compute, inherits `LlamaAttention`]: `L2/attention.py` (q/k/v bias=True, o_proj no bias)
  - **`Qwen2MoeExperts`** [compute, inherits `MixtralExperts`]: `L1/moe_grouped_gemm.py` (3D-tensor experts, fused grouped gemm)
  - **`Qwen2MoeTopKRouter`** [compute]: `L1/linear.py + L1/topk_softmax.py` (gate Linear + softmax-then-topk routing with optional renormalization)
  - **`Qwen2MoeSparseMoeBlock`** [wiring]: wires `Qwen2MoeTopKRouter`, `Qwen2MoeExperts`, plus shared expert (`Qwen2MoeMLP`) and shared expert gate — matches `L2/shared_expert_moe.py`
  - **`Qwen2MoeDecoderLayer`** [wiring, inherits `LlamaDecoderLayer`]: wires `Qwen2MoeAttention`, `Qwen2MoeSparseMoeBlock` (or `Qwen2MoeMLP` for dense layers), `Qwen2MoeRMSNorm` (×2)
  - **`Qwen2MoeModel`** [wiring, inherits `MixtralModel`]: wires `Qwen2MoeDecoderLayer`, `Qwen2MoeRotaryEmbedding`, `Qwen2MoeRMSNorm`; direct `L1/embedding.py`
  - **`Qwen2MoeForCausalLM`** [wiring, inherits `MixtralForCausalLM`]: wires `Qwen2MoeModel`; direct `L1/linear.py` (lm_head)
- **task heads (3)**: ForSequenceClassification, ForTokenClassification, ForQuestionAnswering — base + linear (per-task)

## qwen2_vl
- **src**: modeling_qwen2_vl.py
- **hidden_act**: quick_gelu (vision); silu (text decoder)
- **status**: composable (kb-nano `L4/qwen2_vl.py`)
- **classes**:
  - **`Qwen2VLRMSNorm`** [compute]: `L1/rms_norm.py`
  - **`Qwen2VLRotaryEmbedding`** [compute]: `L1/mrope.py` (M-RoPE for text decoder)
  - **`VisionRotaryEmbedding`** [compute]: `L1/vision_rotary_emb.py`
  - **`PatchEmbed`** [compute]: `L1/conv3d.py` (Conv3d for spatio-temporal patches)
  - **`PatchMerger`** [compute]: `L2/vision_patch_merger.py` (linear + GeLU + linear with RMSNorm)
  - **`VisionMlp`** [compute]: `L1/linear.py + L1/quickgelu.py + L1/linear.py` (vision MLP with quick_gelu — closest L2 `L2/clip_mlp.py`)
  - **`VisionAttention`** [compute]: `L2/vision_attention.py` (q/k/v with vision RoPE, non-causal)
  - **`Qwen2VLVisionBlock`** [wiring]: wires `VisionAttention`, `VisionMlp`, `nn.LayerNorm` (×2)
  - **`Qwen2MLP`** [compute]: `L2/llama_mlp.py` (SwiGLU with silu)
  - **`Qwen2VLAttention`** [compute]: `L2/attention.py` (Qwen2 attention with M-RoPE; q/k/v bias=True, o_proj no bias)
  - **`Qwen2VLDecoderLayer`** [wiring]: wires `Qwen2VLAttention`, `Qwen2MLP`, `Qwen2VLRMSNorm` (×2)
  - **`Qwen2VisionTransformerPretrainedModel`** [wiring]: wires `PatchEmbed`, `VisionRotaryEmbedding`, `Qwen2VLVisionBlock`, `PatchMerger`
  - **`Qwen2VLTextModel`** [wiring]: wires `Qwen2VLDecoderLayer`, `Qwen2VLRotaryEmbedding`, `Qwen2VLRMSNorm`; direct `L1/embedding.py`
  - **`Qwen2VLModel`** [wiring]: wires `Qwen2VisionTransformerPretrainedModel`, `Qwen2VLTextModel`
  - **`Qwen2VLForConditionalGeneration`** [wiring]: wires `Qwen2VLModel`; direct `L1/linear.py` (lm_head)
