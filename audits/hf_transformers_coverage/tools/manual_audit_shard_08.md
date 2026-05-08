## lfm2
- **src**: modular_lfm2.py
- **status**: composable
- **rationale**: Decoder-LM with hybrid attention/short-conv layers; all primitives (RMSNorm, RoPE, SwiGLU MLP, GQA attention with q/k LayerNorm, causal_conv1d) exist in kb-nano L1/L2.
- **classes**:
  - **`Lfm2RMSNorm`** [compute]: `L1/rms_norm.py` (Standard Llama-style RMSNorm.)
  - **`Lfm2RotaryEmbedding`** [compute]: `L1/rotary_emb.py` (Standard NeoX-style RoPE inverse frequencies.)
  - **`Lfm2MLP`** [compute]: `L2/llama_mlp.py`, `L1/silu_and_mul.py` (SwiGLU: w2(silu(w1(x)) * w3(x)) - identical to LlamaMLP shape (w1=gate, w3=up, w2=down).)
  - **`Lfm2Attention`** [compute]: `L2/attention.py`, `L1/rms_norm.py`, `L1/rotary_emb.py` (GQA with per-head Q/K RMSNorm before RoPE, then attention. Maps to LlamaAttention with qk_norm=True (Qwen3-style).)
  - **`Lfm2ShortConv`** [compute]: `L1/causal_conv1d.py`, `L1/linear.py` (Depth-wise causal conv1d gated by linear in_proj/out_proj; uses causal_conv1d_fn/_update kernels matched by L1/causal_conv1d.py.)
  - **`Lfm2DecoderLayer`** [wiring]: Pure wiring of attn/conv + MLP + RMSNorm residuals.
  - **`Lfm2Model`** [wiring]: Wiring: embeddings + decoder stack + final embedding_norm.
  - **`Lfm2ForCausalLM`** [wiring]: Wiring: model + lm_head.

## lfm2_moe
- **src**: modular_lfm2_moe.py
- **status**: composable
- **rationale**: Lfm2 hybrid LM but with sparse MoE FFN on later layers; uses Qwen2MoeExperts pattern (sigmoid routing, top-k) and standard MoE grouped GEMM kernels exist in kb-nano.
- **classes**:
  - **`Lfm2MoeRMSNorm`** [compute]: `L1/rms_norm.py` (Standard RMSNorm.)
  - **`Lfm2MoeRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (Inherits Lfm2 RoPE = standard.)
  - **`Lfm2MoeMLP`** [compute]: `L2/llama_mlp.py`, `L1/silu_and_mul.py` (Same w1/w3/w2 SwiGLU pattern as Lfm2.)
  - **`Lfm2MoeExperts`** [compute]: `L1/moe_grouped_gemm.py`, `L1/silu_and_mul.py` (Stacked expert weights with silu activation; standard fused-MoE grouped-GEMM.)
  - **`Lfm2MoeSparseMoeBlock`** [compute]: `L2/qwen3_moe.py`, `L1/topk_softmax.py`, `L1/sigmoid_topk.py` (Sigmoid+topk router with optional bias and renorm; matches Qwen3-MoE/Lfm2 routing in kb-nano.)
  - **`Lfm2MoeAttention`** [compute]: `L2/attention.py` (Same as Lfm2Attention.)
  - **`Lfm2MoeShortConv`** [compute]: `L1/causal_conv1d.py` (Same as Lfm2ShortConv.)
  - **`Lfm2MoeDecoderLayer`** [wiring]: Wiring: chooses dense MLP or sparse MoE per layer.
  - **`Lfm2MoeModel`** [wiring]: Wiring: embed + decoder stack + embedding_norm.
  - **`Lfm2MoeForCausalLM`** [wiring]: Wiring.

## lfm2_vl
- **src**: modular_lfm2_vl.py
- **status**: composable
- **rationale**: Vision-language wrapper around Lfm2 LM and Siglip vision tower; only new compute is a tiny Linear+GELU+Linear projector with optional LayerNorm and pixel-unshuffle reshape.
- **classes**:
  - **`Lfm2VlMultiModalProjector`** [compute]: `L1/linear.py`, `L1/layer_norm.py`, `L1/gelu.py` (LayerNorm + Linear + activation + Linear, no SwiGLU.)
  - **`Lfm2VlModel`** [wiring]: Wiring: vision_tower + projector + language_model orchestrator.
  - **`Lfm2VlForConditionalGeneration`** [wiring]: Wiring: model + lm_head.

## lightglue
- **src**: modular_lightglue.py
- **status**: partial
- **rationale**: Keypoint-matching graph network with depth-confidence early stopping, point-pruning, log-double-softmax assignment, and a SuperPoint detector backbone — no kb-nano L4 / L2 covers this domain.
- **classes**:
  - **`LightGluePositionalEncoder`** [compute]: Keypoint-matching graph network with depth-confidence early stopping, point-pruning, log-double-softmax assignment, and a SuperPoint detector backbone — no kb-nano L4 / L2 covers this domain.
  - **`LightGlueAttention`** [compute]: `L2/attention.py` (Self/cross attention reusing apply_rotary_pos_emb on (cos,sin)=keypoint embedding; mappable to encoder attention but feeds bespoke pipeline.)
  - **`LightGlueMLP`** [compute]: `L1/linear.py`, `L1/layer_norm.py`, `L1/gelu.py` (fc1 -> LayerNorm -> activation -> fc2.)
  - **`LightGlueTransformerLayer`** [wiring]: Wiring of self/cross attn + MLPs.
  - **`LightGlueMatchAssignmentLayer`** [wiring]: Match-assignment with sigmoid_log_double_softmax over similarity matrix; no kb-nano kernel.
  - **`LightGlueTokenConfidenceLayer`** [wiring]: Linear + sigmoid confidence head.
  - **`LightGlueForKeypointMatching`** [wiring]: End-to-end pipeline with SuperPoint detector, transformer layers, pruning and early-stop loop.

## lilt
- **src**: modeling_lilt.py
- **status**: partial
- **partial_reason**: LiltSelfAttention runs two parallel QKV streams (text + layout) and adds the scaled-dot-product scores before softmax; the cross-stream score addition has no kb-nano L2 equivalent. Implemented in HF via raw torch.matmul + nn.Softmax, so it works on stock PyTorch but lacks a fused kernel.
- **rationale**: BERT-style encoder with a dual text+layout attention stream that sums the attention scores across both streams; no kb-nano kernel for the dual-stream layout-aware attention.
- **classes**:
  - **`LiltSelfAttention`** [compute]: LiltSelfAttention runs two parallel QKV streams (text + layout) and adds the scaled-dot-product scores before softmax; the cross-stream score addition has no kb-nano L2 equivalent. Implemented in HF v
  - **`LiltTextEmbeddings`** [compute]: `L2/encoder_embeddings.py` (Token + position + token-type embeddings + LayerNorm.)
  - **`LiltLayoutEmbeddings`** [compute]: `L1/embedding.py`, `L1/linear.py`, `L1/layer_norm.py` (Box-coord embeddings concatenated, projected, LayerNorm.)
  - **`LiltSelfOutput`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (Linear + dropout + LayerNorm residual (BERT style).)
  - **`LiltAttention`** [wiring]: Wiring around LiltSelfAttention + LiltSelfOutput (sibling-class pattern).
  - **`LiltIntermediate`** [compute]: `L2/encoder_mlp.py` (fc1 + GELU.)
  - **`LiltOutput`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (fc2 + LayerNorm residual.)
  - **`LiltLayer`** [wiring]: Wiring.
  - **`LiltEncoder`** [wiring]: Wiring: stack of layers.
  - **`LiltPooler`** [wiring]: Wiring: dense + tanh on [CLS].
  - **`LiltModel`** [wiring]: Wiring.
  - **`LiltForSequenceClassification`** [wiring]: Wiring.
  - **`LiltForTokenClassification`** [wiring]: Wiring.
  - **`LiltClassificationHead`** [wiring]: Wiring.
  - **`LiltForQuestionAnswering`** [wiring]: Wiring.

## llama
- **src**: modeling_llama.py
- **status**: kb_nano_l4
- **rationale**: Standard Llama-3 decoder LM with full kb-nano L4 pipeline at L4/llama.py.
- **classes**:
  - **`LlamaRMSNorm`** [compute]: `L1/rms_norm.py` (Standard RMSNorm.)
  - **`LlamaRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (NeoX-style RoPE.)
  - **`LlamaMLP`** [compute]: `L2/llama_mlp.py`, `L1/silu_and_mul.py` (SwiGLU MLP.)
  - **`LlamaAttention`** [compute]: `L2/attention.py` (GQA q/k/v/o projections + RoPE + attention.)
  - **`LlamaDecoderLayer`** [compute]: `L3/llama_decoder.py` (Decoder layer wiring.)
  - **`LlamaModel`** [wiring]: Wiring.
  - **`LlamaForCausalLM`** [wiring]: Wiring.

## llama4
- **src**: modeling_llama4.py
- **status**: kb_nano_l4
- **rationale**: Llama-4 Scout MoE with NoPE/temperature tuning and weight-less QK norm; covered by L4/llama4.py.
- **classes**:
  - **`Llama4TextExperts`** [compute]: `L2/llama4_moe.py`, `L1/moe_grouped_gemm.py`, `L1/silu_and_mul.py` (Stacked SwiGLU expert weights; uses fused MoE grouped GEMM.)
  - **`Llama4TextMLP`** [compute]: `L2/llama_mlp.py`, `L1/silu_and_mul.py` (SwiGLU MLP (shared expert).)
  - **`Llama4TextL2Norm`** [compute]: `L1/l2_norm.py` (Weight-less L2 norm (used as QK norm).)
  - **`Llama4TextRMSNorm`** [compute]: `L1/rms_norm.py` (Llama-style RMSNorm.)
  - **`Llama4Router`** [compute]: `L1/linear.py` (Router is a Linear; sigmoid+top-1 routing wrapped in MoE block.)
  - **`Llama4TextMoe`** [compute]: `L2/llama4_moe.py`, `L1/sigmoid_topk.py` (Top-1 sigmoid routing + shared expert combine.)
  - **`Llama4TextRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (Standard RoPE inverse frequencies.)
  - **`Llama4TextAttention`** [compute]: `L2/attention.py`, `L1/l2_norm.py` (GQA with optional NoPE, weight-less L2 QK norm, attention-temperature tuning. Covered by LlamaAttention(nope/use_weightless_qk_norm/attn_temperature_tuning).)
  - **`Llama4TextDecoderLayer`** [compute]: `L3/llama4_decoder.py` (Decoder layer wiring.)
  - **`Llama4TextModel`** [wiring]: Wiring.
  - **`Llama4ForCausalLM`** [wiring]: Wiring.
  - **`Llama4VisionMLP2`** [compute]: `L1/linear.py`, `L1/gelu.py` (Two Linear + GELU stack used by projector.)
  - **`Llama4MultiModalProjector`** [compute]: `L1/linear.py` (Linear projector.)
  - **`Llama4VisionPixelShuffleMLP`** [wiring]: Pixel-shuffle reshape + MLP.
  - **`Llama4VisionAttention`** [compute]: `L2/vision_attention.py` (Vision attention with 2D rotary embedding.)
  - **`Llama4VisionMLP`** [compute]: `L2/vision_mlp.py`, `L1/gelu.py` (Two Linear + GELU vision MLP.)
  - **`Llama4VisionEncoderLayer`** [wiring]: Wiring.
  - **`Llama4VisionEncoder`** [wiring]: Wiring.
  - **`Llama4UnfoldConvolution`** [wiring]: Unfold + Linear.
  - **`Llama4VisionRotaryEmbedding`** [compute]: `L1/vision_rotary_emb.py` (2D vision RoPE.)
  - **`Llama4VisionModel`** [wiring]: Wiring.
  - **`Llama4ForConditionalGeneration`** [wiring]: Wiring.

## llava
- **src**: modeling_llava.py
- **status**: composable
- **rationale**: Wrapper around an auto-loaded text decoder + auto-loaded vision tower with a 2-layer GELU projector; only own kernel is the Linear+act+Linear projector.
- **classes**:
  - **`LlavaMultiModalProjector`** [compute]: `L1/linear.py`, `L1/gelu.py` (Linear -> activation -> Linear projector.)
  - **`LlavaModel`** [wiring]: Wiring: vision_tower + projector + language_model.
  - **`LlavaForConditionalGeneration`** [wiring]: Wiring.

## llava_next
- **src**: modeling_llava_next.py
- **status**: composable
- **rationale**: LLaVA-Next wraps an auto vision encoder + auto LM with a 2-layer projector; AnyRes patch unpadding is python reshapes. No new kernels needed.
- **classes**:
  - **`LlavaNextMultiModalProjector`** [compute]: `L1/linear.py`, `L1/gelu.py` (2-layer projector.)
  - **`LlavaNextModel`** [wiring]: Wiring: AnyRes + projector + LM.
  - **`LlavaNextForConditionalGeneration`** [wiring]: Wiring.

## llava_next_video
- **src**: modular_llava_next_video.py
- **status**: composable
- **rationale**: LlavaNext + a temporal video pooling layer (avg pool over time) + 2-layer projector; underlying compute is auto-loaded backbones plus standard ops.
- **classes**:
  - **`LlavaNextVideoPooler`** [compute]: `L1/avg_pool2d.py`, `L1/linear.py` (Spatial average pool / conv + projection over video frames.)
  - **`LlavaNextVideoMultiModalProjector`** [compute]: `L1/linear.py`, `L1/gelu.py` (2-layer projector.)
  - **`LlavaNextVideoModel`** [wiring]: Wiring.
  - **`LlavaNextVideoForConditionalGeneration`** [wiring]: Wiring.

## llava_onevision
- **src**: modular_llava_onevision.py
- **status**: composable
- **rationale**: LLaVA-OneVision = LlavaNext + AnyRes for video; only own kernel is a 2-layer projector. Auto-loaded backbones supply the heavy compute.
- **classes**:
  - **`LlavaOnevisionMultiModalProjector`** [compute]: `L1/linear.py`, `L1/gelu.py` (2-layer projector.)
  - **`LlavaOnevisionModel`** [wiring]: Wiring.
  - **`LlavaOnevisionForConditionalGeneration`** [wiring]: Wiring.

## longformer
- **src**: modeling_longformer.py
- **status**: partial
- **rationale**: Sliding-chunk local attention with global-attention tokens — bespoke attention pattern requiring custom CUDA-style kernels (sliding-chunk QK matmul, padded diagonal mask handling).
- **classes**:
  - **`LongformerSelfAttention`** [compute]: Sliding-chunk local attention with global-attention tokens — bespoke attention pattern requiring custom CUDA-style kernels (sliding-chunk QK matmul, padded diagonal mask handling).
  - **`LongformerEmbeddings`** [compute]: `L2/encoder_embeddings.py` (Standard token+position+token-type embeddings + LayerNorm.)
  - **`LongformerSelfOutput`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (Standard projection + LayerNorm + residual.)
  - **`LongformerAttention`** [wiring]: Wiring around LongformerSelfAttention + Output (sibling pattern).
  - **`LongformerIntermediate`** [compute]: `L2/encoder_mlp.py` (fc1 + GELU.)
  - **`LongformerOutput`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (fc2 + LayerNorm residual.)
  - **`LongformerLayer`** [wiring]: Wiring.
  - **`LongformerEncoder`** [wiring]: Wiring.
  - **`LongformerPooler`** [wiring]: Wiring.
  - **`LongformerLMHead`** [wiring]: Wiring.
  - **`LongformerModel`** [wiring]: Wiring.
  - **`LongformerForMaskedLM`** [wiring]: Wiring.
  - **`LongformerForSequenceClassification`** [wiring]: Wiring.
  - **`LongformerClassificationHead`** [wiring]: Wiring.
  - **`LongformerForQuestionAnswering`** [wiring]: Wiring.
  - **`LongformerForTokenClassification`** [wiring]: Wiring.
  - **`LongformerForMultipleChoice`** [wiring]: Wiring.

## longt5
- **src**: modeling_longt5.py
- **status**: partial
- **rationale**: LongT5 adds Local and TransientGlobal block-sparse attention variants on top of T5 relative-bias attention; no kb-nano kernel for the local-block / transient-global attention.
- **classes**:
  - **`LongT5LocalAttention`** [compute]: LongT5 adds Local and TransientGlobal block-sparse attention variants on top of T5 relative-bias attention; no kb-nano kernel for the local-block / transient-global attention.
  - **`LongT5LayerNorm`** [compute]: `L1/t5_layer_norm.py` (T5-style RMSNorm without centering.)
  - **`LongT5DenseActDense`** [compute]: `L2/t5_dense.py` (wi -> act -> wo (T5 dense).)
  - **`LongT5DenseGatedActDense`** [compute]: `L2/t5_dense.py`, `L1/gelu_and_mul.py` (Gated GELU MLP (T5v1.1 / Flan).)
  - **`LongT5LayerFF`** [wiring]: Wiring.
  - **`LongT5Attention`** [compute]: `L2/t5_attention.py` (Standard T5 relative-bias attention.)
  - **`LongT5TransientGlobalAttention`** [wiring]: Local + transient global tokens with side relative-bias; no kb-nano kernel.
  - **`LongT5LayerSelfAttention`** [wiring]: Wiring.
  - **`LongT5LayerLocalSelfAttention`** [wiring]: Wiring around LocalAttention.
  - **`LongT5LayerTransientGlobalSelfAttention`** [wiring]: Wiring.
  - **`LongT5LayerCrossAttention`** [wiring]: Wiring.
  - **`LongT5Block`** [wiring]: Wiring.
  - **`LongT5Stack`** [wiring]: Wiring.
  - **`LongT5Model`** [wiring]: Wiring.
  - **`LongT5ForConditionalGeneration`** [wiring]: Wiring.
  - **`LongT5EncoderModel`** [wiring]: Wiring.

## luke
- **src**: modeling_luke.py
- **status**: composable
- **rationale**: RoBERTa-style encoder with extra entity stream; LukeSelfAttention is BERT-style QKV + softmax with separate entity Q/K/V, all standard ops - mappable to encoder primitives.
- **classes**:
  - **`LukeEmbeddings`** [compute]: `L2/encoder_embeddings.py` (RoBERTa-style word + position + token-type embeddings + LayerNorm.)
  - **`LukeEntityEmbeddings`** [compute]: `L1/embedding.py`, `L1/linear.py`, `L1/layer_norm.py` (Entity embedding + position + token-type + projection + LayerNorm.)
  - **`LukeSelfAttention`** [compute]: `L2/encoder_attention.py` (BERT-style QKV attention extended to entity stream by additional projections; same compute primitives.)
  - **`LukeSelfOutput`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (Standard projection + LayerNorm residual.)
  - **`LukeAttention`** [wiring]: Sibling-class wrapper around SelfAttention.
  - **`LukeIntermediate`** [compute]: `L2/encoder_mlp.py` (fc1 + activation.)
  - **`LukeOutput`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (fc2 + LayerNorm residual.)
  - **`LukeLayer`** [wiring]: Wiring.
  - **`LukeEncoder`** [wiring]: Wiring.
  - **`LukePooler`** [wiring]: Wiring.
  - **`EntityPredictionHeadTransform`** [compute]: `L1/linear.py`, `L1/layer_norm.py`, `L1/gelu.py` (Linear + activation + LayerNorm.)
  - **`EntityPredictionHead`** [wiring]: Wiring.
  - **`LukeModel`** [wiring]: Wiring.
  - **`LukeLMHead`** [wiring]: Wiring.
  - **`LukeForMaskedLM`** [wiring]: Wiring.
  - **`LukeForEntityClassification`** [wiring]: Wiring.
  - **`LukeForEntityPairClassification`** [wiring]: Wiring.
  - **`LukeForEntitySpanClassification`** [wiring]: Wiring.
  - **`LukeForSequenceClassification`** [wiring]: Wiring.
  - **`LukeForTokenClassification`** [wiring]: Wiring.
  - **`LukeForQuestionAnswering`** [wiring]: Wiring.
  - **`LukeForMultipleChoice`** [wiring]: Wiring.

## lxmert
- **src**: modeling_lxmert.py
- **status**: composable
- **rationale**: BERT-style cross-modal encoder. Standard QKV attention (LxmertAttention) + cross-attention layers + GeLU MLPs; all primitives map to encoder kernels.
- **classes**:
  - **`GeLU`** [compute]: `L1/gelu.py` (Wraps F.gelu.)
  - **`LxmertEmbeddings`** [compute]: `L2/encoder_embeddings.py` (BERT-style word + position + token-type embeddings + LayerNorm.)
  - **`LxmertAttention`** [compute]: `L2/encoder_attention.py` (Standard QKV attention (used as both self- and cross-attention).)
  - **`LxmertAttentionOutput`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (Linear + LayerNorm residual.)
  - **`LxmertCrossAttentionLayer`** [wiring]: Wiring.
  - **`LxmertSelfAttentionLayer`** [wiring]: Wiring.
  - **`LxmertIntermediate`** [compute]: `L2/encoder_mlp.py` (fc1 + GELU.)
  - **`LxmertOutput`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (fc2 + LayerNorm residual.)
  - **`LxmertLayer`** [wiring]: Wiring.
  - **`LxmertXLayer`** [wiring]: Wiring (cross-modality block).
  - **`LxmertVisualFeatureEncoder`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (Two Linear + LayerNorm projections for visual feats.)
  - **`LxmertEncoder`** [wiring]: Wiring.
  - **`LxmertPooler`** [wiring]: Wiring.
  - **`LxmertPredictionHeadTransform`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (Linear + activation + LayerNorm.)
  - **`LxmertLMPredictionHead`** [wiring]: Wiring.
  - **`LxmertVisualAnswerHead`** [wiring]: Wiring.
  - **`LxmertVisualObjHead`** [wiring]: Wiring.
  - **`LxmertPreTrainingHeads`** [wiring]: Wiring.
  - **`LxmertModel`** [wiring]: Wiring.
  - **`LxmertForPreTraining`** [wiring]: Wiring.
  - **`LxmertForQuestionAnswering`** [wiring]: Wiring.

## m2m_100
- **src**: modeling_m2m_100.py
- **status**: composable
- **rationale**: Encoder-decoder MT model with sinusoidal position embed, standard BART-style multi-head attention (self + cross). Maps to whisper/bart attention family in kb-nano.
- **classes**:
  - **`M2M100ScaledWordEmbedding`** [compute]: `L1/embedding.py` (Embedding scaled by sqrt(d_model).)
  - **`M2M100SinusoidalPositionalEmbedding`** [compute]: `L1/sinusoidal_embed.py` (Fixed sinusoidal positional encoding.)
  - **`M2M100Attention`** [compute]: `L2/whisper_attention.py` (BART-style self/cross attention identical to Whisper attention pattern.)
  - **`M2M100EncoderLayer`** [wiring]: Wiring.
  - **`M2M100DecoderLayer`** [wiring]: Wiring.
  - **`M2M100Encoder`** [wiring]: Wiring.
  - **`M2M100Decoder`** [wiring]: Wiring.
  - **`M2M100Model`** [wiring]: Wiring.
  - **`M2M100ForConditionalGeneration`** [wiring]: Wiring.

## mamba
- **src**: modeling_mamba.py
- **status**: kb_nano_l4
- **rationale**: Mamba v1 selective SSM LM. Full L4 pipeline at L4/mamba.py; mixer in L2/mamba_mixer.py uses kb-nano causal_conv1d/selective_scan ops.
- **classes**:
  - **`MambaMixer`** [compute]: `L2/mamba_mixer.py`, `L1/causal_conv1d.py` (Selective SSM mixer; uses causal_conv1d_fn/_update + selective_scan.)
  - **`MambaRMSNorm`** [compute]: `L1/rms_norm.py` (Standard RMSNorm.)
  - **`MambaBlock`** [compute]: `L3/mamba_decoder.py` (Wiring: norm + mixer + residual.)
  - **`MambaModel`** [wiring]: Wiring.
  - **`MambaForCausalLM`** [wiring]: Wiring.

## mamba2
- **src**: modeling_mamba2.py
- **status**: kb_nano_l4
- **rationale**: Mamba2 SSD LM. Full L4 pipeline at L4/mamba2.py; mixer in L2/mamba2_mixer.py uses kb-nano causal_conv1d, mamba_chunk_scan, selective_state_update, rms_norm_gated.
- **classes**:
  - **`MambaRMSNormGated`** [compute]: `L1/rms_norm_gated.py` (Gated RMSNorm used by Mamba2.)
  - **`Mamba2Mixer`** [compute]: `L2/mamba2_mixer.py`, `L1/causal_conv1d.py`, `L1/rms_norm_gated.py` (SSD mixer with chunk-scan and selective state update.)
  - **`Mamba2RMSNorm`** [compute]: `L1/rms_norm.py` (Standard RMSNorm.)
  - **`Mamba2Block`** [compute]: `L3/mamba2_decoder.py` (Wiring: norm + mixer + residual.)
  - **`Mamba2Model`** [wiring]: Wiring.
  - **`Mamba2ForCausalLM`** [wiring]: Wiring.

## marian
- **src**: modeling_marian.py
- **status**: composable
- **rationale**: Marian MT (BART-style) encoder-decoder; sinusoidal pos embed + standard attention (self/cross). Same family as Whisper attention.
- **classes**:
  - **`MarianSinusoidalPositionalEmbedding`** [compute]: `L1/sinusoidal_embed.py` (Sinusoidal positional embedding.)
  - **`MarianAttention`** [compute]: `L2/whisper_attention.py` (BART-style multi-head attention.)
  - **`MarianEncoderLayer`** [wiring]: Wiring.
  - **`MarianDecoderLayer`** [wiring]: Wiring.
  - **`MarianEncoder`** [wiring]: Wiring.
  - **`MarianDecoder`** [wiring]: Wiring.
  - **`MarianModel`** [wiring]: Wiring.
  - **`MarianMTModel`** [wiring]: Wiring.
  - **`MarianDecoderWrapper`** [wiring]: Wiring.
  - **`MarianForCausalLM`** [wiring]: Wiring.

## markuplm
- **src**: modeling_markuplm.py
- **status**: composable
- **rationale**: BERT-style encoder with extra XPath embedding; standard self-attention + GELU MLP. All ops map to encoder primitives.
- **classes**:
  - **`XPathEmbeddings`** [compute]: `L1/embedding.py`, `L1/linear.py` (Embedding lookup + Linear projection per XPath component.)
  - **`MarkupLMEmbeddings`** [compute]: `L2/encoder_embeddings.py`, `L1/embedding.py` (Word + position + token-type + xpath embeddings + LayerNorm.)
  - **`MarkupLMSelfOutput`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (Linear + LayerNorm residual.)
  - **`MarkupLMIntermediate`** [compute]: `L2/encoder_mlp.py` (fc1 + GELU.)
  - **`MarkupLMOutput`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (fc2 + LayerNorm residual.)
  - **`MarkupLMPooler`** [wiring]: Wiring.
  - **`MarkupLMPredictionHeadTransform`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (Linear + activation + LayerNorm.)
  - **`MarkupLMLMPredictionHead`** [wiring]: Wiring.
  - **`MarkupLMOnlyMLMHead`** [wiring]: Wiring.
  - **`MarkupLMSelfAttention`** [compute]: `L2/encoder_attention.py` (BERT-style QKV attention with sdpa dispatch.)
  - **`MarkupLMAttention`** [wiring]: Wiring around SelfAttention + SelfOutput.
  - **`MarkupLMLayer`** [wiring]: Wiring.
  - **`MarkupLMEncoder`** [wiring]: Wiring.
  - **`MarkupLMModel`** [wiring]: Wiring.
  - **`MarkupLMForQuestionAnswering`** [wiring]: Wiring.
  - **`MarkupLMForTokenClassification`** [wiring]: Wiring.
  - **`MarkupLMForSequenceClassification`** [wiring]: Wiring.

## maskformer
- **src**: modeling_maskformer.py
- **status**: partial
- **rationale**: Instance-segmentation model with bespoke FPN pixel decoder, DETR-style decoder, Hungarian matcher, dice/focal losses, and a small mask-head ConvNet — many components have no kb-nano kernel.
- **classes**:
  - **`MaskFormerDetrDecoderLayer`** [compute]: Instance-segmentation model with bespoke FPN pixel decoder, DETR-style decoder, Hungarian matcher, dice/focal losses, and a small mask-head ConvNet — many components have no kb-nano kernel.
  - **`MaskFormerDetrLearnedPositionEmbedding`** [compute]: `L1/embedding.py` (Two embedding tables for x/y.)
  - **`MaskFormerDetrSelfAttention`** [compute]: `L2/whisper_attention.py` (Standard MHA with positional bias added.)
  - **`MaskFormerDetrCrossAttention`** [compute]: `L2/whisper_attention.py` (Standard MHA cross-attention.)
  - **`MaskFormerDetrMLP`** [compute]: `L2/encoder_mlp.py` (Linear -> activation -> Linear.)
  - **`MaskFormerDetrConvBlock`** [compute]: `L1/conv2d.py`, `L1/group_norm.py` (Conv2d + GroupNorm.)
  - **`MaskFormerDetrFPNFusionStage`** [compute]: `L1/conv2d.py`, `L1/interpolate.py` (1x1 Conv adapter + bilinear upsample + sum.)
  - **`MaskFormerDetrMaskHeadSmallConv`** [wiring]: Bespoke conv stack with multi-head attention map; no kb-nano kernel.
  - **`MaskFormerDetrMHAttentionMap`** [wiring]: Bespoke 2D attention map producing per-query attention masks.
  - **`MaskFormerDetrDecoder`** [wiring]: Wiring.
  - **`MaskFormerHungarianMatcher`** [wiring]: Bipartite matcher (scipy.optimize.linear_sum_assignment); not a GPU kernel.
  - **`MaskFormerLoss`** [wiring]: Dice + focal loss aggregator.
  - **`MaskFormerFPNConvLayer`** [compute]: `L1/conv2d.py`, `L1/group_norm.py` (Conv2d + GroupNorm + ReLU.)
  - **`MaskFormerFPNLayer`** [wiring]: Wiring.
  - **`MaskFormerFPNModel`** [wiring]: Wiring.
  - **`MaskFormerPixelDecoder`** [wiring]: Wiring (FPN + 3x3 mask projection).
  - **`MaskFormerSinePositionEmbedding`** [wiring]: Bespoke 2D sine pos-embed.
  - **`PredictionBlock`** [wiring]: Wiring.
  - **`MaskformerMLPPredictionHead`** [wiring]: Stacked Linear+ReLU mask predictor; no kb-nano kernel.
  - **`MaskFormerPixelLevelModule`** [wiring]: Wiring.
  - **`MaskFormerTransformerModule`** [wiring]: Wiring.
  - **`MaskFormerModel`** [wiring]: Wiring.
  - **`MaskFormerForInstanceSegmentation`** [wiring]: Wiring.

## maskformer_swin
- **src**: modeling_maskformer_swin.py
- **status**: partial
- **rationale**: Original Swin V1 backbone (relative-position-bias window attention with shifted windows, drop-path, patch merging) — kb-nano only has Swin V2 (cosine attention with continuous position bias), which is a different attention formulation.
- **classes**:
  - **`MaskFormerSwinSelfAttention`** [compute]: no kb-nano kernel — Original Swin V1 backbone (relative-position-bias window attention with shifted windows, drop-path, patch merging) — kb-nano only has Swin V2 (cosine attention with continuous position bias), which is
  - **`MaskFormerSwinEmbeddings`** [compute]: `L1/conv2d.py`, `L1/layer_norm.py` (Patch embed via Conv2d + LayerNorm.)
  - **`MaskFormerSwinPatchEmbeddings`** [compute]: `L1/conv2d.py` (Conv2d patch embed.)
  - **`MaskFormerSwinPatchMerging`** [wiring]: Concat 4 patches + LayerNorm + Linear (Swin V1 variant).
  - **`MaskFormerSwinDropPath`** [wiring]: Stochastic depth (drop-path); no kb-nano kernel.
  - **`MaskFormerSwinSelfOutput`** [compute]: `L1/linear.py` (Linear + dropout.)
  - **`MaskFormerSwinAttention`** [wiring]: Wiring.
  - **`MaskFormerSwinIntermediate`** [compute]: `L1/linear.py`, `L1/gelu.py` (fc1 + GELU.)
  - **`MaskFormerSwinOutput`** [compute]: `L1/linear.py` (fc2 + dropout.)
  - **`MaskFormerSwinLayer`** [wiring]: Wiring with shifted-window cyclic shift.
  - **`MaskFormerSwinStage`** [wiring]: Wiring.
  - **`MaskFormerSwinEncoder`** [wiring]: Wiring.
  - **`MaskFormerSwinModel`** [wiring]: Wiring.
  - **`MaskFormerSwinBackbone`** [wiring]: Wiring.

## mbart
- **src**: modeling_mbart.py
- **status**: composable
- **rationale**: BART-style multilingual encoder-decoder. Learned positional embedding, scaled word embedding, standard self/cross attention.
- **classes**:
  - **`MBartLearnedPositionalEmbedding`** [compute]: `L1/embedding.py` (Learned position embedding.)
  - **`MBartScaledWordEmbedding`** [compute]: `L1/embedding.py` (Embedding scaled by sqrt(d_model).)
  - **`MBartAttention`** [compute]: `L2/whisper_attention.py` (BART-style multi-head attention.)
  - **`MBartEncoderLayer`** [wiring]: Wiring.
  - **`MBartDecoderLayer`** [wiring]: Wiring.
  - **`MBartClassificationHead`** [wiring]: Wiring.
  - **`MBartEncoder`** [wiring]: Wiring.
  - **`MBartDecoder`** [wiring]: Wiring.
  - **`MBartModel`** [wiring]: Wiring.
  - **`MBartForConditionalGeneration`** [wiring]: Wiring.
  - **`MBartForSequenceClassification`** [wiring]: Wiring.
  - **`MBartForQuestionAnswering`** [wiring]: Wiring.
  - **`MBartDecoderWrapper`** [wiring]: Wiring.
  - **`MBartForCausalLM`** [wiring]: Wiring.

## megatron_bert
- **src**: modeling_megatron_bert.py
- **status**: composable
- **rationale**: BERT-derivative with Megatron's pre-LayerNorm placement. Standard QKV self-attention + GELU MLP — encoder kernels apply.
- **classes**:
  - **`MegatronBertEmbeddings`** [compute]: `L2/encoder_embeddings.py` (Word + position + token-type + LayerNorm.)
  - **`MegatronBertSelfAttention`** [compute]: `L2/encoder_attention.py` (BERT-style QKV attention.)
  - **`MegatronBertSelfOutput`** [compute]: `L1/linear.py` (Linear + dropout (residual handled by caller).)
  - **`MegatronBertAttention`** [wiring]: Wiring.
  - **`MegatronBertIntermediate`** [compute]: `L2/encoder_mlp.py` (fc1 + GELU.)
  - **`MegatronBertOutput`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (fc2 + LayerNorm residual.)
  - **`MegatronBertLayer`** [wiring]: Wiring.
  - **`MegatronBertEncoder`** [wiring]: Wiring.
  - **`MegatronBertPooler`** [wiring]: Wiring.
  - **`MegatronBertPredictionHeadTransform`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (Linear + activation + LayerNorm.)
  - **`MegatronBertLMPredictionHead`** [wiring]: Wiring.
  - **`MegatronBertOnlyMLMHead`** [wiring]: Wiring.
  - **`MegatronBertOnlyNSPHead`** [wiring]: Wiring.
  - **`MegatronBertPreTrainingHeads`** [wiring]: Wiring.
  - **`MegatronBertModel`** [wiring]: Wiring.
  - **`MegatronBertForPreTraining`** [wiring]: Wiring.
  - **`MegatronBertForCausalLM`** [wiring]: Wiring.
  - **`MegatronBertForMaskedLM`** [wiring]: Wiring.
  - **`MegatronBertForNextSentencePrediction`** [wiring]: Wiring.
  - **`MegatronBertForSequenceClassification`** [wiring]: Wiring.
  - **`MegatronBertForMultipleChoice`** [wiring]: Wiring.
  - **`MegatronBertForTokenClassification`** [wiring]: Wiring.
  - **`MegatronBertForQuestionAnswering`** [wiring]: Wiring.

## mgp_str
- **src**: modeling_mgp_str.py
- **status**: composable
- **rationale**: Vision Transformer backbone for scene-text recognition with three A3 character/bigram/wordpiece heads. ViT backbone is composable but the A3 attention head and DropPath are bespoke.
- **classes**:
  - **`MgpstrDropPath`** [wiring]: Stochastic depth — no kb-nano kernel.
  - **`MgpstrEmbeddings`** [compute]: `L1/conv2d.py`, `L1/embedding.py` (Conv2d patch embed + cls token + position embedding.)
  - **`MgpstrMlp`** [compute]: `L2/vit_encoder_mlp.py`, `L1/gelu.py` (Two Linear + GELU.)
  - **`MgpstrAttention`** [compute]: `L2/vit_encoder_attention.py` (Standard ViT QKV attention.)
  - **`MgpstrLayer`** [wiring]: Wiring.
  - **`MgpstrEncoder`** [wiring]: Wiring.
  - **`MgpstrA3Module`** [wiring]: Bespoke conv1d attention head producing per-class outputs.
  - **`MgpstrModel`** [wiring]: Wiring.
  - **`MgpstrForSceneTextRecognition`** [wiring]: Wiring.

## mimi
- **src**: modeling_mimi.py
- **status**: partial
- **partial_reason**: MimiVectorQuantization / MimiEuclideanCodebook / MimiResidualVectorQuantizer perform nearest-neighbour codebook lookup with EMA updates; no kb-nano L1 for VQ. The Conv1d/ConvTranspose1d wrappers use nn.utils.weight_norm and a runtime padding cache; not provided as a single kb-nano kernel.
- **rationale**: Audio neural codec (encoder + transformer + decoder + residual VQ). The transformer (RoPE + GQA + SwiGLU) maps to kb-nano attention/MLP, but the convolution stack uses weight_norm and asymmetric/causal padding caches, and the residual vector-quantizer is bespoke.
- **classes**:
  - **`MimiResnetBlock`** [compute]: MimiVectorQuantization / MimiEuclideanCodebook / MimiResidualVectorQuantizer perform nearest-neighbour codebook lookup with EMA updates; no kb-nano L1 for VQ. The Conv1d/ConvTranspose1d wrappers use n
  - **`MimiConv1d`** [compute]: `L1/conv1d.py` (Conv1d with asymmetric/causal padding + weight_norm; conv kernel exists but the wrapper is bespoke.)
  - **`MimiConvTranspose1d`** [compute]: `L1/conv_transpose1d.py` (ConvTranspose1d with asymmetric/causal trim + weight_norm.)
  - **`MimiEncoder`** [wiring]: Wiring (downsample stack).
  - **`MimiLayerScale`** [wiring]: Per-channel learnable scale (gamma * x).
  - **`MimiRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (Standard NeoX RoPE.)
  - **`MimiMLP`** [compute]: `L2/llama_mlp.py`, `L1/silu_and_mul.py` (SwiGLU MLP.)
  - **`MimiAttention`** [compute]: `L2/attention.py` (GQA QKV + RoPE + attention.)
  - **`MimiFlashAttention2`** [compute]: `L2/attention.py` (FA2 dispatch wrapper.)
  - **`MimiSdpaAttention`** [compute]: `L2/attention.py` (SDPA dispatch wrapper.)
  - **`MimiTransformerLayer`** [wiring]: Wiring.
  - **`MimiTransformerModel`** [wiring]: Wiring.
  - **`MimiDecoder`** [wiring]: Wiring (upsample stack).
  - **`MimiEuclideanCodebook`** [wiring]: Bespoke EMA-updated nearest-neighbour codebook.
  - **`MimiVectorQuantization`** [wiring]: Wraps codebook encode/decode.
  - **`MimiResidualVectorQuantizer`** [wiring]: Iterative residual quantization across N codebooks.
  - **`MimiSplitResidualVectorQuantizer`** [wiring]: Semantic + acoustic split RVQ.
  - **`MimiModel`** [wiring]: Wiring.

## minicpmv4_6
- **src**: modular_minicpmv4_6.py
- **status**: composable
- **rationale**: Vision-language model: SigLIP vision encoder (composable via L2/siglip_*) + a window-attention merger that uses varlen attention + a downsample MLP, plus an Lfm2 LM. All compute primitives exist.
- **classes**:
  - **`MiniCPMV4_6VisionEmbeddings`** [compute]: `L1/conv2d.py`, `L1/embedding.py` (Conv2d patch embed + position embedding lookup.)
  - **`MiniCPMV4_6VisionMLP`** [compute]: `L2/siglip_mlp.py`, `L1/gelu.py` (Two Linear + GELU (SigLIP style).)
  - **`MiniCPMV4_6VisionAttention`** [compute]: `L2/siglip_attention.py`, `L1/flash_attn_varlen.py` (QKV with varlen attention via cu_seqlens.)
  - **`MiniCPMV4_6VisionEncoderLayer`** [wiring]: Wiring.
  - **`MiniCPMV4_6VisionEncoder`** [wiring]: Wiring.
  - **`MiniCPMV4_6ViTWindowAttentionMerger`** [compute]: `L1/flash_attn_varlen.py`, `L1/layer_norm.py`, `L1/linear.py`, `L1/gelu.py` (LayerNorm + windowed varlen attention + per-image residual MLP merger.)
  - **`MiniCPMV4_6VisionModel`** [wiring]: Wiring: embeddings + encoder + window merger insert at insert_layer_id.
  - **`MiniCPMV4_6DownsampleMLP`** [compute]: `L1/linear.py`, `L1/gelu.py` (Two Linear + GELU.)
  - **`MiniCPMV4_6Merger`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (Iterative downsample/merge with LayerNorm + Linear.)
  - **`MiniCPMV4_6Model`** [wiring]: Wiring: vision_model + merger + Lfm2 language_model.
  - **`MiniCPMV4_6ForConditionalGeneration`** [wiring]: Wiring.
