## lilt
- **src**: modeling_lilt.py
- **hidden_act**: gelu
- **status**: composable
- **classes**:
  - **`LiltTextEmbeddings`** [compute]: `L2/encoder_embeddings.py` (BERT-style word + position + token_type + LayerNorm + Dropout, with padding-aware position ids)
  - **`LiltLayoutEmbeddings`** [compute]: `L1/embedding.py + L1/linear.py + L1/layer_norm.py` (six 2D bbox embeds concatenated -> linear -> add box-position embed -> LayerNorm)
  - **`LiltSelfAttention`** [compute]: `L1/linear.py + L1/dense_attention.py` (no exact L2 match — dual-stream text+layout attention with shared softmax+matmul, sums tmp scores across streams; channel_shrink_ratio for layout heads)
  - **`LiltSelfOutput`** [compute]: `L2/encoder_attention.py` (dense + LayerNorm + residual; identical to BertSelfOutput)
  - **`LiltAttention`** [wiring]: wires `LiltSelfAttention`, `LiltSelfOutput` (×2: separate `output` and `layout_output` for the two streams)
  - **`LiltIntermediate`** [compute]: `L1/linear.py + L1/gelu.py` (BertIntermediate copy; `hidden_act=gelu`)
  - **`LiltOutput`** [compute]: `L2/encoder_attention.py` (dense + LayerNorm + residual; identical structure to BertOutput)
  - **`LiltLayer`** [wiring]: wires `LiltAttention`, `LiltIntermediate` (×2 for text+layout streams), `LiltOutput` (×2)
  - **`LiltEncoder`** [wiring]: wires `LiltLayer`
  - **`LiltPooler`** [compute]: `L1/linear.py + L1/tanh.py`
  - **`LiltModel`** [wiring]: wires `LiltTextEmbeddings`, `LiltLayoutEmbeddings`, `LiltEncoder`, optional `LiltPooler`
  - **`LiltClassificationHead`** [compute]: `L1/linear.py + L1/tanh.py + L1/linear.py`
- **task heads (3)**: ForQuestionAnswering, ForSequenceClassification, ForTokenClassification — base + linear (per-task)

## llama
- **src**: modeling_llama.py
- **hidden_act**: silu
- **status**: kb_nano_l4 (`L4/llama.py`)
- **classes**:
  - **`LlamaRMSNorm`** [compute]: `L1/rms_norm.py`
  - **`LlamaRotaryEmbedding`** [compute]: `L1/rotary_emb.py`
  - **`LlamaMLP`** [compute]: `L2/llama_mlp.py` (gate_proj * silu(gate)→ down_proj; SwiGLU)
  - **`LlamaAttention`** [compute]: `L2/attention.py` (q/k/v/o, RoPE, KV cache, GQA dispatch via ALL_ATTENTION_FUNCTIONS)
  - **`LlamaDecoderLayer`** [wiring]: wires `LlamaAttention`, `LlamaMLP`, `LlamaRMSNorm` (×2)
  - **`LlamaModel`** [wiring]: wires `LlamaDecoderLayer`, `LlamaRMSNorm`, `LlamaRotaryEmbedding`; direct `L1/embedding.py`
  - **`LlamaForCausalLM`** [wiring]: wires `LlamaModel`; direct `L1/linear.py` (lm_head)
- **task heads (3)**: ForSequenceClassification, ForQuestionAnswering, ForTokenClassification — base + linear (per-task)

## llama4
- **src**: modeling_llama4.py
- **hidden_act**: silu (text); gelu (vision)
- **status**: kb_nano_l4 (`L4/llama4.py`)
- **classes**:
  - **`Llama4TextExperts`** [compute]: `L2/llama4_moe.py` (fused experts inside Llama4MoE; bmm-based gate_up + silu + down per expert; `L1/silu.py`)
  - **`Llama4TextMLP`** [compute]: `L2/llama_mlp.py` (Phi3 SwiGLU pattern: gate_proj * silu(gate) → down_proj; used for shared expert)
  - **`Llama4TextL2Norm`** [compute]: `L1/l2_norm.py` (RMS-norm without learnable weight; QK norm)
  - **`Llama4TextRMSNorm`** [compute]: `L1/rms_norm.py`
  - **`Llama4Router`** [compute, inherits `nn.Linear`]: `L1/linear.py + L1/sigmoid_topk.py` (top-k logits scattered, then sigmoid)
  - **`Llama4TextMoe`** [compute]: `L2/llama4_moe.py` (sigmoid top-1 routing + shared expert)
  - **`Llama4TextRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (Llama-style RoPE; emits complex freqs_cis)
  - **`Llama4TextAttention`** [compute]: `L2/llama4_attention.py` (NoPE/RoPE per-layer toggle, optional weight-less QK norm, attn_temperature_tuning)
  - **`Llama4TextDecoderLayer`** [wiring]: wires `Llama4TextAttention`, `Llama4TextMoe` or `Llama4TextMLP`, `Llama4TextRMSNorm` (×2)
  - **`Llama4TextModel`** [wiring]: wires `Llama4TextDecoderLayer`, `Llama4TextRMSNorm`, `Llama4TextRotaryEmbedding`; direct `L1/embedding.py`
  - **`Llama4ForCausalLM`** [wiring]: wires `Llama4TextModel`; direct `L1/linear.py` (lm_head)
  - **`Llama4VisionMLP2`** [compute]: `L1/linear.py + L1/gelu.py + L1/linear.py + L1/gelu.py` (fc1 → GELU → fc2 → GELU; final activation applied to fc2 output)
  - **`Llama4MultiModalProjector`** [compute]: `L1/linear.py` (single linear)
  - **`Llama4VisionPixelShuffleMLP`** [wiring]: wires `Llama4VisionMLP2`; pixel_shuffle reshape
  - **`Llama4VisionAttention`** [compute]: `L2/clip_attention.py` (no exact match — bidirectional q/k/v/o with complex 2D RoPE; non-causal SDPA dispatch; no kb-nano kernel implements complex-valued vision RoPE, so structurally `L1/linear.py + L1/dense_attention.py`)
  - **`Llama4VisionMLP`** [compute]: `L1/linear.py + L1/gelu.py + L1/linear.py` (fc1 → GELU → fc2; ViT-style)
  - **`Llama4VisionEncoderLayer`** [wiring]: wires `Llama4VisionAttention`, `Llama4VisionMLP`; direct `L1/layer_norm.py` (×2)
  - **`Llama4VisionEncoder`** [wiring]: wires `Llama4VisionEncoderLayer`
  - **`Llama4UnfoldConvolution`** [compute]: `L1/linear.py` + `nn.Unfold` (unfold-then-linear patch embedding; no L1 unfold op)
  - **`Llama4VisionRotaryEmbedding`** [compute]: complex 2D vision rope (no exact L1; closest `L1/vision_rotary_emb.py` for 2D vision RoPE but uses real-valued sin/cos)
  - **`Llama4VisionModel`** [wiring]: wires `Llama4UnfoldConvolution`, `Llama4VisionEncoder`, `Llama4VisionPixelShuffleMLP`, `Llama4VisionRotaryEmbedding`; direct `L1/layer_norm.py` (×2)
  - **`Llama4ForConditionalGeneration`** [wiring]: wires `Llama4VisionModel`, `Llama4MultiModalProjector`, `Llama4ForCausalLM`

## llava
- **src**: modeling_llava.py
- **hidden_act**: gelu (projector)
- **status**: composable (vision projector + language model wiring; backbones come from `AutoModel`)
- **classes**:
  - **`LlavaMultiModalProjector`** [compute]: `L1/linear.py + L1/gelu.py + L1/linear.py`
  - **`LlavaModel`** [wiring]: wires `AutoModel(vision_config)`, `LlavaMultiModalProjector`, `AutoModel(text_config)`
  - **`LlavaForConditionalGeneration`** [wiring]: wires `LlavaModel`; direct `L1/linear.py` (lm_head)

## llava_next
- **src**: modeling_llava_next.py
- **hidden_act**: gelu (projector)
- **status**: composable
- **classes**:
  - **`LlavaNextMultiModalProjector`** [compute]: `L1/linear.py + L1/gelu.py + L1/linear.py`
  - **`LlavaNextModel`** [wiring]: wires `AutoModel(vision_config)`, `LlavaNextMultiModalProjector`, `AutoModel(text_config)`; image_newline parameter
  - **`LlavaNextForConditionalGeneration`** [wiring]: wires `LlavaNextModel`; direct `L1/linear.py` (lm_head)

## llava_next_video
- **src**: modeling_llava_next_video.py (and modular_llava_next_video.py)
- **hidden_act**: gelu (projector)
- **status**: composable
- **classes**:
  - **`LlavaNextVideoPooler`** [compute, wires `nn.AvgPool2d|nn.MaxPool2d|nn.Conv2d`]: `L1/avg_pool2d.py` or `L1/max_pool2d.py` or `L1/conv2d.py` (mode-selectable spatial pool; reshape-based)
  - **`LlavaNextVideoMultiModalProjector`** [compute, inherits `LlavaNextMultiModalProjector`]: `L1/linear.py + L1/gelu.py + L1/linear.py`
  - **`LlavaNextVideoModel`** [wiring, inherits `LlavaNextModel`]: wires `AutoModel(vision_config)`, `LlavaNextVideoMultiModalProjector`, `LlavaNextVideoPooler`, `AutoModel(text_config)`
  - **`LlavaNextVideoForConditionalGeneration`** [wiring, inherits `LlavaNextForConditionalGeneration`]: wires `LlavaNextVideoModel`; direct `L1/linear.py` (lm_head)

## llava_onevision
- **src**: modeling_llava_onevision.py (and modular_llava_onevision.py)
- **hidden_act**: gelu (projector)
- **status**: composable
- **classes**:
  - **`LlavaOnevisionMultiModalProjector`** [compute]: `L1/linear.py + L1/gelu.py + L1/linear.py`
  - **`LlavaOnevisionModel`** [wiring, inherits `LlavaNextVideoModel`]: wires `AutoModel(vision_config)`, `LlavaOnevisionMultiModalProjector`, `AutoModel(text_config)`
  - **`LlavaOnevisionForConditionalGeneration`** [wiring, inherits `LlavaNextVideoForConditionalGeneration`]: wires `LlavaOnevisionModel`; direct `L1/linear.py` (lm_head)

## longformer
- **src**: modeling_longformer.py
- **hidden_act**: gelu
- **status**: partial (sliding-window/global attention has no kb-nano kernel)
- **classes**:
  - **`LongformerEmbeddings`** [compute]: `L2/encoder_embeddings.py` (word + position + token_type + LayerNorm; padding-aware position ids)
  - **`LongformerSelfAttention`** [compute]: no kb-nano kernel — sliding-window local + global attention with separate `query_global`/`key_global`/`value_global` projections, custom chunked overlap matmul; structurally `L1/linear.py + L1/softmax.py` (no exact L1/L2 for sliding-window pattern)
  - **`LongformerSelfOutput`** [compute]: `L2/encoder_attention.py` (BertSelfOutput-shaped: dense + LayerNorm + residual)
  - **`LongformerAttention`** [wiring]: wires `LongformerSelfAttention`, `LongformerSelfOutput`
  - **`LongformerIntermediate`** [compute]: `L1/linear.py + L1/gelu.py`
  - **`LongformerOutput`** [compute]: `L2/encoder_attention.py` (BertOutput shape: dense + LayerNorm + residual)
  - **`LongformerLayer`** [wiring]: wires `LongformerAttention`, `LongformerIntermediate`, `LongformerOutput`
  - **`LongformerEncoder`** [wiring]: wires `LongformerLayer`
  - **`LongformerPooler`** [compute]: `L1/linear.py + L1/tanh.py`
  - **`LongformerLMHead`** [compute]: `L1/linear.py + L1/gelu.py + L1/layer_norm.py + L1/linear.py`
  - **`LongformerModel`** [wiring]: wires `LongformerEmbeddings`, `LongformerEncoder`, optional `LongformerPooler`
  - **`LongformerForMaskedLM`** [wiring]: wires `LongformerModel`, `LongformerLMHead`
  - **`LongformerClassificationHead`** [compute]: `L1/linear.py + L1/tanh.py + L1/linear.py`
- **task heads (4)**: ForMultipleChoice, ForQuestionAnswering, ForSequenceClassification, ForTokenClassification — base + linear (per-task)

## longt5
- **src**: modeling_longt5.py
- **hidden_act**: relu (default `feed_forward_proj="relu"`; `dense_act_fn` resolved from this)
- **status**: partial (T5-shape ops covered; local + transient-global attention have no kb-nano kernel)
- **classes**:
  - **`LongT5LayerNorm`** [compute]: `L1/t5_layer_norm.py` (RMS-style, no centering/bias)
  - **`LongT5DenseActDense`** [compute]: `L2/t5_dense.py` (no exact match — non-gated 2-layer FFN; closest is `L1/linear.py + L1/relu.py + L1/linear.py`)
  - **`LongT5DenseGatedActDense`** [compute]: `L2/t5_dense.py` (gated wi_0 * wi_1 with relu/gelu)
  - **`LongT5LayerFF`** [wiring]: wires `LongT5DenseGatedActDense` or `LongT5DenseActDense`, `LongT5LayerNorm`
  - **`LongT5Attention`** [compute]: `L2/t5_attention.py` (T5 self/cross attention with relative_attention_bias)
  - **`LongT5LocalAttention`** [compute]: no kb-nano kernel — block-local windowed attention with relative-position bias; structurally `L1/linear.py + L1/softmax.py`
  - **`LongT5TransientGlobalAttention`** [compute]: no kb-nano kernel — local windows + transient global tokens with side bias; structurally `L1/linear.py + L1/softmax.py`
  - **`LongT5LayerSelfAttention`** [wiring]: wires `LongT5Attention`, `LongT5LayerNorm` (decoder self-attn variant)
  - **`LongT5LayerLocalSelfAttention`** [wiring]: wires `LongT5LocalAttention`, `LongT5LayerNorm`
  - **`LongT5LayerTransientGlobalSelfAttention`** [wiring]: wires `LongT5TransientGlobalAttention`, `LongT5LayerNorm`
  - **`LongT5LayerCrossAttention`** [wiring]: wires `LongT5Attention`, `LongT5LayerNorm`
  - **`LongT5Block`** [wiring]: wires `LongT5LayerSelfAttention` or `LongT5LayerLocalSelfAttention` or `LongT5LayerTransientGlobalSelfAttention`, optional `LongT5LayerCrossAttention`, `LongT5LayerFF`
  - **`LongT5Stack`** [wiring]: wires `LongT5Block`, `LongT5LayerNorm`; direct `L1/embedding.py`
  - **`LongT5Model`** [wiring]: wires `LongT5Stack` (×2: encoder + decoder); direct `L1/embedding.py` (shared)
  - **`LongT5ForConditionalGeneration`** [wiring]: wires `LongT5Stack` (×2); direct `L1/embedding.py` (shared), `L1/linear.py` (lm_head)
  - **`LongT5EncoderModel`** [wiring]: wires `LongT5Stack` (encoder only); direct `L1/embedding.py`

## luke
- **src**: modeling_luke.py
- **hidden_act**: gelu
- **status**: partial (entity-aware attention with 4× query streams is bespoke; no exact kb-nano kernel)
- **classes**:
  - **`LukeEmbeddings`** [compute]: `L2/encoder_embeddings.py` (word + position + token_type + LayerNorm; padding-aware position ids)
  - **`LukeEntityEmbeddings`** [compute]: `L1/embedding.py + L1/linear.py + L1/layer_norm.py` (entity_embed → optional dense → + position-mean + token_type → LayerNorm)
  - **`LukeSelfAttention`** [compute]: no kb-nano kernel — entity-aware: word+entity concat for K/V, four parallel Q projections (w2w/w2e/e2w/e2e); structurally `L1/linear.py + L1/softmax.py + L1/linear.py`
  - **`LukeSelfOutput`** [compute]: `L2/encoder_attention.py` (BertSelfOutput-shaped)
  - **`LukeAttention`** [wiring]: wires `LukeSelfAttention`, `LukeSelfOutput`; concat/split for word+entity streams
  - **`LukeIntermediate`** [compute]: `L1/linear.py + L1/gelu.py`
  - **`LukeOutput`** [compute]: `L2/encoder_attention.py` (BertOutput-shaped)
  - **`LukeLayer`** [wiring]: wires `LukeAttention`, `LukeIntermediate`, `LukeOutput`
  - **`LukeEncoder`** [wiring]: wires `LukeLayer`
  - **`LukePooler`** [compute]: `L1/linear.py + L1/tanh.py`
  - **`EntityPredictionHeadTransform`** [compute]: `L1/linear.py + L1/gelu.py + L1/layer_norm.py`
  - **`EntityPredictionHead`** [wiring]: wires `EntityPredictionHeadTransform`; direct `L1/linear.py`
  - **`LukeModel`** [wiring]: wires `LukeEmbeddings`, `LukeEntityEmbeddings`, `LukeEncoder`, `LukePooler`
  - **`LukeLMHead`** [compute]: `L1/linear.py + L1/gelu.py + L1/layer_norm.py + L1/linear.py`
  - **`LukeForMaskedLM`** [wiring]: wires `LukeModel`, `LukeLMHead`, `EntityPredictionHead`
- **task heads (8)**: ForEntityClassification, ForEntityPairClassification, ForEntitySpanClassification, ForMultipleChoice, ForQuestionAnswering, ForSequenceClassification, ForTokenClassification — base + linear (per-task)

## lxmert
- **src**: modeling_lxmert.py
- **hidden_act**: gelu
- **status**: partial (lang/vision/cross self+cross attention all share `LxmertAttention` shape; no exact L2 BERT-shape match because cross-attn is mid-stack)
- **classes**:
  - **`GeLU`** [compute]: `L1/gelu.py`
  - **`LxmertEmbeddings`** [compute]: `L2/encoder_embeddings.py` (word + position + token_type + LayerNorm)
  - **`LxmertAttention`** [compute]: `L1/linear.py + L1/dense_attention.py` (q/k/v with separate ctx_dim for cross-attn; standard scaled dot-product softmax; no exact L2 because BERT-style needs same-input QKV)
  - **`LxmertAttentionOutput`** [compute]: `L2/encoder_attention.py` (BertSelfOutput-shaped: dense + LayerNorm + residual)
  - **`LxmertCrossAttentionLayer`** [wiring]: wires `LxmertAttention`, `LxmertAttentionOutput` (cross-attn between modalities)
  - **`LxmertSelfAttentionLayer`** [wiring]: wires `LxmertAttention`, `LxmertAttentionOutput`
  - **`LxmertIntermediate`** [compute]: `L1/linear.py + L1/gelu.py`
  - **`LxmertOutput`** [compute]: `L2/encoder_attention.py` (BertOutput-shaped)
  - **`LxmertLayer`** [wiring]: wires `LxmertSelfAttentionLayer`, `LxmertIntermediate`, `LxmertOutput`
  - **`LxmertXLayer`** [wiring]: wires `LxmertCrossAttentionLayer`, `LxmertSelfAttentionLayer` (×2: lang+visn), `LxmertIntermediate` (×2), `LxmertOutput` (×2)
  - **`LxmertVisualFeatureEncoder`** [compute]: `L1/linear.py + L1/layer_norm.py + L1/linear.py + L1/layer_norm.py` (visual+pos features summed and dropped)
  - **`LxmertEncoder`** [wiring]: wires `LxmertVisualFeatureEncoder`, `LxmertLayer` (l_layers + r_layers), `LxmertXLayer` (x_layers)
  - **`LxmertPooler`** [compute]: `L1/linear.py + L1/tanh.py`
  - **`LxmertPredictionHeadTransform`** [compute]: `L1/linear.py + L1/gelu.py + L1/layer_norm.py`
  - **`LxmertLMPredictionHead`** [wiring]: wires `LxmertPredictionHeadTransform`; direct `L1/linear.py`
  - **`LxmertVisualAnswerHead`** [compute]: `L1/linear.py + L1/gelu.py + L1/layer_norm.py + L1/linear.py` (Sequential MLP)
  - **`LxmertVisualObjHead`** [wiring]: wires `LxmertPredictionHeadTransform`; direct `L1/linear.py` (one per visual loss key)
  - **`LxmertPreTrainingHeads`** [wiring]: wires `LxmertLMPredictionHead`; direct `L1/linear.py` (seq_relationship)
  - **`LxmertModel`** [wiring]: wires `LxmertEmbeddings`, `LxmertEncoder`, `LxmertPooler`
- **task heads (2)**: ForPreTraining, ForQuestionAnswering — base + linear (per-task)

## m2m_100
- **src**: modeling_m2m_100.py
- **hidden_act**: relu (`activation_function`)
- **status**: partial (encoder/decoder with cross-attn; standard BART shape — kb-nano lacks an encoder-decoder text engine)
- **classes**:
  - **`M2M100ScaledWordEmbedding`** [compute, inherits `nn.Embedding`]: `L1/embedding.py` (× embed_scale)
  - **`M2M100SinusoidalPositionalEmbedding`** [compute]: `L1/sinusoidal_embed.py + L1/embedding.py` (no exact match — sinusoidal weights with index_select)
  - **`M2M100Attention`** [compute]: `L2/clip_attention.py` (no exact match — BART-style q/k/v/o with KV cache, encoder-decoder cache support; closest is multi-head attn with `L1/linear.py + L1/dense_attention.py`)
  - **`M2M100EncoderLayer`** [wiring]: wires `M2M100Attention`; direct `L1/layer_norm.py` (×2), `L1/linear.py` (fc1/fc2), `L1/relu.py`
  - **`M2M100DecoderLayer`** [wiring]: wires `M2M100Attention` (×2: self + encoder cross); direct `L1/layer_norm.py` (×3), `L1/linear.py` (fc1/fc2), `L1/relu.py`
  - **`M2M100Encoder`** [wiring]: wires `M2M100ScaledWordEmbedding`, `M2M100SinusoidalPositionalEmbedding`, `M2M100EncoderLayer`; direct `L1/layer_norm.py`
  - **`M2M100Decoder`** [wiring]: wires `M2M100ScaledWordEmbedding`, `M2M100SinusoidalPositionalEmbedding`, `M2M100DecoderLayer`; direct `L1/layer_norm.py`
  - **`M2M100Model`** [wiring]: wires `M2M100Encoder`, `M2M100Decoder`; direct `L1/embedding.py` (shared)
  - **`M2M100ForConditionalGeneration`** [wiring]: wires `M2M100Model`; direct `L1/linear.py` (lm_head)

## mamba
- **src**: modeling_mamba.py
- **hidden_act**: silu
- **status**: kb_nano_l4 (`L4/mamba.py`)
- **classes**:
  - **`MambaMixer`** [compute]: `L2/mamba_mixer.py` (selective SSM with conv1d + in_proj + x_proj/dt_proj + selective_scan/selective_state_update; uses `L1/causal_conv1d.py`, `L1/silu.py`)
  - **`MambaRMSNorm`** [compute]: `L1/rms_norm.py`
  - **`MambaBlock`** [wiring]: wires `MambaMixer`, `MambaRMSNorm`; residual add
  - **`MambaModel`** [wiring]: wires `MambaBlock`, `MambaRMSNorm`; direct `L1/embedding.py`
  - **`MambaForCausalLM`** [wiring]: wires `MambaModel`; direct `L1/linear.py` (lm_head)

## mamba2
- **src**: modeling_mamba2.py
- **hidden_act**: silu
- **status**: kb_nano_l4 (`L4/mamba2.py`)
- **classes**:
  - **`MambaRMSNormGated`** [compute]: `L1/rms_norm_gated.py` (RMSNorm with optional silu(gate) modulation)
  - **`Mamba2Mixer`** [compute]: `L2/mamba2_mixer.py` (SSD/Mamba2 mixer: conv1d + in_proj for [z, x, B, C, dt] + chunk_scan_combined + RMSNormGated; uses `L1/causal_conv1d.py`, `L1/silu.py`)
  - **`Mamba2RMSNorm`** [compute]: `L1/rms_norm.py`
  - **`Mamba2Block`** [wiring]: wires `Mamba2Mixer`, `Mamba2RMSNorm`; residual add
  - **`Mamba2Model`** [wiring]: wires `Mamba2Block`, `Mamba2RMSNorm`; direct `L1/embedding.py`
  - **`Mamba2ForCausalLM`** [wiring]: wires `Mamba2Model`; direct `L1/linear.py` (lm_head)

## marian
- **src**: modeling_marian.py
- **hidden_act**: gelu (`activation_function`)
- **status**: partial (BART-shape encoder-decoder; no kb-nano LM-translation engine)
- **classes**:
  - **`MarianSinusoidalPositionalEmbedding`** [compute, inherits `nn.Embedding`]: `L1/sinusoidal_embed.py + L1/embedding.py` (frozen sin/cos table; non-interleaved sin then cos halves)
  - **`MarianAttention`** [compute]: `L1/linear.py + L1/dense_attention.py` (BART-style q/k/v/out with optional encoder-decoder cache; no exact L2 match)
  - **`MarianEncoderLayer`** [wiring]: wires `MarianAttention`; direct `L1/layer_norm.py` (×2), `L1/linear.py` (fc1/fc2), `L1/gelu.py`
  - **`MarianDecoderLayer`** [wiring]: wires `MarianAttention` (×2: self + encoder cross); direct `L1/layer_norm.py` (×3), `L1/linear.py` (fc1/fc2), `L1/gelu.py`
  - **`MarianEncoder`** [wiring]: wires `MarianSinusoidalPositionalEmbedding`, `MarianEncoderLayer`; direct `L1/embedding.py` (× embed_scale)
  - **`MarianDecoder`** [wiring]: wires `MarianSinusoidalPositionalEmbedding`, `MarianDecoderLayer`; direct `L1/embedding.py` (× embed_scale)
  - **`MarianModel`** [wiring]: wires `MarianEncoder`, `MarianDecoder`; direct `L1/embedding.py` (shared/separate)
  - **`MarianMTModel`** [wiring]: wires `MarianModel`; direct `L1/linear.py` (lm_head, optional bias)
  - **`MarianDecoderWrapper`** [wiring]: wires `MarianDecoder`
  - **`MarianForCausalLM`** [wiring]: wires `MarianDecoderWrapper`; direct `L1/linear.py` (lm_head)

## markuplm
- **src**: modeling_markuplm.py
- **hidden_act**: gelu
- **status**: composable
- **classes**:
  - **`XPathEmbeddings`** [compute]: `L1/embedding.py + L1/linear.py + L1/relu.py + L1/linear.py + L1/linear.py` (xpath tag + subs embeds → unitseq2_inner → ReLU → inner2emb)
  - **`MarkupLMEmbeddings`** [wiring + compute]: wires `XPathEmbeddings`; direct `L1/embedding.py` (×3 word/position/token_type) + `L1/layer_norm.py`
  - **`MarkupLMSelfOutput`** [compute]: `L2/encoder_attention.py` (BertSelfOutput-shaped)
  - **`MarkupLMIntermediate`** [compute]: `L1/linear.py + L1/gelu.py`
  - **`MarkupLMOutput`** [compute]: `L2/encoder_attention.py` (BertOutput-shaped)
  - **`MarkupLMPooler`** [compute]: `L1/linear.py + L1/tanh.py`
  - **`MarkupLMPredictionHeadTransform`** [compute]: `L1/linear.py + L1/gelu.py + L1/layer_norm.py`
  - **`MarkupLMLMPredictionHead`** [wiring]: wires `MarkupLMPredictionHeadTransform`; direct `L1/linear.py`
  - **`MarkupLMOnlyMLMHead`** [wiring]: wires `MarkupLMLMPredictionHead`
  - **`MarkupLMSelfAttention`** [compute]: `L2/encoder_attention.py` (q/k/v + dispatch via ALL_ATTENTION_FUNCTIONS; same as Align/Bert pattern)
  - **`MarkupLMAttention`** [wiring]: wires `MarkupLMSelfAttention`, `MarkupLMSelfOutput`
  - **`MarkupLMLayer`** [wiring]: wires `MarkupLMAttention`, `MarkupLMIntermediate`, `MarkupLMOutput`
  - **`MarkupLMEncoder`** [wiring]: wires `MarkupLMLayer`
  - **`MarkupLMModel`** [wiring]: wires `MarkupLMEmbeddings`, `MarkupLMEncoder`, optional `MarkupLMPooler`
- **task heads (3)**: ForQuestionAnswering, ForSequenceClassification, ForTokenClassification — base + linear (per-task)

## maskformer
- **src**: modeling_maskformer.py (and modular_maskformer.py); also embeds modeling_maskformer_swin.py for the Swin backbone
- **hidden_act**: relu (`activation_function` for DETR decoder); gelu (Swin backbone `hidden_act`)
- **status**: partial (DETR-style decoder cross-attention with object-query pos embeds; mask head uses Conv2d/GroupNorm; no kb-nano L4 for MaskFormer)
- **classes**:
  - **`MaskFormerDetrLearnedPositionEmbedding`** [compute]: `L1/embedding.py` (×2: row/col) + cat/permute (no L1 for joint 2D learned pos)
  - **`MaskFormerDetrSelfAttention`** [compute]: `L1/linear.py + L1/dense_attention.py` (no exact L2 — DETR-style: pos embed added to Q+K only, not V)
  - **`MaskFormerDetrCrossAttention`** [compute]: `L1/linear.py + L1/dense_attention.py` (no exact L2 — query gets one pos embed, key gets encoder pos embed, value none)
  - **`MaskFormerDetrMLP`** [compute]: `L1/linear.py + L1/relu.py + L1/linear.py` (2-layer FFN)
  - **`MaskFormerDetrDecoderLayer`** [wiring]: wires `MaskFormerDetrSelfAttention`, `MaskFormerDetrCrossAttention`, `MaskFormerDetrMLP`; direct `L1/layer_norm.py` (×3)
  - **`MaskFormerDetrConvBlock`** [compute]: `L1/conv2d.py + L1/group_norm.py + L1/relu.py`
  - **`MaskFormerDetrFPNFusionStage`** [wiring]: wires `MaskFormerDetrConvBlock`; direct `L1/conv2d.py` (1×1 fpn_adapter), `L1/interpolate.py`
  - **`MaskFormerDetrMaskHeadSmallConv`** [wiring]: wires `MaskFormerDetrConvBlock` (×2), `MaskFormerDetrFPNFusionStage` (×3); direct `L1/conv2d.py` (output_conv)
  - **`MaskFormerDetrMHAttentionMap`** [compute]: `L1/linear.py + L1/conv2d.py + L1/softmax.py` (q_proj on tokens, k_proj as 1×1 conv on spatial keys; returns attention map only)
  - **`MaskFormerDetrDecoder`** [wiring]: wires `MaskFormerDetrDecoderLayer`; direct `L1/layer_norm.py`
  - **`MaskFormerHungarianMatcher`** [compute]: bipartite matching loss helper (CPU; not a kernel — autograd loss component)
  - **`MaskFormerLoss`** [compute]: loss module (not a kernel)
  - **`MaskFormerFPNConvLayer`** [compute]: `L1/conv2d.py + L1/group_norm.py + L1/relu.py`
  - **`MaskFormerFPNLayer`** [wiring]: wires `MaskFormerFPNConvLayer`; direct `L1/conv2d.py` (1×1 proj), `L1/group_norm.py`, `L1/interpolate.py`
  - **`MaskFormerFPNModel`** [wiring]: wires `MaskFormerFPNConvLayer` (stem), `MaskFormerFPNLayer`
  - **`MaskFormerPixelDecoder`** [wiring]: wires `MaskFormerFPNModel`; direct `L1/conv2d.py` (mask_projection)
  - **`MaskFormerSinePositionEmbedding`** [compute]: `L1/sinusoidal_embed.py` (no exact match — 2D sin/cos position; cumsum-based)
  - **`PredictionBlock`** [wiring]: wires linear + activation
  - **`MaskformerMLPPredictionHead`** [wiring]: wires `PredictionBlock` (× num_layers); ReLU between layers
  - **`MaskFormerPixelLevelModule`** [wiring]: wires `AutoBackbone(backbone_config)`, `MaskFormerPixelDecoder`
  - **`MaskFormerTransformerModule`** [wiring]: wires `MaskFormerSinePositionEmbedding`, `MaskFormerDetrDecoder`; direct `L1/embedding.py` (queries), optional `L1/conv2d.py` (input_projection)
  - **`MaskFormerModel`** [wiring]: wires `MaskFormerPixelLevelModule`, `MaskFormerTransformerModule`
- **task heads (1)**: ForInstanceSegmentation — base + Maskformer heads (per-task)

## maskformer_swin
- **src**: modeling_maskformer_swin.py (lives inside `maskformer/`; modeled separately under its own config)
- **hidden_act**: gelu
- **status**: partial (Swin-style windowed attention with relative position bias; no kb-nano kernel for shifted-window MSA, but L2/swinv2_window_attention.py is closest)
- **classes**:
  - **`MaskFormerSwinEmbeddings`** [wiring + compute]: wires `MaskFormerSwinPatchEmbeddings`; direct `L1/layer_norm.py`, optional learned 2D pos embeds, `L1/interpolate.py` (bicubic for variable-resolution)
  - **`MaskFormerSwinPatchEmbeddings`** [compute]: `L1/conv2d.py` (kernel=stride=patch_size)
  - **`MaskFormerSwinPatchMerging`** [compute]: `L1/linear.py + L1/layer_norm.py` (4× channel concat then 2× projection)
  - **`MaskFormerSwinDropPath`** [compute]: stochastic depth (L1 dropout-like; `L1/dropout.py` not exact)
  - **`MaskFormerSwinSelfAttention`** [compute]: no kb-nano kernel — relative-position-bias windowed self-attn; closest is `L2/swinv2_window_attention.py` for the shifted-window pattern
  - **`MaskFormerSwinSelfOutput`** [compute]: `L1/linear.py` (no LayerNorm here; LayerNorm moved to layer)
  - **`MaskFormerSwinAttention`** [wiring]: wires `MaskFormerSwinSelfAttention`, `MaskFormerSwinSelfOutput`
  - **`MaskFormerSwinIntermediate`** [compute]: `L1/linear.py + L1/gelu.py`
  - **`MaskFormerSwinOutput`** [compute]: `L1/linear.py` (no residual here)
  - **`MaskFormerSwinLayer`** [wiring]: wires `MaskFormerSwinAttention`, `MaskFormerSwinIntermediate`, `MaskFormerSwinOutput`, optional `MaskFormerSwinDropPath`; direct `L1/layer_norm.py` (×2)
  - **`MaskFormerSwinStage`** [wiring]: wires `MaskFormerSwinLayer`, optional `MaskFormerSwinPatchMerging` downsample
  - **`MaskFormerSwinEncoder`** [wiring]: wires `MaskFormerSwinStage`
  - **`MaskFormerSwinModel`** [wiring]: wires `MaskFormerSwinEmbeddings`, `MaskFormerSwinEncoder`
  - **`MaskFormerSwinBackbone`** [wiring, inherits `BackboneMixin`, `MaskFormerSwinPreTrainedModel`]: backbone wrapper around `MaskFormerSwinModel`

## mbart
- **src**: modeling_mbart.py
- **hidden_act**: gelu (`activation_function`)
- **status**: partial (BART-shape encoder-decoder with separate pre/post layer norms; no kb-nano LM-translation engine)
- **classes**:
  - **`MBartLearnedPositionalEmbedding`** [compute, inherits `nn.Embedding`]: `L1/embedding.py` (with +offset=2 padding shift)
  - **`MBartScaledWordEmbedding`** [compute, inherits `nn.Embedding`]: `L1/embedding.py` (× embed_scale)
  - **`MBartAttention`** [compute]: `L1/linear.py + L1/dense_attention.py` (BART q/k/v/out with optional encoder-decoder cache; no exact L2 match)
  - **`MBartEncoderLayer`** [wiring]: wires `MBartAttention`; direct `L1/layer_norm.py` (×2: pre-attn + post-attn), `L1/linear.py` (fc1/fc2), `L1/gelu.py`
  - **`MBartDecoderLayer`** [wiring]: wires `MBartAttention` (×2), direct `L1/layer_norm.py` (×3), `L1/linear.py` (fc1/fc2), `L1/gelu.py`
  - **`MBartClassificationHead`** [compute]: `L1/linear.py + L1/tanh.py + L1/linear.py`
  - **`MBartEncoder`** [wiring]: wires `MBartScaledWordEmbedding`, `MBartLearnedPositionalEmbedding`, `MBartEncoderLayer`; direct `L1/layer_norm.py` (×2)
  - **`MBartDecoder`** [wiring]: wires `MBartScaledWordEmbedding`, `MBartLearnedPositionalEmbedding`, `MBartDecoderLayer`; direct `L1/layer_norm.py` (×2)
  - **`MBartModel`** [wiring]: wires `MBartEncoder`, `MBartDecoder`; direct `L1/embedding.py` (shared)
  - **`MBartForConditionalGeneration`** [wiring]: wires `MBartModel`; direct `L1/linear.py` (lm_head)
  - **`MBartDecoderWrapper`** [wiring]: wires `MBartDecoder`
  - **`MBartForCausalLM`** [wiring]: wires `MBartDecoderWrapper`; direct `L1/linear.py` (lm_head)
- **task heads (2)**: ForQuestionAnswering, ForSequenceClassification — base + linear (per-task)

## megatron_bert
- **src**: modeling_megatron_bert.py
- **hidden_act**: gelu
- **status**: composable (BERT shape with LayerNorm moved into the layer)
- **classes**:
  - **`MegatronBertEmbeddings`** [compute]: `L2/encoder_embeddings.py` (word + position + token_type; no LayerNorm here — LN is moved to per-layer)
  - **`MegatronBertSelfAttention`** [compute]: `L2/encoder_attention.py` (q/k/v with optional encoder-decoder cache for cross-attn; BERT-shaped)
  - **`MegatronBertSelfOutput`** [compute]: `L1/linear.py` + residual (no LayerNorm — moved to MegatronBertAttention.ln)
  - **`MegatronBertAttention`** [wiring]: wires `MegatronBertSelfAttention`, `MegatronBertSelfOutput`; direct `L1/layer_norm.py` (pre-attn ln, Megatron-shape)
  - **`MegatronBertIntermediate`** [compute]: `L1/linear.py + L1/gelu.py`
  - **`MegatronBertOutput`** [compute]: `L1/linear.py` + residual (no LayerNorm — moved to MegatronBertLayer.ln)
  - **`MegatronBertLayer`** [wiring]: wires `MegatronBertAttention`, `MegatronBertIntermediate`, `MegatronBertOutput`, optional `MegatronBertAttention` (cross); direct `L1/layer_norm.py` (pre-FFN ln)
  - **`MegatronBertEncoder`** [wiring]: wires `MegatronBertLayer`; direct `L1/layer_norm.py` (final ln)
  - **`MegatronBertPooler`** [compute]: `L1/linear.py + L1/tanh.py`
  - **`MegatronBertPredictionHeadTransform`** [compute]: `L1/linear.py + L1/gelu.py + L1/layer_norm.py`
  - **`MegatronBertLMPredictionHead`** [wiring]: wires `MegatronBertPredictionHeadTransform`; direct `L1/linear.py`
  - **`MegatronBertOnlyMLMHead`** [wiring]: wires `MegatronBertLMPredictionHead`
  - **`MegatronBertOnlyNSPHead`** [compute]: `L1/linear.py`
  - **`MegatronBertPreTrainingHeads`** [wiring]: wires `MegatronBertLMPredictionHead`; direct `L1/linear.py` (seq_relationship)
  - **`MegatronBertModel`** [wiring]: wires `MegatronBertEmbeddings`, `MegatronBertEncoder`, optional `MegatronBertPooler`
  - **`MegatronBertForMaskedLM`** [wiring]: wires `MegatronBertModel`, `MegatronBertOnlyMLMHead`
  - **`MegatronBertForCausalLM`** [wiring]: wires `MegatronBertModel`, `MegatronBertOnlyMLMHead`; direct `L1/linear.py` (lm_head)
- **task heads (6)**: ForPreTraining, ForNextSentencePrediction, ForMultipleChoice, ForQuestionAnswering, ForSequenceClassification, ForTokenClassification — base + linear (per-task)

## mgp_str
- **src**: modeling_mgp_str.py
- **hidden_act**: n/a (`mlp_ratio=4.0`; activations are nn.GELU directly)
- **status**: partial (ViT-style encoder + A3 head; no kb-nano L4)
- **classes**:
  - **`MgpstrDropPath`** [compute]: stochastic depth (no exact L1 — closest is `L1/dropout.py`)
  - **`MgpstrEmbeddings`** [compute]: `L1/conv2d.py + L1/embedding.py` (patch projection + cls token + learned pos)
  - **`MgpstrMlp`** [compute]: `L1/linear.py + L1/gelu.py + L1/linear.py` (2-layer FFN with nn.GELU)
  - **`MgpstrAttention`** [compute]: `L1/linear.py + L1/dense_attention.py` (fused QKV with single linear, ViT-style)
  - **`MgpstrLayer`** [wiring]: wires `MgpstrAttention`, `MgpstrMlp`, optional `MgpstrDropPath`; direct `L1/layer_norm.py` (×2)
  - **`MgpstrEncoder`** [wiring]: wires `MgpstrLayer`
  - **`MgpstrA3Module`** [compute]: `L1/layer_norm.py + L1/conv2d.py + L1/conv2d.py + L1/softmax.py + L1/conv2d.py + L1/layer_norm.py` (token learner: 1×1 conv chain → softmax attentions → einsum)
  - **`MgpstrModel`** [wiring]: wires `MgpstrEmbeddings`, `MgpstrEncoder`
- **task heads (1)**: ForSceneTextRecognition — base + linear (per-task)

## mimi
- **src**: modeling_mimi.py
- **hidden_act**: gelu
- **status**: partial (audio codec: SEANet conv encoder/decoder + transformer + RVQ; no kb-nano L4)
- **classes**:
  - **`MimiConv1d`** [compute]: `L1/conv1d.py` (asymmetric/causal padding wrapper around nn.Conv1d)
  - **`MimiConvTranspose1d`** [compute]: `L1/conv_transpose1d.py` (with right-trim for causal)
  - **`MimiResnetBlock`** [wiring]: wires `MimiConv1d` (×2), optional shortcut `MimiConv1d`; direct `L1/elu.py`
  - **`MimiEncoder`** [wiring]: wires `MimiConv1d`, `MimiResnetBlock`; direct `L1/elu.py`
  - **`MimiLayerScale`** [compute]: scalar parameter multiply (no exact L1 — closest is `L1/tensor_ops.py` or just elementwise mul)
  - **`MimiRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (Llama-shape)
  - **`MimiMLP`** [compute]: `L1/linear.py + L1/gelu.py + L1/linear.py` (CLIP-style fc1→act→fc2; closest L2 is `L2/clip_mlp.py` but uses gelu not quickgelu)
  - **`MimiAttention`** [compute]: `L1/linear.py + L1/dense_attention.py + L1/store_kvcache.py` (Llama-style q/k/v/o with RoPE + KV cache + sliding window; no exact L2 because Mimi uses sliding-window causal mask)
  - **`MimiFlashAttention2`** [compute, inherits `MimiAttention`]: same kernels; flash variant
  - **`MimiSdpaAttention`** [compute, inherits `MimiAttention`]: same kernels; sdpa variant
  - **`MimiTransformerLayer`** [wiring]: wires one of the `MimiAttention` variants, `MimiMLP`, `MimiLayerScale` (×2); direct `L1/layer_norm.py` (×2)
  - **`MimiTransformerModel`** [wiring]: wires `MimiTransformerLayer`
  - **`MimiDecoder`** [wiring]: wires `MimiConv1d`, `MimiConvTranspose1d`, `MimiResnetBlock`; direct `L1/elu.py`
  - **`MimiEuclideanCodebook`** [compute]: cdist-based nearest-centroid (no exact L1 — uses `torch.cdist`)
  - **`MimiVectorQuantization`** [wiring]: wires `MimiEuclideanCodebook`
  - **`MimiResidualVectorQuantizer`** [wiring]: wires `MimiVectorQuantization` (× num_quantizers)
  - **`MimiSplitResidualVectorQuantizer`** [wiring]: wires `MimiResidualVectorQuantizer` (×2: semantic + acoustic)
  - **`MimiModel`** [wiring]: wires `MimiEncoder`, `MimiTransformerModel` (×2: encoder + decoder transformers), `MimiDecoder`, `MimiSplitResidualVectorQuantizer`; direct `L1/conv1d.py` (down/upsample)

## minicpmv4_6
- **src**: modeling_minicpmv4_6.py (and modular_minicpmv4_6.py)
- **hidden_act**: gelu_pytorch_tanh (vision config); inherited from text backbone for LM
- **status**: partial (SigLIP-style vision encoder with NaViT-style variable-resolution + ViT window attention merger + multi-stage downsample MLP; LM is inherited from text config via `AutoModel`)
- **classes**:
  - **`MiniCPMV4_6VisionEmbeddings`** [compute, inherits `Idefics3VisionEmbeddings`]: `L1/conv2d.py + L1/embedding.py` (patch embed with bucketize-based 2D position from variable-size targets)
  - **`MiniCPMV4_6VisionMLP`** [compute, inherits `SiglipMLP`]: `L2/siglip_mlp.py` (fc1 → gelu_pytorch_tanh → fc2)
  - **`MiniCPMV4_6VisionAttention`** [compute, inherits `VisionAttention`]: `L2/vision_attention.py` (q/k/v/out non-causal with cu_seqlens for varlen, falls back to per-chunk SDPA otherwise)
  - **`MiniCPMV4_6VisionEncoderLayer`** [wiring, inherits `SiglipEncoderLayer`]: wires `MiniCPMV4_6VisionAttention`, `MiniCPMV4_6VisionMLP`; direct `L1/layer_norm.py` (×2)
  - **`MiniCPMV4_6VisionEncoder`** [wiring, inherits `SiglipEncoder`]: wires `MiniCPMV4_6VisionEncoderLayer`
  - **`MiniCPMV4_6ViTWindowAttentionMerger`** [wiring + compute]: wires `MiniCPMV4_6VisionAttention`; direct `L1/layer_norm.py` (×2), `L1/linear.py` (×2), `L1/gelu.py` (gelu_pytorch_tanh) — windowed attn + 2×2 patch merger MLP
  - **`MiniCPMV4_6VisionModel`** [wiring]: wires `MiniCPMV4_6VisionEmbeddings`, `MiniCPMV4_6VisionEncoder`, `MiniCPMV4_6ViTWindowAttentionMerger`; direct `L1/layer_norm.py`
  - **`MiniCPMV4_6DownsampleMLP`** [compute]: `L1/layer_norm.py + L1/linear.py + L1/gelu.py + L1/linear.py` (4× channel merge MLP)
  - **`MiniCPMV4_6Merger`** [wiring]: wires `MiniCPMV4_6DownsampleMLP` (× merger_times); recursive 2×2 spatial merge
  - **`MiniCPMV4_6Model`** [wiring, inherits `Lfm2VlModel`]: wires `MiniCPMV4_6VisionModel`, `MiniCPMV4_6Merger`, `AutoModel(text_config)`
  - **`MiniCPMV4_6ForConditionalGeneration`** [wiring]: wires `MiniCPMV4_6Model`; direct `L1/linear.py` (lm_head)

## minimax
- **src**: modeling_minimax.py (and modular_minimax.py)
- **hidden_act**: silu
- **status**: partial (Mixtral-shape with hybrid full/lightning linear-attention layers; lightning attn has no kb-nano kernel)
- **classes**:
  - **`MiniMaxRMSNorm`** [compute, inherits `MixtralRMSNorm`]: `L1/rms_norm.py`
  - **`MiniMaxLightningAttention`** [compute]: no kb-nano kernel — block-wise linear-attention with exponential slope decay (intra-block dense + inter-block linear cache); structurally `L1/linear.py + L1/silu.py` plus block matmuls
  - **`MiniMaxRotaryEmbedding`** [compute, inherits `Gemma2RotaryEmbedding`]: `L1/rotary_emb.py`
  - **`MiniMaxAttention`** [compute, inherits `MixtralAttention`]: `L2/attention.py` (Llama-shape Q/K/V/O + RoPE + KV cache + sliding window)
  - **`MiniMaxTopKRouter`** [compute, inherits `MixtralTopKRouter`]: `L1/linear.py + L1/softmax.py + L1/top_k_per_row.py` (softmax-then-topk router; renormalize)
  - **`MiniMaxExperts`** [compute]: `L2/mixtral_moe.py` (per-expert gate_up + silu*up + down loop with index_add; consolidated as fused experts variant)
  - **`MiniMaxSparseMoeBlock`** [wiring]: wires `MiniMaxTopKRouter`, `MiniMaxExperts`
  - **`MiniMaxDecoderLayer`** [wiring, inherits `MixtralDecoderLayer`]: wires `MiniMaxAttention` or `MiniMaxLightningAttention` (per layer_type), `MiniMaxSparseMoeBlock`, `MiniMaxRMSNorm` (×2); also α/β scale factors on residual and MLP branches
  - **`MiniMaxModel`** [wiring, inherits `MixtralModel`]: wires `MiniMaxDecoderLayer`, `MiniMaxRMSNorm`, `MiniMaxRotaryEmbedding`; direct `L1/embedding.py`
  - **`MiniMaxForCausalLM`** [wiring, inherits `MixtralForCausalLM`]: wires `MiniMaxModel`; direct `L1/linear.py` (lm_head)
- **task heads (3)**: ForSequenceClassification, ForTokenClassification, ForQuestionAnswering — base + linear (per-task)

## mistral
- **src**: modeling_mistral.py (and modular_mistral.py)
- **hidden_act**: silu
- **status**: kb_nano_l4 (consolidated under `L2/llama_mlp.py + L2/attention.py`; mistral uses sliding-window kwarg into Llama-shape attention)
- **classes**:
  - **`MistralMLP`** [compute]: `L2/llama_mlp.py` (gate_proj * silu(gate) → down_proj; same as LlamaMLP)
  - **`MistralAttention`** [compute]: `L2/attention.py` (Llama-shape q/k/v/o + RoPE + KV cache; passes `sliding_window` to attention_interface)
  - **`MistralRMSNorm`** [compute]: `L1/rms_norm.py`
  - **`MistralDecoderLayer`** [wiring]: wires `MistralAttention`, `MistralMLP`, `MistralRMSNorm` (×2)
  - **`MistralRotaryEmbedding`** [compute]: `L1/rotary_emb.py`
  - **`MistralModel`** [wiring]: wires `MistralDecoderLayer`, `MistralRMSNorm`, `MistralRotaryEmbedding`; direct `L1/embedding.py`
  - **`MistralForCausalLM`** [wiring]: wires `MistralModel`; direct `L1/linear.py` (lm_head)
- **task heads (3)**: ForSequenceClassification, ForTokenClassification, ForQuestionAnswering — base + linear (per-task)

## mistral3
- **src**: modeling_mistral3.py (and modular_mistral3.py)
- **hidden_act**: gelu (projector); silu (text); gelu (vision)
- **status**: composable (vision projector wires AutoModel vision + text)
- **classes**:
  - **`Mistral3RMSNorm`** [compute]: `L1/rms_norm.py`
  - **`Mistral3PatchMerger`** [compute]: `L1/linear.py` + `nn.functional.unfold` (no L1 unfold op; spatial-merge unfold-and-project)
  - **`Mistral3MultiModalProjector`** [wiring + compute]: wires `Mistral3RMSNorm`, `Mistral3PatchMerger`; direct `L1/linear.py + L1/gelu.py + L1/linear.py`
  - **`Mistral3Model`** [wiring]: wires `AutoModel(vision_config)`, `Mistral3MultiModalProjector`, `AutoModel(text_config)`
  - **`Mistral3ForConditionalGeneration`** [wiring]: wires `Mistral3Model`; direct `L1/linear.py` (lm_head)

## mixtral
- **src**: modeling_mixtral.py (and modular_mixtral.py)
- **hidden_act**: silu
- **status**: kb_nano_l4 (`L4/mixtral.py`)
- **classes**:
  - **`MixtralExperts`** [compute]: `L2/mixtral_moe.py` (per-expert gate_up + silu*up + down loop with index_add; consolidates with `L2/fused_experts.py` for fused path)
  - **`MixtralTopKRouter`** [compute]: `L1/linear.py + L1/softmax.py + L1/top_k_per_row.py` (softmax-then-topk; renormalize)
  - **`MixtralSparseMoeBlock`** [wiring]: wires `MixtralTopKRouter`, `MixtralExperts`
  - **`MixtralRMSNorm`** [compute]: `L1/rms_norm.py`
  - **`MixtralRotaryEmbedding`** [compute]: `L1/rotary_emb.py`
  - **`MixtralAttention`** [compute]: `L2/attention.py` (Llama-shape Q/K/V/O + RoPE + KV cache; sliding_window kwarg)
  - **`MixtralDecoderLayer`** [wiring]: wires `MixtralAttention`, `MixtralSparseMoeBlock`, `MixtralRMSNorm` (×2)
  - **`MixtralModel`** [wiring]: wires `MixtralDecoderLayer`, `MixtralRMSNorm`, `MixtralRotaryEmbedding`; direct `L1/embedding.py`
  - **`MixtralForCausalLM`** [wiring]: wires `MixtralModel`; direct `L1/linear.py` (lm_head)
- **task heads (3)**: ForSequenceClassification, ForTokenClassification, ForQuestionAnswering — base + linear (per-task)
