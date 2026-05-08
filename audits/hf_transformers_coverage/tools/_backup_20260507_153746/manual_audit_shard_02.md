# Manual Audit Shard 02 (bloom .. ctrl)

## bloom
- **src**: modeling_bloom.py
- **hidden_act**: n/a (uses custom BloomGelu)
- **status**: partial
- **classes**:
  - **`BloomGelu`** [compute]: `L1/gelu.py` (custom Megatron-style approx-gelu; closest L1 op)
  - **`BloomAttention`** [compute]: `L1/linear.py + L1/dense_attention.py` (no exact L2 match — fused QKV linear + ALiBi-baddbmm + softmax + bmm + dense; ALiBi not handled by L2/attention.py which assumes RoPE)
  - **`BloomMLP`** [wiring]: wires `BloomGelu`; direct `L1/linear.py` x2 (dense_h_to_4h, dense_4h_to_h) — 2-layer MLP, no SwiGLU gate
  - **`BloomBlock`** [wiring]: wires `BloomAttention`, `BloomMLP`; direct `L1/layer_norm.py` x2 (input_layernorm, post_attention_layernorm)
  - **`BloomModel`** [wiring]: wires `BloomBlock`; direct `L1/embedding.py` (word_embeddings), `L1/layer_norm.py` x2 (word_embeddings_layernorm, ln_f)
  - **`BloomForCausalLM`** [wiring]: wires `BloomModel`; direct `L1/linear.py` (lm_head)
- **task heads (3)**: ForSequenceClassification, ForTokenClassification, ForQuestionAnswering — base + linear (per-task)

## blt
- **src**: modeling_blt.py (and modular_blt.py)
- **hidden_act**: silu (in all sub-configs)
- **status**: partial
- **classes**:
  - **`BltMLP`** [compute]: `L2/llama_mlp.py` (gate_proj/up_proj/down_proj SwiGLU pattern with silu)
  - **`BltRMSNorm`** [compute]: `L1/rms_norm.py` (standard RMSNorm; doc says "equivalent to T5LayerNorm" but math has the rsqrt+weight pattern)
  - **`BltRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (default-RoPE with interleaved layout; close enough — uses repeat_interleave instead of cat but inv-freq computation matches)
  - **`BltTransformerLayer`** [wiring]: wires `BltSelfAttention`, `BltMLP`, `BltRMSNorm` x2 (input_layernorm, post_attention_layernorm)
  - **`BltSelfAttention`** [compute]: `L2/attention.py` (q/k/v/o + GQA + RoPE + KV cache + ALL_ATTENTION_FUNCTIONS dispatch — Llama-style)
  - **`BltCrossAttention`** [compute]: `L1/linear.py + L1/rms_norm.py + L1/dense_attention.py` (no exact L2 match — q_norm/k_norm before q/k projection, no RoPE, no KV cache, residual on output)
  - **`BltLocalEncoder`** [wiring]: wires `BltTransformerLayer`, `BltRotaryEmbedding`, `BltCrossAttention`; direct `L1/embedding.py` (embed_tokens), `L1/linear.py` (patch_embedding_projection)
  - **`BltLocalDecoder`** [wiring]: wires `BltTransformerLayer`, `BltRotaryEmbedding`, `BltCrossAttention`, `BltRMSNorm` (norm); direct `L1/linear.py` (patch_embedding_projection)
  - **`BltGlobalTransformer`** [wiring]: wires `BltTransformerLayer`, `BltRotaryEmbedding`; direct `L1/linear.py` (token_embedding_projection, optional)
  - **`BltPatcher`** [wiring]: wires `BltTransformerLayer`, `BltRotaryEmbedding`, `BltRMSNorm`; direct `L1/embedding.py` (embed_tokens), `L1/linear.py` (lm_head)
  - **`BltModel`** [wiring]: wires `BltLocalEncoder`, `BltGlobalTransformer`, `BltLocalDecoder`, optional `BltPatcher`; direct `L1/embedding.py` (encoder_hash_tok_embedding)
  - **`BltForCausalLM`** [wiring]: wires `BltModel`; direct `L1/linear.py` (lm_head)
- **task heads (0)**: none

## bridgetower
- **src**: modeling_bridgetower.py
- **hidden_act**: gelu (text and vision configs)
- **status**: partial
- **classes**:
  - **`BridgeTowerResidualAttention`** [compute]: `L1/linear.py + L1/quickgelu.py + L1/layer_norm.py + L1/dense_attention.py` (CLIP-style block: nn.MultiheadAttention with QuickGELU MLP and pre-norm; no exact L2 match — clip_attention.py separates q/k/v while this uses fused MHA)
  - **`BridgeTowerTransformer`** [wiring]: wires `BridgeTowerResidualAttention` (×N)
  - **`BridgeTowerVisionEmbeddings`** [compute]: `L1/conv2d.py + L1/embedding.py` (Conv2d patch_embedding + nn.Embedding pos + class token)
  - **`BridgeTowerVisionTransformer`** [wiring]: wires `BridgeTowerVisionEmbeddings`, `BridgeTowerTransformer`; direct `L1/layer_norm.py` (ln_pre, ln_post, optional ln_separate ModuleList)
  - **`BridgeTowerLinkTower`** [compute]: `L1/layer_norm.py` (LayerNorm of weighted-sum of two streams; no exact L2 match)
  - **`BridgeTowerSelfOutput`** [compute, copied from BertSelfOutput]: `L2/encoder_attention.py` (dense + LayerNorm + residual)
  - **`BridgeTowerIntermediate`** [compute, copied from BertIntermediate]: `L1/linear.py + L1/gelu.py` (just half of an encoder MLP)
  - **`BridgeTowerOutput`** [compute, copied from BertOutput]: `L1/linear.py + L1/layer_norm.py` (encoder MLP output: dense + LayerNorm + residual)
  - **`BridgeTowerPooler`** [compute, copied from BertPooler]: `L1/linear.py + L1/tanh.py`
  - **`BridgeTowerSelfAttention`** [compute, copied from RobertaSelfAttention]: `L2/encoder_attention.py` (q/k/v + ALL_ATTENTION_FUNCTIONS dispatch)
  - **`BridgeTowerCrossAttention`** [compute, copied from RobertaCrossAttention]: `L1/linear.py + L1/dense_attention.py` (cross-attn with EncoderDecoderCache; no exact L2 match)
  - **`BridgeTowerAttention`** [wiring, copied from BertAttention]: wires `BridgeTowerSelfAttention` or `BridgeTowerCrossAttention`, `BridgeTowerSelfOutput`
  - **`BridgeTowerBertCrossLayer`** [wiring]: wires `BridgeTowerAttention` (self + cross), `BridgeTowerIntermediate`, `BridgeTowerOutput`
  - **`BridgeTowerTextLayer`** [wiring, copies BertLayer]: wires `BridgeTowerAttention`, optional `BridgeTowerAttention` (cross), `BridgeTowerIntermediate`, `BridgeTowerOutput`
  - **`BridgeTowerTextEncoder`** [wiring]: wires `BridgeTowerTextLayer`
  - **`BridgeTowerTextEmbeddings`** [compute, copied from RobertaEmbeddings]: `L2/encoder_embeddings.py` (BERT-style word + position + token_type + LayerNorm)
  - **`BridgeTowerVisionModel`** [wiring]: wires `BridgeTowerVisionTransformer`
  - **`BridgeTowerTextModel`** [wiring]: wires `BridgeTowerTextEmbeddings`, `BridgeTowerTextEncoder`, optional `BridgeTowerPooler`
  - **`BridgeTowerModel`** [wiring]: wires `BridgeTowerVisionModel`, `BridgeTowerTextModel`, `BridgeTowerBertCrossLayer` (×2 for image/text), `BridgeTowerLinkTower` (×2N), `BridgeTowerPooler`; direct `L1/linear.py` (token_type_embeddings projection), `L1/layer_norm.py`
  - **`BridgeTowerPredictionHeadTransform`** [compute]: `L1/linear.py + L1/gelu.py + L1/layer_norm.py`
  - **`BridgeTowerMLMHead`** [wiring]: wires `BridgeTowerPredictionHeadTransform`; direct `L1/linear.py` (decoder)
  - **`BridgeTowerITMHead`** [compute]: `L1/linear.py`
  - **`BridgeTowerContrastiveHead`** [compute]: `L1/linear.py`
  - **`BridgeTowerForMaskedLM`** [wiring]: wires `BridgeTowerModel`, `BridgeTowerMLMHead`
  - **`BridgeTowerForImageAndTextRetrieval`** [wiring]: wires `BridgeTowerModel`, `BridgeTowerITMHead`
  - **`BridgeTowerForContrastiveLearning`** [wiring]: wires `BridgeTowerModel`, `BridgeTowerContrastiveHead`, `BridgeTowerITMHead`
- **task heads (0)**: none (ForMaskedLM, ForImageAndTextRetrieval, ForContrastiveLearning are primary forward paths)

## bros
- **src**: modeling_bros.py
- **hidden_act**: gelu
- **status**: partial
- **classes**:
  - **`BrosPositionalEmbedding1D`** [compute]: `L1/sinusoidal_embed.py` (sinusoidal 1d positional encoding for bbox; closest L1 op)
  - **`BrosPositionalEmbedding2D`** [wiring]: wires `BrosPositionalEmbedding1D` (×2: x_pos_emb, y_pos_emb)
  - **`BrosBboxEmbeddings`** [wiring]: wires `BrosPositionalEmbedding2D`; direct `L1/linear.py` (bbox_projection)
  - **`BrosTextEmbeddings`** [compute]: `L2/encoder_embeddings.py` (BERT-style word + position + token_type + LayerNorm)
  - **`BrosSelfAttention`** [compute]: `L1/linear.py + L1/dense_attention.py` (q/k/v with bbox-positional bias added to attention scores; no exact L2 match — encoder_attention.py doesn't add bbox bias)
  - **`BrosSelfOutput`** [compute, copied from BertSelfOutput]: `L2/encoder_attention.py` (dense + LayerNorm + residual)
  - **`BrosAttention`** [wiring]: wires `BrosSelfAttention`, `BrosSelfOutput`
  - **`BrosIntermediate`** [compute, copied from BertIntermediate]: `L1/linear.py + L1/gelu.py`
  - **`BrosOutput`** [compute]: `L1/linear.py + L1/layer_norm.py` (dense + LayerNorm + residual)
  - **`BrosLayer`** [wiring]: wires `BrosAttention`, optional `BrosAttention` (cross), `BrosIntermediate`, `BrosOutput`
  - **`BrosPooler`** [compute, copied from BertPooler]: `L1/linear.py + L1/tanh.py`
  - **`BrosRelationExtractor`** [compute]: `L1/linear.py` (×2 query/key) (relation-score head; no exact L2 match)
  - **`BrosEncoder`** [wiring]: wires `BrosLayer`
  - **`BrosModel`** [wiring]: wires `BrosTextEmbeddings`, `BrosBboxEmbeddings`, `BrosEncoder`, `BrosPooler`
- **task heads (3)**: ForTokenClassification, BrosSpadeEEForTokenClassification, BrosSpadeELForTokenClassification — base + linear / RelationExtractor (per-task)

## camembert
- **src**: modeling_camembert.py (and modular_camembert.py — inherits from Roberta)
- **hidden_act**: gelu
- **status**: composable
- **classes**:
  - **`CamembertEmbeddings`** [compute, copies RobertaEmbeddings]: `L2/encoder_embeddings.py` (BERT-style word + position + token_type + LayerNorm; Roberta uses padding_idx in pos embedding)
  - **`CamembertSelfAttention`** [compute, copies RobertaSelfAttention]: `L2/encoder_attention.py` (q/k/v + ALL_ATTENTION_FUNCTIONS dispatch)
  - **`CamembertCrossAttention`** [compute, copies RobertaCrossAttention]: `L1/linear.py + L1/dense_attention.py` (cross-attn with EncoderDecoderCache)
  - **`CamembertSelfOutput`** [compute, copies BertSelfOutput]: `L2/encoder_attention.py` (dense + LayerNorm + residual)
  - **`CamembertAttention`** [wiring, copies BertAttention]: wires `CamembertSelfAttention` or `CamembertCrossAttention`, `CamembertSelfOutput`
  - **`CamembertIntermediate`** [compute, copies BertIntermediate]: `L1/linear.py + L1/gelu.py`
  - **`CamembertOutput`** [compute]: `L1/linear.py + L1/layer_norm.py` (dense + LayerNorm + residual)
  - **`CamembertLayer`** [wiring]: wires `CamembertAttention`, optional `CamembertAttention` (cross), `CamembertIntermediate`, `CamembertOutput`
  - **`CamembertLMHead`** [compute, copies RobertaLMHead]: `L1/linear.py + L1/gelu.py + L1/layer_norm.py + L1/linear.py` (dense + gelu + LayerNorm + decoder)
  - **`CamembertEncoder`** [wiring]: wires `CamembertLayer`
  - **`CamembertPooler`** [compute, copies BertPooler]: `L1/linear.py + L1/tanh.py`
  - **`CamembertModel`** [wiring, inherits `RobertaModel`]: wires `CamembertEmbeddings`, `CamembertEncoder`, optional `CamembertPooler`
  - **`CamembertClassificationHead`** [compute]: `L1/linear.py + L1/tanh.py + L1/linear.py` (dense + tanh + dropout + out_proj)
  - **`CamembertForMaskedLM`** [wiring, inherits `RobertaForMaskedLM`]: wires `CamembertModel`, `CamembertLMHead`
  - **`CamembertForCausalLM`** [wiring, inherits `RobertaForCausalLM`]: wires `CamembertModel`, `CamembertLMHead`
- **task heads (4)**: ForSequenceClassification, ForMultipleChoice, ForTokenClassification, ForQuestionAnswering — base + linear (per-task)

## canine
- **src**: modeling_canine.py
- **hidden_act**: gelu
- **status**: partial
- **classes**:
  - **`CanineEmbeddings`** [compute]: `L1/embedding.py + L1/layer_norm.py` (multi-hash bucket char embeddings + position + token_type + LayerNorm; no exact L2 match — uses hash buckets instead of single word_embedding lookup)
  - **`CharactersToMolecules`** [compute]: `L1/conv1d.py + L1/gelu.py + L1/layer_norm.py` (strided conv1d for downsampling chars to molecules)
  - **`ConvProjection`** [compute]: `L1/conv1d.py + L1/gelu.py + L1/layer_norm.py` (conv1d with same-padding to project upsampled features)
  - **`CanineSelfAttention`** [compute]: `L1/linear.py + L1/dense_attention.py` (q/k/v with from/to-tensor pair for flexible cross-attn; no exact L2 match — encoder_attention.py expects single hidden_states input)
  - **`CanineSelfOutput`** [compute]: `L2/encoder_attention.py` (dense + LayerNorm + residual)
  - **`CanineAttention`** [wiring]: wires `CanineSelfAttention`, `CanineSelfOutput` (with optional local-chunked attention loop)
  - **`CanineIntermediate`** [compute]: `L1/linear.py + L1/gelu.py`
  - **`CanineOutput`** [compute]: `L1/linear.py + L1/layer_norm.py`
  - **`CanineLayer`** [wiring]: wires `CanineAttention`, `CanineIntermediate`, `CanineOutput`
  - **`CanineEncoder`** [wiring]: wires `CanineLayer`
  - **`CaninePooler`** [compute]: `L1/linear.py + L1/tanh.py`
  - **`CaninePredictionHeadTransform`** [compute]: `L1/linear.py + L1/gelu.py + L1/layer_norm.py`
  - **`CanineLMPredictionHead`** [wiring]: wires `CaninePredictionHeadTransform`; direct `L1/linear.py` (decoder)
  - **`CanineOnlyMLMHead`** [wiring]: wires `CanineLMPredictionHead`
  - **`CanineModel`** [wiring]: wires `CanineEmbeddings`, `CharactersToMolecules` (downsample), `CanineEncoder` (×3: char init, deep transformer, char final), `ConvProjection` (upsample), `CaninePooler`
- **task heads (4)**: ForSequenceClassification, ForMultipleChoice, ForTokenClassification, ForQuestionAnswering — base + linear (per-task)

## chameleon
- **src**: modeling_chameleon.py
- **hidden_act**: silu
- **status**: partial
- **classes**:
  - **`ChameleonRMSNorm`** [compute]: `L1/rms_norm.py` (standard RMSNorm)
  - **`ChameleonRotaryEmbedding`** [compute, copies LlamaRotaryEmbedding]: `L1/rotary_emb.py`
  - **`ChameleonMLP`** [compute, copies LlamaMLP]: `L2/llama_mlp.py` (gate/up/down with silu)
  - **`ChameleonLayerNorm`** [compute]: `L1/layer_norm.py` (LayerNorm with last-dim-only normalization; close enough)
  - **`ChameleonAttention`** [compute]: `L2/attention.py` (q/k/v/o with q_norm/k_norm before RoPE — not supported by L2/attention.py which doesn't apply per-head LayerNorm; nearest match)
  - **`ChameleonDecoderLayer`** [wiring, copies LlamaDecoderLayer]: wires `ChameleonAttention`, `ChameleonMLP`, `ChameleonRMSNorm` x2
  - **`ChameleonSwinDecoderLayer`** [wiring]: wires `ChameleonAttention`, `ChameleonMLP`, `ChameleonRMSNorm` x2 (post-norm variant; norms applied after attn/mlp)
  - **`ChameleonVQVAEVectorQuantizer`** [compute]: `L1/embedding.py` (codebook lookup + L2 distance argmin)
  - **`ChameleonVQVAEEncoderConvDownsample`** [compute]: `L1/conv2d.py` (with manual asymmetric padding)
  - **`ChameleonVQVAEEncoderResnetBlock`** [compute]: `L1/group_norm.py + L1/sigmoid.py + L1/conv2d.py + L1/group_norm.py + L1/sigmoid.py + L1/conv2d.py` (GN + swish + conv x2 with optional shortcut conv2d)
  - **`ChameleonVQVAEEncoderAttnBlock`** [compute]: `L1/group_norm.py + L1/conv2d.py + L1/dense_attention.py + L1/conv2d.py` (1x1 conv q/k/v + attention + proj_out; no exact L2 match)
  - **`ChameleonVQVAEEncoder`** [wiring]: wires `ChameleonVQVAEEncoderResnetBlock`, `ChameleonVQVAEEncoderAttnBlock`, `ChameleonVQVAEEncoderConvDownsample`; direct `L1/conv2d.py` (conv_in, conv_out), `L1/group_norm.py` (norm_out)
  - **`ChameleonVQVAE`** [wiring]: wires `ChameleonVQVAEEncoder`, `ChameleonVQVAEVectorQuantizer`; direct `L1/conv2d.py` (quant_conv, post_quant_conv)
  - **`ChameleonModel`** [wiring]: wires `ChameleonDecoderLayer` (or `ChameleonSwinDecoderLayer`), `ChameleonVQVAE`, `ChameleonRMSNorm` (norm), `ChameleonRotaryEmbedding`; direct `L1/embedding.py` (embed_tokens)
  - **`ChameleonForConditionalGeneration`** [wiring]: wires `ChameleonModel`; direct `L1/linear.py` (lm_head)
- **task heads (0)**: none (ForConditionalGeneration is primary)

## chinese_clip
- **src**: modeling_chinese_clip.py (and modular_chinese_clip.py)
- **hidden_act**: gelu (text), quick_gelu (vision)
- **status**: composable
- **classes**:
  - **`ChineseCLIPTextEmbeddings`** [compute]: `L2/encoder_embeddings.py` (BERT-style word + position + token_type + LayerNorm)
  - **`ChineseCLIPVisionEmbeddings`** [compute]: `L1/conv2d.py + L1/embedding.py` (Conv2d patch + class token + position embedding)
  - **`ChineseCLIPTextSelfAttention`** [compute]: `L2/encoder_attention.py` (q/k/v + ALL_ATTENTION_FUNCTIONS dispatch)
  - **`ChineseCLIPTextSelfOutput`** [compute]: `L2/encoder_attention.py` (dense + LayerNorm + residual)
  - **`ChineseCLIPTextAttention`** [wiring]: wires `ChineseCLIPTextSelfAttention`, `ChineseCLIPTextSelfOutput`
  - **`ChineseCLIPVisionAttention`** [compute]: `L2/clip_attention.py` (q/k/v + out_proj, non-causal CLIP-style)
  - **`ChineseCLIPTextIntermediate`** [compute]: `L1/linear.py + L1/gelu.py`
  - **`ChineseCLIPTextOutput`** [compute]: `L1/linear.py + L1/layer_norm.py`
  - **`ChineseCLIPVisionMLP`** [compute]: `L2/clip_mlp.py` (fc1 → quickgelu → fc2)
  - **`ChineseCLIPTextLayer`** [wiring]: wires `ChineseCLIPTextAttention`, `ChineseCLIPTextIntermediate`, `ChineseCLIPTextOutput`
  - **`ChineseCLIPVisionLayer`** [wiring]: wires `ChineseCLIPVisionAttention`, `ChineseCLIPVisionMLP`; direct `L1/layer_norm.py` x2
  - **`ChineseCLIPTextPooler`** [compute]: `L1/linear.py + L1/tanh.py`
  - **`ChineseCLIPTextEncoder`** [wiring]: wires `ChineseCLIPTextLayer`
  - **`ChineseCLIPVisionEncoder`** [wiring]: wires `ChineseCLIPVisionLayer`
  - **`ChineseCLIPVisionModel`** [wiring]: wires `ChineseCLIPVisionEmbeddings`, `ChineseCLIPVisionEncoder`; direct `L1/layer_norm.py` (pre_layrnorm, post_layernorm)
  - **`ChineseCLIPTextModel`** [wiring]: wires `ChineseCLIPTextEmbeddings`, `ChineseCLIPTextEncoder`, optional `ChineseCLIPTextPooler`
  - **`ChineseCLIPModel`** [wiring]: wires `ChineseCLIPVisionModel`, `ChineseCLIPTextModel`; direct `L1/linear.py` (visual_projection, text_projection), logit_scale parameter
- **task heads (0)**: none

## chmv2
- **src**: modeling_chmv2.py (and modular_chmv2.py)
- **hidden_act**: n/a (DPT-style depth estimation; uses ReLU and backbone)
- **status**: partial
- **classes**:
  - **`CHMv2ReassembleLayer`** [compute]: `L1/conv2d.py + L1/conv_transpose2d.py` (1x1 projection conv + optional ConvTranspose2d upsample or strided Conv2d downsample)
  - **`CHMv2ReassembleStage`** [wiring]: wires `CHMv2ReassembleLayer`; direct `L1/linear.py + L1/gelu.py` (optional readout_projects with Linear+GELU)
  - **`CHMv2PreActResidualLayer`** [compute]: `L1/relu.py + L1/conv2d.py + L1/relu.py + L1/conv2d.py` (pre-act ReLU+Conv x2 with residual)
  - **`CHMv2FeatureFusionLayer`** [wiring]: wires `CHMv2PreActResidualLayer` (×2 if not first layer, ×1 if first); direct `L1/conv2d.py` (projection), `L1/interpolate.py` (bilinear)
  - **`CHMv2UpsampleConvHead`** [compute]: `L1/conv2d.py + L1/interpolate.py + L1/conv2d.py + L1/relu.py + L1/conv2d.py` (Conv3x3 + bilinear upsample + Conv3x3 + ReLU + Conv1x1)
  - **`CHMv2Head`** [wiring]: wires `CHMv2ReassembleStage`, `CHMv2FeatureFusionLayer`, `CHMv2UpsampleConvHead`; direct `L1/conv2d.py` (convs ModuleList)
  - **`CHMv2FeaturesToDepth`** [compute]: `L1/relu.py + L1/softmax.py + L1/sigmoid.py` (depth-bin normalization; no exact L2 match — custom mixlog/softmax/sigmoid/linear depth-bin head)
  - **`CHMv2ForDepthEstimation`** [wiring]: wires backbone (load_backbone — typically DINOv3ViT), `CHMv2Head`, `CHMv2FeaturesToDepth`
- **task heads (0)**: none (ForDepthEstimation is primary)

## clap
- **src**: modeling_clap.py
- **hidden_act**: gelu (text and audio); projection_hidden_act: relu
- **status**: partial
- **classes**:
  - **`ClapDropPath`** [compute]: stochastic depth (no L1 op; uses dropout-like rand+floor; closest is `L1/dropout.py`)
  - **`ClapAudioAFFBlock`** [compute]: `L1/conv2d.py + L1/batch_norm2d.py + L1/relu.py + L1/conv2d.py + L1/batch_norm2d.py + L1/adaptive_avg_pool2d.py + L1/sigmoid.py` (attentional feature fusion: local + global branches with sigmoid gate)
  - **`ClapAudioPatchEmbed`** [wiring]: wires optional `ClapAudioAFFBlock` (fusion); direct `L1/conv2d.py` (proj, optional mel_conv2d), `L1/layer_norm.py` (norm)
  - **`ClapAudioSelfAttention`** [compute, copies SwinSelfAttention]: `L2/swinv2_window_attention.py` (q/k/v + relative_position_bias_table; closest match — Swin v1 vs v2 differs slightly in scale)
  - **`ClapAudioSelfOutput`** [compute, copies SwinSelfOutput]: `L1/linear.py` (dense + dropout, no LayerNorm)
  - **`ClapAudioAttention`** [wiring, copies SwinAttention]: wires `ClapAudioSelfAttention`, `ClapAudioSelfOutput`
  - **`ClapAudioIntermediate`** [compute, copies SwinIntermediate]: `L1/linear.py + L1/gelu.py`
  - **`ClapAudioOutput`** [compute, copies SwinOutput]: `L1/linear.py` (dense + dropout)
  - **`ClapAudioLayer`** [wiring, copies SwinLayer]: wires `ClapAudioAttention`, `ClapAudioIntermediate`, `ClapAudioOutput`, `ClapDropPath`; direct `L1/layer_norm.py` x2 (layernorm_before, layernorm_after)
  - **`ClapAudioStage`** [wiring, copies SwinStage]: wires `ClapAudioLayer`, optional `ClapAudioPatchMerging`
  - **`ClapAudioPatchMerging`** [compute, copies SwinPatchMerging]: `L2/swinv2_patch_merging.py` (closest match; concat 4 sub-patches + LN + Linear reduce 4d→2d)
  - **`ClapAudioEncoder`** [wiring]: wires `ClapAudioPatchEmbed`, `ClapAudioStage`; direct `L1/batch_norm2d.py` (batch_norm), `L1/layer_norm.py` (norm), `L1/adaptive_avg_pool1d.py` (avgpool)
  - **`ClapProjectionLayer`** [compute]: `L1/linear.py + L1/relu.py + L1/linear.py`
  - **`ClapTextEmbeddings`** [compute, copies RobertaEmbeddings]: `L2/encoder_embeddings.py` (BERT-style word + position + token_type + LayerNorm)
  - **`ClapTextSelfAttention`** [compute, copies AlignTextSelfAttention]: `L2/encoder_attention.py` (q/k/v + ALL_ATTENTION_FUNCTIONS dispatch)
  - **`ClapTextSelfOutput`** [compute, copies BertSelfOutput]: `L2/encoder_attention.py` (dense + LayerNorm + residual)
  - **`ClapTextAttention`** [wiring, copies AlignTextAttention]: wires `ClapTextSelfAttention`, `ClapTextSelfOutput`
  - **`ClapTextIntermediate`** [compute, copies BertIntermediate]: `L1/linear.py + L1/gelu.py`
  - **`ClapTextOutput`** [compute, copies BertOutput]: `L1/linear.py + L1/layer_norm.py`
  - **`ClapTextLayer`** [wiring]: wires `ClapTextAttention`, `ClapTextIntermediate`, `ClapTextOutput`
  - **`ClapTextEncoder`** [wiring]: wires `ClapTextLayer`
  - **`ClapTextPooler`** [compute, copies BertPooler]: `L1/linear.py + L1/tanh.py`
  - **`ClapAudioModel`** [wiring]: wires `ClapAudioEncoder`
  - **`ClapTextModel`** [wiring]: wires `ClapTextEmbeddings`, `ClapTextEncoder`, optional `ClapTextPooler`
  - **`ClapModel`** [wiring]: wires `ClapTextModel`, `ClapAudioModel`, `ClapProjectionLayer` (×2: text and audio); logit_scale parameters
  - **`ClapTextModelWithProjection`** [wiring]: wires `ClapTextModel`, `ClapProjectionLayer`
  - **`ClapAudioModelWithProjection`** [wiring]: wires `ClapAudioModel`, `ClapProjectionLayer`
- **task heads (0)**: none

## clip
- **src**: modeling_clip.py
- **hidden_act**: quick_gelu (text and vision)
- **status**: composable
- **classes**:
  - **`CLIPVisionEmbeddings`** [compute]: `L1/conv2d.py + L1/embedding.py` (Conv2d patch + class token + position embedding)
  - **`CLIPTextEmbeddings`** [compute]: `L1/embedding.py + L1/embedding.py` (token_embedding + position_embedding, no LayerNorm)
  - **`CLIPAttention`** [compute]: `L2/clip_attention.py` (q/k/v + out_proj, non-causal CLIP-style; ALL_ATTENTION_FUNCTIONS dispatch)
  - **`CLIPMLP`** [compute]: `L2/clip_mlp.py` (fc1 → quick_gelu → fc2)
  - **`CLIPEncoderLayer`** [wiring]: wires `CLIPAttention`, `CLIPMLP`; direct `L1/layer_norm.py` x2 (layer_norm1, layer_norm2)
  - **`CLIPEncoder`** [wiring]: wires `CLIPEncoderLayer`
  - **`CLIPTextModel`** [wiring]: wires `CLIPTextEmbeddings`, `CLIPEncoder`; direct `L1/layer_norm.py` (final_layer_norm)
  - **`CLIPVisionModel`** [wiring]: wires `CLIPVisionEmbeddings`, `CLIPEncoder`; direct `L1/layer_norm.py` (pre_layrnorm, post_layernorm)
  - **`CLIPModel`** [wiring]: wires `CLIPVisionModel`, `CLIPTextModel`; direct `L1/linear.py` (visual_projection, text_projection), logit_scale parameter
  - **`CLIPTextModelWithProjection`** [wiring]: wires `CLIPTextModel`; direct `L1/linear.py` (text_projection)
  - **`CLIPVisionModelWithProjection`** [wiring]: wires `CLIPVisionModel`; direct `L1/linear.py` (visual_projection)
- **task heads (1)**: ForImageClassification — base + linear (per-task)

## clipseg
- **src**: modeling_clipseg.py
- **hidden_act**: quick_gelu (text and vision); decoder_hidden_act: quick_gelu (but decoder.layers use relu via decoder_config override)
- **status**: composable
- **classes**:
  - **`CLIPSegVisionEmbeddings`** [compute]: `L1/conv2d.py + L1/embedding.py` (Conv2d patch + class token + pos embedding; same as CLIPVisionEmbeddings)
  - **`CLIPSegTextEmbeddings`** [compute]: `L1/embedding.py + L1/embedding.py` (token + position, no LayerNorm)
  - **`CLIPSegAttention`** [compute]: `L2/clip_attention.py` (q/k/v + out_proj, non-causal; same as CLIPAttention)
  - **`CLIPSegMLP`** [compute]: `L2/clip_mlp.py` (fc1 → quick_gelu → fc2)
  - **`CLIPSegEncoderLayer`** [wiring]: wires `CLIPSegAttention`, `CLIPSegMLP`; direct `L1/layer_norm.py` x2 (pre-norm)
  - **`CLIPSegDecoderLayer`** [wiring]: wires `CLIPSegAttention`, `CLIPSegMLP`; direct `L1/layer_norm.py` x2 (post-norm: norms applied after attn/mlp)
  - **`CLIPSegEncoder`** [wiring]: wires `CLIPSegEncoderLayer`
  - **`CLIPSegDecoder`** [wiring]: wires `CLIPSegDecoderLayer`; direct `L1/linear.py` (film_mul, film_add, reduces ModuleList), `L1/conv_transpose2d.py` (transposed_convolution), optional `L1/conv2d.py + L1/relu.py + L1/conv_transpose2d.py + L1/relu.py + L1/conv_transpose2d.py` (complex variant)
  - **`CLIPSegTextModel`** [wiring]: wires `CLIPSegTextEmbeddings`, `CLIPSegEncoder`; direct `L1/layer_norm.py` (final_layer_norm)
  - **`CLIPSegVisionModel`** [wiring]: wires `CLIPSegVisionEmbeddings`, `CLIPSegEncoder`; direct `L1/layer_norm.py` (pre_layrnorm, post_layernorm)
  - **`CLIPSegModel`** [wiring]: wires `CLIPSegVisionModel`, `CLIPSegTextModel`; direct `L1/linear.py` (visual_projection, text_projection), logit_scale parameter
  - **`CLIPSegForImageSegmentation`** [wiring]: wires `CLIPSegModel`, `CLIPSegDecoder`
- **task heads (0)**: none (ForImageSegmentation is primary)

## clvp
- **src**: modeling_clvp.py
- **hidden_act**: gelu (encoder MLP gate); decoder uses activation_function="gelu_new"
- **status**: partial
- **classes**:
  - **`ClvpRMSNorm`** [compute]: `L1/rms_norm.py` (standard RMSNorm)
  - **`ClvpRotaryPositionalEmbedding`** [compute]: `L1/rotary_emb.py` (custom partial-rotary; closest L1 op — caches embeddings, uses `cat((freqs, freqs), -1)`)
  - **`ClvpSelfAttention`** [compute]: `L1/linear.py + L1/rotary_emb.py + L1/dense_attention.py + L1/store_kvcache.py` (partial-rotary applied to q/k/v split, then concat — no exact L2 match)
  - **`ClvpGatedLinearUnit`** [compute]: `L1/linear.py + L1/gelu.py` (single linear → 2x intermediate split into hidden/gate, gate*activation; close to GeGLU but proj outputs both halves)
  - **`ClvpEncoderMLP`** [wiring]: wires `ClvpGatedLinearUnit`; direct `L1/linear.py` (fc2)
  - **`ClvpEncoderLayer`** [wiring]: wires `ClvpSelfAttention`, `ClvpEncoderMLP`, `ClvpRMSNorm` x2 (input_rmsnorm, post_attention_rmsnorm)
  - **`ClvpSequenceSummary`** [compute, copies XLMSequenceSummary]: `L1/linear.py + L1/tanh.py` (or activation; pooled summary head)
  - **`ClvpDecoderMLP`** [compute, copies GPT2MLP]: `L1/linear.py + L1/gelu.py + L1/linear.py` (Conv1D-as-Linear with gelu_new)
  - **`ClvpDecoderLayer`** [wiring]: wires `ClvpSelfAttention`, `ClvpDecoderMLP`; direct `L1/layer_norm.py` x2 (input_layernorm, post_attention_layernorm)
  - **`ClvpConditioningEncoder`** [wiring]: wires `ClvpSelfAttention` (mel_attn_blocks); direct `L1/embedding.py` (text_token_embedding, text_position_embedding), `L1/conv1d.py` (mel_conv), `L1/group_norm.py` (group_norms ModuleList)
  - **`ClvpEncoder`** [wiring]: wires `ClvpEncoderLayer`, `ClvpRotaryPositionalEmbedding`, optional `ClvpSequenceSummary`; direct `L1/embedding.py` (token_embedding), `L1/rms_norm.py` (final_layer_norm), `L1/linear.py` (projection)
  - **`ClvpDecoder`** [wiring]: wires `ClvpDecoderLayer`, `ClvpConditioningEncoder`; direct `L1/embedding.py` (input_embeds_layer, position_embeds_layer), `L1/layer_norm.py` (layer_norm)
  - **`ClvpModel`** [wiring]: wires `ClvpDecoder`
  - **`ClvpForCausalLM`** [wiring]: wires `ClvpModel`, `ClvpConditioningEncoder`; direct `L1/linear.py` (lm_head)
  - **`ClvpModelForConditionalGeneration`** [wiring]: wires `ClvpEncoder` (×2: text and speech), `ClvpForCausalLM`
- **task heads (0)**: none

## codegen
- **src**: modeling_codegen.py
- **hidden_act**: gelu_new (activation_function)
- **status**: partial
- **classes**:
  - **`CodeGenAttention`** [compute]: `L1/linear.py + L1/sinusoidal_embed.py + L1/rotary_emb.py + L1/dense_attention.py + L1/store_kvcache.py` (fused QKV with mp_num=4 split, partial-rotary, GPTJ-style parallel attn+mlp; no exact L2 match — L2/attention.py uses GQA / non-fused)
  - **`CodeGenMLP`** [compute, copies GPTJMLP]: `L1/linear.py + L1/gelu.py + L1/linear.py` (fc_in → gelu_new → fc_out)
  - **`CodeGenBlock`** [wiring, copies GPTJBlock]: wires `CodeGenAttention`, `CodeGenMLP`; direct `L1/layer_norm.py` (ln_1) — parallel attn+mlp with single pre-norm (GPT-J style)
  - **`CodeGenModel`** [wiring]: wires `CodeGenBlock`; direct `L1/embedding.py` (wte), `L1/layer_norm.py` (ln_f)
  - **`CodeGenForCausalLM`** [wiring]: wires `CodeGenModel`; direct `L1/linear.py` (lm_head)
- **task heads (0)**: none

## cohere
- **src**: modeling_cohere.py (and modular_cohere.py)
- **hidden_act**: silu
- **status**: partial
- **classes**:
  - **`CohereLayerNorm`** [compute]: `L1/layer_norm.py` (centered LN with mean+variance and weight-only scale, no bias; close enough to standard LayerNorm with bias=False)
  - **`CohereRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (default-RoPE with interleaved layout; close enough)
  - **`CohereMLP`** [compute]: `L2/llama_mlp.py` (gate/up/down with silu, SwiGLU pattern)
  - **`CohereAttention`** [compute]: `L2/attention.py` (q/k/v/o with optional QKNorm via CohereLayerNorm; closest match — L2/attention.py doesn't apply per-head LayerNorm)
  - **`CohereDecoderLayer`** [wiring]: wires `CohereAttention`, `CohereMLP`, `CohereLayerNorm` (input_layernorm) — parallel attn+mlp with single pre-norm (residual + attn + mlp)
  - **`CohereModel`** [wiring]: wires `CohereDecoderLayer`, `CohereRotaryEmbedding`, `CohereLayerNorm` (norm); direct `L1/embedding.py` (embed_tokens)
  - **`CohereForCausalLM`** [wiring]: wires `CohereModel`; direct `L1/linear.py` (lm_head)
- **task heads (0)**: none

## cohere2
- **src**: modeling_cohere2.py (and modular_cohere2.py)
- **hidden_act**: silu
- **status**: partial
- **classes**:
  - **`Cohere2RotaryEmbedding`** [compute]: `L1/rotary_emb.py` (default-RoPE with interleaved layout)
  - **`Cohere2LayerNorm`** [compute]: `L1/layer_norm.py` (centered LN, weight-only, no bias)
  - **`Cohere2Attention`** [compute]: `L2/attention.py` (q/k/v/o with sliding-window for sliding_attention layers, no QKNorm; standard Llama-like with sliding-window)
  - **`Cohere2MLP`** [compute]: `L2/llama_mlp.py` (gate/up/down with silu)
  - **`Cohere2DecoderLayer`** [wiring]: wires `Cohere2Attention`, `Cohere2MLP`, `Cohere2LayerNorm` (input_layernorm) — parallel attn+mlp with single pre-norm
  - **`Cohere2Model`** [wiring]: wires `Cohere2DecoderLayer`, `Cohere2RotaryEmbedding`, `Cohere2LayerNorm` (norm); direct `L1/embedding.py` (embed_tokens)
  - **`Cohere2ForCausalLM`** [wiring]: wires `Cohere2Model`; direct `L1/linear.py` (lm_head)
- **task heads (0)**: none

## cohere2_vision
- **src**: modeling_cohere2_vision.py (and modular_cohere2_vision.py)
- **hidden_act**: n/a (uses nn.SiLU directly in projector)
- **status**: partial
- **classes**:
  - **`Cohere2VisionMultiModalProjector`** [compute]: `L1/linear.py + L1/silu.py + L1/linear.py` (pixel_shuffle reshape + linear_1 + SwiGLU split (chunk 2) + linear_2; closest to swiglu_mlp)
  - **`Cohere2VisionModel`** [wiring]: wires vision_tower (AutoModel — typically SigLIP), `Cohere2VisionMultiModalProjector`, language_model (AutoModel — typically Cohere2)
  - **`Cohere2VisionForConditionalGeneration`** [wiring]: wires `Cohere2VisionModel`; direct `L1/linear.py` (lm_head)
- **task heads (0)**: none

## cohere_asr
- **src**: modeling_cohere_asr.py (and modular_cohere_asr.py)
- **hidden_act**: relu (decoder MLP); encoder uses silu
- **status**: partial
- **classes**:
  - **`CohereAsrDecoderMLP`** [compute]: `L1/linear.py + L1/relu.py + L1/linear.py` (fc1 → relu → fc2)
  - **`CohereAsrSelfAttention`** [compute]: `L2/attention.py` (q/k/v/o with GQA; no RoPE applied — closest match — Llama-style without rotary)
  - **`CohereAsrCrossAttention`** [compute]: `L1/linear.py + L1/dense_attention.py` (cross-attn with EncoderDecoderCache; no exact L2 match)
  - **`CohereAsrDecoderLayer`** [wiring]: wires `CohereAsrSelfAttention`, `CohereAsrCrossAttention`, `CohereAsrDecoderMLP`; direct `L1/layer_norm.py` x3 (input_layernorm, post_attention_layernorm, final_layernorm)
  - **`CohereAsrDecoder`** [wiring]: wires `CohereAsrDecoderLayer`; direct `L1/embedding.py` (embed_tokens, pos_emb), `L1/layer_norm.py` x2 (norm, embedding_layernorm), `L1/linear.py` (proj from encoder hidden)
  - **`CohereAsrModel`** [wiring]: wires encoder (AutoModel), `CohereAsrDecoder`
  - **`CohereAsrForConditionalGeneration`** [wiring]: wires `CohereAsrModel`; direct `L1/linear.py` (proj_out)
- **task heads (0)**: none

## colmodernvbert
- **src**: modeling_colmodernvbert.py (and modular_colmodernvbert.py)
- **hidden_act**: n/a (thin wrapper around ModernVBert VLM)
- **status**: composable
- **classes**:
  - **`ColModernVBertForRetrieval`** [wiring]: wires vlm (AutoModel — ModernVBert); direct `L1/linear.py` (embedding_proj_layer); compute is L2 normalization on embeddings
- **task heads (0)**: none

## colpali
- **src**: modeling_colpali.py (and modular_colpali.py)
- **hidden_act**: n/a (thin wrapper around PaliGemma VLM)
- **status**: composable
- **classes**:
  - **`ColPaliForRetrieval`** [wiring]: wires vlm (AutoModel — PaliGemma); direct `L1/linear.py` (embedding_proj_layer); compute is L2 normalization on embeddings
- **task heads (0)**: none

## colqwen2
- **src**: modeling_colqwen2.py (and modular_colqwen2.py)
- **hidden_act**: n/a (thin wrapper around Qwen2-VL VLM)
- **status**: composable
- **classes**:
  - **`ColQwen2ForRetrieval`** [wiring]: wires vlm (AutoModel — Qwen2-VL); direct `L1/linear.py` (embedding_proj_layer); compute is L2 normalization on embeddings
- **task heads (0)**: none

## conditional_detr
- **src**: modeling_conditional_detr.py
- **hidden_act**: relu (activation_function)
- **status**: partial
- **classes**:
  - **`ConditionalDetrFrozenBatchNorm2d`** [compute]: `L1/frozen_batch_norm2d.py` (frozen BN with fixed running stats)
  - **`ConditionalDetrConvEncoder`** [wiring]: wires backbone (load_backbone — typically ResNet); replaces nn.BatchNorm2d with `ConditionalDetrFrozenBatchNorm2d`
  - **`ConditionalDetrSinePositionEmbedding`** [compute]: `L1/sinusoidal_embed.py` (2D sine pos embedding, generalized for images)
  - **`ConditionalDetrLearnedPositionEmbedding`** [compute]: `L1/embedding.py` x2 (row_embeddings, column_embeddings)
  - **`ConditionalDetrSelfAttention`** [compute]: `L1/linear.py + L1/dense_attention.py` (q/k get position-added input, v doesn't; non-causal; no exact L2 match)
  - **`ConditionalDetrDecoderSelfAttention`** [compute]: `L1/linear.py + L1/dense_attention.py` (separate content/pos projections for q/k, summed; no exact L2 match)
  - **`ConditionalDetrDecoderCrossAttention`** [compute]: `L1/linear.py + L1/dense_attention.py` (concatenates q_pos_sine + q_content; doubled head_dim; no exact L2 match)
  - **`ConditionalDetrMLP`** [compute]: `L1/linear.py + L1/relu.py + L1/linear.py` (fc1 → relu → fc2 with dropout)
  - **`ConditionalDetrEncoderLayer`** [wiring]: wires `ConditionalDetrSelfAttention`, `ConditionalDetrMLP`; direct `L1/layer_norm.py` x2 (self_attn_layer_norm, final_layer_norm — post-norm)
  - **`ConditionalDetrDecoderLayer`** [wiring]: wires `ConditionalDetrDecoderSelfAttention`, `ConditionalDetrDecoderCrossAttention`, `ConditionalDetrMLP`; direct `L1/layer_norm.py` x3
  - **`ConditionalDetrMLPPredictionHead`** [compute]: `L1/linear.py + L1/relu.py + L1/linear.py + ...` (N-layer MLP with relu between layers)
  - **`ConditionalDetrConvBlock`** [compute]: `L1/conv2d.py + L1/group_norm.py + L1/relu.py`
  - **`ConditionalDetrFPNFusionStage`** [wiring]: wires `ConditionalDetrConvBlock`; direct `L1/conv2d.py` (fpn_adapter), `L1/interpolate.py`
  - **`ConditionalDetrMaskHeadSmallConv`** [wiring]: wires `ConditionalDetrConvBlock` x2, `ConditionalDetrFPNFusionStage` x3; direct `L1/conv2d.py` (output_conv)
  - **`ConditionalDetrMHAttentionMap`** [compute]: `L1/linear.py + L1/softmax.py` (q/k linear projections + softmax — returns attention map only, no value multiply)
  - **`ConditionalDetrEncoder`** [wiring]: wires `ConditionalDetrEncoderLayer`
  - **`ConditionalDetrDecoder`** [wiring]: wires `ConditionalDetrDecoderLayer`; direct `L1/layer_norm.py`, `L1/linear.py`, `ConditionalDetrMLPPredictionHead`
  - **`ConditionalDetrModel`** [wiring]: wires `ConditionalDetrConvEncoder`, `ConditionalDetrEncoder`, `ConditionalDetrDecoder`, `ConditionalDetrSinePositionEmbedding` (or learned); direct `L1/linear.py` (input_projection), `L1/embedding.py` (query_position_embeddings)
- **task heads (2)**: ForObjectDetection, ForSegmentation — base + linear / mask head (per-task)

## convbert
- **src**: modeling_convbert.py
- **hidden_act**: gelu
- **status**: partial
- **classes**:
  - **`ConvBertEmbeddings`** [compute]: `L2/encoder_embeddings.py` (BERT-style word + position + token_type + LayerNorm with separate embedding_size)
  - **`SeparableConv1D`** [compute]: `L1/conv1d.py + L1/conv1d.py` (depthwise + pointwise separable conv1d with bias param)
  - **`ConvBertSelfAttention`** [compute]: `L1/linear.py + L1/conv1d.py + L1/dense_attention.py` (mixed conv-attention: standard q/k/v/softmax PLUS a parallel conv branch with key_conv_attn_layer + softmax-weighted unfold; output is concat of both)
  - **`ConvBertSelfOutput`** [compute]: `L2/encoder_attention.py` (dense + LayerNorm + residual)
  - **`ConvBertAttention`** [wiring]: wires `ConvBertSelfAttention`, `ConvBertSelfOutput`
  - **`GroupedLinearLayer`** [compute]: `L1/linear.py` (grouped linear with reshape; closest L1 op)
  - **`ConvBertIntermediate`** [compute]: `L1/linear.py + L1/gelu.py` (or GroupedLinearLayer + gelu)
  - **`ConvBertOutput`** [compute]: `L1/linear.py + L1/layer_norm.py`
  - **`ConvBertLayer`** [wiring]: wires `ConvBertAttention`, optional `ConvBertAttention` (cross), `ConvBertIntermediate`, `ConvBertOutput`
  - **`ConvBertEncoder`** [wiring]: wires `ConvBertLayer`
  - **`ConvBertPredictionHeadTransform`** [compute]: `L1/linear.py + L1/gelu.py + L1/layer_norm.py`
  - **`ConvBertSequenceSummary`** [compute]: `L1/linear.py + L1/tanh.py` (pooled summary head; copies XLMSequenceSummary)
  - **`ConvBertModel`** [wiring]: wires `ConvBertEmbeddings`, `ConvBertEncoder`; direct optional `L1/linear.py` (embeddings_project for separate embedding_size)
  - **`ConvBertGeneratorPredictions`** [compute]: `L1/linear.py + L1/gelu.py + L1/layer_norm.py`
  - **`ConvBertClassificationHead`** [compute]: `L1/linear.py + L1/tanh.py + L1/linear.py`
  - **`ConvBertForMaskedLM`** [wiring]: wires `ConvBertModel`, `ConvBertGeneratorPredictions`; direct `L1/linear.py` (generator_lm_head)
- **task heads (4)**: ForSequenceClassification, ForMultipleChoice, ForTokenClassification, ForQuestionAnswering — base + linear (per-task)

## convnext
- **src**: modeling_convnext.py
- **hidden_act**: gelu
- **status**: partial
- **classes**:
  - **`ConvNextDropPath`** [compute]: stochastic depth (no exact L1; closest is `L1/dropout.py`)
  - **`ConvNextLayerNorm`** [compute]: `L1/layer_norm.py` (LayerNorm with channels_first/channels_last data format)
  - **`ConvNextEmbeddings`** [compute]: `L1/conv2d.py + L1/layer_norm.py` (patch_embeddings + LN)
  - **`ConvNextLayer`** [compute]: `L1/conv2d.py + L1/layer_norm.py + L1/linear.py + L1/gelu.py + L1/linear.py` (depthwise conv + LN + pwconv1 + gelu + pwconv2 + layer_scale + drop_path + residual; ConvNeXt v1 block — same shape as L2/convnextv2_block.py but no GRN)
  - **`ConvNextStage`** [wiring]: wires `ConvNextLayer`, `ConvNextLayerNorm`; direct `L1/conv2d.py` (downsample)
  - **`ConvNextEncoder`** [wiring]: wires `ConvNextStage`
  - **`ConvNextModel`** [wiring]: wires `ConvNextEmbeddings`, `ConvNextEncoder`; direct `L1/layer_norm.py` (layernorm)
- **task heads (1)**: ForImageClassification — base + linear (per-task)

## convnextv2
- **src**: modeling_convnextv2.py
- **hidden_act**: gelu
- **status**: composable
- **classes**:
  - **`ConvNextV2DropPath`** [compute]: stochastic depth (no exact L1; closest is `L1/dropout.py`)
  - **`ConvNextV2GRN`** [compute]: `L1/grn.py` (Global Response Normalization)
  - **`ConvNextV2LayerNorm`** [compute]: `L1/layer_norm.py` (LayerNorm with channels_first/last support)
  - **`ConvNextV2Embeddings`** [compute]: `L1/conv2d.py + L1/layer_norm.py`
  - **`ConvNextV2Layer`** [compute]: `L2/convnextv2_block.py` (dwconv + LN + pwconv1 + gelu + GRN + pwconv2 + drop_path + residual)
  - **`ConvNextV2Stage`** [wiring]: wires `ConvNextV2Layer`, `ConvNextV2LayerNorm`; direct `L1/conv2d.py` (downsample)
  - **`ConvNextV2Encoder`** [wiring]: wires `ConvNextV2Stage`
  - **`ConvNextV2Model`** [wiring]: wires `ConvNextV2Embeddings`, `ConvNextV2Encoder`; direct `L1/layer_norm.py` (layernorm)
- **task heads (1)**: ForImageClassification — base + linear (per-task)

## cpmant
- **src**: modeling_cpmant.py
- **hidden_act**: gelu (uses torch.nn.GELU directly in DenseGatedACT)
- **status**: partial
- **classes**:
  - **`CpmAntLayerNorm`** [compute]: `L1/rms_norm.py` (RMSNorm despite the name)
  - **`CpmAntAttention`** [compute]: `L1/linear.py + L1/dense_attention.py + L1/store_kvcache.py` (q/k/v/o with absolute position bias added to scores; no exact L2 match — encoder_attention.py doesn't add position bias)
  - **`CpmAntSelfAttentionBlock`** [wiring]: wires `CpmAntLayerNorm`, `CpmAntAttention` — pre-norm with residual
  - **`CpmAntDenseGatedACT`** [compute]: `L1/linear.py + L1/gelu.py + L1/linear.py` (gate*hidden GeGLU pattern with GELU)
  - **`CpmAntFeedForward`** [wiring]: wires `CpmAntDenseGatedACT`; direct `L1/linear.py` (w_out)
  - **`CpmAntFFNBlock`** [wiring]: wires `CpmAntLayerNorm`, `CpmAntFeedForward` — pre-norm with residual
  - **`CpmAntTransformerBlock`** [wiring]: wires `CpmAntSelfAttentionBlock`, `CpmAntFFNBlock`
  - **`CpmAntEncoder`** [wiring]: wires `CpmAntTransformerBlock`, `CpmAntLayerNorm` (output_layernorm)
  - **`CpmAntIntermediate`** [compute]: `L1/linear.py + L1/gelu.py`
  - **`CpmAntSegmentPositionEmbedding`** [compute]: `L1/embedding.py` (relative position bucket lookup; closest L1 op)
  - **`CpmAntOutput`** [compute]: `L1/linear.py + L1/layer_norm.py`
  - **`CpmAntModel`** [wiring]: wires `CpmAntEncoder`, `CpmAntSegmentPositionEmbedding`; direct `L1/embedding.py` (input_embedding, segment_embedding, prompt_embedding)
  - **`CpmAntForCausalLM`** [wiring]: wires `CpmAntModel`; direct `L1/linear.py` (lm_head, optional)
- **task heads (0)**: none

## csm
- **src**: modeling_csm.py (and modular_csm.py)
- **hidden_act**: silu
- **status**: partial
- **classes**:
  - **`CsmRMSNorm`** [compute]: `L1/rms_norm.py`
  - **`CsmRotaryEmbedding`** [compute]: `L1/rotary_emb.py`
  - **`CsmMLP`** [compute]: `L2/llama_mlp.py` (gate/up/down with silu)
  - **`CsmAttention`** [compute]: `L2/attention.py` (q/k/v/o with RoPE + KV cache + GQA — Llama-style)
  - **`CsmDecoderLayer`** [wiring]: wires `CsmAttention`, `CsmMLP`, `CsmRMSNorm` x2 (input_layernorm, post_attention_layernorm)
  - **`CsmDepthDecoderModel`** [wiring]: wires `CsmDecoderLayer`, `CsmRotaryEmbedding`, `CsmRMSNorm` (norm); direct `L1/embedding.py` (embed_tokens), `L1/linear.py` (inputs_embeds_projector)
  - **`CsmCodebooksHead`** [compute]: `L1/linear.py` (codebook-specific weight per layer; closest L1 op)
  - **`CsmDepthDecoderForCausalLM`** [wiring]: wires `CsmDepthDecoderModel`, `CsmCodebooksHead`
  - **`CsmBackboneModelEmbeddings`** [compute]: `L1/embedding.py` (multi-codebook embedding)
  - **`CsmBackboneModel`** [wiring]: wires `CsmDecoderLayer`, `CsmBackboneModelEmbeddings`, `CsmRotaryEmbedding`, `CsmRMSNorm`
  - **`CsmForConditionalGeneration`** [wiring]: wires `CsmBackboneModel`, `CsmDepthDecoderForCausalLM`; direct `L1/linear.py` (lm_head)
- **task heads (0)**: none

## ctrl
- **src**: modeling_ctrl.py
- **hidden_act**: n/a (uses ReLU and GELU hardcoded)
- **status**: partial
- **classes**:
  - **`MultiHeadAttention`** [compute]: `L1/linear.py + L1/dense_attention.py + L1/store_kvcache.py` (q/k/v/o with sinusoidal positional encoding added externally; closest L1 ops)
  - **`EncoderLayer`** [wiring]: wires `MultiHeadAttention`; direct `L1/layer_norm.py` x2 (layernorm1, layernorm2), `L1/linear.py` x2 (ffn) and `L1/relu.py` (CTRL uses standard 2-layer FFN with ReLU)
  - **`CTRLModel`** [wiring]: wires `EncoderLayer`; direct `L1/embedding.py` (w), `L1/sinusoidal_embed.py` (positional encoding), `L1/layer_norm.py` (layernorm)
  - **`CTRLLMHeadModel`** [wiring]: wires `CTRLModel`; direct `L1/linear.py` (lm_head)
- **task heads (1)**: ForSequenceClassification — base + linear (per-task)

