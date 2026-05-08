## blip_2
- **src**: modeling_blip_2.py
- **status**: composable
- **rationale**: Vision encoder is CLIP-style MHA + GELU MLP; Q-Former is BERT-derived self/cross attention with standard linear/softmax; LM head is OPT/T5 (composable). All compute maps to existing kb-nano L1/L2 ops.
- **classes**:
  - **`Blip2VisionEmbeddings`** [compute]: `L1/conv2d.py`, `L1/embedding.py` (Conv2d patch embed + class token + positional embedding)
  - **`Blip2Attention`** [compute]: `L2/encoder_attention.py`, `L1/linear.py`, `L1/softmax.py` (MHA with fused QKV linear; same compute as encoder_attention's EncoderSelfAttention (no causal, no rotary))
  - **`Blip2MLP`** [compute]: `L2/encoder_mlp.py`, `L1/gelu.py` (fc1 -> ACT2FN(gelu) -> fc2; matches EncoderIntermediate/Output pattern)
  - **`Blip2EncoderLayer`** [wiring]: Wiring: layer_norm + Blip2Attention + layer_norm + Blip2MLP + residuals
  - **`Blip2Encoder`** [wiring]: Wiring: stack of Blip2EncoderLayer
  - **`Blip2VisionModel`** [wiring]: Wiring around embeddings + encoder + post-layernorm
  - **`Blip2QFormerMultiHeadAttention`** [compute]: `L2/encoder_attention.py`, `L1/softmax.py` (BERT-style Q/K/V split MHA with optional cross-attention; same compute as EncoderSelfAttention)
  - **`Blip2QFormerSelfOutput`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (dense + layernorm + residual)
  - **`Blip2QFormerAttention`** [wiring]: Wiring: SelfAttention + SelfOutput
  - **`Blip2QFormerIntermediate`** [compute]: `L1/linear.py`, `L1/gelu.py` (BERT-style intermediate dense + gelu)
  - **`Blip2QFormerOutput`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (dense + layernorm + residual)
  - **`Blip2QFormerLayer`** [wiring]: Wiring: self attn + (cross attn) + intermediate + output
  - **`Blip2QFormerEncoder`** [wiring]: Wiring: stack of QFormerLayer
  - **`Blip2TextEmbeddings`** [compute]: `L1/embedding.py`, `L1/layer_norm.py` (token+pos+type embeddings + layer norm)
  - **`Blip2QFormerModel`** [wiring]: Wiring around embeddings + encoder
  - **`Blip2Model`** [wiring]: Wiring of vision + qformer + text projection + LM
  - **`Blip2TextModelWithProjection`** [wiring]: Wiring + final linear projection
  - **`Blip2VisionModelWithProjection`** [wiring]: Wiring + final linear projection
  - **`Blip2ForConditionalGeneration`** [wiring]: Wiring: vision + qformer + LM (OPT/T5)
  - **`Blip2ForImageTextRetrieval`** [wiring]: Wiring + retrieval logic

## bloom
- **src**: modeling_bloom.py
- **status**: partial
- **rationale**: ALiBi-biased multi-head attention via additive bias on attention scores -- supported by L1/dense_attention or L1/flash_attn_decode (alibi_slopes). MLP is fc1->gelu_new->fc2; LayerNorm + fused QKV linear.
- **classes**:
  - **`BloomBlock`** [compute]: no kb-nano kernel — ALiBi-biased multi-head attention via additive bias on attention scores -- supported by L1/dense_attention or L1/flash_attn_decode (alibi_slopes). MLP is fc1->gelu_new->fc2; LayerNorm + fused QKV line
  - **`BloomGelu`** [compute]: `L1/gelu.py` (Custom gelu_new wrapper; gelu kernel approximation matches the gelu L1 op)
  - **`BloomAttention`** [compute]: `L1/linear.py`, `L1/dense_attention.py`, `L1/softmax.py` (Fused QKV linear, MHA with ALiBi additive bias on attention scores; ALiBi handled as attn_mask add via dense_attention/flash_attn)
  - **`BloomMLP`** [compute]: `L1/linear.py`, `L1/gelu.py` (dense_h_to_4h -> gelu -> dense_4h_to_h; encoder MLP shape but with gelu_new)
  - **`BloomModel`** [wiring]: Wiring: word embedding + LayerNorm + ALiBi build + stack of blocks
  - **`BloomForCausalLM`** [wiring]: Wiring: BloomModel + lm_head
  - **`BloomForSequenceClassification`** [wiring]: Wiring + classification head
  - **`BloomForTokenClassification`** [wiring]: Wiring + token classification head
  - **`BloomForQuestionAnswering`** [wiring]: Wiring + QA head

## blt
- **src**: modular_blt.py
- **status**: composable
- **rationale**: BLT extends Mllama: Llama-style self-attn + cross-attn with QK-RMSNorm. SwiGLU MLP, RMSNorm, interleaved RoPE.
- **classes**:
  - **`BltMLP`** [compute]: `L2/llama_mlp.py` (SwiGLU: gate*up -> down; matches LlamaMLP)
  - **`BltRMSNorm`** [compute]: `L1/rms_norm.py` (Standard RMSNorm (Llama-family))
  - **`BltRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (Interleaved RoPE (uses repeat_interleave instead of cat); supported via rotary_emb is_neox=False)
  - **`BltSelfAttention`** [compute]: `L2/attention.py` (Llama-style GQA self-attention with RoPE; LlamaAttention pattern)
  - **`BltCrossAttention`** [compute]: `L2/whisper_attention.py`, `L1/rms_norm.py` (Cross-attention with QK norm; uses encoder hidden states; whisper_attention.WhisperDecoderCrossAttention pattern (KV from cross_attention_states); custom q_norm/k_norm via rms_norm)
  - **`BltTransformerLayer`** [wiring]: Wiring: rms_norm + self_attn + rms_norm + mlp
  - **`BltLocalEncoder`** [wiring]: Wiring: stack of BltTransformerLayer + cross-attn between encoder and global
  - **`BltLocalDecoder`** [wiring]: Wiring: stack of BltTransformerLayer with cross-attn
  - **`BltGlobalTransformer`** [wiring]: Wiring: stack of BltTransformerLayer (patch-level)
  - **`BltPatcher`** [wiring]: Wiring: dynamic byte-patching by entropy
  - **`BltModel`** [wiring]: Wiring: patcher + local_encoder + global + local_decoder
  - **`BltForCausalLM`** [wiring]: Wiring: BltModel + lm_head

## bridgetower
- **src**: modeling_bridgetower.py
- **status**: partial
- **rationale**: Vision tower: CLIP-style ResidualAttention using nn.MultiheadAttention; text tower: BERT-style self/cross attention; cross-modal layers compose the two. Compute is plain MHA with QuickGELU MLP.
- **classes**:
  - **`BridgeTowerAttention`** [wiring]: Sibling-wrapper around BridgeTowerSelfAttention (or BridgeTowerCrossAttention) + BridgeTowerSelfOutput. Per guideline 11 the bare *Attention class is wiring; the kernel lives on the sibling *SelfAttention.
  - **`BridgeTowerResidualAttention`** [compute]: `L1/linear.py`, `L1/dense_attention.py`, `L1/quickgelu.py`, `L1/layer_norm.py` (nn.MultiheadAttention compute = standard MHA with separate Q/K/V linears + dense_attention; QuickGELU MLP)
  - **`BridgeTowerTransformer`** [wiring]: Wiring: stack of ResidualAttention
  - **`BridgeTowerVisionEmbeddings`** [compute]: `L1/conv2d.py`, `L1/embedding.py` (Conv2d patch embed + class token + position embedding)
  - **`BridgeTowerVisionTransformer`** [wiring]: Wiring: embeddings + ln_pre + transformer + ln_post
  - **`BridgeTowerLinkTower`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (Cross-modal linking: dense + layernorm)
  - **`BridgeTowerSelfOutput`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (BERT self-output: dense + dropout + layernorm + residual)
  - **`BridgeTowerIntermediate`** [compute]: `L1/linear.py`, `L1/gelu.py` (BERT intermediate: dense + ACT2FN)
  - **`BridgeTowerOutput`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (BERT output: dense + dropout + layernorm + residual)
  - **`BridgeTowerPooler`** [wiring]: Wiring: pool first token + dense + tanh
  - **`BridgeTowerSelfAttention`** [compute]: `L2/encoder_attention.py` (BERT-style Q/K/V MHA with split projections; same as EncoderSelfAttention)
  - **`BridgeTowerCrossAttention`** [compute]: `L1/linear.py`, `L1/dense_attention.py`, `L1/softmax.py` (BERT-style cross-attention: query from hidden, K/V from encoder hidden; standard MHA)
  - **`BridgeTowerBertCrossLayer`** [wiring]: Wiring: self_attn + cross_attn + intermediate + output
  - **`BridgeTowerTextLayer`** [wiring]: Wiring: BERT layer with optional cross
  - **`BridgeTowerTextEncoder`** [wiring]: Wiring: stack of TextLayer
  - **`BridgeTowerTextEmbeddings`** [compute]: `L1/embedding.py`, `L1/layer_norm.py` (BERT-style: token+pos+type embed + layernorm)
  - **`BridgeTowerVisionModel`** [wiring]: Wiring around vision transformer
  - **`BridgeTowerTextModel`** [wiring]: Wiring around text encoder
  - **`BridgeTowerModel`** [wiring]: Wiring: vision + text + LinkTowers + cross-modal layers
  - **`BridgeTowerPredictionHeadTransform`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (MLM head: dense + ACT2FN + layernorm)
  - **`BridgeTowerMLMHead`** [compute]: `L1/linear.py` (Wiring: transform + decoder linear)
  - **`BridgeTowerITMHead`** [compute]: `L1/linear.py` (Image-text matching head: linear classifier)
  - **`BridgeTowerForMaskedLM`** [wiring]: Wiring + MLM head
  - **`BridgeTowerForImageAndTextRetrieval`** [wiring]: Wiring + ITM head
  - **`BridgeTowerContrastiveHead`** [compute]: `L1/linear.py` (Linear projection head)
  - **`BridgeTowerForContrastiveLearning`** [wiring]: Wiring + contrastive heads

## bros
- **src**: modeling_bros.py
- **status**: composable
- **rationale**: BERT-derived encoder with sinusoidal 2D bbox positional bias added to attention scores. All compute (linear, layernorm, gelu, MHA + bias add) maps to existing kb-nano L1/L2 ops.
- **classes**:
  - **`BrosPositionalEmbedding1D`** [wiring]: Sinusoidal embedding with sin/cos -- pure tensor ops; no kernel needed beyond elementwise sin/cos
  - **`BrosPositionalEmbedding2D`** [wiring]: Wiring: applies 1D positional embed to x and y coords
  - **`BrosBboxEmbeddings`** [compute]: `L1/linear.py` (Wiring: 2D pos embed + linear projection)
  - **`BrosTextEmbeddings`** [compute]: `L1/embedding.py`, `L1/layer_norm.py` (BERT-style word+pos+type embed + layernorm)
  - **`BrosSelfAttention`** [compute]: `L2/encoder_attention.py`, `L1/softmax.py` (BERT-style MHA with bbox positional bias added to attention scores; same as EncoderSelfAttention plus bias-add)
  - **`BrosSelfOutput`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (BERT self-output)
  - **`BrosAttention`** [wiring]: Wiring: SelfAttention + SelfOutput (rule 11 sibling pattern)
  - **`BrosIntermediate`** [compute]: `L1/linear.py`, `L1/gelu.py` (BERT intermediate: dense + gelu)
  - **`BrosOutput`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (BERT output: dense + layernorm)
  - **`BrosLayer`** [wiring]: Wiring: BERT-style layer
  - **`BrosPooler`** [wiring]: Wiring: dense + tanh on CLS token
  - **`BrosRelationExtractor`** [compute]: `L1/linear.py` (Wiring: linear projections for relation extraction)
  - **`BrosEncoder`** [wiring]: Wiring: stack of BrosLayer
  - **`BrosModel`** [wiring]: Wiring: text + bbox embeddings + encoder
  - **`BrosForTokenClassification`** [wiring]: Wiring + classifier
  - **`BrosSpadeEEForTokenClassification`** [wiring]: Wiring + entity extraction head
  - **`BrosSpadeELForTokenClassification`** [wiring]: Wiring + entity linking head

## camembert
- **src**: modeling_camembert.py
- **status**: composable
- **rationale**: Camembert is RoBERTa-derived (BERT-style encoder); all compute classes map to encoder_attention/encoder_mlp/embeddings.
- **classes**:
  - **`CamembertEmbeddings`** [compute]: `L1/embedding.py`, `L1/layer_norm.py` (RoBERTa token+pos+type embed + layernorm)
  - **`CamembertSelfAttention`** [compute]: `L2/encoder_attention.py` (BERT-style Q/K/V split MHA; same as EncoderSelfAttention)
  - **`CamembertCrossAttention`** [compute]: `L1/linear.py`, `L1/dense_attention.py`, `L1/softmax.py` (BERT-style cross-attention with K/V from encoder_hidden)
  - **`CamembertSelfOutput`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (BERT self-output)
  - **`CamembertAttention`** [wiring]: Wiring: SelfAttention + SelfOutput
  - **`CamembertIntermediate`** [compute]: `L2/encoder_mlp.py` (BERT intermediate: dense + gelu)
  - **`CamembertOutput`** [compute]: `L2/encoder_mlp.py` (BERT output: dense + layernorm)
  - **`CamembertLayer`** [wiring]: Wiring: BERT-style layer
  - **`CamembertLMHead`** [compute]: `L1/linear.py`, `L1/layer_norm.py`, `L1/gelu.py` (MLM head: dense + gelu + layernorm + decoder)
  - **`CamembertEncoder`** [wiring]: Wiring: stack of CamembertLayer
  - **`CamembertPooler`** [wiring]: Wiring: dense + tanh on CLS
  - **`CamembertModel`** [wiring]: Wiring: embeddings + encoder + pooler
  - **`CamembertForMaskedLM`** [wiring]: Wiring + MLM head
  - **`CamembertClassificationHead`** [compute]: `L1/linear.py` (Wiring: dense + tanh + linear projection)
  - **`CamembertForSequenceClassification`** [wiring]: Wiring + classification head
  - **`CamembertForMultipleChoice`** [wiring]: Wiring + multi-choice head
  - **`CamembertForTokenClassification`** [wiring]: Wiring + token classifier
  - **`CamembertForQuestionAnswering`** [wiring]: Wiring + QA head
  - **`CamembertForCausalLM`** [wiring]: Wiring + LM head

## canine
- **src**: modeling_canine.py
- **status**: composable
- **rationale**: BERT-style encoder with characters-to-molecules downsampling via Conv1d, ConvProjection upsampling, and optional block-wise local attention (chunked SDPA). All compute primitives exist (conv1d, layernorm, encoder_attention, gelu).
- **classes**:
  - **`CanineEmbeddings`** [compute]: `L1/embedding.py`, `L1/layer_norm.py` (Hashed character embeddings + position + type + layernorm)
  - **`CharactersToMolecules`** [compute]: `L1/conv1d.py`, `L1/layer_norm.py`, `L1/gelu.py` (Strided Conv1d downsample + activation + layernorm)
  - **`ConvProjection`** [compute]: `L1/conv1d.py`, `L1/layer_norm.py`, `L1/gelu.py` (Padded Conv1d + activation + layernorm + dropout)
  - **`CanineSelfAttention`** [compute]: `L2/encoder_attention.py`, `L1/softmax.py` (BERT-style MHA with separate from/to_tensor (cross-attn enabled))
  - **`CanineSelfOutput`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (BERT self-output)
  - **`CanineAttention`** [wiring]: Wiring: SelfAttention + SelfOutput; optional block-wise chunking via Python slicing
  - **`CanineIntermediate`** [compute]: `L2/encoder_mlp.py` (BERT intermediate)
  - **`CanineOutput`** [compute]: `L2/encoder_mlp.py` (BERT output)
  - **`CanineLayer`** [wiring]: Wiring: BERT-style layer
  - **`CanineEncoder`** [wiring]: Wiring: stack of CanineLayer
  - **`CaninePooler`** [wiring]: Wiring: dense + tanh
  - **`CaninePredictionHeadTransform`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (MLM transform)
  - **`CanineLMPredictionHead`** [wiring]: Wiring + decoder linear
  - **`CanineOnlyMLMHead`** [wiring]: Wiring around LMPredictionHead
  - **`CanineModel`** [wiring]: Wiring: embeddings + chars2mol + encoder + projection
  - **`CanineForSequenceClassification`** [wiring]: Wiring + classifier
  - **`CanineForMultipleChoice`** [wiring]: Wiring
  - **`CanineForTokenClassification`** [wiring]: Wiring
  - **`CanineForQuestionAnswering`** [wiring]: Wiring

## chameleon
- **src**: modeling_chameleon.py
- **status**: composable
- **rationale**: Decoder = Llama-style GQA attention with QK-LayerNorm + SwiGLU MLP + RMSNorm + RoPE. VQ-VAE encoder = Conv2d + GroupNorm + bmm-attention block. All compute exists in kb-nano L1.
- **classes**:
  - **`ChameleonRMSNorm`** [compute]: `L1/rms_norm.py` (Standard Llama-style RMSNorm)
  - **`ChameleonRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (NeoX-style RoPE with optional dynamic scaling)
  - **`ChameleonMLP`** [compute]: `L2/llama_mlp.py` (SwiGLU: down(act(gate) * up); same as LlamaMLP)
  - **`ChameleonLayerNorm`** [compute]: `L1/layer_norm.py` (LayerNorm computed only over last dim per head; underlying op is layer_norm)
  - **`ChameleonAttention`** [compute]: `L2/attention.py`, `L1/layer_norm.py` (Llama-style GQA attention with Q/K LayerNorm; LlamaAttention pattern + extra norm)
  - **`ChameleonDecoderLayer`** [wiring]: Wiring: rmsnorm + attn + rmsnorm + mlp
  - **`ChameleonSwinDecoderLayer`** [wiring]: Wiring: Swin-style post-norm decoder layer
  - **`ChameleonVQVAEVectorQuantizer`** [compute]: `L1/embedding.py` (Codebook lookup via embedding + nearest-neighbor distance)
  - **`ChameleonVQVAEEncoderConvDownsample`** [compute]: `L1/conv2d.py` (Strided Conv2d with manual asymmetric pad)
  - **`ChameleonVQVAEEncoderResnetBlock`** [compute]: `L1/conv2d.py`, `L1/group_norm.py`, `L1/sigmoid.py` (GroupNorm + Conv2d + sigmoid (silu) + residual)
  - **`ChameleonVQVAEEncoderAttnBlock`** [compute]: `L1/conv2d.py`, `L1/group_norm.py`, `L1/bmm.py`, `L1/softmax.py` (Self-attention via 1x1 conv Q/K/V + bmm + softmax)
  - **`ChameleonVQVAEEncoder`** [wiring]: Wiring: cascade of resnet/attn blocks
  - **`ChameleonVQVAE`** [wiring]: Wiring: encoder + quantizer
  - **`ChameleonModel`** [wiring]: Wiring: VQVAE + token embeddings + decoder layers + final norm
  - **`ChameleonForConditionalGeneration`** [wiring]: Wiring + lm_head

## chinese_clip
- **src**: modular_chinese_clip.py
- **status**: composable
- **rationale**: Text encoder is BERT-style (encoder_attention/encoder_mlp), vision encoder is CLIP-style (clip_attention/clip_mlp). All compute maps to existing kb-nano L1/L2 ops.
- **classes**:
  - **`ChineseCLIPTextEmbeddings`** [compute]: `L1/embedding.py`, `L1/layer_norm.py` (BERT-style word+pos+type embed + layernorm)
  - **`ChineseCLIPVisionEmbeddings`** [compute]: `L1/conv2d.py`, `L1/embedding.py` (Conv2d patch embed + class token + position embed)
  - **`ChineseCLIPTextSelfAttention`** [compute]: `L2/encoder_attention.py` (BERT-style Q/K/V split MHA)
  - **`ChineseCLIPTextSelfOutput`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (BERT self-output)
  - **`ChineseCLIPTextAttention`** [wiring]: Wiring: SelfAttention + SelfOutput
  - **`ChineseCLIPVisionAttention`** [compute]: `L2/clip_attention.py` (CLIP-style MHA with separate Q/K/V; matches CLIPAttention)
  - **`ChineseCLIPTextIntermediate`** [compute]: `L2/encoder_mlp.py` (BERT intermediate)
  - **`ChineseCLIPTextOutput`** [compute]: `L2/encoder_mlp.py` (BERT output)
  - **`ChineseCLIPVisionMLP`** [compute]: `L2/clip_mlp.py` (CLIP-style fc1 + ACT2FN(quickgelu) + fc2)
  - **`ChineseCLIPTextLayer`** [wiring]: Wiring: BERT-style layer
  - **`ChineseCLIPVisionLayer`** [wiring]: Wiring: CLIP-style layer
  - **`ChineseCLIPTextPooler`** [wiring]: Wiring: dense + tanh
  - **`ChineseCLIPTextEncoder`** [wiring]: Wiring: stack of TextLayer
  - **`ChineseCLIPVisionEncoder`** [wiring]: Wiring: stack of VisionLayer
  - **`ChineseCLIPVisionModel`** [wiring]: Wiring around vision encoder
  - **`ChineseCLIPTextModel`** [wiring]: Wiring around text encoder
  - **`ChineseCLIPModel`** [wiring]: Wiring: text + vision + projection

## chmv2
- **src**: modular_chmv2.py
- **status**: partial
- **rationale**: Depth estimation model that loads its backbone via HF AutoBackbone (load_backbone) -- requires a separate vision backbone (DPT-style) that kb-nano does not expose as an L4 pipeline. The DPT head itself (Reassemble + Fusion + UpsampleConv) is composable, but without a backbone the model cannot run.
- **classes**:
  - **`CHMv2ReassembleStage`** [compute]: no kb-nano kernel — Depth estimation model that loads its backbone via HF AutoBackbone (load_backbone) -- requires a separate vision backbone (DPT-style) that kb-nano does not expose as an L4 pipeline. The DPT head itsel
  - **`CHMv2ReassembleLayer`** [compute]: `L1/conv2d.py`, `L1/conv_transpose2d.py` (1x1 Conv2d + ConvTranspose2d/Conv2d resize)
  - **`CHMv2PreActResidualLayer`** [compute]: `L1/conv2d.py`, `L1/relu.py` (ReLU + Conv2d (x2) + residual)
  - **`CHMv2FeatureFusionLayer`** [compute]: `L1/conv2d.py`, `L1/interpolate.py` (Bilinear interp + residual + 1x1 Conv2d)
  - **`CHMv2UpsampleConvHead`** [compute]: `L1/conv2d.py`, `L1/relu.py`, `L1/interpolate.py` (Conv -> bilinear upsample -> Conv -> ReLU -> Conv)
  - **`CHMv2Head`** [wiring]: Wiring: reassemble + convs + fusion + depth conv head
  - **`CHMv2FeaturesToDepth`** [wiring]: Wiring: depth-bin conversion (softmax + dot product)
  - **`CHMv2ForDepthEstimation`** [wiring]: Wiring: backbone (load_backbone) + head + features-to-depth

## clap
- **src**: modeling_clap.py
- **status**: composable
- **rationale**: Audio tower is Swin-style window attention with relative position bias (table lookup) + RoBERTa text tower. Compute primitives (linear, softmax, dense_attention, layernorm, conv2d, dropout) all exist in kb-nano L1. Note: Clap audio uses original Swin attention (relative_position_bias_table), not SwinV2's CPB-MLP, so L2/swinv2_window_attention does not apply directly.
- **classes**:
  - **`ClapDropPath`** [wiring]: Stochastic depth -- pure tensor ops
  - **`ClapAudioAFFBlock`** [compute]: `L1/conv2d.py`, `L1/sigmoid.py` (Attentional feature fusion: small conv block + sigmoid gate)
  - **`ClapAudioPatchEmbed`** [compute]: `L1/conv2d.py`, `L1/layer_norm.py` (Conv2d patch embed + optional layernorm)
  - **`ClapAudioSelfAttention`** [compute]: `L1/linear.py`, `L1/softmax.py`, `L1/dense_attention.py` (Window MHA with table-lookup relative position bias added to scores)
  - **`ClapAudioSelfOutput`** [compute]: `L1/linear.py` (Dense + dropout)
  - **`ClapAudioAttention`** [wiring]: Wiring: SelfAttention + SelfOutput
  - **`ClapAudioIntermediate`** [compute]: `L1/linear.py`, `L1/gelu.py` (Swin intermediate: dense + ACT2FN)
  - **`ClapAudioOutput`** [compute]: `L1/linear.py` (Swin output: dense + dropout)
  - **`ClapAudioLayer`** [wiring]: Wiring: pre-LN + window-attn + droppath + LN + MLP
  - **`ClapAudioStage`** [wiring]: Wiring: stack of ClapAudioLayer per stage
  - **`ClapAudioPatchMerging`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (Spatial 2x downsample via concat-and-linear)
  - **`ClapAudioEncoder`** [wiring]: Wiring: stages + patch merging
  - **`ClapProjectionLayer`** [compute]: `L1/linear.py`, `L1/relu.py` (Linear + ReLU + Linear projection head)
  - **`ClapTextEmbeddings`** [compute]: `L1/embedding.py`, `L1/layer_norm.py` (RoBERTa-style word+pos+type embed)
  - **`ClapTextSelfAttention`** [compute]: `L2/encoder_attention.py` (BERT-style MHA)
  - **`ClapTextSelfOutput`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (BERT self-output)
  - **`ClapTextAttention`** [wiring]: Wiring
  - **`ClapTextIntermediate`** [compute]: `L2/encoder_mlp.py` (BERT intermediate)
  - **`ClapTextOutput`** [compute]: `L2/encoder_mlp.py` (BERT output)
  - **`ClapTextLayer`** [wiring]: Wiring
  - **`ClapTextEncoder`** [wiring]: Wiring
  - **`ClapTextPooler`** [wiring]: Wiring: dense + tanh
  - **`ClapAudioModel`** [wiring]: Wiring around audio encoder
  - **`ClapTextModel`** [wiring]: Wiring around text encoder
  - **`ClapModel`** [wiring]: Wiring: audio + text + projections
  - **`ClapTextModelWithProjection`** [wiring]: Wiring + projection head
  - **`ClapAudioModelWithProjection`** [wiring]: Wiring + projection head

## clip
- **src**: modeling_clip.py
- **status**: composable
- **rationale**: CLIP text and vision encoders use separate Q/K/V MHA + QuickGELU MLP; both have direct kb-nano L2 mappings (clip_attention, clip_mlp).
- **classes**:
  - **`CLIPVisionEmbeddings`** [compute]: `L1/conv2d.py`, `L1/embedding.py` (Conv2d patch embed + class token + position embed)
  - **`CLIPTextEmbeddings`** [compute]: `L2/clip_mlp.py` (CLIPTextEmbeddings is defined inside L2/clip_mlp.py)
  - **`CLIPAttention`** [compute]: `L2/clip_attention.py` (CLIP-style MHA with separate Q/K/V projections; verified __init__ + forward match)
  - **`CLIPMLP`** [compute]: `L2/clip_mlp.py` (fc1 + QuickGELU + fc2)
  - **`CLIPEncoderLayer`** [wiring]: Wiring: layer_norm + attn + layer_norm + mlp
  - **`CLIPEncoder`** [wiring]: Wiring: stack of EncoderLayer
  - **`CLIPTextModel`** [wiring]: Wiring around text encoder
  - **`CLIPVisionModel`** [wiring]: Wiring around vision encoder
  - **`CLIPModel`** [wiring]: Wiring: text + vision + projections
  - **`CLIPTextModelWithProjection`** [wiring]: Wiring + projection
  - **`CLIPVisionModelWithProjection`** [wiring]: Wiring + projection
  - **`CLIPForImageClassification`** [wiring]: Wiring + classifier

## clipseg
- **src**: modular_clipseg.py
- **status**: composable
- **rationale**: CLIPSeg shares CLIP text+vision encoders (clip_attention/clip_mlp). The decoder uses CLIPEncoderLayer with a small per-layer projection MLP and a Conv2d transpose mask head; all primitives exist.
- **classes**:
  - **`CLIPSegVisionEmbeddings`** [compute]: `L1/conv2d.py`, `L1/embedding.py` (Same as CLIPVisionEmbeddings)
  - **`CLIPSegTextEmbeddings`** [compute]: `L2/clip_mlp.py` (Same as CLIPTextEmbeddings)
  - **`CLIPSegAttention`** [compute]: `L2/clip_attention.py` (Identical to CLIPAttention)
  - **`CLIPSegMLP`** [compute]: `L2/clip_mlp.py` (fc1 + ACT2FN + fc2)
  - **`CLIPSegEncoderLayer`** [wiring]: Wiring: pre-norm + attn + pre-norm + mlp
  - **`CLIPSegDecoderLayer`** [wiring]: Wiring: post-norm decoder layer (same kernels)
  - **`CLIPSegEncoder`** [wiring]: Wiring
  - **`CLIPSegDecoder`** [compute]: `L1/conv_transpose2d.py`, `L1/conv2d.py`, `L1/linear.py` (Wiring: per-layer projections + decoder layers + transpose-conv mask head)
  - **`CLIPSegTextModel`** [wiring]: Wiring around text encoder
  - **`CLIPSegVisionModel`** [wiring]: Wiring around vision encoder
  - **`CLIPSegModel`** [wiring]: Wiring: text + vision + projections
  - **`CLIPSegForImageSegmentation`** [wiring]: Wiring: CLIPSegModel + CLIPSegDecoder

## clvp
- **src**: modeling_clvp.py
- **status**: composable
- **rationale**: CLVP encoder/decoder use RMSNorm, partial RoPE (rotary applied to q/k/value subsets), GLU MLP (gated chunk pattern). All compute primitives (rms_norm, rotary_emb, linear, gelu, dense_attention, softmax) exist.
- **classes**:
  - **`ClvpRMSNorm`** [compute]: `L1/rms_norm.py` (Standard Llama-style RMSNorm)
  - **`ClvpRotaryPositionalEmbedding`** [compute]: `L1/rotary_emb.py` (Standard RoPE inv_freq table)
  - **`ClvpSelfAttention`** [compute]: `L1/linear.py`, `L1/dense_attention.py`, `L1/rotary_emb.py`, `L1/softmax.py` (MHA with partial RoPE (applied to q, k AND v slices); standard linear + sdpa-equivalent)
  - **`ClvpGatedLinearUnit`** [compute]: `L1/linear.py`, `L1/gelu_and_mul.py` (GLU: chunk(2) -> first * gelu(second); canonical GeGLU pattern; per guideline 3 GeGLU MLPs use gelu_and_mul, not bare gelu.)
  - **`ClvpEncoderMLP`** [wiring]: Wiring: GLU + Linear
  - **`ClvpEncoderLayer`** [wiring]: Wiring: rmsnorm + attn + rmsnorm + mlp
  - **`ClvpSequenceSummary`** [compute]: `L1/linear.py` (Wiring: pooling + projection)
  - **`ClvpDecoderMLP`** [compute]: `L1/linear.py`, `L1/gelu.py` (fc + gelu_new + fc)
  - **`ClvpDecoderLayer`** [wiring]: Wiring: layer_norm + attn + layer_norm + mlp
  - **`ClvpConditioningEncoder`** [compute]: `L1/conv1d.py`, `L1/group_norm.py` (Wiring: text/audio conditioning encoder)
  - **`ClvpEncoder`** [wiring]: Wiring: stack of EncoderLayer
  - **`ClvpDecoder`** [wiring]: Wiring: stack of DecoderLayer
  - **`ClvpModel`** [wiring]: Wiring around decoder
  - **`ClvpForCausalLM`** [wiring]: Wiring + lm_head
  - **`ClvpModelForConditionalGeneration`** [wiring]: Wiring: encoder + decoder + heads

## codegen
- **src**: modeling_codegen.py
- **status**: partial
- **rationale**: GPT-J derivative: fused QKV with mp_num=4 split, partial NeoX RoPE (rotary_dim), MHA with no GQA, fc1+gelu_new+fc2 MLP, LayerNorm. All compute primitives exist (linear, rotary_emb, gelu, layer_norm, dense_attention).
- **classes**:
  - **`CodeGenBlock`** [compute]: no kb-nano kernel — GPT-J derivative: fused QKV with mp_num=4 split, partial NeoX RoPE (rotary_dim), MHA with no GQA, fc1+gelu_new+fc2 MLP, LayerNorm. All compute primitives exist (linear, rotary_emb, gelu, layer_norm, d
  - **`CodeGenAttention`** [compute]: `L1/linear.py`, `L1/rotary_emb.py`, `L1/dense_attention.py`, `L1/softmax.py` (Fused QKV linear with mp_num=4 reshape, partial rotary on first rotary_dim, MHA with KV cache)
  - **`CodeGenMLP`** [compute]: `L1/linear.py`, `L1/gelu.py` (fc_in + gelu_new + fc_out + dropout)
  - **`CodeGenModel`** [wiring]: Wiring: embeddings + dropout + stack of blocks + ln_f
  - **`CodeGenForCausalLM`** [wiring]: Wiring + lm_head

## cohere
- **src**: modular_cohere.py
- **status**: partial
- **rationale**: Llama-derived: GQA attention with optional QK-LayerNorm, SwiGLU MLP, custom CohereLayerNorm (centered LayerNorm = standard nn.LayerNorm without bias), interleaved RoPE. All primitives exist.
- **classes**:
  - **`CohereDecoderLayer`** [compute]: no kb-nano kernel — Llama-derived: GQA attention with optional QK-LayerNorm, SwiGLU MLP, custom CohereLayerNorm (centered LayerNorm = standard nn.LayerNorm without bias), interleaved RoPE. All primitives exist.
  - **`CohereLayerNorm`** [compute]: `L1/layer_norm.py` (Mean-subtracting LayerNorm; same as nn.LayerNorm fp32 path; layer_norm L1 op handles via standard params)
  - **`CohereRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (Interleaved RoPE (repeat_interleave instead of cat); supported via rotary_emb is_neox=False)
  - **`CohereMLP`** [compute]: `L2/llama_mlp.py` (SwiGLU: down(silu(gate) * up); same as LlamaMLP)
  - **`CohereAttention`** [compute]: `L2/attention.py`, `L1/layer_norm.py` (GQA Llama-style attention with optional Q/K LayerNorm (CohereLayerNorm))
  - **`CohereModel`** [wiring]: Wiring: stack of CohereDecoderLayer + final CohereLayerNorm
  - **`CohereForCausalLM`** [wiring]: Wiring + lm_head + logit_scale

## cohere2
- **src**: modular_cohere2.py
- **status**: partial
- **rationale**: Cohere2 extends Cohere with sliding-window attention (SWA), Gemma2-style hybrid local/global attention. Same compute primitives as Cohere; sliding window handled as attention mask in dense_attention.
- **classes**:
  - **`Cohere2DecoderLayer`** [compute]: no kb-nano kernel — Cohere2 extends Cohere with sliding-window attention (SWA), Gemma2-style hybrid local/global attention. Same compute primitives as Cohere; sliding window handled as attention mask in dense_attention.
  - **`Cohere2RotaryEmbedding`** [compute]: `L1/rotary_emb.py` (Same as Cohere RoPE)
  - **`Cohere2LayerNorm`** [compute]: `L1/layer_norm.py` (Same as Cohere LayerNorm)
  - **`Cohere2Attention`** [compute]: `L2/attention.py` (GQA attention with optional sliding-window mask per layer)
  - **`Cohere2Model`** [wiring]: Wiring: stack of Cohere2DecoderLayer + final norm + alternating SWA/global
  - **`Cohere2ForCausalLM`** [wiring]: Wiring + lm_head

## cohere2_vision
- **src**: modular_cohere2_vision.py
- **status**: partial
- **rationale**: VLM pipeline: SigLIP-style vision tower + Cohere2 LM + multi-modal projector (pixel-shuffle + SwiGLU). All sub-modules map to existing kb-nano kernels (siglip_attention/mlp, cohere2 stack, llama_mlp pattern for projector).
- **classes**:
  - **`Cohere2VisionModel`** [compute]: no kb-nano kernel — VLM pipeline: SigLIP-style vision tower + Cohere2 LM + multi-modal projector (pixel-shuffle + SwiGLU). All sub-modules map to existing kb-nano kernels (siglip_attention/mlp, cohere2 stack, llama_mlp p
  - **`Cohere2VisionMultiModalProjector`** [compute]: `L1/linear.py`, `L1/silu_and_mul.py` (Pixel-shuffle + linear + chunked SwiGLU + linear; per guideline 3 chunked SwiGLU uses silu_and_mul, not bare silu.)
  - **`Cohere2VisionForConditionalGeneration`** [wiring]: Wiring + lm_head

## cohere_asr
- **src**: modular_cohere_asr.py
- **status**: composable
- **rationale**: Whisper-style ASR derived from Moonshine: encoder (conv stem + self-attention) + decoder (self+cross attention with separate caches) + LayerNorm + ACT2FN MLP. All compute maps to whisper_attention pattern + standard linear/gelu/layer_norm.
- **classes**:
  - **`CohereAsrDecoderMLP`** [compute]: `L2/whisper_mlp.py` (fc1 + ACT2FN + fc2)
  - **`CohereAsrSelfAttention`** [compute]: `L2/whisper_attention.py` (GQA self-attention (causal) with separate Q/K/V; matches WhisperDecoderSelfAttention pattern)
  - **`CohereAsrCrossAttention`** [compute]: `L2/whisper_attention.py` (Cross-attention with K/V from encoder hidden states; matches WhisperDecoderCrossAttention pattern)
  - **`CohereAsrDecoderLayer`** [wiring]: Wiring: layer_norm + self_attn + layer_norm + cross_attn + layer_norm + mlp
  - **`CohereAsrDecoder`** [wiring]: Wiring: token+pos embed + stack of DecoderLayer + final LayerNorm
  - **`CohereAsrModel`** [wiring]: Wiring: encoder (Moonshine-style conv+attn) + decoder
  - **`CohereAsrForConditionalGeneration`** [wiring]: Wiring + lm_head

## colmodernvbert
- **src**: modular_colmodernvbert.py
- **status**: composable
- **rationale**: Pure retrieval wrapper around an underlying ModernVBert/ColPali VLM. Adds embedding projection + L2 normalization. No new compute primitives needed beyond the underlying VLM kernels.
- **classes**:
  - **`ColModernVBertForRetrieval`** [compute]: `L1/linear.py` (Wiring: AutoModel.from_config(vlm_config) + linear projection + L2 norm)

## colpali
- **src**: modular_colpali.py
- **status**: composable
- **rationale**: Retrieval wrapper around PaliGemma VLM (AutoModel.from_config). Adds linear embedding projection + L2 normalization. No compute beyond the underlying VLM.
- **classes**:
  - **`ColPaliForRetrieval`** [compute]: `L1/linear.py` (Wiring: AutoModel(vlm_config) + linear projection + L2 norm)

## colqwen2
- **src**: modular_colqwen2.py
- **status**: composable
- **rationale**: Retrieval wrapper around Qwen2-VL VLM. Same pattern as ColPali: AutoModel + projection + L2 norm.
- **classes**:
  - **`ColQwen2ForRetrieval`** [compute]: `L1/linear.py` (Wiring: AutoModel(vlm_config) + linear projection + L2 norm)

## conditional_detr
- **src**: modular_conditional_detr.py
- **status**: partial
- **rationale**: DETR-derived object detector that loads its CNN backbone via transformers.backbone_utils.load_backbone (typically ResNet from timm or AutoBackbone). kb-nano has no backbone-loading abstraction. The transformer encoder/decoder itself is composable (standard MHA + content/position-split decoder attn + linear projections), but without a backbone the model cannot run end-to-end.
- **classes**:
  - **`ConditionalDetrConvEncoder`** [compute]: no kb-nano kernel — DETR-derived object detector that loads its CNN backbone via transformers.backbone_utils.load_backbone (typically ResNet from timm or AutoBackbone). kb-nano has no backbone-loading abstraction. The tr
  - **`ConditionalDetrFrozenBatchNorm2d`** [compute]: `L1/frozen_batch_norm2d.py` (FrozenBatchNorm matches kb-nano L1)
  - **`ConditionalDetrSinePositionEmbedding`** [wiring]: Sin/cos 2D positional encoding -- pure tensor ops
  - **`ConditionalDetrLearnedPositionEmbedding`** [compute]: `L1/embedding.py` (Learned 2D position embeddings)
  - **`ConditionalDetrSelfAttention`** [compute]: `L1/linear.py`, `L1/dense_attention.py` (MHA with position embeddings added to Q/K (not V))
  - **`ConditionalDetrDecoderSelfAttention`** [compute]: `L1/linear.py`, `L1/dense_attention.py` (Content+position split projections combined for Q/K)
  - **`ConditionalDetrDecoderCrossAttention`** [compute]: `L1/linear.py`, `L1/dense_attention.py` (Concat-doubled Q/K for content+sine; standard SDPA on doubled head dim)
  - **`ConditionalDetrMLP`** [compute]: `L1/linear.py`, `L1/relu.py` (fc1 + ACT2FN + fc2)
  - **`ConditionalDetrEncoderLayer`** [wiring]: Wiring
  - **`ConditionalDetrDecoderLayer`** [wiring]: Wiring
  - **`ConditionalDetrMLPPredictionHead`** [compute]: `L1/linear.py`, `L1/relu.py` (Stack of linear + ReLU)
  - **`ConditionalDetrConvBlock`** [compute]: `L1/conv2d.py`, `L1/group_norm.py`, `L1/relu.py` (Conv + GroupNorm + ReLU)
  - **`ConditionalDetrFPNFusionStage`** [wiring]: Wiring: conv blocks + interpolate
  - **`ConditionalDetrMaskHeadSmallConv`** [compute]: `L1/conv2d.py`, `L1/group_norm.py`, `L1/relu.py` (Mask-head conv stack)
  - **`ConditionalDetrMHAttentionMap`** [compute]: `L1/linear.py`, `L1/softmax.py` (MHA-derived spatial attention map for segmentation)
  - **`ConditionalDetrEncoder`** [wiring]: Wiring
  - **`ConditionalDetrDecoder`** [wiring]: Wiring
  - **`ConditionalDetrModel`** [wiring]: Wiring: backbone + encoder + decoder
  - **`ConditionalDetrForObjectDetection`** [wiring]: Wiring + class/bbox heads
  - **`ConditionalDetrForSegmentation`** [wiring]: Wiring + mask head

## convbert
- **src**: modeling_convbert.py
- **status**: composable
- **rationale**: BERT-derived encoder with span-based dynamic convolutions inside attention (SeparableConv1D + GroupedLinearLayer + softmax). All primitives exist (conv1d, linear, softmax, layer_norm, gelu).
- **classes**:
  - **`ConvBertEmbeddings`** [compute]: `L1/embedding.py`, `L1/layer_norm.py` (BERT-style word+pos+type embed + layernorm + projection)
  - **`SeparableConv1D`** [compute]: `L1/conv1d.py` (Depthwise + pointwise conv1d)
  - **`ConvBertSelfAttention`** [compute]: `L2/encoder_attention.py`, `L1/conv1d.py`, `L1/softmax.py` (BERT self-attn + parallel span-conv branch (SeparableConv1D) merged)
  - **`ConvBertSelfOutput`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (BERT self-output)
  - **`ConvBertAttention`** [wiring]: Wiring: SelfAttention + SelfOutput
  - **`GroupedLinearLayer`** [compute]: `L1/linear.py` (Block-diagonal linear realised via reshape + bmm/linear; uses linear primitive)
  - **`ConvBertIntermediate`** [compute]: `L1/linear.py`, `L1/gelu.py` (BERT intermediate (with optional grouped linear))
  - **`ConvBertOutput`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (BERT output (with optional grouped linear))
  - **`ConvBertLayer`** [wiring]: Wiring: BERT-style layer
  - **`ConvBertEncoder`** [wiring]: Wiring: stack of ConvBertLayer
  - **`ConvBertPredictionHeadTransform`** [compute]: `L1/linear.py`, `L1/layer_norm.py`, `L1/gelu.py` (MLM transform)
  - **`ConvBertSequenceSummary`** [compute]: `L1/linear.py` (Pooling + summary)
  - **`ConvBertModel`** [wiring]: Wiring: embeddings + encoder
  - **`ConvBertGeneratorPredictions`** [compute]: `L1/linear.py`, `L1/layer_norm.py`, `L1/gelu.py` (Generator MLM head transform)
  - **`ConvBertForMaskedLM`** [wiring]: Wiring + generator + MLM
  - **`ConvBertClassificationHead`** [compute]: `L1/linear.py` (Classification head)
  - **`ConvBertForSequenceClassification`** [wiring]: Wiring + classifier
  - **`ConvBertForMultipleChoice`** [wiring]: Wiring
  - **`ConvBertForTokenClassification`** [wiring]: Wiring
  - **`ConvBertForQuestionAnswering`** [wiring]: Wiring + QA head

## convnext
- **src**: modeling_convnext.py
- **status**: composable
- **rationale**: Pure convolutional backbone: depthwise Conv2d + LayerNorm + pointwise Conv2d + GELU + DropPath. All primitives in L1 (conv2d, layer_norm, gelu).
- **classes**:
  - **`ConvNextDropPath`** [wiring]: Stochastic depth -- pure tensor ops
  - **`ConvNextLayerNorm`** [compute]: `L1/layer_norm.py`, `L1/layer_norm2d.py` (Channels-last and channels-first layernorm)
  - **`ConvNextEmbeddings`** [compute]: `L1/conv2d.py`, `L1/layer_norm.py` (Patch-embed Conv2d + layernorm)
  - **`ConvNextLayer`** [compute]: `L1/conv2d.py`, `L1/layer_norm.py`, `L1/linear.py`, `L1/gelu.py` (Depthwise Conv2d + LayerNorm + pointwise (1x1) + GELU + pointwise + DropPath)
  - **`ConvNextStage`** [wiring]: Wiring: optional downsample + stack of ConvNextLayer
  - **`ConvNextEncoder`** [wiring]: Wiring: stages
  - **`ConvNextModel`** [wiring]: Wiring: embeddings + encoder + final layernorm
  - **`ConvNextForImageClassification`** [wiring]: Wiring + classifier
  - **`ConvNextBackbone`** [wiring]: Wiring around ConvNextModel for backbone use

## convnextv2
- **src**: modeling_convnextv2.py
- **status**: kb_nano_l4
- **rationale**: ConvNeXt V2 has a dedicated kb-nano L4 pipeline at tasks/baseline/L4/convnextv2.py.
- **classes**:
  - **`ConvNextV2DropPath`** [wiring]: Stochastic depth
  - **`ConvNextV2GRN`** [compute]: `L1/grn.py` (Global Response Normalization (V2-specific))
  - **`ConvNextV2LayerNorm`** [compute]: `L1/layer_norm.py`, `L1/layer_norm2d.py` (Channels-last and channels-first layernorm)
  - **`ConvNextV2Embeddings`** [compute]: `L1/conv2d.py`, `L1/layer_norm.py` (Patch-embed + layernorm)
  - **`ConvNextV2Layer`** [compute]: `L3/convnextv2_stage.py` (Depthwise conv + LayerNorm + pointwise + GELU + GRN + pointwise + DropPath)
  - **`ConvNextV2Stage`** [compute]: `L3/convnextv2_stage.py` (Wiring: downsample + stack of layers)
  - **`ConvNextV2Encoder`** [wiring]: Wiring: stages
  - **`ConvNextV2Model`** [compute]: `L4/convnextv2.py` (Top-level pipeline; exists as kb-nano L4 pipeline)
  - **`ConvNextV2ForImageClassification`** [compute]: `L4/convnextv2.py` (Wiring + classifier; exposed via L4 pipeline)
  - **`ConvNextV2Backbone`** [wiring]: Wiring for backbone use

## cpmant
- **src**: modeling_cpmant.py
- **status**: composable
- **rationale**: Llama-style RMSNorm + Gated GELU MLP + standard MHA with relative-position bias added to scores. All compute (linear, rms_norm, gelu, softmax, dense_attention) exists in L1.
- **classes**:
  - **`CpmAntLayerNorm`** [compute]: `L1/rms_norm.py` (RMSNorm (despite the class name); same compute as standard rms_norm)
  - **`CpmAntAttention`** [compute]: `L1/linear.py`, `L1/dense_attention.py`, `L1/softmax.py` (Standard MHA with separate Q/K/V + relative position bias added; masked_fill for padding)
  - **`CpmAntSelfAttentionBlock`** [wiring]: Wiring: rmsnorm + attn + residual
  - **`CpmAntDenseGatedACT`** [compute]: `L1/linear.py`, `L1/gelu_and_mul.py` (GeGLU: gelu(w_0(x)) * w_1(x); per guideline 3 GeGLU MLPs use gelu_and_mul, not bare gelu.)
  - **`CpmAntFeedForward`** [wiring]: Wiring: GeGLU + linear out
  - **`CpmAntFFNBlock`** [wiring]: Wiring: rmsnorm + ffn + residual
  - **`CpmAntTransformerBlock`** [wiring]: Wiring: SelfAttentionBlock + FFNBlock
  - **`CpmAntEncoder`** [wiring]: Wiring: stack of TransformerBlock + final rmsnorm
  - **`CpmAntIntermediate`** [compute]: `L1/linear.py`, `L1/gelu.py` (Linear + ACT2FN)
  - **`CpmAntSegmentPositionEmbedding`** [compute]: `L1/embedding.py` (Segment + relative-position bucket embedding (T5-like))
  - **`CpmAntOutput`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (Linear + layernorm)
  - **`CpmAntModel`** [wiring]: Wiring: embeddings + segment_pos + encoder
  - **`CpmAntForCausalLM`** [wiring]: Wiring + lm_head
