## idefics
- **src**: modeling_idefics.py, vision.py, perceiver.py
- **hidden_act**: silu (text), gelu (vision)
- **status**: composable
- **classes**:
  - **`IdeficsVisionEmbeddings`** [compute]: `L1/conv2d.py + L1/embedding.py` (Conv2d patch embed + class token + learned position embedding with optional bicubic interpolation)
  - **`IdeficsVisionAttention`** [compute]: `L2/clip_attention.py` (q/k/v + ALL_ATTENTION_FUNCTIONS dispatch, non-causal, CLIP-style)
  - **`IdeficsVisionMLP`** [compute]: `L2/clip_mlp.py` (fc1 -> ACT2FN[gelu] -> fc2; CLIP-style MLP)
  - **`IdeficsVisionEncoderLayer`** [wiring]: wires `IdeficsVisionAttention`, `IdeficsVisionMLP`, `nn.LayerNorm` (x2)
  - **`IdeficsVisionEncoder`** [wiring]: wires `IdeficsVisionEncoderLayer`
  - **`IdeficsVisionTransformer`** [wiring]: wires `IdeficsVisionEmbeddings`, `IdeficsVisionEncoder`; direct `L1/layer_norm.py` (pre + post layernorm)
  - **`IdeficsPerceiverResampler`** [wiring]: wires `IdeficsPerceiverAttention`, `IdeficsMLP`; direct `L1/layer_norm.py` (final), latent params
  - **`IdeficsPerceiverAttention`** [compute]: `L1/linear.py + L1/layer_norm.py + L1/dense_attention.py` (Flamingo-style cross-attn with [context, latents] keys; stabilized softmax; no exact L2 match)
  - **`IdeficsMLP` (perceiver.py)** [compute]: `L1/layer_norm.py + L1/linear.py + L1/relu.py + L1/linear.py` (LN -> fc -> ReLU -> c_proj; no exact L2 match)
  - **`IdeficsDecoupledEmbedding`** [compute, inherits `nn.Embedding`]: `L1/embedding.py` (x2) (frozen base + trainable additional embedding lookup; no exact L2 match)
  - **`IdeficsDecoupledLinear`** [compute, inherits `nn.Linear`]: `L1/linear.py` (x2) (frozen base + trainable additional projection; no exact L2 match)
  - **`IdeficsRMSNorm`** [compute]: `L1/rms_norm.py` (Llama-style RMSNorm)
  - **`IdeficsEmbedding`** [compute]: `L1/rotary_emb.py` (Llama RoPE)
  - **`IdeficsMLP` (modeling_idefics.py)** [compute]: `L2/llama_mlp.py` (SwiGLU: down_proj(silu(gate_proj) * up_proj))
  - **`IdeficsAttention`** [compute]: `L2/attention.py` (Llama-style causal attn with RoPE + KV cache; cross-attn variant w/ optional QK RMSNorm; no exact L2 match for cross-attn variant)
  - **`IdeficsDecoderLayer`** [wiring]: wires `IdeficsAttention`, `IdeficsMLP`, `IdeficsRMSNorm` (x2)
  - **`IdeficsGatedCrossAttentionLayer`** [wiring]: wires `IdeficsAttention` (cross-attn), `IdeficsMLP`, `IdeficsRMSNorm` (x2); direct `nn.Tanh` + learnable alpha gates (no exact match — gated tanh residual)
  - **`IdeficsModel`** [wiring]: wires `IdeficsDecoupledEmbedding`, `IdeficsVisionTransformer`, `IdeficsPerceiverResampler` (optional), `IdeficsDecoderLayer`, `IdeficsGatedCrossAttentionLayer`, `IdeficsRMSNorm` (final)
  - **`IdeficsForVisionText2Text`** [wiring]: wires `IdeficsModel`; direct `IdeficsDecoupledLinear` (lm_head)

## idefics2
- **src**: modeling_idefics2.py
- **hidden_act**: gelu_pytorch_tanh (vision/perceiver), silu (text via Mistral)
- **status**: composable
- **classes**:
  - **`Idefics2VisionEmbeddings`** [compute]: `L1/conv2d.py + L1/embedding.py` (NaViT-style: Conv2d patch embed + bucketized 2D position embedding for variable-resolution images)
  - **`Idefics2VisionAttention`** [compute]: `L2/siglip_attention.py` (SigLIP-style non-causal q/k/v + ALL_ATTENTION_FUNCTIONS dispatch)
  - **`Idefics2VisionMLP`** [compute]: `L2/siglip_mlp.py` (fc1 -> gelu_pytorch_tanh -> fc2)
  - **`Idefics2MLP`** [compute]: `L2/llama_mlp.py` (SwiGLU: down_proj(silu(gate_proj) * up_proj); generic gate-up-down)
  - **`Idefics2MultiheadAttentionPoolingHead`** [compute]: `L1/linear.py + L1/layer_norm.py` + `Idefics2MLP` (uses torch.nn.MultiheadAttention; no exact L2 match — attention pooling head)
  - **`Idefics2EncoderLayer`** [wiring]: wires `Idefics2VisionAttention`, `Idefics2VisionMLP`, `nn.LayerNorm` (x2)
  - **`Idefics2Encoder`** [wiring]: wires `Idefics2EncoderLayer`
  - **`Idefics2VisionTransformer`** [wiring]: wires `Idefics2VisionEmbeddings`, `Idefics2Encoder`; direct `L1/layer_norm.py` (post_layernorm)
  - **`Idefics2RMSNorm`** [compute]: `L1/rms_norm.py`
  - **`Idefics2PerceiverAttention`** [compute]: `L1/linear.py + L1/dense_attention.py` (cross-attn over [context,latents] keys w/ GQA; no exact L2 match)
  - **`Idefics2PerceiverLayer`** [wiring]: wires `Idefics2RMSNorm` (x3), `Idefics2PerceiverAttention`, `Idefics2MLP`
  - **`Idefics2PerceiverResampler`** [wiring]: wires `Idefics2PerceiverLayer`, `Idefics2RMSNorm`; latents param
  - **`Idefics2Connector`** [wiring]: wires `Idefics2MLP` (modality_projection), `Idefics2PerceiverResampler`
  - **`Idefics2Model`** [wiring]: wires `Idefics2VisionTransformer`, `Idefics2Connector`, AutoModel text decoder
  - **`Idefics2ForConditionalGeneration`** [wiring]: wires `Idefics2Model`; direct `L1/linear.py` (lm_head)

## idefics3
- **src**: modeling_idefics3.py
- **hidden_act**: gelu_pytorch_tanh (vision); text via AutoModel (Llama3)
- **status**: composable
- **classes**:
  - **`Idefics3VisionEmbeddings`** [compute]: `L1/conv2d.py + L1/embedding.py` (NaViT-style patch embed; same as Idefics2)
  - **`Idefics3VisionAttention`** [compute]: `L2/siglip_attention.py` (SigLIP non-causal, copied from Siglip)
  - **`Idefics3VisionMLP`** [compute]: `L2/siglip_mlp.py` (fc1 -> gelu_pytorch_tanh -> fc2)
  - **`Idefics3SimpleMLP`** [compute]: `L1/linear.py` (single linear projection only)
  - **`Idefics3EncoderLayer`** [wiring]: wires `Idefics3VisionAttention`, `Idefics3VisionMLP`, `nn.LayerNorm` (x2)
  - **`Idefics3Encoder`** [wiring]: wires `Idefics3EncoderLayer`
  - **`Idefics3RMSNorm`** [compute]: `L1/rms_norm.py`
  - **`Idefics3Connector`** [wiring]: wires `Idefics3SimpleMLP`; pixel-shuffle reshape (no kernel — only view/permute)
  - **`Idefics3VisionTransformer`** [wiring]: wires `Idefics3VisionEmbeddings`, `Idefics3Encoder`; direct `L1/layer_norm.py` (post_layernorm)
  - **`Idefics3Model`** [wiring]: wires `Idefics3VisionTransformer`, `Idefics3Connector`, AutoModel text decoder
  - **`Idefics3ForConditionalGeneration`** [wiring]: wires `Idefics3Model`; direct `L1/linear.py` (lm_head)

## ijepa
- **src**: modeling_ijepa.py, modular_ijepa.py
- **hidden_act**: gelu
- **status**: composable
- **classes**:
  - **`IJepaPatchEmbeddings`** [compute]: `L1/conv2d.py` (Conv2d patch projection, no class token, no positions)
  - **`IJepaEmbeddings`** [wiring/compute]: wires `IJepaPatchEmbeddings`; direct learned position embedding param + optional mask token, no CLS (interpolation supports dynamic shapes)
  - **`IJepaSelfAttention`** [compute]: `L2/encoder_attention.py` (BERT-style q/k/v + ALL_ATTENTION_FUNCTIONS dispatch, non-causal, no out_proj)
  - **`IJepaSelfOutput`** [compute]: `L2/encoder_attention.py` (dense + dropout; residual added in IJepaLayer not here)
  - **`IJepaAttention`** [wiring]: wires `IJepaSelfAttention`, `IJepaSelfOutput`
  - **`IJepaIntermediate`** [compute]: `L1/linear.py + L1/gelu.py` (dense -> ACT2FN[gelu]; half of encoder MLP)
  - **`IJepaOutput`** [compute]: `L1/linear.py` (dense + dropout + residual; half of encoder MLP)
  - **`IJepaLayer`** [wiring]: wires `IJepaAttention`, `IJepaIntermediate`, `IJepaOutput`, `nn.LayerNorm` (x2; layernorm_before/after)
  - **`IJepaEncoder`** [wiring]: wires `IJepaLayer`
  - **`IJepaPooler`** [compute]: `L1/linear.py` + ACT2FN[pooler_act] (typically tanh -> `L1/tanh.py`)
  - **`IJepaModel`** [wiring]: wires `IJepaEmbeddings`, `IJepaEncoder`, optional `IJepaPooler`; direct `L1/layer_norm.py` (final layernorm)
- **task heads (1)**: ForImageClassification — base + linear (per-task)

## imagegpt
- **src**: modeling_imagegpt.py
- **hidden_act** (`activation_function`): quick_gelu
- **status**: composable
- **classes**:
  - **`ImageGPTLayerNorm`** [compute]: `L1/t5_layer_norm.py` (T5-style RMSNorm — no centering, weight only, no bias)
  - **`ImageGPTAttention`** [compute]: `L1/linear.py + L1/dense_attention.py + L1/store_kvcache.py` (GPT-2-style with Conv1D fused QKV; causal mask via tril buffer; optional cross-attn; no exact L2 match — Conv1D-based QKV is GPT2 not BERT)
  - **`ImageGPTMLP`** [compute]: `L1/linear.py + L1/quickgelu.py` (Conv1D c_fc -> quick_gelu -> Conv1D c_proj; GPT-2 style 2-layer FFN with non-standard activation)
  - **`ImageGPTBlock`** [wiring]: wires `ImageGPTAttention`, `ImageGPTMLP`, `ImageGPTLayerNorm` (x2 or x3 with crossattention), optional cross-attn `ImageGPTAttention`
  - **`ImageGPTModel`** [wiring]: wires `nn.Embedding` (wte, wpe), `ImageGPTBlock`, `ImageGPTLayerNorm` (ln_f); direct `L1/embedding.py` (x2)
  - **`ImageGPTForCausalImageModeling`** [wiring]: wires `ImageGPTModel`; direct `L1/linear.py` (lm_head)
- **task heads (1)**: ForImageClassification — base + linear (per-task)

## informer
- **src**: modeling_informer.py, modular_informer.py
- **hidden_act** (`activation_function`): gelu
- **status**: composable
- **classes**:
  - **`InformerFeatureEmbedder`** [compute]: `L1/embedding.py` (multiple categorical embeddings concatenated)
  - **`InformerStdScaler`** [compute]: pure tensor ops (mean/var-based standardization; no learnable params; no kb-nano kernel needed)
  - **`InformerMeanScaler`** [compute]: pure tensor ops (weighted-average scaling; no learnable params)
  - **`InformerNOPScaler`** [compute]: pure tensor ops (identity scaler returning ones)
  - **`InformerSinusoidalPositionalEmbedding`** [compute, inherits `nn.Embedding`]: `L1/embedding.py` (frozen sinusoidal weights)
  - **`InformerValueEmbedding`** [compute]: `L1/linear.py` (single bias-free linear projection)
  - **`InformerAttention`** [compute]: `L1/linear.py + L1/dense_attention.py + L1/store_kvcache.py` (Bart-style q/k/v + ALL_ATTENTION_FUNCTIONS dispatch w/ EncoderDecoderCache; no exact L2 match — encoder-decoder seq2seq)
  - **`InformerProbSparseAttention`** [compute]: `L1/linear.py + L1/dense_attention.py` (Informer ProbSparse: top-k query selection w/ KL-divergence; specialized; no exact L2 match)
  - **`InformerConvLayer`** [wiring/compute]: direct `L1/conv1d.py + L1/elu.py + L1/max_pool1d.py` (downConv -> BatchNorm1d -> ELU -> MaxPool1d; uses `nn.BatchNorm1d` — no L1 file for it)
  - **`InformerEncoderLayer`** [wiring]: wires `InformerAttention` or `InformerProbSparseAttention`, `nn.LayerNorm` (x2); direct `L1/linear.py` (fc1, fc2) + `L1/gelu.py`
  - **`InformerDecoderLayer`** [wiring]: wires `InformerAttention` (self+cross) or `InformerProbSparseAttention`, `nn.LayerNorm` (x3); direct `L1/linear.py` (fc1, fc2) + `L1/gelu.py`
  - **`InformerEncoder`** [wiring]: wires `InformerValueEmbedding`, `InformerSinusoidalPositionalEmbedding`, `InformerEncoderLayer`, optional `InformerConvLayer` chain; direct `L1/layer_norm.py`
  - **`InformerDecoder`** [wiring]: wires `InformerValueEmbedding`, `InformerSinusoidalPositionalEmbedding`, `InformerDecoderLayer`; direct `L1/layer_norm.py`
  - **`InformerModel`** [wiring]: wires scaler, optional `InformerFeatureEmbedder`, `InformerEncoder`, `InformerDecoder`
  - **`InformerForPrediction`** [wiring]: wires `InformerModel`, distribution-output projection (param projection + StudentT/Normal/NegBin head; no exact kb-nano kernel)

## instructblip
- **src**: modeling_instructblip.py
- **hidden_act**: gelu (vision and qformer)
- **status**: composable
- **classes**:
  - **`InstructBlipVisionEmbeddings`** [compute]: `L1/conv2d.py + L1/embedding.py` (Conv2d patch embed + class token + learned position embedding param + bicubic interp)
  - **`InstructBlipAttention`** [compute]: `L1/linear.py + L1/dense_attention.py` (BLIP-style fused QKV, optional q/v bias only, non-causal, projection out; no exact L2 — fused QKV not in standard CLIP attention)
  - **`InstructBlipMLP`** [compute]: `L2/clip_mlp.py` (fc1 -> ACT2FN[gelu] -> fc2; CLIP-pattern MLP)
  - **`InstructBlipEncoderLayer`** [wiring]: wires `InstructBlipAttention`, `InstructBlipMLP`, `nn.LayerNorm` (x2)
  - **`InstructBlipEncoder`** [wiring]: wires `InstructBlipEncoderLayer`
  - **`InstructBlipVisionModel`** [wiring]: wires `InstructBlipVisionEmbeddings`, `InstructBlipEncoder`; direct `L1/layer_norm.py` (post_layernorm)
  - **`InstructBlipQFormerMultiHeadAttention`** [compute]: `L2/encoder_attention.py` (BERT-style q/k/v with explicit softmax; supports cross-attn to vision encoder)
  - **`InstructBlipQFormerSelfOutput`** [compute]: `L2/encoder_attention.py` (BERT-style dense + LayerNorm + residual)
  - **`InstructBlipQFormerAttention`** [wiring]: wires `InstructBlipQFormerMultiHeadAttention`, `InstructBlipQFormerSelfOutput`
  - **`InstructBlipQFormerIntermediate`** [compute]: `L1/linear.py + L1/gelu.py` (BERT-style; copied from BertIntermediate)
  - **`InstructBlipQFormerOutput`** [compute]: `L1/linear.py + L1/layer_norm.py` (dense + LayerNorm + residual; copied from BertOutput)
  - **`InstructBlipQFormerLayer`** [wiring]: wires `InstructBlipQFormerAttention` (self), optional `InstructBlipQFormerAttention` (cross), `InstructBlipQFormerIntermediate` (x2 — query & text), `InstructBlipQFormerOutput` (x2)
  - **`InstructBlipQFormerEncoder`** [wiring]: wires `InstructBlipQFormerLayer`
  - **`InstructBlipQFormerEmbeddings`** [compute]: `L2/encoder_embeddings.py` (word + position + LayerNorm; BERT-style minus token_type)
  - **`InstructBlipQFormerModel`** [wiring]: wires `InstructBlipQFormerEmbeddings`, `InstructBlipQFormerEncoder`
  - **`InstructBlipModel`** [wiring]: wires `InstructBlipVisionModel`, `InstructBlipQFormerModel`, AutoModel language model; direct `L1/linear.py` (language_projection); query_tokens param
  - **`InstructBlipForConditionalGeneration`** [wiring]: wires `InstructBlipVisionModel`, `InstructBlipQFormerModel`, AutoModelForCausalLM/Seq2SeqLM; direct `L1/linear.py` (language_projection)

## instructblipvideo
- **src**: modeling_instructblipvideo.py, modular_instructblipvideo.py
- **hidden_act**: gelu (vision and qformer)
- **status**: composable
- **classes** (mirrors instructblip; per modular file most classes inherit unchanged):
  - **`InstructBlipVideoVisionEmbeddings`** [compute, inherits `InstructBlipVisionEmbeddings`]: `L1/conv2d.py + L1/embedding.py` (Conv2d patch + class token + position param; same as Idefics/InstructBlip)
  - **`InstructBlipVideoAttention`** [compute, inherits `InstructBlipAttention`]: `L1/linear.py + L1/dense_attention.py` (BLIP-style fused QKV non-causal)
  - **`InstructBlipVideoMLP`** [compute, inherits `InstructBlipMLP`]: `L2/clip_mlp.py` (fc1 -> gelu -> fc2)
  - **`InstructBlipVideoEncoderLayer`** [wiring]: wires `InstructBlipVideoAttention`, `InstructBlipVideoMLP`, `nn.LayerNorm` (x2)
  - **`InstructBlipVideoEncoder`** [wiring]: wires `InstructBlipVideoEncoderLayer`
  - **`InstructBlipVideoVisionModel`** [wiring]: wires `InstructBlipVideoVisionEmbeddings`, `InstructBlipVideoEncoder`; direct `L1/layer_norm.py`
  - **`InstructBlipVideoQFormerMultiHeadAttention`** [compute]: `L2/encoder_attention.py` (BERT-style q/k/v with explicit softmax; supports cross-attn)
  - **`InstructBlipVideoQFormerSelfOutput`** [compute]: `L2/encoder_attention.py` (dense + LayerNorm + residual)
  - **`InstructBlipVideoQFormerAttention`** [wiring]: wires `InstructBlipVideoQFormerMultiHeadAttention`, `InstructBlipVideoQFormerSelfOutput`
  - **`InstructBlipVideoQFormerIntermediate`** [compute]: `L1/linear.py + L1/gelu.py`
  - **`InstructBlipVideoQFormerOutput`** [compute]: `L1/linear.py + L1/layer_norm.py`
  - **`InstructBlipVideoQFormerLayer`** [wiring]: wires `InstructBlipVideoQFormerAttention` (self+optional cross), `InstructBlipVideoQFormerIntermediate` (x2), `InstructBlipVideoQFormerOutput` (x2)
  - **`InstructBlipVideoQFormerEncoder`** [wiring]: wires `InstructBlipVideoQFormerLayer`
  - **`InstructBlipVideoQFormerEmbeddings`** [compute]: `L2/encoder_embeddings.py` (word + position + LayerNorm)
  - **`InstructBlipVideoQFormerModel`** [wiring]: wires `InstructBlipVideoQFormerEmbeddings`, `InstructBlipVideoQFormerEncoder`
  - **`InstructBlipVideoModel`** [wiring]: wires `InstructBlipVideoVisionModel`, `InstructBlipVideoQFormerModel`, AutoModel language model; direct `L1/linear.py` (language_projection); query_tokens param; processes per-frame video
  - **`InstructBlipVideoForConditionalGeneration`** [wiring]: wires `InstructBlipVideoVisionModel`, `InstructBlipVideoQFormerModel`, AutoModelForCausalLM/Seq2SeqLM; direct `L1/linear.py`

## internvl
- **src**: modeling_internvl.py, modular_internvl.py
- **hidden_act**: gelu (vision); projector_hidden_act: gelu
- **status**: composable
- **classes**:
  - **`InternVLVisionRMSNorm`** [compute, inherits `LlamaRMSNorm`]: `L1/rms_norm.py`
  - **`InternVLVisionAttention`** [compute, inherits `JanusVisionAttention`]: `L1/linear.py + L1/dense_attention.py` (separate q/k/v, optional QK RMSNorm, projection_layer + dropout, non-causal; no exact L2 — adds projection_dropout & QK norm vs siglip_attention)
  - **`InternVLVisionPatchEmbeddings`** [compute]: `L1/conv2d.py` (Conv2d patch projection only)
  - **`InternVLVisionEmbeddings`** [wiring/compute]: wires `InternVLVisionPatchEmbeddings`; direct CLS token + optional mask token + optional learned position embedding (with bicubic interp)
  - **`InternVLVisionMLP`** [compute, inherits `CLIPMLP`]: `L2/clip_mlp.py` (fc1 -> ACT2FN[gelu] -> fc2)
  - **`InternVLVisionLayer`** [wiring/compute]: wires `InternVLVisionAttention`, `InternVLVisionMLP`, NORM2FN[norm_type] (x2; LayerNorm or RMSNorm); direct lambda_1/lambda_2 layer-scale params (timm-block style)
  - **`InternVLVisionEncoder`** [wiring]: wires `InternVLVisionLayer`
  - **`InternVLVisionModel`** [wiring]: wires `InternVLVisionEmbeddings`, `InternVLVisionEncoder`; direct `nn.Identity` or `L1/layer_norm.py` final
  - **`InternVLMultiModalProjector`** [compute]: `L1/layer_norm.py + L1/linear.py + L1/gelu.py + L1/linear.py` (LN -> linear_1 -> ACT2FN[gelu] -> linear_2)
  - **`InternVLModel`** [wiring, inherits `LlavaModel`]: wires AutoModel `vision_tower`, `InternVLMultiModalProjector`, AutoModel `language_model`
  - **`InternVLForConditionalGeneration`** [wiring, inherits `LlavaForConditionalGeneration`]: wires `InternVLModel`; direct `L1/linear.py` (lm_head)

## jamba
- **src**: modeling_jamba.py, modular_jamba.py
- **hidden_act**: silu
- **status**: kb_nano_l4 (`L4/jamba.py`)
- **classes**:
  - **`JambaRMSNorm`** [compute]: `L1/rms_norm.py`
  - **`JambaAttention`** [compute]: `L2/jamba_attention.py` (causal MHA, no RoPE, no QK-norm; Llama-style separate q/k/v + ALL_ATTENTION_FUNCTIONS)
  - **`JambaMambaMixer`** [compute]: `L2/jamba_mamba_mixer.py` (Mamba v1 selective scan with per-layer dt/B/C RMSNorms)
  - **`JambaMLP`** [compute]: `L2/jamba_mlp.py` (SwiGLU: fused gate_up_proj -> SiluAndMul -> down_proj)
  - **`JambaExperts`** [compute]: kb-nano L1 grouped GEMM ops (mirrors `L2/jamba_moe.py` per-expert GEMMs)
  - **`JambaSparseMoeBlock`** [compute]: `L2/jamba_moe.py` (sparse MoE with softmax+top-k routing; `JambaExperts` as sub-experts)
  - **`JambaAttentionDecoderLayer`** [wiring]: wires `JambaAttention`, `JambaMLP` or `JambaSparseMoeBlock`, `JambaRMSNorm` (x2)
  - **`JambaMambaDecoderLayer`** [wiring]: wires `JambaMambaMixer`, `JambaMLP` or `JambaSparseMoeBlock`, `JambaRMSNorm` (x2)
  - **`JambaModel`** [wiring]: wires `nn.Embedding` (embed_tokens), `JambaAttentionDecoderLayer` / `JambaMambaDecoderLayer` per `layers_block_type`, `JambaRMSNorm` (final norm)
  - **`JambaForCausalLM`** [wiring]: wires `JambaModel`; direct `L1/linear.py` (lm_head)
- **task heads (1)**: ForSequenceClassification — base + linear (per-task)

## janus
- **src**: modeling_janus.py, modular_janus.py
- **hidden_act**: gelu (vision and VQ-VAE)
- **status**: composable
- **classes**:
  - **`JanusVisionEmbeddings`** [compute]: `L1/conv2d.py + L1/embedding.py` (Conv2d patch embed + learned position embedding, no class token; bicubic interp)
  - **`JanusVisionAttention`** [compute]: `L1/linear.py + L1/dense_attention.py` (separate q/k/v + optional QK LayerNorm + projection_layer + projection_dropout, non-causal; no exact L2 match)
  - **`JanusVisionMLP`** [compute]: `L2/clip_mlp.py` (fc1 -> ACT2FN[gelu] -> dropout -> fc2 -> dropout; CLIP-style with extra dropouts)
  - **`JanusVisionEncoderLayer`** [wiring]: wires `JanusVisionAttention`, `JanusVisionMLP`, `nn.LayerNorm` (x2)
  - **`JanusVisionEncoder`** [wiring]: wires `JanusVisionEncoderLayer`
  - **`JanusVisionModel`** [wiring]: wires `JanusVisionEmbeddings`, `JanusVisionEncoder`; direct `L1/layer_norm.py` (post_layernorm)
  - **`JanusVisionAlignerMLP`** [compute]: `L1/linear.py + L1/gelu.py + L1/linear.py` (multi-layer; depth-deep MLP with gelu between layers)
  - **`JanusVQVAEVectorQuantizer`** [compute]: `L1/embedding.py` + custom L2-distance argmin codebook lookup (no exact kb-nano kernel — VQ codebook quantization)
  - **`JanusVQVAEResnetBlock`** [compute]: `L1/group_norm.py + L1/sigmoid.py + L1/conv2d.py` (x2) (Swish via x*sigmoid(x), GroupNorm + Conv2d residual block)
  - **`JanusVQVAEAttnBlock`** [compute]: `L1/group_norm.py + L1/conv2d.py + L1/bmm.py + L1/softmax.py` (1x1 Conv-based q/k/v over spatial dims with bmm-attention; no exact L2 match)
  - **`JanusVQVAEConvDownsample`** [compute]: `L1/conv2d.py` (asymmetric pad + Conv2d stride-2)
  - **`JanusVQVAEConvUpsample`** [compute]: `L1/conv2d.py` (Conv2d after spatial nearest-up; uses F.interpolate)
  - **`JanusVQVAEMidBlock`** [wiring]: wires `JanusVQVAEResnetBlock` (x2), `JanusVQVAEAttnBlock`
  - **`JanusVQVAEEncoder`** [wiring]: wires `JanusVQVAEResnetBlock`, `JanusVQVAEAttnBlock`, `JanusVQVAEConvDownsample`, `JanusVQVAEMidBlock`; direct `L1/conv2d.py` (conv_in, conv_out), `L1/group_norm.py` (norm_out)
  - **`JanusVQVAEDecoder`** [wiring]: wires `JanusVQVAEResnetBlock`, `JanusVQVAEAttnBlock`, `JanusVQVAEConvUpsample`, `JanusVQVAEMidBlock`; direct `L1/conv2d.py`, `L1/group_norm.py`
  - **`JanusVQVAE`** [wiring]: wires `JanusVQVAEEncoder`, `JanusVQVAEVectorQuantizer`, `JanusVQVAEDecoder`; direct `L1/conv2d.py` (quant_conv, post_quant_conv)
  - **`JanusVQVAEAlignerMLP`** [compute]: `L1/linear.py + L1/gelu.py + L1/linear.py` (multi-layer MLP; same shape as JanusVisionAlignerMLP)
  - **`JanusVQVAEHead`** [compute]: `L1/linear.py + L1/gelu.py + L1/linear.py` (proj_out -> ACT2FN[gelu] -> vision_head)
  - **`JanusModel`** [wiring]: wires `JanusVisionModel`, `JanusVisionAlignerMLP`, `JanusVQVAE`, `JanusVQVAEAlignerMLP`, `JanusVQVAEHead`, AutoModel language model
  - **`JanusForConditionalGeneration`** [wiring]: wires `JanusModel`; direct `L1/linear.py` (lm_head)

## jetmoe
- **src**: modeling_jetmoe.py, modular_jetmoe.py
- **hidden_act** (`activation_function`): silu
- **status**: composable
- **classes**:
  - **`JetMoeRMSNorm`** [compute]: `L1/rms_norm.py`
  - **`JetMoeRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (Llama RoPE; supports rope_type variants)
  - **`JetMoeParallelExperts`** [compute]: per-expert F.linear loop over expert weights `[num_experts, output_size, input_size]` (could map to `L1/moe_grouped_gemm.py` for Triton fused; current HF impl is naive sequential; no exact L2 match)
  - **`JetMoeTopKGating`** [compute]: `L1/linear.py + L1/softmax.py` (top-k routing with sort/scatter for token-expert grouping; specialized; no exact L2 match)
  - **`JetMoeMoE`** [wiring/compute]: wires `JetMoeTopKGating`, `JetMoeParallelExperts` (input_linear x2 fused), `JetMoeParallelExperts` (output_linear); SwiGLU activation between (gate*up); residual bias param; specialized — no exact kb-nano L2 match (closest: `L1/moe_grouped_gemm.py` + `L1/silu_and_mul.py`)
  - **`JetMoeMoA`** [compute]: Mixture-of-Attention experts: per-expert q-projection on input, per-expert o-projection on output; uses `JetMoeParallelExperts` (x2); router shared with attention; specialized — unique to JetMoe, no kb-nano kernel
  - **`JetMoeAttention`** [wiring/compute]: wires `JetMoeMoA`, `nn.Linear` (kv_proj fused KV); applies RoPE and ALL_ATTENTION_FUNCTIONS dispatch; uses MoA for q/o projections — no exact L2 match (specialized MoA attention)
  - **`JetMoeDecoderLayer`** [wiring]: wires `JetMoeAttention`, `JetMoeMoE`, `JetMoeRMSNorm` (x2)
  - **`JetMoeModel`** [wiring]: wires `nn.Embedding`, `JetMoeDecoderLayer`, `JetMoeRMSNorm`, `JetMoeRotaryEmbedding`
  - **`JetMoeForCausalLM`** [wiring]: wires `JetMoeModel`; direct `L1/linear.py` (lm_head)
- **task heads (1)**: ForSequenceClassification — base + linear (per-task)

## kimi_linear
- **src**: not present in this Transformers pin (no `models/kimi_linear/` folder); kb-nano has L4 implementation `L4/kimi_linear.py` with L2 ops (`L2/kda_attention.py`, `L2/kimi_delta_attention.py`, `L2/mla_attention.py`, `L2/kimi_moe.py`, `L3/kimi_linear_decoder.py`).
- **status**: not_in_hf

## kimi_vl
- **src**: not present in this Transformers pin (no `models/kimi_vl/` folder).
- **status**: not_in_hf

## kimi_vlm
- **src**: not present in this Transformers pin (no `models/kimi_vlm/` folder).
- **status**: not_in_hf

## kosmos2
- **src**: modeling_kosmos2.py
- **hidden_act** (text `activation_function`): gelu; vision `hidden_act`: quick_gelu
- **status**: composable
- **classes**:
  - **`Kosmos2VisionEmbeddings`** [compute]: `L1/conv2d.py + L1/embedding.py` (Conv2d patch + class token + learned position embedding with bicubic interp; CLIP-style)
  - **`Kosmos2VisionAttention`** [compute]: `L2/clip_attention.py` (CLIP-style q/k/v with non-causal ALL_ATTENTION_FUNCTIONS dispatch)
  - **`Kosmos2VisionMLP`** [compute]: `L1/linear.py + L1/quickgelu.py + L1/linear.py` (CLIP-style fc1 -> ACT2FN[quick_gelu] -> fc2)
  - **`Kosmos2VisionEncoderLayer`** [wiring]: wires `Kosmos2VisionAttention`, `Kosmos2VisionMLP`, `nn.LayerNorm` (x2)
  - **`Kosmos2VisionEncoder`** [wiring]: wires `Kosmos2VisionEncoderLayer`
  - **`Kosmos2VisionTransformer`** [wiring]: wires `Kosmos2VisionEmbeddings`, `Kosmos2VisionEncoder`; direct `L1/layer_norm.py` (pre + post layernorm)
  - **`Kosmos2TextSinusoidalPositionalEmbedding`** [compute]: pure-tensor sinusoidal embedding lookup with `register_buffer('weights')`; no learnable params
  - **`KosmosTextAttention`** [compute]: `L1/linear.py + L1/dense_attention.py + L1/store_kvcache.py` (Bart-style q/k/v + EncoderDecoderCache; optional inner_attn_ln; no exact L2 — encoder-decoder seq2seq)
  - **`Kosmos2TextFFN`** [compute]: `L1/linear.py + L1/gelu.py + L1/layer_norm.py + L1/linear.py` (fc1 -> gelu -> ffn_layernorm -> fc2; non-standard with intermediate LN)
  - **`Kosmos2TextBlock`** [wiring]: wires `KosmosTextAttention` (self), optional cross-attn `KosmosTextAttention`, `Kosmos2TextFFN`, `nn.LayerNorm` (x2 or x3)
  - **`Kosmos2TextTransformer`** [wiring]: wires `nn.Embedding`, `Kosmos2TextSinusoidalPositionalEmbedding`, `Kosmos2TextBlock`, `nn.LayerNorm` (final)
  - **`Kosmos2VisionModel`** [wiring]: wires `Kosmos2VisionTransformer`
  - **`Kosmos2TextModel`** [wiring]: wires `Kosmos2TextTransformer`
  - **`Kosmos2TextForCausalLM`** [wiring]: wires `Kosmos2TextTransformer`; direct `L1/linear.py` (lm_head)
  - **`Kosmos2ImageToTextProjection`** [wiring/compute]: wires `KosmosTextAttention` (cross-attn over [features, latent_query]); direct `L1/linear.py` (dense), `latent_query` param
  - **`Kosmos2Model`** [wiring]: wires `Kosmos2TextModel`, `Kosmos2VisionModel`, `Kosmos2ImageToTextProjection`
  - **`Kosmos2ForConditionalGeneration`** [wiring]: wires `Kosmos2TextForCausalLM`, `Kosmos2VisionModel`, `Kosmos2ImageToTextProjection`

## kosmos2_5
- **src**: modeling_kosmos2_5.py
- **hidden_act** (text `activation_function`): gelu; vision `dense_act_fn`: gelu_new
- **status**: composable
- **classes**:
  - **`Kosmos2_5LayerNorm`** [compute]: `L1/t5_layer_norm.py` (T5-style RMSNorm — variance-only, no centering, no bias)
  - **`Kosmos2_5VisionEmbeddings`** [compute]: `L1/linear.py + L1/embedding.py` (Pix2Struct-style: linear patch projection + row + column embeddings; flattened-patches input format)
  - **`Kosmos2_5VisionMlp`** [compute]: `L2/t5_dense.py` (T5DenseGatedActDense: wi_0 -> gelu_new gate * wi_1 linear -> wo; SwiGLU/GeGLU pattern)
  - **`Kosmos2_5VisionAttention`** [compute]: `L1/linear.py + L1/dense_attention.py` (separate q/k/v + output, non-causal, no bias)
  - **`Kosmos2_5VisionLayer`** [wiring]: wires `Kosmos2_5VisionAttention`, `Kosmos2_5VisionMlp`, `Kosmos2_5LayerNorm` (x2)
  - **`Kosmos2_5VisionEncoder`** [wiring]: wires `Kosmos2_5VisionLayer`
  - **`Kosmos2_5TextSinusoidalPositionalEmbedding`** [compute]: register_buffer sinusoidal weights (no learnable params)
  - **`Kosmos2_5TextFFN`** [compute]: `L1/linear.py + L1/gelu.py + L1/layer_norm.py + L1/linear.py` (fc1 -> gelu -> ffn_layernorm -> fc2; non-standard with intermediate LN)
  - **`Kosmos2_5TextAttention`** [compute]: `L1/linear.py + L1/dense_attention.py + L1/store_kvcache.py` (Bart-style q/k/v + DynamicCache; pre-scales query, no scaling in attn; no exact L2 match)
  - **`Kosmos2_5TextBlock`** [wiring]: wires `Kosmos2_5TextAttention`, `Kosmos2_5TextFFN`, `nn.LayerNorm` (x2)
  - **`Kosmos2_5TextTransformer`** [wiring]: wires `nn.Embedding` (embed_tokens, segment_emb), `Kosmos2_5TextSinusoidalPositionalEmbedding`, `Kosmos2_5TextBlock`, `nn.LayerNorm`
  - **`Kosmos2_5VisionModel`** [wiring]: wires `Kosmos2_5VisionEmbeddings`, `Kosmos2_5VisionEncoder`; direct `Kosmos2_5LayerNorm`
  - **`Kosmos2_5TextModel`** [wiring]: wires `Kosmos2_5TextTransformer`
  - **`Kosmos2_5TextForCausalLM`** [wiring]: wires `Kosmos2_5TextTransformer`; direct `L1/linear.py` (lm_head)
  - **`Kosmos2_5ImageToTextProjection`** [wiring/compute]: direct `L1/linear.py` (dense), `latent_query` param + cross-attn module (uses `Kosmos2_5TextAttention`)
  - **`Kosmos2_5Model`** [wiring]: wires `Kosmos2_5TextModel`, `Kosmos2_5VisionModel`, `Kosmos2_5ImageToTextProjection`
  - **`Kosmos2_5ForConditionalGeneration`** [wiring]: wires `Kosmos2_5TextForCausalLM`, `Kosmos2_5VisionModel`, `Kosmos2_5ImageToTextProjection`

## kyutai_speech_to_text
- **src**: modeling_kyutai_speech_to_text.py, modular_kyutai_speech_to_text.py
- **hidden_act**: silu
- **status**: composable
- **classes**:
  - **`KyutaiSpeechToTextFlexibleLinear`** [compute]: per-codebook stacked-weight linear (selects per-token weight via gather; specialized; no exact kb-nano kernel — multi-codebook batched GEMM)
  - **`KyutaiSpeechToTextEmbeddings`** [wiring/compute]: direct `L1/embedding.py` (single embed_tokens spanning text+audio codebooks); per-codebook offset addition before lookup
  - **`KyutaiSpeechToTextRMSNorm`** [compute]: `L1/rms_norm.py`
  - **`KyutaiSpeechToTextLinear`** [wiring]: wires `nn.Linear` or `KyutaiSpeechToTextFlexibleLinear` (use_flexible_linear flag)
  - **`KyutaiSpeechToTextRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (Llama RoPE; supports rope_type variants)
  - **`KyutaiSpeechToTextGatingMLP`** [compute]: `L2/llama_mlp.py` (SwiGLU: fc1 -> chunk-2 -> ACT2FN[silu] gate * up -> fc2; fc1/fc2 may be FlexibleLinear for depth decoder)
  - **`KyutaiSpeechToTextAttention`** [compute]: `L2/attention.py` (Llama-style q/k/v + GQA + RoPE + KV cache; q/k/v/o use `KyutaiSpeechToTextLinear` for codebook flexibility; explicit eager softmax in default path)
  - **`KyutaiSpeechToTextFlashAttention2`** [compute, inherits `KyutaiSpeechToTextAttention`]: same as parent w/ flash_attn varlen path
  - **`KyutaiSpeechToTextSdpaAttention`** [compute, inherits `KyutaiSpeechToTextAttention`]: same as parent w/ SDPA path
  - **`KyutaiSpeechToTextDecoderLayer`** [wiring]: wires `KyutaiSpeechToTextAttention`, `KyutaiSpeechToTextGatingMLP`, `KyutaiSpeechToTextRMSNorm` (x2)
  - **`KyutaiSpeechToTextModel`** [wiring]: wires `KyutaiSpeechToTextEmbeddings`, `KyutaiSpeechToTextDecoderLayer`, `KyutaiSpeechToTextRMSNorm`
  - **`KyutaiSpeechToTextForConditionalGeneration`** [wiring]: wires `KyutaiSpeechToTextModel`, AutoModel `codec_model` (Mimi); direct `L1/linear.py` (lm_head)

## lasr
- **src**: modeling_lasr.py, modular_lasr.py
- **hidden_act**: silu
- **status**: composable
- **classes**:
  - **`LasrEncoderSubsampling`** [compute]: `L1/linear.py + L1/relu.py + L1/conv1d.py` (dense_0 -> ReLU -> Conv1d (x2) -> ReLU -> dense_1; mel-bin to hidden subsampling)
  - **`LasrEncoderRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (Llama RoPE; supports rope_type variants)
  - **`LasrEncoderAttention`** [compute]: `L2/attention.py` (q/k/v + GQA + RoPE + ALL_ATTENTION_FUNCTIONS dispatch; non-causal — ASR encoder)
  - **`LasrEncoderConvolutionModule`** [compute]: `L1/conv1d.py + L1/silu.py` (Conformer convolution: pointwise_conv -> GLU -> depthwise_conv -> BatchNorm1d -> silu -> pointwise_conv; specialized — uses nn.BatchNorm1d, no kb-nano kernel)
  - **`LasrEncoderFeedForward`** [compute]: `L1/linear.py + L1/silu.py + L1/linear.py` (linear1 -> ACT2FN[silu] -> dropout -> linear2; standard 2-layer FFN)
  - **`LasrEncoderBlock`** [wiring]: wires `LasrEncoderFeedForward` (x2), `LasrEncoderAttention`, `LasrEncoderConvolutionModule`, `nn.LayerNorm` (x5); Conformer-style with weighted residuals
  - **`LasrEncoder`** [wiring]: wires `LasrEncoderSubsampling`, `LasrEncoderRotaryEmbedding`, `LasrEncoderBlock`
  - **`LasrForCTC`** [wiring]: wires `LasrEncoder`; direct `L1/linear.py` (CTC head)
- **task heads (1)**: ForCTC — CTC (per-task)

## layoutlm
- **src**: modeling_layoutlm.py
- **hidden_act**: gelu
- **status**: composable
- **classes**:
  - **`LayoutLMEmbeddings`** [compute]: `L1/embedding.py + L1/layer_norm.py` (word + 1D position + 4 2D bbox positions (x,y,h,w) + token_type + LayerNorm; extends BERT's encoder embeddings with bbox features; no exact L2 match)
  - **`LayoutLMSelfAttention`** [compute]: `L2/encoder_attention.py` (BERT-style q/k/v + ALL_ATTENTION_FUNCTIONS dispatch; no out_proj — that's in SelfOutput)
  - **`LayoutLMSelfOutput`** [compute]: `L2/encoder_attention.py` (dense + LayerNorm + residual; copied from BertSelfOutput)
  - **`LayoutLMAttention`** [wiring]: wires `LayoutLMSelfAttention`, `LayoutLMSelfOutput`
  - **`LayoutLMIntermediate`** [compute]: `L1/linear.py + L1/gelu.py` (BERT-style)
  - **`LayoutLMOutput`** [compute]: `L1/linear.py + L1/layer_norm.py` (dense + LayerNorm + residual)
  - **`LayoutLMLayer`** [wiring]: wires `LayoutLMAttention`, `LayoutLMIntermediate`, `LayoutLMOutput`
  - **`LayoutLMEncoder`** [wiring]: wires `LayoutLMLayer`
  - **`LayoutLMPooler`** [compute]: `L1/linear.py + L1/tanh.py` (BERT pooler)
  - **`LayoutLMPredictionHeadTransform`** [compute]: `L1/linear.py + L1/gelu.py + L1/layer_norm.py`
  - **`LayoutLMLMPredictionHead`** [wiring]: wires `LayoutLMPredictionHeadTransform`; direct `L1/linear.py` (decoder)
  - **`LayoutLMOnlyMLMHead`** [wiring]: wires `LayoutLMLMPredictionHead`
  - **`LayoutLMModel`** [wiring]: wires `LayoutLMEmbeddings`, `LayoutLMEncoder`, `LayoutLMPooler`
  - **`LayoutLMForMaskedLM`** [wiring]: wires `LayoutLMModel`, `LayoutLMOnlyMLMHead`
- **task heads (3)**: ForSequenceClassification, ForTokenClassification, ForQuestionAnswering — base + linear (per-task)

## layoutlmv2
- **src**: modeling_layoutlmv2.py
- **hidden_act**: gelu
- **status**: composable
- **classes**:
  - **`LayoutLMv2Embeddings`** [compute]: `L1/embedding.py + L1/layer_norm.py` (word + 1D position + 4 2D bbox positions + token_type + LayerNorm; coordinates+shapes split with smaller embed dims; no exact L2 match)
  - **`LayoutLMv2SelfAttention`** [compute]: `L1/linear.py + L1/dense_attention.py` (BERT-style q/k/v; optional fast_qkv fused QKV with q/v bias params; explicit relative position bias addition (rel_pos + rel_2d_pos); no ALL_ATTENTION_FUNCTIONS dispatch; no exact L2 match)
  - **`LayoutLMv2Attention`** [wiring]: wires `LayoutLMv2SelfAttention`, `LayoutLMv2SelfOutput`
  - **`LayoutLMv2SelfOutput`** [compute]: `L2/encoder_attention.py` (dense + LayerNorm + residual; BERT pattern)
  - **`LayoutLMv2Intermediate`** [compute]: `L1/linear.py + L1/gelu.py`
  - **`LayoutLMv2Output`** [compute]: `L1/linear.py + L1/layer_norm.py`
  - **`LayoutLMv2Layer`** [wiring]: wires `LayoutLMv2Attention`, `LayoutLMv2Intermediate`, `LayoutLMv2Output`
  - **`LayoutLMv2Encoder`** [wiring/compute]: wires `LayoutLMv2Layer`; direct `L1/linear.py` (rel_pos_bias, rel_pos_x_bias, rel_pos_y_bias for relative position bias bucketing); computes 1D and 2D relative positions
  - **`LayoutLMv2VisualBackbone`** [wiring]: wraps detectron2 FPN backbone; direct `nn.AvgPool2d` or `nn.AdaptiveAvgPool2d`; not a pure-kb-nano model — depends on external `detectron2` package
  - **`LayoutLMv2Pooler`** [compute]: `L1/linear.py + L1/tanh.py`
  - **`LayoutLMv2Model`** [wiring]: wires `LayoutLMv2Embeddings`, `LayoutLMv2VisualBackbone`, `LayoutLMv2Encoder`, `LayoutLMv2Pooler`; direct `L1/linear.py` (visual_proj), `L1/layer_norm.py` (visual_LayerNorm); requires detectron2
- **task heads (3)**: ForSequenceClassification, ForTokenClassification, ForQuestionAnswering — base + linear (per-task)

## layoutlmv3
- **src**: modeling_layoutlmv3.py
- **hidden_act**: gelu
- **status**: composable
- **classes**:
  - **`LayoutLMv3PatchEmbeddings`** [compute]: `L1/conv2d.py + L1/interpolate.py` (Conv2d patch projection with optional bicubic position embedding interpolation)
  - **`LayoutLMv3TextEmbeddings`** [compute]: `L1/embedding.py + L1/layer_norm.py` (word + token_type + 1D position + 4 2D bbox positions; cat-style (v2) bbox; no exact L2 match)
  - **`LayoutLMv3SelfAttention`** [compute]: `L1/linear.py + L1/dense_attention.py` (BERT-style q/k/v with explicit CogView attention (PB-Relax stabilized softmax) + relative + 2D position bias; no exact L2 match)
  - **`LayoutLMv3SelfOutput`** [compute]: `L2/encoder_attention.py` (RoBERTa pattern: dense + LayerNorm + residual)
  - **`LayoutLMv3Attention`** [wiring]: wires `LayoutLMv3SelfAttention`, `LayoutLMv3SelfOutput`
  - **`LayoutLMv3Layer`** [wiring]: wires `LayoutLMv3Attention`, `LayoutLMv3Intermediate`, `LayoutLMv3Output`
  - **`LayoutLMv3Encoder`** [wiring/compute]: wires `LayoutLMv3Layer`; direct `L1/linear.py` (rel_pos_bias, rel_pos_x_bias, rel_pos_y_bias)
  - **`LayoutLMv3Intermediate`** [compute]: `L1/linear.py + L1/gelu.py` (RoBERTa pattern)
  - **`LayoutLMv3Output`** [compute]: `L1/linear.py + L1/layer_norm.py`
  - **`LayoutLMv3Model`** [wiring]: wires optional `LayoutLMv3TextEmbeddings`, optional `LayoutLMv3PatchEmbeddings`, `LayoutLMv3Encoder`; direct `L1/layer_norm.py` (LayerNorm, norm), cls_token + pos_embed params
  - **`LayoutLMv3ClassificationHead`** [compute]: `L1/linear.py + L1/tanh.py + L1/linear.py` (head used by For* tasks)
- **task heads (3)**: ForTokenClassification, ForQuestionAnswering, ForSequenceClassification — base + linear (per-task)

## led
- **src**: modeling_led.py
- **hidden_act** (`activation_function`): gelu
- **status**: composable
- **classes**:
  - **`LEDLearnedPositionalEmbedding`** [compute, inherits `nn.Embedding`]: `L1/embedding.py` (offset-2 learned positional embedding)
  - **`LEDEncoderSelfAttention`** [compute]: Longformer sliding-window attention with global tokens; specialized chunked/windowed attention with sparse pattern; no kb-nano kernel (Longformer-specific)
  - **`LEDEncoderAttention`** [wiring]: wires `LEDEncoderSelfAttention`; direct `L1/linear.py` (output projection)
  - **`LEDDecoderAttention`** [compute]: `L1/linear.py + L1/dense_attention.py + L1/store_kvcache.py` (Bart-style q/k/v + EncoderDecoderCache; explicit bmm-based attention; no exact L2 — encoder-decoder seq2seq)
  - **`LEDEncoderLayer`** [wiring]: wires `LEDEncoderAttention`, `nn.LayerNorm` (x2); direct `L1/linear.py` (fc1, fc2) + `L1/gelu.py`
  - **`LEDDecoderLayer`** [wiring]: wires `LEDDecoderAttention` (self+cross), `nn.LayerNorm` (x3); direct `L1/linear.py` (fc1, fc2) + `L1/gelu.py`
  - **`LEDClassificationHead`** [compute]: `L1/linear.py + L1/tanh.py + L1/linear.py`
  - **`LEDEncoder`** [wiring]: wires `nn.Embedding`, `LEDLearnedPositionalEmbedding`, `LEDEncoderLayer`, `nn.LayerNorm`
  - **`LEDDecoder`** [wiring]: wires `nn.Embedding`, `LEDLearnedPositionalEmbedding`, `LEDDecoderLayer`, `nn.LayerNorm`
  - **`LEDModel`** [wiring]: wires `LEDEncoder`, `LEDDecoder`; shared embedding
  - **`LEDForConditionalGeneration`** [wiring]: wires `LEDModel`; direct `L1/linear.py` (lm_head)
- **task heads (2)**: ForQuestionAnswering, ForSequenceClassification — base + LEDClassificationHead (per-task)

## levit
- **src**: modeling_levit.py
- **hidden_act**: hardswish (hardcoded `nn.Hardswish()` throughout)
- **status**: composable
- **classes**:
  - **`LevitConvEmbeddings`** [compute]: `L1/conv2d.py + L1/batch_norm2d.py` (Conv2d + BatchNorm2d)
  - **`LevitPatchEmbeddings`** [wiring]: wires `LevitConvEmbeddings` (x4); direct `nn.Hardswish` (`L1/hardswish.py`) (x3); progressive Conv2d-BN-Hardswish stem
  - **`MLPLayerWithBN`** [compute]: `L1/linear.py + L1/batch_norm2d.py` (Linear + BN1d on flattened features; uses `nn.BatchNorm1d`)
  - **`LevitSubsample`** [compute]: pure spatial slicing reshape (no kernel; just reshape+stride indexing)
  - **`LevitAttention`** [compute]: `L1/linear.py + L1/batch_norm2d.py + L1/hardswish.py` (Linear+BN qkv -> attention with bucketed positional bias + Hardswish + Linear+BN projection; specialized — no exact L2 match)
  - **`LevitAttentionSubsample`** [compute]: `L1/linear.py + L1/batch_norm2d.py + L1/hardswish.py` (separate keys_values + queries_subsample paths with stride; specialized)
  - **`LevitMLPLayer`** [compute]: `L1/linear.py + L1/batch_norm2d.py + L1/hardswish.py` (linear_up -> Hardswish -> linear_down; 2x expansion)
  - **`LevitResidualLayer`** [wiring]: wires inner module; adds residual with optional drop_path
  - **`LevitStage`** [wiring]: wires `LevitAttention`, `LevitMLPLayer`, optional `LevitAttentionSubsample`, `LevitResidualLayer` wrapping
  - **`LevitEncoder`** [wiring]: wires `LevitStage`
  - **`LevitClassificationLayer`** [compute]: `L1/batch_norm2d.py + L1/linear.py` (BatchNorm1d + Linear; classification head)
  - **`LevitModel`** [wiring]: wires `LevitPatchEmbeddings`, `LevitEncoder`
- **task heads (2)**: ForImageClassification, ForImageClassificationWithTeacher — base + LevitClassificationLayer (per-task)

## lfm2
- **src**: modeling_lfm2.py, modular_lfm2.py
- **hidden_act**: silu (hardcoded `F.silu` in MLP)
- **status**: composable
- **classes**:
  - **`Lfm2RMSNorm`** [compute]: `L1/rms_norm.py`
  - **`Lfm2RotaryEmbedding`** [compute]: `L1/rotary_emb.py` (Llama RoPE; supports rope_type variants)
  - **`Lfm2MLP`** [compute]: `L2/llama_mlp.py` (SwiGLU: w2(silu(w1(x)) * w3(x)); standard SwiGLU pattern)
  - **`Lfm2Attention`** [compute]: `L2/attention.py` (Llama-style q/k/v + GQA + RoPE + KV cache + Q/K RMSNorm; QK-norm Llama variant)
  - **`Lfm2ShortConv`** [compute]: `L1/causal_conv1d.py + L1/linear.py` (Mamba-style short conv: in_proj -> chunk B,C,x -> B*x -> causal_conv1d -> C*conv_out -> out_proj; uses causal_conv1d_fn / causal_conv1d_update fast path)
  - **`Lfm2DecoderLayer`** [wiring]: wires `Lfm2Attention` (full_attention layers) or `Lfm2ShortConv` (conv layers), `Lfm2MLP`, `Lfm2RMSNorm` (x2)
  - **`Lfm2Model`** [wiring]: wires `nn.Embedding`, `Lfm2DecoderLayer`, `Lfm2RotaryEmbedding`, `Lfm2RMSNorm` (embedding_norm)
  - **`Lfm2ForCausalLM`** [wiring]: wires `Lfm2Model`; direct `L1/linear.py` (lm_head)

## lfm2_moe
- **src**: modeling_lfm2_moe.py, modular_lfm2_moe.py
- **hidden_act**: silu (hardcoded `F.silu` via Lfm2MLP)
- **status**: composable
- **classes**:
  - **`Lfm2MoeRMSNorm`** [compute, inherits `LlamaRMSNorm`]: `L1/rms_norm.py`
  - **`Lfm2MoeRotaryEmbedding`** [compute, inherits `Lfm2RotaryEmbedding`]: `L1/rotary_emb.py`
  - **`Lfm2MoeMLP`** [compute, inherits `Lfm2MLP`]: `L2/llama_mlp.py` (SwiGLU)
  - **`Lfm2MoeExperts`** [compute, inherits `Qwen2MoeExperts`]: kb-nano L1 grouped GEMM ops (`L1/moe_grouped_gemm.py` patterns) — per-expert SwiGLU experts
  - **`Lfm2MoeSparseMoeBlock`** [wiring/compute]: `L2/qwen3_moe.py` style or sigmoid-topk routing; wires `Lfm2MoeExperts` with router gate (custom Lfm2 routing — sigmoid top-k variant; closest kb-nano: `L1/sigmoid_topk.py + L2/shared_expert_moe.py`)
  - **`Lfm2MoeAttention`** [compute, inherits `Lfm2Attention`]: `L2/attention.py` (Llama-style + QK-norm + GQA + RoPE)
  - **`Lfm2MoeShortConv`** [compute, inherits `Lfm2ShortConv`]: `L1/causal_conv1d.py + L1/linear.py` (Mamba-style short conv with in_proj/out_proj)
  - **`Lfm2MoeDecoderLayer`** [wiring, inherits `Lfm2DecoderLayer`]: wires `Lfm2MoeAttention` or `Lfm2MoeShortConv`, `Lfm2MoeMLP` or `Lfm2MoeSparseMoeBlock`, `Lfm2MoeRMSNorm` (x2)
  - **`Lfm2MoeModel`** [wiring, inherits `MixtralModel`]: wires `nn.Embedding`, `Lfm2MoeDecoderLayer`, `Lfm2MoeRotaryEmbedding`, `Lfm2MoeRMSNorm`
  - **`Lfm2MoeForCausalLM`** [wiring, inherits `LlamaForCausalLM`]: wires `Lfm2MoeModel`; direct `L1/linear.py` (lm_head)

## lfm2_vl
- **src**: modeling_lfm2_vl.py, modular_lfm2_vl.py
- **hidden_act**: gelu (projector); text/vision via AutoModel children
- **status**: composable
- **classes**:
  - **`Lfm2VlMultiModalProjector`** [compute]: `L1/layer_norm.py + L1/linear.py + L1/gelu.py + L1/linear.py` (pixel-unshuffle reshape -> optional LN -> linear_1 -> ACT2FN[gelu] -> linear_2)
  - **`Lfm2VlModel`** [wiring, inherits `LlavaModel`]: wires AutoModel `vision_tower`, `Lfm2VlMultiModalProjector`, AutoModel `language_model`
  - **`Lfm2VlForConditionalGeneration`** [wiring, inherits `LlavaForConditionalGeneration`]: wires `Lfm2VlModel`; direct `L1/linear.py` (lm_head)

## lightglue
- **src**: modeling_lightglue.py, modular_lightglue.py
- **hidden_act**: gelu
- **status**: composable
- **classes**:
  - **`LightGluePositionalEncoder`** [compute]: `L1/linear.py` (single 2-d projector + cos/sin computation; specialized — RoPE-style position encoding from keypoint coords)
  - **`LightGlueAttention`** [compute, inherits `LlamaAttention`]: `L2/attention.py` (Llama-style q/k/v + GQA + RoPE + cross-attn variant; supports encoder_hidden_states for image-pair cross-attention)
  - **`LightGlueMLP`** [compute, inherits `CLIPMLP`]: `L1/linear.py + L1/layer_norm.py + L1/gelu.py + L1/linear.py` (fc1 -> LayerNorm -> ACT2FN[gelu] -> fc2; differs from CLIP — adds intermediate LN)
  - **`LightGlueTransformerLayer`** [wiring]: wires `LightGlueAttention` (self+cross), `LightGlueMLP` (x2; self_mlp + cross_mlp); SuperGlue-style image-pair cross-attention via flip
  - **`LightGlueMatchAssignmentLayer`** [compute]: `L1/linear.py` (final_projection + matchability) + custom `sigmoid_log_double_softmax` scoring (specialized; no kb-nano kernel)
  - **`LightGlueTokenConfidenceLayer`** [compute]: `L1/linear.py + L1/sigmoid.py` (single linear + sigmoid for token confidence)
  - **`LightGlueForKeypointMatching`** [wiring]: wires `LightGluePositionalEncoder`, `LightGlueTransformerLayer` (x num_layers), `LightGlueMatchAssignmentLayer`, optional `LightGlueTokenConfidenceLayer`; direct keypoint normalization, SuperPoint backbone via AutoModel
- **task heads (1)**: ForKeypointDetection — base + scoring (per-task)

