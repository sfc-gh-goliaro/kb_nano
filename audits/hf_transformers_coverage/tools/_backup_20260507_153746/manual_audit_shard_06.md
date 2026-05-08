# Manual audit shard 06 (glm_image .. ibert)

## glm_image
- **src**: modeling_glm_image.py (and modular_glm_image.py)
- **hidden_act**: silu (text), gelu (vision)
- **status**: composable
- **classes**:
  - **`GlmImageVisionMLP`** [compute, inherits `SiglipMLP`]: `L2/siglip_mlp.py` (modular says inherits SiglipMLP; expanded body matches fc1 -> gelu -> fc2)
  - **`GlmImageVisionAttention`** [compute, inherits `Glm4vVisionAttention`]: `L1/linear.py + L1/dense_attention.py` (no exact L2 match — qkv merged, non-causal varlen via cu_seqlens, dispatch via ALL_ATTENTION_FUNCTIONS)
  - **`GlmImageVisionPatchEmbed`** [compute, inherits `Glm4vVisionPatchEmbed`]: `L1/conv2d.py` (Conv2d patch projection)
  - **`GlmImageVisionEmbeddings`** [compute, inherits `Glm4vVisionEmbeddings`]: `L1/embedding.py + L1/grid_sample.py` (interpolated 2D position embedding)
  - **`GlmImageVisionBlock`** [wiring, inherits `Glm4vVisionBlock`]: wires `nn.LayerNorm`(x2), `GlmImageVisionAttention`, `GlmImageVisionMLP`; direct `L1/layer_norm.py`
  - **`GlmImageTextAttention`** [compute, inherits `Glm4vMoeTextAttention`]: `L2/attention.py` (q_proj/k_proj/v_proj/o_proj + RoPE + KV cache + dispatch; causal)
  - **`GlmImageVQVAEVectorQuantizer`** [compute, inherits `ChameleonVQVAEVectorQuantizer`]: `L1/embedding.py` + custom L2-norm distance (no exact L2 match — VQ codebook lookup)
  - **`GlmImageVQVAE`** [wiring, inherits `ChameleonVQVAE`]: wires `GlmImageVQVAEVectorQuantizer`; direct `L1/conv2d.py` (quant_conv, post_quant_conv)
  - **`GlmImageVisionModel`** [wiring, inherits `Glm4vVisionModel`]: wires `GlmImageVisionEmbeddings`, `GlmImageVisionPatchEmbed`, `GlmImageVisionBlock` (xN)
  - **`GlmImageRMSNorm`** [compute]: `L1/rms_norm.py`
  - **`GlmImageTextRotaryEmbedding`** [compute]: `L1/mrope.py` (3D positions, mrope_section split)
  - **`GlmImageTextMLP`** [compute]: `L2/llama_mlp.py` (gate_up_proj merged, silu*up, down_proj — SwiGLU pattern)
  - **`GlmImageTextDecoderLayer`** [wiring]: wires `GlmImageTextAttention`, `GlmImageTextMLP`, `GlmImageRMSNorm` (x4 — input/post_attn/post_self_attn/post_mlp)
  - **`GlmImageTextModel`** [wiring, inherits `Glm4vTextModel`]: wires `GlmImageTextDecoderLayer` (xN), `GlmImageRMSNorm`, `GlmImageTextRotaryEmbedding`; direct `L1/embedding.py` (embed_tokens)
  - **`GlmImageModel`** [wiring, inherits `Glm4vModel`]: wires `GlmImageVisionModel`, `GlmImageTextModel`, `GlmImageVQVAE`
  - **`GlmImageForConditionalGeneration`** [wiring]: wires `GlmImageModel`; direct `L1/linear.py` (lm_head)

## glm_moe_dsa
- **src**: modeling_glm_moe_dsa.py (and modular_glm_moe_dsa.py)
- **hidden_act**: silu
- **status**: composable
- **classes**:
  - **`GlmMoeDsaRMSNorm`** [compute, inherits `Glm4MoeRMSNorm`]: `L1/rms_norm.py`
  - **`GlmMoeDsaIndexer`** [compute]: `L1/linear.py + L1/layer_norm.py + L1/rotary_emb.py + L1/sparse_attn_indexer.py` (no exact L2 match — DSA top-k token selection via wq_b/wk + RoPE + bf16 score; closest is `L2/sparse_attn_indexer.py` if present)
  - **`GlmMoeDsaAttention`** [compute]: `L2/deepseek_mla_attention.py` (MLA with q_lora + kv_a_proj_with_mqa + kv_b_proj + RoPE on pe split + indexer-driven top-k mask) — MLA pattern matches deepseek; DSA mask is an additional mask op on top
  - **`GlmMoeDsaMLP`** [compute]: `L2/llama_mlp.py` (gate_proj, up_proj, down_proj, silu — SwiGLU)
  - **`GlmMoeDsaTopkRouter`** [compute]: `L1/linear.py` (router_logits = F.linear(x, weight))
  - **`GlmMoeDsaNaiveMoe`** [compute]: `L1/moe_grouped_gemm.py` (per-expert gate_up_proj + silu*up + down_proj loop; kb-nano fused MoE replaces this naive loop)
  - **`GlmMoeDsaMoE`** [wiring, inherits `Glm4MoeMoE`]: wires `GlmMoeDsaNaiveMoe`, `GlmMoeDsaTopkRouter`, `GlmMoeDsaMLP` (shared experts) — overall: `L2/shared_expert_moe.py` (sigmoid gate + group topk + shared expert add)
  - **`GlmMoeDsaDecoderLayer`** [wiring, inherits `Glm4MoeLiteDecoderLayer`]: wires `GlmMoeDsaAttention`, `GlmMoeDsaMoE` (sparse) or `GlmMoeDsaMLP` (dense), `GlmMoeDsaRMSNorm` (x2)
  - **`GlmMoeDsaRotaryEmbedding`** [compute]: `L1/yarn_rotary_emb.py` (rope_type may be yarn for DeepSeek-style; falls back to default = `L1/rotary_emb.py`)
  - **`GlmMoeDsaModel`** [wiring, inherits `Glm4MoeModel`]: wires `GlmMoeDsaDecoderLayer` (xN), `GlmMoeDsaRMSNorm`, `GlmMoeDsaRotaryEmbedding`; direct `L1/embedding.py` (embed_tokens)
  - **`GlmMoeDsaForCausalLM`** [wiring, inherits `Glm4MoeForCausalLM`]: wires `GlmMoeDsaModel`; direct `L1/linear.py` (lm_head)

## glm_ocr
- **src**: modeling_glm_ocr.py (and modular_glm_ocr.py)
- **hidden_act**: silu (text), silu (vision)
- **status**: composable
- **classes**:
  - **`GlmOcrRMSNorm`** [compute, inherits `Glm4vRMSNorm`]: `L1/rms_norm.py`
  - **`GlmOcrVisionMlp`** [compute, inherits `Glm4VisionMlp`]: `L2/llama_mlp.py` (gate_proj, up_proj, down_proj, silu — SwiGLU; with bias)
  - **`GlmOcrTextAttention`** [compute, inherits `Glm4vTextAttention`]: `L2/attention.py` (q_proj/k_proj/v_proj/o_proj, GQA, RoPE interleaved, KV cache)
  - **`GlmOcrTextMLP`** [compute]: `L2/llama_mlp.py` (merged gate_up_proj, silu, down_proj — SwiGLU)
  - **`GlmOcrTextDecoderLayer`** [wiring, inherits `Glm4vTextDecoderLayer`]: wires `GlmOcrTextAttention`, `GlmOcrTextMLP`, `GlmOcrRMSNorm` (x4)
  - **`GlmOcrVisionRotaryEmbedding`** [compute]: `L1/vision_rotary_emb.py` (1D inv_freq for vision)
  - **`GlmOcrVisionAttention`** [compute, inherits `Glm4vVisionAttention`]: `L1/linear.py + L1/rms_norm.py + L1/rotary_emb.py + L1/dense_attention.py` (no exact L2 match — qkv merged, q_norm/k_norm RMSNorm, vision RoPE, varlen via cu_seqlens)
  - **`GlmOcrVisionBlock`** [wiring, inherits `Glm4vVisionBlock`]: wires `GlmOcrRMSNorm` (x2), `GlmOcrVisionAttention`, `GlmOcrVisionMlp`
  - **`GlmOcrVisionPatchMerger`** [compute, inherits `Glm4vVisionPatchMerger`]: `L1/linear.py + L1/layer_norm.py + L1/gelu.py + L1/silu.py` (proj + LN + GELU then SwiGLU-style gate*up + down)
  - **`GlmOcrVisionPatchEmbed`** [compute]: `L1/conv3d.py` (Conv3d patch projection — temporal+spatial)
  - **`GlmOcrVisionModel`** [wiring, inherits `Glm4vVisionModel`]: wires `GlmOcrVisionPatchEmbed`, `GlmOcrVisionRotaryEmbedding`, `GlmOcrVisionBlock` (xN), `GlmOcrVisionPatchMerger`, `GlmOcrRMSNorm`; direct `L1/conv2d.py` (downsample)
  - **`GlmOcrTextModel`** [wiring, inherits `Glm4vTextModel`]: wires `GlmOcrTextDecoderLayer` (xN), `GlmOcrRMSNorm`, `GlmOcrTextRotaryEmbedding`; direct `L1/embedding.py` (embed_tokens)
  - **`GlmOcrTextRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (default rope; partial_rotary_factor)
  - **`GlmOcrModel`** [wiring, inherits `Glm4vModel`]: wires `GlmOcrVisionModel`, `GlmOcrTextModel`
  - **`GlmOcrForConditionalGeneration`** [wiring, inherits `Glm4vForConditionalGeneration`]: wires `GlmOcrModel`; direct `L1/linear.py` (lm_head)

## glmasr
- **src**: modeling_glmasr.py (and modular_glmasr.py)
- **hidden_act**: gelu (encoder), gelu (projector)
- **status**: composable
- **classes**:
  - **`GlmAsrRotaryEmbedding`** [compute, inherits `GlmRotaryEmbedding`]: `L1/rotary_emb.py`
  - **`GlmAsrAttention`** [compute, inherits `LlamaAttention`]: `L2/attention.py` (q/k/v/o linears, GQA, RoPE, no causal mask, no KV cache — encoder-only attention)
  - **`GlmAsrMLP`** [compute]: `L2/encoder_mlp.py` (fc1 -> gelu -> fc2)
  - **`GlmAsrEncoderLayer`** [wiring]: wires `GlmAsrAttention`, `GlmAsrMLP`, `nn.LayerNorm` (x2); direct `L1/layer_norm.py`
  - **`GlmAsrEncoder`** [wiring, inherits `AudioFlamingo3PreTrainedModel`]: wires `GlmAsrEncoderLayer` (xN), `GlmAsrRotaryEmbedding`, `nn.LayerNorm`; direct `L1/conv1d.py` (x2 — conv1, conv2), `L1/gelu.py`
  - **`GlmAsrMultiModalProjector`** [compute, inherits `AudioFlamingo3MultiModalProjector`]: `L1/linear.py + L1/gelu.py + L1/linear.py` (linear_1 -> gelu -> linear_2)
  - **`GlmAsrForConditionalGeneration`** [wiring, inherits `AudioFlamingo3ForConditionalGeneration`]: wires `GlmAsrEncoder` (audio_tower), `GlmAsrMultiModalProjector`, language model (Glm via AutoModelForCausalLM)

## glpn
- **src**: modeling_glpn.py
- **hidden_act**: gelu
- **status**: composable
- **classes**:
  - **`GLPNDropPath`** [compute]: `L1/dropout.py` (stochastic depth — drop_path)
  - **`GLPNOverlapPatchEmbeddings`** [compute]: `L1/conv2d.py + L1/layer_norm.py` (Conv2d patch embed with overlap + LN)
  - **`GLPNEfficientSelfAttention`** [compute]: `L1/linear.py + L1/conv2d.py + L1/layer_norm.py + L1/dense_attention.py` (no exact L2 match — SegFormer-style with sequence reduction via Conv2d on KV)
  - **`GLPNSelfOutput`** [compute]: `L1/linear.py` (dense projection + dropout)
  - **`GLPNAttention`** [wiring]: wires `GLPNEfficientSelfAttention`, `GLPNSelfOutput`
  - **`GLPNDWConv`** [compute]: `L1/conv2d.py` (depthwise 3x3 conv with groups=dim)
  - **`GLPNMixFFN`** [compute]: `L1/linear.py + L1/conv2d.py + L1/gelu.py + L1/linear.py` (dense1 -> dwconv -> gelu -> dense2 — Segformer Mix-FFN)
  - **`GLPNLayer`** [wiring]: wires `nn.LayerNorm` (x2), `GLPNAttention`, `GLPNMixFFN`, optional `GLPNDropPath`; direct `L1/layer_norm.py`
  - **`GLPNEncoder`** [wiring]: wires `GLPNOverlapPatchEmbeddings` (xN), `GLPNLayer` (hierarchical), `nn.LayerNorm`
  - **`GLPNModel`** [wiring]: wires `GLPNEncoder`
  - **`GLPNSelectiveFeatureFusion`** [compute]: `L1/conv2d.py + L1/batch_norm2d.py + L1/relu.py + L1/sigmoid.py` (3 conv layers with BN/ReLU then sigmoid attention map and weighted fusion)
  - **`GLPNDecoderStage`** [wiring]: wires `GLPNSelectiveFeatureFusion`, `nn.Conv2d`, `nn.Upsample`; direct `L1/conv2d.py + L1/interpolate.py`
  - **`GLPNDecoder`** [wiring]: wires `GLPNDecoderStage` (xN), `nn.Upsample`
  - **`GLPNDepthEstimationHead`** [compute]: `L1/conv2d.py + L1/relu.py + L1/conv2d.py + L1/sigmoid.py` (head Conv-ReLU-Conv then sigmoid * max_depth)
  - **`GLPNForDepthEstimation`** [wiring]: wires `GLPNModel`, `GLPNDecoder`, `GLPNDepthEstimationHead`

## got_ocr2
- **src**: modeling_got_ocr2.py (and modular_got_ocr2.py)
- **hidden_act**: gelu (vision); text uses qwen2 (silu)
- **status**: composable
- **classes**:
  - **`GotOcr2MLPBlock`** [compute, inherits `SamMLPBlock`]: `L2/sam3_vit_mlp.py` (or `L2/encoder_mlp.py` — lin1 -> gelu -> lin2)
  - **`GotOcr2VisionAttention`** [compute, inherits `SamVisionAttention`]: `L2/sam3_vit_attention.py` (qkv merged, decomposed rel-pos h/w embeddings, manual softmax+matmul) — SAM-style window attention
  - **`GotOcr2VisionLayer`** [wiring, inherits `SamVisionLayer`]: wires `nn.LayerNorm` (x2), `GotOcr2VisionAttention`, `GotOcr2MLPBlock`; window partition/unpartition logic; direct `L1/layer_norm.py`
  - **`GotOcr2PatchEmbeddings`** [compute]: `L1/conv2d.py` (Conv2d patch projection)
  - **`GotOcr2LayerNorm`** [compute]: `L1/layer_norm.py` (channels_first/last LayerNorm)
  - **`GotOcr2VisionNeck`** [compute]: `L1/conv2d.py + L1/layer_norm.py + L1/conv2d.py + L1/layer_norm.py` (Conv-LN-Conv-LN neck)
  - **`GotOcr2VisionEncoder`** [wiring, inherits `SamVisionEncoder`]: wires `GotOcr2PatchEmbeddings`, `GotOcr2VisionLayer` (xN), `GotOcr2VisionNeck`; direct `L1/embedding.py` (pos_embed Parameter)
  - **`GotOcr2MultiModalProjector`** [compute]: `L1/conv2d.py + L1/conv2d.py + L1/linear.py` (conv_upsampler1 -> conv_upsampler2 -> linear)
  - **`GotOcr2Model`** [wiring, inherits `LlavaModel`]: wires `GotOcr2VisionEncoder` (vision_tower), `GotOcr2MultiModalProjector`, language model (Qwen2 via AutoModel)
  - **`GotOcr2ForConditionalGeneration`** [wiring, inherits `LlavaForConditionalGeneration`]: wires `GotOcr2Model`; direct `L1/linear.py` (lm_head)

## gpt2
- **src**: modeling_gpt2.py
- **hidden_act**: gelu_new (activation_function)
- **status**: composable
- **classes**:
  - **`GPT2Attention`** [compute]: `L1/linear.py + L1/dense_attention.py + L1/store_kvcache.py` (no exact L2 match — uses `Conv1D` (=Linear with weight transposed) for c_attn (qkv merged) + c_proj; supports cross-attention; KV cache via DynamicCache/EncoderDecoderCache)
  - **`GPT2MLP`** [compute]: `L2/encoder_mlp.py` (closest fit — c_fc -> gelu_new -> c_proj using Conv1D; same shape as fc1->act->fc2) — note Conv1D is just a Linear variant
  - **`GPT2Block`** [wiring]: wires `nn.LayerNorm` (x2 + optional cross), `GPT2Attention`, optional `GPT2Attention(is_cross_attention=True)`, `GPT2MLP`; direct `L1/layer_norm.py`
  - **`GPT2SequenceSummary`** [compute]: `L1/linear.py + L1/tanh.py` (sequence summary head — last/first/mean/cls_index pooling + optional Linear + activation)
  - **`GPT2Model`** [wiring]: wires `GPT2Block` (xN), `nn.Embedding` (wte, wpe), `nn.LayerNorm` (ln_f); direct `L1/embedding.py + L1/layer_norm.py`
  - **`GPT2LMHeadModel`** [wiring]: wires `GPT2Model`; direct `L1/linear.py` (lm_head, tied to wte)
  - **`GPT2DoubleHeadsModel`** [wiring]: wires `GPT2Model`, `GPT2SequenceSummary`; direct `L1/linear.py` (lm_head)
- **task heads (3)**: ForSequenceClassification, ForTokenClassification, ForQuestionAnswering — base + linear (per-task)

## gpt_bigcode
- **src**: modeling_gpt_bigcode.py
- **hidden_act**: gelu_pytorch_tanh (activation_function)
- **status**: composable
- **classes**:
  - **`GPTBigCodeAttention`** [compute]: `L1/linear.py + L1/dense_attention.py + L1/store_kvcache.py` (no exact L2 match — multi-query attention (kv_heads=1) or MHA, qkv merged via c_attn (Linear), supports cross-attention; closest is L2/attention.py without RoPE)
  - **`GPTBigCodeMLP`** [compute]: `L2/encoder_mlp.py` (c_fc -> gelu_pytorch_tanh -> c_proj — like Siglip but generic)
  - **`GPTBigCodeBlock`** [wiring]: wires `nn.LayerNorm` (x2), `GPTBigCodeAttention`, optional cross-attn `GPTBigCodeAttention(is_cross=True)`, `GPTBigCodeMLP`; direct `L1/layer_norm.py`
  - **`GPTBigCodeModel`** [wiring]: wires `GPTBigCodeBlock` (xN), `nn.Embedding` (wte, wpe), `nn.LayerNorm`; direct `L1/embedding.py + L1/layer_norm.py`
  - **`GPTBigCodeForCausalLM`** [wiring]: wires `GPTBigCodeModel`; direct `L1/linear.py` (lm_head)
- **task heads (2)**: ForSequenceClassification, ForTokenClassification — base + linear (per-task)

## gpt_neo
- **src**: modeling_gpt_neo.py
- **hidden_act**: gelu_new (activation_function)
- **status**: composable
- **classes**:
  - **`GPTNeoSelfAttention`** [compute]: `L1/linear.py + L1/dense_attention.py + L1/store_kvcache.py` (no exact L2 match — separate q_proj/k_proj/v_proj Linears (no merged c_attn), local or global mask via attention_type)
  - **`GPTNeoFlashAttention2`** [compute, inherits `GPTNeoSelfAttention`]: same as parent — flash impl variant
  - **`GPTNeoAttention`** [wiring]: wires `GPTNeoSelfAttention` (chooses local/global) — wrapper that selects attention type per layer
  - **`GPTNeoMLP`** [compute]: `L2/encoder_mlp.py` (c_fc -> gelu_new -> c_proj)
  - **`GPTNeoBlock`** [wiring]: wires `nn.LayerNorm` (x2), `GPTNeoAttention`, `GPTNeoMLP`; direct `L1/layer_norm.py`
  - **`GPTNeoModel`** [wiring]: wires `GPTNeoBlock` (xN), `nn.Embedding` (wte, wpe), `nn.LayerNorm`; direct `L1/embedding.py + L1/layer_norm.py`
  - **`GPTNeoForCausalLM`** [wiring]: wires `GPTNeoModel`; direct `L1/linear.py` (lm_head)
- **task heads (3)**: ForSequenceClassification, ForTokenClassification, ForQuestionAnswering — base + linear (per-task)

## gpt_neox
- **src**: modeling_gpt_neox.py
- **hidden_act**: gelu
- **status**: composable
- **classes**:
  - **`GPTNeoXMLP`** [compute]: `L2/encoder_mlp.py` (dense_h_to_4h -> gelu -> dense_4h_to_h)
  - **`GPTNeoXRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (default RoPE; partial_rotary_factor)
  - **`GPTNeoXAttention`** [compute]: `L2/attention.py` (qkv merged via query_key_value Linear, RoPE, KV cache, causal — close to LlamaAttention but with merged QKV instead of separate q/k/v_proj)
  - **`GPTNeoXLayer`** [wiring]: wires `nn.LayerNorm` (x2), `GPTNeoXAttention`, `GPTNeoXMLP`; supports parallel_residual; direct `L1/layer_norm.py`
  - **`GPTNeoXModel`** [wiring]: wires `GPTNeoXLayer` (xN), `GPTNeoXRotaryEmbedding`, `nn.Embedding`, `nn.LayerNorm`; direct `L1/embedding.py + L1/layer_norm.py`
  - **`GPTNeoXForCausalLM`** [wiring]: wires `GPTNeoXModel`; direct `L1/linear.py` (embed_out)
- **task heads (3)**: ForSequenceClassification, ForTokenClassification, ForQuestionAnswering — base + linear (per-task)

## gpt_neox_japanese
- **src**: modeling_gpt_neox_japanese.py
- **hidden_act**: gelu
- **status**: composable
- **classes**:
  - **`GPTNeoXJapaneseRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (default RoPE; partial_rotary_factor)
  - **`GPTNeoXJapaneseAttention`** [compute]: `L2/attention.py` (qkv merged via query_key_value Linear, partial RoPE, KV cache; per-layer use_bias on dense; uses baddbmm in eager) — close to GPTNeoX with optional bias parameter
  - **`GPTNeoXJapaneseMLP`** [compute]: `L2/encoder_mlp.py` (dense_h_to_4h -> gelu -> dense_4h_to_h)
  - **`GPTNeoXJapaneseLayer`** [wiring]: wires `nn.LayerNorm` (x2), `GPTNeoXJapaneseAttention`, `GPTNeoXJapaneseMLP`; bias_dropout_add residual; direct `L1/layer_norm.py + L1/dropout.py`
  - **`GPTNeoXJapaneseModel`** [wiring]: wires `GPTNeoXJapaneseLayer` (xN), `GPTNeoXJapaneseRotaryEmbedding`, `nn.Embedding`, `nn.LayerNorm`; direct `L1/embedding.py + L1/layer_norm.py`
  - **`GPTNeoXJapaneseForCausalLM`** [wiring]: wires `GPTNeoXJapaneseModel`; direct `L1/linear.py` (embed_out)

## gpt_oss
- **src**: modeling_gpt_oss.py (and modular_gpt_oss.py)
- **hidden_act**: silu
- **status**: kb_nano_l4 (`L4/gpt_oss.py`)
- **classes**:
  - **`GptOssRMSNorm`** [compute, inherits `LlamaRMSNorm`]: `L1/rms_norm.py`
  - **`GptOssExperts`** [compute]: `L1/mxfp4_moe.py` (MXFP4-quantized experts in checkpoint; bf16 path is `L1/moe_grouped_gemm.py` — gate_up clamp + sigmoid GLU + bias)
  - **`GptOssTopKRouter`** [compute]: `L1/linear.py + L1/topk_softmax.py` (router_logits + topk + softmax over top values)
  - **`GptOssMLP`** [wiring]: wires `GptOssTopKRouter`, `GptOssExperts` — `L2/gpt_oss_moe.py`
  - **`GptOssRotaryEmbedding`** [compute]: `L1/yarn_rotary_emb.py` (yarn scaling supported via rope_type) or `L1/rotary_emb.py` (default)
  - **`GptOssAttention`** [compute, inherits `Qwen2Attention`]: `L2/gpt_oss_attention.py` or `L2/attention.py` (q/k/v/o linears with bias, sliding window, attention sinks (per-head learnable bias))
  - **`GptOssDecoderLayer`** [wiring, inherits `LlamaDecoderLayer`]: wires `GptOssAttention`, `GptOssMLP`, `GptOssRMSNorm` (x2)
  - **`GptOssModel`** [wiring, inherits `MixtralModel`]: wires `GptOssDecoderLayer` (xN), `GptOssRMSNorm`, `GptOssRotaryEmbedding`; direct `L1/embedding.py`
  - **`GptOssForCausalLM`** [wiring, inherits `MixtralForCausalLM`]: wires `GptOssModel`; direct `L1/linear.py` (lm_head)
- **task heads (2)**: ForSequenceClassification, ForTokenClassification — base + linear (per-task)

## gptj
- **src**: modeling_gptj.py
- **hidden_act**: gelu_new (activation_function)
- **status**: composable
- **classes**:
  - **`GPTJAttention`** [compute]: `L1/linear.py + L1/sinusoidal_embed.py + L1/rotary_emb.py + L1/dense_attention.py + L1/store_kvcache.py` (no exact L2 match — separate q/k/v_proj/out_proj (no bias), partial rotary on rotary_dim, sinusoidal position cache, fp32 attention compute)
  - **`GPTJFlashAttention2`** [compute, inherits `GPTJAttention`]: same as parent — flash variant
  - **`GPTJMLP`** [compute]: `L2/encoder_mlp.py` (fc_in -> gelu_new -> fc_out)
  - **`GPTJBlock`** [wiring]: wires `nn.LayerNorm`, `GPTJAttention`, `GPTJMLP` (parallel residual: attn+mlp+residual all summed)
  - **`GPTJModel`** [wiring]: wires `GPTJBlock` (xN), `nn.Embedding` (wte), `nn.LayerNorm` (ln_f); direct `L1/embedding.py + L1/layer_norm.py`
  - **`GPTJForCausalLM`** [wiring]: wires `GPTJModel`; direct `L1/linear.py` (lm_head)
- **task heads (2)**: ForSequenceClassification, ForQuestionAnswering — base + linear (per-task)

## granite
- **src**: modeling_granite.py (and modular_granite.py)
- **hidden_act**: silu
- **status**: composable
- **classes**:
  - **`GraniteAttention`** [compute, inherits `LlamaAttention`]: `L2/attention.py` (q/k/v/o linears, GQA, RoPE, KV cache; uses attention_multiplier as scaling instead of head_dim^-0.5)
  - **`GraniteRMSNorm`** [compute]: `L1/rms_norm.py`
  - **`GraniteMLP`** [compute]: `L2/llama_mlp.py` (gate_proj, up_proj, down_proj, silu — SwiGLU)
  - **`GraniteDecoderLayer`** [wiring, inherits `LlamaDecoderLayer`]: wires `GraniteAttention`, `GraniteMLP`, `GraniteRMSNorm` (x2); residual_multiplier scaling on residual
  - **`GraniteRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (default RoPE)
  - **`GraniteModel`** [wiring, inherits `LlamaModel`]: wires `GraniteDecoderLayer` (xN), `GraniteRMSNorm`, `GraniteRotaryEmbedding`; direct `L1/embedding.py` (embed_tokens, scaled by embedding_multiplier)
  - **`GraniteForCausalLM`** [wiring, inherits `LlamaForCausalLM`]: wires `GraniteModel`; direct `L1/linear.py` (lm_head, divided by logits_scaling)

## granite_speech
- **src**: modeling_granite_speech.py
- **hidden_act**: silu (encoder uses SiLU directly)
- **status**: composable
- **classes**:
  - **`GraniteSpeechEncoderProjector`** [wiring]: wires Q-Former (BLIP-2 via AutoModel), `nn.Linear` (linear); direct `L1/linear.py + L1/embedding.py` (query Parameter)
  - **`GraniteSpeechConformerFeedForward`** [compute]: `L1/layer_norm.py + L1/linear.py + L1/silu.py + L1/linear.py` (pre_norm -> up_proj -> silu -> down_proj — Conformer FFN)
  - **`GraniteSpeechConformerAttention`** [compute]: `L1/layer_norm.py + L1/linear.py + L1/embedding.py + L1/dense_attention.py` (no exact L2 match — Shaw relative pos embed via einsum, blocked context-size attention via SDPA)
  - **`GraniteSpeechConformerDepthWiseConv1d`** [compute]: `L1/conv1d.py` (depthwise 1D conv with manual padding)
  - **`GraniteSpeechConformerConvModule`** [compute]: `L1/layer_norm.py + L1/conv1d.py + L1/conv1d.py + L1/silu.py + L1/batch_norm2d.py + L1/conv1d.py` (Conformer conv: norm -> up_conv -> GLU -> depth_conv -> SiLU+BN -> down_conv) — note nn.GLU is not in kb-nano L1 list, falls back to chunk + sigmoid + mul
  - **`GraniteSpeechConformerBlock`** [wiring]: wires `GraniteSpeechConformerFeedForward` (x2 — ff1 + ff2), `GraniteSpeechConformerAttention`, `GraniteSpeechConformerConvModule`, `nn.LayerNorm` (post_norm); 0.5 scale on ff residuals
  - **`GraniteSpeechCTCEncoder`** [wiring]: wires `GraniteSpeechConformerBlock` (xN); direct `L1/linear.py` (input_linear, out, out_mid), `L1/softmax.py` (mid output)
  - **`GraniteSpeechForConditionalGeneration`** [wiring]: wires `GraniteSpeechCTCEncoder`, `GraniteSpeechEncoderProjector`, language model (Granite via AutoModelForCausalLM)

## granitemoe
- **src**: modeling_granitemoe.py
- **hidden_act**: silu
- **status**: composable
- **classes**:
  - **`GraniteMoeRMSNorm`** [compute]: `L1/rms_norm.py`
  - **`GraniteMoeRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (default RoPE)
  - **`GraniteMoeParallelExperts`** [compute]: `L1/moe_grouped_gemm.py` (per-expert F.linear loop on contiguous batches; closest fused kernel is grouped GEMM)
  - **`GraniteMoeTopKGating`** [compute]: `L1/linear.py + L1/topk_softmax.py` (linear router -> topk -> softmax over top-k -> sort by expert id)
  - **`GraniteMoeMoE`** [wiring]: wires `GraniteMoeParallelExperts` (x2 — input_linear, output_linear), `GraniteMoeTopKGating`; SwiGLU activation between (silu*chunk2); closest L2: `L2/mixtral_moe.py` or `L2/llama4_moe.py`
  - **`GraniteMoeAttention`** [compute]: `L2/attention.py` (q/k/v/o linears, GQA, RoPE, KV cache; attention_multiplier scaling)
  - **`GraniteMoeDecoderLayer`** [wiring]: wires `GraniteMoeAttention`, `GraniteMoeMoE` (block_sparse_moe), `GraniteMoeRMSNorm` (x2); residual_multiplier
  - **`GraniteMoeModel`** [wiring]: wires `GraniteMoeDecoderLayer` (xN), `GraniteMoeRMSNorm`, `GraniteMoeRotaryEmbedding`; direct `L1/embedding.py` (embed_tokens with embedding_multiplier)
  - **`GraniteMoeForCausalLM`** [wiring]: wires `GraniteMoeModel`; direct `L1/linear.py` (lm_head, divided by logits_scaling)

## granitemoehybrid
- **src**: modeling_granitemoehybrid.py
- **hidden_act**: silu
- **status**: composable
- **classes**:
  - **`GraniteMoeHybridAttention`** [compute]: `L2/attention.py` (q/k/v/o linears, GQA, RoPE, KV cache; attention_multiplier scaling) — same as GraniteMoeAttention
  - **`GraniteMoeHybridMambaLayer`** [compute]: `L2/mamba2_mixer.py` (Mamba2 SSM with conv1d + selective scan + RMS-norm-gated output projection)
  - **`GraniteMoeHybridRMSNormGated`** [compute]: `L1/rms_norm_gated.py` (RMS norm with optional silu(gate) multiply)
  - **`GraniteMoeHybridMLP`** [compute]: `L2/llama_mlp.py` (input_linear (merged gate*2) -> silu+chunk*chunk -> output_linear — SwiGLU shared expert)
  - **`GraniteMoeHybridRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (default RoPE)
  - **`GraniteMoeHybridParallelExperts`** [compute]: `L1/moe_grouped_gemm.py` (per-expert F.linear loop)
  - **`GraniteMoeHybridTopKGating`** [compute]: `L1/linear.py + L1/topk_softmax.py` (linear router -> topk -> softmax over top-k)
  - **`GraniteMoeHybridMoE`** [wiring]: wires `GraniteMoeHybridParallelExperts` (x2), `GraniteMoeHybridTopKGating`; SwiGLU activation between
  - **`GraniteMoeHybridRMSNorm`** [compute]: `L1/rms_norm.py`
  - **`GraniteMoeHybridDecoderLayer`** [wiring]: wires `GraniteMoeHybridAttention` OR `GraniteMoeHybridMambaLayer` (per layer_type), `GraniteMoeHybridMoE` (block_sparse_moe, optional), `GraniteMoeHybridMLP` (shared_mlp), `GraniteMoeHybridRMSNorm` (x2); residual_multiplier
  - **`GraniteMoeHybridModel`** [wiring]: wires `GraniteMoeHybridDecoderLayer` (xN, mixed attn/mamba), `GraniteMoeHybridRMSNorm`, `GraniteMoeHybridRotaryEmbedding`; direct `L1/embedding.py`
  - **`GraniteMoeHybridForCausalLM`** [wiring]: wires `GraniteMoeHybridModel`; direct `L1/linear.py` (lm_head)

## granitemoeshared
- **src**: modeling_granitemoeshared.py
- **hidden_act**: silu
- **status**: composable
- **classes**:
  - **`GraniteMoeSharedMLP`** [compute]: `L2/llama_mlp.py` (input_linear (merged gate*2) -> silu*chunk -> output_linear — SwiGLU shared expert)
  - **`GraniteMoeSharedRMSNorm`** [compute]: `L1/rms_norm.py`
  - **`GraniteMoeSharedParallelExperts`** [compute]: `L1/moe_grouped_gemm.py`
  - **`GraniteMoeSharedTopKGating`** [compute]: `L1/linear.py + L1/topk_softmax.py`
  - **`GraniteMoeSharedMoE`** [wiring]: wires `GraniteMoeSharedParallelExperts` (x2), `GraniteMoeSharedTopKGating`; SwiGLU activation — closest L2: `L2/shared_expert_moe.py` when combined with shared MLP at decoder level
  - **`GraniteMoeSharedAttention`** [compute]: `L2/attention.py` (q/k/v/o linears, GQA, RoPE, KV cache; attention_multiplier scaling)
  - **`GraniteMoeSharedDecoderLayer`** [wiring]: wires `GraniteMoeSharedAttention`, `GraniteMoeSharedMoE` (block_sparse_moe), `GraniteMoeSharedMLP` (shared_mlp), `GraniteMoeSharedRMSNorm` (x2); residual_multiplier
  - **`GraniteMoeSharedRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (default RoPE)
  - **`GraniteMoeSharedModel`** [wiring]: wires `GraniteMoeSharedDecoderLayer` (xN), `GraniteMoeSharedRMSNorm`, `GraniteMoeSharedRotaryEmbedding`; direct `L1/embedding.py`
  - **`GraniteMoeSharedForCausalLM`** [wiring]: wires `GraniteMoeSharedModel`; direct `L1/linear.py` (lm_head, divided by logits_scaling)

## grounding_dino
- **src**: modeling_grounding_dino.py
- **hidden_act**: relu (activation_function)
- **status**: composable
- **classes**:
  - **`MultiScaleDeformableAttention`** [compute]: `L1/rtdetrv2_deformable_attention.py` (multi-scale deformable attention compute kernel)
  - **`GroundingDinoFrozenBatchNorm2d`** [compute]: `L1/frozen_batch_norm2d.py`
  - **`GroundingDinoConvEncoder`** [wiring]: wires backbone (timm/swin via AutoBackbone), `GroundingDinoFrozenBatchNorm2d`
  - **`GroundingDinoConvModel`** [wiring]: wires `GroundingDinoConvEncoder`, position_embedding
  - **`GroundingDinoSinePositionEmbedding`** [compute]: `L1/sinusoidal_embed.py` (2D sine pos embed)
  - **`GroundingDinoLearnedPositionEmbedding`** [compute]: `L1/embedding.py` (2 nn.Embeddings for row/col)
  - **`GroundingDinoMultiscaleDeformableAttention`** [compute]: `L2/rtdetrv2_deformable_attention.py` (sampling_offsets + attention_weights via Linear -> kernel call -> output_proj)
  - **`GroundingDinoTextEnhancerLayer`** [wiring]: wires `GroundingDinoMultiheadAttention`, `nn.LayerNorm` (x2), `nn.Linear` (x2 — fc1, fc2); direct `L1/relu.py` (FFN with relu)
  - **`GroundingDinoBiMultiHeadAttention`** [compute]: `L1/linear.py + L1/dense_attention.py` (no exact L2 match — bidirectional cross-attention between vision and text features; vision_proj/text_proj K/V/Q linears + manual attention compute)
  - **`GroundingDinoDropPath`** [compute]: `L1/dropout.py` (stochastic depth)
  - **`GroundingDinoFusionLayer`** [wiring]: wires `GroundingDinoBiMultiHeadAttention`, `nn.LayerNorm` (x2), optional `GroundingDinoDropPath`; learnable scale parameters for residuals
  - **`GroundingDinoDeformableLayer`** [wiring]: wires `GroundingDinoMultiscaleDeformableAttention`, `nn.LayerNorm` (x2), `nn.Linear` (x2 — fc1, fc2); direct `L1/relu.py` (FFN)
  - **`GroundingDinoEncoderLayer`** [wiring]: wires `GroundingDinoFusionLayer`, `GroundingDinoDeformableLayer` (vision), `GroundingDinoTextEnhancerLayer` (text)
  - **`GroundingDinoMultiheadAttention`** [compute]: `L1/linear.py + L1/dense_attention.py + L1/store_kvcache.py` (no exact L2 match — q/k/v/o linears + standard MHA, used for cross-attention; closest L2: encoder_attention)
  - **`GroundingDinoDecoderLayer`** [wiring]: wires `GroundingDinoMultiheadAttention` (self-attn), `GroundingDinoMultiscaleDeformableAttention` (cross-attn), `nn.LayerNorm` (x4), `nn.Linear` (x2 — fc1, fc2); direct `L1/relu.py`
  - **`GroundingDinoContrastiveEmbedding`** [compute]: `L1/linear.py` (contrastive logit projection)
  - **`GroundingDinoEncoder`** [wiring]: wires `GroundingDinoEncoderLayer` (xN)
  - **`GroundingDinoDecoder`** [wiring]: wires `GroundingDinoDecoderLayer` (xN), `nn.LayerNorm`
  - **`GroundingDinoModel`** [wiring]: wires `GroundingDinoConvModel`, `GroundingDinoEncoder`, `GroundingDinoDecoder`, text backbone (BERT via AutoModel); direct `L1/linear.py` (input_proj, level_embed)
  - **`GroundingDinoMLPPredictionHead`** [compute]: `L1/linear.py + L1/relu.py` (3-layer MLP for box prediction)
  - **`GroundingDinoForObjectDetection`** [wiring]: wires `GroundingDinoModel`, `GroundingDinoMLPPredictionHead` (x2 — bbox_embed, class_embed), `GroundingDinoContrastiveEmbedding`

## groupvit
- **src**: modeling_groupvit.py
- **hidden_act**: quick_gelu (vision), gelu (text)
- **status**: composable
- **classes**:
  - **`GroupViTCrossAttentionLayer`** [wiring]: wires `GroupViTAttention`, `nn.LayerNorm` (x2), `GroupViTMLP`
  - **`GroupViTAssignAttention`** [compute]: `L1/linear.py + L1/softmax.py` (no exact L2 match — soft assignment with gumbel/hard softmax over key dim)
  - **`GroupViTTokenAssign`** [wiring]: wires `nn.LayerNorm` (x4), `GroupViTMixerMLP`, `GroupViTCrossAttentionLayer`, `GroupViTAssignAttention`, `GroupViTMLP`
  - **`GroupViTPatchEmbeddings`** [compute]: `L1/conv2d.py` (Conv2d patch projection)
  - **`GroupViTVisionEmbeddings`** [compute]: `L1/conv2d.py + L1/embedding.py + L1/layer_norm.py` (patch_embed + position_embeddings parameter + LN)
  - **`GroupViTTextEmbeddings`** [compute]: `L1/embedding.py + L1/embedding.py` (token + position embeddings)
  - **`GroupViTStage`** [wiring]: wires `GroupViTEncoderLayer` (x N_depth), optional `GroupViTTokenAssign`, `nn.LayerNorm`
  - **`GroupViTMLP`** [compute]: `L2/clip_mlp.py` (vision: fc1 -> quickgelu -> fc2) or `L2/encoder_mlp.py` (text: fc1 -> gelu -> fc2)
  - **`GroupViTMixerMLP`** [compute, inherits `GroupViTMLP`]: same as parent but transposed input
  - **`GroupViTAttention`** [compute]: `L2/clip_attention.py` (q/k/v/o linears, dispatch via ALL_ATTENTION_FUNCTIONS, supports cross-attention via encoder_hidden_states)
  - **`GroupViTEncoderLayer`** [wiring]: wires `nn.LayerNorm` (x2), `GroupViTAttention`, `GroupViTMLP`
  - **`GroupViTVisionEncoder`** [wiring]: wires `GroupViTStage` (xN, hierarchical with grouping)
  - **`GroupViTTextEncoder`** [wiring]: wires `GroupViTEncoderLayer` (xN)
  - **`GroupViTTextTransformer`** [wiring]: wires `GroupViTTextEmbeddings`, `GroupViTTextEncoder`, `nn.LayerNorm`
  - **`GroupViTTextModel`** [wiring]: wires `GroupViTTextTransformer`
  - **`GroupViTVisionTransformer`** [wiring]: wires `GroupViTVisionEmbeddings`, `GroupViTVisionEncoder`, `nn.LayerNorm`
  - **`GroupViTVisionModel`** [wiring]: wires `GroupViTVisionTransformer`
  - **`GroupViTModel`** [wiring]: wires `GroupViTTextTransformer`, `GroupViTVisionTransformer`; direct `L1/linear.py` (text_projection, visual_projection), CLIP-style logit_scale parameter

## helium
- **src**: modeling_helium.py (and modular_helium.py)
- **hidden_act**: silu
- **status**: composable
- **classes**:
  - **`HeliumRMSNorm`** [compute]: `L1/rms_norm.py` (note: same as Llama RMSNorm)
  - **`HeliumRotaryEmbedding`** [compute, inherits `LlamaRotaryEmbedding`]: `L1/rotary_emb.py`
  - **`HeliumMLP`** [compute, inherits `LlamaMLP`]: `L2/llama_mlp.py` (gate_proj, up_proj, down_proj, silu — SwiGLU)
  - **`HeliumAttention`** [compute, inherits `GraniteAttention`]: `L2/attention.py` (q/k/v/o linears, GQA, RoPE, KV cache)
  - **`HeliumDecoderLayer`** [wiring, inherits `LlamaDecoderLayer`]: wires `HeliumAttention`, `HeliumMLP`, `HeliumRMSNorm` (x2)
  - **`HeliumModel`** [wiring, inherits `LlamaModel`]: wires `HeliumDecoderLayer` (xN), `HeliumRMSNorm`, `HeliumRotaryEmbedding`; direct `L1/embedding.py`
  - **`HeliumForCausalLM`** [wiring, inherits `GemmaForCausalLM`]: wires `HeliumModel`; direct `L1/linear.py` (lm_head)
- **task heads (2)**: ForSequenceClassification, ForTokenClassification — base + linear (per-task)

## hgnet_v2
- **src**: modeling_hgnet_v2.py (and modular_hgnet_v2.py)
- **hidden_act**: relu
- **status**: composable
- **classes**:
  - **`HGNetV2LearnableAffineBlock`** [compute]: scale * x + bias (just an affine; closest L1: `L1/tensor_ops.py` or inline as scale Parameter mul + bias add)
  - **`HGNetV2ConvLayer`** [compute, inherits `RTDetrResNetConvLayer`]: `L1/conv2d.py + L1/batch_norm2d.py + L1/relu.py` + optional `HGNetV2LearnableAffineBlock`
  - **`HGNetV2ConvLayerLight`** [wiring]: wires `HGNetV2ConvLayer` (x2 — pointwise + depthwise)
  - **`HGNetV2Embeddings`** [wiring]: wires `HGNetV2ConvLayer` (x5 — stem stages); direct `L1/max_pool2d.py`
  - **`HGNetV2BasicLayer`** [wiring]: wires `HGNetV2ConvLayer` or `HGNetV2ConvLayerLight` (xN), aggregation `nn.Sequential` of `HGNetV2ConvLayer` (x2); residual + drop_path
  - **`HGNetV2Stage`** [wiring]: wires optional `HGNetV2ConvLayer` (downsample), `HGNetV2BasicLayer` (xN_blocks)
  - **`HGNetV2Encoder`** [wiring]: wires `HGNetV2Stage` (xN_stages)
  - **`HGNetV2Backbone`** [wiring, inherits `BackboneMixin`]: wires `HGNetV2Embeddings`, `HGNetV2Encoder`
- **task heads (1)**: ForImageClassification — base + linear (per-task)

## hiera
- **src**: modeling_hiera.py
- **hidden_act**: gelu
- **status**: composable
- **classes**:
  - **`HieraPatchEmbeddings`** [compute]: `L1/conv2d.py` (Conv2d patch projection with optional masked conv for MAE)
  - **`HieraEmbeddings`** [wiring]: wires `HieraPatchEmbeddings`; direct `L1/embedding.py` (position_embeddings Parameter), `L1/interpolate.py` (bicubic interp on pos embed)
  - **`HieraMaskUnitAttention`** [compute]: `L1/linear.py + L1/dense_attention.py` (no exact L2 match — qkv merged Linear; supports query_stride via maxpool reduction; mask-unit (windowed) or global attention)
  - **`HieraDropPath`** [compute]: `L1/dropout.py` (stochastic depth)
  - **`HieraMlp`** [compute]: `L2/encoder_mlp.py` (fc1 -> gelu -> fc2)
  - **`HieraLayer`** [wiring]: wires `nn.LayerNorm` (x2), `HieraMaskUnitAttention`, `HieraMlp`, optional `HieraDropPath`, optional `nn.Linear` (proj when hidden_size changes)
  - **`HieraStage`** [wiring]: wires `HieraLayer` (xN_depth)
  - **`HieraEncoder`** [wiring]: wires `HieraStage` (xN_stages)
  - **`HieraPooler`** [compute]: `L1/linear.py + L1/layer_norm.py` (head pooling — mean + LN + linear)
  - **`HieraModel`** [wiring]: wires `HieraEmbeddings`, `HieraEncoder`, optional `HieraPooler`
  - **`HieraDecoder`** [wiring]: wires `HieraLayer` (xN), `nn.LayerNorm`; direct `L1/linear.py` (decoder embed and pred)
  - **`HieraMultiScaleHead`** [compute]: `L1/conv2d.py + L1/linear.py` (multi-scale fusion convs + final linear)
  - **`HieraForPreTraining`** [wiring]: wires `HieraModel`, `HieraDecoder`, `HieraMultiScaleHead`
  - **`HieraBackbone`** [wiring, inherits `BackboneMixin`]: wires `HieraEmbeddings`, `HieraEncoder`
- **task heads (1)**: ForImageClassification — base + linear (per-task)

## hubert
- **src**: modeling_hubert.py (and modular_hubert.py)
- **hidden_act**: gelu
- **status**: composable
- **classes**:
  - **`HubertPositionalConvEmbedding`** [compute]: `L1/conv1d.py + L1/gelu.py` (Conv1d (groups=num_conv_pos_embedding_groups) + GELU + same-pad)
  - **`HubertSamePadLayer`** [compute, inherits `Wav2Vec2SamePadLayer`]: simple slicing for odd kernel — no kernel needed
  - **`HubertNoLayerNormConvLayer`** [compute]: `L1/conv1d.py + L1/gelu.py`
  - **`HubertLayerNormConvLayer`** [compute]: `L1/conv1d.py + L1/layer_norm.py + L1/gelu.py`
  - **`HubertGroupNormConvLayer`** [compute]: `L1/conv1d.py + L1/group_norm.py + L1/gelu.py`
  - **`HubertFeatureEncoder`** [wiring, inherits `Wav2Vec2FeatureEncoder`]: wires `HubertGroupNormConvLayer` or `HubertLayerNormConvLayer` or `HubertNoLayerNormConvLayer` (xN — feature_extract_norm)
  - **`HubertFeatureProjection`** [compute]: `L1/layer_norm.py + L1/linear.py + L1/dropout.py` (LN + Linear + dropout)
  - **`HubertAttention`** [compute]: `L2/whisper_attention.py` (q/k/v/o linears, supports cross-attention via key_value_states; closest L2 is whisper which has same enc/dec/cross variants; closest fit for self-attn-only is `L2/encoder_attention.py`)
  - **`HubertFeedForward`** [compute]: `L2/encoder_mlp.py` (intermediate_dense -> gelu -> output_dense + dropouts)
  - **`HubertEncoderLayer`** [wiring]: wires `HubertAttention`, `nn.LayerNorm` (x2), `HubertFeedForward`
  - **`HubertEncoder`** [wiring, inherits `Wav2Vec2Encoder`]: wires `HubertPositionalConvEmbedding`, `nn.LayerNorm`, `HubertEncoderLayer` (xN)
  - **`HubertAttnAdapterLayer`** [compute]: `L1/linear.py + L1/relu.py + L1/layer_norm.py` (residual adapter)
  - **`HubertEncoderLayerStableLayerNorm`** [wiring]: wires `HubertAttention`, `nn.LayerNorm` (x2), `HubertFeedForward`, optional `HubertAttnAdapterLayer`
  - **`HubertEncoderStableLayerNorm`** [wiring, inherits `Wav2Vec2EncoderStableLayerNorm`]: wires `HubertPositionalConvEmbedding`, `HubertEncoderLayerStableLayerNorm` (xN), `nn.LayerNorm`
  - **`HubertModel`** [wiring, inherits `Wav2Vec2Model`]: wires `HubertFeatureEncoder`, `HubertFeatureProjection`, `HubertEncoder` or `HubertEncoderStableLayerNorm`
- **task heads (3)**: ForCTC, ForSequenceClassification — base + linear (per-task)

## hunyuan_v1_dense
- **src**: modeling_hunyuan_v1_dense.py (and modular_hunyuan_v1_dense.py)
- **hidden_act**: silu
- **status**: composable
- **classes**:
  - **`HunYuanDenseV1RMSNorm`** [compute, inherits `LlamaRMSNorm`]: `L1/rms_norm.py`
  - **`HunYuanDenseV1MLP`** [compute, inherits `LlamaMLP`]: `L2/llama_mlp.py` (gate_proj, up_proj, down_proj, silu — SwiGLU)
  - **`HunYuanDenseV1Attention`** [compute, inherits `LlamaAttention`]: `L2/attention.py` (q/k/v/o linears, GQA, RoPE, KV cache; QK-norm via RMSNorm)
  - **`HunYuanDenseV1DecoderLayer`** [wiring, inherits `LlamaDecoderLayer`]: wires `HunYuanDenseV1Attention`, `HunYuanDenseV1MLP`, `HunYuanDenseV1RMSNorm` (x2)
  - **`HunYuanDenseV1RotaryEmbedding`** [compute, inherits `LlamaRotaryEmbedding`]: `L1/rotary_emb.py`
  - **`HunYuanDenseV1Model`** [wiring, inherits `LlamaModel`]: wires `HunYuanDenseV1DecoderLayer` (xN), `HunYuanDenseV1RMSNorm`, `HunYuanDenseV1RotaryEmbedding`; direct `L1/embedding.py`
  - **`HunYuanDenseV1ForCausalLM`** [wiring, inherits `LlamaForCausalLM`]: wires `HunYuanDenseV1Model`; direct `L1/linear.py` (lm_head)
- **task heads (1)**: ForSequenceClassification — base + linear (per-task)

## hunyuan_v1_moe
- **src**: modeling_hunyuan_v1_moe.py
- **hidden_act**: silu
- **status**: composable
- **classes**:
  - **`HunYuanMoEV1RMSNorm`** [compute]: `L1/rms_norm.py`
  - **`HunYuanMoEV1MLP`** [compute]: `L2/llama_mlp.py` (gate_proj, up_proj, down_proj, silu — SwiGLU)
  - **`HunYuanMoEV1Attention`** [compute]: `L2/attention.py` (q/k/v/o linears, GQA, RoPE, KV cache; QK-norm via RMSNorm)
  - **`HunYuanMoEV1Gate`** [compute]: `L1/linear.py + L1/topk_softmax.py` (top-k routing)
  - **`HunYuanMoEV1Experts`** [compute]: `L1/moe_grouped_gemm.py` (per-expert gate*up + silu activation; closest fused MoE kernel)
  - **`HunYuanMoEV1Moe`** [wiring]: wires `HunYuanMoEV1Gate`, `HunYuanMoEV1Experts`, optional shared `HunYuanMoEV1MLP` — closest L2: `L2/shared_expert_moe.py`
  - **`HunYuanMoEV1DecoderLayer`** [wiring]: wires `HunYuanMoEV1Attention`, `HunYuanMoEV1Moe`, `HunYuanMoEV1RMSNorm` (x2)
  - **`HunYuanMoEV1RotaryEmbedding`** [compute]: `L1/rotary_emb.py` (default RoPE; supports scaling types)
  - **`HunYuanMoEV1Model`** [wiring]: wires `HunYuanMoEV1DecoderLayer` (xN), `HunYuanMoEV1RMSNorm`, `HunYuanMoEV1RotaryEmbedding`; direct `L1/embedding.py`
  - **`HunYuanMoEV1ForCausalLM`** [wiring]: wires `HunYuanMoEV1Model`; direct `L1/linear.py` (lm_head)
- **task heads (1)**: ForSequenceClassification — base + linear (per-task)

## hy_v3
- **src**: modeling_hy_v3.py (and modular_hy_v3.py)
- **hidden_act**: silu
- **status**: composable
- **classes**:
  - **`HYV3RMSNorm`** [compute, inherits `LlamaRMSNorm`]: `L1/rms_norm.py`
  - **`HYV3RotaryEmbedding`** [compute, inherits `LlamaRotaryEmbedding`]: `L1/rotary_emb.py`
  - **`HYV3MLP`** [compute, inherits `LlamaMLP`]: `L2/llama_mlp.py` (gate_proj, up_proj, down_proj, silu — SwiGLU)
  - **`HYV3Attention`** [compute, inherits `ApertusAttention`]: `L2/attention.py` (q/k/v/o linears, GQA, RoPE, KV cache)
  - **`HYV3TopKRouter`** [compute, inherits `MixtralTopKRouter`]: `L1/linear.py + L1/topk_softmax.py`
  - **`HYV3Experts`** [compute, inherits `Qwen3MoeExperts`]: `L1/moe_grouped_gemm.py`
  - **`HYV3MoE`** [wiring, inherits `MiniMaxM2SparseMoeBlock`]: wires `HYV3TopKRouter`, `HYV3Experts`; closest L2: `L2/qwen3_moe.py` or `L2/mixtral_moe.py`
  - **`HYV3DecoderLayer`** [wiring, inherits `DeepseekV3DecoderLayer`]: wires `HYV3Attention`, `HYV3MoE` or `HYV3MLP` (per layer_type), `HYV3RMSNorm` (x2)
  - **`HYV3Model`** [wiring, inherits `MiniMaxM2Model`]: wires `HYV3DecoderLayer` (xN), `HYV3RMSNorm`, `HYV3RotaryEmbedding`; direct `L1/embedding.py`
  - **`HYV3ForCausalLM`** [wiring, inherits `LlamaForCausalLM`]: wires `HYV3Model`; direct `L1/linear.py` (lm_head)

## ibert
- **src**: modeling_ibert.py
- **hidden_act**: gelu
- **status**: composable
- **classes**:
  - **`IBertEmbeddings`** [compute]: `L2/encoder_embeddings.py` (word + position + token_type + LayerNorm + Dropout — BERT-style)
  - **`IBertSelfAttention`** [compute]: `L2/encoder_attention.py` (BERT-style q/k/v + dispatch; quantization params unused at fp inference)
  - **`IBertSelfOutput`** [compute]: `L2/encoder_attention.py` (dense + LayerNorm + residual — BERT-style)
  - **`IBertAttention`** [wiring]: wires `IBertSelfAttention`, `IBertSelfOutput` — `L2/encoder_attention.py` (full wrapper)
  - **`IBertIntermediate`** [compute]: `L1/linear.py + L1/gelu.py` (just half of an encoder MLP)
  - **`IBertOutput`** [compute]: `L1/linear.py + L1/layer_norm.py` (dense + LN + residual)
  - **`IBertLayer`** [wiring]: wires `IBertAttention`, `IBertIntermediate`, `IBertOutput`
  - **`IBertEncoder`** [wiring]: wires `IBertLayer` (xN)
  - **`IBertPooler`** [compute]: `L1/linear.py + L1/tanh.py` (BERT-style first-token pooler)
  - **`IBertModel`** [wiring]: wires `IBertEmbeddings`, `IBertEncoder`, optional `IBertPooler`
  - **`IBertLMHead`** [compute]: `L1/linear.py + L1/gelu.py + L1/layer_norm.py + L1/linear.py` (BERT MLM head)
  - **`IBertClassificationHead`** [compute]: `L1/linear.py + L1/tanh.py + L1/linear.py` (sentence classification head)
  - **`IBertForMaskedLM`** [wiring]: wires `IBertModel`, `IBertLMHead`
- **task heads (4)**: ForSequenceClassification, ForMultipleChoice, ForTokenClassification, ForQuestionAnswering — base + linear (per-task)

