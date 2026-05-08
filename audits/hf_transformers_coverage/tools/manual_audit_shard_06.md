## glm46v
- **src**: modular_glm46v.py
- **status**: partial
- **rationale**: Pure subclass of Glm4v: vision blocks + Glm4v text decoder; only the processor differs. All compute classes inherit Glm4v* with no overrides.
- **classes**:
  - **`Glm46VModel`** [compute]: no kb-nano kernel — Pure subclass of Glm4v: vision blocks + Glm4v text decoder; only the processor differs. All compute classes inherit Glm4v* with no overrides.
  - **`Glm46VForConditionalGeneration`** [wiring]: wiring; identical to Glm4v.

## glm4_moe
- **src**: modular_glm4_moe.py
- **status**: partial
- **rationale**: GLM-4 MoE = Llama-style attention (Cohere bias config) + DeepSeekV3 MoE (top-k softmax router with shared expert) + RMSNorm + Llama RoPE. Maps cleanly to L2/attention.py + L2/shared_expert_moe.py + L1/rms_norm + L1/rotary_emb.
- **classes**:
  - **`Glm4MoeDecoderLayer`** [compute]: no kb-nano kernel — GLM-4 MoE = Llama-style attention (Cohere bias config) + DeepSeekV3 MoE (top-k softmax router with shared expert) + RMSNorm + Llama RoPE. Maps cleanly to L2/attention.py + L2/shared_expert_moe.py + L1
  - **`Glm4MoeRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (Standard NeoX RoPE (Glm RoPE follows Llama).)
  - **`Glm4MoeAttention`** [compute]: `L2/attention.py` (GQA + bias + RoPE + o_proj; LlamaAttention covers this with bias=True.)
  - **`Glm4MoeMLP`** [compute]: `L2/llama_mlp.py` (Shared/dense MLP = SwiGLU gate_up + SiluAndMul + down (matches LlamaMLP).)
  - **`Glm4MoeTopkRouter`** [compute]: `L2/shared_expert_moe.py` (DeepseekV3-style top-k group router with bias correction; embedded inside shared_expert_moe.)
  - **`Glm4MoeRMSNorm`** [compute]: `L1/rms_norm.py` (Standard Llama RMSNorm.)
  - **`Glm4MoeModel`** [wiring]: wiring; embed + layers + norm.
  - **`Glm4MoeForCausalLM`** [wiring]: wiring; lm_head.

## glm4_moe_lite
- **src**: modular_glm4_moe_lite.py
- **status**: composable
- **rationale**: Smaller variant of glm4_moe; reuses Glm4Moe* classes plus DeepseekV3-naive MoE layer. All compute paths covered by L2/attention + L2/shared_expert_moe + L1 norms.
- **classes**:
  - **`Glm4MoeLiteAttention`** [compute]: `L2/deepseek_mla_attention.py` (Glm4-Lite is actually MLA: q_a_proj+q_b_proj LoRA, kv_a_proj_with_mqa+kv_b_proj split, qk_rope_head_dim+qk_nope_head_dim, qk_head_dim sum. Maps to L2/deepseek_mla_attention.py, not LlamaAttention; rope_interleave=True default is supported by the MLA kernel via is_neox_style flag.)
  - **`Glm4MoeLiteMLP`** [compute]: `L2/llama_mlp.py` (Shared SwiGLU MLP.)
  - **`Glm4MoeLiteTopkRouter`** [compute]: `L2/shared_expert_moe.py` (Top-k group router.)
  - **`Glm4MoeLiteRMSNorm`** [compute]: `L1/rms_norm.py` (Standard RMSNorm.)
  - **`Glm4MoeLiteNaiveMoe`** [compute]: `L2/shared_expert_moe.py` (Top-k routed experts; SharedExpertMoE handles this.)
  - **`Glm4MoeLiteMoE`** [compute]: `L2/shared_expert_moe.py` (Routed + shared expert composite.)
  - **`Glm4MoeLiteDecoderLayer`** [wiring]: wiring.
  - **`Glm4MoeLiteModel`** [wiring]: wiring.
  - **`Glm4MoeLiteForCausalLM`** [wiring]: wiring.

## glm4v
- **src**: modular_glm4v.py
- **status**: partial
- **rationale**: GLM-4V = Glm4 (Llama-style) text decoder with M-RoPE for multimodal positions + Qwen2.5-VL-style vision encoder (Conv3d patch embed + 2D RoPE + SwiGLU vision MLP). All ops map to existing kb-nano kernels (mrope, vision_rotary_emb, llama_mlp, attention, conv3d, rms_norm).
- **classes**:
  - **`Glm4vVisionBlock`** [compute]: no kb-nano kernel — GLM-4V = Glm4 (Llama-style) text decoder with M-RoPE for multimodal positions + Qwen2.5-VL-style vision encoder (Conv3d patch embed + 2D RoPE + SwiGLU vision MLP). All ops map to existing kb-nano kern
  - **`Glm4vRMSNorm`** [compute]: `L1/rms_norm.py` (Standard Llama RMSNorm.)
  - **`Glm4VisionMlp`** [compute]: `L2/llama_mlp.py` (Vision SwiGLU MLP (gate/up/down with SiLU activation), matches LlamaMLP.)
  - **`Glm4vVisionPatchEmbed`** [compute]: `L1/conv3d.py`, `L2/vision_patch_embed.py` (Conv3d projection over (T,H,W).)
  - **`Glm4vVisionRotaryEmbedding`** [compute]: `L1/vision_rotary_emb.py` (2D vision RoPE.)
  - **`Glm4vVisionPatchMerger`** [compute]: `L2/vision_patch_merger.py` (proj + LN + GELU + SwiGLU down (gate/up/down).)
  - **`Glm4vVisionEmbeddings`** [compute]: `L1/embedding.py`, `L1/grid_sample.py` (Position embedding + bicubic grid_sample for adaptive resolution.)
  - **`Glm4vVisionAttention`** [compute]: `L2/vision_attention.py` (QKV-fused vision attention with cu_seqlens varlen support.)
  - **`Glm4vTextRotaryEmbedding`** [compute]: `L1/mrope.py` (M-RoPE for 3D (T,H,W) positions.)
  - **`Glm4vTextAttention`** [compute]: `L2/attention.py` (GQA + bias on Q/K/V + interleaved RoPE.)
  - **`Glm4vTextMLP`** [compute]: `L2/llama_mlp.py` (SwiGLU gate_up/down.)
  - **`Glm4vTextDecoderLayer`** [wiring]: wiring; 4 RMSNorms (input/post-attn/post-self-attn/post-mlp) + attn + mlp.
  - **`Glm4vVisionModel`** [wiring]: wiring; vision tower.
  - **`Glm4vTextModel`** [wiring]: wiring; text tower.
  - **`Glm4vModel`** [wiring]: wiring; combines vision + text.

## glm4v_moe
- **src**: modular_glm4v_moe.py
- **status**: partial
- **partial_reason**: configuration_glm4v_moe.py defaults `partial_rotary_factor = 0.5`; modeling_glm4v_moe.py:apply_rotary_pos_emb slices q[..., :rotary_dim] / q[..., rotary_dim:] before rotating. kb-nano L1/rotary_emb.py rotates the full head; partial-rotary needs external slicing. Same gap as glm4/glm4_moe.
- **rationale**: GLM-4V MoE = same GLM-4V vision tower (mrope + 2D vision RoPE) + text side combining GLM-4 attention with GLM-4 MoE (DeepseekV3-style top-k + shared expert). Vision path composes; MoE composes via shared_expert_moe; attention path needs partial-rotary slicing. (Note: text RoPE here is NeoX-style with rotate_half + partial-rotary, not interleaved as the earlier audit incorrectly stated.)
- **classes**:
  - **`Glm4vMoeTextAttention`** [compute]: `L2/attention.py` (GQA + bias + interleaved RoPE; LlamaAttention with bias.)
  - **`Glm4vMoeTextTopkRouter`** [compute]: `L2/shared_expert_moe.py` (DeepseekV3-style top-k softmax with bias correction.)
  - **`Glm4vMoeTextNaiveMoe`** [compute]: `L2/shared_expert_moe.py` (Routed experts.)
  - **`Glm4vMoeTextMoE`** [compute]: `L2/shared_expert_moe.py` (Routed + shared composite.)
  - **`Glm4vMoeTextMLP`** [compute]: `L2/llama_mlp.py` (Shared SwiGLU MLP.)
  - **`Glm4vMoeTextDecoderLayer`** [wiring]: wiring.
  - **`Glm4vMoeVisionRotaryEmbedding`** [compute]: `L1/vision_rotary_emb.py` (2D vision RoPE.)
  - **`Glm4vMoeVisionModel`** [wiring]: wiring; vision tower.
  - **`Glm4vMoeTextModel`** [wiring]: wiring; text tower.
  - **`Glm4vMoeForConditionalGeneration`** [wiring]: wiring; combined model + lm_head.

## glm_image
- **src**: modular_glm_image.py
- **status**: partial
- **rationale**: GLM-Image adds a Chameleon-style VQVAE for image generation. The VQVAE has bespoke ResNet/Conv2d blocks and a vector quantizer with EMA codebook updates — kb-nano has no Chameleon VQVAE kernels and no L4 pipeline for it.
- **classes**:
  - **`GlmImageVisionBlock`** [compute]: no kb-nano kernel — GLM-Image adds a Chameleon-style VQVAE for image generation. The VQVAE has bespoke ResNet/Conv2d blocks and a vector quantizer with EMA codebook updates — kb-nano has no Chameleon VQVAE kernels and no
  - **`GlmImageVisionMLP`** [compute]: `L2/siglip_mlp.py` (fc1 + GELU + fc2.)
  - **`GlmImageVisionAttention`** [compute]: `L2/vision_attention.py` (QKV-fused vision attention.)
  - **`GlmImageVisionPatchEmbed`** [compute]: `L1/conv3d.py`, `L2/vision_patch_embed.py` (Conv3d patch embed.)
  - **`GlmImageVisionEmbeddings`** [compute]: `L1/embedding.py`, `L1/grid_sample.py` (2D pos embed + grid_sample.)
  - **`GlmImageTextAttention`** [compute]: `L2/attention.py` (GQA + bias + interleaved RoPE.)
  - **`GlmImageVQVAEVectorQuantizer`** [wiring]: Vector quantization codebook with EMA — no kb-nano kernel.
  - **`GlmImageVQVAE`** [wiring]: VQVAE encoder/decoder using Chameleon ResNet blocks — no kb-nano equivalent.
  - **`GlmImageVisionModel`** [wiring]: wiring.
  - **`GlmImageTextModel`** [wiring]: wiring.
  - **`GlmImageModel`** [wiring]: wiring.
  - **`GlmImageForConditionalGeneration`** [wiring]: wiring with VQVAE for image gen.

## glm_moe_dsa
- **src**: modular_glm_moe_dsa.py
- **status**: composable
- **rationale**: GLM MoE DSA = MLA attention + DeepSeek Sparse Attention indexer + GLM-4 MoE (top-k + shared expert). Maps to L2/deepseek_mla_attention + L2/sparse_attn_indexer + L2/shared_expert_moe.
- **classes**:
  - **`GlmMoeDsaRMSNorm`** [compute]: `L1/rms_norm.py` (Standard Llama RMSNorm.)
  - **`GlmMoeDsaIndexer`** [compute]: `L2/sparse_attn_indexer.py` (DSA indexer: wq_b, wk, k_norm, weights_proj, top-k token scoring.)
  - **`GlmMoeDsaAttention`** [compute]: `L2/deepseek_mla_attention.py` (MLA: q_a/q_b, kv_a/kv_b LoRA + interleaved RoPE on PE-half + DSA indexer mask.)
  - **`GlmMoeDsaDecoderLayer`** [wiring]: wiring; norm -> MLA+DSA -> norm -> MoE.
  - **`GlmMoeDsaModel`** [wiring]: wiring; embed + layers + norm.
  - **`GlmMoeDsaForCausalLM`** [wiring]: wiring; lm_head.

## glm_ocr
- **src**: modular_glm_ocr.py
- **status**: partial
- **rationale**: GLM-OCR = pure subclass of Glm4v vision + text components, inheritance only (no new compute). Same as Glm4v structurally.
- **classes**:
  - **`GlmOcrTextDecoderLayer`** [compute]: no kb-nano kernel — GLM-OCR = pure subclass of Glm4v vision + text components, inheritance only (no new compute). Same as Glm4v structurally.
  - **`GlmOcrRMSNorm`** [compute]: `L1/rms_norm.py` (Standard RMSNorm.)
  - **`GlmOcrVisionMlp`** [compute]: `L2/llama_mlp.py` (SwiGLU vision MLP.)
  - **`GlmOcrTextAttention`** [compute]: `L2/attention.py` (GQA + bias + interleaved RoPE.)
  - **`GlmOcrVisionAttention`** [compute]: `L2/vision_attention.py` (QKV-fused vision attention with varlen.)
  - **`GlmOcrVisionBlock`** [wiring]: wiring.
  - **`GlmOcrVisionPatchMerger`** [compute]: `L2/vision_patch_merger.py` (proj + LN + GELU + SwiGLU.)
  - **`GlmOcrVisionModel`** [wiring]: wiring.
  - **`GlmOcrTextModel`** [wiring]: wiring.
  - **`GlmOcrModel`** [wiring]: wiring.
  - **`GlmOcrForConditionalGeneration`** [wiring]: wiring; lm_head.

## glmasr
- **src**: modular_glmasr.py
- **status**: partial
- **rationale**: GLM-ASR adds an audio Conformer encoder (AudioFlamingo3 style) with depthwise Conv1d + GLU + BatchNorm1d + Shaw relative positional embeddings. kb-nano has no Conformer/audio-encoder kernels (no audio_flamingo or whisper-non-attention modules covering this).
- **classes**:
  - **`GlmAsrEncoderLayer`** [compute]: no kb-nano kernel — GLM-ASR adds an audio Conformer encoder (AudioFlamingo3 style) with depthwise Conv1d + GLU + BatchNorm1d + Shaw relative positional embeddings. kb-nano has no Conformer/audio-encoder kernels (no audio
  - **`GlmAsrRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (Standard NeoX RoPE.)
  - **`GlmAsrAttention`** [compute]: `L2/attention.py` (Llama-style GQA attention for the text decoder of the projected audio.)
  - **`GlmAsrMLP`** [compute]: `L2/llama_mlp.py` (SwiGLU MLP.)
  - **`GlmAsrEncoder`** [wiring]: Conformer audio encoder with depthwise conv1d + GLU + Shaw relpos — no kb-nano kernel.
  - **`GlmAsrMultiModalProjector`** [wiring]: Q-Former style audio→text projector; no kb-nano equivalent.
  - **`GlmAsrForConditionalGeneration`** [wiring]: wiring; combined audio encoder + text decoder.

## glpn
- **src**: modeling_glpn.py
- **status**: composable
- **rationale**: GLPN = SegFormer-style vision encoder for monocular depth estimation. Uses Conv2d patch embed, efficient self-attention (with conv-based sequence reduction), MixFFN with depthwise conv, GELU. All ops exist as kb-nano L1 (conv2d, linear, layer_norm, softmax, gelu).
- **classes**:
  - **`GLPNDropPath`** [compute]: `L1/dropout.py` (Stochastic depth via dropout + scale; trivial.)
  - **`GLPNOverlapPatchEmbeddings`** [compute]: `L1/conv2d.py`, `L1/layer_norm.py` (Conv2d + LN.)
  - **`GLPNEfficientSelfAttention`** [compute]: `L1/linear.py`, `L1/conv2d.py`, `L1/softmax.py`, `L1/layer_norm.py` (Q/K/V projections; sequence reduction via Conv2d + LN; manual scaled dot-product.)
  - **`GLPNSelfOutput`** [compute]: `L1/linear.py` (Output projection.)
  - **`GLPNAttention`** [wiring]: wiring; SelfAttention + SelfOutput.
  - **`GLPNDWConv`** [compute]: `L1/conv2d.py` (Depthwise Conv2d (groups=dim).)
  - **`GLPNMixFFN`** [compute]: `L1/linear.py`, `L1/gelu.py` (fc1 + DWConv + GELU + fc2.)
  - **`GLPNLayer`** [wiring]: wiring; norm + attn + drop_path + norm + MixFFN.
  - **`GLPNEncoder`** [wiring]: wiring; multi-stage encoder.
  - **`GLPNModel`** [wiring]: wiring.
  - **`GLPNSelectiveFeatureFusion`** [compute]: `L1/conv2d.py`, `L1/sigmoid.py`, `L1/relu.py` (Conv2d + sigmoid + ReLU fusion gate.)
  - **`GLPNDecoderStage`** [compute]: `L1/conv2d.py`, `L1/interpolate.py` (Conv + upsample.)
  - **`GLPNDecoder`** [wiring]: wiring.
  - **`GLPNDepthEstimationHead`** [compute]: `L1/conv2d.py`, `L1/sigmoid.py` (Conv2d + ReLU + Conv2d + sigmoid.)
  - **`GLPNForDepthEstimation`** [wiring]: wiring.

## got_ocr2
- **src**: modular_got_ocr2.py
- **status**: partial
- **rationale**: GOT-OCR-2 vision tower is the SAM ViT encoder, which uses decomposed relative positional embeddings (MViT-v2 / Shaw style) with custom einsum scoring. kb-nano has SAM3 vision attention (uses 2D RoPE, not decomposed relpos), so the SAM-style attention math is not covered by an existing kernel.
- **classes**:
  - **`GotOcr2VisionAttention`** [compute]: no kb-nano kernel — GOT-OCR-2 vision tower is the SAM ViT encoder, which uses decomposed relative positional embeddings (MViT-v2 / Shaw style) with custom einsum scoring. kb-nano has SAM3 vision attention (uses 2D RoPE, 
  - **`GotOcr2MLPBlock`** [compute]: `L2/sam3_vit_mlp.py` (fc1 + GELU + fc2 — close to SAM3 vit MLP.)
  - **`GotOcr2VisionLayer`** [wiring]: wiring; LN + attn + LN + MLP with windowed partition.
  - **`GotOcr2VisionEncoder`** [wiring]: wiring; vision tower.
  - **`GotOcr2MultiModalProjector`** [compute]: `L1/conv2d.py`, `L1/linear.py` (Conv2d upsamplers + linear projector.)
  - **`GotOcr2Model`** [wiring]: wiring; vision_tower + multi_modal_projector + Qwen text decoder.
  - **`GotOcr2ForConditionalGeneration`** [wiring]: wiring; lm_head + image embedding fusion.

## gpt2
- **src**: modeling_gpt2.py
- **status**: composable
- **rationale**: Classic GPT-2: learned positional embeddings, multi-head attention with bias on c_attn (Conv1D-style packed QKV), GELU MLP, LayerNorm. All standard PyTorch ops; encoder-attention.py covers the bias=True attention pattern with absolute positions.
- **classes**:
  - **`GPT2Attention`** [compute]: `L2/encoder_attention.py` (Causal multi-head attention with packed QKV (Conv1D), bias, optional cross-attention. EncoderSelfAttention covers the QKV-projection + softmax pattern (with causal mask).)
  - **`GPT2MLP`** [compute]: `L2/encoder_mlp.py` (fc1 (Conv1D) + GELU + fc2 (Conv1D) — same as encoder MLP.)
  - **`GPT2Block`** [wiring]: wiring; LN + attn + LN + MLP.
  - **`GPT2Model`** [wiring]: wiring; wte + wpe + blocks + LN.
  - **`GPT2LMHeadModel`** [wiring]: wiring.
  - **`GPT2DoubleHeadsModel`** [wiring]: wiring.
  - **`GPT2ForSequenceClassification`** [wiring]: wiring.
  - **`GPT2ForTokenClassification`** [wiring]: wiring.
  - **`GPT2ForQuestionAnswering`** [wiring]: wiring.

## gpt_bigcode
- **src**: modeling_gpt_bigcode.py
- **status**: composable
- **rationale**: GPT-BigCode = GPT-2 with multi-query attention (single K/V head shared across heads). Same compute primitives as GPT-2 (Conv1D-style projections + GELU MLP + LN), just with KV broadcast.
- **classes**:
  - **`GPTBigCodeAttention`** [compute]: `L2/encoder_attention.py` (MQA causal attention; encoder_attention with kv_heads=1 broadcast.)
  - **`GPTBigCodeMLP`** [compute]: `L2/encoder_mlp.py` (fc1 + GELU + fc2.)
  - **`GPTBigCodeBlock`** [wiring]: wiring.
  - **`GPTBigCodeModel`** [wiring]: wiring.
  - **`GPTBigCodeForCausalLM`** [wiring]: wiring.
  - **`GPTBigCodeForSequenceClassification`** [wiring]: wiring.
  - **`GPTBigCodeForTokenClassification`** [wiring]: wiring.

## gpt_neo
- **src**: modeling_gpt_neo.py
- **status**: composable
- **rationale**: GPT-Neo = GPT-2-like with alternating local/global attention windows. Uses standard linear projections + softmax + GELU MLP + LN. encoder_attention covers it.
- **classes**:
  - **`GPTNeoSelfAttention`** [compute]: `L2/encoder_attention.py` (Multi-head attention with sliding window mask; standard linear projections.)
  - **`GPTNeoFlashAttention2`** [compute]: `L2/encoder_attention.py` (FA2 path; same compute, different backend.)
  - **`GPTNeoAttention`** [wiring]: wiring; chooses local vs global SelfAttention.
  - **`GPTNeoMLP`** [compute]: `L2/encoder_mlp.py` (fc1 + GELU + fc2.)
  - **`GPTNeoBlock`** [wiring]: wiring.
  - **`GPTNeoModel`** [wiring]: wiring.
  - **`GPTNeoForCausalLM`** [wiring]: wiring.
  - **`GPTNeoForSequenceClassification`** [wiring]: wiring.
  - **`GPTNeoForTokenClassification`** [wiring]: wiring.
  - **`GPTNeoForQuestionAnswering`** [wiring]: wiring.

## gpt_neox
- **src**: modular_gpt_neox.py
- **status**: partial
- **rationale**: GPT-NeoX = LlamaModel base with NeoX-style RoPE on first part of head_dim, separate q/k/v projections fused into qkv linear, GELU MLP. Maps cleanly to L2/attention + L2/llama_mlp + L1/rotary_emb.
- **classes**:
  - **`GPTNeoXLayer`** [compute]: no kb-nano kernel — GPT-NeoX = LlamaModel base with NeoX-style RoPE on first part of head_dim, separate q/k/v projections fused into qkv linear, GELU MLP. Maps cleanly to L2/attention + L2/llama_mlp + L1/rotary_emb.
  - **`GPTNeoXMLP`** [compute]: `L2/encoder_mlp.py` (dense_h_to_4h + GELU + dense_4h_to_h — two-layer MLP, NOT SwiGLU.)
  - **`GPTNeoXRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (Standard NeoX RoPE (partial-dim variant).)
  - **`GPTNeoXAttention`** [compute]: `L2/attention.py` (Fused QKV linear + NeoX RoPE on first rotary_ndims of head_dim + dense output. LlamaAttention with bias=True.)
  - **`GPTNeoXModel`** [wiring]: wiring.
  - **`GPTNeoXForCausalLM`** [wiring]: wiring.
  - **`GPTNeoXForSequenceClassification`** [wiring]: wiring.
  - **`GPTNeoXForTokenClassification`** [wiring]: wiring.
  - **`GPTNeoXForQuestionAnswering`** [wiring]: wiring.

## gpt_neox_japanese
- **src**: modeling_gpt_neox_japanese.py
- **status**: composable
- **rationale**: GPT-NeoX Japanese = GPT-NeoX with bias-only output projection and slightly different layout; same compute primitives (NeoX RoPE on partial dim, GELU MLP, LN). Encoder-style attention with bias.
- **classes**:
  - **`GPTNeoXJapaneseRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (NeoX RoPE (partial-dim).)
  - **`GPTNeoXJapaneseAttention`** [compute]: `L2/attention.py` (Fused QKV + RoPE + dense output. LlamaAttention pattern.)
  - **`GPTNeoXJapaneseMLP`** [compute]: `L2/encoder_mlp.py` (dense_h_to_4h + GELU + dense_4h_to_h.)
  - **`GPTNeoXJapaneseLayer`** [wiring]: wiring.
  - **`GPTNeoXJapaneseModel`** [wiring]: wiring.
  - **`GPTNeoXJapaneseForCausalLM`** [wiring]: wiring.

## gpt_oss
- **src**: modular_gpt_oss.py
- **status**: kb_nano_l4
- **rationale**: Existing standalone L4 pipeline at L4/gpt_oss.py targets openai/gpt-oss-20b: 24 layers alternating sliding/full attention, MXFP4 MoE, YaRN RoPE, attention sinks.
- **classes**:
  - **`GptOssRMSNorm`** [compute]: `L1/rms_norm.py` (Standard Llama RMSNorm.)
  - **`GptOssExperts`** [compute]: `L1/mxfp4_moe.py`, `L2/gpt_oss_moe.py` (MXFP4-quantized expert weights with OAI SwiGLU (clamped gate, sigmoid·gate gate*alpha, (up+1)*glu).)
  - **`GptOssTopKRouter`** [compute]: `L2/gpt_oss_moe.py` (Linear router + top-k softmax.)
  - **`GptOssMLP`** [compute]: `L2/gpt_oss_moe.py` (wiring; router + experts.)
  - **`GptOssRotaryEmbedding`** [compute]: `L1/yarn_rotary_emb.py` (YaRN-NeoX RoPE.)
  - **`GptOssAttention`** [compute]: `L2/gpt_oss_attention.py`, `L2/attention.py` (GQA + bias + sliding window (even layers) + attention sinks.)
  - **`GptOssDecoderLayer`** [wiring]: wiring.
  - **`GptOssModel`** [wiring]: wiring.
  - **`GptOssForCausalLM`** [wiring]: wiring.
  - **`GptOssForSequenceClassification`** [wiring]: wiring.
  - **`GptOssForTokenClassification`** [wiring]: wiring.

## gptj
- **src**: modeling_gptj.py
- **status**: partial
- **rationale**: GPT-J = parallel residual: attention and MLP run in parallel from same input, summed with residual. Uses GPT-J-style RoPE (rotary on first rotary_dim of head_dim, NeoX layout) without bias on q/k/v. All ops standard.
- **classes**:
  - **`GPTJBlock`** [compute]: no kb-nano kernel — GPT-J = parallel residual: attention and MLP run in parallel from same input, summed with residual. Uses GPT-J-style RoPE (rotary on first rotary_dim of head_dim, NeoX layout) without bias on q/k/v. A
  - **`GPTJAttention`** [compute]: `L2/attention.py` (MHA with separate q/k/v projections (no bias) + GPT-J RoPE on first rotary_dim. LlamaAttention with bias=False.)
  - **`GPTJFlashAttention2`** [compute]: `L2/attention.py` (FA2 path; same compute.)
  - **`GPTJMLP`** [compute]: `L2/encoder_mlp.py` (fc_in + GELU + fc_out — two-layer with GELU.)
  - **`GPTJModel`** [wiring]: wiring.
  - **`GPTJForCausalLM`** [wiring]: wiring.
  - **`GPTJForSequenceClassification`** [wiring]: wiring.
  - **`GPTJForQuestionAnswering`** [wiring]: wiring.

## granite
- **src**: modular_granite.py
- **status**: composable
- **rationale**: Granite = LlamaModel with attention_multiplier (replaces 1/sqrt(d_k) scaling) and residual_multiplier on residual adds. Identical kernel layout otherwise: SwiGLU MLP, RMSNorm, NeoX RoPE.
- **classes**:
  - **`GraniteAttention`** [compute]: `L2/attention.py` (GQA + RoPE + custom scaling factor (attention_multiplier).)
  - **`GraniteDecoderLayer`** [wiring]: wiring; norm + attn + scaled residual + norm + mlp + scaled residual.
  - **`GraniteModel`** [wiring]: wiring.
  - **`GraniteForCausalLM`** [wiring]: wiring.

## granite_speech
- **src**: modeling_granite_speech.py
- **status**: partial
- **rationale**: Granite Speech encoder is a Conformer with depthwise Conv1d + GLU + BatchNorm1d + Shaw relative positional embeddings (einsum-based pos_attn). kb-nano has no Conformer block and no Shaw relpos primitive.
- **classes**:
  - **`GraniteSpeechConformerAttention`** [compute]: no kb-nano kernel — Granite Speech encoder is a Conformer with depthwise Conv1d + GLU + BatchNorm1d + Shaw relative positional embeddings (einsum-based pos_attn). kb-nano has no Conformer block and no Shaw relpos primiti
  - **`GraniteSpeechEncoderProjector`** [wiring]: Q-Former-style projector with cross-attention; no kb-nano kernel.
  - **`GraniteSpeechConformerFeedForward`** [compute]: `L1/linear.py`, `L1/silu.py`, `L1/layer_norm.py` (LN + linear + SiLU + linear with dropout — primitives exist but composite is novel.)
  - **`GraniteSpeechConformerDepthWiseConv1d`** [compute]: `L1/conv1d.py` (Depthwise Conv1d (groups=chan_in).)
  - **`GraniteSpeechConformerConvModule`** [wiring]: Conv1d + GLU + DWConv1d + SiLU + BatchNorm1d + Conv1d. GLU activation has no kb-nano L1 op; BatchNorm1d also missing.
  - **`GraniteSpeechConformerBlock`** [wiring]: wiring; FF1 + Attn + Conv + FF2 + LN.
  - **`GraniteSpeechCTCEncoder`** [wiring]: Stack of Conformer blocks; no kb-nano equivalent.
  - **`GraniteSpeechForConditionalGeneration`** [wiring]: wiring; encoder + projector + Granite text decoder.

## granitemoe
- **src**: modular_granitemoe.py
- **status**: composable
- **rationale**: GraniteMoE = Granite (Llama + multipliers) attention + JetMoe-style ParallelExperts MoE (top-k softmax, scatter routing, fused gate/up/down). Maps to L2/attention + L2/mixtral_moe (both use the w13/w2 grouped-experts pattern) + L1/rms_norm.
- **classes**:
  - **`GraniteMoeRMSNorm`** [compute]: `L1/rms_norm.py` (Standard RMSNorm.)
  - **`GraniteMoeRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (NeoX RoPE.)
  - **`GraniteMoeParallelExperts`** [compute]: `L1/moe_grouped_gemm.py` (Per-expert linear; matches the grouped-GEMM kernel's per-expert weight slabs.)
  - **`GraniteMoeTopKGating`** [compute]: `L1/topk_softmax.py`, `L1/moe_align.py` (Top-k softmax router with scatter-based grouping.)
  - **`GraniteMoeMoE`** [compute]: `L2/mixtral_moe.py`, `L2/fused_experts.py` (input_linear (gate/up packed) + activation*gate + output_linear with index_add gather. Same pattern as Mixtral MoE.)
  - **`GraniteMoeAttention`** [compute]: `L2/attention.py` (Llama-style attention with attention_multiplier scaling.)
  - **`GraniteMoeDecoderLayer`** [wiring]: wiring; norm + attn + scaled residual + norm + MoE + scaled residual.
  - **`GraniteMoeModel`** [wiring]: wiring.
  - **`GraniteMoeForCausalLM`** [wiring]: wiring.

## granitemoehybrid
- **src**: modular_granitemoehybrid.py
- **status**: composable
- **rationale**: GraniteMoE Hybrid alternates attention layers and Bamba-style Mamba2 layers with shared MoE + dense MLP. All ops covered: L2/attention + L2/mamba2_mixer (Bamba is Mamba2) + L2/shared_expert_moe / L2/mixtral_moe + L1/rms_norm_gated + L1/rms_norm.
- **classes**:
  - **`GraniteMoeHybridAttention`** [compute]: `L2/attention.py` (GQA + optional RoPE (NoPE for some layers) + dense output.)
  - **`GraniteMoeHybridMambaLayer`** [compute]: `L2/mamba2_mixer.py` (Bamba mixer = Mamba2 SSD (in_proj, conv1d, A_log, D, dt_bias, norm, out_proj).)
  - **`GraniteMoeHybridRMSNormGated`** [compute]: `L1/rms_norm_gated.py` (Gated RMSNorm used in Mamba2 mixer.)
  - **`GraniteMoeHybridMLP`** [compute]: `L2/llama_mlp.py` (input_linear (gate/up packed) + SwiGLU + output_linear; equivalent to LlamaMLP.)
  - **`GraniteMoeHybridRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (Standard NeoX RoPE.)
  - **`GraniteMoeHybridMoE`** [compute]: `L2/mixtral_moe.py` (JetMoe-style top-k routed experts with optional shared expert.)
  - **`GraniteMoeHybridDecoderLayer`** [wiring]: wiring; switches between attention and mamba based on layers_block_type.
  - **`GraniteMoeHybridModel`** [wiring]: wiring.
  - **`GraniteMoeHybridForCausalLM`** [wiring]: wiring.

## granitemoeshared
- **src**: modular_granitemoeshared.py
- **status**: composable
- **rationale**: GraniteMoE Shared = GraniteMoE + a parallel shared MLP added to the routed-MoE output. SharedExpertMoE pattern; all kernels available.
- **classes**:
  - **`GraniteMoeSharedMLP`** [compute]: `L2/llama_mlp.py` (input_linear (gate/up packed) + SwiGLU + output_linear; equivalent to LlamaMLP.)
  - **`GraniteMoeSharedDecoderLayer`** [wiring]: wiring; attn + (routed_MoE + shared_MLP) parallel MLP.
  - **`GraniteMoeSharedModel`** [wiring]: wiring.
  - **`GraniteMoeSharedForCausalLM`** [wiring]: wiring.

## grounding_dino
- **src**: modular_grounding_dino.py
- **status**: partial
- **partial_reason**: MultiScaleDeformableAttention compute matches L1/rtdetrv2_deformable_attention.py, but GroundingDinoBiMultiHeadAttention (text-vision cross-attention with separate Q/K/V/projection for both modalities), GroundingDinoFusionLayer, and GroundingDinoContrastiveEmbedding rely on bespoke matmul/softmax patterns. These use only standard torch primitives (linear/softmax/matmul) so they fall back to torch — implementable but not currently kb-nano composable.
- **rationale**: Grounding DINO = DETR-style detector with a multi-scale deformable attention + bidirectional cross-attention between text and image features. The deformable attention has a kb-nano L1 kernel (rtdetrv2_deformable_attention) but the bi-multihead attention with text-image fusion, sine 2D position embeddings, contrastive embedding head, and text encoder integration are bespoke compositions not present in kb-nano.
- **classes**:
  - **`GroundingDinoConvEncoder`** [compute]: no kb-nano kernel — MultiScaleDeformableAttention compute matches L1/rtdetrv2_deformable_attention.py, but GroundingDinoBiMultiHeadAttention (text-vision cross-attention with separate Q/K/V/projection for both modalities
  - **`GroundingDinoFrozenBatchNorm2d`** [compute]: `L1/frozen_batch_norm2d.py` (Frozen affine batch norm.)
  - **`GroundingDinoSinePositionEmbedding`** [wiring]: Sine 2D position embedding via cumsum + temperature scaling; pure torch elementwise — implementable from L1 primitives.
  - **`GroundingDinoLearnedPositionEmbedding`** [compute]: `L1/embedding.py` (Two embedding tables + concat.)
  - **`GroundingDinoMultiscaleDeformableAttention`** [compute]: `L1/rtdetrv2_deformable_attention.py`, `L2/rtdetrv2_deformable_attention.py` (Multi-scale deformable attention with sampling_offsets + attention_weights + grid_sample style aggregation. RT-DETR L1 kernel matches.)
  - **`GroundingDinoTextEnhancerLayer`** [compute]: `L2/encoder_attention.py`, `L2/encoder_mlp.py` (Self-attention + FFN over text — encoder pattern.)
  - **`GroundingDinoBiMultiHeadAttention`** [wiring]: Bidirectional vision-text cross attention with separate Q/K/V/proj for both modalities; no kb-nano L2 wrapper.
  - **`GroundingDinoFusionLayer`** [wiring]: wiring; LN + bi-attention + drop-path.
  - **`GroundingDinoDeformableLayer`** [wiring]: wiring; deformable attn + FFN.
  - **`GroundingDinoEncoderLayer`** [wiring]: wiring.
  - **`GroundingDinoMultiheadAttention`** [compute]: `L2/encoder_attention.py` (Standard multi-head attention.)
  - **`GroundingDinoDecoderLayer`** [wiring]: wiring; cross-attn + self-attn + FFN.
  - **`GroundingDinoContrastiveEmbedding`** [wiring]: Linear projection + cosine similarity matmul against text embeddings; pure torch.
  - **`GroundingDinoEncoder`** [wiring]: wiring.
  - **`GroundingDinoDecoder`** [wiring]: wiring.
  - **`GroundingDinoModel`** [wiring]: wiring; combines backbone + encoder + decoder + text encoder.
  - **`GroundingDinoMLPPredictionHead`** [compute]: `L1/linear.py`, `L1/relu.py` (Multi-layer MLP.)
  - **`GroundingDinoForObjectDetection`** [wiring]: wiring; class heads + bbox heads.

## groupvit
- **src**: modeling_groupvit.py
- **status**: partial
- **rationale**: GroupViT = CLIP-text + GroupViT vision (with token-grouping cross-attention and Gumbel softmax assign). Standard linear + softmax + LN + GELU MLP composition. clip_attention + clip_mlp cover the text side; vision uses standard MHA with extra group-token assign attention (also pure torch ops).
- **classes**:
  - **`GroupViTCrossAttentionLayer`** [compute]: no kb-nano kernel — GroupViT = CLIP-text + GroupViT vision (with token-grouping cross-attention and Gumbel softmax assign). Standard linear + softmax + LN + GELU MLP composition. clip_attention + clip_mlp cover the text 
  - **`GroupViTAssignAttention`** [compute]: `L1/linear.py`, `L1/softmax.py` (Token-to-group cross-attention with hard Gumbel softmax — uses linear projections + softmax + matmul (all kb-nano L1).)
  - **`GroupViTTokenAssign`** [wiring]: wiring; mlp_inter + cross_attn + mlp_channels.
  - **`GroupViTPatchEmbeddings`** [compute]: `L1/conv2d.py`, `L2/vision_patch_embed.py` (Conv2d patch embed.)
  - **`GroupViTVisionEmbeddings`** [compute]: `L1/embedding.py`, `L2/vision_patch_embed.py` (Patch embed + position embed.)
  - **`GroupViTTextEmbeddings`** [compute]: `L1/embedding.py` (Token embed + position embed.)
  - **`GroupViTStage`** [wiring]: wiring; stack of GroupViTEncoderLayer + group projector + downsample.
  - **`GroupViTMLP`** [compute]: `L2/clip_mlp.py` (fc1 + activation (QuickGELU/GELU) + fc2.)
  - **`GroupViTMixerMLP`** [compute]: `L2/clip_mlp.py` (MLP applied with transposed input — same primitives.)
  - **`GroupViTAttention`** [compute]: `L2/clip_attention.py` (Standard CLIP-style multi-head attention with q/k/v projections + bias.)
  - **`GroupViTEncoderLayer`** [wiring]: wiring; LN + attn + LN + MLP.
  - **`GroupViTVisionEncoder`** [wiring]: wiring; multi-stage.
  - **`GroupViTTextEncoder`** [wiring]: wiring.
  - **`GroupViTTextTransformer`** [wiring]: wiring.
  - **`GroupViTTextModel`** [wiring]: wiring.
  - **`GroupViTVisionTransformer`** [wiring]: wiring.
  - **`GroupViTVisionModel`** [wiring]: wiring.
  - **`GroupViTModel`** [wiring]: wiring; vision + text + projection heads.

## helium
- **src**: modular_helium.py
- **status**: partial
- **rationale**: Helium = Llama-family with HeliumRMSNorm (fp32 cast inside) + GraniteAttention base (no attention_multiplier override here — rebound to 1/sqrt(d_k)) + HeliumMLP (= LlamaMLP) + interleaved RoPE (rotate_half via stack). All maps to L2/attention + L2/llama_mlp + L1/rms_norm + L1/rotary_emb.
- **classes**:
  - **`HeliumDecoderLayer`** [compute]: no kb-nano kernel — Helium = Llama-family with HeliumRMSNorm (fp32 cast inside) + GraniteAttention base (no attention_multiplier override here — rebound to 1/sqrt(d_k)) + HeliumMLP (= LlamaMLP) + interleaved RoPE (rotate
  - **`HeliumRMSNorm`** [compute]: `L1/rms_norm.py` (Standard Llama RMSNorm (fp32 variance compute).)
  - **`HeliumRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (Standard NeoX/Llama RoPE.)
  - **`HeliumMLP`** [compute]: `L2/llama_mlp.py` (SwiGLU gate_up + SiluAndMul + down.)
  - **`HeliumAttention`** [compute]: `L2/attention.py` (GQA + bias-free o_proj + 1/sqrt(d_k) scaling.)
  - **`HeliumModel`** [wiring]: wiring.
  - **`HeliumForCausalLM`** [wiring]: wiring.
  - **`HeliumForSequenceClassification`** [wiring]: wiring.
  - **`HeliumForTokenClassification`** [wiring]: wiring.

## hgnet_v2
- **src**: modular_hgnet_v2.py
- **status**: composable
- **rationale**: HGNetV2 = ConvNet backbone for RT-DETR family. Pure Conv2d + BatchNorm2d + ReLU + MaxPool2d + LearnableAffineBlock (scale*x + bias). All kb-nano L1 ops exist (conv2d, batch_norm2d, relu, max_pool2d).
- **classes**:
  - **`HGNetV2LearnableAffineBlock`** [wiring]: scale * x + bias (learnable scalar params); trivial elementwise.
  - **`HGNetV2ConvLayer`** [compute]: `L1/conv2d.py`, `L1/batch_norm2d.py`, `L1/relu.py` (Conv2d + BN + activation + optional LearnableAffineBlock.)
  - **`HGNetV2ConvLayerLight`** [wiring]: wiring; conv1x1 + DWConv.
  - **`HGNetV2Embeddings`** [compute]: `L1/max_pool2d.py` (Stem of multiple ConvLayers + MaxPool2d.)
  - **`HGNetV2BasicLayer`** [wiring]: wiring; sequence of ConvLayers + concat aggregation + 1x1 conv squeeze/excite + drop_path.
  - **`HGNetV2Stage`** [wiring]: wiring; downsample + N basic layers.
  - **`HGNetV2Encoder`** [wiring]: wiring; multi-stage.
  - **`HGNetV2Backbone`** [wiring]: wiring; backbone for detection.
  - **`HGNetV2ForImageClassification`** [wiring]: wiring; backbone + classifier head.
