## afmoe
- **src**: modular_afmoe.py
- **status**: composable
- **rationale**: Llama-style GQA attention with QK-norm + sigmoid output gate (matches Qwen3-Next pattern); shared+routed sigmoid-router MoE matches shared_expert_moe; standard SwiGLU MLP and RMSNorm.
- **classes**:
  - **`AfmoeRMSNorm`** [compute]: `L1/rms_norm.py` (Standard RMSNorm: variance + rsqrt + weight; identical to L1/rms_norm.py.)
  - **`AfmoeMLP`** [compute]: `L2/llama_mlp.py`, `L1/silu_and_mul.py` (SwiGLU MLP (gate_proj + up_proj + down_proj with SiLU-and-Mul) — matches LlamaMLP.)
  - **`AfmoeTokenChoiceRouter`** [compute]: `L1/linear.py`, `L1/sigmoid.py` (Linear gate -> sigmoid -> topk with expert_bias for selection only -> renormalize -> scale. Matches the sigmoid+correction_bias path of L2/shared_expert_moe.py's gating.)
  - **`AfmoeExperts`** [compute]: `L1/moe_grouped_gemm.py` (Standard fused-MoE GEMM over routed experts; same as Qwen2MoeExperts pattern handled by L1/moe_grouped_gemm.py.)
  - **`AfmoeSparseMoeBlock`** [compute]: `L2/shared_expert_moe.py` (Routed experts + always-active shared_experts SwiGLU MLP. Matches SharedExpertMoE with routing='sigmoid', correction_bias=True (expert_bias param applied to selection), renormalize=True, route_scale multiplier.)
  - **`AfmoeAttention`** [compute]: `L2/qwen3_next_attention.py`, `L1/rms_norm.py`, `L1/rotary_emb.py` (GQA Llama attention with per-head QK-RMSNorm, optional sliding window (for 'sliding_attention' layer types), RoPE only on local layers, plus sigmoid output gate (gate_proj * sigmoid then mul output). Same pattern as Qwen3NextAttention; differs only in norm class (standard RMSNorm not GemmaRMSNorm).)
  - **`AfmoeDecoderLayer`** [wiring]: Wiring: dual norm (pre+post) around attention and MLP.
  - **`AfmoeModel`** [wiring]: Wiring: stacks decoder layers, applies optional muP scaling and final norm.
  - **`AfmoeForCausalLM`** [wiring]: Wiring: lm_head + model.

## aimv2
- **src**: modular_aimv2.py
- **status**: composable
- **rationale**: SigLIP-derived ViT vision encoder + CLIP-derived text encoder, with LlamaMLP/RMSNorm replacing the original MLP/LN. All compute leaves map to existing kb-nano kernels.
- **classes**:
  - **`Aimv2RMSNorm`** [compute]: `L1/rms_norm.py` (Standard Llama RMSNorm.)
  - **`Aimv2MLP`** [compute]: `L2/llama_mlp.py`, `L1/silu_and_mul.py` (SwiGLU MLP per LlamaMLP.)
  - **`Aimv2VisionEmbeddings`** [compute]: `L1/conv2d.py`, `L1/embedding.py`, `L1/rms_norm.py` (Conv2d patch_embed + RMSNorm + learned position embed (or 2D sincos). Sum of basic ops.)
  - **`Aimv2TextEmbeddings`** [compute]: `L1/embedding.py` (Token + position embedding.)
  - **`Aimv2Attention`** [compute]: `L2/siglip_attention.py` (Separate q/k/v + dense bidirectional attention; matches SigLIPAttention (which uses DenseAttention).)
  - **`Aimv2EncoderLayer`** [wiring]: Wiring: pre-norm attention + pre-norm MLP.
  - **`Aimv2Encoder`** [wiring]: Wiring: stacks encoder layers.
  - **`Aimv2AttentionPoolingHead`** [compute]: `L1/dense_attention.py`, `L1/linear.py` (cls_token query + k_proj/v_proj of patch tokens, F.scaled_dot_product_attention, mean pool, output_proj. Standard MHA cross-attn primitives.)
  - **`Aimv2VisionModel`** [wiring]: Wiring.
  - **`Aimv2TextModel`** [wiring]: Wiring.
  - **`Aimv2Model`** [wiring]: Wiring: vision_model + text_model + projections.

## albert
- **src**: modeling_albert.py
- **status**: composable
- **rationale**: BERT-derived encoder: separate q/k/v + LayerNorm + ACT2FN MLP. Maps cleanly to encoder_attention.py + encoder_mlp.py + bert_embeddings.
- **classes**:
  - **`AlbertEmbeddings`** [compute]: `L2/bert_embeddings.py` (Word + position + token_type embeddings + LayerNorm — same as BertEmbeddings.)
  - **`AlbertAttention`** [compute]: `L2/encoder_attention.py`, `L1/layer_norm.py` (Separate q/k/v projections, eager_attention_forward, dense out + LayerNorm(residual + out). Matches EncoderSelfAttention with the post-attn LayerNorm folded in by AlbertLayer wiring.)
  - **`AlbertLayer`** [compute]: `L2/encoder_mlp.py` (Two-layer ACT2FN feed-forward (ffn -> act -> ffn_output) wrapped with LayerNorm. The FFN itself maps to EncoderIntermediate+EncoderOutput pattern; the surrounding wiring is composition.)
  - **`AlbertLayerGroup`** [wiring]: Wiring: stacks AlbertLayers.
  - **`AlbertTransformer`** [wiring]: Wiring: groups + embedding hidden mapping.
  - **`AlbertModel`** [wiring]: Wiring.
  - **`AlbertMLMHead`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (Dense + activation + LayerNorm + decoder linear.)
  - **`AlbertSOPHead`** [compute]: `L1/linear.py` (Single linear classifier.)

## align
- **src**: modeling_align.py
- **status**: partial
- **partial_reason**: EfficientNet vision encoder is structurally similar to but not the same as kb-nano's EfficientNetV2 building blocks (different block sequence, depthwise stride/padding handling, dropout-based stochastic depth). Compute primitives all exist in PyTorch (conv2d, batch_norm, sigmoid, adaptive avg pool) but no kb-nano module wraps the EfficientNet-B7 cascade used by Align.
- **rationale**: Vision tower is EfficientNet (depthwise separable conv, batchnorm, squeeze-excite). The expansion/depthwise/SE blocks rely on torch.nn Conv2d/BatchNorm2d ops with Align-specific block sequences not packaged as a kb-nano module.
- **classes**:
  - **`AlignVisionBlock`** [compute]: no kb-nano kernel — EfficientNet vision encoder is structurally similar to but not the same as kb-nano's EfficientNetV2 building blocks (different block sequence, depthwise stride/padding handling, dropout-based stochast
  - **`AlignVisionEmbeddings`** [compute]: `L1/conv2d.py`, `L1/batch_norm2d.py` (ZeroPad + Conv2d + BatchNorm2d + activation.)
  - **`AlignVisionDepthwiseConv2d`** [compute]: `L1/conv2d.py` (Conv2d with groups=in_channels.)
  - **`AlignVisionExpansionLayer`** [compute]: `L1/conv2d.py`, `L1/batch_norm2d.py` (1x1 expand conv + BN + activation.)
  - **`AlignVisionDepthwiseLayer`** [compute]: `L1/conv2d.py`, `L1/batch_norm2d.py` (Depthwise conv + BN + activation.)
  - **`AlignVisionSqueezeExciteLayer`** [compute]: `L2/efficientnetv2_squeeze_excite.py` (AdaptiveAvgPool2d + reduce conv + act + expand conv + sigmoid; same shape as EfficientNetV2 SE block.)
  - **`AlignVisionFinalBlockLayer`** [compute]: `L1/conv2d.py`, `L1/batch_norm2d.py` (Project conv + BN + dropout + skip add.)
  - **`AlignVisionEncoder`** [wiring]: Wiring: stages of blocks.
  - **`AlignTextEmbeddings`** [compute]: `L2/bert_embeddings.py` (BERT-style embeddings.)
  - **`AlignTextSelfAttention`** [compute]: `L2/encoder_attention.py` (BERT-style separate q/k/v attention.)
  - **`AlignTextSelfOutput`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (Dense + LayerNorm.)
  - **`AlignTextAttention`** [wiring]: Wiring: SelfAttention + SelfOutput.
  - **`AlignTextIntermediate`** [compute]: `L2/encoder_mlp.py` (Linear + GELU.)
  - **`AlignTextOutput`** [compute]: `L2/encoder_mlp.py` (Linear + LayerNorm + residual.)
  - **`AlignTextLayer`** [wiring]: Wiring.
  - **`AlignTextEncoder`** [wiring]: Wiring.
  - **`AlignTextPooler`** [compute]: `L1/linear.py`, `L1/tanh.py` (Linear + tanh on CLS token.)
  - **`AlignTextModel`** [wiring]: Wiring.
  - **`AlignVisionModel`** [wiring]: Wiring.
  - **`AlignModel`** [wiring]: Wiring.

## altclip
- **src**: modular_altclip.py
- **status**: composable
- **rationale**: Roberta (BERT-derived) text encoder + CLIP vision encoder; both halves map to existing encoder_attention/clip_attention/clip_mlp/bert_embeddings modules.
- **classes**:
  - **`AltRobertaEmbeddings`** [compute]: `L2/bert_embeddings.py`, `L2/xlm_roberta_embeddings.py` (Token + position + token_type + LayerNorm; Roberta-style embeddings.)
  - **`AltRobertaSelfAttention`** [compute]: `L2/encoder_attention.py` (BERT-derived q/k/v attention via ChineseCLIPText -> Bert lineage.)
  - **`AltRobertaSelfOutput`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (Dense + LayerNorm.)
  - **`AltRobertaAttention`** [wiring]: Wiring: SelfAttention + SelfOutput (BERT sibling-class pattern).
  - **`AltRobertaIntermediate`** [compute]: `L2/encoder_mlp.py` (Linear + GELU.)
  - **`AltRobertaOutput`** [compute]: `L2/encoder_mlp.py` (Linear + LayerNorm + residual.)
  - **`AltRobertaLayer`** [wiring]: Wiring.
  - **`AltRobertaEncoder`** [wiring]: Wiring.
  - **`AltRobertaPooler`** [compute]: `L1/linear.py`, `L1/tanh.py` (Linear + tanh on CLS token.)
  - **`AltCLIPAttention`** [compute]: `L2/clip_attention.py` (CLIP self-attention with separate q/k/v + scaled dot product.)
  - **`AltCLIPMLP`** [compute]: `L2/clip_mlp.py` (Two-layer fc1+QuickGELU+fc2.)
  - **`AltCLIPEncoderLayer`** [wiring]: Wiring.
  - **`AltCLIPEncoder`** [wiring]: Wiring.
  - **`AltCLIPVisionEmbeddings`** [compute]: `L1/conv2d.py`, `L1/embedding.py` (Conv2d patch embed + cls token + position embedding.)
  - **`AltCLIPVisionModel`** [wiring]: Wiring.
  - **`AltRobertaModel`** [wiring]: Wiring.
  - **`AltCLIPTextModel`** [wiring]: Wiring.
  - **`AltCLIPModel`** [wiring]: Wiring: vision + text + projections.

## apertus
- **src**: modular_apertus.py
- **status**: partial
- **rationale**: Uses xIELU activation (a learnable activation introduced in the Apertus paper) as ACT2CLS['xielu'](dtype=...). xIELU is not in PyTorch's standard activation set and has no kb-nano kernel.
- **classes**:
  - **`ApertusDecoderLayer`** [compute]: Uses xIELU activation (a learnable activation introduced in the Apertus paper) as ACT2CLS['xielu'](dtype=...). xIELU is not in PyTorch's standard activation set and has no kb-nano kernel.
  - **`ApertusMLP`** [wiring]: Two-layer MLP up_proj -> xielu -> down_proj. xielu activation has no kb-nano equivalent.
  - **`ApertusRMSNorm`** [compute]: `L1/rms_norm.py` (Standard Llama RMSNorm.)
  - **`ApertusRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (Standard Llama rotary embedding (with llama3 RoPE scaling parameters).)
  - **`ApertusAttention`** [compute]: `L2/attention.py`, `L1/rms_norm.py` (Llama attention with QK-RMSNorm. LlamaAttention covers the structure; QK-norm follows the same pattern as Qwen3 attention.)
  - **`ApertusModel`** [wiring]: Wiring.
  - **`ApertusForCausalLM`** [wiring]: Wiring.
  - **`ApertusForTokenClassification`** [wiring]: Wiring.

## arcee
- **src**: modular_arcee.py
- **status**: partial
- **partial_reason**: ArceeMLP is up_proj -> ACT2FN['relu2'] -> down_proj (no gate), i.e. a two-layer MLP with squared-ReLU activation. kb-nano provides L1/squared_relu.py and the fused L1/squared_relu_and_mul.py used in BitNet's gated MLP, but no L2 module for the bare two-layer squared-ReLU MLP. The compute is trivially expressible in PyTorch from these L1 ops, so the gap is a missing L2 wrapper rather than a missing primitive.
- **rationale**: Arcee is Llama with NemotronMLP (two-layer up_proj -> activation -> down_proj). Activation is 'relu2' (squared ReLU) which kb-nano has as L1/squared_relu.py BUT wired only for the SwiGLU-fused L1/squared_relu_and_mul.py form — there is no two-layer (no-gate) ReLU2 MLP class in kb-nano L2.
- **classes**:
  - **`ArceeForCausalLM`** [compute]: no kb-nano kernel — ArceeMLP is up_proj -> ACT2FN['relu2'] -> down_proj (no gate), i.e. a two-layer MLP with squared-ReLU activation. kb-nano provides L1/squared_relu.py and the fused L1/squared_relu_and_mul.py used in B
  - **`ArceeMLP`** [compute]: `L1/linear.py`, `L1/squared_relu.py` (up_proj -> squared_relu -> down_proj. Squared-ReLU primitive exists; assembled MLP class does not.)
  - **`ArceeForSequenceClassification`** [wiring]: Wiring.
  - **`ArceeForQuestionAnswering`** [wiring]: Wiring.
  - **`ArceeForTokenClassification`** [wiring]: Wiring.

## aria
- **src**: modular_aria.py
- **status**: partial
- **partial_reason**: AriaCrossAttention uses torch.nn.MultiheadAttention as a black box (with batch_first=True). PyTorch implements it via _scaled_dot_product_attention, so the compute is available via L1/dense_attention.py, but there is no kb-nano L2 wrapper for nn.MultiheadAttention's exact interface (separate q/k/v projections + multihead_attn module + dense linear). Vision tower (Idefics3VisionTransformer) also is not packaged as a kb-nano L4.
- **rationale**: Aria text model is Llama + shared+routed MoE (matches shared_expert_moe). However the AriaProjector cross-attention uses nn.MultiheadAttention with arbitrary attn_mask broadcast, and the vision tower is an Idefics3 ViT (not directly mapped in kb-nano).
- **classes**:
  - **`AriaCrossAttention`** [compute]: no kb-nano kernel — AriaCrossAttention uses torch.nn.MultiheadAttention as a black box (with batch_first=True). PyTorch implements it via _scaled_dot_product_attention, so the compute is available via L1/dense_attention.
  - **`AriaTextRMSNorm`** [compute]: `L1/rms_norm.py` (Standard Llama RMSNorm.)
  - **`AriaProjectorMLP`** [compute]: `L1/linear.py`, `L1/gelu.py` (linear_in -> gelu_new -> linear_out.)
  - **`AriaProjector`** [wiring]: Wiring: query parameters + cross_attn + LN + AriaProjectorMLP.
  - **`AriaSharedExpertsMLP`** [compute]: `L2/llama_mlp.py`, `L1/silu_and_mul.py` (SwiGLU MLP with intermediate=intermediate_size*num_shared_experts.)
  - **`AriaGroupedExpertsGemm`** [compute]: `L1/moe_grouped_gemm.py` (Grouped expert GEMM (per-expert batched matmul). HF references the grouped_gemm library and falls back to a sequential Python loop; the underlying op is moe_grouped_gemm.)
  - **`AriaExperts`** [compute]: `L2/fused_experts.py` (Permute tokens by expert, fc1 grouped GEMM -> SiLU*gate -> fc2 grouped GEMM, unpermute and weight-sum. Same compute as FusedExperts.)
  - **`AriaTextMoELayer`** [compute]: `L2/shared_expert_moe.py` (Linear router + softmax topk -> AriaExperts (routed) + AriaSharedExpertsMLP (shared, always-active) -> sum. Matches SharedExpertMoE with routing='softmax', shared expert active.)
  - **`AriaTextAttention`** [compute]: `L2/attention.py` (Standard LlamaAttention.)
  - **`AriaTextDecoderLayer`** [wiring]: Wiring.
  - **`AriaTextModel`** [wiring]: Wiring.
  - **`AriaTextForCausalLM`** [wiring]: Wiring.
  - **`AriaModel`** [wiring]: Wiring.
  - **`AriaForConditionalGeneration`** [wiring]: Wiring.

## audio_spectrogram_transformer
- **src**: modeling_audio_spectrogram_transformer.py
- **status**: composable
- **rationale**: ViT-style encoder over audio mel spectrogram patches. Standard separate q/k/v + LayerNorm + GELU MLP — matches encoder_attention + encoder_mlp.
- **classes**:
  - **`ASTEmbeddings`** [compute]: `L1/conv2d.py`, `L1/embedding.py` (Conv2d patch embed + cls token + distillation token + position embeddings.)
  - **`ASTPatchEmbeddings`** [compute]: `L1/conv2d.py` (Conv2d-based patch embedding.)
  - **`ASTSelfAttention`** [compute]: `L2/encoder_attention.py` (Separate q/k/v + ALL_ATTENTION_FUNCTIONS dispatch (eager_attention_forward fallback). EncoderSelfAttention covers the layout.)
  - **`ASTSelfOutput`** [compute]: `L1/linear.py` (Dense + dropout (residual added by ASTLayer).)
  - **`ASTAttention`** [wiring]: Wiring: SelfAttention + SelfOutput (sibling pattern).
  - **`ASTIntermediate`** [compute]: `L2/encoder_mlp.py` (Dense + GELU.)
  - **`ASTOutput`** [compute]: `L2/encoder_mlp.py` (Dense + dropout + residual add.)
  - **`ASTLayer`** [wiring]: Wiring: pre-norm attention + pre-norm MLP.
  - **`ASTEncoder`** [wiring]: Wiring.
  - **`ASTModel`** [wiring]: Wiring.
  - **`ASTMLPHead`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (LayerNorm + linear classifier.)

## audioflamingo3
- **src**: modular_audioflamingo3.py
- **status**: composable
- **rationale**: Whisper encoder layers + Voxtral conditional generation pattern. AudioFlamingo3 inherits attention/encoder_layer from Whisper directly, and the projector is a two-layer linear+activation+linear.
- **classes**:
  - **`AudioFlamingo3Attention`** [compute]: `L2/whisper_attention.py` (Whisper-style encoder/decoder attention.)
  - **`AudioFlamingo3EncoderLayer`** [wiring]: Wiring: attention + LN + MLP.
  - **`AudioFlamingo3Encoder`** [wiring]: Wiring: stacked encoder layers + average pool + final LN. Pure composition.
  - **`AudioFlamingo3MultiModalProjector`** [compute]: `L1/linear.py`, `L1/gelu.py` (linear_1 -> ACT2FN -> linear_2.)
  - **`AudioFlamingo3ForConditionalGeneration`** [wiring]: Wiring: audio_tower + projector + LM.

## autoformer
- **src**: modeling_autoformer.py
- **status**: partial
- **partial_reason**: AutoformerAttention replaces SDPA with autocorrelation: q/k FFT -> conjugate multiply -> inverse FFT -> top-k autocorrelation delay aggregation via torch.gather/roll. PyTorch supplies torch.fft.rfft/irfft, but kb-nano has no FFT primitive and no autocorrelation-attention kernel.
- **rationale**: AutoformerAttention is FFT-based AutoCorrelation (torch.fft.rfft / irfft + top-k delay aggregation), not standard scaled-dot-product attention. AutoformerSeriesDecompositionLayer also uses torch.nn.AvgPool1d for trend/seasonal decomposition.
- **classes**:
  - **`AutoformerAttention`** [compute]: no kb-nano kernel — AutoformerAttention replaces SDPA with autocorrelation: q/k FFT -> conjugate multiply -> inverse FFT -> top-k autocorrelation delay aggregation via torch.gather/roll. PyTorch supplies torch.fft.rfft/i
  - **`AutoformerFeatureEmbedder`** [compute]: `L1/embedding.py` (Concat of Embedding tables for categorical features.)
  - **`AutoformerStdScaler`** [wiring]: Standardisation: mean/std over time. Pure tensor ops.
  - **`AutoformerMeanScaler`** [wiring]: Mean-only scaling. Pure tensor ops.
  - **`AutoformerNOPScaler`** [wiring]: No-op scaler; identity.
  - **`AutoformerSinusoidalPositionalEmbedding`** [compute]: `L1/sinusoidal_embed.py` (Sinusoidal positional embedding table.)
  - **`AutoformerValueEmbedding`** [compute]: `L1/linear.py` (Linear projection of value features.)
  - **`AutoformerSeriesDecompositionLayer`** [compute]: `L1/avg_pool1d.py` (AvgPool1d-based trend extraction; seasonal = input - trend.)
  - **`AutoformerLayernorm`** [compute]: `L1/layer_norm.py` (Standard LayerNorm wrapped to subtract a global mean.)
  - **`AutoformerEncoderLayer`** [wiring]: Wiring.
  - **`AutoformerDecoderLayer`** [wiring]: Wiring.
  - **`AutoformerEncoder`** [wiring]: Wiring.
  - **`AutoformerDecoder`** [wiring]: Wiring.
  - **`AutoformerModel`** [wiring]: Wiring.
  - **`AutoformerForPrediction`** [wiring]: Wiring.

## aya_vision
- **src**: modular_aya_vision.py
- **status**: composable
- **rationale**: LLaVA-derived vision-language wrapper: vision tower (SigLIP) + projector (SwiGLU) + Cohere/Llama text. Compute leaves are linear/silu/layer_norm.
- **classes**:
  - **`AyaVisionMultiModalProjector`** [compute]: `L1/layer_norm.py`, `L1/linear.py`, `L1/silu_and_mul.py` (pixel_shuffle (reshape only) + LayerNorm + linear_1 -> chunk -> silu(gate)*x (SwiGLU split) + linear_2.)
  - **`AyaVisionModel`** [wiring]: Wiring: vision_tower + projector + LM.

## bamba
- **src**: modular_bamba.py
- **status**: partial
- **partial_reason**: BambaConfig hardcodes `partial_rotary_factor = 0.5` (configuration_bamba.py). kb-nano L1/rotary_emb.py rotates the full head_dim; partial-rotary requires either external q_rot/q_pass slicing in user code or a Gemma4-style proportional embedding wrapper. Same gap as phi/persimmon/glm — applied here for consistency.
- **rationale**: Hybrid Mamba2 + Llama attention decoder. BambaMixer is Mamba2 (causal_conv1d + SSM + RMSNormGated); BambaAttention is LlamaAttention with partial-rotary 0.5; MLP is SwiGLU. Mamba2 + RMSNormGated + SwiGLU + LlamaAttention all map to existing kernels; the partial-rotary path needs external slicing (decomposable from L1 ops + standard PyTorch).
- **classes**:
  - **`BambaRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (Standard NeoX RoPE.)
  - **`BambaAttention`** [compute]: `L2/attention.py` (Standard Llama GQA attention.)
  - **`BambaRMSNormGated`** [compute]: `L1/rms_norm_gated.py` (Mamba2-style gated RMSNorm.)
  - **`BambaMixer`** [compute]: `L2/mamba2_mixer.py`, `L1/causal_conv1d.py`, `L1/rms_norm_gated.py` (Mamba2 mixer: in_proj -> conv1d -> SSM -> norm_gated -> out_proj. Same as kb-nano Mamba2Mixer.)
  - **`BambaMLP`** [compute]: `L2/llama_mlp.py`, `L1/silu_and_mul.py` (Standard SwiGLU MLP.)
  - **`BambaRMSNorm`** [compute]: `L1/rms_norm.py` (Standard Llama RMSNorm.)
  - **`BambaDecoderLayer`** [wiring]: Wiring: per-layer mamba mixer or attention + MLP.
  - **`BambaModel`** [wiring]: Wiring.
  - **`BambaForCausalLM`** [wiring]: Wiring.

## bark
- **src**: modeling_bark.py
- **status**: composable
- **rationale**: GPT-NeoX style fused-QKV self-attention (causal or full) + 4x GELU MLP. Maps to encoder_attention.py (with fused QKV variant) and encoder_mlp.py-style ACT2FN MLP. Bark composes Semantic/Coarse/Fine sub-models that are all standard transformer decoders.
- **classes**:
  - **`BarkSelfAttention`** [compute]: `L1/linear.py`, `L1/dense_attention.py`, `L1/softmax.py` (Fused att_proj (3*hidden) -> split q/k/v -> dense attention with optional causal mask. Same compute as DenseAttention (manual softmax variant).)
  - **`BarkSelfFlashAttention2`** [compute]: `L1/flash_attn_dense.py` (Same projection layout but uses _flash_attention_forward.)
  - **`BarkMLP`** [compute]: `L1/linear.py`, `L1/gelu.py` (in_proj (4x hidden) + GELU + out_proj. Two-layer GELU MLP.)
  - **`BarkBlock`** [wiring]: Wiring.
  - **`BarkCausalModel`** [wiring]: Wiring.
  - **`BarkSemanticModel`** [wiring]: Wiring.
  - **`BarkCoarseModel`** [wiring]: Wiring.
  - **`BarkFineModel`** [wiring]: Wiring.
  - **`BarkModel`** [wiring]: Wiring (composes the three sub-models + EnCodec).

## bart
- **src**: modeling_bart.py
- **status**: composable
- **rationale**: Standard BART encoder-decoder attention (q/k/v + EncoderDecoderCache) — directly matches whisper_attention's three sibling classes (encoder self-attn, decoder self-attn, decoder cross-attn).
- **classes**:
  - **`BartLearnedPositionalEmbedding`** [compute]: `L1/embedding.py` (Learned position embedding with offset.)
  - **`BartScaledWordEmbedding`** [compute]: `L1/embedding.py` (Embedding * scale.)
  - **`BartAttention`** [compute]: `L2/whisper_attention.py` (Encoder/decoder attention with optional cross-attention via key_value_states. Identical pattern to whisper_attention's three classes.)
  - **`BartEncoderLayer`** [wiring]: Wiring: self-attn + LN + FFN + LN.
  - **`BartDecoderLayer`** [wiring]: Wiring: self-attn + LN + cross-attn + LN + FFN + LN.
  - **`BartClassificationHead`** [compute]: `L1/linear.py` (Dense + tanh + dense classifier.)
  - **`BartEncoder`** [wiring]: Wiring.
  - **`BartDecoder`** [wiring]: Wiring.
  - **`BartModel`** [wiring]: Wiring.
  - **`BartForConditionalGeneration`** [wiring]: Wiring.
  - **`BartForSequenceClassification`** [wiring]: Wiring.
  - **`BartForQuestionAnswering`** [wiring]: Wiring.
  - **`BartForCausalLM`** [wiring]: Wiring.

## beit
- **src**: modeling_beit.py
- **status**: composable
- **rationale**: ViT-style encoder with relative position bias (added as attn_mask to SDPA), separate q/k/v projections, GELU MLP, optional layer-scale gammas. Maps to encoder_attention + encoder_mlp + bias add as attn_mask.
- **classes**:
  - **`BeitEmbeddings`** [compute]: `L1/conv2d.py`, `L1/embedding.py` (Patch embed + cls token + optional mask token + position embeddings.)
  - **`BeitPatchEmbeddings`** [compute]: `L1/conv2d.py` (Conv2d-based patch embedding.)
  - **`BeitSelfAttention`** [compute]: `L2/encoder_attention.py`, `L1/softmax.py` (Separate q/k/v + matmul + softmax + matmul, with optional relative_position_bias added to attn scores.)
  - **`BeitSdpaSelfAttention`** [compute]: `L2/encoder_attention.py`, `L1/sdpa.py` (Same projections + F.scaled_dot_product_attention with attn_bias for relative position bias.)
  - **`BeitSelfOutput`** [compute]: `L1/linear.py` (Dense + dropout.)
  - **`BeitAttention`** [wiring]: Wiring: SelfAttention + SelfOutput (sibling pattern).
  - **`BeitIntermediate`** [compute]: `L2/encoder_mlp.py` (Dense + GELU.)
  - **`BeitOutput`** [compute]: `L2/encoder_mlp.py` (Dense + dropout.)
  - **`BeitLayer`** [wiring]: Wiring: pre-LN attention + pre-LN MLP with optional layer-scale (lambda_1, lambda_2) and DropPath.
  - **`BeitRelativePositionBias`** [compute]: `L1/embedding.py` (Indexed lookup of a learned bias table per (head, q_pos, k_pos) pair.)
  - **`BeitEncoder`** [wiring]: Wiring.
  - **`BeitModel`** [wiring]: Wiring.
  - **`BeitPooler`** [compute]: `L1/layer_norm.py` (Mean pool + LN (or cls).)
  - **`BeitForMaskedImageModeling`** [wiring]: Wiring.
  - **`BeitForImageClassification`** [wiring]: Wiring.
  - **`BeitForSemanticSegmentation`** [wiring]: Wiring (uses ConvModule + UperHead/FCNHead).
  - **`BeitConvModule`** [compute]: `L1/conv2d.py`, `L1/batch_norm2d.py` (Conv2d + BN + activation.)
  - **`BeitPyramidPoolingBlock`** [compute]: `L1/adaptive_avg_pool2d.py` (Adaptive pool + ConvModule.)
  - **`BeitPyramidPoolingModule`** [wiring]: Wiring.
  - **`BeitUperHead`** [wiring]: Wiring.
  - **`BeitFCNHead`** [wiring]: Wiring.
  - **`BeitBackbone`** [wiring]: Wiring.

## bert
- **src**: modeling_bert.py
- **status**: composable
- **rationale**: Canonical BERT: separate q/k/v + LayerNorm + GELU MLP. Encoder-attention/encoder-mlp/bert-embeddings cover every compute leaf.
- **classes**:
  - **`BertEmbeddings`** [compute]: `L2/bert_embeddings.py` (Word + position + token_type + LayerNorm.)
  - **`BertSelfAttention`** [compute]: `L2/encoder_attention.py` (Separate q/k/v projections + ALL_ATTENTION_FUNCTIONS dispatch (eager_attention_forward fallback).)
  - **`BertCrossAttention`** [compute]: `L2/whisper_attention.py` (Cross-attention with EncoderDecoderCache (encoder K/V cached). Same as whisper decoder cross-attention pattern.)
  - **`BertSelfOutput`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (Dense + dropout + LayerNorm(residual + out).)
  - **`BertAttention`** [wiring]: Wiring: SelfAttention + SelfOutput (sibling pattern).
  - **`BertIntermediate`** [compute]: `L2/encoder_mlp.py` (Dense + GELU.)
  - **`BertOutput`** [compute]: `L2/encoder_mlp.py` (Dense + dropout + LayerNorm(residual + out).)
  - **`BertLayer`** [wiring]: Wiring.
  - **`BertEncoder`** [wiring]: Wiring.
  - **`BertPooler`** [compute]: `L1/linear.py`, `L1/tanh.py` (Linear + tanh on CLS token.)
  - **`BertPredictionHeadTransform`** [compute]: `L1/linear.py`, `L1/layer_norm.py`, `L1/gelu.py` (Dense + GELU + LayerNorm.)
  - **`BertLMPredictionHead`** [wiring]: Wiring: transform + decoder.
  - **`BertModel`** [wiring]: Wiring.
  - **`BertForPreTraining`** [wiring]: Wiring.
  - **`BertLMHeadModel`** [wiring]: Wiring.
  - **`BertForMaskedLM`** [wiring]: Wiring.
  - **`BertForNextSentencePrediction`** [wiring]: Wiring.
  - **`BertForSequenceClassification`** [wiring]: Wiring.
  - **`BertForMultipleChoice`** [wiring]: Wiring.
  - **`BertForTokenClassification`** [wiring]: Wiring.
  - **`BertForQuestionAnswering`** [wiring]: Wiring.

## bert_generation
- **src**: modeling_bert_generation.py
- **status**: composable
- **rationale**: Decoder variant of BERT — same compute leaves as bert (separate q/k/v + LayerNorm + GELU MLP), just trained with causal mask for generation.
- **classes**:
  - **`BertGenerationSelfOutput`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (Dense + dropout + LayerNorm.)
  - **`BertGenerationSelfAttention`** [compute]: `L2/encoder_attention.py` (BERT-style separate q/k/v attention.)
  - **`BertGenerationCrossAttention`** [compute]: `L2/whisper_attention.py` (Cross-attention with EncoderDecoderCache.)
  - **`BertGenerationAttention`** [wiring]: Wiring.
  - **`BertGenerationIntermediate`** [compute]: `L2/encoder_mlp.py` (Dense + GELU.)
  - **`BertGenerationOutput`** [compute]: `L2/encoder_mlp.py` (Dense + LayerNorm.)
  - **`BertGenerationLayer`** [wiring]: Wiring.
  - **`BertEncoder`** [wiring]: Wiring.
  - **`BertGenerationEmbeddings`** [compute]: `L2/bert_embeddings.py` (Word + position + LayerNorm (no token_type).)
  - **`BertGenerationEncoder`** [wiring]: Wiring.
  - **`BertGenerationOnlyLMHead`** [compute]: `L1/linear.py` (LM head linear.)
  - **`BertGenerationDecoder`** [wiring]: Wiring.

## big_bird
- **src**: modeling_big_bird.py
- **status**: partial
- **partial_reason**: BigBirdBlockSparseAttention combines global tokens + sliding window + random-blocks attention via masked dense matmuls on grouped block tensors. PyTorch supplies the underlying ops (matmul, softmax, gather), but kb-nano has no block-sparse attention kernel and no L2 wrapper for this specific compute pattern. The full-attention BigBirdSelfAttention path is composable.
- **rationale**: BigBirdBlockSparseAttention is a custom block-sparse attention combining global, sliding-window, and random-block attention patterns. Implemented in pure PyTorch (no custom CUDA kernel), but kb-nano has no equivalent block-sparse attention module.
- **classes**:
  - **`BigBirdBlockSparseAttention`** [compute]: no kb-nano kernel — BigBirdBlockSparseAttention combines global tokens + sliding window + random-blocks attention via masked dense matmuls on grouped block tensors. PyTorch supplies the underlying ops (matmul, softmax, g
  - **`BigBirdEmbeddings`** [compute]: `L2/bert_embeddings.py` (Same as BertEmbeddings.)
  - **`BigBirdSelfAttention`** [compute]: `L2/encoder_attention.py` (Standard BERT-style separate q/k/v full attention (used in 'original_full' mode).)
  - **`BigBirdSelfOutput`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (Dense + LayerNorm + residual.)
  - **`BigBirdAttention`** [wiring]: Wiring: chooses self-attention variant + SelfOutput.
  - **`BigBirdIntermediate`** [compute]: `L2/encoder_mlp.py` (Dense + GELU.)
  - **`BigBirdOutput`** [compute]: `L2/encoder_mlp.py` (Dense + LayerNorm.)
  - **`BigBirdLayer`** [wiring]: Wiring.
  - **`BigBirdEncoder`** [wiring]: Wiring.
  - **`BigBirdModel`** [wiring]: Wiring.
  - **`BigBirdForPreTraining`** [wiring]: Wiring.
  - **`BigBirdForMaskedLM`** [wiring]: Wiring.
  - **`BigBirdForCausalLM`** [wiring]: Wiring.
  - **`BigBirdForSequenceClassification`** [wiring]: Wiring.
  - **`BigBirdForMultipleChoice`** [wiring]: Wiring.
  - **`BigBirdForTokenClassification`** [wiring]: Wiring.
  - **`BigBirdForQuestionAnswering`** [wiring]: Wiring.

## bigbird_pegasus
- **src**: modeling_bigbird_pegasus.py
- **status**: partial
- **partial_reason**: BigBirdPegasusBlockSparseAttention reuses the BigBird block-sparse pattern (global + sliding window + random blocks). No kb-nano equivalent. The decoder full attention is composable via whisper_attention.
- **rationale**: Same block-sparse attention pattern as big_bird (BigBirdPegasusBlockSparseAttention), but in an encoder-decoder architecture. Encoder uses block-sparse; decoder uses standard full attention.
- **classes**:
  - **`BigBirdPegasusBlockSparseAttention`** [compute]: BigBirdPegasusBlockSparseAttention reuses the BigBird block-sparse pattern (global + sliding window + random blocks). No kb-nano equivalent. The decoder full attention is composable via whisper_attent
  - **`BigBirdPegasusLearnedPositionalEmbedding`** [compute]: `L1/embedding.py` (Learned position embedding.)
  - **`BigBirdPegasusScaledWordEmbedding`** [compute]: `L1/embedding.py` (Embedding * scale.)
  - **`BigBirdPegasusSelfAttention`** [compute]: `L2/encoder_attention.py` (Standard q/k/v attention (full mode).)
  - **`BigBirdPegasusEncoderAttention`** [wiring]: Wiring: dispatches between full and block-sparse self-attention.
  - **`BigBirdPegasusDecoderAttention`** [compute]: `L2/whisper_attention.py` (Standard decoder self/cross attention.)
  - **`BigBirdPegasusEncoderLayer`** [wiring]: Wiring.
  - **`BigBirdPegasusDecoderLayer`** [wiring]: Wiring.
  - **`BigBirdPegasusClassificationHead`** [compute]: `L1/linear.py` (Dense + tanh + dense classifier.)
  - **`BigBirdPegasusEncoder`** [wiring]: Wiring.
  - **`BigBirdPegasusDecoder`** [wiring]: Wiring.
  - **`BigBirdPegasusModel`** [wiring]: Wiring.
  - **`BigBirdPegasusForConditionalGeneration`** [wiring]: Wiring.
  - **`BigBirdPegasusForSequenceClassification`** [wiring]: Wiring.
  - **`BigBirdPegasusForQuestionAnswering`** [wiring]: Wiring.
  - **`BigBirdPegasusForCausalLM`** [wiring]: Wiring.

## biogpt
- **src**: modular_biogpt.py
- **status**: composable
- **rationale**: Pure BART-style decoder lineage (BartAttention, BartDecoderLayer). Maps directly to whisper_attention pattern.
- **classes**:
  - **`BioGptLearnedPositionalEmbedding`** [compute]: `L1/embedding.py` (Learned position embedding with offset (OPT/BART style).)
  - **`BioGptScaledWordEmbedding`** [compute]: `L1/embedding.py` (Embedding * scale.)
  - **`BioGptAttention`** [compute]: `L2/whisper_attention.py` (BART-style q/k/v attention with optional cross-attention.)
  - **`BioGptDecoderLayer`** [wiring]: Wiring: self-attn + LN + FFN + LN.
  - **`BioGptModel`** [wiring]: Wiring.
  - **`BioGptForCausalLM`** [wiring]: Wiring.
  - **`BioGptForTokenClassification`** [wiring]: Wiring.
  - **`BioGptForSequenceClassification`** [wiring]: Wiring.

## bit
- **src**: modeling_bit.py
- **status**: partial
- **partial_reason**: WeightStandardizedConv2d standardizes the conv weights (batch_norm on weight tensor) on every forward pass. Implemented with PyTorch's nn.functional.batch_norm + conv2d but no kb-nano L1 op fuses or wraps this; existing L1/conv2d.py is a vanilla conv. DynamicPad2d also computes per-input padding which is a torch op but not packaged.
- **rationale**: Big Transfer ResNet-v2 with WeightStandardizedConv2d (custom: applies BatchNorm to weights at every forward) + GroupNorm + dynamic 'SAME' padding. WeightStandardizedConv2d is a custom op; kb-nano's Conv2d does not standardize weights.
- **classes**:
  - **`BitPreActivationBottleneckLayer`** [compute]: WeightStandardizedConv2d standardizes the conv weights (batch_norm on weight tensor) on every forward pass. Implemented with PyTorch's nn.functional.batch_norm + conv2d but no kb-nano L1 op fuses or w
  - **`WeightStandardizedConv2d`** [compute]: `L1/conv2d.py`, `L1/batch_norm2d.py` (Conv2d with weight standardization via F.batch_norm on the weight tensor each forward. Primitives exist but are not packaged as a single kb-nano module.)
  - **`BitGroupNormActivation`** [compute]: `L1/group_norm.py` (GroupNorm + activation.)
  - **`DynamicPad2d`** [wiring]: Dynamic per-input padding for 'SAME' padding mode. Pure torch ops; no kb-nano wrapper.
  - **`BitMaxPool2d`** [compute]: `L1/max_pool2d.py` (MaxPool2d with optional dynamic padding.)
  - **`BitEmbeddings`** [wiring]: Wiring: stem conv + GN + MaxPool.
  - **`BitDropPath`** [wiring]: Stochastic depth (Bernoulli mask). Pure tensor ops.
  - **`BitBottleneckLayer`** [wiring]: Wiring: WS-Conv x3 + GN + skip.
  - **`BitDownsampleConv`** [wiring]: Wiring.
  - **`BitStage`** [wiring]: Wiring: stack of bottleneck layers.
  - **`BitEncoder`** [wiring]: Wiring.
  - **`BitModel`** [wiring]: Wiring.
  - **`BitForImageClassification`** [wiring]: Wiring.
  - **`BitBackbone`** [wiring]: Wiring.

## bitnet
- **src**: modular_bitnet.py
- **status**: kb_nano_l4
- **rationale**: kb-nano has tasks/baseline/L4/bitnet.py whose docstring header explicitly targets 'microsoft/bitnet-b1.58-2B-4T' (W1.58A8 native 1.58-bit weights + 8-bit activations) — the same model family as this HF folder.
- **classes**:
  - **`BitNetRMSNorm`** [compute]: `L1/bitnet_rms_norm.py` (Standard RMSNorm; bitnet uses the bitnet-specific quantized variant for sub-norms inside attention/MLP.)
  - **`BitNetMLP`** [compute]: `L2/bitnet_mlp.py`, `L1/bitnet_linear.py`, `L1/bitnet_rms_norm.py`, `L1/squared_relu_and_mul.py` (GemmaMLP layout (gate/up/down) with extra ffn_sub_norm (RMSNorm) inserted after the gated activation. Bitnet kb-nano variant uses bitnet_linear (W1.58A8 quantized linear) and squared-ReLU activation.)
  - **`BitNetAttention`** [compute]: `L2/bitnet_attention.py`, `L1/bitnet_linear.py`, `L1/bitnet_rms_norm.py`, `L1/rotary_emb.py` (Llama GQA attention with extra attn_sub_norm (RMSNorm) inserted between attention output and o_proj. Bitnet variant uses bitnet_linear for q/k/v/o projections.)
  - **`BitNetDecoderLayer`** [wiring]: Wiring.
  - **`BitNetModel`** [wiring]: Wiring.
  - **`BitNetForCausalLM`** [wiring]: Wiring.

## blenderbot
- **src**: modeling_blenderbot.py
- **status**: composable
- **rationale**: Standard BART-style enc-dec attention. Maps cleanly to whisper_attention's three sibling classes.
- **classes**:
  - **`BlenderbotLearnedPositionalEmbedding`** [compute]: `L1/embedding.py` (Learned position embedding.)
  - **`BlenderbotScaledWordEmbedding`** [compute]: `L1/embedding.py` (Embedding * scale.)
  - **`BlenderbotAttention`** [compute]: `L2/whisper_attention.py` (BART/Whisper-style q/k/v with optional cross-attention via key_value_states.)
  - **`BlenderbotEncoderLayer`** [wiring]: Wiring.
  - **`BlenderbotDecoderLayer`** [wiring]: Wiring.
  - **`BlenderbotEncoder`** [wiring]: Wiring.
  - **`BlenderbotDecoder`** [wiring]: Wiring.
  - **`BlenderbotModel`** [wiring]: Wiring.
  - **`BlenderbotForConditionalGeneration`** [wiring]: Wiring.
  - **`BlenderbotForCausalLM`** [wiring]: Wiring.

## blenderbot_small
- **src**: modeling_blenderbot_small.py
- **status**: composable
- **rationale**: Same BART-style enc-dec architecture as Blenderbot, smaller hidden size. Maps to whisper_attention.
- **classes**:
  - **`BlenderbotSmallLearnedPositionalEmbedding`** [compute]: `L1/embedding.py` (Learned position embedding.)
  - **`BlenderbotSmallAttention`** [compute]: `L2/whisper_attention.py` (BART/Whisper-style q/k/v with optional cross-attention.)
  - **`BlenderbotSmallEncoderLayer`** [wiring]: Wiring.
  - **`BlenderbotSmallDecoderLayer`** [wiring]: Wiring.
  - **`BlenderbotSmallEncoder`** [wiring]: Wiring.
  - **`BlenderbotSmallDecoder`** [wiring]: Wiring.
  - **`BlenderbotSmallModel`** [wiring]: Wiring.
  - **`BlenderbotSmallForConditionalGeneration`** [wiring]: Wiring.
  - **`BlenderbotSmallForCausalLM`** [wiring]: Wiring.

## blip/blip
- **src**: modeling_blip.py
- **status**: composable
- **rationale**: BLIP vision encoder is ViT-style with fused QKV (matches vit_encoder_attention) + two-layer GELU MLP (matches clip_mlp/encoder_mlp).
- **classes**:
  - **`BlipVisionEmbeddings`** [compute]: `L1/conv2d.py`, `L1/embedding.py` (Conv2d patch embed + cls token + position embeddings.)
  - **`BlipTextEmbeddings`** [compute]: `L1/embedding.py` (Token + position embedding.)
  - **`BlipAttention`** [compute]: `L2/vit_encoder_attention.py` (Fused qkv linear + dense attention with manual softmax. Same layout as VitEncoderAttention.)
  - **`BlipMLP`** [compute]: `L2/clip_mlp.py` (fc1 + ACT2FN + fc2. Same as CLIPMLP.)
  - **`BlipEncoderLayer`** [wiring]: Wiring.
  - **`BlipEncoder`** [wiring]: Wiring.
  - **`BlipVisionModel`** [wiring]: Wiring.
  - **`BlipModel`** [wiring]: Wiring.
  - **`BlipForConditionalGeneration`** [wiring]: Wiring.
  - **`BlipForQuestionAnswering`** [wiring]: Wiring.
  - **`BlipForImageTextRetrieval`** [wiring]: Wiring.

## blip/blip_text
- **src**: modeling_blip_text.py
- **status**: composable
- **rationale**: BERT-derived text encoder with optional cross-attention to vision features. Maps to encoder_attention (self) + whisper_attention (cross) + bert_embeddings + encoder_mlp.
- **classes**:
  - **`BlipTextEmbeddings`** [compute]: `L2/bert_embeddings.py` (Word + position + LayerNorm.)
  - **`BlipTextSelfAttention`** [compute]: `L2/encoder_attention.py`, `L2/whisper_attention.py` (Separate q/k/v with optional cross-attention via encoder_hidden_states. Self-attn path matches encoder_attention; cross-attn path matches whisper_attention's cross-attention class.)
  - **`BlipTextSelfOutput`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (Dense + LayerNorm.)
  - **`BlipTextAttention`** [wiring]: Wiring.
  - **`BlipTextIntermediate`** [compute]: `L2/encoder_mlp.py` (Dense + GELU.)
  - **`BlipTextOutput`** [compute]: `L2/encoder_mlp.py` (Dense + LayerNorm.)
  - **`BlipTextLayer`** [wiring]: Wiring: self-attn + optional cross-attn + FFN.
  - **`BlipTextEncoder`** [wiring]: Wiring.
  - **`BlipTextPooler`** [compute]: `L1/linear.py`, `L1/tanh.py` (CLS-token linear + tanh.)
  - **`BlipTextPredictionHeadTransform`** [compute]: `L1/linear.py`, `L1/layer_norm.py`, `L1/gelu.py` (Dense + GELU + LayerNorm.)
  - **`BlipTextLMPredictionHead`** [wiring]: Wiring.
  - **`BlipTextModel`** [wiring]: Wiring.
  - **`BlipTextLMHeadModel`** [wiring]: Wiring.
