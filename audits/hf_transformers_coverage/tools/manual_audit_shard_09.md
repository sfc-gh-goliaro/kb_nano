## minimax
- **src**: modular_minimax.py
- **status**: partial
- **partial_reason**: MiniMaxLightningAttention requires a custom linear-attention kernel with block-wise (Q@K^T)*V intra/inter decomposition and decay buffers; HF realises it via plain torch.matmul/torch.split loops. No kb-nano L1 covers this; closest is L1/chunk_gated_delta_rule but the formulation differs.
- **rationale**: MiniMax interleaves Mixtral-style full attention layers (composable) with MiniMaxLightningAttention - a custom block-wise linear attention with per-head decay rates that has no kb-nano kernel.
- **classes**:
  - **`MiniMaxLightningAttention`** [compute]: no kb-nano kernel — MiniMaxLightningAttention requires a custom linear-attention kernel with block-wise (Q@K^T)*V intra/inter decomposition and decay buffers; HF realises it via plain torch.matmul/torch.split loops. No k
  - **`MiniMaxRMSNorm`** [compute]: `L1/rms_norm.py` (Standard Llama-family RMSNorm (inherits from MixtralRMSNorm).)
  - **`MiniMaxRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (Standard NeoX/Llama RoPE inherited via Gemma2.)
  - **`MiniMaxAttention`** [compute]: `L2/attention.py` (Plain Mixtral attention (Llama-family GQA + RoPE). Maps to LlamaAttention.)
  - **`MiniMaxTopKRouter`** [compute]: `L1/topk_softmax.py` (Standard top-k softmax router.)
  - **`MiniMaxSparseMoeBlock`** [compute]: `L2/mixtral_moe.py` (Mixtral-style fused SwiGLU MoE block.)
  - **`MiniMaxDecoderLayer`** [wiring]: Wiring: alternates LightningAttention vs MixtralAttention based on layer_types, plus alpha/beta-scaled residuals.
  - **`MiniMaxModel`** [wiring]: Wiring/Model class.
  - **`MiniMaxForCausalLM`** [wiring]: Wiring/head class.

## mistral
- **src**: modular_mistral.py
- **status**: composable
- **rationale**: Mistral inherits Llama-family blocks; every leaf compute (Mistral attention with sliding window, SwiGLU MLP, RMSNorm, RoPE) maps to existing kb-nano L2/L1 kernels.
- **classes**:
  - **`MistralMLP`** [compute]: `L2/llama_mlp.py`, `L1/silu_and_mul.py` (SwiGLU gate_up + down with silu activation; identical to Llama.)
  - **`MistralAttention`** [compute]: `L2/attention.py`, `L1/rotary_emb.py`, `L1/flash_attn_prefill.py`, `L1/flash_attn_decode.py` (GQA Llama attention with sliding_window flag. LlamaAttention in kb-nano (L2/attention.py) covers this.)
  - **`MistralDecoderLayer`** [wiring]: Wiring layer: input_layernorm -> self_attn -> residual -> post_attention_layernorm -> mlp -> residual.
  - **`MistralModel`** [wiring]: Wiring class.
  - **`MistralForCausalLM`** [wiring]: Wiring/head class.

## mistral3
- **src**: modular_mistral3.py
- **status**: partial
- **partial_reason**: Mistral3PatchMerger relies on torch.nn.functional.unfold (sliding-window patch extraction). No kb-nano L1 op for unfold; PyTorch has it natively. Vision tower + projector also depend on Pixtral encoder (not in shard).
- **rationale**: Mistral3 is Pixtral vision tower + Mistral text via LLaVA wrapping. The Mistral3PatchMerger uses torch.nn.functional.unfold for spatial merging which has no kb-nano kernel.
- **classes**:
  - **`Mistral3PatchMerger`** [compute]: Mistral3PatchMerger relies on torch.nn.functional.unfold (sliding-window patch extraction). No kb-nano L1 op for unfold; PyTorch has it natively. Vision tower + projector also depend on Pixtral encode
  - **`Mistral3RMSNorm`** [compute]: `L1/rms_norm.py` (Standard RMSNorm.)
  - **`Mistral3MultiModalProjector`** [compute]: `L1/rms_norm.py`, `L1/linear.py` (Norm + PatchMerger + Linear-act-Linear projector. Composes the unsupported PatchMerger.)
  - **`Mistral3Model`** [wiring]: Wiring: vision_tower + multi_modal_projector + language_model.
  - **`Mistral3ForConditionalGeneration`** [wiring]: Top-level VLM wiring.

## mixtral
- **src**: modular_mixtral.py
- **status**: kb_nano_l4
- **rationale**: kb-nano has L4/mixtral.py, a Mixtral-8x7B pipeline using shared TP layers with L2/mixtral_moe.py for the MoE block.
- **classes**:
  - **`MixtralExperts`** [compute]: `L1/moe_grouped_gemm.py`, `L2/fused_experts.py` (SwiGLU experts with gate_up_proj/down_proj 3D tensors; kb-nano fused_experts handles this efficiently.)
  - **`MixtralTopKRouter`** [compute]: `L1/topk_softmax.py` (Linear -> softmax(fp32) -> topk -> normalize. Standard top-k router.)
  - **`MixtralSparseMoeBlock`** [compute]: `L2/mixtral_moe.py` (Composes router + experts; matches kb-nano L2/mixtral_moe.MixtralMoE.)
  - **`MixtralRMSNorm`** [compute]: `L1/rms_norm.py` (Standard RMSNorm.)
  - **`MixtralRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (Standard NeoX RoPE.)
  - **`MixtralAttention`** [compute]: `L2/attention.py` (GQA + sliding window attention (Llama-family).)
  - **`MixtralDecoderLayer`** [wiring]: Wiring: norm -> attn -> residual -> norm -> moe -> residual.
  - **`MixtralModel`** [wiring]: Wiring class.
  - **`MixtralForCausalLM`** [wiring]: Wiring/head class.

## mllama
- **src**: modeling_mllama.py
- **status**: partial
- **partial_reason**: MllamaPrecomputedAspectRatioEmbedding/MllamaPrecomputedPositionEmbedding are bespoke gated tile-aware embeddings. MllamaTextCrossAttention has tanh-gated residuals, q/k RMSNorm, and a custom cross-attention cache pattern (update only on first call) that no kb-nano L2 covers. PyTorch primitives suffice; no missing kernel beyond compositional.
- **rationale**: Llama 3.2-Vision with text+vision towers and gated cross-attention. The vision tower uses tile/aspect-ratio gated embeddings with no kb-nano analog; cross-attention has q_norm/k_norm with non-trivial cache update logic and no kb-nano L2 cross-attention kernel covers it.
- **classes**:
  - **`MllamaVisionEncoderLayer`** [compute]: no kb-nano kernel — MllamaPrecomputedAspectRatioEmbedding/MllamaPrecomputedPositionEmbedding are bespoke gated tile-aware embeddings. MllamaTextCrossAttention has tanh-gated residuals, q/k RMSNorm, and a custom cross-att
  - **`MllamaPrecomputedAspectRatioEmbedding`** [wiring]: Aspect-ratio-id Embedding with gated tanh; bespoke.
  - **`MllamaPrecomputedPositionEmbedding`** [wiring]: Per-tile position + token position with tanh-gated mixing; bespoke.
  - **`MllamaVisionMLP`** [compute]: `L2/clip_mlp.py` (Two-layer fc1+act+fc2 (CLIP-style).)
  - **`MllamaVisionAttention`** [compute]: `L2/clip_attention.py` (Standard MHA without GQA/RoPE; CLIP-style encoder attention.)
  - **`MllamaVisionEncoder`** [wiring]: Wiring stack of encoder layers.
  - **`MllamaTextRMSNorm`** [compute]: `L1/rms_norm.py` (Standard Llama RMSNorm.)
  - **`MllamaTextCrossAttention`** [wiring]: Cross-attention with q_norm/k_norm and per-image cache update; no kb-nano cross-attention L2 matches this pattern.
  - **`MllamaTextSelfAttention`** [compute]: `L2/attention.py` (Standard Llama GQA self-attention with RoPE.)
  - **`MllamaTextMLP`** [compute]: `L2/llama_mlp.py`, `L1/silu_and_mul.py` (SwiGLU gate_proj/up_proj/down_proj.)
  - **`MllamaSelfAttentionDecoderLayer`** [wiring]: Standard self-attn decoder layer wiring.
  - **`MllamaCrossAttentionDecoderLayer`** [wiring]: Wiring with tanh gates around cross-attn/ffn.
  - **`MllamaRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (Standard Llama RoPE.)
  - **`MllamaVisionModel`** [wiring]: Wiring.
  - **`MllamaTextModel`** [wiring]: Wiring.
  - **`MllamaForCausalLM`** [wiring]: Wiring/head.
  - **`MllamaModel`** [wiring]: Wiring.
  - **`MllamaForConditionalGeneration`** [wiring]: Wiring/head.

## mobilebert
- **src**: modeling_mobilebert.py
- **status**: composable
- **rationale**: MobileBERT uses NoNorm (custom non-normalizing affine layer) gated by config, plus trigram_input embedding using F.pad. Bottleneck linears are standard but NoNorm has no kb-nano kernel; falls back to torch primitives.
- **classes**:
  - **`NoNorm`** [wiring]: Affine-only NoNorm: x * weight + bias. No kb-nano equivalent.
  - **`MobileBertEmbeddings`** [compute]: `L1/embedding.py` (Word + position + token_type embeddings, optional trigram via F.pad+Linear, then LayerNorm.)
  - **`MobileBertSelfAttention`** [compute]: `L2/encoder_attention.py` (BERT-style encoder self-attention (Q,K,V Linear with bias, no RoPE, no causal mask).)
  - **`MobileBertSelfOutput`** [wiring]: Linear + (NoNorm or LayerNorm) + residual; calls into possibly-NoNorm path.
  - **`MobileBertAttention`** [wiring]: Wiring: SelfAttention + SelfOutput.
  - **`MobileBertIntermediate`** [compute]: `L1/linear.py` (Dense + activation.)
  - **`OutputBottleneck`** [wiring]: Bottleneck Linear + (NoNorm or LayerNorm) + residual.
  - **`MobileBertOutput`** [wiring]: Wiring: Linear + norm + bottleneck path.
  - **`BottleneckLayer`** [wiring]: Linear + NoNorm/LayerNorm; bottleneck downprojection.
  - **`Bottleneck`** [wiring]: Wiring around BottleneckLayer producing q/k/v/layer_input variants.
  - **`FFNOutput`** [wiring]: Linear + norm + residual.
  - **`FFNLayer`** [wiring]: Wiring intermediate + FFNOutput.
  - **`MobileBertLayer`** [wiring]: Layer wiring: Bottleneck -> Attention -> FFN(s) -> Output.
  - **`MobileBertEncoder`** [wiring]: Wiring stack of MobileBertLayer.
  - **`MobileBertPooler`** [wiring]: Optional Linear+tanh on CLS token.
  - **`MobileBertModel`** [wiring]: Wiring.

## mobilenet_v1
- **src**: modeling_mobilenet_v1.py
- **status**: composable
- **rationale**: Pure CNN: depthwise/pointwise Conv2d + BatchNorm2d + ReLU6 (Conv2d, BatchNorm2d, activation all available in kb-nano L1). The optional TF-style padding is composed via torch primitives; standard PyTorch path.
- **classes**:
  - **`MobileNetV1ConvLayer`** [compute]: `L1/conv2d.py`, `L1/batch_norm2d.py`, `L1/relu.py` (Conv2d + BatchNorm2d + activation (ReLU6 typically, available via L1 ops).)
  - **`MobileNetV1Model`** [wiring]: Wiring: stem + 13 depthwise/pointwise blocks + global avg pool.
  - **`MobileNetV1ForImageClassification`** [wiring]: Wiring: backbone + classifier Linear.

## mobilenet_v2
- **src**: modeling_mobilenet_v2.py
- **status**: composable
- **rationale**: MobileNetV2 inverted residual blocks built from Conv2d+BatchNorm2d+ReLU6; kb-nano covers all L1 ops. Optional ASPP/DeepLabV3 segmentation head also composes from Conv/BN/interpolate (interpolate exists in L1).
- **classes**:
  - **`MobileNetV2ConvLayer`** [compute]: `L1/conv2d.py`, `L1/batch_norm2d.py`, `L1/relu.py` (Conv2d+BN+activation.)
  - **`MobileNetV2InvertedResidual`** [wiring]: Wiring: expand 1x1 + depthwise 3x3 + project 1x1 with optional residual.
  - **`MobileNetV2Stem`** [wiring]: Wiring stem.
  - **`MobileNetV2Model`** [wiring]: Wiring.
  - **`MobileNetV2ForImageClassification`** [wiring]: Wiring/head.
  - **`MobileNetV2DeepLabV3Plus`** [compute]: `L1/interpolate.py` (ASPP head with Conv/BN + interpolate.)
  - **`MobileNetV2ForSemanticSegmentation`** [wiring]: Wiring/head.

## mobilevit
- **src**: modeling_mobilevit.py
- **status**: composable
- **rationale**: MobileViT mixes MobileNetV2 inverted residuals with a standard Transformer (MHA + LayerNorm + MLP). All compute primitives (Conv2d, BN, LayerNorm, dense MHA via DenseAttention, interpolate, fc1+act+fc2) exist in kb-nano L1/L2. Folding/unfolding is plain reshape/transpose.
- **classes**:
  - **`MobileViTConvLayer`** [compute]: `L1/conv2d.py`, `L1/batch_norm2d.py` (Conv2d+BN+activation.)
  - **`MobileViTInvertedResidual`** [wiring]: Wiring inverted residual.
  - **`MobileViTMobileNetLayer`** [wiring]: Wiring stack of inverted residuals.
  - **`MobileViTSelfAttention`** [compute]: `L2/encoder_attention.py`, `L1/dense_attention.py` (Vanilla MHA (Q,K,V Linear + softmax). Maps to encoder self-attention pattern.)
  - **`MobileViTSelfOutput`** [compute]: `L1/linear.py` (Output Linear + dropout.)
  - **`MobileViTAttention`** [wiring]: Wiring: SelfAttention + SelfOutput.
  - **`MobileViTIntermediate`** [compute]: `L1/linear.py` (Linear + activation.)
  - **`MobileViTOutput`** [compute]: `L1/linear.py` (Linear + dropout + residual.)
  - **`MobileViTTransformerLayer`** [wiring]: Wiring pre-LN transformer block.
  - **`MobileViTTransformer`** [wiring]: Wiring stack.
  - **`MobileViTLayer`** [compute]: `L1/interpolate.py` (Wiring: downsample + conv + transformer + folding/unfolding (uses interpolate).)
  - **`MobileViTEncoder`** [wiring]: Wiring.
  - **`MobileViTModel`** [wiring]: Wiring.
  - **`MobileViTForImageClassification`** [wiring]: Wiring/head.
  - **`MobileViTASPPPooling`** [compute]: `L1/global_avg_pool2d.py`, `L1/interpolate.py` (Global avg pool + Conv + interpolate.)
  - **`MobileViTASPP`** [wiring]: Wiring.
  - **`MobileViTDeepLabV3`** [wiring]: Wiring.
  - **`MobileViTForSemanticSegmentation`** [wiring]: Wiring/head.

## mobilevitv2
- **src**: modeling_mobilevitv2.py
- **status**: partial
- **partial_reason**: MobileViTV2LinearSelfAttention is a custom O(N) attention: softmax over single-query channel + element-wise (key * scores) sum + relu(value)*context. No kb-nano kernel matches; pure torch primitives suffice.
- **rationale**: MobileViTV2 introduces LinearSelfAttention (single-query softmax + element-wise broadcast) and uses nn.GroupNorm(num_groups=1) for normalization. No kb-nano kernel implements MobileViTV2 linear separable attention; group_norm exists.
- **classes**:
  - **`MobileViTV2MobileNetLayer`** [compute]: MobileViTV2LinearSelfAttention is a custom O(N) attention: softmax over single-query channel + element-wise (key * scores) sum + relu(value)*context. No kb-nano kernel matches; pure torch primitives s
  - **`MobileViTV2ConvLayer`** [compute]: `L1/conv2d.py`, `L1/batch_norm2d.py` (Conv+BN+act.)
  - **`MobileViTV2InvertedResidual`** [wiring]: Wiring.
  - **`MobileViTV2LinearSelfAttention`** [wiring]: Custom O(N) linear attention; not in kb-nano.
  - **`MobileViTV2FFN`** [wiring]: Wiring two 1x1 ConvLayers as MLP.
  - **`MobileViTV2TransformerLayer`** [compute]: `L1/group_norm.py` (Wiring: GroupNorm(1) + LinearSelfAttention + FFN.)
  - **`MobileViTV2Transformer`** [wiring]: Wiring.
  - **`MobileViTV2Layer`** [wiring]: Wiring.
  - **`MobileViTV2Encoder`** [wiring]: Wiring.
  - **`MobileViTV2Model`** [wiring]: Wiring.
  - **`MobileViTV2ForImageClassification`** [wiring]: Wiring/head.
  - **`MobileViTV2ASPPPooling`** [compute]: `L1/global_avg_pool2d.py`, `L1/interpolate.py` (Standard ASPP pooling.)
  - **`MobileViTV2ASPP`** [wiring]: Wiring.
  - **`MobileViTV2DeepLabV3`** [wiring]: Wiring.
  - **`MobileViTV2ForSemanticSegmentation`** [wiring]: Wiring/head.

## modernbert
- **src**: modular_modernbert.py
- **status**: partial
- **partial_reason**: Encoder self-attention with both RoPE on Q/K and sliding-window masking is not implemented in kb-nano L2; HF uses ALL_ATTENTION_FUNCTIONS path with sliding_window kwarg. kb-nano flash_attn_prefill supports causal sliding window but encoder_attention.py (DenseAttention path) does not wire RoPE + window. GLU MLP without down_proj fusion is also not in kb-nano L2 (closest L2/swiglu_mlp.py uses Linear gate/up/down, not single chunked Wi).
- **rationale**: ModernBERT is encoder-only with RoPE attention + sliding-window attention pattern + GLU MLP. kb-nano L2/encoder_attention.py does not apply RoPE inside the attention call and does not propagate sliding_window. The components exist as separate L1 ops but no L2 wires RoPE-encoder attention with sliding-window mask.
- **classes**:
  - **`ModernBertAttention`** [compute]: Encoder self-attention with both RoPE on Q/K and sliding-window masking is not implemented in kb-nano L2; HF uses ALL_ATTENTION_FUNCTIONS path with sliding_window kwarg. kb-nano flash_attn_prefill sup
  - **`ModernBertEmbeddings`** [compute]: `L1/embedding.py`, `L1/layer_norm.py` (Embedding + LayerNorm + dropout.)
  - **`ModernBertMLP`** [wiring]: GLU: Wi(x) -> chunk -> act(input)*gate -> Wo. Single packed Wi; closest is swiglu_mlp.py but the projection layout is fused-different.
  - **`ModernBertRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (Standard NeoX RoPE.)
  - **`ModernBertEncoderLayer`** [wiring]: Wiring: norm -> attn -> residual -> norm -> mlp -> residual.
  - **`ModernBertModel`** [wiring]: Wiring.
  - **`ModernBertPredictionHead`** [wiring]: Wiring.

## modernbert_decoder
- **src**: modular_modernbert_decoder.py
- **status**: partial
- **partial_reason**: Same GLU MLP issue as modernbert (packed Wi); attention itself maps to LlamaAttention with LayerNorm replacing RMSNorm. kb-nano L2/llama_mlp.py uses gate_proj/up_proj/down_proj layout, not chunked Wi.
- **rationale**: Decoder variant of ModernBert with causal RoPE attention + sliding window + LayerNorm + GLU MLP. The split-QKV attention (vs packed Wqkv in encoder) makes it closer to LlamaAttention, but with LayerNorm and GLU MLP. LayerNorm with bias and the GLU pattern (Wi->chunk->act*gate->Wo) don't match L2/llama_mlp.py directly.
- **classes**:
  - **`ModernBertDecoderLayer`** [compute]: Same GLU MLP issue as modernbert (packed Wi); attention itself maps to LlamaAttention with LayerNorm replacing RMSNorm. kb-nano L2/llama_mlp.py uses gate_proj/up_proj/down_proj layout, not chunked Wi.
  - **`ModernBertDecoderEmbeddings`** [compute]: `L1/embedding.py`, `L1/layer_norm.py` (Inherited from ModernBertEmbeddings.)
  - **`ModernBertDecoderMLP`** [wiring]: Same packed-Wi GLU MLP as ModernBertMLP.
  - **`ModernBertDecoderRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (Standard NeoX RoPE.)
  - **`ModernBertDecoderAttention`** [compute]: `L2/attention.py` (Causal MHA with separate q/k/v_proj + RoPE + sliding window. Compute matches LlamaAttention except output is Wo.)
  - **`ModernBertDecoderModel`** [wiring]: Wiring.

## modernvbert
- **src**: modular_modernvbert.py
- **status**: partial
- **rationale**: ModernVBert composes a SigLIP vision tower with a ModernBert text encoder via AutoModel.from_config. The ModernBertConnector does pixel-shuffle + Linear, but ModernBert (text) itself is partial (sliding-window+RoPE encoder attention with no kb-nano L2). The AutoModel-driven composition mirrors AutoBackbone — no kb-nano equivalent for the load-bearing path.
- **classes**:
  - **`ModernVBertModel`** [compute]: no kb-nano kernel — ModernVBert composes a SigLIP vision tower with a ModernBert text encoder via AutoModel.from_config. The ModernBertConnector does pixel-shuffle + Linear, but ModernBert (text) itself is partial (slidi
  - **`ModernVBertConnector`** [compute]: `L1/linear.py` (Pixel-shuffle (pure tensor op) + modality_projection Linear.)
  - **`ModernVBertForMaskedLM`** [wiring]: Wiring/head.
  - **`ModernVBertForSequenceClassification`** [wiring]: Wiring/head.
  - **`ModernVBertForTokenClassification`** [wiring]: Wiring/head.

## moonshine
- **src**: modular_moonshine.py
- **status**: partial
- **partial_reason**: MoonshineAttention requires RoPE before the SDPA call (encoder-decoder path with cross-attention), and head_dim padding to a multiple. kb-nano whisper_attention.py has no RoPE pre-application; llama-family attention.py is causal-only. Pure-torch primitives compose it but no L2 matches.
- **rationale**: Moonshine is a Whisper-like encoder-decoder with RoPE applied inside Q/K projections. kb-nano L2/whisper_attention.py does NOT apply RoPE; it serves vanilla Whisper-style attention. So the attention compute does not map to an existing L2.
- **classes**:
  - **`MoonshineAttention`** [compute]: no kb-nano kernel — MoonshineAttention requires RoPE before the SDPA call (encoder-decoder path with cross-attention), and head_dim padding to a multiple. kb-nano whisper_attention.py has no RoPE pre-application; llama-f
  - **`MoonshineEncoderMLP`** [compute]: `L2/whisper_mlp.py` (Two-layer fc1+act+fc2 MLP.)
  - **`MoonshineDecoderMLP`** [wiring]: Single fc1 producing 2*intermediate, chunk into gate+main, act(gate)*main, fc2. SwiGLU-like but layout differs from L2/llama_mlp.py (single fc1 vs gate+up split).
  - **`MoonshineRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (Standard NeoX RoPE.)
  - **`MoonshineEncoderLayer`** [wiring]: Wiring.
  - **`MoonshineDecoderLayer`** [wiring]: Wiring with self_attn + encoder_attn + mlp.
  - **`MoonshineEncoder`** [wiring]: Wiring stack.
  - **`MoonshineDecoder`** [wiring]: Wiring stack.
  - **`MoonshineModel`** [wiring]: Wiring.
  - **`MoonshineForConditionalGeneration`** [wiring]: Wiring/head.

## moonshine_streaming
- **src**: modular_moonshine_streaming.py
- **status**: partial
- **partial_reason**: MoonshineStreamingFrameCMVN, MoonshineStreamingAsinhCompression, MoonshineStreamingCausalConv1d (with mask propagation), and MoonshineStreamingLayerNorm (unit-offset gamma) all compose via torch ops; no kb-nano L1 implements these specifically. Sliding-window mask via flex_attention also unsupported.
- **rationale**: Moonshine-streaming adds CMVN normalization, asinh compression, causal Conv1d with masked padding, and a unit-offset LayerNorm. None of these have kb-nano L1 wrappers; they compose from torch primitives. Inherits Moonshine attention which is also partial.
- **classes**:
  - **`MoonshineStreamingEncoderAttention`** [compute]: no kb-nano kernel — MoonshineStreamingFrameCMVN, MoonshineStreamingAsinhCompression, MoonshineStreamingCausalConv1d (with mask propagation), and MoonshineStreamingLayerNorm (unit-offset gamma) all compose via torch ops; 
  - **`MoonshineStreamingFrameCMVN`** [wiring]: Per-frame mean/RMS normalization. Pure torch.
  - **`MoonshineStreamingAsinhCompression`** [wiring]: asinh(exp(log_k)*x); pure torch.
  - **`MoonshineStreamingCausalConv1d`** [compute]: `L1/conv1d.py` (Conv1d with left-pad and mask-aware downsampling; base Conv1d covered.)
  - **`MoonshineStreamingLayerNorm`** [compute]: `L1/layer_norm.py` (LayerNorm without affine + parameter gamma with unit offset; LayerNorm covered, gamma offset is wiring.)
  - **`MoonshineStreamingEncoderMLP`** [compute]: `L2/whisper_mlp.py` (Inherited; same two-layer MLP.)
  - **`MoonshineStreamingEncoderLayer`** [wiring]: Wiring.
  - **`MoonshineStreamingEncoderEmbedder`** [wiring]: Wiring CMVN + compression + Conv stack.
  - **`MoonshineStreamingEncoder`** [wiring]: Wiring.
  - **`MoonshineStreamingDecoder`** [wiring]: Wiring (same Moonshine decoder).
  - **`MoonshineStreamingModel`** [wiring]: Wiring.
  - **`MoonshineStreamingForConditionalGeneration`** [wiring]: Wiring/head.

## moshi
- **src**: modeling_moshi.py
- **status**: partial
- **partial_reason**: MoshiFlexibleLinear (per-codebook 3D weight bank with torch.index_select + batched matmul) is a custom op without kb-nano equivalent; closest is moe_grouped_gemm but the routing is different (codebook index, not top-k). The depth decoder relies on this throughout.
- **rationale**: Moshi (audio LLM) introduces MoshiFlexibleLinear (per-codebook stacked weights with index_select+matmul) and MoshiGatingMLP variant. The flexible-linear path has no kb-nano kernel; standard MoshiAttention also wraps MoshiLinear which can be flexible.
- **classes**:
  - **`MoshiAttention`** [compute]: no kb-nano kernel — MoshiFlexibleLinear (per-codebook 3D weight bank with torch.index_select + batched matmul) is a custom op without kb-nano equivalent; closest is moe_grouped_gemm but the routing is different (codebook
  - **`MoshiRMSNorm`** [compute]: `L1/rms_norm.py` (Standard RMSNorm with float promotion.)
  - **`MoshiFlexibleLinear`** [wiring]: Per-codebook weight bank + index_select + batched matmul. No kb-nano kernel.
  - **`MoshiLinear`** [compute]: `L1/linear.py` (Wraps either nn.Linear or MoshiFlexibleLinear; partial when use_flexible_linear=True.)
  - **`MoshiRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (Standard NeoX RoPE.)
  - **`MoshiGatingMLP`** [wiring]: fc1 -> chunk -> act(gate)*main -> fc2 using either nn.Linear or MoshiFlexibleLinear. Single packed fc1 pattern not in kb-nano L2.
  - **`MoshiDecoderLayer`** [wiring]: Wiring.
  - **`MoshiDepthDecoder`** [wiring]: Wiring/head; uses MoshiFlexibleLinear pervasively.
  - **`MoshiModel`** [wiring]: Wiring.
  - **`MoshiForCausalLM`** [wiring]: Wiring/head.
  - **`MoshiForConditionalGeneration`** [wiring]: Wiring with audio_encoder dependency.

## mpnet
- **src**: modeling_mpnet.py
- **status**: partial
- **partial_reason**: Relative position bias added to attention scores: kb-nano flash_attn/dense_attention have no additive-bias parameter, and t5_attention.py is T5-specific (not the same bucket function used by MPNet which has its own compute_position_bias). Addition is pure-torch but no L2 wires it into encoder MHA.
- **rationale**: MPNet uses BERT-like encoder attention but adds T5-style learned relative position bias (added to attention scores). kb-nano encoder_attention.py does not support an additive position bias; T5-style bias lives in L2/t5_attention.py with a different relative-bucket scheme.
- **classes**:
  - **`MPNetSelfAttention`** [compute]: no kb-nano kernel — Relative position bias added to attention scores: kb-nano flash_attn/dense_attention have no additive-bias parameter, and t5_attention.py is T5-specific (not the same bucket function used by MPNet whi
  - **`MPNetEmbeddings`** [compute]: `L1/embedding.py`, `L1/layer_norm.py` (Token + position embedding + LayerNorm + dropout.)
  - **`MPNetAttention`** [wiring]: Wiring: SelfAttention + LayerNorm + residual.
  - **`MPNetIntermediate`** [compute]: `L1/linear.py` (Linear + activation.)
  - **`MPNetOutput`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (Linear + LayerNorm + residual.)
  - **`MPNetLayer`** [wiring]: Wiring.
  - **`MPNetEncoder`** [wiring]: Wiring + relative_attention_bias Embedding + compute_position_bias method.
  - **`MPNetPooler`** [wiring]: CLS Linear + tanh.
  - **`MPNetModel`** [wiring]: Wiring.
  - **`MPNetLMHead`** [wiring]: Wiring.
  - **`MPNetClassificationHead`** [wiring]: Wiring.

## mpt
- **src**: modeling_mpt.py
- **status**: partial
- **partial_reason**: build_mpt_alibi_tensor produces an additive [n_heads, q_len, k_len] bias added to attention scores. kb-nano flash kernels lack ALiBi support; torch.matmul-based softmax is the only path. Per consistency reminder: ALiBi-as-additive-bias is partial.
- **rationale**: MPT uses ALiBi additive position bias on attention scores. kb-nano flash_attn_* and dense_attention do not support alibi_slopes; HF computes the bias as torch tensor and adds it manually.
- **classes**:
  - **`MptAttention`** [compute]: no kb-nano kernel — build_mpt_alibi_tensor produces an additive [n_heads, q_len, k_len] bias added to attention scores. kb-nano flash kernels lack ALiBi support; torch.matmul-based softmax is the only path. Per consisten
  - **`MptMLP`** [compute]: `L2/whisper_mlp.py` (Two-layer up_proj+GELU+down_proj (no gate). Maps to whisper-style MLP.)
  - **`MptBlock`** [wiring]: Wiring: LayerNorm(no bias) -> attn -> residual -> LayerNorm -> ffn.
  - **`MptModel`** [wiring]: Wiring; precomputes ALiBi tensor.
  - **`MptForCausalLM`** [wiring]: Wiring/head.

## mra
- **src**: modeling_mra.py
- **status**: unsupported
- **unsupported_reason**: Sparse multi-resolution analysis attention uses MraSampledDenseMatMul / MraSparseDenseMatMul / MraReduceSum bound to a CUDA extension (load_cuda_kernels). kb-nano has no sparse-block attention kernel; the closest sparse_mm.py is a generic sparse matmul, not MRA's structured block-sparse pattern.
- **rationale**: MRA uses a custom CUDA kernel (kernels-community 'mra' op) loaded via load_cuda_kernels for sparse multi-resolution attention. The mra2_attention path calls into the loaded CUDA module for sampled-dense matmul; no kb-nano kernel implements this scheme.
- **classes**:
  - **`MraSelfAttention`** [compute]: no kb-nano kernel — Sparse multi-resolution analysis attention uses MraSampledDenseMatMul / MraSparseDenseMatMul / MraReduceSum bound to a CUDA extension (load_cuda_kernels). kb-nano has no sparse-block attention kernel;
  - **`MraEmbeddings`** [compute]: `L1/embedding.py`, `L1/layer_norm.py` (Standard BERT embeddings.)
  - **`MraSelfOutput`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (Standard Linear + LayerNorm + residual.)
  - **`MraAttention`** [wiring]: Wiring.
  - **`MraIntermediate`** [compute]: `L1/linear.py` (Linear + activation.)
  - **`MraOutput`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (Linear + LayerNorm + residual.)
  - **`MraLayer`** [wiring]: Wiring.
  - **`MraEncoder`** [wiring]: Wiring.
  - **`MraModel`** [wiring]: Wiring.

## mt5
- **src**: modeling_mt5.py
- **status**: partial
- **partial_reason**: T5 decoder cross-attention with relative-position bias has no kb-nano L2 (kb-nano whisper_attention.py is BART-style without T5 relative bias; t5_attention.py is encoder self-attn). Per consistency reminder: T5 cross-attn unsupported in kb-nano.
- **rationale**: MT5 is multilingual T5 (encoder-decoder with relative position bias and gated activations). kb-nano L4/t5_encoder.py covers T5 encoder via L3/t5_block.py + L2/t5_attention.py; the MT5 decoder with cross-attention has no kb-nano L2/L3.
- **classes**:
  - **`MT5LayerSelfAttention`** [compute]: T5 decoder cross-attention with relative-position bias has no kb-nano L2 (kb-nano whisper_attention.py is BART-style without T5 relative bias; t5_attention.py is encoder self-attn). Per consistency re
  - **`MT5LayerNorm`** [compute]: `L1/t5_layer_norm.py` (RMS-style T5 layer norm (no centering).)
  - **`MT5DenseActDense`** [compute]: `L2/t5_dense.py` (wi -> act -> wo (no gating).)
  - **`MT5DenseGatedActDense`** [compute]: `L2/t5_dense.py` (wi_0 / wi_1 -> act(gate)*main -> wo. Gated SiLU/GeLU; T5-style.)
  - **`MT5LayerFF`** [wiring]: Wiring norm + DenseAct(Gated)Dense + residual.
  - **`MT5Attention`** [compute]: `L2/t5_attention.py` (Self-attention with optional relative position bias. Encoder path matches kb-nano L2/t5_attention.py; cross-attention path (no rel bias) does not match kb-nano (whisper_attention is BART-style).)
  - **`MT5LayerCrossAttention`** [wiring]: Wiring norm + MT5Attention(no rel bias) + residual; no kb-nano T5 cross-attn.
  - **`MT5Block`** [compute]: `L3/t5_block.py` (Encoder block matches L3/t5_block.py; decoder block (with cross-attention) is not in kb-nano L3.)
  - **`MT5Stack`** [wiring]: Wiring; encoder maps to L4/t5_encoder.py.
  - **`MT5Model`** [wiring]: Wiring.
  - **`MT5ForConditionalGeneration`** [wiring]: Wiring/head.
  - **`MT5EncoderModel`** [wiring]: Wiring; uses encoder-only path (covered by L4/t5_encoder.py).

## musicgen
- **src**: modeling_musicgen.py
- **status**: partial
- **partial_reason**: Per-codebook embedding tables summed at the input + sinusoidal positional embeddings + delay-pattern in autoregressive generation. The attention itself (self + cross) maps to whisper_attention.py, but the codebook input layer and audio_encoder dependency are model-level patterns kb-nano L4 does not provide.
- **rationale**: Musicgen is a BART/Whisper-style encoder-decoder text-to-music LLM with sinusoidal positional embeddings and cross-attention. The decoder self+cross-attention pattern matches L2/whisper_attention.py family, BUT MusicGen wraps it around T5-encoder-conditioning + per-codebook embedding sums that have no kb-nano analog at the model level.
- **classes**:
  - **`MusicgenDecoderLayer`** [compute]: no kb-nano kernel — Per-codebook embedding tables summed at the input + sinusoidal positional embeddings + delay-pattern in autoregressive generation. The attention itself (self + cross) maps to whisper_attention.py, but
  - **`MusicgenSinusoidalPositionalEmbedding`** [compute]: `L1/sinusoidal_embed.py` (Sinusoidal positional embedding with index_select.)
  - **`MusicgenAttention`** [compute]: `L2/whisper_attention.py` (MHA with optional cross-attention via key_value_states. Matches whisper_attention.py family (self/decoder self/cross variants).)
  - **`MusicgenDecoder`** [wiring]: Wiring with per-codebook embedding sum + sinusoidal pos.
  - **`MusicgenModel`** [wiring]: Wiring.
  - **`MusicgenForCausalLM`** [wiring]: Wiring/head with multi-codebook lm_heads.
  - **`MusicgenForConditionalGeneration`** [wiring]: Wiring; depends on text_encoder + audio_encoder.

## musicgen_melody
- **src**: modeling_musicgen_melody.py
- **status**: partial
- **partial_reason**: Same as musicgen — per-codebook embedding sum and conditional generation wiring not in kb-nano. Attention itself composes from whisper_attention.py.
- **rationale**: Same architecture family as Musicgen (BART-style decoder + per-codebook embeddings + sinusoidal pos), with melody conditioning prepended to inputs instead of cross-attention. Attention maps to L2/whisper_attention.py but model-level wiring (multi-codebook + melody conditioning) has no kb-nano L4.
- **classes**:
  - **`MusicgenMelodyDecoderLayer`** [compute]: no kb-nano kernel — Same as musicgen — per-codebook embedding sum and conditional generation wiring not in kb-nano. Attention itself composes from whisper_attention.py.
  - **`MusicgenMelodySinusoidalPositionalEmbedding`** [compute]: `L1/sinusoidal_embed.py` (Sinusoidal positional embedding.)
  - **`MusicgenMelodyAttention`** [compute]: `L2/whisper_attention.py` (MHA (self only here; melody is prepended to inputs).)
  - **`MusicgenMelodyDecoder`** [wiring]: Wiring.
  - **`MusicgenMelodyModel`** [wiring]: Wiring.
  - **`MusicgenMelodyForCausalLM`** [wiring]: Wiring/head.
  - **`MusicgenMelodyForConditionalGeneration`** [wiring]: Wiring.

## mvp
- **src**: modeling_mvp.py
- **status**: partial
- **partial_reason**: MvpAttention supports an attn_prompt argument that prepends learned prompt tensors to key_states/value_states within the attention call. kb-nano whisper_attention.py has no prompt-prepend hook. Pure-torch composition possible but no L2 covers it.
- **rationale**: MVP is BART-style encoder-decoder with optional learned prompt tokens prepended to K/V inside attention. The base attention maps to L2/whisper_attention.py family, but the attn_prompt path (prepend prompts to K/V before SDPA) is not modeled in any kb-nano L2.
- **classes**:
  - **`MvpAttention`** [compute]: no kb-nano kernel — MvpAttention supports an attn_prompt argument that prepends learned prompt tensors to key_states/value_states within the attention call. kb-nano whisper_attention.py has no prompt-prepend hook. Pure-t
  - **`MvpLearnedPositionalEmbedding`** [compute]: `L1/embedding.py` (Learned positional embedding.)
  - **`MvpEncoderLayer`** [wiring]: Wiring.
  - **`MvpDecoderLayer`** [wiring]: Wiring with cross-attn.
  - **`MvpClassificationHead`** [wiring]: Wiring.
  - **`MvpPrompt`** [wiring]: Generates per-layer K/V prompts via small MLP; bespoke.
  - **`MvpEncoder`** [wiring]: Wiring.
  - **`MvpDecoder`** [wiring]: Wiring.
  - **`MvpModel`** [wiring]: Wiring.
  - **`MvpForConditionalGeneration`** [wiring]: Wiring/head.
  - **`MvpForCausalLM`** [wiring]: Wiring/head.

## nanochat
- **src**: modular_nanochat.py
- **status**: partial
- **partial_reason**: NanoChatRMSNorm = Llama4TextL2Norm = pure F.normalize(p=2) without learned weight; kb-nano L1/l2_norm.py covers F.normalize but is used in RWKV7 context, not as a transformer pre-norm. Custom rotate_half((x2, -x1) order) means the standard kb-nano rotary_emb kernel produces a different result. RoPE-then-Norm ordering on q/k is also inverted from typical Llama. No kb-nano L2 fuses CLIP-MLP + L2-norm + custom-rotate Llama-attn.
- **rationale**: NanoChat uses Llama4TextL2Norm (L2-normalize without learnable weight) as its norm, a CLIP-style 2-layer MLP (fc1->act->fc2 not SwiGLU), and a non-standard rotate_half implementation (x2, -x1) different from the kb-nano L1/rotary_emb.py convention. Each individual op exists but the combination diverges from any kb-nano L2.
- **classes**:
  - **`NanoChatAttention`** [compute]: NanoChatRMSNorm = Llama4TextL2Norm = pure F.normalize(p=2) without learned weight; kb-nano L1/l2_norm.py covers F.normalize but is used in RWKV7 context, not as a transformer pre-norm. Custom rotate_h
  - **`NanoChatRMSNorm`** [compute]: `L1/l2_norm.py` (Pure F.normalize(p=2); no learned weight (unlike standard RMSNorm).)
  - **`NanoChatRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (Standard NeoX RoPE buffer/forward.)
  - **`NanoChatMLP`** [compute]: `L2/clip_mlp.py` (Two-layer fc1+act+fc2 MLP (CLIP-style).)
  - **`NanoChatDecoderLayer`** [wiring]: Wiring.
  - **`NanoChatModel`** [wiring]: Wiring.
  - **`NanoChatForCausalLM`** [wiring]: Wiring/head.

## nemotron
- **src**: modeling_nemotron.py
- **status**: partial
- **partial_reason**: NemotronLayerNorm1P (F.layer_norm with weight+1 reparam) has no kb-nano kernel. Partial-RoPE (rotates only first int(head_dim*partial_rotary_factor) channels) not implemented in L1/rotary_emb.py which assumes full head_dim rotation. squared_relu MLP without gating uses L1/squared_relu.py, but no fused L2.
- **rationale**: Nemotron uses NemotronLayerNorm1P (LayerNorm with weight+1 trick), partial RoPE (rotates only first rot_dim of head_dim and concats q_pass), squared_relu activation, and a non-gated 2-layer MLP. LayerNorm1P has no kb-nano kernel; partial RoPE not in kb-nano L1/rotary_emb.py.
- **classes**:
  - **`NemotronAttention`** [compute]: NemotronLayerNorm1P (F.layer_norm with weight+1 reparam) has no kb-nano kernel. Partial-RoPE (rotates only first int(head_dim*partial_rotary_factor) channels) not implemented in L1/rotary_emb.py which
  - **`NemotronLayerNorm1P`** [wiring]: LayerNorm where weight is implicitly weight+1; no kb-nano variant.
  - **`NemotronRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (Standard inv_freq RoPE buffer.)
  - **`NemotronMLP`** [compute]: `L1/squared_relu.py` (up_proj -> squared_relu -> down_proj (non-gated). Squared ReLU is the typical activation; L1/squared_relu.py is the non-fused op.)
  - **`NemotronDecoderLayer`** [wiring]: Wiring with LayerNorm1P pre/post.
  - **`NemotronModel`** [wiring]: Wiring.
  - **`NemotronForCausalLM`** [wiring]: Wiring/head.

## nemotron_h
- **src**: modular_nemotron_h.py
- **status**: partial
- **partial_reason**: NemotronHExperts uses non-gated up_proj+act+down_proj (not SwiGLU); kb-nano fused_experts/L2 mixtral_moe.py and shared_expert_moe.py both assume gated experts. NemotronHMoE adds optional fc1_latent_proj/fc2_latent_proj wrapping the experts. No kb-nano L2 implements non-gated experts with latent projection.
- **rationale**: Nemotron-H is a hybrid Mamba2 + attention + MoE LLM. Mamba2 mixer maps to L2/mamba2_mixer.py; attention inherits Jamba (L2/jamba_attention.py). The MoE is non-gated experts with optional latent projection (different from standard SwiGLU MoE) and uses shared_experts pattern; no kb-nano L2 covers non-gated MoE with latent projection.
- **classes**:
  - **`NemotronHBlock`** [compute]: no kb-nano kernel — NemotronHExperts uses non-gated up_proj+act+down_proj (not SwiGLU); kb-nano fused_experts/L2 mixtral_moe.py and shared_expert_moe.py both assume gated experts. NemotronHMoE adds optional fc1_latent_pr
  - **`NemotronHMamba2Mixer`** [compute]: `L2/mamba2_mixer.py` (Mamba2 mixer (Conv1d + in_proj + ssm + RMSNormGated + out_proj). Maps to mamba2_mixer.py.)
  - **`NemotronHRMSNorm`** [compute]: `L1/rms_norm.py` (Standard Llama RMSNorm.)
  - **`NemotronHMLP`** [wiring]: Non-gated up_proj+act+down_proj MLP.
  - **`NemotronHExperts`** [wiring]: Non-gated MoE experts (no gate_proj). No kb-nano L2 implements this.
  - **`NemotronHMoE`** [wiring]: Custom MoE: optional latent projection + shared_experts + non-gated experts. No kb-nano L2.
  - **`NemotronHTopkRouter`** [compute]: `L1/topk_softmax.py` (DeepseekV3-style topk router.)
  - **`NemotronHAttention`** [compute]: `L2/jamba_attention.py` (Inherits Jamba attention (no RoPE, GQA).)
  - **`NemotronHModel`** [wiring]: Wiring.
  - **`NemotronHForCausalLM`** [wiring]: Wiring/head.

## nllb_moe
- **src**: modeling_nllb_moe.py
- **status**: partial
- **partial_reason**: NllbMoeTop2Router implements fairseq-style capacity-based Top-2 routing (cumsum < capacity, batch-prioritized, gumbel sampling) — semantically different from kb-nano top-k routers (L1/topk_softmax.py, grouped_topk.py). Experts stored as ModuleDict (not 3D weight bank) so moe_grouped_gemm.py / fused_experts.py do not apply. The gating MLP is non-gated DenseActDense (fc1+act+fc2). Plus BART-style attention with cross-attention which is partially covered by whisper_attention.py.
- **rationale**: NLLB-MoE is a BART-style encoder-decoder with a custom Top2Router (capacity-based with batch-prioritized routing, token dropping, normalize-before-drop policies). The expert dispatch with capacity is not in any kb-nano MoE L2; experts themselves are per-expert nn.ModuleDict not stacked tensors.
- **classes**:
  - **`NllbMoeEncoderLayer`** [compute]: no kb-nano kernel — NllbMoeTop2Router implements fairseq-style capacity-based Top-2 routing (cumsum < capacity, batch-prioritized, gumbel sampling) — semantically different from kb-nano top-k routers (L1/topk_softmax.py,
  - **`NllbMoeScaledWordEmbedding`** [compute]: `L1/embedding.py` (Embedding scaled by sqrt(d_model).)
  - **`NllbMoeSinusoidalPositionalEmbedding`** [compute]: `L1/sinusoidal_embed.py` (Sinusoidal positional embedding.)
  - **`NllbMoeTop2Router`** [wiring]: Fairseq-style Top-2 capacity router. No kb-nano equivalent.
  - **`NllbMoeDenseActDense`** [wiring]: Non-gated fc1+act+dropout+fc2 expert (per-expert).
  - **`NllbMoeExperts`** [wiring]: Per-expert ModuleDict + manual scatter dispatch. No grouped-GEMM fast path possible.
  - **`NllbMoeSparseMLP`** [wiring]: Wiring: Top2Router + per-expert dispatch.
  - **`NllbMoeAttention`** [compute]: `L2/whisper_attention.py` (BART-style MHA (self/cross). Maps to whisper_attention family.)
  - **`NllbMoeDecoderLayer`** [wiring]: Wiring with self_attn + cross_attn + (sparse|dense) ffn.
  - **`NllbMoeEncoder`** [wiring]: Wiring.
  - **`NllbMoeDecoder`** [wiring]: Wiring.
  - **`NllbMoeModel`** [wiring]: Wiring.
  - **`NllbMoeForConditionalGeneration`** [wiring]: Wiring/head.
