## dinat
- **src**: modeling_dinat.py
- **status**: unsupported
- **unsupported_reason**: Uses external `natten.functional.natten2dqkrpb` and `natten2dav` CUDA kernels for sliding/dilated 2D neighborhood attention. kb-nano has no neighborhood attention primitive (closest is dense SDPA, which has different semantics). Requires natten-style fused kernel.
- **rationale**: NeighborhoodAttention requires natten library kernels (natten2dqkrpb, natten2dav); no kb-nano equivalent.
- **classes**:
  - **`NeighborhoodAttention`** [compute]: Uses external `natten.functional.natten2dqkrpb` and `natten2dav` CUDA kernels for sliding/dilated 2D neighborhood attention. kb-nano has no neighborhood attention primitive (closest is dense SDPA, whi
  - **`DinatEmbeddings`** [wiring]: wiring: token + pos embed sum
  - **`DinatPatchEmbeddings`** [compute]: `L1/conv2d.py` (Conv2d patch projection)
  - **`DinatDownsampler`** [compute]: `L1/conv2d.py`, `L1/layer_norm.py` (Conv2d strided downsample + LayerNorm)
  - **`DinatDropPath`** [wiring]: stochastic depth (training-only); no-op at inference
  - **`NeighborhoodAttentionOutput`** [compute]: `L1/linear.py` (Linear + dropout)
  - **`NeighborhoodAttentionModule`** [wiring]: wiring around NeighborhoodAttention + output
  - **`DinatIntermediate`** [compute]: `L1/linear.py`, `L1/gelu.py` (Linear + GELU (act configurable))
  - **`DinatOutput`** [compute]: `L1/linear.py` (Linear + dropout)
  - **`DinatLayer`** [wiring]: wiring decoder block
  - **`DinatStage`** [wiring]: wiring stage
  - **`DinatEncoder`** [wiring]: wiring encoder
  - **`DinatModel`** [wiring]: wiring full model
  - **`DinatForImageClassification`** [wiring]: wiring classifier head
  - **`DinatBackbone`** [wiring]: wiring backbone interface

## dinov2
- **src**: modeling_dinov2.py
- **status**: composable
- **rationale**: Standard ViT-style encoder with separate Q/K/V SDPA, GELU MLP, optional fused-input SwiGLU FFN; all primitives exist in kb-nano.
- **classes**:
  - **`Dinov2Embeddings`** [wiring]: wiring: CLS + patch + pos embed
  - **`Dinov2PatchEmbeddings`** [compute]: `L1/conv2d.py` (Conv2d patch projection)
  - **`Dinov2SelfAttention`** [compute]: `L2/encoder_attention.py` (separate Q/K/V Linear + non-causal SDPA matches EncoderSelfAttention)
  - **`Dinov2SelfOutput`** [compute]: `L1/linear.py` (Linear out + dropout)
  - **`Dinov2Attention`** [wiring]: wiring SelfAttention + SelfOutput
  - **`Dinov2LayerScale`** [wiring]: scalar param multiply (elementwise); covered by tensor_ops
  - **`Dinov2DropPath`** [wiring]: stochastic depth (training-only)
  - **`Dinov2MLP`** [compute]: `L2/encoder_mlp.py` (fc1 -> activation -> fc2 (GELU); matches EncoderIntermediate+Output style)
  - **`Dinov2SwiGLUFFN`** [compute]: `L1/silu_and_mul.py`, `L1/linear.py` (fused weights_in into 2 chunks then SiLU(x1)*x2 -> weights_out; canonical SwiGLU pattern; per guideline 3 SwiGLU MLPs use silu_and_mul, not bare silu.)
  - **`Dinov2Layer`** [wiring]: wiring transformer block
  - **`Dinov2Encoder`** [wiring]: wiring layer stack
  - **`Dinov2Model`** [wiring]: wiring model
  - **`Dinov2ForImageClassification`** [wiring]: wiring classifier head
  - **`Dinov2Backbone`** [wiring]: wiring backbone

## dinov2_with_registers
- **src**: modular_dinov2_with_registers.py
- **status**: composable
- **rationale**: Same as Dinov2 but with register tokens prepended in embeddings; all leaf compute classes still map to encoder_attention + encoder_mlp/SwiGLU primitives.
- **classes**:
  - **`Dinov2WithRegistersPatchEmbeddings`** [compute]: `L1/conv2d.py` (Conv2d patch projection)
  - **`Dinov2WithRegistersEmbeddings`** [wiring]: wiring: CLS + register tokens + patch + pos embed
  - **`Dinov2WithRegistersSelfAttention`** [compute]: `L2/encoder_attention.py` (separate Q/K/V + SDPA)
  - **`Dinov2WithRegistersSelfOutput`** [compute]: `L1/linear.py` (Linear + dropout)
  - **`Dinov2WithRegistersAttention`** [wiring]: wiring
  - **`Dinov2WithRegistersLayerScale`** [wiring]: scalar gate multiply
  - **`Dinov2WithRegistersDropPath`** [wiring]: stochastic depth
  - **`Dinov2WithRegistersMLP`** [compute]: `L2/encoder_mlp.py` (fc1 + GELU + fc2)
  - **`Dinov2WithRegistersSwiGLUFFN`** [compute]: `L1/silu_and_mul.py`, `L1/linear.py` (fused-in SwiGLU; per guideline 3 SwiGLU MLPs use silu_and_mul, not bare silu.)
  - **`Dinov2WithRegistersLayer`** [wiring]: wiring block
  - **`Dinov2WithRegistersEncoder`** [wiring]: wiring stack
  - **`Dinov2WithRegistersModel`** [wiring]: wiring model
  - **`Dinov2WithRegistersForImageClassification`** [wiring]: wiring head
  - **`Dinov2WithRegistersBackbone`** [wiring]: wiring backbone

## dinov3_convnext
- **src**: modeling_dinov3_convnext.py
- **status**: composable
- **rationale**: ConvNeXt-style hierarchical CNN: Conv2d (depthwise + pointwise), LayerNorm, GELU. All primitives exist in kb-nano L1.
- **classes**:
  - **`DINOv3ConvNextDropPath`** [wiring]: stochastic depth
  - **`DINOv3ConvNextLayerNorm`** [compute]: `L1/layer_norm.py` (channels-first LayerNorm; reshape + layer_norm)
  - **`DINOv3ConvNextLayer`** [compute]: `L1/conv2d.py`, `L1/layer_norm.py`, `L1/linear.py`, `L1/gelu.py` (depthwise Conv2d + LayerNorm + Linear + GELU + Linear (ConvNeXt block))
  - **`DINOv3ConvNextStage`** [wiring]: wiring: downsample + N layers
  - **`DINOv3ConvNextEncoder`** [wiring]: wiring stage stack
  - **`DINOv3ConvNextModel`** [wiring]: wiring model
  - **`DINOv3ConvNextBackbone`** [wiring]: wiring backbone

## dinov3_vit
- **src**: modeling_dinov3_vit.py
- **status**: kb_nano_l4
- **rationale**: kb-nano L4/dinov3.py implements DINOv3 ViT (7B/16) with EvaBlock, dinov3_rope, layer-scale, SwiGLU MLP, register tokens.
- **classes**:
  - **`DINOv3ViTEmbeddings`** [wiring]: wiring: patch + CLS + register tokens (L4 handles)
  - **`DINOv3ViTRopePositionEmbedding`** [compute]: `L1/dinov3_rope.py` (DINOv3 2D RoPE matches kb-nano L1/dinov3_rope.py)
  - **`DINOv3ViTAttention`** [compute]: `L2/eva_attention.py` (separate Q/K/V + 2D RoPE + SDPA matches EvaAttention)
  - **`DINOv3ViTLayerScale`** [wiring]: scalar gate multiply (in EvaBlock)
  - **`DINOv3ViTDropPath`** [wiring]: stochastic depth
  - **`DINOv3ViTMLP`** [compute]: `L1/linear.py`, `L1/gelu.py` (up_proj + act + down_proj (non-gated))
  - **`DINOv3ViTGatedMLP`** [compute]: `L2/swiglu_mlp.py` (gate_proj + up_proj + SiLU-mul + down_proj matches SwiGLUMlp)
  - **`DINOv3ViTLayer`** [compute]: `L3/eva_block.py` (pre-norm + attn + layer-scale + droppath + MLP matches EvaBlock)
  - **`DINOv3ViTEncoder`** [wiring]: wiring layer stack (L4 handles)
  - **`DINOv3ViTModel`** [wiring]: wiring model (L4 handles)
  - **`DINOv3ViTBackbone`** [wiring]: wiring backbone

## distilbert
- **src**: modeling_distilbert.py
- **status**: composable
- **rationale**: BERT-shaped encoder with separate Q/K/V Linear + non-causal SDPA + 2-layer FFN with configurable activation; covered by encoder_attention + encoder_mlp.
- **classes**:
  - **`Embeddings`** [wiring]: wiring: word + pos sinusoidal embed
  - **`DistilBertSelfAttention`** [compute]: `L2/encoder_attention.py` (separate Q/K/V + non-causal SDPA + out_lin)
  - **`FFN`** [compute]: `L2/encoder_mlp.py` (lin1 + activation + lin2 (GELU/relu))
  - **`TransformerBlock`** [wiring]: wiring block
  - **`Transformer`** [wiring]: wiring encoder stack
  - **`DistilBertModel`** [wiring]: wiring model
  - **`DistilBertForMaskedLM`** [wiring]: wiring + LM head
  - **`DistilBertForSequenceClassification`** [wiring]: wiring + classifier
  - **`DistilBertForQuestionAnswering`** [wiring]: wiring + QA head
  - **`DistilBertForTokenClassification`** [wiring]: wiring + token cls
  - **`DistilBertForMultipleChoice`** [wiring]: wiring + MC head

## doge
- **src**: modular_doge.py
- **status**: partial
- **partial_reason**: DogeAttention.prepare_dynamic_mask uses torch.topk + scatter to build a sparse attention mask each step (no kb-nano kernel for DMA). DogeCDMoE uses two nn.Embedding(num_experts, hidden_size) tables + matmul to materialize per-token expert weights, which has no shared-expert / grouped-MoE kb-nano equivalent. Both run in plain PyTorch.
- **rationale**: DogeAttention adds Dynamic-Mask-Attention (dt_proj + softplus + topk + scatter mask) on top of standard SDPA; DogeCDMoE is custom retrieval-MoE using nn.Embedding indexing. The math sits on torch.nn / torch.topk fallbacks; no kb-nano kernel for DMA mask gen or product-key MoE retrieval.
- **classes**:
  - **`DogeDecoderLayer`** [compute]: DogeAttention.prepare_dynamic_mask uses torch.topk + scatter to build a sparse attention mask each step (no kb-nano kernel for DMA). DogeCDMoE uses two nn.Embedding(num_experts, hidden_size) tables + 
  - **`DogeRMSNorm`** [compute]: `L1/rms_norm.py` (Llama-style RMSNorm)
  - **`DogeRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (standard NeoX RoPE)
  - **`DogeAttention`** [compute]: `L2/attention.py` (Q/K/V + GQA + RoPE matches LlamaAttention; but additionally builds dynamic-mask via dt_proj + softplus + topk (no kb-nano DMA kernel; falls back to torch ops))
  - **`DogeMLP`** [compute]: `L2/llama_mlp.py` (SwiGLU MLP)
  - **`DogeCDMoE`** [wiring]: Custom Cross-Domain MoE: product-key retrieval via two nn.Embedding tables + per-token matmul; no kb-nano grouped-MoE / shared-expert kernel matches this layout. Implemented with plain torch.matmul + nn.Embedding.
  - **`DogeModel`** [wiring]: wiring stack
  - **`DogeForCausalLM`** [wiring]: wiring + LM head
  - **`DogeForSequenceClassification`** [wiring]: wiring + classifier

## donut
- **src**: modeling_donut_swin.py
- **status**: composable
- **rationale**: Donut-Swin v1 encoder: window-partitioned SDPA with relative-position-bias add, configurable-act FFN, patch merging via Linear. All math expressible with kb-nano L2/encoder_attention (additive bias mask), L1/conv2d, L1/linear, L1/layer_norm, L1/gelu.
- **classes**:
  - **`DonutSwinEmbeddings`** [wiring]: wiring: patch embed + abs pos + dropout
  - **`DonutSwinPatchEmbeddings`** [compute]: `L1/conv2d.py` (Conv2d patch projection)
  - **`DonutSwinPatchMerging`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (concat 2x2 + Linear + LayerNorm)
  - **`DonutSwinDropPath`** [wiring]: stochastic depth
  - **`DonutSwinSelfAttention`** [compute]: `L2/encoder_attention.py` (separate Q/K/V + non-causal SDPA with additive relative-position-bias and optional shifted-window mask (both treated as additive attn_mask))
  - **`DonutSwinSelfOutput`** [compute]: `L1/linear.py` (Linear + dropout)
  - **`DonutSwinAttention`** [wiring]: wiring SelfAttention + SelfOutput
  - **`DonutSwinIntermediate`** [compute]: `L1/linear.py`, `L1/gelu.py` (Linear + GELU)
  - **`DonutSwinOutput`** [compute]: `L1/linear.py` (Linear + dropout)
  - **`DonutSwinLayer`** [wiring]: wiring shifted-window block
  - **`DonutSwinStage`** [wiring]: wiring stage
  - **`DonutSwinEncoder`** [wiring]: wiring encoder
  - **`DonutSwinModel`** [wiring]: wiring model
  - **`DonutSwinForImageClassification`** [wiring]: wiring classifier

## dots1
- **src**: modular_dots1.py
- **status**: composable
- **rationale**: Qwen3-style attention (Q/K-norm + RoPE) + DeepseekV3 shared-expert MoE wrapper; all parts have kb-nano kernels (attention.py, deepseek_moe.py / shared_expert_moe.py, llama_mlp.py).
- **classes**:
  - **`Dots1RMSNorm`** [compute]: `L1/rms_norm.py` (standard RMSNorm)
  - **`Dots1RotaryEmbedding`** [compute]: `L1/rotary_emb.py` (standard NeoX RoPE)
  - **`Dots1Attention`** [compute]: `L2/attention.py` (Q/K-norm + GQA + RoPE matches LlamaAttention configuration)
  - **`Dots1MLP`** [compute]: `L2/llama_mlp.py` (SwiGLU MLP (gate_up + SiluAndMul + down))
  - **`Dots1TopkRouter`** [compute]: `L1/grouped_topk.py`, `L1/sigmoid_topk.py` (DeepseekV3 grouped top-k routing)
  - **`Dots1MoE`** [compute]: `L2/deepseek_moe.py`, `L2/shared_expert_moe.py` (shared expert + routed experts pattern)
  - **`Dots1DecoderLayer`** [wiring]: wiring block
  - **`Dots1Model`** [wiring]: wiring stack
  - **`Dots1ForCausalLM`** [wiring]: wiring + LM head

## dpr
- **src**: modeling_dpr.py
- **status**: composable
- **rationale**: DPREncoder/Reader/Predictor are pure wrappers around BertModel + Linear projections; all compute lives in BERT primitives (encoder_attention, encoder_mlp).
- **classes**:
  - **`DPREncoder`** [wiring]: wraps BertModel + projection Linear
  - **`DPRSpanPredictor`** [wiring]: wraps DPREncoder + 2 Linears for span prediction
  - **`DPRContextEncoder`** [wiring]: wiring: DPREncoder for context
  - **`DPRQuestionEncoder`** [wiring]: wiring: DPREncoder for question
  - **`DPRReader`** [wiring]: wiring: DPRSpanPredictor

## dpt
- **src**: modeling_dpt.py
- **status**: composable
- **rationale**: Dense Prediction Transformer: ViT backbone (separate Q/K/V SDPA + GELU MLP) + Reassemble (Conv2d/ConvTranspose2d) + FeatureFusion (Conv2d + BatchNorm2d + ReLU). All primitives exist in kb-nano L1.
- **classes**:
  - **`DPTViTHybridEmbeddings`** [wiring]: wiring: backbone-feature + CLS + pos embed
  - **`DPTViTEmbeddings`** [wiring]: wiring: patch + CLS + pos embed
  - **`DPTViTPatchEmbeddings`** [compute]: `L1/conv2d.py` (Conv2d patch projection)
  - **`DPTSelfAttention`** [compute]: `L2/encoder_attention.py` (separate Q/K/V + non-causal SDPA)
  - **`DPTViTSelfOutput`** [compute]: `L1/linear.py` (Linear + dropout)
  - **`DPTViTAttention`** [wiring]: wiring SelfAttention + SelfOutput
  - **`DPTViTIntermediate`** [compute]: `L1/linear.py`, `L1/gelu.py` (Linear + GELU)
  - **`DPTViTOutput`** [compute]: `L1/linear.py` (Linear + dropout + residual)
  - **`DPTViTLayer`** [wiring]: wiring block
  - **`DPTReassembleStage`** [wiring]: wiring: reshape + readout-projects + reassemble layers
  - **`DPTReassembleLayer`** [compute]: `L1/conv2d.py`, `L1/conv_transpose2d.py` (1x1 Conv2d + ConvTranspose2d (upsample) or Conv2d (downsample))
  - **`DPTFeatureFusionStage`** [wiring]: wiring stage
  - **`DPTPreActResidualLayer`** [compute]: `L1/conv2d.py`, `L1/relu.py`, `L1/batch_norm2d.py` (ReLU + Conv2d (+BN) + ReLU + Conv2d (+BN))
  - **`DPTFeatureFusionLayer`** [compute]: `L1/conv2d.py`, `L1/interpolate.py` (residual fusion + Conv2d + bilinear interp upsample)
  - **`DPTViTEncoder`** [wiring]: wiring layer stack
  - **`DPTModel`** [wiring]: wiring model
  - **`DPTViTPooler`** [wiring]: wiring: take CLS + tanh
  - **`DPTNeck`** [wiring]: wiring: reassemble + fusion
  - **`DPTDepthEstimationHead`** [compute]: `L1/conv2d.py`, `L1/relu.py`, `L1/interpolate.py` (Conv2d -> upsample -> Conv2d -> ReLU -> Conv2d -> ReLU)
  - **`DPTForDepthEstimation`** [wiring]: wiring full depth pipeline
  - **`DPTSemanticSegmentationHead`** [compute]: `L1/conv2d.py`, `L1/batch_norm2d.py`, `L1/relu.py` (Conv2d + BN + ReLU + Conv2d)
  - **`DPTAuxiliaryHead`** [compute]: `L1/conv2d.py`, `L1/batch_norm2d.py`, `L1/relu.py` (aux Conv2d head)
  - **`DPTForSemanticSegmentation`** [wiring]: wiring full seg pipeline

## edgetam
- **src**: modular_edgetam.py
- **status**: composable
- **rationale**: EdgeTam is a SAM2 derivative: Sam2 Vision encoder (RepViT-style), Sam2 prompt encoder, Sam2 mask decoder with two-way attention. EdgeTam-only classes are thin overrides; all underlying compute is SDPA + Linear + Conv2d + LayerNorm + GELU, expressible with kb-nano primitives. No SAM2 L4 pipeline exists.
- **classes**:
  - **`EdgeTamLayerNorm`** [compute]: `L1/layer_norm.py` (channels-first LayerNorm)
  - **`EdgeTamAttention`** [compute]: `L2/encoder_attention.py` (Sam2 cross/self attention reduces to separate Q/K/V SDPA)
  - **`EdgeTamTwoWayAttentionBlock`** [wiring]: wiring two-way attn block
  - **`EdgeTamFeedForward`** [compute]: `L2/encoder_mlp.py` (Linear + ReLU + Linear feed-forward)
  - **`EdgeTamVisionModel`** [wiring]: wiring vision encoder
  - **`EdgeTamModel`** [wiring]: wiring full SAM-style model

## edgetam_video
- **src**: modular_edgetam_video.py
- **status**: composable
- **rationale**: EdgeTam video extends EdgeTam with 2D-RoPE memory attention, perceiver resampler and memory encoder. All custom classes use SDPA + Linear + Conv2d + 2D rotary apply (expressible with rotary_emb + tensor ops). No SAM2-video L4 pipeline; close cousin sam3_video.py exists for SAM3 only.
- **classes**:
  - **`EdgeTamVideoLayerNorm`** [compute]: `L1/layer_norm.py` (channels-first LayerNorm)
  - **`EdgeTamVideoMemoryFuserCXBlock`** [compute]: `L1/conv2d.py`, `L1/layer_norm.py`, `L1/gelu.py`, `L1/linear.py` (ConvNeXt-style fuser block)
  - **`EdgeTamVideoVisionRotaryEmbedding`** [wiring]: axial 2D rotary frequency build (no cuda kernel)
  - **`EdgeTamVideoAttention`** [compute]: `L2/encoder_attention.py` (standard SDPA)
  - **`EdgeTamVideoRoPESelfAttention`** [compute]: `L2/encoder_attention.py`, `L1/vision_rotary_emb.py` (Q/K/V + 2D RoPE apply + non-causal SDPA; per guideline 5 vision 2D RoPE uses vision_rotary_emb, not bare rotary_emb.)
  - **`EdgeTamVideoRoPECrossAttention`** [compute]: `L1/linear.py`, `L1/sdpa.py`, `L1/vision_rotary_emb.py` (cross-attn Q (image) <- K,V (memory) + per-side 2D RoPE; per guideline 5 vision 2D RoPE uses vision_rotary_emb.)
  - **`EdgeTamVideoTwoWayAttentionBlock`** [wiring]: wiring two-way attn block
  - **`EdgeTamVideoPositionEmbeddingSine`** [wiring]: sinusoidal 2D position embed (LRU-cached)
  - **`EdgeTamVideoMemoryEncoder`** [wiring]: wiring memory encoder
  - **`EdgeTamVideoFeedForward`** [compute]: `L2/encoder_mlp.py` (Linear+ReLU+Linear FFN)
  - **`EdgeTamVideoMemoryAttentionMLP`** [compute]: `L1/linear.py`, `L1/relu.py` (Linear + activation + Linear)
  - **`EdgeTamVideoMemoryAttentionLayer`** [wiring]: wiring: RoPE self-attn + RoPE cross-attn + MLP
  - **`EdgeTamVideoMemoryAttention`** [wiring]: wiring memory-attn stack
  - **`EdgeTamVideoPerceiverMLP`** [compute]: `L1/linear.py`, `L1/gelu.py` (fc1 + GELU + fc2)
  - **`EdgeTamVideoPerceiverAttention`** [compute]: `L2/encoder_attention.py` (perceiver cross-attn (Q from learnt latents, KV from input))
  - **`EdgeTamVideoPerceiverEncoderLayer`** [wiring]: wiring perceiver layer
  - **`EdgeTamVideoPerceiverResampler`** [wiring]: wiring perceiver resampler
  - **`EdgeTamVideoModel`** [wiring]: wiring full video model

## efficientloftr
- **src**: modeling_efficientloftr.py
- **status**: composable
- **rationale**: EfficientLoFTR keypoint matcher: RepVGG-style CNN backbone (Conv2d + BN), aggregated attention with 2D RoPE on aggregated tokens (SDPA), fine-fusion convs, spatial expectation head. All ops covered by kb-nano L1 (conv2d, batch_norm2d, relu, linear, layer_norm, sdpa, rotary_emb).
- **classes**:
  - **`EfficientLoFTRRotaryEmbedding`** [wiring]: compute 2D rotary cos/sin (CPU/GPU index math; no cuda kernel)
  - **`EfficientLoFTRConvNormLayer`** [compute]: `L1/conv2d.py`, `L1/batch_norm2d.py`, `L1/relu.py` (Conv2d + BN + (optional) ReLU)
  - **`EfficientLoFTRRepVGGBlock`** [compute]: `L1/conv2d.py`, `L1/batch_norm2d.py`, `L1/relu.py` (RepVGG: parallel 3x3 + 1x1 + identity convs)
  - **`EfficientLoFTRRepVGGStage`** [wiring]: wiring stage
  - **`EfficientLoFTRepVGG`** [wiring]: wiring backbone
  - **`EfficientLoFTRAggregationLayer`** [compute]: `L1/conv2d.py`, `L1/layer_norm.py` (depthwise Conv2d aggregation + LayerNorm)
  - **`EfficientLoFTRAttention`** [compute]: `L2/encoder_attention.py`, `L1/vision_rotary_emb.py` (Q/K/V + 2D RoPE + SDPA (self or cross); per guideline 5 vision 2D RoPE uses vision_rotary_emb, not bare rotary_emb.)
  - **`EfficientLoFTRMLP`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (concat + 2-layer MLP + LayerNorm)
  - **`EfficientLoFTRAggregatedAttention`** [wiring]: wiring: aggregate -> attention -> MLP
  - **`EfficientLoFTRLocalFeatureTransformerLayer`** [wiring]: wiring self/cross aggregated attn
  - **`EfficientLoFTRLocalFeatureTransformer`** [wiring]: wiring transformer stack
  - **`EfficientLoFTROutConvBlock`** [compute]: `L1/conv2d.py`, `L1/batch_norm2d.py`, `L1/relu.py` (Conv-norm-act stack)
  - **`EfficientLoFTRFineFusionLayer`** [compute]: `L1/conv2d.py`, `L1/interpolate.py` (fine fusion via Conv2d + bilinear upsample)
  - **`EfficientLoFTRModel`** [wiring]: wiring backbone + transformer + fine fusion
  - **`EfficientLoFTRForKeypointMatching`** [wiring]: wiring full matcher (uses spatial_expectation2d helper)

## efficientnet
- **src**: modeling_efficientnet.py
- **status**: composable
- **rationale**: EfficientNet v1 stem + MBConv blocks (expansion + depthwise + squeeze-excite + final). All ops are Conv2d / depthwise Conv2d / BatchNorm2d / Sigmoid / configurable activation, all in kb-nano L1. Note: kb-nano L4/efficientnetv2.py exists but targets v2 (different block layout).
- **classes**:
  - **`EfficientNetEmbeddings`** [compute]: `L1/conv2d.py`, `L1/batch_norm2d.py` (ZeroPad + Conv2d stem + BN + activation)
  - **`EfficientNetDepthwiseConv2d`** [compute]: `L1/conv2d.py` (Conv2d with groups=in_channels (depthwise))
  - **`EfficientNetExpansionLayer`** [compute]: `L1/conv2d.py`, `L1/batch_norm2d.py` (1x1 Conv2d + BN + activation)
  - **`EfficientNetDepthwiseLayer`** [compute]: `L1/conv2d.py`, `L1/batch_norm2d.py` (ZeroPad + depthwise Conv2d + BN + activation)
  - **`EfficientNetSqueezeExciteLayer`** [compute]: `L2/efficientnetv2_squeeze_excite.py` (global avg pool + 1x1 reduce + activation + 1x1 expand + sigmoid (matches v2 SE module))
  - **`EfficientNetFinalBlockLayer`** [compute]: `L1/conv2d.py`, `L1/batch_norm2d.py` (1x1 Conv2d + BN + drop_path + skip)
  - **`EfficientNetBlock`** [wiring]: wiring MBConv block
  - **`EfficientNetEncoder`** [wiring]: wiring block stack
  - **`EfficientNetModel`** [wiring]: wiring model
  - **`EfficientNetForImageClassification`** [wiring]: wiring + classifier

## electra
- **src**: modeling_electra.py
- **status**: composable
- **rationale**: BERT-shaped encoder (separate Q/K/V SDPA, GELU FFN), discriminator/generator heads. All compute classes covered by encoder_attention + encoder_mlp.
- **classes**:
  - **`ElectraEmbeddings`** [wiring]: wiring: word + pos + tok-type embed + LayerNorm
  - **`ElectraSelfAttention`** [compute]: `L2/encoder_attention.py` (BERT-style separate Q/K/V SDPA)
  - **`ElectraCrossAttention`** [compute]: `L2/encoder_attention.py` (cross-attn variant of EncoderSelfAttention)
  - **`ElectraSelfOutput`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (Linear + dropout + LayerNorm + residual)
  - **`ElectraAttention`** [wiring]: wiring SelfAttention + SelfOutput
  - **`ElectraIntermediate`** [compute]: `L1/linear.py`, `L1/gelu.py` (Linear + GELU)
  - **`ElectraOutput`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (Linear + dropout + LayerNorm)
  - **`ElectraLayer`** [wiring]: wiring block
  - **`ElectraEncoder`** [wiring]: wiring encoder stack
  - **`ElectraDiscriminatorPredictions`** [compute]: `L1/linear.py` (Linear + activation + Linear)
  - **`ElectraGeneratorPredictions`** [compute]: `L1/linear.py`, `L1/gelu.py`, `L1/layer_norm.py` (Linear + GELU + LayerNorm)
  - **`ElectraModel`** [wiring]: wiring model
  - **`ElectraClassificationHead`** [compute]: `L1/linear.py` (2 Linears + dropout)
  - **`ElectraSequenceSummary`** [wiring]: wiring pooled summary
  - **`ElectraForSequenceClassification`** [wiring]: wiring + cls head
  - **`ElectraForPreTraining`** [wiring]: wiring + discriminator head
  - **`ElectraForMaskedLM`** [wiring]: wiring + generator head
  - **`ElectraForTokenClassification`** [wiring]: wiring + token cls
  - **`ElectraForQuestionAnswering`** [wiring]: wiring + QA head
  - **`ElectraForMultipleChoice`** [wiring]: wiring + MC head
  - **`ElectraForCausalLM`** [wiring]: wiring + LM head

## emu3
- **src**: modular_emu3.py
- **status**: composable
- **rationale**: Llama-style text decoder (LlamaAttention + LlamaMLP) plus Emu3 VQ-VAE for image tokens (Conv3d + GroupNorm + SiglipAttention + ChameleonVQVAE blocks). All compute primitives have kb-nano kernels (attention.py, llama_mlp.py, conv3d, group_norm, siglip_attention).
- **classes**:
  - **`Emu3Attention`** [compute]: `L2/attention.py` (LlamaAttention with RoPE + GQA)
  - **`Emu3DecoderLayer`** [wiring]: wiring decoder block
  - **`Emu3VQVAEVectorQuantizer`** [compute]: `L1/embedding.py` (Embedding codebook + nearest-neighbor lookup)
  - **`Emu3VQVAEEncoderConvDownsample`** [compute]: `L1/conv2d.py` (Conv2d strided down)
  - **`Emu3VQVAEEncoderConvUpsample`** [compute]: `L1/conv2d.py`, `L1/interpolate.py` (interpolate + Conv2d up)
  - **`Emu3VQVAEConv3d`** [compute]: `L1/conv3d.py` (Conv3d wrapper)
  - **`Emu3VQVAESpatialNorm`** [compute]: `L1/group_norm.py`, `L1/conv2d.py`, `L1/interpolate.py` (GroupNorm + spatial-conditional convs)
  - **`Emu3VQVAETemporalUpsample`** [compute]: `L1/conv3d.py`, `L1/interpolate.py` (temporal Conv3d + interp)
  - **`Emu3VQVAETemporalDownsample`** [compute]: `L1/conv3d.py` (Conv3d strided)
  - **`Emu3VQVAETemporalResnetBlock`** [compute]: `L1/conv3d.py`, `L1/group_norm.py`, `L1/silu.py` (Conv3d + GroupNorm + SiLU residual)
  - **`Emu3VQVAEResnetBlock`** [compute]: `L1/conv2d.py`, `L1/group_norm.py`, `L1/silu.py` (Conv2d + GroupNorm + SiLU residual)
  - **`Emu3VQVAEAttentionBlock`** [compute]: `L2/siglip_attention.py` (SigLIP-style multi-head SDPA)
  - **`Emu3VQVAEGroupNorm`** [compute]: `L1/group_norm.py` (GroupNorm with optional spatial conditioning)
  - **`Emu3VQVAEMiddleBlock`** [wiring]: wiring resnet + attn + resnet
  - **`Emu3VQVAEDownBlock`** [wiring]: wiring downsample stage
  - **`Emu3VQVAEUpBlock`** [wiring]: wiring upsample stage
  - **`Emu3VQVAEEncoder`** [wiring]: wiring VAE encoder
  - **`Emu3VQVAEDecoder`** [wiring]: wiring VAE decoder
  - **`Emu3VQVAE`** [wiring]: wiring VQ-VAE module
  - **`Emu3TextModel`** [wiring]: wiring Llama text stack
  - **`Emu3ForCausalLM`** [wiring]: wiring Llama LM head
  - **`Emu3Model`** [wiring]: wiring multimodal model
  - **`Emu3ForConditionalGeneration`** [wiring]: wiring full multimodal generation

## encodec
- **src**: modeling_encodec.py
- **status**: composable
- **rationale**: SEANet audio codec: weight-norm Conv1d/ConvTranspose1d + ELU + ResidualBlock + LSTM + EuclideanCodebook (residual VQ). All ops covered by kb-nano L1 (conv1d, conv_transpose1d, lstm, elu, embedding).
- **classes**:
  - **`EncodecConv1d`** [compute]: `L1/conv1d.py` (weight-normed Conv1d wrapper)
  - **`EncodecConvTranspose1d`** [compute]: `L1/conv_transpose1d.py` (weight-normed ConvTranspose1d wrapper)
  - **`EncodecLSTM`** [compute]: `L1/lstm.py` (nn.LSTM with residual)
  - **`EncodecResnetBlock`** [compute]: `L1/conv1d.py`, `L1/elu.py` (ELU + Conv1d + ELU + Conv1d residual)
  - **`EncodecEncoder`** [wiring]: wiring SEANet encoder
  - **`EncodecDecoder`** [wiring]: wiring SEANet decoder
  - **`EncodecEuclideanCodebook`** [wiring]: Euclidean nearest-neighbor codebook lookup (cdist + argmin + embedding)
  - **`EncodecVectorQuantization`** [wiring]: wiring single-VQ
  - **`EncodecResidualVectorQuantizer`** [wiring]: wiring residual VQ stack
  - **`EncodecModel`** [wiring]: wiring full codec

## encoder_decoder
- **src**: modeling_encoder_decoder.py
- **status**: composable
- **rationale**: EncoderDecoderModel is a pure wrapper that delegates to any encoder + decoder pair (e.g. BERT->GPT2). It contributes no compute kernels of its own.
- **classes**:
  - **`EncoderDecoderModel`** [wiring]: wiring: holds encoder + decoder modules and routes hidden states between them

## eomt
- **src**: modular_eomt.py
- **status**: composable
- **rationale**: EoMT = Mask2Former-style universal segmentation built on a Dinov2 ViT backbone with SigLIP-style attention; queries are interleaved as extra tokens. Compute = encoder_attention/SDPA + encoder_mlp + Conv2d + LayerNorm + bilinear interp.
- **classes**:
  - **`EomtPatchEmbeddings`** [compute]: `L1/conv2d.py` (Conv2d patch projection)
  - **`EomtEmbeddings`** [wiring]: wiring: patch + CLS + register tokens + pos embed
  - **`EomtAttention`** [compute]: `L2/siglip_attention.py` (SigLIP separate Q/K/V SDPA)
  - **`EomtLayerScale`** [wiring]: scalar gate
  - **`EomtLayer`** [wiring]: wiring transformer block
  - **`EomtLayerNorm2d`** [compute]: `L1/layer_norm.py` (channels-first LayerNorm via permute)
  - **`EomtScaleLayer`** [compute]: `L1/conv2d.py`, `L1/conv_transpose2d.py` (ConvTranspose2d upsample + Conv2d)
  - **`EomtScaleBlock`** [wiring]: wiring scale layers
  - **`EomtMaskHead`** [compute]: `L1/linear.py`, `L1/gelu.py` (Linear + GELU + Linear + GELU + Linear)
  - **`EomtForUniversalSegmentation`** [wiring]: wiring full segmentation pipeline

## eomt_dinov3
- **src**: modular_eomt_dinov3.py
- **status**: composable
- **rationale**: EoMT applied on a DINOv3-ViT backbone (with 2D RoPE) plus the same Mask2Former-style segmentation head as EoMT. Backbone matches kb-nano L4/dinov3.py components but the pipeline as a whole (queries + scale block + mask head) is not an L4. All compute classes have kb-nano kernels.
- **classes**:
  - **`EomtDinov3Attention`** [compute]: `L2/eva_attention.py` (Q/K/V + 2D RoPE + SDPA)
  - **`EomtDinov3Embeddings`** [wiring]: wiring: patch + CLS + registers + queries-interleaved
  - **`EomtDinov3Layer`** [compute]: `L3/eva_block.py` (EVA pre-norm block + layer-scale + SwiGLU/MLP)
  - **`EomtDinov3LayerScale`** [wiring]: scalar gate
  - **`EomtDinov3RotaryEmbedding`** [compute]: `L1/dinov3_rope.py` (DINOv3 2D RoPE)
  - **`EomtDinov3ForUniversalSegmentation`** [wiring]: wiring full segmentation model

## ernie
- **src**: modular_ernie.py
- **status**: composable
- **rationale**: Pure inheritance from BERT (encoder_attention + encoder_mlp + bert_embeddings) with task-id embedding added; no new compute kernels.
- **classes**:
  - **`ErnieEmbeddings`** [compute]: `L2/bert_embeddings.py` (BERT embeddings + task-type embedding sum)
  - **`ErnieSelfAttention`** [compute]: `L2/encoder_attention.py` (BERT SDPA)
  - **`ErnieCrossAttention`** [compute]: `L2/encoder_attention.py` (BERT cross-attn)
  - **`ErnieLayer`** [compute]: `L3/bert_layer.py` (BERT block)
  - **`ErniePooler`** [wiring]: wiring CLS pooling + tanh
  - **`ErnieLMPredictionHead`** [wiring]: wiring LM head
  - **`ErnieEncoder`** [compute]: `L3/bert_encoder.py` (wiring layer stack)
  - **`ErnieModel`** [compute]: `L3/bert_model.py` (wiring model)
  - **`ErnieForPreTraining`** [wiring]: wiring + pretraining heads
  - **`ErnieForCausalLM`** [wiring]: wiring + LM head
  - **`ErnieForMaskedLM`** [wiring]: wiring + MLM head
  - **`ErnieForNextSentencePrediction`** [wiring]: wiring + NSP head
  - **`ErnieForSequenceClassification`** [wiring]: wiring + cls head
  - **`ErnieForMultipleChoice`** [wiring]: wiring + MC head
  - **`ErnieForTokenClassification`** [wiring]: wiring + token cls
  - **`ErnieForQuestionAnswering`** [wiring]: wiring + QA head

## ernie4_5
- **src**: modular_ernie4_5.py
- **status**: composable
- **rationale**: Llama-shaped decoder: LlamaAttention + LlamaMLP + custom RoPE (Olmo-style); all kernels exist (attention.py + llama_mlp.py + rotary_emb).
- **classes**:
  - **`Ernie4_5RotaryEmbedding`** [compute]: `L1/rotary_emb.py` (standard NeoX RoPE)
  - **`Ernie4_5MLP`** [compute]: `L2/llama_mlp.py` (SwiGLU MLP)
  - **`Ernie4_5Attention`** [compute]: `L2/attention.py` (Llama attention with GQA + RoPE)
  - **`Ernie4_5ForCausalLM`** [wiring]: wiring + LM head

## ernie4_5_moe
- **src**: modular_ernie4_5_moe.py
- **status**: composable
- **rationale**: Llama-style attention + Mixtral expert pattern (top-K router + per-expert SwiGLU MLP) with extra correction-bias router (Ernie4_5_MoeStatics). Maps to attention.py + mixtral_moe.py.
- **classes**:
  - **`Ernie4_5_MoeRMSNorm`** [compute]: `L1/rms_norm.py` (RMSNorm)
  - **`Ernie4_5_MoeMLP`** [compute]: `L2/llama_mlp.py` (SwiGLU MLP per expert)
  - **`Ernie4_5_MoeRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (RoPE)
  - **`Ernie4_5_MoeAttention`** [compute]: `L2/attention.py` (Llama attention)
  - **`Ernie4_5_MoeStatics`** [wiring]: static correction-bias buffers; no compute
  - **`Ernie4_5_MoeExperts`** [compute]: `L1/moe_grouped_gemm.py` (grouped per-expert SwiGLU using Mixtral expert kernel)
  - **`Ernie4_5_MoeTopKRouter`** [compute]: `L1/topk_softmax.py` (router gate + top-k softmax with correction bias)
  - **`Ernie4_5_MoeSparseMoeBlock`** [compute]: `L2/mixtral_moe.py` (router + experts dispatch + (optional) shared expert add)
  - **`Ernie4_5_MoeDecoderLayer`** [wiring]: wiring block
  - **`Ernie4_5_MoeModel`** [wiring]: wiring stack
  - **`Ernie4_5_MoeForCausalLM`** [wiring]: wiring + LM head

## ernie4_5_vl_moe
- **src**: modular_ernie4_5_vl_moe.py
- **status**: partial
- **rationale**: VL extension of Ernie4_5_Moe: Qwen2-VL/2.5-VL vision tower + Ernie4_5 MoE text decoder + variable-resolution resampler. All compute uses kernels already present (vision_attention, vision_mlp, attention.py, mixtral_moe, llama_mlp).
- **classes**:
  - **`Ernie4_5_VLMoeMoeBlock`** [compute]: no kb-nano kernel — VL extension of Ernie4_5_Moe: Qwen2-VL/2.5-VL vision tower + Ernie4_5 MoE text decoder + variable-resolution resampler. All compute uses kernels already present (vision_attention, vision_mlp, attentio
  - **`Ernie4_5_VLMoeTextRotaryEmbedding`** [compute]: `L1/mrope.py` (M-RoPE for VL)
  - **`Ernie4_5_VLMoeTextAttention`** [compute]: `L2/attention.py` (Llama-style attn)
  - **`Ernie4_5_VLMoeRMSNorm`** [compute]: `L1/rms_norm.py` (RMSNorm)
  - **`Ernie4_5_VLMoeMLP`** [compute]: `L2/llama_mlp.py` (SwiGLU MLP)
  - **`Ernie4_5_VLMoeMoeStatics`** [wiring]: router bias buffers
  - **`Ernie4_5_VLMoeMoeTopKRouter`** [compute]: `L1/topk_softmax.py` (top-k router)
  - **`Ernie4_5_VLMoeMoeExperts`** [compute]: `L1/moe_grouped_gemm.py` (grouped expert GEMM)
  - **`Ernie4_5_VLMoeSparseMoeBlock`** [compute]: `L2/mixtral_moe.py` (router + experts)
  - **`Ernie4_5_VLMoeDecoderLayer`** [wiring]: wiring block
  - **`Ernie4_5_VLMoeVisionAttention`** [compute]: `L2/vision_attention.py` (Qwen2.5-VL vision attention)
  - **`Ernie4_5_VLMoeVisionBlock`** [wiring]: wiring vision block
  - **`Ernie4_5_VLMoeTextModel`** [wiring]: wiring text stack
  - **`Ernie4_5VLVisionMLP`** [compute]: `L2/vision_mlp.py` (Qwen vision MLP)
  - **`Ernie4_5_VLMoePatchEmbed`** [compute]: `L1/conv2d.py` (Qwen2.5-VL patch embed)
  - **`Ernie4_5_VLMoeVisionRotaryEmbedding`** [compute]: `L1/vision_rotary_emb.py` (vision 2D RoPE)
  - **`Ernie4_5_VLMoeVisionTransformerPretrainedModel`** [wiring]: wiring vision tower
  - **`Ernie4_5_VLMoeVisionMLP`** [compute]: `L1/linear.py`, `L1/silu_and_mul.py` (gate + up + SiLU + down (SwiGLU vision MLP); per guideline 3 SwiGLU MLPs use silu_and_mul.)
  - **`Ernie4_5_VLMoeVariableResolutionResamplerModel`** [wiring]: wiring variable-resolution resampler (Linear + LayerNorm + SiLU + averaging)
  - **`Ernie4_5_VLMoeModel`** [wiring]: wiring multimodal model
  - **`Ernie4_5_VLMoeForConditionalGeneration`** [wiring]: wiring conditional generation

## esm
- **src**: modeling_esm.py
- **status**: composable
- **rationale**: BERT-shaped protein encoder with rotary position embeddings (optional). Q is pre-scaled (instead of post-scaling K@V) for ESM rotary correctness, but the math is still separate-Q/K/V SDPA with optional RoPE.
- **classes**:
  - **`EsmRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (GPT-NeoX-style RoPE (interleaved-half))
  - **`EsmContactPredictionHead`** [compute]: `L1/linear.py`, `L1/sigmoid.py` (Linear + sigmoid over symmetrized averaged attentions)
  - **`EsmEmbeddings`** [wiring]: wiring: word + (optional abs pos) + layernorm/dropout
  - **`EsmSelfAttention`** [compute]: `L2/encoder_attention.py`, `L1/rotary_emb.py` (separate Q/K/V SDPA with optional RoPE; Q is pre-scaled to match ESM RoPE convention)
  - **`EsmSelfOutput`** [compute]: `L1/linear.py` (Linear + dropout + residual)
  - **`EsmAttention`** [wiring]: wiring LayerNorm + SelfAttention + SelfOutput
  - **`EsmIntermediate`** [compute]: `L1/linear.py`, `L1/gelu.py` (Linear + GELU)
  - **`EsmOutput`** [compute]: `L1/linear.py` (Linear + dropout + residual)
  - **`EsmLayer`** [wiring]: wiring block
  - **`EsmEncoder`** [wiring]: wiring stack
  - **`EsmPooler`** [wiring]: wiring: take CLS + tanh
  - **`EsmModel`** [wiring]: wiring model
  - **`EsmForMaskedLM`** [wiring]: wiring + MLM head
  - **`EsmLMHead`** [compute]: `L1/linear.py`, `L1/gelu.py`, `L1/layer_norm.py` (Linear + GELU + LayerNorm + Linear)
  - **`EsmForSequenceClassification`** [wiring]: wiring + cls head
  - **`EsmForTokenClassification`** [wiring]: wiring + token cls head
  - **`EsmClassificationHead`** [compute]: `L1/linear.py` (Linear + tanh + Linear)

## eurobert
- **src**: modular_eurobert.py
- **status**: composable
- **rationale**: EuroBert is a pure Llama-shaped encoder (LlamaAttention + LlamaMLP + LlamaRMSNorm) with bidirectional masking; all primitives covered by attention.py + llama_mlp.py + rms_norm.
- **classes**:
  - **`EuroBertRMSNorm`** [compute]: `L1/rms_norm.py` (Llama RMSNorm)
  - **`EuroBertAttention`** [compute]: `L2/attention.py` (Llama attention (bidirectional, no causal mask))
  - **`EuroBertModel`** [wiring]: wiring stack
  - **`EuroBertForMaskedLM`** [wiring]: wiring + MLM head
  - **`EuroBertForSequenceClassification`** [wiring]: wiring + cls head
  - **`EuroBertForTokenClassification`** [wiring]: wiring + token cls head
