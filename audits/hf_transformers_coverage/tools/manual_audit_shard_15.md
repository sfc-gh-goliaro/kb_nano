## univnet
- **src**: modeling_univnet.py
- **status**: partial
- **rationale**: Location-variable convolution (LVC) and weight_norm parametrization with custom einsum-based pooling-conv have no kb-nano equivalent.
- **classes**:
  - **`UnivNetLvcResidualBlock`** [compute]: Location-variable convolution (LVC) and weight_norm parametrization with custom einsum-based pooling-conv have no kb-nano equivalent.
  - **`UnivNetKernelPredictorResidualBlock`** [compute]: `L1/conv1d.py`, `L1/leaky_relu.py` (Conv1d + leaky_relu residual block; primitives exist.)
  - **`UnivNetKernelPredictor`** [wiring]: Stack of Conv1d layers wrapped with nn.utils.weight_norm parametrization; kb-nano has no weight_norm primitive.
  - **`UnivNetLvcBlock`** [wiring]: Wires UnivNetKernelPredictor with multiple LvcResidualBlocks; depends on the unsupported children.
  - **`UnivNetModel`** [wiring]: Wiring; depends on unsupported LVC blocks.

## upernet
- **src**: modeling_upernet.py
- **status**: composable
- **rationale**: UperNet semantic segmentation head built from Conv2d + BatchNorm2d + ReLU + AdaptiveAvgPool2d + bilinear interpolate; all kb-nano L1 primitives exist.
- **classes**:
  - **`UperNetConvModule`** [compute]: `L1/conv2d.py`, `L1/batch_norm2d.py`, `L1/relu.py` (Conv2d -> BatchNorm2d -> ReLU.)
  - **`UperNetPyramidPoolingBlock`** [compute]: `L1/adaptive_avg_pool2d.py` (AdaptiveAvgPool2d + UperNetConvModule.)
  - **`UperNetPyramidPoolingModule`** [compute]: `L1/interpolate.py` (List of pooling blocks + bilinear interpolate to input resolution.)
  - **`UperNetHead`** [compute]: `L1/conv2d.py`, `L1/interpolate.py` (PSP + FPN + classifier; pure conv2d composition.)
  - **`UperNetFCNHead`** [compute]: `L1/conv2d.py` (FCN auxiliary head; conv2d stack.)
  - **`UperNetForSemanticSegmentation`** [wiring]: Wiring backbone + head; depends on a separate vision backbone.

## uvdoc
- **src**: modular_uvdoc.py
- **status**: composable
- **rationale**: Pure Conv2d + BatchNorm2d + activation ResNet-style backbone with bridge dilated convolutions; PointPositions head uses conv2d. All primitives in kb-nano L1.
- **classes**:
  - **`UVDocConvLayer`** [compute]: `L1/conv2d.py`, `L1/batch_norm2d.py`, `L1/relu.py` (Conv2d -> BatchNorm2d -> activation.)
  - **`UVDocResidualBlock`** [wiring]: Composes UVDocConvLayer; wiring.
  - **`UVDocResNetStage`** [wiring]: Wiring of residual blocks.
  - **`UVDocResNet`** [wiring]: Wiring of resnet head + stages.
  - **`UVDocBridgeBlock`** [wiring]: Sequence of dilated UVDocConvLayer modules; wiring.
  - **`UVDocPointPositions2D`** [compute]: `L1/conv2d.py` (Two Conv2d layers for point regression.)
  - **`UVDocBridge`** [wiring]: Wiring.
  - **`UVDocBackbone`** [wiring]: Wiring.
  - **`UVDocHead`** [compute]: `L1/conv2d.py` (Conv2d head.)
  - **`UVDocModel`** [wiring]: Wiring.

## vaultgemma
- **src**: modular_vaultgemma.py
- **status**: composable
- **rationale**: Modular re-export of Gemma2 components (RMSNorm + SwiGLU MLP + GQA attention + sliding-window decoder layer); maps to gemma_dense_attention + llama_mlp + gemma_rms_norm primitives.
- **classes**:
  - **`VaultGemmaRMSNorm`** [compute]: `L1/gemma_rms_norm.py` (Gemma RMSNorm variant (1+weight scaling).)
  - **`VaultGemmaMLP`** [compute]: `L2/llama_mlp.py`, `L1/silu_and_mul.py` (SwiGLU MLP (gate_up -> silu_and_mul -> down).)
  - **`VaultGemmaAttention`** [compute]: `L2/attention.py`, `L1/rotary_emb.py` (Causal GQA attention with sliding-window support; LlamaAttention covers this pattern.)
  - **`VaultGemmaDecoderLayer`** [wiring]: Wiring with input_layernorm + self_attn + pre_feedforward_layernorm + mlp.
  - **`VaultGemmaForCausalLM`** [wiring]: Wiring.

## vibevoice_acoustic_tokenizer
- **src**: modular_vibevoice_acoustic_tokenizer.py
- **status**: composable
- **rationale**: ConvNext-1d encoder/decoder with CausalConv1d + CausalConvTranspose1d + RMSNorm + simple FFN; all primitives in kb-nano L1.
- **classes**:
  - **`VibeVoiceAcousticTokenizerRMSNorm`** [compute]: `L1/rms_norm.py` (Standard Llama-style RMSNorm.)
  - **`VibeVoiceAcousticTokenizerFeedForward`** [compute]: `L1/linear.py`, `L1/gelu.py` (linear -> activation -> linear (non-SwiGLU two-layer FFN).)
  - **`VibeVoiceAcousticTokenizerCausalConv1d`** [compute]: `L1/conv1d.py`, `L1/causal_conv1d.py` (Conv1d with left causal padding (pad + conv); kb-nano has Conv1d and CausalConv1d L1 ops.)
  - **`VibeVoiceAcousticTokenizerCausalConvTranspose1d`** [compute]: `L1/conv_transpose1d.py` (ConvTranspose1d with right-trim causal handling.)
  - **`VibeVoiceAcousticTokenizerConvNext1dLayer`** [wiring]: Wiring: norm + mixer (CausalConv1d) + ffn with layer_scale gammas.
  - **`VibeVoiceAcousticTokenizerEncoderStem`** [wiring]: Wiring.
  - **`VibeVoiceAcousticTokenizerEncoderLayer`** [wiring]: Wiring.
  - **`VibeVoiceAcousticTokenizerEncoderModel`** [wiring]: Wiring.
  - **`VibeVoiceAcousticTokenizerDecoderStem`** [wiring]: Wiring.
  - **`VibeVoiceAcousticTokenizerDecoderLayer`** [wiring]: Wiring with CausalConvTranspose1d + ConvNext1dLayer stack.
  - **`VibeVoiceAcousticTokenizerDecoderModel`** [wiring]: Wiring.
  - **`VibeVoiceAcousticTokenizerModel`** [wiring]: Wiring (encoder + decoder).

## vibevoice_asr
- **src**: modular_vibevoice_asr.py
- **status**: composable
- **rationale**: ASR wrapper composing acoustic+semantic tokenizers (VibeVoice acoustic tokenizer composable) with a multimodal projector (Linear+RMSNorm) and a generic LLM (AudioFlamingo3 / Qwen2-style); all components composable.
- **classes**:
  - **`VibeVoiceAsrRMSNorm`** [compute]: `L1/rms_norm.py` (Standard Llama/Qwen2-style RMSNorm.)
  - **`VibeVoiceAsrMultiModalProjector`** [compute]: `L1/linear.py`, `L1/rms_norm.py` (Two parallel paths of Linear + RMSNorm + Linear, summed.)
  - **`VibeVoiceAsrForConditionalGeneration`** [wiring]: Wiring: composes tokenizer encoders + projector + language model.

## video_llama_3
- **src**: modular_video_llama_3.py
- **status**: composable
- **rationale**: Vision encoder is Siglip-style with separate Q/K/V + 2D vision RoPE + cu_seqlens varlen attention; LM extends Qwen2VL. All primitives present (siglip-attn pattern, vision_rotary_emb, flash_attn_varlen) but no exact L2 wrapper. LM side has L4 qwen2_vl pipeline support.
- **classes**:
  - **`VideoLlama3VisionRotaryEmbedding`** [compute]: `L1/vision_rotary_emb.py` (2D vision RoPE with merge-size-aware position id construction.)
  - **`VideoLlama3VisionEmbeddings`** [compute]: `L1/conv2d.py` (Patch embedding via Conv2d.)
  - **`VideoLlama3VisionMLP`** [compute]: `L2/siglip_mlp.py` (Inherits SigLIP MLP (fc1+act+fc2).)
  - **`VideoLlama3VisionAttention`** [compute]: `L2/siglip_attention.py`, `L1/vision_rotary_emb.py`, `L1/flash_attn_varlen.py` (Separate Q/K/V (Siglip-style) + 2D vision rope + cu_seqlens varlen attention; primitives exist.)
  - **`VideoLlama3VisionEncoderLayer`** [wiring]: Pre-norm wiring around attention + MLP.
  - **`VideoLlama3VisionEncoder`** [wiring]: Wiring of layer stack.
  - **`VideoLlama3VisionModel`** [wiring]: Wiring: rope + embeddings + encoder + post_layernorm + pixel_unshuffle.
  - **`VideoLlama3Projector`** [compute]: `L1/linear.py` (Linear projection from vision feature dim to LM hidden.)
  - **`VideoLlama3Model`** [wiring]: Wiring: vision tower + projector + Qwen2VL LM (which has L4 qwen2_vl).
  - **`VideoLlama3ForConditionalGeneration`** [wiring]: Wiring.

## video_llava
- **src**: modeling_video_llava.py
- **status**: composable
- **rationale**: Pure wrapper: vision tower (CLIP-style) + multimodal projector (Linear+act+Linear) + Llama LM. All primitives exist in kb-nano.
- **classes**:
  - **`VideoLlavaMultiModalProjector`** [compute]: `L1/linear.py`, `L1/gelu.py` (Linear -> activation -> Linear projection.)
  - **`VideoLlavaModel`** [wiring]: Wiring: video_tower + image_tower + projector + language_model.
  - **`VideoLlavaForConditionalGeneration`** [wiring]: Wiring.

## videomae
- **src**: modeling_videomae.py
- **status**: composable
- **rationale**: ViT-clone (separate Q/K/V dense attention, fc1+GELU+fc2 MLP, layernorm). Pre-/post-norm decoder for masked image modeling reuses same blocks. All primitives in kb-nano.
- **classes**:
  - **`VideoMAEEmbeddings`** [wiring]: Wiring: tubelet patch embedding + sinusoidal position embeddings.
  - **`VideoMAEPatchEmbeddings`** [compute]: `L1/conv3d.py` (Tubelet patch embedding via Conv3d.)
  - **`VideoMAESelfAttention`** [compute]: `L2/encoder_attention.py`, `L1/dense_attention.py` (Separate Q/K/V projections + SDPA attention (non-causal); matches encoder_attention.py:EncoderSelfAttention pattern.)
  - **`VideoMAEAttention`** [wiring]: Wiring: SelfAttention + SelfOutput (BERT/ViT sibling-class wrapper).
  - **`VideoMAEIntermediate`** [compute]: `L2/encoder_mlp.py` (Linear + GELU; matches EncoderIntermediate.)
  - **`VideoMAEOutput`** [compute]: `L2/encoder_mlp.py` (Linear + dropout + residual (without LayerNorm — pre-norm pattern); analogous to EncoderOutput minus LayerNorm.)
  - **`VideoMAELayer`** [wiring]: Wiring: pre-norm attention + pre-norm intermediate/output.
  - **`VideoMAEEncoder`** [wiring]: Wiring.
  - **`VideoMAEModel`** [wiring]: Wiring.
  - **`VideoMAEDecoder`** [wiring]: Wiring of decoder layers.
  - **`VideoMAEForPreTraining`** [wiring]: Wiring.
  - **`VideoMAEForVideoClassification`** [wiring]: Wiring.

## videomt
- **src**: modeling_videomt.py
- **status**: composable
- **rationale**: Inference path is DINOv2/EoMT-style ViT encoder (separate Q/K/V attention + fc1+GELU+fc2 or SwiGLU MLP). Hungarian matcher uses scipy linear_sum_assignment but is training-loss only (decorated torch.no_grad and not on inference path). LayerScale, DropPath, LayerNorm2d are simple element-wise ops.
- **classes**:
  - **`VideomtPatchEmbeddings`** [compute]: `L1/conv2d.py` (Patch embedding via Conv2d.)
  - **`VideomtEmbeddings`** [wiring]: Wiring: patch + positional + cls token.
  - **`VideomtMLP`** [compute]: `L2/encoder_mlp.py` (fc1 + activation + fc2 (non-SwiGLU).)
  - **`VideomtGatedMLP`** [compute]: `L2/llama_mlp.py`, `L1/silu_and_mul.py` (SwiGLU pattern: weights_in -> chunk -> silu(x1)*x2 -> weights_out.)
  - **`VideomtAttention`** [compute]: `L2/encoder_attention.py` (Separate Q/K/V + SDPA, non-causal; matches encoder_attention.py.)
  - **`VideomtSwiGLUFFN`** [compute]: `L2/llama_mlp.py`, `L1/silu_and_mul.py` (SwiGLU FFN.)
  - **`VideomtLayer`** [wiring]: Wiring with norm + attn + layer_scale + drop_path + norm + mlp + layer_scale.
  - **`VideomtLayerScale`** [wiring]: Element-wise scaling parameter.
  - **`VideomtScaleLayer`** [compute]: `L1/conv_transpose2d.py`, `L1/conv2d.py` (ConvTranspose2d + Conv2d feature pyramid scaling.)
  - **`VideomtScaleBlock`** [wiring]: Wiring of ScaleLayer blocks.
  - **`VideomtMaskHead`** [compute]: `L1/linear.py`, `L1/gelu.py` (MLP for mask query embeddings.)
  - **`VideomtForUniversalSegmentation`** [wiring]: Wiring: backbone + scale block + mask head.

## vilt
- **src**: modeling_vilt.py
- **status**: composable
- **rationale**: BERT/ViT-style encoder for vision-and-language transformer (separate Q/K/V SDPA, fc1+GELU+fc2 MLP, LayerNorm). All primitives in kb-nano.
- **classes**:
  - **`ViltEmbeddings`** [wiring]: Wiring: patch + position + token-type + modality embeddings.
  - **`TextEmbeddings`** [compute]: `L2/bert_embeddings.py` (BERT-style word + position + token type embeddings + LayerNorm.)
  - **`ViltPatchEmbeddings`** [compute]: `L1/conv2d.py` (Patch embedding via Conv2d.)
  - **`ViltSelfAttention`** [compute]: `L2/encoder_attention.py` (Separate Q/K/V + SDPA, non-causal.)
  - **`ViltAttention`** [wiring]: Wiring: SelfAttention + SelfOutput.
  - **`ViltIntermediate`** [compute]: `L2/encoder_mlp.py` (Linear + GELU.)
  - **`ViltOutput`** [compute]: `L2/encoder_mlp.py` (Linear + dropout + residual (no LayerNorm in this variant).)
  - **`ViltLayer`** [wiring]: Wiring.
  - **`ViltEncoder`** [wiring]: Wiring.
  - **`ViltModel`** [wiring]: Wiring.
  - **`ViltPooler`** [compute]: `L1/linear.py`, `L1/tanh.py` (Linear + tanh on [CLS].)
  - **`ViltForMaskedLM`** [wiring]: Wiring.
  - **`ViltMLMHead`** [compute]: `L1/linear.py` (Prediction head.)

## vipllava
- **src**: modular_vipllava.py
- **status**: composable
- **rationale**: Pure wrapper: CLIP vision tower + multimodal projector (LayerNorm + Linear + act + Linear) + Llama LM. All primitives present.
- **classes**:
  - **`VipLlavaMultiModalProjector`** [compute]: `L1/layer_norm.py`, `L1/linear.py`, `L1/gelu.py` (LayerNorm + Linear + activation + Linear over concatenated multi-layer vision features.)
  - **`VipLlavaModel`** [wiring]: Wiring: vision_tower + projector + language_model.
  - **`VipLlavaForConditionalGeneration`** [wiring]: Wiring.

## vision_encoder_decoder
- **src**: modeling_vision_encoder_decoder.py
- **status**: composable
- **rationale**: Pure wrapper that composes any vision encoder (ViT-like) with any text decoder (BART/T5/GPT2-like). No own kernel.
- **classes**:
  - **`VisionEncoderDecoderModel`** [wiring]: Wiring around an external encoder + decoder.

## vision_text_dual_encoder
- **src**: modeling_vision_text_dual_encoder.py
- **status**: composable
- **rationale**: CLIP-style dual-encoder wrapper with logit_scale; composes any vision + text encoder. No own kernel.
- **classes**:
  - **`VisionTextDualEncoderModel`** [wiring]: Wiring.

## visual_bert
- **src**: modeling_visual_bert.py
- **status**: composable
- **rationale**: BERT-style encoder over text + visual feature embeddings. Standard BERT attention (separate Q/K/V SDPA) + fc1+GELU+fc2 + LayerNorm. RegionToPhraseAttention is a single-head bare matmul-softmax block; composable.
- **classes**:
  - **`VisualBertEmbeddings`** [compute]: `L2/bert_embeddings.py` (BERT-style token + position + token-type + visual embeddings.)
  - **`VisualBertSelfAttention`** [compute]: `L2/encoder_attention.py` (Standard BERT separate Q/K/V + SDPA.)
  - **`VisualBertAttention`** [wiring]: Wiring: SelfAttention + SelfOutput.
  - **`VisualBertIntermediate`** [compute]: `L2/encoder_mlp.py` (Linear + GELU.)
  - **`VisualBertOutput`** [compute]: `L2/encoder_mlp.py` (Linear + dropout + LayerNorm (residual).)
  - **`VisualBertLayer`** [wiring]: Wiring.
  - **`VisualBertEncoder`** [wiring]: Wiring.
  - **`VisualBertPooler`** [compute]: `L1/linear.py`, `L1/tanh.py` (Linear + tanh on [CLS].)
  - **`VisualBertPredictionHeadTransform`** [compute]: `L1/linear.py`, `L1/gelu.py`, `L1/layer_norm.py` (Linear + activation + LayerNorm.)
  - **`VisualBertLMPredictionHead`** [wiring]: Wiring.
  - **`VisualBertPreTrainingHeads`** [wiring]: Wiring.
  - **`VisualBertModel`** [wiring]: Wiring.
  - **`VisualBertRegionToPhraseAttention`** [compute]: `L1/linear.py`, `L1/sdpa.py` (Single-head matmul + softmax cross-attention block; composable from primitive ops.)

## vit
- **src**: modeling_vit.py
- **status**: composable
- **rationale**: Standard ViT: separate Q/K/V dense attention (encoder_attention.py), fc1+GELU+fc2 MLP (encoder_mlp.py), LayerNorm. Patch embed via Conv2d.
- **classes**:
  - **`ViTEmbeddings`** [wiring]: Wiring: patch embedding + cls token + position embeddings.
  - **`ViTPatchEmbeddings`** [compute]: `L1/conv2d.py` (Patch embedding via Conv2d.)
  - **`ViTSelfAttention`** [compute]: `L2/encoder_attention.py` (Separate Q/K/V + SDPA, non-causal; exactly matches EncoderSelfAttention.)
  - **`ViTAttention`** [wiring]: Wiring (sibling-class wrapper around SelfAttention).
  - **`ViTIntermediate`** [compute]: `L2/encoder_mlp.py` (Linear + GELU.)
  - **`ViTOutput`** [compute]: `L2/encoder_mlp.py` (Linear + dropout + residual (pre-norm style: layernorm is in ViTLayer).)
  - **`ViTLayer`** [wiring]: Wiring: pre-norm attn + pre-norm intermediate/output.
  - **`ViTEncoder`** [wiring]: Wiring.
  - **`ViTModel`** [wiring]: Wiring.
  - **`ViTPooler`** [compute]: `L1/linear.py`, `L1/tanh.py` (Linear + tanh on [CLS].)
  - **`ViTForMaskedImageModeling`** [wiring]: Wiring.
  - **`ViTForImageClassification`** [wiring]: Wiring.

## vit_mae
- **src**: modeling_vit_mae.py
- **status**: composable
- **rationale**: ViT-clone (separate Q/K/V SDPA, fc1+GELU+fc2 MLP) used for masked autoencoder pretraining. Decoder mirrors encoder structure with sin-cos pos embeddings.
- **classes**:
  - **`ViTMAEEmbeddings`** [wiring]: Wiring: patch embed + cls token + sincos pos embed + random masking.
  - **`ViTMAEPatchEmbeddings`** [compute]: `L1/conv2d.py` (Patch embedding via Conv2d.)
  - **`ViTMAESelfAttention`** [compute]: `L2/encoder_attention.py` (Separate Q/K/V + SDPA non-causal.)
  - **`ViTMAEAttention`** [wiring]: Wiring.
  - **`ViTMAEIntermediate`** [compute]: `L2/encoder_mlp.py` (Linear + GELU.)
  - **`ViTMAEOutput`** [compute]: `L2/encoder_mlp.py` (Linear + dropout + residual.)
  - **`ViTMAELayer`** [wiring]: Wiring.
  - **`ViTMAEEncoder`** [wiring]: Wiring.
  - **`ViTMAEModel`** [wiring]: Wiring.
  - **`ViTMAEDecoder`** [wiring]: Wiring of decoder layers.
  - **`ViTMAEForPreTraining`** [wiring]: Wiring.

## vit_msn
- **src**: modeling_vit_msn.py
- **status**: composable
- **rationale**: ViT-clone for self-supervised masked siamese networks; identical encoder structure to ViT (Copied-from comments confirm). All primitives in kb-nano.
- **classes**:
  - **`ViTMSNEmbeddings`** [wiring]: Wiring: patch + cls + position embeddings.
  - **`ViTMSNPatchEmbeddings`** [compute]: `L1/conv2d.py` (Patch embedding via Conv2d.)
  - **`ViTMSNSelfAttention`** [compute]: `L2/encoder_attention.py` (Separate Q/K/V + SDPA.)
  - **`ViTMSNAttention`** [wiring]: Wiring.
  - **`ViTMSNIntermediate`** [compute]: `L2/encoder_mlp.py` (Linear + GELU.)
  - **`ViTMSNOutput`** [compute]: `L2/encoder_mlp.py` (Linear + dropout + residual.)
  - **`ViTMSNLayer`** [wiring]: Wiring.
  - **`ViTMSNEncoder`** [wiring]: Wiring.
  - **`ViTMSNModel`** [wiring]: Wiring.
  - **`ViTMSNForImageClassification`** [wiring]: Wiring.

## vitdet
- **src**: modeling_vitdet.py
- **status**: composable
- **rationale**: Vit detection backbone with fused QKV + decomposed relative position bias + windowed attention + ResNet bottleneck blocks. Attention uses bare matmul/softmax (rel-pos bias). All ops are composable from kb-nano L1 (linear, sdpa, conv2d, layer_norm); no exact-match L2 wrapper exists for the rel-pos variant.
- **classes**:
  - **`VitDetEmbeddings`** [compute]: `L1/conv2d.py` (Patch embed via Conv2d + optional absolute positional embeddings.)
  - **`VitDetAttention`** [compute]: `L1/linear.py`, `L1/softmax.py` (Fused QKV via Linear + decomposed relative position bias added before softmax. No fused L2 module; composes from L1 primitives.)
  - **`VitDetLayerNorm`** [compute]: `L1/layer_norm2d.py` (Channel-axis LayerNorm for (N,C,H,W); kb-nano has L1/layer_norm2d.py.)
  - **`VitDetResBottleneckBlock`** [compute]: `L1/conv2d.py` (Three Conv2d + LayerNorm + activation residual block.)
  - **`VitDetMlp`** [compute]: `L2/encoder_mlp.py` (fc1 + activation + fc2.)
  - **`VitDetLayer`** [wiring]: Wiring: window-partition + attention + window-unpartition + MLP + residual.
  - **`VitDetEncoder`** [wiring]: Wiring.
  - **`VitDetModel`** [wiring]: Wiring.
  - **`VitDetBackbone`** [wiring]: Wiring.

## vitmatte
- **src**: modeling_vitmatte.py
- **status**: composable
- **rationale**: Image matting head: conv2d + batchnorm + relu + bilinear interpolate over a backbone's feature pyramid. All primitives in kb-nano.
- **classes**:
  - **`VitMatteBasicConv3x3`** [compute]: `L1/conv2d.py`, `L1/batch_norm2d.py`, `L1/relu.py` (Conv2d 3x3 + BatchNorm2d + ReLU.)
  - **`VitMatteConvStream`** [wiring]: Wiring of BasicConv3x3 stack.
  - **`VitMatteFusionBlock`** [compute]: `L1/interpolate.py` (Bilinear upsample + concat + conv block.)
  - **`VitMatteHead`** [compute]: `L1/conv2d.py` (Conv2d head.)
  - **`VitMatteDetailCaptureModule`** [wiring]: Wiring of conv stream + fusion blocks + head.
  - **`VitMatteForImageMatting`** [wiring]: Wiring.

## vitpose
- **src**: modeling_vitpose.py
- **status**: composable
- **rationale**: Pose estimation head over a ViT backbone. Decoders are Conv2d/ConvTranspose2d + BatchNorm + ReLU. All primitives present.
- **classes**:
  - **`VitPoseSimpleDecoder`** [compute]: `L1/conv2d.py`, `L1/interpolate.py`, `L1/relu.py` (Bilinear upsample + Conv2d + ReLU.)
  - **`VitPoseClassicDecoder`** [compute]: `L1/conv_transpose2d.py`, `L1/conv2d.py`, `L1/batch_norm2d.py`, `L1/relu.py` (ConvTranspose2d + BatchNorm + ReLU + Conv2d.)
  - **`VitPoseForPoseEstimation`** [wiring]: Wiring of backbone + decoder.

## vitpose_backbone
- **src**: modeling_vitpose_backbone.py
- **status**: composable
- **rationale**: ViT backbone with a NaiveMoE variant (mask-and-multiply per expert iteration). Standard ViT attention + fc1+act+fc2 MLP and an alternative MoE MLP. All primitives in kb-nano.
- **classes**:
  - **`VitPoseBackbonePatchEmbeddings`** [compute]: `L1/conv2d.py` (Conv2d patch embedding.)
  - **`VitPoseBackboneEmbeddings`** [wiring]: Wiring: patch + cls + position.
  - **`VitPoseBackboneSelfAttention`** [compute]: `L2/encoder_attention.py` (Separate Q/K/V SDPA.)
  - **`VitPoseBackboneAttention`** [wiring]: Wiring.
  - **`VitPoseNaiveMoe`** [compute]: `L1/linear.py` (Iterates over experts and masks; not a fused MoE kernel but composable from primitive Linear ops.)
  - **`VitPoseBackboneMoeMLP`** [wiring]: Wiring of fc1 + act + fc2 + NaiveMoe expert merge.
  - **`VitPoseBackboneMLP`** [compute]: `L2/encoder_mlp.py` (fc1 + activation + fc2.)
  - **`VitPoseBackboneLayer`** [wiring]: Wiring.
  - **`VitPoseBackboneEncoder`** [wiring]: Wiring.
  - **`VitPoseBackbone`** [wiring]: Wiring.

## vits
- **src**: modeling_vits.py
- **status**: partial
- **partial_reason**: All compute primitives (Conv1d, ConvTranspose1d, sigmoid, tanh, relu, leaky_relu, Linear, softmax) exist in kb-nano. However: (1) nn.utils.weight_norm parametrization on conv layers is a PyTorch-only utility with no kb-nano primitive; (2) fused_add_tanh_sigmoid_multiply is implemented as elementwise sigmoid*tanh in PyTorch with no fused kb-nano kernel; (3) the piecewise rational quadratic spline used by VitsConvFlow / VitsStochasticDurationPredictor is a long sequence of PyTorch elementwise ops; (4) VitsAttention's relative-position-to-absolute-position trick uses bare bmm/matmul + softmax with no flash-attn fast-path. Inference still works via the underlying primitives but several fused-op opportunities are unimplemented.
- **rationale**: VITS TTS pipeline composes WaveNet + HiFi-GAN + normalizing-flow coupling (residual coupling, dilated depth-separable conv, conv-flow piecewise rational quadratic spline) + relative-position attention. Most ops are conv1d/sigmoid/tanh, but the piecewise rational quadratic transform inside the stochastic duration predictor is bespoke flow math implemented entirely in PyTorch primitives.
- **classes**:
  - **`VitsPosteriorEncoder`** [compute]: no kb-nano kernel — All compute primitives (Conv1d, ConvTranspose1d, sigmoid, tanh, relu, leaky_relu, Linear, softmax) exist in kb-nano. However: (1) nn.utils.weight_norm parametrization on conv layers is a PyTorch-only 
  - **`VitsWaveNet`** [compute]: `L1/conv1d.py`, `L1/sigmoid.py`, `L1/tanh.py` (Stack of dilated Conv1d + fused_add_tanh_sigmoid_multiply (sigmoid(a)*tanh(b)) gating; weight_norm-wrapped convs.)
  - **`HifiGanResidualBlock`** [compute]: `L1/conv1d.py`, `L1/leaky_relu.py` (Conv1d + leaky_relu residual block (weight_norm-wrapped).)
  - **`VitsHifiGan`** [compute]: `L1/conv1d.py`, `L1/conv_transpose1d.py`, `L1/leaky_relu.py`, `L1/tanh.py` (HiFi-GAN vocoder with ConvTranspose1d upsamples + residual blocks + tanh output (weight_norm-wrapped).)
  - **`VitsResidualCouplingLayer`** [wiring]: Affine coupling using WaveNet; bespoke normalizing-flow algebra with chunk + WaveNet + scale/translate.
  - **`VitsResidualCouplingBlock`** [wiring]: Wiring of coupling flows.
  - **`VitsDilatedDepthSeparableConv`** [compute]: `L1/conv1d.py`, `L1/layer_norm.py`, `L1/gelu.py` (Depth-separable Conv1d + LayerNorm + GELU.)
  - **`VitsConvFlow`** [wiring]: Conv-based input to a piecewise rational quadratic spline transform; spline math implemented in pure PyTorch elementwise ops with no kb-nano kernel.
  - **`VitsElementwiseAffine`** [wiring]: x * exp(scale) + translate flow.
  - **`VitsStochasticDurationPredictor`** [wiring]: Stochastic flow-based duration predictor using piecewise rational quadratic spline; bespoke math in PyTorch.
  - **`VitsDurationPredictor`** [compute]: `L1/conv1d.py`, `L1/relu.py`, `L1/layer_norm.py` (Conv1d + ReLU + LayerNorm + Conv1d projection.)
  - **`VitsAttention`** [compute]: `L1/linear.py`, `L1/softmax.py` (Separate Q/K/V + relative position embeddings (custom relative<->absolute index trick) + bmm-softmax-bmm. No flash-attn path because of rel-pos addition.)
  - **`VitsFeedForward`** [compute]: `L1/conv1d.py` (Conv1d-based FFN with explicit padding.)
  - **`VitsEncoderLayer`** [wiring]: Wiring.
  - **`VitsEncoder`** [wiring]: Wiring.
  - **`VitsTextEncoder`** [wiring]: Wiring.
  - **`VitsModel`** [wiring]: Wiring: text encoder + duration predictor + flow + posterior encoder + HiFi-GAN.

## vivit
- **src**: modeling_vivit.py
- **status**: composable
- **rationale**: Video Vision Transformer = ViT-clone over tubelet patch embeddings. Same separate Q/K/V SDPA + fc1+GELU+fc2 MLP + LayerNorm. All primitives present.
- **classes**:
  - **`VivitTubeletEmbeddings`** [compute]: `L1/conv3d.py` (Tubelet patch embedding via Conv3d.)
  - **`VivitEmbeddings`** [wiring]: Wiring.
  - **`VivitSelfAttention`** [compute]: `L2/encoder_attention.py` (Separate Q/K/V SDPA.)
  - **`VivitAttention`** [wiring]: Wiring.
  - **`VivitIntermediate`** [compute]: `L2/encoder_mlp.py` (Linear + GELU.)
  - **`VivitOutput`** [compute]: `L2/encoder_mlp.py` (Linear + dropout + residual.)
  - **`VivitLayer`** [wiring]: Wiring.
  - **`VivitEncoder`** [wiring]: Wiring.
  - **`VivitPooler`** [compute]: `L1/linear.py`, `L1/tanh.py` (Linear + tanh on [CLS].)
  - **`VivitModel`** [wiring]: Wiring.
  - **`VivitForVideoClassification`** [wiring]: Wiring.

## vjepa2
- **src**: modeling_vjepa2.py
- **status**: kb_nano_l4
- **rationale**: Existing L4 pipeline tasks/baseline/L4/vjepa2.py targets V-JEPA 2; uses L2/vjepa2_embeddings + L3/vjepa2_layer + L3/vjepa2_pooler + L3/vjepa2_predictor.
- **classes**:
  - **`VJEPA2PatchEmbeddings3D`** [compute]: `L2/vjepa2_embeddings.py`, `L1/conv3d.py` (3D patch embedding; matches L2/vjepa2_embeddings.py:VJEPA2PatchEmbeddings3D.)
  - **`VJEPA2Embeddings`** [compute]: `L2/vjepa2_embeddings.py` (Embedding wrapper; matches L2/vjepa2_embeddings.py:VJEPA2Embeddings.)
  - **`VJEPA2RopeAttention`** [compute]: `L2/vjepa2_attention.py`, `L1/vjepa2_rope.py` (Multi-head attention with V-JEPA 2 spatio-temporal RoPE.)
  - **`VJEPA2MLP`** [compute]: `L2/vjepa2_mlp.py` (fc1 + activation + fc2.)
  - **`VJEPA2Layer`** [compute]: `L3/vjepa2_layer.py` (Encoder layer; norm + attn + mlp.)
  - **`VJEPA2Encoder`** [wiring]: Wiring of layer stack.
  - **`VJEPA2PredictorEmbeddings`** [wiring]: Wiring.
  - **`VJEPA2Predictor`** [compute]: `L3/vjepa2_predictor.py` (Predictor module covered by L3/vjepa2_predictor.py.)
  - **`VJEPA2PoolerSelfAttention`** [compute]: `L2/vjepa2_attention.py` (Self-attention used in attentive pooler.)
  - **`VJEPA2PoolerCrossAttention`** [compute]: `L2/vjepa2_attention.py` (Cross-attention used in attentive pooler.)
  - **`VJEPA2PoolerSelfAttentionLayer`** [wiring]: Wiring.
  - **`VJEPA2PoolerCrossAttentionLayer`** [wiring]: Wiring.
  - **`VJEPA2AttentivePooler`** [compute]: `L3/vjepa2_pooler.py` (AttentivePooler covered by L3/vjepa2_pooler.py.)
  - **`VJEPA2Model`** [wiring]: Wiring assembled in L4/vjepa2.py.
  - **`VJEPA2ForVideoClassification`** [wiring]: Wiring.

## voxtral
- **src**: modular_voxtral.py
- **status**: composable
- **rationale**: Voxtral = Whisper-style audio encoder (Qwen2Audio-derived) + multimodal projector + Llama LM. Whisper attention + Whisper MLP + Llama LM all available in kb-nano.
- **classes**:
  - **`VoxtralAttention`** [compute]: `L2/whisper_attention.py` (Whisper-family encoder self-attention.)
  - **`VoxtralEncoderLayer`** [wiring]: Wiring of encoder layer.
  - **`VoxtralEncoder`** [compute]: `L1/conv1d.py`, `L1/gelu.py` (Wiring: Conv1d + GELU stem + position embeddings + transformer layers.)
  - **`VoxtralMultiModalProjector`** [compute]: `L1/linear.py`, `L1/gelu.py` (Linear + activation + Linear projector.)
  - **`VoxtralForConditionalGeneration`** [wiring]: Wiring: encoder + projector + Llama LM.

## voxtral_realtime
- **src**: modular_voxtral_realtime.py
- **status**: composable
- **rationale**: Voxtral-realtime = streaming audio encoder (CausalConv1d + Mistral-style attention with biased Q/V) + Mistral-derived text decoder with AdaRMSNorm time-conditioning. All compute primitives (linear, rms_norm, conv1d with causal pad, RoPE, GELU, GQA attention) are in kb-nano.
- **classes**:
  - **`VoxtralRealtimeRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (Standard Llama-style RoPE.)
  - **`VoxtralRealtimeCausalConv1d`** [compute]: `L1/conv1d.py`, `L1/causal_conv1d.py` (Conv1d with left causal padding.)
  - **`VoxtralRealtimeRMSNorm`** [compute]: `L1/rms_norm.py` (Standard Llama-family RMSNorm.)
  - **`VoxtralRealtimeAttention`** [compute]: `L2/attention.py` (GQA attention with Q/V having bias and K not (Whisper-style); LlamaAttention covers GQA, biases are config flags.)
  - **`VoxtralRealtimeMLP`** [compute]: `L2/llama_mlp.py`, `L1/silu_and_mul.py` (SwiGLU MLP with bias on down_proj.)
  - **`VoxtralRealtimeEmbedder`** [wiring]: Wiring: 2x CausalConv1d + GELU.
  - **`VoxtralRealtimeEncoderLayer`** [wiring]: Wiring: pre-norm self-attn + pre-norm MLP.
  - **`VoxtralRealtimeEncoder`** [wiring]: Wiring: embedder + layer stack + final norm + RoPE.
  - **`VoxtralRealtimeTextAdaRmsNorm`** [compute]: `L1/linear.py`, `L1/gelu.py` (Linear + GELU + Linear modulation conditioning network.)
  - **`VoxtralRealtimeTextAttention`** [compute]: `L2/attention.py` (Standard Mistral GQA attention.)
  - **`VoxtralRealtimeTextMLP`** [compute]: `L2/llama_mlp.py`, `L1/silu_and_mul.py` (SwiGLU MLP.)
  - **`VoxtralRealtimeTextDecoderLayer`** [wiring]: Wiring with extra t_cond modulation via ada_rms_norm.
  - **`VoxtralRealtimeTextModel`** [wiring]: Wiring.
  - **`VoxtralRealtimeTextForCausalLM`** [wiring]: Wiring.
  - **`VoxtralRealtimeTimeEmbedding`** [compute]: `L1/sinusoidal_embed.py` (Sinusoidal time embedding.)
  - **`VoxtralRealtimeMultiModalProjector`** [compute]: `L1/linear.py`, `L1/gelu.py` (Linear + activation + Linear.)
  - **`VoxtralRealtimeForConditionalGeneration`** [wiring]: Wiring.
