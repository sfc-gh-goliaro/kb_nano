# Manual audit shard 15 — video_llama_3 through xcodec

## video_llama_3
- **src**: modeling_video_llama_3.py, modular_video_llama_3.py
- **hidden_act**: gelu_pytorch_tanh (vision); text uses Qwen2 (silu)
- **status**: composable
- **classes**:
  - **`VideoLlama3VisionRotaryEmbedding`** [compute]: builds 2D vision rope cos/sin from grid_thw. Closest kb-nano: `L1/vision_rotary_emb.py` (no exact match — uses custom pixel_unshuffle-aware grid)
  - **`VideoLlama3VisionEmbeddings`** [compute]: `L1/conv2d.py` (patch_embedding, no position embedding here)
  - **`VideoLlama3VisionMLP`** [compute, modular inherits `SiglipMLP`]: `L2/siglip_mlp.py` (fc1 + gelu_pytorch_tanh + fc2)
  - **`VideoLlama3VisionAttention`** [compute, modular inherits `SiglipAttention`]: `L2/vision_attention.py` (variable-length cu_seqlens, vision rope, non-causal — closer to Qwen2VL vision attention than plain SigLIP)
  - **`VideoLlama3VisionEncoderLayer`** [wiring]: wires `VideoLlama3VisionAttention`, `VideoLlama3VisionMLP`; direct `L1/layer_norm.py` (×2)
  - **`VideoLlama3VisionEncoder`** [wiring]: wires `VideoLlama3VisionEncoderLayer`
  - **`VideoLlama3VisionModel`** [wiring]: wires `VideoLlama3VisionRotaryEmbedding`, `VideoLlama3VisionEmbeddings`, `VideoLlama3VisionEncoder`; direct `L1/layer_norm.py` (post_layernorm), pixel_unshuffle uses interpolate (no exact L1)
  - **`VideoLlama3Projector`** [compute]: `L1/linear.py + L1/gelu.py + L1/linear.py` (nn.Sequential of Linear, GELU, Linear)
  - **`VideoLlama3Model`** [wiring]: wires vision_model (AutoModel), `VideoLlama3Projector`, language_model (AutoModel — Qwen2)
  - **`VideoLlama3ForConditionalGeneration`** [wiring]: wires `VideoLlama3Model`; direct `L1/linear.py` (lm_head)

## video_llava
- **src**: modeling_video_llava.py
- **hidden_act**: projector_hidden_act=gelu (vision tower CLIP); text uses LLaMA (silu)
- **status**: composable
- **classes**:
  - **`VideoLlavaMultiModalProjector`** [compute]: `L1/linear.py + L1/gelu.py + L1/linear.py`
  - **`VideoLlavaModel`** [wiring]: wires video_tower, image_tower (AutoModel — CLIP), `VideoLlavaMultiModalProjector`, language_model (AutoModel — LLaMA)
  - **`VideoLlavaForConditionalGeneration`** [wiring]: wires `VideoLlavaModel`; direct `L1/linear.py` (lm_head)

## videomae
- **src**: modeling_videomae.py
- **hidden_act**: gelu
- **status**: composable
- **classes**:
  - **`VideoMAEEmbeddings`** [compute]: wires `VideoMAEPatchEmbeddings`; adds fixed sinusoidal position embedding (no L1 — buffer add) and applies bool_masked_pos masking
  - **`VideoMAEPatchEmbeddings`** [compute]: `L1/conv3d.py` (Conv3d projection over tubelets)
  - **`VideoMAESelfAttention`** [compute]: `L2/encoder_attention.py` (q/k/v + dispatch via ALL_ATTENTION_FUNCTIONS, non-causal — ViT-style)
  - **`VideoMAESelfOutput`** [compute]: `L1/linear.py` (dense; residual added in parent layer)
  - **`VideoMAEAttention`** [wiring]: wires `VideoMAESelfAttention`, `VideoMAESelfOutput`
  - **`VideoMAEIntermediate`** [compute]: `L1/linear.py + L1/gelu.py` (no exact L2 match — half of an encoder MLP)
  - **`VideoMAEOutput`** [compute]: `L1/linear.py` (dense + residual)
  - **`VideoMAELayer`** [wiring]: wires `VideoMAEAttention`, `VideoMAEIntermediate`, `VideoMAEOutput`; direct `L1/layer_norm.py` (×2)
  - **`VideoMAEEncoder`** [wiring]: wires `VideoMAELayer`
  - **`VideoMAEModel`** [wiring]: wires `VideoMAEEmbeddings`, `VideoMAEEncoder`; direct `L1/layer_norm.py` (optional)
  - **`VideoMAEDecoder`** [wiring]: wires `VideoMAELayer`; direct `L1/layer_norm.py`, `L1/linear.py` (head)
  - **`VideoMAEForPreTraining`** [wiring]: wires `VideoMAEModel`, `VideoMAEDecoder`; direct `L1/linear.py` (encoder_to_decoder)
- **task heads (1)**: ForVideoClassification — base + linear (per-task)

## videomt
- **src**: modeling_videomt.py, modular_videomt.py
- **hidden_act**: gelu
- **status**: composable
- **classes**:
  - **`VideomtPatchEmbeddings`** [compute]: `L1/conv2d.py`
  - **`VideomtEmbeddings`** [wiring]: wires `VideomtPatchEmbeddings`; direct `L1/embedding.py` (position_embeddings), parameter buffers (cls/register/mask tokens)
  - **`VideomtMLP`** [compute]: `L1/linear.py + L1/gelu.py + L1/linear.py` (2-layer fc, no exact L2 — close to encoder_mlp but split differently)
  - **`VideomtGatedMLP`** [compute]: SwiGLU pattern (silu(x1)*x2) — `L2/llama_mlp.py` analog (single weights_in chunked)
  - **`VideomtAttention`** [compute]: `L2/encoder_attention.py` (q/k/v + dispatch via ALL_ATTENTION_FUNCTIONS, non-causal)
  - **`VideomtDropPath`** [compute]: stochastic depth (no exact L1 — bernoulli mask)
  - **`VideomtSwiGLUFFN`** [compute]: `L2/llama_mlp.py` analog (silu(x1)*x2 SwiGLU)
  - **`VideomtLayer`** [wiring]: wires `VideomtAttention`, `VideomtMLP`/`VideomtSwiGLUFFN`, `VideomtLayerScale` (×2), `VideomtDropPath`; direct `L1/layer_norm.py` (×2)
  - **`VideomtLayerScale`** [compute]: `L1/tensor_ops.py` (scalar mul by learnable lambda1)
  - **`VideomtLayerNorm2d`** [compute]: `L1/layer_norm.py` (with permutes for NCHW)
  - **`VideomtScaleLayer`** [compute]: `L1/conv_transpose2d.py + L1/gelu.py + L1/conv2d.py` + `VideomtLayerNorm2d`
  - **`VideomtScaleBlock`** [wiring]: wires `VideomtScaleLayer`
  - **`VideomtMaskHead`** [compute]: `L1/linear.py + L1/gelu.py + L1/linear.py + L1/gelu.py + L1/linear.py` (3-layer MLP)
  - **`VideomtForUniversalSegmentation`** [wiring]: wires `VideomtEmbeddings`, `VideomtLayer` (×N), `VideomtScaleBlock`, `VideomtMaskHead`; direct `L1/layer_norm.py`, `L1/embedding.py` (query), `L1/linear.py` (class_predictor, query_updater)
- **task heads (1)**: ForUniversalSegmentation — base + linear (per-task)

## vilt
- **src**: modeling_vilt.py
- **hidden_act**: gelu
- **status**: composable
- **classes**:
  - **`ViltEmbeddings`** [wiring]: wires `TextEmbeddings`, `ViltPatchEmbeddings`; direct `L1/embedding.py` (token_type_embeddings); cls_token+position_embeddings as parameters; visual_embed uses interpolate (no exact L1)
  - **`TextEmbeddings`** [compute]: `L2/encoder_embeddings.py` (BERT-style word + position + token_type + LayerNorm + Dropout)
  - **`ViltPatchEmbeddings`** [compute]: `L1/conv2d.py`
  - **`ViltSelfAttention`** [compute]: `L2/encoder_attention.py` (q/k/v + softmax, BERT-style; non-causal)
  - **`ViltSelfOutput`** [compute]: `L1/linear.py` (dense; residual in parent)
  - **`ViltAttention`** [wiring]: wires `ViltSelfAttention`, `ViltSelfOutput`
  - **`ViltIntermediate`** [compute]: `L1/linear.py + L1/gelu.py`
  - **`ViltOutput`** [compute]: `L1/linear.py` (dense + residual)
  - **`ViltLayer`** [wiring]: wires `ViltAttention`, `ViltIntermediate`, `ViltOutput`; direct `L1/layer_norm.py` (×2)
  - **`ViltEncoder`** [wiring]: wires `ViltLayer`
  - **`ViltModel`** [wiring]: wires `ViltEmbeddings`, `ViltEncoder`, `ViltPooler`; direct `L1/layer_norm.py`
  - **`ViltPooler`** [compute]: `L1/linear.py + L1/tanh.py`
  - **`ViltPredictionHeadTransform`** [compute]: `L1/linear.py + L1/gelu.py + L1/layer_norm.py`
  - **`ViltMLMHead`** [wiring]: wires `ViltPredictionHeadTransform`; direct `L1/linear.py`
  - **`ViltForMaskedLM`** [wiring]: wires `ViltModel`, `ViltMLMHead`
- **task heads (4)**: ForQuestionAnswering, ForImageAndTextRetrieval, ForImagesAndTextClassification, ForTokenClassification — base + linear (per-task)

## vipllava
- **src**: modeling_vipllava.py, modular_vipllava.py
- **hidden_act**: projector_hidden_act=gelu (vision tower CLIP uses quickgelu); text uses LLaMA (silu)
- **status**: composable
- **classes**:
  - **`VipLlavaMultiModalProjector`** [compute]: `L1/layer_norm.py + L1/linear.py + L1/gelu.py + L1/linear.py`
  - **`VipLlavaModel`** [wiring]: wires vision_tower (AutoModel — CLIP), `VipLlavaMultiModalProjector`, language_model (AutoModel — LLaMA)
  - **`VipLlavaForConditionalGeneration`** [wiring]: wires `VipLlavaModel`; direct `L1/linear.py` (lm_head)

## vision_encoder_decoder
- **src**: modeling_vision_encoder_decoder.py
- **hidden_act**: N/A (generic wiring class — encoder/decoder configs vary)
- **status**: composable
- **classes**:
  - **`VisionEncoderDecoderModel`** [wiring]: wires encoder (AutoModel — vision encoder), decoder (AutoModelForCausalLM); direct `L1/linear.py` (optional enc_to_dec_proj when hidden_sizes differ)

## vision_text_dual_encoder
- **src**: modeling_vision_text_dual_encoder.py
- **hidden_act**: N/A (generic wiring class)
- **status**: composable
- **classes**:
  - **`VisionTextDualEncoderModel`** [wiring]: wires vision_model (AutoModel/CLIPVisionModel), text_model (AutoModel); direct `L1/linear.py` (visual_projection, text_projection); learnable scalar logit_scale

## visual_bert
- **src**: modeling_visual_bert.py
- **hidden_act**: gelu
- **status**: composable
- **classes**:
  - **`VisualBertEmbeddings`** [compute]: `L2/encoder_embeddings.py` (BERT-style word + position + token_type + LayerNorm) + extra `L1/embedding.py` (visual_token_type_embeddings, visual_position_embeddings) + `L1/linear.py` (visual_projection)
  - **`VisualBertSelfAttention`** [compute]: `L2/encoder_attention.py` (q/k/v + softmax, BERT-style; non-causal)
  - **`VisualBertSelfOutput`** [compute]: `L2/encoder_attention.py` (dense + LayerNorm + residual)
  - **`VisualBertAttention`** [wiring]: wires `VisualBertSelfAttention`, `VisualBertSelfOutput`
  - **`VisualBertIntermediate`** [compute]: `L1/linear.py + L1/gelu.py`
  - **`VisualBertOutput`** [compute]: `L1/linear.py + L1/layer_norm.py` (dense + LayerNorm + residual; encoder_mlp pattern's second half)
  - **`VisualBertLayer`** [wiring]: wires `VisualBertAttention`, `VisualBertIntermediate`, `VisualBertOutput`
  - **`VisualBertEncoder`** [wiring]: wires `VisualBertLayer`
  - **`VisualBertPooler`** [compute]: `L1/linear.py + L1/tanh.py`
  - **`VisualBertPredictionHeadTransform`** [compute]: `L1/linear.py + L1/gelu.py + L1/layer_norm.py`
  - **`VisualBertLMPredictionHead`** [wiring]: wires `VisualBertPredictionHeadTransform`; direct `L1/linear.py` (decoder)
  - **`VisualBertPreTrainingHeads`** [wiring]: wires `VisualBertLMPredictionHead`; direct `L1/linear.py` (seq_relationship)
  - **`VisualBertModel`** [wiring]: wires `VisualBertEmbeddings`, `VisualBertEncoder`, `VisualBertPooler`
  - **`VisualBertForPreTraining`** [wiring]: wires `VisualBertModel`, `VisualBertPreTrainingHeads`
  - **`VisualBertRegionToPhraseAttention`** [compute]: `L1/linear.py` (q/k) + manual softmax-less score (no exact L2 — single-head additive mask)
  - **`VisualBertForRegionToPhraseAlignment`** [wiring]: wires `VisualBertModel`, `VisualBertPreTrainingHeads`, `VisualBertRegionToPhraseAttention`
- **task heads (3)**: ForMultipleChoice, ForQuestionAnswering, ForVisualReasoning — base + linear (per-task)

## vit
- **src**: modeling_vit.py
- **hidden_act**: gelu
- **status**: composable
- **classes**:
  - **`ViTEmbeddings`** [compute]: wires `ViTPatchEmbeddings`; cls_token + position_embeddings parameters; supports interpolate (no exact L2 — partial match to `L2/vision_patch_embed.py`)
  - **`ViTPatchEmbeddings`** [compute]: `L1/conv2d.py`
  - **`ViTSelfAttention`** [compute]: `L2/encoder_attention.py` (q/k/v + dispatch via ALL_ATTENTION_FUNCTIONS, non-causal — vit_encoder_attention.py is closer match)
  - **`ViTSelfOutput`** [compute]: `L1/linear.py` (dense; residual in parent)
  - **`ViTAttention`** [wiring]: wires `ViTSelfAttention`, `ViTSelfOutput`
  - **`ViTIntermediate`** [compute]: `L1/linear.py + L1/gelu.py`
  - **`ViTOutput`** [compute]: `L1/linear.py` (dense + residual)
  - **`ViTLayer`** [wiring]: wires `ViTAttention`, `ViTIntermediate`, `ViTOutput`; direct `L1/layer_norm.py` (×2). Closer composite: `L3/vit_encoder_block.py`
  - **`ViTEncoder`** [wiring]: wires `ViTLayer`
  - **`ViTModel`** [wiring]: wires `ViTEmbeddings`, `ViTEncoder`, optional `ViTPooler`; direct `L1/layer_norm.py`
  - **`ViTPooler`** [compute]: `L1/linear.py + L1/tanh.py` (pooler_act=tanh by default)
  - **`ViTForMaskedImageModeling`** [wiring]: wires `ViTModel`; direct `L1/conv2d.py` + PixelShuffle (no exact L1 for PixelShuffle)
- **task heads (1)**: ForImageClassification — base + linear (per-task)

## vit_mae
- **src**: modeling_vit_mae.py
- **hidden_act**: gelu
- **status**: composable
- **classes**:
  - **`ViTMAEEmbeddings`** [compute]: wires `ViTMAEPatchEmbeddings`; cls_token + position_embeddings parameters; random_masking via argsort/gather; supports interpolate_pos_encoding (no exact L2)
  - **`ViTMAEPatchEmbeddings`** [compute]: `L1/conv2d.py`
  - **`ViTMAESelfAttention`** [compute]: `L2/encoder_attention.py` (q/k/v + dispatch via ALL_ATTENTION_FUNCTIONS, non-causal)
  - **`ViTMAESelfOutput`** [compute]: `L1/linear.py`
  - **`ViTMAEAttention`** [wiring]: wires `ViTMAESelfAttention`, `ViTMAESelfOutput`
  - **`ViTMAEIntermediate`** [compute]: `L1/linear.py + L1/gelu.py`
  - **`ViTMAEOutput`** [compute]: `L1/linear.py` (dense + residual)
  - **`ViTMAELayer`** [wiring]: wires `ViTMAEAttention`, `ViTMAEIntermediate`, `ViTMAEOutput`; direct `L1/layer_norm.py` (×2)
  - **`ViTMAEEncoder`** [wiring]: wires `ViTMAELayer`
  - **`ViTMAEModel`** [wiring]: wires `ViTMAEEmbeddings`, `ViTMAEEncoder`; direct `L1/layer_norm.py`
  - **`ViTMAEDecoder`** [wiring]: wires `ViTMAELayer` (×N); direct `L1/linear.py` (decoder_embed, decoder_pred), `L1/layer_norm.py` (decoder_norm); mask_token + decoder_pos_embed parameters
  - **`ViTMAEForPreTraining`** [wiring]: wires `ViTMAEModel`, `ViTMAEDecoder`

## vit_msn
- **src**: modeling_vit_msn.py
- **hidden_act**: gelu
- **status**: composable
- **classes**:
  - **`ViTMSNEmbeddings`** [compute]: wires `ViTMSNPatchEmbeddings`; cls_token + position_embeddings parameters; supports interpolate_pos_encoding (no exact L2 — same shape as ViTEmbeddings)
  - **`ViTMSNPatchEmbeddings`** [compute]: `L1/conv2d.py`
  - **`ViTMSNSelfAttention`** [compute]: `L2/encoder_attention.py` (q/k/v + dispatch, non-causal — ViT-style)
  - **`ViTMSNSelfOutput`** [compute]: `L1/linear.py`
  - **`ViTMSNAttention`** [wiring]: wires `ViTMSNSelfAttention`, `ViTMSNSelfOutput`
  - **`ViTMSNIntermediate`** [compute]: `L1/linear.py + L1/gelu.py`
  - **`ViTMSNOutput`** [compute]: `L1/linear.py` (dense + residual)
  - **`ViTMSNLayer`** [wiring]: wires `ViTMSNAttention`, `ViTMSNIntermediate`, `ViTMSNOutput`; direct `L1/layer_norm.py` (×2)
  - **`ViTMSNEncoder`** [wiring]: wires `ViTMSNLayer`
  - **`ViTMSNModel`** [wiring]: wires `ViTMSNEmbeddings`, `ViTMSNEncoder`; direct `L1/layer_norm.py`
- **task heads (1)**: ForImageClassification — base + linear (per-task)

## vitdet
- **src**: modeling_vitdet.py
- **hidden_act**: gelu
- **status**: composable
- **classes**:
  - **`VitDetEmbeddings`** [compute]: `L1/conv2d.py` (projection); position_embeddings parameter; uses interpolate (no exact L2)
  - **`VitDetAttention`** [compute]: fused qkv linear + manual softmax + decomposed relative-position embeddings (rel_pos_h, rel_pos_w einsum) — no exact L2 match (specialized window-attention with rel-pos); decomposes to `L1/linear.py + L1/softmax.py` plus custom rel-pos bias (no L1)
  - **`VitDetDropPath`** [compute]: stochastic depth (no exact L1)
  - **`VitDetLayerNorm`** [compute]: `L1/layer_norm.py` (channel-wise variant — close to layer_norm2d.py)
  - **`VitDetResBottleneckBlock`** [compute]: `L1/conv2d.py` (×3) + `VitDetLayerNorm` (×3) + `L1/gelu.py` (×2) + residual
  - **`VitDetMlp`** [compute]: `L1/linear.py + L1/gelu.py + L1/linear.py` (2-layer MLP)
  - **`VitDetLayer`** [wiring]: wires `VitDetAttention`, `VitDetMlp`, `VitDetDropPath`, optional `VitDetResBottleneckBlock`; direct `L1/layer_norm.py` (×2); window_partition/unpartition (no exact L1)
  - **`VitDetEncoder`** [wiring]: wires `VitDetLayer`
  - **`VitDetModel`** [wiring]: wires `VitDetEmbeddings`, `VitDetEncoder`
  - **`VitDetBackbone`** [wiring]: wires `VitDetEmbeddings`, `VitDetEncoder`; direct `L1/layer_norm.py` per output

## vitmatte
- **src**: modeling_vitmatte.py
- **hidden_act**: N/A (uses ReLU directly; backbone has its own hidden_act)
- **status**: composable
- **classes**:
  - **`VitMatteBasicConv3x3`** [compute]: `L1/conv2d.py + L1/batch_norm2d.py + L1/relu.py`
  - **`VitMatteConvStream`** [wiring]: wires `VitMatteBasicConv3x3` (×N)
  - **`VitMatteFusionBlock`** [compute]: interpolate + cat + `VitMatteBasicConv3x3` (no exact L1 for interpolate)
  - **`VitMatteHead`** [compute]: `L1/conv2d.py + L1/batch_norm2d.py + L1/relu.py + L1/conv2d.py` (Sequential matting_convs)
  - **`VitMatteDetailCaptureModule`** [wiring]: wires `VitMatteConvStream`, `VitMatteFusionBlock` (×N), `VitMatteHead`; direct sigmoid on output
  - **`VitMatteForImageMatting`** [wiring]: wires backbone (load_backbone), `VitMatteDetailCaptureModule`

## vitpose
- **src**: modeling_vitpose.py
- **hidden_act**: N/A (ReLU directly; backbone has its own)
- **status**: composable
- **classes**:
  - **`VitPoseSimpleDecoder`** [compute]: `L1/relu.py + L1/conv2d.py` (Upsample = interpolate, no exact L1)
  - **`VitPoseClassicDecoder`** [compute]: `L1/conv_transpose2d.py + L1/batch_norm2d.py + L1/relu.py + L1/conv_transpose2d.py + L1/batch_norm2d.py + L1/relu.py + L1/conv2d.py`
  - **`VitPoseForPoseEstimation`** [wiring]: wires backbone (load_backbone), `VitPoseSimpleDecoder`/`VitPoseClassicDecoder`

## vitpose_backbone
- **src**: modeling_vitpose_backbone.py
- **hidden_act**: gelu
- **status**: composable
- **classes**:
  - **`VitPoseBackbonePatchEmbeddings`** [compute]: `L1/conv2d.py`
  - **`VitPoseBackboneEmbeddings`** [compute]: wires `VitPoseBackbonePatchEmbeddings`; cls_token + position_embeddings parameters
  - **`VitPoseBackboneSelfAttention`** [compute]: `L2/encoder_attention.py` (q/k/v + dispatch via ALL_ATTENTION_FUNCTIONS, non-causal — ViT-style)
  - **`VitPoseBackboneSelfOutput`** [compute]: `L1/linear.py`
  - **`VitPoseBackboneAttention`** [wiring]: wires `VitPoseBackboneSelfAttention`, `VitPoseBackboneSelfOutput`
  - **`VitPoseNaiveMoe`** [compute]: `L1/linear.py` (×N experts) + manual selection by indices (no exact MoE L1 match — naive expert routing)
  - **`VitPoseBackboneMoeMLP`** [compute]: `L1/linear.py + L1/gelu.py + L1/linear.py` + `VitPoseNaiveMoe` + cat
  - **`VitPoseBackboneMLP`** [compute]: `L1/linear.py + L1/gelu.py + L1/linear.py` (2-layer MLP)
  - **`VitPoseBackboneLayer`** [wiring]: wires `VitPoseBackboneAttention`, `VitPoseBackboneMLP`/`VitPoseBackboneMoeMLP`; direct `L1/layer_norm.py` (×2)
  - **`VitPoseBackboneEncoder`** [wiring]: wires `VitPoseBackboneLayer`
  - **`VitPoseBackbone`** [wiring]: wires `VitPoseBackboneEmbeddings`, `VitPoseBackboneEncoder`

## vits
- **src**: modeling_vits.py
- **hidden_act**: relu (for VitsFeedForward)
- **status**: composable
- **classes**:
  - **`VitsWaveNet`** [compute]: `L1/conv1d.py` (×N in_layers, res_skip_layers; weight_norm — no exact L1) + custom fused_add_tanh_sigmoid_multiply (no L1)
  - **`VitsPosteriorEncoder`** [wiring]: wires `VitsWaveNet`; direct `L1/conv1d.py` (conv_pre, conv_proj)
  - **`HifiGanResidualBlock`** [compute]: `L1/conv1d.py` (×N convs1, ×N convs2) + `L1/leaky_relu.py` (×2N) + residual
  - **`VitsHifiGan`** [wiring]: wires `HifiGanResidualBlock` (×N); direct `L1/conv1d.py` (conv_pre, conv_post, optional cond), `L1/conv_transpose1d.py` (×N upsampler), `L1/leaky_relu.py`, `L1/tanh.py`
  - **`VitsResidualCouplingLayer`** [wiring]: wires `VitsWaveNet`; direct `L1/conv1d.py` (conv_pre, conv_post)
  - **`VitsResidualCouplingBlock`** [wiring]: wires `VitsResidualCouplingLayer` (×N flows)
  - **`VitsDilatedDepthSeparableConv`** [compute]: `L1/conv1d.py` (×N dilated, ×N pointwise) + `L1/layer_norm.py` (×2N) + `L1/gelu.py` (×2N)
  - **`VitsConvFlow`** [wiring]: wires `VitsDilatedDepthSeparableConv`; direct `L1/conv1d.py` (conv_pre, conv_proj); rational_quadratic_spline (no L1)
  - **`VitsElementwiseAffine`** [compute]: scalar affine + log-det (no exact L1 — translate/log_scale parameters)
  - **`VitsStochasticDurationPredictor`** [wiring]: wires `VitsDilatedDepthSeparableConv`, `VitsConvFlow`, `VitsElementwiseAffine`; direct `L1/conv1d.py` (conv_pre, conv_proj)
  - **`VitsDurationPredictor`** [compute]: `L1/conv1d.py + L1/relu.py + L1/layer_norm.py + L1/conv1d.py + L1/relu.py + L1/layer_norm.py + L1/conv1d.py` (proj)
  - **`VitsAttention`** [compute]: q/k/v + bmm-style attention with relative position embeddings (emb_rel_k, emb_rel_v) — no exact L2 match (specialized; closest concept: `L2/t5_attention.py` (rel-pos-bias))
  - **`VitsFeedForward`** [compute]: `L1/conv1d.py + L1/relu.py + L1/conv1d.py` (Conv1d-based FFN with relu activation)
  - **`VitsEncoderLayer`** [wiring]: wires `VitsAttention`, `VitsFeedForward`; direct `L1/layer_norm.py` (×2)
  - **`VitsEncoder`** [wiring]: wires `VitsEncoderLayer`
  - **`VitsTextEncoder`** [wiring]: wires `VitsEncoder`; direct `L1/embedding.py` (embed_tokens), `L1/conv1d.py` (project)
  - **`VitsModel`** [wiring]: wires `VitsTextEncoder`, `VitsResidualCouplingBlock`, `VitsHifiGan`, `VitsStochasticDurationPredictor`/`VitsDurationPredictor`, `VitsPosteriorEncoder`; direct `L1/embedding.py` (optional embed_speaker)

## vivit
- **src**: modeling_vivit.py
- **hidden_act**: gelu_fast
- **status**: composable
- **classes**:
  - **`VivitTubeletEmbeddings`** [compute]: `L1/conv3d.py` (Conv3d projection over tubelets)
  - **`VivitEmbeddings`** [compute]: wires `VivitTubeletEmbeddings`; cls_token + position_embeddings parameters
  - **`VivitSelfAttention`** [compute]: `L2/encoder_attention.py` (q/k/v + dispatch via ALL_ATTENTION_FUNCTIONS, non-causal — ViT-style)
  - **`VivitSelfOutput`** [compute]: `L1/linear.py`
  - **`VivitAttention`** [wiring]: wires `VivitSelfAttention`, `VivitSelfOutput`
  - **`VivitIntermediate`** [compute]: `L1/linear.py + L1/gelu.py` (gelu_fast)
  - **`VivitOutput`** [compute]: `L1/linear.py` (dense + residual)
  - **`VivitLayer`** [wiring]: wires `VivitAttention`, `VivitIntermediate`, `VivitOutput`; direct `L1/layer_norm.py` (×2)
  - **`VivitEncoder`** [wiring]: wires `VivitLayer`
  - **`VivitPooler`** [compute]: `L1/linear.py + L1/tanh.py`
  - **`VivitModel`** [wiring]: wires `VivitEmbeddings`, `VivitEncoder`, optional `VivitPooler`; direct `L1/layer_norm.py`
- **task heads (1)**: ForVideoClassification — base + linear (per-task)

## vjepa2
- **src**: modeling_vjepa2.py
- **hidden_act**: gelu
- **status**: kb_nano_l4 (has dedicated `L4/vjepa2.py`)
- **classes**:
  - **`VJEPA2PatchEmbeddings3D`** [compute]: `L1/conv3d.py` (Conv3d projection over tubelets)
  - **`VJEPA2Embeddings`** [wiring]: wires `VJEPA2PatchEmbeddings3D`. Closer match: `L2/vjepa2_embeddings.py`
  - **`VJEPA2RopeAttention`** [compute]: `L2/vjepa2_attention.py` (rope-based vision attention)
  - **`VJEPA2DropPath`** [compute]: stochastic depth (no exact L1)
  - **`VJEPA2MLP`** [compute]: `L2/vjepa2_mlp.py`
  - **`VJEPA2Layer`** [wiring]: wires `VJEPA2RopeAttention`, `VJEPA2MLP`, `VJEPA2DropPath`; direct `L1/layer_norm.py` (×2). Closer match: `L3/vjepa2_layer.py`
  - **`VJEPA2Encoder`** [wiring]: wires `VJEPA2Layer`
  - **`VJEPA2PredictorEmbeddings`** [compute]: `L1/linear.py` (predictor_embeddings); mask_tokens parameter; concat context + target
  - **`VJEPA2Predictor`** [wiring]: wires `VJEPA2PredictorEmbeddings`, `VJEPA2Layer` (×N). Closer: `L3/vjepa2_predictor.py`
  - **`VJEPA2PoolerSelfAttention`** [compute]: `L2/encoder_attention.py` (q/k/v + dispatch via ALL_ATTENTION_FUNCTIONS, non-causal — SigLIP-like)
  - **`VJEPA2PoolerCrossAttention`** [compute]: same as above but cross-attention with no out_proj — `L2/encoder_attention.py` analog
  - **`VJEPA2PoolerSelfAttentionLayer`** [wiring]: wires `VJEPA2PoolerSelfAttention`, `VJEPA2MLP`; direct `L1/layer_norm.py` (×2)
  - **`VJEPA2PoolerCrossAttentionLayer`** [wiring]: wires `VJEPA2PoolerCrossAttention`, `VJEPA2MLP`; direct `L1/layer_norm.py` (×2)
  - **`VJEPA2AttentivePooler`** [wiring]: wires `VJEPA2PoolerSelfAttentionLayer` (×N), `VJEPA2PoolerCrossAttentionLayer`; query_tokens parameter. Closer: `L3/vjepa2_pooler.py`
  - **`VJEPA2Model`** [wiring]: wires `VJEPA2Embeddings`, `VJEPA2Encoder`, `VJEPA2Predictor`; direct `L1/layer_norm.py`
- **task heads (1)**: ForVideoClassification — base + linear (per-task)

## voxtral
- **src**: modeling_voxtral.py, modular_voxtral.py
- **hidden_act**: projector_hidden_act=gelu (encoder activation_function configurable; text uses LLaMA silu)
- **status**: composable
- **classes**:
  - **`VoxtralAttention`** [compute, modular inherits `WhisperAttention`]: `L2/whisper_attention.py` (encoder variant — non-causal, with k_proj bias=False)
  - **`VoxtralEncoderLayer`** [wiring, modular inherits `WhisperEncoderLayer`]: wires `VoxtralAttention`; direct `L1/linear.py` (fc1, fc2), `L1/gelu.py`, `L1/layer_norm.py` (×2)
  - **`VoxtralEncoder`** [wiring, modular inherits `WhisperEncoder`]: wires `VoxtralEncoderLayer`; direct `L1/conv1d.py` (×2 conv1, conv2), `L1/gelu.py` (×2), `L1/embedding.py` (embed_positions), `L1/layer_norm.py`, `L1/avg_pool1d.py` (avg_pooler)
  - **`VoxtralMultiModalProjector`** [compute]: `L1/linear.py + L1/gelu.py + L1/linear.py`
  - **`VoxtralForConditionalGeneration`** [wiring]: wires `VoxtralEncoder`, `VoxtralMultiModalProjector`, language_model (AutoModel — LLaMA); direct `L1/linear.py` (lm_head)

## voxtral_realtime
- **src**: modeling_voxtral_realtime.py, modular_voxtral_realtime.py
- **hidden_act**: silu (encoder MLP and text MLP both); projector_hidden_act=gelu
- **status**: composable
- **classes**:
  - **`VoxtralRealtimeRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (or `L1/yarn_rotary_emb.py` if rope_type != "default")
  - **`VoxtralRealtimeCausalConv1d`** [compute]: `L1/causal_conv1d.py` (left-padded, supports streaming cache)
  - **`VoxtralRealtimeRMSNorm`** [compute]: `L1/rms_norm.py`
  - **`VoxtralRealtimeAttention`** [compute]: `L2/attention.py` (causal, with sliding_window option, RoPE, KV cache; encoder)
  - **`VoxtralRealtimeMLP`** [compute]: `L2/llama_mlp.py` (SwiGLU: silu(gate)*up → down; with bias on down_proj)
  - **`VoxtralRealtimeEmbedder`** [compute]: `L1/causal_conv1d.py + L1/gelu.py + L1/causal_conv1d.py + L1/gelu.py` (×2 conv1, conv2 with cache, then permute)
  - **`VoxtralRealtimeEncoderLayer`** [wiring]: wires `VoxtralRealtimeAttention`, `VoxtralRealtimeMLP`, `VoxtralRealtimeRMSNorm` (×2)
  - **`VoxtralRealtimeEncoder`** [wiring]: wires `VoxtralRealtimeEmbedder`, `VoxtralRealtimeEncoderLayer`, `VoxtralRealtimeRotaryEmbedding`; direct `VoxtralRealtimeRMSNorm`
  - **`VoxtralRealtimeTextAdaRmsNorm`** [compute]: `L1/linear.py + L1/gelu.py + L1/linear.py` (small MLP for adaptive scale)
  - **`VoxtralRealtimeTextAttention`** [compute]: `L2/attention.py` (causal, RoPE, KV cache; LLaMA-style)
  - **`VoxtralRealtimeTextMLP`** [compute]: `L2/llama_mlp.py` (SwiGLU)
  - **`VoxtralRealtimeTextDecoderLayer`** [wiring]: wires `VoxtralRealtimeTextAttention`, `VoxtralRealtimeTextMLP`, `VoxtralRealtimeRMSNorm` (×2), `VoxtralRealtimeTextAdaRmsNorm`; multiplicative t_cond modulation
  - **`VoxtralRealtimeTextModel`** [wiring]: wires `VoxtralRealtimeTextDecoderLayer`, `VoxtralRealtimeRotaryEmbedding`; direct `L1/embedding.py` (embed_tokens), `VoxtralRealtimeRMSNorm`
  - **`VoxtralRealtimeTextForCausalLM`** [wiring]: wires `VoxtralRealtimeTextModel`; direct `L1/linear.py` (lm_head)
  - **`VoxtralRealtimeTimeEmbedding`** [compute]: sinusoidal time embedding (no exact L1 — close to `L1/sinusoidal_embed.py`)
  - **`VoxtralRealtimeMultiModalProjector`** [compute]: `L1/linear.py + L1/gelu.py + L1/linear.py` (similar to VoxtralMultiModalProjector)
  - **`VoxtralRealtimeForConditionalGeneration`** [wiring]: wires `VoxtralRealtimeEncoder`, `VoxtralRealtimeMultiModalProjector`, `VoxtralRealtimeTextForCausalLM`, `VoxtralRealtimeTimeEmbedding`

## wav2vec2
- **src**: modeling_wav2vec2.py
- **hidden_act**: gelu
- **status**: composable
- **classes**:
  - **`Wav2Vec2NoLayerNormConvLayer`** [compute]: `L1/conv1d.py + L1/gelu.py`
  - **`Wav2Vec2LayerNormConvLayer`** [compute]: `L1/conv1d.py + L1/layer_norm.py + L1/gelu.py`
  - **`Wav2Vec2GroupNormConvLayer`** [compute]: `L1/conv1d.py + L1/group_norm.py + L1/gelu.py`
  - **`Wav2Vec2PositionalConvEmbedding`** [compute]: `L1/conv1d.py` (weight-norm grouped conv) + `Wav2Vec2SamePadLayer` + `L1/gelu.py`
  - **`Wav2Vec2SamePadLayer`** [compute]: tail-trim slice (no L1 — shape op)
  - **`Wav2Vec2FeatureEncoder`** [wiring]: wires `Wav2Vec2GroupNormConvLayer`/`Wav2Vec2NoLayerNormConvLayer`/`Wav2Vec2LayerNormConvLayer` (per feat_extract_norm)
  - **`Wav2Vec2FeatureProjection`** [compute]: `L1/layer_norm.py + L1/linear.py`
  - **`Wav2Vec2Attention`** [compute]: `L2/encoder_attention.py` analog (q/k/v + dispatch via ALL_ATTENTION_FUNCTIONS, non-causal — Bart-family encoder attention)
  - **`Wav2Vec2FeedForward`** [compute]: `L1/linear.py + L1/gelu.py + L1/linear.py` (intermediate_dense + activation + output_dense)
  - **`Wav2Vec2EncoderLayer`** [wiring]: wires `Wav2Vec2Attention`, `Wav2Vec2FeedForward`; direct `L1/layer_norm.py` (×2)
  - **`Wav2Vec2EncoderLayerStableLayerNorm`** [wiring]: wires `Wav2Vec2Attention`, `Wav2Vec2FeedForward`, optional `Wav2Vec2AttnAdapterLayer`; direct `L1/layer_norm.py` (×2)
  - **`Wav2Vec2Encoder`** [wiring]: wires `Wav2Vec2PositionalConvEmbedding`, `Wav2Vec2EncoderLayer`; direct `L1/layer_norm.py`
  - **`Wav2Vec2EncoderStableLayerNorm`** [wiring]: wires `Wav2Vec2PositionalConvEmbedding`, `Wav2Vec2EncoderLayerStableLayerNorm`; direct `L1/layer_norm.py`
  - **`Wav2Vec2GumbelVectorQuantizer`** [compute]: `L1/linear.py` (weight_proj) + gumbel_softmax + codevector lookup (no exact L1 for gumbel/quantization)
  - **`Wav2Vec2Adapter`** [wiring]: wires `Wav2Vec2AdapterLayer`; direct `L1/linear.py`, `L1/layer_norm.py` (optional proj/proj_layer_norm)
  - **`Wav2Vec2AdapterLayer`** [compute]: `L1/conv1d.py` + GLU (no exact L1 — F.glu split-and-mul)
  - **`Wav2Vec2AttnAdapterLayer`** [compute]: `L1/layer_norm.py + L1/linear.py + L1/relu.py + L1/linear.py`
  - **`Wav2Vec2Model`** [wiring]: wires `Wav2Vec2FeatureEncoder`, `Wav2Vec2FeatureProjection`, `Wav2Vec2Encoder`/`Wav2Vec2EncoderStableLayerNorm`, optional `Wav2Vec2Adapter`
  - **`Wav2Vec2ForPreTraining`** [wiring]: wires `Wav2Vec2Model`, `Wav2Vec2GumbelVectorQuantizer`; direct `L1/linear.py`
  - **`TDNNLayer`** [compute]: `L1/conv1d.py + L1/relu.py` (used in XVector head)
  - **`AMSoftmaxLoss`** [skip — Loss]
- **task heads (4)**: ForCTC, ForSequenceClassification, ForAudioFrameClassification, ForXVector — base + linear (per-task)

## wav2vec2_bert
- **src**: modeling_wav2vec2_bert.py
- **hidden_act**: swish (= silu)
- **status**: composable
- **classes**:
  - **`Wav2Vec2BertRotaryPositionalEmbedding`** [compute]: `L1/rotary_emb.py` (custom; sin-cos table)
  - **`Wav2Vec2BertRelPositionalEmbedding`** [compute]: relative position table (no exact L1 — sinusoidal pos table for transformer-XL relative)
  - **`Wav2Vec2BertFeatureProjection`** [compute]: `L1/layer_norm.py + L1/linear.py`
  - **`Wav2Vec2BertFeedForward`** [compute]: `L1/linear.py + L1/silu.py + L1/linear.py`
  - **`Wav2Vec2BertConvolutionModule`** [compute]: `L1/layer_norm.py + L1/conv1d.py` (pointwise) + GLU + `L1/conv1d.py` (depthwise) + `L1/layer_norm.py + L1/silu.py + L1/conv1d.py` (pointwise2). Conformer convolution module — no exact L2 match
  - **`Wav2Vec2BertSelfAttention`** [compute]: q/k/v with optional rotary or relative-position embedding (Transformer-XL-style) — no exact L2 match (specialized; closest concept: `L2/t5_attention.py` for rel-pos)
  - **`Wav2Vec2BertEncoderLayer`** [wiring]: wires `Wav2Vec2BertFeedForward` (×2 ffn1, ffn2), `Wav2Vec2BertSelfAttention`, `Wav2Vec2BertConvolutionModule`; direct `L1/layer_norm.py` (×5)
  - **`Wav2Vec2BertEncoder`** [wiring]: wires `Wav2Vec2BertRotaryPositionalEmbedding`/`Wav2Vec2BertRelPositionalEmbedding`, `Wav2Vec2BertEncoderLayer`; direct `L1/layer_norm.py`
  - **`Wav2Vec2BertAdapter`** [wiring]: wires `Wav2Vec2BertAdapterLayer`; direct `L1/linear.py`, `L1/layer_norm.py` (optional)
  - **`Wav2Vec2BertAdapterLayer`** [compute]: `L1/conv1d.py` + GLU + `L1/conv1d.py` (residual variant)
  - **`Wav2Vec2BertModel`** [wiring]: wires `Wav2Vec2BertFeatureProjection`, `Wav2Vec2BertEncoder`, optional `Wav2Vec2BertAdapter`
  - **`TDNNLayer`** [compute]: `L1/conv1d.py + L1/relu.py`
  - **`AMSoftmaxLoss`** [skip — Loss]
- **task heads (4)**: ForCTC, ForSequenceClassification, ForAudioFrameClassification, ForXVector — base + linear (per-task)

## wav2vec2_conformer
- **src**: modeling_wav2vec2_conformer.py
- **hidden_act**: gelu
- **status**: composable
- **classes**:
  - **`Wav2Vec2ConformerSamePadLayer`** [compute]: tail-trim slice (no L1)
  - **`Wav2Vec2ConformerPositionalConvEmbedding`** [compute]: `L1/conv1d.py` (weight-norm grouped) + `Wav2Vec2ConformerSamePadLayer` + `L1/gelu.py`
  - **`Wav2Vec2ConformerRotaryPositionalEmbedding`** [compute]: `L1/rotary_emb.py` analog
  - **`Wav2Vec2ConformerRelPositionalEmbedding`** [compute]: relative position table (no exact L1)
  - **`Wav2Vec2ConformerNoLayerNormConvLayer`** [compute]: `L1/conv1d.py + L1/gelu.py`
  - **`Wav2Vec2ConformerLayerNormConvLayer`** [compute]: `L1/conv1d.py + L1/layer_norm.py + L1/gelu.py`
  - **`Wav2Vec2ConformerGroupNormConvLayer`** [compute]: `L1/conv1d.py + L1/group_norm.py + L1/gelu.py`
  - **`Wav2Vec2ConformerFeatureEncoder`** [wiring]: wires conv layers (per feat_extract_norm)
  - **`Wav2Vec2ConformerFeatureProjection`** [compute]: `L1/layer_norm.py + L1/linear.py`
  - **`Wav2Vec2ConformerFeedForward`** [compute]: `L1/linear.py + L1/gelu.py + L1/linear.py`
  - **`Wav2Vec2ConformerConvolutionModule`** [compute]: `L1/layer_norm.py + L1/conv1d.py` (pointwise) + GLU + `L1/conv1d.py` (depthwise) + `L1/batch_norm2d.py` (or layer_norm) + `L1/silu.py` + `L1/conv1d.py` (pointwise2)
  - **`Wav2Vec2ConformerSelfAttention`** [compute]: q/k/v with optional rotary or relative-position embedding — no exact L2 match (Conformer style)
  - **`Wav2Vec2ConformerEncoderLayer`** [wiring]: wires `Wav2Vec2ConformerFeedForward` (×2), `Wav2Vec2ConformerSelfAttention`, `Wav2Vec2ConformerConvolutionModule`; direct `L1/layer_norm.py` (×5)
  - **`Wav2Vec2ConformerEncoder`** [wiring]: wires `Wav2Vec2ConformerRotaryPositionalEmbedding`/`Wav2Vec2ConformerRelPositionalEmbedding`, `Wav2Vec2ConformerEncoderLayer`; direct `L1/layer_norm.py`
  - **`Wav2Vec2ConformerGumbelVectorQuantizer`** [compute]: same as Wav2Vec2GumbelVectorQuantizer
  - **`Wav2Vec2ConformerAdapter`** [wiring]: wires `Wav2Vec2ConformerAdapterLayer`
  - **`Wav2Vec2ConformerAdapterLayer`** [compute]: `L1/conv1d.py` + GLU
  - **`Wav2Vec2ConformerModel`** [wiring]: wires `Wav2Vec2ConformerFeatureEncoder`, `Wav2Vec2ConformerFeatureProjection`, `Wav2Vec2ConformerEncoder`, optional `Wav2Vec2ConformerAdapter`
  - **`Wav2Vec2ConformerForPreTraining`** [wiring]: wires `Wav2Vec2ConformerModel`, `Wav2Vec2ConformerGumbelVectorQuantizer`; direct `L1/linear.py`
  - **`TDNNLayer`** [compute]: `L1/conv1d.py + L1/relu.py`
  - **`AMSoftmaxLoss`** [skip — Loss]
- **task heads (4)**: ForCTC, ForSequenceClassification, ForAudioFrameClassification, ForXVector — base + linear (per-task)

## wavlm
- **src**: modeling_wavlm.py
- **hidden_act**: gelu
- **status**: composable
- **classes**:
  - **`WavLMSamePadLayer`** [compute]: tail-trim slice (no L1)
  - **`WavLMPositionalConvEmbedding`** [compute]: `L1/conv1d.py` (weight-norm grouped) + `WavLMSamePadLayer` + `L1/gelu.py`
  - **`WavLMFeatureProjection`** [compute]: `L1/layer_norm.py + L1/linear.py`
  - **`WavLMAttention`** [compute]: q/k/v + relative position bias (with relative_attention_bias) + gated relative position bias — no exact L2 match (specialized; closest concept: T5 rel-pos)
  - **`WavLMFeedForward`** [compute]: `L1/linear.py + L1/gelu.py + L1/linear.py`
  - **`WavLMEncoderLayer`** [wiring]: wires `WavLMAttention`, `WavLMFeedForward`; direct `L1/layer_norm.py` (×2)
  - **`WavLMEncoderLayerStableLayerNorm`** [wiring]: wires `WavLMAttention`, `WavLMFeedForward`; direct `L1/layer_norm.py` (×2)
  - **`WavLMEncoder`** [wiring]: wires `WavLMPositionalConvEmbedding`, `WavLMEncoderLayer`; direct `L1/layer_norm.py`
  - **`WavLMEncoderStableLayerNorm`** [wiring]: wires `WavLMPositionalConvEmbedding`, `WavLMEncoderLayerStableLayerNorm`; direct `L1/layer_norm.py`
  - **`WavLMGumbelVectorQuantizer`** [compute]: same as Wav2Vec2GumbelVectorQuantizer
  - **`WavLMNoLayerNormConvLayer`** [compute]: `L1/conv1d.py + L1/gelu.py`
  - **`WavLMLayerNormConvLayer`** [compute]: `L1/conv1d.py + L1/layer_norm.py + L1/gelu.py`
  - **`WavLMGroupNormConvLayer`** [compute]: `L1/conv1d.py + L1/group_norm.py + L1/gelu.py`
  - **`WavLMFeatureEncoder`** [wiring]: wires conv layers
  - **`WavLMAdapterLayer`** [compute]: `L1/conv1d.py` + GLU
  - **`WavLMAdapter`** [wiring]: wires `WavLMAdapterLayer`
  - **`WavLMModel`** [wiring]: wires `WavLMFeatureEncoder`, `WavLMFeatureProjection`, `WavLMEncoder`/`WavLMEncoderStableLayerNorm`, optional `WavLMAdapter`
  - **`TDNNLayer`** [compute]: `L1/conv1d.py + L1/relu.py`
  - **`AMSoftmaxLoss`** [skip — Loss]
- **task heads (4)**: ForCTC, ForSequenceClassification, ForAudioFrameClassification, ForXVector — base + linear (per-task)

## whisper
- **src**: modeling_whisper.py
- **hidden_act**: gelu (activation_function)
- **status**: kb_nano_l4 (`L4/whisper.py` exists)
- **classes**:
  - **`WhisperPositionalEmbedding`** [compute]: `L1/embedding.py` (subclass of nn.Embedding with custom forward)
  - **`WhisperAttention`** [compute]: `L2/whisper_attention.py` (encoder + decoder + cross variants supported)
  - **`WhisperEncoderLayer`** [wiring]: wires `WhisperAttention`; direct `L1/linear.py` (fc1, fc2), `L1/gelu.py`, `L1/layer_norm.py` (×2). Closer composite: `L3/whisper_encoder_layer.py`
  - **`WhisperDecoderLayer`** [wiring]: wires `WhisperAttention` (self), `WhisperAttention` (cross, encoder_attn); direct `L1/linear.py` (fc1, fc2), `L1/gelu.py`, `L1/layer_norm.py` (×3). Closer: `L3/whisper_decoder_layer.py`
  - **`WhisperEncoder`** [wiring]: wires `WhisperEncoderLayer`; direct `L1/conv1d.py` (×2 conv1, conv2), `L1/gelu.py` (×2), `WhisperPositionalEmbedding`, `L1/layer_norm.py`, `L1/avg_pool1d.py` (avg_pooler)
  - **`WhisperDecoder`** [wiring]: wires `WhisperDecoderLayer`; direct `L1/embedding.py` (embed_tokens), `WhisperPositionalEmbedding`, `L1/layer_norm.py`
  - **`WhisperModel`** [wiring]: wires `WhisperEncoder`, `WhisperDecoder`
  - **`WhisperForConditionalGeneration`** [wiring]: wires `WhisperModel`; direct `L1/linear.py` (proj_out as lm_head)
  - **`WhisperDecoderWrapper`** [wiring]: wires `WhisperDecoder`
  - **`WhisperForCausalLM`** [wiring]: wires `WhisperDecoderWrapper`; direct `L1/linear.py` (proj_out)
- **task heads (1)**: ForAudioClassification — base + linear (per-task)

## x_clip
- **src**: modeling_x_clip.py
- **hidden_act**: quick_gelu (vision and text); prompt_hidden_act=quick_gelu
- **status**: composable
- **classes**:
  - **`XCLIPVisionEmbeddings`** [compute]: `L1/conv2d.py` (patch_embedding) + `L1/embedding.py` (position_embedding); class_embedding parameter
  - **`XCLIPTextEmbeddings`** [compute]: `L1/embedding.py` (token_embedding) + `L1/embedding.py` (position_embedding) — no LayerNorm
  - **`XCLIPAttention`** [compute]: `L2/clip_attention.py` (q/k/v + dispatch via ALL_ATTENTION_FUNCTIONS, non-causal — CLIP-style)
  - **`XCLIPMLP`** [compute]: `L2/clip_mlp.py` (fc1 + quickgelu + fc2)
  - **`XCLIPEncoderLayer`** [wiring]: wires `XCLIPAttention`, `XCLIPMLP`; direct `L1/layer_norm.py` (×2)
  - **`XCLIPDropPath`** [compute]: stochastic depth (no exact L1)
  - **`XCLIPVisionEncoderLayer`** [wiring]: wires `XCLIPAttention` (×2 self_attn + message_attn), `XCLIPMLP`, `XCLIPDropPath`; direct `L1/linear.py` (message_fc), `L1/layer_norm.py` (×3)
  - **`XCLIPEncoder`** [wiring]: wires `XCLIPEncoderLayer`
  - **`XCLIPVisionEncoder`** [wiring]: wires `XCLIPVisionEncoderLayer`
  - **`XCLIPTextModel`** [wiring]: wires `XCLIPTextEmbeddings`, `XCLIPEncoder`; direct `L1/layer_norm.py`
  - **`XCLIPVisionModel`** [wiring]: wires `XCLIPVisionEmbeddings`, `XCLIPVisionEncoder`; direct `L1/layer_norm.py` (×2 pre_layrnorm + post_layernorm)
  - **`XCLIPMultiframeIntegrationTransformer`** [wiring]: wires `XCLIPEncoder`; position_embedding parameter; mean pooling
  - **`XCLIPCrossAttention`** [compute]: q/k/v cross-attention (no out_proj — `proj` afterwards), no causal — `L2/clip_attention.py` analog with separate kv input
  - **`PromptGeneratorLayer`** [wiring]: wires `XCLIPCrossAttention`; direct `L1/linear.py + ACT + L1/linear.py` (Sequential mlp), `L1/layer_norm.py` (×2)
  - **`XCLIPPromptGenerator`** [wiring]: wires `PromptGeneratorLayer`; direct `L1/layer_norm.py`; alpha parameter
  - **`XCLIPModel`** [wiring]: wires `XCLIPTextModel`, `XCLIPVisionModel`, `XCLIPMultiframeIntegrationTransformer`, `XCLIPPromptGenerator`; direct `L1/linear.py` (visual_projection, text_projection), `L1/layer_norm.py` (prompts_visual_layernorm); learnable scalar logit_scale, prompts_visual_projection parameter

## xcodec
- **src**: modeling_xcodec.py
- **hidden_act**: N/A (uses ELU directly; backbone DAC + HuBERT/Wav2Vec2 have their own)
- **status**: composable
- **classes**:
  - **`XcodecResidualUnit`** [compute]: `L1/elu.py + L1/conv1d.py + L1/elu.py + L1/conv1d.py` + residual
  - **`XcodecSemanticEncoderBlock`** [wiring]: wires `XcodecResidualUnit` (×N); direct `L1/conv1d.py`
  - **`SemanticEncoder`** [wiring]: wires `XcodecSemanticEncoderBlock` (×N); direct `L1/conv1d.py`
  - **`SemanticDecoderBlock`** [wiring]: wires `XcodecResidualUnit` (×N); direct `L1/conv1d.py` or `L1/conv_transpose1d.py`
  - **`SemanticDecoder`** [wiring]: wires `SemanticDecoderBlock` (×N); direct `L1/conv1d.py` (×2)
  - **`XcodecEuclideanCodebook`** [compute]: codebook lookup via Euclidean distance + `L1/embedding.py` (decode); buffers (no L1 for quantize)
  - **`XcodecVectorQuantization`** [wiring]: wires `XcodecEuclideanCodebook`
  - **`XcodecResidualVectorQuantization`** [wiring]: wires `XcodecVectorQuantization` (×N)
  - **`XcodecModel`** [wiring]: wires acoustic_encoder/decoder (AutoModel — DAC), `SemanticEncoder`, `SemanticDecoder`, semantic_model (AutoModel — HuBERT/Wav2Vec2), `XcodecResidualVectorQuantization`; direct `L1/linear.py` (fc, fc1, fc2)
