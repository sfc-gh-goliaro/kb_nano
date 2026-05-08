## csm
- **src**: modular_csm.py
- **status**: composable
- **rationale**: Llama-style backbone + LlamaForCausalLM-style depth decoder with codebook-specific linear head and audio token sum embedding; all compute is Llama-family + Linear/Embedding.
- **classes**:
  - **`CsmRMSNorm`** [compute]: `L1/rms_norm.py` (Llama RMSNorm pass-through.)
  - **`CsmRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (Standard Llama RoPE pass-through.)
  - **`CsmMLP`** [compute]: `L2/llama_mlp.py`, `L1/silu_and_mul.py` (SwiGLU gate_up_proj -> SiluAndMul -> down_proj.)
  - **`CsmAttention`** [compute]: `L2/attention.py` (Llama attention pass-through (LlamaAttention).)
  - **`CsmDecoderLayer`** [wiring]: Wiring: norm + attn + mlp.
  - **`CsmDepthDecoderModel`** [wiring]: Wiring: embed + projector + decoder layers + norm.
  - **`CsmCodebooksHead`** [compute]: `L1/linear.py` (Per-codebook linear projection (nn.functional.linear).)
  - **`CsmDepthDecoderForCausalLM`** [wiring]: Wiring around model + codebook head.
  - **`CsmBackboneModelEmbeddings`** [compute]: `L1/embedding.py` (Audio-token offset embedding then sum across codebooks.)
  - **`CsmBackboneModel`** [wiring]: Wiring around backbone embeddings + Llama layers.
  - **`CsmForConditionalGeneration`** [wiring]: Wiring: text embed + audio codec encode + backbone + depth decoder + lm_head.

## ctrl
- **src**: modeling_ctrl.py
- **status**: composable
- **rationale**: Custom causal transformer with sinusoidal positional encoding, MHA with bare matmul-softmax-matmul, and Sequential FFN with ReLU; all leaf ops (linear/sdpa-or-matmul/relu/layer_norm/embedding) exist in kb-nano.
- **classes**:
  - **`MultiHeadAttention`** [compute]: `L1/linear.py`, `L1/sdpa.py`, `L1/softmax.py` (Three Linear projections (Wq/Wk/Wv) + scaled dot-product attention + dense output projection; can use sdpa primitive.)
  - **`EncoderLayer`** [compute]: `L1/layer_norm.py`, `L1/linear.py`, `L1/relu.py` (LayerNorm + MHA + LayerNorm + Sequential(Linear/ReLU/Linear); all primitives exist.)
  - **`CTRLModel`** [compute]: `L1/embedding.py`, `L1/sinusoidal_embed.py`, `L1/layer_norm.py` (Token embedding + sinusoidal positional encoding (kb-nano L1/sinusoidal_embed.py exists) + stacked encoder layers + final layer norm.)
  - **`CTRLLMHeadModel`** [wiring]: Wiring: transformer + lm_head Linear.
  - **`CTRLForSequenceClassification`** [wiring]: Wiring: transformer + classifier Linear.

## cvt
- **src**: modeling_cvt.py
- **status**: composable
- **rationale**: Convolutional Vision Transformer: depthwise Conv2d projections for QKV + LayerNorm + standard scaled-dot-product attention + GELU MLP; all leaf ops exist (conv2d, batch_norm2d, layer_norm, sdpa, gelu, linear).
- **classes**:
  - **`CvtDropPath`** [compute]: `L1/dropout.py` (Stochastic depth (training-only).)
  - **`CvtEmbeddings`** [wiring]: Wiring: ConvEmbeddings + Dropout.
  - **`CvtConvEmbeddings`** [compute]: `L1/conv2d.py`, `L1/layer_norm.py` (Patch embedding via Conv2d + LayerNorm.)
  - **`CvtSelfAttentionConvProjection`** [compute]: `L1/conv2d.py`, `L1/batch_norm2d.py` (Depthwise Conv2d + BatchNorm2d for QKV projection.)
  - **`CvtSelfAttentionLinearProjection`** [wiring]: Pure reshape/permute, no parameters.
  - **`CvtSelfAttentionProjection`** [wiring]: Wiring around Conv + Linear projection.
  - **`CvtSelfAttention`** [compute]: `L1/linear.py`, `L1/sdpa.py`, `L1/softmax.py` (Multi-head attention with conv-projected QKV + Linear + scaled dot-product (einsum form).)
  - **`CvtSelfOutput`** [compute]: `L1/linear.py` (Linear + dropout.)
  - **`CvtAttention`** [wiring]: Wiring around CvtSelfAttention + CvtSelfOutput.
  - **`CvtIntermediate`** [compute]: `L1/linear.py`, `L1/gelu.py` (Linear + GELU.)
  - **`CvtOutput`** [compute]: `L1/linear.py` (Linear + dropout + residual.)
  - **`CvtLayer`** [wiring]: Wiring: layernorm + attn + drop_path + layernorm + intermediate + output.
  - **`CvtStage`** [wiring]: Wiring: embedding + cls_token + layers.
  - **`CvtEncoder`** [wiring]: Wiring: stages.

## cwm
- **src**: modular_cwm.py
- **status**: composable
- **rationale**: CWM is Llama with Llama-3 RoPE scaling and per-layer sliding/full attention pattern; same Llama attention/MLP/RMSNorm primitives apply.
- **classes**:
  - **`CwmRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (Standard Llama-style rotary embedding (Llama-3 scaling supported in rotary_emb).)
  - **`CwmAttention`** [compute]: `L2/attention.py` (Llama attention with q/k/v Linear projections (no bias); supports sliding window via attention.py LlamaAttention.)
  - **`CwmDecoderLayer`** [wiring]: Wiring: norm + attn + mlp; uses LlamaMLP via parent.
  - **`CwmModel`** [wiring]: Wiring: embed_tokens + layers + norm with sliding/full mask map.
  - **`CwmForCausalLM`** [wiring]: Wiring: model + lm_head.

## d_fine
- **src**: modular_d_fine.py
- **status**: composable
- **rationale**: D-Fine extends RT-DETR-V2 with a deformable cross-attention, integral box head, gating, and HGNetv2 backbone; all kernels (deformable attention, conv2d/batch_norm2d, layer_norm, linear, softmax, gelu/relu/silu) exist in kb-nano (e.g. L1/rtdetrv2_deformable_attention.py).
- **classes**:
  - **`DFineMLP`** [compute]: `L1/linear.py`, `L1/relu.py` (Stack of Linear + activation.)
  - **`DFineGate`** [compute]: `L1/linear.py`, `L1/sigmoid.py`, `L1/layer_norm.py` (Linear + sigmoid + LayerNorm + gated combine.)
  - **`DFineFrozenBatchNorm2d`** [compute]: `L1/frozen_batch_norm2d.py` (Frozen BN with kb-nano kernel.)
  - **`DFineMultiscaleDeformableAttention`** [compute]: `L1/rtdetrv2_deformable_attention.py`, `L2/rtdetrv2_deformable_attention.py`, `L1/linear.py`, `L1/softmax.py` (Same multi-scale deformable attention as RT-DETR-V2; uses kb-nano deformable attention primitive.)
  - **`DFineConvNormLayer`** [compute]: `L2/rtdetrv2_conv_norm.py`, `L1/conv2d.py`, `L1/batch_norm2d.py` (Conv2d + BN.)
  - **`DFineRepVggBlock`** [compute]: `L2/rtdetrv2_repvgg_block.py` (RepVGG block reuses RT-DETR-V2 kernel.)
  - **`DFineCSPRepLayer`** [compute]: `L2/rtdetrv2_csp_rep_layer.py` (CSP layer.)
  - **`DFineRepNCSPELAN4`** [wiring]: Wiring: ConvNormLayer + CSPRepLayer stack.
  - **`DFineSCDown`** [wiring]: Wiring: two ConvNormLayers.
  - **`DFineEncoderLayer`** [compute]: `L2/rtdetrv2_encoder_layer.py` (RT-DETR encoder layer + DFineMLP.)
  - **`DFineAIFILayer`** [compute]: `L2/rtdetrv2_layers.py` (AIFI pass-through.)
  - **`DFineIntegral`** [compute]: `L1/softmax.py`, `L1/linear.py` (softmax + linear for integral bbox.)
  - **`DFineLQE`** [compute]: `L1/softmax.py`, `L1/linear.py` (softmax + topk + linear.)
  - **`DFineDecoderLayer`** [wiring]: Wiring: self-attn + cross-attn (deformable) + gateway + MLP.
  - **`DFineMLPPredictionHead`** [compute]: `L2/rtdetrv2_mlp_head.py` (MLP prediction head pass-through.)
  - **`DFineHybridEncoder`** [compute]: `L3/rtdetrv2_hybrid_encoder.py` (Wiring around AIFI layers + FPN/PAN.)
  - **`DFineDecoder`** [compute]: `L3/rtdetrv2_decoder.py` (Wiring around decoder layers with FDR refinement.)

## dab_detr
- **src**: modeling_dab_detr.py
- **status**: partial
- **rationale**: Standard DETR-style enc-dec with conditional cross-attention with q/k content/position projections; all compute is matmul/softmax/linear/layer_norm/conv (via load_backbone) — all primitives exist.
- **classes**:
  - **`DabDetrConvEncoder`** [compute]: no kb-nano kernel — Standard DETR-style enc-dec with conditional cross-attention with q/k content/position projections; all compute is matmul/softmax/linear/layer_norm/conv (via load_backbone) — all primitives exist.
  - **`DabDetrFrozenBatchNorm2d`** [compute]: `L1/frozen_batch_norm2d.py` (Frozen BN.)
  - **`DabDetrConvModel`** [wiring]: Wiring: convolutional encoder + position embed.
  - **`DabDetrSinePositionEmbedding`** [compute]: `L1/sinusoidal_embed.py` (2D sine position embedding (custom).)
  - **`DetrAttention`** [compute]: `L1/linear.py`, `L1/sdpa.py`, `L1/softmax.py` (Standard MHA with q/k/v/o linear projections; can use sdpa primitive.)
  - **`DabDetrAttention`** [compute]: `L1/linear.py`, `L1/sdpa.py`, `L1/softmax.py` (Cross-attention with separate dim for q/k vs v; matmul/softmax/matmul.)
  - **`DabDetrDecoderLayerSelfAttention`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (Q/K content+position content/pos linear + attn + LayerNorm.)
  - **`DabDetrDecoderLayerCrossAttention`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (Cross-attention with anchor projections.)
  - **`DabDetrDecoderLayerFFN`** [compute]: `L1/linear.py`, `L1/relu.py`, `L1/layer_norm.py` (Linear + activation + Linear + LayerNorm.)
  - **`DabDetrEncoderLayer`** [wiring]: Wiring: self-attn + FFN.
  - **`DabDetrDecoderLayer`** [wiring]: Wiring: self-attn + cross-attn + FFN.
  - **`DabDetrMLP`** [compute]: `L1/linear.py` (Stack of Linear + ReLU.)
  - **`DabDetrEncoder`** [wiring]: Wiring: stack of encoder layers.
  - **`DabDetrDecoder`** [wiring]: Wiring: stack of decoder layers.
  - **`DabDetrModel`** [wiring]: Wiring: backbone + encoder + decoder.
  - **`DabDetrMHAttentionMap`** [compute]: `L1/linear.py`, `L1/conv2d.py`, `L1/softmax.py` (Multi-head attention map for segmentation.)
  - **`DabDetrForObjectDetection`** [wiring]: Wiring: model + class/bbox heads.

## dac
- **src**: modeling_dac.py
- **status**: partial
- **partial_reason**: Snake1d activation (x + (1/(alpha+eps)) * sin(alpha*x).pow(2)) has no fused kb-nano kernel; would fall back to torch elementwise sin/pow/mul/add.
- **rationale**: DAC audio codec uses Snake1d activation (x + 1/alpha * sin(alpha*x)^2) — kb-nano has silu/gelu/relu/swiglu but no Snake1d. Other ops (Conv1d, ConvTranspose1d, Embedding, mse_loss, F.normalize) are composable.
- **classes**:
  - **`DacEncoder`** [compute]: no kb-nano kernel — Snake1d activation (x + (1/(alpha+eps)) * sin(alpha*x).pow(2)) has no fused kb-nano kernel; would fall back to torch elementwise sin/pow/mul/add.
  - **`Snake1d`** [wiring]: Snake activation (sin-based) — no kb-nano fused kernel; runs via torch elementwise ops.
  - **`DacVectorQuantize`** [compute]: `L1/conv1d.py`, `L1/embedding.py`, `L1/l2_norm.py` (Conv1d in/out projection + Embedding codebook + L2-normalized nearest-neighbor.)
  - **`DacResidualUnit`** [compute]: `L1/conv1d.py` (Wiring: Snake1d + Conv1d (dilated).)
  - **`DacEncoderBlock`** [compute]: `L1/conv1d.py` (Wiring: residual units + Snake + strided Conv1d.)
  - **`DacDecoderBlock`** [compute]: `L1/conv_transpose1d.py` (Wiring: Snake + ConvTranspose1d + residual units.)
  - **`DacResidualVectorQuantizer`** [wiring]: Wiring: list of DacVectorQuantize.
  - **`DacDecoder`** [wiring]: Wiring: Conv1d + DacDecoderBlock stack + Snake + final Conv1d.
  - **`DacModel`** [wiring]: Wiring: encoder + quantizer + decoder.

## data2vec_audio
- **src**: modular_data2vec_audio.py
- **status**: composable
- **rationale**: Inherits from Wav2Vec2: Conv1d feature extractor + LayerNorm + GELU + transformer encoder with positional Conv embedding; all primitives exist (conv1d/layer_norm/gelu/sdpa/linear).
- **classes**:
  - **`Data2VecAudioConvLayer`** [compute]: `L1/conv1d.py`, `L1/layer_norm.py`, `L1/gelu.py` (Conv1d + LayerNorm + GELU activation.)
  - **`Data2VecAudioPadLayer`** [wiring]: Pure slice/pad (no kernel).
  - **`Data2VecAudioPositionalConvLayer`** [compute]: `L1/conv1d.py`, `L1/layer_norm.py`, `L1/gelu.py` (Grouped Conv1d positional embedding + LayerNorm + activation.)
  - **`Data2VecAudioPositionalConvEmbedding`** [wiring]: Wiring: stack of positional conv layers.
  - **`Data2VecAudioFeatureEncoder`** [wiring]: Wiring: stack of conv layers.
  - **`Data2VecAudioFeatureProjection`** [compute]: `L1/layer_norm.py`, `L1/linear.py` (LayerNorm + Linear.)
  - **`Data2VecAudioEncoder`** [wiring]: Wiring: BERT-style encoder layers (Wav2Vec2).
  - **`Data2VecAudioAdapter`** [compute]: `L1/conv1d.py` (Conv1d adapter layers.)
  - **`Data2VecAudioModel`** [wiring]: Wiring: feature_extractor + projection + encoder + adapter.
  - **`Data2VecAudioForCTC`** [wiring]: Wiring: model + lm_head Linear.
  - **`Data2VecAudioForSequenceClassification`** [wiring]: Wiring: model + classifier.
  - **`Data2VecAudioForAudioFrameClassification`** [wiring]: Wiring: model + classifier.
  - **`Data2VecAudioForXVector`** [wiring]: Wiring: model + TDNN/Linear.

## data2vec_text
- **src**: modular_data2vec_text.py
- **status**: composable
- **rationale**: Inherits from RoBERTa (BERT-derived): encoder self-attention with token+position+type embeddings; all kernels via encoder_attention.py + encoder_mlp.py + bert_embeddings.py.
- **classes**:
  - **`Data2VecTextEmbeddings`** [compute]: `L2/bert_embeddings.py`, `L1/embedding.py`, `L1/layer_norm.py` (Token + position + type embed + LayerNorm.)
  - **`Data2VecTextSelfAttention`** [compute]: `L2/encoder_attention.py` (BERT-style self-attention.)
  - **`Data2VecTextCrossAttention`** [compute]: `L2/encoder_attention.py` (BERT-style cross-attention.)
  - **`Data2VecTextLayer`** [compute]: `L3/bert_layer.py` (BERT-style encoder layer wiring.)
  - **`Data2VecTextLMHead`** [compute]: `L1/linear.py`, `L1/layer_norm.py`, `L1/gelu.py` (MLM head: Linear + LayerNorm + Linear.)
  - **`Data2VecTextClassificationHead`** [compute]: `L1/linear.py` (Linear + tanh + Linear.)
  - **`Data2VecTextModel`** [compute]: `L3/bert_model.py` (Wiring: embeddings + encoder + pooler.)

## data2vec_vision
- **src**: modeling_data2vec_vision.py
- **status**: composable
- **rationale**: BeiT-style ViT with relative position bias and SDPA attention; ops are conv2d (patch embed), layer_norm, linear, gelu, sdpa — all primitives exist.
- **classes**:
  - **`Data2VecVisionDropPath`** [compute]: `L1/dropout.py` (Stochastic depth.)
  - **`Data2VecVisionEmbeddings`** [compute]: `L1/embedding.py` (Patch + cls token + position embeddings.)
  - **`Data2VecVisionPatchEmbeddings`** [compute]: `L1/conv2d.py` (Conv2d patch embedding.)
  - **`Data2VecVisionSelfAttention`** [compute]: `L1/linear.py`, `L1/sdpa.py`, `L1/softmax.py` (QKV linears + scaled-dot-product attention with relative position bias.)
  - **`Data2VecVisionSdpaSelfAttention`** [compute]: `L1/sdpa.py` (SDPA-backed variant of self-attention.)
  - **`Data2VecVisionSelfOutput`** [compute]: `L1/linear.py` (Linear output projection.)
  - **`Data2VecVisionAttention`** [wiring]: Wiring: self-attention + output.
  - **`Data2VecVisionIntermediate`** [compute]: `L1/linear.py`, `L1/gelu.py` (Linear + GELU.)
  - **`Data2VecVisionOutput`** [compute]: `L1/linear.py` (Linear output projection.)
  - **`Data2VecVisionLayer`** [wiring]: Wiring: layernorm + attn + layernorm + intermediate + output.
  - **`Data2VecVisionRelativePositionBias`** [compute]: `L1/embedding.py` (Embedding-based relative position bias table.)
  - **`Data2VecVisionEncoder`** [wiring]: Wiring: layers.
  - **`Data2VecVisionModel`** [wiring]: Wiring: embeddings + encoder + pooler.
  - **`Data2VecVisionPooler`** [compute]: `L1/layer_norm.py` (LayerNorm pooler.)
  - **`Data2VecVisionForImageClassification`** [wiring]: Wiring: model + classifier Linear.
  - **`Data2VecVisionConvModule`** [compute]: `L1/conv2d.py`, `L1/batch_norm2d.py`, `L1/relu.py` (Conv2d + BN + ReLU.)
  - **`Data2VecVisionPyramidPoolingBlock`** [compute]: `L1/adaptive_avg_pool2d.py` (Adaptive avg pool + ConvModule.)
  - **`Data2VecVisionPyramidPoolingModule`** [compute]: `L1/interpolate.py` (Pyramid of pooling blocks + interpolate.)
  - **`Data2VecVisionUperHead`** [wiring]: Wiring: pyramid + FPN-style head.
  - **`Data2VecVisionFCNHead`** [wiring]: Wiring: ConvModule + classifier.
  - **`Data2VecVisionForSemanticSegmentation`** [wiring]: Wiring: backbone + UperHead + auxiliary FCN head.

## dbrx
- **src**: modular_dbrx.py
- **status**: composable
- **rationale**: DBRX is Llama-style attention with clipped fused QKV (Wqkv) + Mixtral-style MoE (DbrxExpertGLU is SwiGLU expert with weights w1/v1/w2); all primitives exist (linear/sdpa/softmax/layer_norm/silu/silu_and_mul + grouped_gemm/moe_align).
- **classes**:
  - **`DbrxRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (Standard Llama RoPE.)
  - **`DbrxAttention`** [compute]: `L2/attention.py`, `L1/linear.py`, `L1/rotary_emb.py` (Fused Wqkv linear with clip + RoPE + sdpa-equivalent attention + out_proj.)
  - **`DbrxExpertGLU`** [compute]: `L1/silu_and_mul.py`, `L1/linear.py` (SwiGLU expert: act(w1·x) * (v1·x), then w2·result.)
  - **`DbrxExperts`** [compute]: `L1/moe_grouped_gemm.py`, `L1/moe_align.py` (Per-expert dispatch via top-k routing.)
  - **`DbrxRouter`** [compute]: `L1/linear.py` (Linear router.)
  - **`DbrxFFN`** [compute]: `L2/mixtral_moe.py`, `L1/softmax.py`, `L1/topk_softmax.py` (MoE wiring: router + experts (top-k softmax).)
  - **`DbrxNormAttentionNorm`** [compute]: `L1/layer_norm.py` (Wiring: norm + attn + norm.)
  - **`DbrxBlock`** [wiring]: Wiring: norm_attn_norm + ffn (MoE).
  - **`DbrxModel`** [wiring]: Wiring: wte + blocks + norm_f.
  - **`DbrxForCausalLM`** [wiring]: Wiring: transformer + lm_head.

## deberta
- **src**: modeling_deberta.py
- **status**: composable
- **rationale**: DeBERTa uses Disentangled Self-Attention (content + content-to-position + position-to-content gather-based scores). All ops are linear/matmul/softmax/embedding/layer_norm — composable using bare primitives though no fused DeBERTa kernel exists.
- **classes**:
  - **`DebertaLayerNorm`** [compute]: `L1/layer_norm.py` (Standard mean-centered LayerNorm.)
  - **`DebertaSelfOutput`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (Linear + dropout + LayerNorm with residual.)
  - **`DisentangledSelfAttention`** [compute]: `L1/linear.py`, `L1/softmax.py`, `L1/embedding.py` (Q/K/V Linear + content scores + c2p (content-to-position) + p2c (position-to-content) gather-based relative position attention; uses gather/matmul/softmax.)
  - **`DebertaEmbeddings`** [compute]: `L2/bert_embeddings.py`, `L1/embedding.py`, `L1/layer_norm.py` (Word + position + type embeddings + LayerNorm.)
  - **`DebertaAttention`** [wiring]: Wiring: DisentangledSelfAttention + SelfOutput.
  - **`DebertaIntermediate`** [compute]: `L1/linear.py`, `L1/gelu.py` (Linear + GELU.)
  - **`DebertaOutput`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (Linear + dropout + LayerNorm with residual.)
  - **`DebertaLayer`** [wiring]: Wiring: attention + intermediate + output.
  - **`DebertaEncoder`** [compute]: `L1/embedding.py`, `L1/layer_norm.py` (Wiring: layers + relative position embeddings.)
  - **`DebertaModel`** [wiring]: Wiring: embeddings + encoder.
  - **`ContextPooler`** [compute]: `L1/linear.py` (Linear + activation pooler.)
  - **`DebertaForMaskedLM`** [wiring]: Wiring: model + MLM head.
  - **`DebertaForSequenceClassification`** [wiring]: Wiring: model + pooler + classifier.
  - **`DebertaForTokenClassification`** [wiring]: Wiring: model + classifier.
  - **`DebertaForQuestionAnswering`** [wiring]: Wiring: model + qa_outputs Linear.

## deberta_v2
- **src**: modeling_deberta_v2.py
- **status**: partial
- **rationale**: DeBERTa-v2 extends DeBERTa with bucket-style relative position attention + ConvLayer; same compute primitives (linear/matmul/softmax/layer_norm/conv1d/embedding) all available.
- **classes**:
  - **`DebertaV2Attention`** [compute]: no kb-nano kernel — DeBERTa-v2 extends DeBERTa with bucket-style relative position attention + ConvLayer; same compute primitives (linear/matmul/softmax/layer_norm/conv1d/embedding) all available.
  - **`DebertaV2SelfOutput`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (Linear + dropout + LayerNorm.)
  - **`DisentangledSelfAttention`** [compute]: `L1/linear.py`, `L1/softmax.py`, `L1/embedding.py` (v2 disentangled attention with bucket position, c2p + p2c.)
  - **`DebertaV2Intermediate`** [compute]: `L1/linear.py`, `L1/gelu.py` (Linear + GELU.)
  - **`DebertaV2Output`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (Linear + dropout + LayerNorm.)
  - **`DebertaV2Layer`** [wiring]: Wiring: attn + intermediate + output.
  - **`ConvLayer`** [compute]: `L1/conv1d.py`, `L1/layer_norm.py` (Conv1d + LayerNorm + dropout for first-layer enhancement.)
  - **`DebertaV2Embeddings`** [compute]: `L2/bert_embeddings.py`, `L1/embedding.py`, `L1/layer_norm.py` (Embeddings + optional projection + LayerNorm.)
  - **`DebertaV2Encoder`** [compute]: `L1/layer_norm.py`, `L1/embedding.py` (Wiring: layers + ConvLayer + relative position embeddings.)
  - **`DebertaV2Model`** [wiring]: Wiring: embeddings + encoder.
  - **`ContextPooler`** [compute]: `L1/linear.py` (Linear + activation pooler.)
  - **`DebertaV2ForMaskedLM`** [wiring]: Wiring: model + MLM head.
  - **`DebertaV2ForSequenceClassification`** [wiring]: Wiring: model + pooler + classifier.
  - **`DebertaV2ForTokenClassification`** [wiring]: Wiring: model + classifier.
  - **`DebertaV2ForQuestionAnswering`** [wiring]: Wiring: model + qa_outputs Linear.
  - **`DebertaV2ForMultipleChoice`** [wiring]: Wiring: model + pooler + classifier.

## decision_transformer
- **src**: modeling_decision_transformer.py
- **status**: composable
- **rationale**: GPT-2 style transformer (Conv1D fused QKV + MLP, LayerNorm, GELU/sdpa) wrapped for offline RL with state/action/return embeddings. All ops are standard.
- **classes**:
  - **`DecisionTransformerGPT2Attention`** [compute]: `L1/linear.py`, `L1/sdpa.py`, `L1/softmax.py` (GPT-2 style Conv1D-based attention (fused QKV) + sdpa.)
  - **`DecisionTransformerGPT2MLP`** [compute]: `L1/linear.py`, `L1/gelu.py` (Conv1D fc + activation + Conv1D proj.)
  - **`DecisionTransformerGPT2Block`** [wiring]: Wiring: ln_1 + attn + ln_2 + mlp + optional cross-attn.
  - **`DecisionTransformerGPT2Model`** [wiring]: Wiring: embeddings + blocks + ln_f.
  - **`DecisionTransformerModel`** [compute]: `L1/embedding.py`, `L1/linear.py`, `L1/layer_norm.py`, `L1/tanh.py` (Wiring: state/action/return embeddings + GPT2 transformer + heads (Linear+tanh).)

## deepseek_v2
- **src**: modular_deepseek_v2.py
- **status**: composable
- **rationale**: DeepSeek V2 is MLA attention + MoE with shared experts (Llama-style); kb-nano has L2/deepseek_mla_attention.py + L2/shared_expert_moe.py + L2/llama_mlp.py. Note: V2 uses torch.polar (complex RoPE) which differs slightly from kb-nano YarnRotaryEmbedding (which is real-valued); the polar form can be expressed as cos/sin pairs (composable).
- **classes**:
  - **`DeepseekV2Experts`** [compute]: `L1/moe_grouped_gemm.py`, `L1/silu_and_mul.py` (Qwen2-MoE style expert with SwiGLU; uses moe_grouped_gemm primitive.)
  - **`DeepseekV2Moe`** [compute]: `L2/shared_expert_moe.py`, `L2/deepseek_moe.py`, `L1/topk_softmax.py`, `L1/grouped_topk.py` (MoE with greedy / group_limited_greedy top-k routing + shared experts.)
  - **`DeepseekV2MLP`** [compute]: `L2/llama_mlp.py`, `L1/silu_and_mul.py` (SwiGLU dense MLP.)
  - **`DeepseekV2RMSNorm`** [compute]: `L1/rms_norm.py` (RMSNorm.)
  - **`DeepseekV2RotaryEmbedding`** [compute]: `L1/yarn_rotary_emb.py`, `L1/rotary_emb.py` (Complex (polar) form RoPE — can be realised via real cos/sin pairs in kb-nano.)
  - **`DeepseekV2Attention`** [compute]: `L2/deepseek_mla_attention.py`, `L2/mla_attention_impl.py`, `L1/rms_norm.py` (MLA: q_a_proj + q_b_proj + kv_a_proj_with_mqa + kv_b_proj + rope on partial dims; same pattern as kb-nano DeepSeekMLAAttention.)
  - **`DeepseekV2DecoderLayer`** [wiring]: Wiring: norm + MLA + norm + MoE-or-MLP.
  - **`DeepseekV2Model`** [wiring]: Wiring: embed + layers + norm.
  - **`DeepseekV2ForCausalLM`** [wiring]: Wiring: model + lm_head.
  - **`DeepseekV2ForSequenceClassification`** [wiring]: Wiring: model + classifier.

## deepseek_v3
- **src**: modular_deepseek_v3.py
- **status**: kb_nano_l4
- **rationale**: L4/deepseek.py is a full standalone DeepSeek V3.2 pipeline (MLA + MoE + grouped routing + YARN RoPE + FP8 + DSA indexer) targeting this exact family.
- **classes**:
  - **`DeepseekV3RMSNorm`** [compute]: `L1/rms_norm.py` (RMSNorm.)
  - **`DeepseekV3RotaryEmbedding`** [compute]: `L1/yarn_rotary_emb.py` (YARN-scaled RoPE.)
  - **`DeepseekV3MLP`** [compute]: `L2/llama_mlp.py`, `L1/silu_and_mul.py` (SwiGLU MLP.)
  - **`DeepseekV3TopkRouter`** [compute]: `L1/linear.py` (Linear router with e_score_correction_bias.)
  - **`DeepseekV3NaiveMoe`** [compute]: `L1/moe_grouped_gemm.py`, `L1/silu_and_mul.py` (Top-k experts with SwiGLU.)
  - **`DeepseekV3MoE`** [compute]: `L2/shared_expert_moe.py`, `L2/deepseek_moe.py`, `L1/sigmoid_topk.py`, `L1/grouped_topk.py` (MoE with sigmoid + group + shared experts.)
  - **`DeepseekV3Attention`** [compute]: `L2/deepseek_mla_attention.py`, `L2/mla_attention_impl.py`, `L1/yarn_rotary_emb.py` (MLA with optional q_lora_rank, YARN scaling, partial RoPE on rope_head_dim.)
  - **`DeepseekV3DecoderLayer`** [compute]: `L3/deepseek_decoder.py` (Wiring: norm + MLA + norm + MoE-or-MLP (matches L3).)
  - **`DeepseekV3Model`** [wiring]: Wiring: embed + layers + norm.
  - **`DeepseekV3ForCausalLM`** [compute]: `L4/deepseek.py` (L4 DeepSeek pipeline target.)
  - **`DeepseekV3ForSequenceClassification`** [wiring]: Wiring: model + classifier.
  - **`DeepseekV3ForTokenClassification`** [wiring]: Wiring: model + classifier.

## deepseek_v4
- **src**: modular_deepseek_v4.py
- **status**: partial
- **rationale**: DeepSeek V4 introduces novel Heavily Compressed Attention (HCA), Compressed Sparse Attention (CSA), Indexer for sparse attention, multi-rope-type Laguna-style rotary, HyperConnection routing, hash router, grouped output projection — none of these have kb-nano kernels.
- **classes**:
  - **`DeepseekV4Attention`** [compute]: no kb-nano kernel — DeepSeek V4 introduces novel Heavily Compressed Attention (HCA), Compressed Sparse Attention (CSA), Indexer for sparse attention, multi-rope-type Laguna-style rotary, HyperConnection routing, hash rou
  - **`DeepseekV4RMSNorm`** [compute]: `L1/rms_norm.py` (RMSNorm.)
  - **`DeepseekV4UnweightedRMSNorm`** [compute]: `L1/rms_norm_native.py` (RMSNorm without weight (rsqrt + scale only).)
  - **`DeepseekV4RotaryEmbedding`** [wiring]: Multi-rope-type (per layer-type) RoPE — not in kb-nano.
  - **`DeepseekV4HCACache`** [wiring]: Custom cache layer for compressed long-range entries; no kb-nano equivalent.
  - **`DeepseekV4CSACache`** [wiring]: Adds indexer + overlap state; no kb-nano equivalent.
  - **`DeepseekV4GroupedLinear`** [wiring]: Block-diagonal grouped linear (g groups bmm); no kb-nano kernel.
  - **`DeepseekV4HCACompressor`** [wiring]: Heavily Compressed Attention compressor — novel sparse-attention compressor; no kb-nano kernel.
  - **`DeepseekV4Indexer`** [wiring]: Sparse-attention indexer — different from V3 DSA indexer; no kb-nano equivalent for this V4 form.
  - **`DeepseekV4CSACompressor`** [wiring]: Compressed Sparse Attention compressor; no kb-nano kernel.
  - **`DeepseekV4HyperConnection`** [wiring]: Per-layer hyper-connection routing of residual streams; bespoke.
  - **`DeepseekV4HyperHead`** [wiring]: Hyper head; bespoke.
  - **`DeepseekV4MLP`** [compute]: `L2/llama_mlp.py` (SwiGLU MLP.)
  - **`DeepseekV4Experts`** [compute]: `L1/moe_grouped_gemm.py` (Mixtral-style experts.)
  - **`DeepseekV4TopKRouter`** [compute]: `L1/topk_softmax.py` (Top-k softmax router.)
  - **`DeepseekV4HashRouter`** [wiring]: Hash-based router; bespoke.
  - **`DeepseekV4SparseMoeBlock`** [wiring]: Wiring: router + experts.
  - **`DeepseekV4DecoderLayer`** [wiring]: Wiring: HyperConnection + attention + MoE.
  - **`DeepseekV4Model`** [wiring]: Wiring: embed + layers + norm.
  - **`DeepseekV4ForCausalLM`** [wiring]: Wiring: model + lm_head.

## deepseek_vl
- **src**: modular_deepseek_vl.py
- **status**: composable
- **rationale**: VL composes a SigLIP vision encoder + Llama text + a 2-layer MLP aligner (Linear+GELU+Linear); all parts available.
- **classes**:
  - **`DeepseekVLAligner`** [compute]: `L1/linear.py`, `L1/gelu.py` (Linear + GELU + Linear aligner.)
  - **`DeepseekVLModel`** [wiring]: Wiring: vision_model (AutoModel: SigLIP) + aligner + language_model (AutoModel: Llama).
  - **`DeepseekVLForConditionalGeneration`** [wiring]: Wiring: model + lm_head.

## deepseek_vl_hybrid
- **src**: modular_deepseek_vl_hybrid.py
- **status**: partial
- **rationale**: Hybrid adds a SAM vision encoder branch with neck + DeepseekVLSamVisionProj (Conv2d twice) and a hybrid aligner (two Linear projections + GELU + Linear); all primitives exist (Conv2d/interpolate/Linear/GELU).
- **classes**:
  - **`DeepseekVLHybridModel`** [compute]: no kb-nano kernel — Hybrid adds a SAM vision encoder branch with neck + DeepseekVLSamVisionProj (Conv2d twice) and a hybrid aligner (two Linear projections + GELU + Linear); all primitives exist (Conv2d/interpolate/Linea
  - **`DeepseekVLHybridLayerNorm`** [compute]: `L1/layer_norm2d.py` (SAM-style 2D LayerNorm pass-through.)
  - **`DeepseekVLSamVisionNeck`** [compute]: `L1/conv2d.py`, `L1/layer_norm2d.py` (Conv2d + LayerNorm neck for SAM features.)
  - **`DeepseekVLSamVisionProj`** [compute]: `L1/conv2d.py`, `L1/interpolate.py` (Bilinear interpolate + two strided Conv2d.)
  - **`DeepseekVLHybridAligner`** [compute]: `L1/linear.py`, `L1/gelu.py` (Two Linear projections (low-res/high-res) + GELU + final Linear.)
  - **`DeepseekVLHybridForConditionalGeneration`** [wiring]: Wiring: model + lm_head.

## deformable_detr
- **src**: modular_deformable_detr.py
- **status**: partial
- **rationale**: Deformable DETR uses MultiScaleDeformableAttention via grid_sample + standard transformer enc-dec. kb-nano has L1/grid_sample.py and L1/L2 rtdetrv2_deformable_attention.py (used for the same family).
- **classes**:
  - **`DeformableDetrConvEncoder`** [compute]: no kb-nano kernel — Deformable DETR uses MultiScaleDeformableAttention via grid_sample + standard transformer enc-dec. kb-nano has L1/grid_sample.py and L1/L2 rtdetrv2_deformable_attention.py (used for the same family).
  - **`MultiScaleDeformableAttention`** [compute]: `L1/grid_sample.py`, `L1/rtdetrv2_deformable_attention.py` (Bilinear sampling + weighted sum across multi-scale feature maps; kb-nano grid_sample primitive applies.)
  - **`DeformableDetrFrozenBatchNorm2d`** [compute]: `L1/frozen_batch_norm2d.py` (Frozen BN.)
  - **`DeformableDetrSinePositionEmbedding`** [compute]: `L1/sinusoidal_embed.py` (2D sine position embedding.)
  - **`DeformableDetrLearnedPositionEmbedding`** [compute]: `L1/embedding.py` (Learned 2D position embedding.)
  - **`DeformableDetrSelfAttention`** [compute]: `L1/linear.py`, `L1/sdpa.py` (Standard MHA.)
  - **`DeformableDetrMultiscaleDeformableAttention`** [compute]: `L2/rtdetrv2_deformable_attention.py`, `L1/rtdetrv2_deformable_attention.py`, `L1/linear.py`, `L1/softmax.py` (Sampling offsets + attention weights + multi-scale deformable kernel.)
  - **`DeformableDetrMLP`** [compute]: `L1/linear.py` (Stacked Linear + ReLU.)
  - **`DeformableDetrEncoderLayer`** [wiring]: Wiring: deformable self-attn + FFN.
  - **`DeformableDetrDecoderLayer`** [wiring]: Wiring: self-attn + cross-attn (deformable) + FFN.
  - **`DeformableDetrEncoder`** [wiring]: Wiring: encoder layers.
  - **`DeformableDetrDecoder`** [wiring]: Wiring: decoder layers + box refinement.
  - **`DeformableDetrModel`** [wiring]: Wiring: backbone + position + encoder + decoder.
  - **`DeformableDetrMLPPredictionHead`** [compute]: `L1/linear.py` (Stacked Linear MLP.)
  - **`DeformableDetrForObjectDetection`** [wiring]: Wiring: model + class/bbox heads.

## deimv2
- **src**: modular_deimv2.py
- **status**: composable
- **rationale**: Deim v2 inherits from D-Fine (which extends RT-DETR-V2) and uses DINOv3 ConvEncoder + RMSNorm + SwiGLU MLP + DFineMultiscaleDeformableAttention; all components exist in kb-nano (deformable attn, DINOv3 backbone is L4 dinov3).
- **classes**:
  - **`Deimv2RMSNorm`** [compute]: `L1/rms_norm.py` (RMSNorm.)
  - **`Deimv2SwiGLUFFN`** [compute]: `L2/llama_mlp.py`, `L1/silu_and_mul.py` (SwiGLU MLP.)
  - **`Deimv2Gate`** [compute]: `L1/linear.py`, `L1/sigmoid.py`, `L1/layer_norm.py` (Gating module pass-through.)
  - **`Deimv2MLP`** [compute]: `L1/linear.py` (Stacked Linear + activation.)
  - **`Deimv2MultiscaleDeformableAttention`** [compute]: `L2/rtdetrv2_deformable_attention.py`, `L1/rtdetrv2_deformable_attention.py` (Same deformable attention as D-Fine.)
  - **`Deimv2ConvNormLayer`** [compute]: `L2/rtdetrv2_conv_norm.py` (Conv + BN.)
  - **`Deimv2RepVggBlock`** [compute]: `L2/rtdetrv2_repvgg_block.py` (RepVGG block.)
  - **`Deimv2CSPRepLayer`** [compute]: `L2/rtdetrv2_csp_rep_layer.py` (CSP layer.)
  - **`Deimv2RepNCSPELAN5`** [wiring]: Wiring: ConvNormLayer + CSPRepLayer.
  - **`Deimv2SCDown`** [wiring]: Wiring: two ConvNormLayers.
  - **`Deimv2EncoderLayer`** [compute]: `L2/rtdetrv2_encoder_layer.py` (Encoder layer.)
  - **`Deimv2AIFILayer`** [compute]: `L2/rtdetrv2_layers.py` (AIFI.)
  - **`Deimv2SpatialTuningAdapter`** [compute]: `L1/conv2d.py`, `L1/layer_norm.py`, `L1/linear.py` (Spatial tuning adapter (Conv + LayerNorm + Linear).)
  - **`Deimv2ConvEncoder`** [wiring]: Wiring: backbone.
  - **`Deimv2DINOv3ConvEncoder`** [compute]: `L4/dinov3.py` (Uses DINOv3 backbone (kb-nano L4).)
  - **`Deimv2Integral`** [compute]: `L1/softmax.py`, `L1/linear.py` (Integral bbox head.)
  - **`Deimv2LQE`** [compute]: `L1/softmax.py`, `L1/linear.py` (LQE.)
  - **`Deimv2DecoderLayer`** [wiring]: Wiring: self-attn + deformable cross + gate + MLP.
  - **`Deimv2LiteEncoder`** [wiring]: Wiring: lite encoder.
  - **`Deimv2HybridEncoder`** [compute]: `L3/rtdetrv2_hybrid_encoder.py` (Hybrid encoder.)
  - **`Deimv2Decoder`** [compute]: `L3/rtdetrv2_decoder.py` (Decoder with FDR.)

## deit
- **src**: modeling_deit.py
- **status**: composable
- **rationale**: DeiT is a standard ViT (patch embedding via Conv2d, fused QKV linear + LayerNorm + GELU MLP, learned position + cls/dist tokens); kb-nano has vit_encoder_attention.py + vit_encoder_mlp.py.
- **classes**:
  - **`DeiTEmbeddings`** [compute]: `L2/vision_patch_embed.py`, `L1/embedding.py` (Patch + cls + distillation tokens + learned position embeddings.)
  - **`DeiTPatchEmbeddings`** [compute]: `L1/conv2d.py` (Conv2d patch projection.)
  - **`DeiTSelfAttention`** [compute]: `L2/vit_encoder_attention.py`, `L1/linear.py`, `L1/sdpa.py` (Standard ViT QKV + sdpa attention.)
  - **`DeiTSelfOutput`** [compute]: `L1/linear.py` (Output projection Linear.)
  - **`DeiTAttention`** [wiring]: Wiring: self-attn + output.
  - **`DeiTIntermediate`** [compute]: `L1/linear.py`, `L1/gelu.py` (Linear + GELU.)
  - **`DeiTOutput`** [compute]: `L1/linear.py` (Linear projection.)
  - **`DeiTLayer`** [compute]: `L3/vit_encoder_block.py` (Wiring: layernorm + attn + layernorm + mlp.)
  - **`DeiTEncoder`** [wiring]: Wiring: layers.
  - **`DeiTModel`** [wiring]: Wiring: embeddings + encoder + layernorm + pooler.
  - **`DeiTPooler`** [compute]: `L1/linear.py`, `L1/tanh.py` (Linear + tanh pooler.)
  - **`DeiTForMaskedImageModeling`** [wiring]: Wiring: model + Conv2d decoder.
  - **`DeiTForImageClassification`** [wiring]: Wiring: model + classifier Linear.
  - **`DeiTForImageClassificationWithTeacher`** [wiring]: Wiring: model + cls + distillation Linear heads.

## depth_anything
- **src**: modeling_depth_anything.py
- **status**: composable
- **rationale**: DepthAnything composes a vision backbone (DINOv2 etc. via AutoBackbone) + reassemble (Conv1x1 + ConvTranspose) + feature fusion (residual blocks with Conv2d/BN) + depth estimation head; all primitives exist (conv2d/conv_transpose2d/batch_norm2d/relu/interpolate).
- **classes**:
  - **`DepthAnythingReassembleLayer`** [compute]: `L1/conv2d.py`, `L1/conv_transpose2d.py` (Conv1x1 + optional ConvTranspose for upsampling.)
  - **`DepthAnythingReassembleStage`** [wiring]: Wiring: reassemble layers.
  - **`DepthAnythingPreActResidualLayer`** [compute]: `L1/conv2d.py`, `L1/relu.py` (Pre-act ReLU + Conv2d residual.)
  - **`DepthAnythingFeatureFusionLayer`** [compute]: `L1/conv2d.py`, `L1/interpolate.py` (Wiring: residual + interpolate + Conv2d projection.)
  - **`DepthAnythingFeatureFusionStage`** [wiring]: Wiring: fusion layers.
  - **`DepthAnythingNeck`** [wiring]: Wiring: reassemble + fusion.
  - **`DepthAnythingDepthEstimationHead`** [compute]: `L1/conv2d.py`, `L1/relu.py`, `L1/interpolate.py` (Conv2d + interpolate + Conv2d + ReLU + Conv2d head.)
  - **`DepthAnythingForDepthEstimation`** [wiring]: Wiring: backbone + neck + head.

## depth_pro
- **src**: modeling_depth_pro.py
- **status**: composable
- **rationale**: DepthPro composes multi-scale ViT (AutoModel patch encoder) + image encoder + feature upsample (Conv + ConvTranspose stacks) + DPT-style depth head + FOV head; all kernels (conv2d/conv_transpose2d/interpolate/sdpa/layer_norm/linear/gelu/relu) exist.
- **classes**:
  - **`DepthProPatchEncoder`** [compute]: `L1/interpolate.py` (Wiring: AutoModel ViT patch encoder + bilinear interpolate.)
  - **`DepthProImageEncoder`** [wiring]: Wiring: AutoModel image encoder + reconstruct features.
  - **`DepthProEncoder`** [wiring]: Wiring: patch encoder + image encoder.
  - **`DepthProFeatureUpsampleBlock`** [compute]: `L1/conv2d.py`, `L1/conv_transpose2d.py` (Conv2d + ConvTranspose2d upsample.)
  - **`DepthProFeatureUpsample`** [wiring]: Wiring: feature upsample blocks.
  - **`DepthProFeatureProjection`** [compute]: `L1/conv2d.py` (Conv2d feature projection.)
  - **`DepthProNeck`** [wiring]: Wiring: feature upsample + projection.
  - **`DepthProModel`** [wiring]: Wiring: encoder + neck.
  - **`DepthProPreActResidualLayer`** [compute]: `L1/conv2d.py`, `L1/relu.py` (Pre-act residual.)
  - **`DepthProFeatureFusionLayer`** [wiring]: Wiring: residual + interpolate.
  - **`DepthProFeatureFusionStage`** [wiring]: Wiring: fusion layers.
  - **`DepthProFovEncoder`** [wiring]: Wiring: AutoModel ViT + interpolate.
  - **`DepthProFovHead`** [compute]: `L1/linear.py`, `L1/conv2d.py` (FOV head: Conv + Linear.)
  - **`DepthProFovModel`** [wiring]: Wiring: FovEncoder + FovHead.
  - **`DepthProDepthEstimationHead`** [compute]: `L1/conv2d.py`, `L1/relu.py`, `L1/interpolate.py` (Conv2d + interpolate + Conv2d + ReLU + Conv2d head.)
  - **`DepthProForDepthEstimation`** [wiring]: Wiring: model + fusion + depth head + fov model.

## detr
- **src**: modeling_detr.py
- **status**: partial
- **rationale**: Standard DETR enc-dec: ResNet backbone (AutoBackbone) + sinusoidal/learned position embeddings + standard MHA self/cross attention + MLP. All primitives exist.
- **classes**:
  - **`DetrConvEncoder`** [compute]: no kb-nano kernel — Standard DETR enc-dec: ResNet backbone (AutoBackbone) + sinusoidal/learned position embeddings + standard MHA self/cross attention + MLP. All primitives exist.
  - **`DetrFrozenBatchNorm2d`** [compute]: `L1/frozen_batch_norm2d.py` (Frozen BN.)
  - **`DetrSinePositionEmbedding`** [compute]: `L1/sinusoidal_embed.py` (2D sinusoidal position embedding.)
  - **`DetrLearnedPositionEmbedding`** [compute]: `L1/embedding.py` (Learned 2D position embedding.)
  - **`DetrSelfAttention`** [compute]: `L1/linear.py`, `L1/sdpa.py`, `L1/softmax.py` (Standard MHA with separate Q/K/V Linear.)
  - **`DetrCrossAttention`** [compute]: `L1/linear.py`, `L1/sdpa.py`, `L1/softmax.py` (Cross-attention from queries to encoder features.)
  - **`DetrMLP`** [compute]: `L1/linear.py`, `L1/relu.py` (Linear + ReLU + Linear.)
  - **`DetrEncoderLayer`** [wiring]: Wiring: self-attn + FFN.
  - **`DetrDecoderLayer`** [wiring]: Wiring: self-attn + cross-attn + FFN.
  - **`DetrConvBlock`** [compute]: `L1/conv2d.py`, `L1/group_norm.py`, `L1/relu.py` (Conv + GroupNorm + ReLU.)
  - **`DetrFPNFusionStage`** [wiring]: Wiring: ConvBlocks + interpolate.
  - **`DetrMaskHeadSmallConv`** [compute]: `L1/conv2d.py`, `L1/group_norm.py` (Mask head conv stack.)
  - **`DetrMHAttentionMap`** [compute]: `L1/linear.py`, `L1/conv2d.py`, `L1/softmax.py` (Multi-head attention map for segmentation.)
  - **`DetrEncoder`** [wiring]: Wiring: encoder layers.
  - **`DetrDecoder`** [wiring]: Wiring: decoder layers.
  - **`DetrModel`** [wiring]: Wiring: backbone + transformer.
  - **`DetrMLPPredictionHead`** [compute]: `L1/linear.py` (Stacked Linear MLP.)
  - **`DetrForObjectDetection`** [wiring]: Wiring: model + class/bbox heads.
  - **`DetrForSegmentation`** [wiring]: Wiring: detection + mask head.

## dia
- **src**: modular_dia.py
- **status**: composable
- **rationale**: Dia (TTS) is a Llama-style enc-dec with Phi3 SwiGLU MLP and standard self/cross attention; all components covered (LlamaAttention, llama_mlp, rms_norm, rotary_emb, multichannel embedding via L1/embedding).
- **classes**:
  - **`DiaMultiChannelEmbedding`** [compute]: `L1/embedding.py` (Embedding with per-channel offset; sum across channels.)
  - **`DiaMLP`** [compute]: `L2/llama_mlp.py`, `L1/silu_and_mul.py` (SwiGLU MLP (Phi3 = same shape as LlamaMLP).)
  - **`DiaRMSNorm`** [compute]: `L1/rms_norm.py` (RMSNorm.)
  - **`DiaRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (Llama RoPE.)
  - **`DiaSelfAttention`** [compute]: `L2/attention.py` (Llama self-attention.)
  - **`DiaCrossAttention`** [compute]: `L1/linear.py`, `L1/sdpa.py`, `L1/rms_norm.py` (Cross-attention from decoder to encoder hidden states.)
  - **`DiaEncoderLayer`** [wiring]: Wiring: norm + self-attn + norm + mlp.
  - **`DiaEncoder`** [wiring]: Wiring: embed + layers + norm.
  - **`DiaDecoderLayer`** [wiring]: Wiring: self-attn + cross-attn + mlp.
  - **`DiaDecoder`** [wiring]: Wiring: multichannel embedding + decoder layers + norm.
  - **`DiaModel`** [wiring]: Wiring: encoder + decoder.
  - **`DiaForConditionalGeneration`** [wiring]: Wiring: model + lm_head per channel.

## diffllama
- **src**: modular_diffllama.py
- **status**: unsupported
- **rationale**: DiffLlama is Llama with Differential Transformer attention (two SDPA halves combined as attn1 - lambda * attn2 + GroupNorm). All ops are SDPA / matmul / softmax / linear / RMSNorm / SwiGLU MLP — composable using sdpa primitive twice.
- **classes**:
  - **`DiffLlamaDecoderLayer`** [compute]: no kb-nano kernel — DiffLlama is Llama with Differential Transformer attention (two SDPA halves combined as attn1 - lambda * attn2 + GroupNorm). All ops are SDPA / matmul / softmax / linear / RMSNorm / SwiGLU MLP — compo
  - **`DiffLlamaMLP`** [compute]: `L2/llama_mlp.py`, `L1/silu_and_mul.py` (SwiGLU MLP (Mistral = Llama).)
  - **`DiffLlamaRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (Llama RoPE.)
  - **`DiffLlamaAttention`** [compute]: `L1/linear.py`, `L1/sdpa.py`, `L1/softmax.py`, `L1/rms_norm.py` (Differential attention: split V into two halves, compute attn1 - lambda*attn2, apply unweighted RMSNorm; uses linear + matmul + softmax + RMSNorm primitives.)
  - **`DiffLlamaFlashAttention2`** [compute]: `L1/flash_attn_prefill.py`, `L1/flash_attn_decode.py` (FlashAttention2 variant; calls _flash_attention_forward twice (once per V half).)
  - **`DiffLlamaSdpaAttention`** [compute]: `L1/sdpa.py` (SDPA-backed differential attention.)
  - **`DiffLlamaModel`** [wiring]: Wiring: embed + layers + norm.
  - **`DiffLlamaForCausalLM`** [wiring]: Wiring: model + lm_head.
  - **`DiffLlamaForSequenceClassification`** [wiring]: Wiring: model + classifier.
  - **`DiffLlamaForQuestionAnswering`** [wiring]: Wiring: model + qa_outputs.
  - **`DiffLlamaForTokenClassification`** [wiring]: Wiring: model + classifier.
