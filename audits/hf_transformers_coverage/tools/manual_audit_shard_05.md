## evolla
- **src**: modular_evolla.py
- **status**: partial
- **rationale**: Multimodal protein-text LM with bespoke ESM (BERT-like) protein encoder + Perceiver-style sequence compressor + cross-attention sequence aligner injected into Llama decoder; the resampler/aligner cross-attention layouts are bespoke and not in kb-nano.
- **classes**:
  - **`EvollaSaProtAttention`** [compute]: no kb-nano kernel — Multimodal protein-text LM with bespoke ESM (BERT-like) protein encoder + Perceiver-style sequence compressor + cross-attention sequence aligner injected into Llama decoder; the resampler/aligner cros
  - **`EvollaSaProtEmbeddings`** [wiring]: Token + position embedding wiring for ESM-style protein encoder; composes.
  - **`EvollaSaProtRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (Standard NeoX-style rotary embedding.)
  - **`EvollaSaProtSelfAttention`** [compute]: `L2/encoder_attention.py` (ESM self-attention is BERT-like (q/k/v/o linears with rotary applied to q/k); maps to EncoderSelfAttention with added rotary outside.)
  - **`EvollaSaProtSelfOutput`** [wiring]: Linear+dropout+residual wiring.
  - **`EvollaSaProtIntermediate`** [compute]: `L2/encoder_mlp.py` (Linear + GELU; matches EncoderIntermediate.)
  - **`EvollaSaProtOutput`** [compute]: `L2/encoder_mlp.py` (Linear + dropout + LayerNorm + residual; matches EncoderOutput.)
  - **`EvollaSaProtLayer`** [wiring]: Wiring of attention + intermediate + output.
  - **`EvollaSaProtEncoder`** [wiring]: Stack of layers; composes.
  - **`EvollaSaProtPooler`** [wiring]: Linear + tanh pooler; composes.
  - **`EvollaSaProtProteinEncoder`** [wiring]: Top-level ESM protein encoder model wiring; composes.
  - **`EvollaSequenceCompressorAttention`** [wiring]: Perceiver-style cross-attention: concat(image_features, latents) for kv, dense matmul + max-trick softmax + boolean masked_fill; bespoke compute, no kb-nano equivalent.
  - **`EvollaFeedForward`** [wiring]: LayerNorm + Linear + GELU + Linear; composable from L1 ops but no exact L2 wrapper (encoder_mlp does fc1+gelu+fc2 with terminal LN, not pre-norm).
  - **`EvollaSequenceCompressorResampler`** [wiring]: Perceiver Resampler with learned latents; bespoke wiring + uses EvollaSequenceCompressorAttention which has no kb-nano kernel.
  - **`EvollaProteinEncoder`** [wiring]: Wires ESM encoder + resampler; composes.
  - **`EvollaSequenceAlignerCrossAttention`** [wiring]: Multi-modality cross-attention that concatenates protein+structure+msa kv with custom gates and FFN gating; bespoke.
  - **`EvollaRMSNorm`** [compute]: `L1/rms_norm.py` (Standard Llama RMSNorm.)
  - **`EvollaRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (Standard NeoX rotary.)
  - **`EvollaMLP`** [compute]: `L2/llama_mlp.py` (Standard SwiGLU MLP.)
  - **`EvollaAttention`** [compute]: `L2/attention.py` (Standard Llama attention.)
  - **`EvollaDecoderLayer`** [wiring]: Decoder block with optional cross-attn injection from sequence aligner; wiring.
  - **`EvollaModel`** [wiring]: Top model wiring; composes.
  - **`EvollaForProteinText2Text`** [wiring]: LM head + generation wiring; composes.

## exaone4
- **src**: modular_exaone4.py
- **status**: composable
- **rationale**: Llama-family decoder with Olmo2-style post-norm decoder layer and QK-RMSNorm; sliding-window + global attention. All ops map to existing kb-nano L1/L2 kernels.
- **classes**:
  - **`Exaone4RMSNorm`** [compute]: `L1/rms_norm.py` (Standard Llama RMSNorm.)
  - **`Exaone4RotaryEmbedding`** [compute]: `L1/rotary_emb.py` (Default RoPE; same NeoX rotation as Llama.)
  - **`Exaone4Attention`** [compute]: `L2/attention.py` (GQA with QK-RMSNorm and per-layer global/sliding pattern; matches LlamaAttention(qk_norm=True, sliding_window=...).)
  - **`Exaone4MLP`** [compute]: `L2/llama_mlp.py` (Olmo2MLP is the same gate/up/down SwiGLU pattern.)
  - **`Exaone4DecoderLayer`** [wiring]: Post-norm decoder block wiring.
  - **`Exaone4Model`** [wiring]: Top model wiring.
  - **`Exaone4ForCausalLM`** [wiring]: LM head + generation wiring.
  - **`Exaone4ForSequenceClassification`** [wiring]: Classification head wiring.
  - **`Exaone4ForTokenClassification`** [wiring]: Token classification head wiring.
  - **`Exaone4ForQuestionAnswering`** [wiring]: QA head wiring.

## exaone4_5
- **src**: modular_exaone4_5.py
- **status**: partial
- **rationale**: EXAONE 4.5 is a Qwen2.5-VL multimodal model with vision encoder + 2D-RoPE + GQA. The vision tower components (ViT-style with 2D RoPE) and the multimodal projector pipeline have no kb-nano L4 wrapper or vision-encoder L2/L3 stack equivalent.
- **classes**:
  - **`Exaone4_5_VisionBlock`** [compute]: EXAONE 4.5 is a Qwen2.5-VL multimodal model with vision encoder + 2D-RoPE + GQA. The vision tower components (ViT-style with 2D RoPE) and the multimodal projector pipeline have no kb-nano L4 wrapper o
  - **`Exaone4_5_PatchEmbed`** [compute]: `L2/vision_patch_embed.py` (Vision patch embedding via Conv3d; kb-nano has VisionPatchEmbed for similar pattern.)
  - **`Exaone4_5_VisionRotaryEmbedding`** [compute]: `L1/vision_rotary_emb.py` (2D vision RoPE; kb-nano has VisionRotaryEmb.)
  - **`Exaone4_5_PatchMerger`** [compute]: `L2/vision_patch_merger.py` (Spatial merging via reshape + Linear.)
  - **`Exaone4_5_VisionAttention`** [compute]: `L2/vision_attention.py` (Vision attention with cu_seqlens varlen + GQA QKV; close to vision_attention but bespoke GQA QKV layout makes drop-in non-trivial.)
  - **`Exaone4_5_MLP`** [compute]: `L2/llama_mlp.py` (Standard SwiGLU MLP.)
  - **`Exaone4_5_VisionModel`** [wiring]: Vision tower wiring; no end-to-end Qwen2.5-VL vision pipeline in kb-nano.
  - **`Exaone4_5_Model`** [wiring]: Multimodal model wiring (vision + AutoModel text).
  - **`Exaone4_5_ForConditionalGeneration`** [wiring]: Top-level conditional generation; no kb-nano equivalent for Qwen2.5-VL multimodal pipeline.

## exaone_moe
- **src**: modular_exaone_moe.py
- **status**: composable
- **rationale**: Exaone4 backbone (LlamaAttention with QK-norm + sliding) + DeepSeek V3-style MoE block (TopkRouter with score-correction-bias + grouped experts + shared expert). All map to existing kb-nano L2 kernels.
- **classes**:
  - **`ExaoneMoeAttention`** [compute]: `L2/attention.py` (Same as Exaone4Attention (qk-norm + sliding GQA).)
  - **`ExaoneMoeMLP`** [compute]: `L2/llama_mlp.py` (Standard SwiGLU MLP used as dense layer / shared expert.)
  - **`ExaoneMoeTopkRouter`** [wiring]: Routing logic with score correction bias; matches DeepseekV3 router used inside L2/shared_expert_moe.py.
  - **`ExaoneMoeExperts`** [compute]: `L1/moe_grouped_gemm.py` (Naive MoE expert MLPs; backed by grouped-GEMM L1 kernel.)
  - **`ExaoneMoeSparseMoEBlock`** [compute]: `L2/shared_expert_moe.py` (Routed experts + shared expert block; matches L2/shared_expert_moe.py pattern (DeepSeek/Qwen3-Next family).)
  - **`ExaoneMoeDecoderLayer`** [wiring]: Decoder block wiring with dense-or-sparse MLP per layer.
  - **`ExaoneMoeModel`** [wiring]: Top model wiring.
  - **`ExaoneMoeForCausalLM`** [wiring]: LM head wiring.

## falcon
- **src**: modeling_falcon.py
- **status**: partial
- **rationale**: Falcon supports both RoPE (new arch) and ALiBi (legacy 7B/40B) attention biases plus parallel-attention layer topology. kb-nano attention kernels (LlamaAttention/Attention impl) have no ALiBi additive bias support.
- **classes**:
  - **`FalconAttention`** [compute]: no kb-nano kernel — Falcon supports both RoPE (new arch) and ALiBi (legacy 7B/40B) attention biases plus parallel-attention layer topology. kb-nano attention kernels (LlamaAttention/Attention impl) have no ALiBi additive
  - **`FalconLinear`** [compute]: `L1/linear.py` (Plain linear with explicit transpose; equivalent to Linear.)
  - **`FalconRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (Standard NeoX rotary.)
  - **`FalconFlashAttention2`** [wiring]: FA2 backend variant of FalconAttention; same gap.
  - **`FalconMLP`** [compute]: `L2/encoder_mlp.py` (Two-layer dense_h_to_4h -> GELU -> dense_4h_to_h, matches non-SwiGLU encoder_mlp pattern (no LayerNorm in MLP).)
  - **`FalconDecoderLayer`** [wiring]: Parallel-attention block wiring with optional second LN; composes.
  - **`FalconModel`** [wiring]: Top model wiring.
  - **`FalconForCausalLM`** [wiring]: LM head wiring.
  - **`FalconForSequenceClassification`** [wiring]: Classification head wiring.
  - **`FalconForTokenClassification`** [wiring]: Token classification head wiring.
  - **`FalconForQuestionAnswering`** [wiring]: QA head wiring.

## falcon_h1
- **src**: modular_falcon_h1.py
- **status**: composable
- **rationale**: Llama-style attention (with key_multiplier scaling) interleaved with Mamba2 mixer (FalconH1Mixer = Mamba2Mixer with optional gated RMSNorm and ssm_in/zxbcdt multipliers). Both pieces map to kb-nano kernels.
- **classes**:
  - **`FalconH1RotaryEmbedding`** [compute]: `L1/rotary_emb.py` (Standard NeoX rotary.)
  - **`FalconH1Attention`** [compute]: `L2/attention.py` (GQA Llama attention with extra key_multiplier; small numeric tweak that fits LlamaAttention with a key scaling outside (variant of Llama).)
  - **`FalconH1RMSNormGated`** [compute]: `L1/rms_norm_gated.py` (Gated RMSNorm with n_groups + optional pre/post gate; matches kb-nano rms_norm_gated.)
  - **`FalconH1Mixer`** [compute]: `L2/mamba2_mixer.py` (Mamba2 SSD mixer with extra ssm/zxbcdt multipliers; structurally identical to Mamba2Mixer with extra elementwise scaling layers.)
  - **`FalconH1MLP`** [compute]: `L2/llama_mlp.py` (Standard SwiGLU MLP.)
  - **`FalconH1RMSNorm`** [compute]: `L1/rms_norm.py` (Standard RMSNorm.)
  - **`FalconH1DecoderLayer`** [wiring]: Hybrid Mamba/Attention decoder layer wiring (similar to Jamba pattern).
  - **`FalconH1Model`** [wiring]: Top model wiring.
  - **`FalconH1ForCausalLM`** [wiring]: LM head wiring.

## falcon_mamba
- **src**: modular_falcon_mamba.py
- **status**: composable
- **rationale**: Falcon-Mamba is vanilla Mamba v1 with extra RMSNorm applied to B/C/dt; the underlying mixer/cache pattern matches kb-nano's mamba L4 pipeline (the rms-on-bcdt is a small variant).
- **classes**:
  - **`FalconMambaMixer`** [compute]: `L2/mamba_mixer.py` (Identical to MambaMixer with extra rms_forward applied to B, C and dt before the SSM (small numeric variant of MambaMixer).)
  - **`FalconMambaRMSNorm`** [compute]: `L1/rms_norm.py` (Standard RMSNorm.)
  - **`FalconMambaBlock`** [wiring]: Wires norm + mixer + residual.
  - **`FalconMambaModel`** [wiring]: Top model wiring; mirrors L4/mamba.py.
  - **`FalconMambaForCausalLM`** [wiring]: LM head wiring; mirrors L4/mamba.py.

## fast_vlm
- **src**: modular_fast_vlm.py
- **status**: unsupported
- **unsupported_reason**: Vision tower is FastViT pulled from timm via timm_wrapper config; kb-nano has no FastViT/timm L4 nor an L3 stack for it. Multimodal projector and Llava plumbing also have no kb-nano equivalent.
- **rationale**: Llava-style multimodal wrapper over a Qwen2 LM + FastViT (timm) vision tower. The vision backbone is loaded via timm_wrapper, which kb-nano cannot run.
- **classes**:
  - **`FastVlmMultiModalProjector`** [compute]: Vision tower is FastViT pulled from timm via timm_wrapper config; kb-nano has no FastViT/timm L4 nor an L3 stack for it. Multimodal projector and Llava plumbing also have no kb-nano equivalent.
  - **`FastVlmModel`** [wiring]: Llava-style vision_tower + multi_modal_projector + language_model wiring; no kb-nano Llava pipeline.
  - **`FastVlmForConditionalGeneration`** [wiring]: Top-level conditional generation; no kb-nano equivalent.

## fastspeech2_conformer
- **src**: modeling_fastspeech2_conformer.py
- **status**: partial
- **rationale**: Conformer-based TTS encoder with relative positional encoding (Transformer-XL style) attention plus duration/pitch/energy predictors and HiFi-GAN vocoder. The relative-position attention with learnable u/v biases and the conformer convolution module are bespoke; HiFi-GAN is a custom transposed-conv vocoder.
- **classes**:
  - **`FastSpeech2ConformerBatchNormConvLayer`** [compute]: no kb-nano kernel — Conformer-based TTS encoder with relative positional encoding (Transformer-XL style) attention plus duration/pitch/energy predictors and HiFi-GAN vocoder. The relative-position attention with learnabl
  - **`FastSpeech2ConformerDurationPredictor`** [wiring]: Conv1d + LayerNorm + ReLU + Linear stack; composable from L1 ops but no L2/L3 wrapper.
  - **`FastSpeech2ConformerSpeechDecoderPostnet`** [wiring]: Postnet wiring; depends on BatchNormConvLayer.
  - **`FastSpeech2ConformerPredictorLayer`** [wiring]: Conv1d + ReLU + LayerNorm + Dropout.
  - **`FastSpeech2ConformerVariancePredictor`** [wiring]: Stack of PredictorLayers + projection.
  - **`FastSpeech2ConformerVarianceEmbedding`** [wiring]: Conv1d embedding.
  - **`FastSpeech2ConformerAttention`** [wiring]: Relative-position multi-head attention with u/v biases (Transformer-XL); no kb-nano kernel for this rel-pos formulation.
  - **`FastSpeech2ConformerConvolutionModule`** [wiring]: Conformer pointwise+depthwise conv + BatchNorm1d + Swish; uses BatchNorm1d not in kb-nano.
  - **`FastSpeech2ConformerEncoderLayer`** [wiring]: Conformer block wiring (FFN + Attn + ConvModule + FFN + LN).
  - **`FastSpeech2ConformerMultiLayeredConv1d`** [wiring]: Conv1d-based positionwise FFN.
  - **`FastSpeech2ConformerRelPositionalEncoding`** [wiring]: Relative sinusoidal positional encoding; bespoke to conformer.
  - **`FastSpeech2ConformerEncoder`** [wiring]: Stack of encoder layers.
  - **`FastSpeech2ConformerLoss`** [wiring]: Loss; not inference.
  - **`FastSpeech2ConformerModel`** [wiring]: Top model wiring.
  - **`HifiGanResidualBlock`** [wiring]: WeightNorm-wrapped Conv1d residual block; no kb-nano wrapper.
  - **`FastSpeech2ConformerHifiGan`** [wiring]: HiFi-GAN vocoder (transposed-conv stacks + residual blocks); no kb-nano L4 for HiFi-GAN.
  - **`FastSpeech2ConformerWithHifiGan`** [wiring]: Wires the TTS encoder + HiFi-GAN vocoder.

## flaubert
- **src**: modeling_flaubert.py
- **status**: partial
- **rationale**: Flaubert (XLM-derived) uses BART-style attention with separate q_lin/k_lin/v_lin and supports both encoder and cross-attention with EncoderDecoderCache. The kb-nano encoder_attention.py does not support cross-attention, and the bare q/k/v separate projection layout differs from QKVParallelLinear in attention.py.
- **classes**:
  - **`MultiHeadAttention`** [compute]: no kb-nano kernel — Flaubert (XLM-derived) uses BART-style attention with separate q_lin/k_lin/v_lin and supports both encoder and cross-attention with EncoderDecoderCache. The kb-nano encoder_attention.py does not suppo
  - **`TransformerFFN`** [wiring]: Two-layer Linear + GELU/ReLU + dropout; close to encoder_mlp.py but no LayerNorm inside.
  - **`FlaubertPredLayer`** [wiring]: Adaptive softmax / linear LM head wiring.
  - **`FlaubertPoolerStartLogits`** [wiring]: Linear pooler for SQuAD; composable from Linear.
  - **`FlaubertPoolerEndLogits`** [wiring]: Linear + tanh + LN + linear pooler.
  - **`FlaubertPoolerAnswerClass`** [wiring]: Pooler MLP; composable.
  - **`FlaubertSQuADHead`** [wiring]: Top-level SQuAD head wiring.
  - **`FlaubertSequenceSummary`** [wiring]: Pool-then-linear head; composable.
  - **`FlaubertModel`** [wiring]: Top model wiring with sinusoidal pos embed + sequential MultiHeadAttention layers.
  - **`FlaubertWithLMHeadModel`** [wiring]: LM head wiring.
  - **`FlaubertForSequenceClassification`** [wiring]: Classification head wiring.
  - **`FlaubertForTokenClassification`** [wiring]: Token classification head wiring.
  - **`FlaubertForQuestionAnsweringSimple`** [wiring]: QA head wiring.
  - **`FlaubertForQuestionAnswering`** [wiring]: QA head wiring with SQuAD head.
  - **`FlaubertForMultipleChoice`** [wiring]: Multiple choice head wiring.

## flava
- **src**: modeling_flava.py
- **status**: composable
- **rationale**: FLAVA is a triple-tower (image ViT + text BERT + multimodal BERT) plus a discrete VQ-VAE codebook. The transformer towers are standard ViT/BERT structure that maps to kb-nano EncoderSelfAttention + EncoderIntermediate/Output + Conv2d patch embed. FlavaSelfAttention is BERT-style (separate q/k/v linears + softmax matmul). The codebook is just Conv2d/ReLU stacks.
- **classes**:
  - **`FlavaImageEmbeddings`** [compute]: `L2/vision_patch_embed.py` (Conv2d patch embed + cls token + position embeddings; matches vision_patch_embed pattern.)
  - **`PatchEmbeddings`** [compute]: `L1/conv2d.py` (Plain Conv2d patch projection.)
  - **`FlavaTextEmbeddings`** [compute]: `L2/encoder_embeddings.py` (Token + position + token-type embeddings + LayerNorm; standard BERT embeddings.)
  - **`FlavaSelfAttention`** [compute]: `L2/encoder_attention.py` (BERT-style self-attention with separate q/k/v linears; matches EncoderSelfAttention.)
  - **`FlavaSelfOutput`** [wiring]: Linear + dropout + residual; wiring.
  - **`FlavaAttention`** [wiring]: Wraps SelfAttention + SelfOutput; sibling-wrapper pattern (kernel is on FlavaSelfAttention).
  - **`FlavaIntermediate`** [compute]: `L2/encoder_mlp.py` (Linear + GELU; matches EncoderIntermediate.)
  - **`FlavaOutput`** [compute]: `L2/encoder_mlp.py` (Linear + dropout + LayerNorm + residual; matches EncoderOutput.)
  - **`FlavaLayer`** [wiring]: Pre-LN attention + intermediate/output wiring.
  - **`FlavaEncoder`** [wiring]: Stack of FlavaLayer; composes.
  - **`FlavaPooler`** [wiring]: Linear + tanh pooler.
  - **`FlavaImageModel`** [wiring]: ViT image tower wiring.
  - **`FlavaTextModel`** [wiring]: BERT text tower wiring.
  - **`FlavaMultimodalModel`** [wiring]: Multimodal BERT tower wiring.
  - **`FlavaModel`** [wiring]: Top-level model wiring.
  - **`FlavaImageCodebookResPath`** [wiring]: Conv2d + ReLU residual path; composable from L1 conv2d/relu.
  - **`FlavaImageCodebookBlock`** [wiring]: Conv-residual block wiring.
  - **`FlavaImageCodebookLayerGroup`** [wiring]: Group of codebook blocks; composes.
  - **`FlavaImageCodebook`** [wiring]: VQ-VAE codebook (Conv2d stacks + max_pool); composable.
  - **`FlavaPredictionHeadTransform`** [wiring]: Linear + GELU + LN; composable.
  - **`FlavaMaskedPredictionHead`** [wiring]: MLM head wiring.
  - **`FlavaITMHead`** [wiring]: Image-text matching head.
  - **`FlavaGlobalContrastiveHead`** [wiring]: Contrastive head wiring.
  - **`FlavaForPreTraining`** [wiring]: Pretraining wiring.

## flex_olmo
- **src**: modular_flex_olmo.py
- **status**: composable
- **rationale**: Flex-OLMo = Olmo2 attention + OLMoE MoE block. All components map to existing kb-nano L1/L2 kernels (LlamaAttention with QK-norm pattern, llama_mlp for shared, fused MoE for routed experts).
- **classes**:
  - **`FlexOlmoRMSNorm`** [compute]: `L1/rms_norm.py` (Standard RMSNorm.)
  - **`FlexOlmoRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (Standard NeoX rotary.)
  - **`FlexOlmoMLP`** [compute]: `L2/llama_mlp.py` (Standard SwiGLU MLP.)
  - **`FlexOlmoAttention`** [compute]: `L2/attention.py` (GQA with QK-RMSNorm; matches LlamaAttention(qk_norm=True).)
  - **`FlexOlmoTopKRouter`** [wiring]: Top-k router wiring; standard MoE routing.
  - **`FlexOlmoSparseMoeBlock`** [compute]: `L1/moe_grouped_gemm.py`, `L1/topk_softmax.py` (Standard fused-MoE sparse block (no shared expert); maps to grouped-gemm + topk.)
  - **`FlexOlmoDecoderLayer`** [wiring]: Olmo2-style post-norm decoder block wiring.
  - **`FlexOlmoModel`** [wiring]: Top model wiring.
  - **`FlexOlmoForCausalLM`** [wiring]: LM head wiring.

## florence2
- **src**: modular_florence2.py
- **status**: partial
- **rationale**: Florence-2 = DaViT vision backbone (channel attention + window spatial attention with depth-wise Conv2d positional encoding) + BART-style seq2seq language model. Vision backbone is bespoke DaViT not present in kb-nano; BART seq2seq is also not implemented as an L4.
- **classes**:
  - **`Florence2VisionChannelAttention`** [compute]: no kb-nano kernel — Florence-2 = DaViT vision backbone (channel attention + window spatial attention with depth-wise Conv2d positional encoding) + BART-style seq2seq language model. Vision backbone is bespoke DaViT not p
  - **`Florence2VisionDropPath`** [wiring]: Stochastic depth; not a compute op.
  - **`Florence2VisionLearnedAbsolutePositionEmbedding2D`** [wiring]: Learned 2D position embedding via 2 nn.Embedding tables.
  - **`Florence2VisionPositionalEmbeddingCosine1D`** [wiring]: Sinusoidal 1D positional encoding.
  - **`Florence2VisionMLP`** [wiring]: Vision MLP fc1 + GELU + fc2; close to vision_mlp but no exact match.
  - **`Florence2VisionConvEmbed`** [compute]: `L1/conv2d.py` (Conv2d patch embed + LayerNorm; partial.)
  - **`Florence2VisionChannelBlock`** [wiring]: Wires channel attention + Conv2dPositionEnc + MLP.
  - **`Florence2VisionWindowAttention`** [wiring]: Non-overlapping window attention with explicit pad+reshape; partially related to swinv2_window_attention but window mechanics differ.
  - **`Florence2VisionSpatialBlock`** [wiring]: Wires window attention + ConvPosEnc + MLP.
  - **`Florence2VisionBlock`** [wiring]: Stacks SpatialBlock + ChannelBlock; DaViT block.
  - **`Florence2VisionBackbone`** [wiring]: DaViT backbone with multi-stage downsampling.
  - **`Florence2MultiModalProjector`** [wiring]: Linear projector + image-token features.
  - **`Florence2Model`** [wiring]: Wires DaViT vision + BART seq2seq.
  - **`Florence2ForConditionalGeneration`** [wiring]: Top-level conditional generation.

## fnet
- **src**: modeling_fnet.py
- **status**: partial
- **partial_reason**: FNetBasicFourierTransform uses torch.fft.fftn (or scipy linalg.dft fallback). torch.fft.fftn exists in PyTorch, so a partial port is possible by calling it directly, but kb-nano has no L1 FFT primitive.
- **rationale**: FNet replaces self-attention with a parameter-free 2D Fourier transform via torch.fft.fftn. The rest (BertIntermediate/Output, embeddings) is standard BERT and composable, but kb-nano has no FFT op.
- **classes**:
  - **`FNetLayer`** [compute]: FNetBasicFourierTransform uses torch.fft.fftn (or scipy linalg.dft fallback). torch.fft.fftn exists in PyTorch, so a partial port is possible by calling it directly, but kb-nano has no L1 FFT primitiv
  - **`FNetEmbeddings`** [compute]: `L2/encoder_embeddings.py` (Standard BERT embeddings (token + position + token-type + LN).)
  - **`FNetBasicFourierTransform`** [wiring]: Calls torch.fft.fftn(dim=(1,2)).real; no kb-nano L1 op.
  - **`FNetBasicOutput`** [wiring]: LayerNorm + residual; composable.
  - **`FNetFourierTransform`** [wiring]: Wires basic transform + output.
  - **`FNetIntermediate`** [compute]: `L2/encoder_mlp.py` (Linear + GELU; matches EncoderIntermediate.)
  - **`FNetOutput`** [compute]: `L2/encoder_mlp.py` (Linear + dropout + LN + residual; matches EncoderOutput.)
  - **`FNetEncoder`** [wiring]: Stack of FNetLayer.
  - **`FNetPooler`** [wiring]: Linear + tanh pooler.
  - **`FNetPredictionHeadTransform`** [wiring]: Linear + GELU + LN; composable.
  - **`FNetLMPredictionHead`** [wiring]: MLM head wiring.
  - **`FNetOnlyMLMHead`** [wiring]: MLM head wiring.
  - **`FNetOnlyNSPHead`** [wiring]: NSP head wiring.
  - **`FNetPreTrainingHeads`** [wiring]: Combined MLM + NSP heads.
  - **`FNetModel`** [wiring]: Top model wiring.
  - **`FNetForPreTraining`** [wiring]: Pretraining wiring.
  - **`FNetForMaskedLM`** [wiring]: MLM wiring.
  - **`FNetForNextSentencePrediction`** [wiring]: NSP wiring.
  - **`FNetForSequenceClassification`** [wiring]: Classification wiring.
  - **`FNetForMultipleChoice`** [wiring]: Multiple choice wiring.
  - **`FNetForTokenClassification`** [wiring]: Token classification wiring.
  - **`FNetForQuestionAnswering`** [wiring]: QA wiring.

## focalnet
- **src**: modeling_focalnet.py
- **status**: partial
- **rationale**: FocalNet uses Focal Modulation: a bespoke replacement for self-attention that combines depthwise Conv2d hierarchical context aggregation with gating. No kb-nano kernel implements this pattern, and there is no Focal-Modulation L4.
- **classes**:
  - **`FocalNetModulation`** [compute]: FocalNet uses Focal Modulation: a bespoke replacement for self-attention that combines depthwise Conv2d hierarchical context aggregation with gating. No kb-nano kernel implements this pattern, and the
  - **`FocalNetEmbeddings`** [wiring]: Patch embed + LayerNorm wiring.
  - **`FocalNetPatchEmbeddings`** [compute]: `L1/conv2d.py` (Conv2d patch projection.)
  - **`FocalNetDropPath`** [wiring]: Stochastic depth utility.
  - **`FocalNetMlp`** [wiring]: Linear + GELU + Linear + dropout; close to encoder_mlp but no LayerNorm here.
  - **`FocalNetLayer`** [wiring]: Wires LN + Modulation + LN + MLP + DropPath + residual.
  - **`FocalNetStage`** [wiring]: Stack of layers + downsample.
  - **`FocalNetEncoder`** [wiring]: Stack of stages.
  - **`FocalNetModel`** [wiring]: Top model wiring.
  - **`FocalNetForMaskedImageModeling`** [wiring]: MIM head wiring.
  - **`FocalNetForImageClassification`** [wiring]: Classification head wiring.
  - **`FocalNetBackbone`** [wiring]: Backbone wiring.

## fsmt
- **src**: modeling_fsmt.py
- **status**: partial
- **rationale**: FSMT is a BART-style encoder-decoder seq2seq model with cross-attention. Although kb-nano has whisper_attention.py (3 sibling classes: encoder/decoder/cross), there is no FSMT/BART L4 pipeline and the FSMT Attention class uses (seq, batch, dim) layout with separate q/k/v projs that does not match Whisper's QKVParallelLinear merging.
- **classes**:
  - **`EncoderLayer`** [compute]: no kb-nano kernel — FSMT is a BART-style encoder-decoder seq2seq model with cross-attention. Although kb-nano has whisper_attention.py (3 sibling classes: encoder/decoder/cross), there is no FSMT/BART L4 pipeline and the
  - **`FSMTEncoder`** [wiring]: Sinusoidal-pos encoder stack.
  - **`DecoderLayer`** [wiring]: Wires self-attn + cross-attn + FFN.
  - **`FSMTDecoder`** [wiring]: Decoder stack.
  - **`Attention`** [wiring]: BART-style attention with separate q/k/v/o linears in (T,B,D) layout, supporting both self and cross attention with EncoderDecoderCache. No exact kb-nano wrapper for this layout.
  - **`FSMTModel`** [wiring]: Top encoder-decoder model wiring.
  - **`FSMTForConditionalGeneration`** [wiring]: Seq2seq head wiring.
  - **`SinusoidalPositionalEmbedding`** [compute]: `L1/sinusoidal_embed.py` (Sinusoidal positional embedding; kb-nano has SinusoidalEmbed.)

## funnel
- **src**: modeling_funnel.py
- **status**: partial
- **rationale**: Funnel Transformer uses pooled-query relative-position multi-head attention with per-block q/k stride pooling and a custom learned positional structure. The relative-attention structure (FunnelAttentionStructure with phi/pi/psi/omega bias terms) is bespoke and has no kb-nano kernel.
- **classes**:
  - **`FunnelRelMultiheadAttention`** [compute]: no kb-nano kernel — Funnel Transformer uses pooled-query relative-position multi-head attention with per-block q/k stride pooling and a custom learned positional structure. The relative-attention structure (FunnelAttenti
  - **`FunnelEmbeddings`** [wiring]: Embedding + LN + dropout wiring.
  - **`FunnelAttentionStructure`** [wiring]: Builds relative position embeddings + token-type biases for funnel attention; bespoke.
  - **`FunnelPositionwiseFFN`** [wiring]: Linear + GELU + Linear + LN; composable from L1 ops.
  - **`FunnelLayer`** [wiring]: Wires attention + FFN.
  - **`FunnelEncoder`** [wiring]: Multi-block encoder with downsampling; bespoke wiring.
  - **`FunnelDecoder`** [wiring]: Upsampling decoder wiring.
  - **`FunnelDiscriminatorPredictions`** [wiring]: ELECTRA-style discriminator head.
  - **`FunnelClassificationHead`** [wiring]: Classification head wiring.
  - **`FunnelBaseModel`** [wiring]: Encoder-only model wiring.
  - **`FunnelModel`** [wiring]: Encoder-decoder model wiring.
  - **`FunnelForPreTraining`** [wiring]: Pretraining wiring (ELECTRA-style).
  - **`FunnelForMaskedLM`** [wiring]: MLM wiring.
  - **`FunnelForSequenceClassification`** [wiring]: Classification wiring.
  - **`FunnelForMultipleChoice`** [wiring]: Multiple choice wiring.
  - **`FunnelForTokenClassification`** [wiring]: Token classification wiring.
  - **`FunnelForQuestionAnswering`** [wiring]: QA wiring.

## fuyu
- **src**: modeling_fuyu.py
- **status**: partial
- **rationale**: Fuyu wraps a Persimmon (parallel-attention) language model with a single Linear vision_embed_tokens projecting raw image patches to text embedding space. Persimmon is not in kb-nano, and there is no Fuyu/Persimmon L4 pipeline.
- **classes**:
  - **`FuyuModel`** [compute]: no kb-nano kernel — Fuyu wraps a Persimmon (parallel-attention) language model with a single Linear vision_embed_tokens projecting raw image patches to text embedding space. Persimmon is not in kb-nano, and there is no F
  - **`FuyuForCausalLM`** [wiring]: LM head + generation wiring around FuyuModel.

## gemma
- **src**: modular_gemma.py
- **status**: composable
- **rationale**: Gemma 1 = Llama with GemmaRMSNorm (1+weight) and GELU-tanh activation. Attention/MLP map to LlamaAttention/LlamaMLP; norm maps to L1/gemma_rms_norm.py.
- **classes**:
  - **`GemmaTextScaledWordEmbedding`** [compute]: `L1/embedding.py` (nn.Embedding subclass that scales by sqrt(hidden_size); composable from kb-nano embedding.)
  - **`GemmaRMSNorm`** [compute]: `L1/gemma_rms_norm.py` ((x*w).to(orig_dtype) form with (1+weight) scale; matches L1/gemma_rms_norm.py exactly.)
  - **`GemmaMLP`** [compute]: `L2/llama_mlp.py` (Standard SwiGLU gate_up + down with GELU(tanh approx) instead of SiLU; structure matches llama_mlp.)
  - **`GemmaRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (Standard NeoX rotary.)
  - **`GemmaAttention`** [compute]: `L2/attention.py` (LlamaAttention with optional bidirectional flag.)
  - **`GemmaModel`** [wiring]: Top model wiring with scaled embedding.
  - **`GemmaForCausalLM`** [wiring]: LM head wiring.
  - **`GemmaForSequenceClassification`** [wiring]: Classification head wiring.
  - **`GemmaForTokenClassification`** [wiring]: Token classification wiring.

## gemma2
- **src**: modular_gemma2.py
- **status**: composable
- **rationale**: Gemma 2 = Gemma 1 + interleaved sliding/global attention + softcapping (logit + attention) + pre/post-feedforward norms. Soft-capping is a small additive op handled within attention; everything else maps to existing kb-nano kernels.
- **classes**:
  - **`Gemma2RMSNorm`** [compute]: `L1/gemma_rms_norm.py` (Same as GemmaRMSNorm.)
  - **`Gemma2MLP`** [compute]: `L2/llama_mlp.py` (Standard SwiGLU MLP with GELU-tanh.)
  - **`Gemma2RotaryEmbedding`** [compute]: `L1/rotary_emb.py` (Standard NeoX rotary.)
  - **`Gemma2Attention`** [compute]: `L2/attention.py` (GQA Llama attention with sliding_window + attn_logit_softcapping; kb-nano LlamaAttention supports sliding_window. Softcapping is a small additive math step not covered by the standard fast-attn kernel but inferable as a non-fast-path option.)
  - **`Gemma2DecoderLayer`** [wiring]: Pre/post norm wiring around attention and MLP; composes.
  - **`Gemma2Model`** [wiring]: Top model wiring with sliding/global mask interleave.
  - **`Gemma2ForCausalLM`** [wiring]: LM head wiring with final logit softcapping.
  - **`Gemma2ForSequenceClassification`** [wiring]: Classification head wiring.
  - **`Gemma2ForTokenClassification`** [wiring]: Token classification wiring.

## gemma3
- **src**: modular_gemma3.py
- **status**: composable
- **rationale**: Gemma 3 text path = Gemma2 + per-layer-type RoPE (global vs local) + QK-RMSNorm + bidirectional mask option. Vision path uses SigLIP (mapped to siglip_attention/siglip_mlp). All kernels exist in kb-nano.
- **classes**:
  - **`Gemma3TextScaledWordEmbedding`** [compute]: `L1/embedding.py` (Scaled embedding.)
  - **`Gemma3MLP`** [compute]: `L2/llama_mlp.py` (Standard SwiGLU MLP.)
  - **`Gemma3RMSNorm`** [compute]: `L1/gemma_rms_norm.py` (Gemma RMSNorm.)
  - **`Gemma3RotaryEmbedding`** [compute]: `L1/rotary_emb.py` (Per-layer-type rotary frequencies; standard NeoX rotation per type.)
  - **`Gemma3Attention`** [compute]: `L2/attention.py` (Adds QK-RMSNorm (Gemma3RMSNorm on q/k); LlamaAttention with qk_norm=True and sliding_window matches.)
  - **`Gemma3DecoderLayer`** [wiring]: Pre/post norm wiring.
  - **`Gemma3TextModel`** [wiring]: Top text model with per-type RoPE dispatch.
  - **`Gemma3ForCausalLM`** [wiring]: LM head wiring.
  - **`Gemma3MultiModalProjector`** [wiring]: AvgPool2d + RMSNorm + Linear projector vision->text; composable from L1 avg_pool2d/RMSNorm/Linear.
  - **`Gemma3Model`** [wiring]: Multimodal wiring (SigLIP vision tower + projector + Gemma3 text).
  - **`Gemma3ForConditionalGeneration`** [wiring]: Top-level conditional generation wiring.
  - **`Gemma3ForSequenceClassification`** [wiring]: Classification head wiring.
  - **`Gemma3TextForSequenceClassification`** [wiring]: Text-only classification wiring.

## gemma3n
- **src**: modular_gemma3n.py
- **status**: unsupported
- **unsupported_reason**: Gemma3nTextAltUp (parallel prediction routing with cross-correction), Gemma3nTextLaurelBlock (low-rank residual aug), Gemma3nAudioConformerAttention/RelativePositionEmbedding (USM audio attention), and Gemma3nAudioCumulativeGroupNorm have no kb-nano equivalents. Per-layer KV sharing also requires custom cache plumbing not in infra/context.py.
- **rationale**: Gemma 3n adds bespoke text-stack components: AltUp (alternating updates over 4 parallel predictions), Laurel residual blocks, per-layer KV sharing, activation sparsity, plus a complete USM (Universal Speech Model) audio encoder. None of AltUp/Laurel/USM exist in kb-nano.
- **classes**:
  - **`Gemma3nAudioAttention`** [compute]: no kb-nano kernel — Gemma3nTextAltUp (parallel prediction routing with cross-correction), Gemma3nTextLaurelBlock (low-rank residual aug), Gemma3nAudioConformerAttention/RelativePositionEmbedding (USM audio attention), an
  - **`Gemma3nRMSNorm`** [compute]: `L1/gemma_rms_norm.py` (Gemma-style RMSNorm with optional with_scale; close to gemma_rms_norm but variant on scale handling.)
  - **`Gemma3nAudioRelativePositionEmbedding`** [wiring]: Bespoke USM relative position bias.
  - **`Gemma3nAudioCumulativeGroupNorm`** [wiring]: Time-cumulative group norm; bespoke.
  - **`Gemma3nAudioSSCPConvBlock`** [wiring]: Sub-sample conv projection block; uses Conv2d but bespoke wiring.
  - **`Gemma3nAudioSubSampleConvProjection`** [wiring]: Stack of SSCPConvBlock.
  - **`Gemma3nAudioConformerAttention`** [wiring]: Wraps audio attention with pre/post LN; bespoke.
  - **`Gemma3nAudioConformerFeedForward`** [wiring]: Conformer FFN.
  - **`Gemma3nAudioConformerLightConv1d`** [wiring]: Conformer lightweight Conv1d module.
  - **`Gemma3nAudioConformerBlock`** [wiring]: Audio conformer block wiring.
  - **`Gemma3nTextScaledWordEmbedding`** [compute]: `L1/embedding.py` (Scaled embedding.)
  - **`Gemma3nTextLaurelBlock`** [wiring]: Learned Augmented Residual Layer (low-rank residual); bespoke.
  - **`Gemma3nTextMLP`** [compute]: `L2/llama_mlp.py` (SwiGLU MLP with optional activation sparsity (top-k filter).)
  - **`Gemma3nTextAltUp`** [wiring]: Alternating Updates: predicts 4 parallel hidden states with learned routing; bespoke.
  - **`Gemma3nTextAttention`** [wiring]: Llama-style attention + per-layer KV sharing across global/local layer pairs; KV sharing requires bespoke cache plumbing.
  - **`Gemma3nTextDecoderLayer`** [wiring]: Wires AltUp + Laurel + Attention + MLP + per-layer-input fusion.
  - **`Gemma3nAudioEncoder`** [wiring]: USM audio encoder top model wiring.
  - **`Gemma3nRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (Per-layer-type rotary.)
  - **`Gemma3nTextModel`** [wiring]: Top text model with AltUp + per-layer-input embeddings.
  - **`Gemma3nForCausalLM`** [wiring]: LM head wiring.
  - **`Gemma3nMultimodalEmbedder`** [wiring]: Embedder for vision/audio token offsets.
  - **`Gemma3nModel`** [wiring]: Multimodal model wiring (vision + audio + text).
  - **`Gemma3nForConditionalGeneration`** [wiring]: Top-level conditional generation wiring.

## gemma4
- **src**: modular_gemma4.py
- **status**: kb_nano_l4
- **rationale**: kb-nano has L4/gemma4.py implementing the Gemma4 text-only causal LM (the inner LM of Gemma4ForConditionalGeneration). HF gemma4 also wires audio + vision encoders and a Gemma3n-derived multimodal model on top, but the text path matches.
- **classes**:
  - **`Gemma4ClippableLinear`** [wiring]: Clipped Linear used in audio/vision; not in text path.
  - **`Gemma4RMSNorm`** [compute]: `L1/gemma_rms_norm.py` (Gemma-style RMSNorm; matches kb-nano L1.)
  - **`Gemma4AudioRelPositionalEncoding`** [wiring]: Audio path; not in text L4.
  - **`Gemma4AudioAttention`** [wiring]: Chunked audio attention with rel-pos; not in kb-nano L4 (audio path skipped).
  - **`Gemma4AudioSubSampleConvProjectionLayer`** [wiring]: Audio path.
  - **`Gemma4AudioSubSampleConvProjection`** [wiring]: Audio path.
  - **`Gemma4AudioFeedForward`** [wiring]: Audio path.
  - **`Gemma4AudioCausalConv1d`** [wiring]: Causal padding Conv1d.
  - **`Gemma4AudioLightConv1d`** [wiring]: Audio path.
  - **`Gemma4AudioLayer`** [wiring]: Audio path.
  - **`Gemma4VisionPatchEmbedder`** [wiring]: Vision path; not in text L4.
  - **`Gemma4VisionPooler`** [wiring]: Vision path.
  - **`Gemma4VisionMLP`** [wiring]: Vision path.
  - **`Gemma4VisionRotaryEmbedding`** [wiring]: Vision path 2D rotary.
  - **`Gemma4VisionAttention`** [wiring]: Vision path.
  - **`Gemma4VisionEncoderLayer`** [wiring]: Vision path.
  - **`Gemma4VisionEncoder`** [wiring]: Vision path.
  - **`Gemma4TextMLP`** [compute]: `L2/llama_mlp.py`, `L2/gemma4_mlp.py` (Standard SwiGLU MLP; kb-nano L4 uses Gemma4DecoderLayer with gemma4_mlp/gemma4_attention.)
  - **`Gemma4TextRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (Per-layer-type rotary; kb-nano L4 has Gemma4ProportionalRotaryEmbedding for proportional case.)
  - **`Gemma4TextAttention`** [compute]: `L2/gemma4_attention.py` (Llama-family attention with per-layer-type sliding/global head_dim, partial rotary, k-eq-v option; kb-nano has Gemma4Attention.)
  - **`Gemma4TextExperts`** [compute]: `L1/moe_grouped_gemm.py` (MoE experts with grouped-GEMM; kb-nano L2/gemma4_moe.py.)
  - **`Gemma4TextRouter`** [compute]: `L1/gemma4_routing.py` (Gemma4 routing; kb-nano L1/gemma4_routing.py.)
  - **`Gemma4TextDecoderLayer`** [compute]: `L3/gemma4_decoder.py` (Decoder layer wiring with optional MoE; kb-nano L3/gemma4_decoder.py.)
  - **`Gemma4TextScaledWordEmbedding`** [compute]: `L1/embedding.py` (Scaled embedding.)
  - **`Gemma4TextModel`** [wiring]: Top text model wiring; matches L4/gemma4.py.
  - **`Gemma4ForCausalLM`** [wiring]: LM head wiring; matches L4/gemma4.py.
  - **`Gemma4AudioModel`** [wiring]: Audio encoder top model; not in kb-nano L4.
  - **`Gemma4VisionModel`** [wiring]: Vision encoder top model; not in kb-nano L4.
  - **`Gemma4MultimodalEmbedder`** [wiring]: Multimodal embedder; not in kb-nano L4.
  - **`Gemma4Model`** [wiring]: Multimodal model wiring (vision + audio + text); kb-nano L4 only covers text path.
  - **`Gemma4ForConditionalGeneration`** [wiring]: Top-level conditional generation; kb-nano L4 covers the inner text LM only.

## gemma4_assistant
- **src**: modeling_gemma4_assistant.py
- **status**: composable
- **rationale**: Gemma 4 Assistant is a thin wrapper / variant of Gemma4 (presumably an assistant-tuned model that reuses the same architecture). The same L4/gemma4.py text pipeline applies.
- **classes**:

## git
- **src**: modeling_git.py
- **status**: composable
- **rationale**: GIT = BERT-style text decoder (causal masked) + CLIP-style vision encoder. Text attention matches EncoderSelfAttention pattern (with causal mask added externally), vision attention matches CLIP attention, and visual projection is Linear+LN.
- **classes**:
  - **`GitEmbeddings`** [compute]: `L2/encoder_embeddings.py` (Standard BERT embeddings (token + position + LN + dropout); composable via encoder_embeddings.)
  - **`GitSelfAttention`** [compute]: `L2/encoder_attention.py` (BERT-style self-attention with separate q/k/v linears, scaled-dot-product, optional past_key_values; matches EncoderSelfAttention with causal mask supplied via attention_mask.)
  - **`GitSelfOutput`** [wiring]: Linear + dropout + residual; wiring.
  - **`GitAttention`** [wiring]: Wraps SelfAttention + SelfOutput (sibling-wrapper pattern; kernel is on GitSelfAttention).
  - **`GitIntermediate`** [compute]: `L2/encoder_mlp.py` (Linear + GELU; matches EncoderIntermediate.)
  - **`GitOutput`** [compute]: `L2/encoder_mlp.py` (Linear + dropout + LN + residual; matches EncoderOutput.)
  - **`GitLayer`** [wiring]: Pre-LN attn + intermediate + output wiring.
  - **`GitEncoder`** [wiring]: Stack of GitLayer.
  - **`GitVisionEmbeddings`** [compute]: `L1/conv2d.py`, `L1/embedding.py` (CLIP-style patch embed via Conv2d + class token + position embedding.)
  - **`GitVisionMLP`** [compute]: `L2/clip_mlp.py` (CLIP-style fc1 + QuickGELU + fc2; matches clip_mlp.)
  - **`GitVisionAttention`** [compute]: `L2/clip_attention.py` (CLIP-style self-attention; matches clip_attention.)
  - **`GitVisionEncoderLayer`** [wiring]: CLIP block wiring.
  - **`GitVisionEncoder`** [wiring]: Stack of vision encoder layers.
  - **`GitVisionTransformer`** [wiring]: Vision transformer wiring.
  - **`GitVisionModel`** [wiring]: Top vision model wiring.
  - **`GitProjection`** [wiring]: Linear + LN projector vision -> text dim.
  - **`GitModel`** [wiring]: Top model wiring (vision + text decoder).
  - **`GitForCausalLM`** [wiring]: LM head + generation wiring.

## glm
- **src**: modular_glm.py
- **status**: partial
- **rationale**: GLM = Llama attention + Phi3MLP (gate_up_proj + chunk + down_proj, same SwiGLU pattern as llama_mlp) + interleaved (rotate-pairs) RoPE. The interleave-2 rotary is a variant of NeoX rotary; can be done via reshape outside rotary_emb.
- **classes**:
  - **`GlmForCausalLM`** [compute]: no kb-nano kernel — GLM = Llama attention + Phi3MLP (gate_up_proj + chunk + down_proj, same SwiGLU pattern as llama_mlp) + interleaved (rotate-pairs) RoPE. The interleave-2 rotary is a variant of NeoX rotary; can be done
  - **`GlmMLP`** [compute]: `L2/llama_mlp.py` (Phi3MLP is gate_up_proj (merged) + SiLU + down_proj; identical SwiGLU shape to llama_mlp.)
  - **`GlmRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (Standard NeoX rotary with partial_rotary_factor; kb-nano rotary_emb supports partial.)
  - **`GlmAttention`** [compute]: `L2/attention.py` (Llama attention (overrides o_proj to bias=False which is the default). The interleave-2 apply_rotary path used by Glm differs from kb-nano's rotary_emb (which is concat-half NeoX form); equivalent up to a permutation of head_dim.)
  - **`GlmForSequenceClassification`** [wiring]: Classification head wiring.
  - **`GlmForTokenClassification`** [wiring]: Token classification head wiring.

## glm4
- **src**: modular_glm4.py
- **status**: partial
- **rationale**: GLM-4 = Glm attention + Phi3MLP + extra post-self-attn / post-mlp RMSNorms (sandwich norm). All ops map to existing kb-nano kernels.
- **classes**:
  - **`Glm4DecoderLayer`** [compute]: no kb-nano kernel — GLM-4 = Glm attention + Phi3MLP + extra post-self-attn / post-mlp RMSNorms (sandwich norm). All ops map to existing kb-nano kernels.
  - **`Glm4MLP`** [compute]: `L2/llama_mlp.py` (Standard SwiGLU MLP.)
  - **`Glm4Attention`** [compute]: `L2/attention.py` (Same as Glm attention.)
  - **`Glm4ForCausalLM`** [wiring]: LM head wiring.
  - **`Glm4ForSequenceClassification`** [wiring]: Classification head wiring.
  - **`Glm4ForTokenClassification`** [wiring]: Token classification head wiring.
