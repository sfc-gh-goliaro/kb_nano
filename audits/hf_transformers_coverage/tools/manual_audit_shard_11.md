## persimmon
- **src**: modeling_persimmon.py
- **status**: partial
- **partial_reason**: PersimmonAttention uses fused query_key_value Linear with partial RoPE (rotary_ndims = head_dim*partial_rotary_factor) and per-head QK LayerNorm; PersimmonDecoderLayer uses nn.LayerNorm. kb-nano's L2/attention.py assumes RMSNorm and the L1/rotary_emb supports full-head RoPE only (partial rotary requires manual slicing). torch.nn.LayerNorm fallback is used.
- **rationale**: GPT-NeoX-style decoder-LLM with fused QKV, partial RoPE, LayerNorm (not RMSNorm), GELU two-layer MLP (not SwiGLU), and optional QK LayerNorm. kb-nano L2/L3 LLM stack assumes RMSNorm + SwiGLU; no LayerNorm-decoder attention class.
- **classes**:
  - **`PersimmonAttention`** [compute]: no kb-nano kernel — PersimmonAttention uses fused query_key_value Linear with partial RoPE (rotary_ndims = head_dim*partial_rotary_factor) and per-head QK LayerNorm; PersimmonDecoderLayer uses nn.LayerNorm. kb-nano's L2/
  - **`PersimmonRotaryEmbedding`** [wiring]: Partial-rotary RoPE generator; kb-nano L1/rotary_emb.py builds full-head cos/sin only. Would need a partial-RoPE wrapper to match.
  - **`PersimmonMLP`** [compute]: `L1/linear.py`, `L1/gelu.py` (Two-Linear MLP with GELU (dense_h_to_4h -> gelu -> dense_4h_to_h). Same compute as encoder_mlp.py's intermediate+output but without trailing LayerNorm; can compose from L1 Linear+GELU.)
  - **`PersimmonDecoderLayer`** [wiring]: Wiring: input_layernorm (LayerNorm) -> self_attn -> residual -> post_attention_layernorm -> mlp -> dropout -> residual. composes.
  - **`PersimmonModel`** [wiring]: composes (embeddings + decoder stack + final_layernorm).
  - **`PersimmonForCausalLM`** [wiring]: composes (model + lm_head).

## phi
- **src**: modular_phi.py
- **status**: partial
- **partial_reason**: PhiAttention overrides Q/K/V to separate Linears with bias=True and applies partial RoPE; PhiMLP inherits CLIPMLP (fc1+activation_fn+fc2 with QuickGELU). PhiDecoderLayer uses nn.LayerNorm and a parallel attn+mlp residual. The compute relies on torch.nn.LayerNorm, partial-rotary slicing, and configurable activation that maps to kb-nano L1/quickgelu or L1/gelu but the L2 wiring class doesn't exist.
- **rationale**: Phi-1/Phi-2 decoder-LLM derived from Llama. Uses LayerNorm (not RMS), CLIPMLP-style fc1+act+fc2 (not SwiGLU), partial RoPE on first rotary_ndims dims, optional QK LayerNorm, and parallel attn+mlp residual addition. kb-nano L2/attention.py uses RMSNorm + SwiGLU; no LayerNorm-decoder LLM attention.
- **classes**:
  - **`PhiAttention`** [compute]: no kb-nano kernel — PhiAttention overrides Q/K/V to separate Linears with bias=True and applies partial RoPE; PhiMLP inherits CLIPMLP (fc1+activation_fn+fc2 with QuickGELU). PhiDecoderLayer uses nn.LayerNorm and a parall
  - **`PhiRotaryEmbedding`** [wiring]: Partial-rotary copy of Llama RoPE; same gap as PersimmonRotaryEmbedding.
  - **`PhiMLP`** [compute]: `L2/clip_mlp.py` (Inherits CLIPMLP unchanged: fc1 -> activation -> fc2 (QuickGELU by default). Maps to L2/clip_mlp.py:CLIPMLP.)
  - **`PhiDecoderLayer`** [wiring]: Wiring: input_layernorm (LayerNorm) -> self_attn + mlp added in parallel residual. composes.
  - **`PhiModel`** [wiring]: composes.
  - **`PhiForCausalLM`** [wiring]: composes.

## phi3
- **src**: modular_phi3.py
- **status**: partial
- **partial_reason**: Phi3Attention uses one fused qkv_proj of width Hq*D + 2*Hkv*D, then apply_rotary_pos_emb slices q/k to rotary_dim only and concatenates back the unrotated tail. kb-nano L1/rotary_emb operates on the full head; partial-rotary requires manual slicing or a new kernel.
- **rationale**: Phi-3 decoder-LLM derived from Mistral. Fused QKV, partial RoPE (rotary_dim < head_dim path in apply_rotary_pos_emb), SwiGLU MLP via gate_up_proj chunk, RMSNorm, plus residual dropout. SwiGLU and RMSNorm are covered, but partial-rotary path is not in kb-nano L1/rotary_emb.
- **classes**:
  - **`Phi3DecoderLayer`** [compute]: Phi3Attention uses one fused qkv_proj of width Hq*D + 2*Hkv*D, then apply_rotary_pos_emb slices q/k to rotary_dim only and concatenates back the unrotated tail. kb-nano L1/rotary_emb operates on the f
  - **`Phi3MLP`** [compute]: `L2/llama_mlp.py`, `L1/silu_and_mul.py` (gate_up_proj produces 2*intermediate; chunk into gate, up; up * act(gate) -> down_proj. Same compute as L2/llama_mlp.py:LlamaMLP using L1/silu_and_mul.py.)
  - **`Phi3Attention`** [compute]: `L2/attention.py` (Fused QKV + partial RoPE. L2/attention.py:LlamaAttention covers the QKVParallelLinear + RoPE + Attention pattern with bias and qk_norm options, but the partial-rotary slice is not a wrap-in kwarg; would require a partial-rotary path.)
  - **`Phi3ForCausalLM`** [wiring]: composes.

## phi4_multimodal
- **src**: modular_phi4_multimodal.py
- **status**: partial
- **rationale**: Phi-4 multimodal = Phi-3 LLM + SigLIP vision tower + Conformer-style audio encoder with NeMo conv subsampling, depth-wise separable Conv1d, GLU pointwise conv, relative attention bias, and mean-variance norm. Conformer audio block, NeMo conv subsampling, and the GLU pointwise conv have no kb-nano equivalents.
- **classes**:
  - **`Phi4MultimodalAudioAttention`** [compute]: no kb-nano kernel — Phi-4 multimodal = Phi-3 LLM + SigLIP vision tower + Conformer-style audio encoder with NeMo conv subsampling, depth-wise separable Conv1d, GLU pointwise conv, relative attention bias, and mean-varian
  - **`Phi4MultimodalVisionMLP`** [compute]: `L2/siglip_mlp.py` (Inherits SigLIP MLP (fc1 -> GELU -> fc2).)
  - **`Phi4MultimodalVisionAttention`** [compute]: `L2/siglip_attention.py` (SigLIP-style attention (Q/K/V separate) — close analog in L2/siglip_attention.py.)
  - **`Phi4MultimodalImageEmbedding`** [wiring]: Image-feature projector; composes.
  - **`Phi4MultimodalAudioMLP`** [wiring]: FFN inside Conformer; depends on the larger Conformer block which has no kb-nano analog.
  - **`Phi4MultimodalAudioDepthWiseSeparableConv1d`** [wiring]: Depthwise + pointwise Conv1d; kb-nano L1/conv1d exists but Conformer wiring not packaged.
  - **`Phi4MultimodalAudioGluPointWiseConv`** [wiring]: Pointwise Conv1d producing 2*C channels then GLU split — bespoke.
  - **`Phi4MultimodalAudioConvModule`** [wiring]: Conformer ConvModule wrapping LayerNorm + GLU + DW-sep + activation + pointwise.
  - **`Phi4MultimodalAudioConformerEncoderLayer`** [wiring]: Macaron Conformer layer: 0.5*FFN -> attn -> conv -> 0.5*FFN -> norm. No kb-nano L3 analog.
  - **`Phi4MultimodalAudioNemoConvSubsampling`** [wiring]: NeMo-style Conv2d-stacked time subsampling; bespoke.
  - **`Phi4MultimodalAudioRelativeAttentionBias`** [wiring]: Custom relative position bias for audio attention.
  - **`Phi4MultimodalAudioMeanVarianceNormLayer`** [wiring]: Per-feature mean-variance normalization.
  - **`Phi4MultimodalAudioModel`** [wiring]: composes Conformer stack.
  - **`Phi4MultimodalAudioEmbedding`** [wiring]: Audio feature -> hidden projector; composes.
  - **`Phi4MultimodalDecoderLayer`** [wiring]: Inherits Phi3DecoderLayer; same partial-rotary gap.
  - **`Phi4MultimodalFeatureEmbedding`** [wiring]: Combines image + audio + token embeddings; composes.
  - **`Phi4MultimodalModel`** [wiring]: composes.
  - **`Phi4MultimodalForCausalLM`** [wiring]: composes.

## phimoe
- **src**: modular_phimoe.py
- **status**: partial
- **rationale**: Mixtral-derived MoE LLM but with bespoke sparsemixer router (Heun's-third-order gradient estimator wrapping a custom torch.autograd.Function PhimoeMultiplier) and nn.LayerNorm in place of RMSNorm. The sparsemixer top-k has no kb-nano analog; standard L1/topk_softmax / sigmoid_topk does not capture the two-pass Gumbel/jitter logic. LayerNorm on the decoder also breaks the existing LLM stack.
- **classes**:
  - **`PhimoeSparseMoeBlock`** [compute]: no kb-nano kernel — Mixtral-derived MoE LLM but with bespoke sparsemixer router (Heun's-third-order gradient estimator wrapping a custom torch.autograd.Function PhimoeMultiplier) and nn.LayerNorm in place of RMSNorm. The
  - **`PhimoeRotaryEmbedding`** [wiring]: Long/short mscale switch on RoPE; partial gap with kb-nano L1/yarn_rotary_emb.py.
  - **`PhimoeAttention`** [compute]: `L2/attention.py` (Pure inherit of LlamaAttention; would map to L2/attention.py but the surrounding LayerNorm decoder breaks composition.)
  - **`PhimoeExperts`** [compute]: `L2/mixtral_moe.py` (Mixtral-style expert FFN; analog exists in L2/mixtral_moe.py.)
  - **`PhimoeTopKRouter`** [wiring]: Linear router that calls sparsemixer; no kb-nano analog for sparsemixer.
  - **`PhimoeDecoderLayer`** [wiring]: Overrides input_layernorm/post_attention_layernorm to nn.LayerNorm — outside kb-nano LLM stack.
  - **`PhimoeModel`** [wiring]: composes.
  - **`PhimoeForCausalLM`** [wiring]: composes.

## pi0
- **src**: modular_pi0.py
- **status**: kb_nano_l4
- **rationale**: kb-nano L4 pipeline tasks/baseline/L4/pi0.py exists and explicitly targets the HF PI0ForConditionalGeneration (PaliGemma + flow-matching action expert), per its docstring header.
- **classes**:
  - **`PI0TimestepEmbeddings`** [wiring]: Sinusoidal timestep embedding; pi0 L4 builds an equivalent embedder inline.
  - **`PI0ActionTimeEmbedding`** [compute]: `L2/pi0_action_embed.py` (State + action + time projections fused via SiLU-MLP. L2/pi0_action_embed.py covers this.)
  - **`PI0Model`** [wiring]: composes (PaliGemma VLM + DiT action expert).
  - **`PI0ForConditionalGeneration`** [wiring]: composes (flow-matching action prediction loop). L4/pi0.py owns the pipeline.

## pix2struct
- **src**: modeling_pix2struct.py
- **status**: partial
- **partial_reason**: Pix2StructTextLayerCrossAttention wraps a Pix2StructTextAttention configured for cross-attention with relative bias. kb-nano L2/t5_attention.py implements only T5SelfAttention; cross-attention with the T5 relative-position-bias path is not present. The vision side (Pix2StructVisionAttention with no relative bias) and the text side self-attention + gated FFN map to existing kernels, but cross-attention is missing.
- **rationale**: Pix2Struct = ViT-style vision encoder with T5-LayerNorm + T5GatedActDense MLP, plus a T5-style text decoder with self-attention AND cross-attention with relative position bias. kb-nano has T5 self-attention and gated FFN, but no T5 cross-attention class.
- **classes**:
  - **`Pix2StructVisionLayer`** [compute]: Pix2StructTextLayerCrossAttention wraps a Pix2StructTextAttention configured for cross-attention with relative bias. kb-nano L2/t5_attention.py implements only T5SelfAttention; cross-attention with th
  - **`Pix2StructLayerNorm`** [compute]: `L1/t5_layer_norm.py` (RMS-style norm without mean centering — same as T5LayerNorm; matches L1/t5_layer_norm.py.)
  - **`Pix2StructVisionEmbeddings`** [wiring]: Patch projection (Linear) + 2D row/col embedding sum. composes.
  - **`Pix2StructVisionAttention`** [compute]: `L2/encoder_attention.py` (Q/K/V separate, no relative bias, non-causal — encoder-style attention. Maps to L2/encoder_attention.py:EncoderSelfAttention pattern (no bias version).)
  - **`Pix2StructVisionMlp`** [compute]: `L2/t5_dense.py` (T5-style gated FFN (wi_0, wi_1, wo) — same as L2/t5_dense.py:T5DenseGatedActDense.)
  - **`Pix2StructVisionEncoder`** [wiring]: composes.
  - **`Pix2StructVisionModel`** [wiring]: composes.
  - **`Pix2StructTextDenseGatedActDense`** [compute]: `L2/t5_dense.py` (Same as T5GatedActDense.)
  - **`Pix2StructTextLayerFF`** [wiring]: composes (Pix2StructLayerNorm + DenseGatedActDense + dropout).
  - **`Pix2StructTextAttention`** [compute]: `L2/t5_attention.py` (T5-style attention with relative position bias; kb-nano L2/t5_attention.py covers self-attention but not cross-attention configuration.)
  - **`Pix2StructTextLayerSelfAttention`** [wiring]: composes (LayerNorm + T5Attention + dropout + residual).
  - **`Pix2StructTextLayerCrossAttention`** [wiring]: composes around Pix2StructTextAttention configured as cross-attention; missing cross-attention kernel.
  - **`Pix2StructTextBlock`** [wiring]: composes.
  - **`Pix2StructTextModel`** [wiring]: composes.
  - **`Pix2StructForConditionalGeneration`** [wiring]: composes (vision + text model + lm_head).

## pixio
- **src**: modular_pixio.py
- **status**: composable
- **rationale**: ViT-derived vision encoder/backbone (DINOv2 lineage). Standard Q/K/V separate Linears with optional bias, LayerNorm, two-Linear MLP with GELU, plus DropPath. Maps to existing kb-nano encoder kernels.
- **classes**:
  - **`PixioPatchEmbeddings`** [compute]: `L2/vision_patch_embed.py` (Conv2d patch projection; same as ViT/DINOv2 patch embed.)
  - **`PixioEmbeddings`** [wiring]: composes (token + register + interpolated pos embedding).
  - **`PixioSelfAttention`** [compute]: `L2/encoder_attention.py` (Q/K/V separate Linears with optional qkv_bias; eager_attention_forward with SDPA. Same shape as L2/encoder_attention.py:EncoderSelfAttention.)
  - **`PixioAttention`** [wiring]: Wiring: PixioSelfAttention + PixioSelfOutput. composes.
  - **`PixioDropPath`** [compute]: `L1/dropout.py` (Stochastic depth — eval-time identity; uses L1/dropout for training-time variant if needed.)
  - **`PixioMLP`** [compute]: `L2/encoder_mlp.py` (Two-Linear + GELU FFN — same shape as L2/encoder_mlp.py.)
  - **`PixioLayer`** [wiring]: composes (norm1 + attention + drop_path + residual + norm2 + mlp + drop_path + residual).
  - **`PixioEncoder`** [wiring]: composes.
  - **`PixioModel`** [wiring]: composes.
  - **`PixioBackbone`** [wiring]: composes (selects intermediate hidden states for downstream tasks).

## pixtral
- **src**: modeling_pixtral.py
- **status**: partial
- **partial_reason**: PixtralRotaryEmbedding precomputes inv_freq per (h,w) position by interleaving freqs[::2] (h dim) and freqs[1::2] (w dim) and indexing by position_ids. kb-nano L1/vision_rotary_emb.py builds cos_sin from a 1D max_grid_size table indexed via grid_thw_list with spatial_merge_size — Qwen2-VL semantics, not Pixtral's interleaved h/w split. Substituting requires a torch fallback (or a new RoPE table builder).
- **rationale**: Pixtral vision tower uses 2D (h,w) RoPE built on a precomputed (h,w)-indexed inv_freq table — different layout from kb-nano's L1/vision_rotary_emb (which encodes Qwen2-VL grid_thw layouts). The MLP (SwiGLU), RMSNorm, and Q/K/V structure map to existing kernels.
- **classes**:
  - **`PixtralAttentionLayer`** [compute]: PixtralRotaryEmbedding precomputes inv_freq per (h,w) position by interleaving freqs[::2] (h dim) and freqs[1::2] (w dim) and indexing by position_ids. kb-nano L1/vision_rotary_emb.py builds cos_sin f
  - **`PixtralRotaryEmbedding`** [wiring]: Bespoke 2D RoPE; not covered by L1/vision_rotary_emb.
  - **`PixtralAttention`** [compute]: `L2/encoder_attention.py` (Q/K/V separate, no causal mask, then RoPE applied via apply_rotary_pos_emb. Encoder-style; matches L2/encoder_attention.py shape if RoPE applied externally.)
  - **`PixtralMLP`** [compute]: `L2/llama_mlp.py`, `L1/silu_and_mul.py` (Llama-style SwiGLU FFN (gate_proj, up_proj, down_proj). Maps to L2/llama_mlp.py.)
  - **`PixtralRMSNorm`** [compute]: `L1/rms_norm.py` (Standard RMSNorm.)
  - **`PixtralTransformer`** [wiring]: composes.
  - **`PixtralVisionModel`** [wiring]: composes (patch_conv + ln_pre + RoPE + transformer).

## plbart
- **src**: modeling_plbart.py
- **status**: composable
- **rationale**: BART-style encoder-decoder (Programming Language BART). PLBartAttention implements both self- and cross-attention with separate Q/K/V Linears and bias=True; PLBartEncoderLayer is fc1+activation+fc2 (encoder-style). All compute maps to existing whisper/encoder kernels.
- **classes**:
  - **`PLBartScaledWordEmbedding`** [compute]: `L1/embedding.py` (nn.Embedding scaled by sqrt(d_model). Maps to L1/embedding.py + scalar mul.)
  - **`PLBartLearnedPositionalEmbedding`** [compute]: `L1/embedding.py` (Learned positional embedding offset by 2.)
  - **`PLBartAttention`** [compute]: `L2/whisper_attention.py` (BART attention covers self- and cross-attention with bias=True Q/K/V projections; same structure as L2/whisper_attention.py (3 sibling classes: encoder self, decoder self, cross).)
  - **`PLBartEncoderLayer`** [wiring]: composes (self_attn + LN + fc1 + act + fc2 + LN).
  - **`PLBartEncoder`** [wiring]: composes.
  - **`PLBartDecoderLayer`** [wiring]: composes (self_attn + cross_attn + fc1 + act + fc2 + LN).
  - **`PLBartDecoder`** [wiring]: composes.
  - **`PLBartModel`** [wiring]: composes.
  - **`PLBartForConditionalGeneration`** [wiring]: composes.
  - **`PLBartClassificationHead`** [compute]: `L1/linear.py` (Linear + tanh + Linear classification head.)
  - **`PLBartForSequenceClassification`** [wiring]: composes.
  - **`PLBartForCausalLM`** [wiring]: composes.

## poolformer
- **src**: modeling_poolformer.py
- **status**: composable
- **rationale**: MetaFormer baseline: stages of (patch embed via Conv2d) + repeated PoolFormer blocks where the 'token-mixer' is AvgPool2d minus identity, followed by a Conv2d-Conv2d FFN with activation. Norms are GroupNorm(1, C). All ops have direct kb-nano analogs.
- **classes**:
  - **`PoolFormerDropPath`** [compute]: `L1/dropout.py` (Stochastic depth; eval-time identity.)
  - **`PoolFormerEmbeddings`** [compute]: `L1/conv2d.py` (Conv2d projection + optional norm. composes.)
  - **`PoolFormerGroupNorm`** [compute]: `L1/group_norm.py` (GroupNorm(num_groups=1).)
  - **`PoolFormerPooling`** [compute]: `L1/avg_pool2d.py` (AvgPool2d(stride=1, padding=k//2, count_include_pad=False) - identity. L1/avg_pool2d supports count_include_pad.)
  - **`PoolFormerOutput`** [compute]: `L1/conv2d.py` (1x1 Conv2d -> ACT -> 1x1 Conv2d FFN.)
  - **`PoolFormerLayer`** [wiring]: composes (norm -> pool -> drop_path -> residual; norm -> ffn -> drop_path -> residual; with optional layer_scale).
  - **`PoolFormerEncoder`** [wiring]: composes.
  - **`PoolFormerModel`** [wiring]: composes.
  - **`PoolFormerFinalPooler`** [compute]: `L1/linear.py` (Linear + tanh.)
  - **`PoolFormerForImageClassification`** [wiring]: composes.

## pop2piano
- **src**: modeling_pop2piano.py
- **status**: partial
- **partial_reason**: Pop2PianoLayerCrossAttention wraps Pop2PianoAttention configured for cross-attention. kb-nano L2/t5_attention.py implements T5SelfAttention only; cross-attention path (separate K/V from encoder hidden states with relative bias) is not implemented. Self-attention, gated/non-gated FFN, and T5 LayerNorm map to L2/t5_attention.py / L2/t5_dense.py / L1/t5_layer_norm.py.
- **rationale**: T5-derived encoder-decoder for music transcription. Uses T5LayerNorm, T5DenseGatedActDense (or T5DenseActDense), T5-style attention with relative bias, AND cross-attention. kb-nano L2/t5_attention covers self-attention only; cross-attention with relative bias is missing.
- **classes**:
  - **`Pop2PianoLayerSelfAttention`** [compute]: Pop2PianoLayerCrossAttention wraps Pop2PianoAttention configured for cross-attention. kb-nano L2/t5_attention.py implements T5SelfAttention only; cross-attention path (separate K/V from encoder hidden
  - **`Pop2PianoLayerNorm`** [compute]: `L1/t5_layer_norm.py` (Same as T5LayerNorm.)
  - **`Pop2PianoDenseActDense`** [compute]: `L2/t5_dense.py` (Same as T5DenseActDense.)
  - **`Pop2PianoDenseGatedActDense`** [compute]: `L2/t5_dense.py` (Same as T5DenseGatedActDense.)
  - **`Pop2PianoLayerFF`** [wiring]: composes.
  - **`Pop2PianoAttention`** [compute]: `L2/t5_attention.py` (T5-style attention; only self-attention is in kb-nano.)
  - **`Pop2PianoLayerCrossAttention`** [wiring]: composes; missing cross-attention kernel.
  - **`Pop2PianoBlock`** [wiring]: composes.
  - **`Pop2PianoStack`** [wiring]: composes.
  - **`Pop2PianoConcatEmbeddingToMel`** [wiring]: Mel feature concatenation + Linear projection.
  - **`Pop2PianoForConditionalGeneration`** [wiring]: composes.

## pp_doclayout_v2
- **src**: modular_pp_doclayout_v2.py
- **status**: composable
- **rationale**: RT-DETR-derived document-layout detector with an additional reading-order LayoutLMv3 sub-encoder. Uses MultiscaleDeformableAttention (kb-nano L1/L2 rtdetrv2_deformable_attention exists), RTDetr backbone/encoder/decoder (kb-nano L3/rtdetrv2_*), and LayoutLMv3-style self-attention for reading-order classification (encoder-style, composable).
- **classes**:
  - **`PPDocLayoutV2GlobalPointer`** [compute]: `L1/linear.py` (Linear projection + sinusoidal positional encoding for span pointer; composes with L1 ops.)
  - **`PPDocLayoutV2PositionRelationEmbedding`** [compute]: `L1/linear.py`, `L1/embedding.py` (Embedding + Linear for relation features.)
  - **`PPDocLayoutV2ReadingOrderSelfAttention`** [compute]: `L2/encoder_attention.py` (BERT-derived self-attention with optional 1D/2D relative position bias; matches encoder_attention.py shape.)
  - **`PPDocLayoutV2ReadingOrderSelfOutput`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (Linear + dropout + LayerNorm; composes.)
  - **`PPDocLayoutV2ReadingOrderIntermediate`** [compute]: `L2/encoder_mlp.py` (Linear + GELU.)
  - **`PPDocLayoutV2ReadingOrderOutput`** [compute]: `L2/encoder_mlp.py` (Linear + dropout + LayerNorm.)
  - **`PPDocLayoutV2ReadingOrderAttention`** [wiring]: composes (SelfAttention + SelfOutput).
  - **`PPDocLayoutV2ReadingOrderLayer`** [wiring]: composes.
  - **`PPDocLayoutV2ReadingOrderEncoder`** [wiring]: composes.
  - **`PPDocLayoutV2TextEmbeddings`** [compute]: `L2/bert_embeddings.py` (BERT-style word + position + token-type + LayerNorm; matches L2/bert_embeddings.py.)
  - **`MultiScaleDeformableAttention`** [compute]: `L1/rtdetrv2_deformable_attention.py` (Deformable attention reference impl; kb-nano has the CUDA kernel in L1/rtdetrv2_deformable_attention.)
  - **`PPDocLayoutV2MultiscaleDeformableAttention`** [compute]: `L2/rtdetrv2_deformable_attention.py` (RT-DETR-style multiscale deformable attention wrapper.)
  - **`PPDocLayoutV2ReadingOrder`** [wiring]: composes (text emb + reading-order encoder + classifier head).
  - **`PPDocLayoutV2MLPPredictionHead`** [compute]: `L2/rtdetrv2_mlp_head.py` (RT-DETR MLP head.)
  - **`PPDocLayoutV2MLP`** [compute]: `L2/rtdetrv2_mlp_head.py` (FFN inside hybrid encoder; covered.)
  - **`PPDocLayoutV2FrozenBatchNorm2d`** [compute]: `L1/frozen_batch_norm2d.py` (Direct match.)
  - **`PPDocLayoutV2SelfAttention`** [compute]: `L2/rtdetrv2_multihead_attention.py` (RT-DETR encoder self-attention.)
  - **`PPDocLayoutV2ConvEncoder`** [compute]: `L3/rtdetrv2_backbone.py` (RT-DETR ResNet-PResNet backbone.)
  - **`PPDocLayoutV2ConvNormLayer`** [compute]: `L2/rtdetrv2_conv_norm.py` (Conv + BN + activation.)
  - **`PPDocLayoutV2EncoderLayer`** [compute]: `L2/rtdetrv2_encoder_layer.py` (composes.)
  - **`PPDocLayoutV2RepVggBlock`** [compute]: `L2/rtdetrv2_repvgg_block.py` (Direct match.)
  - **`PPDocLayoutV2CSPRepLayer`** [compute]: `L2/rtdetrv2_csp_rep_layer.py` (Direct match.)
  - **`PPDocLayoutV2DecoderLayer`** [compute]: `L3/rtdetrv2_decoder.py` (RT-DETR decoder layer with deformable cross-attention.)
  - **`PPDocLayoutV2SinePositionEmbedding`** [compute]: `L1/sinusoidal_embed.py` (2D sine positional embedding.)
  - **`PPDocLayoutV2AIFILayer`** [compute]: `L3/rtdetrv2_hybrid_encoder.py` (AIFI = single-scale intra-feature attention layer in hybrid encoder.)
  - **`PPDocLayoutV2HybridEncoder`** [compute]: `L3/rtdetrv2_hybrid_encoder.py` (composes.)

## pp_doclayout_v3
- **src**: modular_pp_doclayout_v3.py
- **status**: composable
- **rationale**: Pure RT-DETR variant for document layout: ConvNeXt/PResNet backbone, hybrid encoder (AIFI + CSP-RepLayer), deformable-attention decoder, and a MaskFeatFPN scale-mask head. All compute kernels reuse kb-nano's RTDetr L1/L2/L3 stack.
- **classes**:
  - **`PPDocLayoutV3GlobalPointer`** [compute]: `L1/linear.py` (Same as v2.)
  - **`MultiScaleDeformableAttention`** [compute]: `L1/rtdetrv2_deformable_attention.py` (Reference deformable attention.)
  - **`PPDocLayoutV3MultiscaleDeformableAttention`** [compute]: `L2/rtdetrv2_deformable_attention.py` (RT-DETR multiscale deformable attention.)
  - **`PPDocLayoutV3MLPPredictionHead`** [compute]: `L2/rtdetrv2_mlp_head.py` (RT-DETR MLP head.)
  - **`PPDocLayoutV3ConvLayer`** [compute]: `L1/conv2d.py`, `L1/batch_norm2d.py` (Conv2d + BN + activation.)
  - **`PPDocLayoutV3ScaleHead`** [compute]: `L1/conv2d.py` (Per-scale mask head.)
  - **`PPDocLayoutV3MaskFeatFPN`** [compute]: `L1/conv2d.py`, `L1/interpolate.py` (FPN with bilinear upsample + Conv fusion.)
  - **`PPDocLayoutV3MLP`** [compute]: `L2/rtdetrv2_mlp_head.py` (FFN.)
  - **`PPDocLayoutV3SelfAttention`** [compute]: `L2/rtdetrv2_multihead_attention.py` (RT-DETR encoder self-attention.)
  - **`PPDocLayoutV3ConvNormLayer`** [compute]: `L2/rtdetrv2_conv_norm.py` (Conv + BN + activation.)
  - **`PPDocLayoutV3EncoderLayer`** [compute]: `L2/rtdetrv2_encoder_layer.py` (composes.)
  - **`PPDocLayoutV3RepVggBlock`** [compute]: `L2/rtdetrv2_repvgg_block.py` (Direct match.)
  - **`PPDocLayoutV3CSPRepLayer`** [compute]: `L2/rtdetrv2_csp_rep_layer.py` (Direct match.)
  - **`PPDocLayoutV3SinePositionEmbedding`** [compute]: `L1/sinusoidal_embed.py` (2D sine positional embedding.)
  - **`PPDocLayoutV3AIFILayer`** [compute]: `L3/rtdetrv2_hybrid_encoder.py` (Single-scale intra-feature attention.)
  - **`PPDocLayoutV3HybridEncoder`** [compute]: `L3/rtdetrv2_hybrid_encoder.py` (composes.)
  - **`PPDocLayoutV3DecoderLayer`** [compute]: `L3/rtdetrv2_decoder.py` (RT-DETR decoder layer.)
  - **`PPDocLayoutV3Decoder`** [compute]: `L3/rtdetrv2_decoder.py` (composes.)
  - **`PPDocLayoutV3FrozenBatchNorm2d`** [compute]: `L1/frozen_batch_norm2d.py` (Direct match.)
  - **`PPDocLayoutV3ConvEncoder`** [compute]: `L3/rtdetrv2_backbone.py` (RT-DETR backbone (PResNet/ConvNeXt path).)
  - **`PPDocLayoutV3Model`** [compute]: `L3/rtdetrv2_model.py` (composes.)
  - **`PPDocLayoutV3ForObjectDetection`** [wiring]: composes.

## pp_formulanet
- **src**: modular_pp_formulanet.py
- **status**: partial
- **partial_reason**: PPFormulaNetMultiModalProjector wraps Florence-2 projection (custom). PPFormulaNetVisionAttention inherits SLANeXtVisionAttention (custom encoder attention). PPFormulaNetAttention + PPFormulaNetDecoderLayer follow MBart enc-dec with cross-attention; kb-nano L2/whisper_attention.py covers BART-style cross-attention but the SLANeXt vision attention variant + Florence2 projector are not present. Falls back to torch.nn.MultiheadAttention-style compute.
- **rationale**: FormulaNet = SLANeXt vision encoder + MBart-style text decoder for math-formula recognition. Vision side is custom SLANeXt attention (with bias) + Florence-2-style projector. Text side is MBart decoder with self+cross attention. Cross-attention with relative position bias for text decode is partially missing (kb-nano has whisper-style cross attention but the projector + SLANeXt attention need reverification).
- **classes**:
  - **`PPFormulaNetVisionAttention`** [compute]: no kb-nano kernel — PPFormulaNetMultiModalProjector wraps Florence-2 projection (custom). PPFormulaNetVisionAttention inherits SLANeXtVisionAttention (custom encoder attention). PPFormulaNetAttention + PPFormulaNetDecode
  - **`PPFormulaNetMultiModalProjector`** [wiring]: Florence-2 projector; bespoke.
  - **`PPFormulaNetMLPBlock`** [compute]: `L2/encoder_mlp.py` (fc1 + activation + fc2.)
  - **`PPFormulaNetVisionLayer`** [wiring]: composes.
  - **`PPFormulaNetPatchEmbeddings`** [compute]: `L2/vision_patch_embed.py` (Conv2d patch projection.)
  - **`PPFormulaNetLayerNorm`** [compute]: `L1/layer_norm.py` (LayerNorm with bias-on-zero handling.)
  - **`PPFormulaNetVisionNeck`** [wiring]: composes (interpolate + Linear projection).
  - **`PPFormulaNetVisionModel`** [wiring]: composes.
  - **`PPFormulaNetLearnedPositionalEmbedding`** [compute]: `L1/embedding.py` (Learned positional embedding.)
  - **`PPFormulaNetScaledWordEmbedding`** [compute]: `L1/embedding.py` (Embedding scaled by sqrt(d_model).)
  - **`PPFormulaNetAttention`** [compute]: `L2/whisper_attention.py` (MBart-style attention; kb-nano whisper_attention covers self+cross.)
  - **`PPFormulaNetDecoderLayer`** [wiring]: composes.
  - **`PPFormulaNetTextModel`** [wiring]: composes.
  - **`PPFormulaNetModel`** [wiring]: composes (vision + projector + text decoder).
  - **`PPFormulaNetForConditionalGeneration`** [wiring]: composes.

## pp_lcnet
- **src**: modeling_pp_lcnet.py
- **status**: composable
- **rationale**: Lightweight CV backbone. Stack of depthwise-separable Conv layers (3x3 DW + 1x1 PW + BN + HardSwish) with optional SqueezeExcitation. All ops have direct kb-nano analogs.
- **classes**:
  - **`PPLCNetConvLayer`** [compute]: `L1/conv2d.py`, `L1/batch_norm2d.py`, `L1/hardswish.py` (Conv2d + BN + HardSwish.)
  - **`PPLCNetDepthwiseSeparableConvLayer`** [compute]: `L1/conv2d.py`, `L1/batch_norm2d.py` (DW Conv2d (groups=in_ch) + 1x1 Conv + optional SE.)
  - **`PPLCNetSqueezeExcitationModule`** [compute]: `L1/global_avg_pool2d.py`, `L1/conv2d.py`, `L1/sigmoid.py` (AdaptiveAvgPool + 1x1 Conv + ReLU + 1x1 Conv + Hardsigmoid.)
  - **`PPLCNetBlock`** [wiring]: composes.
  - **`PPLCNetEncoder`** [wiring]: composes.
  - **`PPLCNetBackbone`** [wiring]: composes.
  - **`PPLCNetForImageClassification`** [wiring]: composes.

## pp_lcnet_v3
- **src**: modular_pp_lcnet_v3.py
- **status**: partial
- **partial_reason**: PPLCNetV3LearnableAffineBlock applies learnable scale * x + learnable bias as a separate parameter pair. PPLCNetV3LearnableRepLayer reparameterizes a stack of (DWConv + LearnableAffineBlock) into a single conv at inference (RepVGG-style) with affine. Standard L1/conv2d + L1/linear cover the underlying conv but not the runtime fold-in pattern; falls back to torch.nn computations.
- **rationale**: PPLCNet variant adding LearnableAffineBlock (gamma * x + beta scaling), LearnableRepLayer (conv with reparameterizable affine scaling per stage), and LearnableActivation. The learnable affine + reparameterized conv path is bespoke; no kb-nano analog.
- **classes**:
  - **`PPLCNetV3LearnableAffineBlock`** [compute]: no kb-nano kernel — PPLCNetV3LearnableAffineBlock applies learnable scale * x + learnable bias as a separate parameter pair. PPLCNetV3LearnableRepLayer reparameterizes a stack of (DWConv + LearnableAffineBlock) into a si
  - **`PPLCNetV3ConvLayer`** [compute]: `L1/conv2d.py`, `L1/batch_norm2d.py`, `L1/hardswish.py` (Conv2d + BN + HardSwish.)
  - **`PPLCNetV3ActLearnableAffineBlock`** [wiring]: Activation + LearnableAffineBlock; depends on missing primitive.
  - **`PPLCNetV3LearnableRepLayer`** [wiring]: Reparameterized DW conv with multiple branches + learnable affine; no kb-nano analog.
  - **`PPLCNetV3SqueezeExcitationModule`** [compute]: `L1/global_avg_pool2d.py`, `L1/conv2d.py` (Same as PPLCNet SE.)
  - **`PPLCNetV3DepthwiseSeparableConvLayer`** [compute]: `L1/conv2d.py` (Depthwise + pointwise conv; underlying ops covered.)
  - **`PPLCNetV3Block`** [wiring]: composes.
  - **`PPLCNetV3Backbone`** [wiring]: composes.
  - **`PPLCNetV3Encoder`** [wiring]: composes.

## pp_ocrv5_mobile_det
- **src**: modular_pp_ocrv5_mobile_det.py
- **status**: composable
- **rationale**: Mobile text-detection model with a backbone + Neck (FPN with SqueezeExcitation residual blocks) + Conv-BN segmentation head. All ops are Conv2d / BN / HardSwish / Hardsigmoid / SE — direct kb-nano matches.
- **classes**:
  - **`PPOCRV5MobileDetSqueezeExcitationModule`** [compute]: `L1/global_avg_pool2d.py`, `L1/conv2d.py`, `L1/sigmoid.py` (Standard SE.)
  - **`PPOCRV5MobileDetResidualSqueezeExcitationLayer`** [compute]: `L1/conv2d.py` (Residual SE block.)
  - **`PPOCRV5MobileDetNeck`** [compute]: `L1/conv2d.py`, `L1/interpolate.py` (FPN with bilinear upsample + Conv fusion.)
  - **`PPOCRV5MobileDetConvBatchnormLayer`** [compute]: `L1/conv2d.py`, `L1/batch_norm2d.py` (Conv + BN + activation.)
  - **`PPOCRV5MobileDetHead`** [compute]: `L1/conv2d.py`, `L1/conv_transpose2d.py` (Segmentation head with deconv.)
  - **`PPOCRV5MobileDetModel`** [wiring]: composes.
  - **`PPOCRV5MobileDetForObjectDetection`** [wiring]: composes.

## pp_ocrv5_mobile_rec
- **src**: modular_pp_ocrv5_mobile_rec.py
- **status**: composable
- **rationale**: Mobile text-recognition with SVTR-style attention encoder + GroupNorm-Conv backbone. SVTR attention is a CLIP-style encoder layer (own attention + MLP) — composable from L2/encoder_attention + L2/clip_mlp shapes.
- **classes**:
  - **`PPOCRV5MobileRecAttention`** [compute]: `L2/encoder_attention.py` (Q/K/V separate (Blip2-derived), encoder-style.)
  - **`PPOCRV5MobileRecMLP`** [compute]: `L2/encoder_mlp.py` (fc1 + activation + fc2.)
  - **`PPOCRV5MobileRecBlock`** [wiring]: composes (CLIPEncoderLayer-style).
  - **`PPOCRV5MobileRecConvLayer`** [compute]: `L1/conv2d.py`, `L1/batch_norm2d.py` (Conv + BN + activation.)
  - **`PPOCRV5MobileRecEncoderWithSVTR`** [wiring]: composes (Conv stem + SVTR transformer blocks).
  - **`PPOCRV5MobileRecModel`** [wiring]: composes.
  - **`PPOCRV5MobileRecHead`** [compute]: `L1/linear.py` (CTC classifier head.)
  - **`PPOCRV5MobileRecForTextRecognition`** [wiring]: composes.

## pp_ocrv5_server_det
- **src**: modular_pp_ocrv5_server_det.py
- **status**: composable
- **rationale**: Server text-detection: PResNet/HGNet backbone + IntraclassBlock (Conv stack) + Conv-BN neck + segmentation head. Plain Conv2d/BN/activation primitives.
- **classes**:
  - **`PPOCRV5ServerDetIntraclassBlock`** [compute]: `L1/conv2d.py` (Stacked Conv2d blocks.)
  - **`PPOCRV5ServerDetNeck`** [compute]: `L1/conv2d.py`, `L1/interpolate.py` (FPN-style neck.)
  - **`PPOCRV5ServerDetConvBatchnormLayer`** [compute]: `L1/conv2d.py`, `L1/batch_norm2d.py` (Conv + BN + activation.)
  - **`PPOCRV5ServerDetSegmentationHead`** [compute]: `L1/conv2d.py`, `L1/conv_transpose2d.py` (Segmentation head.)
  - **`PPOCRV5ServerDetLocalModule`** [compute]: `L1/conv2d.py` (Local feature module.)
  - **`PPOCRV5ServerDetHead`** [wiring]: composes.
  - **`PPOCRV5ServerDetModel`** [wiring]: composes.
  - **`PPOCRV5ServerDetForObjectDetection`** [wiring]: composes.

## pp_ocrv5_server_rec
- **src**: modular_pp_ocrv5_server_rec.py
- **status**: composable
- **rationale**: Server text-recognition with SVTR-style transformer encoder (CLIPEncoderLayer-derived block + Blip2-derived attention) over a ResNet stem. Encoder-style attention + FocalNet MLP variant; composable from kb-nano L2/encoder_attention + L2/encoder_mlp + Conv2d/BN.
- **classes**:
  - **`PPOCRV5ServerRecBlock`** [wiring]: CLIPEncoderLayer-style block; composes.
  - **`PPOCRV5ServerRecAttention`** [compute]: `L2/encoder_attention.py` (Blip2-derived encoder attention (Q/K/V separate).)
  - **`PPOCRV5ServerRecConvLayer`** [compute]: `L1/conv2d.py`, `L1/batch_norm2d.py` (Conv + BN + activation.)
  - **`PPOCRV5ServerRecHead`** [compute]: `L1/linear.py` (CTC classifier head.)
  - **`PPOCRV5ServerRecMLP`** [compute]: `L2/encoder_mlp.py` (FocalNet MLP (fc1 + activation + fc2).)
  - **`PPOCRV5ServerRecEncoderWithSVTR`** [wiring]: composes.
  - **`PPOCRV5ServerRecModel`** [wiring]: composes.
  - **`PPOCRV5ServerRecForTextRecognition`** [wiring]: composes.

## prompt_depth_anything
- **src**: modeling_prompt_depth_anything.py
- **status**: partial
- **rationale**: Depth-Anything depth estimator with prompt-depth fusion. Backbone is loaded externally via load_backbone (e.g. DPT/DINOv2). Compute classes are pure Conv2d + ReLU + bilinear upsample + residual fusion; no novel ops.
- **classes**:
  - **`PromptDepthAnythingFeatureFusionStage`** [compute]: Depth-Anything depth estimator with prompt-depth fusion. Backbone is loaded externally via load_backbone (e.g. DPT/DINOv2). Compute classes are pure Conv2d + ReLU + bilinear upsample + residual fusion
  - **`PromptDepthAnythingLayer`** [compute]: `L1/conv2d.py`, `L1/relu.py` (3 stacked Conv2d + ReLU.)
  - **`PromptDepthAnythingPreActResidualLayer`** [compute]: `L1/conv2d.py`, `L1/relu.py` (ReLU + Conv -> ReLU + Conv with residual.)
  - **`PromptDepthAnythingFeatureFusionLayer`** [compute]: `L1/conv2d.py`, `L1/interpolate.py` (Residual fusion + bilinear upsample + Conv.)
  - **`PromptDepthAnythingDepthEstimationHead`** [compute]: `L1/conv2d.py`, `L1/relu.py`, `L1/interpolate.py` (Conv -> upsample -> Conv -> ReLU -> Conv head.)
  - **`PromptDepthAnythingReassembleLayer`** [compute]: `L1/conv2d.py`, `L1/conv_transpose2d.py` (DPT reassemble (1x1 conv + optional resize via conv_transpose2d).)
  - **`PromptDepthAnythingReassembleStage`** [wiring]: composes.
  - **`PromptDepthAnythingNeck`** [wiring]: composes.
  - **`PromptDepthAnythingForDepthEstimation`** [wiring]: composes (load_backbone + neck + head).

## prophetnet
- **src**: modeling_prophetnet.py
- **status**: partial
- **rationale**: ProphetNet uses NgramSelfAttention with stream-level n-gram prediction, custom relative-position-bucket bias projected via a learned Linear(hidden -> num_buckets*num_heads), and the self-attention attends to a main stream + n predict streams. No kb-nano analog for the n-gram stream prediction logic or its relative-bias projection.
- **classes**:
  - **`ProphetNetNgramSelfAttention`** [compute]: no kb-nano kernel — ProphetNet uses NgramSelfAttention with stream-level n-gram prediction, custom relative-position-bucket bias projected via a learned Linear(hidden -> num_buckets*num_heads), and the self-attention att
  - **`ProphetNetPositionalEmbeddings`** [compute]: `L1/embedding.py` (Learned positional embeddings.)
  - **`ProphetNetAttention`** [compute]: `L2/whisper_attention.py` (Self/cross-attention with separate Q/K/V; same shape as Whisper attention.)
  - **`ProphetNetFeedForward`** [compute]: `L2/encoder_mlp.py` (fc1 + activation + dropout + fc2.)
  - **`ProphetNetEncoderLayer`** [wiring]: composes.
  - **`ProphetNetDecoderLayer`** [wiring]: composes; depends on missing NgramSelfAttention.
  - **`ProphetNetEncoder`** [wiring]: composes.
  - **`ProphetNetDecoder`** [wiring]: composes.
  - **`ProphetNetModel`** [wiring]: composes.
  - **`ProphetNetForConditionalGeneration`** [wiring]: composes.
  - **`ProphetNetForCausalLM`** [wiring]: composes.
  - **`ProphetNetDecoderWrapper`** [wiring]: composes.

## pvt
- **src**: modeling_pvt.py
- **status**: composable
- **rationale**: Pyramid Vision Transformer with EfficientSelfAttention: Q comes from full-res hidden, K/V come from a Conv2d-spatially-reduced (sequences_reduction_ratio) hidden + LayerNorm. All ops (Linear, Conv2d, LayerNorm, softmax, matmul) have kb-nano analogs.
- **classes**:
  - **`PvtDropPath`** [compute]: `L1/dropout.py` (Stochastic depth.)
  - **`PvtPatchEmbeddings`** [compute]: `L1/conv2d.py`, `L1/layer_norm.py` (Conv2d patch projection + position embedding.)
  - **`PvtSelfOutput`** [compute]: `L1/linear.py` (Linear + dropout.)
  - **`PvtEfficientSelfAttention`** [compute]: `L1/linear.py`, `L1/conv2d.py`, `L1/layer_norm.py`, `L1/softmax.py` (Q from hidden, KV from Conv2d-reduced (stride=ratio) hidden + LayerNorm. Composable from primitives but no L2 wrapper.)
  - **`PvtAttention`** [wiring]: composes.
  - **`PvtFFN`** [compute]: `L2/encoder_mlp.py` (fc1 + activation + fc2.)
  - **`PvtLayer`** [wiring]: composes.
  - **`PvtEncoder`** [wiring]: composes.
  - **`PvtModel`** [wiring]: composes.
  - **`PvtForImageClassification`** [wiring]: composes.

## pvt_v2
- **src**: modeling_pvt_v2.py
- **status**: composable
- **rationale**: PvT v2 adds DepthWiseConv inside the FFN (ConvFeedForwardNetwork) and uses overlapping-patch embeddings. Self-attention has the same Conv2d-spatial-reduction pattern as v1 (replaced by AvgPool when ratio == -1 and adaptive pooling). All composable from kb-nano L1.
- **classes**:
  - **`PvtV2DropPath`** [compute]: `L1/dropout.py` (Stochastic depth.)
  - **`PvtV2OverlapPatchEmbeddings`** [compute]: `L1/conv2d.py`, `L1/layer_norm.py` (Overlapping Conv2d patch projection.)
  - **`PvtV2DepthWiseConv`** [compute]: `L1/conv2d.py` (DW Conv2d (groups=in_ch).)
  - **`PvtV2SelfAttention`** [compute]: `L1/linear.py`, `L1/conv2d.py`, `L1/layer_norm.py`, `L1/softmax.py`, `L1/adaptive_avg_pool2d.py` (Q/K/V separate; KV reduced by Conv2d (stride=sr_ratio) or adaptive AvgPool2d when linear_attention enabled.)
  - **`PvtV2ConvFeedForwardNetwork`** [compute]: `L1/linear.py`, `L1/conv2d.py` (Linear + DWConv + activation + Linear.)
  - **`PvtV2BlockLayer`** [wiring]: composes.
  - **`PvtV2EncoderLayer`** [wiring]: composes.
  - **`PvtV2Encoder`** [wiring]: composes.
  - **`PvtV2Model`** [wiring]: composes.
  - **`PvtV2ForImageClassification`** [wiring]: composes.
  - **`PvtV2Backbone`** [wiring]: composes.

## qianfan_ocr
- **src**: modular_qianfan_ocr.py
- **status**: partial
- **partial_reason**: QianfanOCRVisionAttention applies QianfanOCRVisionRMSNorm to Q and K before the per-head reshape (use_qk_norm flag). kb-nano L2/encoder_attention.py does not support QK-norm in the vision branch (LlamaAttention-style qk_norm exists in L2/attention.py but that targets decoder Llama). The text-side InternVL backbone is Llama-family (composable via L2/attention.py + L2/llama_mlp.py).
- **rationale**: Qianfan OCR = InternVL clone (vision encoder + Llama-style text decoder + multimodal projector). Vision attention uses Q/K/V separate Linears with optional QK-RMSNorm and a custom QianfanOCRVisionRMSNorm. The QK-RMSNorm vision attention variant + OCR-specific multimodal projector are not directly in kb-nano L2 (encoder_attention has no QK-norm).
- **classes**:
  - **`QianfanOCRVisionAttention`** [compute]: QianfanOCRVisionAttention applies QianfanOCRVisionRMSNorm to Q and K before the per-head reshape (use_qk_norm flag). kb-nano L2/encoder_attention.py does not support QK-norm in the vision branch (Llam
  - **`QianfanOCRDropPath`** [compute]: `L1/dropout.py` (Stochastic depth.)
  - **`QianfanOCRVisionRMSNorm`** [compute]: `L1/rms_norm.py` (Standard RMSNorm.)
  - **`QianfanOCRVisionMLP`** [compute]: `L2/encoder_mlp.py` (fc1 + activation + fc2.)
  - **`QianfanOCRVisionLayer`** [wiring]: composes.
  - **`QianfanOCRVisionPatchEmbeddings`** [compute]: `L2/vision_patch_embed.py` (Conv2d patch projection.)
  - **`QianfanOCRVisionEmbeddings`** [wiring]: composes.
  - **`QianfanOCRVisionModel`** [wiring]: composes.
  - **`QianfanOCRMultiModalProjector`** [compute]: `L1/linear.py` (Vision -> text hidden projection (Linear stack).)
  - **`QianfanOCRModel`** [wiring]: composes (vision + projector + Llama text).
  - **`QianfanOCRForConditionalGeneration`** [wiring]: composes.

## qwen2
- **src**: modular_qwen2.py
- **status**: composable
- **rationale**: Qwen2 is Llama-family with bias=True on Q/K/V projections, optional sliding-window per layer_type, and SwiGLU MLP. Maps cleanly onto L2/attention.py:LlamaAttention (which has bias and sliding_window kwargs) + L2/llama_mlp.py + L1/rms_norm + L1/rotary_emb. There is no qwen2-specific L4, but every leaf op exists in kb-nano.
- **classes**:
  - **`Qwen2MLP`** [compute]: `L2/llama_mlp.py`, `L1/silu_and_mul.py` (SwiGLU FFN (gate_proj, up_proj, down_proj).)
  - **`Qwen2RotaryEmbedding`** [compute]: `L1/rotary_emb.py` (Standard NeOX-style RoPE; same as L1/rotary_emb.)
  - **`Qwen2Attention`** [compute]: `L2/attention.py` (Llama attention with bias=True on Q/K/V (bias=False on o_proj) and optional sliding window. L2/attention.py:LlamaAttention exposes bias and sliding_window kwargs.)
  - **`Qwen2DecoderLayer`** [wiring]: composes (RMSNorm + attention + RMSNorm + MLP).
  - **`Qwen2Model`** [wiring]: composes.
  - **`Qwen2ForCausalLM`** [wiring]: composes.
  - **`Qwen2ForSequenceClassification`** [wiring]: composes.
  - **`Qwen2ForTokenClassification`** [wiring]: composes.
  - **`Qwen2ForQuestionAnswering`** [wiring]: composes.
