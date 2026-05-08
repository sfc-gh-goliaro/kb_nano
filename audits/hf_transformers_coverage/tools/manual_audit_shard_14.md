## stablelm
- **src**: modeling_stablelm.py
- **status**: partial
- **partial_reason**: StableLmLayerNormPerHead applies a separate nn.LayerNorm to each attention head via nn.ModuleList split/cat; kb-nano has L1/layer_norm.py but no per-head fused variant. Partial-rotary application (rotary on first rotary_ndims, identity on the pass-through tail) is also not exposed by L1/rotary_emb.py — both fall back to torch.nn / native ops.
- **rationale**: Llama-style decoder with partial-rotary RoPE and per-head LayerNorm (StableLmLayerNormPerHead) plus parallel-residual decoder layers; the per-head LayerNorm and split partial-RoPE handling are not implemented as kb-nano kernels.
- **classes**:
  - **`StableLmDecoderLayer`** [compute]: no kb-nano kernel — StableLmLayerNormPerHead applies a separate nn.LayerNorm to each attention head via nn.ModuleList split/cat; kb-nano has L1/layer_norm.py but no per-head fused variant. Partial-rotary application (rot
  - **`StableLmRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (Standard NeoX-style RoPE cos/sin generator; same compute as kb-nano L1/rotary_emb.py.)
  - **`StableLmMLP`** [compute]: `L2/llama_mlp.py`, `L1/silu_and_mul.py` (SwiGLU FFN: down(act(gate(x)) * up(x)) — matches L2/llama_mlp.py with fused L1/silu_and_mul.py (config.hidden_act = 'silu').)
  - **`StableLmLayerNormPerHead`** [wiring]: Per-head nn.LayerNorm via ModuleList(num_heads) + split/cat. No equivalent fused per-head LayerNorm in kb-nano L1.
  - **`StableLmAttention`** [compute]: `L2/attention.py`, `L1/rotary_emb.py` (GQA attention with optional qk_layernorm and partial-rotary (rotary_ndims slice). Bulk QKV+SDPA+O matches L2/attention.py:LlamaAttention but partial-rotary slicing and per-head LN are not in kb-nano.)
  - **`StableLmModel`** [wiring]: Wiring.
  - **`StableLmForCausalLM`** [wiring]: Wiring + LM head.

## starcoder2
- **src**: modular_starcoder2.py
- **status**: composable
- **rationale**: Mistral-derived decoder with sliding-window attention and a non-SwiGLU two-layer MLP (fc1+act+fc2 with bias); maps to LlamaAttention (which already supports sliding window via interface kwarg) and an encoder-style MLP in kb-nano.
- **classes**:
  - **`Starcoder2MLP`** [compute]: `L2/encoder_mlp.py`, `L1/gelu.py` (Two-layer fc1->act->fc2 with bias and residual dropout — same shape/compute as L2/encoder_mlp.py (Intermediate+Output without LN); GPT-Bigcode hidden_act is gelu_pytorch_tanh approximated by L1/gelu.py.)
  - **`Starcoder2Attention`** [compute]: `L2/attention.py`, `L1/rotary_emb.py` (GQA + RoPE + sliding_window via attention_interface kwarg — same compute path as L2/attention.py:LlamaAttention; sliding window honored by underlying SDPA.)
  - **`Starcoder2DecoderLayer`** [wiring]: Wiring.
  - **`Starcoder2Model`** [wiring]: Wiring.
  - **`Starcoder2ForCausalLM`** [wiring]: Wiring + LM head.

## superglue
- **src**: modeling_superglue.py
- **status**: partial
- **partial_reason**: SuperGlueMultiLayerPerceptron uses nn.BatchNorm1d on transposed channels — kb-nano has L1/batch_norm2d.py but no BatchNorm1d. The Sinkhorn matching for the final assignment (in SuperGlueForKeypointMatching) is also a custom optimization-style op not exposed in kb-nano; both fall back to torch.nn / iterative ops.
- **rationale**: Keypoint-matching GNN built on a BERT-style attention block with BatchNorm1d MLPs and a Sinkhorn final assignment; vanilla Conv-free — attention maps to encoder_attention but the BatchNorm1d MLP, the GNN cross/self propagation pattern, and the (out-of-class) Sinkhorn matching have no kb-nano equivalents.
- **classes**:
  - **`SuperGlueKeypointEncoder`** [compute]: no kb-nano kernel — SuperGlueMultiLayerPerceptron uses nn.BatchNorm1d on transposed channels — kb-nano has L1/batch_norm2d.py but no BatchNorm1d. The Sinkhorn matching for the final assignment (in SuperGlueForKeypointMat
  - **`SuperGlueMultiLayerPerceptron`** [compute]: `L1/linear.py`, `L1/relu.py` (Linear -> BatchNorm1d -> ReLU. Linear and ReLU are in kb-nano; nn.BatchNorm1d is not.)
  - **`SuperGlueSelfAttention`** [compute]: `L2/encoder_attention.py` (BERT-style Q/K/V Linear + softmax(QK^T/sqrt(d))V — same compute as EncoderSelfAttention. Supports cross-attention via encoder_hidden_states branch (matches L2 sibling pattern).)
  - **`SuperGlueSelfOutput`** [compute]: `L1/linear.py` (Just nn.Linear — no LayerNorm (unlike Bert).)
  - **`SuperGlueAttention`** [wiring]: Wiring around SuperGlueSelfAttention + SuperGlueSelfOutput (sibling-class wrapper, rule 11).
  - **`SuperGlueAttentionalPropagation`** [wiring]: Wiring: attention + concat + MLP stack.
  - **`SuperGlueAttentionalGNN`** [wiring]: Wiring: alternating self/cross propagation layers.
  - **`SuperGlueFinalProjection`** [compute]: `L1/linear.py` (Single Linear projection.)
  - **`SuperGlueForKeypointMatching`** [wiring]: Wiring + Sinkhorn matching helper functions; Sinkhorn iterative log-domain solver has no kb-nano equivalent.

## superpoint
- **src**: modeling_superpoint.py
- **status**: composable
- **rationale**: Pure-conv keypoint detector (SuperPoint VGG-style encoder + score/descriptor decoders). Conv2d/ReLU/MaxPool2d are all in kb-nano L1, but the decoder uses F.grid_sample for keypoint sampling and depends on F.normalize and ad-hoc NMS, which have no kb-nano kernels.
- **classes**:
  - **`SuperPointConvBlock`** [compute]: `L1/conv2d.py`, `L1/relu.py`, `L1/max_pool2d.py` (Two Conv2d 3x3 + ReLU + optional 2x2 MaxPool — all primitives present.)
  - **`SuperPointEncoder`** [wiring]: Wiring of 4 SuperPointConvBlock stages.
  - **`SuperPointInterestPointDecoder`** [compute]: `L1/conv2d.py`, `L1/relu.py` (Conv2d + ReLU + softmax + NMS (custom python). The NMS / keypoint-extraction post-processing (simple_nms, remove_keypoints_from_borders, top_k_keypoints) is not a kb-nano kernel; only the pre-NMS conv path maps cleanly.)
  - **`SuperPointDescriptorDecoder`** [compute]: `L1/conv2d.py`, `L1/relu.py`, `L1/grid_sample.py`, `L1/l2_norm.py` (Conv2d + ReLU + F.normalize + F.grid_sample bilinear sampling. L1/grid_sample.py and L1/l2_norm.py exist; the keypoint pre/post-scaling logic stays in python.)
  - **`SuperPointForKeypointDetection`** [wiring]: Wiring.

## swiftformer
- **src**: modeling_swiftformer.py
- **status**: partial
- **partial_reason**: SwiftFormerEfficientAdditiveAttention computes a single global query via softmax(Q @ w_g) -> sum, then proj(global * key) — this 'efficient additive attention' pattern is not a kb-nano kernel and falls back to torch ops (matmul + softmax + F.normalize). L1/l2_norm.py covers the normalize step but the surrounding additive-attention algebra has no kb-nano kernel.
- **rationale**: Hybrid CNN+attention image classifier. Patch embed and conv encoder rely on Conv2d+BatchNorm2d (in kb-nano), but SwiftFormerEfficientAdditiveAttention uses F.normalize plus a learnable global-query projection (additive attention rather than QK^T attention) — that custom attention pattern has no kb-nano equivalent.
- **classes**:
  - **`SwiftFormerEncoderBlock`** [compute]: no kb-nano kernel — SwiftFormerEfficientAdditiveAttention computes a single global query via softmax(Q @ w_g) -> sum, then proj(global * key) — this 'efficient additive attention' pattern is not a kb-nano kernel and fall
  - **`SwiftFormerPatchEmbedding`** [compute]: `L1/conv2d.py`, `L1/batch_norm2d.py`, `L1/relu.py` (Two Conv2d (stride 2) + BatchNorm2d + ReLU — all primitives present.)
  - **`SwiftFormerEmbeddings`** [compute]: `L1/conv2d.py`, `L1/batch_norm2d.py` (Down-sampling Conv2d + BatchNorm2d.)
  - **`SwiftFormerConvEncoder`** [compute]: `L1/conv2d.py`, `L1/batch_norm2d.py`, `L1/gelu.py` (Depthwise 3x3 + BN + 1x1 + GELU + 1x1 + layer-scale residual. All primitives present (depthwise via groups=dim arg to L1/conv2d.py).)
  - **`SwiftFormerMlp`** [compute]: `L1/conv2d.py`, `L1/batch_norm2d.py`, `L1/gelu.py` (BN + 1x1 Conv2d + GELU + 1x1 Conv2d — primitives exist.)
  - **`SwiftFormerEfficientAdditiveAttention`** [compute]: `L1/linear.py`, `L1/l2_norm.py`, `L1/softmax.py` (Custom 'efficient additive attention' (Q,K Linear, F.normalize, learnable global w_g, softmax, broadcast). Primitives are present but the pattern as a whole is not assembled as a kb-nano L2 kernel.)
  - **`SwiftFormerLocalRepresentation`** [compute]: `L1/conv2d.py`, `L1/batch_norm2d.py`, `L1/gelu.py` (Same as SwiftFormerConvEncoder without drop_path.)
  - **`SwiftFormerStage`** [wiring]: Wiring.
  - **`SwiftFormerEncoder`** [wiring]: Wiring.
  - **`SwiftFormerForImageClassification`** [wiring]: Wiring + classifier head.

## swin
- **src**: modeling_swin.py
- **status**: partial
- **partial_reason**: SwinSelfAttention uses standard scaled dot-product attention plus a learnable relative_position_bias_table (Embedding-like) indexed by relative_position_index — kb-nano has L2/swinv2_window_attention.py for V2 (cosine + CPB MLP) but no V1 (additive table-bias) variant. The bias is added to attn scores using torch indexing/permute that has no kb-nano L1 wrapper.
- **rationale**: Hierarchical Swin Transformer V1 with windowed attention and an additive learned relative position bias (table indexed by relative_position_index). kb-nano provides L2/swinv2_window_attention.py for the V2 variant (cosine attention + CPB), but the V1 dot-product + learned-bias-table form is not implemented as a kb-nano kernel.
- **classes**:
  - **`SwinSelfAttention`** [compute]: no kb-nano kernel — SwinSelfAttention uses standard scaled dot-product attention plus a learnable relative_position_bias_table (Embedding-like) indexed by relative_position_index — kb-nano has L2/swinv2_window_attention.
  - **`SwinEmbeddings`** [wiring]: Wiring around SwinPatchEmbeddings + nn.LayerNorm + optional learned absolute pos.
  - **`SwinPatchEmbeddings`** [compute]: `L1/conv2d.py` (Conv2d patch projection with optional pad — Conv2d primitive exists.)
  - **`SwinPatchMerging`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (Strided gather + concat + LayerNorm + Linear (4C -> 2C). Note: differs from SwinV2 patch merging (different LN order) — kb-nano L2/swinv2_patch_merging.py is the V2 variant, not V1.)
  - **`SwinSelfOutput`** [compute]: `L1/linear.py` (Single Linear + dropout.)
  - **`SwinAttention`** [wiring]: Sibling-wrapper around SwinSelfAttention + SwinSelfOutput (rule 11).
  - **`SwinIntermediate`** [compute]: `L1/linear.py`, `L1/gelu.py` (Linear + GELU (config.hidden_act = 'gelu').)
  - **`SwinOutput`** [compute]: `L1/linear.py` (Linear + dropout (no LN here; LN happens elsewhere in SwinLayer).)
  - **`SwinLayer`** [wiring]: Wiring with cyclic shift + window partition + reverse — control-flow only.
  - **`SwinStage`** [wiring]: Wiring.
  - **`SwinEncoder`** [wiring]: Wiring.
  - **`SwinModel`** [wiring]: Wiring.
  - **`SwinForImageClassification`** [wiring]: Wiring + classifier head.
  - **`SwinForMaskedImageModeling`** [wiring]: Wiring + decoder head.
  - **`SwinBackbone`** [wiring]: Wiring.

## swin2sr
- **src**: modeling_swin2sr.py
- **status**: composable
- **rationale**: Swin2SR adopts the SwinV2 windowed attention with cosine + continuous-position-bias (CPB) for super-resolution; the per-window attention compute matches kb-nano L2/swinv2_window_attention.py, but the SR-specific Upsample / PixelShuffle / nearest-conv upsampler heads have no kb-nano equivalent (PixelShuffle in particular).
- **classes**:
  - **`Swin2SREmbeddings`** [wiring]: Wiring.
  - **`Swin2SRPatchEmbeddings`** [compute]: `L1/conv2d.py` (Conv2d patch projection.)
  - **`Swin2SRPatchUnEmbeddings`** [wiring]: Reshape only — no kernel.
  - **`Swin2SRPatchMerging`** [compute]: `L2/swinv2_patch_merging.py` (Standard Swin patch merging — matches L2/swinv2_patch_merging.py.)
  - **`Swin2SRSelfAttention`** [compute]: `L2/swinv2_window_attention.py` (SwinV2-style cosine-attention with logit_scale + CPB MLP — same compute as L2/swinv2_window_attention.py.)
  - **`Swin2SRSelfOutput`** [compute]: `L1/linear.py` (Linear projection.)
  - **`Swin2SRAttention`** [wiring]: Sibling-wrapper (rule 11).
  - **`Swin2SRIntermediate`** [compute]: `L1/linear.py`, `L1/gelu.py` (Linear + GELU.)
  - **`Swin2SROutput`** [compute]: `L1/linear.py` (Linear.)
  - **`Swin2SRLayer`** [wiring]: Wiring with shift + window-partition control flow.
  - **`Swin2SRStage`** [wiring]: Wiring.
  - **`Swin2SREncoder`** [wiring]: Wiring.
  - **`Swin2SRModel`** [wiring]: Wiring.
  - **`Upsample`** [wiring]: Conv2d + nn.PixelShuffle stack — PixelShuffle is not in kb-nano.
  - **`UpsampleOneStep`** [wiring]: Conv2d + nn.PixelShuffle.
  - **`PixelShuffleUpsampler`** [wiring]: Conv2d + Upsample (uses PixelShuffle internally).
  - **`NearestConvUpsampler`** [wiring]: F.interpolate(mode='nearest') + Conv2d — nearest-mode interpolate is not a kb-nano kernel.
  - **`PixelShuffleAuxUpsampler`** [wiring]: Same — PixelShuffle dependency.
  - **`Swin2SRForImageSuperResolution`** [wiring]: Wiring.

## swinv2
- **src**: modeling_swinv2.py
- **status**: kb_nano_l4
- **rationale**: L4/swinv2.py is the dedicated kb-nano pipeline for SwinV2 (timm swinv2_large_window12_192 reference); it composes L2/swinv2_window_attention.py (cosine + CPB) and L3/swinv2_block.py / swinv2_stage.py.
- **classes**:
  - **`Swinv2Embeddings`** [wiring]: Wiring; covered by L4/swinv2.py.
  - **`Swinv2PatchEmbeddings`** [compute]: `L1/conv2d.py` (Conv2d patch projection — covered in L4/swinv2.py.)
  - **`Swinv2PatchMerging`** [compute]: `L2/swinv2_patch_merging.py` (Same as kb-nano L2.)
  - **`Swinv2SelfAttention`** [compute]: `L2/swinv2_window_attention.py` (SwinV2 cosine attention + CPB — same compute as kb-nano L2.)
  - **`Swinv2SelfOutput`** [compute]: `L1/linear.py` (Linear projection.)
  - **`Swinv2Attention`** [wiring]: Sibling-wrapper (rule 11).
  - **`Swinv2Intermediate`** [compute]: `L1/linear.py`, `L1/gelu.py` (Linear + GELU.)
  - **`Swinv2Output`** [compute]: `L1/linear.py` (Linear.)
  - **`Swinv2Layer`** [compute]: `L3/swinv2_block.py` (Wiring covered by L3/swinv2_block.py.)
  - **`Swinv2Stage`** [compute]: `L3/swinv2_stage.py` (Stage covered by L3/swinv2_stage.py.)
  - **`Swinv2Encoder`** [wiring]: Wiring.
  - **`Swinv2Model`** [wiring]: Wiring.
  - **`Swinv2ForImageClassification`** [wiring]: Wiring + classifier.
  - **`Swinv2ForMaskedImageModeling`** [wiring]: Wiring + decoder head (PixelShuffle path not in L4/swinv2.py focus).
  - **`Swinv2Backbone`** [wiring]: Wiring.

## switch_transformers
- **src**: modular_switch_transformers.py
- **status**: partial
- **partial_reason**: SwitchTransformersTop1Router applies token-priority cumsum + capacity overflow masking and SwitchTransformersExperts loops over experts with index_add_; the capacity-limited Switch routing is structurally different from standard fused-MoE (top-k softmax) covered by L1/moe_grouped_gemm.py. Decoder cross-attention (T5LayerCrossAttention) also has no kb-nano L2 (kb-nano covers only T5 self-attention encoder side).
- **rationale**: T5 with a top-1 router-based MoE FFN (capacity-limited). Self/cross-attention and dense FFN map to T5 kernels (encoder-only in kb-nano), but the SwitchTransformersTop1Router (capacity-mask + cumsum + scatter expert routing) and SwitchTransformersExperts (loop with index_add_) have no kb-nano fused kernel; standard MoE kernels (L1/moe_grouped_gemm.py) target token-choose-experts, not Switch's expert-choose-tokens with capacity overflow.
- **classes**:
  - **`SwitchTransformersLayerCrossAttention`** [compute]: no kb-nano kernel — SwitchTransformersTop1Router applies token-priority cumsum + capacity overflow masking and SwitchTransformersExperts loops over experts with index_add_; the capacity-limited Switch routing is structur
  - **`SwitchTransformersTop1Router`** [wiring]: Capacity-limited top-1 routing with cumsum + one-hot — no kb-nano kernel.
  - **`SwitchTransformersLayerNorm`** [compute]: `L1/t5_layer_norm.py` (Inherits T5LayerNorm — same as kb-nano L1/t5_layer_norm.py.)
  - **`SwitchTransformersDenseActDense`** [compute]: `L2/t5_dense.py` (Inherits T5DenseActDense — covered by L2/t5_dense.py:T5DenseActDense.)
  - **`SwitchTransformersExperts`** [wiring]: Loop over experts with per-expert SwitchTransformersDenseActDense + index_add_ scatter; not the fused-MoE shape that L1/moe_grouped_gemm.py expects.
  - **`SwitchTransformersSparseMLP`** [wiring]: Wiring around router + experts.
  - **`SwitchTransformersLayerFF`** [wiring]: Wiring (LN + dense_or_sparse_mlp).
  - **`SwitchTransformersAttention`** [compute]: `L2/t5_attention.py` (Inherits T5Attention — kb-nano L2/t5_attention.py covers self-attention; cross-attention path not in kb-nano L2.)
  - **`SwitchTransformersLayerSelfAttention`** [compute]: `L3/t5_block.py` (Inherits T5LayerSelfAttention — covered by kb-nano L3/t5_block.py:T5LayerSelfAttention.)
  - **`SwitchTransformersBlock`** [wiring]: Wiring (encoder uses self+ff; decoder adds cross).
  - **`SwitchTransformersStack`** [wiring]: Wiring.
  - **`SwitchTransformersModel`** [wiring]: Wiring.
  - **`SwitchTransformersForConditionalGeneration`** [wiring]: Wiring + LM head.
  - **`SwitchTransformersEncoderModel`** [wiring]: Wiring.

## t5
- **src**: modeling_t5.py
- **status**: partial
- **partial_reason**: T5LayerCrossAttention (T5Attention with key_value_states from encoder + EncoderDecoderCache) is not implemented in kb-nano (L2/t5_attention.py covers only T5SelfAttention); the relative-bias bucket helper exists encoder-side but the cross-attn path with cross KV cache is missing. Note: kb-nano DOES have an L4 pipeline (L4/t5_encoder.py) but it targets the encoder-only T5XXL used by FLUX, not the full encoder-decoder, so this folder is partial overall.
- **rationale**: T5 encoder-decoder. The encoder side (T5LayerNorm, T5DenseActDense / T5DenseGatedActDense, T5LayerSelfAttention with relative bias) is covered by kb-nano L1/t5_layer_norm.py + L2/t5_dense.py + L2/t5_attention.py + L3/t5_block.py (and the L4/t5_encoder.py pipeline reflects this). The decoder T5LayerCrossAttention has no kb-nano kernel.
- **classes**:
  - **`T5LayerCrossAttention`** [compute]: no kb-nano kernel — T5LayerCrossAttention (T5Attention with key_value_states from encoder + EncoderDecoderCache) is not implemented in kb-nano (L2/t5_attention.py covers only T5SelfAttention); the relative-bias bucket he
  - **`T5LayerNorm`** [compute]: `L1/t5_layer_norm.py` (RMS-style norm with no centering and weight-only scale — matches L1/t5_layer_norm.py.)
  - **`T5DenseActDense`** [compute]: `L2/t5_dense.py` (wi -> act -> wo dense FFN — matches L2/t5_dense.py:T5DenseActDense.)
  - **`T5DenseGatedActDense`** [compute]: `L2/t5_dense.py` (Gated act(wi_0) * wi_1 -> wo — matches L2/t5_dense.py:T5DenseGatedActDense (MergedColumnParallel handles wi_0+wi_1).)
  - **`T5LayerFF`** [compute]: `L3/t5_block.py` (LN + DenseActDense/Gated + residual — wiring covered by L3/t5_block.py:T5LayerFF.)
  - **`T5Attention`** [compute]: `L2/t5_attention.py` (Covers self-attention with relative-bias buckets. Cross-attn case (key_value_states + EncoderDecoderCache) is in this same class but not implemented in kb-nano L2/t5_attention.py.)
  - **`T5LayerSelfAttention`** [compute]: `L3/t5_block.py` (LN + T5Attention + residual — wiring matches L3/t5_block.py:T5LayerSelfAttention.)
  - **`T5Block`** [compute]: `L3/t5_block.py` (Wiring; encoder-side covered by L3/t5_block.py. Decoder-side adds T5LayerCrossAttention which is missing.)
  - **`T5ClassificationHead`** [compute]: `L1/linear.py` (Linear + dropout + Linear.)
  - **`T5Stack`** [wiring]: Wiring (encoder side covered by L4/t5_encoder.py:T5Stack; decoder not).
  - **`T5Model`** [wiring]: Wiring.
  - **`T5ForConditionalGeneration`** [wiring]: Wiring + LM head.
  - **`T5EncoderModel`** [wiring]: Encoder-only model; covered structurally by L4/t5_encoder.py.
  - **`T5ForSequenceClassification`** [wiring]: Wiring.
  - **`T5ForTokenClassification`** [wiring]: Wiring.
  - **`T5ForQuestionAnswering`** [wiring]: Wiring.

## t5gemma
- **src**: modular_t5gemma.py
- **status**: partial
- **partial_reason**: T5GemmaCrossAttention (Gemma2Attention with overridden forward that pulls cross KV from encoder_hidden_states + EncoderDecoderCache.cross_attention_cache + soft-cap + sliding_window=None) has no kb-nano L2 equivalent; L2/attention.py covers self-attention only.
- **rationale**: T5-style encoder-decoder built atop Gemma2 (RMSNorm + GeGLU MLP + Gemma2 RoPE + sliding-window self-attn). Self-attn maps to LlamaAttention-family (Gemma2 variant), but T5GemmaCrossAttention overrides forward with EncoderDecoderCache cross-KV — kb-nano has no cross-attention kernel for this family.
- **classes**:
  - **`T5GemmaCrossAttention`** [compute]: no kb-nano kernel — T5GemmaCrossAttention (Gemma2Attention with overridden forward that pulls cross KV from encoder_hidden_states + EncoderDecoderCache.cross_attention_cache + soft-cap + sliding_window=None) has no kb-na
  - **`T5GemmaRMSNorm`** [compute]: `L1/gemma_rms_norm.py` (Inherits Gemma2RMSNorm — matches kb-nano L1/gemma_rms_norm.py.)
  - **`T5GemmaMLP`** [compute]: `L2/llama_mlp.py`, `L1/gelu_and_mul.py` (GeGLU FFN (act(gate)*up -> down) with dropout — Gemma uses gelu_pytorch_tanh so L1/gelu_and_mul.py('tanh') applies; same shape as L2/llama_mlp.py.)
  - **`T5GemmaRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (Standard NeoX-RoPE generator.)
  - **`T5GemmaSelfAttention`** [compute]: `L2/attention.py`, `L1/rotary_emb.py`, `L1/gemma_rms_norm.py` (Inherits Gemma2Attention with is_causal toggled by is_decoder — same compute as L2/attention.py:LlamaAttention (Gemma2 variant supports sliding window/soft-cap via interface kwargs).)
  - **`T5GemmaEncoderLayer`** [wiring]: Wiring (pre/post norm pattern).
  - **`T5GemmaDecoderLayer`** [wiring]: Wiring (extra cross-attn block).
  - **`T5GemmaClassificationHead`** [compute]: `L1/linear.py` (Linear + dropout.)
  - **`T5GemmaLMHead`** [compute]: `L1/linear.py` (Linear.)
  - **`T5GemmaEncoder`** [wiring]: Wiring.
  - **`T5GemmaDecoder`** [wiring]: Wiring.
  - **`T5GemmaModel`** [wiring]: Wiring.
  - **`T5GemmaEncoderModel`** [wiring]: Wiring.
  - **`T5GemmaForConditionalGeneration`** [wiring]: Wiring.
  - **`T5GemmaForSequenceClassification`** [wiring]: Wiring.
  - **`T5GemmaForTokenClassification`** [wiring]: Wiring.

## t5gemma2
- **src**: modular_t5gemma2.py
- **status**: partial
- **partial_reason**: T5Gemma2MergedAttention concatenates [self_KV | cross_KV] along the seq dim and runs a single fused softmax (with masking); kb-nano L2/attention.py is vanilla self-attention, no merged self+cross variant exists. Q/K RMS-norm is present (matches Gemma3 attention) but the merged-attention concat path is novel.
- **rationale**: Encoder-decoder built atop Gemma3 with a custom T5Gemma2MergedAttention that fuses self+cross attention by concatenating self KV with cross KV in a single softmax. Self-attention maps to LlamaAttention-family kernels but the merged self+cross attention pattern is unique and not in kb-nano.
- **classes**:
  - **`T5Gemma2MergedAttention`** [compute]: T5Gemma2MergedAttention concatenates [self_KV | cross_KV] along the seq dim and runs a single fused softmax (with masking); kb-nano L2/attention.py is vanilla self-attention, no merged self+cross vari
  - **`T5Gemma2RMSNorm`** [compute]: `L1/gemma_rms_norm.py` (Inherits Gemma3RMSNorm — same as kb-nano L1/gemma_rms_norm.py.)
  - **`T5Gemma2MLP`** [compute]: `L2/llama_mlp.py`, `L1/gelu_and_mul.py` (GeGLU FFN with dropout — same shape as L2/llama_mlp.py with L1/gelu_and_mul.py('tanh').)
  - **`T5Gemma2RotaryEmbedding`** [compute]: `L1/rotary_emb.py` (Standard NeoX-RoPE generator.)
  - **`T5Gemma2SelfAttention`** [compute]: `L2/attention.py`, `L1/rotary_emb.py`, `L1/gemma_rms_norm.py` (Inherits Gemma3Attention (with is_causal=False for encoder) — same compute as L2/attention.py:LlamaAttention with q/k norm.)
  - **`T5Gemma2EncoderLayer`** [wiring]: Wiring (pass-through inheritance).
  - **`T5Gemma2DecoderLayer`** [wiring]: Wiring (replaces self-attn with merged attn).
  - **`T5Gemma2LMHead`** [compute]: `L1/linear.py` (Linear.)
  - **`T5Gemma2ClassificationHead`** [compute]: `L1/linear.py` (Linear + dropout.)
  - **`T5Gemma2MultiModalProjector`** [wiring]: Vision-projector wiring (not implemented in kb-nano for Gemma3).
  - **`T5Gemma2TextScaledWordEmbedding`** [compute]: `L1/embedding.py` (Embedding scaled by sqrt(d) — covered by L1/embedding.py + scalar mul.)
  - **`T5Gemma2TextEncoder`** [wiring]: Wiring.
  - **`T5Gemma2Encoder`** [wiring]: Wiring.
  - **`T5Gemma2Decoder`** [wiring]: Wiring.
  - **`T5Gemma2Model`** [wiring]: Wiring.
  - **`T5Gemma2ForConditionalGeneration`** [wiring]: Wiring.
  - **`T5Gemma2ForSequenceClassification`** [wiring]: Wiring.
  - **`T5Gemma2ForTokenClassification`** [wiring]: Wiring.

## table_transformer
- **src**: modeling_table_transformer.py
- **status**: partial
- **partial_reason**: TableTransformerAttention adds object_queries to hidden_states and spatial_position_embeddings to key_value_states *before* the q_proj/k_proj projections (DETR's 'with_pos_embed' pattern), then runs bmm-flattened cross-attn — kb-nano has no DETR-style pos-additive cross-attn kernel. Bipartite-matching loss + Hungarian assignment in TableTransformerForObjectDetection has no kb-nano equivalent.
- **rationale**: DETR-style enc-dec object detector: ResNet-with-FrozenBN backbone + sinusoidal/learned pos embed + cross-attended decoder. Frozen BN exists in kb-nano (L1/frozen_batch_norm2d.py); but the DETR-attention with object_queries / spatial_position_embeddings (Q+pos, K+pos before projection) and the BMM-flattened multi-head reshape have no kb-nano L2 wrapper.
- **classes**:
  - **`TableTransformerConvEncoder`** [compute]: no kb-nano kernel — TableTransformerAttention adds object_queries to hidden_states and spatial_position_embeddings to key_value_states *before* the q_proj/k_proj projections (DETR's 'with_pos_embed' pattern), then runs b
  - **`TableTransformerFrozenBatchNorm2d`** [compute]: `L1/frozen_batch_norm2d.py` (Frozen BN with running stats — matches kb-nano L1/frozen_batch_norm2d.py.)
  - **`TableTransformerConvModel`** [wiring]: Wiring.
  - **`TableTransformerSinePositionEmbedding`** [wiring]: Sinusoidal pos embed (math: sin/cos of position grid) — no kb-nano L1; falls back to torch ops.
  - **`TableTransformerLearnedPositionEmbedding`** [compute]: `L1/embedding.py` (nn.Embedding for row+column.)
  - **`TableTransformerAttention`** [wiring]: DETR-style attention with q/k position-additive projection — no kb-nano kernel.
  - **`TableTransformerEncoderLayer`** [wiring]: Wiring (self-attn + fc1+act+fc2 + LN).
  - **`TableTransformerDecoderLayer`** [wiring]: Wiring (self-attn + cross-attn + fc1+act+fc2 + LN).
  - **`TableTransformerEncoder`** [wiring]: Wiring.
  - **`TableTransformerDecoder`** [wiring]: Wiring.
  - **`TableTransformerModel`** [wiring]: Wiring.
  - **`TableTransformerForObjectDetection`** [wiring]: Wiring + Hungarian-matching loss; loss & matching not in kb-nano.
  - **`TableTransformerMLPPredictionHead`** [compute]: `L1/linear.py`, `L1/relu.py` (Stacked Linear + ReLU.)

## tapas
- **src**: modeling_tapas.py
- **status**: partial
- **partial_reason**: TapasEmbeddings.forward uses IndexMap / ProductIndexMap / reduce_min / gather (segment-aware reductions for cell-relative position) — these operate on per-token table indices and have no kb-nano L1 / L2 wrapper. The downstream cell-selection / aggregation heads in TapasForQuestionAnswering also use custom probabilistic reductions.
- **rationale**: BERT-derived encoder for tabular question answering with extra token-type embeddings (one per table-structure feature) and IndexMap segment-reduce ops. Self-attn / FF map to encoder kernels, but TapasEmbeddings' segmented/reset position-id logic (IndexMap, ProductIndexMap, reduce_min, gather) has no kb-nano equivalent.
- **classes**:
  - **`TapasAttention`** [wiring]: Sibling-wrapper around TapasSelfAttention + TapasSelfOutput. Per guideline 11 the bare *Attention class is wiring; kernel lives on TapasSelfAttention. (The IndexMap / segment-reduce gap lives on TapasEmbeddings.)
  - **`TapasEmbeddings`** [wiring]: Word + position + N token-type embeddings (config.type_vocab_sizes) + LayerNorm + dropout, plus optional reset-position-per-cell branch using IndexMap/reduce_min/gather — no kb-nano kernel for the segment reductions.
  - **`TapasSelfAttention`** [compute]: `L2/encoder_attention.py` (BERT-style Q/K/V Linear + softmax(QK^T/sqrt(d))V — same compute as EncoderSelfAttention.)
  - **`TapasSelfOutput`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (Linear + dropout + LayerNorm(residual).)
  - **`TapasIntermediate`** [compute]: `L2/encoder_mlp.py` (Linear + GELU — same as EncoderIntermediate.)
  - **`TapasOutput`** [compute]: `L2/encoder_mlp.py` (Linear + dropout + LayerNorm — same as EncoderOutput.)
  - **`TapasLayer`** [wiring]: Wiring.
  - **`TapasEncoder`** [wiring]: Wiring.
  - **`TapasPooler`** [compute]: `L1/linear.py`, `L1/tanh.py` (Linear + Tanh on first token.)
  - **`TapasPredictionHeadTransform`** [compute]: `L1/linear.py`, `L1/gelu.py`, `L1/layer_norm.py` (Linear + act + LayerNorm.)
  - **`TapasLMPredictionHead`** [wiring]: Wiring.
  - **`TapasOnlyMLMHead`** [wiring]: Wiring.
  - **`TapasModel`** [wiring]: Wiring.
  - **`TapasForMaskedLM`** [wiring]: Wiring.
  - **`TapasForQuestionAnswering`** [wiring]: Wiring + cell-selection / aggregation heads (custom segmented losses).
  - **`TapasForSequenceClassification`** [wiring]: Wiring.

## textnet
- **src**: modeling_textnet.py
- **status**: composable
- **rationale**: Pure-conv text-detection backbone: TextNetConvLayer (Conv2d+BN+act) and TextNetRepConvLayer (re-parameterizable conv block) compose Conv2d/BatchNorm2d/ReLU primitives that all exist in kb-nano L1.
- **classes**:
  - **`TextNetConvLayer`** [compute]: `L1/conv2d.py`, `L1/batch_norm2d.py`, `L1/relu.py` (Conv2d -> BN -> activation (config.activation_function in {relu, ...}). All primitives present.)
  - **`TextNetRepConvLayer`** [compute]: `L1/conv2d.py`, `L1/batch_norm2d.py`, `L1/relu.py` (Re-parameterizable block: main Conv2d+BN + auxiliary 1xK / Kx1 / 1x1 + identity BN — all Conv2d/BN/ReLU primitives present (the rep-fold is a math identity over conv parameters, no special op).)
  - **`TextNetStage`** [wiring]: Wiring.
  - **`TextNetEncoder`** [wiring]: Wiring.
  - **`TextNetModel`** [wiring]: Wiring + global-avg-pool head (L1/global_avg_pool2d.py).
  - **`TextNetForImageClassification`** [wiring]: Wiring + classifier head.
  - **`TextNetBackbone`** [wiring]: Wiring.

## time_series_transformer
- **src**: modeling_time_series_transformer.py
- **status**: partial
- **partial_reason**: TimeSeriesTransformerAttention is BART-style (bmm-flatten + Q*scaling + LayerNorm-around) — kb-nano L2/whisper_attention.py covers the closest pattern but the per-class structure (no relative bias, no RoPE, plain multi-head with key_value_states for cross-attn) does not 1:1 match a kb-nano L2 file. Distribution-output projection (parameter projections for NegBin/StudentT) has no kb-nano kernel.
- **rationale**: Probabilistic forecasting transformer (BART-style enc-dec) with std/mean scalers and lagged-features input; attention is the BART-style bmm-flattened MHA without RoPE. Encoder/decoder attention has no kb-nano L2 (no whisper/bart-style covers this with cross-attn properly), and the probabilistic distribution head (Negative-Binomial / Student-T parametric output) is not in kb-nano.
- **classes**:
  - **`TimeSeriesTransformerAttention`** [compute]: TimeSeriesTransformerAttention is BART-style (bmm-flatten + Q*scaling + LayerNorm-around) — kb-nano L2/whisper_attention.py covers the closest pattern but the per-class structure (no relative bias, no
  - **`TimeSeriesFeatureEmbedder`** [compute]: `L1/embedding.py` (List of nn.Embedding modules concatenated.)
  - **`TimeSeriesStdScaler`** [wiring]: Compute mean/std with weighted sums + clamp + division — pure tensor ops, no kb-nano kernel needed but no dedicated wrapper either.
  - **`TimeSeriesMeanScaler`** [wiring]: Same — pure tensor ops.
  - **`TimeSeriesNOPScaler`** [wiring]: Identity scaler.
  - **`TimeSeriesSinusoidalPositionalEmbedding`** [compute]: `L1/embedding.py` (Sinusoidal lookup as embedding.)
  - **`TimeSeriesValueEmbedding`** [compute]: `L1/linear.py` (Single Linear projection.)
  - **`TimeSeriesTransformerEncoderLayer`** [wiring]: Wiring (self-attn + fc1+act+fc2 + LN).
  - **`TimeSeriesTransformerDecoderLayer`** [wiring]: Wiring (self+cross+fc).
  - **`TimeSeriesTransformerEncoder`** [wiring]: Wiring.
  - **`TimeSeriesTransformerDecoder`** [wiring]: Wiring.
  - **`TimeSeriesTransformerModel`** [wiring]: Wiring.
  - **`TimeSeriesTransformerForPrediction`** [wiring]: Wiring + parametric distribution output head (Negative-Binomial / Student-T) — no kb-nano kernel.

## timesfm
- **src**: modeling_timesfm.py
- **status**: partial
- **rationale**: TimesFM forecasting decoder uses learnable per-dimension query scaling (softplus(scaling) * 1.4427/sqrt(d_h)) inside attention — a custom op with no PyTorch built-in equivalent; the MLP also embeds its layernorm and a paddings-mask multiplication that does not match kb-nano L2.
- **classes**:
  - **`TimesFmAttention`** [compute]: no kb-nano kernel — TimesFM forecasting decoder uses learnable per-dimension query scaling (softplus(scaling) * 1.4427/sqrt(d_h)) inside attention — a custom op with no PyTorch built-in equivalent; the MLP also embeds it
  - **`TimesFmMLP`** [wiring]: LN + gate_proj(relu) + down_proj + optional paddings mask + residual — non-standard FFN; primitives exist but no fused kb-nano L2.
  - **`TimesFmResidualBlock`** [compute]: `L1/linear.py`, `L1/silu.py` (input_layer + SiLU + output_layer + residual_layer (skip projection).)
  - **`TimesFmRMSNorm`** [compute]: `L1/t5_layer_norm.py` (Author comment: 'equivalent to T5LayerNorm' — RMS-style with no centering, weight-only scale; matches L1/t5_layer_norm.py.)
  - **`TimesFmPositionalEmbedding`** [wiring]: Sinusoidal pos embed with inv_timescales.
  - **`TimesFmDecoderLayer`** [wiring]: Wiring.
  - **`TimesFmModel`** [wiring]: Wiring.
  - **`TimesFmModelForPrediction`** [wiring]: Wiring + quantile / mean output heads.

## timesfm2_5
- **src**: modeling_timesfm2_5.py
- **status**: partial
- **rationale**: TimesFM 2.5 keeps the per-dim learnable softplus query-scaling from v1 plus adds Q/K RMS-norms and standard NeoX RoPE; the per-dim scaling is the same op-level gap and has no kb-nano equivalent.
- **classes**:
  - **`TimesFm2_5Attention`** [compute]: TimesFM 2.5 keeps the per-dim learnable softplus query-scaling from v1 plus adds Q/K RMS-norms and standard NeoX RoPE; the per-dim scaling is the same op-level gap and has no kb-nano equivalent.
  - **`TimesFm2_5MLP`** [compute]: `L2/encoder_mlp.py` (Two-layer fc1+act+fc2 with bias — same shape as L2/encoder_mlp.py (without LN-output).)
  - **`TimesFm2_5ResidualBlock`** [compute]: `L1/linear.py` (Linear + act + Linear + residual_proj.)
  - **`TimesFm2_5RMSNorm`** [compute]: `L1/t5_layer_norm.py` (RMS-style (no centering) — matches L1/t5_layer_norm.py.)
  - **`TimesFm2_5RotaryEmbedding`** [compute]: `L1/rotary_emb.py` (Standard NeoX-style RoPE.)
  - **`TimesFm2_5DecoderLayer`** [wiring]: Wiring (pre+post norms around attn and FFN).
  - **`TimesFm2_5PositionalEmbedding`** [wiring]: Sinusoidal pos embed.
  - **`TimesFm2_5Model`** [wiring]: Wiring.
  - **`TimesFm2_5ModelForPrediction`** [wiring]: Wiring + quantile head.

## timesformer
- **src**: modeling_timesformer.py
- **status**: composable
- **rationale**: ViT-style video transformer with divided space-time attention. The attention block is plain QKV-Linear -> softmax(QK^T/sqrt(d))V (no relative bias) which matches encoder_attention; intermediate / output use ViT-style fc1+GELU+fc2; conv2d patch embed is L1.
- **classes**:
  - **`TimesformerPatchEmbeddings`** [compute]: `L1/conv2d.py` (Conv2d patch projection.)
  - **`TimesformerEmbeddings`** [wiring]: Wiring (cls + patch + spatial pos + temporal pos).
  - **`TimesformerSelfAttention`** [compute]: `L2/encoder_attention.py` (Fused QKV Linear + softmax(QK^T/sqrt(d))V — same compute path as EncoderSelfAttention (qkv merged but math is identical).)
  - **`TimesformerSelfOutput`** [compute]: `L1/linear.py` (Linear + dropout.)
  - **`TimeSformerAttention`** [wiring]: Sibling-wrapper (rule 11).
  - **`TimesformerIntermediate`** [compute]: `L2/vit_encoder_mlp.py` (Linear + GELU + dropout — ViT-style FFN.)
  - **`TimesformerOutput`** [compute]: `L1/linear.py` (Linear + dropout (no LN).)
  - **`TimesformerLayer`** [wiring]: Wiring (divided space-time attention).
  - **`TimesformerEncoder`** [wiring]: Wiring.
  - **`TimesformerModel`** [wiring]: Wiring.
  - **`TimesformerForVideoClassification`** [wiring]: Wiring + classifier.

## timm_backbone
- **src**: modeling_timm_backbone.py
- **status**: unsupported
- **unsupported_reason**: TimmBackbone delegates the entire forward to a timm-loaded model (`requires_backends(self, 'timm')`, `timm.create_model(config.backbone, features_only=True)`). kb-nano does not embed timm; only the L4/swinv2.py / L4/dinov3.py / L4/vjepa2.py target specific timm models with reimplemented kernels.
- **rationale**: TimmBackbone is a thin wrapper around the external timm library (`import timm` + `timm.create_model`). All compute is delegated to timm models; kb-nano has no equivalent runtime for arbitrary timm backbones.
- **classes**:
  - **`TimmBackbone`** [compute]: no kb-nano kernel — TimmBackbone delegates the entire forward to a timm-loaded model (`requires_backends(self, 'timm')`, `timm.create_model(config.backbone, features_only=True)`). kb-nano does not embed timm; only the L4

## timm_wrapper
- **src**: modeling_timm_wrapper.py
- **status**: unsupported
- **unsupported_reason**: TimmWrapperModel / TimmWrapperForImageClassification call into the timm library to instantiate and run the model; kb-nano has no equivalent runtime.
- **rationale**: TimmWrapperModel is a Transformers wrapper around an arbitrary timm.create_model classifier; entire forward is delegated to timm.
- **classes**:
  - **`TimmWrapperModel`** [compute]: no kb-nano kernel — TimmWrapperModel / TimmWrapperForImageClassification call into the timm library to instantiate and run the model; kb-nano has no equivalent runtime.
  - **`TimmWrapperForImageClassification`** [wiring]: External timm wrapper + classification head.

## trocr
- **src**: modeling_trocr.py
- **status**: partial
- **partial_reason**: TrOCRAttention is BART-style (Q*scaling -> bmm-flattened multi-head SDPA) with EncoderDecoderCache for cross-attn — kb-nano L2/whisper_attention.py is the closest sibling but is structured around Whisper's three sibling classes; the bare 'TrOCRAttention' single-class form for self+cross does not match a kb-nano file 1:1, and the cross-KV cache update path differs.
- **rationale**: BART-style decoder for OCR (TrOCRDecoder + TrOCRForCausalLM) with sinusoidal/learned pos embed and BART attention (bmm-flattened MHA, with both self and cross attention via key_value_states). TrOCRAttention's bmm + EncoderDecoderCache cross-attn flow is closest to L2/whisper_attention.py but the exact class structure (decoder-only that takes encoder_hidden_states) is not implemented as a kb-nano L2.
- **classes**:
  - **`TrOCRAttention`** [compute]: TrOCRAttention is BART-style (Q*scaling -> bmm-flattened multi-head SDPA) with EncoderDecoderCache for cross-attn — kb-nano L2/whisper_attention.py is the closest sibling but is structured around Whis
  - **`TrOCRLearnedPositionalEmbedding`** [compute]: `L1/embedding.py` (Learned pos embed with offset=2 (BART convention).)
  - **`TrOCRScaledWordEmbedding`** [compute]: `L1/embedding.py` (Embedding with output scale.)
  - **`TrOCRSinusoidalPositionalEmbedding`** [wiring]: Custom sinusoidal pos table; no kb-nano L1 wrapper.
  - **`TrOCRDecoderLayer`** [wiring]: Wiring (self + optional cross + fc1+act+fc2).
  - **`TrOCRDecoder`** [wiring]: Wiring.
  - **`TrOCRDecoderWrapper`** [wiring]: Wiring.
  - **`TrOCRForCausalLM`** [wiring]: Wiring + LM head.

## tvp
- **src**: modeling_tvp.py
- **status**: partial
- **partial_reason**: TvpVisionModel uses Transformers' load_backbone for an external ResNet (not a kb-nano backbone). TvpFrameDownPadPrompter / TvpFramePadPrompter implement learnable padding around video frames as nn.Parameter padding masks — no kb-nano equivalent. TvpAttention is a 'fused' BERT-attention that includes the post-attention dense+LN inside the same class (unusual) which still maps to L2/encoder_attention.py + L2/encoder_mlp.py-style ops but is structurally combined.
- **rationale**: Video grounding model: vision backbone (ResNet via load_backbone) + text+visual BERT-style encoder + temporal prompt + IoU/L1 grounding head. Encoder attention/MLP map to encoder_attention/encoder_mlp, but the load_backbone-driven ResNet (TvpVisionModel) and the visual prompter padding modules (TvpFrame*Prompter) have no kb-nano equivalents.
- **classes**:
  - **`TvpEncodeLayer`** [compute]: TvpVisionModel uses Transformers' load_backbone for an external ResNet (not a kb-nano backbone). TvpFrameDownPadPrompter / TvpFramePadPrompter implement learnable padding around video frames as nn.Par
  - **`TvpLoss`** [wiring]: IoU/L1 grounding loss; not inference-relevant.
  - **`TvpVisionModel`** [wiring]: Wraps load_backbone(config) — typically a ResNet from transformers. No kb-nano backbone.
  - **`TvpVisualInputEmbedding`** [wiring]: Wiring (3D pos embeds + LN + dropout).
  - **`TvpTextInputEmbeddings`** [compute]: `L2/bert_embeddings.py` (BERT-style word + position + token-type embeddings + LN + dropout — same structure as L2/bert_embeddings.py.)
  - **`TvpAttention`** [compute]: `L2/encoder_attention.py` (BERT-style Q/K/V Linear + SDPA, then dense+LN(residual) inside the same class. The attention math is identical to EncoderSelfAttention; the post-attn dense+LN matches encoder_mlp's output.)
  - **`TvpIntermediate`** [compute]: `L2/encoder_mlp.py` (Linear + act — same as EncoderIntermediate.)
  - **`TvpOutputLayer`** [compute]: `L2/encoder_mlp.py` (Linear + dropout + LN(residual) — same as EncoderOutput.)
  - **`TvpEncoder`** [wiring]: Wiring.
  - **`TvpPooler`** [compute]: `L1/linear.py`, `L1/tanh.py` (Linear + Tanh on first token.)
  - **`TvpFrameDownPadPrompter`** [wiring]: Learnable padding around video frames; pure nn.Parameter manipulation but no fused kb-nano kernel.
  - **`TvpFramePadPrompter`** [wiring]: Same — visual prompter padding.
  - **`TvpModel`** [wiring]: Wiring.
  - **`TvpVideoGroundingHead`** [compute]: `L1/linear.py`, `L1/relu.py` (MLP-style head.)
  - **`TvpForVideoGrounding`** [wiring]: Wiring.

## udop
- **src**: modeling_udop.py
- **status**: partial
- **partial_reason**: UdopLayerCrossAttention has no kb-nano kernel (kb-nano L2/t5_attention.py covers only self-attention with relative bias). RelativePositionBiasHorizontal / RelativePositionBiasVertical compute bbox-based 2D relative-position buckets (uses bbox coordinates instead of token indices) — no kb-nano equivalent. UdopPatchEmbeddings is a standard Conv2d patch projection (covered by L1/conv2d.py).
- **rationale**: UDOP is T5 with extra UdopCellEmbeddings + Relative-Position-Bias{1D, Horizontal, Vertical} for document layout, and full enc-dec with cross-attention. Self-attn / dense FFN / T5 layer-norm map to kb-nano T5 kernels, but UdopLayerCrossAttention and the layout-relative-bias variants (Horizontal/Vertical 2D layout buckets) have no kb-nano equivalent.
- **classes**:
  - **`UdopLayerCrossAttention`** [compute]: UdopLayerCrossAttention has no kb-nano kernel (kb-nano L2/t5_attention.py covers only self-attention with relative bias). RelativePositionBiasHorizontal / RelativePositionBiasVertical compute bbox-bas
  - **`UdopPatchEmbeddings`** [compute]: `L1/conv2d.py` (Conv2d patch projection.)
  - **`UdopLayerNorm`** [compute]: `L1/t5_layer_norm.py` (T5-style RMS norm — same as L1/t5_layer_norm.py.)
  - **`UdopDenseActDense`** [compute]: `L2/t5_dense.py` (T5DenseActDense — same as kb-nano L2.)
  - **`UdopDenseGatedActDense`** [compute]: `L2/t5_dense.py` (T5DenseGatedActDense — same as kb-nano L2.)
  - **`UdopLayerFF`** [compute]: `L3/t5_block.py` (Wiring — same as L3/t5_block.py:T5LayerFF.)
  - **`UdopAttention`** [compute]: `L2/t5_attention.py` (T5-style attention with relative bias; self-attn covered by L2/t5_attention.py, cross-attn path here not.)
  - **`UdopLayerSelfAttention`** [compute]: `L3/t5_block.py` (Same as L3/t5_block.py:T5LayerSelfAttention.)
  - **`UdopBlock`** [wiring]: Wiring (encoder uses self+ff; decoder adds cross).
  - **`UdopCellEmbeddings`** [compute]: `L1/embedding.py` (Two nn.Embedding tables (rows + cols) summed — covered by L1/embedding.py.)
  - **`RelativePositionBiasBase`** [wiring]: Abstract bias base — wiring.
  - **`RelativePositionBias1D`** [wiring]: Token-index relative bias (same buckets as T5) — encoder-side bucket logic exists in kb-nano L2/t5_attention.py but as part of self-attn. Standalone-bias module not in kb-nano.
  - **`RelativePositionBiasHorizontal`** [wiring]: Bbox-x relative bias — no kb-nano equivalent.
  - **`RelativePositionBiasVertical`** [wiring]: Bbox-y relative bias — no kb-nano equivalent.
  - **`RelativePositionBiasAggregated`** [wiring]: Wiring (sum over bias variants).
  - **`UdopStack`** [wiring]: Wiring (encoder side covered by L4/t5_encoder.py-style; decoder side missing cross-attn).
  - **`UdopModel`** [wiring]: Wiring.
  - **`UdopForConditionalGeneration`** [wiring]: Wiring + LM head.
  - **`UdopEncoderModel`** [wiring]: Wiring.

## umt5
- **src**: modeling_umt5.py
- **status**: partial
- **partial_reason**: UMT5LayerCrossAttention has no kb-nano kernel (L2/t5_attention.py covers only self-attention). The relative-bias-per-layer variation is computed inside UMT5Attention itself; kb-nano's L2/t5_attention.py is also self-attn only with the standard first-layer-bias pattern, so per-layer-bias for the encoder works but cross-attn does not.
- **rationale**: UMT5 = T5 with per-layer (not first-layer-only) relative attention bias. Self-attention + dense FFN + T5 layer-norm map to kb-nano T5 kernels; cross-attention (UMT5LayerCrossAttention) has no kb-nano L2 kernel.
- **classes**:
  - **`UMT5LayerCrossAttention`** [compute]: UMT5LayerCrossAttention has no kb-nano kernel (L2/t5_attention.py covers only self-attention). The relative-bias-per-layer variation is computed inside UMT5Attention itself; kb-nano's L2/t5_attention.
  - **`UMT5LayerNorm`** [compute]: `L1/t5_layer_norm.py` (Same as T5LayerNorm — matches L1/t5_layer_norm.py.)
  - **`UMT5DenseActDense`** [compute]: `L2/t5_dense.py` (Same as T5DenseActDense — matches kb-nano L2.)
  - **`UMT5DenseGatedActDense`** [compute]: `L2/t5_dense.py` (Same as T5DenseGatedActDense — matches kb-nano L2.)
  - **`UMT5LayerFF`** [compute]: `L3/t5_block.py` (Wiring — same as L3/t5_block.py:T5LayerFF.)
  - **`UMT5Attention`** [compute]: `L2/t5_attention.py` (T5-style attention with relative bias; self-attn covered by kb-nano L2 (cross-attn branch here is not).)
  - **`UMT5LayerSelfAttention`** [compute]: `L3/t5_block.py` (Same as L3/t5_block.py:T5LayerSelfAttention.)
  - **`UMT5Block`** [wiring]: Wiring (encoder/decoder).
  - **`UMT5ClassificationHead`** [compute]: `L1/linear.py` (Linear + dropout + Linear.)
  - **`UMT5Stack`** [wiring]: Wiring.
  - **`UMT5Model`** [wiring]: Wiring.
  - **`UMT5ForConditionalGeneration`** [wiring]: Wiring + LM head.
  - **`UMT5EncoderModel`** [wiring]: Wiring.
  - **`UMT5ForSequenceClassification`** [wiring]: Wiring.
  - **`UMT5ForTokenClassification`** [wiring]: Wiring.
  - **`UMT5ForQuestionAnswering`** [wiring]: Wiring.

## unispeech
- **src**: modular_unispeech.py
- **status**: partial
- **partial_reason**: Wav2Vec2PositionalConvEmbedding uses nn.utils.weight_norm-parametrized Conv1d (weight reparameterization not exposed in kb-nano L1/conv1d.py). Wav2Vec2GumbelVectorQuantizer uses nn.functional.gumbel_softmax — no kb-nano kernel. Wav2Vec2Attention is BART-style (k_proj/v_proj/q_proj/out_proj plain MHA without RoPE / GQA) — kb-nano has no L2 wrapper for this exact form (closest is whisper_attention.py but that targets the full Whisper enc-dec). Conv feature extractor (Wav2Vec2GroupNormConvLayer) needs L1/group_norm.py (which exists) but the multi-conv stack is not assembled as a kb-nano L2.
- **rationale**: Wav2Vec2-derived speech encoder: stacked Conv1d feature extractor (with weight-norm conv-pos-embed), BART-style attention encoder, and Gumbel-softmax vector quantizer for pretraining. Conv1d primitives exist (L1/conv1d.py) but the weight-normalized Conv1d positional embedding, group-norm-conv-extractor stack, BART-style attention, and Gumbel-softmax quantizer are not implemented as kb-nano kernels.
- **classes**:
  - **`UniSpeechFeatureEncoder`** [compute]: Wav2Vec2PositionalConvEmbedding uses nn.utils.weight_norm-parametrized Conv1d (weight reparameterization not exposed in kb-nano L1/conv1d.py). Wav2Vec2GumbelVectorQuantizer uses nn.functional.gumbel_s
  - **`UniSpeechPositionalConvEmbedding`** [wiring]: weight-norm Conv1d positional embedding — no kb-nano kernel for weight-norm parametrization.
  - **`UniSpeechFeatureProjection`** [compute]: `L1/layer_norm.py`, `L1/linear.py` (LN + Linear + dropout — primitives present.)
  - **`UniSpeechEncoder`** [wiring]: Stack of Wav2Vec2EncoderLayer (BART-style attention + fc1+act+fc2 + LN). Attention not in kb-nano L2.
  - **`UniSpeechEncoderStableLayerNorm`** [wiring]: Same — pre-LN variant.
  - **`UniSpeechGumbelVectorQuantizer`** [wiring]: Gumbel-softmax vector quantizer — no kb-nano kernel.
  - **`UniSpeechModel`** [wiring]: Wiring.
  - **`UniSpeechForPreTraining`** [wiring]: Wiring + contrastive loss.
  - **`UniSpeechForCTC`** [wiring]: Wiring + CTC head.
  - **`UniSpeechForSequenceClassification`** [wiring]: Wiring.

## unispeech_sat
- **src**: modular_unispeech_sat.py
- **status**: partial
- **partial_reason**: Same as unispeech: Wav2Vec2-style weight-norm Conv1d positional embedding, multi-stage Conv1d feature encoder with group-norm, BART-style Wav2Vec2Attention, and Gumbel-softmax vector quantizer are not implemented as kb-nano kernels.
- **rationale**: UniSpeech-SAT is the speaker-aware variant of UniSpeech inheriting the same Wav2Vec2 stack (weight-norm Conv1d pos, conv feature encoder, BART-style attention encoder, Gumbel-softmax quantizer). Same gaps as unispeech.
- **classes**:
  - **`UniSpeechSatFeatureEncoder`** [compute]: no kb-nano kernel — Same as unispeech: Wav2Vec2-style weight-norm Conv1d positional embedding, multi-stage Conv1d feature encoder with group-norm, BART-style Wav2Vec2Attention, and Gumbel-softmax vector quantizer are not
  - **`UniSpeechSatPositionalConvEmbedding`** [wiring]: weight-norm Conv1d — no kb-nano weight-norm wrapper.
  - **`UniSpeechSatFeatureProjection`** [compute]: `L1/layer_norm.py`, `L1/linear.py` (LN + Linear + dropout.)
  - **`UniSpeechSatEncoder`** [wiring]: BART-style attention encoder stack — no kb-nano L2.
  - **`UniSpeechSatEncoderStableLayerNorm`** [wiring]: Pre-LN variant.
  - **`UniSpeechSatGumbelVectorQuantizer`** [wiring]: Gumbel-softmax vector quantizer — no kb-nano kernel.
  - **`UniSpeechSatModel`** [wiring]: Wiring.
  - **`UniSpeechSatForPreTraining`** [wiring]: Wiring + contrastive loss.
  - **`UniSpeechSatForCTC`** [wiring]: Wiring + CTC head.
  - **`UniSpeechSatForSequenceClassification`** [wiring]: Wiring.
  - **`UniSpeechSatForAudioFrameClassification`** [wiring]: Wiring.
  - **`UniSpeechSatForXVector`** [wiring]: Wiring + speaker-embedding head.
