## qwen2_5_omni
- **src**: modeling_qwen2_5_omni.py
- **status**: kb_nano_l4
- **rationale**: L4/qwen2_5_omni.py implements the Thinker (text/image/video/audio) path that targets this HF folder; speech generation (Talker, Token2Wav, BigVGAN, DiT) is intentionally outside kb-nano scope per the L4 docstring.
- **classes**:
  - **`Qwen2_5OmniAudioAttention`** [compute]: `L2/whisper_attention.py` (Encoder-style bidirectional attention with bias=True on Q/V/O and no bias on K, mirroring WhisperAttention; matches WhisperEncoderSelfAttention pattern (no KV cache, dense full attention).)
  - **`Qwen2_5OmniAudioEncoderLayer`** [wiring]: Wiring: self_attn + LayerNorm + fc1/act/fc2 + LayerNorm. Inherits Qwen2AudioEncoderLayer pattern.
  - **`SinusoidsPositionEmbedding`** [wiring]: Pure-PyTorch sinusoidal position embedding, no kb-nano kernel needed (precomputed cosine/sine table lookup).
  - **`Qwen2_5OmniAudioEncoder`** [wiring]: Wiring: stacks audio encoder layers + conv stem + sinusoidal pos.
  - **`Qwen2_5OmniVisionAttention`** [compute]: `L2/vision_attention.py` (Vision encoder MHA with cu_seqlens varlen + 2D vision RoPE; matches VisionAttention which uses FlashAttnPrefill + QKVParallelLinear with vision rotary.)
  - **`Qwen2_5OmniMLP`** [compute]: `L2/llama_mlp.py` (SwiGLU gate_proj/up_proj/down_proj with SiLU activation; standard kb-nano LlamaMLP fused via SiluAndMul.)
  - **`Qwen2_5OmniVisionBlock`** [wiring]: Wiring: norm1 + VisionAttention + norm2 + MLP.
  - **`Qwen2_5_VisionRotaryEmbedding`** [compute]: `L1/vision_rotary_emb.py` (2D vision rotary embedding for image patches; matches kb-nano VisionRotaryEmbedding.)
  - **`Qwen2_5_VisionPatchEmbed`** [compute]: `L2/vision_patch_embed.py` (Conv3d patch embedding for video/image; matches VisionPatchEmbed.)
  - **`Qwen2_5OmniPatchMerger`** [compute]: `L2/vision_patch_merger.py` (RMSNorm + linear merger for spatial patch downsample; matches VisionPatchMerger.)
  - **`Qwen2_5OmniVisionEncoder`** [wiring]: Wiring: vision blocks + merger.
  - **`Qwen2_5OmniRotaryEmbedding`** [compute]: `L1/mrope.py` (Multimodal RoPE for text with mrope_section split (temporal/H/W); matches kb-nano MRotaryEmbedding.)
  - **`Qwen2_5OmniAttention`** [compute]: `L2/attention.py` (Qwen2-style attention with bias=True on QKV, sliding_window per layer_type, MRoPE; matches LlamaAttention(bias=True, sliding_window=...) variant. The MRoPE is applied via apply_multimodal_rotary_pos_emb (the L1/mrope op handles the mrope_section split).)
  - **`Qwen2MLP`** [compute]: `L2/llama_mlp.py` (Standard SwiGLU MLP.)
  - **`Qwen2_5OmniDecoderLayer`** [wiring]: Wiring: input_layernorm + self_attn + post_attention_layernorm + mlp.
  - **`Qwen2_5OmniThinkerTextModel`** [wiring]: Wiring: embed_tokens + decoder layers + final norm.
  - **`Qwen2_5OmniThinkerForConditionalGeneration`** [wiring]: Top-level wiring for Thinker (text + vision + audio).
  - **`Qwen2_5OmniTalkerModel`** [wiring]: Wiring for the Talker speech generator (out of kb-nano scope per L4 docstring).
  - **`Qwen2_5OmniTalkerForConditionalGeneration`** [wiring]: Top-level Talker wiring; not in kb-nano L4.
  - **`Qwen2_5OmniDiTRotaryEmbedding`** [wiring]: DiT speech-codec RoPE, used only in Token2WavDiTModel (out of scope).
  - **`TimeDelayNetBlock`** [wiring]: ECAPA-TDNN speaker-embed block for speech generation; out of kb-nano scope.
  - **`Res2NetBlock`** [wiring]: Res2Net block for ECAPA speaker embed; speech-only.
  - **`SqueezeExcitationBlock`** [wiring]: Standard SE block (1D); out of kb-nano scope (Talker).
  - **`AttentiveStatisticsPooling`** [wiring]: Speech speaker-pooling; out of scope.
  - **`SqueezeExcitationRes2NetBlock`** [wiring]: Composite SE+Res2Net for ECAPA; out of scope.
  - **`ECAPA_TimeDelayNet`** [wiring]: ECAPA-TDNN backbone for speaker embedding; out of scope.
  - **`DiTInputEmbedding`** [wiring]: DiT input embedder for Token2Wav; out of scope.
  - **`DiTCodecEmbedding`** [wiring]: DiT codec token embed; out of scope.
  - **`Qwen2_5_OmniAdaLayerNormZero`** [wiring]: AdaLN-Zero variant for DiT (Token2Wav); out of scope.
  - **`Qwen2_5_OmniAdaLayerNormZero_Final`** [wiring]: Final AdaLN-Zero for DiT; out of scope.
  - **`DiTMLP`** [wiring]: DiT MLP for Token2Wav; out of scope.
  - **`DiTAttention`** [wiring]: DiT attention for Token2Wav (3D rotary); out of scope.
  - **`DiTTimestepEmbedding`** [wiring]: DiT timestep sinusoidal embedding; out of scope.
  - **`DiTDecoderLayer`** [wiring]: DiT decoder layer; out of scope.
  - **`SnakeBeta`** [wiring]: BigVGAN snake-beta activation; out of scope.
  - **`AMPBlock`** [wiring]: BigVGAN AMP block; out of scope.
  - **`Qwen2_5OmniToken2WavBigVGANModel`** [wiring]: BigVGAN vocoder for speech generation; out of scope.
  - **`Qwen2_5OmniToken2WavDiTModel`** [wiring]: DiT-based codec-to-mel model; out of scope.
  - **`Qwen2_5OmniToken2WavModel`** [wiring]: Top-level Token2Wav wrapper; out of scope.
  - **`Qwen2_5OmniForConditionalGeneration`** [wiring]: Top-level Omni wiring (Thinker + Talker + Token2Wav). The Thinker portion maps to L4/qwen2_5_omni.py.

## qwen2_5_vl
- **src**: modular_qwen2_5_vl.py
- **status**: composable
- **rationale**: Qwen2.5-VL shares its language model with Qwen2-VL (M-RoPE) and adds a SiLU+RMSNorm vision encoder; L4/qwen2_vl.py is the canonical pipeline target and L4/qwen25_vl_encoder.py provides a text-only encoder slice. Per HF v4-mapping convention the Qwen2-VL pipeline serves Qwen2.5-VL too.
- **classes**:
  - **`Qwen2_5_VLRMSNorm`** [compute]: `L1/rms_norm.py` (Standard RMSNorm; same as LlamaRMSNorm.)
  - **`Qwen2_5_VLMLP`** [compute]: `L2/llama_mlp.py` (SwiGLU gate/up/down with SiLU; standard LlamaMLP.)
  - **`Qwen2_5_VisionPatchEmbed`** [compute]: `L2/vision_patch_embed.py` (Conv3d patch embedding; matches VisionPatchEmbed.)
  - **`Qwen2_5_VisionRotaryEmbedding`** [compute]: `L1/vision_rotary_emb.py` (2D vision RoPE; matches kb-nano VisionRotaryEmbedding.)
  - **`Qwen2_5_VLPatchMerger`** [compute]: `L2/vision_patch_merger.py` (RMSNorm + 2-layer MLP merger; matches VisionPatchMerger.)
  - **`Qwen2_5_VLVisionAttention`** [compute]: `L2/vision_attention.py` (Vision encoder MHA with vision RoPE + cu_seqlens varlen; matches VisionAttention (FlashAttnPrefill backend).)
  - **`Qwen2_5_VLVisionBlock`** [wiring]: Wiring: norm1 + Attn + norm2 + MLP.
  - **`Qwen2_5_VisionTransformerPretrainedModel`** [wiring]: Wiring: stacks vision blocks + merger; uses windowed attention metadata.
  - **`Qwen2_5_VLModel`** [wiring]: Wiring; inherits all forward logic from Qwen2VLModel (vision + text).
  - **`Qwen2_5_VLForConditionalGeneration`** [wiring]: Top-level wiring for VL conditional generation.

## qwen2_audio
- **src**: modeling_qwen2_audio.py
- **status**: composable
- **rationale**: Qwen2-Audio is a Whisper-style audio encoder + projector wired into a Qwen2 LLM (text decoder lives in qwen2 folder). All components map to existing kb-nano kernels: WhisperEncoderSelfAttention pattern + LayerNorm + 2-layer encoder MLP + linear projector.
- **classes**:
  - **`Qwen2AudioAttention`** [compute]: `L2/whisper_attention.py` (Copied from WhisperAttention: bidirectional encoder MHA with bias=True on Q/V/O, no bias on K. Maps to WhisperEncoderSelfAttention which uses FlashAttnPrefill non-causal.)
  - **`Qwen2AudioEncoderLayer`** [wiring]: Wiring: self_attn + LayerNorm + fc1/act/fc2 + LayerNorm. Activation is configurable; standard composition.
  - **`Qwen2AudioEncoder`** [wiring]: Wiring: conv stem + sinusoidal position + stacked encoder layers + final LayerNorm.
  - **`Qwen2AudioMultiModalProjector`** [compute]: `L1/linear.py` (Single Linear from audio_hidden -> text_hidden; standard Linear.)
  - **`Qwen2AudioForConditionalGeneration`** [wiring]: Wiring: audio encoder + projector + Qwen2 LM.

## qwen2_moe
- **src**: modular_qwen2_moe.py
- **status**: composable
- **rationale**: Qwen2-MoE composes LlamaAttention(bias=True), Qwen2MoeMLP=LlamaMLP, and a sparse MoE block with shared expert + sigmoid shared-expert gate; identical structural pattern to Qwen3-Next MoE which is implemented as L2/shared_expert_moe.py. No L4 exists for the dense Qwen2-MoE family.
- **classes**:
  - **`Qwen2MoeRMSNorm`** [compute]: `L1/rms_norm.py` (Standard RMSNorm.)
  - **`Qwen2MoeRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (Standard NeoX rotary.)
  - **`Qwen2MoeMLP`** [compute]: `L2/llama_mlp.py` (SwiGLU MLP gate/up/down.)
  - **`Qwen2MoeAttention`** [compute]: `L2/attention.py` (Q/K/V have bias (qkv_bias=True), O has no bias; sliding_window per layer_type. Maps to LlamaAttention(bias=True, sliding_window=...).)
  - **`Qwen2MoeExperts`** [compute]: `L2/fused_experts.py`, `L1/moe_grouped_gemm.py` (Per-expert SwiGLU; routed-experts grouped GEMM via FusedExperts.)
  - **`Qwen2MoeTopKRouter`** [compute]: `L1/topk_softmax.py` (Linear gate -> softmax -> top-k -> renormalize; standard topk_softmax pattern.)
  - **`Qwen2MoeSparseMoeBlock`** [compute]: `L2/shared_expert_moe.py` (Shared expert (LlamaMLP with shared_intermediate) + sigmoid shared-gate + routed experts; this is exactly the SharedExpertMoE pattern with shared_expert_attr_name='shared_expert', shared_expert_gate=True, routing='softmax'.)
  - **`Qwen2MoeDecoderLayer`** [wiring]: Wiring: norm + self_attn + norm + mlp (sparse or dense by layer_idx).
  - **`Qwen2MoeModel`** [wiring]: Wiring: embed + decoder stack + final norm.
  - **`Qwen2MoeForCausalLM`** [wiring]: Wiring: model + lm_head.

## qwen2_vl
- **src**: modeling_qwen2_vl.py
- **status**: kb_nano_l4
- **rationale**: L4/qwen2_vl.py docstring explicitly targets Qwen2-VL: vision encoder + Qwen2 language model with M-RoPE.
- **classes**:
  - **`Qwen2VLRMSNorm`** [compute]: `L1/rms_norm.py` (Standard RMSNorm.)
  - **`Qwen2VLRotaryEmbedding`** [compute]: `L1/mrope.py` (Multimodal rotary with mrope_section split (T/H/W); matches kb-nano MRotaryEmbedding.)
  - **`VisionRotaryEmbedding`** [compute]: `L1/vision_rotary_emb.py` (2D vision rotary used inside the patch encoder.)
  - **`PatchEmbed`** [compute]: `L2/vision_patch_embed.py` (Conv3d patch embedding.)
  - **`PatchMerger`** [compute]: `L2/vision_patch_merger.py` (RMSNorm + 2-layer MLP merger.)
  - **`VisionMlp`** [compute]: `L2/vision_mlp.py` (fc1 + QuickGELU + fc2 (Qwen2-VL default activation); matches VisionMLP.)
  - **`VisionAttention`** [compute]: `L2/vision_attention.py` (QKV merged, vision RoPE, cu_seqlens varlen; matches VisionAttention with FlashAttnPrefill.)
  - **`Qwen2VLVisionBlock`** [wiring]: Wiring: norm + attn + norm + mlp.
  - **`Qwen2MLP`** [compute]: `L2/llama_mlp.py` (SwiGLU.)
  - **`Qwen2VLAttention`** [compute]: `L2/attention.py` (Standard Qwen2 attention with QKV bias=True and MRoPE; LlamaAttention(bias=True) consumes positions through MRotaryEmbedding.)
  - **`Qwen2VLDecoderLayer`** [wiring]: Wiring: norm + attn + norm + mlp.
  - **`Qwen2VisionTransformerPretrainedModel`** [wiring]: Wiring.
  - **`Qwen2VLTextModel`** [wiring]: Wiring: embed + decoder stack + norm.
  - **`Qwen2VLModel`** [wiring]: Wiring: vision + text + multimodal merging.
  - **`Qwen2VLForConditionalGeneration`** [wiring]: Top-level wiring.

## qwen3
- **src**: modular_qwen3.py
- **status**: composable
- **rationale**: Qwen3 dense (text-only) is Qwen2 architecture + per-head QK-norm; identical pattern to Qwen3-VL/Qwen3-Next text path. LlamaAttention(qk_norm=True) + LlamaMLP cover all the compute. No standalone L4 yet (subsumed by qwen3_vl/qwen3_next paths and llama-style engines).
- **classes**:
  - **`Qwen3RMSNorm`** [compute]: `L1/rms_norm.py` (Standard RMSNorm.)
  - **`Qwen3MLP`** [compute]: `L2/llama_mlp.py` (SwiGLU gate/up/down.)
  - **`Qwen3RotaryEmbedding`** [compute]: `L1/rotary_emb.py` (Standard NeoX rotary.)
  - **`Qwen3Attention`** [compute]: `L2/attention.py` (QK-norm (per-head RMSNorm on Q and K, head_dim only) + sliding_window per layer_type; matches LlamaAttention(qk_norm=True, sliding_window=...).)
  - **`Qwen3ForCausalLM`** [wiring]: Top-level wiring.

## qwen3_5
- **src**: modular_qwen3_5.py
- **status**: partial
- **partial_reason**: Qwen3_5GatedDeltaNet uses split in_proj_qkv/in_proj_z/in_proj_b/in_proj_a Linear projections; kb-nano L2/qwen3_next_gdn_attention.py expects fused in_proj_qkvz/in_proj_ba and would need a small wrapper to consume split projections. The underlying recurrence kernels exist.
- **rationale**: Qwen3.5 is Qwen3-Next + new GatedDeltaNet projection layout. The recurrence kernels (CausalConv1d, GDNChunkPrefill/Recurrent, RMSNormGated) all exist in kb-nano L1 and L2/qwen3_next_gdn_attention.py, but the new in_proj_qkv / in_proj_z / in_proj_b / in_proj_a split projection layout in Qwen3_5GatedDeltaNet is not directly expressed by the existing kb-nano L2 (which expects fused in_proj_qkvz/in_proj_ba). The HF impl uses torch.split + nn.Linear for the new layout — pure-torch wiring, not a missing kernel.
- **classes**:
  - **`Qwen3_5DecoderLayer`** [compute]: Qwen3_5GatedDeltaNet uses split in_proj_qkv/in_proj_z/in_proj_b/in_proj_a Linear projections; kb-nano L2/qwen3_next_gdn_attention.py expects fused in_proj_qkvz/in_proj_ba and would need a small wrappe
  - **`Qwen3_5VisionRotaryEmbedding`** [compute]: `L1/vision_rotary_emb.py` (Vision rotary.)
  - **`Qwen3_5TextRotaryEmbedding`** [compute]: `L1/mrope.py` (Multimodal rotary on text path.)
  - **`Qwen3_5GatedDeltaNet`** [compute]: `L1/causal_conv1d.py`, `L1/gdn_recurrence.py`, `L1/rms_norm_gated.py` (Same GDN recurrence as Qwen3-Next, but with separate in_proj_qkv / in_proj_z / in_proj_b / in_proj_a Linears (instead of fused in_proj_qkvz/in_proj_ba). Recurrence kernels are reused; the projection split is a wiring change.)
  - **`Qwen3_5Attention`** [compute]: `L2/qwen3_next_attention.py` (Identical to Qwen3-Next full-attention layer (per-head QK-norm + partial RoPE + output gating).)
  - **`Qwen3_5MLP`** [compute]: `L2/llama_mlp.py` (SwiGLU MLP.)
  - **`Qwen3_5RMSNorm`** [compute]: `L1/gemma_rms_norm.py` (Qwen3-Next uses GemmaRMSNorm convention (weight + 1).)
  - **`Qwen3_5VisionModel`** [wiring]: Wiring; same as Qwen3-VL vision.
  - **`Qwen3_5TextModel`** [wiring]: Wiring: embed + hybrid decoder stack.
  - **`Qwen3_5Model`** [wiring]: Wiring: vision + text + multimodal merging.
  - **`Qwen3_5ForCausalLM`** [wiring]: Top-level wiring (text-only).
  - **`Qwen3_5ForConditionalGeneration`** [wiring]: Top-level wiring (text+vision).

## qwen3_5_moe
- **src**: modular_qwen3_5_moe.py
- **status**: partial
- **partial_reason**: Inherits Qwen3_5GatedDeltaNet split projection layout (in_proj_qkv/z/b/a); same wiring gap as qwen3_5. MoE block uses Qwen3MoeSparseMoeBlock pattern (no shared expert) which is L2/qwen3_moe.py.
- **rationale**: Qwen3.5-MoE combines Qwen3.5 GatedDeltaNet (split projection layout) + Qwen3-Next attention + Qwen3-VL-MoE routing. All MoE components (FusedExperts, top-k softmax routing) exist in kb-nano, and Qwen3-Next attention is fully covered, but the new GDN projection split is the same gap as in qwen3_5.
- **classes**:
  - **`Qwen3_5MoeDecoderLayer`** [compute]: no kb-nano kernel — Inherits Qwen3_5GatedDeltaNet split projection layout (in_proj_qkv/z/b/a); same wiring gap as qwen3_5. MoE block uses Qwen3MoeSparseMoeBlock pattern (no shared expert) which is L2/qwen3_moe.py.
  - **`Qwen3_5MoeVisionRotaryEmbedding`** [compute]: `L1/vision_rotary_emb.py` (Vision rotary.)
  - **`Qwen3_5MoeTextRotaryEmbedding`** [compute]: `L1/mrope.py` (Multimodal rotary.)
  - **`Qwen3_5MoeGatedDeltaNet`** [compute]: `L1/causal_conv1d.py`, `L1/gdn_recurrence.py`, `L1/rms_norm_gated.py` (Same GDN recurrence with split projection layout; recurrence kernels exist.)
  - **`Qwen3_5MoeAttention`** [compute]: `L2/qwen3_next_attention.py` (Qwen3-Next full attention with per-head QK-norm + partial RoPE + output gating.)
  - **`Qwen3_5MoeMLP`** [compute]: `L2/llama_mlp.py` (SwiGLU MLP.)
  - **`Qwen3_5MoeExperts`** [compute]: `L2/fused_experts.py` (Routed experts via grouped GEMM.)
  - **`Qwen3_5MoeTopKRouter`** [compute]: `L1/topk_softmax.py` (Top-k softmax routing.)
  - **`Qwen3_5MoeSparseMoeBlock`** [compute]: `L2/shared_expert_moe.py` (Shared expert + sigmoid gate + routed experts (Qwen3-Next pattern).)
  - **`Qwen3_5MoeForCausalLM`** [wiring]: Top-level wiring (text-only MoE).
  - **`Qwen3_5MoeForConditionalGeneration`** [wiring]: Top-level wiring (text+vision MoE).

## qwen3_moe
- **src**: modular_qwen3_moe.py
- **status**: composable
- **rationale**: Qwen3-MoE = Qwen3 dense attention (QK-norm) + sparse MoE without shared expert. Maps cleanly to L3/qwen3_moe_decoder.py which composes LlamaAttention(qk_norm=True) + L2/qwen3_moe.py (Qwen3MoE class with FusedExperts).
- **classes**:
  - **`Qwen3MoeAttention`** [compute]: `L2/attention.py` (Same as Qwen3Attention (QK-norm + sliding window optional).)
  - **`Qwen3MoeMLP`** [compute]: `L2/llama_mlp.py` (SwiGLU.)
  - **`Qwen3MoeExperts`** [compute]: `L2/fused_experts.py` (Per-expert SwiGLU via grouped GEMM.)
  - **`Qwen3MoeTopKRouter`** [compute]: `L1/topk_softmax.py` (Linear -> softmax -> top-k -> renormalize.)
  - **`Qwen3MoeSparseMoeBlock`** [compute]: `L2/qwen3_moe.py` (Routed-only MoE (no shared expert); kb-nano L2/qwen3_moe.py implements exactly this pattern with optional FP8 quant.)
  - **`Qwen3MoeRMSNorm`** [compute]: `L1/rms_norm.py` (Standard RMSNorm.)
  - **`Qwen3MoeDecoderLayer`** [wiring]: Wiring; matches L3/qwen3_moe_decoder.py.
  - **`Qwen3MoeModel`** [wiring]: Wiring.
  - **`Qwen3MoeForCausalLM`** [wiring]: Top-level wiring.

## qwen3_next
- **src**: modular_qwen3_next.py
- **status**: kb_nano_l4
- **rationale**: L4/qwen3_next.py docstring explicitly targets Qwen/Qwen3-Next-80B-A3B-Instruct (3:1 GDN+full attention layers, MoE with shared expert).
- **classes**:
  - **`Qwen3NextRMSNorm`** [compute]: `L1/gemma_rms_norm.py` (GemmaRMSNorm convention (weight + 1).)
  - **`Qwen3NextRMSNormGated`** [compute]: `L1/rms_norm_gated.py` (Gated RMSNorm used by GDN linear attention output.)
  - **`Qwen3NextAttention`** [compute]: `L2/qwen3_next_attention.py` (Per-head QK-norm + partial RoPE (25%) + output gating (sigmoid); matches kb-nano Qwen3NextAttention exactly.)
  - **`Qwen3NextGatedDeltaNet`** [compute]: `L2/qwen3_next_gdn_attention.py`, `L1/causal_conv1d.py`, `L1/gdn_recurrence.py`, `L1/rms_norm_gated.py` (GDN linear attention; matches kb-nano Qwen3NextGDNAttention.)
  - **`Qwen3NextMLP`** [compute]: `L2/llama_mlp.py` (SwiGLU.)
  - **`Qwen3NextSparseMoeBlock`** [compute]: `L2/qwen3_next_moe.py`, `L2/shared_expert_moe.py` (512-routed-experts top-10 softmax + shared expert with sigmoid gate; matches Qwen3NextMoE which uses FusedExperts + AllReduce.)
  - **`Qwen3NextDecoderLayer`** [wiring]: Wiring: alternates GDN linear attn vs full attn; mirrored in L3/qwen3_next_decoder.py.
  - **`Qwen3NextModel`** [wiring]: Wiring.
  - **`Qwen3NextForCausalLM`** [wiring]: Top-level wiring.

## qwen3_omni_moe
- **src**: modular_qwen3_omni_moe.py
- **status**: partial
- **partial_reason**: Code2Wav stack (CausalConvNet, CausalTransConvNet, ConvNeXtBlock, SnakeBeta-based decoder, AMP block, BigVGAN-style decoder) is a speech vocoder/codec that has no kb-nano equivalent. The transformer layers (Code2WavAttention, Code2WavMlp, Code2WavRMSNorm) reuse Qwen3 patterns and are individually composable, but the surrounding upsample/convnet decoder is missing.
- **rationale**: Thinker text path is Qwen3-MoE (composable). Vision/audio encoders mirror Qwen3-VL-MoE/Qwen2.5-Omni patterns (composable). Talker (text + code predictor) is text-only Qwen3-style; covered. But Code2Wav speech generation (transformer + ConvNeXt + SnakeBeta + decoder upsampling) is speech-only and out of kb-nano scope, similar to qwen2_5_omni Token2Wav.
- **classes**:
  - **`Qwen3OmniMoeAudioEncoder`** [compute]: Code2Wav stack (CausalConvNet, CausalTransConvNet, ConvNeXtBlock, SnakeBeta-based decoder, AMP block, BigVGAN-style decoder) is a speech vocoder/codec that has no kb-nano equivalent. The transformer l
  - **`Qwen3OmniMoeAudioAttention`** [compute]: `L2/whisper_attention.py` (Whisper-style audio MHA.)
  - **`Qwen3OmniMoeVisionAttention`** [compute]: `L2/vision_attention.py` (Vision MHA + 2D RoPE + cu_seqlens.)
  - **`Qwen3OmniMoeVisionPatchMerger`** [compute]: `L2/vision_patch_merger.py` (Patch merger.)
  - **`Qwen3OmniMoeVisionRotaryEmbedding`** [compute]: `L1/vision_rotary_emb.py` (Vision RoPE.)
  - **`Qwen3OmniMoeVisionEncoder`** [wiring]: Wiring.
  - **`Qwen3OmniMoeThinkerTextRotaryEmbedding`** [compute]: `L1/mrope.py` (Multimodal rotary.)
  - **`Qwen3OmniMoeThinkerTextExperts`** [compute]: `L2/fused_experts.py` (Routed experts grouped GEMM.)
  - **`Qwen3OmniMoeThinkerTextTopKRouter`** [compute]: `L1/topk_softmax.py` (Routing.)
  - **`Qwen3OmniMoeThinkerTextSparseMoeBlock`** [compute]: `L2/qwen3_moe.py` (Routed-only MoE (no shared expert).)
  - **`Qwen3OmniMoeThinkerTextAttention`** [compute]: `L2/attention.py` (Qwen3 QK-norm attention.)
  - **`Qwen3OmniMoeThinkerTextDecoderLayer`** [wiring]: Wiring.
  - **`Qwen3OmniMoeThinkerTextModel`** [wiring]: Wiring.
  - **`Qwen3OmniMoeThinkerForConditionalGeneration`** [wiring]: Top-level Thinker wiring.
  - **`Qwen3OmniMoeTalkerCodePredictorAttention`** [compute]: `L2/attention.py` (Qwen3 QK-norm attention.)
  - **`Qwen3OmniMoeTalkerCodePredictorDecoderLayer`** [wiring]: Wiring.
  - **`Qwen3OmniMoeTalkerCodePredictorModel`** [wiring]: Wiring.
  - **`Qwen3OmniMoeTalkerTextSparseMoeBlock`** [compute]: `L2/shared_expert_moe.py` (Qwen2-Moe-style shared expert MoE.)
  - **`Qwen3OmniMoeTalkerDecoderLayer`** [wiring]: Wiring.
  - **`Qwen3OmniMoeTalkerModel`** [wiring]: Wiring.
  - **`Qwen3OmniMoeCausalConvNet`** [compute]: `L1/causal_conv1d.py` (Causal Conv1D for code-to-wave decoder; kb-nano CausalConv1d covers it.)
  - **`Qwen3OmniMoeCausalTransConvNet`** [compute]: `L1/conv_transpose1d.py` (Causal transposed conv1d for upsampling; kb-nano ConvTranspose1d covers it.)
  - **`Qwen3OmniMoeConvNeXtBlock`** [wiring]: ConvNeXt 1D block (depthwise conv + LayerNorm + pointwise) used inside Code2Wav decoder; no direct kb-nano L2 wrapper for 1D ConvNeXt-style block.
  - **`Qwen3OmniMoeCode2WavAttention`** [compute]: `L2/attention.py` (Qwen3 attention reused inside Code2Wav.)
  - **`Qwen3OmniMoeCode2WavMlp`** [compute]: `L2/llama_mlp.py` (SwiGLU.)
  - **`Qwen3OmniMoeCode2WavRMSNorm`** [compute]: `L1/rms_norm.py` (RMSNorm.)
  - **`Qwen3OmniMoeCode2WavLayerScale`** [wiring]: Per-channel learnable scaling; trivial torch op.
  - **`Qwen3OmniMoeCode2WavTransformerLayer`** [wiring]: Wiring.
  - **`Qwen3OmniMoeCode2WavTransformerModel`** [wiring]: Wiring.
  - **`SnakeBeta`** [wiring]: BigVGAN snake-beta activation; no kb-nano kernel.
  - **`Qwen3OmniMoeCode2WavDecoderResidualUnit`** [wiring]: Residual unit inside the speech decoder; no kb-nano composite.
  - **`Qwen3OmniMoeCode2WavDecoderBlock`** [wiring]: Decoder block wiring.
  - **`Qwen3OmniMoeCode2Wav`** [wiring]: Top-level Code2Wav wrapper; speech vocoder out of scope.
  - **`Qwen3OmniMoeForConditionalGeneration`** [wiring]: Top-level wiring (Thinker + Talker + Code2Wav).

## qwen3_vl
- **src**: modular_qwen3_vl.py
- **status**: kb_nano_l4
- **rationale**: L4/qwen3_vl.py docstring explicitly targets Qwen3-VL: vision encoder with DeepStack + Qwen3 language model with M-RoPE.
- **classes**:
  - **`Qwen3VLVisionMLP`** [compute]: `L2/vision_mlp.py` (fc1 + SiLU + fc2 (Qwen3-VL uses SiLU instead of QuickGELU); VisionMLP supports configurable activation.)
  - **`Qwen3VLVisionPatchEmbed`** [compute]: `L2/vision_patch_embed.py` (Conv3d patch embedding.)
  - **`Qwen3VLVisionRotaryEmbedding`** [compute]: `L1/vision_rotary_emb.py` (Vision RoPE.)
  - **`Qwen3VLVisionPatchMerger`** [compute]: `L2/vision_patch_merger.py` (Patch merger.)
  - **`Qwen3VLVisionAttention`** [compute]: `L2/vision_attention.py` (Vision MHA with cu_seqlens.)
  - **`Qwen3VLVisionBlock`** [wiring]: Wiring.
  - **`Qwen3VLTextRotaryEmbedding`** [compute]: `L1/mrope.py` (Interleaved M-RoPE for text path.)
  - **`Qwen3VLTextAttention`** [compute]: `L2/attention.py` (Per-head QK-norm + RoPE; matches LlamaAttention(qk_norm=True). Verified the forward applies q_norm/k_norm before transpose then RoPE.)
  - **`Qwen3VLTextDecoderLayer`** [wiring]: Wiring; mirrored by L3/llama_decoder.py.
  - **`Qwen3VLVisionModel`** [wiring]: Wiring.
  - **`Qwen3VLTextModel`** [wiring]: Wiring.
  - **`Qwen3VLModel`** [wiring]: Wiring: vision + text + multimodal merging.
  - **`Qwen3VLForConditionalGeneration`** [wiring]: Top-level wiring.

## qwen3_vl_moe
- **src**: modular_qwen3_vl_moe.py
- **status**: kb_nano_l4
- **rationale**: L4/qwen3_vl_moe.py docstring explicitly targets Qwen3-VL-MoE: Qwen3VisionTransformer + Qwen3-MoE language model (128 FP8 experts, top-8).
- **classes**:
  - **`Qwen3VLMoeTextRMSNorm`** [compute]: `L1/rms_norm.py` (Standard RMSNorm.)
  - **`Qwen3VLMoeTextExperts`** [compute]: `L2/fused_experts.py` (Routed experts grouped GEMM.)
  - **`Qwen3VLMoeTextTopKRouter`** [compute]: `L1/topk_softmax.py` (Top-k softmax routing.)
  - **`Qwen3VLMoeTextSparseMoeBlock`** [compute]: `L2/qwen3_moe.py` (Routed-only MoE (no shared expert) with FP8 quant.)
  - **`Qwen3VLMoeTextAttention`** [compute]: `L2/attention.py` (Qwen3 QK-norm attention.)
  - **`Qwen3VLMoeTextDecoderLayer`** [wiring]: Wiring; matches L3/qwen3_moe_decoder.py.
  - **`Qwen3VLMoeVisionRotaryEmbedding`** [compute]: `L1/vision_rotary_emb.py` (Vision RoPE.)
  - **`Qwen3VLMoeVisionAttention`** [compute]: `L2/vision_attention.py` (Vision MHA.)
  - **`Qwen3VLMoeVisionBlock`** [wiring]: Wiring.
  - **`Qwen3VLMoeVisionModel`** [wiring]: Wiring.
  - **`Qwen3VLMoeTextModel`** [wiring]: Wiring.
  - **`Qwen3VLMoeForConditionalGeneration`** [wiring]: Top-level wiring.

## recurrent_gemma
- **src**: modeling_recurrent_gemma.py
- **status**: partial
- **partial_reason**: configuration_recurrent_gemma.py sets `partial_rotary_factor = 0.5`. RecurrentGemmaSdpaAttention does q_rot, q_pass split and rotates only q_rot. kb-nano L2/attention.py forwards full q,k to rotary_emb; standard L1/RotaryEmbedding rotates the full head_dim. Partial-rotary on Griffin requires either external slicing in user code or a Gemma4-style proportional rotary subclass. (This is the same gap as phi/persimmon — applied here for consistency; the earlier audit's claim that "partial-rotary is a chunked-RoPE wrap that uses the same L1 rotary kernel" is correct only if the user adds external slicing.)
- **rationale**: Griffin-style hybrid: SDPA attention with partial-rotary 0.5 + RG-LRU recurrent block + standard SwiGLU MLP. RG-LRU, RMSNorm (Gemma), SwiGLU, Llama attention primitives all map to existing kernels; partial-rotary needs external slicing on top of L1/rotary_emb.
- **classes**:
  - **`RecurrentGemmaRMSNorm`** [compute]: `L1/gemma_rms_norm.py` (Gemma-style RMSNorm (weight + 1).)
  - **`RecurrentGemmaRotaryEmbedding`** [compute]: `L1/rotary_emb.py` (Standard NeoX RoPE on the rotary slice; the partial-rotary q_rot/q_pass split happens in RecurrentGemmaSdpaAttention's forward, not in the embedding itself.)
  - **`RecurrentGemmaSdpaAttention`** [compute]: `L2/attention.py` (GQA SDPA with partial-rotary (q_rot, q_pass split done inside forward) + bias on o_proj only. kb-nano L2/attention.py does NOT slice q/k before calling rotary_emb; partial-rotary requires either external slicing or a Gemma4-style proportional rotary wrapper. Same gap as phi/persimmon.)
  - **`RecurrentGemmaRglru`** [compute]: `L1/rg_lru.py` (Real-Gated Linear Recurrent Unit; kb-nano L1/rg_lru.py is the faithful re-implementation per its docstring.)
  - **`RecurrentGemmaRecurrentBlock`** [compute]: `L1/causal_conv1d.py`, `L1/rg_lru.py` (linear_x + linear_y -> CausalConv1d -> RG-LRU -> linear_out; CausalConv1d + RGLRU exist as kb-nano L1 ops.)
  - **`RecurrentGemmaMlp`** [compute]: `L2/llama_mlp.py` (GeGLU-style up/gate (with GELU activation) + down; LlamaMLP supports SwiGLU pattern (silu_and_mul) but Gemma uses GELU — covered by L1/gelu_and_mul.py if needed. Standard SwiGLU MLP.)
  - **`RecurrentGemmaDecoderLayer`** [wiring]: Wiring: norm + (recurrent_block or attention) + norm + mlp.
  - **`RecurrentGemmaModel`** [wiring]: Wiring.
  - **`RecurrentGemmaForCausalLM`** [wiring]: Top-level wiring.

## reformer
- **src**: modeling_reformer.py
- **status**: partial
- **partial_reason**: LSH/local chunked attention with bucketing and sort/unsort logic has no kb-nano equivalent; the underlying ops (matmul, softmax, gather) are torch primitives but no kb-nano L2 module wraps the LSH attention pattern.
- **rationale**: Reformer's LSH self-attention performs LSH bucketing + chunked attention with sort/unsort tricks; pure-PyTorch (no custom CUDA), but no kb-nano L2 wraps the LSH/chunked-attention pattern. LocalSelfAttention is similarly chunked attention with stride. The reversible residual function is a custom torch.autograd.Function. All falls back to torch ops, so partial rather than unsupported.
- **classes**:
  - **`LSHSelfAttention`** [compute]: LSH/local chunked attention with bucketing and sort/unsort logic has no kb-nano equivalent; the underlying ops (matmul, softmax, gather) are torch primitives but no kb-nano L2 module wraps the LSH att
  - **`AxialPositionEmbeddings`** [compute]: `L1/embedding.py` (Two factorized embedding tables broadcast over axial positions; embedding lookup.)
  - **`PositionEmbeddings`** [compute]: `L1/embedding.py` (Standard learned positional embedding.)
  - **`ReformerEmbeddings`** [wiring]: Wiring: token embed + position embed + dropout.
  - **`LocalSelfAttention`** [wiring]: Chunked local attention with stride; no kb-nano L2 wrapper.
  - **`ReformerSelfOutput`** [compute]: `L1/linear.py` (Output linear projection.)
  - **`ReformerAttention`** [wiring]: Wiring: pre-norm + (LSH or Local) + output.
  - **`ReformerFeedForwardDense`** [compute]: `L1/linear.py` (fc1 + activation.)
  - **`ReformerFeedForwardOutput`** [compute]: `L1/linear.py` (fc2 + dropout.)
  - **`ChunkReformerFeedForward`** [wiring]: Wiring: pre-norm + dense + output, optionally chunked over seq dim.
  - **`ReformerLayer`** [wiring]: Wiring: reversible attention + feed-forward.
  - **`ReformerEncoder`** [wiring]: Stacks layers with reversible residual.
  - **`ReformerOnlyLMHead`** [compute]: `L1/linear.py` (LM head linear.)
  - **`ReformerModel`** [wiring]: Wiring.
  - **`ReformerModelWithLMHead`** [wiring]: Wiring.

## regnet
- **src**: modeling_regnet.py
- **status**: composable
- **rationale**: RegNet (X/Y) is a CNN classifier: Conv2d + BatchNorm + ReLU + adaptive avg pool + linear head, with optional Squeeze-Excitation. All ops have kb-nano L1 equivalents (conv2d, batch_norm2d, relu, adaptive_avg_pool2d, linear). Squeeze-Excite block is structurally similar to L2/efficientnetv2_squeeze_excite.py.
- **classes**:
  - **`RegNetConvLayer`** [compute]: `L1/conv2d.py`, `L1/batch_norm2d.py`, `L1/relu.py` (Conv2d + BatchNorm2d + ReLU.)
  - **`RegNetEmbeddings`** [wiring]: Wiring: stem ConvLayer.
  - **`RegNetShortCut`** [compute]: `L1/conv2d.py`, `L1/batch_norm2d.py` (1x1 conv + batchnorm shortcut for downsampling.)
  - **`RegNetSELayer`** [compute]: `L1/adaptive_avg_pool2d.py`, `L1/conv2d.py`, `L1/relu.py` (Squeeze-Excitation: GAP -> 1x1 conv -> ReLU -> 1x1 conv -> sigmoid -> scale.)
  - **`RegNetXLayer`** [wiring]: Wiring: 3 convs (1x1 -> 3x3 grouped -> 1x1) + shortcut + ReLU.
  - **`RegNetYLayer`** [wiring]: Wiring: like X-layer but with SE block.
  - **`RegNetStage`** [wiring]: Wiring: stacks layers within a stage.
  - **`RegNetEncoder`** [wiring]: Wiring: stacks stages.
  - **`RegNetModel`** [wiring]: Wiring: embeddings + encoder + adaptive avg pool.
  - **`RegNetForImageClassification`** [wiring]: Wiring: backbone + linear head.

## rembert
- **src**: modeling_rembert.py
- **status**: composable
- **rationale**: RemBERT is a BERT variant with embedding-input-projection and a slightly larger embedding dim. All compute classes map to encoder kernels: EncoderSelfAttention, EncoderMLP-equivalent, BertEmbeddings.
- **classes**:
  - **`RemBertEmbeddings`** [compute]: `L2/encoder_embeddings.py`, `L2/bert_embeddings.py` (Token + token-type + position embeddings + LayerNorm + (input projection from input_embedding_size to hidden_size); the projection is a single Linear, the rest matches BertEmbeddings.)
  - **`RemBertPooler`** [compute]: `L1/linear.py` (Linear + tanh on first token.)
  - **`RemBertSelfAttention`** [compute]: `L2/encoder_attention.py` (Standard BERT self-attention (Q/K/V Linear with bias) -> matches EncoderSelfAttention.)
  - **`RemBertSelfOutput`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (Linear + dropout + LayerNorm (residual).)
  - **`RemBertAttention`** [wiring]: Wiring: SelfAttention + SelfOutput (sibling-class wrapper, see methodology rule 11).
  - **`RemBertIntermediate`** [compute]: `L1/linear.py`, `L1/gelu.py` (fc1 + GELU activation.)
  - **`RemBertOutput`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (fc2 + dropout + LayerNorm (residual).)
  - **`RemBertLayer`** [wiring]: Wiring: attention + intermediate + output.
  - **`RemBertEncoder`** [wiring]: Wiring: input projection + stacked layers.
  - **`RemBertPredictionHeadTransform`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (Dense + activation + LayerNorm.)
  - **`RemBertLMPredictionHead`** [compute]: `L1/linear.py` (Dense + decoder linear.)
  - **`RemBertModel`** [wiring]: Wiring.

## resnet
- **src**: modeling_resnet.py
- **status**: composable
- **rationale**: Standard ResNet: Conv2d + BatchNorm2d + ReLU + MaxPool + adaptive avg pool + linear head. Basic and BottleNeck residual layers are stacks of these. All ops covered by kb-nano L1 (conv2d, batch_norm2d, relu, max_pool2d, adaptive_avg_pool2d, linear).
- **classes**:
  - **`ResNetConvLayer`** [compute]: `L1/conv2d.py`, `L1/batch_norm2d.py`, `L1/relu.py` (Conv2d + BatchNorm2d + activation.)
  - **`ResNetEmbeddings`** [compute]: `L1/max_pool2d.py` (Stem ConvLayer + 3x3 MaxPool stride 2.)
  - **`ResNetShortCut`** [compute]: `L1/conv2d.py`, `L1/batch_norm2d.py` (1x1 conv + batchnorm.)
  - **`ResNetBasicLayer`** [wiring]: Wiring: 2 ConvLayers + shortcut + ReLU.
  - **`ResNetBottleNeckLayer`** [wiring]: Wiring: 1x1 -> 3x3 -> 1x1 ConvLayers + shortcut.
  - **`ResNetStage`** [wiring]: Wiring: stacks layers.
  - **`ResNetEncoder`** [wiring]: Wiring: stacks stages.
  - **`ResNetModel`** [wiring]: Wiring: embeddings + encoder + adaptive avg pool.
  - **`ResNetForImageClassification`** [wiring]: Wiring: backbone + linear head.
  - **`ResNetBackbone`** [wiring]: Wiring.

## roberta
- **src**: modular_roberta.py
- **status**: composable
- **rationale**: RoBERTa is BERT with a different position-embedding offset and no token-type embeddings; inherits everything else from BERT. All compute maps to L2/encoder_attention.py + L2/encoder_mlp.py + L2/bert_embeddings.py.
- **classes**:
  - **`RobertaEmbeddings`** [compute]: `L2/bert_embeddings.py` (Word + position + (no token-type) + LayerNorm; matches BertEmbeddings with adjusted padding offset.)
  - **`RobertaSelfAttention`** [compute]: `L2/encoder_attention.py` (Inherits BertSelfAttention; standard encoder self-attention.)
  - **`RobertaCrossAttention`** [compute]: `L2/encoder_attention.py` (Cross-attention variant of EncoderSelfAttention.)
  - **`RobertaLayer`** [wiring]: Wiring: attention + intermediate + output.
  - **`RobertaModel`** [wiring]: Wiring: embeddings + encoder + pooler.
  - **`RobertaForCausalLM`** [wiring]: Wiring.
  - **`RobertaForMaskedLM`** [wiring]: Wiring.
  - **`RobertaLMHead`** [compute]: `L1/linear.py`, `L1/gelu.py`, `L1/layer_norm.py` (Dense + GELU + LayerNorm + decoder.)
  - **`RobertaForSequenceClassification`** [wiring]: Wiring.
  - **`RobertaForMultipleChoice`** [wiring]: Wiring.
  - **`RobertaForTokenClassification`** [wiring]: Wiring.
  - **`RobertaClassificationHead`** [compute]: `L1/linear.py` (Dense + tanh + dropout + classifier.)
  - **`RobertaForQuestionAnswering`** [wiring]: Wiring.

## roberta_prelayernorm
- **src**: modeling_roberta_prelayernorm.py
- **status**: composable
- **rationale**: Pre-LayerNorm variant of RoBERTa: LayerNorm before attention/FFN instead of after. The compute primitives (encoder attention, MLP, layer_norm) are identical to RoBERTa; only the residual order changes (a wiring-level difference in *Layer/*Output classes).
- **classes**:
  - **`RobertaPreLayerNormEmbeddings`** [compute]: `L2/bert_embeddings.py` (Same as RobertaEmbeddings.)
  - **`RobertaPreLayerNormSelfAttention`** [compute]: `L2/encoder_attention.py` (Standard encoder self-attention.)
  - **`RobertaPreLayerNormCrossAttention`** [compute]: `L2/encoder_attention.py` (Cross-attention variant.)
  - **`RobertaPreLayerNormSelfOutput`** [compute]: `L1/linear.py` (Linear + dropout (no LayerNorm here -- moved before attention).)
  - **`RobertaPreLayerNormAttention`** [wiring]: Wiring: pre-norm + SelfAttention + SelfOutput (sibling-class wrapper).
  - **`RobertaPreLayerNormIntermediate`** [compute]: `L1/linear.py`, `L1/gelu.py` (Pre-norm + fc1 + GELU.)
  - **`RobertaPreLayerNormOutput`** [compute]: `L1/linear.py` (fc2 + dropout.)
  - **`RobertaPreLayerNormLayer`** [wiring]: Wiring.
  - **`RobertaPreLayerNormEncoder`** [wiring]: Wiring.
  - **`RobertaPreLayerNormPooler`** [compute]: `L1/linear.py` (Linear + tanh.)
  - **`RobertaPreLayerNormModel`** [wiring]: Wiring.
  - **`RobertaPreLayerNormForCausalLM`** [wiring]: Wiring.
  - **`RobertaPreLayerNormForMaskedLM`** [wiring]: Wiring.
  - **`RobertaPreLayerNormLMHead`** [compute]: `L1/linear.py`, `L1/gelu.py`, `L1/layer_norm.py` (Dense + GELU + LayerNorm + decoder.)
  - **`RobertaPreLayerNormForSequenceClassification`** [wiring]: Wiring.
  - **`RobertaPreLayerNormClassificationHead`** [compute]: `L1/linear.py` (Dense + tanh + classifier.)

## roc_bert
- **src**: modeling_roc_bert.py
- **status**: composable
- **rationale**: RoCBert = BERT with extra phonetic + shape embeddings (sums into word embeddings). All other compute classes (SelfAttention, CrossAttention, Layer, Encoder, Pooler) are BERT-equivalent and map to encoder kernels.
- **classes**:
  - **`RoCBertEmbeddings`** [compute]: `L1/embedding.py`, `L1/layer_norm.py` (Word + token-type + position + phonetic + shape embeddings (all Linear/Embedding lookups) summed and LayerNormed.)
  - **`RoCBertSelfAttention`** [compute]: `L2/encoder_attention.py` (Standard BERT self-attention.)
  - **`RoCBertCrossAttention`** [compute]: `L2/encoder_attention.py` (Cross-attention variant of encoder MHA.)
  - **`RoCBertSelfOutput`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (Linear + dropout + LayerNorm.)
  - **`RoCBertAttention`** [wiring]: Wiring: SelfAttention + SelfOutput (sibling-class wrapper).
  - **`RoCBertIntermediate`** [compute]: `L1/linear.py`, `L1/gelu.py` (fc1 + GELU.)
  - **`RoCBertOutput`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (fc2 + dropout + LayerNorm.)
  - **`RoCBertLayer`** [wiring]: Wiring.
  - **`RoCBertEncoder`** [wiring]: Wiring.
  - **`RoCBertPooler`** [compute]: `L1/linear.py` (Linear + tanh.)
  - **`RoCBertPredictionHeadTransform`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (Dense + activation + LayerNorm.)
  - **`RoCBertLMPredictionHead`** [compute]: `L1/linear.py` (Dense + decoder.)
  - **`RoCBertModel`** [wiring]: Wiring.

## roformer
- **src**: modeling_roformer.py
- **status**: partial
- **partial_reason**: EncoderSelfAttention in kb-nano (L2/encoder_attention.py) does not apply rotary position embeddings inside the bidirectional encoder; would need a small variant that calls L1/rotary_emb.py on Q/K (and optionally V) before SDPA. The L1 RoPE op exists but is not wired into encoder attention.
- **rationale**: RoFormer is BERT with rotary position embedding inside encoder self-attention. Standard NeoX RoPE exists in kb-nano (L1/rotary_emb.py), but EncoderSelfAttention in kb-nano does not currently apply RoPE inside the encoder forward — the encoder kernel is bias-only Linear projections + dense attention. RoFormer also has an optional rotary_value flag (apply RoPE to V too), which is non-standard.
- **classes**:
  - **`RoFormerAttention`** [wiring]: Sibling-wrapper around RoFormerSelfAttention + RoFormerSelfOutput. Per guideline 11 the bare *Attention class is wiring; the missing-encoder-RoPE gap lives on RoFormerSelfAttention below.
  - **`RoFormerSinusoidalPositionalEmbedding`** [wiring]: Pure-PyTorch sinusoidal lookup; small custom op.
  - **`RoFormerEmbeddings`** [compute]: `L2/bert_embeddings.py` (Word + token-type + LayerNorm (no position embedding -- RoPE applied in attention).)
  - **`RoFormerSelfAttention`** [compute]: `L1/rotary_emb.py` (Encoder self-attention with RoPE applied inside forward (apply_rotary_position_embeddings); L1 RoPE op exists but encoder attention wiring needs an integration. Optional rotary_value applies RoPE to V as well.)
  - **`RoFormerSelfOutput`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (Standard.)
  - **`RoFormerIntermediate`** [compute]: `L1/linear.py`, `L1/gelu.py` (fc1 + GELU.)
  - **`RoFormerOutput`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (fc2 + dropout + LayerNorm.)
  - **`RoFormerLayer`** [wiring]: Wiring.
  - **`RoFormerEncoder`** [wiring]: Wiring.
  - **`RoFormerSequenceSummary`** [wiring]: Wiring.
  - **`RoFormerPredictionHeadTransform`** [compute]: `L1/linear.py`, `L1/layer_norm.py` (Standard.)
  - **`RoFormerLMPredictionHead`** [compute]: `L1/linear.py` (Dense + decoder.)
  - **`RoFormerModel`** [wiring]: Wiring.

## rt_detr/rt_detr
- **src**: modular_rt_detr.py
- **status**: composable
- **rationale**: RT-DETR (v1) shares all components with RT-DETR-V2 except the deformable attention variant: v1 uses DeformableDetr-style multi-scale deformable attention which performs the same bilinear sampling as kb-nano L1/rtdetrv2_deformable_attention.py. Hybrid encoder (CSP-Rep + AIFI), decoder, RepVggBlock, FrozenBatchNorm, ConvNormLayer all have direct kb-nano L2 wrappers. There is no L4 dedicated to v1 (L4/rtdetrv2.py targets v2).
- **classes**:
  - **`RTDetrMLP`** [compute]: `L2/rtdetrv2_mlp_head.py` (fc1 + activation + dropout + fc2 + dropout; matches generic 2-layer MLP.)
  - **`RTDetrFrozenBatchNorm2d`** [compute]: `L1/frozen_batch_norm2d.py` (Inference-only batchnorm; kb-nano FrozenBatchNorm2d covers this.)
  - **`RTDetrSelfAttention`** [compute]: `L2/rtdetrv2_multihead_attention.py` (DETR-style multi-head attention with optional position embeddings.)
  - **`RTDetrConvEncoder`** [compute]: `L2/rtdetrv2_resnet.py` (Wiring: ResNet-style backbone (HGNetv2 or ResNet).)
  - **`RTDetrConvNormLayer`** [compute]: `L2/rtdetrv2_conv_norm.py` (Conv2d + BatchNorm + activation.)
  - **`RTDetrEncoderLayer`** [compute]: `L2/rtdetrv2_encoder_layer.py` (Wiring: norm + attn + norm + ffn.)
  - **`RTDetrRepVggBlock`** [compute]: `L2/rtdetrv2_repvgg_block.py` (RepVGG block (3x3 + 1x1 + identity at train, fused at inference).)
  - **`RTDetrCSPRepLayer`** [compute]: `L2/rtdetrv2_csp_rep_layer.py` (CSP-style residual stack with RepVgg blocks.)
  - **`RTDetrMultiscaleDeformableAttention`** [compute]: `L1/rtdetrv2_deformable_attention.py`, `L2/rtdetrv2_deformable_attention.py` (Multi-scale deformable attention via bilinear sampling; the v1 method='default' path is what kb-nano's MultiScaleDeformableAttentionV2 implements.)
  - **`RTDetrDecoderLayer`** [wiring]: Wiring: self_attn + deformable cross_attn + ffn.
  - **`RTDetrSinePositionEmbedding`** [wiring]: Pure-PyTorch sinusoidal pos embedding for 2D queries.
  - **`RTDetrAIFILayer`** [wiring]: Wiring: applies one EncoderLayer to the highest-resolution feature with sine pos.
  - **`RTDetrMLPPredictionHead`** [compute]: `L2/rtdetrv2_mlp_head.py` (Multi-layer MLP head for box regression.)
  - **`RTDetrHybridEncoder`** [compute]: `L3/rtdetrv2_hybrid_encoder.py` (Wiring: AIFI + CSP-Rep + FPN/PAN.)
  - **`RTDetrDecoder`** [compute]: `L3/rtdetrv2_decoder.py` (Wiring: stacked DecoderLayers with intermediate refinement.)
  - **`RTDetrModel`** [compute]: `L3/rtdetrv2_model.py` (Wiring: backbone + hybrid encoder + decoder.)
  - **`RTDetrForObjectDetection`** [wiring]: Top-level wiring.

## rt_detr/rt_detr_resnet
- **src**: modeling_rt_detr_resnet.py
- **status**: composable
- **rationale**: RT-DETR's HGNetv2/ResNet backbone reimplemented as a sibling modeling file. Same compute as standard ResNet (Conv2d + BatchNorm2d + ReLU + MaxPool + adaptive_avg_pool + linear).
- **classes**:
  - **`RTDetrResNetConvLayer`** [compute]: `L1/conv2d.py`, `L1/batch_norm2d.py`, `L1/relu.py` (Standard Conv + BN + activation.)
  - **`RTDetrResNetEmbeddings`** [compute]: `L1/max_pool2d.py` (Stem ConvLayers + MaxPool.)
  - **`RTDetrResNetShortCut`** [compute]: `L1/conv2d.py`, `L1/batch_norm2d.py` (1x1 conv + BN downsample.)
  - **`RTDetrResNetBasicLayer`** [wiring]: Wiring: 2 ConvLayers + shortcut + ReLU.
  - **`RTDetrResNetBottleNeckLayer`** [wiring]: Wiring: bottleneck.
  - **`RTDetrResNetStage`** [wiring]: Wiring.
  - **`RTDetrResNetEncoder`** [wiring]: Wiring.
  - **`RTDetrResNetBackbone`** [wiring]: Wiring.

## rt_detr_v2
- **src**: modular_rt_detr_v2.py
- **status**: kb_nano_l4
- **rationale**: L4/rtdetrv2.py docstring explicitly targets RT-DETR-V2 object detection; backbone, hybrid encoder, decoder, and v2-specific multi-scale deformable attention all live in kb-nano L1/L2/L3.
- **classes**:
  - **`RTDetrV2MultiscaleDeformableAttention`** [compute]: `L1/rtdetrv2_deformable_attention.py`, `L2/rtdetrv2_deformable_attention.py` (V2-specific deformable attention with n_points_scale, offset_scale, method dispatch ('default' / 'discrete'). Matches kb-nano MultiScaleDeformableAttentionV2 + RTDetrV2MultiscaleDeformableAttention.)
  - **`RTDetrV2DecoderLayer`** [wiring]: Wiring inherited from RTDetrDecoderLayer.
  - **`RTDetrV2Decoder`** [compute]: `L3/rtdetrv2_decoder.py` (Wiring inherited.)
  - **`RTDetrV2Model`** [compute]: `L3/rtdetrv2_model.py` (Wiring.)
  - **`RTDetrV2MLPPredictionHead`** [compute]: `L2/rtdetrv2_mlp_head.py` (MLP head.)
  - **`RTDetrV2ForObjectDetection`** [wiring]: Top-level wiring (matched by L4/rtdetrv2.py).

## rwkv
- **src**: modeling_rwkv.py
- **status**: unsupported
- **unsupported_reason**: RwkvLinearAttention is implemented via an external rwkv_cuda_kernel CUDA extension (forward / forward_with_state with bf16 variants). RWKV-v4 WKV recurrence is not equivalent to RWKV-7 (different time-decay formulation), so kb-nano's RWKV-7 kernels do not substitute. Adding RWKV-v4 would require a new fused recurrence kernel.
- **rationale**: RWKV (v1-v4) uses a custom CUDA kernel rwkv_cuda_kernel.forward / rwkv_cuda_kernel.forward_with_state for its WKV attention (cumulative time-decay). The kernel layout differs from RWKV-7 (which kb-nano covers via L1/chunk_rwkv7.py and L1/fused_recurrent_rwkv7.py). No kb-nano kernel implements the v4 WKV recurrence.
- **classes**:
  - **`RwkvLinearAttention`** [compute]: RwkvLinearAttention is implemented via an external rwkv_cuda_kernel CUDA extension (forward / forward_with_state with bf16 variants). RWKV-v4 WKV recurrence is not equivalent to RWKV-7 (different time
  - **`RwkvSelfAttention`** [wiring]: Wraps the WKV CUDA kernel with time_decay/time_first/time_mix parameters; no kb-nano composite.
  - **`RwkvFeedForward`** [compute]: `L1/linear.py` (Channel-mix (time_mix + key/value linear with squared ReLU activation); could compose from L1/linear.py + L1/squared_relu.py but the surrounding time-mix wiring has no L2 wrapper.)
  - **`RwkvBlock`** [wiring]: Wiring: norm + time-mix attention + norm + channel-mix ffn.
  - **`RwkvModel`** [wiring]: Wiring.
  - **`RwkvForCausalLM`** [wiring]: Top-level wiring.

## sam
- **src**: modeling_sam.py
- **status**: partial
- **partial_reason**: SamVisionAttention's decomposed 2D relative position embeddings (rel_pos_h / rel_pos_w with F.interpolate + einsum to add to attention scores) have no kb-nano L2 equivalent. The base attention compute (qk@k.T + softmax + p@v) is composable from DenseAttention but the rel-pos addition step is not wrapped.
- **rationale**: SAM (v1) has SamAttention (standard SDPA, composable via DenseAttention) and SamVisionAttention with decomposed relative position embeddings (rel_pos_h / rel_pos_w + interpolation + einsum to add to attention scores). The rel-pos decomposition uses pure torch (F.interpolate + einsum) but no kb-nano kernel wraps this MViTv2-style 2D rel-pos attention; SAM3 (kb-nano L4) uses different attention (RoPE + windowed). A SAM-v1 port would need to either reuse encoder_attention with a manual rel-pos add or build a new 2D-rel-pos attention L2 kernel.
- **classes**:
  - **`SamTwoWayAttentionBlock`** [compute]: SamVisionAttention's decomposed 2D relative position embeddings (rel_pos_h / rel_pos_w with F.interpolate + einsum to add to attention scores) have no kb-nano L2 equivalent. The base attention compute
  - **`SamPatchEmbeddings`** [compute]: `L1/conv2d.py` (Single Conv2d patch embedding.)
  - **`SamMLPBlock`** [compute]: `L1/linear.py`, `L1/gelu.py` (fc1 + GELU + fc2; standard 2-layer encoder MLP.)
  - **`SamLayerNorm`** [compute]: `L1/layer_norm.py`, `L1/layer_norm2d.py` (LayerNorm with channels_first/last support; kb-nano LayerNorm2d covers channels_first variant.)
  - **`SamAttention`** [compute]: `L1/dense_attention.py` (Standard MHA with downsample_rate (internal_dim = hidden_size / downsample_rate); composable via DenseAttention SDPA.)
  - **`SamTwoWayTransformer`** [wiring]: Wiring: stacks TwoWayAttentionBlocks for mask decoder.
  - **`SamFeedForward`** [compute]: `L1/linear.py` (Multi-layer MLP head (configurable layers).)
  - **`SamMaskDecoder`** [wiring]: Wiring: TwoWayTransformer + IoU/mask token + upsampling + hypernetworks.
  - **`SamPositionalEmbedding`** [wiring]: Positional encoding via random gaussian features; small custom op.
  - **`SamMaskEmbedding`** [compute]: `L1/conv2d.py` (Conv2d-based mask embedding.)
  - **`SamPromptEncoder`** [wiring]: Wiring: point/box/mask prompt embeddings.
  - **`SamVisionAttention`** [wiring]: Eager attention (qk^T/sqrt(d) + softmax + p@v) with optional decomposed 2D rel-pos embeddings (rel_pos_h, rel_pos_w via F.interpolate + einsum). The rel-pos add to attn_weights has no kb-nano kernel.
  - **`SamVisionSdpaAttention`** [wiring]: SDPA variant of the same; same rel-pos gap.
  - **`SamVisionLayer`** [wiring]: Wiring: norm + windowed attention + norm + MLP.
  - **`SamVisionNeck`** [compute]: `L1/conv2d.py`, `L1/layer_norm2d.py` (Conv2d + LayerNorm2d + Conv2d neck.)
  - **`SamVisionEncoder`** [wiring]: Wiring: patch embed + pos embed + vision blocks + neck.
  - **`SamVisionModel`** [wiring]: Wiring.
  - **`SamModel`** [wiring]: Top-level wiring (vision encoder + prompt encoder + mask decoder).
