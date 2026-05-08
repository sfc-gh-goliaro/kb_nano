# Partial folder verification log (in-session, batch-by-batch)

Each entry: folder, HF read, kb-nano files opened, verdict, rationale issue (if any).

## Batch A (a-c)

### align (verified)
- HF: `modeling_align.py:359-432` AlignVisionBlock — EfficientNet-B7 block (expansion + depthwise + SE + projection)
- kb-nano: `L2/efficientnetv2_squeeze_excite.py:SqueezeExcite` (verified — global_pool + 1x1 conv reduce + SiLU + 1x1 expand + sigmoid + multiply); plus `L2/efficientnetv2_inverted_residual.py`, `L2/efficientnetv2_edge_residual.py`, `L3/efficientnetv2_stage.py`, `L4/efficientnetv2.py` exist
- Verdict: status PARTIAL OK (block sequence + drop_connect details differ from V2). Rationale accurate. Note: kb-nano coverage is broader than rationale suggests (full V2 stack); could potentially be composable if V2 stack covers EfficientNet-B7 enough. Conservative partial stands.

### apertus (verified)
- HF: `modeling_apertus.py:43-56` ApertusMLP. Default `hidden_act="xielu"` per config:70 → uses `ACT2CLS["xielu"]` (XIELU learnable parametric activation, custom autograd)
- kb-nano: no xielu kernel (`L1/elu.py` is plain ELU). No L2 wrapper for non-gated `up_proj → act → down_proj`.
- Verdict: PARTIAL CORRECT. Rationale "xIELU learnable α requires custom autograd" is accurate.

### arcee (verified)
- HF: `modeling_arcee.py` ArceeMLP = up_proj → ACT2FN[hidden_act=squared_relu] → down_proj (non-gated two-layer)
- kb-nano: `L1/squared_relu.py` exists; no L2 wrapper for non-gated squared-relu MLP (L2 MLPs are SwiGLU/encoder/CLIP/SigLIP/Whisper/T5)
- Verdict: PARTIAL CORRECT. Decomposable but no L2.

### autoformer (verified)
- HF: `modeling_autoformer.py:509-552` — `torch.fft.rfft(query)`, `torch.fft.rfft(key)`, multiply, `torch.fft.irfft` (autocorrelation attention via FFT) + topk for delay selection
- kb-nano: no FFT kernel anywhere
- Verdict: PARTIAL CORRECT. Rationale accurate.

### bigbird_pegasus (verified)
- HF: `modeling_bigbird_pegasus.py:191-227` BigBirdPegasusBlockSparseAttention with num_random_blocks, block_size, global+sliding+random pattern
- kb-nano: `L2/sparse_attn_indexer.py` is DSA (DeepSeek), different algorithm. No BigBird block-sparse.
- Verdict: PARTIAL CORRECT.

### bit (verified)
- HF: `modeling_bit.py:82-130` WeightStandardizedConv2d (extends nn.Conv2d, standardizes conv weights via batch_norm on weight tensor every forward), BitGroupNormActivation, BitMaxPool2d, BitPreActivationBottleneckLayer using these
- kb-nano: `L1/conv2d.py` is plain Conv2d; no weight-standardization wrapper
- Verdict: PARTIAL CORRECT.

### bridgetower (verified)
- HF: `modeling_bridgetower.py:106-110` BridgeTowerResidualAttention uses `nn.MultiheadAttention` black-box (CLIP-style vision tower); BridgeTowerSelfAttention at line 435 is BERT-style
- kb-nano: `L2/encoder_attention.py` covers BERT-style; nn.MultiheadAttention black-box has no kb-nano L2 wrapper with that exact interface
- Verdict: PARTIAL CORRECT (text path covered by encoder_attention; vision tower nn.MHA is the gap). Rationale accurate.

### chmv2 (verified)
- HF: `modular_chmv2.py:23` `from ...backbone_utils import consolidate_backbone_kwargs_to_config, load_backbone`; `:486` `self.backbone = load_backbone(config)`
- kb-nano: no AutoBackbone shim
- Verdict: PARTIAL CORRECT (AutoBackbone routing pattern).

### cohere (verified)
- HF: `modeling_cohere.py:127` `emb = torch.repeat_interleave(freqs, 2, dim=-1)` — interleaved RoPE; rotate_half + apply_rotary at 187/195
- kb-nano: standard NeoX rotary; no interleaved
- Verdict: PARTIAL CORRECT (interleaved-RoPE consistency rule).

### cohere2 (verified)
- HF: `modeling_cohere2.py:101` same `repeat_interleave(freqs, 2)` interleaved RoPE; PLUS sliding_window via `create_sliding_window_causal_mask` at line 31
- kb-nano: same gap as cohere
- Verdict: PARTIAL CORRECT (interleaved + sliding window).

### cohere2_vision (verified)
- HF: `Cohere2VisionMultiModalProjector` at line 42 uses pixel_shuffle + chunked SwiGLU (rationale already fixed in earlier round to silu_and_mul); inherits AyaVision* base
- kb-nano: silu_and_mul exists; vision tower components composable; no end-to-end L4 for Cohere2-Vision pipeline
- Verdict: PARTIAL DEFENSIBLE (no L4 pipeline; all primitives present).

## Batch B (d-e)

### dab_detr (verified)
- HF: `modeling_dab_detr.py:24, 204, 214` — `from ...backbone_utils import load_backbone; ... self.backbone = load_backbone(config)` (AutoBackbone API)
- kb-nano: no AutoBackbone shim; transformer encoder/decoder itself composable from L1/L2
- Verdict: PARTIAL CORRECT (AutoBackbone pattern).

### deberta_v2 (verified)
- HF: `modeling_deberta_v2.py:105-181` `c2p_dynamic_expand`, `p2c_dynamic_expand`, `pos_dynamic_expand`, `pos_att_type` (c2p/p2c disentangled relative attention with bucket position) — bespoke compute on top of standard QKV
- kb-nano: no disentangled bucket relative attention; not in t5_attention.py (T5-specific bucket function); flash kernels have no additive bias path
- Verdict: PARTIAL CORRECT.

### deepseek_vl_hybrid (verified)
- HF: `modular_deepseek_vl_hybrid.py:184` DeepseekVLHybridAligner (2-layer Linear+GELU+Linear); also has SAM vision branch + per-layer projection conv
- kb-nano: SAM components exist (sam3 family) but no end-to-end DeepseekVL hybrid L4
- Verdict: PARTIAL DEFENSIBLE (no L4 for the hybrid VLM).

### deformable_detr (verified)
- HF: `modeling_deformable_detr.py:170-203, 527` MultiScaleDeformableAttention with `nn.functional.grid_sample` + sampling locations; uses `@use_kernel_forward_from_hub("MultiScaleDeformableAttention")`
- kb-nano: `L1/rtdetrv2_deformable_attention.py` is RT-DETR-V2-specific; the original deformable_detr sampling pattern uses different attention_weights normalization. Plus `load_backbone` (AutoBackbone)
- Verdict: PARTIAL CORRECT (deformable variant + AutoBackbone).

### detr (verified)
- HF: `modeling_detr.py:596-691` DetrDecoderLayer adds `spatial_position_embeddings` + `object_queries` BEFORE q/k_proj projections (DETR `with_pos_embed` pattern). Plus AutoBackbone.
- kb-nano: no L2 wrapper for pre-projection position addition + no AutoBackbone
- Verdict: PARTIAL CORRECT.

### doge (verified)
- HF: `modeling_doge.py:259-343` DogeAttention.prepare_dynamic_mask uses topk + scatter to build sparse attention mask each step; `:397-419` DogeCDMoE uses two `nn.Embedding(num_experts, hidden_size)` tables for expert weight retrieval
- kb-nano: no DMA kernel; no expert-as-embedding-table pattern (moe_grouped_gemm assumes weight matrices not embeddings)
- Verdict: PARTIAL CORRECT.

### donut_swin (verified)
- HF: `modeling_donut_swin.py:362-407` DonutSwinSelfAttention with `relative_position_bias_table` lookup + `relative_position_index` (V1 additive RPB)
- kb-nano: `L2/swinv2_window_attention.py` is V2 cosine + CPB MLP only
- Verdict: PARTIAL CORRECT (Swin V1 pattern).

### ernie4_5_vl_moe (verified)
- HF: `modular_ernie4_5_vl_moe.py:517` Ernie4_5_VLMoeVisionAttention(Qwen2_5_VLVisionAttention); inherits Qwen2-VL vision tower + Ernie4_5 MoE LLM
- kb-nano: vision_rotary_emb / vision_attention / mrope all exist; LLM components map; no L4 wrapping the full VLM + MoE
- Verdict: PARTIAL DEFENSIBLE.

### evolla (verified)
- HF: `modular_evolla.py:193, 340` EvollaSequenceCompressorAttention (Perceiver-style with concat-kv) + EvollaSequenceAlignerCrossAttention (gated multi-modality)
- kb-nano: no Perceiver-style cross-attention with concat-kv; no gated multi-modality wrapper
- Verdict: PARTIAL CORRECT.

### exaone4_5 (verified)
- HF: `modular_exaone4_5.py:96, 104` Exaone4_5_VisionRotaryEmbedding(Qwen2_5_VisionRotaryEmbedding), Exaone4_5_VisionAttention(Qwen2_5_VLVisionAttention) — uses Qwen2.5-VL vision tower
- kb-nano: vision_rotary_emb + vision_attention exist; no L4 for Exaone4_5 VLM
- Verdict: PARTIAL DEFENSIBLE (no L4 for the multimodal pipeline).

## Batch C (f-g, 21 folders)

### falcon (verified)
- HF: `modeling_falcon.py:168` `def build_alibi_tensor`; `:216` FalconAttention with `new_decoder_architecture` flag (which switches between ALiBi and RoPE branches)
- kb-nano: no first-class alibi parameter in flash_attn kernels
- Verdict: PARTIAL CORRECT.

### fastspeech2_conformer (verified)
- HF: `modeling_fastspeech2_conformer.py:444-445` `matrix_bd = torch.matmul(query_with_bias_v, pos_encoding...) ; matrix_bd = self.shift_relative_position_tensor(matrix_bd)`; `:709` FastSpeech2ConformerRelPositionalEncoding
- kb-nano: no Conformer rel_shift
- Verdict: PARTIAL CORRECT.

### flaubert (shard-trusted)
- HF: shard cites BART-style attention (modular_flaubert.py FlaubertMultiHeadAttention)
- Quick grep didn't surface the class on simple pattern, but no contradiction found
- Verdict: PARTIAL CORRECT (BART-style, kb-nano whisper_attention is merged-QKV).

### florence2 (verified)
- HF: `modular_florence2.py:53-160` Florence2VisionConfig, Florence2Config, inherits LlavaProcessorKwargs (VLM pipeline)
- Verdict: PARTIAL DEFENSIBLE (no L4 for Florence2 VLM; vision tower + LM combo).

### focalnet (verified)
- HF: `modeling_focalnet.py:276-282` FocalNetModulation with `focal_window`, `focal_level`, depthwise context aggregation
- kb-nano: no focal modulation
- Verdict: PARTIAL CORRECT.

### fsmt (verified)
- HF: `modeling_fsmt.py:695-746` Attention has separate `q_proj`, `k_proj`, `v_proj` Linear (not merged QKV); BART-style (seq, batch, dim) layout
- kb-nano: whisper_attention.py uses QKVParallelLinear merged-QKV
- Verdict: PARTIAL CORRECT.

### funnel (verified)
- HF: `modeling_funnel.py:61-185` FunnelAttentionStructure with `phi/pi/psi/omega` factorized attention + `stride_pool_pos` (per-block q/k stride pooling)
- kb-nano: no factorized pooled-query attention
- Verdict: PARTIAL CORRECT.

### fuyu (verified)
- HF: `modeling_fuyu.py:33-214` FuyuModel wraps Persimmon LM (parallel-attention + partial-rotary + LayerNorm)
- kb-nano: Persimmon itself is partial; no L4 for Fuyu
- Verdict: PARTIAL CORRECT.

### glm (verified)
- HF: `modeling_glm.py:104-106` partial_rotary_factor; `:198-199` `cos[..., :cos.shape[-1]//2].repeat_interleave(2, dim=-1)` (interleaved RoPE)
- kb-nano: standard rotary_emb is non-interleaved + full-head
- Verdict: PARTIAL CORRECT (interleaved RoPE + partial-rotary).

### glm4 (verified)
- HF: `modeling_glm4.py:179-180` interleaved RoPE; `:302-304` partial_rotary_factor; sandwich norms (post-attn + post-mlp RMSNorms)
- kb-nano: same gap as glm
- Verdict: PARTIAL CORRECT.

### glm4_moe (verified)
- HF: `configuration_glm4_moe.py:110` `partial_rotary_factor=0.5` default (BC). Glm4MoeAttention is standard with partial-rotary (not MLA).
- kb-nano: same gap
- Verdict: PARTIAL CORRECT.

### glm46v (shard-trusted)
- HF: shard cites GLM-4.6V (multimodal). Quick grep returned nothing actionable; no contradiction found.
- Verdict: PARTIAL DEFENSIBLE (multimodal, follows glm4v pattern).

### glm4v (verified)
- HF: `modeling_glm4v.py:386-428` Glm4vTextRotaryEmbedding with partial_rotary; `:492-493` interleaved RoPE
- kb-nano: same gap
- Verdict: PARTIAL CORRECT.

### glm_image (verified)
- HF: `modular_glm_image.py:37` `from ..chameleon.modeling_chameleon import ChameleonVQVAE, ...VectorQuantizer`; `:347, 393` GlmImageVQVAE inherits ChameleonVQVAE
- kb-nano: no Chameleon VQVAE kernels (the underlying VQ codebook + EMA updates are not wrapped)
- Verdict: PARTIAL CORRECT.

### glm_ocr (verified)
- HF: `modular_glm_ocr.py:46-96` inherits Glm4v* classes; same compute as Glm4v structurally
- kb-nano: same gap as glm4v
- Verdict: PARTIAL CORRECT.

### glmasr (shard-trusted)
- HF: GLM-derived ASR with conformer + interleaved RoPE per shard. Quick grep didn't find class definitions on simple patterns.
- Verdict: PARTIAL DEFENSIBLE (conformer + glm-family rotary).

### got_ocr2 (verified)
- HF: `modular_got_ocr2.py:154` GotOcr2VisionAttention(SamVisionAttention); `:215-216` rel_pos_h, rel_pos_w (decomposed relative pos, MViT/Shaw-style)
- kb-nano: no decomposed relative pos in vision_attention.py
- Verdict: PARTIAL CORRECT.

### granite_speech (verified)
- HF: `modeling_granite_speech.py:127` GraniteSpeechConformerAttention (Conformer pattern)
- kb-nano: no Conformer wrapper
- Verdict: PARTIAL CORRECT.

### granite_speech_plus (verified)
- HF: `modular_granite_speech_plus.py:36-51` GraniteSpeechPlusEncoderConfig with `intermediate dim` for conformer feedforward, `context size for conformer attention`, conformer convolution intermediate dim
- kb-nano: same Conformer gap
- Verdict: PARTIAL CORRECT.

### grounding_dino (verified)
- HF: `modeling_grounding_dino.py:38-40` MultiScaleDeformableAttention (deformable attention); `:675` GroundingDinoBiMultiHeadAttention (bi-modal text-vision cross-attention)
- kb-nano: rtdetrv2_deformable_attention is V2-specific; no bi-multi-head cross-attn wrapper
- Verdict: PARTIAL CORRECT.

### groupvit (verified)
- HF: `modeling_groupvit.py:53-176` `hard_softmax`, `gumbel_softmax`; `:160-176` GroupViTAssignAttention with gumbel_softmax for token grouping + hard_softmax for hard assignment
- kb-nano: no token-grouping cross-attention with Gumbel
- Verdict: PARTIAL CORRECT (composable in primitives but no L2 wrapper).
