# Reclassification of partial/unsupported folders — Shard B

Reviewed against the paper's looser definition: a folder is **composable** if
kb-nano has all the L1 compute primitives, even without an L2 wrapper.

## Reclassifications

nystromformer: unsupported — uses iterative Moore-Penrose pseudo-inverse on softmax kernels (`iterative_inv`), a bespoke compute primitive not present as an L1 op.
olmo_hybrid: composable — Llama-style attention + Qwen3-Next-style GatedDeltaNet linear attention; both compute primitives covered by `gdn_recurrence`/`chunk_gated_delta_rule` and `flash_attn_*`/`linear`/`rms_norm`.
ovis2: composable — vision tower is SigLIP-style (`siglip_*` L2 + `dense_attention`/`linear`/`gelu`); LM tower is via AutoModel and only routes to existing kb-nano text models.
pe_audio: unsupported — DAC audio codec uses Snake1d activation (`x + sin(alpha*x)^2/alpha`) which is not a kb-nano L1 op; no existing snake activation primitive.
pe_audio_video: composable — transformer encoder is plain ViT-style (linear/dense_attention/gelu/layer_norm); audio/video sub-encoders go through AutoModel.
pe_video: composable — plain ViT transformer encoder reuses existing primitives; vision sub-encoder via AutoModelForImageClassification.
phi4_multimodal: composable — text Phi3-style and vision SigLIP-style use existing primitives; audio relative-attention-bias is a learned table + add to scores (already representable as embedding + bias-add); depthwise Conv1d is `conv1d`.
pp_doclayout_v2: unsupported — LayoutLMv3-style reading-order encoder needs a 2D relative-attention-bias bucket lookup with its own bias table (T5/DeBERTa-style relative_attention_bias) plus GlobalPointer head; not currently a kb-nano primitive.
pp_doclayout_v3: composable — RT-DETR-style detector reuses `rtdetrv2_deformable_attention` L1 op + standard conv/attention primitives; mask FPN is plain conv2d.
prompt_depth_anything: composable — DPT-style depth head is conv2d/convtranspose2d/relu/interpolate, all available as L1 ops; ViT backbone reuses standard primitives.
qwen2_5_omni: unsupported — talker DiT + token2wav BigVGAN vocoder use components (Snake1d-like, anti-aliasing filters, weight-norm conv chains) without exact kb-nano primitives.
qwen3_omni_moe: unsupported — code2wav BigVGAN-style vocoder needs anti-aliased activation primitives not in kb-nano.
rwkv: unsupported — RWKV-1/4 wkv recurrence uses a custom CUDA kernel (`rwkv_cuda_kernel`) distinct from the RWKV-7 ops kb-nano has.
sam_hq: composable — windowed ViT-Det attention with relative-position embeddings plus SAM mask decoder; all primitives (linear/dense_attention/conv2d/layer_norm/gather) exist; relative-pos bias is an embedding lookup added to scores.
segformer: composable — efficient self-attention is just spatial-reduction (Conv2d/LayerNorm) before standard SDPA; mix-FFN is `linear`+`conv2d`+`gelu`; all primitives exist.
seggpt: composable — ViT with rel_pos einsum is an embedding-lookup + matmul (`bmm`) + add; all primitives exist.
smolvlm: composable — SigLIP-style vision via `siglip_*` L2 + Llama-family LM; no new compute primitive needed.
squeezebert: composable — BERT variant using grouped Conv1d in place of Linear; `conv1d` (with groups) + `layer_norm` + `gelu` + `dense_attention` all present.
xlnet: unsupported — two-stream attention with relative-shift positional encoding is a bespoke per-step recombination kernel not realizable from existing dense_attention without new shifting/gather logic.
xlstm: unsupported — matrix-LSTM (mLSTM) chunkwise/sequence kernels from external `xlstm.xlstm_large` package have no kb-nano analogue.
yoso: unsupported — custom CUDA `lsh_cumulation` and `fast_hash` kernels for LSH-based attention are not in kb-nano.
zamba: composable — Mamba1 selective-scan covered by kb-nano's `mamba_mixer` L2 (uses same `selective_scan_fn`); shared transformer block uses standard `dense_attention`/`linear`/`rms_norm`.
zamba2: composable — uses Mamba2 SSD chunked scan which maps to kb-nano's `mamba2_mixer` L2 + standard attention primitives; LoRA-style adapters are linear+linear add.
zoedepth: unsupported — log-binomial softmax + attractor layers + bin-centers post-processing is bespoke depth-estimation compute without kb-nano analogues.
autoformer: unsupported — AutoCorrelation attention via `torch.fft.rfft`/`irfft` requires FFT, not a kb-nano primitive.
efficientloftr: composable — RepVGG conv blocks + standard self/cross attention; all primitives (`conv2d`/`batch_norm2d`/`linear`/`dense_attention`) exist.
encodec: composable — uses `elu` (kb-nano has `L1/elu.py`) + `conv1d` + `conv_transpose1d` + LSTM (`L1/lstm.py` exists) for residual codec stacks; weight-norm is a parametrization, not a separate primitive.
fastspeech2_conformer: unsupported — conformer relative-shift attention with bias matrix-bd reshape is a bespoke compute pattern (per caveat list).
funnel: unsupported — pooling-based attention with relative-shift gather and factorized/unfactorized relative-pos schemes; uses bespoke `_relative_shift_gather` not in kb-nano.
omdet_turbo: composable — uses `MultiScaleDeformableAttention` which kb-nano has as `rtdetrv2_deformable_attention` L1; remaining components are standard Conv/RepVGG/MHA.
oneformer: composable — Mask2Former-style universal segmentation reuses `rtdetrv2_deformable_attention` for pixel decoder + standard cross-attention/MLP; all primitives exist.
openai: composable — legacy GPT-1 with Conv1D-based projections (a 1D linear-equivalent), gelu_new, LayerNorm; all primitives present.
opt: composable — standard pre-Llama decoder with learned positional embeddings + LayerNorm + ReLU/GELU + standard attention; all primitives exist.
owlv2: composable — OWL-ViT detector uses CLIP-style ViT (linear/dense_attention/quick_gelu/layer_norm) + simple box/class heads; all primitives exist.
owlvit: composable — same as owlv2: CLIP-style ViT + linear detection heads; all primitives exist.
parakeet: unsupported — Conformer relative-shift attention with `matrix_bd @ relative_k` shift logic is a bespoke kernel pattern (per caveat list); plus depthwise Conv1d is fine but the relative-shift is the blocker.
patchtsmixer: composable — TSMixer is just gated MLP + BatchNorm + simple attention block; all primitives (`linear`/`gelu`/`batch_norm2d`/`dense_attention`) exist.
patchtst: composable — patch-based time-series transformer with standard attention/linear/gelu/layer_norm; all primitives exist.
pegasus: composable — standard BART-style encoder-decoder with sinusoidal positional embeddings + cross-attention; all primitives exist (linear/dense_attention/gelu/layer_norm/embedding).
pegasus_x: unsupported — block-local + global-stagger attention has structurally distinct chunked compute that needs custom block reshape kernels not currently in kb-nano.
perceiver: composable — cross-attention + latent self-attention + standard MLPs; all primitives (linear/dense_attention/gelu/layer_norm) exist; multimodal preprocessors reuse conv2d/conv1d/embedding.
reformer: unsupported — LSH attention with bucketing/hashing/sort is a bespoke kernel (per caveat list).
seamless_m4t: unsupported — Conformer speech encoder with relative-shift attention + HiFi-GAN vocoder; relative-shift kernel and HiFi-GAN anti-aliased activation/conv stacks not in kb-nano.
seamless_m4t_v2: unsupported — extends seamless_m4t with additional conv-based T2U decoder; same conformer relative-shift blocker.
sew: composable — Wav2Vec2-style with squeeze-feature 1D conv stack + standard attention; all primitives (`conv1d`/`gelu`/`layer_norm`/`dense_attention`) exist.
sew_d: unsupported — DeBERTa disentangled-attention with c2p/p2c relative-position bias decomposition is a bespoke attention compute path not in kb-nano.
slanet: composable — PP-LCNet/CSP-PAN backbone uses standard conv2d/hardswish; AttentionGRUCell is decomposable into 3 linear + sigmoid/tanh + elementwise (all L1 ops); no new fused kernel needed.
slanext: composable — SAM-style ViT + AttentionGRUCell head; ViT uses standard attention/linear/gelu/layer_norm; GRUCell decomposes into existing element-wise/linear primitives.
speech_encoder_decoder: composable — generic wrapper combining arbitrary encoder/decoder; if its sub-models are individually composable, this wrapper is too (no new primitive of its own).
speech_to_text: composable — CNN subsampler (conv1d) + standard Transformer enc-dec with sinusoidal positional embedding + cross-attention; all primitives exist.
speecht5: unsupported — speech prenets/postnets and HiFi-GAN vocoder use anti-aliased upsampling + weight-norm conv stacks not currently captured by a kb-nano primitive.
superglue: composable — keypoint matching is just self/cross multi-head attention + MLPs + Sinkhorn iterations (matrix multiplications); all primitives exist.
superpoint: composable — VGG-style CNN with conv2d/relu/maxpool2d for keypoint detection; all primitives exist.
swiftformer: composable — depth-wise Conv2d + 1x1 Conv2d + LayerNorm + additive-attention (linear+softmax+linear); all primitives exist.
timm_backbone: unsupported — explicit delegation to external `timm` library; not realizable from kb-nano primitives without re-implementing each timm model.
timm_wrapper: unsupported — same `timm` delegation as timm_backbone.

## Summary

- Total reviewed: 57
- New status distribution:
  - composable: 34 (olmo_hybrid, ovis2, pe_audio_video, pe_video, phi4_multimodal, pp_doclayout_v3, prompt_depth_anything, sam_hq, segformer, seggpt, smolvlm, squeezebert, zamba, zamba2, efficientloftr, encodec, omdet_turbo, oneformer, openai, opt, owlv2, owlvit, patchtsmixer, patchtst, pegasus, perceiver, sew, slanet, slanext, speech_encoder_decoder, speech_to_text, superglue, superpoint, swiftformer)
  - partial: 0
  - unsupported: 23 (nystromformer, pe_audio, pp_doclayout_v2, qwen2_5_omni, qwen3_omni_moe, rwkv, xlnet, xlstm, yoso, zoedepth, autoformer, fastspeech2_conformer, funnel, parakeet, pegasus_x, reformer, seamless_m4t, seamless_m4t_v2, sew_d, speecht5, timm_backbone, timm_wrapper)

- Folders RECLASSIFIED (was partial/unsupported -> now composable, 34 total):
  olmo_hybrid, ovis2, pe_audio_video, pe_video, phi4_multimodal,
  pp_doclayout_v3, prompt_depth_anything, sam_hq, segformer, seggpt,
  smolvlm, squeezebert, zamba, zamba2, efficientloftr, encodec,
  omdet_turbo, oneformer, openai, opt, owlv2, owlvit, patchtsmixer,
  patchtst, pegasus, perceiver, sew, slanet, slanext,
  speech_encoder_decoder, speech_to_text, superglue, superpoint,
  swiftformer

- Folders KEPT THE SAME status (still unsupported, 23 total):
  nystromformer, pe_audio, pp_doclayout_v2, qwen2_5_omni, qwen3_omni_moe,
  rwkv, xlnet, xlstm, yoso, zoedepth, autoformer, fastspeech2_conformer,
  funnel, parakeet, pegasus_x, reformer, seamless_m4t, seamless_m4t_v2,
  sew_d, speecht5, timm_backbone, timm_wrapper
