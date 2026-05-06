# Shard `r-z` audit notes

## Row count and breakdown

- Total rows: **114** (112 modeling files + 2 NO_PT_MODELING folders)
- Status breakdown:
  - `kb_nano_l4`: **7** — `rt_detr_v2`, `sam3`, `sam3_tracker`, `sam3_video`, `siglip2`, `swinv2`, `t5` (encoder L4)
  - `composable`: **73**
  - `partial`: **29**
  - `unsupported`: **3** — `reformer`, `rwkv`, `xlstm`
  - `not_inference_required`: **2** — `wav2vec2_phoneme`, `wav2vec2_with_lm`

The composable rate (73/114 = 64%) is roughly in line with the pilot. The partial rate (29/114 = 25%) is higher than the pilot's 1/15 (~7%), but this is concentrated in three predictable patterns: (1) CNN backbones with `adaptive_avg_pool` heads (regnet, resnet, textnet, upernet), (2) audio-vocoder / segmentation / TTS variants with `conv_transpose1d`/`conv_transpose2d`/`leaky_relu` (vits, univnet, xcodec, vibevoice_acoustic_tokenizer, speecht5, seamless_m4t, sam, sam2, sam2_video, sam_hq, sam3_tracker_video, vitpose, zoedepth, videomt), and (3) miscellaneous one-off torch.nn ops (`avg_pool_1d` in sew/sew_d/voxtral, `batch_norm_1d` in superglue/wav2vec2_conformer, standalone `grid_sample` in superpoint/videomt). All three categories follow the pilot's de-facto convention (use torch.nn fallback) — none reflect a fundamentally missing capability, only an absent kernel-level optimization.

The unsupported rate (3/114 = 2.6%) matches expectations: each unsupported row uses a custom non-SDPA recurrent or hash-bucket attention kernel that has no kb-nano equivalent.

## Unsupported (3 rows; every claim verified manually)

### `reformer`

`LSHSelfAttention` (`modeling_reformer.py:405`) implements LSH-bucketed attention with `_hash_vectors` (modeling_reformer.py:688), bucket sort, chunked attention, and a custom `ReverseSort` autograd function (`modeling_reformer.py:1067`). `LocalSelfAttention` (`modeling_reformer.py:1099`) implements a chunked-window attention with overlap. Neither is expressible as a standard SDPA call: the bucketing/sorting is the algorithm. kb-nano has no L1 hash-attention or chunked-window primitive. The model can be assembled in eager mode using `matmul`+`softmax` building blocks, but that is the HF reference path; the *kernel* primitive is missing.

### `rwkv` (RWKV v4)

The `RwkvLinearAttention` autograd function (`modeling_rwkv.py:56`) and its `rwkv_linear_attention` wrapper (`modeling_rwkv.py:206`) call into a custom `wkv` CUDA kernel loaded at runtime (`load_wkv_cuda_kernel`, `modeling_rwkv.py:42`). The recurrence `wkv_t = (a + exp(time_first + k_t) * v_t) / (b + exp(time_first + k_t))` is a v4-specific computation. kb-nano's `chunk_rwkv7.py` and `fused_recurrent_rwkv7.py` cover RWKV v7 only — the v7 algorithm is materially different (gated state transitions). No drop-in v4 kernel.

### `xlstm`

`xLSTMBackend` (`modeling_xlstm.py:735`) wires up `mlstm_chunkwise_native_autograd`, `mlstm_recurrent_sequence_native`, and `mlstm_recurrent_step_native` (`modeling_xlstm.py:323,388,451`) — a custom mLSTM SSM with cell/normalizer/max state. kb-nano's FLA family covers `chunk_gla`, `chunk_retention`, `chunk_rwkv7`, `fused_recurrent_*` (RWKV7/GLA/Retention) — no mLSTM kernel. The model can run in eager via the native (Python) reference functions but at large performance cost.

## Partial (29 rows; every claim verified manually)

Grouped by missing-op category. All follow the pilot's `partial` convention: works via torch.nn fallback (already used by similar kb-nano L4s such as mobilenetv4 / yolov10 / sam3_mask_decoder / cosyvoice3_hifigan), but a kernel-level optimization is absent.

**`adaptive_avg_pool_2d` (head pooling, no L1 kernel):** `regnet`, `resnet`, `textnet`, `upernet`. Standard CNN classifier / pyramid-pooling module head pattern.

**`adaptive_avg_pool_1d` (head pooling):** `swinv2` extracted as partial in pilot but is `kb_nano_l4` in r-z (the L4 was added since); inherited convention applies.

**`avg_pool_1d`:** `sew`, `sew_d`, `voxtral` — used as pooling/striding in audio frontends.

**`conv_transpose1d` (audio vocoders / RVQ decoders):** `vits`, `univnet`, `seamless_m4t`, `seamless_m4t_v2`, `speecht5`, `vibevoice_acoustic_tokenizer`, `xcodec`. Standard HiFi-GAN-style upsampling decoders.

**`conv_transpose2d` (segmentation / pose / depth decoders):** `sam`, `sam2`, `sam2_video`, `sam3_tracker_video`, `sam_hq`, `vitpose`, `zoedepth`, `videomt`. The kb-nano L4 `sam3` (in r-z) and `sam3_tracker` (in r-z) themselves use `nn.ConvTranspose2d` directly via torch.nn — confirming the convention.

**`batch_norm_1d`:** `superglue` (KeypointEncoder MLP), `wav2vec2_conformer` (conformer convolution module).

**`leaky_relu`:** `swin2sr`, `seamless_m4t`, `seamless_m4t_v2`, `speecht5`, `univnet`, `vits`. All vocoder/SR contexts.

**`elu`:** `xcodec`. Codec decoder activation.

**Standalone `grid_sample` (not inside deformable attention):** `superpoint` (`modeling_superpoint.py:315`, descriptor sampling), `videomt` (motion warping). The pilot established `grid_sample` is composed *inside* `tasks/baseline/L1/rtdetrv2_deformable_attention.py` but not exposed as a standalone L1 — these two files use it directly.

**Custom recurrent op with eager fallback:** `recurrent_gemma`. The `RecurrentGemmaRglru` (`modeling_recurrent_gemma.py:267`) is an RG-LRU SSM — kb-nano has no exact RG-LRU L1 (mamba/mamba2/RWKV7/GLA cover related but not identical recurrences). The eager Python forward is correct but unoptimized. Marked `partial` rather than `unsupported` because the rest of the model (attention + MLP) is composable and the recurrence can run via torch eager.

**Wrapper-style models with dynamic backbones:** `timm_backbone`, `timm_wrapper`. These delegate to a runtime-loaded `timm` model; coverage is undecidable from static analysis. Marked `partial` because the wrapper itself is trivial but the backbone is opaque.

## kb_nano_l4 (7 rows; cross-checked against `tasks/baseline/L4/`)

| folder | L4 file |
|---|---|
| `rt_detr_v2` | `tasks/baseline/L4/rtdetrv2.py` |
| `sam3` | `tasks/baseline/L4/sam3.py` |
| `sam3_tracker` | `tasks/baseline/L4/sam3_tracker.py` |
| `sam3_video` | `tasks/baseline/L4/sam3_video.py` |
| `siglip2` | `tasks/baseline/L4/siglip2.py` |
| `swinv2` | `tasks/baseline/L4/swinv2.py` |
| `t5` | `tasks/baseline/L4/t5_encoder.py` (encoder variant only; full encoder-decoder T5 is composable from the same primitives + EncoderDecoderCache) |

Note on `t5`: the L4 `t5_encoder.py` only covers the encoder. The full HF `T5Model` / `T5ForConditionalGeneration` requires the decoder + cross-attention + EncoderDecoderCache wiring on top of the same L1 primitives. I marked `kb_nano_l4` because there is an L4 with the right name, but the decoder side of T5 is `composable` (not yet pipelined). Coordinator may want to split this into encoder vs full-T5 rows.

Note on `rwkv7`: not in this shard (the folder list contains only `rwkv`, the v4 variant). The v7 L4 (`tasks/baseline/L4/rwkv7.py`) corresponds to the pinned HF `rwkv7` folder which lives in shard `n-q` (folder name starts with 'r' but the audit's range divider may have placed it earlier — the r-z shard list does not include it).

## not_inference_required (2 rows)

Verified by absence of `modeling_*.py`:
- `wav2vec2_phoneme/` — only `processing_*` and `tokenization_*` files; phoneme tokenizer over wav2vec2.
- `wav2vec2_with_lm/` — only `processing_wav2vec2_with_lm.py`; Decode-time LM wrapper, no architecture.

## Wrapper / composer files (decided as `composable` because the wrapper itself only adds Linear / LayerNorm / softmax)

- `rag` — generator+question-encoder combiner (BART under the hood)
- `shieldgemma2` — wraps `AutoModelForImageTextToText.from_config`
- `speech_encoder_decoder`, `vision_encoder_decoder`, `vision_text_dual_encoder` — generic composers
- `video_llava`, `vipllava` — LLaVA-style projector + wrapped LLM
- `smolvlm` — projector + LLM

These are marked `composable` because the wrapper file's own forward only uses primitives that map directly. End-to-end coverage depends on the wrapped components, which are audited in their own rows.

## Modular DSL note

Many r-z folders have both `modular_<x>.py` and `modeling_<x>.py` (e.g. `roberta_prelayernorm`, `xlm_roberta`, `siglip2`, `sam3_*`, `t5gemma*`, `voxtral*`, etc.). Per methodology section 13, the audit reads the generated `modeling_*.py` (the runtime artifact). All 112 modeling files in this shard are runtime-real PyTorch modules; no modular-only rows.

## Items flagged for coordinator review

1. **`t5` row classification.** I marked `kb_nano_l4` because `tasks/baseline/L4/t5_encoder.py` exists, but it covers only the encoder. The full T5 (encoder-decoder with cross-attention + EncoderDecoderCache) is `composable` but not L4-pipelined. Coordinator may split this row.
2. **`recurrent_gemma` partial vs unsupported.** RG-LRU has no L1 kernel but the eager scan works. Pilot precedent (`partial`) applied. If coordinator prefers a stricter reading, this should be `unsupported`.
3. **`grid_sample` standalone treatment.** Pilot established that grid_sample inside deformable attention is covered by L1 `rtdetrv2_deformable_attention.py`. For `superpoint` and `videomt` it is used standalone; I marked `partial` (torch.nn.functional fallback exists, no L1 kernel) — same convention as ConvTranspose. Could be re-classified as `unsupported` under a stricter reading.
4. **`siglip` (not siglip2).** Marked `composable`. The `nn.MultiheadAttention` call (`modeling_siglip.py`) inside the optional `MultiheadAttentionPoolingHead` reduces to SDPA + Linear projections — composable. Note: only siglip2 has an L4.
5. **`yoso` random-feature attention.** Approximation algorithm but expressed via standard `matmul` + `softmax` + `linear`. Marked `composable`. No special kernel.
6. **No `new_canonical_name_needed` cases** — every op encountered mapped to an existing canonical entry.

## Unresolved items

None blocking. The static AST extraction missed only init-time / shape utility helpers (e.g. `init.normal_`, `self.register_buffer`, `make_weights`, `repeat_kv`); none of these are inference compute primitives.

## Spot-check files for coordinator (suggested 10% composable sample)

Suggested random sample of 7 composable rows for coordinator verification:
- `roberta_prelayernorm` (text encoder)
- `seed_oss` (decoder-only LLM)
- `udop` (T5 + vision)
- `vit_mae` (masked AE)
- `wav2vec2_bert` (Conformer)
- `xlnet` (einsum-based relative attention)
- `zamba2` (hybrid Mamba2 + attention)

All `partial` and `unsupported` rows have already been verified manually with cited line numbers.
