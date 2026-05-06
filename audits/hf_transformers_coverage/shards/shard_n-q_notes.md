# Shard `n-q` notes

## Row count and breakdown

- **Total rows:** 74 (one per modeling file / NO_PT_MODELING folder)
- **Folders covered:** 74 (matches `_folders_n-q.txt`)
- **Records in `_extract_n-q.jsonl`:** 70 (4 NO_PT_MODELING folders absent from extract; handled per methodology)

| status | count |
|---|---|
| `composable` | 52 |
| `partial` | 16 |
| `not_inference_required` | 4 |
| `kb_nano_l4` | 2 |
| `unsupported` | 0 |

`partial` rate: 16/74 = 21.6%, vs pilot 6.7%. The elevated rate is concentrated in two well-bounded buckets:
1. **PaddlePaddle CNN family (`pp_lcnet`, `pp_lcnet_v3`, `pp_ocrv5_*_det`, `prompt_depth_anything`):** uses `nn.AdaptiveAvgPool2d`, `nn.Hardsigmoid`, and/or `nn.ConvTranspose2d` in classifier/segmentation heads — exactly the gaps the pilot already documented for `data2vec_vision`.
2. **GatedDeltaNet hybrid LLMs (`olmo_hybrid`, `qwen3_5`, `qwen3_5_moe`, `qwen3_next`):** uses `fla.ops.gated_delta_rule.chunk_gated_delta_rule`, which has no kb-nano L1 wrapper today (the L1 FLA family covers GLA, retention, and RWKV7 but not gated_delta_rule).

The remaining `partial`s are time-series (`patchtst`, `patchtsmixer`, `parakeet`) using `nn.BatchNorm1d` (no L1 kernel; same convention as adaptive_avg_pool — torch.nn fallback acceptable), `pvt_v2` (uses `nn.AdaptiveAvgPool2d` for spatial reduction in attention), `perception_lm` (uses `F.adaptive_avg_pool2d` in projector — same gap as `data2vec_vision` from pilot), and the omni-modal vocoders `qwen2_5_omni` / `qwen3_omni_moe` (BigVGAN/Code2Wav use `nn.ConvTranspose1d`).

`kb_nano_l4` rows: `qwen2_5_vl` (matches existing L4 `qwen2_vl.py` + `qwen25_vl_encoder.py`), `qwen3_vl` (matches `qwen3_vl.py` L4).

`not_inference_required` (NO_PT_MODELING) rows: `nllb`, `nougat`, `phobert`, `pp_chart2table`. All four are tokenizer-only or modular wrappers re-using a parent architecture.

## Per-`partial` justification

Every `partial` row was verified manually by reading the cited HF source line.

1. **`olmo_hybrid`** (`modeling_olmo_hybrid.py:50,631,791`) — `OlmoHybridGatedDeltaNet` calls `chunk_gated_delta_rule` (FLA gated delta rule). kb-nano `tasks/baseline/L1/` has `chunk_gla.py`, `chunk_retention.py`, `chunk_rwkv7.py`, plus `fused_recurrent_*` siblings — but no `chunk_gated_delta_rule.py`. Causal_conv1d is covered via vLLM in `tasks/baseline/L2/mamba_mixer.py:35`. Gap: gated_delta_rule wrapper. *Fundamental* gap (a real kernel is missing), but small (one new L1 class wrapping `fla.ops.gated_delta_rule`).

2. **`parakeet`** (`modeling_parakeet.py:125,155,268`) — `ParakeetEncoderConvolutionModule` uses `nn.BatchNorm1d` in the conformer conv block. No L1 `BatchNorm1d`. *Cosmetic-ish*: project convention is to use torch.nn fallback for this kind of head/normalization (same as adaptive_avg_pool*). `ctc_loss` is training-only; `log_softmax` at inference is a torch builtin. `new_canonical_name_needed: batch_norm_1d`.

3. **`patchtsmixer`** (`modeling_patchtsmixer.py:66,692`) — `PatchTSMixer*` uses `nn.BatchNorm1d` as one of two configurable norms (the other is LayerNorm). Same gap as parakeet.

4. **`patchtst`** (`modeling_patchtst.py:160,583`) — same: `nn.BatchNorm1d` is a configurable norm. Falls back to torch.nn.

5. **`perception_lm`** (`modeling_perception_lm.py:37,50,56`) — `PerceptionLMAdaptiveAvgPooling` calls `F.adaptive_avg_pool2d` directly inside `MultiModalProjector`. Same gap as `data2vec_vision` (pilot row 10c). Treat as `partial` per pilot convention.

6. **`pp_lcnet`** (`modeling_pp_lcnet.py:120,125,174`) — `PPLCNetSqueezeExcitationModule` uses `nn.AdaptiveAvgPool2d(1)` and `nn.Hardsigmoid`. Both have no L1 kernel. `new_canonical_name_needed: hardsigmoid`. Same partial convention.

7. **`pp_lcnet_v3`** — derivative of pp_lcnet; same gap.

8. **`pp_ocrv5_mobile_det`** (`modeling_pp_ocrv5_mobile_det.py:182,237`) — DBNet text-detection head with `nn.ConvTranspose2d`. Per pilot: ConvTranspose* is a `partial` flag (no L1 kernel). Backbone is pp_lcnet-like → also has AdaptiveAvgPool2d.

9. **`pp_ocrv5_server_det`** (`modeling_pp_ocrv5_server_det.py:252,307`) — server-grade DBNet head; same `nn.ConvTranspose2d`. `nn.Upsample` composes from `F.interpolate` (passthrough — Interpolate L1).

10. **`prompt_depth_anything`** (`modeling_prompt_depth_anything.py:257`) — `PromptDepthAnythingReassembleLayer` uses `nn.ConvTranspose2d` to upsample DPT-style features. Load-bearing for depth output (not just a head). Partial.

11. **`pvt_v2`** (`modeling_pvt_v2.py:144`) — uses `nn.AdaptiveAvgPool2d(7)` to spatially reduce K/V before attention (one of pvt_v2's distinguishing tricks). Falls back to torch.nn; same convention as data2vec_vision.

12. **`qwen2_5_omni`** (`modeling_qwen2_5_omni.py:3298,3322`) — Token2Wav BigVGAN audio decoder uses `nn.ConvTranspose1d` for upsampling. Note: text+vision+audio paths are otherwise composable; only the vocoder is partial.

13. **`qwen3_5`** (`modeling_qwen3_5.py:61,358,631`) — same chunk_gated_delta_rule gap as olmo_hybrid; vision tower + decoder + GatedDeltaNet hybrid.

14. **`qwen3_5_moe`** (`modeling_qwen3_5_moe.py:62,359,632`) — MoE variant of qwen3_5; same gap.

15. **`qwen3_next`** (`modeling_qwen3_next.py:52,498,815`) — text-only Qwen3 with GatedDeltaNet + attention + MoE; same chunk_gated_delta_rule gap.

16. **`qwen3_omni_moe`** (`modeling_qwen3_omni_moe.py`) — Code2Wav vocoder uses `nn.ConvTranspose1d`. Otherwise composable (omni-modal: text+vision+audio+talker+MoE).

## NO_PT_MODELING / modular-only

| folder | reason |
|---|---|
| `nllb` | tokenizer-only; reuses parent NLLB architecture (BART-style) lives elsewhere in the original repo. |
| `nougat` | processor/tokenizer-only; reuses Donut/MBart architectures. |
| `phobert` | tokenizer-only; reuses RoBERTa. |
| `pp_chart2table` | per inventory: modular-only or no-modeling on this commit; reuses parent VLM. |

## `new_canonical_name_needed` flags

The following canonical op names are referenced by HF code in this shard but are NOT in `tools/canonical_to_kb_nano.csv`. Coordinator decision needed on whether to add formal entries:

| canonical name | example HF | kb-nano status |
|---|---|---|
| `batch_norm_1d` | `parakeet`, `patchtst`, `patchtsmixer` | no L1; same torch.nn convention as `batch_norm_2d` already in L1 |
| `hardsigmoid` | `pp_lcnet`, `pp_lcnet_v3` | no L1; trivial torch builtin elementwise; can mirror `sigmoid.py` |
| `multihead_attention` | `oneformer`, `phi4_multimodal` | no L1 wrapper, but composable from kb-nano `dense_attention.py` + `linear.py` |
| `chunk_gated_delta_rule` | `olmo_hybrid`, `qwen3_next`, `qwen3_5`, `qwen3_5_moe` | no L1 wrapper; FLA family in kb-nano covers GLA/retention/RWKV7 but not gated_delta_rule |
| `elu` | `owlvit`, `owlv2` | only used as logit-scale activation in head (`elu(x)+1`); not load-bearing |
| `unfold` | `phi4_multimodal` | torch builtin (rearranges memory); not a compute primitive |
| `upsample` | `pp_doclayout_v3`, `pp_ocrv5_server_det` | composes from `F.interpolate` → Interpolate L1 |
| `mamba_scan` | `nemotron_h` | mapped — `mamba_chunk_scan_combined` is provided by `tasks/baseline/L2/mamba2_mixer.py` (vLLM-backed) |

Recommend adding canonical entries for `batch_norm_1d` (`partial - no L1 kernel; torch.nn.BatchNorm1d fallback`), `hardsigmoid` (`unsupported on origin/experiments; trivial to add`), `chunk_gated_delta_rule` (`unsupported on origin/experiments; missing FLA wrapper`), `multihead_attention` (`composable from sdpa+linear`), and `elu` (`unsupported on origin/experiments; trivial`).

## Coverage of unresolved items

The extractor's `unresolved_top` field surfaced common HF helpers (`apply_rotary_pos_emb`, `repeat_kv`, `create_causal_mask`, `ALL_ATTENTION_FUNCTIONS.get_interface`, `eager_attention_forward`, `dynamic_rope_update`, `use_kernel_func_from_hub`, `use_kernelized_func`, `merge_with_config_defaults`, `maybe_autocast`, `capture_outputs`). These are dispatchers / annotations / mask builders, not compute primitives — they delegate to ops the extractor already canonicalizes. No new canonical_name needed.

## Multi-class / multi-variant flags

Several files contain multiple `*ForX` heads (e.g. `qwen2`, `phi`, `olmo`, all CausalLM + Sequence/Token/QA classification). Per pilot convention the row's `support_status` reflects the load-bearing decoder body; classifier heads are wiring (Linear + tanh/softmax) and do not change status. Where a single file contains multiple inference modes that diverge in primitive coverage, the most-restrictive status is reported and the variant is flagged in `notes`. No such case occurred in this shard (all multi-head files share the same primitive set across variants).

## Audit caveats

- **Modular DSL:** nearly every Qwen / Olmo / Phi / Nemotron file in this shard is generated from a `modular_*.py` source. The audit reads the generated `modeling_*.py` per methodology §13.
- **`paligemma`, `pi0`, `perception_lm`, `ovis2`, `qwen*_vl*`, `paddleocr_vl`, `qianfan_ocr`, `pp_formulanet`, `phi4_multimodal`, `qwen2_5_omni`, `qwen3_omni_moe`, `pix2struct`** delegate to sub-models via `AutoModel.from_config(...)`. Coverage of those sub-models depends on the underlying vision/text/audio backbone architecture (typically SigLIP / ViT / Llama / Qwen / Whisper / Ernie / Gemma / DiT). Per methodology §2 the audit is per-modeling-file; the sub-model coverage is captured in *its own* row. The wrapper file's status reflects the wrapper's own primitive use.
- **Deformable attention v1 vs v2:** `omdet_turbo`, `pp_doclayout_v2`, `pp_doclayout_v3`, `oneformer` use multi_scale_deformable_attention (v1 normalization). Per pilot finding 5 + 6, kb-nano's `MultiScaleDeformableAttentionV2` with `method="default"` is bit-identical to v1 — composable, not partial.
- **`encoder_decoder_cache`** appears in `nllb_moe`, `pegasus`, `pegasus_x`, `plbart`, `pix2struct`, `pop2piano`, `pp_formulanet`, `prophetnet`. Per pilot finding 1: composable from two stacked KV caches; not a missing primitive.
- **`F.unfold` (phi4_multimodal):** torch builtin tensor reshape; not a compute primitive.
