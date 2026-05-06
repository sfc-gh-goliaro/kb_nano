# Shard `j-m` audit notes

## Counts

- **Folders in shard**: 78 (per `_folders_j-m.txt`)
- **Modeling files in shard**: 74 (one folder, `maskformer`, has 2 modeling files; 5 folders are `NO_PT_MODELING`)
- **Rows written**: 79 (74 modeling rows + 5 `not_inference_required` placeholders)

## Status breakdown

| status | count |
|---|---|
| `kb_nano_l4` | 3 |
| `composable` | 60 |
| `partial` | 10 |
| `unsupported` | 1 |
| `not_inference_required` | 5 |
| **total** | **79** |

`partial` rate = 10/74 modeling rows = 13.5% (within pilot tolerance — pilot was 1/15 ≈ 6.7%, but this shard is heavy on vision-CNN families that all share the same `AdaptiveAvgPool2d` head fallback).

`unsupported` rate = 1/74 = 1.4% (mra only).

## `kb_nano_l4` rows

- **llama4** → `tasks/baseline/L4/llama4.py` exists.
- **mamba2** → `tasks/baseline/L4/mamba2.py` exists.
- **mixtral** → `tasks/baseline/L4/mixtral.py` exists.

## `partial` rows (each verified by reading the cited HF lines)

### lasr (BatchNorm1d in conv module)
LASR is a Conformer-style speech encoder. The convolution module instantiates `nn.BatchNorm1d` at `modeling_lasr.py:299` for normalization between depthwise/pointwise conv layers. There is no kb-nano L1 kernel for `BatchNorm1d` (only `BatchNorm2d` exists at `tasks/baseline/L1/batch_norm2d.py`). Inference path is affected on every layer's conv module. Standard convention (per pilot for ConvTranspose/AdaptiveAvgPool) is to flag `partial` since torch fallback works. CTC loss is training-only.

### layoutlmv2 (AdaptiveAvgPool2d head + detectron2 backbone)
The visual feature extractor depends on `detectron2` (META_ARCH_REGISTRY). The pooled feature head at `modeling_layoutlmv2.py:508` uses `nn.AdaptiveAvgPool2d` (or `nn.AvgPool2d` at line 501 when deterministic algorithms are enabled). Two concerns are flagged: (a) `adaptive_avg_pool_2d` has no L1 kernel (consistent with mobilenetv4 L4 convention), and (b) the detectron2 ResNet visual backbone is outside the kb-nano kernel surface entirely. The text encoder portion is composable.

### levit (BatchNorm1d + Hardswish)
LeViT uses `nn.BatchNorm1d` in MLP-style heads (lines 129, 462) and `nn.Hardswish` activations throughout (lines 92/97/102/162/239/309). Neither has a kb-nano L1 class. The convolutional stem uses `nn.BatchNorm2d` (covered) and `nn.Conv2d` (covered).

### lw_detr (ConvTranspose2d in upsampling neck)
LwDetr's detection neck adds `nn.ConvTranspose2d` for resolution upsampling at `modeling_lw_detr.py:569` and `:571`. No L1 kernel for `conv_transpose2d`. The ViT backbone, deformable-attention decoder, and other ops are composable. Same convention as `data2vec_vision` segmentation head from the pilot. The extracted op set also includes `grid_sample`; this is composed inside `tasks/baseline/L1/rtdetrv2_deformable_attention.py` so its use here (inside deformable attention) is covered, but a standalone `grid_sample` call would not be.

### mask2former (nn.MultiheadAttention + deformable v1)
Two distinct partial flags:
1. `nn.MultiheadAttention` is used directly at `modeling_mask2former.py:1585` (called at line 1611 and again later in the decoder). kb-nano has no wrapper for the bare `nn.MultiheadAttention` call signature; users would need to reimplement the head as `Linear(QKV) + DenseAttention + Linear(O)`.
2. `multi_scale_deformable_attention` at `modeling_mask2former.py:2013` (and elsewhere in the pixel decoder) is the v1 normalization variant, while kb-nano has v2 in `tasks/baseline/L1/rtdetrv2_deformable_attention.py`. Same numerical-drift caveat as the deformable_detr pilot row.

### maskformer (modeling_maskformer_swin.py — AdaptiveAvgPool1d in pooler)
The Swin backbone variant for MaskFormer ends in an `nn.AdaptiveAvgPool1d` pooler at `modeling_maskformer_swin.py:729`. Per the pilot's swin row, this is a head-only fallback — classified `composable` (NOT in the partial-row paragraph here; just noting). The inner attention/conv ops are all covered.

### mimi (ConvTranspose1d + ELU)
Mimi is a neural audio codec. The decoder upsamples via `nn.ConvTranspose1d` (wrapped as `MimiConvTranspose1d` at `modeling_mimi.py:354`, with the actual `nn.ConvTranspose1d` at line 370). Decoder blocks also use `nn.ELU` activations throughout (lines 428, 473, 478, 1155, 1165). Neither has a kb-nano L1 class. The middle transformer is composable (rotary, sdpa, kv_cache).

### mobilenet_v1 (AdaptiveAvgPool2d in pooler)
Pure depthwise-separable CNN classifier. The pooler is `nn.AdaptiveAvgPool2d` at `modeling_mobilenet_v1.py:184`. No L1 kernel. Same convention as mobilenetv4 L4 (which itself uses `nn.AdaptiveAvgPool2d` directly).

### mobilenet_v2 (AdaptiveAvgPool2d in pooler + DeepLab head)
Inverted-residual CNN. `nn.AdaptiveAvgPool2d` at line 320 (pooler) and line 437 (DeepLabV3+ ASPP module global pool for the segmentation head). No L1 kernel.

### mobilevit (AdaptiveAvgPool2d in seg head)
Hybrid CNN+ViT. `nn.AdaptiveAvgPool2d(output_size=1)` at `modeling_mobilevit.py:755` is used in the segmentation head's global pooler. No L1 kernel.

### mobilevitv2 (AdaptiveAvgPool2d in seg head)
Same as mobilevit. `nn.AdaptiveAvgPool2d` at line 732. The patch unfolding and folding use `nn.functional.unfold/fold` (both torch builtins, passthrough — not flagged).

## `unsupported` rows (verified)

### mra (custom CUDA kernel from kernels-community/mra)
MRA's sparse self-attention requires the `kernels-community/mra` HuggingFace Hub kernel, loaded at `modeling_mra.py:57` via `integrations.hub_kernels.get_kernel`. Three calls into this custom kernel are load-bearing on the inference path:
- `mra_cuda_kernel.index_max(...)` at line 82 (used for sparse softmax stability)
- `mra_cuda_kernel.mm_to_sparse(...)` at line 148 (block-sparse Q·Kᵀ)
- `mra_cuda_kernel.sparse_dense_mm(...)` at line 186 (sparse times dense GEMM)

These are not torch builtins and have no kb-nano coverage. Building MRA in kb-nano would require porting these three kernels (block-sparse top-K + sparse-dense matmul). Flagging `new_canonical_name_needed: mra_sparse_attention` (or three sub-ops).

## `not_inference_required` rows

- **layoutxlm**: only `modular_layoutxlm.py` + tokenizer/processor — no `modeling_*.py`. Wraps LayoutLMv2 architecture at runtime (which is itself `partial`).
- **mbart50**: only `tokenization_mbart50.py`. Wraps MBart at inference.
- **megatron_gpt2**: only checkpoint-conversion scripts. Wraps GPT2 at inference.
- **mluke**: only tokenization + checkpoint conversion. Wraps Luke at inference.
- **myt5**: only tokenization + checkpoint conversion. Wraps MT5 at inference.

## Multimodal-wrapper rows (composable, with caveats)

A large class of "multimodal wrapper" files in this shard delegate vision and language to sub-models loaded via `AutoModel.from_config` and only contribute a small projection layer themselves. These are uniformly composable:

- **lfm2_vl** (line 150/153 — vision + lfm2)
- **lighton_ocr** (modular DSL, vision + LLM via auto)
- **llava** (line 133/136 — CLIP vision + Llama text)
- **llava_next** (line 256/263)
- **llava_next_video** (line 311/318 — adds AvgPool2d/MaxPool2d for frame pooling, both covered)
- **llava_onevision** (line 271/278)
- **mistral3** (modular DSL, vision + Mistral text)
- **modernvbert** (line 212/216 — SigLIP-like vision + ModernBERT text)
- **musicflamingo** (audio encoder + LLM)
- **musicgen** (text encoder + audio encoder + decoder LM)
- **musicgen_melody** (same shape as musicgen)
- **lfm2_vl** etc.

Coverage in each case is "wrapper itself is composable; sub-model coverage depends on the chosen vision/text/audio model." This is the convention established by the pilot (e.g. qwen2_vl is `kb_nano_l4` because its sub-models are also covered).

## EncoderDecoderCache rows

Rows that import `EncoderDecoderCache`: kosmos2, led, longt5, m2m_100, marian, mbart, megatron_bert, moonshine, moonshine_streaming, mt5, musicgen, musicgen_melody, mvp. Per pilot finding 1, this composes from kb-nano's existing paged KV cache — `composable`.

## einsum-using rows

`einsum` is a torch builtin (lowers to matmul/permute). Used in:
- janus (line 527, VQVAE embedding distance computation)
- led (lines 428/510/561, sliding-window attention)
- longformer (lines 783/865/916, sliding-window/global attention)
- longt5 (lines 454/624/849, T5 attention scores)
- mask2former (line 2013, mask logits computation)
- maskformer (lines 1895/1911, mask logits)
- mgp_str (line 274, character grouping)

None of these rows are `partial` because of einsum — it's covered by the passthrough rule.

## `act2fn_dynamic` and `attention_dispatcher`

Most rows have `act2fn_dynamic` (config-selected activation). Per pilot decision 6, kb-nano covers gelu/silu/relu/tanh/quick_gelu/squared_relu/sigmoid/hardswish*/elu*/log_sigmoid/mish*. Asterisked entries are missing — **hardswish** and **elu** caused the levit and mimi `partial` rows respectively. Other models with `act2fn_dynamic` resolve to one of the supported activations in their canonical configs.

`attention_dispatcher` (`ALL_ATTENTION_FUNCTIONS`) — kb-nano supports sdpa+flash+eager, which suffices unless a model is flex-only. None of the models in this shard are flex-only.

## `new_canonical_name_needed` flags

- `mra_sparse_attention` (or its three sub-ops `mra_index_max`, `mra_mm_to_sparse`, `mra_sparse_dense_mm`) — the only `unsupported` row in the shard.
- (informational) `batch_norm_1d` — already exists in the AST extractor as `batch_norm_1d`, but no entry in `tools/canonical_to_kb_nano.csv`. Adding a row would help (it's used by lasr and levit). Behaves like BatchNorm2d but for 1D inputs.
- (informational) `hardswish` — already in the AST extractor as `hardswish`, but no entry in `tools/canonical_to_kb_nano.csv`. Used by levit (and probably others outside this shard via mobilenetv3-style stems).
- (informational) `elu` — used by mimi. Same status: AST extractor recognizes it; no canonical-map entry.
- (informational) `mamba_scan` — used by mamba2. Same kind: extractor flags it; map currently has `selective_scan` entry that covers Mamba1, and Mamba2 chunked scan is in `tasks/baseline/L2/mamba2_mixer.py`.

## Unresolved / ambiguity items

- **lw_detr `grid_sample`**: I marked it as a partial-related concern in the row, but the standalone usage location was not pinpointed inside the file. The file uses both deformable attention (where grid_sample is composed inside the kb-nano L1) and likely a separate use elsewhere. If standalone use is purely in deformable attention, this `partial` flag is too strong — but the conv_transpose2d flag is solid.
- **mask2former v1 deformable attention**: The "v1 vs v2" ambiguity is the same as in the pilot's `deformable_detr` row. The pilot resolved deformable_detr to `composable` after a side-by-side check of normalization math; mask2former's pixel-decoder deformable attention should be inspected with the same method to either downgrade my `partial` to `composable` or confirm `partial`.
- **lasr glu**: `nn.functional.glu` (gated linear unit) is used inside the convolution module (referenced in `unresolved_top`). It is composable from `chunk + sigmoid + multiply`, which are all torch builtins, so no kernel-level concern. Did not affect classification.
- **layoutlmv2 detectron2**: I included `detectron2_backbone` as a partial flag because the visual feature extractor is a non-pytorch-built-in dependency. If the audit's scope is "compute primitives only" then the detectron2 dependency is out-of-scope and the row is purely partial-due-to-AdaptiveAvgPool2d. Coordinator may want to demote that note.

## Sub-variant notes

- **layoutlmv2**: `LayoutLMv2Model` (no visual backbone needed if visual features are pre-extracted) is composable; `LayoutLMv2ForXxx` heads that go through the detectron2 backbone are `partial`.
- **mobilenet_v2**: `ForImageClassification` and `ForSemanticSegmentation` both partial (same AdaptiveAvgPool2d issue).
- **mobilevit / mobilevitv2**: `ForImageClassification` is composable (no AdaptiveAvgPool2d in classifier head — uses `mean(dim=...)`); `ForSemanticSegmentation` is partial. Lines cited are in the seg head, so the row-level status is `partial`. (The existing mobilenetv4 L4 in kb-nano uses `nn.AdaptiveAvgPool2d` even in classifier head, so the convention holds.)
- **mask2former**: both heads use the bare `nn.MultiheadAttention` and pixel-decoder deformable attention; both partial. No sub-variant difference.
- **mimi**: only one architecture class (`MimiModel`), so the partial flag covers everything.

## Modular DSL rows

Multiple rows in this shard come from generated `modeling_*.py` files (the project's modular DSL). Each was audited from the generated file as the runtime artifact, per methodology section 13. Notable examples:
- jais2, lighton_ocr, mistral3, ministral3, mistral4, lfm2, lfm2_moe, lfm2_vl, modernvbert, kyutai_speech_to_text — all generated from `modular_*.py`. The classes and ops audited match the generated file.

## Quality checklist

- [x] Every `partial` row has a verified `file:line` reference into the pinned HF commit.
- [x] Every `unsupported` row has a manually verified custom-kernel call site with three concrete `file:line` references.
- [x] No row cites a kb-nano `file:line` that is not on `audit/hf-transformers-coverage`.
- [x] `kb_nano_l4` rows (3 total) are confirmed by `ls tasks/baseline/L4/`.
- [x] All 5 `NO_PT_MODELING` folders verified by `ls` of the folder.
