# kb-nano coverage of Hugging Face Transformers — summary

**HF source:** `huggingface/transformers` @ `da6c53e431f7c9ef0691239d4ce89b0f711ecad7`.
**kb-nano support surface:** `origin/experiments` @ `11aa838`.
**Audit:** static-analysis + manual review of HF modeling files vs kb-nano L1/L2/L3 operator surface.

## Inventory denominators

| denominator | count |
|---|---:|
| HF model folders under `models/` | 465 |
| folders with any PyTorch `modeling_*.py` | 442 |
| **distinct PyTorch modeling files (sum across folders)** — **headline denominator** | **448** |
| folders with no PyTorch modeling at all | 21 |
| folders with `modular_*.py` but no PyTorch modeling | 2 |

## Headline coverage (modeling-file denominator = 448)

| status | count | % of 448 |
|---|---:|---:|
| `kb_nano_l4` (already an L4 pipeline) | 17 | 3.8% |
| `composable` (existing L1/L2/L3 + wiring) | 422 | 94.2% |
| `partial` (one or more ops via torch.nn fallback) | 4 | 0.9% |
| `unsupported` (new primitive needed) | 4 | 0.9% |
| `not_inference_required` (no PyTorch modeling) | 24 | — |

**"Coverage"**, defined as `kb_nano_l4 + composable`, is **439 / 448 = 98.0%**.
**"Coverage including partial"** is **443 / 448 = 98.9%**.

## Coverage by modality

| modality | kb_nano_l4 | composable | partial | unsupported | not_inference_required | total |
|---|---:|---:|---:|---:|---:|---:|
| audio | 1 | 42 | 0 | 0 | 2 | 45 |
| audio+text | 0 | 7 | 0 | 0 | 0 | 7 |
| detection | 1 | 19 | 0 | 0 | 0 | 20 |
| multimodal | 7 | 95 | 0 | 0 | 2 | 104 |
| none | 0 | 0 | 0 | 0 | 9 | 9 |
| other | 0 | 4 | 0 | 0 | 1 | 5 |
| robotics | 0 | 1 | 0 | 0 | 0 | 1 |
| segmentation | 0 | 8 | 0 | 0 | 0 | 8 |
| structure | 0 | 1 | 0 | 0 | 0 | 1 |
| text | 6 | 164 | 1 | 4 | 6 | 181 |
| text+layout | 0 | 2 | 0 | 0 | 1 | 3 |
| text+layout+vision | 0 | 1 | 1 | 0 | 0 | 2 |
| text+structure | 0 | 1 | 0 | 0 | 0 | 1 |
| time-series | 0 | 5 | 0 | 0 | 0 | 5 |
| unknown | 0 | 0 | 0 | 0 | 3 | 3 |
| vision | 2 | 72 | 2 | 0 | 0 | 76 |

## Top missing/partial primitives (frequency table)

| canonical op | frequency | example HF files |
|---|---:|---|
| `timm_dynamic_backbone` | 2 | timm_backbone/modeling_timm_backbone.py;timm_wrapper/modeling_timm_wrapper.py |
| `detectron2_backbone` | 1 | layoutlmv2/modeling_layoutlmv2.py |
| `mra_sparse_kernels` | 1 | mra/modeling_mra.py |
| `rg_lru_scan` | 1 | recurrent_gemma/modeling_recurrent_gemma.py |
| `lsh_self_attention` | 1 | reformer/modeling_reformer.py |
| `local_self_attention` | 1 | reformer/modeling_reformer.py |
| `wkv_linear_attention_v4` | 1 | rwkv/modeling_rwkv.py |
| `mlstm_chunkwise_kernel` | 1 | xlstm/modeling_xlstm.py |
| `mlstm_recurrent_sequence` | 1 | xlstm/modeling_xlstm.py |
| `mlstm_recurrent_step` | 1 | xlstm/modeling_xlstm.py |
