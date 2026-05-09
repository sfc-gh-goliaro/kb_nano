# Paper-Quality HF Transformers x kb-nano Re-Audit Methodology

## Scope

Primary target: the final audit universe implied by the pinned HF source tree at:

- `/tmp/hf_transformers_pinned/src/transformers/models`

The row universe is derived from actual `modeling_*.py` files, excluding `auto/modeling_auto.py`. Prior audit artifacts are comparison targets, not sources of truth.

## Evidence Standard

A row is not verified unless the audit records:

- HF source files and line ranges opened.
- kb-nano source files and line ranges opened.
- The load-bearing compute path: `__init__`, `forward`, helper functions, imported parent classes, decorators, and config branches needed for inference compute.
- A status decision with confidence.
- Any ambiguity rule applied.

`rg`/glob may locate files or patterns, but cannot by itself verify a status.

## Status Definitions

### `kb_nano_l4`

Use only when a kb-nano `L4` file targets the same HF model family or an explicitly equivalent runtime surface, and the L4 path covers the folder's load-bearing inference path. Scoped L4s must be labeled as scoped, not promoted to full-folder L4.

### `composable`

Use when every load-bearing compute class in the HF row maps to existing kb-nano L1/L2/L3 primitives or wrappers with compatible semantics:

- same activation and MLP topology,
- same norm semantics,
- compatible attention layout and bias path,
- compatible RoPE variant,
- compatible cache/state behavior,
- compatible projection layout or a documented weight-packing transform,
- no missing required routing/wrapper logic.

### `partial`

Use when the model is implementable with PyTorch/tensor primitives but kb-nano lacks a matching primitive/wrapper for at least one load-bearing compute pattern.

### `unsupported`

Use when the active/default inference path requires a hard external runtime dependency, external CUDA package, or non-torch custom kernel not represented in kb-nano. If an external dependency is optional and a torch fallback exists, prefer `partial` unless the default path hard-requires the dependency.

## Fixed Ambiguity Rules

### R1: Additive attention-bias generators

Learned RPB, ALiBi, decomposed 2D relative position, DeBERTa c2p/p2c bias, LayoutLMv3 2D bias, and similar generated score-bias paths are `partial` unless kb-nano has a matching wrapper for the generator and placement. A generic `attn_mask` parameter is not sufficient by itself.

### R2: AutoBackbone / unconstrained AutoModel

`load_backbone()` or unconstrained `AutoModel.from_config()` in the active row is `partial` unless the row is explicitly scoped to a concrete child config already covered. If the default child is `timm`, `detectron2`, `natten`, `xlstm`, or another hard external dependency, classify as `unsupported`.

### R3: `weight_norm`

Active `nn.utils.weight_norm` or `torch.nn.utils.parametrizations.weight_norm` is `partial` unless kb-nano has a weight-norm-aware wrapper or documented loader fold preserving inference semantics.

### R4: MLP topology and activation identity

SwiGLU, GeGLU, non-gated GELU, non-gated squared-ReLU, gated squared-ReLU, QuickGELU, Snake, xIELU, and tanh-approx GELU are not interchangeable. A row is `partial` if only lower-level activations exist but no matching wrapper exists and the methodology requires wrapper-level parity.

### R5: RoPE variants

Standard NeoX full-head RoPE, interleaved RoPE, partial q/pass RoPE, M-RoPE, vision 2D RoPE, YaRN, DINOv3 RoPE, and Gemma4 proportional RoPE are distinct. Config-only partial-RoPE evidence is not enough; runtime must actually rotate fewer than `head_dim` channels.

### R6: Cache/state semantics

Paged KV cache, encoder-decoder cross cache, recurrent state, streaming Conv1d padding cache, and codec state caches are distinct. Plain Conv/attention kernels do not imply cache-wrapper coverage.

### R7: External dependency rule

Hard default dependency on `timm`, `detectron2`, `natten`, `xlstm`, `kernels-community/*`, or equivalent external runtime means `unsupported`, unless a torch fallback is active by default. `torchaudio` is to be logged as a human-judgment point and classified both ways in headline sensitivity if encountered.

### R8: Wrapper vs primitive policy

When a folder is mathematically decomposable from L1 ops but no kb-nano L2/L3 wrapper exists for the actual family interface, classify as `partial` under strict paper-quality rules. If a looser primitive-only headline is desired, report it separately as a sensitivity analysis.

## Required Outputs

- `paper_quality_reaudit_report.md`
- `paper_quality_reaudit_evidence.json`
- `paper_quality_reaudit_rules.md`
- headline counts with sensitivity cases
- explicit list of unchecked or low-confidence rows

