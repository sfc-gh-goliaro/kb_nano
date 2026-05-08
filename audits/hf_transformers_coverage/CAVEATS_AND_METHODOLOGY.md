# Caveats, methodology, and what the numbers mean

> **TL;DR.** The headline numbers (27 / 237 / 171 / 12 = 447; strict 59.1%, loose
> 97.3%, unsupported 2.7%) are robust under the rules described below.
> Individual cells in the appendix table — specifically the kb-nano kernel
> column — were verified against a fixed methodology. v12 (the round before
> this writeup) closed an inconsistency: 6 folders that had been left
> `composable` despite using `partial_rotary_factor < 1.0` were demoted to
> `partial`, matching how phi/persimmon/glm/etc. were already classified.
> See §3 "Post-v11 fixes" below.

This document captures (a) the methodology used for the re-audit, (b) the
distinctions that the audit *did* and *did not* enforce strictly, and (c) every
known caveat that a reviewer should be aware of before citing the numbers.
It is intended to be the single self-contained methodology answer for the
paper appendix.

---

## 1. Methodology

### 1.1 Definitions (locked at the start of the re-audit)

For each HF folder containing at least one PyTorch `modeling_*.py` (excluding
`auto/`):

- **`kb_nano_l4`** — there is a kb-nano `tasks/baseline/L4/<file>.py` whose
  docstring header explicitly targets the same model family. Verified by
  reading the L4 file's first 30 lines.
- **`composable`** — every compute class in the HF folder maps to an existing
  kb-nano L1/L2/L3 kernel that implements the same compute pattern. "Same
  pattern" means: same activation, same norm variant, same projection layout,
  same attention type, same op sequence. Verified by opening the kb-nano file
  and confirming its `__init__` + `forward` match the HF class.
- **`partial`** — at least one compute class needs a torch op (e.g.,
  `torch.nn.LayerNorm` in a decoder, `torch.fft`, partial-rotary slicing,
  ALiBi-as-attn-bias, BatchNorm1d, Conformer rel_shift) that exists in
  `torch.*` but does not have a kb-nano kernel. The model can run via
  PyTorch fallback today; a tuned kernel would move it to composable.
- **`unsupported`** — at least one compute class needs (a) a custom CUDA
  kernel from `kernels-community/*`, (b) an external library (timm,
  natten, detectron2, xlstm), or (c) a genuinely novel compute primitive
  (custom autograd Function with non-trivial math) that has no kb-nano
  *or* `torch.*` equivalent.

### 1.2 Source-of-truth pinning

- HF source: `huggingface/transformers` @ commit
  `da6c53e431f7c9ef0691239d4ce89b0f711ecad7`, shallow-cloned to
  `/tmp/hf_transformers_pinned/`. All HF `file:line` references in the audit
  point into this checkout.
- kb-nano: branch `audit/hf-transformers-coverage` cut from `origin/experiments`
  @ `11aa838`. All kb-nano `file:line` references point into this branch's
  working tree.

### 1.3 Audit pipeline

1. **First pass (16 parallel sub-agents).** Each sub-agent received the same
   prompt (`tools/SUBAGENT_PROMPT.md`) covering the four-status definitions,
   the 12 kernel-mapping rules (silu-vs-silu_and_mul, RoPE variants, attention
   backends, norm variants, etc.), and a worked example. 425 folders audited.
2. **Cross-verify round 1 (5 verifiers).** Independent agents re-checked
   239 folders (priority on `partial` and `unsupported` rows, plus randomly
   sampled `composable`).
3. **Phase-2 deep source reads (slices 1–6, 8).** Re-read full HF + kb-nano
   source for 222 folders where the first-pass rationale was thin or where
   cross-pattern consistency mattered (e.g., all ALiBi users; all
   partial-rotary users).
4. **Slice 7 (recovery).** Discovered that the first pass dropped 20 folders
   due to a sharding bug (alphabetical splitting failed for late-added folders
   like `mask2former`, `lw_detr`, `mistral4`). Audited the missing 20.
5. **v11 additions.** `esm/` and `donut/` are multi-modeling folders; the
   first pass treated them as single rows. Added `esmfold` and `donut_swin`
   as separate rows. Final count: 447.
6. **Coordinator-only manual reads.** I personally source-read 107 folders
   end-to-end (HF + kb-nano), prioritizing every `unsupported`, every
   `partial` flagged by cross-verifiers, every status-change between
   first-pass and cross-verifier, and a 10% random sample of `composable`.

### 1.4 What was strict vs. lenient

| Concern | Strict / lenient | Why |
|---|---|---|
| Status (4 labels) | **strict** | This is the headline number; every status was either source-verified or cross-verifier confirmed. |
| Kernel-mapping by family (decoder vs encoder vs CLIP-text vs SigLIP vs Whisper vs T5) | **strict** | Each family has a distinct kb-nano L2 wrapper; getting it wrong would mislead the reader. |
| `silu_and_mul` vs `silu` for SwiGLU MLPs (rule 3) | **strict** (after v11 sweep — 4 nits fixed) | kb-nano really has both kernels; SwiGLU should reference the fused one. |
| RoPE variant (yarn / mrope / vision-2D / dinov3 / standard) | **strict** (3 vision-2D nits fixed in v11 sweep) | Different kb-nano L1 file per variant. |
| Norm variant (rms_norm / t5_layer_norm / bitnet_rms_norm / gemma_rms_norm / layer_norm) | **strict** (no nits found) | Different kb-nano L1 file per variant. |
| `*Attention` (sibling-wrapper) tagged `[wiring]` not `[compute]` | **lenient** (4 cases left as compute then re-tagged as wiring; render is identical) | The renderer treats both tags the same in the appendix; this is internal markdown bookkeeping. |
| `*PreTrainedModel`/`*Cache`/`*Output` dataclass exclusion (rule 12) | **strict** for ModelOutput dataclasses; **lenient** for nn.Module sublayers that happen to end with "Output" (e.g., `BertOutput` is a real Linear+Dropout+LayerNorm sublayer and was correctly listed as compute). | The guideline as written is ambiguous about nn.Module sublayers vs `BaseModelOutput*` dataclasses. |
| Multi-modeling folders | **strict** | `blip/blip` and `blip/blip_text` audited as separate rows. Same for `data2vec_*`, `donut/donut_swin`, `esm/esmfold`, `maskformer_swin`, `rt_detr/rt_detr_resnet`. |
| AutoBackbone / `load_backbone()` routing | **consistent** rule (partial, not unsupported) — see §9 below + `intermediate/CONSISTENCY_AUDIT.md` | Originally inconsistent across slices; resolved with the rule "infrastructure routing, not a compute gap". |
| ALiBi via `attn_mask` injection | **consistent** rule (partial) — see §9 below + `intermediate/CONSISTENCY_AUDIT.md` | Originally split between composable and unsupported across shards; resolved at partial. |
| Interleaved RoPE (GLM-style) | **consistent** rule (partial) | kb-nano `L1/rotary_emb.py` is non-interleaved; GLM/codegen/cohere/gptj need a partial. |
| Partial-rotary (`partial_rotary_factor`) | **consistent** rule (partial) | kb-nano RoPE rotates the full head dimension; partial-rotary needs slicing. |

### 1.5 What "strict" means in practice (a worked example)

Take `dinov2.Dinov2SwiGLUFFN` (now corrected). HF source:
```python
class Dinov2SwiGLUFFN(nn.Module):
    def __init__(self, config):
        ...
        self.weights_in = nn.Linear(in_features, 2 * hidden_features, bias=True)
        self.weights_out = nn.Linear(hidden_features, out_features, bias=True)
    def forward(self, hidden_state):
        hidden_state = self.weights_in(hidden_state)
        x1, x2 = hidden_state.chunk(2, dim=-1)
        hidden = nn.functional.silu(x1) * x2     # ← SwiGLU
        return self.weights_out(hidden)
```

kb-nano has both `L1/silu.py` (just `out = silu(x)`) and `L1/silu_and_mul.py`
(fused `out = silu(x[..., :H]) * x[..., H:]`). The strict mapping is the fused
kernel. The first-pass agent listed bare `L1/silu.py + L1/linear.py`; the
post-v11 sweep corrected this to `L1/silu_and_mul.py + L1/linear.py`. The
folder's status (`composable`) is unchanged because the kernel exists either
way.

---

## 2. The four denominators (and why we use 447)

| denominator | what it counts | when used |
|---:|---|---|
| **466** | All filesystem entries in `models/` (incl. `__init__.py`, etc.) | wrong: includes non-model files. |
| **465** | All HF model directories | wrong: includes 23 tokenizer/processor-only folders that have no PyTorch modeling code. |
| **442** | Folders with at least one `modeling_*.py` (excl. `_old`) | folder-level — secondary number. |
| **448** | Distinct PyTorch `modeling_*.py` files | multi-modeling folders (`blip`, `data2vec`, `donut`, `esm`, `maskformer`, `rt_detr`) expand to multiple files. |
| **447** | 448 minus `auto/__init__.py` (AutoModel registry, not a model) | **canonical denominator**. |

Filesystem ground-truth at `da6c53e4`:
```
$ ls /tmp/hf_transformers_pinned/src/transformers/models/ | wc -l           # 466
$ ls -d /tmp/hf_transformers_pinned/src/transformers/models/*/ | wc -l       # 465
$ find /tmp/hf_transformers_pinned/src/transformers/models/ -maxdepth 2 \
    -name "modeling_*.py" -not -name "*_old.py" | wc -l                      # 448
```

The audit covers 447 of those 448 (every PyTorch modeling file except `auto/`).

---

## 3. Post-v11 kernel-mapping fixes (this round)

A re-audit pass against the 12 kernel-mapping rules surfaced 9 file-mapping
inaccuracies in individual cells of the appendix tex. None changed any
folder's status.

| folder | class | before | after | rule |
|---|---|---|---|---|
| `dinov2` | `Dinov2SwiGLUFFN` | `L1/silu.py + L1/linear.py` | `L1/silu_and_mul.py + L1/linear.py` | 3 |
| `dinov2_with_registers` | `Dinov2WithRegistersSwiGLUFFN` | same | same | 3 |
| `cohere2_vision` | `Cohere2VisionMultiModalProjector` | `L1/linear.py + L1/silu.py` | `L1/linear.py + L1/silu_and_mul.py` | 3 |
| `ernie4_5_vl_moe` | `Ernie4_5_VLMoeVisionMLP` | `L1/linear.py + L1/silu.py` | `L1/linear.py + L1/silu_and_mul.py` | 3 |
| `clvp` | `ClvpGatedLinearUnit` | `L1/linear.py + L1/gelu.py` | `L1/linear.py + L1/gelu_and_mul.py` | 3 |
| `cpmant` | `CpmAntDenseGatedACT` | same | `L1/linear.py + L1/gelu_and_mul.py` | 3 |
| `edgetam_video` | `EdgeTamVideoRoPESelfAttention` | `L2/encoder_attention.py + L1/rotary_emb.py` | `L2/encoder_attention.py + L1/vision_rotary_emb.py` | 5 |
| `edgetam_video` | `EdgeTamVideoRoPECrossAttention` | `… + L1/rotary_emb.py` | `… + L1/vision_rotary_emb.py` | 5 |
| `efficientloftr` | `EfficientLoFTRAttention` | `L2/encoder_attention.py + L1/rotary_emb.py` | `L2/encoder_attention.py + L1/vision_rotary_emb.py` | 5 |

In addition, 4 sibling-wrapper classes (rule 11) were re-tagged from
`[compute]` to `[wiring]` for internal markdown consistency
(`bridgetower.BridgeTowerAttention`, `roformer.RoFormerAttention`,
`tapas.TapasAttention`, `xlm_roberta_xl.XLMRobertaXLAttention`). Render output
is identical; this is bookkeeping.

After re-rendering: 447 rows, kernel column corrected, status counts
unchanged at this stage (27 / 243 / 165 / 12).

### Post-v11 status changes (v12 partial-rotary consistency)

A subsequent guideline-by-guideline re-audit found that the rule
"partial-rotary requires either external slicing or Gemma4-style proportional
embedding → mark as `partial`" had been applied to phi/persimmon/glm/etc.
but **not** to 6 other folders with `partial_rotary_factor < 1.0`. Demoted
for consistency:

| folder | v11 | v12 | partial-rotary evidence |
|---|---|---|---|
| `bamba` | composable | partial | `configuration_bamba.py` hardcodes `partial_rotary_factor = 0.5` |
| `glm4v_moe` | composable | partial | `configuration_glm4v_moe.py` defaults `partial_rotary_factor = 0.5`; modeling does q_rot/q_pass split |
| `laguna` | composable | partial | full_attention layers use `partial_rotary_factor = 0.5` (sliding uses 1.0) |
| `musicflamingo` | composable | partial | Qwen2 LM rope_parameters set `partial_rotary_factor = 0.2` |
| `recurrent_gemma` | composable | partial | `partial_rotary_factor = 0.5`; Griffin SDPA does q_rot/q_pass externally |
| `solar_open` | composable | partial | inherits GLM4-MoE BC default `partial_rotary_factor = 0.5` |

Triple cross-check after v12 re-render: 447 rows in json/csv/tex,
counts (27 / 237 / 171 / 12), 0 three-way mismatches.

### Why kb-nano L1/rotary_emb.py does not cover partial-rotary directly

`L1/rotary_emb.RotaryEmbedding.__init__` builds `inv_freq = 1.0 / rope_theta**(arange(0, head_dim, 2)/head_dim)` — i.e., it always rotates the full head_dim. The CUDA kernel applies pairs across the entire head. There is also `Gemma4ProportionalRotaryEmbedding` (line 158) which handles partial-rotary by **padding** `inv_freq` with zeros for the non-rotated tail (Gemma4-specific approach). Neither covers the phi/glm-style q_rot/q_pass external-slice pattern out of the box. To call `L1/rotary_emb.py` on a `partial_rotary_factor < 1.0` model, the user code must:
1. Slice `q_rot = q[..., :rotary_dim]; q_pass = q[..., rotary_dim:]` (and same for k).
2. Call `RotaryEmbedding(rotary_dim)(positions, q_rot, k_rot)`.
3. Concatenate `q_out = cat([q_rot, q_pass], dim=-1)`.

This decomposes from L1 ops + standard PyTorch — hence `partial`, not `unsupported`. To make these folders truly `composable` would require a `PartialRotaryAttention` L2 wrapper that does the slice/cat inside `forward`. That wrapper does not currently exist in kb-nano.

---

## 4. Known limitations of the audit (read before citing)

1. **Static, not runtime.** The audit reads source. It does not certify
   byte-correctness against HF on a real workload, nor does it measure
   throughput. A `composable` model may run slowly via torch fallback today.
2. **`partial` is "decomposable, not yet wrapped".** A partial folder will
   run in PyTorch (the relevant op is in `torch.*`); kb-nano just doesn't
   have a tuned kernel for that variant yet. Examples: partial-rotary RoPE,
   interleaved-RoPE, ALiBi slopes, BatchNorm1d, Conformer rel_shift.
3. **`unsupported` excludes "could be added with effort".** All 12
   unsupported folders need either an external library or a custom CUDA
   kernel. Not "I'd need to write more PyTorch."
4. **L4 promotion is conservative.** Three judgment calls were made
   conservatively rather than aggressively:
   - `qwen2_5_vl` is `composable`, not `kb_nano_l4`, even though
     `tasks/baseline/L4/qwen2_vl.py` would in principle run it (Qwen2.5-VL
     inherits the LM from Qwen2-VL). The L4 docstring header says
     "Qwen2-VL" without explicitly mentioning 2.5; promoting would push
     L4 from 27 to 28 and strict from 59.1% to 59.3%.
   - `sam3_tracker_video` is `kb_nano_l4` via `tasks/baseline/L4/sam3_tracker.py`
     (Sam3TrackerPredictor handles the video-tracker workflow). This is the
     looser side of the same judgment call as qwen2_5_vl.
   - `dinov3_vit` is `kb_nano_l4` via `tasks/baseline/L4/dinov3.py` (Eva-arch
     DINOv3 7B/16). The L4 file's docstring explicitly targets the ViT variant,
     so promotion is solid.
5. **Multi-modeling folders' rows are uneven.** `blip` has 2 rows
   (`blip/blip`, `blip/blip_text`). `data2vec` has 3 (`audio`, `text`,
   `vision`). `donut`, `esm` have 2 each. `maskformer` has 2
   (`maskformer`, `maskformer_swin`). `rt_detr` has 2. The denominator
   447 reflects this expansion.
6. **Multi-modeling folders not in the audit.** The HF source has
   `modular_<x>.py` files that are *generators* — they produce
   `modeling_<x>.py` at install time. The audit reads `modeling_<x>.py`
   (what runs), with cross-reference to `modular_<x>.py` for inheritance
   lineage.
7. **Borderline `partial` vs `composable` calls.** Some folders sit on
   the boundary, depending on whether you require a kb-nano L2 wrapper
   for the family or only that L1 primitives exist. The audit chose the
   stricter side (partial) for: BART-style encoder-decoder with separate
   q/k/v projections (`fsmt`), encoder folders that use Llama-style
   attention with `is_causal=False` flag (left as composable when the
   audit could verify the flag is honored, e.g., `eurobert`).
8. **Paper text claim is not reproducible from the table.** The submitted
   paper cites 421 / 96.2% / 7 unsupported. The submitted table actually
   has 425 rows; 421 was an informal "after collapsing multi-modeling
   folders" derivation that was never executed. This re-audit's 447 / 59.1%
   strict / 12 unsupported is the corrected answer (see
   `NUMBER_DRIFT_RECONCILIATION.md` for the full chain).

---

## 5. Reproducibility

Every status verdict has a 3-source paper trail in `audit_evidence.csv`:
`fp_verdict` (first-pass shard), `xv1_verdict` (cross-verifier round 1),
`p2_verdict` (phase-2 deep read). Disagreements between the three were
adjudicated by the coordinator (me); the `i_personally_read` column flags
which 107 folders the coordinator source-read end-to-end.

The full markdown shards (`tools/manual_audit_shard_{01..17}.md`) have the
per-class compute-vs-wiring tag, the kb-nano file mapping, and the
one-line rationale that goes into the rendered tex. To re-render after
any shard edit:

```bash
python audits/hf_transformers_coverage/tools/md_to_tex.py
# -> writes audits/hf_transformers_coverage/hf_coverage_rows.tex
```

---

## 6. Trust footprint

A reviewer who wants to verify *one* folder spends ~5 minutes:
1. Open `audit_evidence.csv`, find the folder's row, read all 4 verdict
   columns.
2. Open `tools/manual_audit_shard_NN.md`, find the folder, read the
   per-class breakdown.
3. Open `/tmp/hf_transformers_pinned/src/transformers/models/<folder>/modeling_*.py`
   and the kb-nano files cited in the breakdown.
4. Compare.

A reviewer who wants to verify the *headline* spends ~30 seconds:
1. `wc -l hf_coverage_rows.tex` ≈ 4000 lines, 447 entries.
2. `grep -c '\\xmark' hf_coverage_rows.tex` = 12 unsupported.
3. `grep -c '\$\\bullet\$' hf_coverage_rows.tex` = 27 L4.

---

## 7. The 27 L4 promotions (per-folder rationale)

Every L4 promotion was verified by reading the kb-nano `tasks/baseline/L4/<file>.py`
docstring header against the HF `modeling_<folder>.py`. Table below pairs each
HF folder with the L4 file that runs it.

| HF folder | kb-nano L4 file | evidence (L4 docstring) |
|---|---|---|
| `bitnet` | `L4/bitnet.py` | targets `microsoft/bitnet-b1.58-2B-4T` (W1.58A8, GQA, NeoX RoPE, squared-ReLU MLP) |
| `convnextv2` | `L4/convnextv2.py` | "ConvNeXtV2 image classification model" |
| `deepseek_v3` | `L4/deepseek.py` | "Standalone DeepSeek V3.2 model implementation. Supports MLA, MoE, DSA" |
| `dinov3_vit` | `L4/dinov3.py` | "DINOv3 7B/16 Eva vision encoder"; targets timm `vit_7b_patch16_dinov3.lvd1689m` |
| `gemma4` | `L4/gemma4.py` | Gemma4 with proportional partial-rotary RoPE |
| `gpt_oss` | `L4/gpt_oss.py` | "openai/gpt-oss-20b": MXFP4 experts + YaRN RoPE + sliding window + attention sinks |
| `jamba` | `L4/jamba.py` | Jamba hybrid Transformer/Mamba |
| `llama` | `L4/llama.py` | Llama 3.1 |
| `llama4` | `L4/llama4.py` | Llama 4 (NoPE layers + temperature tuning + weightless QK norm) |
| `mamba` | `L4/mamba.py` | Mamba (selective scan) |
| `mamba2` | `L4/mamba2.py` | Mamba2 SSD |
| `mixtral` | `L4/mixtral.py` | Mixtral 8x7B SparseMoeBlock |
| `pi0` | `L4/pi0.py` | "Pi0 vision-language-action model" |
| `qwen2_5_omni` | `L4/qwen2_5_omni.py` | "Qwen2.5-Omni Thinker model" |
| `qwen2_vl` | `L4/qwen2_vl.py` | "Qwen2-VL: vision encoder + Qwen2 LM with M-RoPE" |
| `qwen3_next` | `L4/qwen3_next.py` | Qwen3-Next (per-head QK-norm + partial RoPE + output gating) |
| `qwen3_vl` | `L4/qwen3_vl.py` | Qwen3-VL |
| `qwen3_vl_moe` | `L4/qwen3_vl_moe.py` | Qwen3-VL MoE |
| `rt_detr_v2` | `L4/rtdetrv2.py` | RT-DETR v2 |
| `sam3` | `L4/sam3.py` | SAM3 base (text-prompted segmentation) |
| `sam3_tracker` | `L4/sam3_tracker.py` | Sam3TrackerBase + Sam3TrackerPredictor |
| `sam3_tracker_video` | `L4/sam3_tracker.py` | Sam3TrackerPredictor handles the video-tracker workflow |
| `sam3_video` | `L4/sam3_video.py` | wraps Sam3TrackerPredictor for video |
| `siglip2` | `L4/siglip2.py` | timm `naflexvit_so400m_patch16_siglip.v2_webli` |
| `swinv2` | `L4/swinv2.py` | SwinV2 (cosine attention + CPB MLP) |
| `vjepa2` | `L4/vjepa2.py` | V-JEPA 2 |
| `whisper` | `L4/whisper.py` | Whisper (encoder-decoder, 3 sibling attention classes) |

### L4 calls intentionally NOT promoted (kept as `composable`)

| HF folder | reason |
|---|---|
| `qwen2_5_vl` | `L4/qwen2_5_vl_encoder.py` is a **text-encoder slice** for HunyuanVideo-1.5, not the full Qwen2.5-VL VL pipeline. Qwen2.5-VL inherits the LM from Qwen2-VL so `L4/qwen2_vl.py` would in principle run it, but the L4 docstring header says "Qwen2-VL", not 2.5. Kept conservative; promoting would push L4 from 27 → 28 and strict from 59.1% → 59.3%. |
| `falcon_mamba` | HF `modeling_falcon_mamba.py:60-62, 267-269, 367-369` calls `rms_forward` on B/C/dt before the SSM scan; kb-nano `L4/mamba.py` doesn't do this extra normalization. Architecture differs. |
| `gemma4_assistant` | Couldn't verify equivalence to `gemma4` from header comments alone. Conservative. |
| `sam2_video` | The 3 SAM3-named L4 files (`sam3_video.py`, `sam3_tracker.py`, `sam3.py`) explicitly cite `sam3/model/sam3_*.py` as their reference; SAM2 is a different generation. SAM2 falls to composable via L1/L2 sam* primitives. |

---

## 8. The 12 unsupported folders (per-folder rationale, canonical v12)

Each requires either (a) a custom CUDA kernel from `kernels-community/*`,
(b) an external library (timm, natten, detectron2, xlstm), or (c) genuinely
novel compute with no `torch.*` equivalent. None of the 12 can run "as-is"
in kb-nano without writing new code or installing external deps.

### Custom CUDA via `kernels-community/*` (3)

| folder | HF source | dependency |
|---|---|---|
| `mra` | `mra/modeling_mra.py:51-57` `mra_cuda_kernel = get_kernel("kernels-community/mra")` | `index_max`, `mm_to_sparse`, `sparse_dense_mm` |
| `rwkv` | `rwkv/modeling_rwkv.py:42-52` `rwkv_cuda_kernel = get_kernel("kernels-community/rwkv")` | wkv_cuda forward/backward (v4-specific; kb-nano has only v7 in `L1/rwkv7_recurrence.py`) |
| `yoso` | `yoso/modeling_yoso.py:51-57` `yoso = get_kernel("kernels-community/yoso")` | `fast_hash`, `lsh_cumulation` |

### External library hard-imports (5)

| folder | HF source | dependency |
|---|---|---|
| `dinat` | `dinat/modeling_dinat.py:38-39` `from natten.functional import natten2dav, natten2dqkrpb` | `natten` (sliding/dilated 2D neighborhood attention) |
| `timm_backbone` | `timm_backbone/modeling_timm_backbone.py:29, 54` | `timm.create_model(...)` |
| `timm_wrapper` | `timm_wrapper/modeling_timm_wrapper.py:28, 63` | `timm.create_model(...)` |
| `fast_vlm` | `configuration_fast_vlm.py:66` defaults `vision_config.model_type = "timm_wrapper"` | transitively requires timm |
| `layoutlmv2` | `layoutlmv2/modeling_layoutlmv2.py:42-43` `is_detectron2_available(); import detectron2` | `detectron2` (Mask R-CNN visual feature extractor) |

### External library + bespoke compute (1)

| folder | HF source | dependency |
|---|---|---|
| `xlstm` | `xlstm/modeling_xlstm.py:34-36` `from xlstm.xlstm_large.model import mLSTMBlock` | external `xlstm` package's mLSTM chunkwise/sequence/step kernels (no kb-nano L1 equivalent; kb-nano `lstm.py` is plain LSTM) |

### Genuinely novel compute primitive (3)

| folder | HF source | what's novel |
|---|---|---|
| `diffllama` | `modeling_diffllama.py:182-273` `lambda_init_fn`, `lambda_full = lambda_1 - lambda_2 + self.lambda_init`, `attn_output = attn_output1 - lambda_full * attn_output2` | Differential Attention: paired softmax with learnable λ subtraction. Not in any standard library. |
| `ibert` | `modeling_ibert.py:39` `from .quant_modules import IntGELU, IntLayerNorm, IntSoftmax, QuantAct, QuantEmbedding, QuantLinear` | Integer-arithmetic GELU/Softmax/LayerNorm (8-bit fixed-point emulation). kb-nano has fp8/bitnet quant but not int-arith emulation. |
| `gemma3n` | `modular_gemma3n.py:316` `class Gemma3nVisionConfig(TimmWrapperConfig)` + `gemma3n/modeling_gemma3n.py:151+` Gemma3nAudio* (relative position bias + Conformer) + AltUp + Laurel | Compounding gaps: timm-backed vision (timm_wrapper transitively) + audio Conformer with relative position embedding + AltUp ladder + Laurel skip-connection structure. |


---

## 9. Cross-pattern judgment calls (the ambiguous decisions)

This table consolidates every recurring pattern where the partial-vs-composable
or composable-vs-unsupported call was genuinely ambiguous, plus the
consistency rule applied. These rules were synthesized from cross-agent
disagreements (see `intermediate/CONSISTENCY_AUDIT.md` for the per-folder
agent verdict log) and applied uniformly across all matching folders.

### Patterns ruled `partial` (decomposable from torch + L1, but no kb-nano L2 wrapper)

| pattern | rule rationale | example folders |
|---|---|---|
| **partial-rotary** (`partial_rotary_factor < 1.0`) | kb-nano `L1/rotary_emb.RotaryEmbedding` rotates the full head_dim. `Gemma4ProportionalRotaryEmbedding` handles the proportional case but not the q_rot/q_pass split. Needs external slicing in user code. | phi, phi3, persimmon, fuyu, gpt_neox, nemotron, moonshine, stablelm, qwen3_5, deepseek_v4, glm, glm4, glm4_moe, glm4v, glmasr, **bamba**, **glm4v_moe**, **laguna**, **musicflamingo**, **recurrent_gemma**, **solar_open** (bold = the 6 v12 demotions) |
| **interleaved RoPE** (`cos[..., :d//2].repeat_interleave(2, dim=-1)`) | kb-nano `L1/rotary_emb` is standard NeoX (`rotate_half`-style, not interleaved). GLM family uses interleaved. | glm, glm4, glm4v, glm_ocr, glmasr, codegen, gptj, cohere, codestral; deepseek_mla path interleaved RoPE IS supported via `is_neox_style=False` |
| **LayerNorm in decoder LLM** (Phi/Persimmon, not RMSNorm) | kb-nano `L2/attention.py` expects RMSNorm. LayerNorm-decoder needs a different norm wrapper. | phi, persimmon |
| **ALiBi** (added to attn scores via `build_alibi_tensor`) | kb-nano has no alibi kernel; works through `attn_mask` injection but no dedicated wrapper. | bloom, falcon (when `alibi=True`) |
| **AutoBackbone / `load_backbone()`** | Infrastructure routing (HF AutoModel registry), not a compute gap. The backbone *itself* is composable when its model_type maps to a kb-nano-supported family; the routing layer needs config-driven dispatch which kb-nano doesn't have. | conditional_detr, oneformer, omdet_turbo, modernvbert, chmv2, lightglue, grounding_dino, pix2struct (etc.) — **but folders that route to timm/detectron2 escalate to unsupported** |
| **BART-style separate q/k/v projections** | kb-nano `L2/whisper_attention.py` uses `QKVParallelLinear` (merged QKV). BART-family uses 3 separate Linear projections + `(seq, batch, dim)` layout. Decomposable but no L2 wrapper for that exact layout. | fsmt, mbart, marian, blenderbot (when matched against whisper_attention strict layout); composable when audited against `nn.Linear` decomposition |
| **T5 cross-attention** (`T5LayerCrossAttention`) | kb-nano `L2/t5_attention.py` implements self-attention with relative bias; cross-attn variant (with key_value_states from encoder + EncoderDecoderCache) is not wrapped. T5 encoder-only path is composable; decoder makes the folder partial. | t5, mt5, longt5, umt5, t5gemma, t5gemma2, switch_transformers, pop2piano, udop, pix2struct |
| **Conformer relative-position rel_shift** (`matrix_bd shift_relative_position_tensor`) | Decomposes from `gather` + `bmm` + `softmax` (all in L1) but no kb-nano kernel implements the rel_shift index gymnastics. | wav2vec2_conformer, fastspeech2_conformer, granite_speech, granite_speech_plus |
| **Swin V1 relative-position-bias windowed attention** | kb-nano `L2/swinv2_window_attention.py` is V2-specific (cosine attention + CPB MLP). V1 uses additive `relative_position_bias_table`. | swin, donut_swin, maskformer_swin |
| **BatchNorm1d** | kb-nano has `L1/batch_norm2d.py` only; BN1d is a torch.nn primitive but no kb-nano wrapper. | levit, efficientnet (when BN1d is on the path), wav2vec2_conformer (depthwise conv path) |
| **weight_norm (`torch.nn.utils.weight_norm`)** | Reparametrization; decomposes from L1 ops but no kb-nano wrapper. | dac, encodec, vits, mimi, fastspeech2_conformer |
| **Custom 2D position encoding** (Fourier basis, segment-aware index gather) | Decomposes from arange + einsum but no kb-nano kernel. | perceiver (Fourier), tapas (IndexMap segment reduce), reformer (LSH) |
| **`torch.fft.{rfft, irfft, fft, fftn}`** | torch builtin; no kb-nano FFT kernel. | autoformer, fnet |
| **MoE with bespoke routing** (JetMoe MoA, identity-experts, etc.) | The MoE kernel exists (`L1/moe_grouped_gemm` etc.) but the routing layer needs custom logic. | jetmoe (MoA), longcat_flash (identity-experts), zamba2 (custom mixer) |
| **Sliding-window / chunked attention with attn_mask** | Decomposes via mask injection. kb-nano supports `sliding_window` flag in `LlamaAttention`; folders without that flag in their L2 wrapper are partial. | mistral has the flag (composable); big_bird, bigbird_pegasus, longformer use block-sparse + global pattern (partial) |
| **Snake1d activation** | Pure-torch activation, no kb-nano wrapper. | dac, encodec |
| **`nn.GRUCell`** | RNN cell; no kb-nano LSTM/GRU wrapper for cell-level recurrence. | bark, dac, vits |

### Patterns ruled `composable` (no compute gap)

| pattern | rule rationale | example folders |
|---|---|---|
| **`nn.MultiheadAttention`** | Decomposes to separate Q/K/V Linear + sdpa + output Linear; all in L1/L2. | bridgetower, ctrl, flaubert, xlm, blip2 q-former |
| **CLIP qkv split** (separate q/k/v Linear) | Maps to `L2/clip_attention.py` which uses separate projections (not merged). | clip, altclip, x_clip |
| **mamba variants with custom mixer wiring** | If the underlying selective-scan / causal_conv1d / RMSNormGated primitives all map, the wiring is composable. | mamba, falcon_mamba (composable since v9 reaudit), bamba (until v12 partial-rotary demotion) |

### Patterns ruled `unsupported` (custom CUDA / external lib)

| pattern | rule rationale | example folders |
|---|---|---|
| **`kernels-community/*` CUDA kernel** | External CUDA dependency; no `torch.*` fallback. | mra, rwkv (v4), yoso |
| **`timm.create_model`** | Hard import of external `timm`. | timm_backbone, timm_wrapper, fast_vlm (transitive) |
| **`detectron2`** | Hard import of external `detectron2`. | layoutlmv2 |
| **`natten.functional`** | External CUDA library for neighborhood attention. | dinat |
| **`xlstm.xlstm_large.model`** | External `xlstm` package's mLSTM kernels. | xlstm |
| **Bespoke autograd.Function with novel math** | Differential attention (paired softmax + λ subtraction); integer-arithmetic GELU/Softmax. | diffllama, ibert |
| **Compounding gaps** (multiple unsupported sub-systems) | E.g. timm vision tower + Conformer audio + AltUp ladder. | gemma3n |

