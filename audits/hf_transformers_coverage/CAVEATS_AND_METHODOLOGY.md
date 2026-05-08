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

---

## 10. Per-folder partial gap rationale (all 171 folders)

§9 lists the recurring patterns; this section enumerates every partial
folder, grouped by primary gap, with a one-line rationale for each.
For full per-class breakdowns see `tools/manual_audit_shard_*.md` and
`audit_evidence.csv`.

### partial-rotary RoPE (`partial_rotary_factor < 1.0`; q_rot/q_pass split) (23)

> kb-nano `L1/rotary_emb.RotaryEmbedding` rotates the full head_dim. Needs external slicing in user code or a Gemma4-style proportional rotary wrapper.

- `bamba` — BambaConfig hardcodes `partial_rotary_factor = 0.5` (configuration_bamba.py). kb-nano L1/rotary_emb.py rotates the full head_dim; partial-rotary requires either external q_rot/q_pass slicing in user code or a Gemma4-style proportional embedding wr...
- `codegen` — GPT-J derivative: fused QKV with mp_num=4 split, partial NeoX RoPE (rotary_dim), MHA with no GQA, fc1+gelu_new+fc2 MLP, LayerNorm. All compute primitives exist (linear, rotary_emb, gelu, layer_norm, dense_attention).
- `fuyu` — Fuyu wraps a Persimmon (parallel-attention) language model with a single Linear vision_embed_tokens projecting raw image patches to text embedding space. Persimmon is not in kb-nano, and there is no Fuyu/Persimmon L4 pipeline.
- `glm` — GLM = Llama attention + Phi3MLP (gate_up_proj + chunk + down_proj, same SwiGLU pattern as llama_mlp) + interleaved (rotate-pairs) RoPE. The interleave-2 rotary is a variant of NeoX rotary; can be done via reshape outside rotary_emb.
- `glm4` — GLM-4 = Glm attention + Phi3MLP + extra post-self-attn / post-mlp RMSNorms (sandwich norm). All ops map to existing kb-nano kernels.
- `glm46v` — (see shard)
- `glm4_moe` — GLM-4 MoE = Llama-style attention (Cohere bias config) + DeepSeekV3 MoE (top-k softmax router with shared expert) + RMSNorm + Llama RoPE. Maps cleanly to L2/attention.py + L2/shared_expert_moe.py + L1/rms_norm + L1/rotary_emb.
- `glm4v` — GLM-4V = Glm4 (Llama-style) text decoder with M-RoPE for multimodal positions + Qwen2.5-VL-style vision encoder (Conv3d patch embed + 2D RoPE + SwiGLU vision MLP). All ops map to existing kb-nano kernels (mrope, vision_rotary_emb, llama_mlp, atten...
- `glm4v_moe` — configuration_glm4v_moe.py defaults `partial_rotary_factor = 0.5`; modeling_glm4v_moe.py:apply_rotary_pos_emb slices q[..., :rotary_dim] / q[..., rotary_dim:] before rotating. kb-nano L1/rotary_emb.py rotates the full head; partial-rotary needs ex...
- `glm_image` — GLM-Image adds a Chameleon-style VQVAE for image generation. The VQVAE has bespoke ResNet/Conv2d blocks and a vector quantizer with EMA codebook updates — kb-nano has no Chameleon VQVAE kernels and no L4 pipeline for it.
- `glm_ocr` — GLM-OCR = pure subclass of Glm4v vision + text components, inheritance only (no new compute). Same as Glm4v structurally.
- `glmasr` — GLM-ASR adds an audio Conformer encoder (AudioFlamingo3 style) with depthwise Conv1d + GLU + BatchNorm1d + Shaw relative positional embeddings. kb-nano has no Conformer/audio-encoder kernels (no audio_flamingo or whisper-non-attention modules cove...
- `gpt_neox` — GPT-NeoX = LlamaModel base with NeoX-style RoPE on first part of head_dim, separate q/k/v projections fused into qkv linear, GELU MLP. Maps cleanly to L2/attention + L2/llama_mlp + L1/rotary_emb.
- `gptj` — GPT-J = parallel residual: attention and MLP run in parallel from same input, summed with residual. Uses GPT-J-style RoPE (rotary on first rotary_dim of head_dim, NeoX layout) without bias on q/k/v. All ops standard.
- `laguna` — configuration_laguna.py sets `full_attention.partial_rotary_factor = 0.5` (sliding_attention uses 1.0). kb-nano L1/rotary_emb.py rotates the full head_dim; partial-rotary on the full-attention layers requires either Gemma4-style proportional embed...
- `musicflamingo` — configuration_musicflamingo.py sets `partial_rotary_factor = 0.2` for the Qwen2 LM rope_parameters; standard L1/rotary_emb rotates the full head_dim, so the LM path needs external q_rot/q_pass slicing or Gemma4-style proportional rotary. Custom Mu...
- `nemotron` — NemotronLayerNorm1P (F.layer_norm with weight+1 reparam) has no kb-nano kernel. Partial-RoPE (rotates only first int(head_dim*partial_rotary_factor) channels) not implemented in L1/rotary_emb.py which assumes full head_dim rotation. squared_relu M...
- `persimmon` — (see shard)
- `phi` — PhiAttention overrides Q/K/V to separate Linears with bias=True and applies partial RoPE; PhiMLP inherits CLIPMLP (fc1+activation_fn+fc2 with QuickGELU). PhiDecoderLayer uses nn.LayerNorm and a parallel attn+mlp residual. The compute relies on tor...
- `phi3` — Phi3Attention uses one fused qkv_proj of width Hq*D + 2*Hkv*D, then apply_rotary_pos_emb slices q/k to rotary_dim only and concatenates back the unrotated tail. kb-nano L1/rotary_emb operates on the full head; partial-rotary requires manual slicin...
- `phi4_multimodal` — Phi-4 multimodal = Phi-3 LLM + SigLIP vision tower + Conformer-style audio encoder with NeMo conv subsampling, depth-wise separable Conv1d, GLU pointwise conv, relative attention bias, and mean-variance norm. Conformer audio block, NeMo conv subsa...
- `recurrent_gemma` — configuration_recurrent_gemma.py sets `partial_rotary_factor = 0.5`. RecurrentGemmaSdpaAttention does q_rot, q_pass split and rotates only q_rot. kb-nano L2/attention.py forwards full q,k to rotary_emb; standard L1/RotaryEmbedding rotates the full...
- `solar_open` — configuration_solar_open.py inherits the GLM4-MoE BC default `kwargs.setdefault("partial_rotary_factor", 0.5)`. Standard L1/rotary_emb rotates the full head_dim; partial-rotary needs external slicing or Gemma4-style proportional rotary. Same gap a...

### Interleaved RoPE (`cos[..., :d//2].repeat_interleave(2, dim=-1)`) (2)

> kb-nano `L1/rotary_emb` applies rotate_half-style NeoX rotation; the GLM/Cohere interleaved variant needs a different rotation pattern. (Already covered above for glm-family in the partial-rotary group; listed here for non-glm interleaved cases.)

- `cohere` — Llama-derived: GQA attention with optional QK-LayerNorm, SwiGLU MLP, custom CohereLayerNorm (centered LayerNorm = standard nn.LayerNorm without bias), interleaved RoPE. All primitives exist.
- `helium` — Helium = Llama-family with HeliumRMSNorm (fp32 cast inside) + GraniteAttention base (no attention_multiplier override here — rebound to 1/sqrt(d_k)) + HeliumMLP (= LlamaMLP) + interleaved RoPE (rotate_half via stack). All maps to L2/attention + L2...

### LayerNorm-decoder (Phi/Persimmon/GPT-NeoX, not RMSNorm) (1)

> kb-nano `L2/attention.py` expects `RMSNorm` and SwiGLU MLP. LayerNorm-decoder paths need a separate L2 wrapper. Often combined with parallel-attention residual.

- `mpt` — build_mpt_alibi_tensor produces an additive [n_heads, q_len, k_len] bias added to attention scores. kb-nano flash kernels lack ALiBi support; torch.matmul-based softmax is the only path. Per consistency reminder: ALiBi-as-additive-bias is partial.

### ALiBi positional bias (`build_alibi_tensor` injection) (2)

> kb-nano has no ALiBi kernel; the slopes can be injected via `attn_mask` but no dedicated wrapper / fused path.

- `bloom` — ALiBi-biased multi-head attention via additive bias on attention scores -- supported by L1/dense_attention or L1/flash_attn_decode (alibi_slopes). MLP is fc1->gelu_new->fc2; LayerNorm + fused QKV linear.
- `falcon` — Falcon supports both RoPE (new arch) and ALiBi (legacy 7B/40B) attention biases plus parallel-attention layer topology. kb-nano attention kernels (LlamaAttention/Attention impl) have no ALiBi additive bias support.

### AutoBackbone / `load_backbone()` infrastructure routing (14)

> HF AutoModel registry config-driven dispatch; the backbone *itself* is composable when its model_type maps to a kb-nano-supported family but kb-nano has no AutoBackbone shim. Folders that route to timm/detectron2 escalate to unsupported.

- `chmv2` — Depth estimation model that loads its backbone via HF AutoBackbone (load_backbone) -- requires a separate vision backbone (DPT-style) that kb-nano does not expose as an L4 pipeline. The DPT head itself (Reassemble + Fusion + UpsampleConv) is compo...
- `conditional_detr` — DETR-derived object detector that loads its CNN backbone via transformers.backbone_utils.load_backbone (typically ResNet from timm or AutoBackbone). kb-nano has no backbone-loading abstraction. The transformer encoder/decoder itself is composable...
- `dab_detr` — Standard DETR-style enc-dec with conditional cross-attention with q/k content/position projections; all compute is matmul/softmax/linear/layer_norm/conv (via load_backbone) — all primitives exist.
- `detr` — Standard DETR enc-dec: ResNet backbone (AutoBackbone) + sinusoidal/learned position embeddings + standard MHA self/cross attention + MLP. All primitives exist.
- `mask2former` — Pixel encoder is loaded via AutoBackbone (load_backbone) — kb-nano has no AutoBackbone/timm equivalent. Decoder also uses nn.MultiheadAttention for cross-attn.
- `mm_grounding_dino` — Inherits Grounding DINO conv encoder which calls load_backbone(config) for the (Swin) image backbone, plus loads BERT text backbone via AutoModel.from_config. AutoBackbone has no kb-nano equivalent.
- `modernvbert` — ModernVBert composes a SigLIP vision tower with a ModernBert text encoder via AutoModel.from_config. The ModernBertConnector does pixel-shuffle + Linear, but ModernBert (text) itself is partial (sliding-window+RoPE encoder attention with no kb-nan...
- `omdet_turbo` — Open-vocab detection model: depends on AutoBackbone (timm) for vision tower, AutoModel for text, plus multi-scale deformable attention (MSDA v1, kernels-community kernel) and a custom hybrid encoder/decoder. No L4 pipeline; kb-nano deformable atte...
- `oneformer` — OneFormer universal segmentation depends on multi-scale deformable attention, bare nn.MultiheadAttention (Transformer decoder cross-attn), Hungarian matcher (scipy), and AutoBackbone (timm/swin/dinat). No kb-nano equivalent for the cross-attention...
- `perception_lm` — Vision tower is loaded via AutoModel.from_config with model_args['embed_dim'] suggesting a custom timm-like perception encoder; no fixed kb-nano vision pipeline corresponds. Adaptive avg pool 2d exists in kb-nano (L1/adaptive_avg_pool2d.py).
- `prompt_depth_anything` — Depth-Anything depth estimator with prompt-depth fusion. Backbone is loaded externally via load_backbone (e.g. DPT/DINOv2). Compute classes are pure Conv2d + ReLU + bilinear upsample + residual fusion; no novel ops.
- `sam` — SamVisionAttention's decomposed 2D relative position embeddings (rel_pos_h / rel_pos_w with F.interpolate + einsum to add to attention scores) have no kb-nano L2 equivalent. The base attention compute (qk@k.T + softmax + p@v) is composable from De...
- `sam_hq` — SAM-HQ adds high-quality output token with Hiera-style ViT vision encoder + standard SAM prompt encoder + two-way transformer + HQ-augmented mask decoder. All compute primitives (Conv2d, LayerNorm, Linear, attention, GELU, interpolate) exist in kb...
- `tvp` — TvpVisionModel uses Transformers' load_backbone for an external ResNet (not a kb-nano backbone). TvpFrameDownPadPrompter / TvpFramePadPrompter implement learnable padding around video frames as nn.Parameter padding masks — no kb-nano equivalent. T...

### BART-style separate q/k/v projections + (seq, batch, dim) layout (7)

> kb-nano `L2/whisper_attention.py` uses `QKVParallelLinear` (merged QKV). BART-family uses 3 separate Linear projections + (seq, batch, dim) layout. Decomposable from L1 ops but no L2 wrapper for that exact layout.

- `flaubert` — Flaubert (XLM-derived) uses BART-style attention with separate q_lin/k_lin/v_lin and supports both encoder and cross-attention with EncoderDecoderCache. The kb-nano encoder_attention.py does not support cross-attention, and the bare q/k/v separate...
- `florence2` — Florence-2 = DaViT vision backbone (channel attention + window spatial attention with depth-wise Conv2d positional encoding) + BART-style seq2seq language model. Vision backbone is bespoke DaViT not present in kb-nano; BART seq2seq is also not imp...
- `fsmt` — FSMT is a BART-style encoder-decoder seq2seq model with cross-attention. Although kb-nano has whisper_attention.py (3 sibling classes: encoder/decoder/cross), there is no FSMT/BART L4 pipeline and the FSMT Attention class uses (seq, batch, dim) la...
- `lilt` — LiltSelfAttention runs two parallel QKV streams (text + layout) and adds the scaled-dot-product scores before softmax; the cross-stream score addition has no kb-nano L2 equivalent. Implemented in HF via raw torch.matmul + nn.Softmax, so it works o...
- `pp_formulanet` — PPFormulaNetMultiModalProjector wraps Florence-2 projection (custom). PPFormulaNetVisionAttention inherits SLANeXtVisionAttention (custom encoder attention). PPFormulaNetAttention + PPFormulaNetDecoderLayer follow MBart enc-dec with cross-attentio...
- `time_series_transformer` — TimeSeriesTransformerAttention is BART-style (bmm-flatten + Q*scaling + LayerNorm-around) — kb-nano L2/whisper_attention.py covers the closest pattern but the per-class structure (no relative bias, no RoPE, plain multi-head with key_value_states f...
- `trocr` — TrOCRAttention is BART-style (Q*scaling -> bmm-flattened multi-head SDPA) with EncoderDecoderCache for cross-attn — kb-nano L2/whisper_attention.py is the closest sibling but is structured around Whisper's three sibling classes; the bare 'TrOCRAtt...

### T5 cross-attention (`T5LayerCrossAttention` not wrapped) (10)

> kb-nano `L2/t5_attention.py` implements self-attention with relative bias; the cross-attn variant (with `key_value_states` from encoder + `EncoderDecoderCache`) is not wrapped. Encoder-only path is composable; decoder makes the folder partial.

- `longt5` — LongT5 adds Local and TransientGlobal block-sparse attention variants on top of T5 relative-bias attention; no kb-nano kernel for the local-block / transient-global attention.
- `mt5` — T5 decoder cross-attention with relative-position bias has no kb-nano L2 (kb-nano whisper_attention.py is BART-style without T5 relative bias; t5_attention.py is encoder self-attn). Per consistency reminder: T5 cross-attn unsupported in kb-nano.
- `pix2struct` — Pix2StructTextLayerCrossAttention wraps a Pix2StructTextAttention configured for cross-attention with relative bias. kb-nano L2/t5_attention.py implements only T5SelfAttention; cross-attention with the T5 relative-position-bias path is not present...
- `pop2piano` — Pop2PianoLayerCrossAttention wraps Pop2PianoAttention configured for cross-attention. kb-nano L2/t5_attention.py implements T5SelfAttention only; cross-attention path (separate K/V from encoder hidden states with relative bias) is not implemented....
- `switch_transformers` — SwitchTransformersTop1Router applies token-priority cumsum + capacity overflow masking and SwitchTransformersExperts loops over experts with index_add_; the capacity-limited Switch routing is structurally different from standard fused-MoE (top-k s...
- `t5` — T5LayerCrossAttention (T5Attention with key_value_states from encoder + EncoderDecoderCache) is not implemented in kb-nano (L2/t5_attention.py covers only T5SelfAttention); the relative-bias bucket helper exists encoder-side but the cross-attn pat...
- `t5gemma` — T5GemmaCrossAttention (Gemma2Attention with overridden forward that pulls cross KV from encoder_hidden_states + EncoderDecoderCache.cross_attention_cache + soft-cap + sliding_window=None) has no kb-nano L2 equivalent; L2/attention.py covers self-a...
- `t5gemma2` — T5Gemma2MergedAttention concatenates [self_KV \| cross_KV] along the seq dim and runs a single fused softmax (with masking); kb-nano L2/attention.py is vanilla self-attention, no merged self+cross variant exists. Q/K RMS-norm is present (matches Ge...
- `udop` — UdopLayerCrossAttention has no kb-nano kernel (kb-nano L2/t5_attention.py covers only self-attention with relative bias). RelativePositionBiasHorizontal / RelativePositionBiasVertical compute bbox-based 2D relative-position buckets (uses bbox coor...
- `umt5` — UMT5LayerCrossAttention has no kb-nano kernel (L2/t5_attention.py covers only self-attention). The relative-bias-per-layer variation is computed inside UMT5Attention itself; kb-nano's L2/t5_attention.py is also self-attn only with the standard fir...

### Conformer relative-position rel_shift / `matrix_bd shift_relative_position_tensor` (12)

> Decomposes from `gather` + `bmm` + `softmax` (all in L1) but no kb-nano kernel implements the rel_shift index gymnastics. Common in audio/Conformer encoders.

- `fastspeech2_conformer` — Conformer-based TTS encoder with relative positional encoding (Transformer-XL style) attention plus duration/pitch/energy predictors and HiFi-GAN vocoder. The relative-position attention with learnable u/v biases and the conformer convolution modu...
- `granite_speech` — Granite Speech encoder is a Conformer with depthwise Conv1d + GLU + BatchNorm1d + Shaw relative positional embeddings (einsum-based pos_attn). kb-nano has no Conformer block and no Shaw relpos primitive.
- `granite_speech_plus` — GraniteSpeechConformerConvModule uses nn.BatchNorm1d (granite_speech/modeling_granite_speech.py:226) — kb-nano only has L1/batch_norm2d.py; BatchNorm1d would fall back to torch.nn.BatchNorm1d.
- `lasr` — Conformer convolution module (inherited from Parakeet) uses nn.BatchNorm1d as the in-conv normalization. kb-nano has BatchNorm2d, GroupNorm, FrozenBatchNorm2d but no BatchNorm1d. ReLU + nn.Conv1d (subsampling) are present.
- `parakeet` — ParakeetEncoderAttention adds learnable bias_u/bias_v (Transformer-XL style) and applies _rel_shift on relative positional logits before SDPA; this fused (matrix_ac + matrix_bd) addition pattern is not exposed as a kb-nano kernel. FastSpeech2Confo...
- `seamless_m4t` — Conformer relative-position attention + GLU activation + custom rel-pos shift are torch-native (matmul/softmax/Linear) but no fused kb-nano kernel; SeamlessM4TVariancePredictor uses standard Conv1d + LN.
- `seamless_m4t_v2` — Conformer relative-position attention has no fused kb-nano kernel (same as v1); rest is torch-native and composable.
- `sew_d` — DisentangledSelfAttention (DeBERTa-v2 style content/pos/c2p/p2c scoring) and ConvLayer relative-pos handling have no fused kb-nano kernel; rely on torch matmul/gather/softmax/Linear.
- `wav2vec2` — (see shard)
- `wav2vec2_bert` — Wav2Vec2BertSelfAttention is a Conformer attention with Transformer-XL relative bias (pos_bias_u/pos_bias_v) and a custom rel-shift; no kb-nano L2 attention implements relative bias attention.
- `wav2vec2_conformer` — Wav2Vec2ConformerSelfAttention implements Transformer-XL relative bias attention (linear_pos + pos_bias_u/v + rel-shift) and the ConvolutionModule needs nn.functional.glu; no kb-nano L2 covers either compute.
- `wavlm` — WavLMAttention uses gated relative position bias (T5-style buckets gated by GRU-style projection of hidden states) and calls F.multi_head_attention_forward; no kb-nano L2 covers this gated relative bias attention.

### Swin V1 `relative_position_bias_table` windowed attention (3)

> kb-nano `L2/swinv2_window_attention.py` is V2-specific (cosine attention + CPB MLP). V1 uses additive `relative_position_bias_table` indexed via `relative_position_index`.

- `donut_swin` — Swin V1 windowed attention uses relative_position_bias_table lookup (additive bias on attention scores). kb-nano L2/swinv2_window_attention.py is V2-only (cosine attention + CPB MLP) — different math.
- `maskformer_swin` — Original Swin V1 backbone (relative-position-bias window attention with shifted windows, drop-path, patch merging) — kb-nano only has Swin V2 (cosine attention with continuous position bias), which is a different attention formulation.
- `swin` — SwinSelfAttention uses standard scaled dot-product attention plus a learnable relative_position_bias_table (Embedding-like) indexed by relative_position_index — kb-nano has L2/swinv2_window_attention.py for V2 (cosine + CPB MLP) but no V1 (additiv...

### BatchNorm1d (no kb-nano BN1d wrapper; only BN2d) (4)

> `L1/batch_norm2d.py` exists; BN1d is a torch.nn primitive but no kb-nano wrapper.

- `hubert` — BatchNorm1d (used in HubertPositionalConvEmbedding when conv_pos_batch_norm=True) and Conv1d weight_norm parametrization have no kb-nano L1 equivalent. GroupNorm is available, BatchNorm2d is, but BatchNorm1d is missing.
- `levit` — MLPLayerWithBN applies nn.BatchNorm1d after every Linear; kb-nano has BatchNorm2d but no BatchNorm1d. LevitAttention/LevitAttentionSubsample also add a learned 2D positional attention_biases tensor to attention scores before softmax — kb-nano flas...
- `speecht5` — SpeechT5RelativePositionalEncoding requires custom shift-relative-pos handling (no fused kb-nano kernel); SpeechT5BatchNormConvLayer uses nn.BatchNorm1d which has no kb-nano kernel (only BN2d in L1).
- `superglue` — SuperGlueMultiLayerPerceptron uses nn.BatchNorm1d on transposed channels — kb-nano has L1/batch_norm2d.py but no BatchNorm1d. The Sinkhorn matching for the final assignment (in SuperGlueForKeypointMatching) is also a custom optimization-style op n...

### `torch.nn.utils.weight_norm` parametrization (4)

> Reparametrization decomposes from L1 ops but no kb-nano wrapper for `WeightNorm`.

- `kyutai_speech_to_text` — The codec_model (Mimi) uses MimiConv1d with streaming padding cache and weight-normalized Conv1d/ConvTranspose1d (audio codec primitives). kb-nano has Conv1d but no weight_norm parametrization or streaming padding cache. The Llama text decoder por...
- `mimi` — MimiVectorQuantization / MimiEuclideanCodebook / MimiResidualVectorQuantizer perform nearest-neighbour codebook lookup with EMA updates; no kb-nano L1 for VQ. The Conv1d/ConvTranspose1d wrappers use nn.utils.weight_norm and a runtime padding cache...
- `univnet` — (see shard)
- `vits` — All compute primitives (Conv1d, ConvTranspose1d, sigmoid, tanh, relu, leaky_relu, Linear, softmax) exist in kb-nano. However: (1) nn.utils.weight_norm parametrization on conv layers is a PyTorch-only utility with no kb-nano primitive; (2) fused_ad...

### Custom 2D / segment-aware position encoding (5)

> Fourier basis (perceiver), IndexMap segment-reduce (tapas), LSH bucketing (reformer), torch.fft (autoformer / fnet) — decomposable from arange + einsum + fft but no kb-nano kernel for these patterns.

- `autoformer` — AutoformerAttention replaces SDPA with autocorrelation: q/k FFT -> conjugate multiply -> inverse FFT -> top-k autocorrelation delay aggregation via torch.gather/roll. PyTorch supplies torch.fft.rfft/irfft, but kb-nano has no FFT primitive and no a...
- `fnet` — FNetBasicFourierTransform uses torch.fft.fftn (or scipy linalg.dft fallback). torch.fft.fftn exists in PyTorch, so a partial port is possible by calling it directly, but kb-nano has no L1 FFT primitive.
- `perceiver` — Fourier-based position encoding (PerceiverFourierPositionEncoding) builds frequency basis with linspace + cos/sin; not in kb-nano. The cross-attention pattern (latent_array attend to inputs) uses generic attention but kb-nano doesn't have a Percei...
- `reformer` — LSH/local chunked attention with bucketing and sort/unsort logic has no kb-nano equivalent; the underlying ops (matmul, softmax, gather) are torch primitives but no kb-nano L2 module wraps the LSH attention pattern.
- `tapas` — TapasEmbeddings.forward uses IndexMap / ProductIndexMap / reduce_min / gather (segment-aware reductions for cell-relative position) — these operate on per-token table indices and have no kb-nano L1 / L2 wrapper. The downstream cell-selection / agg...

### Block-sparse / sliding-window / chunked attention (12)

> BigBird (block-sparse), Longformer (sliding + global), Cohere2 (sliding window without flag in L2 wrapper), LED, ModernBERT, Pegasus-X, Nystromformer (Moore-Penrose pseudo-inverse). Decomposes via `attn_mask` injection but no kb-nano L2 wrapper for these routing patterns.

- `big_bird` — BigBirdBlockSparseAttention combines global tokens + sliding window + random-blocks attention via masked dense matmuls on grouped block tensors. PyTorch supplies the underlying ops (matmul, softmax, gather), but kb-nano has no block-sparse attenti...
- `bigbird_pegasus` — BigBirdPegasusBlockSparseAttention reuses the BigBird block-sparse pattern (global + sliding window + random blocks). No kb-nano equivalent. The decoder full attention is composable via whisper_attention.
- `cohere2` — Cohere2 extends Cohere with sliding-window attention (SWA), Gemma2-style hybrid local/global attention. Same compute primitives as Cohere; sliding window handled as attention mask in dense_attention.
- `led` — LED uses Longformer-style sliding-window self-attention (O(N*W)) with global attention on a subset of tokens. The chunked sliding-window matmul algorithm (_sliding_chunks_query_key_matmul, _sliding_chunks_matmul_attn_probs_value) and global-token...
- `longformer` — Sliding-chunk local attention with global-attention tokens — bespoke attention pattern requiring custom CUDA-style kernels (sliding-chunk QK matmul, padded diagonal mask handling).
- `mistral3` — Mistral3PatchMerger relies on torch.nn.functional.unfold (sliding-window patch extraction). No kb-nano L1 op for unfold; PyTorch has it natively. Vision tower + projector also depend on Pixtral encoder (not in shard).
- `modernbert` — Encoder self-attention with both RoPE on Q/K and sliding-window masking is not implemented in kb-nano L2; HF uses ALL_ATTENTION_FUNCTIONS path with sliding_window kwarg. kb-nano flash_attn_prefill supports causal sliding window but encoder_attenti...
- `modernbert_decoder` — Same GLU MLP issue as modernbert (packed Wi); attention itself maps to LlamaAttention with LayerNorm replacing RMSNorm. kb-nano L2/llama_mlp.py uses gate_proj/up_proj/down_proj layout, not chunked Wi.
- `moonshine` — MoonshineAttention requires RoPE before the SDPA call (encoder-decoder path with cross-attention), and head_dim padding to a multiple. kb-nano whisper_attention.py has no RoPE pre-application; llama-family attention.py is causal-only. Pure-torch p...
- `moonshine_streaming` — MoonshineStreamingFrameCMVN, MoonshineStreamingAsinhCompression, MoonshineStreamingCausalConv1d (with mask propagation), and MoonshineStreamingLayerNorm (unit-offset gamma) all compose via torch ops; no kb-nano L1 implements these specifically. Sl...
- `nystromformer` — Nystromformer's self-attention is a custom Nystrom approximation requiring iterative pseudo-inverse computation and depthwise conv2d residual — no kb-nano kernel approximates this.
- `pegasus_x` — PegasusXGlobalLocalAttention implements custom block-wise local attention with cross-attention to global tokens via einsum (BHGF/BHXF, BHGX/BHXF). The block-local + global-token pattern has no kb-nano kernel; would need a custom L1/L2 or torch fal...

### MoE with bespoke routing logic (9)

> The MoE expert kernel (`L1/moe_grouped_gemm.py`, `L2/shared_expert_moe.py`, etc.) exists but the routing layer needs custom logic (JetMoe Mixture-of-Attention, longcat identity-experts, NLLB-MoE conditional expert, switch_transformers token routing).

- `jetmoe` — JetMoeMoA wraps the attention computation with input-side routed q-projection and output-side routed combine, with per-expert weights stored in JetMoeParallelExperts (looped F.linear per expert). kb-nano has no equivalent for routing query project...
- `longcat_flash` — LongcatFlashExperts has zero-compute (Identity) experts (modular_longcat_flash.py:97, 134-135) — kb-nano L1/moe_grouped_gemm.py and L2/deepseek_moe.py have no pass-through identity-expert path; the routing+identity branch falls back to torch index...
- `nemotron_h` — NemotronHExperts uses non-gated up_proj+act+down_proj (not SwiGLU); kb-nano fused_experts/L2 mixtral_moe.py and shared_expert_moe.py both assume gated experts. NemotronHMoE adds optional fc1_latent_proj/fc2_latent_proj wrapping the experts. No kb-...
- `nllb_moe` — NllbMoeTop2Router implements fairseq-style capacity-based Top-2 routing (cumsum < capacity, batch-prioritized, gumbel sampling) — semantically different from kb-nano top-k routers (L1/topk_softmax.py, grouped_topk.py). Experts stored as ModuleDict...
- `olmo` — OlmoAttention applies torch.Tensor.clamp_ on Q/K/V if config.clip_qkv is set; kb-nano L2/attention.py:LlamaAttention has no clip_qkv option. clamp_ is a torch primitive but isn't exposed through the kb-nano attention class.
- `olmoe` — OlmoeAttention applies torch.Tensor.clamp_ on Q/K/V if config.clip_qkv is set; same gap as olmo. kb-nano L2/attention.py:LlamaAttention has no clip_qkv knob.
- `phimoe` — Mixtral-derived MoE LLM but with bespoke sparsemixer router (Heun's-third-order gradient estimator wrapping a custom torch.autograd.Function PhimoeMultiplier) and nn.LayerNorm in place of RMSNorm. The sparsemixer top-k has no kb-nano analog; stand...
- `qwen3_5` — Qwen3_5GatedDeltaNet uses split in_proj_qkv/in_proj_z/in_proj_b/in_proj_a Linear projections; kb-nano L2/qwen3_next_gdn_attention.py expects fused in_proj_qkvz/in_proj_ba and would need a small wrapper to consume split projections. The underlying...
- `qwen3_5_moe` — Inherits Qwen3_5GatedDeltaNet split projection layout (in_proj_qkv/z/b/a); same wiring gap as qwen3_5. MoE block uses Qwen3MoeSparseMoeBlock pattern (no shared expert) which is L2/qwen3_moe.py.

### Snake1d / xIELU / non-standard activation (7)

> Snake1d (sin-based, used in audio codecs); xIELU (apertus/arcee learnable α); squared-ReLU + non-gated MLP without an L2 wrapper (jais2-style).

- `apertus` — Uses xIELU activation (a learnable activation introduced in the Apertus paper) as ACT2CLS['xielu'](dtype=...). xIELU is not in PyTorch's standard activation set and has no kb-nano kernel.
- `arcee` — ArceeMLP is up_proj -> ACT2FN['relu2'] -> down_proj (no gate), i.e. a two-layer MLP with squared-ReLU activation. kb-nano provides L1/squared_relu.py and the fused L1/squared_relu_and_mul.py used in BitNet's gated MLP, but no L2 module for the bar...
- `dac` — Snake1d activation (x + (1/(alpha+eps)) * sin(alpha*x).pow(2)) has no fused kb-nano kernel; would fall back to torch elementwise sin/pow/mul/add.
- `pe_audio` — DacEncoder uses Snake1d activation: x + (1/alpha) * sin^2(alpha * x). Not in kb-nano L1; pure torch primitives but no fused kernel.
- `pe_audio_video` — PeAudioVideoMaskedGroupNorm uses torch.masked.mean/var for padding-aware GroupNorm; kb-nano has L1/group_norm.py but no masked variant. Plus AutoModel coupling for sub-encoders.
- `pe_video` — Inherits PeAudioVideoMaskedGroupNorm using torch.masked.mean/var; missing in kb-nano. Plus AutoModel coupling for video sub-encoder.
- `qwen3_omni_moe` — Code2Wav stack (CausalConvNet, CausalTransConvNet, ConvNeXtBlock, SnakeBeta-based decoder, AMP block, BigVGAN-style decoder) is a speech vocoder/codec that has no kb-nano equivalent. The transformer layers (Code2WavAttention, Code2WavMlp, Code2Wav...

### `nn.MultiheadAttention` black-box wrapper (3)

> Decomposes to separate Q/K/V Linear + sdpa + output Linear (covered by `L2/encoder_attention.py`), but the folders use the opaque `nn.MultiheadAttention` class and the audit was conservative.

- `aria` — AriaCrossAttention uses torch.nn.MultiheadAttention as a black box (with batch_first=True). PyTorch implements it via _scaled_dot_product_attention, so the compute is available via L1/dense_attention.py, but there is no kb-nano L2 wrapper for nn.M...
- `bridgetower` — Vision tower: CLIP-style ResidualAttention using nn.MultiheadAttention; text tower: BERT-style self/cross attention; cross-modal layers compose the two. Compute is plain MHA with QuickGELU MLP.
- `idefics2` — SigLIP-style vision encoder + Perceiver resampler with cross-attn + AutoModel text decoder + Idefics2Connector (linear projection). MultiheadAttentionPoolingHead uses torch.nn.MultiheadAttention which is plain SDPA. All primitives present.

### Mamba / SSM variant with custom mixer wiring (3)

> Zamba/Zamba2 use Mamba2 with bespoke `Zamba2RMSNormGated` and fused `in_proj` of `intermediate + conv_dim + num_heads`; MiniMax uses lightning attention.

- `minimax` — (see shard)
- `zamba` — Multi-head Mamba mixer (per-head x_proj_weight / dt_proj_weight / dt_proj_bias / A_log / D over n_mamba_heads) is not realised by any L2 mamba mixer; kb-nano supports single-head Mamba via L2/mamba_mixer.py and L2/jamba_mamba_mixer.py but a multi-...
- `zamba2` — Zamba2MambaMixer uses a custom layout (in_proj fused output with intermediate + conv_dim + num_heads, group-norm-gated activation via Zamba2RMSNormGated) that is not reproduced by kb-nano's L2/mamba2_mixer.py; kb-nano has the underlying L1 ops (ca...

### Vision encoder w/o existing kb-nano L4 (multimodal pipelines) (11)

> These VLMs combine a vision tower (often Qwen2-VL/2.5-VL or SigLIP-style) with a text decoder; all compute primitives are present but there is no end-to-end L4 pipeline tying them together. Composable in spirit; partial because the wiring isn't materialized.

- `cohere2_vision` — VLM pipeline: SigLIP-style vision tower + Cohere2 LM + multi-modal projector (pixel-shuffle + SwiGLU). All sub-modules map to existing kb-nano kernels (siglip_attention/mlp, cohere2 stack, llama_mlp pattern for projector).
- `deepseek_vl_hybrid` — Hybrid adds a SAM vision encoder branch with neck + DeepseekVLSamVisionProj (Conv2d twice) and a hybrid aligner (two Linear projections + GELU + Linear); all primitives exist (Conv2d/interpolate/Linear/GELU).
- `evolla` — (see shard)
- `exaone4_5` — EXAONE 4.5 is a Qwen2.5-VL multimodal model with vision encoder + 2D-RoPE + GQA. The vision tower components (ViT-style with 2D RoPE) and the multimodal projector pipeline have no kb-nano L4 wrapper or vision-encoder L2/L3 stack equivalent.
- `got_ocr2` — GOT-OCR-2 vision tower is the SAM ViT encoder, which uses decomposed relative positional embeddings (MViT-v2 / Shaw style) with custom einsum scoring. kb-nano has SAM3 vision attention (uses 2D RoPE, not decomposed relpos), so the SAM-style attent...
- `mllama` — MllamaPrecomputedAspectRatioEmbedding/MllamaPrecomputedPositionEmbedding are bespoke gated tile-aware embeddings. MllamaTextCrossAttention has tanh-gated residuals, q/k RMSNorm, and a custom cross-attention cache pattern (update only on first call...
- `mvp` — MvpAttention supports an attn_prompt argument that prepends learned prompt tensors to key_states/value_states within the attention call. kb-nano whisper_attention.py has no prompt-prepend hook. Pure-torch composition possible but no L2 covers it.
- `ovis2` — Ovis2VisionModel.forward calls nn.functional.gumbel_softmax (when tokenize_function='gumbel_argmax') which is not in kb-nano L1; it's a torch primitive but no fused kernel.
- `paddleocr_vl` — OCR-focused VLM combining Ernie4.5 (Llama-style) LLM + Qwen2.5-Omni attention + Qwen2-VL RoPE + SigLIP vision MLP + VideoLlama3 vision attention. All architectural pieces map to existing kb-nano L2/L3 (encoder/llama-style attention, SwiGLU MLP, vi...
- `pixtral` — PixtralRotaryEmbedding precomputes inv_freq per (h,w) position by interleaving freqs[::2] (h dim) and freqs[1::2] (w dim) and indexing by position_ids. kb-nano L1/vision_rotary_emb.py builds cos_sin from a 1D max_grid_size table indexed via grid_t...
- `qianfan_ocr` — QianfanOCRVisionAttention applies QianfanOCRVisionRMSNorm to Q and K before the per-head reshape (use_qk_norm flag). kb-nano L2/encoder_attention.py does not support QK-norm in the vision branch (LlamaAttention-style qk_norm exists in L2/attention...

### CNN / vision backbone structural mismatch (6)

> EfficientNet-style or BiT-style CNN where the building blocks differ from kb-nano's ConvNeXtV2 / EfficientNetV2 (different block ordering, depthwise stride/padding, weight standardization, etc.).

- `align` — EfficientNet vision encoder is structurally similar to but not the same as kb-nano's EfficientNetV2 building blocks (different block sequence, depthwise stride/padding handling, dropout-based stochastic depth). Compute primitives all exist in PyTo...
- `bit` — WeightStandardizedConv2d standardizes the conv weights (batch_norm on weight tensor) on every forward pass. Implemented with PyTorch's nn.functional.batch_norm + conv2d but no kb-nano L1 op fuses or wraps this; existing L1/conv2d.py is a vanilla c...
- `focalnet` — FocalNet uses Focal Modulation: a bespoke replacement for self-attention that combines depthwise Conv2d hierarchical context aggregation with gating. No kb-nano kernel implements this pattern, and there is no Focal-Modulation L4.
- `mobilevitv2` — MobileViTV2LinearSelfAttention is a custom O(N) attention: softmax over single-query channel + element-wise (key * scores) sum + relu(value)*context. No kb-nano kernel matches; pure torch primitives suffice.
- `pp_lcnet_v3` — PPLCNetV3LearnableAffineBlock applies learnable scale * x + learnable bias as a separate parameter pair. PPLCNetV3LearnableRepLayer reparameterizes a stack of (DWConv + LearnableAffineBlock) into a single conv at inference (RepVGG-style) with affi...
- `swiftformer` — SwiftFormerEfficientAdditiveAttention computes a single global query via softmax(Q @ w_g) -> sum, then proj(global * key) — this 'efficient additive attention' pattern is not a kb-nano kernel and falls back to torch ops (matmul + softmax + F.norma...

### Bespoke novel attention (decomposable, no kb-nano L2) (5)

> DeepSeek V4 (HCA + CSA + Indexer + HyperConnection); Doge (Dynamic Mask Attention); ESMFold (AlphaFold-style triangle attention); FocalNet (focal modulation); Funnel (pooled-query relative-pos); Informer (ProbSparse).

- `deepseek_v4` — DeepSeek V4 introduces novel Heavily Compressed Attention (HCA), Compressed Sparse Attention (CSA), Indexer for sparse attention, multi-rope-type Laguna-style rotary, HyperConnection routing, hash router, grouped output projection — none of these...
- `doge` — DogeAttention.prepare_dynamic_mask uses torch.topk + scatter to build a sparse attention mask each step (no kb-nano kernel for DMA). DogeCDMoE uses two nn.Embedding(num_experts, hidden_size) tables + matmul to materialize per-token expert weights,...
- `esmfold` — EsmFoldTriangleAttention is AlphaFold-style triangular attention (no kb-nano L2 wrapper). The compute decomposes from torch primitives (matmul + softmax + tensor reshape) but no kb-nano kernel implements the triangular pattern.
- `funnel` — Funnel Transformer uses pooled-query relative-position multi-head attention with per-block q/k stride pooling and a custom learned positional structure. The relative-attention structure (FunnelAttentionStructure with phi/pi/psi/omega bias terms) i...
- `informer` — Informer's defining contribution is InformerProbSparseAttention: random key sampling, sparsity measurement (max - mean) on Q-K_sample, top-u query selection, sparse attention only on top-u queries with cumsum-based context for the rest. This algor...

### DETR-family deformable / cross-modal grounding (3)

> kb-nano has `L1/rtdetrv2_deformable_attention.py` for the V2-specific path; older deformable_detr / grounding_dino use a different deformable variant + bi-modal cross-attention without kb-nano L2 wrappers.

- `deformable_detr` — Deformable DETR uses MultiScaleDeformableAttention via grid_sample + standard transformer enc-dec. kb-nano has L1/grid_sample.py and L1/L2 rtdetrv2_deformable_attention.py (used for the same family).
- `grounding_dino` — MultiScaleDeformableAttention compute matches L1/rtdetrv2_deformable_attention.py, but GroundingDinoBiMultiHeadAttention (text-vision cross-attention with separate Q/K/V/projection for both modalities), GroundingDinoFusionLayer, and GroundingDinoC...
- `maskformer` — Instance-segmentation model with bespoke FPN pixel decoder, DETR-style decoder, Hungarian matcher, dice/focal losses, and a small mask-head ConvNet — many components have no kb-nano kernel.

### Encoder-decoder seq2seq + non-trivial bias / temporal head (3)

> BART/Pegasus-style models with custom temporal heads (musicgen / pop2piano), or seq2seq with extra bias in the cross-attention path.

- `musicgen` — Per-codebook embedding tables summed at the input + sinusoidal positional embeddings + delay-pattern in autoregressive generation. The attention itself (self + cross) maps to whisper_attention.py, but the codebook input layer and audio_encoder dep...
- `musicgen_melody` — Same as musicgen — per-codebook embedding sum and conditional generation wiring not in kb-nano. Attention itself composes from whisper_attention.py.
- `prophetnet` — ProphetNet uses NgramSelfAttention with stream-level n-gram prediction, custom relative-position-bucket bias projected via a learned Linear(hidden -> num_buckets*num_heads), and the self-attention attends to a main stream + n predict streams. No k...

### Per-token / additive attention bias (DeBERTa, LayoutLM, RoFormer) (3)

> kb-nano flash-attn kernels do not support additive attention bias (DeBERTa-v2 bucket relative pos, LayoutLMv3 rel_pos + rel_2d_pos, RoFormer encoder-RoPE). Decomposable from `dense_attention.py` but no L2 wrapper.

- `deberta_v2` — DeBERTa-v2 extends DeBERTa with bucket-style relative position attention + ConvLayer; same compute primitives (linear/matmul/softmax/layer_norm/conv1d/embedding) all available.
- `layoutlmv3` — LayoutLMv3SelfAttention adds rel_pos + rel_2d_pos to attention scores before softmax (additive bias not supported by kb-nano flash kernels) and uses the CogView numerical-stability softmax variant (not in kb-nano L1/softmax.py). Otherwise the rest...
- `roformer` — EncoderSelfAttention in kb-nano (L2/encoder_attention.py) does not apply rotary position embeddings inside the bidirectional encoder; would need a small variant that calls L1/rotary_emb.py on Q/K (and optionally V) before SDPA. The L1 RoPE op exis...

### Time-series-specific (FFT autocorrelation, ProbSparse, hierarchical patches) (4)

> TimeSeriesTransformer / PatchTST / PatchTSMixer / Informer / Autoformer use bespoke time-series structures.

- `patchtsmixer` — PatchTSMixerBatchNorm uses nn.BatchNorm1d. kb-nano has L1/batch_norm2d.py but no batch_norm1d.py; would need an additional L1 op (trivial wrapper around F.batch_norm).
- `patchtst` — PatchTSTBatchNorm uses nn.BatchNorm1d (L1 only has batch_norm2d.py). Needs new L1 op or torch fallback.
- `timesfm` — TimesFM forecasting decoder uses learnable per-dimension query scaling (softplus(scaling) * 1.4427/sqrt(d_h)) inside attention — a custom op with no PyTorch built-in equivalent; the MLP also embeds its layernorm and a paddings-mask multiplication...
- `timesfm2_5` — TimesFM 2.5 keeps the per-dim learnable softplus query-scaling from v1 plus adds Q/K RMS-norms and standard NeoX RoPE; the per-dim scaling is the same op-level gap and has no kb-nano equivalent.

### Pre-projection position addition (DETR `with_pos_embed`) (1)

> TableTransformer / DETR add object_queries / spatial_position_embeddings *before* q_proj/k_proj. Decomposable but the sequencing is non-standard for kb-nano L2 wrappers.

- `table_transformer` — TableTransformerAttention adds object_queries to hidden_states and spatial_position_embeddings to key_value_states *before* the q_proj/k_proj projections (DETR's 'with_pos_embed' pattern), then runs bmm-flattened cross-attn — kb-nano has no DETR-s...

### XLNet permutation language modeling (two-stream relative attention) (1)

> XLNetRelativeAttention is Transformer-XL two-stream attention with `g`/`h` streams, segment embeddings, and a custom rel_shift. Bespoke; decomposable but no kb-nano L2 equivalent.

- `xlnet` — XLNetRelativeAttention is two-stream Transformer-XL relative attention with permutation-language-modeling g/h streams, segment embeddings, and a custom rel_shift; bespoke compute with no kb-nano L2 equivalent.

### OCR / table-parsing layouts (2)

> SLANet/SLANeXt use bespoke document-table head; pp_formulanet has custom math-OCR head.

- `slanet` — SLANetAttentionGRUCell (and SLANetSLAHead) use nn.GRUCell — torch handles the recurrence but kb-nano has no GRU kernel (only L1/lstm.py); rest of pipeline (Conv2d, depthwise conv, hardswish, Linear) is composable.
- `slanext` — SLANeXtAttentionGRUCell uses nn.GRUCell which has no kb-nano kernel (kb-nano only ships L1/lstm.py for RNN family).

### Other unique gaps (11)

Folders whose gap doesn't fit any of the groups above. Each has a
one-off rationale.

- `ernie4_5_vl_moe` — VL extension of Ernie4_5_Moe: Qwen2-VL/2.5-VL vision tower + Ernie4_5 MoE text decoder + variable-resolution resampler. All compute uses kernels already present (vision_attention, vision_mlp, attention.py, mixtral_moe, llama_mlp).
- `groupvit` — GroupViT = CLIP-text + GroupViT vision (with token-grouping cross-attention and Gumbel softmax assign). Standard linear + softmax + LN + GELU MLP composition. clip_attention + clip_mlp cover the text side; vision uses standard MHA with extra group...
- `lightglue` — Keypoint-matching graph network with depth-confidence early stopping, point-pruning, log-double-softmax assignment, and a SuperPoint detector backbone — no kb-nano L4 / L2 covers this domain.
- `moshi` — MoshiFlexibleLinear (per-codebook 3D weight bank with torch.index_select + batched matmul) is a custom op without kb-nano equivalent; closest is moe_grouped_gemm but the routing is different (codebook index, not top-k). The depth decoder relies on...
- `mpnet` — Relative position bias added to attention scores: kb-nano flash_attn/dense_attention have no additive-bias parameter, and t5_attention.py is T5-specific (not the same bucket function used by MPNet which has its own compute_position_bias). Addition...
- `nanochat` — NanoChatRMSNorm = Llama4TextL2Norm = pure F.normalize(p=2) without learned weight; kb-nano L1/l2_norm.py covers F.normalize but is used in RWKV7 context, not as a transformer pre-norm. Custom rotate_half((x2, -x1) order) means the standard kb-nano...
- `sam3_lite_text` — RepMixer/MobileOneBlock structure relies on nn.BatchNorm2d (kb-nano L1/batch_norm2d.py exists but the depthwise-conv reparameterization wrapper has no fused kb-nano kernel) and nn.functional.interpolate handled in pure torch.
- `stablelm` — (see shard)
- `unispeech` — Wav2Vec2PositionalConvEmbedding uses nn.utils.weight_norm-parametrized Conv1d (weight reparameterization not exposed in kb-nano L1/conv1d.py). Wav2Vec2GumbelVectorQuantizer uses nn.functional.gumbel_softmax — no kb-nano kernel. Wav2Vec2Attention i...
- `unispeech_sat` — Same as unispeech: Wav2Vec2-style weight-norm Conv1d positional embedding, multi-stage Conv1d feature encoder with group-norm, BART-style Wav2Vec2Attention, and Gumbel-softmax vector quantizer are not implemented as kb-nano kernels.
- `zoedepth` — LogBinomialSoftmax uses torch.lgamma to compute log binomial coefficients (log_binom = lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1)); torch supplies lgamma but kb-nano has no L1 lgamma kernel.

<!-- Total: 171 partial folders documented -->
