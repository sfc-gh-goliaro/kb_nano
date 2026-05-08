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
| AutoBackbone / `load_backbone()` routing | **consistent** rule (partial, not unsupported) — see CONSISTENCY_AUDIT.md | Originally inconsistent across slices; resolved with the rule "infrastructure routing, not a compute gap". |
| ALiBi via `attn_mask` injection | **consistent** rule (partial) — see CONSISTENCY_AUDIT.md | Originally split between composable and unsupported across shards; resolved at partial. |
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
# -> writes both MENTOR_REVIEW_full_audit.tex and hf_coverage_rows.tex
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
