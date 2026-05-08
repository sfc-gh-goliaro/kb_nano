# Audit verification guidelines (hard rules from past mistakes)

These rules exist because prior audit passes — and several of my own attempts —
took shortcuts that produced wrong rationales, wrong status calls, and false
"100% verified" claims. Read this before doing any verification work.

## Rule 0: HARD NO on filename-only or regex-only verification

**A folder is not verified until you have actually opened the relevant files.**

Specifically forbidden patterns (these have produced documented errors):

1. **"The class name says X so it must be X."** Filename ≠ implementation.
   - Example failure: agent claimed `maskformer_swin` was correctly classified
     because "Swin V1" — but didn't open `L2/swinv2_window_attention.py` to
     verify the V1-vs-V2 distinction. Actual verification requires reading
     the kb-nano file and confirming "cosine attention + CPB MLP" (V2) vs
     "additive `relative_position_bias_table`" (V1).
2. **"grep matched, so the kernel exists."** A grep hit on a kernel name doesn't
   tell you what compute the kernel implements. You must open the file and
   read the `forward` method.
3. **"The shard rationale says X."** v4-era shard rationales were optimistic.
   They've been found wrong in:
   - `paddleocr_vl` — PaddleOCRDecoderLayer was a wiring class tagged `[compute]`,
     making the renderer treat it as missing primitive when it isn't
   - `esmfold` — claimed "no kb-nano L2 wrapper" but kb-nano has
     `L2/alphafold3_triangle_attention.py` with literally identical signature
     and line-by-line-identical forward method
   - `gpt_neox`/`gptj`/`codegen` — claimed "Maps cleanly" but missed
     parallel-residual + LayerNorm-decoder + non-SwiGLU MLP gaps
   - `bloom` — claimed "supported via attn_mask" but kb-nano has no
     first-class alibi parameter; it's a fragile workaround
4. **"Agent says verified."** Agent fan-outs were caught taking shortcuts:
   - Batch 1 + Batch 3 of the 6-agent pass claimed "100% accuracy across
     24 folders" with shallow file reads (2-6 kb-nano files per folder;
     load-bearing files unopened)
   - **Any "100% accuracy" claim from an agent on >5 folders is suspicious
     by default** — verify a sample personally before trusting

## Rule 1: For every folder, open both sides

Each folder verification requires:

1. **HF side.** Open `/tmp/hf_transformers_pinned/src/transformers/models/<folder>/modeling_<folder>.py` (and `modular_<folder>.py` if both exist). Read the forward method of the class(es) the audit shard called out as `[compute]`. Reading 30-50 lines around the class is enough; full-file reads are unnecessary.
2. **kb-nano side.** For every `L?/<file>.py` cited by the shard, open the file. Read its `__init__` and `forward`. Verify the compute matches what HF does.
3. **Coverage gap check.** `ls /home/olu/kb_nano/tasks/baseline/L1/`, `L2/`, `L3/`, `L4/` and ask: is there a kb-nano file with related semantics that the audit didn't cite? (esmfold-style false negative.)

## Rule 2: Status decisions require evidence on both sides

For each folder, decide:

| call | requires |
|---|---|
| `kb_nano_l4` | A kb-nano `L4/<file>.py` whose docstring header explicitly targets the same model family. Open the L4 file, confirm. |
| `composable` | Every `[compute]` class in the audit shard maps to an existing kb-nano L1/L2/L3 kernel that **implements the same compute pattern** (not just same name). |
| `partial` | At least one `[compute]` class needs a torch primitive for which kb-nano has no matched wrapper. The folder runs in PyTorch (not unsupported) but a tuned kb-nano kernel is missing. |
| `unsupported` | At least one `[compute]` class needs (a) a custom CUDA kernel from `kernels-community/*`, (b) an external library (timm, natten, detectron2, xlstm), or (c) genuinely novel compute with no `torch.*` equivalent. |

## Rule 3: Twelve recurring kernel-mapping rules

These are the prior audit's 12 rules, restated to be unambiguous:

1. **Filename ≠ implementation.** Open the file. Confirm `__init__` + `forward`.
2. **Inheritance ≠ structural identity.** A class inheriting `LlamaAttention` may override `forward`; read the child body.
3. **silu vs silu_and_mul.** SwiGLU MLPs (`act(gate) * up`) use `L1/silu_and_mul.py`, NOT bare `L1/silu.py`. Same for `gelu_and_mul`, `squared_relu_and_mul`.
4. **Norm variants.** RMSNorm (`L1/rms_norm.py`) ≠ T5LayerNorm (`L1/t5_layer_norm.py`, no centering) ≠ BitNetRMSNorm (`L1/bitnet_rms_norm.py`) ≠ GemmaRMSNorm (`L1/gemma_rms_norm.py`, weight + 1) ≠ LayerNorm (`L1/layer_norm.py`, with centering).
5. **RoPE variants.**
   - Standard NeoX: `L1/rotary_emb.py:RotaryEmbedding` — rotates the **full** head_dim
   - YaRN (DeepSeek V3, GPT-OSS, derivatives): `L1/yarn_rotary_emb.py`
   - M-RoPE (Qwen-VL family): `L1/mrope.py`
   - Vision 2D RoPE: `L1/vision_rotary_emb.py`
   - DINOv3 RoPE: `L1/dinov3_rope.py`
   - Sinusoidal (BART/Whisper): `L1/sinusoidal_embed.py`
   - **partial-rotary (`partial_rotary_factor < 1.0`)**: standard `RotaryEmbedding` rotates the FULL head; partial-rotary needs either external `q_rot/q_pass` slicing in user code or `Gemma4ProportionalRotaryEmbedding` (specific to Gemma4 proportional pattern). Folders that use partial-rotary go to `partial`.
   - **Interleaved RoPE (GLM-style `cos[..., :d//2].repeat_interleave(2, dim=-1)`)**: kb-nano standard is `rotate_half`-style NeoX, not interleaved → `partial`. EXCEPT when used inside `L2/deepseek_mla_attention.py` which supports interleaved via `is_neox_style=False`.
6. **Attention backends.**
   - Decoder-only (Llama, Qwen, Mistral, GPT-OSS, Mixtral, etc.): `L2/attention.py:LlamaAttention`
   - Encoder family (BERT, ViT, BEiT, BERT-derived): `L2/encoder_attention.py`
   - CLIP-text: `L2/clip_attention.py` (separate q/k/v Linear, QuickGELU MLP)
   - SigLIP: `L2/siglip_attention.py`
   - Whisper / BART-style: `L2/whisper_attention.py` (3 sibling classes: encoder/decoder/cross — but **only with merged QKV layout**)
   - T5: `L2/t5_attention.py:T5SelfAttention` — **self-attention only**; `T5LayerCrossAttention` is NOT covered
   - DeepSeek MLA: `L2/deepseek_mla_attention.py` — supports interleaved RoPE via `is_neox_style` flag
   - Deformable DETR-V2: `L1/rtdetrv2_deformable_attention.py` + `L2/rtdetrv2_deformable_attention.py`
   - SwinV2 windowed: `L2/swinv2_window_attention.py` — **V2 cosine + CPB only, not V1 RPB**
   - Vision (Qwen2-VL etc.): `L2/vision_attention.py`
   - AlphaFold3 triangle: `L2/alphafold3_triangle_attention.py` (signature `(c_in, c_hidden, no_heads, starting, inf)`) — **same signature as ESMFold's AF2 variant**; the audit was wrong to say "no kb-nano triangle attention"
   - SAM3 RoPE: `L1/sam3_rope_attention.py`
   - DSA indexer (DeepSeek Sparse Attention): `L2/sparse_attn_indexer.py` — **NOT BigBird block-sparse, different algorithm**
7. **MoE expert kernels.**
   - MXFP4 (GPT-OSS): `L1/mxfp4_moe.py`
   - Standard fused-MoE: `L1/moe_grouped_gemm.py` + `L2/fused_experts.py` + `L2/mixtral_moe.py`
   - Shared-expert pattern (DeepSeek/Kimi/Qwen3-Next): `L2/shared_expert_moe.py`
   - Gemma4 routing: `L1/gemma4_routing.py`
   - GPT-OSS top-k router: `L2/gpt_oss_moe.py`
   - JetMoe Mixture-of-Attention (MoA): **no kb-nano kernel** (only MoE-MLP exists)
8. **Linear variants.**
   - BitNet → `L1/bitnet_linear.py`
   - FP8 → `L1/fp8_linear.py`
   - Standard → `L1/linear.py`
9. **MLP variants.**
   - SwiGLU `gate_up → SiluAndMul → down`: `L2/llama_mlp.py`
   - Two-layer `fc1 → activation → fc2`: `L2/encoder_mlp.py` (BERT-family, GELU) / `L2/clip_mlp.py` (CLIP-text, QuickGELU) / `L2/siglip_mlp.py` (SigLIP, GELU) / `L2/whisper_mlp.py` (BART/Whisper) / `L2/t5_dense.py` (T5)
   - **Vision encoder MLPs (ViT, BEiT, DINOv2 plain) are fc1+GELU+fc2 — NOT llama_mlp**
   - **GPT-NeoX/GPT-J/CodeGen use fc1+GELU+fc2, NOT SwiGLU**
10. **Wiring vs compute.** Skip from listing entirely:
    - `*PreTrainedModel`, `*Config`, `*Output` (the dataclass kind, e.g., `BaseModelOutput*`), `*Cache`, `*Mixin`, processor / tokenizer classes
    - **NOTE**: BERT-family `*SelfOutput` and `*Output` are `nn.Module` sublayers (Linear+Dropout+LayerNorm), NOT ModelOutput dataclasses — these ARE legitimate compute classes
    
    Mark as `[wiring]` (no kb-nano files): `*Block`, `*Layer`, `*DecoderLayer`, `*EncoderLayer`, `*Encoder`, `*Decoder`, `*Model`, `*ForXxx`, `*Pooler`, `*Head`, `*Embeddings` when just sum of token+pos+norm.
11. **Sibling-class attention wrapper.** When the same folder has both `*SelfAttention` AND bare `*Attention` (BERT/Beit/ViT pattern), the bare `*Attention` is a wiring wrapper around `*SelfAttention` + `*SelfOutput`. Mark bare `*Attention` as `[wiring]`. The kernel is on `*SelfAttention`.
12. **Skip these classes entirely.** Never list as `[compute]`: `*PreTrainedModel`, `*Config`, `*Output` (dataclass), `*OutputWith*`, `*Cache`, `*Mixin`, processor/tokenizer classes.

## Rule 4: Common gap patterns and the rule that applies

These are the consistency rules that resolve cross-folder ambiguity. **Apply uniformly.**

| pattern | folders go to | because |
|---|---|---|
| `partial_rotary_factor < 1.0` (default <1) | partial | kb-nano `L1/rotary_emb` rotates full head; needs external slicing |
| Interleaved RoPE (GLM-family `repeat_interleave(2)`) | partial | kb-nano standard is rotate_half NeoX, not interleaved |
| LayerNorm in decoder LLM (phi-style, not RMSNorm) | partial | kb-nano `L2/attention.py` expects RMSNorm |
| Parallel-residual `attn + mlp + h` (GPT-J, GPT-NeoX, CodeGen) | partial | no kb-nano L2 wraps this layer pattern |
| ALiBi via `build_alibi_tensor` | partial | kb-nano flash kernels have no first-class alibi parameter; attn_mask injection is a fragile workaround |
| AutoBackbone / `load_backbone()` | partial | no kb-nano AutoBackbone shim. (Folders that route to timm/detectron2 escalate to unsupported) |
| BART-style separate q/k/v + (seq, batch, dim) | partial | kb-nano `L2/whisper_attention.py` is QKVParallelLinear merged-QKV layout |
| T5 cross-attention (`T5LayerCrossAttention`) | partial | kb-nano `L2/t5_attention.py:T5SelfAttention` is self-attn only |
| Conformer rel_shift / `matrix_bd shift_relative_position_tensor` | partial | no kb-nano Conformer wrapper |
| Swin V1 `relative_position_bias_table` | partial | kb-nano `L2/swinv2_window_attention.py` is V2 cosine + CPB only |
| BatchNorm1d | partial | kb-nano has only `L1/batch_norm2d.py` |
| `torch.nn.utils.weight_norm` parametrize | partial | no kb-nano weight_norm wrapper |
| `torch.fft.{rfft, irfft, fft, fftn}` | partial | no kb-nano FFT kernel |
| Custom 2D pos enc (Fourier, IndexMap, LSH) | partial | no kb-nano kernel |
| Snake1d / xIELU / non-standard activation | partial | no kb-nano wrapper for these activations |
| `nn.MultiheadAttention` black-box wrapper | partial (composable in spirit; partial because audit was conservative) | decomposes to L1 ops |
| `kernels-community/*` CUDA kernel | unsupported | external CUDA, no torch.* fallback |
| `timm.create_model` | unsupported | external lib |
| `detectron2` | unsupported | external lib |
| `natten.functional` | unsupported | external CUDA library |
| Bespoke autograd Function with novel math | unsupported | no torch.* equivalent |

## Rule 5: How to log verification

For every folder you verify, add an entry to a verification log JSON with:

```json
{
  "<folder>": {
    "hf_files_opened": ["modeling_<folder>.py:<line_range>", ...],
    "kb_nano_files_opened": ["L?/<file>.py", ...],
    "compute_class_in_HF_verified": "<class name + line>",
    "kb_nano_kernel_match": "<kb-nano file that matches, or 'no match'>",
    "status_call": "kb_nano_l4 | composable | partial | unsupported",
    "rationale_one_line": "<verified rationale>",
    "issues_with_prior_audit": "<discrepancies found, if any>"
  }
}
```

**No entry without opened files.** If you can't open both sides, mark
`"status_call": "uncertain"` and explain.

## Rule 6: Specific findings to remember

Documented errors that any future audit must NOT repeat:

1. **paddleocr_vl** (post-v12 fix): a wiring class tagged `[compute]` triggers
   the renderer's "Missing primitive: " path. Always re-tag wiring classes as
   `[wiring]`.
2. **esmfold** (post-v12 fix): kb-nano has `L2/alphafold3_triangle_attention.py`
   with the SAME `(c_in, c_hidden, no_heads, starting, inf)` signature as
   HF's ESMFold variant. AF3 vs AF2 reference is debatable; kernel existence
   is not. Don't claim "no kb-nano triangle attention".
3. **gpt_neox / gptj / codegen** (post-v12 fix): "Maps cleanly to L2/attention
   + L2/llama_mlp" is wrong. These use parallel-residual + LayerNorm + fc1+
   GELU+fc2 — none of which match kb-nano's RMSNorm + sequential-residual +
   SwiGLU L2/attention.py.
4. **bloom**: "Supported via attn_mask" understates. kb-nano flash kernels
   have no first-class alibi parameter. Workaround is fragile.
5. **maskformer_swin**: V1, not V2. Don't trust agent claims that opened
   only `L1/conv2d.py` etc. — verify against `L2/swinv2_window_attention.py`.

## Rule 7: When you find a status flip, propagate everywhere

A status flip means updating:

1. `_reaudit_final_v11.json` `final_status` field
2. `audit_evidence.csv` `final_status_v11` column
3. The shard markdown's `**status**:` field
4. Re-render `tools/md_to_tex.py` to regenerate `hf_coverage_rows.tex`
5. Verify triple cross-check (json/csv/tex) returns 0 mismatches

If a flip changes the headline numbers, update README.md and CAVEATS_AND_METHODOLOGY.md accordingly.

## Rule 8: Honesty about session capacity

A full per-folder verification of all 447 folders is ~15-25 hours of focused
work at 2-3 minutes per folder. A single session can realistically cover
30-80 folders. Track progress in the log JSON and be explicit about what's
verified vs. carried-forward from prior audits.
