# Per-folder paper-grade audit task

You are auditing a batch of HuggingFace Transformers folders to map every
HF compute class to the kb-nano kernel file(s) that would actually run that
compute. This is for a paper appendix; output must be paper-grade rigorous,
NOT regex-pattern-matched.

## Inputs

- HF source pinned at `/tmp/hf_transformers_pinned/src/transformers/models/<folder>/`
  (commit `da6c53e4`). For each folder, prefer `modular_<folder>.py` (explicit
  inheritance) over `modeling_<folder>.py` (generated, expanded).
- kb-nano file inventory at `/home/olu/kb_nano/tasks/baseline/{L1,L2,L3,L4}/`.
- This guidelines doc + the LESSONS_LEARNED.md.

## Mandatory rules (read carefully — these are ALL recurring mistakes)

1. **Filename ≠ implementation.** Before listing kb-nano file `X` in a row,
   open `X` with the Read tool and confirm its `__init__` + `forward` actually
   implement the same compute as the HF class. Docstring grep is NOT enough.
2. **Inheritance ≠ structural identity.** When `XAttention(LlamaAttention)`
   inherits, check if X overrides `forward` or `__init__`. If yes, the parent's
   structure may not survive — read the child's body.
3. **No silu vs silu_and_mul confusion.** SwiGLU MLPs (`act(gate) * up`) use
   `L1/silu_and_mul.py` (or `gelu_and_mul.py`, `squared_relu_and_mul.py`),
   NOT bare `silu.py`. The fused form is what kb-nano actually runs.
4. **Norm variants matter.** T5 uses `L1/t5_layer_norm.py` (no centering).
   BitNet uses `L1/bitnet_rms_norm.py`. Gemma uses `L1/gemma_rms_norm.py`.
   Standard Llama-family uses `L1/rms_norm.py`. Don't confuse them.
5. **RoPE variants matter.** YaRN-NeoX (GPT-OSS) and YaRN-DeepSeek share
   `L1/yarn_rotary_emb.py` (two classes inside). M-RoPE for Qwen-VL uses
   `L1/mrope.py`. Vision 2D RoPE uses `L1/vision_rotary_emb.py`. DINOv3 RoPE
   uses `L1/dinov3_rope.py`.
6. **Attention backends matter.** Decoder-family (Llama, Qwen, Mistral, GPT-OSS,
   Mixtral) uses `L2/attention.py:LlamaAttention`. Encoder-family (BERT, ViT,
   BEiT, CLIP-like) uses `L2/encoder_attention.py:EncoderSelfAttention`. CLIP
   uses `L2/clip_attention.py`. SigLIP uses `L2/siglip_attention.py`. Whisper
   uses `L2/whisper_attention.py` (3 sibling classes for encoder/decoder/cross).
   T5 uses `L2/t5_attention.py` (relative position bias). MLA (DeepSeek) uses
   `L2/deepseek_mla_attention.py`. Deformable attention uses
   `L1/rtdetrv2_deformable_attention.py`.
7. **MoE expert kernels.** MXFP4 quantized (GPT-OSS) uses `L1/mxfp4_moe.py`.
   Standard fused-MoE uses `L1/moe_grouped_gemm.py`. Shared-expert pattern
   (DeepSeek, Kimi-Linear, Qwen3-Next) uses `L2/shared_expert_moe.py`.
   Arch-specific MoE blocks: `L2/<arch>_moe.py` (mixtral, jamba, gpt_oss,
   gemma4, deepseek, kimi, llama4, qwen3, qwen3_next).
8. **Linear variants.** BitNet uses `L1/bitnet_linear.py:{BitLinear,
   BitLinearMerged}` (1.58-bit). Standard linear uses `L1/linear.py:Linear`
   (or `BMM`/`Matmul` for non-parametric). FP8 linear uses `L1/fp8_linear.py`.
9. **MLP variants.** SwiGLU pattern (`gate_up → SiluAndMul → down`) uses
   `L2/llama_mlp.py:LlamaMLP` (or arch-specific override). Two-layer
   `fc1 → activation → fc2` is NOT SwiGLU and uses `L2/clip_mlp.py`,
   `L2/siglip_mlp.py`, `L2/whisper_mlp.py`, `L2/encoder_mlp.py`, or
   `L2/t5_dense.py` depending on family. Vision encoder MLPs (ViT, BEiT,
   DINOv2, etc.) are fc1+GELU+fc2 — NOT llama_mlp.py.
10. **Wiring classes mark as `composes`.** Classes whose forward is pure
    composition of other classes' compute (no own kernel): `*DecoderLayer`,
    `*EncoderLayer`, `*Block`, `*Encoder`, `*Decoder`, `*Model`, `*ForXxx`,
    `*Pooler`, `*Head`, `*Embeddings` (when just sum of token+pos+norm).
    Use `composes` (not a kb-nano file) for these — they don't have their
    own kernel; they call others.
11. **Sibling-class attention wrapper.** When the same folder has both
    `*SelfAttention` AND `*Attention` classes (e.g. Beit/ViT/BERT pattern),
    the bare `*Attention` is a wiring class around `*SelfAttention`. Mark
    bare `*Attention` as `composes`. The actual kernel is on the
    `*SelfAttention` row.
12. **Skip these classes entirely.** Never list them in output:
    `*PreTrainedModel`, `*Config`, `*Output`, `*OutputWith*`, `*Cache`,
    `*Mixin` (e.g. GenerationMixin), processor / tokenizer classes.

## Per-folder process — DEEP ANALYSIS, NO SHORTCUTS

For each folder in your batch, you MUST:

1. **Read BOTH `modular_<folder>.py` AND `modeling_<folder>.py` if both exist.**
   Modular shows inheritance; modeling shows the expanded compute. Cross-reference
   them — if a class is `pass` in modular but has a custom forward in modeling,
   the modeling forward is the truth. Do NOT skip the modeling file even when
   modular is present.

2. **Read every class's `__init__` AND `forward` in full.** Inheritance can
   be misleading; look at actual sub-modules instantiated and ops called.
   Don't shortcut by reading just the class declaration line.

3. **For EVERY kb-nano file you list, you MUST `Read` it** (open and read
   the contents — `__init__` + `forward` at minimum, often more). Confirm:
   - Same activation function (silu vs silu_and_mul, gelu vs quickgelu, etc.)
   - Same norm variant (rms vs t5 vs layer)
   - Same projection layout (separate q/k/v vs fused qkv)
   - Same attention type (causal vs non-causal vs sliding-window vs MLA)
   - Same op sequence in forward
   If you cannot verify the file matches by reading it, do NOT list it —
   either find a different file or mark `composes`. Filename match is
   NOT sufficient.

4. **Inspect each L2/L3 file's L1 imports.** If you map an HF class to an
   L2 file, the user expects every L1 op the HF class invokes to be
   contained in that L2 file's `from ..L1 import` lines. Verify this.
   Example: `LlamaAttention -> L2/attention.py` is only correct if
   `L2/attention.py` imports the L1 ops Llama actually uses (linear,
   rms_norm, rotary_emb, attention_impl which uses flash_attn).

5. Filter out the skip classes (rule 12).

6. For each remaining class, determine the role and the kb-nano kernel(s)
   the compute would invoke. Apply rules 1–11.

7. **Document inheritance chain** — list every cross-arch parent class
   (e.g., `AfmoeMLP(Qwen2MoeMLP)`). This is informative for the reader.

8. **Self-check before reporting.** For each row in your output:
   - Did you READ the kb-nano file (not just glance at filename)?
   - Did you READ the HF source class's forward (not just signature)?
   - Are the L1 deps of the kb-nano file consistent with what the HF class needs?
   If ANY of these are no, redo that row.

9. **Don't take shortcuts.** Each subagent runs once and produces 28 rows.
   The user explicitly asked for paper-grade rigor. If your context is
   getting tight, prioritize reading the modular/modeling source over
   rendering more output. Skip a row (with explanation) rather than
   submitting an unread / unverified mapping.

## Output format

Return ONE JSON object covering all folders in your batch:

```json
{
  "<folder1>": {
    "src_file": "modular_<folder1>.py" or "modeling_<folder1>.py",
    "classes": [
      {
        "name": "<ClassName>",
        "line": <int line in src_file>,
        "bases": ["<ParentClass>", ...],
        "kb_nano_files": ["tasks/baseline/L1/foo.py", ...] or [],
        "rationale": "<one-line why this maps here, including any caveats>"
      },
      ...
    ],
    "inheritance_summary": ["<Class>(<CrossArchParent>)", ...],
    "evidence_lines": ["<folder1>/modular_<folder1>.py:<line>", ...]
  },
  "<folder2>": {...},
  ...
}
```

- `kb_nano_files` is `[]` for `composes` (wiring classes); otherwise list the
  files you VERIFIED by reading them.
- `rationale` should explain key inheritance / variant decisions. Examples:
  "inherits LlamaRotaryEmbedding; standard RoPE",
  "QFormer-style MHA (BERT-derived); routes to encoder_attention",
  "wiring class around <ClassName>SelfAttention".

## Reporting honesty

- If you cannot find a kb-nano file that matches a class's compute, leave
  `kb_nano_files: []` and explain in rationale (the audit will mark it as
  composes or partial).
- Don't claim a kb-nano file you didn't open. If you ran out of context to
  read a file, omit it from the row and say so in rationale.
- For multi-modeling folders (e.g. `data2vec`, `rt_detr`), audit each
  `modeling_*.py` separately as a sibling key under the folder.
