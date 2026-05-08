# Re-audit methodology — sanity check on `afmoe` and `aimv2`

**For mentor review.** Two worked examples showing exactly what the new
audit pass will include and exclude per HF folder. If the methodology is
sound, I'll scale this to all 448 HF modeling files.

---

## Why a re-audit is needed

The original audit only captured **L1 leaf primitives** (`linear`, `sdpa`,
`rms_norm`, `silu`, ...) for every HF row. It missed:

1. **The HF class names themselves** (`LlamaAttention`, `LlamaMLP`,
   `BertLayer`, ...) — only top-level `*ForCausalLM` / `*Model` were captured.
2. **The kb-nano L2/L3/L4 hierarchy** — the canonical-op map was L1-only,
   so subagents never linked HF blocks to kb-nano composites like
   `L2/llama_mlp.py`, `L3/llama_decoder.py`, `L4/llama.py`.
3. **`modular_<arch>.py` inheritance** — these files explicitly state things
   like `class AfmoeAttention(LlamaAttention)`, telling us afmoe IS a
   Llama-family attention with no new compute. The audit only read
   `modeling_<arch>.py` (the generated, expanded file) and missed this.

Net effect: only **40/151 L1 (26.5%), 5/175 L2 (2.9%), 0/82 L3 (0.0%),
10/54 L4 (18.5%)** of kb-nano files were referenced from any HF row, even
though many more should be.

---

## Methodology (proposed) — applied to every HF folder

### Phase 1: Source extraction

For folder `<f>`:

1. If `modular_<f>.py` exists → use it (preferred — explicit inheritance).
   Otherwise use `modeling_<f>.py`.
2. AST-extract every top-level `class` definition + its base class(es).
3. Skip boilerplate: `*PreTrainedModel` (just `_init_weights`), `*Config`
   (configuration), `*Output` (return-type containers), generic mixins.

### Phase 2: Per-class kb-nano mapping

For each remaining HF class, classify by role from the class name suffix
and inheritance:

| HF class suffix | Role | kb-nano lookup order |
|---|---|---|
| `*RMSNorm`, `*LayerNorm`, `*RotaryEmbedding`, `*Linear`, ... | L1 primitive | `tasks/baseline/L1/<canonical>.py` |
| `*Attention`, `*MLP`, `*Embeddings`, `*Experts`, `*Router`, `*MoEBlock`, `*PoolingHead` | L2 composite | `tasks/baseline/L2/<f>_<role>.py` (folder-prefixed); else `<inherited-from>_<role>.py` (e.g. siglip_attention.py for an `*Attention(SiglipAttention)`); else mark as composable from L1 primitives |
| `*Layer`, `*DecoderLayer`, `*EncoderLayer`, `*Block`, `*Encoder`, `*Decoder` | L3 layer | `tasks/baseline/L3/<f>_<role>.py`; else `<inherited-from>_<role>.py` |
| `*Model`, `*ForCausalLM`, `*ForXxx`, `*Vision*Model`, `*TextModel` | L4 pipeline | `tasks/baseline/L4/<f>.py`; else mark as composable from L3 layers |

**Direct match** = kb-nano filename exactly matches `<folder>_<role>.py`.
**Inherited match** = HF class inherits from another arch's class (e.g.
`AfmoeMLP(Qwen2MoeMLP)`) and kb-nano has the parent file.
**Composable** = no single kb-nano file matches; row decomposes via L1.

### Phase 3: What I INCLUDE in the row

- `architecture_classes`: every non-boilerplate HF class (CamelCase, as in source)
- `mapped_kb_nano`: every kb-nano file that maps to one of those classes,
  with the canonical op name + class name (`llama_mlp→tasks/baseline/L2/llama_mlp.py:LlamaMLP`)
- `notes`: modular inheritance chain (e.g. "Afmoe = Llama + Qwen2Moe + GptOss")
- `evidence_hf`: line numbers for the load-bearing classes
- `support_status`: see decision rules below

### Phase 4: What I EXCLUDE (with reason)

| Excluded | Reason |
|---|---|
| `*PreTrainedModel`, `PreTrainedModel` | Pure boilerplate (no compute) |
| `*Config`, `*VisionConfig`, `*TextConfig` | Configuration only |
| `*Output`, `ModelOutput` namedtuples | Return-type containers |
| `GradientCheckpointingLayer` (the bare base) | Generic gradient-checkpointing wrapper, captured via the subclass |
| Helper functions (`_prepare_4d_attention_mask`, `_init_weights`) | Sub-routines, captured via the calling class |
| Mixins (`GenerationMixin`) | Inference orchestration, not compute |

### Phase 5: Status rules

- `kb_nano_l4` = direct file `tasks/baseline/L4/<f>.py` exists implementing `*Model` or `*ForXxx`
- `composable` = every required class has either a direct kb-nano file OR is composable from kb-nano L1 primitives
- `partial` = at least one class requires an op that has no kb-nano equivalent AND torch.nn fallback exists (e.g. `detectron2_backbone`)
- `unsupported` = at least one class requires a custom CUDA kernel or external library not in kb-nano (e.g. `mra_sparse_kernels`)

---

## Worked example 1: `afmoe`

### Source files

- `modular_afmoe.py` (460 lines) — preferred
- `modeling_afmoe.py` (693 lines) — generated from modular

### Modular inheritance chain (key insight)

```
AfmoeRotaryEmbedding(LlamaRotaryEmbedding)
AfmoeRMSNorm(GptOssRMSNorm)
AfmoeMLP(Qwen2MoeMLP)
AfmoeAttention(LlamaAttention)
AfmoeExperts(Qwen2MoeExperts)
AfmoeForCausalLM(LlamaForCausalLM, AfmoePreTrainedModel, GenerationMixin)
```

Reading: afmoe IS a **Llama-family** decoder-only model with **Qwen2Moe-style
MoE** swapped in for the MLP, using **GptOss-style RMSNorm**. All three parent
architectures already have full kb-nano coverage.

### Per-class mapping

| # | HF class | kb-nano file | match type | rationale |
|---:|---|---|---|---|
| 1 | `AfmoeRotaryEmbedding` | `L1/rotary_emb.py:RotaryEmbedding` | **direct** | LlamaRotaryEmbedding-equivalent |
| 2 | `AfmoeRMSNorm` | `L1/rms_norm.py:RMSNorm` | **direct** | GptOssRMSNorm uses standard RMS formula |
| 3 | `AfmoeMLP` | `L2/llama_mlp.py:LlamaMLP` | **inherited** | Qwen2MoeMLP and LlamaMLP have identical structure (gate_up_proj → SiluAndMul → down_proj — verified by reading kb-nano file) |
| 4 | `AfmoeTokenChoiceRouter` | `L1/grouped_topk.py:GroupedTopK` (+ `L1/linear.py`) | **composable** | router = Linear → softmax → topk; kb-nano covers via grouped_topk |
| 5 | `AfmoeExperts` | `L1/moe_grouped_gemm.py:MoeGroupedGemm` | **direct** | Qwen2MoeExperts kernel is the fused MoE GEMM that kb-nano implements (verified — Triton fused MoE in kb-nano file) |
| 6 | `AfmoeSparseMoeBlock` | (composes #4 + #5 + reduce) | **composable** | No L2 wrapper in kb-nano; composition is router + experts + sum |
| 7 | `AfmoeAttention` | `L1/dense_attention.py:DenseAttention` (+ `L1/linear.py`, `L1/store_kvcache.py`, `L1/rotary_emb.py`) | **composable** | LlamaAttention; kb-nano has NO `*_attention.py` for llama-family — uses L1 SDPA directly |
| 8 | `AfmoeDecoderLayer` | `L3/llama_decoder.py:LlamaDecoder` | **inherited** | Inherits GradientCheckpointingLayer; the actual block structure (norm → attn → norm → MoE) matches Llama decoder; kb-nano `llama_decoder.py` covers this pattern |
| 9 | `AfmoeModel` | (composes #1–#8 + embedding) | **composable** | No L4 pipeline; composable |
| 10 | `AfmoeForCausalLM` | (composes #9 + LM head) | **composable** | No L4 pipeline; composable |

### What's INCLUDED

- **Operators (10)**: `AfmoeAttention, AfmoeDecoderLayer, AfmoeExperts, AfmoeForCausalLM, AfmoeMLP, AfmoeModel, AfmoeRMSNorm, AfmoeRotaryEmbedding, AfmoeSparseMoeBlock, AfmoeTokenChoiceRouter`
- **kb-nano L1 (8)**: `dense_attention, embedding, grouped_topk, linear, moe_grouped_gemm, rms_norm, rotary_emb, store_kvcache`
- **kb-nano L2 (1)**: `llama_mlp`
- **kb-nano L3 (1)**: `llama_decoder`
- **kb-nano L4 (0)**: none (no `L4/afmoe.py`)
- **Notes**: "modular inheritance: Afmoe = Llama + Qwen2Moe + GptOss; no L4 pipeline; composable from existing parts"

### What's EXCLUDED

- `AfmoePreTrainedModel` (boilerplate, only `_init_weights`)
- `Aimv2Output`/etc. style namedtuples (none in afmoe)
- HF helper functions inside classes (`_prepare_4d_attention_mask`, etc.)

### Status: `composable`

---

## Worked example 2: `aimv2`

### Source files

- `modular_aimv2.py` (preferred)
- `modeling_aimv2.py` (generated)

### Modular inheritance chain

```
Aimv2RMSNorm(LlamaRMSNorm)
Aimv2MLP(LlamaMLP)
Aimv2VisionEmbeddings(nn.Module)        # custom (vision patch embed)
Aimv2TextEmbeddings(CLIPTextEmbeddings)
Aimv2Attention(SiglipAttention)
Aimv2EncoderLayer(GradientCheckpointingLayer)   # custom block structure
Aimv2Encoder(SiglipEncoder)
Aimv2AttentionPoolingHead(nn.Module)    # cross-attn pool with latent query
Aimv2VisionModel(Aimv2PreTrainedModel)
Aimv2TextModel(Aimv2PreTrainedModel)
Aimv2Model(CLIPModel)
```

Reading: aimv2 is a **vision-language encoder** that mixes **SigLIP**
(attention + encoder) + **Llama** (RMSNorm + MLP, i.e. SwiGLU MLP) + **CLIP**
(text embeddings + the dual-tower model wiring) + a custom **attention
pooling head** for the vision representation.

### Per-class mapping

| # | HF class | kb-nano file | match type | rationale |
|---:|---|---|---|---|
| 1 | `Aimv2RMSNorm` | `L1/rms_norm.py:RMSNorm` | **direct** | LlamaRMSNorm-equivalent |
| 2 | `Aimv2MLP` | `L2/llama_mlp.py:LlamaMLP` | **inherited** | LlamaMLP — SwiGLU MLP (gate, up, down with SiLU) — verified by reading kb-nano file |
| 3 | `Aimv2VisionEmbeddings` | `L1/conv2d.py:Conv2d` + `L1/embedding.py` (composable) | **composable** | Custom: patch embed (Conv2d) + position embed (Embedding); composes from L1 |
| 4 | `Aimv2TextEmbeddings` | (composable from L1 embedding + L1 layer_norm) | **composable** | CLIPTextEmbeddings = token + position embedding sum + LayerNorm |
| 5 | `Aimv2Attention` | `L2/siglip_attention.py:SigLIPAttention` | **direct** | SiglipAttention; kb-nano file mirrors HF SigLIP attention exactly (verified — q/k/v/o + DenseAttention, no positional encoding) |
| 6 | `Aimv2EncoderLayer` | (composable from L2 attention + L2 mlp + L1 layer_norm) | **composable** | Custom block: norm → SiglipAttention → norm → LlamaMLP — the union of SigLIP encoder block + Llama MLP. No single L3 file; composable from existing pieces |
| 7 | `Aimv2Encoder` | (composable from N × `Aimv2EncoderLayer`) | **composable** | Stack of #6; SiglipEncoder pattern |
| 8 | `Aimv2AttentionPoolingHead` | `L2/attention_pool.py:AttentionPoolLatent` | **direct** | Cross-attention pooling with learnable latent query; kb-nano file is exactly this (verified — same docstring, same forward pattern as timm AttentionPoolLatent) |
| 9 | `Aimv2VisionModel` | (composes #3 + #6 + #7 + #8) | **composable** | No L4 pipeline |
| 10 | `Aimv2TextModel` | (composes #4 + #6 + #7) | **composable** | No L4 pipeline |
| 11 | `Aimv2Model` | (composes #9 + #10 + projection heads) | **composable** | CLIPModel-style dual tower |

### What's INCLUDED

- **Operators (11)**: `Aimv2Attention, Aimv2AttentionPoolingHead, Aimv2EncoderLayer, Aimv2Encoder, Aimv2MLP, Aimv2Model, Aimv2RMSNorm, Aimv2TextEmbeddings, Aimv2TextModel, Aimv2VisionEmbeddings, Aimv2VisionModel`
- **kb-nano L1 (5)**: `conv2d, embedding, layer_norm, linear, rms_norm` (also `silu` via `silu_and_mul` inside `llama_mlp.py`, `dense_attention` via `siglip_attention.py`)
- **kb-nano L2 (3)**: `attention_pool, llama_mlp, siglip_attention`
- **kb-nano L3 (0)**: none (no aimv2-specific or siglip encoder layer used directly; aimv2 uses a custom encoder block)
- **kb-nano L4 (0)**: none
- **Notes**: "vision-language encoder; modular inheritance: SigLIP (attn, encoder) + Llama (RMSNorm, SwiGLU MLP) + CLIP (text embed, dual tower); attention pooling via timm-style latent-query pool"

### What's EXCLUDED

- `Aimv2VisionConfig`, `Aimv2TextConfig`, `Aimv2Config` (configuration only)
- `Aimv2Output` (ModelOutput namedtuple)
- `Aimv2PreTrainedModel` (boilerplate)

### Status: `composable`

---

## Summary table — mentor sanity check

| | afmoe | aimv2 |
|---|---|---|
| **HF source** | modeling+modular | modeling+modular |
| **HF classes (incl. all blocks)** | 10 | 11 |
| **HF classes the OLD audit captured** | 3 (just *Model + *ForCausalLM + *PreTrainedModel) | 0 (just architecture_classes was empty) |
| **kb-nano L1 direct** | 8 | 5 |
| **kb-nano L2 direct/inherited** | 1 (llama_mlp) | 3 (attention_pool, llama_mlp, siglip_attention) |
| **kb-nano L3 direct/inherited** | 1 (llama_decoder) | 0 |
| **kb-nano L4 direct** | 0 | 0 |
| **Verdict** | composable (Llama + Qwen2Moe + GptOss hybrid; no L4) | composable (SigLIP + Llama + CLIP hybrid; no L4) |
| **Coverage gain vs old audit** | +1 L2 (llama_mlp) + 1 L3 (llama_decoder) + 1 L1 (moe_grouped_gemm) + 7 HF class names in operators | +3 L2 + 11 HF class names + 1 modular inheritance note |

---

## Decision points the mentor should confirm

1. **Inheritance-based mapping is fair.** When `AfmoeMLP(Qwen2MoeMLP)`
   inherits from a parent that's structurally identical to kb-nano's
   `LlamaMLP` (gate-up-down SwiGLU), I map it to `L2/llama_mlp.py`. Is
   this aggressive-but-honest, or too aggressive?

2. **`*Attention` decomposition.** kb-nano has NO `llama_attention.py` —
   the Llama attention is composed at runtime from L1 (Linear + DenseAttention
   + StoreKVCache + RotaryEmbedding). For afmoe (LlamaAttention-based) I
   list those L1 files as the mapping. Correct?

3. **Custom blocks (Aimv2EncoderLayer).** When HF defines a custom block
   that doesn't directly inherit from another arch, I mark it composable
   from existing kb-nano L1/L2 (SiglipAttention + LlamaMLP + LayerNorm).
   No single kb-nano file matches. Correct verdict?

4. **Excluded classes.** `*PreTrainedModel`, `*Config`, `*Output`,
   `GradientCheckpointingLayer`, `GenerationMixin` — agree these add no
   compute info?

5. **The `evidence_hf` column.** Currently has 3 line numbers per row
   (one per top-level class). Should the new audit cite a line for EVERY
   HF class in operators? (More evidence; longer rows.)

If methodology approved, I'll run it on all 448 modeling files. Estimated
runtime: ~5 minutes (pure AST, no LLM). Output replaces the current CSV.
