# Manual paper-grade audit — write the final table rows yourself

You are auditing a paper-grade HF Transformers × kb-nano coverage table. The
script-based and corrections-based approaches have failed to reach paper-grade
accuracy. The user has explicitly asked for a fully manual review per class.

You will write the final table rows directly, reading every relevant HF source
file and verifying every kb-nano file before claiming a mapping.

## Inputs

- HF source: `/tmp/hf_transformers_pinned/src/transformers/models/<folder>/`
  - For each folder, ALWAYS read `modeling_<folder>.py` (the expanded form
    that runs at inference). This is the source of truth for compute.
  - If `modular_<folder>.py` exists, also read it for inheritance lineage —
    the parent class names are useful for the table. But the COMPUTE lives
    in modeling_*.py.
  - If `configuration_<folder>.py` exists, look up `hidden_act` default to
    resolve `ACT2FN` activations.
  - For multi-modeling folders (e.g. `data2vec/modeling_data2vec_audio.py`,
    `rt_detr/modeling_rt_detr_resnet.py`), audit each modeling file
    separately under its own subkey.
- kb-nano files: `/home/olu/kb_nano/tasks/baseline/{L1,L2,L3,L4}/`
  - Compact digest at `/home/olu/kb_nano/audits/hf_transformers_coverage/tools/kb_nano_digest.txt`
    (one line per file: `path: docstring`)
  - Read the actual file before claiming any mapping.
- Your shard list: `/home/olu/kb_nano/audits/hf_transformers_coverage/tools/rewrite_shard_{N}.txt`
  (newline-delimited folder keys)

## Per-folder workflow

For each folder in your shard, in alphabetical order:

1. **List the classes** in `modeling_<folder>.py` via `grep -n "^class "` so
   you know the scope.
2. **Read configuration_<folder>.py** to find the `hidden_act` default
   (silu/gelu/etc.). Note it for activation resolution.
3. **Read modular_<folder>.py** if present — note inheritance for each class.
4. **For each class** (skipping boilerplate — see rule below):
   a. Read its `__init__` and `forward`.
   b. Identify whether it's COMPUTE (own forward does math on tensors) or
      WIRING (forward just sequences sub-modules).
   c. List the direct sub-modules from `__init__` (any `self.X = ...`).
   d. For COMPUTE classes: pick the one kb-nano file (or small composition)
      whose `__init__` and `forward` match. **Read the kb-nano file** before
      claiming the match — open it, confirm same activation / norm / projection
      / attention type / op sequence.
   e. For WIRING classes: list the kb-nano kernels invoked directly
      (e.g. `nn.Linear` in __init__) PLUS the names of HF sub-module classes
      it instantiates.

5. **Skip these classes** (do not include rows for them):
   - `*PreTrainedModel`, `*Config`, `*Output`, `*OutputWith*`, `*Cache`
   - `*Mixin` (e.g. GenerationMixin, BackboneMixin)
   - `*Kwargs`, `*Loss`, `*Function` (autograd Functions)
   - Multi-task heads ending in `For{X}` where X is in {SequenceClassification,
     TokenClassification, QuestionAnswering, MultipleChoice, NextSentencePrediction,
     PreTraining, ImageClassification, ObjectDetection, SemanticSegmentation,
     AudioClassification, VideoClassification, AudioFrameClassification,
     UniversalSegmentation, InstanceSegmentation, MaskGeneration, KeypointDetection,
     XVector, CTC, DocumentQuestionAnswering, TableQuestionAnswering,
     VisualQuestionAnswering, CausalImageModeling, MaskedImageModeling}.
     For these, emit ONE row per folder labeled `task heads (N)` that lists
     the suffixes.
   - **KEEP these `For*` heads** (primary forward path): `ForCausalLM`,
     `ForConditionalGeneration`, `ForMaskedLM`.

## Decision tree for kb-nano kernel selection

Read the relevant kb-nano file before mapping! These are heuristics, not rules:

**Norms:**
- Standard RMSNorm (Llama-family) → `L1/rms_norm.py`
- T5 layer norm (no centering) → `L1/t5_layer_norm.py`
- BitNet RMSNorm → `L1/bitnet_rms_norm.py`
- Gemma RMSNorm → `L1/gemma_rms_norm.py`
- nn.LayerNorm → `L1/layer_norm.py`
- nn.GroupNorm → `L1/group_norm.py`
- nn.BatchNorm2d → `L1/batch_norm2d.py`

**RoPE:**
- Standard Llama RoPE → `L1/rotary_emb.py`
- YaRN-NeoX (GPT-OSS), YaRN-DeepSeek → `L1/yarn_rotary_emb.py`
- M-RoPE (Qwen-VL) → `L1/mrope.py`
- 2D vision RoPE → `L1/vision_rotary_emb.py`
- DINOv3 RoPE → `L1/dinov3_rope.py`

**Attention (read the source carefully):**
- Decoder causal w/ RoPE + KV cache (Llama, Qwen, Mistral, GPT-OSS, Mixtral) → `L2/attention.py`
- BERT-style `*Attention` wrapper (`self.self = X + self.output = Y`) → `L2/encoder_attention.py`
- BERT-style `*SelfAttention` (q/k/v + dispatch) → `L2/encoder_attention.py`
- BERT-style `*SelfOutput` (dense + LayerNorm + residual) → `L2/encoder_attention.py`
- SigLIP non-causal → `L2/siglip_attention.py`
- CLIP (causal option) → `L2/clip_attention.py`
- Whisper enc/dec/cross (3 variants) → `L2/whisper_attention.py`
- T5 with rel-pos-bias → `L2/t5_attention.py`
- DeepSeek MLA → `L2/deepseek_mla_attention.py`
- Deformable (RT-DETR-V2) → `L1/rtdetrv2_deformable_attention.py`
- BitNet → `L2/bitnet_attention.py`

**MLPs:**
- SwiGLU (`gate_proj * silu(gate_up_proj) → down_proj` or similar gate*up pattern) → `L2/llama_mlp.py`
- BERT-encoder pattern (`fc1 → activation → fc2` split into Intermediate + Output classes) → `L2/encoder_mlp.py`
- CLIP MLP (fc1 → quickgelu → fc2) → `L2/clip_mlp.py`
- SigLIP MLP (fc1 → gelu_pytorch_tanh → fc2) → `L2/siglip_mlp.py`
- Whisper MLP (fc1 → gelu → fc2) → `L2/whisper_mlp.py`
- T5 dense (gated, with relative scoring) → `L2/t5_dense.py`
- BitNet MLP → `L2/bitnet_mlp.py`
- 2-layer with non-standard activation → decompose to L1 (`linear + activation`)

**MoE:**
- Standard fused MoE → `L1/moe_grouped_gemm.py`
- Shared-expert (DeepSeek, Kimi-Linear, Qwen3-Next) → `L2/shared_expert_moe.py`
- MXFP4 quantized (GPT-OSS) → `L1/mxfp4_moe.py`
- Arch-specific block: `L2/<arch>_moe.py` (mixtral, jamba, gpt_oss, gemma4, deepseek, kimi, llama4, qwen3, qwen3_next)

**Activations (resolved from `ACT2FN[hidden_act]`):**
- silu, swish → `L1/silu.py` (or `L1/silu_and_mul.py` if SwiGLU pattern)
- gelu, gelu_new, gelu_pytorch_tanh, gelu_fast → `L1/gelu.py` (or `L1/gelu_and_mul.py` for GeGLU)
- quick_gelu, quickgelu → `L1/quickgelu.py`
- relu → `L1/relu.py`
- relu2, relu_squared → `L1/squared_relu.py` (or `L1/squared_relu_and_mul.py` for gated)
- tanh → `L1/tanh.py`
- sigmoid → `L1/sigmoid.py`

**Other primitives:**
- nn.Linear → `L1/linear.py`
- nn.Embedding → `L1/embedding.py`
- nn.Conv1d/2d/3d → `L1/conv1d.py`/`conv2d.py`/`conv3d.py`
- nn.MaxPool2d → `L1/max_pool2d.py`
- nn.AdaptiveAvgPool2d / nn.AvgPool2d → `L1/avg_pool2d.py`
- nn.AvgPool1d → `L1/avg_pool1d.py`
- F.scaled_dot_product_attention call (no KV cache) → `L1/dense_attention.py`
- KV cache update → `L1/store_kvcache.py`

**Embeddings (Vision):**
- BERT-style `*Embeddings` (word + position + token_type + LayerNorm + Dropout) → `L2/encoder_embeddings.py`
- Vision patch embed (Conv2d + position) → compute kernels: `L1/conv2d.py`, `L1/embedding.py`,
  optionally `L1/layer_norm.py` or `L1/rms_norm.py` depending on the model

## On compute vs wiring — read this carefully

Every class falls into one of three rendering categories:

**1. Compute with a single kb-nano kernel match.** The class's `forward` does
math on tensors and matches an existing kb-nano file's `__init__` + `forward`
exactly (or close enough that the underlying L1/L2 ops are the same). Render:

```
- **`AfmoeRMSNorm`** [compute]: `L1/rms_norm.py`
- **`AfmoeMLP`** [compute, inherits `Qwen2MoeMLP`]: `L2/llama_mlp.py`
```

**2. Compute that decomposes to L1.** The class's `forward` is small and
matches no exact L2 file, but is built from L1 primitives. Render with `+`
between L1 paths and a "(no exact L2 match)" note if needed:

```
- **`AltRobertaAttention`** [compute]: `L1/linear.py + L1/dense_attention.py + L1/store_kvcache.py` (no exact L2 match)
- **`BertIntermediate`** [compute]: `L1/linear.py + L1/gelu.py` (just half of an encoder MLP)
```

**3. Wiring: a class with no kernel of its own that just sequences sub-modules
in `forward`.** The class's `__init__` instantiates other classes and the
`forward` calls them in order. Render as `wires X, Y, Z` listing the HF
sub-module class names. Each sub-module is mapped on its own row, so the
reader can transitively trace the kernel breakdown. Also list any direct
kb-nano kernels the wiring class itself uses (e.g., a final `nn.Linear`):

```
- **`BertModel`** [wiring]: wires `BertEmbeddings`, `BertEncoder`, `BertPooler`
- **`BertForMaskedLM`** [wiring]: wires `BertModel`, `BertOnlyMLMHead`
- **`AfmoeForCausalLM`** [wiring]: wires `AfmoeModel`; direct `L1/linear.py` (lm_head)
- **`BambaDecoderLayer`** [wiring]: wires `BambaAttention` (or `BambaMixer` per layer_type), `BambaMLP`, `BambaRMSNorm` (×2)
```

**Rules of thumb for compute vs wiring:**
- Class typically COMPUTES if its `forward` does arithmetic directly: matmul,
  add, mul, normalization math, softmax, activation.
- Class typically WIRES if its `forward` body is just `x = self.A(x); x =
  self.B(x); return x` — purely calls sub-modules, no arithmetic.
- Edge case: a class that calls one sub-module AND adds a residual is still
  basically wiring. Mark it `[wiring]` with "wires X" plus a note about the
  residual.
- `*Layer` / `*EncoderLayer` / `*DecoderLayer` / `*Block` / `*Stage` are
  usually wiring.
- `*Encoder` / `*Decoder` / `*Model` / `*ForXxx` are always wiring.
- `*Embeddings` is often a HYBRID — it instantiates `nn.Embedding` directly
  AND adds a `LayerNorm` in `__init__`. Render as compute with the kernels
  it directly uses: `[compute]: L1/embedding.py + L1/layer_norm.py + L1/embedding.py`.
  (Or map to `L2/encoder_embeddings.py` if it's the BERT-style word+pos+token_type
  pattern.)

## Output format

Write your output to a single markdown file at
`/home/olu/kb_nano/audits/hf_transformers_coverage/tools/manual_audit_shard_{N}.md`.

Format per folder:

```markdown
## <folder_key>
- **src**: modeling_<folder>.py (and modular_<folder>.py if exists)
- **hidden_act**: <silu|gelu|...> (from configuration_<folder>.py)
- **status**: composable | kb_nano_l4 | partial | unsupported
- **classes**:
  - **`<ClassName>`** [compute|wiring] [inherits `<Parent>`]: <kb-nano paths or "wires <ChildA>, <ChildB>, ..."> [notes]
  - ...
- **task heads (N)**: ForX, ForY, ForZ — base + linear (per-task)
```

Example:

```markdown
## bert
- **src**: modeling_bert.py
- **hidden_act**: gelu
- **status**: composable
- **classes**:
  - **`BertEmbeddings`** [compute]: `L2/encoder_embeddings.py`
  - **`BertSelfAttention`** [compute]: `L2/encoder_attention.py` (q/k/v + dispatch via ALL_ATTENTION_FUNCTIONS)
  - **`BertSelfOutput`** [compute]: `L2/encoder_attention.py` (dense + LayerNorm + residual)
  - **`BertAttention`** [compute]: `L2/encoder_attention.py` (wrapper: self.self + self.output)
  - **`BertIntermediate`** [compute]: `L1/linear.py + L1/gelu.py` (no exact L2 match — encoder_mlp.py covers Intermediate+Output but BertIntermediate is just one half)
  - **`BertLayer`** [wiring]: wires `BertAttention`, `BertIntermediate`, `BertOutput`, optional `BertCrossAttention`
  - **`BertEncoder`** [wiring]: wires `BertLayer`
  - **`BertPooler`** [compute]: `L1/linear.py + L1/tanh.py`
  - **`BertPredictionHeadTransform`** [compute]: `L1/linear.py + L1/gelu.py + L1/layer_norm.py`
  - **`BertLMPredictionHead`** [wiring]: wires `BertPredictionHeadTransform`; direct `L1/linear.py`
  - **`BertOnlyMLMHead`** [wiring]: wires `BertLMPredictionHead`
  - **`BertModel`** [wiring]: wires `BertEmbeddings`, `BertEncoder`, `BertPooler`
  - **`BertForMaskedLM`** [wiring]: wires `BertModel`, `BertOnlyMLMHead`
  - **`BertLMHeadModel`** [wiring]: wires `BertModel`, `BertOnlyMLMHead`; direct `L1/linear.py`
- **task heads (5)**: ForMultipleChoice, ForNextSentencePrediction, ForQuestionAnswering, ForSequenceClassification, ForTokenClassification — base + linear (per-task)
```

## Verification rules

1. **Read the actual file.** Do not claim a kb-nano kernel without opening
   the file and confirming `__init__` + `forward` match. Don't grep first lines
   and guess.
2. **Don't invent kernel paths.** Every path you write must exist (check the
   digest or `kb_nano_files.txt`).
3. **Don't trust class names alone.** A class named `*MLP` could be SwiGLU
   (gate*up) or 2-layer (fc1→act→fc2) — check the body.
4. **Don't double-count conditional sub-modules.** If `self.crossattention =
   X` is inside `if self.add_cross_attention:`, note it as "optional cross-attn"
   but don't double-count.
5. **Skip torch builtins from "wires":** `nn.Dropout`, `nn.Identity`,
   `nn.ZeroPad2d`, `nn.Sigmoid`, `nn.Tanh` (used as instances), and
   `nn.ModuleList` (just a container) — skip.
6. **For modular files**: if the modular file shows `class X(Parent): pass`,
   the compute is inherited. Read the parent class in its modeling file
   (e.g. `from ..llama.modeling_llama import LlamaAttention` → read llama
   modeling). Or note it as "inherits Parent" and use the parent's mapping.

## Reporting

After writing the markdown file, post a short report:
- Folders reviewed
- Total classes mapped
- Any folder you couldn't fully verify (and why)
- Confidence level
