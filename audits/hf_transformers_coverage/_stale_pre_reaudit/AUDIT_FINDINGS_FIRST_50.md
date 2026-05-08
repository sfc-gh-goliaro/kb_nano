# Audit findings — first 50 folders

Manual audit by reading both HF source (`/tmp/hf_transformers_pinned/src/transformers/models/<folder>/`) and kb-nano source (`/home/olu/kb_nano/tasks/baseline/`).

Format per folder:
- ✓ row OK (kernel mapping matches HF compute)
- ✗ row WRONG (followed by what's wrong + correction)
- ? row needs review

## 1. afmoe
Source: modular_afmoe.py (full re-read)
- AfmoeRotaryEmbedding `(LlamaRotaryEmbedding) pass` → L1/rotary_emb.py ✓
- AfmoeRMSNorm `(GptOssRMSNorm) pass` → L1/rms_norm.py ✓ (GptOssRMSNorm is plain RMSNorm)
- AfmoeMLP `(Qwen2MoeMLP) pass` → L2/llama_mlp.py ✓ (Qwen2MoeMLP is SwiGLU pattern)
- AfmoeTokenChoiceRouter (own __init__: nn.Linear gate; forward: linear→sigmoid→topk→gather→sum→mul) → L1/linear.py ✓ (only kernel is the gate linear; rest is small tensor math)
- AfmoeExperts `(Qwen2MoeExperts) pass` → L1/moe_grouped_gemm.py ✓
- AfmoeSparseMoeBlock (router + shared_experts AfmoeMLP + experts AfmoeExperts) → L2/shared_expert_moe.py ✓
- AfmoeAttention `(LlamaAttention)` overrides __init__ adding q_norm/k_norm (RMSNorm) + gate_proj (Linear); forward applies Q/K norm + sigmoid gate → {L2/attention.py, L1/rms_norm.py} ⚠ MINOR: doesn't list extra L1/linear for gate_proj or sigmoid kernel; but L2/attention.py includes linear internally and sigmoid is an elementwise op not a kernel. Acceptable.
- AfmoeDecoderLayer composes: L2/attention.py + 5× L1/rms_norm.py + L2/shared_expert_moe.py + L2/llama_mlp.py ✓ (4 layer norms + 1 from AfmoeAttention's q/k_norm = 5; mlp is conditionally MoE or dense)
- AfmoeModel composes: L1/embedding.py + AfmoeDecoderLayer (wiring) + L1/rms_norm.py + L1/rotary_emb.py ✓
- AfmoeForCausalLM composes: AfmoeModel (wiring) + L1/linear.py ✓

Verdict: **all 10 rows correct.**

## 2. aimv2
Source: modular_aimv2.py (full re-read) + cross-checked CLIPTextEmbeddings in clip/modeling_clip.py
- Aimv2RMSNorm `(LlamaRMSNorm) pass` → L1/rms_norm.py ✓
- Aimv2MLP `(LlamaMLP) pass` → L2/llama_mlp.py ✓ (LlamaMLP is SwiGLU)
- Aimv2VisionEmbeddings (own __init__: patch_embed=Conv2d, rms_norm=Aimv2RMSNorm, position_embedding=Embedding) → {L1/conv2d.py, L1/embedding.py} **⚠ MISSING `L1/rms_norm.py`** — the rms_norm submodule is dropped from the row.
- Aimv2TextEmbeddings `(CLIPTextEmbeddings) pass` → {L1/embedding.py, L1/layer_norm.py} **✗ WRONG** — CLIPTextEmbeddings has only 2× nn.Embedding (token + position), no LayerNorm. Should be just L1/embedding.py.
- Aimv2Attention `(SiglipAttention)` overrides only 4× nn.Linear k/v/q/out_proj → L2/siglip_attention.py ✓ (caveat: kb-nano file is hardcoded `causal=False`; aimv2 text attention is causal — would need a small parameter fix at adoption time, but the underlying L1 ops cover it)
- Aimv2EncoderLayer (own attention+ffn+2 RMSNorms) → composes: L2/siglip_attention.py + L2/llama_mlp.py + 2× L1/rms_norm.py ✓
- Aimv2Encoder `(SiglipEncoder) pass` → DROPPED (was wiring) — but that lost the link `Aimv2Encoder = stack of Aimv2EncoderLayer`. Reader has to grep SiglipEncoder.
- Aimv2AttentionPoolingHead (own __init__: 3× nn.Linear, cls_token Param; forward calls `F.scaled_dot_product_attention`) → composes: 3× L1/linear.py **⚠ MISSING `L1/dense_attention.py`** — the SDPA call in forward is not represented.
- Aimv2VisionModel composes: L1/conv2d.py + L1/embedding.py + Aimv2Encoder (wiring) + L1/rms_norm.py + Aimv2AttentionPoolingHead (wiring). Inherits the missing rms_norm bug from Aimv2VisionEmbeddings.
- Aimv2TextModel composes: L1/embedding.py + L1/layer_norm.py + Aimv2Encoder (wiring) + L1/rms_norm.py — has the spurious layer_norm.py from Aimv2TextEmbeddings bug.
- Aimv2Model `(CLIPModel)` overrides __init__ with 2× nn.Linear + 2× sub-model (`Aimv2VisionModel._from_config(...)`, `Aimv2TextModel._from_config(...)`). Tex shows: composes: 2× L1/linear.py **⚠ MISSING vision_model + text_model wiring refs** — AST extractor doesn't follow `Class._from_config()` method-call pattern.

**aimv2 errors to fix in JSON:**
1. Aimv2TextEmbeddings: drop `L1/layer_norm.py`
2. Aimv2VisionEmbeddings: add `L1/rms_norm.py`
3. Aimv2AttentionPoolingHead: add `L1/dense_attention.py`
4. Aimv2Model: missing wiring (post-rendering issue, AST limitation)

## 3. albert
Source: modeling_albert.py (no modular)
- AlbertEmbeddings (own __init__: word/pos/token_type Embedding + LayerNorm + Dropout) → L2/encoder_embeddings.py ✓ (matches BERT-style EncoderEmbeddingsBase)
- AlbertAttention (own __init__: q/k/v/dense Linear + LayerNorm; forward dispatches via ALL_ATTENTION_FUNCTIONS, returns dense+LayerNorm(residual+attn)) → L2/encoder_attention.py ✓ (matches EncoderAttention which wraps SelfAttention + SelfOutput)
- AlbertLayer (own __init__: full_layer_layer_norm + attention=AlbertAttention + ffn=Linear + ffn_output=Linear + activation=ACT2FN; forward composes attention+ffn+activation+layer_norm) → L2/encoder_mlp.py **✗ WRONG** — AlbertLayer is wiring around encoder_attention + encoder_mlp + extra LayerNorm. The subagent's own rationale literally says "wiring", but the JSON has kb_nano_files=['L2/encoder_mlp.py']. Should be empty (wiring class).
- AlbertLayerGroup (ModuleList of AlbertLayer) → L2/encoder_mlp.py **✗ WRONG** — pure wiring. JSON has empty list, but the row got mapped to encoder_mlp.py somehow. (Actually: JSON has [], and renderer falls back to wiring; row gets dropped now after pure-wiring filter — the tex showing it as encoder_mlp.py is from the previous render. Need re-render to confirm.)
- AlbertTransformer composes: L1/linear.py + AlbertLayerGroup (wiring) ✓
- AlbertModel composes: L2/encoder_embeddings.py + AlbertTransformer (wiring) + L1/linear.py + L1/tanh.py ✓ (pooler does linear + tanh)
- AlbertMLMHead composes: L1/layer_norm.py + 2× L1/linear.py — source has bias param + LayerNorm + dense + decoder + activation. Has activation (gelu/relu) — **⚠ MISSING activation.py**.
- AlbertSOPHead → L1/linear.py — source has dropout + classifier (Linear). Dropout is no-op for inference. ✓
- task heads (5) — collapsed to ForX-with-linear pattern ✓

**albert errors to fix:**
1. AlbertLayer: change kb_nano_files to [] (it's wiring) — currently subagent set to encoder_mlp.py
2. AlbertMLMHead: add activation kernel (gelu/silu depending on config; default is gelu)

## 4. align (first 19 rows)
Source: modeling_align.py
- AlignVisionEmbeddings → {L1/conv2d.py, L1/batch_norm2d.py} **⚠ MISSING activation** — source has self.activation = ACT2FN[hidden_act] (silu for efficientnet). ZeroPad2d is parameter-free padding, no kernel needed.
- AlignVisionDepthwiseConv2d (nn.Conv2d) → L1/conv2d.py ✓ (depthwise via groups=in_channels)
- AlignVisionExpansionLayer composes: L1/conv2d.py + L1/batch_norm2d.py **⚠ MISSING activation** (expand_act = ACT2FN[hidden_act])
- AlignVisionDepthwiseLayer composes: ZeroPad2d + L1/conv2d.py + L1/batch_norm2d.py **⚠ MISSING activation** (depthwise_act); also `ZeroPad2d` is rendered as a class name but it's a torch builtin — should not appear in the composes list.
- AlignVisionSqueezeExciteLayer → L2/efficientnetv2_squeeze_excite.py — source has AdaptiveAvgPool2d + 2× Conv2d + act_reduce + Sigmoid. Need to verify L2 file matches.
- AlignVisionFinalBlockLayer composes: L1/conv2d.py + L1/batch_norm2d.py ✓ (project_conv + project_bn + Dropout — dropout is no-op for inference)
- AlignVisionBlock → L2/efficientnetv2_inverted_residual.py — need to verify
- AlignVisionEncoder → L2/efficientnetv2_inverted_residual.py (via AlignVisionBlock chain) — this is the resolved single-kernel result of the dropped wiring row, OK
- AlignTextEmbeddings → L2/encoder_embeddings.py (BERT-style, has word+pos+token_type + LayerNorm + Dropout) ✓
- AlignTextSelfAttention → L2/encoder_attention.py (BERT-style q/k/v + DenseAttention) ✓ (assuming standard BERT pattern)
- AlignTextSelfOutput → L2/encoder_attention.py (dense + LayerNorm) ✓ (matches EncoderSelfOutput in same file)
- AlignTextAttention → {L1/linear.py, L1/dense_attention.py, L1/store_kvcache.py} (compose; no exact L2 match) — **⚠ this is an "Attention" name without exact L2 match, but actually L2/encoder_attention.py:EncoderAttention IS this exact pattern (self + output). Should map to L2/encoder_attention.py, not decompose.**
- AlignTextIntermediate → L2/encoder_mlp.py ✓
- AlignTextOutput → L2/encoder_mlp.py ✓
- AlignTextLayer composes: L1/linear.py + L1/dense_attention.py + L1/store_kvcache.py + 2× L2/encoder_mlp.py — picks up the buggy AlignTextAttention decomposition. Also missing the LayerNorm typically used in BERT layers. **⚠ INCOMPLETE**
- AlignTextEncoder composes: AlignTextLayer (wiring) ✓ (after drop, dangling ref but acceptable)
- AlignTextPooler composes: L1/linear.py + L1/tanh.py ✓
- AlignTextModel composes: L2/encoder_embeddings.py + AlignTextEncoder (wiring) + AlignTextPooler (wiring) ✓
- AlignVisionModel composes: L1/conv2d.py + L1/batch_norm2d.py + AlignVisionEncoder (wiring) + L1/avg_pool2d.py + L1/max_pool2d.py — **⚠ MISSING activation** (top conv stem has activation too)

**align errors to fix:**
1. Activation functions (silu for efficientnet) systematically missing across all AlignVision* rows
2. AlignTextAttention: should map to L2/encoder_attention.py:EncoderAttention (which exactly matches the BERT-style attention+output pattern), not decompose to L1
3. ZeroPad2d torch builtin leaking into composes list (cosmetic but wrong — should be filtered)

---

# Systematic issues found across 4 folders

These bugs likely repeat across many of the remaining 46 folders:

1. **Activation captured via `ACT2FN[config.hidden_act]` is invisible to AST extractor.**
   The pattern `self.activation = ACT2FN[config.hidden_act]` is a Subscript, not a Call. AST walker treats it as not-a-class, so the activation kernel (silu / gelu / relu / quickgelu / etc.) is dropped. Affects: every vision block, every BERT-style encoder layer's MLP, AlbertMLMHead, all align AlignVision*Layer rows, etc.

2. **Subagent sometimes maps wiring classes to L2 files.**
   E.g., AlbertLayer's rationale literally says "wiring" but kb_nano_files=['L2/encoder_mlp.py']. The subagent picked the FFN portion's L2 file as the whole-layer mapping. Other folders may have similar patterns (look for *Layer / *Block classes whose rationale says "wiring" but kb_nano_files is non-empty).

3. **BERT-style `*Attention` (the wrapper around *SelfAttention + *SelfOutput) is decomposed to L1 instead of mapping to L2/encoder_attention.py:EncoderAttention.**
   E.g., AlignTextAttention. The decompose_compute_class fallback fires because the class name ends with "Attention" but no exact L2 match was set. But L2/encoder_attention.py:EncoderAttention IS the exact match.

4. **Cross-arch inheritance (e.g. CLIPTextEmbeddings) mappings sometimes copy wrong kernels.**
   E.g., Aimv2TextEmbeddings inherits CLIPTextEmbeddings (no LayerNorm) but JSON claims `{embedding, layer_norm}`. Subagent likely confused with BERT-style embeddings.

5. **Multimodal models (`AriaModel`, `Aimv2Model`, etc.) miss vision/text sub-model refs because they use `Class._from_config()` instead of `Class(config)`.** AST walker doesn't unwrap classmethod calls.

6. **`ZeroPad2d` (torch builtin) leaks into composes list as an HF class name rather than being filtered.**

---

## Honest assessment

I audited 4 of the first 50 folders deeply. Found 5 systematic issue classes that almost certainly recur in the other 46. My current confidence: **the table is broadly correct but has an estimated 10-30% per-row error rate on missing kernels (mainly activations) and a smaller (~5%) rate of mismapped wiring-vs-direct rows.**

To make this paper-grade I'd need to:
- Fix the AST extractor to handle ACT2FN subscripts (re-resolve config.hidden_act to a kernel)
- Fix the AST extractor to unwrap `_from_config` method calls
- Add a JSON post-pass that catches "wiring classes mistakenly mapped to L2 files"
- Re-audit the bert / clip / siglip / efficientnet inheritance graph (since many folders inherit from these)

Cannot honestly call the audit done without these fixes.


