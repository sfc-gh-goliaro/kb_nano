## xglm
- **src**: modeling_xglm.py
- **hidden_act**: gelu (config field `activation_function`, default "gelu")
- **status**: composable
- **classes**:
  - **`XGLMScaledWordEmbedding`** [compute, inherits `nn.Embedding`]: `L1/embedding.py` (multiplies output by embed_scale)
  - **`XGLMSinusoidalPositionalEmbedding`** [compute]: precomputed sin/cos table + index_select; closest is `L1/sinusoidal_embed.py` (no exact L2 match — table is built and `index_select`'d, not standard rotary)
  - **`XGLMAttention`** [compute]: BART-style enc-dec attention with optional cross-attn, q/k/v + bmm + softmax + bmm + out_proj. No KV-cache RoPE. Closest is `L2/whisper_attention.py` (similar enc/dec/cross variants), but XGLM is decoder-causal only with optional cross-attn. Decompose: `L1/linear.py + L1/dense_attention.py + L1/store_kvcache.py` (no exact L2 match)
  - **`XGLMDecoderLayer`** [wiring]: wires `XGLMAttention` (self-attn), optional `XGLMAttention` (encoder cross-attn), `nn.LayerNorm` (×2 or ×3 with cross-attn), `nn.Linear` (fc1, fc2), activation `gelu`; direct `L1/layer_norm.py + L1/linear.py + L1/gelu.py`
  - **`XGLMModel`** [wiring]: wires `XGLMScaledWordEmbedding`, `XGLMSinusoidalPositionalEmbedding`, `XGLMDecoderLayer` (×N), final `nn.LayerNorm`; direct `L1/layer_norm.py`
  - **`XGLMForCausalLM`** [wiring]: wires `XGLMModel`; direct `L1/linear.py` (lm_head)

## xlm
- **src**: modeling_xlm.py
- **hidden_act**: gelu (config `gelu_activation=True` selects `gelu`, else `relu`)
- **status**: composable
- **classes**:
  - **`MultiHeadAttention`** [compute]: BART-style enc/dec attn — q_lin/k_lin/v_lin Linear + matmul + softmax + matmul + out_lin. With optional KV cache via EncoderDecoderCache. No RoPE. Decompose: `L1/linear.py + L1/dense_attention.py + L1/store_kvcache.py` (no exact L2 match — closest is `L2/whisper_attention.py` for the cross-attn flavor)
  - **`TransformerFFN`** [compute]: `L1/linear.py + L1/gelu.py + L1/linear.py` (2-layer MLP fc1->gelu->fc2; gelu by config, else relu) — closest `L2/encoder_mlp.py` but XLM applies LayerNorm post-FFN inside XLMModel, not as part of the FFN class
  - **`XLMPredLayer`** [compute]: cross-entropy/adaptive-softmax head; `L1/linear.py` (when `asm=False`); when `asm=True` uses `nn.AdaptiveLogSoftmaxWithLoss` (no kb-nano kernel for adaptive softmax)
  - **`XLMModel`** [wiring]: wires `MultiHeadAttention` (×N), `TransformerFFN` (×N), `nn.LayerNorm` (layer_norm1×N, layer_norm2×N, layer_norm_emb), `nn.Embedding` (position_embeddings, lang_embeddings optional, embeddings); direct `L1/embedding.py + L1/layer_norm.py`
  - **`XLMWithLMHeadModel`** [wiring]: wires `XLMModel`, `XLMPredLayer`
- **task heads (5)**: ForSequenceClassification, ForQuestionAnsweringSimple, ForQuestionAnswering, ForTokenClassification, ForMultipleChoice — base + linear (per-task)

## xlm_roberta
- **src**: modeling_xlm_roberta.py (modular_xlm_roberta.py present — most classes inherit from RoBERTa)
- **hidden_act**: gelu
- **status**: composable
- **classes**:
  - **`XLMRobertaEmbeddings`** [compute]: word + token_type + position + LayerNorm + Dropout (RoBERTa style — uses `padding_idx`-aware position id construction). Maps to `L2/xlm_roberta_embeddings.py`
  - **`XLMRobertaSelfAttention`** [compute]: q/k/v Linear + dispatch via ALL_ATTENTION_FUNCTIONS, optional KV cache update. Maps to `L2/encoder_attention.py`
  - **`XLMRobertaCrossAttention`** [compute]: q/k/v cross-attn with EncoderDecoderCache. Decompose: `L1/linear.py + L1/dense_attention.py + L1/store_kvcache.py` (no exact L2 match — encoder_attention.py is self-attn only)
  - **`XLMRobertaSelfOutput`** [compute]: dense + dropout + LayerNorm(x + residual). Maps to `L2/encoder_attention.py` (self.output portion)
  - **`XLMRobertaAttention`** [compute]: wrapper containing `self.self` (self/cross) + `self.output`. Maps to `L2/encoder_attention.py`
  - **`XLMRobertaIntermediate`** [compute]: `L1/linear.py + L1/gelu.py`
  - **`XLMRobertaOutput`** [compute]: dense + dropout + LayerNorm(x + residual). `L1/linear.py + L1/layer_norm.py`
  - **`XLMRobertaLayer`** [wiring]: wires `XLMRobertaAttention` (self), optional `XLMRobertaAttention` (cross), `XLMRobertaIntermediate`, `XLMRobertaOutput`. Maps to `L3/xlm_roberta_layer.py`
  - **`XLMRobertaEncoder`** [wiring]: wires `XLMRobertaLayer` (×N). Maps to `L3/xlm_roberta_encoder.py`
  - **`XLMRobertaPooler`** [compute]: `L1/linear.py + L1/tanh.py` (first-token pool)
  - **`XLMRobertaLMHead`** [compute]: `L1/linear.py + L1/gelu.py + L1/layer_norm.py + L1/linear.py`
  - **`XLMRobertaModel`** [wiring]: wires `XLMRobertaEmbeddings`, `XLMRobertaEncoder`, optional `XLMRobertaPooler`. Maps to `L3/xlm_roberta_model.py`
  - **`XLMRobertaForCausalLM`** [wiring]: wires `XLMRobertaModel`, `XLMRobertaLMHead`
  - **`XLMRobertaForMaskedLM`** [wiring]: wires `XLMRobertaModel`, `XLMRobertaLMHead`
- **task heads (4)**: ForSequenceClassification, ForMultipleChoice, ForTokenClassification, ForQuestionAnswering — base + linear (per-task)

## xlm_roberta_xl
- **src**: modeling_xlm_roberta_xl.py
- **hidden_act**: gelu
- **status**: composable
- **classes**:
  - **`XLMRobertaXLEmbeddings`** [compute]: word + token_type + position + Dropout (no LayerNorm here — moved to encoder). Decompose: `L1/embedding.py + L1/embedding.py + L1/embedding.py` (no exact L2 — differs from `L2/xlm_roberta_embeddings.py` by lacking LayerNorm)
  - **`XLMRobertaXLSelfAttention`** [compute]: same as RoBERTa SelfAttention. Maps to `L2/encoder_attention.py`
  - **`XLMRobertaXLCrossAttention`** [compute]: q/k/v cross-attn with EncoderDecoderCache. Decompose: `L1/linear.py + L1/dense_attention.py + L1/store_kvcache.py` (no exact L2 match)
  - **`XLMRobertaXLSelfOutput`** [compute]: dense + dropout + (x + residual) — no LayerNorm here (pre-norm variant). `L1/linear.py` (just linear+residual)
  - **`XLMRobertaXLAttention`** [compute]: pre-norm wrapper: applies `self_attn_layer_norm` first, then `self.self`, then `self.output` (dense + residual). Decompose: `L1/layer_norm.py + L2/encoder_attention.py` (no exact L2 — pre-norm flavor)
  - **`XLMRobertaXLIntermediate`** [compute]: `L1/linear.py + L1/gelu.py`
  - **`XLMRobertaXLOutput`** [compute]: dense + (x + residual). `L1/linear.py` (residual add)
  - **`XLMRobertaXLLayer`** [wiring]: wires `XLMRobertaXLAttention`, `XLMRobertaXLIntermediate`, `XLMRobertaXLOutput`, plus a `nn.LayerNorm` applied before intermediate (pre-norm); direct `L1/layer_norm.py`
  - **`XLMRobertaXLEncoder`** [wiring]: wires `XLMRobertaXLLayer` (×N), final `nn.LayerNorm`; direct `L1/layer_norm.py`
  - **`XLMRobertaXLPooler`** [compute]: `L1/linear.py + L1/tanh.py`
  - **`XLMRobertaXLLMHead`** [compute]: `L1/linear.py + L1/gelu.py + L1/layer_norm.py + L1/linear.py`
  - **`XLMRobertaXLClassificationHead`** [compute]: `L1/linear.py + L1/tanh.py + L1/linear.py` (drops first token, dense, tanh, dropout, out_proj)
  - **`XLMRobertaXLModel`** [wiring]: wires `XLMRobertaXLEmbeddings`, `XLMRobertaXLEncoder`, optional `XLMRobertaXLPooler`
  - **`XLMRobertaXLForCausalLM`** [wiring]: wires `XLMRobertaXLModel`, `XLMRobertaXLLMHead`
  - **`XLMRobertaXLForMaskedLM`** [wiring]: wires `XLMRobertaXLModel`, `XLMRobertaXLLMHead`
- **task heads (4)**: ForSequenceClassification, ForMultipleChoice, ForTokenClassification, ForQuestionAnswering — base + linear (per-task)

## xlnet
- **src**: modeling_xlnet.py
- **hidden_act**: gelu (config `ff_activation`, default "gelu")
- **status**: partial
- **classes**:
  - **`XLNetRelativeAttention`** [compute]: relative-positional two-stream attention with content + position + segment scoring via einsum, plus rel_shift, two query streams (h-stream and g-stream), nn.LayerNorm post-attention. **Architecturally specific** — closest is `L2/t5_attention.py` (rel-pos-bias) but XLNet's mechanism is materially different (two-stream, segment embeddings, position-based key, q/k/v/o/r raw-tensor projections rather than nn.Linear). Decompose: `L1/dense_attention.py + L1/layer_norm.py` (no exact L2 match — XLNet rel-attn is unique)
  - **`XLNetFeedForward`** [compute]: layer_1 -> activation (gelu) -> dropout -> layer_2 -> dropout -> LayerNorm(x + residual). `L1/linear.py + L1/gelu.py + L1/linear.py + L1/layer_norm.py` (post-norm 2-layer MLP — close to encoder MLP pattern but with internal residual+norm)
  - **`XLNetLayer`** [wiring]: wires `XLNetRelativeAttention`, `XLNetFeedForward`
  - **`XLNetModel`** [wiring]: wires `nn.Embedding` (word_embedding), `XLNetLayer` (×N); direct `L1/embedding.py`
  - **`XLNetLMHeadModel`** [wiring]: wires `XLNetModel`; direct `L1/linear.py` (lm_loss)
- **task heads (5)**: ForSequenceClassification, ForTokenClassification, ForMultipleChoice, ForQuestionAnsweringSimple, ForQuestionAnswering — base + linear (per-task)

## xlstm
- **src**: modeling_xlstm.py
- **hidden_act**: n/a (uses fixed `nn.SiLU()` for FFN gate, fixed `nn.Sigmoid()` for output gate; config has no `hidden_act`)
- **status**: partial (Mamba-style; xLSTM uses an external `xlstm.xlstm_large` package's RMSNorm + mLSTMBlock when available — when unavailable the in-file fallback classes are used)
- **classes** (in-file fallback path, only entered when external `xlstm` package not installed):
  - **`xLSTMRMSNorm`** [compute]: standard RMSNorm with optional weight + optional bias (`force_float32_reductions`). Maps to `L1/rms_norm.py` (closest; bias variant not in kb-nano)
  - **`xLSTMMultiHeadLayerNorm`** [compute]: per-head LayerNorm over last dim then reshape (centered, var-based). Decompose: `L1/layer_norm.py` (applied per-head) — no exact L2 match
  - **`xLSTMBackend`** [compute]: dispatches to mLSTM kernel backends (recurrent / chunkwise). External; closest in spirit to `L1/lstm.py` but xLSTM-specific (no kb-nano match for matrix-LSTM exponential gating)
  - **`xLSTMFeedForward`** [compute]: SwiGLU-style gated FFN using `nn.SiLU()` (proj_up_gate * silu(proj_up) -> proj_down or fused gate_z). Maps to `L2/llama_mlp.py` (SwiGLU pattern; activation hard-coded to silu)
  - **`xLSTMLayer`** [compute]: q/k/v/igate/fgate/ogate Linear projections + soft_cap on gates + xLSTMBackend + xLSTMMultiHeadLayerNorm + sigmoid(o_preact)*h_norm + out_proj. **Architecturally specific** — no kb-nano kernel for matrix-LSTM with exponential gates. Decompose: `L1/linear.py + L1/sigmoid.py + L1/layer_norm.py` (no exact L2 match)
  - **`xLSTMBlock`** [wiring, GradientCheckpointingLayer]: wires `xLSTMRMSNorm` (×2: norm_mlstm, norm_ffn), `xLSTMLayer`, `xLSTMFeedForward`
  - **`xLSTMModel`** [wiring]: wires `nn.Embedding` (embeddings), `xLSTMBlock` (×N), `xLSTMRMSNorm` (out_norm); direct `L1/embedding.py`
  - **`xLSTMForCausalLM`** [wiring]: wires `xLSTMModel`; direct `L1/linear.py` (lm_head)

## xmod
- **src**: modeling_xmod.py (no modular file)
- **hidden_act**: gelu
- **status**: composable
- **classes**:
  - **`XmodEmbeddings`** [compute]: word + token_type + position + LayerNorm + Dropout (RoBERTa-style). Maps to `L2/xlm_roberta_embeddings.py` (RoBERTa embeddings family)
  - **`XmodSelfAttention`** [compute]: q/k/v Linear + dispatch via ALL_ATTENTION_FUNCTIONS (RoBERTa-style). Maps to `L2/encoder_attention.py`
  - **`XmodCrossAttention`** [compute]: q/k/v cross-attn with EncoderDecoderCache. Decompose: `L1/linear.py + L1/dense_attention.py + L1/store_kvcache.py` (no exact L2 match)
  - **`XmodSelfOutput`** [compute]: dense + dropout + (x + residual). `L1/linear.py`
  - **`XmodAttention`** [compute]: pre/post-norm wrapper: applies `self.output.LayerNorm` if `pre_norm`, else after. `L1/layer_norm.py + L2/encoder_attention.py` (close but pre/post-norm variant)
  - **`XmodIntermediate`** [compute]: `L1/linear.py + L1/gelu.py`
  - **`XmodAdapter`** [compute]: per-language bottleneck adapter (dense1 -> gelu -> dense2). `L1/linear.py + L1/gelu.py + L1/linear.py`
  - **`XmodOutput`** [compute]: dense + dropout + residual + per-language adapter dispatch (looks up `adapter_modules[lang_key]`, applies `XmodAdapter`, optional adapter_layer_norm). Decompose: `L1/linear.py + L1/layer_norm.py + (per-lang) XmodAdapter` (no kb-nano kernel — XMOD-specific language adapter routing)
  - **`XmodLayer`** [wiring, GradientCheckpointingLayer]: wires `XmodAttention`, optional `XmodAttention` (cross-attn), `XmodIntermediate`, `XmodOutput` (adapter-dispatch)
  - **`XmodEncoder`** [wiring]: wires `XmodLayer` (×N), optional final `nn.LayerNorm` (pre-norm); direct `L1/layer_norm.py`
  - **`XmodPooler`** [compute]: `L1/linear.py + L1/tanh.py`
  - **`XmodLMHead`** [compute]: `L1/linear.py + L1/gelu.py + L1/layer_norm.py + L1/linear.py`
  - **`XmodClassificationHead`** [compute]: `L1/linear.py + L1/tanh.py + L1/linear.py`
  - **`XmodModel`** [wiring]: wires `XmodEmbeddings`, `XmodEncoder`, optional `XmodPooler`
  - **`XmodForCausalLM`** [wiring]: wires `XmodModel`, `XmodLMHead`
  - **`XmodForMaskedLM`** [wiring]: wires `XmodModel`, `XmodLMHead`
- **task heads (4)**: ForSequenceClassification, ForMultipleChoice, ForTokenClassification, ForQuestionAnswering — base + linear (per-task)

## yolos
- **src**: modeling_yolos.py
- **hidden_act**: gelu
- **status**: composable
- **classes**:
  - **`YolosEmbeddings`** [compute]: cls token + detection_tokens + patch_embeddings + position_embeddings (with bicubic interpolation). Decompose: `L1/embedding.py + L1/conv2d.py + L2/vision_pos_embed_interpolate.py` (close — uses `nn.functional.interpolate` for pos-embed resize)
  - **`InterpolateInitialPositionEmbeddings`** [compute]: bicubic resize of patch position embeddings, concat cls + patch + det. Decompose: uses `F.interpolate`; closest is `L2/vision_pos_embed_interpolate.py`
  - **`InterpolateMidPositionEmbeddings`** [compute]: same idea but operates over depth dimension (mid-stage pos embeds). Decompose: same as above
  - **`YolosPatchEmbeddings`** [compute]: Conv2d + flatten/transpose. `L1/conv2d.py` (close — `L2/vision_patch_embed.py` covers ViT patch embed pattern)
  - **`YolosSelfAttention`** [compute]: ViT-style q/k/v Linear (no causal) + dispatch via ALL_ATTENTION_FUNCTIONS. Maps to `L2/vit_encoder_attention.py` (ViT family) or `L2/encoder_attention.py`
  - **`YolosSelfOutput`** [compute]: dense + dropout (no LayerNorm — applied externally as ViT pre-norm). `L1/linear.py`
  - **`YolosAttention`** [compute]: ViT-style wrapper containing `self.attention` + `self.output`. Maps to `L2/vit_encoder_attention.py`
  - **`YolosIntermediate`** [compute]: `L1/linear.py + L1/gelu.py`
  - **`YolosOutput`** [compute]: `L1/linear.py` (dense + dropout + residual; no LayerNorm — pre-norm)
  - **`YolosLayer`** [wiring, GradientCheckpointingLayer]: wires `YolosAttention`, `YolosIntermediate`, `YolosOutput`, two `nn.LayerNorm` (layernorm_before, layernorm_after); direct `L1/layer_norm.py`. Maps to `L3/vit_encoder_block.py` shape
  - **`YolosEncoder`** [wiring]: wires `YolosLayer` (×N), optional `mid_position_embeddings` Parameter + `InterpolateMidPositionEmbeddings`
  - **`YolosModel`** [wiring]: wires `YolosEmbeddings`, `YolosEncoder`, final `nn.LayerNorm`, optional `YolosPooler`; direct `L1/layer_norm.py`
  - **`YolosPooler`** [compute]: `L1/linear.py + L1/tanh.py`
  - **`YolosMLPPredictionHead`** [compute]: stack of N `nn.Linear` with ReLU between. `L1/linear.py + L1/relu.py` (×N)
- **task heads (1)**: ForObjectDetection — base + YolosMLPPredictionHead (×2: class + bbox) (per-task)

## yoso
- **src**: modeling_yoso.py
- **hidden_act**: gelu
- **status**: partial (custom CUDA LSH cumulation kernel — no kb-nano equivalent)
- **classes**:
  - **`YosoCumulation`** / **`YosoLSHCumulation`** [autograd.Function] — skipped per rule (autograd Functions)
  - **`YosoEmbeddings`** [compute]: word + position + token_type + LayerNorm + Dropout (Nystromformer-style; offsets position_ids by +2). Maps to `L2/encoder_embeddings.py` (BERT-style)
  - **`YosoSelfAttention`** [compute]: q/k/v Linear + LSH-cumulation custom CUDA kernel (`YosoCumulation` / `YosoLSHCumulation`) for sparse attention; optional Conv2d. **Architecturally specific** — no kb-nano kernel for YOSO LSH bernoulli sampling. Decompose: `L1/linear.py + L1/conv2d.py` (no exact L2 match — YOSO LSH attn is unique)
  - **`YosoSelfOutput`** [compute]: dense + dropout + LayerNorm(x + residual) — copy of BertSelfOutput. Maps to `L2/encoder_attention.py` (self.output portion)
  - **`YosoAttention`** [compute]: wrapper containing `self.self` + `self.output`. Wires self/output components; the LSH kernel makes this not directly map to encoder_attention.py
  - **`YosoIntermediate`** [compute]: `L1/linear.py + L1/gelu.py` (BertIntermediate copy)
  - **`YosoOutput`** [compute]: dense + dropout + LayerNorm(x + residual) — BertOutput copy. `L1/linear.py + L1/layer_norm.py`
  - **`YosoLayer`** [wiring, GradientCheckpointingLayer]: wires `YosoAttention`, `YosoIntermediate`, `YosoOutput`
  - **`YosoEncoder`** [wiring]: wires `YosoLayer` (×N)
  - **`YosoPredictionHeadTransform`** [compute]: `L1/linear.py + L1/gelu.py + L1/layer_norm.py` (BertPredictionHeadTransform copy)
  - **`YosoLMPredictionHead`** [wiring]: wires `YosoPredictionHeadTransform`; direct `L1/linear.py`
  - **`YosoOnlyMLMHead`** [wiring]: wires `YosoLMPredictionHead`
  - **`YosoModel`** [wiring]: wires `YosoEmbeddings`, `YosoEncoder`
  - **`YosoForMaskedLM`** [wiring]: wires `YosoModel`, `YosoOnlyMLMHead`
  - **`YosoClassificationHead`** [compute]: `L1/linear.py + L1/tanh.py + L1/linear.py`
- **task heads (4)**: ForSequenceClassification, ForMultipleChoice, ForTokenClassification, ForQuestionAnswering — base + linear (per-task)

## youtu
- **src**: modeling_youtu.py (modular_youtu.py present — `YoutuConfig(DeepseekV3Config)`, `YoutuMLP(Qwen3MLP)`, `YoutuAttention(DeepseekV3Attention)`, `YoutuDecoderLayer(LlamaDecoderLayer)`, `YoutuModel(LlamaModel)`, `YoutuForCausalLM(LlamaForCausalLM)`)
- **hidden_act**: silu
- **status**: composable
- **classes**:
  - **`YoutuRMSNorm`** [compute]: standard RMSNorm. Maps to `L1/rms_norm.py`
  - **`YoutuRotaryEmbedding`** [compute, inherits `LlamaRotaryEmbedding` (modular)]: standard Llama RoPE with dynamic scaling. Maps to `L1/rotary_emb.py` (or `L1/yarn_rotary_emb.py` when YaRN is selected via rope_parameters)
  - **`YoutuMLP`** [compute, inherits `Qwen3MLP` (modular)]: SwiGLU (gate_proj * silu * up_proj -> down_proj). Maps to `L2/llama_mlp.py`
  - **`YoutuAttention`** [compute, inherits `DeepseekV3Attention` (modular)]: MLA-style with q_lora_rank/kv_lora_rank, qk_nope/qk_rope split, kv_b_proj, RoPE on rotary part. Maps to `L2/deepseek_mla_attention.py`
  - **`YoutuDecoderLayer`** [wiring, inherits `LlamaDecoderLayer` (modular)]: wires `YoutuAttention`, `YoutuMLP`, `YoutuRMSNorm` (×2: input_layernorm, post_attention_layernorm)
  - **`YoutuModel`** [wiring, inherits `LlamaModel` (modular)]: wires `nn.Embedding` (embed_tokens), `YoutuDecoderLayer` (×N), `YoutuRMSNorm` (norm), `YoutuRotaryEmbedding` (rotary_emb); direct `L1/embedding.py`
  - **`YoutuForCausalLM`** [wiring, inherits `LlamaForCausalLM` (modular)]: wires `YoutuModel`; direct `L1/linear.py` (lm_head)

## zamba
- **src**: modeling_zamba.py (no modular file)
- **hidden_act**: gelu (also `hidden_mamba_act` for mamba activation, default "silu")
- **status**: partial (Mamba/SSM — uses `mamba-ssm` and `causal-conv1d` external CUDA kernels for fast path; slow_forward fallback)
- **classes**:
  - **`ZambaRMSNorm`** [compute]: standard RMSNorm (variance-based, weight only). Maps to `L1/rms_norm.py`
  - **`ZambaAttention`** [compute]: standard Mistral-style attention with q/k/v/o Linear, GQA via num_key_value_groups, scaling=`(head_dim/2)**-0.5` (unique due to attention_hidden_size=2*hidden_size concat trick), dispatch via ALL_ATTENTION_FUNCTIONS. No RoPE in Zamba v1. Maps to `L2/attention.py` (close — but no RoPE; scaling factor differs)
  - **`ZambaMambaMixer`** [compute]: Mamba selective-scan SSM with multi-head (n_mamba_heads), in_proj/x_proj/dt_proj/out_proj, conv1d, A_log/D parameters, optionally fast `selective_scan_fn`/`causal_conv1d_fn`. Maps to kb-nano Mamba ops in `L4/mamba.py` chain; closest L1 ops are `L1/causal_conv1d.py` and `L1/silu.py`. (no exact L2 match for n_mamba_heads multi-head Mamba — kb-nano `L4/mamba.py` is single-head)
  - **`ZambaMLP`** [compute, copy of MistralMLP]: SwiGLU (gate_proj * gelu * up_proj -> down_proj; activation per `hidden_act`=gelu — note: this uses gelu rather than silu despite the SwiGLU shape). Maps to `L2/llama_mlp.py` (SwiGLU pattern; activation differs)
  - **`ZambaAttentionDecoderLayer`** [wiring]: wires `ZambaAttention` (self_attn), `ZambaMLP` (feed_forward), `ZambaRMSNorm` (×2: input_layernorm, pre_ff_layernorm); concatenates `[hidden_states, original_hidden_states]` before input_layernorm
  - **`ZambaMambaDecoderLayer`** [wiring, GradientCheckpointingLayer]: wires `ZambaMambaMixer`, `ZambaRMSNorm`
  - **`ZambaHybridLayer`** [wiring, GradientCheckpointingLayer]: wires `ZambaAttentionDecoderLayer` (shared_transf), `nn.Linear` (linear), `ZambaMambaDecoderLayer` (mamba_decoder); direct `L1/linear.py`
  - **`ZambaModel`** [wiring]: wires `nn.Embedding` (embed_tokens), mix of `ZambaHybridLayer` and `ZambaMambaDecoderLayer` per `layers_block_type`, `ZambaRMSNorm` (final_layernorm); direct `L1/embedding.py`
  - **`ZambaForCausalLM`** [wiring]: wires `ZambaModel`; direct `L1/linear.py` (lm_head)
- **task heads (1)**: ForSequenceClassification — base + linear (per-task)

## zamba2
- **src**: modeling_zamba2.py (modular_zamba2.py present)
- **hidden_act**: gelu
- **status**: partial (Mamba2 with SSD chunked scan, plus shared transformer with LoRA-style adapters)
- **classes**:
  - **`Zamba2RMSNormGated`** [compute]: RMSNorm with optional silu(gate) modulation, group-wise normalization (group_size). Closest is `L1/rms_norm_gated.py`
  - **`Zamba2RMSNorm`** [compute]: standard RMSNorm. Maps to `L1/rms_norm.py`
  - **`Zamba2RotaryEmbedding`** [compute]: standard Llama-style RoPE (only used when `use_mem_rope=True`). Maps to `L1/rotary_emb.py`
  - **`Zamba2Attention`** [compute]: q/k/v/o Linear with optional shared adapter `nn.Sequential(Linear,Linear)` (LoRA-style) for tied transformer blocks; GQA; optional RoPE; scaling=`(head_dim/2)**-0.5`. Maps to `L2/attention.py` plus per-block adapter logic (no exact kb-nano kernel — adapter routing is Zamba2-specific)
  - **`Zamba2MambaMixer`** [compute]: Mamba2 SSD-style chunked scan; uses external `mamba_chunk_scan_combined` / `selective_state_update` kernels. Closest kb-nano is `L4/mamba2.py` chain (no exact L2 match for Mamba2 mixer)
  - **`Zamba2MLP`** [compute]: gate_up_proj fused Linear + per-block adapter `nn.Sequential(Linear,Linear)` (LoRA-style) + activation (gelu) on chunked gate -> down_proj. Maps to `L2/llama_mlp.py` (SwiGLU shape; gelu activation; adapters are Zamba2-specific addition)
  - **`Zamba2AttentionDecoderLayer`** [wiring]: wires `Zamba2Attention`, `Zamba2MLP`, `Zamba2RMSNorm` (×2)
  - **`Zamba2MambaDecoderLayer`** [wiring, GradientCheckpointingLayer]: wires `Zamba2MambaMixer`, `Zamba2RMSNorm`
  - **`Zamba2HybridLayer`** [wiring, GradientCheckpointingLayer]: wires `Zamba2AttentionDecoderLayer` (shared_transformer), `nn.Linear` (linear), `Zamba2MambaDecoderLayer` (mamba_decoder); direct `L1/linear.py`
  - **`Zamba2Model`** [wiring]: wires `nn.Embedding`, mix of `Zamba2HybridLayer`/`Zamba2MambaDecoderLayer`, `Zamba2RMSNorm` (final_layernorm), optional `Zamba2RotaryEmbedding`; direct `L1/embedding.py`
  - **`Zamba2ForCausalLM`** [wiring]: wires `Zamba2Model`; direct `L1/linear.py` (lm_head)
- **task heads (1)**: ForSequenceClassification — base + linear (per-task)

## zoedepth
- **src**: modeling_zoedepth.py (no modular file)
- **hidden_act**: gelu (used for ReassembleStage readout_projects)
- **status**: partial (depth-estimation specific — many heads have no kb-nano equivalent: log-binomial softmax, attractor layers)
- **classes**:
  - **`ZoeDepthReassembleStage`** [compute, copied from DPT]: wires `ZoeDepthReassembleLayer` (×N), optional `nn.Sequential(nn.Linear, gelu)` per stage when `readout_type=="project"`. Operates on hidden states from backbone. `L1/linear.py + L1/gelu.py` for readout
  - **`ZoeDepthReassembleLayer`** [compute, copied from DPT]: `nn.Conv2d` (1x1 projection) + `nn.ConvTranspose2d` / `nn.Conv2d` / Identity (resize). `L1/conv2d.py + L1/conv_transpose2d.py`
  - **`ZoeDepthFeatureFusionStage`** [wiring]: wires `ZoeDepthFeatureFusionLayer` (×N)
  - **`ZoeDepthPreActResidualLayer`** [compute]: ReLU + Conv2d + optional BatchNorm2d + ReLU + Conv2d + optional BatchNorm2d + residual. `L1/relu.py + L1/conv2d.py + L1/batch_norm2d.py`
  - **`ZoeDepthFeatureFusionLayer`** [compute, wiring-ish]: wires `ZoeDepthPreActResidualLayer` (×2), `nn.Conv2d` (projection); uses `F.interpolate` for upsample; direct `L1/conv2d.py + L1/interpolate.py`
  - **`ZoeDepthNeck`** [wiring]: wires `ZoeDepthReassembleStage`, `ZoeDepthFeatureFusionStage`, `nn.Conv2d` (×N convs); direct `L1/conv2d.py`
  - **`ZoeDepthRelativeDepthEstimationHead`** [compute]: stack of `nn.Conv2d` + `nn.Upsample` + ReLU. `L1/conv2d.py + L1/relu.py + L1/interpolate.py`
  - **`LogBinomialSoftmax`** [compute]: log binomial distribution with stirling approximation (no kb-nano kernel). Decompose: pure tensor ops (log, softmax) — `L1/softmax.py`
  - **`ZoeDepthConditionalLogBinomialSoftmax`** [compute]: `nn.Sequential(Conv2d, GELU, Conv2d, Softplus)` MLP + `LogBinomialSoftmax`. `L1/conv2d.py + L1/gelu.py + L1/softplus.py`
  - **`ZoeDepthSeedBinRegressor`** [compute]: Conv2d + ReLU + Conv2d + (ReLU or Softplus). `L1/conv2d.py + L1/relu.py` (or `L1/softplus.py`)
  - **`ZoeDepthAttractorLayer`** [compute]: Conv2d + ReLU + Conv2d + ReLU then attractor math (`inv_attractor` tensor ops, `F.interpolate`). `L1/conv2d.py + L1/relu.py + L1/interpolate.py` (no exact match — attractor math is ZoeDepth-specific)
  - **`ZoeDepthAttractorLayerUnnormed`** [compute]: same shape as Attractor but with Softplus instead of ReLU at end. `L1/conv2d.py + L1/relu.py + L1/softplus.py + L1/interpolate.py`
  - **`ZoeDepthProjector`** [compute]: Conv2d + ReLU + Conv2d. `L1/conv2d.py + L1/relu.py`
  - **`ZoeDepthMultiheadAttention`** [compute]: standard q/k/v/out_proj Linear + softmax(QK^T/sqrt(d))V (non-causal, batch-first). Decompose: `L1/linear.py + L1/dense_attention.py` (close to `L2/encoder_attention.py` self-only but no LayerNorm)
  - **`ZoeDepthTransformerEncoderLayer`** [wiring]: wires `ZoeDepthMultiheadAttention`, `nn.Linear` (×2), `nn.LayerNorm` (×2), activation (relu); direct `L1/linear.py + L1/layer_norm.py + L1/relu.py`
  - **`ZoeDepthPatchTransformerEncoder`** [wiring]: wires `ZoeDepthTransformerEncoderLayer` (×N), `nn.Conv2d` (embedding_convPxP); direct `L1/conv2d.py`
  - **`ZoeDepthMLPClassifier`** [compute]: `L1/linear.py + L1/relu.py + L1/linear.py`
  - **`ZoeDepthMultipleMetricDepthEstimationHeads`** [wiring]: wires `nn.Conv2d`, `ZoeDepthPatchTransformerEncoder`, `ZoeDepthMLPClassifier`, `ZoeDepthSeedBinRegressor` (per-config), `ZoeDepthProjector` (×N), `ZoeDepthAttractorLayer`/`Unnormed` (×N), `ZoeDepthConditionalLogBinomialSoftmax`
  - **`ZoeDepthMetricDepthEstimationHead`** [wiring]: same family as the multi version but single config; wires `nn.Conv2d`, `ZoeDepthSeedBinRegressor`, `ZoeDepthProjector` (×N), `ZoeDepthAttractorLayer`/`Unnormed` (×N), `ZoeDepthConditionalLogBinomialSoftmax`
- **task heads (1)**: ForDepthEstimation — uses `load_backbone(config)` (a backbone, e.g. BEiT) + `ZoeDepthNeck` + `ZoeDepthRelativeDepthEstimationHead` + (multi or single) `ZoeDepthMetricDepthEstimationHead` (per-task)
