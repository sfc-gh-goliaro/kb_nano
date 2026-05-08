qwen3_omni_moe: composable — Code2Wav DAC vocoder uses SnakeBeta (x + sin^2(alpha*x)/alpha, pure elementwise) plus Conv1d/ConvTranspose1d/Linear/RMSNorm; all primitives present in L1.
rag: composable — pure wrapper that delegates to AutoModel question_encoder (BERT/DPR) and AutoModelForSeq2SeqLM generator (BART/T5); no new compute, only tensor concat/marginalization.
reformer: unsupported — LSH self-attention with stable argsort, hash bucketing, chunked local/LSH attention pattern; no LSH primitive in L1 and not decomposable into simple ops.
rwkv: unsupported — explicitly imports kernels-community/rwkv (wkv_cuda forward/backward) for the time-mixing scan; no rwkv-v4 wkv L1 op (only rwkv7 variants).
seamless_m4t: partial — Conformer self-attention with relative-position embeddings needs the rel-shift trick (matmul + index/roll) on top of standard sdpa+conv1d+linear which are all in L1.
seamless_m4t_v2: partial — Conformer with relative_key embeddings reduces to einsum("bhld,lrd->bhlr") added to attn scores, i.e. one bmm + scale on top of L1 sdpa/linear/conv1d.
sew_d: partial — DeBERTa-v2-style disentangled (c2p/p2c) attention requires gather + bmm + log-bucket position index on top of standard linear/layer_norm/gelu (all in L1).
speecht5: partial — standard sdpa with additive relative_position_bias (bucketized embedding lookup) plus HiFi-GAN vocoder using only conv1d/conv_transpose1d/leaky_relu/group_norm/batch_norm which are in L1; the rel-pos bucket gather is the one missing piece.
timm_backbone: unsupported — requires the external `timm` library at import time; backbone is constructed via timm.create_model and is not implemented in HF source.
timm_wrapper: unsupported — same as timm_backbone, hard-imports `timm` and wraps a timm model; no in-tree compute kernels to map.
xlnet: partial — XLNetRelativeAttention two-stream is plain einsum/bmm + relative-shift reshape on top of linear/layer_norm/softmax (all in L1); only the rel-shift index gather is missing.
xlstm: unsupported — mLSTM chunkwise/recurrent kernels (matC/vecN/scaMinter state updates) are a novel matrix-LSTM scan; no matching L1 op (kb-nano has GLA/RWKV7/retention/gdn/kda but not mLSTM).
yoso: unsupported — explicitly imports kernels-community/yoso for fast_hash and lsh_cumulation; LSH-based attention approximation has no L1 counterpart.
zoedepth: composable — attractor / projector / metric-bin / log-binomial heads are stacks of Conv2d, ConvTranspose2d, Linear, softmax, relu, sigmoid — all primitives in L1.
