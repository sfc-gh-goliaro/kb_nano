apertus: composable — Llama-style decoder; rotary + RMSNorm + MLP + standard attention all in L1.
big_bird: composable — block-sparse attention is gather + bmm + softmax (all L1 primitives, no custom CUDA).
bigbird_pegasus: composable — BigBird block-sparse attention plus seq2seq encoder-decoder; pure-torch primitives only.
bloom: composable — ALiBi bias add to attention scores composes from L1 dense_attention + linear; no new primitive.
blt: composable — byte-level transformer using standard attention/linear/embedding/RMSNorm.
bridgetower: composable — nn.MultiheadAttention in BridgeTowerLinkTower decomposes to L1 linear + sdpa + linear; no new primitive.
bros: composable — BERT-style encoder with 2D box-position embeddings (extra embedding tables); standard primitives only.
canine: composable — character-level encoder; nn.MaxPool1d already covered by tasks/baseline/L1/max_pool1d.py.
chameleon: composable — Llama-style text decoder + VQ-VAE image tokenizer with conv2d/groupnorm; all L1 primitives present.
chmv2: composable — DPT-style depth model; nn.ConvTranspose2d covered by tasks/baseline/L1/conv_transpose2d.py.
clap: composable — RoBERTa text tower + Swin-style audio tower; AdaptiveAvgPool1d/2d both have L1 wrappers.
clvp: composable — speaker-conditioned text/voice encoder using standard attention + RMSNorm + rotary.
codegen: composable — GPT-J-style decoder with rotary; standard primitives only.
cohere: composable — Llama-style decoder with QK-LayerNorm; all primitives in L1.
cohere2: composable — Cohere2 sliding-window attention is mask + dense attention; standard primitives only.
cohere2_vision: composable — vision tower (ViT-style) + Cohere2 text decoder; standard primitives only.
cohere_asr: composable — Cohere2 backbone + audio encoder using conv1d/2d + attention; all L1 primitives present.
conditional_detr: composable — DETR variant with cross-attention; standard linear+attention+layer_norm primitives.
convbert: composable — span-based dynamic conv (unfold + linear + softmax); all primitives composable from L1 conv1d + linear + softmax.
convnext: composable — depthwise conv2d + layer_norm + GELU + linear; all in L1.
cpmant: composable — encoder/decoder with relative-position bias; standard linear+attention+layer_norm.
csm: composable — Conditional Speech Model; Llama-style backbone + audio codec head using standard primitives.
ctrl: composable — GPT-style decoder with sinusoidal positions; standard primitives only.
dac: composable — Descript Audio Codec; nn.ConvTranspose1d covered by tasks/baseline/L1/conv_transpose1d.py.
deberta: composable — disentangled-attention BERT; gather + matmul + softmax all in L1.
deberta_v2: composable — same as deberta plus log-bucket relative positions; pure torch primitives.
deepseek_v4: composable — DeepSeek MLA + MoE; existing L2 deepseek_mla_attention + deepseek_moe primitives cover it.
dinat: composable — Dilated Neighborhood Attention via gather/window-extract loop; AdaptiveAvgPool1d in L1; no NATTEN custom kernel needed.
doge: composable — Doge-LM style decoder using standard attention + RMSNorm + rotary.
donut: composable — Swin v1-style document encoder; AdaptiveAvgPool1d covered by L1.
dpt: composable — DPT depth head; nn.ConvTranspose2d covered by tasks/baseline/L1/conv_transpose2d.py.
edgetam: composable — SAM-2 derivative; nn.ConvTranspose2d in mask decoder covered by L1 conv_transpose2d.
edgetam_video: composable — video extension of edgetam with memory attention; same conv_transpose2d covered by L1.
efficientnet: composable — Squeeze-and-Excite uses AdaptiveAvgPool2d, already in L1.
emu3: partial — image tokenizer uses nn.BatchNorm3d (lines 452, 459); kb-nano L1 only has BatchNorm2d, needs simple BatchNorm3d torch.nn fallback wrapper.
eomt: composable — EomtScaleLayer uses nn.ConvTranspose2d, covered by L1 conv_transpose2d (grid_sample on line 119 is training-only matcher).
eomt_dinov3: composable — DINOv3-backed eomt; same nn.ConvTranspose2d in L1.
ernie4_5_vl_moe: composable — Ernie 4.5 VL with MoE; standard MoE + attention + vision encoder primitives all in L1.
evolla: composable — protein-language model; standard attention/MLP/embedding primitives.
exaone4_5: composable — LG ExaOne v4.5 decoder; Llama-style with RMSNorm + rotary, all L1.
falcon: composable — Falcon decoder with rotary + multi-query attention; all L1 primitives.
falcon_h1: composable — Falcon hybrid (Mamba2 + attention); existing L1 mamba ops + standard attention.
florence2: composable — Florence-2 vision-language with DaViT vision encoder; standard primitives only.
fnet: unsupported — FNet uses torch.fft.fft / torch.fft.fftn for Fourier mixing (lines 67-79); no FFT primitive in kb-nano L1 (explicitly listed as unsupported example).
focalnet: composable — focal modulation encoder; AdaptiveAvgPool1d in head pooler covered by L1.
gemma3n: composable — Conformer audio encoder uses einsum already decomposed to matmul (line 281+); standard primitives only.
gemma4_assistant: composable — speculative-decode helper for gemma4; reuses existing gemma4 L2 ops.
longformer: composable — sliding-chunks attention via pad + matmul + softmax; all primitives in L1.
longt5: composable — T5 with local + transient-global attention; existing L2 t5_attention plus pad/matmul/softmax suffice.
luke: composable — entity-aware BERT with two embedding tables and entity-attention biases; standard primitives only.
lxmert: composable — two-tower vision-language transformer; standard primitives only.
m2m_100: composable — multilingual seq2seq with sinusoidal embeddings; all L1 primitives.
marian: composable — encoder-decoder with sinusoidal positions; all standard primitives.
maskformer: composable — DETR-style + ResNet/Swin backbone; AdaptiveAvgPool1d covered by L1, ConvTranspose2d covered by L1.
maskformer_swin: composable — Swin v1 backbone variant; AdaptiveAvgPool1d covered by L1, no other unusual ops.
mbart: composable — multilingual seq2seq based on BART; standard primitives only.
mgp_str: composable — Multi-Granularity Prediction Scene Text Recognition; ViT encoder + character heads with standard primitives.
mimi: composable — Mimi neural codec; nn.ConvTranspose1d covered by L1 conv_transpose1d, nn.ELU covered by L1 elu.
minicpmv4_6: composable — MiniCPM-V 4.6 multimodal; vision encoder + LLM with standard primitives.
minimax: composable — MiniMax MoE with LightningAttention (chunked linear-attention via matmul/exp); all primitives in L1.
mra: unsupported — MRA self-attention requires kernels-community/mra CUDA extension (mra_cuda_kernel.index_max, mm_to_sparse, sparse_dense_mm) at lines 82/148/186; genuinely needs new compute primitive.
