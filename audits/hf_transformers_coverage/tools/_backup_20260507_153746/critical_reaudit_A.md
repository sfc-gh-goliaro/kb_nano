autoformer: partial — AutoCorrelation block calls torch.fft.rfft/irfft (line 509-512); kb-nano has no FFT L1 op but torch.fft is built-in, all other ops (linear, layernorm, conv1d, softmax, topk) are in L1.
emu3: partial — VQVAE decoder uses nn.BatchNorm3d (line 452, 459); kb-nano has BatchNorm2d only, BN3d composes trivially as 3 BatchNorm2d slices; Conv3d, attention, RMSNorm, rotary are all in L1.
fastspeech2_conformer: partial — Conformer relative-position attention (matrix_bd shift_relative_position_tensor, line 389-445) decomposes to bmm+gather+softmax; Conv1d, GLU, LayerNorm all in L1; needs only a small gather/index helper.
fnet: partial — FNetBasicFourierTransform calls torch.fft.fft/fftn (line 78, 149); no FFT L1 op in kb-nano but torch.fft is built-in; all other ops (linear, layernorm, gelu) are in L1.
funnel: composable — uses nn.functional.avg_pool2d/max_pool2d (line 271-275, both in L1), torch.gather for relative-shift (line 171, 180), plus standard linear/layernorm/softmax all in L1.
mlp_mixer: composable — folder absent in HF Transformers pinned snapshot; kb-nano's L4/mobilenetv4-style MLP-Mixer would only need linear+layernorm+gelu+transpose, all in L1 (no missing primitive).
mobilenetv4: composable — folder absent in HF Transformers pinned snapshot; kb-nano already has L4/mobilenetv4.py built from L1 conv2d/batch_norm2d/hardswish/adaptive_avg_pool2d (no missing primitive).
mra: unsupported — loads kernels-community/mra custom CUDA kernel (line 57) for index_max/mm_to_sparse/sparse_dense_mm (line 82, 148, 186); genuinely new compute primitive with no L1 equivalent.
nystromformer: composable — iterative_inv (Moore-Penrose, line 141-160) is pure matmul+identity; landmark attention is matmul+softmax; Conv2d residual all in L1.
parakeet: composable — Conformer with Conv2d subsampling, GLU, relative-position attention (rel_shift uses pad/view, line 357-363); all primitives (conv1d, conv2d, linear, layernorm, softmax, rotary) in L1.
pe_audio: composable — Snake1d activation (line 47-59) is pure elementwise (x + sin^2(alpha*x)/alpha); Conv1d/ConvTranspose1d, RMSNorm, rotary attention all in L1.
pegasus_x: composable — PegasusXGlobalLocalAttention (line 273) is just reshape-into-blocks + dense_attention + concat global tokens (line 466-493); all decomposable from L1 dense_attention/sdpa + softmax.
pp_doclayout_v2: composable — MultiScaleDeformableAttention uses nn.functional.grid_sample (line 603) which kb-nano has as L1 (grid_sample.py + rtdetrv2_deformable_attention.py); Conv2d, LayerNorm, attention all in L1.
qwen2_5_omni: composable — SnakeBeta activation (line 3071, pure elementwise like Snake1d), Conv1d/Conv2d/Conv3d/ConvTranspose1d, M-RoPE (kb-nano has mrope.py), audio/vision/talker/code2wav all decompose to existing L1 ops.
