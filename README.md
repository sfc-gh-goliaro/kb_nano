# kb-nano

A standalone, high-performance inference engine supporting **LLMs** (Llama 3.1, Mixtral-8x7B, GPT-OSS, Kimi-Linear, Qwen3-Next, Qwen2-VL, Qwen3-VL), **diffusion models** (FLUX.1-dev, SDXL, HunyuanVideo-1.5), **detection models** (YOLOv10, RTDetrV2), **vision encoders** (SigLIP-2, DINOv3, SwinV2), **segmentation models** (SAM3.1), **protein structure prediction** (OpenFold3), audio models (Whisper), and **TTS models** (CosyVoice3) with tensor parallelism. No vLLM dependency at runtime — just PyTorch, Triton, and Flash Attention.

## Features

- **Llama 3.1** (8B, 70B) with frequency-scaled RoPE
- **Mixtral-8x7B** with fused Triton MoE grouped-GEMM kernels
- **GPT-OSS** (20B, 120B) MXFP4-quantized MoE with native Triton inference, YaRN RoPE, attention sinks, and sliding window
- **Kimi-Linear-48B-A3B-Instruct** hybrid recurrent + attention MoE: KDA (Kimi Delta Attention) linear-attention layers fused with MLA (Multi-head Latent Attention) full-attention layers, sigmoid-gated 256-expert MoE (top-8) with shared expert and `e_score_correction_bias` routing
- **Qwen3-Next-80B-A3B-Instruct** hybrid recurrent + attention MoE: 3:1 GDN (Gated Delta Net) linear attention to full attention with output gating + per-head QK-norm + partial RoPE, softmax-gated 512-expert MoE (top-10) with sigmoid-gated shared expert
- **FLUX.1-dev** diffusion transformer (text-to-image) with Flash Attention
- **SDXL** (Stable Diffusion XL) UNet-based text-to-image with dual CLIP text encoders
- **HunyuanVideo-1.5** 3D video diffusion transformer (text-to-video) with dual-stream joint attention, M-RoPE, and Qwen2.5-VL text encoder
- **YOLOv10** (`jameslahm/yolov10n`) NMS-free object detection with rank sorting
- **RTDetrV2** (`PekingU/rtdetr_v2_r101vd`) real-time detection transformer with deformable attention
- **Qwen2-VL / Qwen3-VL** vision-language models with image and video support
- **SigLIP-2** (google/siglip2-so400m-patch16-naflex) NaFlexVit vision encoder with MAP attention pooling
- **DINOv3** (facebook/dinov3-vit7b16-pretrain-lvd1689m) Eva vision encoder with 2D RoPE, SwiGLU MLP, and register tokens
- **SwinV2** (timm/swinv2_large_window12_192.ms_in22k) hierarchical vision transformer with shifted-window cosine attention and continuous position bias
- **SAM3.1** (facebook/sam3.1) image/video segmentation with ViT backbone, fusion encoder, detection decoder, and segmentation head
- **Whisper** (large-v3) encoder-decoder speech-to-text with batched inference and paged cross-attention KV cache
- **CosyVoice3** (Fun-CosyVoice3-0.5B-2512) text-to-speech with flow matching DiT + HiFi-GAN vocoder
- **OpenFold3** (OpenFold/OpenFold3) protein structure prediction with MSA module, PairFormer, diffusion sampling, and atom attention
- **Tensor parallelism** (TP) with custom IPC-based all-reduce for multi-GPU inference
- **Paged KV cache** with Triton store kernels (LLM models)
- **CUDA graph capture** for decode steps (LLM models)
- **Flash Attention** for both prefill/paged decode (LLMs) and non-causal bidirectional attention (diffusion)
- Greedy and top-p sampling (LLMs); flow-match Euler discrete scheduling (diffusion)
- **Layered operator architecture** (L1 single-kernel ops through L4 full models) with clean separation of concerns
- **Benchmarking suite** for evaluating custom CUDA/Triton/PyTorch kernels at 4 abstraction levels
- **vllm-omni comparison benchmark** for FLUX diffusion and HunyuanVideo-1.5 video diffusion
- **vllm-omni comparison benchmark** for CosyVoice3 TTS (SEED-TTS-Eval dataset)
- **diffusers comparison benchmark** for SDXL diffusion
- **Detection benchmark** for YOLOv10 and RTDetrV2
- **timm comparison benchmark** for SigLIP-2, DINOv3, and SwinV2 vision encoders (ImageNet-1K validation)
- **facebook/sam3 comparison benchmark** for SAM3.1 segmentation
- **OpenFold/OpenFold3 comparison benchmark** for protein structure prediction

## Project Structure

```
├── tasks/                      # Benchmarkable operators & models
│   ├── baseline/               # Reference implementations (the code to beat)
│   │   ├── L1/                 # Single-kernel ops
|   |   |   ├── ...
│   │   │   └── csrc/           # CUDA/C++ kernel sources (JIT-compiled)
│   │   │       └── ...
│   │   ├── L2/                 # Multi-op blocks
│   │   │   └── ...
│   │   ├── L3/                 # Decoder layers
│   │   │   └── ...
│   │   └── L4/                 # Full models
│   │       └── ...
│   └── candidate/              # Generated replacement kernels (gitignored)
│       ├── README.md           # Instructions
│       └── L1/, L2/, ...       # Organized by level, named after the operator
├── infra/                      # Non-benchmarkable infrastructure
│   ├── context.py              # Global inference context (paged KV cache coordination)
│   └── tp.py                   # TP helper utilities (_tp_size, _tp_rank)
├── bench/                      # Benchmarking suite
│   ├── kernels/                # Isolated kernel-level benchmarking
│   ├── eval/                   # Multi-model evaluation sweep
│   ├── e2e/                    # End-to-end throughput/latency benchmarks
│   └── tracking/               # MLflow experiment tracking API
├── agent/                      # LLM-powered kernel generation agent
│   ├── agent.py               # CLI agent: generates kernels via Claude, benchmarks them
│   └── llm_api.py             # Corvo LLM endpoint helper (async + sync)
├── engine.py                   # Batched inference engine with paged KV cache and TP
├── weight_loader.py            # HuggingFace safetensors weight loading with TP sharding
└── tests/                      # Test suite
    ├── test_bench.py           # Bench module tests (discovery, replacement, kernel and E2E integration)
    ├── bench_vllm.py           # Multi-scenario throughput + latency + alignment benchmark vs vLLM
    ├── bench_vllm_omni.py     # Diffusion (FLUX / HunyuanVideo) and TTS (CosyVoice3) benchmark: kb-nano vs vllm-omni
    ├── bench_diffusers.py     # SDXL diffusion benchmark: kb-nano vs diffusers + torch.compile
    ├── bench_timm.py          # Vision encoder benchmark: kb-nano vs timm (SigLIP-2, DINOv3, SwinV2)
    ├── test_sam.py            # SAM3 segmentation benchmark: kb-nano vs facebook/sam3 reference
    ├── bench_openfold3.py     # OpenFold3 protein structure benchmark: kb-nano vs OpenFold reference
    ├── utils/                  # Post-processing and visualization
    │   └── parse_vllm_bench_results.py  # Generate tables and plots from bench_vllm.py results
    └── debug/                  # Profiling and debugging scripts
```

## Quick Start

```bash
# Clone the repo
git clone git@github.com:sfc-gh-goliaro/kb-nano.git
cd kb-nano

# Install
pip install .

# Now all commands work from any directory:
kb_nano kernels --list
kb_nano eval --help
kb_nano e2e throughput --help

# Or use python -m from any directory:
python -m kb_nano kernels --list
python -m kb_nano eval --help
```

### Benchmarking vs vLLM (LLMs)

```bash
# Throughput + latency + alignment benchmark vs vLLM
python tests/bench_vllm.py --model meta-llama/Llama-3.1-8B-Instruct

# With tensor parallelism
python tests/bench_vllm.py \
    --model meta-llama/Llama-3.1-70B-Instruct --tp 4

# GPT-OSS MoE (MXFP4 quantized)
python tests/bench_vllm.py \
    --model openai/gpt-oss-120b --tp 2

# Kimi-Linear-48B-A3B (hybrid KDA + MLA MoE)
python tests/bench_vllm.py \
    --model moonshotai/Kimi-Linear-48B-A3B-Instruct --tp 2

# Qwen3-Next-80B-A3B (hybrid GDN + full-attention MoE)
python tests/bench_vllm.py \
    --model Qwen/Qwen3-Next-80B-A3B-Instruct --tp 2

# Whisper speech-to-text
python tests/bench_vllm.py --model openai/whisper-large-v3

# Bench module tests (unit tests + GPU integration)
python tests/test_bench.py

# Bench module unit tests only (no GPU required)
python tests/test_bench.py --unit-only
```

### Benchmarking vs vllm-omni (Diffusion)

```bash
# FLUX.1-dev: throughput + latency + correctness benchmark vs vllm-omni
python tests/bench_vllm_omni.py --model black-forest-labs/FLUX.1-dev

# HunyuanVideo-1.5: text-to-video benchmark vs vllm-omni
python tests/bench_vllm_omni.py --model hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v

# kb-nano only (skip vllm-omni comparison)
python tests/bench_vllm_omni.py --skip-vllm-omni

# Override batch size for FLUX scenarios
python tests/bench_vllm_omni.py --batch-size 2

# Skip throughput or latency phases
python tests/bench_vllm_omni.py --skip-throughput
python tests/bench_vllm_omni.py --skip-latency

# Save results to a specific directory
python tests/bench_vllm_omni.py --output-dir tests/results/B200/FLUX.1-dev
```

### Benchmarking vs vllm-omni (TTS / CosyVoice3)

```bash
# CosyVoice3: throughput + latency benchmark vs vllm-omni (SEED-TTS-Eval dataset)
python tests/bench_vllm_omni.py --model FunAudioLLM/Fun-CosyVoice3-0.5B-2512

# kb-nano only
python tests/bench_vllm_omni.py --model FunAudioLLM/Fun-CosyVoice3-0.5B-2512 --skip-vllm-omni
```

### Benchmarking vs diffusers (SDXL Diffusion)

```bash
# SDXL: throughput + latency + correctness benchmark vs diffusers + torch.compile
python tests/bench_diffusers.py --model stabilityai/stable-diffusion-xl-base-1.0

# Eager mode (for correctness measurement without compile divergence)
python tests/bench_diffusers.py --enforce-eager

# kb-nano only (skip diffusers comparison)
python tests/bench_diffusers.py --skip-diffusers

# Save results to a specific directory
python tests/bench_diffusers.py --output-dir tests/results/B200/stable-diffusion-xl-base-1.0
```

### Benchmarking vs timm (Vision Encoders)

Supported models:

| Short Name | Model ID |
|------------|----------|
| SigLIP-2 | `google/siglip2-so400m-patch16-naflex` |
| DINOv3 | `facebook/dinov3-vit7b16-pretrain-lvd1689m` |
| SwinV2 | `timm/swinv2_large_window12_192.ms_in22k` |

```bash
# Run a single model benchmark (throughput + latency + correctness vs timm)
python tests/bench_timm.py --model google/siglip2-so400m-patch16-naflex
python tests/bench_timm.py --model facebook/dinov3-vit7b16-pretrain-lvd1689m
python tests/bench_timm.py --model timm/swinv2_large_window12_192.ms_in22k

# Run all three in parallel on separate GPUs
CUDA_VISIBLE_DEVICES=0 python tests/bench_timm.py --model timm/swinv2_large_window12_192.ms_in22k &
CUDA_VISIBLE_DEVICES=1 python tests/bench_timm.py --model facebook/dinov3-vit7b16-pretrain-lvd1689m &
CUDA_VISIBLE_DEVICES=2 python tests/bench_timm.py --model google/siglip2-so400m-patch16-naflex &
wait

# kb-nano only (skip timm comparison)
python tests/bench_timm.py --model google/siglip2-so400m-patch16-naflex --skip-timm

# Use a different dataset (default: ILSVRC/imagenet-1k validation)
python tests/bench_timm.py --model google/siglip2-so400m-patch16-naflex --dataset food101 --dataset-split validation

# Override number of images or resolution
python tests/bench_timm.py --model google/siglip2-so400m-patch16-naflex --num-images 1000 --resolution 512
```

### Benchmarking vs facebook/sam3 (Segmentation)

```bash
# SAM3.1: throughput + latency + correctness benchmark vs facebook/sam3 reference
# Uses 100 images (SACo-Gold + SACo-VEval), 20 latency iterations, all modalities
python tests/test_sam.py

# Custom number of images
python tests/test_sam.py --num-items 200

# Image-only or video-only modality
python tests/test_sam.py --modality image
python tests/test_sam.py --modality video

# kb-nano only (skip reference comparison)
python tests/test_sam.py --skip-reference

# Skip latency phase
python tests/test_sam.py --skip-latency
```

### Benchmarking detection models

```bash
# YOLOv10: throughput + latency + alignment benchmark vs official THU-MIG/yolov10
python tests/bench_detection.py --model jameslahm/yolov10n --use-fp16

# RTDetrV2: throughput + latency + alignment benchmark vs transformers
python tests/bench_detection.py --model PekingU/rtdetr_v2_r101vd --use-fp16

# kb-nano only
python tests/bench_detection.py --model jameslahm/yolov10n --use-fp16 --skip-reference

# Save results to a specific directory
python tests/bench_detection.py --model jameslahm/yolov10n --use-fp16 --output-dir tests/results/H200/yolov10n
python tests/bench_detection.py --model PekingU/rtdetr_v2_r101vd --use-fp16 --output-dir tests/results/H200/rtdetr_v2_r101vd
```

### Benchmarking vs OpenFold (Protein Structure Prediction)

```bash
# OpenFold3: throughput + latency + correctness benchmark vs OpenFold reference
# Uses real pre-computed MSA data from OpenProteinSet (downloaded from S3 on first run)
python tests/bench_openfold3.py

# kb-nano only (skip reference comparison)
python tests/bench_openfold3.py --skip-reference

# Skip latency phase
python tests/bench_openfold3.py --skip-latency

# Override number of queries per scenario
python tests/bench_openfold3.py --num-seqs 100

# Enable torch.compile for potential speedup
python tests/bench_openfold3.py --torch-compile
```

The protein structure benchmark measures:
- **Throughput**: tokens/sec across short (100-200 residues), medium (200-400), long (400-700), and extra-long (700+) protein chains
- **Latency**: per-structure prediction latency with median stats
- **Correctness**: percentage of queries meeting all correctness requirements (atom position cosine similarity >= 0.95, RMSD < 0.5A, pLDDT correlation >= 0.99, PAE cosine similarity >= 0.95)

The diffusion benchmark measures:
- **Throughput**: images/sec (FLUX) or videos/sec (HunyuanVideo) at various resolutions
- **Latency**: per-image/video latency with P50 percentile stats
- **Correctness**: per-batch latent cosine similarity (FLUX) or per-prompt decoded-frame PSNR and cosine similarity (HunyuanVideo)

## Benchmarking

The benchmark suite lets you evaluate custom kernel implementations at 4 abstraction levels:

- **L1** — Single-kernel ops (e.g. `rms_norm`, `linear`, `rotary_emb`)
- **L2** — Multi-op blocks (e.g. `attention`, `llama_mlp`, `mixtral_moe`)
- **L3** — Decoder layers (e.g. `llama_decoder`, `mixtral_decoder`)
- **L4** — Full models (e.g. `llama`, `mixtral`)

```bash
# List all targets and which models use them
kb_nano kernels --map

# List targets at a specific level
kb_nano kernels --list --level 1

# Benchmark a candidate kernel from tasks/candidate/
kb_nano kernels --target rms_norm

# Results are auto-saved with timestamps:
#   bench/results/kernels_20260313_143022.json
# Override with --output-json:
kb_nano kernels --target rms_norm --output-json my_results.json
```

The model-to-operator mapping is derived automatically from the import graph — no manual annotations needed.

## LLM Kernel Agent

The agent uses Claude Opus 4.6 to automatically generate replacement kernels for any operator level, then benchmarks them against the baseline using the bench suite.

```bash
# Generate all L1 kernels for Llama, benchmark them
kb_nano agent \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --level 1

# Force CUDA-only kernels (no Triton/PyTorch builtins)
kb_nano agent \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --level 1 --cuda-only

# Mixtral with tensor parallelism
kb_nano agent \
    --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --level 2 --tp 4

# Custom retry limit and LLM model
kb_nano agent \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --level 1 --max-retries 3 --llm-model claude-opus-4-6
```

The agent discovers operators, generates replacements, validates they compile, patches them all into the model simultaneously, and reports token match rate and speedup. Failed kernels are retried up to `--max-retries` times with error feedback to the LLM.

## Experiment Tracking

kb_nano automatically logs all benchmark runs and agent runs to [MLflow](https://mlflow.org) for experiment tracking and kernel lineage.

```bash
# List recent tracked runs
kb_nano history

# Show history for a specific operator
kb_nano history --op rms_norm

# Show best-ever speedup for each operator
kb_nano history --best

# Launch the MLflow web UI
kb_nano mlflow-ui
# Open http://localhost:5000
```

Sample output from `kb_nano history`:

```
======================================================================
  RECENT TRACKED RUNS
======================================================================
  TIMESTAMP          RUN NAME                            KEY METRICS
  ──────────────────────────────────────────────────────────────────
  2026-03-16 16:44   e2e_throughput_Llama-3.1-8B         tokens_per_second=15000.0
  2026-03-16 16:44   kernels_rms_norm                    avg_speedup=1.64x  total_passed=2  total_failed=0
  2026-03-16 16:44   agent_L1_Llama-3.1-8B               e2e_speedup=1.15x  e2e_token_match_rate=97.0%
======================================================================
```

Sample output from `kb_nano history --best`:

```
======================================================================
  BEST SPEEDUP PER OPERATOR (from kernel benchmarks)
======================================================================
  OPERATOR                  BEST SPEEDUP DATE               RUN ID
  ──────────────────────────────────────────────────────────────────
  rms_norm                         1.64x 2026-03-16 16:44   a1b2c3d4
  rotary_emb                       1.22x 2026-03-15 11:30   d3e4f5a6
  silu_and_mul                     1.45x 2026-03-14 09:55   b7c8d9e0
======================================================================
```

### Tracking API

Any kernel optimization agent or script can use the tracking API:

```python
from kb_nano.bench.tracking import tracker

with tracker.start_run("my-optimization-v3", params={
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "level": 1,
    "strategy": "triton-fused",
}):
    # Log generated kernel source code as artifact
    tracker.log_kernel("rms_norm", level=1, code=kernel_source)

    # Log kernel benchmark results (accepts KernelBenchResult directly)
    tracker.log_kernel_bench(bench_result)

    # Log any custom metrics
    tracker.log_metrics({"compile_time_s": 12.3})
```

Tracking data is stored locally in `mlruns/` (gitignored). If `mlflow` is not installed, all tracking calls are silently skipped.

### MLflow Web UI

```bash
kb_nano mlflow-ui
# Open http://localhost:5000
```

The UI shows all tracked runs under the `kb_nano` experiment. You can:

- **Sort and filter** runs by metrics (`metrics.e2e_speedup > 1.0`) or parameters (`params.level = "1"`, `tags.tier = "agent"`)
- **Inspect** any run to see its parameters, per-operator metrics, and generated kernel source code (stored as artifacts under `kernels/`)
- **Compare** multiple runs side-by-side to see how metrics and kernel code evolved across iterations
- **Download** kernel artifacts to recover high-performing kernels from previous runs

Each run is tagged with a `tier` (`agent`, `kernel`, `eval`, or `e2e`) indicating which command produced it. See the [user guide](docs/user_guide.md#mlflow-web-ui) for detailed UI walkthrough.

## Dependencies

- Python 3.10+
- PyTorch 2.x with CUDA
- Triton
- Flash Attention (`flash-attn`)
- SGLang kernel library (`sgl-kernel`) — fused RMSNorm, SiLU, RoPE, MoE ops
- FlashInfer (`flashinfer-python`) — TRTLLM-gen attention kernels on Blackwell+
- Hugging Face (`transformers`, `huggingface_hub`, `safetensors`)
- aiohttp (for the LLM kernel agent)
- MLflow (experiment tracking — gracefully skipped if not installed)
- timm (pytorch-image-models — only needed for vision encoder comparison tests)
- vLLM (only needed for running comparison tests)
- matplotlib (only needed for benchmark plotting)
- ultralytics (YOLOv10 baseline benchmarking)

### Optional detection benchmark dependencies

```bash
pip install ultralytics==8.4.35
git clone https://github.com/THU-MIG/yolov10.git third_party/yolov10
```

### GPU architecture and library compatibility

The pre-built wheels for `torch`, `sgl-kernel`, `flash-attn`, and `vllm` must all
agree on the same PyTorch ABI and CUDA variant, otherwise you get `undefined symbol`
errors at import time.

**Tested stack (Blackwell B200, CUDA 13.0):**

```bash
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
    --index-url https://download.pytorch.org/whl/cu130
pip install "sgl-kernel==0.3.21+cu130" \
    --find-links https://docs.sglang.io/whl/cu130/sgl-kernel/
pip install vllm==0.16.0
pip install flash-attn
```

The engine auto-selects the attention backend based on GPU compute capability:
**sm_100+** (Blackwell) uses TRTLLM-gen decode kernels via FlashInfer;
**sm_90 and below** (Hopper, Ampere) uses Flash Attention.

## Performance

Run `tests/bench_vllm.py` to reproduce. Workload uses random token IDs with `ignore_eos=True`, both engines with full optimizations (`enforce_eager=False`).

**Hardware: 4x NVIDIA H200 (NVLink)**

### Llama 3.1

| Model | TP | Seqs | Input/Output | vLLM (tok/s) | Ours (tok/s) | Ratio |
|-------|---:|-----:|:------------:|-------------:|-------------:|------:|
| Llama-3.1-8B  | 1 |  256 | 1024/1024 |  9,623 |  9,150 | 0.95x |
| Llama-3.1-8B  | 4 |  128 |  512/256  | 16,468 | 16,605 | 1.01x |
| Llama-3.1-8B  | 4 |  256 | 1024/1024 | 21,052 | 20,492 | 0.97x |

### Mixtral-8x7B

| Model | TP | Seqs | Input/Output | vLLM (tok/s) | Ours (tok/s) | Ratio |
|-------|---:|-----:|:------------:|-------------:|-------------:|------:|
| Mixtral-8x7B | 4 |   64 |  512/256  |  3,397 |  4,401 | 1.30x |
| Mixtral-8x7B | 4 |  128 |  512/256  |  4,720 |  7,230 | 1.53x |
| Mixtral-8x7B | 4 |  256 | 1024/1024 |  9,769 |  9,852 | 1.01x |

### GPT-OSS (MXFP4)

Run `tests/bench_vllm.py --model openai/gpt-oss-120b --tp 2` to reproduce. 1000 sequences per scenario, `temperature=0`. Expert weights remain in packed MXFP4 uint8 format — no dequantization at any point. Uses vLLM's Triton `matmul_ogs` kernel for fused MoE, FlashAttention 3 for attention sinks and sliding window, and YaRN RoPE.

Throughput:

| Model | TP | Scenario | Input/Output | vLLM (tok/s) | Ours (tok/s) | Ratio | Avg Match Tokens |
|-------|---:|----------|:------------:|-------------:|-------------:|------:|-----------------:|
| gpt-oss-20b  | 2 | prefill-heavy | 1024/512  |  11,554 |   9,675 | 0.84x | 460.4/512 |
| gpt-oss-20b  | 2 | balanced      |  512/512  |  12,974 |  10,907 | 0.84x | 437.1/512 |
| gpt-oss-20b  | 2 | decode-heavy  |  512/1024 |  13,175 |  12,035 | 0.91x | 925.6/1024 |
| gpt-oss-120b | 2 | prefill-heavy | 1024/512  |  10,761 |  13,265 | **1.23x** | 112.3/512 |
| gpt-oss-120b | 2 | balanced      |  512/512  |  16,300 |  17,274 | **1.06x** | 124.1/512 |
| gpt-oss-120b | 2 | decode-heavy  |  512/1024 |  15,236 |  19,186 | **1.26x** | 213.0/1024 |

Latency (128 output tokens, 5 iterations):

| Model | TP | Scenario | Batch Size | vLLM median | Ours median | vLLM ms/tok | Ours ms/tok | Ratio |
|-------|---:|----------|---:|------------:|------------:|------------:|------------:|------:|
| gpt-oss-20b  | 2 | single-request | 1  | 0.446s | 0.554s | 3.48 | 4.32 | 0.80x |
| gpt-oss-20b  | 2 | fixed-batch-32 | 32 | 0.665s | 0.743s | 0.16 | 0.18 | 0.90x |
| gpt-oss-120b | 2 | single-request | 1  | 0.661s | 0.794s | 5.17 | 6.21 | 0.83x |
| gpt-oss-120b | 2 | fixed-batch-32 | 32 | 3.044s | 1.393s | 0.74 | 0.34 | **2.19x** |

The 120B model shows strong throughput advantages (1.06-1.26x) and a 2.19x batched latency speedup. The lower token match rate for the 120B model is expected: with 128 experts and top-4 routing, small numerical differences in router logits cause different expert selections, which cascade into divergent outputs. The 20B model (32 experts) shows higher match rates (85-90%).

**Hardware: 4x NVIDIA B200 (NVLink)**

Run `tests/bench_vllm.py` to reproduce. Three scenarios per model, 1000 sequences each, `temperature=0`.

### Llama 3.1

| Model | TP | Scenario | Input/Output | vLLM (tok/s) | Ours (tok/s) | Ratio |
|-------|---:|----------|:------------:|-------------:|-------------:|------:|
| Llama-3.1-8B  | 1 | prefill-heavy | 1024/512  | 14,448 | 15,039 | **1.04x** |
| Llama-3.1-8B  | 1 | balanced      |  512/512  | 25,075 | 23,734 | 0.95x |
| Llama-3.1-8B  | 1 | decode-heavy  |  512/1024 | 22,830 | 22,928 | **1.00x** |
| Llama-3.1-8B  | 4 | prefill-heavy | 1024/512  | 38,843 | 36,873 | 0.95x |
| Llama-3.1-8B  | 4 | balanced      |  512/512  | 47,568 | 52,756 | **1.11x** |
| Llama-3.1-8B  | 4 | decode-heavy  |  512/1024 | 52,219 | 56,589 | **1.08x** |
| Llama-3.1-70B | 4 | prefill-heavy | 1024/512  |  7,939 |  7,200 | 0.91x |
| Llama-3.1-70B | 4 | balanced      |  512/512  | 11,622 | 10,542 | 0.91x |
| Llama-3.1-70B | 4 | decode-heavy  |  512/1024 | 13,847 | 12,251 | 0.88x |

### Mixtral-8x7B

| Model | TP | Scenario | Input/Output | vLLM (tok/s) | Ours (tok/s) | Ratio |
|-------|---:|----------|:------------:|-------------:|-------------:|------:|
| Mixtral-8x7B | 4 | prefill-heavy | 1024/512  | 15,060 | 23,064 | **1.53x** |
| Mixtral-8x7B | 4 | balanced      |  512/512  | 20,530 | 33,443 | **1.63x** |
| Mixtral-8x7B | 4 | decode-heavy  |  512/1024 | 24,728 | 37,761 | **1.53x** |

### Qwen2-VL / Qwen3-VL (VLM)

Throughput (1000 sequences per scenario, `temperature=0`, `max_model_len=16896`):

| Model | TP | Scenario | Output | vLLM (tok/s) | Ours (tok/s) | Ratio | Avg Match Tokens |
|-------|---:|----------|-------:|-------------:|-------------:|------:|-----------------:|
| Qwen2-VL-7B  | 1 | text-only | 1024 | 35,385 | 35,510 | **1.00x** | 933.3/1024 |
| Qwen2-VL-7B  | 1 | image     |  512 | 15,266 | 14,851 | **0.97x** | 294.5/512 |
| Qwen2-VL-7B  | 1 | video     |  512 |  3,240 |  2,414 | 0.75x | 390.5/512 |
| Qwen3-VL-8B  | 1 | text-only | 1024 | 20,887 | 20,279 | 0.97x | 888.0/1024 |
| Qwen3-VL-8B  | 1 | image     |  512 | 15,590 | 13,156 | 0.84x | 114.4/512 |
| Qwen3-VL-8B  | 1 | video     |  512 |  3,710 |  8,804 | **2.37x** | 103.0/512 |

Latency (batch size 1, 128 output tokens, 5 iterations):

| Model | TP | Scenario | vLLM median | Ours median | Ratio |
|-------|---:|----------|------------:|------------:|------:|
| Qwen2-VL-7B  | 1 | single-image | 0.486s | 0.518s | 0.94x |
| Qwen2-VL-7B  | 1 | single-video | 0.539s | 0.682s | 0.79x |
| Qwen3-VL-8B  | 1 | single-image | 0.559s | 0.578s | 0.97x |
| Qwen3-VL-8B  | 1 | single-video | 0.613s | 0.593s | **1.03x** |

### Whisper (large-v3)

Throughput (full LibriSpeech `test.clean` — 2,620 utterances, 324 minutes of audio, `temperature=0`, `enforce_eager=True`, 448 max output tokens):

| Model | TP | Seqs | Audio | vLLM (tok/s) | Ours (tok/s) | Ratio | Avg Match Tokens |
|-------|---:|-----:|------:|-------------:|-------------:|------:|-----------------:|
| whisper-large-v3 | 1 | 2,620 | 324 min | 8,525 | 8,084 | 0.95x | 388.7/444 |

Latency (448 output tokens, 5 iterations):

| Model | TP | Scenario | Batch Size | vLLM median | Ours median | vLLM ms/tok | Ours ms/tok | Ratio |
|-------|---:|----------|---:|------------:|------------:|------------:|------------:|------:|
| whisper-large-v3 | 1 | single-utterance | 1 | 5.702s | 5.812s | 12.73 | 12.97 | 0.98x |
| whisper-large-v3 | 1 | fixed-batch-32 | 32 | 6.214s | 5.978s | 0.43 | 0.42 | **1.04x** |

### Qwen3-VL FP8 (W8A8 block-quantized)

FP8 support uses block-scaled FP8 GEMM via DeepGEMM. Vision encoder and lm_head remain in BF16; only LLM decoder layers use FP8.

Throughput (1000 sequences per scenario, `temperature=0`, `max_model_len=16896`):

| Model | TP | Scenario | Output | vLLM (tok/s) | Ours (tok/s) | Ratio | Avg Match Tokens |
|-------|---:|----------|-------:|-------------:|-------------:|------:|-----------------:|
| Qwen3-VL-8B-FP8 | 1 | text-only | 1024 |  8,138 | 10,350 | **1.27x** | 795.6/1024 |
| Qwen3-VL-8B-FP8 | 1 | image     |  512 |  7,761 |  7,863 | **1.01x** |  16.2/512 |
| Qwen3-VL-8B-FP8 | 1 | video     |  512 |  2,195 |  6,071 | **2.77x** |  57.7/512 |
| Qwen3-VL-30B-FP8 (MoE) | 1 | image |  512 | 10,598 |  7,612 | 0.72x |  60.8/512 |
| Qwen3-VL-235B-FP8 (MoE) | 4 | text-only | 1024 | 8,262 | 8,213 | 0.99x | 418.5/1024 |
| Qwen3-VL-235B-FP8 (MoE) | 4 | image     |  512 | 6,578 | 4,809 | 0.73x |  94.8/512 |
| Qwen3-VL-235B-FP8 (MoE) | 4 | video     |  512 | 1,581 | 4,675 | **2.96x** |  77.1/512 |

Latency (batch size 1, 128 output tokens, 5 iterations):

| Model | TP | Scenario | vLLM median | Ours median | Ratio |
|-------|---:|----------|------------:|------------:|------:|
| Qwen3-VL-8B-FP8 | 1 | single-image | 0.560s | 0.728s | 0.77x |
| Qwen3-VL-8B-FP8 | 1 | single-video | 0.724s | 0.731s | 0.99x |
| Qwen3-VL-30B-FP8 (MoE) | 1 | single-image | 0.682s | 0.777s | 0.88x |
| Qwen3-VL-235B-FP8 (MoE) | 4 | single-image | 1.433s | 1.808s | 0.79x |
| Qwen3-VL-235B-FP8 (MoE) | 4 | single-video | 1.882s | 1.768s | **1.06x** |

FP8 activation quantization uses a custom Triton kernel for single-launch per-token-group UE8M0 quantization. Pre-allocated shared prefill buffers eliminate dynamic allocation during FP8 prefill, and DeepGEMM is JIT-warmed for both decode and prefill batch sizes. The remaining throughput gap vs vLLM is primarily from vLLM's `torch.compile` + Inductor fusion passes (RMSNorm+quant, SiLU+quant).

### FLUX.1-dev (Diffusion)

Run `tests/bench_vllm_omni.py` to reproduce. Prompts drawn from the full nateraw/parti-prompts (P2) dataset (1632 prompts), shuffled deterministically. Reference engine: vllm-omni 0.16.0. Both engines run in eager mode.

**Hardware: NVIDIA H200**

Throughput (images/sec, eager mode, 28 steps):

| Scenario | Batch | Images | vllm-omni | Ours | Ratio |
|----------|------:|-------:|----------:|-----:|------:|
| 1024x1024 | 4 | 40 | 0.22 | 0.22 | **1.01x** |
| 512x512   | 8 | 80 | 0.72 | 0.72 | **1.00x** |

Latency (single image, 28 steps, median of 5 runs, eager mode):

| Resolution | vllm-omni | Ours | Ratio |
|------------|----------:|-----:|------:|
| 1024x1024  | 4.734s | 4.712s | **1.00x** |
| 512x512    | 1.653s | 1.616s | **1.02x** |

Correctness (eager mode, decoded image space, per-batch cosine similarity):

| Scenario | Mean CosSim | Min CosSim |
|----------|------------:|-----------:|
| 1024x1024, 28 steps | 0.995 | 0.990 |
| 512x512, 28 steps   | 0.994 | 0.986 |

Both engines run in eager mode to ensure numerically comparable outputs. The remaining cosine divergence (~0.5%) is from the CLIP text encoder: kb-nano uses a custom implementation while vllm-omni uses HuggingFace's `CLIPTextModel`, which produces slightly different pooled embeddings (`cos≈0.9999` per token) that compound over 28 denoising steps. On H200 (Hopper), both engines use `flash_attn_func` for attention.

### HunyuanVideo-1.5 (Video Diffusion)

Run `tests/bench_vllm_omni.py --model hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v` to reproduce. Prompts drawn from the Movie Gen Video Bench dataset (1003 prompts), shuffled deterministically. Reference engine: vllm-omni (Omni sync API). Both engines run in eager mode with 30 inference steps, guidance_scale=6.0.

**Hardware: NVIDIA H200**

Throughput (videos/sec, eager mode, 30 steps):

| Scenario | Resolution | Frames | Videos | vllm-omni | Ours | Ratio |
|----------|:----------:|-------:|-------:|----------:|-----:|------:|
| 480p-short  | 480x832 | 25 | 16 | 0.0466 | 0.0454 | 0.97x |
| 480p-medium | 480x832 | 49 |  8 | 0.0221 | 0.0215 | 0.97x |

Latency (single video, 30 steps, median of 5 runs, eager mode):

| Scenario | Resolution | Frames | vllm-omni | Ours | Ratio |
|----------|:----------:|-------:|----------:|-----:|------:|
| single-480p-short  | 480x832 | 25 | 21.504s | 20.462s | **1.05x** |
| single-480p-medium | 480x832 | 49 | 45.165s | 43.262s | **1.04x** |

Correctness (eager mode, decoded video frames, per-prompt cosine similarity):

| Scenario | Prompts | Mean CosSim | Min CosSim | Mean PSNR | Result |
|----------|--------:|------------:|-----------:|----------:|-------:|
| 480p-short  | 16 | 0.925 | 0.833 | 12.96 dB | WARN |
| 480p-medium |  8 | 0.923 | 0.862 | 12.92 dB | WARN |

Correctness is measured in decoded pixel space (both engines produce PIL video frames which are compared as uint8 numpy arrays). The pixel-level cosine similarity of ~0.92 is expected for two independent bf16 implementations: numerical differences in the 30-step denoising loop are amplified by the VAE decoder. For reference, latent-space comparison between kb-nano and HF diffusers yields CosSim=0.986, confirming the transformer backbone is correctly implemented. The pixel-space divergence is dominated by VAE decode amplification and different text encoder implementations (kb-nano uses a custom Qwen2.5-VL paged-attention encoder vs vllm-omni's HuggingFace-based encoder).

### CosyVoice3 (TTS)

Run `tests/bench_vllm_omni.py --model FunAudioLLM/Fun-CosyVoice3-0.5B-2512` to reproduce. Dataset: SEED-TTS-Eval (110 utterances across short/medium/long scenarios). Both engines run in float32 with greedy decoding. Reference engine: vllm-omni 0.16.0.

**Hardware: NVIDIA H200**

Throughput (utterances/sec):

| Scenario | Utts | vllm-omni (utt/s) | vllm-omni RTF | Ours (utt/s) | Ours RTF | Speedup |
|----------|-----:|-----------:|----------:|-----------:|----------:|--------:|
| tts-short   |  27 | 0.46 | 0.436 | 0.99 | 0.201 | **2.15x** |
| tts-medium  | 100 | 0.39 | 0.391 | 0.87 | 0.165 | **2.22x** |
| tts-long    |  50 | 0.38 | 0.389 | 0.78 | 0.164 | **2.03x** |

Latency (single utterance, median of 5 runs):

| Scenario | vllm-omni | Ours | Speedup |
|----------|----------:|-----:|--------:|
| single-utterance | 2.072s | 1.138s | **1.82x** |

Correctness (mel spectrogram cosine similarity, kb-nano vs vllm-omni):

| Scenario | Utts | Median | Mean | P10 | Min |
|----------|-----:|-------:|-----:|----:|----:|
| tts-short   |  27 | 0.892 | 0.847 | 0.692 | 0.171 |
| tts-medium  | 100 | 0.904 | 0.874 | 0.754 | 0.513 |
| tts-long    |  50 | 0.870 | 0.860 | 0.766 | 0.467 |
| **Overall** | 177 | **0.894** | 0.866 | — | — |

Code2Wav equivalence (same speech tokens, same CFM seed): mel cosine similarity **0.999**.

The Code2Wav stage (flow-matching DiT + HiFi-GAN vocoder) produces near-identical output when given the same tokens. The remaining E2E divergence comes from the Talker LLM stage, where kb-nano uses SDPA while vllm-omni uses PagedAttention with TritonAttention kernels — these attention backends accumulate small numerical differences that can cause token sequences to diverge, especially on longer utterances.

### SDXL (Diffusion)

Run `tests/bench_diffusers.py` to reproduce. Prompts drawn from the full nateraw/parti-prompts (P2) dataset (1632 prompts), shuffled deterministically. Reference engine: diffusers 0.31 with `torch.compile(mode="max-autotune")`, eager mode for correctness.

**Hardware: NVIDIA B200**

Throughput (images/sec, eager mode):

| Scenario | Batch | Images | diffusers | Ours | Ratio |
|----------|------:|-------:|----------:|-----:|------:|
| 1024x1024, 50 steps | 1 | 5 | 0.51 | 0.57 | **1.12x** |
| 512x512, 50 steps   | 4 | 20 | 2.03 | 2.62 | **1.29x** |
| 1024x1024, 28 steps | 1 | 5 | 0.92 | 1.01 | **1.10x** |

Latency (single image, 50 steps, median of 5 runs, eager mode):

| Resolution | diffusers | Ours | Ratio |
|------------|----------:|-----:|------:|
| 1024x1024  | 1.914s | 1.753s | **1.09x** |
| 512x512    | 1.906s | 1.370s | **1.39x** |

Correctness (eager mode, latent space, per-batch cosine similarity):

| Scenario | Mean CosSim | Min CosSim |
|----------|------------:|-----------:|
| 1024x1024, 50 steps | 0.990 | 0.967 |
| 1024x1024, 28 steps | 0.987 | 0.977 |
| 512x512, 50 steps   | 0.969 | 0.952 |

Correctness is measured in eager mode with bf16 precision. Both engines use identical model weights (diffusers checkpoint) and the same EulerDiscreteScheduler. The remaining cosine divergence is expected from bf16 accumulation differences across 28-50 denoising steps with CFG guidance_scale=5.0.

### SigLIP-2 (Vision Encoder)

Run `tests/bench_timm.py --model google/siglip2-so400m-patch16-naflex` to reproduce. 10,000 unique images from ImageNet-1K validation (ILSVRC/imagenet-1k), loaded via streaming and preprocessed with Inception normalization (mean/std=0.5). Each image processed exactly once. Reference engine: timm (pytorch-image-models).

**Hardware: NVIDIA H200**

Throughput (images/sec, bf16):

| Scenario | Resolution | Images | timm (img/s) | Ours (img/s) | Ratio |
|----------|:----------:|-------:|-------------:|-------------:|------:|
| default-res | 384x384 | 10,000 | 927 | 832 | 0.90x |
| high-res    | 512x512 | 10,000 | 512 | 494 | 0.96x |

Latency (median of 30 iterations, bf16):

| Scenario | Batch | timm p50 | Ours p50 | Ratio |
|----------|------:|---------:|---------:|------:|
| single-image | 1 | 5.2 ms | 4.7 ms | **1.10x** |
| batch-8 | 8 | 9.9 ms | 9.8 ms | **1.01x** |

Correctness (per-batch embedding cosine similarity):

| Scenario | Batches | Mean CosSim | Min CosSim | Result |
|----------|--------:|------------:|-----------:|-------:|
| default-res | 16 | 1.000 | 1.000 | PASS |
| high-res | 32 | 1.000 | 1.000 | PASS |

### DINOv3 (Vision Encoder)

Run `tests/bench_timm.py --model facebook/dinov3-vit7b16-pretrain-lvd1689m` to reproduce. 1,500 unique images from ImageNet-1K validation, loaded via streaming and preprocessed with ImageNet normalization (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]). Each image processed exactly once. Reference engine: timm (pytorch-image-models).

**Hardware: NVIDIA H200**

Throughput (images/sec, bf16):

| Scenario | Resolution | Images | timm (img/s) | Ours (img/s) | Ratio |
|----------|:----------:|-------:|-------------:|-------------:|------:|
| default-res | 256x256 | 1,500 | 133 | 131 | 0.98x |
| high-res    | 512x512 | 1,500 |  33 |  33 | 0.99x |

Latency (median of 30 iterations, bf16):

| Scenario | Batch | timm p50 | Ours p50 | Ratio |
|----------|------:|---------:|---------:|------:|
| single-image | 1 | 16.4 ms | 14.0 ms | **1.17x** |
| batch-8 | 8 | 64.5 ms | 64.3 ms | **1.00x** |

Correctness (per-batch embedding cosine similarity):

| Scenario | Batches | Mean CosSim | Min CosSim | Result |
|----------|--------:|------------:|-----------:|-------:|
| default-res | 16 | 1.000 | 1.000 | PASS |
| high-res | 32 | 1.000 | 1.000 | PASS |

DINOv3 produces bit-identical embeddings to timm (perfect 1.000 cosine similarity, zero MSE).

### SwinV2 (Vision Encoder)

Run `tests/bench_timm.py --model timm/swinv2_large_window12_192.ms_in22k` to reproduce. 5,000 unique images from ImageNet-1K validation (ILSVRC/imagenet-1k), loaded via streaming and preprocessed with ImageNet normalization (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]). Each image processed exactly once. Reference engine: timm (pytorch-image-models).

**Hardware: NVIDIA H200**

Throughput (images/sec, bf16):

| Scenario | Resolution | Images | timm (img/s) | Ours (img/s) | Ratio |
|----------|:----------:|-------:|-------------:|-------------:|------:|
| default-res | 192x192 | 5,000 | 1,287 | 1,763 | **1.37x** |
| high-res    | 512x512 | 5,000 |   250 |   244 | 0.97x |

Latency (median of 30 iterations, bf16):

| Scenario | Batch | timm p50 | Ours p50 | Ratio |
|----------|------:|---------:|---------:|------:|
| single-image | 1 | 12.7 ms | 14.9 ms | 0.86x |
| batch-8 | 8 | 13.3 ms | 14.9 ms | 0.89x |

Correctness (per-batch embedding cosine similarity):

| Scenario | Batches | Mean CosSim | Min CosSim | Result |
|----------|--------:|------------:|-----------:|-------:|
| default-res | 16 | 1.000 | 1.000 | PASS |
| high-res | 32 | 1.000 | 1.000 | PASS |

SwinV2 produces bit-identical embeddings to timm (perfect 1.000 cosine similarity, zero MSE).

### SAM3.1 (Segmentation)

Run `tests/test_sam.py` to reproduce. 100 images from SACo-Gold and SACo-VEval SmartGlasses (image + video frames), preprocessed to 1008x1008. Reference engine: facebook/sam3 (`build_sam3_image_model`). Both engines load shared pretrained weights; correctness is checked on every image used for throughput. kb-nano implements the full reference architecture including boxRPB, presence_token, DotProductScoring, and geometry encoder.

**Hardware: NVIDIA H200**

Throughput (100 images, full pipeline):

| Scenario | Images | Reference (img/s) | Ours (img/s) | Speedup |
|----------|-------:|-----------:|-----------:|--------:|
| full-pipeline | 100 | 6.48 | 6.79 | **1.05x** |

Latency (median of 20 iterations):

| Scenario | BS | Resolution | Reference p50 | Ours p50 | Speedup |
|----------|---:|-----------:|--------------:|---------:|--------:|
| single-image-1008 | 1 | 1008 | 0.083s | 0.076s | **1.09x** |
| batch-4-image-1008 | 4 | 1008 | 0.082s | 0.076s | **1.09x** |
| single-video-frame-1008 | 1 | 1008 | 0.082s | 0.076s | **1.09x** |

Correctness (100 images, per-element cosine similarity):

| Output | Avg CosSim | Min CosSim | Result |
|--------|----------:|-----------:|-------:|
| Bounding Boxes | 0.980 | 0.959 | PASS |
| Segmentation Masks | 0.949 | 0.871 | PASS |
| Classification Logits | 0.975 | 0.924 | PASS |

The remaining numerical divergence is expected from SDPA vs Flash Attention numerics and bf16/fp32 precision differences accumulated through the deep pipeline (backbone + encoder + decoder + pixel decoder + mask predictor). All metrics pass their thresholds (boxes/logits: mean >= 0.95, min >= 0.90; masks: mean >= 0.90, min >= 0.85).

### YOLOv10 (Detection)

Run `tests/bench_detection.py --model jameslahm/yolov10n` to reproduce. Throughput uses synthetic 640x640 image tensors; the reference baseline is the official `THU-MIG/yolov10` implementation loaded from `third_party/yolov10`.

**Hardware: NVIDIA H200**

| Model | Images | Batch | Reference (img/s) | Ours (img/s) | Ratio |
|-------|-------:|------:|------------------:|-------------:|------:|
| jameslahm/yolov10n | 8 | 2 | 323.66 | 343.75 | **1.06x** |

Latency (median of 3 iterations):

| Batch Size | Reference p50 | Ours p50 | Ratio |
|-----------:|--------------:|---------:|------:|
| 1 | 0.00591s | 0.00592s | 1.00x |
| 4 | 0.01059s | 0.00778s | **1.36x** |

Alignment:

| Output | Metric | Value |
|--------|--------|------:|
| Boxes | Cosine | 1.000 |
| Boxes | MAE | 0.0 |
| Scores | Cosine | 1.000 |
| Scores | MAE | 0.0 |
| Labels | Match Rate | 1.000 |

### RTDetrV2 (Detection)

Run `tests/bench_detection.py --model PekingU/rtdetr_v2_r101vd` to reproduce. Throughput uses synthetic 640x640 image tensors; the reference baseline is `transformers.RTDetrV2ForObjectDetection`.

**Hardware: NVIDIA H200**

| Model | Images | Batch | Reference (img/s) | Ours (img/s) | Ratio |
|-------|-------:|------:|------------------:|-------------:|------:|
| PekingU/rtdetr_v2_r101vd | 8 | 2 | 80.49 | 87.01 | **1.08x** |

Latency (median of 3 iterations):

| Batch Size | Reference p50 | Ours p50 | Ratio |
|-----------:|--------------:|---------:|------:|
| 1 | 0.02305s | 0.02227s | **1.04x** |
| 4 | 0.02374s | 0.02431s | 0.98x |

Alignment:

| Output | Metric | Value |
|--------|--------|------:|
| Boxes | Cosine | 1.000 |
| Boxes | MAE | 0.0 |
| Scores | Cosine | 1.000 |
| Scores | MAE | 0.0 |
| Labels | Match Rate | 1.000 |
### OpenFold3 (Protein Structure Prediction)

Run `tests/bench_openfold3.py` to reproduce. Uses real pre-computed MSA data from [OpenProteinSet](https://registry.opendata.aws/openfold/) (74 protein chains across 4 length buckets). Model: OpenFold/OpenFold3 (368M params, `of3-p2-155k.pt` checkpoint). Both engines use identical code with 10 diffusion rollout steps and 1 recycle. Correctness is verified deterministically (identical seeds, `torch.use_deterministic_algorithms`).

**Hardware: NVIDIA H200**

Throughput (tokens/sec, bf16):

| Scenario | Queries | Residue Range | Reference (tok/s) | Ours (tok/s) | Speedup |
|----------|--------:|--------------:|-------------------:|--------------:|--------:|
| short       |  50 | 100-200 | 146 | 150 | **1.03x** |
| medium      |  20 | 200-400 | 137 | 141 | **1.03x** |
| long        |  10 | 400-700 | 159 | 165 | **1.04x** |
| extra-long  |   5 | 700+    | 150 | 153 | **1.02x** |

Latency (single structure, median of 3 runs):

| Scenario | Tokens | Reference | Ours | Speedup |
|----------|-------:|----------:|-----:|--------:|
| single-short       | 321 | 1.361s | 1.333s | **1.02x** |
| single-medium      | 123 | 0.970s | 0.924s | **1.05x** |
| single-long        | 219 | 0.968s | 0.923s | **1.05x** |
| single-extra-long  | 552 | 2.958s | 2.934s | **1.01x** |

Correctness (deterministic, all queries across all scenarios):

| Metric | Target | Result |
|--------|--------|--------|
| Atom position cosine similarity | mean >= 0.95, min >= 0.90 | **100% pass** |
| Atom position RMSD (Kabsch)     | < 0.5 A                   | **100% pass** |
| pLDDT score correlation         | Pearson r >= 0.99          | **100% pass** |
| PAE matrix cosine similarity    | mean >= 0.95               | **100% pass** |

Both engines run identical model code with the same pretrained weights and deterministic seeding, achieving bitwise reproducibility (100% correctness pass rate). The throughput advantage comes from vectorized block conversion utilities (`torch.gather`-based) in the atom attention module, replacing Python for-loops in the reference implementation.

### Key optimizations

- **Fused RMSNorm**: Uses `sgl_kernel`'s fused residual-add + RMSNorm CUDA kernel, eliminating multiple kernel launches per norm call
- **Fused SiLU-and-Mul**: Single-kernel SiLU activation with gate multiplication
- **Inplace RoPE**: Applies rotary position embeddings in-place with cos/sin cache
- **Fused MoE routing**: `topk_softmax` fuses softmax + top-k into a single kernel for expert routing
- **Triton grouped GEMM**: Custom Triton kernels for MoE expert execution with auto-tuned configs
- **Custom IPC all-reduce**: Replaces NCCL for intra-node TP with direct GPU P2P memory access
- **SHM spin-wait signaling**: Workers spin on a shared-memory sequence counter instead of `multiprocessing.Event`, reducing per-step signaling latency from ~0.48ms to ~0.004ms
- **LM head in CUDA graph**: The LM head projection and local argmax are captured inside the CUDA graph alongside the transformer body
- **Greedy fast path**: For greedy decoding with TP, uses local argmax + small all-gather instead of gathering full logits across ranks
- **Triton FP8 activation quantization**: Single-kernel per-token-group UE8M0 quantization for FP8 inference, with pre-allocated decode buffers for CUDA graph capture and shared prefill buffers to eliminate dynamic allocation
