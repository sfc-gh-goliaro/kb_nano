# kb-nano

A standalone, high-performance LLM inference engine supporting **Llama 3.1** and **Mixtral-8x7B** with tensor parallelism. No vLLM dependency at runtime — just PyTorch, Triton, and Flash Attention.

## Features

- **Llama 3.1** (8B, 70B) with frequency-scaled RoPE
- **Mixtral-8x7B** with fused Triton MoE grouped-GEMM kernels
- **Tensor parallelism** (TP) with custom IPC-based all-reduce for multi-GPU inference
- **Paged KV cache** with Triton store kernels
- **CUDA graph capture** for decode steps
- **Flash Attention** for both prefill and paged decode
- Greedy and top-p sampling
- **Layered operator architecture** (L1 single-kernel ops through L4 full models) with clean separation of concerns
- **Benchmarking suite** for evaluating custom CUDA/Triton/PyTorch kernels at 4 abstraction levels

## Project Structure

```
├── tasks/                      # Benchmarkable operators & models, organized by level
│   ├── L1/                     # Single-kernel ops
│   │   ├── rms_norm.py         # Fused RMSNorm
│   │   ├── silu_and_mul.py     # SiLU activation with gate
│   │   ├── rotary_emb.py       # RoPE (standard + Llama 3.1 frequency scaling)
│   │   ├── store_kvcache.py    # Triton KV cache store kernel
│   │   ├── flash_attn_prefill.py
│   │   ├── flash_attn_decode.py
│   │   ├── allreduce.py        # AllReduce op + custom IPC all-reduce (NCCL fallback)
│   │   ├── linear.py           # F.linear wrapper
│   │   ├── embedding.py        # F.embedding wrapper
│   │   ├── moe_align.py        # MoE token-expert alignment
│   │   ├── moe_sum.py          # Fused MoE sum kernel
│   │   ├── moe_grouped_gemm.py # Triton fused MoE grouped GEMM
│   │   └── csrc/               # CUDA/C++ kernel sources (JIT-compiled)
│   │       └── custom_allreduce_kernels.cu  # P2P cross-device reduction
│   ├── L2/                     # Multi-op blocks
│   │   ├── attention.py        # GQA attention (QKV proj + RoPE + KV cache + flash attn)
│   │   ├── llama_mlp.py        # Llama SwiGLU MLP
│   │   ├── mixtral_moe.py      # Mixtral MoE routing + experts
│   │   ├── fused_experts.py    # Fused expert execution (2x grouped GEMM + SiLU)
│   │   ├── parallel_linear.py  # TP-aware linear layers (Column, Merged, QKV, Row)
│   │   └── parallel_embedding.py # TP-aware embedding and LM head
│   ├── L3/                     # Decoder layers
│   │   ├── llama_decoder.py    # Llama decoder (attention + MLP + norms)
│   │   └── mixtral_decoder.py  # Mixtral decoder (attention + MoE + norms)
│   └── L4/                     # Full models
│       ├── llama.py            # LlamaForCausalLM (config, model, LM head)
│       └── mixtral.py          # MixtralForCausalLM (config, model, LM head)
├── infra/                      # Non-benchmarkable infrastructure
│   ├── context.py              # Global inference context (paged KV cache coordination)
│   └── tp.py                   # TP helper utilities (_tp_size, _tp_rank)
├── bench/                      # Benchmarking suite
│   ├── discovery.py            # Auto-discovers targets via import graph analysis
│   ├── replacement.py          # Class monkey-patching for swapping implementations
│   ├── runner.py               # Benchmark orchestration (baseline vs user)
│   ├── evaluator.py            # KL divergence + speedup metrics
│   └── __main__.py             # CLI entry point
├── example/                    # LLM-powered kernel generation agent
│   ├── agent.py               # CLI agent: generates kernels via Claude, benchmarks them
│   ├── llm_api.py             # Corvo LLM endpoint helper (async + sync)
│   └── _generated_kernels/    # Output directory for LLM-generated kernels (gitignored)
├── engine.py                   # Batched inference engine with paged KV cache and TP
├── weight_loader.py            # HuggingFace safetensors weight loading with TP sharding
└── tests/                      # Test suite
    ├── test_vllm_alignment.py  # Token-level correctness test vs vLLM (eager mode)
    ├── test_bench.py           # Bench module tests (discovery, evaluator, replacement, integration)
    ├── bench_throughput.py     # Throughput benchmark vs vLLM (full speed)
    └── debug/                  # Profiling and debugging scripts
        ├── profile_decode.py
        ├── profile_decode_detail.py
        ├── profile_gap.py
        ├── profile_llama_tp1.py
        ├── profile_llama_tp1_detail.py
        ├── profile_mixtral_detail.py
        ├── tune_moe_gemm.py
        └── bench_moe.py
```

## Quick Start

```bash
# Clone the repo
git clone git@github.com:sfc-gh-goliaro/kb-nano.git
cd kb-nano

# Single model test (vs vLLM)
python tests/test_vllm_alignment.py --model meta-llama/Llama-3.1-8B-Instruct

# Multiple models with tensor parallelism
python tests/test_vllm_alignment.py \
    --model meta-llama/Llama-3.1-70B-Instruct mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --tp 4 --max-tokens 50

# Bench module tests (unit tests + GPU integration)
python tests/test_bench.py

# Bench module unit tests only (no GPU required)
python tests/test_bench.py --unit-only

# Throughput benchmark vs vLLM (both engines at full speed)
python tests/bench_throughput.py --model meta-llama/Llama-3.1-8B-Instruct

# With tensor parallelism and custom workload
python tests/bench_throughput.py \
    --model meta-llama/Llama-3.1-70B-Instruct --tp 4 \
    --num-seqs 256 --max-input-len 1024 --max-output-len 1024
```

## Benchmarking

The benchmark suite lets you evaluate custom kernel implementations at 4 abstraction levels:

- **L1** — Single-kernel ops (e.g. `rms_norm`, `linear`, `rotary_emb`)
- **L2** — Multi-op blocks (e.g. `attention`, `llama_mlp`, `mixtral_moe`)
- **L3** — Decoder layers (e.g. `llama_decoder`, `mixtral_decoder`)
- **L4** — Full models (e.g. `llama`, `mixtral`)

```bash
# List all targets and which models use them
python -m kb_nano.bench --map

# List targets at a specific level
python -m kb_nano.bench --list --level 1

# Benchmark a custom kernel
python -m kb_nano.bench \
    --target rms_norm \
    --user-impl path/to/my_kernel.py:MyRMSNorm
```

The model-to-operator mapping is derived automatically from the import graph — no manual annotations needed.

## LLM Kernel Agent

The agent uses Claude Opus 4.6 to automatically generate replacement kernels for any operator level, then benchmarks them against the baseline using the bench suite.

```bash
# Generate all L1 kernels for Llama, benchmark them
python -m kb_nano.example \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --level 1

# Force CUDA-only kernels (no Triton/PyTorch builtins)
python -m kb_nano.example \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --level 1 --cuda-only

# Mixtral with tensor parallelism
python -m kb_nano.example \
    --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --level 2 --tp 4

# Custom retry limit and LLM model
python -m kb_nano.example \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --level 1 --max-retries 3 --llm-model claude-opus-4-6
```

The agent discovers operators, generates replacements, validates they compile, patches them all into the model simultaneously, and reports KL divergence, token match rate, and speedup. Failed kernels are retried up to `--max-retries` times with error feedback to the LLM.

## Dependencies

- Python 3.10+
- PyTorch 2.x with CUDA
- Triton
- Flash Attention (`flash-attn`)
- Hugging Face (`transformers`, `huggingface_hub`, `safetensors`)
- aiohttp (for the LLM kernel agent)
- vLLM (only needed for running comparison tests)

## Performance

Run `tests/bench_throughput.py` to reproduce. Workload uses random token IDs with `ignore_eos=True`, both engines with full optimizations (`enforce_eager=False`).

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
