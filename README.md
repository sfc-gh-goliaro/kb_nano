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
│   │   ├── linear.py           # F.linear wrapper
│   │   ├── embedding.py        # F.embedding wrapper
│   │   ├── softmax.py          # F.softmax wrapper
│   │   ├── topk.py             # torch.topk wrapper
│   │   ├── moe_align.py        # Triton MoE token-expert alignment
│   │   └── moe_grouped_gemm.py # Triton fused MoE grouped GEMM
│   ├── L2/                     # Multi-op blocks
│   │   ├── attention.py        # GQA attention (QKV proj + RoPE + KV cache + flash attn)
│   │   ├── llama_mlp.py        # Llama SwiGLU MLP
│   │   ├── mixtral_moe.py      # Mixtral MoE routing + experts
│   │   └── fused_experts.py    # Fused expert execution (2x grouped GEMM + SiLU)
│   ├── L3/                     # Decoder layers
│   │   ├── llama_decoder.py    # Llama decoder (attention + MLP + norms)
│   │   └── mixtral_decoder.py  # Mixtral decoder (attention + MoE + norms)
│   └── L4/                     # Full models
│       ├── llama.py            # LlamaForCausalLM (config, model, LM head)
│       └── mixtral.py          # MixtralForCausalLM (config, model, LM head)
├── infra/                      # Non-benchmarkable infrastructure
│   ├── context.py              # Global inference context (paged KV cache coordination)
│   ├── tp.py                   # TP-aware linear layers, embeddings, LM head
│   ├── custom_allreduce.py     # IPC-based custom all-reduce (replaces NCCL for intra-node TP)
│   └── custom_allreduce_kernels.cu  # CUDA kernels for P2P cross-device reduction
├── bench/                      # Benchmarking suite
│   ├── discovery.py            # Auto-discovers targets via import graph analysis
│   ├── replacement.py          # Class monkey-patching for swapping implementations
│   ├── runner.py               # Benchmark orchestration (baseline vs user)
│   ├── evaluator.py            # KL divergence + speedup metrics
│   └── __main__.py             # CLI entry point
├── engine.py                   # Batched inference engine with paged KV cache and TP
├── weight_loader.py            # HuggingFace safetensors weight loading with TP sharding
└── tests/                      # Test suite
    ├── test_vllm_alignment.py  # Token-level correctness test vs vLLM (eager mode)
    ├── test_bench.py           # Bench module tests (discovery, evaluator, replacement, integration)
    ├── bench_throughput.py     # Throughput benchmark vs vLLM (full speed, nano-vllm style)
    └── profile_gap.py          # Detailed prefill/decode timing breakdown vs vLLM
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

## Dependencies

- Python 3.10+
- PyTorch 2.x with CUDA
- Triton
- Flash Attention (`flash-attn`)
- Hugging Face (`transformers`, `huggingface_hub`, `safetensors`)
- vLLM (only needed for running comparison tests)

## Performance

Run `tests/bench_throughput.py` to reproduce. Workload: 256 sequences, random 100-1024 input/output tokens, `ignore_eos=True`, both engines with full optimizations (`enforce_eager=False`).

**Hardware: 4× NVIDIA H200 (NVLink)**

| Model | TP | Seqs | vLLM (tok/s) | Ours (tok/s) | Ratio |
|-------|---:|-----:|-------------:|-------------:|------:|
| Llama-3.1-8B  | 1 |   32 |  2,938 |  2,671 | 0.91× |
| Llama-3.1-8B  | 1 |   64 |  5,324 |  4,840 | 0.91× |
| Llama-3.1-8B  | 1 |  128 |  7,465 |  6,784 | 0.91× |
| Llama-3.1-8B  | 1 |  256 |  9,787 |  9,048 | 0.92× |
| Llama-3.1-8B  | 4 |   32 |  5,274 |  4,896 | 0.93× |
| Llama-3.1-8B  | 4 |   64 |  9,886 |  9,097 | 0.92× |
| Llama-3.1-8B  | 4 |  128 | 14,682 | 14,101 | 0.96× |
| Llama-3.1-8B  | 4 |  256 | 21,067 | 20,608 | 0.98× |
| Llama-3.1-70B | 4 |  256 |  4,470 |  4,408 | 0.99× |

Use `tests/profile_gap.py` for a detailed prefill/decode timing breakdown:

```bash
python -m kb-nano.tests.profile_gap \
    --model meta-llama/Llama-3.1-8B-Instruct --tp 4 \
    --batch-sizes 32 64 128 256
```

### Key optimizations

- **Custom IPC all-reduce**: Replaces NCCL for intra-node TP with direct GPU P2P memory access, matching vLLM's approach
- **SHM spin-wait signaling**: Workers spin on a shared-memory sequence counter instead of `multiprocessing.Event`, reducing per-step signaling latency from ~0.48ms to ~0.004ms
- **LM head in CUDA graph**: The LM head projection and local argmax are captured inside the CUDA graph alongside the transformer body, eliminating extra kernel launch overhead
- **Greedy fast path**: For greedy decoding with TP, uses local argmax + small all-gather instead of gathering full logits across ranks
