# kb-nano

A standalone, high-performance LLM inference engine supporting **Llama 3.1** and **Mixtral-8x7B** with tensor parallelism. No vLLM dependency at runtime — just PyTorch, Triton, and Flash Attention.

## Features

- **Llama 3.1** (8B, 70B) with frequency-scaled RoPE
- **Mixtral-8x7B** with fused Triton MoE grouped-GEMM kernels
- **Tensor parallelism** (TP) via NCCL for multi-GPU inference
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
│   └── tp.py                   # TP-aware linear layers, embeddings, LM head
├── bench/                      # Benchmarking suite
│   ├── discovery.py            # Auto-discovers targets via import graph analysis
│   ├── replacement.py          # Class monkey-patching for swapping implementations
│   ├── runner.py               # Benchmark orchestration (baseline vs user)
│   ├── evaluator.py            # KL divergence + speedup metrics
│   └── __main__.py             # CLI entry point
├── engine.py                   # Batched inference engine with paged KV cache and TP
├── weight_loader.py            # HuggingFace safetensors weight loading with TP sharding
└── test.py                     # Correctness + benchmark tests vs vLLM
```

## Quick Start

```bash
# Clone the repo
git clone git@github.com:sfc-gh-goliaro/kb-nano.git
cd kb-nano

# Single model test
python test.py --model meta-llama/Llama-3.1-8B-Instruct

# Multiple models with tensor parallelism
python test.py \
    --model meta-llama/Llama-3.1-70B-Instruct mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --tp 4 --max-tokens 50
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

Benchmarked with TP=4, 100 max tokens, greedy decoding:

| Model | Mode | vLLM | Ours | Speedup |
|-------|------|------|------|---------|
| Llama 3.1 70B | Sequential | 42.4 tok/s | 57.8 tok/s | 1.36x |
| Llama 3.1 70B | Batched | 170.2 tok/s | 251.4 tok/s | 1.48x |
| Mixtral 8x7B | Sequential | 42.7 tok/s | 129.4 tok/s | 3.03x |
| Mixtral 8x7B | Batched | 139.1 tok/s | 367.4 tok/s | 2.64x |

Llama 3.1 8B and Mixtral 8x7B produce near-identical outputs to vLLM (minor divergences in 1/4 prompts due to bfloat16 numerical differences).
