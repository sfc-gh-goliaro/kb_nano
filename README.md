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

## Project Structure

```
standalone_llama/
├── ops/               # Low-level operators
│   ├── attention.py   # Multi-head attention + KV cache store kernel
│   ├── context.py     # Global inference context (paged KV cache coordination)
│   ├── fused_moe.py   # Triton fused MoE grouped-GEMM kernels
│   ├── norm.py        # RMSNorm, SiluAndMul
│   ├── rotary.py      # Standard RoPE
│   └── tp.py          # TP-aware linear layers, embeddings, LM head
├── models/            # One file per model architecture
│   ├── llama31.py     # Llama 3.1 (config, frequency-scaled RoPE, MLP, decoder)
│   └── mixtral.py     # Mixtral-8x7B (config, MoE layer, decoder)
├── engine.py          # Batched inference engine with paged KV cache and TP
├── weight_loader.py   # HuggingFace safetensors weight loading with TP sharding
└── test.py            # Correctness + benchmark tests vs vLLM
```

## Quick Start

```bash
# Single model test
python -m standalone_llama.test --model meta-llama/Llama-3.1-8B-Instruct

# Multiple models with tensor parallelism
python -m standalone_llama.test \
    --model meta-llama/Llama-3.1-70B-Instruct mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --tp 4 --max-tokens 50
```

## Dependencies

- Python 3.10+
- PyTorch 2.x with CUDA
- Triton
- Flash Attention (`flash-attn`)
- Hugging Face (`transformers`, `huggingface_hub`, `safetensors`)
- vLLM (only needed for running comparison tests)

## Performance

Benchmarked with TP=4, 50 max tokens, greedy decoding:

| Model | Mode | vLLM | Ours | Speedup |
|-------|------|------|------|---------|
| Llama 3.1 70B | Sequential | 41.6 tok/s | 60.6 tok/s | 1.45x |
| Llama 3.1 70B | Batched | 156.9 tok/s | 262.9 tok/s | 1.67x |
| Mixtral 8x7B | Sequential | 44.6 tok/s | 92.2 tok/s | 2.07x |
| Mixtral 8x7B | Batched | 151.3 tok/s | 376.5 tok/s | 2.49x |

Correctness: Llama 3.1 produces bit-exact identical outputs to vLLM. Mixtral has 3/4 exact matches (1 mismatch due to bfloat16 numerical differences in MoE routing).
