# kb-nano User Guide

A standalone, high-performance LLM inference engine with a built-in benchmarking suite for evaluating custom CUDA/Triton/PyTorch kernels. Supports **Llama 3.1** (8B, 70B) and **Mixtral-8x7B** with tensor parallelism across multiple GPUs.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Running Inference](#running-inference)
4. [Architecture Overview](#architecture-overview)
5. [The Operator Hierarchy (L1–L4)](#the-operator-hierarchy-l1l4)
6. [Benchmarking Custom Kernels](#benchmarking-custom-kernels)
7. [Writing a Custom Kernel](#writing-a-custom-kernel)
8. [LLM Kernel Generation Agent](#llm-kernel-generation-agent)
9. [Testing and Validation](#testing-and-validation)
10. [Tensor Parallelism](#tensor-parallelism)
11. [Environment Variables](#environment-variables)
12. [Troubleshooting](#troubleshooting)

---

## Prerequisites

- Python 3.10+
- PyTorch 2.x with CUDA
- Triton
- Flash Attention (`flash-attn`)
- HuggingFace libraries (`transformers`, `huggingface_hub`, `safetensors`)
- `sgl_kernel` (used by baseline L1 kernels like RMSNorm and SiLU)
- `aiohttp` (only for the LLM kernel agent)
- `vLLM` (only for running comparison/correctness tests)
- NVIDIA GPU(s) with sufficient VRAM for the model being loaded

---

## Installation

```bash
git clone git@github.com:sfc-gh-goliaro/kb-nano.git
cd kb-nano
```

No `pip install` step is required. The project is used as a local package — scripts reference `kb_nano` as a Python package from the repository root.

---

## Running Inference

The `LlamaEngine` class in `engine.py` is the main entry point for text generation. It handles model loading, KV cache allocation, CUDA graph capture, batched scheduling, and sampling — all without depending on vLLM at runtime.

### Minimal example

```python
from kb_nano.engine import LlamaEngine, SamplingParams

engine = LlamaEngine(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    tensor_parallel_size=1,       # number of GPUs
    enforce_eager=False,          # False enables CUDA graphs
)

params = SamplingParams(
    temperature=0.0,   # 0.0 = greedy decoding
    top_p=1.0,
    max_tokens=128,
    seed=42,
)

prompts = ["Explain quantum computing in one paragraph."]
outputs = engine.generate(prompts, params)

for out in outputs:
    print(out.generated_text)
```

### SamplingParams options

| Parameter      | Type        | Default | Description                              |
|----------------|-------------|---------|------------------------------------------|
| `temperature`  | `float`     | `0.0`   | Sampling temperature. 0.0 = greedy.      |
| `top_p`        | `float`     | `1.0`   | Nucleus sampling threshold.              |
| `max_tokens`   | `int`       | `512`   | Maximum tokens to generate per sequence. |
| `seed`         | `int\|None` | `None`  | Random seed for reproducibility.         |
| `ignore_eos`   | `bool`      | `False` | Continue generating past EOS token.      |

### GenerationOutput fields

| Field             | Type                          | Description                                  |
|-------------------|-------------------------------|----------------------------------------------|
| `prompt`          | `str`                         | The input prompt.                            |
| `generated_text`  | `str`                         | Decoded generated text.                      |
| `token_ids`       | `list[int]`                   | Generated token IDs.                         |
| `logits_history`  | `list[torch.Tensor] \| None` | Per-step logits (when `collect_logits=True`). |

### Passing raw token IDs

You can pass pre-tokenized inputs as `list[int]` instead of strings:

```python
token_ids = [128000, 2028, 374, 264, 1296]
outputs = engine.generate([token_ids], params)
```

---

## Architecture Overview

kb-nano follows a two-tier design:

**ModelRunner** (one per GPU) handles:
- Model weight loading and TP sharding
- KV cache allocation
- Input preparation (prefill / decode / mixed batches)
- CUDA graph capture and replay
- Forward pass execution
- Inter-rank coordination via shared memory

**LlamaEngine** (rank 0 only) handles:
- Tokenization
- Sequence scheduling and block management (paged KV cache)
- Sampling (greedy, top-p)
- Coordinating prefill and decode batches

For multi-GPU setups, rank-0 serializes method calls to worker ranks via shared memory. Workers spin on a shared-memory sequence counter, which reduces per-step signaling latency from ~0.48ms (using `multiprocessing.Event`) to ~0.004ms.

---

## The Operator Hierarchy (L1–L4)

All model operators live under `tasks/baseline/` and are organized into four abstraction levels. This hierarchy determines what you can benchmark and replace:

### L1 — Single-kernel ops

Individual GPU kernels. The smallest replaceable unit.

| Operator            | File                              | Description                              |
|---------------------|-----------------------------------|------------------------------------------|
| `rms_norm`          | `baseline/L1/rms_norm.py`         | Fused RMSNorm (with optional residual add) |
| `silu_and_mul`      | `baseline/L1/silu_and_mul.py`     | SiLU activation fused with gate multiply |
| `rotary_emb`        | `baseline/L1/rotary_emb.py`       | RoPE with Llama 3.1 frequency scaling   |
| `linear`            | `baseline/L1/linear.py`           | `F.linear` wrapper                       |
| `embedding`         | `baseline/L1/embedding.py`        | `F.embedding` wrapper                    |
| `store_kvcache`     | `baseline/L1/store_kvcache.py`    | Triton KV cache store kernel             |
| `flash_attn_prefill`| `baseline/L1/flash_attn_prefill.py`| Flash Attention for prefill              |
| `flash_attn_decode` | `baseline/L1/flash_attn_decode.py`| Flash Attention for paged decode         |
| `allreduce`         | `baseline/L1/allreduce.py`        | AllReduce + custom IPC implementation    |
| `moe_align`         | `baseline/L1/moe_align.py`        | MoE token-expert alignment               |
| `moe_sum`           | `baseline/L1/moe_sum.py`          | Fused MoE sum kernel                     |
| `moe_grouped_gemm`  | `baseline/L1/moe_grouped_gemm.py` | Triton fused MoE grouped GEMM            |

### L2 — Multi-op blocks

Composed of multiple L1 operators.

| Operator            | File                               | Description                               |
|---------------------|------------------------------------|-------------------------------------------|
| `attention`         | `baseline/L2/attention.py`         | GQA with QKV projection + RoPE + KV cache + Flash Attention |
| `llama_mlp`         | `baseline/L2/llama_mlp.py`         | SwiGLU MLP (gate_up_proj + SiLU + down_proj) |
| `mixtral_moe`       | `baseline/L2/mixtral_moe.py`       | MoE routing + expert execution            |
| `fused_experts`     | `baseline/L2/fused_experts.py`     | Fused expert execution (grouped GEMM + SiLU) |
| `parallel_linear`   | `baseline/L2/parallel_linear.py`   | TP-aware linear layers                    |
| `parallel_embedding`| `baseline/L2/parallel_embedding.py`| TP-aware embedding and LM head            |

### L3 — Decoder layers

A single transformer decoder block.

| Operator            | File                                | Description                                |
|---------------------|-------------------------------------|--------------------------------------------|
| `llama_decoder`     | `baseline/L3/llama_decoder.py`      | Attention + MLP + RMSNorm residual         |
| `mixtral_decoder`   | `baseline/L3/mixtral_decoder.py`    | Attention + MoE + RMSNorm residual         |

### L4 — Full models

Complete model implementations.

| Operator   | File                       | Description             |
|------------|----------------------------|-------------------------|
| `llama`    | `baseline/L4/llama.py`     | LlamaForCausalLM        |
| `mixtral`  | `baseline/L4/mixtral.py`   | MixtralForCausalLM      |

### How the hierarchy works

Higher-level operators compose lower-level ones via standard Python imports. For example:

```
baseline/L4/llama.py (LlamaForCausalLM)
  └── baseline/L3/llama_decoder.py (LlamaDecoderLayer)
        ├── baseline/L2/attention.py (Attention)
        │     ├── baseline/L1/store_kvcache.py
        │     ├── baseline/L1/flash_attn_prefill.py
        │     ├── baseline/L1/flash_attn_decode.py
        │     └── baseline/L2/parallel_linear.py
        │           ├── baseline/L1/linear.py
        │           └── baseline/L1/allreduce.py
        ├── baseline/L2/llama_mlp.py (LlamaMLP)
        │     ├── baseline/L1/silu_and_mul.py
        │     └── baseline/L2/parallel_linear.py
        └── baseline/L1/rms_norm.py
```

When you replace an L1 operator, the change propagates upward through every level that uses it. The bench suite leverages this — it can trace which models use which operators automatically by analyzing the import graph.

---

## Benchmarking Custom Kernels

The benchmark suite (`kb_nano.bench`) lets you swap any operator with your own implementation and measure correctness + performance against the baseline.

### Listing available targets

```bash
# List all benchmarkable targets
python -m kb_nano.bench --list

# Filter by level
python -m kb_nano.bench --list --level 1

# See the full model-to-operator mapping
python -m kb_nano.bench --map
```

The `--map` command shows two views:
1. **Operators by model** — which operators each model (llama31, mixtral) uses
2. **Models by operator** — which models each operator belongs to

### Running a benchmark

```bash
# Auto-discover candidate from tasks/candidate/
python -m kb_nano.bench \
    --target rms_norm \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --max-tokens 50 \
    --tp 1

# Or specify a custom implementation explicitly
python -m kb_nano.bench \
    --target rms_norm \
    --user-impl path/to/my_kernel.py:MyRMSNorm \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --max-tokens 50 \
    --tp 1
```

This will:
1. Build the model with baseline operators and run inference (collecting logits)
2. Monkey-patch the target class with your implementation everywhere it's referenced
3. Rebuild the model with your class and run the same inference
4. Compare outputs using **KL divergence**, **token match rate**, and **wall-clock speedup**

### CLI options

| Flag            | Default | Description                                  |
|-----------------|---------|----------------------------------------------|
| `--target`      | —       | Operator name (e.g. `rms_norm`, `attention`) |
| `--user-impl`   | —       | `path/to/file.py:ClassName` or `module:ClassName` |
| `--model`       | auto    | HuggingFace model name(s). Defaults to all applicable models. |
| `--max-tokens`  | `50`    | Tokens to generate per prompt.               |
| `--tp`          | `1`     | Tensor parallelism degree.                   |
| `--seed`        | `42`    | Random seed.                                 |
| `--num-warmup`  | `1`     | Warmup iterations before timing.             |
| `--num-runs`    | `3`     | Timed runs to average.                       |

### Programmatic API

```python
from kb_nano.bench import benchmark, list_targets

# List targets
for t in list_targets(level=1):
    print(f"L{t.level} {t.name} — used by {t.models}")

# Run a benchmark
from my_kernels import MyRMSNorm

results = benchmark(
    target_name="rms_norm",
    user_impl=MyRMSNorm,
    models=["meta-llama/Llama-3.1-8B-Instruct"],
    max_tokens=50,
)

for r in results:
    print(r.report())
```

### Understanding the results

The `BenchResult` contains:

| Metric              | Good value  | Meaning                                     |
|----------------------|-------------|---------------------------------------------|
| `kl_mean`           | < 0.001     | Mean KL divergence between logit distributions. 0 = identical. |
| `kl_max`            | < 0.01      | Worst-case KL divergence across all steps.  |
| `token_match_rate`  | > 99%       | Fraction of tokens that match exactly.      |
| `speedup`           | > 1.0       | Ratio of baseline_time / user_time. >1 = faster. |

A benchmark is considered **PASS** if `kl_mean < 0.1`. The exit code is non-zero if any result exceeds this threshold.

---

## Writing a Custom Kernel

To create a replacement kernel for benchmarking, you need to write an `nn.Module` subclass that matches the target operator's interface.

### Step 1: Identify the target

```bash
python -m kb_nano.bench --list --level 1
```

Pick a target, then read its source file to understand the `forward` signature. For example, `rms_norm`:

```python
# Baseline: tasks/baseline/L1/rms_norm.py
class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x, residual=None):
        # returns normalized x, or (x, residual) if residual is provided
        ...
```

### Step 2: Write your replacement

Your replacement must:
- Be an `nn.Module` subclass
- Have the **exact same class name** as the target
- Have the **exact same `forward` signature** (parameter names, types, defaults)
- The `__init__` signature should match if you override it
- Produce numerically equivalent (or very close) outputs

```python
# my_rms_norm.py
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x, residual=None):
        if residual is not None:
            x = x + residual
            residual = x

        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        result = (self.weight * x).to(x.dtype)

        if residual is not None:
            return result, residual
        return result
```

### Step 3: Benchmark it

Place your file at `tasks/candidate/L1/rms_norm.py`, then:

```bash
python -m kb_nano.bench --target rms_norm
```

Or specify the path explicitly:

```bash
python -m kb_nano.bench \
    --target rms_norm \
    --user-impl my_rms_norm.py:RMSNorm
```

### Implementation options

You can use any of these approaches in your kernel:

- **Pure PyTorch** — easiest to write, good baseline
- **Triton kernels** — write GPU kernels in Python with `triton.jit`
- **Inline CUDA** — use `torch.utils.cpp_extension.load_inline()` to JIT-compile CUDA C++ code
- **External libraries** — cuBLAS, cutlass, flash_attn, etc.

Avoid importing `vllm`, `sglang`, or `sgl_kernel` in your replacement (the baseline already uses these — the point is to provide an alternative implementation).

### How patching works

The bench suite uses monkey-patching to swap classes. When you target `rms_norm`, it:

1. Finds `RMSNorm` in `tasks/baseline/L1/rms_norm.py`
2. Scans all loaded `kb_nano.*` modules for references to that class
3. Replaces every reference with your class
4. Rebuilds the full model — your class is now used wherever the original was
5. After benchmarking, restores all original references

This means replacing an L1 operator automatically affects all L2/L3/L4 modules that use it.

---

## LLM Kernel Generation Agent

The `kb_nano.example` module contains an autonomous agent that uses Claude to generate replacement kernels, validate them, and benchmark them — all in one command.

### Basic usage

```bash
# Generate all L1 kernels for Llama, benchmark them
python -m kb_nano.example \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --level 1
```

### What the agent does

1. **Discovers** all operators at the specified level for the given model
2. **Generates** replacement kernels in parallel (one LLM call per operator)
3. **Validates** each kernel: imports it, instantiates the class, checks for compile errors
4. **Runs unit tests** with random data to verify numerical correctness
5. **Patches all successful kernels** into the model simultaneously
6. **Runs an end-to-end benchmark**, comparing baseline vs. patched model
7. **Retries** failed kernels with error feedback to the LLM

### CLI options

| Flag              | Default            | Description                                     |
|-------------------|--------------------|------------------------------------------------|
| `--model`         | —                  | HuggingFace model name (required).              |
| `--level`         | —                  | Operator level: 1, 2, 3, or 4 (required).       |
| `--cuda-only`     | `False`            | Force raw CUDA kernels (no Triton/PyTorch).     |
| `--max-retries`   | `5`                | Max attempts per kernel on compile/runtime failure. |
| `--tp`            | `1`                | Tensor parallelism degree.                      |
| `--max-tokens`    | `50`               | Tokens per prompt during benchmarking.          |
| `--llm-model`     | `claude-opus-4-6`  | LLM model for code generation.                  |
| `--skip-unit-tests` | `False`          | Skip per-operator unit tests.                   |

### Examples

```bash
# CUDA-only L1 kernels for Llama
python -m kb_nano.example \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --level 1 --cuda-only

# L2 operators for Mixtral with TP=4
python -m kb_nano.example \
    --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --level 2 --tp 4

# Custom retry limit and LLM
python -m kb_nano.example \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --level 1 --max-retries 3 --llm-model claude-opus-4-6
```

Generated kernels are saved to `tasks/candidate/L{level}/{op_name}.py`. Previous candidates are archived to `tasks/candidate/prev-attempts/<timestamp>/`. CUDA JIT compilation artifacts are cached in `example/_cuda_build_cache/`.

---

## Testing and Validation

### Correctness test (vs. vLLM)

Compares kb-nano's outputs token-by-token against vLLM in eager mode:

```bash
# Single model
python tests/test_vllm_alignment.py --model meta-llama/Llama-3.1-8B-Instruct

# Multiple models with TP
python tests/test_vllm_alignment.py \
    --model meta-llama/Llama-3.1-70B-Instruct \
              mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --tp 4 --max-tokens 50
```

Both engines run in separate subprocesses with `enforce_eager=True` for deterministic comparison. The test reports per-prompt token matches and highlights divergence points.

### Bench module tests

Tests the benchmarking infrastructure itself (discovery, evaluator, replacement patching):

```bash
# Full test suite (includes GPU integration)
python tests/test_bench.py

# Unit tests only (no GPU required)
python tests/test_bench.py --unit-only
```

The integration test runs two benchmarks in subprocesses:
- **Identity test**: patches RMSNorm with itself, expects KL ~ 0
- **Broken test**: patches with an all-zeros implementation, expects KL >> 0

### Throughput benchmark

Measures token throughput against vLLM and sglang at full speed (CUDA graphs enabled):

```bash
# Default workload
python tests/bench_throughput.py --model meta-llama/Llama-3.1-8B-Instruct

# Custom workload with TP
python tests/bench_throughput.py \
    --model meta-llama/Llama-3.1-70B-Instruct --tp 4 \
    --num-seqs 256 --max-input-len 1024 --max-output-len 1024

# Skip vLLM (only run kb-nano)
python tests/bench_throughput.py \
    --model meta-llama/Llama-3.1-8B-Instruct --skip-vllm --skip-sglang

# Cache vLLM results for reuse
python tests/bench_throughput.py \
    --model meta-llama/Llama-3.1-8B-Instruct --save-vllm vllm_results.json
python tests/bench_throughput.py \
    --model meta-llama/Llama-3.1-8B-Instruct --load-vllm vllm_results.json
```

---

## Tensor Parallelism

kb-nano supports multi-GPU inference via tensor parallelism. The TP degree is specified when creating the engine.

### How it works

- **Rank 0** runs the `LlamaEngine` scheduler and coordinates worker ranks
- **Ranks 1..N** run `ModelRunner` instances that block in a spin-wait loop, executing commands from rank 0 via shared memory
- Weights are automatically sharded during loading (column-parallel for QKV/MLP projections, row-parallel for output projections)
- Communication uses a **custom IPC-based all-reduce** for low-latency intra-node reduction (falls back to NCCL if disabled)

### Usage

```python
engine = LlamaEngine(
    model_name="meta-llama/Llama-3.1-70B-Instruct",
    tensor_parallel_size=4,
)
```

Or from the CLI:

```bash
python -m kb_nano.bench --target rms_norm --user-impl my_kernel.py:RMSNorm --tp 4
```

Worker processes are spawned automatically and cleaned up on exit.

---

## Environment Variables

| Variable                      | Default | Description                                    |
|-------------------------------|---------|------------------------------------------------|
| `KB_NANO_NCCL_PORT`          | `29501` | TCP port for NCCL process group initialization.|
| `KB_NANO_PROFILE`            | `0`     | Set to `1` to enable internal profiling timers.|
| `KB_NANO_DISABLE_CUSTOM_AR`  | `0`     | Set to `1` to disable custom IPC all-reduce and use NCCL instead. |

---

## Troubleshooting

### "Not enough GPU memory for KV cache"

The engine allocates KV cache to fill ~90% of available GPU memory after model loading. If you see this error:
- Use a smaller model (8B instead of 70B)
- Increase TP degree to spread memory across more GPUs
- Close other GPU-consuming processes

### NCCL port conflict

If you're running multiple instances, set different ports:
```bash
KB_NANO_NCCL_PORT=29502 python -m kb_nano.bench ...
```

### Custom all-reduce hangs

If multi-GPU inference hangs during all-reduce, disable the custom implementation:
```bash
KB_NANO_DISABLE_CUSTOM_AR=1 python -m kb_nano.bench --tp 4 ...
```

### "No .safetensors files found"

The engine downloads model weights from HuggingFace Hub automatically. Ensure:
- You have network access
- You're authenticated with `huggingface-cli login` for gated models (e.g., Llama)
- The model name is correct (e.g., `meta-llama/Llama-3.1-8B-Instruct`)

### Bench target not found

If `--target xyz` fails with "Unknown bench target", run `python -m kb_nano.bench --list` to see all available targets. Target names match the Python file names under `tasks/baseline/` (without `.py`).

### User implementation class name mismatch

Your replacement class must have the **exact same name** as the baseline class. If the baseline file defines `class RMSNorm`, your file must also define `class RMSNorm`. The bench suite matches classes by name.

### Forward signature mismatch

Runtime errors during benchmarking often mean your `forward` method has a different signature than the baseline. Check the baseline source carefully — all parameter names, order, and defaults must match exactly.
