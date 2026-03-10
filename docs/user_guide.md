# kb-nano User Guide

A standalone LLM inference engine with a benchmarking suite for evaluating custom CUDA/Triton/PyTorch kernels. Supports **Llama 3.1** and **Mixtral-8x7B** with tensor parallelism.

---

## Quick Start

```bash
git clone git@github.com:sfc-gh-goliaro/kb-nano.git
cd kb-nano

# Throughput + latency + alignment benchmark vs vLLM
python tests/bench_vllm.py --model meta-llama/Llama-3.1-8B-Instruct

# E2E throughput benchmark (mirrors vLLM's benchmark_throughput.py)
python -m kb_nano.bench.e2e throughput \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-name random --random-input-len 512 --random-output-len 128

# Kernel-level benchmark (swap an operator, measure correctness + speedup)
python -m kb_nano.bench.kernels --target rms_norm
```

No `pip install` step is required. Run scripts from the repository root; `kb_nano` is used as a local package. See the README for dependency installation.

---

## Benchmarking

kb-nano provides three complementary benchmarking tools.

### E2E Benchmark CLI

`python -m kb_nano.bench.e2e` mirrors vLLM's benchmarking CLI. The same flags work in both tools, making it easy to compare results:

```bash
# Throughput (mirrors `vllm bench throughput`)
python -m kb_nano.bench.e2e throughput \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-name random --random-input-len 512 --random-output-len 128 \
    --num-prompts 200 --tp 4

# Latency
python -m kb_nano.bench.e2e latency \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --input-len 128 --output-len 128

# Evaluate candidate kernels from tasks/candidate/ against baseline
python -m kb_nano.bench.e2e eval \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --input-len 512 --output-len 128 --num-prompts 100
```

### Kernel Benchmark CLI

`python -m kb_nano.bench.kernels` benchmarks a single operator replacement. It monkey-patches the target class, rebuilds the model, runs inference, and compares KL divergence, token match rate, and wall-clock speedup against the baseline.

```bash
# List all benchmarkable targets
python -m kb_nano.bench.kernels --list

# Show model-to-operator mapping
python -m kb_nano.bench.kernels --map

# Auto-discover candidate from tasks/candidate/
python -m kb_nano.bench.kernels --target rms_norm

# Specify a custom implementation explicitly
python -m kb_nano.bench.kernels \
    --target rms_norm \
    --user-impl path/to/my_kernel.py:RMSNorm \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --tp 1
```

A benchmark is **PASS** if `kl_mean < 0.1`. Speedup > 1.0 means your kernel is faster than the baseline.

### vLLM Alignment Test

`tests/bench_vllm.py` runs kb-nano and vLLM side-by-side across three workload scenarios (prefill-heavy, balanced, decode-heavy) plus latency benchmarks, comparing throughput and per-token alignment:

```bash
python tests/bench_vllm.py --model meta-llama/Llama-3.1-8B-Instruct
python tests/bench_vllm.py --model meta-llama/Llama-3.1-70B-Instruct --tp 4

# Latency only (skip throughput)
python tests/bench_vllm.py --model meta-llama/Llama-3.1-8B-Instruct --skip-throughput

# Parse and plot results
python tests/utils/parse_vllm_bench_results.py
```

Results are saved to `tests/results/<GPU>/<model>_tp<N>/results.json`. The parser auto-discovers these files and generates tables and plots in `tests/plots/<GPU>/`.

For a quick correctness check (no throughput measurement), use the `--skip-throughput` flag:

```bash
python tests/bench_vllm.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --skip-throughput --skip-latency
```

---

## LLM Kernel Agent

The agent uses Claude to generate replacement kernels, validate them, and benchmark them end-to-end:

```bash
# Generate all L1 kernels for Llama
python -m kb_nano.example \
    --model meta-llama/Llama-3.1-8B-Instruct --level 1

# CUDA-only kernels (no Triton/PyTorch builtins)
python -m kb_nano.example \
    --model meta-llama/Llama-3.1-8B-Instruct --level 1 --cuda-only

# Mixtral L2 operators with TP
python -m kb_nano.example \
    --model mistralai/Mixtral-8x7B-Instruct-v0.1 --level 2 --tp 4
```

The agent discovers operators at the specified level, generates replacements in parallel, validates compilation and numerical correctness, patches all successful kernels into the model, and reports KL divergence, token match rate, and speedup. Failed kernels are retried with error feedback.

Generated kernels are saved to `tasks/candidate/L{level}/{op_name}.py`.

---

## Design Philosophy

### The L1-L4 Hierarchy

All model operators live under `tasks/baseline/` at four abstraction levels:

- **L1** -- Single-kernel ops (e.g. `rms_norm`, `silu_and_mul`, `rotary_emb`, `linear`)
- **L2** -- Multi-op blocks (e.g. `LlamaAttention`, `LlamaMLP`, `MixtralMoE`, `QKVParallelLinear`)
- **L3** -- Decoder layers (e.g. `LlamaDecoderLayer`, `MixtralDecoderLayer`)
- **L4** -- Full models (e.g. `LlamaForCausalLM`, `MixtralForCausalLM`)

Higher levels compose lower ones via standard Python imports:

```
L4/llama.py (LlamaForCausalLM)
  L3/llama_decoder.py (LlamaDecoderLayer)
    L2/attention.py (LlamaAttention)
      L1/store_kvcache.py, L1/flash_attn_*.py
      L2/parallel_linear.py (QKVParallelLinear, RowParallelLinear)
        L1/linear.py, L1/allreduce.py
    L2/llama_mlp.py (LlamaMLP)
      L1/silu_and_mul.py
      L2/parallel_linear.py
    L1/rms_norm.py
```

When you replace an L1 operator, the change propagates upward through every level that uses it. The bench suite traces these dependencies automatically via import graph analysis.

### Interface Mirroring

Every module's `__init__` and `forward` signatures are designed to mirror the corresponding vLLM module. This makes it easy to port optimizations between the two codebases and ensures candidate kernels can be validated against a well-known reference.

Key conventions:

| kb-nano module | vLLM equivalent | `forward` signature |
|---|---|---|
| `LlamaAttention` | `LlamaAttention` | `(positions, hidden_states) -> Tensor` |
| `LlamaDecoderLayer` | `LlamaDecoderLayer` | `(positions, hidden_states, residual) -> (Tensor, Tensor)` |
| `QKVParallelLinear` | `QKVParallelLinear` | `(x) -> (output, bias)` |
| `RowParallelLinear` | `RowParallelLinear` | `(x) -> (output, bias)` |
| `MergedColumnParallelLinear` | `MergedColumnParallelLinear` | `(x) -> (output, bias)` |
| `VocabParallelEmbedding` | `VocabParallelEmbedding` | `(input_ids) -> Tensor` |
| `ParallelLMHead` | `ParallelLMHead` | `(hidden_states) -> Tensor` |
| `SiluAndMul` | `SiluAndMul` | `(x) -> Tensor` |
| `RMSNorm` | `RMSNorm` | `(x, residual=None) -> Tensor or (Tensor, Tensor)` |

All parallel linear layers return `(output, bias)` tuples to match vLLM. `LlamaAttention` stores `rotary_emb` as an `__init__` parameter (not a `forward` argument), matching vLLM's design where each `LlamaAttention` owns its rotary embedding.

---

## Adding a New Model or Kernel

### Adding a new model

Follow the L1 -> L4 pattern. For a hypothetical "NewModel":

1. **L1**: Identify which single-kernel ops are needed. Reuse existing L1 ops where possible (e.g. `RMSNorm`, `SiluAndMul`). Write new L1 modules only for ops that don't exist yet.
2. **L2**: Compose L1 ops into multi-op blocks (attention, MLP). Mirror the corresponding vLLM module's `__init__` and `forward` signatures.
3. **L3**: Write a decoder layer that combines L2 attention + L2 MLP + L1 normalization.
4. **L4**: Write the full model class (`NewModelForCausalLM`) with embedding, decoder stack, and LM head.

Each module should be a drop-in match for its vLLM counterpart. The bench suite auto-discovers new operators from the import graph.

### Writing a candidate kernel

To write a replacement kernel for benchmarking:

1. Read the baseline source to understand the `forward` signature.
2. Write an `nn.Module` with the **exact same class name** and **exact same `forward` signature**.
3. Place it at `tasks/candidate/L{level}/{op_name}.py` and run the benchmark.

Example -- replacing `RMSNorm`:

```python
# tasks/candidate/L1/rms_norm.py
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

```bash
python -m kb_nano.bench.kernels --target rms_norm
```

You can use pure PyTorch, Triton (`triton.jit`), inline CUDA (`torch.utils.cpp_extension.load_inline`), or external libraries. Avoid importing `vllm` or `sgl_kernel` in your replacement -- the point is to provide an alternative implementation.

---

## Reference

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `KB_NANO_NCCL_PORT` | `29501` | TCP port for NCCL process group initialization. |
| `KB_NANO_PROFILE` | `0` | Set to `1` to enable internal profiling timers. |
| `KB_NANO_DISABLE_CUSTOM_AR` | `0` | Set to `1` to disable custom IPC all-reduce (use NCCL). |

### Troubleshooting

**GPU memory**: The engine fills ~90% of free GPU memory with KV cache. Use a smaller model or increase TP if you run out of memory.

**NCCL port conflict**: Set `KB_NANO_NCCL_PORT=29502` if another instance is using the default port.

**Custom all-reduce hangs**: Disable with `KB_NANO_DISABLE_CUSTOM_AR=1`.

**HuggingFace auth**: Run `huggingface-cli login` for gated models like Llama.

**Bench target not found**: Run `python -m kb_nano.bench.kernels --list` to see available targets. Names match file names under `tasks/baseline/` (without `.py`).

**Class name mismatch**: Your replacement class must have the exact same name as the baseline class (e.g. `RMSNorm`, not `MyRMSNorm`).
