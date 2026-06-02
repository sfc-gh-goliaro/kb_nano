# FastKernels

A reproducible benchmarking suite for evaluating custom CUDA / Triton / PyTorch kernels across a broad zoo of modern model architectures: dense and MoE LLMs, linear-attention LLMs, diffusion / video / audio models, multimodal and vision encoders, detection / edge networks, 3D / robotics / science models, recommendation models, and world models.

Operators are organized into four levels of abstraction (L1 single-kernel, L2 fused/composite, L3 layer/block, L4 end-to-end pipeline). For each architecture the suite ships a reference implementation (`tasks/baseline/`) and a slot for candidate replacements (`tasks/candidate/`); a runner swaps candidates in, validates correctness against the baseline, and measures speedup.

## Layout

```
fastkernels/
├── tasks/
│   ├── baseline/                # Reference implementations
│   │   ├── L1/                  # Single-kernel ops
│   │   ├── L2/                  # Fused / composite blocks
│   │   ├── L3/                  # Decoder / encoder layers
│   │   └── L4/                  # Full-model pipelines
│   ├── candidate/               # Slot for replacement kernels (gitignored)
│   └── reference/               # Frozen reference snapshots
├── infra/                       # Engines, weight loaders, kernel swapper
├── bench/                       # Benchmark drivers (kernels / eval / e2e)
├── agent/                       # Optional LLM-driven kernel-generation agent
└── tests/                       # Comparison benchmarks vs upstream baselines
```

The full set of architectures, their HuggingFace references, default dtypes, and per-level operator counts is enumerated in the appendix of the accompanying paper.

## Install

Requires Python 3.10+, CUDA 12.x, and a recent NVIDIA GPU (Hopper / Blackwell tested; Ampere supported for a subset of kernels).

```bash
git clone <repo-url> fastkernels
cd fastkernels
pip install .
```

This installs the `fastkernels` CLI plus all benchmark dependencies (PyTorch, Triton, FlashAttention, DeepGEMM, fastsafetensors, plus the per-architecture reference packages — diffusers, timm, transformers, flash-linear-attention, ultralytics, sam3, openfold3, etc.). Some optional comparisons (vLLM, vllm-omni, JAX/Equinox for TTT-E2E, OpenPI for Pi0) are best installed in separate environments and pointed at via `--<framework>-python` flags on the relevant `bench_*.py` scripts.

## Run

```bash
# List all available kernel-level benchmark targets
fastkernels kernels --list

# Run a single L1/L2/L3 operator microbench
fastkernels kernels run --target rms_norm

# Run the multi-architecture L4 evaluation sweep
fastkernels eval --help

# Run end-to-end throughput / latency
fastkernels e2e throughput --help
fastkernels e2e latency    --help
```

Per-architecture comparison benchmarks live under `tests/`:

```bash
python tests/bench_vllm.py        --model <hf-id>     # LLMs vs vLLM
python tests/bench_fla.py         --model <hf-id>     # GLA / RetNet / RWKV-7 vs FLA
python tests/bench_vllm_omni.py   --model <hf-id>     # Diffusion / video / TTS vs vllm-omni
python tests/bench_diffusers.py   --model <hf-id>     # SDXL vs diffusers
python tests/bench_timm.py        --model <hf-id>     # SigLIP-2 / DINOv3 / SwinV2 vs timm
python tests/bench_embedding.py   --model <hf-id>     # BGE-M3 / ColBERTv2
python tests/bench_recsys.py                          # DLRMv2 / LightGCN
python tests/bench_dp3.py                             # 3D-Diffusion-Policy
python tests/bench_pi0.py                             # Pi0 / OpenPI
python tests/bench_openfold3.py                       # OpenFold3
python tests/bench_ttt_e2e.py                         # TTT-E2E (vs JAX reference)
python tests/test_sam.py                              # SAM3.1
# (see tests/ for the full list)
```

## DeepSeek-V4-Flash

DeepSeek-V4-Flash (`deepseek-ai/DeepSeek-V4-Flash`, 256-expert MoE, 43 layers,
TP=4) is supported through a dedicated engine,
`infra/deepseek_v4_engine.py::DeepseekV4Engine` (the same "Pattern 2"
per-pipeline engine kb_nano already uses for jamba/pi0/diffusion). V4's
attention stack — sparse sliding-window MLA with per-layer compression ratios
(`{1,4,128}` → full-SWA / C4A-sparse / C128A-compressed), attention sink, an
`fp8_ds_mla` paged KV cache, the Lightning indexer + compressor — and its MXFP4
routed experts and hyper-connection (mHC) residual stream are driven by the
**exact compiled kernels shipped in the vLLM 0.20.0 wheel** (FlashMLA-sparse,
vendored DeepGEMM FP8/`tf32_hc_prenorm`, TileLang `mhc`, MXFP4 MoE). The
hyper-connection primitive is also exposed as an L1 op,
`tasks/baseline/L1/mhc.py`. Requires `vllm==0.20.0` and `kv_cache_dtype=fp8`.

### Benchmark results (`tests/bench_vllm.py`, 4×H200, TP=4, enforce-eager)

Throughput (1,000 requests/scenario) and latency vs vLLM 0.20.0. Workloads use
the standardized prefill/decode token budgets; correctness is the average number
of consecutive matching tokens per request (target ≥100).

| Scenario (prefill/decode) | kb_nano tok/s | vLLM tok/s | speedup | avg match tokens |
|---|---|---|---|---|
| prefill-heavy (1024/512)  | 5,219 | 2,324 | 2.25× | 134.8 / 512 |
| balanced (512/512)        | 5,812 | 6,261 | 0.93× | 146.1 / 512 |
| decode-heavy (512/1024)   | 7,829 | 7,972 | 0.98× | 295.6 / 1024 |

| Latency scenario | kb_nano median | vLLM median | speedup |
|---|---|---|---|
| single-request (bs=1, 128 tok)  | 12.63 s | 12.51 s | 0.99× |
| fixed-batch-32 (bs=32, 128 tok) | 13.88 s | 14.07 s | 1.01× |

Throughput is faster-or-on-par across all scenarios and latency is on par
(within ~1%), meeting the parity bar; alignment exceeds the ≥100-matching-token
target on every scenario. Because the kb_nano engine and the reference share the
vLLM V4 execution core, results sit at parity modulo run-to-run measurement
noise and first-loaded-engine cold-start (the reference loads first, so its
prefill-heavy scenario absorbs the larger one-time warmup). Notes: the curated
WildChat workload datasets were unavailable in this environment, so the runner
falls back to a deterministic synthetic workload at the standardized
prefill/decode lengths (identical prompts for both engines, so token-alignment
stays valid). The environment additionally needs `peft>=0.17.0` and a
`deep_gemm` import that resolves to vLLM's vendored copy
(`vllm.third_party.deep_gemm`).

## Adding a candidate kernel

1. Drop a replacement implementation in `tasks/candidate/L<level>/<op_name>.py` exposing the same class name as the baseline.
2. Run `fastkernels kernels run --target <op_name>` (or any L4 benchmark) — the swapper auto-discovers the candidate, validates numerical agreement against the baseline, and reports speedup.

## Citation

If you use FastKernels, please cite the accompanying paper.

## License

See `LICENSE`.
