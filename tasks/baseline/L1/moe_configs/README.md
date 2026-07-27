# Tuned Triton fused-MoE configs

Per-device kernel configs for the Triton grouped GEMM in
`tasks/baseline/L1/moe_grouped_gemm.py`, discovered by `_get_moe_configs()`.

## Why these are vendored

These files are **static tuning data**, not generated at runtime. Upstream vLLM
produces them offline with `benchmarks/kernels/benchmark_moe.py` (a sweep over
`BLOCK_SIZE_{M,N,K}` / `GROUP_SIZE_M` / `num_warps` / `num_stages`), commits the
winners, and ships them as package data — vLLM's `get_moe_configs()` only ever
*reads* them.

fastkernels previously looked for them in a `vllm_repo/` source checkout beside
this repo. In any environment without that checkout the lookup silently missed
and we fell back to a size heuristic, while vLLM used a hand-tuned config for
the identical `(E, N, dtype, block_shape)` — an unintended handicap in every
head-to-head MoE benchmark, not a deliberate choice.

Copying the JSON keeps the tuned numbers without adding a runtime dependency on
vLLM being importable. Regenerate or extend with the upstream tuner; the file
naming must stay byte-compatible with `_get_config_file_name()`.

## Naming

    E=<num_experts>,N=<intermediate_per_tp>,device_name=<dev>[,dtype=<dt>][,block_shape=[bn,bk]].json

`N` is the per-TP-shard intermediate size, so one model yields different files
per TP degree (Qwen3-VL-235B-A22B: `moe_intermediate_size` 1536 → N=768 at TP2,
N=384 at TP4).

## Provenance

Copied verbatim from vLLM 0.18.0
(`vllm/model_executor/layers/fused_moe/configs/`), Apache-2.0; each file is
sha256-identical to upstream.

Covers every block-FP8 MoE row currently benchmarked, at the TP degree it is
benchmarked at:

| Row | E | N | file |
|---|---|---|---|
| Qwen3-VL-235B-A22B-FP8 @ TP4 | 128 | 384 | `E=128,N=384,…H200/B200` |
| Qwen3-VL-235B-A22B-FP8 @ TP2 | 128 | 768 | `E=128,N=768,…H200/B200` |
| DeepSeek-V3.2 @ TP8 | 256 | 256 | `E=256,N=256,…H200/B200` |

Because a tuned file exists for each of these, they never reach the heuristic
fallback in `_get_default_config`. Upstream ships no `E=128,N=192/1536` or
`E=256,N=512` H200 variant, so TP1/TP8 of the Qwen3-VL family and TP4 DeepSeek
would fall back — none of those are benchmarked configurations.
