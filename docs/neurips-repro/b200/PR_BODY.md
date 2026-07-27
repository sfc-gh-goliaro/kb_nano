# B200 reproduction of the NeurIPS benchmark table

Reproduces `table_results.tex` on 8×B200 (sm_100), adapting the H200 procedure in
`docs/neurips-repro/h200/`. Full working notes, including every negative result and
retraction, are in `docs/neurips-repro/b200/REPRODUCTION.md`.

## Coverage

**46 of 47 rows in `kb_nano_models_unified.csv` have a measured speedup on B200.** The
one gap (BitNet) is missing baseline wiring, not a model that fails to run.

Alignment columns are each row's native metric. **Token-alignment columns are not
directly comparable**: the paper reports one scalar per row, ours is the per-scenario
average matched tokens from whichever scenario the harness extracts, so a lower number
may reflect a different scenario rather than worse agreement. The `cos` rows are
directly comparable.

| Architecture | Paper × | Ours × | Ours/Paper | Paper align | Ours align | Why not / caveat |
|---|---|---|---|---|---|---|
| Llama-3.1 | 1.04 | 1.011 | 0.97 | 408.5 tok | 208.5 tok | |
| DeepSeek-V3.2 | 0.84 | 0.629 | **0.75** | 294.1 tok | 82.3 tok | n=64 only |
| Mixtral | 0.97 | 1.156 | 1.19 | 108.9 tok | 104.3 tok | |
| BitNet 1.58b | 1.12 | **—** | — | Top-20 100% | — | **no baseline wired** (runs; 1.221× vs ref) |
| GPT-OSS 120b MXFP4 | 1.02 | 0.984 | 0.96 | 599.6 tok | 485.7 tok | |
| GPT-OSS 120b (2nd cfg) | 1.02 | 0.968 | 0.95 | 599.6 tok | 59.1 tok | |
| EAGLE-3 | 0.98 | 1.099 | 1.12 | Top-20 100% | 173.6 tok | n=16; needs `sglang` venv |
| Gemma-4 | 1.00 | 0.880 | **0.88** | ≈100% | 93.7 tok | needs `vllm020` venv |
| Mamba | 1.05 | **1.073** | 1.02 | 541.3 tok | 261.2 tok | fixed in this PR |
| Mamba2 | 0.97 | **0.973** | 1.00 | 555.9 tok | 319.5 tok | fixed in this PR |
| RWKV-7 | 1.18 | 0.986 | **0.84** | 593.8 tok | 140.4 tok | improved 0.90→0.99 in this PR |
| GLA | 1.85 | 1.037 | **0.56** | 645.5 tok | 418.9 tok | same kernel as reference; engine-bound |
| RetNet | 1.86 | 1.142 | **0.61** | 647.0 tok | 65.7 tok | |
| Qwen-3-Next | 1.24 | 1.137 | 0.92 | 487.4 tok | 20.0 tok | stale run; alignment now 110 tok/seq |
| Kimi-Linear | 1.20 | 2.769 | 2.31 | Top-20 99.2% | 5.1 tok | n=64; **vLLM 0.18 baseline unstable** |
| TTT-E2E | 1.10 | 0.779 | **0.71** | NLL 8.3e-2 | — | |
| Jamba | 1.02 | 0.988 | 0.97 | 415.4 tok | 95.4 tok | |
| FLUX.1-Dev | 1.01 | 0.965 | 0.96 | cos 0.995 | cos 0.9936 | |
| HunyuanVideo-1.5 | 0.97 | 2.331 | 2.40 | cos 0.924 | cos 0.9320 | |
| SDXL | 1.17 | 1.033 | **0.88** | cos 0.982 | cos 0.9692 | |
| SAM3.1 | 1.05 | 1.073 | 1.02 | 0.980/0.949/0.975 | cos 0.9668 | n=100 |
| Whisper | 0.95 | 1.225 | 1.29 | 388.7/444 | 1748 tok | n=100 |
| CosyVoice3 | 2.13 | 1.930 | 0.91 | cos 0.999 | cos 0.9994 | |
| Qwen2-VL | 0.91 | 0.927 | 1.02 | 539.4 tok | 1024 tok | |
| Qwen3-VL | 1.39 | 1.703 | 1.22 | 368.5 tok | 897.1 tok | |
| Qwen-2.5-Omni | 2.02 | 1.861 | 0.92 | 36.2% EM | 525.0 tok | |
| SigLIP-2 | 0.93 | 1.016 | 1.09 | cos 1.000 | cos 0.9998 | fixed in this PR |
| DINOv3 | 0.99 | 1.439 | 1.45 | cos 1.000 | cos 1.0000 | |
| SwinV2 | 1.17 | 0.990 | **0.85** | cos 1.000 | cos 1.0000 | |
| MobileNetV4 | 1.15 | 1.042 | 0.91 | cos 1.000 | cos 1.0000 | |
| ConvNeXtV2 | 0.99 | 0.970 | 0.98 | cos 1.000 | cos 1.0000 | |
| EfficientNetV2 | 1.05 | 1.047 | 1.00 | cos 1.000 | cos 1.0000 | |
| YOLOv10 | 1.06 | 1.450 | 1.37 | 1.000 ×3 | cos 0.9984 | |
| RTDetrV2 | 1.08 | 0.906 | **0.84** | 1.000 ×3 | cos 1.0000 | r101vd; serial re-run 0.96–1.01× |
| 3DGS | 0.99 | 1.018 | 1.03 | cos 1.000 | cos 1.0000 | needs `gs` venv |
| InstantNGP | 0.97 | 0.981 | 1.01 | cos 1.000 | cos 1.0000 | fox-scene data |
| PointTransformerV3 | 1.00 | 1.172 | 1.17 | cos 0.9796 | cos 0.9908 | needs `ptv3` venv |
| OpenFold3 | 1.03 | 0.997 | 0.97 | 100% pass | pass 100% | |
| Pi0 | 3.48 | 2.477 | **0.71** | cos 0.9997 | cos 1.0000 | needs `openpi` venv |
| DP3 | 1.42 | 1.030 | **0.73** | cos 1.000 | cos 1.0000 | n=100 |
| DLRMv2 | 1.06 | 1.154 | 1.09 | cos 1.000 | **cos 0.5000** | alignment metric suspect |
| LightGCN | 1.03 | 1.016 | 0.99 | cos 1.000 | cos 1.0000 | MovieLens-1M |
| BGE-M3 | 1.06 | 1.497 | 1.41 | cos 0.999 | cos 0.9950 | serial re-measure gave 0.964× |
| ColBERTv2 | 3.08 | 3.936 | 1.28 | cos 0.99999 | cos 0.9950 | |
| LLaDA | 1.07 | 1.088 | 1.02 | 98.35% | cos 0.9999 | |
| Oasis | 1.29 | 1.151 | **0.89** | n/a | cos 0.9962 | |
| V-JEPA 2 | 1.01 | 0.895 | **0.89** | cos 1.000 | cos 1.0000 | |

Rows below 90% of paper: DeepSeek-V3.2, Gemma-4, RWKV-7, GLA, RetNet, TTT-E2E, SDXL,
SwinV2, RTDetrV2, Pi0, DP3, Oasis, V-JEPA 2.

## Fixes in this PR

| Row / area | Before | After | Cause |
|---|---|---|---|
| Mamba | 0.94× | **1.09×** | decode is launch-bound (81% dispatch vs 15% GPU wait); CUDA-graph coverage was capped at `min(max_num_seqs, 256)` while `max_num_seqs`=1024, so most batches replayed no graph |
| Mamba2 | 0.42× | **0.97×** | vLLM passes `is_blackwell=True` to `selective_state_update`, selecting `BLOCK_SIZE_M=32/num_warps=8` for `dstate>64`; Codestral has `dstate=128`, so without the flag we ran the same kernel at an 8× smaller tile |
| Qwen3-Next alignment | 30.1 tok | **110.2 tok** | `in_proj_ba` had a midpoint-splitting TP loader (the Qwen3.5 layout); this checkpoint ships one fused interleaved weight, so the split permuted rows across a boundary that does not exist |
| RWKV-7 | 0.90× | **0.97×** | `LayerNorm(promote_fp32=True)` upcast every hidden state for provably identical output — `F.layer_norm` already accumulates in fp32 for bf16 input |
| Block-FP8 startup | crash | fixed | `_warmup_deepgemm` sliced a column-major scale buffer, giving DeepGEMM a stride of `max_tokens` where its SM100 check requires the actual M. Any block-FP8 model died at startup on B200 under default settings; the bench never saw it because it sets `VLLM_DEEP_GEMM_WARMUP=skip` |
| BitNet | 0.090× | **1.221×** | CUDA graphs were off by default for its bench |
| SigLIP-2 / DINOv3 | 0.82× / 0.93× | 1.02× / 1.44× | `torch.compile` enabled in the timm registry |
| SwinV2 (B200-only) | 0.945× | 1.005×, cos 1.000000 | fp32-promoted norms on the Blackwell path only |
| Qwen3-VL-235B-A22B-FP8 | did not run | 1.818× | enabled on both sides |
| DeepSeek-V3.2 | no result | 0.629× | first measurement on this platform |

**H200 safety.** The Mamba2 and SwinV2 fixes are gated on sm_100 and are no-ops
elsewhere. Mamba graph coverage is derived from *measured* free memory — capture
largest-first, stop below a headroom threshold, treat a per-bucket
`torch.OutOfMemoryError` as "stop here" — so on a smaller card it captures fewer
buckets instead of OOMing; an uncaptured bucket runs eager, making coverage a
performance knob and never a correctness one. Validated on Codestral (captures 896
buckets, unchanged at 1.00/0.95/0.96×), the model the old constant existed to protect.

## Negative and retracted results

Recorded in full because they cost real time and would otherwise be repeated:

- **Mamba2 has no reachable ceiling from scheduling.** A microbenchmark "hard ceiling"
  of ~6,850 tok/s was derived and had to be retracted: it was the *narrow tile's*
  ceiling only. The reference was never bound by it.
- **RWKV-7 prefill budget: retracted.** A sweep of
  `FASTKERNELS_MAX_NUM_BATCHED_TOKENS` appeared to give 0.71→0.90×, but `FLAEngine`
  never reads that variable — all arms ran identical configs and the spread was
  run-to-run variance. The tell was already in the notes: matched tokens were
  byte-identical across arms, which is what a knob that never applied looks like.
- **RTDetrV2 "was never behind": retracted.** Those runs used `rtdetr_v2_r18vd`, copied
  from a usage example; the paper row and all canonical jobs use `r101vd`. Re-measured
  correctly it is 0.96× median / 1.006× best-of-3 — still behind.
- **Gemma-4 "fixed to 0.943×": corrected.** That was `--num-seqs 300`; at the canonical
  n=1000 the same code gives 0.88×.
- **FLA's fused kernels are not universally faster.** `fused_addcmul_rwkv7` is
  reproducibly *slower* for us (varlen 56.9→59.9 ms) and `token_shift` costs a
  device→host sync per call (`cu_seqlens.max().item()`), ~64 per prefill step, which the
  reference never pays because it prefills dense padded batches.
- **Kimi-Linear's baseline is unusable on vLLM 0.18/B200** (parity assert + async race,
  both upstream).
- **Detection `torch.compile` is fast but not exact** (2.29× at boxes cos 0.931): the
  correctness gate compares an argmax over 80 classes and a score top-k over
  mostly-junk detections, so any change in accumulation order reorders them. Left
  opt-in and off by default.

## Reproducing

- Clocks are **not** pinned on this host (`clocks.applications.graphics` 1965 MHz is a
  boost ceiling; idle sits at 120 MHz and `nvidia-smi -lgc` needs root). Any row whose
  timed region is under ~30 s needs **≥3 serial repeats reduced best-of-N per side** —
  observed spread on RTDetrV2 was 0.79–1.33× across serial runs of unchanged code.
- Several benches copy batches host→device *inside* the timed region, so host memory
  state enters the measurement; it adds noise to both arms rather than biasing one.
- Do not run these benches wide when producing reported numbers. Three concurrent
  BGE-M3 runs each write 17 GiB of output tensors immediately after their timed region,
  landing on their neighbours' timed encodes.
- **Six rows need dedicated reference venvs** under `/home/yak/repro_venvs/`: Gemma-4
  (`vllm020`), EAGLE-3 (`sglang`), Pi0 (`openpi`), PointTransformerV3 (`ptv3`), DLRMv2
  (`dlrm`), 3DGS (`gs`). Omitting the `--vllm-python` flag fails with a
  pydantic/transformers error that never mentions the environment.

## Known-weak evidence

- Small-n rows: DeepSeek-V3.2 (n=64), EAGLE-3 (n=16), Kimi-Linear (n=64),
  SAM3.1/Whisper/DP3 (n=100). Kimi-Linear's 2.769× in particular is n=64 against an
  unstable baseline.
- **DLRMv2** reports cos 0.5000 against a paper 1.000 — that looks like a broken metric
  rather than a 50% mismatch, and should be checked before the row is cited.
- **BGE-M3** shows 1.497× here but 0.964× under serial repeats, so the canonical
  directory holds a differently-configured run.
- **Qwen-3-Next** speedup is from a pre-fix run; its alignment is now 110 tok/seq, not
  the 20.0 shown.

## Open leads

- Three rows show the same signature — our operators match or beat the reference while
  the scenario loses: RWKV-7 decode (forward 25.0 ms vs reference 34.7 ms), Mamba
  prefill, and GLA (1.037× against 1.85× on the *same* `chunk_gla` kernel). One
  engine-loop investigation would likely move several rows.
- DeepSeek-V3.2's sparse indexer is **ruled out** as the alignment cause: layer 0's
  top-2048 selection is bit-identical to vLLM's (per-row Jaccard 1.000000), decaying to
  ~0.89 by layer 60 only as upstream drift crosses the top-k boundary. First divergence
  is in layer 0's output.
- `promote_fp32=True` is provably a no-op wherever a norm's weight dtype matches its
  activations, and 12+ other models use that default (DINOv3, CLIP, T5, Whisper, SAM3,
  V-JEPA 2, Hunyuan, EVA, Oasis, AlphaFold3). Gating it on `weight.dtype == x.dtype`
  would remove casting copies repo-wide while keeping DeepSeek's fp32-weight case.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
