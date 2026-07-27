# fastkernels — `table_results.tex` reproduction on B200

Blackwell (8× NVIDIA B200, sm_100) reproduction of the paper's benchmark table,
requested by the NeurIPS reviewers alongside the existing H200 results in
[`../h200/REPRODUCTION.md`](../h200/REPRODUCTION.md).

## 0. Host

| | |
|---|---|
| GPUs | 8× NVIDIA B200 (183 GiB each), sm_100, driver 580.159.03 |
| CUDA toolkits | 13.0 (system default `/usr/local/cuda`), 12.9, 12.0 |
| Base env (conda `dev`) | Python 3.12.12, torch 2.10.0+cu128, vLLM 0.18.0, vllm-omni 0.18.0, transformers 4.57.6, flashinfer 0.6.6, triton 3.6.0, fla 0.5.1, diffusers 0.39.0 |
| fastkernels | installed `-e` |
| `HF_HOME` | `/home/yak/data-fast/huggingface` |

```bash
bash docs/neurips-repro/b200/setup_repro_envs.sh   # reference repos, isolated venvs, instant-ngp, weights
```

Isolated venvs under `~/repro_venvs/` (reference side only; fastkernels always
runs in `dev`): `openpi`, `dlrm`, `ptv3`, `gs`, `vllm020`, `sglang` (0.5.9, for
the EAGLE-3 reference — the base env has no `sglang`/`sgl_kernel`).

The sweep is driven by `b200_repro/sched.py`, a GPU-pool scheduler that packs
jobs onto the 8 GPUs by their TP degree (`jobs_b200_full.json`), with
`watchdog.sh` relaunching it for jobs that failed, up to 3 attempts each.

## 1. Blackwell-specific defects found and fixed

The paper's numbers were produced on H200. Every row below reproduced on H200 but
failed or silently mis-computed on B200, because the sm_100 code paths are newer
and less exercised. All fixes are no-ops on sm_90.

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | Whisper: `Paged KV cache block size must be divisible by 256` | L1 attention ops probed only `is_fa_version_supported(3)`, which is false on B200, so all of them fell back to upstream `flash_attn` (FA2) | `_fa_backend` centralizes the probe: FA3 on Hopper, FA4 on Blackwell, FA2 only as a last resort |
| 2 | Whisper: paged cross-attention read the wrong memory | Cross-attention stored K/V in the TRTLLM **HND** layout (page 16) on Blackwell but read it back with flash_attn's **NHD** paged kernel | Dispatch Whisper cross-attention to `TRTLLMPrefill`/`TRTLLMDecode` like `attention_impl.py` does; teach `TRTLLMPrefill` the non-causal paged case via FlashInfer's `BatchPrefillWithPagedKVCacheWrapper` (the TRTLLM-gen context kernel is causal-only) |
| 3 | GPT-OSS: greedy output diverged from vLLM immediately (**19 of 15652 tokens matched**) | `Attention` passes attention sinks (`s_aux`) and `window_size` as kwargs; `TRTLLMPrefill`/`TRTLLMDecode` swallowed `**kwargs`, so every layer ran with no sink logit and sliding-window layers ran as full attention | Translate and forward both to the TRTLLM kernels' `sinks`/`window_left` using vLLM's convention (`window_left = sliding_window - 1`, sinks in fp32) |
| 4 | Kimi-Linear: `CUBLAS_STATUS_NOT_INITIALIZED` in the MoE router GEMM | The L1 csrc extension linked `-lcublas` from the **system default toolkit (13.0)** while torch 2.10.0+cu128 loads cuBLAS **12**; cuBLAS 13 rejects a cuBLAS-12 handle from `getCurrentCUDABlasHandle()` | `csrc/__init__.py` selects a toolkit whose CUDA major matches torch's (12.9 here) before building |
| 5 | Qwen3-Next: `launch shared memory exceeds current GPU arch sm_100a allowed` (329728 > 232448 bytes) | FA4's Blackwell kernels are TMEM-limited to `head_size <= 128` (plus 192 for MLA); Qwen3-Next's head_dim is 256 | `fa_version_for_head_dim()` mirrors vLLM's rule in `fa_utils.get_flash_attn_version`, resolved per layer (per call for MLA's 192/576) |
| 6 | EAGLE-3: same paged-block-size crash, plus a wrong-layout cache view | `TreeAttnPrefill` gated its vLLM-FA path on `cc[0] == 9`; and tree verification re-views the paged cache at 1-token page granularity, which is only meaningful in NHD | Route tree attention through `_fa_backend`; pin the EAGLE-3 engine to the flash_attn (NHD) backend before the models are built. This also restores CUDA-graph capture, which `_setup_cuda_graphs` skips without FA paged tree attention |
| 7 | Kimi-Linear: `Dense decode MLA is only supported on SM90a architecture` | FlashMLA's dense decode kernel is Hopper-only, and `MLAAttention` used it unconditionally | Add `TRTLLMMLADecode` (FlashInfer `trtllm_batch_decode_with_kv_cache_mla`) and select it on sm100+ for the BF16 dense-decode paths, matching vLLM's `FLASHINFER_MLA` backend and its shape constraints (`qk_nope_head_dim` in {64,128}, page size in {32,64}). Validated to cosine >= 0.999996 against a PyTorch absorbed-MLA reference |
| 8 | Qwen3-Next: `delta rule kernel does not support this device major version: 10` | FlashInfer builds its gated-delta-rule prefill kernel for SM90 only (`gdn_prefill_sm90`), but the gate was `capability[0] >= 9`, which treats Blackwell as Hopper | Gate on exactly `(9, 0)`, matching vLLM's `ChunkGatedDeltaRule` (`is_device_capability(90)`), so B200 takes the already-present Triton/FLA fallback |
| 9 | Kimi-Linear (after defect 7): `v must have shape (total_k, num_heads_k, head_size)` | MLA prefill runs qk at 192 and v at 128. Only FA3-on-Hopper and FA4 take that natively, and `TRACEABLE_FA_AVAILABLE` is gated to FA3, so Blackwell fell through to upstream flash_attn, which rejects it. The gate's comment assumed FA2 was only inadequate for paged KV | Zero-pad v out to q's head dim when the resolved kernel cannot do it, mirroring vLLM's `MLACommonImpl._pad_v` / `_flash_attn_varlen_diff_headdims` (callers already slice back to `v_head_dim`), and translate `return_softmax_lse` to upstream's `return_attn_probs` |
| 10 | RetNet: CUDA OOM on **both** engines | RetNet-2.7B has the largest recurrent state of the FLA rows; at the default 512 scheduler slots the state cache plus its per-step `index_select` gather exhausts 178 GiB. `expandable_segments` did not help -- it is capacity, not fragmentation | Halve the scheduler width on both sides (`--max-num-seqs 256 --ref-max-num-seqs 256`) so the comparison stays symmetric |

Two environment defects were also fixed, outside the repo:

* **`nvidia-cutlass-dsl` version skew** — `nvidia-cutlass-dsl` 4.5.3 shipped a
  `.pth` pointing at its self-contained `python_packages/`, but leftover
  `nvidia-cutlass-dsl-libs-{core,cu12}` 4.6.1 had installed an incomplete
  `dsl_packages/cutlass/` tree with no `__init__.py`. Some processes resolved
  `cutlass` there and died with `cannot import name '_cutlass_ir'`, which broke
  **vLLM's own FA4** (visible as a bge-m3 baseline failure) as well as ours.
  Pinned to a self-contained 4.5.3 and removed the stray tree. 4.6.1 is *not* a
  valid alternative: it breaks `quack-kernels` 0.5.0
  (`cutlass.cute.core has no attribute ThrMma`).
* Missing packages for rows whose references are optional deps: `blobfile`
  (Kimi tokenizer), `torch-geometric` (LightGCN reference).

## 2. Results

Speedup = fastkernels / reference, arithmetic mean over the row's scenarios.
Generated by `b200_repro/compare.py`, which maps every paper row to its result
directory and tolerates the differing per-bench JSON schemas:

```bash
python b200_repro/compare.py            # paper vs B200, flags rows <90% of target
```

**The sweep is still in flight, and the numbers below are not the ones to quote.**
There are four passes, in order:

1. **Coverage** (8 GPUs wide) -- establishes that every row runs and aligns. Fast, but
   the host is ~1.5x oversubscribed at 8-wide, so its throughput numbers are not
   comparable to the paper's. The original rationale here was that fastkernels is
   host-heavier than the references and therefore understated at 8-wide; the clean
   pass shows that was wrong, and the direction is **row-specific** rather than a
   uniform bias:

   | row | 8-wide | quiet | direction |
   |---|---|---|---|
   | RWKV-7 | 1.270x | 0.710x | badly overstated at 8-wide |
   | YOLOv10 | 1.136x | 0.926x | overstated |
   | DINOv3 | 0.963x | 0.934x | ~unchanged |
   | Mamba2 | 0.424x | 0.526x | understated |
   | ConvNeXtV2 | 0.860x | 1.200x | badly understated |

   So "we measured it on a busy box" is not a conservative excuse in either
   direction, and only the quiet pass is quotable.
2. **Quiet re-measurement** (3 GPUs wide) -- the pass whose numbers are meant to be
   quoted. **The first attempt is void**: every row in it finished between 15:02 and
   15:21, which is exactly when the attention A/Bs, the Mamba2 profile and the
   gpt-oss debug runs in §3 were also loading the box. It moved rows in both
   directions (RWKV-7 1.270 -> 0.841, ConvNeXtV2 0.860 -> 1.200), so it cannot be
   quoted either way.
3. **Clean re-measurement** (`jobs_remeasure2.json`, all 17 rows, nothing else
   scheduled) -- in progress, 10/17 landed, every job rc=0. Results so far:

   | row | paper | B200 (quiet) | ratio | alignment |
   |---|---|---|---|---|
   | Llama-3.1 | 1.04x | 0.969x | 0.93 | 27.4/507 |
   | Mamba | 1.05x | 0.933x | 0.89 | 256058 tok |
   | Mamba2 | 0.97x | 0.526x | 0.54 | 315832 tok |
   | RWKV-7 | 1.18x | 0.710x | 0.60 | 140395 tok |
   | GLA | 1.85x | 1.053x | 0.57 | 418912 tok |
   | SDXL | 1.17x | 1.021x | 0.87 | cos 0.9716 |
   | SigLIP-2 | 0.93x | 0.842x | 0.91 | cos 0.9999 |
   | DINOv3 | 0.99x | 0.934x | 0.94 | cos 1.0000 |
   | SwinV2 | 1.17x | 0.874x | 0.75 | cos 1.0000 |
   | YOLOv10 | 1.06x | 0.926x | 0.87 | cos 0.9984 |
   | RTDetrV2 | 1.08x | 1.011x | 0.94 | cos 1.0000 |
   | FLUX.1-Dev | 1.01x | 0.964x | 0.95 | cos 0.9936 |

   The correctness metrics reproduce throughout (embedding/box/latent cosines at or
   near 1.000); the shortfalls are throughput. Llama-3.1 at 0.969x is the honest
   number for that row -- the 1.034x quoted earlier came from the 8-wide pass.
4. **Tuned column** (`jobs_page64.json`) -- the same attention rows at HND page 64,
   which measures 4-12% faster at identical alignment (§3).
5. **Authoritative post-parity pass** (`jobs_quiet1.json` + `jobs_quietN.json`) -- in
   progress, and the one to quote. Everything before it predates the block-table stride
   fix, the page-64 default and the FlashInfer MXFP4 path, all of which change the numbers.
   Split by GPU count deliberately: the 43 single-GPU rows run on a **1-wide pool**, so
   exactly one benchmark is on the machine at a time, and the 3 multi-GPU rows (Mixtral
   TP=4, Qwen3-Next TP=2, gpt-oss-120b TP=2) follow on a 4-wide pool where a TP=4 job
   occupies the box anyway. 1-wide costs wall-clock -- roughly 8 hours for 43 rows, with 7
   GPUs idle -- but 4-wide was measured to cost Llama 6% (0.977x standalone versus 0.917x
   contended), and a table that cannot be quoted is worth less than idle GPUs. BitNet runs
   last with a shortened timeout so its hang cannot stall the other 42 rows.
   `jobs_remeasure` only ever covered the 17 rows judged most host-sensitive, which left
   **~30 table rows still carrying 8-wide coverage numbers**. Given that the 8-wide pass
   is demonstrably off by up to +-40% on individual rows (RWKV-7 1.270x vs 0.710x), those
   cannot be quoted either, so every remaining row is being re-measured. Run 4 wide
   rather than 3 only because Mixtral is TP=4 and a 3-wide pool could never place it.

**Authoritative pass, rows landed so far** (1-wide, one benchmark on the machine at a
time; regenerate the full table with `compare.py` once the pass completes):

| row | paper | B200 | ratio | alignment vs paper |
|---|---|---|---|---|
| Llama-3.1 | 1.04x | 0.987x | 0.95 | 237 / 408.5 |
| GPT-OSS (20b, TP=1) | 1.02x | 0.967x | 0.95 | **640 / 599.6 (107%)** |
| Mamba | 1.05x | 0.950x | 0.90 | 315 / 541.3 |
| Mamba2 | 0.97x | 0.533x | **0.55** | 456 / 555.9 (82%) |
| Whisper | 0.95x | **1.430x** | **1.50** | 69.9 / 388.7 (18%) |
| Qwen2-VL | 0.91x | **0.933x** | **1.03** | **677 / 539.4 (126%)** |

Each agrees with the paired A/Bs measured earlier, so the pass is self-consistent.
Qwen2-VL now **exceeds both** its paper speedup (1.03x of target) and its alignment
(126%), with the text-only scenario at perfect agreement -- 1024.0 of 1024 matched tokens --
against 171 mean before the block-table stride fix. That is the clearest single demonstration
of what that fix bought.

Whisper now **exceeds** its paper speedup by 50% (1.43x against 0.95x), having gained from
both the block-table stride fix and the page-64 default; its low agreement is the
reference-side batch dependence documented above -- our Whisper output is stable across batch
sizes (444.0) where vLLM's is not (47.6). Mamba2
remains the single worst speed row and its cause is already established above: GPU-bound
decode at roughly 3.7x the state-bandwidth floor, with vLLM's own kernels, comparable
concurrency and graph coverage all eliminated.

Rows for which no quiet number will exist, and why: **Kimi-Linear** (vLLM 0.18 cannot
run it on B200 under any MLA backend, so there is no reference to divide by), both
**GPT-OSS** rows under CUDA graphs (they crash; the eager 20b variant *is* in the quiet
pass), **DeepSeek-V3.2** and **Qwen3-VL-235B** (gpus=8, their own phase, gated on BitNet
releasing its reserved GPU), and **BitNet** itself, which is mid-run as a reserved-GPU
orphan.

Rows marked *RAN, NO BASELINE* produced fastkernels throughput with no reference to
divide by (see §1's leaked-memory note) and are re-run automatically by
`next_jobs.py`.


> **Read the `measured` column.** `compare.py` prints the timestamp of the
> result file each row came from and marks anything older than `--after` as
> `STALE`:
>
> ```bash
> python b200_repro/compare.py --after "07-25 08:30"
> ```
>
> The smoke pass wrote results into the same directories at reduced scale, so a
> row whose full-scale job has not finished yet still shows a plausible-looking
> number from smoke. Ten rows were stale at the time of writing — including four
> of the `BEHIND` flags (SigLIP-2, SwinV2, YOLOv10, RTDetrV2) and both GPT-OSS
> rows. Those four say nothing about full-scale behaviour, and an earlier
> revision of this document mistakenly listed three smoke numbers as full-scale
> results.

Current state, generated (not transcribed -- an earlier revision of this file
carried three hand-copied smoke numbers as if they were full-scale results):

```
Category     Row                   paper    B200   ratio  measured      alignment
--------------------------------------------------------------------------------------------------------
Dense/MoE    Llama-3.1              1.04   1.034    0.99  07-25 08:44 OLDER n=1000  total_matching_tokens=27422
Dense/MoE    DeepSeek-V3.2          0.84       —       —  —             (no result yet)
Dense/MoE    Mixtral                0.97   0.918    0.95  07-25 08:34 n=1000  total_matching_tokens=37725
Dense/MoE    BitNet 1.58b           1.12       —       —  —             (RAN, NO BASELINE)
Dense/MoE    GPT-OSS (MXFP4)        1.02   0.833    0.82  07-25 08:20 STALE n=32  total_matching_tokens=18  <-- BEHIND
Dense/MoE    GPT-OSS 20b (extra)    1.02   0.846    0.83  07-25 11:38 n=1000  total_matching_tokens=28812  <-- BEHIND
Dense/MoE    EAGLE-3                0.98       —       —  —             (RAN, NO BASELINE)
Dense/MoE    Gemma-4                1.00       —       —  —             (no result yet)
Linear-attn  Mamba                  1.05   0.896    0.85  07-25 11:47 n=1000  total_matching_tokens=257053  <-- BEHIND
Linear-attn  Mamba2                 0.97   0.424    0.44  07-25 11:46 n=1000  total_matching_tokens=313617  <-- BEHIND
Linear-attn  RWKV-7                 1.18   1.270    1.08  07-25 09:45 n=1000  total_matching_tokens=140395
Linear-attn  GLA                    1.85   1.004    0.54  07-25 09:40 n=1000  total_matching_tokens=418912  <-- BEHIND
Linear-attn  RetNet                 1.86   2.611    1.40  07-25 12:29 n=1000  total_matching_tokens=65742
Linear-attn  Qwen-3-Next            1.24   1.126    0.91  07-25 12:45 n=1000  total_matching_tokens=19985
Linear-attn  Kimi-Linear            1.20   2.099    1.75  07-25 13:55 OLDER n=32  total_matching_tokens=154
Linear-attn  TTT-E2E                1.10   0.781    0.71  07-25 12:02    <-- BEHIND
Linear-attn  Jamba                  1.02   0.996    0.98  07-25 09:37 n=1000  total_matching_tokens=93587
Vision/AV    FLUX.1-Dev             1.01   0.963    0.95  07-25 09:40 n=1632  cos 0.9936
Vision/AV    HunyuanVideo-1.5       0.97   2.052    2.12  07-25 13:11 n=1003  cos 0.9306
Vision/AV    SDXL                   1.17   1.024    0.88  07-25 11:45  cos 0.9746  <-- BEHIND
Vision/AV    SAM3.1                 1.05   1.077    1.03  07-25 09:42 n=100  cos 0.9668
Vision/AV    Whisper                0.95   1.490    1.57  07-25 12:00 n=100  total_matching_tokens=83208
Vision/AV    CosyVoice3             2.13   2.168    1.02  07-25 12:15  code2wav cos 0.9994
Multimodal   Qwen2-VL               0.91   0.972    1.07  07-25 09:33 n=1000  total_matching_tokens=33059
Multimodal   Qwen3-VL               1.39   1.606    1.16  07-25 09:28 n=1000  total_matching_tokens=290362
Multimodal   Qwen-2.5-Omni          2.02   1.277    0.63  07-25 09:41 n=1000  total_matching_tokens=19931  <-- BEHIND
Multimodal   SigLIP-2               0.93   0.649    0.70  07-25 10:47  cos 0.9999  <-- BEHIND
Multimodal   DINOv3                 0.99   0.963    0.97  07-25 10:16  cos 1.0000
Multimodal   SwinV2                 1.17   1.009    0.86  07-25 10:44  cos 1.0000  <-- BEHIND
Edge/Det     MobileNetV4            1.15   0.625    0.54  07-25 10:44  cos 1.0000  <-- BEHIND
Edge/Det     ConvNeXtV2             0.99   0.860    0.87  07-25 10:17  cos 1.0000  <-- BEHIND
Edge/Det     EfficientNetV2         1.05   1.101    1.05  07-25 10:32  cos 1.0000
Edge/Det     YOLOv10                1.06   0.839    0.79  07-25 10:35 n=5000  cos 0.9984  <-- BEHIND
Edge/Det     RTDetrV2               1.08   0.972    0.90  07-25 10:37 n=5000  cos 1.0000  <-- BEHIND
3D/Robotics  3DGS                   0.99   0.996    1.01  07-25 12:03  cos 1.0000
3D/Robotics  InstantNGP             0.97   1.009    1.04  07-25 12:04  cos 1.0000
3D/Robotics  PointTransformerV3     1.00   1.004    1.00  07-25 12:04  cos 0.9908
3D/Robotics  OpenFold3              1.03   0.753    0.73  07-25 10:50  pass_rate=1  <-- BEHIND
3D/Robotics  Pi0                    3.48   2.470    0.71  07-25 12:06  cos 1.0000  <-- BEHIND
3D/Robotics  DP3                    1.42   0.712    0.50  07-25 10:44 n=100  cos 1.0000  <-- BEHIND
Recsys/Spec  DLRMv2                 1.06   1.077    1.02  07-25 12:05  cos 0.5000
Recsys/Spec  LightGCN               1.03   1.033    1.00  07-25 10:44  cos 1.0000
Recsys/Spec  BGE-M3                 1.06   1.194    1.13  07-25 10:48  cos 0.9950
Recsys/Spec  ColBERTv2              3.08   4.004    1.30  07-25 10:47  cos 0.9950
Recsys/Spec  LLaDA                  1.07   1.058    0.99  07-25 14:13  cos 0.9999
World model  Oasis                  1.29   1.180    0.91  07-25 10:54  cos 0.9962
World model  V-JEPA 2               1.01   0.994    0.98  07-25 12:08  cos 1.0000

43/47 rows have results; 17 below 90% of paper
  BEHIND GPT-OSS (MXFP4): paper 1.02x, B200 0.833x
  BEHIND GPT-OSS 20b (extra): paper 1.02x, B200 0.846x
  BEHIND Mamba: paper 1.05x, B200 0.896x
  BEHIND Mamba2: paper 0.97x, B200 0.424x
  BEHIND GLA: paper 1.85x, B200 1.004x
  BEHIND TTT-E2E: paper 1.10x, B200 0.781x
  BEHIND SDXL: paper 1.17x, B200 1.024x
  BEHIND Qwen-2.5-Omni: paper 2.02x, B200 1.277x
  BEHIND SigLIP-2: paper 0.93x, B200 0.649x
  BEHIND SwinV2: paper 1.17x, B200 1.009x
  BEHIND MobileNetV4: paper 1.15x, B200 0.625x
  BEHIND ConvNeXtV2: paper 0.99x, B200 0.860x
  BEHIND YOLOv10: paper 1.06x, B200 0.839x
  BEHIND RTDetrV2: paper 1.08x, B200 0.972x
  BEHIND OpenFold3: paper 1.03x, B200 0.753x
  BEHIND Pi0: paper 3.48x, B200 2.470x
  BEHIND DP3: paper 1.42x, B200 0.712x
```

### Caveat: the sweep runs 8 jobs per host, and the host is oversubscribed

The sweep packs 8 benchmark jobs onto the 8 GPUs. Each job's Python engine takes
~27 cores at peak, so the 192-core host runs at a load average near 300 —
**about 1.5x oversubscribed**. Two consequences matter for the numbers:

* fastkernels is host-heavy relative to vLLM/FLA, so CPU starvation costs it
  more than the reference.
* Within a job the two engines run *sequentially*, so they are measured under
  whatever load happened to be present at the time — the comparison is not
  guaranteed symmetric.

The per-scenario shapes suggest this is real rather than hypothetical: the two
FLA rows each have exactly one anomalous scenario rather than a uniform deficit
(GLA prefill-heavy 0.542x against balanced 1.241x and decode-heavy 1.229x;
RWKV-7 decode-heavy 0.549x against prefill-heavy 2.16x), and GLA's prefill-heavy
wall time (192.7 s) exceeds its balanced wall time (81.7 s) despite producing
*fewer* tokens. A genuine kernel or scheduling deficit would not be confined to
one scenario per row.

Contention is not equally suspect everywhere, and it is worth being precise
about where it bites:

* **LLM / FLA rows** are most exposed. The two sides are *separate
  subprocesses* (a vLLM or FLA worker, then the fastkernels engine) that run at
  different times with different host-work profiles, so they can be measured
  under different load.
* **timm / image-classification / detection rows** are more robust than I first
  assumed. `bench_timm` builds each batch with `_preprocess_batch` **before**
  `torch.cuda.synchronize(); t0 = perf_counter()`, so host-side decode and
  transform are outside the timed region, and both engines are in-process Python
  forwards measured the same way. CPU starvation stretches these jobs' wall
  clock badly (30+ min each when five run at once, versus ~190 s alone) without
  necessarily distorting the ratio. SigLIP-2 at 0.818x and SwinV2 at 0.896x may
  therefore be real deficits rather than artifacts.

They are all in the re-measurement list regardless, since separating the two
explanations costs one clean run each.

So the sweep establishes **coverage** — that every row runs and aligns — but the
throughput numbers quoted for the rebuttal should come from a low-concurrency
pass. `jobs_remeasure.json` holds the affected and headline rows; run it over a
restricted pool so at most a few jobs share the host:

```bash
python sched.py jobs_remeasure.json --pool 0,1,2 --status logs/status_remeasure.json
```

`watchdog.sh` runs this automatically as **phase 2**, entered once phase 1 (the
8-wide coverage sweep, including retries) has nothing left. **Phase 3** is
DeepSeek-V3.2 at TP=8 on its own: it needs all 8 GPUs at once, so running it
earlier would monopolise the pool and delay both the retries and the clean
re-measurement — and it is the one row the H200 reproduction also left blank
(known decode regression). Its timeout is 3 h rather than 6.

Two rows now read as large **overshoots**, and they deserve the same scepticism
as the deficits: CosyVoice3 4.894x against a 2.13x target (one scenario's
reference took 586 s to our 109 s) and HunyuanVideo-1.5 2.770x against 0.97x
(vllm-omni 1040 s to our 369 s for 16 videos). Both alignments reproduce
(HunyuanVideo frame cos 0.927 vs the paper's 0.924), so the *outputs* are right;
it is the reference's throughput that looks anomalous, which is what a saturated
host or a Blackwell fallback path in vllm-omni would both produce. Both are in
the re-measurement list.

Until that pass lands, treat ratios below 0.9 as unconfirmed **and large
overshoots too** — CosyVoice3 reads 4.894x against a 2.13x target, driven by a
scenario where the reference took 586 s against our 109 s. A 2.3x overshoot is no
more credible than an undershoot when the host is saturated, so it is in the
re-measurement list alongside the deficits.

## The alignment column, not the speedup column, is what B200 breaks

The speedups largely reproduce. Token-level *agreement with the reference* does
not, and it does so across almost every row that the paper scores in "avg toks" --
including plain dense Llama-3.1, which has no MoE, no linear attention and a
head_dim of 128. `align.py` compares the paper's Align./Value column against the
newest full-scale run per model (largest `num_seqs` wins, so a 16-seq diagnostic
can never masquerade as a headline number):

```
model                                n  B200 avg toks (per scenario)      mean   paper  ratio
------------------------------------------------------------------------------------------------
AI21-Jamba-Mini-1.7               1000  94/256 147/256 94/256            111.6   415.4   0.27
Kimi-Linear-48B-A3B-Instruct        32  5/501                              4.8       -      -
Llama-3.1-8B-Instruct             1000  27/507 21/511 21/988              23.2   408.5   0.06
Mamba-Codestral-7B-v0.1           1000  316/624 388/638 667/1248         457.2   541.3   0.84
Mixtral-8x7B-Instruct-v0.1        1000  38/624 29/638 46/1248             37.4   108.9   0.34
Qwen2-VL-7B-Instruct              1000  33/1024 25/512 456/512           171.4   539.4   0.32
Qwen2.5-Omni-7B                   1000  20/526 20/512 244/512 8/256       73.0       -      -
Qwen3-Next-80B-A3B-Instruct       1000  20/512 34/526 25/1024             26.1   487.4   0.05
Qwen3-VL-8B-Instruct              1000  290/1024 20/512 17/512           109.0   368.5   0.30
gemma-4-26B-A4B-it                1000  97/508 107/518 104/986           102.7       -      -
gla-2.7B-100B                     1000  419/624 457/638 879/1241         584.8   645.5   0.91
gpt-oss-120b                        32  1/489                              0.6   599.6   0.00
gpt-oss-20b                       1000  29/489 19/493 21/943              22.9   599.6   0.04
mamba-2.8b-hf                     1000  257/607 221/611 475/1210         317.7   555.9   0.57
retnet-2.7B-100B                  1000  66/624 66/638 157/1241            96.4   647.0   0.15
rwkv7-2.9B-g1                     1000  140/552 126/551 158/1084         141.5   593.8   0.24
whisper-large-v3                   100  32/444                            31.8   388.7   0.08

rows agreeing with the reference far less than the paper:
  gpt-oss-120b                         0.6 vs   599.6  (0%)
  gpt-oss-20b                         22.9 vs   599.6  (4%)
  Qwen3-Next-80B-A3B-Instruct         26.1 vs   487.4  (5%)
  Llama-3.1-8B-Instruct               23.2 vs   408.5  (6%)
  whisper-large-v3                    31.8 vs   388.7  (8%)
  retnet-2.7B-100B                    96.4 vs   647.0  (15%)
  rwkv7-2.9B-g1                      141.5 vs   593.8  (24%)
  AI21-Jamba-Mini-1.7                111.6 vs   415.4  (27%)
  Qwen3-VL-8B-Instruct               109.0 vs   368.5  (30%)
  Qwen2-VL-7B-Instruct               171.4 vs   539.4  (32%)
  Mixtral-8x7B-Instruct-v0.1          37.4 vs   108.9  (34%)
```

This is a bigger deal for the rebuttal than any single slow row, so it is worth
being precise about what it is and is not.

**It is not garbage output.** Decoding both sides of Qwen3-Next gives coherent,
fluent, semantically equivalent completions that differ in wording from the first
or second token -- e.g. "Here's the provided data transformed into a clean JSON
array" vs "Here's your data converted into a clean JSON array", both followed by
byte-identical JSON. Greedy decoding turns an arbitrarily small logit difference
into a different token, and from there the sequences never re-converge, so "20
matched tokens" is a measure of when the first tie flipped, not of quality.

**It is batch-occupancy dependent.** Two independent observations:

* Llama-3.1 at `--num-seqs 64` matches 213.5 / 273.6 / 229.7 tokens. The same
  build at 1000 prompts matches 27 / 21 / 21.
* Within a single full-scale run, the divergence point falls monotonically with
  position in the request queue -- later requests are the ones running while the
  batch is fullest:

  | requests | mean divergence index |
  |---|---|
  | 0-199 | 27.9 |
  | 200-399 | 22.2 |
  | 400-599 | 16.9 |
  | 600-799 | 10.9 |
  | 800-999 | 6.1 |

  (`balanced`, 1000 requests; only 17/1000 diverge at token 0, so this is not a
  prompt-handling bug.)

So the mechanism is a scheduling/batching-dependent numerical difference that grows
with concurrency, not a wrong kernel. The open question is what makes it so much
worse on B200 than on H200 -- the H200 reproduction of *this same harness* matched
390.9/444 on Whisper where B200 manages 31.2/444, so the effect is
Blackwell-specific, not a property of the benchmark.

Forcing the NHD path at full scale (`FASTKERNELS_FORCE_NHD=1`, with the pin verified
firing via `[attn] pinned backend: use_trtllm=False block_size=256`) recovers an
order of magnitude of agreement -- 204.7 / 255.8 / 253.8 matched tokens against
27 / 21 / 21 -- at the cost of throughput, 0.58 / 0.64 / 0.70x against 1.03x. So the
attention path is implicated. But four attempts to name *which part* of it all
failed:

1. **The KV page size.** Blackwell gets `block_size=16` (HND, required by TRTLLM-gen)
   where Hopper gets 256, so paged decode does 16x the partial reductions.
   `probe_decode_accuracy.py`: TRTLLM/page-16 and flash_attn/page-256 are both
   2.2e-3 from an fp32 reference, flat from batch 1 to 512. Layout is not an accuracy
   factor.
2. **Decode kernel selection.** vLLM's auto-detection is
   `use_trtllm = num_tokens <= 256` (`vllm/utils/flashinfer.py`), and the full-scale
   Llama log confirms the reference used the FlashInfer wrapper for every decode step
   -- it logs "Using TRTLLM prefill attention (auto-detected)" and never the decode
   equivalent -- while we use TRTLLM-gen at every batch size. The two kernels are
   2.4-3.6e-3 apart (`probe_decode_agreement.py`), more than either is from fp32, so
   this looked decisive. It is not: forcing the reference onto TRTLLM-gen decode at
   every batch size (`FASTKERNELS_VLLM_FORCE_TRTLLM=1`) left agreement at
   27.4 / 21.0 / 21.1, indistinguishable from auto-detect. The knob and the
   observation are worth keeping -- the wrapper is the *more* accurate kernel at
   batch >= 256 (1.66e-3 vs 2.20e-3 from fp32) -- but it does not explain the
   collapse.
3. **Prefill accuracy.** `probe_prefill_accuracy.py`, ragged batch with real
   wildchat-like lengths: TRTLLM-gen paged prefill 1.952e-3 from fp32, flash_attn
   varlen 1.948e-3, only 1.307e-3 from each other -- tighter than the decode pair,
   and uniform across per-request lengths from 137 to 2001 tokens.
4. **Batch-composition sensitivity.** `probe_batch_sensitivity.py` holds one sequence
   fixed and changes only its neighbours: both kernels shift that sequence's output
   by ~3e-3 (TRTLLM 3.02e-3, flash_attn 2.70e-3), and both *saturate* by batch 64
   rather than growing with occupancy. Real, and enough to make bitwise agreement
   impossible at scale, but symmetric between the two paths.

### The actual cause: a column-sliced block table in the eager decode path

**This supersedes the batch-invariance analysis below.** That section measured a
consequence of this bug, not a property of the TRTLLM-gen kernel; it is kept because the
elimination sequence is still useful, but its conclusion was wrong.

`_run_decode_greedy_eager` passed the block table as
`self._eager_block_tables[:n, :bt_cols]` -- a **column slice** of a wider allocation, so
`size(-1) == bt_cols` while `stride(0) == max_num_blocks`. FlashInfer's TRTLLM-gen
launcher derives the block-table row stride from `size(-1)`, not from `stride(0)`, so
every row except row 0 read **entirely the wrong KV pages**.
`probe_blocktable_stride.py` shows it with nothing else in play -- identical page ids,
sliced view versus a contiguous copy:

```
row 0: max abs diff 0          cos 0.99999994
row 1: max abs diff 0.036377   cos 0.00000000   <-- WRONG
...
7/8 rows differ. A stride-aware kernel would give 0/8.
```

Cosine ~0 means orthogonal output: not a rounding difference, different data.

**Why this is B200-only, and why H200 never showed it.** The captured-graph path always
passed the full-width buffer, so the bug only fires on the eager path -- which is reached
when the decode batch exceeds the 512-entry graph ceiling (`max_capture_limit`). The KV
cache is sized from HBM, so B200's 183 GiB admits ~750 concurrent sequences for an 8B
model where H200's 141 GiB admits ~530. B200 crosses the ceiling routinely; H200 never
does. It also explains the signature that had been visible all along and unexplained:
`exact_matches` was 0-1 per 1000 requests, because only the row-0 sequence was ever
computed correctly.

Llama-3.1, 1000 prompts, matched tokens per scenario:

| config | prefill-heavy | balanced | decode-heavy |
|---|---|---|---|
| before, fully eager | 30.1 | 23.5 | 27.2 |
| **after, fully eager** | **475.8** | **510.6** | **887.8** |
| after, default (graph <=512, eager above) | 208.9 | 243.0 | 259.2 |

The fixed eager path **exceeds the paper's H200 figure of 408.5 avg**, at unchanged
throughput (1.02 / 0.96 / 0.94x versus 1.02 / 0.95 / 0.96x). The fix is gated on
`ATTN_BACKEND_CONFIG.use_trtllm` so the H200 flash_attn argument is unchanged byte for
byte.

**Mixtral confirms it, and reproduces both columns.** At its paper TP=4:

| config | speedup | matched tokens | mean |
|---|---|---|---|
| before the fix | 0.918x (8-wide) | 36.8 / 28.7 / 45.8 | 37.1 |
| after, default | 0.90 / 0.91 / 0.90x | 103.1 / 105.0 / 113.3 | **107.1** |
| after, eager | 1.03 / 0.99 / 0.97x | 93.4 / 94.9 / 108.0 | 98.8 |

Paper: **0.97x / 108.9 tok**. The default arm reproduces the alignment almost exactly
(107.1 vs 108.9) and the eager arm reproduces the speedup almost exactly (0.997 vs 0.97),
so between them the row is reproduced on both axes.

Note the graph/eager ordering **reverses** between the two models: for Llama the eager
path aligns far better (476 vs 209), for Mixtral the graph path aligns slightly better
(107 vs 99). So the residual graph-path effect is not a systematic defect in one
direction -- it is per-model variation around the paper value, which weakens the case for
treating the graph path as broken and argues against disabling decode graphs on sm100.

### Still open: the captured-graph path caps agreement at ~209

With the stride bug fixed, the graph path is the limiting one, and identically so whether
or not concurrency is capped to the ceiling (209.0 / 243.1 / 259.1 at
`max_num_seqs=512`). Two candidate explanations tested and refuted:

* **The `max_context_len` convention** -- capture bakes in `max_model_len`, eager passes
  the true batch max, and both reach the kernel as `max_seq_len`.
  `probe_decode_paths.py` runs one real decode batch under each: **rel 0.000e+00**, bit
  identical. Inert.
* **Batch padding to a capture bucket** -- `probe_graph_padding.py` replays real decode
  batches through both paths: 114/114, 152/154 and 194/194 tokens agree. So the graph
  path is not badly wrong; it flips of order 1% of tokens per step, which compounds to
  roughly the observed ceiling (1/0.013 ~ 77 tokens, observed ~209, same order).

The honest limit of that probe: with short generations it only ever observed decode
batches up to 194, while the real 1000-prompt run reaches ~870. The regime that matters
has not been measured, so the next step is a per-step graph-vs-eager trace at the batch
sizes the benchmark actually produces, not another hypothesis.

### What it actually is: the TRTLLM-gen path is not batch-invariant

The mistake in all four attempts above was measuring *our* kernels against fp32 and
against each other. The right question is how each **stack** behaves against
**itself** at two different batch sizes. Mean token index at which a request's output
starts to differ between a batch-64 run and a batch-1000 run, over the same 64
prompts:

| | prefill-heavy | balanced | decode-heavy |
|---|---|---|---|
| fastkernels | 45.6 | 32.5 | 23.5 |
| vLLM | 505.8 | 508.9 | 941.0 |

vLLM reproduces its own output almost exactly regardless of batch size. We do not
agree with *ourselves* past ~30 tokens once the batch is large. Cross-stack agreement
then tracks our self-stability almost exactly -- 198-245 matched tokens at batch 64,
22-40 at batch 1000 -- which is what you would expect when our own batch dependence
is the dominant term. The paper's alignment column is measured at 1000 prompts, so on
B200 it is largely measuring this.

Two controls make the claim specific rather than hand-wavy:

* **Both stacks are perfectly deterministic at fixed batch.** Two independent
  1000-prompt runs agree to 510.6 of 511 tokens, for us and for vLLM alike. So this
  is batch dependence, not nondeterminism.
* **It localises to the TRTLLM-gen kernels, not to the layout.** Self-stability of
  each of our configurations against our own batch-64 run: TRTLLM page-16 **32.5**,
  TRTLLM page-64 **32.5**, flash_attn/NHD **211.2**, vLLM **508.9**. Page size makes
  no difference; the kernel family makes all of it.

And vLLM's own behaviour is the tell. It *switches decode kernels* between those two
scales -- TRTLLM-gen at batch 64, the FlashInfer wrapper at 1000, per its
`num_tokens <= 256` rule -- and still reproduces its own output to 505/507 tokens.
So the wrapper and TRTLLM-gen agree with each other on real data, and the wrapper is
batch-invariant while TRTLLM-gen is not. `probe_batch_sensitivity.py` shows the
mechanism at kernel level: TRTLLM-gen returns bitwise-identical output for a fixed
sequence at batch 1 and 8, then a different answer from batch 64 on, i.e. it selects
a different split/reduction configuration as the batch grows.

**Preemption is not involved**, though it looked like the best candidate:
`Sequence.preempt()` resets `token_ids` to the prompt and clears `generated_ids`,
restarting a sequence instead of resuming it the way vLLM's recompute preemption
does. But the preemption counter added for this shows **zero** preemptions at both 64
and 1000 prompts, so the path never runs here. (The counter is worth keeping, and the
restart-vs-resume difference is worth fixing on its own merits, since it would
corrupt exactly this metric on any host where the KV cache does fill.)

### Sweeping it across rows: both stacks are batch-dependent, differently

`batch_invariance.py` runs the same self-comparison for every row that happens to
have runs at two scales. The comparison is legitimate because
`bench/utils/real_prompts.py` shuffles with a fixed seed and then takes the first
`num_requests`, so a 32-request run uses exactly the prompt prefix of a 1000-request
run.

```
model                                  scales      ours  reference  scenario
----------------------------------------------------------------------------------
Kimi-Linear-48B-A3B-Instruct         32->1000       0.6          -  balanced
Mamba-Codestral-7B-v0.1              32->1000      23.9       23.9  balanced
Mixtral-8x7B-Instruct-v0.1           32->1000       0.1        2.1  balanced
Qwen2-VL-7B-Instruct                 16->1000      28.9          -  text-only
Qwen2.5-Omni-7B                      16->1000      27.1          -  text
Qwen3-Next-80B-A3B-Instruct          16->1000      84.7      101.1  prefill-heavy
Qwen3-VL-8B-Instruct                 16->1000      20.8          -  text-only
gpt-oss-20b                          32->1000       0.0        7.8  balanced
mamba-2.8b-hf                        32->1000       2.2          -  balanced
whisper-large-v3                      16->100     444.0       47.6  librispeech

2/10 rows are markedly less batch-invariant than their reference:
  gpt-oss-20b                      ours     0.0 vs reference     7.8
  Mixtral-8x7B-Instruct-v0.1       ours     0.1 vs reference     2.1
```

This is less tidy than the Llama result and more informative. Batch-invariance is a
property of a *particular stack on a particular model*, and neither side has it in
general:

* **Llama-3.1 (dense)** -- vLLM invariant (508.9), we are not (32.5). Our problem.
* **Mixtral, GPT-OSS (MoE)** -- *neither* side is invariant (0.1 vs 2.1, 0.0 vs 7.8).
  Expert routing plus grouped GEMMs are batch-dependent on both stacks, so at
  1000-request concurrency this metric is close to meaningless for MoE rows -- note
  the paper itself reports only 108.9 matched tokens for Mixtral, the lowest value in
  the table, which is consistent with that.
* **Whisper** -- the reverse: *we* are invariant (444.0, i.e. perfectly stable) and
  the **reference** is not (47.6). So Whisper's 31.2/444 is substantially the vLLM
  side moving, not ours. That materially changes how that row should be presented.
* **Mamba-Codestral** -- both 23.9, and both hold up reasonably against the paper
  (0.84 of target).

Which closes the loop on why H200 saw none of this. On H200 we run flash_attn/NHD,
which measures as the batch-invariant path (211.2 vs TRTLLM's 32.5), against a vLLM
that is also invariant for dense models -- so the two stacks track each other and
Llama reports 408.5. On B200 we switch to TRTLLM-gen, which does not have the
property, and the agreement goes with it.

### What this means for the two columns

The speedup column is unaffected: it is a throughput measurement and does not care
about batch-invariance. The alignment column, at the concurrency the paper uses, is
substantially a measurement of our Blackwell attention path's batch dependence.

Three configurations, full-scale Llama-3.1, same harness:

| config | matched tokens | speedup |
|---|---|---|
| TRTLLM/HND page 16 (default) | 27 / 21 / 21 | 1.03x |
| TRTLLM/HND page 64 | 26 / 20 / 22 | **1.10 / 1.07 / 1.07x** |
| flash_attn/NHD page 256 | 205 / 256 / 254 | 0.58 / 0.64 / 0.70x |

Two actionable results fall out:

* **Page 64 is a free win, confirmed on two models.** Identical alignment, more
  throughput:

  | model | page 16 | page 64 | our tok/s change |
  |---|---|---|---|
  | Llama-3.1 | 1.03x | 1.10 / 1.07 / 1.07x | +7 / +7 / +12% |
  | Mixtral | 1.02 / 1.02 / 0.94x | 1.05 / 1.06 / 0.98x | +4 / +5 / +4% |

  Matched tokens move by less than a token either way (Mixtral 36.8/28.7/45.8 ->
  36.8/28.3/46.9), so this buys throughput without trading agreement. It is a
  one-line change to the `cc[0] >= 10` branch of `AttnBackendConfig.auto_detect`,
  and therefore cannot affect H200 -- but the default is deliberately **left at 16**
  for now so the re-measurement pass reports the configuration the paper describes.
  `jobs_page64.json` sweeps the attention rows at 64 as a separate tuned column,
  scheduled after the clean pass so the two never compete for the host. On the quiet
  host the first two rows are in, and the pin is confirmed firing
  (`use_trtllm=True block_size=64`):

  | row | paper | page 16 | **page 64 (quiet)** | vs paper |
  |---|---|---|---|---|
  | Llama-3.1 | 1.04x | 0.969x (quiet) | **1.077x** | **1.04** |
  | Whisper | 0.95x | 0.809x (NHD arm) | **1.33x** | **1.40** |
  | Qwen2-VL | 0.91x | 0.972x (8-wide) | **1.07x** | **1.18** |
  | Jamba | 1.02x | 0.996x (8-wide) | 0.983x | 0.96 |
  | Qwen-3-Next | 1.24x | 1.127x (8-wide) | 1.107x | 0.89 |
  | Mixtral (TP=4) | 0.97x | — | 0.973x | **1.00** |
  | Gemma-4 | 1.00x | 0.733x | 0.737x | 0.74 (no-op) |

  Three rows go **above their paper target** at page 64 -- Llama-3.1, Whisper and
  Qwen2-VL -- where the as-published page-16 configuration leaves Llama 7% short.
  Matched tokens are unchanged on Llama (25.9/507 vs 27.4/507) and actually *improve*
  on Whisper (31.2 -> 40.3/444). Jamba barely moves, which fits: it is a hybrid whose
  attention layers are a minority. Gemma-4's no-change is the expected result rather than
  a null one -- its head_dim > 256 forces the NHD pin, which takes precedence over the
  page setting, so that row doubles as a consistency check that the knob only affects the
  HND path. Mixtral at its paper TP=4 lands at 0.973x against a 0.97x target -- exactly
  on it (it needed the widened 4-wide pool to be schedulable at all).

  This makes the Blackwell page size the single most valuable change found in this
  reproduction.

### SwinV2: the constant bias recompute was worth ~46%

SwinV2 is a line-for-line port of timm's, so parity is the ceiling -- but both sides
recompute the continuous position bias on every forward (an MLP over the coords table,
a gather, a sigmoid, a permute and a contiguous, once per block per batch) even though
in inference it is a function of fixed buffers and fixed weights. Caching it under
no-grad eval is numerically free and takes it off the hot path.

The correctness check is unambiguous: embedding cosine **1.000000** with **MSE
0.00e+00** on both scenarios, i.e. bit-identical output. Our own throughput at
default-res went 1377.12 -> 2014.66 img/s (+46%), moving the row from 0.82x to 0.96x.

That comparison is not yet trustworthy as a *ratio*, though: timm's own number moved
1679.81 -> 2096.13 img/s (+25%) between the two runs, so the pair is measuring host
conditions as much as the change. `FASTKERNELS_SWINV2_BIAS_CACHE=0` exists purely to
re-run the two arms back to back (`jobs_swinv2b.json`), which is queued. The bit-exact
correctness result stands regardless.
* **The principled fix is to mirror vLLM's heuristic**: keep the HND cache but hand
  decode to `BatchDecodeWithPagedKVCacheWrapper` above 256 tokens, exactly as the
  reference does, keeping TRTLLM-gen for the small-batch case and for sinks (the
  wrapper does not support them). That should restore agreement *and* batch-invariance
  without giving up the Blackwell layout, and the wrapper is the more accurate of the
  two at batch >= 256 anyway (1.66e-3 vs 2.20e-3 from fp32). Not implemented here: it
  needs a `plan()` per step shared across layers and has to stay CUDA-graph-safe,
  which is a real change rather than a flag, and doing it blind at this stage would
  risk the numbers already collected.

Whisper shows the same effect but is not fully explained by it: the NHD path takes it
from 31.2/444 to **65.8/444** (at 0.81x, down from 1.48x). A 2x recovery from the
same cause, with a large residual -- so the Whisper row has an additional
divergence of its own, consistent with the earlier finding that its two new
components are individually correct.

Rows that do hold up are at least consistent with an attention-path cause: GLA (0.91
of paper), Mamba-Codestral (0.84) and Mamba-2.8b (0.57) are the recurrent models,
whose state update does not page a KV cache at all.

### POSTPONED: the monolithic TRTLLM-gen MoE routes internally, and upstream distrusts that

Parked deliberately; recorded so it is not lost.

The reference's malformed 120b output is **not** a detokenization problem. At token-ID
level, request 0: ours `[200005, 35644, 200008, ...]` = `<|channel|> analysis <|message|>`,
vLLM `[220, 200003, 35644, 200008, ...]` = `' ' <|constrain|> analysis <|message|>`. It emits
two tokens where we emit one and picks the wrong control token -- `<|constrain|>` (200003)
instead of `<|channel|>` (200005) -- on 941 of 1000 requests. Allowing up to a 2-token shift
on either side lifts mean agreement only from 0.8 to 7.2 tokens, so it is a bad control-token
prefix *and* real divergence immediately after, not a pure offset.

Why the 120b and not the 20b is **not established**. Both sizes run identical code on both
sides and log the same backend choice ("Using FlashInfer MXFP4 BF16 backend for SM100");
they differ only in expert count (32 -> 128) and depth (24 -> 36).

The lead worth following when this is picked up again: upstream vLLM has documented exactly
this failure mode for the **sibling FP8 kernel**. Issue
[#37591](https://github.com/vllm-project/vllm/issues/37591), fix #37605, "Disable monolithic
TRTLLM MoE for Renormalize routing":

> Renormalize/RenormalizeNaive are excluded: the monolithic kernel's internal routing for
> these methods produces output uncorrelated with the modular kernel's output and with
> Triton kernel's output for Qwen3.5-35B-A3B-FP8.

Qualifications: it patches `experts/trtllm_fp8_moe.py`, not the MXFP4 path
(`trtllm_nvfp4_moe.py` still lists `Renormalize` as supported), and it is **not** in the
installed vLLM 0.18.

**This is a risk on our side too, not just the reference's.**
`trtllm_fp4_block_scale_moe` is the monolithic FP4 member of the same kernel family, we pass
`routing_method_type=1` (Renormalize), and it performs top-k + softmax *internally* -- the
exact combination upstream now refuses for FP8. Our 20b agreeing with vLLM at 486/489 does
not clear it, because both stacks call the same monolithic kernel and would share the defect.
The safer design, and what upstream now does, is to compute the routing ourselves and hand
`topk_weights`/`topk_ids` to a modular kernel instead of letting the monolithic kernel route.
That is a real change to `Mxfp4MoE.forward_flashinfer`, and it is testable against the
existing fp32-dequantised reference (0.45% rel-max at E=128).

### The 120b cannot be alignment-verified against vLLM 0.18 on B200

Both of vLLM's usable sm100 MXFP4 backends fail as a correctness oracle for the
128-expert model, for different reasons:

| reference for 120b TP=1 | reference well-formed | our matched tokens | note |
|---|---|---|---|
| default `SM100_FI_MXFP4_BF16` | **30-70/1000 (3-7%)** | 8.1 / 9.2 / 14.1 | reference opens 94% of completions with a bare space |
| `VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1` | 1000/1000 (100%) | 42.5 / 43.3 / 52.1 | quantises activations to MXFP8, so a different numeric path by construction |

The MXFP8 run also reads 0.23-0.27x on speed, but that is not a regression on our side --
vLLM runs 16,370 tok/s there against 5,315 for us because it is computing in lower
precision, which its own log warns "may impact accuracy".

So: with a malformed reference the comparison is meaningless, and with a well-formed one the
reference is doing different arithmetic. The 120b's agreement number on B200 is therefore
not interpretable against vLLM 0.18 either way, and no amount of work on our side changes
that. What *is* interpretable: the **20b** at TP=1 against vLLM's BF16 backend agrees to
485.9 / 492.7 / 941.9 on the identical code path, which is the evidence that our MXFP4
implementation is correct -- and the 20b is the paper's row.

This also bounds the tensor-parallel finding below: the TP=1-vs-TP=2 contrast is established
on the **20b**, where both configurations have a valid reference. The 120b diverges at ~30-50
tokens in *both* configurations, which is consistent with having no valid reference rather
than with TP being its problem.

### GPT-OSS alignment: the loss is tensor parallelism, and it is intrinsic

The 20b isolates the variable. Same model, same 32 experts, only TP changed:

| gpt-oss-20b | matched tokens | speedup |
|---|---|---|
| TP=1 | **485.9 / 492.7 / 941.9** | 0.94 / 0.98 / 0.98x |
| TP=2 | 93.1 / 107.2 / 117.7 | 0.96 / 0.96 / 0.96x |

So the 120b TP=2 figure (63.1 / 59.6 / 72.3) is this effect, not 128-expert sensitivity --
that hypothesis is dead. Two bisects, both negative:

| variant, 20b TP=2 | matched tokens |
|---|---|
| FlashInfer MXFP4 (current) | 93.1 / 107.2 / 117.7 |
| old Triton MXFP4 (`FASTKERNELS_MXFP4_FLASHINFER=0`, eager) | 83.8 / 101.8 / 106.9 |
| custom all-reduce off (`FASTKERNELS_DISABLE_CUSTOM_AR=1`) | 93.1 / 108.0 / 117.1 |

The loss therefore is **not** caused by the MXFP4 path added in this branch (it predates it
and is common to both MoE backends) and **not** by our custom all-reduce. Output stays 100%
well-formed harmony on both sides with mean divergence 76-88 tokens, so it is drift, not
corruption.

What remains is intrinsic: tensor parallelism splits every row-parallel GEMM into partial
sums whose rounding differs from the unsharded product, and that difference compounds over
24-36 layers. vLLM shards the same way, but not bit-identically, so the two stacks drift
apart faster under TP than without it.

**This is probably not the configuration the paper measured.** Our 20b at TP=1 gives ~640
mean matched tokens against the paper's 599.6 -- i.e. the target is met at TP=1 and missed
by 6x at TP=2 -- and the 120b's MXFP4 weights fit on a single B200 (177 GiB of 183 GiB), so
TP=1 is available for it too. Confirming that needs a TP=1 reference that is not itself
malformed, since vLLM's own TP=1 120b opens 94% of completions with a bare space; a run with
vLLM on its MXFP8 MoE backend is in flight to get one.

Incidental: Triton at TP=2 runs at 0.52-0.73x against the FlashInfer path's 0.96x, which is
independent confirmation that the new path is the right one on Blackwell.

### The paper's GPT-OSS row is the 20b at TP=1 -- and it reproduces

`table_results.tex` names no size, so this had to be reconstructed from the repo. The
decisive artifact is `bench/eval/config.py`'s `MODEL_KEY_TO_DEFAULT_HF`, the canonical
per-architecture model map: **every other entry matches a paper row one-for-one** -- down
to `Qwen/Qwen3-VL-8B-Instruct-FP8` rather than the 235B, `HunyuanVideo-1.5-...-480p_t2v`
rather than 720p -- and it maps `gpt_oss` to **`openai/gpt-oss-20b`**. Supporting evidence:

* `tests/bench_vllm.py` defaults `--tp` to **1**, and neither the README's canonical
  invocation nor the H200 document's commands pass `--tp`;
* `--num-seqs` defaults to 1000, which is the table's "WildChat 1K";
* the table's "prefill / balanced / decode-heavy" are exactly that script's three
  scenarios.

The 120b appears only in `kb_nano_supported_archs.csv` (a planning spreadsheet outside the
repo) and in the kernel shape registry -- not in the eval config.

So the row is **gpt-oss-20b at TP=1**, and on B200 it reproduces:

| | speedup | matched tokens |
|---|---|---|
| paper | 1.02x | 599.6 |
| B200, 20b TP=1 | 0.967x (**95%**) | 485.9 / 492.7 / 941.9, mean **640** (**107%**) |

`compare.py` now maps the row to the 20b and carries the 120b as an extra. An earlier
revision of that file mapped it to the 120b -- my guess, and it inverted this row's verdict
for most of the reproduction.

The 120b at TP=2 remains a legitimate extra result: it runs, matches vLLM's throughput to
3%, and its lower agreement is the tensor-parallel drift documented below, not a defect
specific to that model.

### GPT-OSS: fixed by taking the backend the reference takes

Both sizes now run on Blackwell, under CUDA graphs, at full scale:

| row | speedup | matched tokens | paper |
|---|---|---|---|
| gpt-oss-20b | 0.98 / 0.98 / 0.99x | **487.0/489, 491.0/493, 942.8/943** | 1.02x / 599.6 tok |
| gpt-oss-120b TP=2 | 0.97 / 0.97 / 0.97x | 63.1 / 59.6 / 72.3 | 1.02x / 599.6 tok |

The 20b reaches **99.6% token agreement** on decode-heavy where it previously crashed
under CUDA graphs and matched 23/600 eagerly. The 120b previously did not run at all,
under graphs or eagerly.

The fix was backend selection, not a kernel bug. `Mxfp4MoE` now implements the path the
reference uses on sm100 -- `prepare_weight_flashinfer` (swap adjacent gate/up rows,
per-expert epilogue row shuffle, `nvfp4_block_scale_interleave`, fp8 scale view) and
`forward_flashinfer` (`trtllm_fp4_block_scale_moe`, `routing_method_type=1` = top-k then
softmax, matching `renormalize=True`). Intermediate *and* hidden are padded to 256
instead of 64/unpadded, with the six loaders bounded on the hidden axis; the zero padding
is inert because it survives the kernel's permutation as zeros and an E8M0 byte of 0
times FP4 0 is 0. Everything is gated on `cc[0] == 10`, which is exactly vLLM's
`is_device_capability_family(100)`; H200 reports (9, 0) so the Triton path there is
untouched. `FASTKERNELS_MXFP4_FLASHINFER=0` forces the old path.

Two bugs surfaced while wiring it up, both worth recording:

* slicing the padded hidden axis returns a stride-`H_pad` view, but the `moe_forward`
  custom op's meta function declares a contiguous result -- it must be materialised;
* `tune_max_num_tokens` selects the kernel's tile configuration, so passing the current
  batch size re-selects it every call. That measured 4x *slower* than the Triton path
  until it was pinned to a fixed bound.

**Correction: the "reference is at fault" claim applies to TP=1 only.** I generalised it
from a TP=1 run; `harmony_check.py` scores each side against the harmony format itself
rather than against the other side, and the two configurations behave completely
differently:

| run | ours well-formed | reference well-formed | mean divergence index |
|---|---|---|---|
| 120b TP=1 | 1000/1000 (100%) | **30-70/1000 (3-7%)** | 0.4 - 1.1 |
| **120b TP=2** | 1000/1000 (100%) | **1000/1000 (100%)** | **47.8 / 51.9 / 50.5** |
| 20b TP=1 | 1000/1000 (100%) | 1000/1000 (100%) | 485.9 |

So at TP=1 the reference really is broken -- it opens 94% of completions with a bare space
instead of `<\|channel\|>`, and emits truncated control tokens like
`<\|constrain\|>fanalysis`. That is a genuine vLLM oddity on B200 but it says nothing
about the TP=2 row.

At **TP=2, the configuration that matters, both sides are 100% well-formed** and we diverge
around token 50, giving 63.1 / 59.6 / 72.3 matched tokens against the paper's 599.6. That
is a real numerical divergence on our side of the same order as the other MoE rows, not a
reference defect, and it means **gpt-oss-120b's alignment does not reproduce** even though
the row now runs and matches vLLM's throughput to 3%.

Note the shape of the TP=1 evidence in hindsight: a divergence index of 0.4 means the two
sides differed on the *first* token of nearly every request, which is the signature of a
formatting break rather than accumulated arithmetic error -- and 47.8 at TP=2 is the
signature of accumulated error. I had both numbers before drawing the conclusion and did
not look at them.

### Correction: the page-64 win is ~1%, not 7-12%

The Blackwell HND page size is now 64 rather than 16 in `AttnBackendConfig.auto_detect`
(sm100 branch only, so Hopper keeps flash_attn/NHD at 256). But the magnitude reported
earlier in this document was wrong. A **back-to-back A/B on one GPU**, post-stride-fix:

| Llama-3.1 | prefill-heavy | balanced | decode-heavy | mean |
|---|---|---|---|---|
| page 16 | 1.02x | 0.95x | 0.97x | 0.980 |
| page 64 | 1.04x | 0.96x | 0.97x | 0.990 |

About **1%**, with alignment unchanged (208.5 vs 209.4 matched tokens). The earlier
7-12% figures compared runs taken at different times under different host load -- the
same trap this document already documents twice, and I fell into it again. The change is
kept because it is consistently non-negative and free, not because it is a large win.

### Historical: the crash before it was understood

### GPT-OSS: the crash is in `matmul_ogs`, localised

Both GPT-OSS rows exhausted their retries with `Triton Error [CUDA]: an illegal
memory access was encountered`, surfacing inside Triton's `load_binary` during CUDA
graph capture. That reads like "JIT-compiling a kernel while a capture is active",
but it is not: `capture_cudagraph` already runs an eager warmup forward at every
batch size before capturing, and CUDA errors are asynchronous, so `load_binary` is
merely the first API call after the real fault.

Re-running with `CUDA_LAUNCH_BLOCKING=1` moves the traceback to the actual kernel:

    tasks/baseline/L1/mxfp4_moe.py:300  _fused_experts -> matmul_ogs(...)
    triton_kernels/matmul_ogs.py:574    kernels._p_matmul_ogs[(grid,)](...)
    RuntimeError: Triton Error [CUDA]: an illegal memory access was encountered

So it is the persistent MXFP4 MoE matmul faulting, not graph capture and not our
attention work. Two further observations narrow it:

* It is **intermittent** -- an earlier full-scale 1000-prompt run of gpt-oss-20b
  completed and produced the alignment numbers quoted above (29/489, 19/493, 21/943),
  and the crash reproduces at `--num-seqs 8`, so it is not a scale limit.
* An out-of-bounds access whose occurrence depends on run-to-run luck points at
  index/metadata memory rather than at the matmul itself: `matmul_ogs` indexes
  through `gather_indx`/`scatter_indx` built in `_routing_from_bitmatrix`, so a
  single garbage routing index reads outside the expert weights.

Two hypotheses tested from there, both refuted:

* **Bad routing indices.** `FASTKERNELS_CHECK_MOE_ROUTING=1` validates
  `dispatch_indx`/`combine_indx` bounds and the expert histogram before each
  `matmul_ogs`. Every call passes. (The check cannot run under graph capture -- it
  syncs -- which incidentally shows the routing computation is itself inside the
  captured region.)
* **Triton scratch recycled across graph captures.** Triton asks a user-installed
  allocator for global scratch; vLLM's is a bare `torch.empty` that nobody retains,
  so the block frees the moment the launch returns, and since every capture here
  shares one `graph_pool` a later capture could be handed the same block. Installing
  a persistently-held scratch buffer did not change the crash, so this was reverted
  rather than left in on a hunch.

What *is* established: the row **runs clean in eager mode** and crashes only with
CUDA graphs. So GPT-OSS-20b gets a real measurement rather than a blank row, at full
scale (1000 prompts, `--enforce-eager`):

| scenario | fastkernels | vLLM | speedup | matched |
|---|---|---|---|---|
| prefill-heavy | 9,195 | 8,252 | **1.11x** | 28.9/489 |
| balanced | 10,418 | 11,616 | 0.90x | 18.9/493 |
| decode-heavy | 11,625 | 14,514 | 0.80x | 21.0/943 |

Mean 0.94x against a 1.02x target, *without* CUDA graphs -- so this is a floor for the
row, and the gap to 1.02x is at least partly the graphs it is not allowed to use. The
graph-mode crash stays open as a `triton_kernels` interaction rather than something in
our call.

And the 120b row narrows it further: it faults in the **split-k `reduce()`** inside
`matmul_ogs` *even eagerly*, where 20b faults in the persistent kernel only under
graphs. (Its `rc=-98` was the stall detector correctly reaping the hang left behind
when the TP=2 worker died.)

**Why we are the only one hitting this.** vLLM's `_get_mxfp4_backend`
(`vllm/model_executor/layers/quantization/mxfp4.py`) returns
`SM100_FI_MXFP4_BF16` -- the **FlashInfer** MXFP4 backend -- for any
`is_device_capability_family(100)` host that has FlashInfer, and only falls through to
`Mxfp4Backend.TRITON` when FlashInfer is *absent*. FlashInfer 0.6.6 is installed here,
so vLLM never executes `matmul_ogs` on this machine, while we always do. Our sm100
`opt_flags` constraints match vLLM's `_swizzle_mxfp4` line for line -- but those are
constraints for a path vLLM does not take on Blackwell, so they are effectively
untested there.

That reframes the row: the GPT-OSS gap and the crash are one issue, our MoE backend
selection on Blackwell, not a bug in our call into `matmul_ogs`.

Two follow-ups queued rather than guessed at:

* **Cheap mitigation: tried, refuted.** `FASTKERNELS_MXFP4_SM100_CONSTRAINTS`
  (opt-in, default unchanged) ran both variants on both rows:
  * `is_persistent=0` is **not a legal configuration** -- `matmul_ogs` raises
    `NotImplementedError: Must use persistent kernel and be TMA-compliant for native
    MXFP`. So the persistent kernel is mandatory here, and the 20b fault inside
    `_p_matmul_ogs` cannot be side-stepped that way.
  * `split_k=1` changes nothing: 20b still dies with an illegal access and 120b still
    aborts (NCCL rank 1). Reading `matmul_ogs.py` explains why the constraint could
    not have helped -- the `matmul` scratchpad, and therefore the `reduce()` call, is
    allocated when `split_k > 1` **or** when `scatter_indx is not None and
    n_expts_act > 1`, and GPT-OSS's second matmul always passes a scatter index with
    4 active experts. Hopper's `split_k=1` never removes that path either; it just
    never faults there.

  Worth being explicit that this whole line of attack was chasing the *reporting*
  site. `reduce_kernel`'s `load_binary` is where the sticky asynchronous error
  surfaces; `CUDA_LAUNCH_BLOCKING=1` puts the actual fault in `_p_matmul_ogs`, and
  that kernel faults only under CUDA graphs for 20b (it is clean eagerly) while 120b
  fails either way.
* **The real fix**: a FlashInfer MXFP4 path for sm100 mirroring
  `SM100_FI_MXFP4_BF16`, which is what the reference actually runs. That is a feature,
  not a flag -- it needs the FlashInfer weight preparation
  (`nvfp4_block_scale_interleave`, intermediate-size padding) as well as the different
  forward -- and it would address the crash, the alignment gap and probably the
  speed together.
(Note `_pack_bitmatrix_kernel` in that file is dead code: routing comes from
`triton_kernels.topk`, so it is not a suspect.)

### Mamba2 is GPU-bound, which rules out the scheduling explanations

Mamba2 (Mamba-Codestral) is the worst speed gap in the table: **0.52x** against a
0.97x target (4,804 / 5,231 / 5,544 tok/s versus vLLM's 8,318 / 10,371 / 11,452).
Two earlier hypotheses were about scheduling -- decode CUDA-graph coverage (capping
capture to 256 made it *worse*, 0.424 vs 0.519) and graph bucket choice.

`FASTKERNELS_PROFILE_MAMBA=1` settles where the time actually goes. Over 2368
fast-decode steps (245,065 decode tokens, 21.6 ms/step):

| phase | time | share |
|---|---|---|
| admit | 4.6 ms | 0.0% |
| decode_prep | 45.9 ms | 0.1% |
| gpu_dispatch | 286.6 ms | 0.6% |
| **gpu + d2h wait** | **50,465 ms** | **98.8%** |
| finalize | 274.0 ms | 0.5% |

Host-side work is 1.2% of the loop in total, and the slow (mixed prefill) path runs
only 8 of 2376 steps. So no scheduling fix can win this back.

But the decode kernels are not ours to blame either: `mamba2_mixer.py` calls vLLM's own
`causal_conv1d_update` and `selective_state_update`, and passes
`conv_state_indices` / `state_batch_indices`, so the state is indexed **in place** with
no gather -- vLLM's exact pattern. Per-step work should therefore match the reference's.

**Graph coverage: tried at 256 / 512 / 896, refuted.** `FASTKERNELS_MAX_NUM_BATCHED_TOKENS`
aside, the one structural difference looked like **graph coverage**. `_mamba_graph_bs_list`
caps captured buckets at `min(max_num_seqs, 256)`, with a comment explaining the memory
cost -- but the engine reports *987* state slots for this model on B200
(133.0 MiB/slot), so every decode batch above 256 replays no graph at all. The cap was
chosen against H200's 141 GiB; this host has 183 GiB. `FASTKERNELS_MAMBA_GRAPH_MAX_BS`
(opt-in, default unchanged) raises it. Both settings ran, and the capture logs confirm
they took effect -- 14 buckets max=256 by default, 18 buckets max=512, 21 buckets
max=896 (capped by the 987 slots). It makes no difference at all:

| captured up to | prefill-heavy | balanced | decode-heavy |
|---|---|---|---|
| 256 (default) | 0.58x | 0.51x | 0.49x |
| 512 | 0.57x | 0.50x | 0.48x |
| 896 | 0.56x | 0.49x | 0.47x |

Concurrency capacity is not it either: vLLM reports 144.63 GiB of state cache against
our 128.22 GiB / 987 slots, so we are within ~13% of the reference's parallelism, not
2x short of it. And vLLM's own per-step tensor prep is identical down to the
`A[:, None, None].expand(...).to(torch.float32)` we do every step.

So for Mamba2: same kernels, same per-step prep, comparable concurrency, host work at
1.2%, graph coverage irrelevant -- and still half the reference's throughput. The full-scale profile
(`FASTKERNELS_PROFILE_MAMBA=1`, 1000 prompts, decode-heavy) finally splits it:

| path | steps | tokens | per step | share of wall time |
|---|---|---|---|---|
| fast decode (graph replay) | 1770 | 622,531 | **56.0 ms** (352 tok/step) | 76% |
| slow mixed prefill | 86 | — | **358.1 ms** | **24%** |

Two separate problems, not one:

* **Prefill is 24% of wall time in 4.6% of the steps.** 86 mixed steps against a
  16384-token budget. That is the same knob that was worth +33-53% on RWKV-7, so
  `jobs_mamba2b.json` tries 65536 and 131072 here too.
* **The decode step is ~3.7x off its memory floor.** At 352 slots the state traffic is
  352 x 133 MiB x 2 (read+write) = 91 GiB per step, which is ~15 ms at achievable HBM
  bandwidth against the 56 ms measured -- and 83% of the step is GPU/D2H wait, so it is
  not launch overhead. vLLM reaches 11,343 tok/s on this row, i.e. roughly 2x our
  fast-path rate of 6,286 tok/s, using the *same* kernels. Closing that needs a
  kernel-level trace (nsys) rather than another engine-level knob; every engine-level
  hypothesis is now eliminated.

Worth noting why the earlier "graph coverage" refutation does not settle this: that
test *capped concurrency* to 256 (`FASTKERNELS_MAX_NUM_SEQS=256`), which brings batches
under the cap by shrinking them, and it made things worse (0.424 vs 0.519). Extending
capture upward is the opposite change and has not been tried.

### GLA and friends: our prefill is the reference's own kernel

GLA lands at 0.95 / 1.03 / 1.18x (mean 1.05) against a 1.85x target, and RWKV-7 and
RetNet sit similarly low. `tasks/baseline/L1/chunk_gla.py` is a thin wrapper around
`fla.ops.gla.chunk_gla` -- the *same* kernel the FLA reference calls. So for these rows
there is no kernel advantage to reproduce: the paper's speedup has to come from the
serving engine around it (batching, decode, graphs), and on B200 the prefill-heavy
scenario is where we fall below parity (0.95x) while decode-heavy still wins (1.18x).
This is a different failure from Mamba2's, and reading
`tests/bench_fla.py` points at something concrete. The reference runs
`_continuous_generate` with `max_prefill_tokens=196608`; our engine's
`_detect_scheduling_defaults` gives `max_num_batched_tokens=16384`. For a chunk-based
linear-attention kernel, that is a **12x smaller prefill launch** on our side, against
a reference calling the same kernel -- which is exactly the shape of the result
(behind on prefill-heavy, ahead on decode-heavy). `FASTKERNELS_MAX_NUM_BATCHED_TOKENS`
already exists, so `jobs_fla.json` swept 65536 and 131072 on RWKV-7 and GLA.

**RETRACTED: the sweep changed nothing, so these numbers are variance.**

| | prefill-heavy | balanced | decode-heavy | mean |
|---|---|---|---|---|
| RWKV-7 @ 16384 (default) | 0.60x | 0.94x | 0.59x | 0.71x |
| RWKV-7 @ 65536 | 0.83x | 1.00x | 0.87x | 0.90x |
| RWKV-7 @ 131072 | 0.80x | 0.97x | 0.86x | 0.88x |
| GLA @ 16384 (default) | 0.95x | 1.03x | 1.18x | 1.05x |
| GLA @ 65536 | 0.97x | 1.04x | 1.17x | 1.06x |

`jobs_fla.json` varied only `FASTKERNELS_MAX_NUM_BATCHED_TOKENS` -- and **`FLAEngine`
never reads that variable.** It is consumed by `LlamaEngine` via
`_detect_scheduling_defaults`; the FLA engine takes its prefill knobs as constructor
arguments (`max_prefill_tokens`, default 196608, and `chunked_prefill_size`), which
`tests/bench_fla.py` passes from its own CLI. So all three RWKV-7 rows above ran with
*identical* engine configuration and the spread between them is pure run-to-run
variance on a contended host. The earlier claims built on it -- "+33% prefill / +53%
decode", "65536 beats 131072 so the budget has an optimum" -- are withdrawn. GLA's
"refuted" is the only conclusion that survives, and only because not moving is what a
no-op predicts.

**The tell was in the evidence already recorded here:** matched tokens came back
*byte-identical* across all three settings (140.4/552, 126.1/551, 157.9/1084), which
was written up as confirmation that this was a pure scheduling knob. Byte-identical
output across arms is what you see when the knob does nothing at all. When an
intervention leaves the output bit-for-bit unchanged, the first hypothesis should be
that it was not applied, not that it was cleanly applied.

RWKV-7's real prefill knob is `--chunked-prefill-size` (CLI default 1024).
`max_prefill_tokens` is already 196608, matching the reference's
`_continuous_generate`, so the remaining asymmetry is chunk *granularity*: the
reference prefills each prompt in a single launch while we split it into 1024-token
pieces before the same chunk kernel. That is the sweep that should have been run.

GLA does not move at all, so despite sharing the same reference and the same kernel its
shortfall (1.05x against 1.85x) has a different cause and is still open.

**These rows also expose a measurement trap worth stating plainly.** RWKV-7 measured
1.270x in the 8-wide coverage pass and **0.71x** (0.60 / 0.94 / 0.59) in the clean
low-concurrency pass. The speedup went *down* when the host got quieter, i.e. the FLA
reference is more host-sensitive than we are, so an oversubscribed host flatters us.
The wide pass overstates these rows and only the quiet numbers should be quoted --
which is the opposite of the correction the pass was introduced to make, and worth
saying because it means "we ran it on a busy box" is not a conservative excuse.

### Gemma-4 runs, on a backend it would not have chosen

With the NHD pin (defect 11) and the stale-capture fix, Gemma-4 completes:
**0.733x** against a 1.00x target (0.727 / 0.762 / 0.710 across the three
scenarios), 19-21% token match.

The speedup should be read with the pin in mind: the row only runs *because* we
force it onto flash_attn/NHD, giving up Blackwell's TRTLLM-gen attention. So this
is fastkernels-on-the-fallback-backend versus a vLLM that is free to use whatever
it likes -- a floor for the row, not a like-for-like comparison. Recovering the
TRTLLM path would require the Triton unified kernel to accept an HND cache, or an
HND-capable kernel for head sizes above 256.

Note also that token match is not the paper's metric for this row: the paper
reports a rank score (Top-20 approximately 100%) produced by
`tests/debug/gemma4_rank_align_vllm.py` run inside the vLLM-0.20.1 venv, so the
19-21% figure here is not comparable to it and the rank scorer has not been run on
B200 yet.

### Open: Qwen3-Next runs but barely agrees with vLLM

Speedup is close to the paper (1.18 / 1.11 / 1.09 vs 1.24x) but token agreement
collapses: 20.0/512, 33.6/526, 24.7/1024 against the paper's 487.4 avg. Diverging
around token 20-30 of a greedy decode means a real numerical defect, not drift.

Three hypotheses tested and refuted, so the cause is *not* in the GDN layers:

* **l2norm placement** and **chunk size** in the prefill path (earlier).
* **Decode gating precision.** vLLM fuses the gate into the recurrent update
  (`fused_sigmoid_gating_delta_rule_update`, g and beta in fp32 registers); we
  materialise g as fp32 and beta as **bf16** and call
  `fused_recurrent_gated_delta_rule`. Since the state is multiplied by exp(g) every
  step, a per-step gate error would compound over a 512-token decode -- and vLLM's
  own unit test only checks one step. `probe_gdn_decode.py` iterates both real
  kernels for 512 steps against an fp32 run of the same kernel:

  | path | out cos | state cos |
  |---|---|---|
  | ours (beta bf16, separate kernel) | 0.999992 | 0.999994 |
  | vLLM (gating fused, fp32)         | 0.999993 | 0.999995 |
  | ours vs vLLM, both bf16           | 0.999995 | 0.999996 |

  Both drift from fp32 by the same amount. Switching to the fused entry point --
  which was the obvious "fix" -- would have changed nothing.

Note also that on B200 the FlashInfer GDN prefill kernel is gated off (it is
sm90-only, defect 6), so both prefill and decode GDN now run vLLM's own kernels.
That leaves the hybrid model's *full-attention* layers, which do run on the
Blackwell TRTLLM path, as the remaining suspect; `FASTKERNELS_FORCE_NHD=1` exists
to test exactly that and the A/B is in flight.

### Harness defect: experiments overwrote the baselines they were measured against

`bench_fla`, `bench_jamba` and `bench_timm` default their `--output-dir` to a **fixed**
path (`<model>_fla_tp1/results.json`), not a timestamped run directory the way
`bench_vllm` does. So every experiment re-run of those rows silently replaced the
default-configuration result it was supposed to be compared against: the prefill-budget
sweep clobbered RWKV-7 and GLA, the page-64 sweep clobbered Jamba, and the bias-cache run
clobbered SwinV2. Those four rows' as-published numbers now exist only in the bench logs.

Two fixes: `jobs_rebaseline.json` re-runs all four at default settings before anything
else, and every experiment job file now passes an explicit
`--output-dir .../\_exp_<phase>_<job>` so a tuned run can never land on a baseline again.

This is the kind of error that does not announce itself -- `compare.py` happily reported
the tuned numbers as if they were the default configuration, and the only reason it
surfaced was checking *why* Llama-3.1 had suddenly jumped to 1.078x in the main table.

### compare.py: two reporting bugs

* **EAGLE-3 read as "RAN, NO BASELINE"** for the whole reproduction. `bench_eagle3`
  names its field after the reference (`speedup_vs_sglang`), which no generic key
  matched, so a perfectly good 0.988x sat unread in the file. Now matched via a
  `speedup_vs_*` fallback that excludes `latency_scenarios` (a different quantity).
  Coverage goes 44/47 -> **45/47**.
* **Reduced-scale diagnostics outranked full-scale runs.** Row selection sorted
  candidates by mtime alone and kept the last, so Qwen-3-Next reported **0.804x** from a
  16-sequence attention A/B while its 1000-prompt run (1.106x) sat in the same tree. Now
  ranked by scale first, then recency, matching `align.py`.

### Harness defect: jobs that need more GPUs than the pool has

`sched.py`'s placement loop waits for `len(free) >= job["gpus"]`, which never becomes
true if the job wants more GPUs than the pool contains -- so it spins forever with the
entire pool idle. This bit twice:

* the TP=8 rows (DeepSeek-V3.2, Qwen3-VL-235B) against a 7-GPU pool, because GPU 0 is
  reserved for BitNet;
* `p64_mixtral`, which inherits Mixtral's TP=4, against a 3-wide measurement pool.

Fixed in two places, because either alone leaves the trap open: `sched.py` now rejects
unschedulable jobs up front with `rc=-97` and a log line rather than spinning on them,
and the watchdog defers the TP=8 phase (with a log line) until the reservation clears
instead of launching it into a pool that cannot hold it. The measurement pools that need
a wider pool for one TP=4 row now ask for 4.

The general lesson, the same one as the reservation defect below: a scheduler that
*waits* for an impossible condition looks identical to a scheduler that is busy. Both of
these presented as "GPUs idle, jobs pending, no error anywhere".

### Harness defect: a reservation that only two of three phases honoured

`logs/reserved_gpus.txt` exists so a long orphaned run cannot have its GPU drained
by a relaunch. BitNet lives there -- its Microsoft reference takes ~2.5h. But the
watchdog's phase 2 had its pool hardcoded as `0,1,2` rather than going through
`pool_excluding_reserved`, so the moment phase 1 finished, phase 2's pre-launch
drain killed BitNet on GPU 0 (`exit code -9`) after its reference had completed --
`results.json` was written with `fastkernels: null`. There is no
reference-reuse flag in the bench, so the whole row had to be re-run from scratch.

Two lessons, both now fixed:

* A guard that is not applied uniformly is not a guard. Every pool now comes from
  `pool_excluding_reserved`, which takes an optional width so phase 2 can still ask
  for a narrow 3-GPU pool.
* Editing a running bash script does not change its behaviour. bash had already
  parsed the `while` loop, so the fixed pool logic did not take effect until the
  watchdog was restarted -- it relaunched on `0,1,2` twice after the file was
  correct on disk. Restart, then verify from `ps` that the new pool is in the
  argv, rather than trusting the edit.

### Superseded: the original Qwen3-Next note

With the GDN gate fixed (defect 8) the row completes at 1.097x mean against a
1.24x target (1.170 / 1.076 / 1.044). The problem is alignment: ~20 / 34 / 24
matching tokens per request against the paper's reported 487.4 average, i.e.
greedy output diverges within a few tokens.

Two of the three hypotheses I recorded are now **eliminated** by reading vLLM's
`qwen3_next.py` next to ours:

* *l2norm placement* — our fallback passes `use_qk_l2norm_in_kernel=True` and does
  **not** pre-normalise, exactly matching vLLM's `fla_chunk_gated_delta_rule`
  call. (Only its FlashInfer path pre-normalises with `l2norm_fwd`, and that path
  is Hopper-only for both of us.)
* *chunk size* — both sides call the same `chunk_gated_delta_rule`, so the
  chunking is identical.

What remains is a real structural difference in **decode**:

| | prefill | decode |
|---|---|---|
| vLLM | `chunk_gated_delta_rule` (Triton/FLA on B200) | `fused_sigmoid_gating_delta_rule_update(A_log, a, b, dt_bias, …)` — gating computed **inside** the kernel |
| fastkernels | same | our `_fused_gdn_gating_kernel` computes `g`/`beta`, then `fused_recurrent_gated_delta_rule(g=g, beta=beta, …)` |

For a recurrent model the forget gate compounds multiplicatively at every step, so
a small difference in `g` is exactly the shape of failure observed. **But this
difference also exists on H200**, where the paper reports good alignment — so it
cannot be the whole story on its own, and I am not claiming it as the cause. The
clean test is to switch our decode to `fused_sigmoid_gating_delta_rule_update`
(vLLM's function is importable, and `a`/`b`/`A_log`/`dt_bias` are all already
available at our call site) and re-measure alignment. Deliberately not done blind:
two self-inflicted regressions this session came from changing attention/MLA code
without being able to test it promptly, and the GPUs are currently full.

### Observation: Blackwell shifts the bottleneck to host overhead

The rows landing under their H200 targets are consistently the ones whose GPU
work is small relative to per-step host work — Mamba (worst in *prefill-heavy*,
best in *decode-heavy*), the vision encoders, and the small detectors. Mamba is
the clearest case: it calls vLLM's **own** `causal_conv1d_fn` /
`selective_scan_fn`, so the kernels on both sides are byte-identical and the gap
cannot be kernel quality. B200's kernels finish sooner, so the same Python
scheduling per step is amortized over less GPU time, and vLLM's lower-overhead
C++/async scheduler pulls ahead. This is a property of the harness on faster
hardware rather than of the kernels, and it is the main thing to quantify for
the rebuttal.

## 2b. Second pass: BitNet, Kimi, and the FA2 fallback

### BitNet was never a kernel problem -- it ran eager

`tests/bench_microsoft_bitnet.py` set `enforce_eager = not args.use_kb_cudagraph`
and `--use-kb-cudagraph` was `store_true`, so the engine ran with no CUDA graphs
at batch 1 against a reference whose official int2 M==1 kernels are
graph-captured. A decode step is ~10 ops x num_layers launches, so the row was
pure host-launch overhead: a 1000-prompt run never finished a single scenario
(14,360 s against the reference's 1,232 s).

Measured on 1x B200, 32 prompts, per scenario (prefill / balanced / decode):

| mode | speedups | mean |
|---|---|---|
| eager (old default) | 0.088 0.091 0.090 | **0.090x** |
| CUDA graphs | 1.126 1.259 1.280 | **1.221x** |

Paper target is 1.12x, so graphs clear it. Alignment does not pay for it:
teacher-forced Top-20 under the official direct-decode reference is 1.0 in all
three scenarios -- the metric the paper reports -- and Top-1 is 0.968 where the
reference's own self-consistency is 0.995.

The hazard that motivated the eager default is handled elsewhere: BitLinear picks
bf16 fake-quant vs int2 off the runtime Context, and torch.compile would
specialize that branch during decode capture, so `engine.py` routes compiled
prefill to the eager model.

### Kimi-Linear: vLLM cannot serve it on Blackwell with its default MLA backend

The row looked like it passed for hours. Our side ran, the job exited 0, and
`results.json` was written with only our throughput -- because `run_worker`
returned `None` for the reference and the bench carried on regardless. Collectors
keyed on the missing speedup field then drop the row silently. That is fixed: the
bench now exits non-zero unless `--skip-vllm` was passed deliberately.

The reference's failure is an upstream bug. Kimi-Linear is a hybrid whose
full-attention layers use MLA, so on Blackwell vLLM selects FLASHINFER_MLA, whose
shape check requires `block_num % (128 / block_size) == 0`. `block_num` is the
*block-table width* (`flashinfer/mla.py`: `B_block_table, block_num =
page_table.shape`), i.e. `ceil(longest_seq / page)` -- 45 for a 2879-token
sequence at page 64. vLLM does not pad it; we do, in
`flashinfer_mla_decode._pad_block_table`, which is why only the reference dies.
Violating it without the check reads out of bounds, which is the illegal memory
access originally observed.

No sizing knob fixes this, since the value follows sequence length:
`num_gpu_blocks_override` changed nothing. CUTLASS_MLA carries the same assert
(`cutlass_mla.py`) and FLASHMLA is Hopper-only ("FlashMLA Dense is only supported
on Hopper devices"), leaving TRITON_MLA. Pinned per-model via
`FASTKERNELS_VLLM_ATTENTION_BACKEND` (vLLM 0.18 dropped `VLLM_ATTENTION_BACKEND`
-- it now warns "Unknown vLLM environment variable" -- so the override is an
EngineArgs field).

That is necessary but **not sufficient**, and the second fault is now
characterised. TRITON_MLA completes 256 sequences at `max_model_len` 16896
standalone and completes the exact 45-page width that failed, yet the full bench
run still dies with an illegal access -- reported at
`mla_attention.py:1735`, which is `compute_num_computed_tokens().cpu()`, a D2H
sync. CUDA errors surface at the next synchronisation, so that line is where the
fault is *noticed*, not where it happens; this is also why changing the decode
backend did not help.

Re-running the same configuration under `CUDA_LAUNCH_BLOCKING=1` **completes**. A
fault that vanishes when launches are serialised is a race -- a kernel reading a
buffer another stream is still writing, or a host buffer reused before its copy
retires -- not a shape or indexing error. The original traceback is in
`Worker_TP0`, so it is on the reference's side; our engine runs to completion in
every configuration tried, with and without blocking.

**Conclusion for this row: vLLM 0.18 cannot validly serve Kimi-Linear on
Blackwell.** Its default MLA backend trips a block-table parity assert, CUTLASS_MLA
asserts the same thing, FLASHMLA is Hopper-only, and the one remaining backend
carries an async race. Neither number from the blocking run is usable either:

| scenario | ours | vLLM | "speedup" | align |
|---|---|---|---|---|
| prefill-heavy | 3440 | 1037 | 3.317 | 5.1/525 |
| balanced | 4290 | 1719 | 2.495 | 4.2/519 |
| decode-heavy | 5029 | 2016 | 2.495 | 7.8/1047 |

The 2.769x mean against a 1.20x paper figure is an artefact: serialising launches
costs vLLM's async scheduler far more than it costs our synchronous loop. And the
alignment cannot be read either, because the side being compared against is the one
with the race. The row needs the upstream race fixed, or a newer vLLM, before it can
be measured.

Anyone reading a speedup off this row should note TRITON_MLA is not the backend
vLLM would choose and is slower than the TRTLLM-gen path, so the comparison
flatters us. Treat the alignment as the meaningful number.

### Blackwell forced FA2 in every compiled region

`_fa_backend.TRACEABLE_FA_AVAILABLE` was `FA_VERSION == 3`, so on B200 every
torch.compile region fell back to upstream FA2 -- because FA4's entry point is a
CuTe-DSL `@cute.jit` function that reads `current_stream().cuda_stream`, which
dynamo proxies without that attribute. In an nsys trace of the BGE-M3 encoder with
both engines in one capture, FA2's generic `flash_fwd_kernel` averaged 857 us per
call against 389 us for the sm100 kernel vLLM itself uses, on the same shapes, at
39% of total GPU time.

Dynamo does not need to trace *into* the kernel, only to know its shape
behaviour, so the call is now a custom op with a fake implementation: opaque to
dynamo, and FA4 becomes usable from compiled code. Verified cos 0.999997 against
SDPA and bit-identical output under `torch.compile(fullgraph=True)`.

**But it is not a free win, and the magnitudes are not what single runs
suggested.** FA4 only pays off once the kernel is long enough to amortize its
per-call dispatch cost. With repeats (1-wide, dedicated GPU):

| row | FA2 (opaque off) | FA4 (opaque on) |
|---|---|---|
| ColBERTv2 (~180-token passages) | median **2.086** (1.474-2.523) | median **1.297** (1.256-1.329) |
| BGE-M3 (8192-token documents) | median **0.763** (0.626-0.901) | median **0.549** (0.513-0.586) |

FA4 loses on **both** rows, including the long-document one it was supposed to
help, and on BGE-M3 the two ranges do not overlap -- so this is not noise. The
single 0.978 reading that motivated the change was an outlier; the nsys per-call
figures (857us vs 389us) never translated into end-to-end throughput.

**Why it lost, and what that implies.** `infra/embedding_engine.py` compiles the
encoder with `torch.compile(mode="reduce-overhead", dynamic=True)`, and
`reduce-overhead` means CUDA graphs. Dropping an opaque custom op into a
graph-captured region prevents capture, so the region falls back to per-call eager
execution: the 2.2x faster attention kernel is bought at the cost of graph replay
for the whole encoder, which is a net loss. That is consistent with the measured
0.763 -> 0.549 on BGE-M3 and with FA4 being the *tighter* of the two arms (an eager
path has less run-to-run variance than one whose graph capture is marginal).

So "make FA4 opaque" cannot be the answer for any row whose region is
graph-captured. Getting the sm100 kernel there needs FA4 to be genuinely
dynamo-traceable -- i.e. its CuTe-DSL entry point not reading
`current_stream().cuda_stream` during tracing -- which is the upstream limitation
`TRACEABLE_FA_AVAILABLE` was created to work around in the first place.

The path is therefore **opt-in** (`FASTKERNELS_FA_OPAQUE=1`, with
`FA_OPAQUE_MIN_SEQ` bounding it by sequence length). The mechanism is kept because
it is correct and verified bit-identical under `fullgraph=True`, and because
`TRACEABLE_FA_AVAILABLE` genuinely does force FA2 in every compiled region on
Blackwell -- but nothing enables it until a row is measured to win. Note it is the
FA2 path that is variable here and FA4 that is tight, the opposite of the obvious
guess.

SigLIP-2 and SwinV2 return byte-identical correctness figures under the change,
which proves they never reach this op; their deficits (0.82x and 0.95x, 1-wide)
are elsewhere and still open.

### RTDetrV2 is elementwise-bound, not attention-bound

Profiling the whole run (both engines in one capture) shows attention is nowhere
near the bottleneck:

| share | kernel |
|---|---|
| 20.9% | `at::native::elementwise_kernel<128,4,...>` |
| 20.0% | `at::native::elementwise_kernel<128,4,...>` (second instantiation) |
| 5.7% | `vectorized_elementwise_kernel<8,...>` |
| 5.1% | `cutlass3x_sm100_tensorop_...implicit_gemm_fprop_f16` |
| 5.0% | `vectorized_elementwise_kernel<8,...>` |
| 4.7% | `batch_norm_transform_input_kernel<Half>` |
| 4.5% | `cudnn::...nchwToNhwcKernel<__half>` |

Roughly 55% of GPU time is elementwise, with another 4.5% spent purely converting
NCHW to NHWC. That is a ResNet-101 backbone dominated by small ops and layout
churn, which is why the FA4 attention change moved this row not at all (its
apparent 0.896 -> 0.980 was inside its own run-to-run spread). Closing this row
needs operator fusion and a consistent memory format, not a better attention
kernel.

### SwinV2 is elementwise-bound too

Same picture as RTDetrV2 from a whole-run both-engine capture:

| share | kernel |
|---|---|
| 17.7% | `elementwise_kernel<128,4>` |
| 13.3% | `reduce_kernel<512,1>` |
| 10.7% | `elementwise_kernel<128,4>` |
| 9.3% | `elementwise_kernel<128,4>` |
| 7.8% | `softmax_warp_forward<BFloat16>` |
| 5.5% | `elementwise_kernel<128,4>` |
| 4.2% + 3.8% | `vectorized_layer_norm_kernel` (float, then bf16) |
| 3.3% | GELU `vectorized_elementwise_kernel<8>` |

Over 60% is elementwise / reduce / layernorm and **no GEMM appears in the top
nine**. That is inherent to SwinV2's cosine attention: L2-normalize (a reduce), an
explicit attention matrix, the CPB bias add, then softmax -- there is no fused
flash path to switch to, which is also why timm sits so close to us (0.952x
1-wide). Reaching the 1.17x paper figure means fusing those elementwise chains,
not choosing a better attention kernel.

Taken with RTDetrV2, three of the four "throughput" rows in this group are
fusion problems rather than attention problems, and the attention work done for
BGE-M3 could never have moved them -- consistent with SigLIP-2 and SwinV2
returning byte-identical correctness under the FA4 change.

### SigLIP-2 is attention-heavy, but on the same kernel as its reference

Unlike RTDetrV2 and SwinV2, attention does dominate here -- it is just not a
kernel we can trade up:

| share | kernel |
|---|---|
| 30.5% | `pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<...>>` (54,916 calls, 274us) |
| 11.5% + 11.0% + 7.1% | `nvjet_tst_*` sm100 GEMMs with bias |
| 7.7% | GELU `vectorized_elementwise_kernel<8>` |
| 6.5% | `vectorized_layer_norm_kernel<float>` |
| 6.4% | `unrolled_elementwise_kernel` direct copy |

The attention kernel is in the `pytorch_flash` namespace -- PyTorch's own SDPA
flash backend, an FA2-generation kernel -- and timm reaches it through the same
`F.scaled_dot_product_attention`, so both engines run it. The GEMMs are already
sm100 (`nvjet_*`). So there is no attention-backend divergence behind the 0.820x,
and the remaining candidates are the ~20% in elementwise/layernorm/copies and
naflex's variable-resolution packing overhead.

Note `dense_attention.py` documents that `sdpa_sm100_flash_*` kernels are ~2.7x
faster than cutlass FMHA and that cuDNN needs an explicit
`torch.nn.attention.sdpa_kernel` context to be selected -- worth testing here,
but only as a change that would also move the reference, since both sides share
the path.

### SwinV2's real defect: an fp32 LayerNorm the reference does not run

SwinV2 reproduces at 1.17x on H200, so "elementwise-bound, needs fusion" described
the workload but could not explain a B200-only gap. Subtracting an ours-only nsys
capture from a both-engine capture of the same run found the actual difference --
two kernels we run that timm never does:

| kernel | ours | timm |
|---|---|---|
| `vectorized_layer_norm_kernel<float, float>` | 1690.8 ms, 29,256 calls | 0 ms, 0 calls |
| `unrolled_elementwise_kernel<direct_copy>` | 1012.6 ms, 87,768 calls | ~0, 0 calls |

2.70 s of our 8.92 s -- 30% of GPU time -- while every GEMM and reduce matched
within 1%. The cause is `LayerNorm(promote_fp32=True)`, our default: it upcasts
x/weight/bias to fp32 for the reduction, and the 87,768 `direct_copy` calls are
those conversions. timm runs LayerNorm natively in the model dtype.

That default is load-bearing elsewhere -- DeepSeek-V3.2's indexer `k_norm`, where a
bf16 reduction biases the variance enough to shift the FP8-quantized indexer K cache
and change top-2048 selection in every sparse layer -- so it stays global and only
SwinV2's five norm sites opt out. High-res goes 0.945 -> 1.007 / 1.002 across two
runs with the reference stable to 0.1%; default-res is too noisy to call
(2.44-2.79 s on a ~2.5 s scenario). Correctness is unchanged at cos exactly
1.000000, the expected direction since we now match the reference's dtype rather
than computing in higher precision than it.

Not enough for 1.17x. The remainder is inherent to cosine attention (L2-normalize
is a reduce, the attention matrix is explicit, the CPB bias is an add, then softmax)
and needs fusion in the model code. Whole-model `torch.compile` is not the route:
dynamic shapes hang in sympy (`pow_by_n`, killed at 1611 s) and static shapes
recompile on the trailing partial batch (high-res 0.947 -> 0.525).

RTDetrV2 shows the same fp32-LayerNorm signature -- `<float>` 20,257 calls for us
against `<c10::Half>` 20,184 for transformers -- but the delta is 78 ms, 0.7% of its
time, so it is not worth changing its numerics for. That row's cost really is the
~55% elementwise elsewhere.

### DeepSeek-V3.2: two B200-only bugs stood between it and a first measurement

The row had never produced a number. Two distinct faults, both Blackwell-specific:

1. **FP8 weight postprocess.** `postprocess_fp8_weights_batched` ended in
   `scale_inv[...].copy_(scale_transformed)`, but on Blackwell
   `transform_sf_into_required_layout` packs four UE8M0 exponents per int32, so it
   returns int32 `[E, ., 14]` where `scale_inv` is float32 `[E, ., 56]` -- an
   in-place copy cannot absorb a shape *and* dtype change. It died with "The size of
   tensor a (56) must match the size of tensor b (14)". The 2D path for ordinary
   linear layers already *returned* its transformed scale for the caller to rebind;
   only the 3D MoE path copied in place. Fixed by returning it and rebinding in
   `weight_loader`. Invisible on H200 because UE8M0 is off for sm90 by default
   (vLLM enables it for sm100, and for sm90 only under
   `VLLM_USE_DEEP_GEMM_E8M0_HOPPER`), which makes the transform shape-preserving
   there.

2. **TRTLLM workspace.** `engine.py` called `set_trtllm_workspace` on every module
   with `k_cache`/`v_cache`. `MLAAttention` matches that test but has no such
   setter, and correctly so: its Blackwell decode path allocates and caches its own
   128 MiB buffer in `flashinfer_mla_decode._workspace`, and the only "workspace" it
   exposes is the unrelated BF16 gather buffer for sparse prefill. Guarded with
   `hasattr`, which is what `jamba_engine.py` already did for the same call -- so
   this was two copies of one piece of logic drifting apart.

### DeepSeek-V3.2: what the remaining alignment gap is *not*

With the three load-path faults fixed the row reproduces at 0.624x (paper 0.84) but
aligns at only 86.1 avg matched tokens against a paper value of 294.1, 8/192 exact.

Two candidates tested and eliminated:

- **The DeepGEMM SF layout transform** *was* a real fault (see above) and fixing it
  moved alignment 0.6 -> 86.1. Worth recording how it presented: with the packed
  scales the row ran *faster* -- 0.724x against 0.624x -- because the Triton kernel
  was reading a smaller, wrong tensor. A throughput-only check would have called
  that a success.
- **The UE8M0 weight requant.** The obvious follow-on theory was that requantizing
  block scales to powers of two is also a DeepGEMM-only requirement that costs the
  Triton consumer precision. It does not: skipping it entirely gives 85.0 avg matched
  tokens against 86.1, speedup 0.629 against 0.624, and identical 8/192 exact. The
  requant is numerically irrelevant on this path, so the switch was reverted rather
  than left as a dead env var.

The indexer was then compared against vLLM's line by line, and it matches:
`k_norm` is `LayerNorm(head_dim, eps=1e-6)` on the fp32-promoting default on both
sides; the indexer K cache passes `"ue8m0"` (`indexer_k_cache.py:46`) against vLLM's
`scale_fmt = "ue8m0"`; the query-side `PerTokenGroupQuantFp8` quantizes with UE8M0
scales; `topk_tokens` is `config.index_topk` on both; and `softmax_scale` is
`head_dim ** -0.5` on both. So the static comparison of the indexer is clean too.

The remaining divergence is therefore not any of: the MoE scale layout, the UE8M0
weight requant, or the indexer's norm/quant/top-k configuration. It needs the same
treatment as Qwen3-Next -- runtime comparison of per-layer intermediates against the
reference for a single prompt, finding the first tensor that differs. The indexer's
selected indices are the natural first thing to dump, since: `LayerNorm`'s
`promote_fp32` default exists precisely because a bf16 reduction in the indexer
`k_norm` biases the variance enough to shift the FP8-quantized indexer K cache and
change the top-2048 selection in every sparse layer. An indexer that selects even
slightly different keys produces exactly this signature -- output that is coherent
and roughly the right length but diverges early. Comparing the indexer's selected
indices against vLLM's for a single prompt is the next step, not another sweep of
the MoE quantization path.

### This bench family is too noisy for single-run A/Bs

BGE-M3 run through three code paths that are *identical* for it (8192-token
documents clear every threshold, so all three route to FA4) returned 0.978 /
0.326 / 0.612 -- our throughput swinging 120k-373k input tok/s while the reference
held within a few percent (~370k). Not thermal: all GPUs idle at 120 MHz, 26-29 C,
throttle reason `GpuIdle`, no thermal slowdown. Not workload: the reference is
stable across the same runs.

The cause is warmup coverage. `tests/bench_embedding.py` warmed up on 4 requests
and then timed 1000, and Triton autotune and FA4's CuTe-DSL JIT key on shapes
including batch extent -- so the first wide batch paid its compile cost inside the
timed region, by an amount depending on cache state. Warmup is now 64 requests
(`FASTKERNELS_EMBED_WARMUP`), applied identically to both engines.

Anything measured on this bench before that change, including BGE-M3's headline
0.640x -> 0.978x, is a single-run figure and should not be quoted as a magnitude.

### Mamba2: we are already at the decode kernel's ceiling

The nsys trace is unambiguous -- `_selective_scan_update_kernel` is **89.4%** of
GPU time (51.9 s over 56,407 calls, avg 920 us; 56,406 = 64 layers x ~881 steps).
Both engines import the identical kernel from
`vllm.model_executor.layers.mamba`, and our SSM state is bf16 as vLLM's is
(`mamba_ssm_cache_dtype` defaults to `auto` -> model dtype), so neither kernel
quality nor bytes moved differ.

Microbenchmarking that kernel at Codestral's shapes (128 heads, head_dim 64,
dstate 128) shows near-perfectly linear scaling:

| batch | us/call | us/seq |
|---|---|---|
| 32 | 87.4 | 2.730 |
| 352 | 876.7 | 2.491 |
| 704 | 1745.9 | 2.480 |

Two conclusions. The nsys average (920 us) matches the microbenchmark at batch
~352 (877 us), so the traced batch was ~352 and the kernel is behaving normally.
And because cost is essentially all per-sequence with no fixed component, 64
layers x 2.5 us/seq is a hard ceiling near 6,250 tokens/s *independent of batch
size* -- so larger decode batches cannot help, and the earlier scheduling and
graph-coverage theories are all ruled out. The kernel reaches only ~21% of B200's
peak bandwidth. Whatever vLLM does to beat this, it is not bound by this kernel
the same way; that is where to look next.

### Mamba2: correcting the ceiling claim, and what the paired trace actually shows

An earlier note here concluded from a microbenchmark that we sit at the decode
kernel's hard ceiling. The measured throughputs refute that conclusion:

| | prefill-heavy | balanced | decode-heavy |
|---|---|---|---|
| ours | 4,744 | 5,018 | 5,307 tok/s |
| vLLM | 8,302 | 10,269 | 11,335 tok/s |

`selective_state_update` at Codestral's shapes (128 heads, head_dim 64, dstate
128, bf16) costs ~2.5 us per sequence per layer with no measurable fixed
component -- 2.730 us/seq at batch 32, 2.491 at 352, 2.480 at 704. At 64 layers
that implies ~6,250 tok/s at *any* batch size, which we are just under and vLLM
is well over. So either the kernel is not what bounds vLLM, or the model is wrong.

Subtracting an ours-only capture from a both-engine capture of the identical
64-sequence run (instances):

| kernel | both | ours | vLLM |
|---|---|---|---|
| `_chunk_scan_fwd` | 51,822 | 3,993 | 47,829 |
| `_chunk_state_fwd` | 37,728 | 5,937 | 31,791 |
| `_selective_scan_update` | 5,440 | 1,408 | 4,032 |
| `_state_passing_fwd` | 24,960 | 5,997 | 18,963 |
| `_bmm_chunk_fwd` | 31,359 | 10,614 | 20,745 |

One thing follows safely: vLLM **does** use `selective_state_update`, so the theory
that it avoids that kernel is wrong.

**The instance counts themselves are not usable, and neither is any conclusion
drawn from them.** At 64 sequences the decode-heavy scenario runs ~1,237 steps over
64 layers, so a per-step-per-layer call pattern would emit ~79,000 events; the
trace shows 1,408. Kernels replayed inside a CUDA graph are not emitted as
individual events, and this decode path is graph-captured, so the counts (and hence
the apparent "12x more chunk_scan calls" and "only ~22 steps") measure capture
behaviour rather than work done. Per-kernel time shares from these captures are
suspect for the same reason.

The step profiler reports 98.4% of the decode loop in a counter labelled
`gpu+d2h_wait`, against 0.7% dispatch, 0.1% prep, 0.8% finalize. That counter merges
GPU execution with the D2H wait, so it cannot distinguish a GPU-bound loop from an
idle one -- a genuinely GPU-bound workload reports the same thing. It is not
evidence either way.

**Unresolved.** The one measurement independent of graph capture is the
microbenchmark: ~2.5 us/seq/layer -> ~6,250 tok/s at 64 layers, which we sit just
under (5,307) and vLLM exceeds (11,335). Exceeding it is arithmetically impossible
for a per-step call at any batch size, so vLLM's decode must differ structurally in
a way not yet identified. The state gather is not it -- 826 us with a compact state
and no indices, 877 us with arange, 877 us with scattered indices, so indexing
costs 6%. Note also that at 64 sequences both engines fall well below the ceiling
(2,887 and 3,578 tok/s), so concurrency decides which regime each is in. What *is* established: the kernels are
identical imports on both sides, our SSM state is bf16 as vLLM's is, the kernel
scales linearly with batch, and graph coverage / capture size / batch size are all
ruled out by direct experiment. The next step is an NVTX-annotated capture that
separates per-step host stalls from kernel time, rather than more kernel-summary
comparisons.

### Mamba2: the paper figure is unreachable with a per-step state update

Resolved as far as the arithmetic allows, even though vLLM's mechanism is still
unidentified. The numbers are verified from the raw fields (identical output-token
counts on both sides, 1000 seqs):

| scenario | ours | vLLM |
|---|---|---|
| prefill-heavy | 623,531 tok / 131.45 s = 4,744 | 623,531 tok / 75.10 s = 8,302 |
| balanced | 638,330 tok / 127.21 s = 5,018 | 638,330 tok / 62.16 s = 10,269 |
| decode-heavy | 1,247,834 tok / 235.13 s = 5,307 | 1,247,834 tok / 110.08 s = 11,335 |

`selective_state_update` at Codestral's shapes is *exactly* linear in batch and
achieves only ~1.8 TB/s of B200's ~8 TB/s:

| batch | us/call | us/seq | TB/s (state r+w) |
|---|---|---|---|
| 352 | 829.5 | 2.357 | 1.78 |
| 704 | 1648.2 | 2.341 | 1.79 |
| 1024 | 2391.8 | 2.336 | 1.80 |
| 2048 | 4772.7 | 2.330 | 1.80 |

Because the cost is all per-sequence with no fixed component, one call per layer
per step costs 64 x 2.33 us = 149 us **per token at any batch size**, i.e. a
ceiling of ~6,700 tok/s. We are at 5,307, so ~26% of headroom remains inside that
model -- but spending all of it reaches 0.59x, not 0.97x. vLLM's 11,335 tok/s is
88 us/token, 1.7x below the floor.

**Therefore the paper's 0.97x is not reachable by tuning this path.** Whatever vLLM
does, it is not one `selective_state_update` per layer per decode step, and
matching it requires a different decode strategy rather than a faster call. Ruled
out along the way, each by direct measurement: batch size and concurrency (linear
to 2048), the state gather (826 us compact vs 877 us arange vs 877 us scattered, so
6%), state dtype (bf16 on both sides -- vLLM's `mamba_ssm_cache_dtype` defaults to
`auto` -> model dtype), state shape and layout (vLLM's
`mamba2_state_shape` returns the same `(num_heads/tp, head_dim, state_size)` =
(128, 64, 128)), the kernels themselves (identical imports from
`vllm.model_executor.layers.mamba`), CUDA-graph coverage and capture size, and page
size.

Note also that both engines fall well below the ceiling at 64 sequences (2,887 and
3,578 tok/s), so at low concurrency both are host-bound and the ceiling only binds
at scale.

### Qwen3-Next: nine hypotheses refuted, divergence isolated to the GDN path

Alignment is ~26-30 avg matched tokens against a 487.4 paper value, with 0/6000
exact. What it is *not*:

1. The sm90 FlashInfer GDN gate -- vLLM gates identically
   (`is_device_capability(90)`), so both fall back to Triton/FLA on B200.
2. The kernel -- our fallback calls vLLM's own `chunk_gated_delta_rule` and
   `fused_recurrent_gated_delta_rule`.
3. The MoE backend -- both Triton.
4. The prefill token budget. Our 32768 vs vLLM's 16384 looked decisive, and the
   first dismissal of it was wrong (the paper's H200 run used 32768, but H200 uses
   a *different* GDN kernel, so that argument does not transfer). Tested properly
   on B200: alignment is byte-identical at both budgets (29/37/24), and matching
   vLLM's costs 10% throughput.
5. Page size -- 16 vs 64 gives 30.0 vs 30.1.
6. The full-attention backend -- `FASTKERNELS_FORCE_NHD=1` gives *identical*
   per-scenario alignment, so those layers contribute nothing to the divergence.
7. Preemption destroying recurrent state -- zero preemptions in any run.
8. The GDN recurrent state dtype -- ours is bf16 and so is vLLM's.
9. Batch composition. Small-n runs looked like a recovery (61 at n=1, 63 at n=8)
   but that is a sample size of one to eight, with n=8's mean carried by a single
   scenario reading 142; at n=32 it is 30 and at n=1000 it is 26. There was no
   recovery.

Three further eliminations from a direct read of both implementations:

10. **Decode-path beta precision.** vLLM computes gating separately only when
    `num_prefills > 0` and otherwise feeds raw A_log/a/b/dt_bias into
    `fused_sigmoid_gating_delta_rule_update`, deriving g and beta inside the kernel,
    where we materialized them and wrote beta at `b.dtype` (bf16). We now use the
    same fused kernel. Alignment 30.1 -> 30.3, i.e. no effect -- consistent with the
    divergence being in prefill.
11. **The prefill gating kernel.** Our `_fused_gdn_gating` is byte-for-byte
    equivalent to vLLM's `fused_gdn_gating`: same grid, same `g` fp32 /
    `beta_output` `b.dtype`, same `beta=1.0`, `threshold=20.0`, `num_warps=1`.
12. **qkvz channel ordering into the depthwise conv.** vLLM does project ->
    `fix_query_key_value_ordering` (de-interleave the head-grouped checkpoint
    layout) -> rearrange -> `cat(q,k,v)` -> `causal_conv1d` -> flat split. Ours does
    `_unpack_qkvz` -> `cat` -> `causal_conv1d` -> flat split, in that order, so the
    conv sees the same channel order. A mismatch here would have applied every
    depthwise filter to the wrong channel, which is why it was worth checking.

Also worth recording: **the TP-sharding hypothesis cannot be tested by varying TP on
this model.** Qwen3-Next has 2 KV heads, so TP=4 is rejected outright
("num_key_value_heads must be divisible by tensor_parallel_size"), and TP=1 OOMs --
80B in bf16 is ~160 GiB against 178 GiB before activations. TP=2 is the only
runnable configuration.

### Qwen3-Next: the divergence starts in layer 0

`tests/debug/layer_probe.py` hooks every decoder layer, runs one prompt through each
engine, and reports the first layer whose hidden states differ. Validated on
Llama-3.1-8B first, where all 32 layers come back **bit-identical** (cos 1.000000,
max_abs 0.0) -- so the tool is sound and the noise floor is zero, not 1e-3.

On Qwen3-Next at TP=2, same 13-token prompt, both captures shaped (13, 2048):

| layer | cos | max_abs | rel_l2 |
|---|---|---|---|
| 0 | 0.99150134 | 0.0205 | 0.13011 |
| 1 | 0.97728605 | 0.1147 | 0.22255 |
| 4 | 0.96611801 | 0.0532 | 0.26499 |
| 16 | 0.93114564 | 0.0850 | 0.37906 |
| 47 | 0.99017344 | 5.1875 | 0.13995 |

**Layer 0 already differs**, at 13% relative error. This is not accumulation from a
deeper layer, and relative error then stays in a 0.13-0.38 band across all 48 layers
rather than growing without bound.

`full_attention_interval: 4`, so layers 0-2 are linear-attention (GDN) and layer 3 is
full attention -- layer 0 is a GDN layer. That closes the loop on the earlier
elimination work: it is consistent with `FASTKERNELS_FORCE_NHD=1` producing
*identical* per-scenario alignment, because the full-attention layers were never
implicated.

The next step is a sub-layer probe inside layer 0 -- hooking the conv output, the
gating outputs and the core-attention output separately -- to find which sub-op of a
single GDN block differs. Everything static in that block already matches the
reference (see the eliminations below), so the difference is numerical and now has a
13-token, single-layer, fully deterministic reproduction to work against, which is a
far cheaper thing to bisect than a 1000-sequence benchmark.

Divergence appears at token index 0 for 12-18% of sequences, so it starts in
prefill, and it persists deterministically at n=1. Static comparison of this path is
now exhausted: projections, de-interleave order, conv kernel and channel order,
gating (both paths), the chunk and recurrent kernels, their flags, the state dtype
and layout, and the output norm all match the reference. What remains is runtime
tensor comparison -- instrumenting both engines to dump per-layer intermediates for
one prompt and finding the first that differs. That is engineering, not another
config sweep, and it is where the next person should start.

### Qwen3-Next: found -- `in_proj_ba` was sharded with the wrong loader

The sub-layer probe above (`--sublayer 0`) settled it. Hooking each submodule of
layer 0 separately, on the same 13-token prompt at TP=2:

| unit | cos | rel_l2 |
|---|---|---|
| `input_layernorm` | 1.00000000 | 0.0 |
| `linear_attn.in_proj_qkvz` | 1.00000000 | 0.0 |
| **`linear_attn.in_proj_ba`** | **0.63600000** | **0.79500** |

The input was bit-identical and the *sibling* projection reading that same input was
bit-identical, so this could only be the weight. `in_proj_ba` had a
midpoint-splitting weight loader, which is the **Qwen3.5** layout for checkpoints
shipping separate `in_proj_b` and `in_proj_a` tensors. Qwen3-Next ships one fused
weight interleaved per K-head group, `[b_g0, a_g0, b_g1, a_g1, ...]`; vLLM builds it
as a single output shard for exactly that reason. Splitting at the midpoint permutes
rows across a b/a boundary that does not exist in this checkpoint.

Our own `_unpack_ba` already read the interleaved layout, so the forward pass was
right and only the loader was wrong -- the two disagreed with each other. `b` and `a`
feed the gating that produces `g` (decay) and `beta` (learning rate), so a scrambled
weight corrupts the state update in *every* GDN layer, on every step. Fix: plain
contiguous `ColumnParallelLinear`, which hands each rank whole K-groups and preserves
the interleaving (`65a930d`).

Measured on the 32-sequence bench, same seed and workload:

| | alignment (min/med/max) | exact | speedup |
|---|---|---|---|
| midpoint-split loader | 30.1 (29/37/24) | 0/96 | 0.841x |
| contiguous shard | **110.2** (77/184/70) | 1/96 | 0.845x |

**3.7x better alignment at identical throughput**, and past the 50-60 token target.

Post-fix, the 48-layer probe puts layer 0 at rel_l2 0.0042 (from 0.130) and layers
1-47 fluctuate inside a 0.99-0.9999 cos band without ever growing -- drift, not a
structural defect. Within layer 0 every submodule is now bit-identical except the
`mlp` output at cos 0.99999134, which is *exactly* layer 0's whole-layer cos. So the
residual drift enters through the fused-MoE grouped GEMM, not the GDN path that all
the earlier work was aimed at.

**Retracted:** an intermediate reading of that probe called `mlp.shared_expert`
(cos 0.602, rel_l2 40.6) a second bug. It is not. The two engines put the sigmoid
gate on opposite sides of that module boundary, so the hook compares our pre-gate
output against vLLM's post-gate output -- non-corresponding quantities. The
enclosing `mlp` output agreeing to cos 0.99999134 is what settles it. A sub-module
comparison is only meaningful where both implementations draw the same boundary, and
the enclosing unit is the check on that.

### Mamba2: found -- vLLM asks `selective_state_update` for a Blackwell tile

Everything above is superseded. The kernel is shared, but the *launch config* is not.

`selective_state_update` deliberately does not autotune -- "We don't want autotune
since it will overwrite the state" -- so it picks `BLOCK_SIZE_M` and `num_warps` from
a hand-written table keyed on `dstate`. For `dstate > 64` there are two branches
(`vllm/model_executor/layers/mamba/ops/mamba_ssm.py:410`):

```
dstate > 64 and is_blackwell   -> BLOCK_SIZE_M=32, num_warps=8   # "Optimized for B200"
dstate > 64 and dstate <= 128  -> BLOCK_SIZE_M=4,  num_warps=4
```

Mamba-Codestral has `dstate=128`. vLLM's mixer passes
`is_blackwell=current_platform.is_device_capability_family(100)`
(`mamba_mixer2.py:510,890`), so the reference got the wide tile on B200 and we ran the
same kernel at an 8x smaller one. Measured directly at Codestral's shapes
(`ssu_tile_probe.py`):

| batch | default tile | Blackwell tile | speedup | implied ceiling (64 layers) |
|---|---|---|---|---|
| 32 | 78.1 us | 61.2 us | 1.28x | 6,405 -> 8,169 tok/s |
| 128 | 295.7 us | 112.6 us | 2.63x | 6,763 -> 17,763 tok/s |
| **352** | **803.1 us** | **293.9 us** | **2.73x** | **6,848 -> 18,715 tok/s** |
| 704 | 1599.3 us | 580.3 us | 2.76x | 6,878 -> 18,957 tok/s |

Batch 352 is the batch the production profile observed, and 803 us matches the 920 us
nsys average recorded above. **The ~6,850 tok/s "hard ceiling" derived earlier is real
-- but it is the ceiling of the narrow tile only**, which is why it had to be retracted
as inconsistent with vLLM's 11,335 tok/s. The reference was never bound by it.

End to end, 1000 sequences per scenario (`2c0b4bb`):

| scenario | before | after | vLLM | alignment before -> after |
|---|---|---|---|---|
| prefill-heavy | 0.571x (4,744) | **1.01x** (8,195) | 8,104 | 316 -> 317.4 / 624 |
| balanced | 0.489x (5,018) | **0.96x** (9,698) | 10,135 | 388 -> 382.1 / 638 |
| decode-heavy | 0.468x (5,307) | **0.96x** (10,812) | 11,218 | 667 -> 639.7 / 1248 |

Against a 0.97x paper target, so this row now matches. Alignment is unchanged: a tile
size changes scheduling, not arithmetic, and the two tiles agree to rel_l2 ~1e-5 /
cos 1.0 in bf16.

**Why this stayed hidden, and why H200 was fine.** Every earlier note verified that
both engines *import the identical kernel* and treated that as ruling out kernel
quality -- but the kernel takes a launch-config argument that vLLM was passing and we
were not. The microbenchmark "ceiling", the bandwidth-efficiency puzzle and the
paired-trace analysis were all built on timing *our* call while assuming the reference
made the same one. The H200/B200 asymmetry was the standing clue: a row that matches
the paper on sm90 and halves on sm100 points at something keyed on device capability,
and `is_device_capability_family(100)` is exactly that. It also means the fix carries
no H200 risk -- there the flag is False for both engines, both take `BLOCK_SIZE_M=4`,
and behaviour is identical. Only mamba2 needed it: `mamba_mixer.py` and
`jamba_mamba_mixer.py` are mamba1 with `dstate=16`, which hits the earlier
`dstate <= 16` branch where the flag is not read, and Mamba-Codestral is the only
model in the table built on `Mamba2Mixer`.

**Generalizable lesson:** "same kernel" is not "same call". When a shared kernel is
slower for us than for the reference, diff the *arguments* -- especially any that
select a tile, algorithm or backend -- before concluding anything about the kernel or
the hardware.

### RTDetrV2 and BGE-M3: one measurement artifact, one wrong checkpoint

**RETRACTED: the RTDetrV2 re-measurement below used the wrong model.** All four runs
in this table ran `PekingU/rtdetr_v2_r18vd`, copied from the usage example in
`tests/bench_detection.py`. The paper row, `kb_nano_models_unified.csv`, and all nine
of this repo's canonical RTDetrV2 job definitions use **`PekingU/rtdetr_v2_r101vd`** --
a far larger backbone. The numbers below therefore say nothing about the row they were
meant to correct, and the 0.972x on record (r101vd) stands until re-measured on the
right checkpoint. The `--use-fp16` flag also differs from the canonical jobs. The usage
example has been changed to name r101vd so this trap does not catch the next person.

| run (r18vd, NOT the paper's model) | ours img/s | reference img/s | ratio |
|---|---|---|---|
| a | 1086.4 | 962.2 | 1.13x |
| b | 914.3 | 689.1 | 1.33x |
| c | 1083.6 | 969.6 | 1.12x |
| d | 753.2 | 950.7 | 0.79x |

**Re-measured on r101vd** (serial, 3 repeats, correctness PASS at cos 1.000000):

| run | ours img/s | reference img/s | ratio |
|---|---|---|---|
| r1 | 383.9 | 399.6 | 0.96x |
| r2 | 412.8 | 410.2 | 1.01x |
| r3 | 364.5 | 391.2 | 0.93x |

Best-of-3 per side is **1.006x** and the median ratio **0.96x**, against a 1.08x paper
target -- so **RTDetrV2 is still behind**, by 7-13% depending on the reduction. Better
than the 0.906x on record, but the row is not cleared. Absolute throughput is 364-413
img/s here against 753-1086 for r18vd, which is the scale of the difference that made
the wrong-checkpoint numbers useless.

What does survive from the r18vd runs is checkpoint-independent: our eager detection path
is bit-identical to transformers (boxes cos 1.000000, MAE exactly 0.0), and the
opt-in `torch.compile` path trades that exactness away (boxes cos 0.931, labels 0.724
at 2.29x) because the correctness gate compares an argmax over 80 classes and a
score top-k over mostly-junk detections. Both statements were also measured on r18vd,
so the *magnitudes* need re-checking on r101vd even though the mechanism does not.

**BGE-M3 is at parity, not ahead.** This one used the right model. Serial repeats give
18.028 / 13.986 / 12.778 s on our side against a flat 12.312 / 12.725 / 12.719 s for
vLLM -- best-of-3 per side **0.964x** against a 1.06x paper target. The 1.194x
previously recorded here is not reproducible and should not be cited.

**The measurement lesson still stands.** A 0.79-1.33x spread across serial
runs of unchanged code means a single number from these benches decides nothing. Two
causes, neither fixable from inside the repo:

* Clocks are not actually pinned. `clocks.applications.graphics` is 1965 MHz with
  persistence on, but that is a boost *ceiling*, not a lock -- idle GPUs sit at 120
  MHz with throttle reason `0x1` (GpuIdle), and `nvidia-smi -lgc` needs root, which
  this account does not have. The short benches are the erratic ones (RTDetrV2's timed
  region is ~5 s, BGE-M3's ~17 s) while the 10-minute Mamba2 run is stable, which is
  the signature of clock ramp rather than anything in the code.
* Both benches copy each batch host-to-device *inside* the timed region (~12 GB per
  RTDetrV2 run), so host memory state enters the measurement. Both arms pay it
  equally, so it adds noise and compresses ratios toward 1.0 rather than biasing
  either side.

Practical rule for this host: **at least 3 serial repeats, reduced best-of-N per
side**, for any row whose timed region is under ~30 s. Contention only ever removes
throughput, so each side's maximum converges upward, and the two maxima need not come
from the same run. Parallel runs are fine for exploration but not for a reported
number -- and BGE-M3 shows why: three concurrent runs each write 17 GiB of output
tensors immediately after their timed region, landing squarely on their neighbours'
timed encodes.

### DeepSeek-V3.2: the sparse indexer selects identically at layer 0

The indexer was the leading suspect for the residual alignment gap (86.1 tokens against
the paper's 294.1): it picks the top 2048 tokens per query, so a different selection
changes attention outright, and no activation-level comparison can attribute that.
`tests/debug/layer_probe.py --modules .self_attn.indexer` dumps the selection from each
of the 61 layers on both engines, on a 5344-token prompt (it has to exceed 2048, or the
top-k is not a choice at all and both engines agree trivially). Selections are integer
index sets, so they are scored by per-row Jaccard overlap rather than cosine -- two
selections can agree perfectly as sets and still be ordered differently.

Rows below index 2048 are excluded: with fewer than 2048 candidates everything
available is selected. Over the remaining 3296 rows:

| layer | mean overlap | min | frac rows identical |
|---|---|---|---|
| **0** | **1.000000** | **1.0000** | **1.0000** |
| 1 | 0.994423 | 0.9778 | 0.0118 |
| 2 | 0.993364 | 0.9749 | 0.0076 |
| 5 | 0.987734 | 0.9702 | 0.0012 |
| 27 (worst) | 0.812022 | -- | -- |
| 59 | 0.885106 | 0.6856 | 0.0003 |
| 60 | 0.893034 | 0.6698 | 0.0006 |

Mean across all 61 layers: 0.888636. Both engines always select a full k=2048.

**Layer 0 is bit-identical, so the indexer is not the bug.** Its weights, FP8
quantization, RoPE and top-k all reproduce vLLM's selection exactly when handed the
same input. From layer 1 on the input already differs slightly and a fraction of a
percent of the selected tokens flip across the top-2048 boundary, decaying to ~11%
disagreement by the final layer. That makes the diverging selections a *consequence*
of upstream drift, not its cause -- though they are also an amplifier, since a
different token set changes the attention output rather than perturbing it.

The first divergence is therefore in layer 0's *output*, which is the same place the
Qwen3-Next hunt ended up, and the same sub-layer probe applies.

### Two harness bugs found while getting there

* **`_warmup_deepgemm` passed a mis-strided scale factor** (`fc64e53`). It sliced
  `linear_op._s_buf`, which is physically `(num_groups, max_tokens)` exposed as
  `(max_tokens, num_groups)`, so `[:num_tokens]` keeps `stride(1) == max_tokens` while
  DeepGEMM's SM100 check (`smxx_layout.hpp:201`) requires the stride to equal the
  actual M. It only validated when `num_tokens == max_tokens`. **Any block-FP8 model
  died at startup on B200 under default settings**; `tests/bench_vllm.py` sets
  `VLLM_DEEP_GEMM_WARMUP=skip`, which skips those GEMMs, so the bench never saw it.
  The runtime path was never affected -- `Fp8Linear.forward` allocates a fresh,
  correctly strided scale for the real M on every call.
* **The probe's module matching was too loose and OOM-killed both engines.** A
  substring match for `.indexer` also catches `.indexer_rope_emb` and every child
  (`.indexer.wq`, ...) -- 854 hooks on our side against vLLM's 429 -- and combined with
  61 layer activations at 153 MB each that is ~175 GB of writes. Both arms were killed,
  vLLM's while `np.savez_compressed` was building its archive, which is why a truncated
  `.npz` can appear next to a complete set of `.npy` files. Now matched by `endswith`,
  layer hooks are opt-in, dumps are mmapped, and `--no-npz` skips the archive so
  `--compare` reads the dump directories directly.

**Note on `VLLM_DEEP_GEMM_WARMUP=skip`.** It is set inside *both* worker templates
(`tests/bench_vllm.py:464` for vLLM, `:637` for ours), so it is symmetric by
construction rather than by env inheritance, and it skips only the DeepGEMM JIT
warmup -- CUDA-graph capture, the per-run generate warmup and FlashInfer autotune all
still happen. Two residual asymmetries are worth knowing before quoting fp8 numbers:
first-use JIT then lands *inside* the timed region for both arms, and since the arms
share DeepGEMM's on-disk cache and vLLM runs first, a cold cache makes vLLM pay
compilation that our arm then avoids (a bias toward us, invisible once the cache is
warm). Separately, vLLM runs FlashInfer autotune at startup and we have no equivalent,
which biases the other way. Running both arms at `VLLM_DEEP_GEMM_WARMUP=relax`
(vLLM's own default) moves warmup back out of the timed region for both; `fc64e53` is
a prerequisite, since before it any non-skip setting asserted on B200.

### Canonical-config re-runs of the rows fixed on 07-26

The coverage table reads `tests/results/B200/<model>_tp<N>/`, and the fixes committed on
07-26 were first measured into scratch output dirs, so the table reported them as
failures. Re-run with default settings (no `--output-dir`, n=1000 per scenario):

| row | table before | canonical after | paper | cleared? |
|---|---|---|---|---|
| Mamba | 0.928x | 0.99 / 1.06 / **1.16x** (mean 1.07) | 1.05x | yes |
| Mamba2 | 0.523x | 1.00 / 0.96 / 0.96x (mean **0.97**) | 0.97x | yes |
| RWKV-7 | 0.927x | 0.90 / 1.09 / 0.96x (mean **0.98**) | 1.18x | no |
| Gemma-4 | 0.724x | 0.84 / 0.92 / 0.88x (mean **0.88**) | 1.00x | no |

**Correction on Gemma-4.** It was reported earlier as fixed to 0.943x. That figure is
real but was measured at `--num-seqs 300` (0.947 / 0.926 / 0.963), a non-default
concurrency; at the canonical n=1000 the same code gives 0.88. The
`FASTKERNELS_TRITON_ATTN_MIN_HEAD_DIM=256` per-model default does apply in the
canonical run (the engine logs it), so this is concurrency sensitivity, not a lost fix.
**Quote canonical-config numbers for table rows** -- an exploration sweep at a
different `--num-seqs` is not comparable to the paper row.

**Gemma-4's reference needs its own environment.** The default env's transformers
cannot parse `model_type: gemma4` and the baseline arm dies with a pydantic
`ValidationError` that names transformers rather than the environment. It requires
`--vllm-python /home/yak/repro_venvs/vllm020/bin/python` (vLLM 0.20.1, transformers
5.8.0). `/home/yak/repro_venvs/` also holds `sglang`, `openpi`, `ptv3`, `dlrm` and `gs`,
so EAGLE-3, Pi0, PointTransformerV3, DLRMv2 and 3DGS have the same kind of dependency.
Reproducing the full table therefore has an infrastructure precondition that is not
discoverable from `kb_nano_models_unified.csv` or the bench defaults.

## 3. Notes

- `AttnBackendConfig.auto_detect()` already selected the right Blackwell
  backend (TRTLLM-gen via FlashInfer, HND, page 16); the defects were all in
  ops and engines that did not follow it, or that assumed "not Hopper" implies
  "no vLLM FlashAttention".
- MobileNetV4 is benchmarked by `tests/bench_timm.py`, not
  `tests/bench_image_cls.py` (whose loader only knows ConvNeXtV2 and
  EfficientNetV2).
