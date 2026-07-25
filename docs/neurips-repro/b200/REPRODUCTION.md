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

Each agrees with the paired A/Bs measured earlier, so the pass is self-consistent.
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

**Confirmed for RWKV-7, refuted for GLA.**

| | prefill-heavy | balanced | decode-heavy | mean |
|---|---|---|---|---|
| RWKV-7 @ 16384 (default) | 0.60x | 0.94x | 0.59x | 0.71x |
| RWKV-7 @ **65536** | **0.83x** | **1.00x** | **0.87x** | **0.90x** |
| RWKV-7 @ 131072 | 0.80x | 0.97x | 0.86x | 0.88x |
| GLA @ 16384 (default) | 0.95x | 1.03x | 1.18x | 1.05x |
| GLA @ 65536 | 0.97x | 1.04x | 1.17x | 1.06x |

RWKV-7's own throughput goes from 4,557 to 6,066 tok/s on prefill-heavy (+33%) and
5,376 to 8,245 on decode-heavy (+53%), taking the row from 0.60 to 0.76 of its paper
target. 65536 beats 131072, so the budget has an optimum rather than being
monotone. Matched tokens are **byte-identical** across all three settings
(140.4/552, 126.1/551, 157.9/1084), which is both a check that this is a pure
scheduling knob and a small confirmation that the recurrent rows are batch-invariant --
they page no KV cache, so the alignment story above does not touch them.

GLA does not move at all, so despite sharing the same reference and the same kernel its
shortfall (1.05x against 1.85x) has a different cause and is still open.

These numbers were taken while hunyuan and BitNet were still on the box; the effect is
far larger than contention noise, but the exact figures are queued for a quiet re-run.

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

## 3. Notes

- `AttnBackendConfig.auto_detect()` already selected the right Blackwell
  backend (TRTLLM-gen via FlashInfer, HND, page 16); the defects were all in
  ops and engines that did not follow it, or that assumed "not Hopper" implies
  "no vLLM FlashAttention".
- MobileNetV4 is benchmarked by `tests/bench_timm.py`, not
  `tests/bench_image_cls.py` (whose loader only knows ConvNeXtV2 and
  EfficientNetV2).
