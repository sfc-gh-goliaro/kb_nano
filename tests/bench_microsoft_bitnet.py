#!/usr/bin/env python3
"""Throughput + alignment benchmark for kb-nano BitNet b1.58 vs Microsoft
BitNet GPU lib.

The SOTA reference is the official Microsoft BitNet GPU implementation
(https://github.com/microsoft/BitNet/tree/main/gpu), which provides a
custom W2A8 (1.58-bit weight x int8 activation) CUDA kernel + CUDA-graph
batched generate path.  This is the only implementation that runs the
``microsoft/bitnet-b1.58-2B-4T`` model in its native quantized format on
GPU; the HuggingFace ``transformers`` integration falls back to bf16
matmul and is roughly 5-6x slower, so it is *not* a useful SOTA target.

Both engines run the same 3-scenario LLM workload from
``adding-arch-instructions.md``:

  * prefill-heavy (1024 prefill / 512 decode)
  * balanced     ( 512 prefill / 512 decode)
  * decode-heavy ( 512 prefill / 1024 decode)

with 1000 requests per scenario.  Requests are batched to fit on a
single H200 (kb-nano: full 1000-batch via paged scheduling; SOTA:
``--gen-bsz`` requests per CUDA-graph capture, default 32, looped).

Both engines run greedy decoding (temperature 0, ignore_eos) on the
**same** random token-id prompts and return per-request output token
ids.  The bench then computes per-scenario alignment statistics
(``avg_matching_tokens_per_request``, ``exact_matches``) using the same
code-path as ``bench_vllm.py``.

Setup (one-time):
-----------------
The Microsoft GPU kernel + checkpoint conversion must be done first::

    cd /path/to/microsoft/BitNet/gpu
    bash bitnet_kernels/compile.sh
    huggingface-cli download microsoft/bitnet-b1.58-2B-4T-bf16 \\
        --local-dir checkpoints/bitnet-b1.58-2B-4T-bf16
    python convert_safetensors.py \\
        --safetensors_file checkpoints/bitnet-b1.58-2B-4T-bf16/model.safetensors \\
        --output checkpoints/model_state.pt --model_name 2B
    python convert_checkpoint.py --input checkpoints/model_state.pt
    rm checkpoints/model_state.pt   # only the int2/fp16 splits are needed

Usage:
------
::

    # full benchmark (kb-nano + Microsoft BitNet GPU on all 3 scenarios,
    # 1000 reqs each) -- requires BITNET_REPO env or --bitnet-repo flag.
    BITNET_REPO=/path/to/microsoft/BitNet \\
        python tests/bench_microsoft_bitnet.py

    # smoke run
    python tests/bench_microsoft_bitnet.py --num-prompts 32 --skip-sota

    # kb-nano only
    python tests/bench_microsoft_bitnet.py --skip-sota
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path


_THIS_DIR = Path(__file__).resolve().parent
_PACKAGE_DIR = _THIS_DIR.parent
_PROJECT_ROOT = _PACKAGE_DIR.parent

MODEL_ID = "microsoft/bitnet-b1.58-2B-4T"

SCENARIOS = [
    {"name": "prefill-heavy", "input_len": 1024, "output_len": 512},
    {"name": "balanced",      "input_len": 512,  "output_len": 512},
    {"name": "decode-heavy",  "input_len": 512,  "output_len": 1024},
]


def _detect_gpu_name() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
        ).strip().splitlines()[0]
        for tag in ("B200", "B100", "H200", "H100", "A100", "A10G", "L40S", "L40", "L4"):
            if tag in out:
                return tag
        return out.split()[-1]
    except Exception:
        return "unknown"


def _build_random_token_prompts(num_prompts: int, input_len: int,
                                vocab_size: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    return [
        [rng.randint(2, vocab_size - 1) for _ in range(input_len)]
        for _ in range(num_prompts)
    ]


# ---------------------------------------------------------------------------
# kb-nano subprocess worker.
# Returns per-request output token ids in ``outputs`` so the parent can
# do per-scenario alignment against the SOTA reference.
# ---------------------------------------------------------------------------
KB_WORKER = r'''
import json, sys, time
import torch

with open(sys.argv[1]) as f:
    cfg = json.load(f)
sys.path.insert(0, cfg["project_root"])

# fastsafetensors GDS path is unreliable on some hosts; force the
# threaded safetensors loader so the bench focuses on inference perf.
from kb_nano.infra import weight_loader as _wl
_wl._HAS_FASTSAFETENSORS = False

from kb_nano.infra.engine import LlamaEngine, SamplingParams

engine = LlamaEngine(
    model_name=cfg["model"],
    seed=cfg["seed"],
    enforce_eager=cfg.get("enforce_eager", True),
    tensor_parallel_size=cfg["tp"],
    max_model_len=cfg["max_model_len"],
)

# Warmup
engine.generate([[0] * 16], SamplingParams(temperature=0.0, max_tokens=16))

results = []
for sc in cfg["scenarios"]:
    prompts = sc["prompt_token_ids"]
    out_lens = sc["output_lens"]
    sps = [
        SamplingParams(temperature=0.0, max_tokens=ol, ignore_eos=True)
        for ol in out_lens
    ]
    engine.block_manager.reset()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outs = engine.generate(prompts, sps, use_tqdm=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    n_in = sum(len(p) for p in prompts)
    n_out = sum(len(o.token_ids) for o in outs)
    out_records = [{"token_ids": list(o.token_ids)} for o in outs]
    results.append({
        "name": sc["name"], "elapsed": elapsed,
        "total_input_tokens": n_in, "total_output_tokens": n_out,
        "num_prompts": len(prompts),
        "outputs": out_records,
    })
    print(f"[kb] {sc['name']:>14}: {elapsed:7.2f}s  "
          f"in={n_in:>8d}  out={n_out:>8d}  "
          f"throughput={(n_in + n_out)/elapsed:>8.1f} tok/s",
          flush=True)

with open(cfg["output_file"], "w") as f:
    json.dump({
        "throughput": results,
        "memory_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2),
    }, f)
'''


# ---------------------------------------------------------------------------
# Microsoft BitNet GPU subprocess worker (official SOTA).
#
# The official lib hard-pins (gen_bsz, prompt_length, gen_length) at
# CUDA-graph capture time, so we re-build a FastGen instance per
# scenario and loop ``ceil(num_prompts / gen_bsz)`` graph replays per
# scenario.  Build cost (compile_prefill + compile_generate) is excluded
# from the timed window; only ``generate_all`` is timed.  Per-prompt
# outputs are captured into ``outputs`` for alignment scoring.
#
# Requires the local copy of ``vllm_repo/BitNet/gpu/generate.py`` to
# have the ``output[kv_seqlen - 1, :]`` flat-index bug fixed (see
# diff applied alongside this bench).  Without that fix all batched
# prompts decode prompt 0's logit and outputs are useless for
# per-prompt alignment.
# ---------------------------------------------------------------------------
SOTA_WORKER = r'''
import json, math, os, sys, time
import torch

with open(sys.argv[1]) as f:
    cfg = json.load(f)

bitnet_repo = cfg["bitnet_repo"]
gpu_dir = os.path.join(bitnet_repo, "gpu")
sys.path.insert(0, gpu_dir)
# the official model.py loads ./bitnet_kernels/libbitnet.so via a
# relative path, so we have to chdir into ./gpu/ before importing it.
os.chdir(gpu_dir)

import generate as _bitnet_generate
import model as _bitnet_model

ckpt_dir = cfg["ckpt_dir"]
gen_bsz = int(cfg["gen_bsz"])
torch.cuda.set_device(0)

results = []
for sc in cfg["scenarios"]:
    prompts = sc["prompt_token_ids"]
    out_lens = sc["output_lens"]
    assert all(ol == out_lens[0] for ol in out_lens), \
        "Microsoft BitNet GPU worker requires uniform output length per scenario"
    in_len = len(prompts[0])
    assert all(len(p) == in_len for p in prompts), \
        "Microsoft BitNet GPU worker requires uniform input length per scenario"
    out_len = out_lens[0]

    print(f"[sota] building FastGen for "
          f"gen_bsz={gen_bsz}, prompt_len={in_len}, gen_len={out_len}...",
          flush=True)
    build_t0 = time.perf_counter()
    args = _bitnet_generate.GenArgs(
        prompt_length=in_len, gen_length=out_len, gen_bsz=gen_bsz,
    )
    g = _bitnet_generate.FastGen.build(ckpt_dir, args, "cuda:0")
    # Upstream generate.py expects ``tokenizer.eot_id`` for the
    # early-stop check, but the bundled Llama-3 tiktoken Tokenizer only
    # exposes ``eos_id``.  We force-disable early-stop by setting an
    # unreachable id so the bench runs ``gen_length`` decode iterations
    # for every prompt (ignore_eos=True semantics, matching kb-nano).
    g.tokenizer.eot_id = -1
    torch.cuda.synchronize()
    print(f"[sota] build took {time.perf_counter() - build_t0:.1f}s",
          flush=True)

    # Warmup
    warm = [prompts[0][:in_len]] * gen_bsz
    g.generate_all(warm, use_cuda_graphs=True, use_sampling=False)
    torch.cuda.synchronize()

    n_batches = math.ceil(len(prompts) / gen_bsz)
    n_in = 0
    n_out = 0
    elapsed_total = 0.0
    out_records = [None] * len(prompts)
    for bi in range(n_batches):
        batch = prompts[bi * gen_bsz:(bi + 1) * gen_bsz]
        # Pad the last batch with copies of the first prompt so the
        # CUDA graph shapes still match.  These padded outputs are
        # NOT counted toward the throughput numerator/denominator and
        # are NOT written into out_records.
        real_count = len(batch)
        while len(batch) < gen_bsz:
            batch.append(batch[0])

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _, answers = g.generate_all(batch, use_cuda_graphs=True, use_sampling=False)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        elapsed_total += dt
        n_in += real_count * in_len
        n_out += real_count * out_len

        for j in range(real_count):
            ans = list(answers[j])[:out_len]
            out_records[bi * gen_bsz + j] = {"token_ids": ans}

    assert all(r is not None for r in out_records)

    results.append({
        "name": sc["name"], "elapsed": elapsed_total,
        "total_input_tokens": n_in, "total_output_tokens": n_out,
        "num_prompts": len(prompts), "gen_bsz": gen_bsz,
        "outputs": out_records,
    })
    print(f"[sota] {sc['name']:>14}: {elapsed_total:7.2f}s  "
          f"in={n_in:>8d}  out={n_out:>8d}  "
          f"throughput={(n_in + n_out) / elapsed_total:>8.1f} tok/s",
          flush=True)

    # Free per-scenario state before the next FastGen build.
    del g
    torch.cuda.empty_cache()

with open(cfg["output_file"], "w") as f:
    json.dump({
        "throughput": results,
        "memory_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2),
    }, f)
'''


def run_worker(script: str, config: dict, label: str) -> dict | None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, dir="/tmp") as f:
        f.write(script)
        wpath = f.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config, f)
        cpath = f.name
    out_path = config["output_file"]
    try:
        print(f"\n{'-' * 70}\n  {label}\n{'-' * 70}", flush=True)
        r = subprocess.run([sys.executable, wpath, cpath], timeout=14400)
        if r.returncode != 0:
            print(f"  ERROR: {label} exit code {r.returncode}", flush=True)
            return None
        with open(out_path) as f:
            return json.load(f)
    finally:
        os.unlink(wpath)
        os.unlink(cpath)
        if os.path.exists(out_path):
            os.unlink(out_path)


# ---------------------------------------------------------------------------
# Alignment check (mirrors bench_vllm.py::compute_alignment).
# ---------------------------------------------------------------------------
def compute_alignment(
    a_outputs: list[dict],
    b_outputs: list[dict],
) -> dict:
    """Compare per-request token_ids. Returns alignment statistics."""
    total_seqs = len(a_outputs)
    exact_matches = 0
    total_matching_tokens = 0
    total_output_tokens = 0

    for a, b in zip(a_outputs, b_outputs):
        a_ids = a["token_ids"]
        b_ids = b["token_ids"]
        out_len = max(len(a_ids), len(b_ids))
        total_output_tokens += out_len

        if a_ids == b_ids:
            exact_matches += 1
            total_matching_tokens += len(a_ids)
        else:
            min_len = min(len(a_ids), len(b_ids))
            matching = sum(1 for j in range(min_len) if a_ids[j] == b_ids[j])
            total_matching_tokens += matching

    avg_matching = total_matching_tokens / total_seqs if total_seqs else 0
    avg_output_len = total_output_tokens / total_seqs if total_seqs else 0

    return {
        "exact_matches": exact_matches,
        "total_seqs": total_seqs,
        "total_matching_tokens": total_matching_tokens,
        "total_output_tokens": total_output_tokens,
        "avg_matching_tokens_per_request": avg_matching,
        "avg_output_len": avg_output_len,
    }


def _print_throughput_table(label: str, data: dict):
    print(f"\n{'=' * 70}\n  {label}\n{'=' * 70}")
    print(f"{'scenario':>14}  {'elapsed(s)':>10}  {'in':>10}  {'out':>10}  "
          f"{'tok/s':>10}")
    for r in data["throughput"]:
        tps = (r["total_input_tokens"] + r["total_output_tokens"]) / r["elapsed"]
        print(f"{r['name']:>14}  {r['elapsed']:>10.2f}  "
              f"{r['total_input_tokens']:>10d}  {r['total_output_tokens']:>10d}  "
              f"{tps:>10.1f}")
    if "memory_gb" in data:
        print(f"\n  Memory: {data['memory_gb']} GB")


def _print_summary_table(sota_data: dict, kb_data: dict,
                         alignments: dict[str, dict]):
    print(f"\n{'=' * 100}")
    print("  SUMMARY (kb-nano vs Microsoft BitNet GPU)")
    print(f"{'=' * 100}")
    header = (
        f"  {'SCENARIO':<16} {'IN':>5} {'OUT':>5} "
        f"{'KB-NANO tok/s':>15} {'SOTA tok/s':>12} {'SPEEDUP':>8} "
        f"{'AVG MATCH TOKS':>15} {'EXACT':>10}"
    )
    print(header)
    print(f"  {'-' * 96}")
    for sr, kr in zip(sota_data["throughput"], kb_data["throughput"]):
        sota_tps = (sr["total_input_tokens"]
                    + sr["total_output_tokens"]) / sr["elapsed"]
        kb_tps = (kr["total_input_tokens"]
                  + kr["total_output_tokens"]) / kr["elapsed"]
        a = alignments.get(sr["name"], {})
        avg_match = a.get("avg_matching_tokens_per_request", 0)
        avg_out = a.get("avg_output_len", 0)
        match_str = f"{avg_match:.1f}/{avg_out:.0f}" if avg_out else "N/A"
        exact_str = (f"{a.get('exact_matches', 0)}/{a.get('total_seqs', 0)}"
                     if a else "N/A")
        # input length printed from the prompt records (not stored on
        # the throughput dict).
        in_len = sr["total_input_tokens"] // sr["num_prompts"] if sr["num_prompts"] else 0
        out_len = sr["total_output_tokens"] // sr["num_prompts"] if sr["num_prompts"] else 0
        print(
            f"  {sr['name']:<16} {in_len:>5} {out_len:>5} "
            f"{kb_tps:>15,.0f} {sota_tps:>12,.0f} "
            f"{kb_tps / sota_tps:>7.2f}x "
            f"{match_str:>15} {exact_str:>10}"
        )
    print(f"{'=' * 100}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--num-prompts", type=int, default=1000,
                    help="Requests per scenario (default 1000 per "
                         "adding-arch-instructions.md)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--vocab-size", type=int, default=128256)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--bitnet-repo",
                    default=os.environ.get(
                        "BITNET_REPO",
                        "/home/yak/vllm_repo/BitNet"),
                    help="Path to the Microsoft BitNet repo "
                         "(must contain gpu/checkpoints/model_state_int2.pt and "
                         "gpu/bitnet_kernels/libbitnet.so)")
    ap.add_argument("--gen-bsz", type=int, default=32,
                    help="CUDA-graph batch size for the Microsoft BitNet "
                         "GPU worker (memory scales with gen_bsz x max_seq)")
    ap.add_argument("--skip-sota", action="store_true",
                    help="Skip the Microsoft BitNet GPU SOTA reference run")
    ap.add_argument("--skip-kb", action="store_true",
                    help="Skip kb-nano (SOTA only)")
    ap.add_argument("--output-dir", type=str, default=None,
                    help="Directory to save per-scenario outputs and "
                         "alignment json (default: tests/results/<gpu>/"
                         "<model>_bitnet)")
    args = ap.parse_args()

    gpu = _detect_gpu_name()
    print("=" * 70)
    print(f"  BitNet bench: {args.model}")
    print(f"  GPU: {gpu} | num_prompts/scenario: {args.num_prompts}")
    print("=" * 70)

    if args.output_dir is None:
        short = args.model.split("/")[-1]
        args.output_dir = str(
            _PACKAGE_DIR / "tests" / "results" / gpu / f"{short}_bitnet"
        )
    os.makedirs(args.output_dir, exist_ok=True)

    scenarios = []
    for sc in SCENARIOS:
        prompt_ids = _build_random_token_prompts(
            args.num_prompts, sc["input_len"], args.vocab_size, args.seed,
        )
        scenarios.append({
            "name": sc["name"],
            "prompt_token_ids": prompt_ids,
            "output_lens": [sc["output_len"]] * args.num_prompts,
        })

    sota_data = None
    if not args.skip_sota:
        ckpt_dir = os.path.join(args.bitnet_repo, "gpu", "checkpoints")
        kernel_so = os.path.join(args.bitnet_repo, "gpu",
                                 "bitnet_kernels", "libbitnet.so")
        int2_pt = os.path.join(ckpt_dir, "model_state_int2.pt")
        fp16_pt = os.path.join(ckpt_dir, "model_state_fp16.pt")
        missing = [p for p in (kernel_so, int2_pt, fp16_pt) if not os.path.isfile(p)]
        if missing:
            print(f"\n[skip-sota] Microsoft BitNet GPU artifacts missing: "
                  f"{missing}.  See module docstring for setup instructions.\n")
        else:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                sota_out = f.name
            sota_cfg = {
                "model": args.model, "seed": args.seed,
                "scenarios": scenarios, "output_file": sota_out,
                "bitnet_repo": args.bitnet_repo, "ckpt_dir": ckpt_dir,
                "gen_bsz": args.gen_bsz,
            }
            sota_data = run_worker(
                SOTA_WORKER, sota_cfg,
                f"Microsoft BitNet GPU SOTA [{args.model}, "
                f"gen_bsz={args.gen_bsz}]")
            if sota_data:
                _print_throughput_table(
                    "Microsoft BitNet GPU (W1.58A8 official kernel)",
                    sota_data)

    kb_data = None
    if not args.skip_kb:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            kb_out = f.name
        kb_cfg = {
            "model": args.model, "seed": args.seed, "tp": args.tp,
            "scenarios": scenarios, "output_file": kb_out,
            "max_model_len": args.max_model_len,
            "project_root": str(_PROJECT_ROOT),
            "enforce_eager": True,
        }
        kb_data = run_worker(KB_WORKER, kb_cfg, f"kb-nano [{args.model}]")
        if kb_data:
            _print_throughput_table("kb-nano (W1.58A8 native int8 kernel)",
                                    kb_data)

    alignments: dict[str, dict] = {}
    if sota_data and kb_data:
        for sr, kr in zip(sota_data["throughput"], kb_data["throughput"]):
            assert sr["name"] == kr["name"]
            sota_outs = sr.get("outputs") or []
            kb_outs = kr.get("outputs") or []
            if not sota_outs or not kb_outs:
                continue
            alignments[sr["name"]] = compute_alignment(kb_outs, sota_outs)

        _print_summary_table(sota_data, kb_data, alignments)

        # Persist per-scenario records.
        for sr, kr in zip(sota_data["throughput"], kb_data["throughput"]):
            sdir = os.path.join(args.output_dir, sr["name"])
            os.makedirs(sdir, exist_ok=True)
            with open(os.path.join(sdir, "sota_outputs.json"), "w") as f:
                json.dump(sr, f)
            with open(os.path.join(sdir, "kb_nano_outputs.json"), "w") as f:
                json.dump(kr, f)
            with open(os.path.join(sdir, "alignment.json"), "w") as f:
                json.dump(alignments.get(sr["name"], {}), f, indent=2)
        print(f"\n  Per-scenario outputs + alignment saved under: "
              f"{args.output_dir}")


if __name__ == "__main__":
    main()
