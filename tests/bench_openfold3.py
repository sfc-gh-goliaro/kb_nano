#!/usr/bin/env python3
"""
End-to-end benchmark: kb-nano OpenFold3 vs reference openfold3.

Compares the FULL inference pipeline (InputEmbedder → MSA → PairFormer →
AuxHeads → DiffusionSampling) rather than just the trunk, matching what
bench_vllm.py does for LLMs (full generate()) and bench_diffusers.py does
for diffusion (full denoise loop).

Metrics:
  1. End-to-end correctness: shared weights, compare trunk outputs (s, z)
     and head logits via cosine similarity.
  2. Latency: median wall-clock time for full model.forward() with warmup.
  3. Throughput: tokens/second for the full inference pipeline.

Workload scale matches other kb_nano benchmarks:
  - bench_vllm.py: 1000 sequences × 1024 input + 512 output tokens
  - bench_diffusers.py: 40 images at 1024×1024, 50 inference steps
  - bench_openfold3.py: up to 2048 tokens, 48 PairFormer blocks, 1024 MSA
    seqs, 5 diffusion rollout steps — full AF3 architecture scale

Usage:
    CUDA_VISIBLE_DEVICES=7 python tests/bench_openfold3.py
    CUDA_VISIBLE_DEVICES=7 python tests/bench_openfold3.py --workload complex_large
    CUDA_VISIBLE_DEVICES=7 python tests/bench_openfold3.py --skip-reference
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

_THIS_DIR = Path(__file__).resolve().parent
_PACKAGE_DIR = _THIS_DIR.parent

sys.path.insert(0, str(_PACKAGE_DIR))

# ---------------------------------------------------------------------------
# Workloads — calibrated to real AF3 inference sizes
# ---------------------------------------------------------------------------

WORKLOADS = {
    "sanity": {
        "name": "sanity",
        "n_tokens": 256,
        "pf_blocks": 4,
        "msa_blocks": 2,
        "n_msa_seqs": 16,
        "no_rollout_steps": 5,
        "num_warmup": 3,
        "num_iters": 10,
        "description": "256 tokens, 4 PF blocks — fast sanity check",
    },

    "monomer_small": {
        "name": "monomer_small",
        "n_tokens": 384,
        "pf_blocks": 48,
        "msa_blocks": 4,
        "n_msa_seqs": 512,
        "no_rollout_steps": 5,
        "num_warmup": 2,
        "num_iters": 5,
        "description": "384 tokens, full 48-block AF3 — small monomer",
    },

    "complex_medium": {
        "name": "complex_medium",
        "n_tokens": 768,
        "pf_blocks": 48,
        "msa_blocks": 4,
        "n_msa_seqs": 1024,
        "no_rollout_steps": 5,
        "num_warmup": 2,
        "num_iters": 5,
        "description": "768 tokens, full 48-block AF3 — medium heteromer",
    },

    "complex_large": {
        "name": "complex_large",
        "n_tokens": 1536,
        "pf_blocks": 48,
        "msa_blocks": 4,
        "n_msa_seqs": 1024,
        "no_rollout_steps": 5,
        "num_warmup": 1,
        "num_iters": 3,
        "description": "1536 tokens, full 48-block AF3 — large assembly",
    },

    "max_capacity": {
        "name": "max_capacity",
        "n_tokens": 2048,
        "pf_blocks": 48,
        "msa_blocks": 4,
        "n_msa_seqs": 1024,
        "no_rollout_steps": 5,
        "num_warmup": 1,
        "num_iters": 3,
        "description": "2048 tokens, full 48-block AF3 — near max AF3 capacity",
    },
}

CORRECTNESS_THRESHOLDS = {
    "trunk_s": 0.99,
    "trunk_z": 0.99,
    "plddt_logits": 0.95,
    "distogram_logits": 0.95,
    "pae_logits": 0.95,
    "pde_logits": 0.95,
}

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _print(msg: str = ""):
    print(msg, flush=True)


def detect_gpu() -> str:
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


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.reshape(-1).double()
    b_flat = b.reshape(-1).double()
    denom = a_flat.norm() * b_flat.norm()
    if denom < 1e-12:
        return 0.0
    return (torch.dot(a_flat, b_flat) / denom).item()


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def randomize_weights(module: torch.nn.Module):
    for p in module.parameters():
        p.data.normal_(0, 0.02)


def transfer_matching_weights(src: torch.nn.Module, dst: torch.nn.Module) -> int:
    """Copy all state-dict entries with matching keys and shapes."""
    src_sd = src.state_dict()
    dst_sd = dst.state_dict()
    n = 0
    for key in src_sd:
        if key in dst_sd and src_sd[key].shape == dst_sd[key].shape:
            dst_sd[key] = src_sd[key]
            n += 1
    dst.load_state_dict(dst_sd)
    return n


# ---------------------------------------------------------------------------
# Build models
# ---------------------------------------------------------------------------

def build_kb_pairformer(cfg: dict, device: str, dtype: torch.dtype):
    from kb_nano.tasks.baseline.L3.openfold3_pairformer import PairFormerStack

    return PairFormerStack(
        c_s=384, c_z=128,
        c_hidden_pair_bias=24, no_heads_pair_bias=16,
        c_hidden_mul=128, c_hidden_pair_att=32, no_heads_pair=4,
        no_blocks=cfg["pf_blocks"],
        transition_n=4, pair_dropout=0.0, inf=1e9,
    ).to(device=device, dtype=dtype).eval()


def build_ref_pairformer(cfg: dict, device: str, dtype: torch.dtype):
    from openfold3.core.model.latent.pairformer import PairFormerStack

    return PairFormerStack(
        c_s=384, c_z=128,
        c_hidden_pair_bias=24, no_heads_pair_bias=16,
        c_hidden_mul=128, c_hidden_pair_att=32, no_heads_pair=4,
        no_blocks=cfg["pf_blocks"],
        transition_type="swiglu", transition_n=4,
        pair_dropout=0.0, fuse_projection_weights=False,
        blocks_per_ckpt=None, inf=1e9,
    ).to(device=device, dtype=dtype).eval()


def build_kb_heads(device: str, dtype: torch.dtype):
    from kb_nano.tasks.baseline.L3.openfold3_heads import AuxiliaryHeads
    return AuxiliaryHeads(c_s=384, c_z=128).to(device=device, dtype=dtype).eval()


def build_ref_heads(device: str, dtype: torch.dtype):
    from kb_nano.tasks.baseline.L3.openfold3_heads import AuxiliaryHeads
    return AuxiliaryHeads(c_s=384, c_z=128).to(device=device, dtype=dtype).eval()


def build_kb_full_model(cfg: dict, device: str, dtype: torch.dtype):
    from kb_nano.tasks.baseline.L4.openfold3 import OpenFold3Config, OpenFold3Model

    config = OpenFold3Config(
        pairformer_no_blocks=cfg["pf_blocks"],
        msa_no_blocks=cfg["msa_blocks"],
        diff_no_blocks=4,
        no_rollout_steps=cfg.get("no_rollout_steps", 5),
        num_recycles=1,
    )
    return OpenFold3Model(config).to(device=device, dtype=dtype).eval(), config


def build_ref_full_model(cfg: dict, device: str, dtype: torch.dtype):
    """Build the full reference OpenFold3 model from its config module."""
    from openfold3.projects.of3_all_atom.config.model_config import (
        model_config as base_config,
    )
    from openfold3.projects.of3_all_atom.model import OpenFold3

    config = copy.deepcopy(base_config)

    config.architecture.pairformer.no_blocks = cfg["pf_blocks"]
    config.architecture.msa.msa_module.no_blocks = cfg["msa_blocks"]
    config.architecture.diffusion_module.diffusion_transformer.no_blocks = 4

    config.architecture.shared.diffusion.no_full_rollout_steps = cfg.get(
        "no_rollout_steps", 5
    )

    config.settings.memory.eval.use_deepspeed_evo_attention = False
    config.settings.memory.eval.use_cueq_triangle_kernels = False

    model = OpenFold3(config).to(device=device, dtype=dtype).eval()
    return model, config


def _make_kb_batch(cfg: dict, kb_config, device: str, dtype: torch.dtype) -> dict:
    """Create a synthetic input batch for kb-nano's OpenFold3Model."""
    n_tokens = cfg["n_tokens"]
    batch = {
        "token_features": torch.randn(1, n_tokens, kb_config.c_token_embedder,
                                      device=device, dtype=dtype),
        "residue_index": torch.arange(n_tokens, device=device).unsqueeze(0).float(),
        "token_mask": torch.ones(1, n_tokens, device=device, dtype=dtype),
        "atom_mask": torch.ones(1, n_tokens, device=device, dtype=dtype),
    }
    if cfg["n_msa_seqs"] > 0:
        batch["msa"] = torch.randn(1, cfg["n_msa_seqs"], n_tokens, kb_config.c_m,
                                   device=device, dtype=dtype)
        batch["msa_mask"] = torch.ones(1, cfg["n_msa_seqs"], n_tokens,
                                       device=device, dtype=dtype)
    return batch


def _make_ref_batch(cfg: dict, device: str, dtype: torch.dtype) -> dict:
    """Create a synthetic input batch for the reference OpenFold3.

    Builds all required features directly (without importing openfold3's
    heavy data pipeline which pulls in pdbeccdutils, biotite, etc.).
    Uses a fixed atoms_per_token=5 for all tokens to simulate realistic
    atom counts while keeping the batch construction simple.
    """
    n_tokens = cfg["n_tokens"]
    n_msa = max(cfg["n_msa_seqs"], 4)
    atoms_per_token = 5
    n_atom = n_tokens * atoms_per_token

    start_atom_index = torch.arange(0, n_atom, atoms_per_token).unsqueeze(0).int()
    num_atoms_per_token = torch.full((1, n_tokens), atoms_per_token).int()

    atom_to_token = torch.arange(n_tokens).repeat_interleave(atoms_per_token)
    atom_to_token = atom_to_token.unsqueeze(0).int()

    ref_space_uid = atom_to_token.clone()

    features = {
        "residue_index": torch.arange(n_tokens).unsqueeze(0).int(),
        "token_index": torch.arange(n_tokens).unsqueeze(0).int(),
        "asym_id": torch.ones(1, n_tokens).int(),
        "entity_id": torch.ones(1, n_tokens).int(),
        "sym_id": torch.ones(1, n_tokens).int(),
        "restype": torch.nn.functional.one_hot(
            torch.randint(0, 32, (n_tokens,)), 32
        ).unsqueeze(0).int(),
        "is_protein": torch.ones(1, n_tokens).int(),
        "is_dna": torch.zeros(1, n_tokens).int(),
        "is_rna": torch.zeros(1, n_tokens).int(),
        "is_ligand": torch.zeros(1, n_tokens).int(),
        "is_atomized": torch.zeros(1, n_tokens).int(),
        "ref_pos": torch.randn(1, n_atom, 3).float(),
        "ref_mask": torch.ones(1, n_atom).int(),
        "ref_element": torch.ones(1, n_atom, 119).int(),
        "ref_charge": torch.ones(1, n_atom).float(),
        "ref_atom_name_chars": torch.ones(1, n_atom, 4, 64).int(),
        "ref_space_uid": ref_space_uid,
        "msa": torch.ones(1, n_msa, n_tokens, 32).int(),
        "has_deletion": torch.ones(1, n_msa, n_tokens).float(),
        "deletion_value": torch.ones(1, n_msa, n_tokens).float(),
        "profile": torch.ones(1, n_tokens, 32).float(),
        "deletion_mean": torch.ones(1, n_tokens).float(),
        "template_restype": torch.ones(1, 0, n_tokens, 32).int(),
        "template_pseudo_beta_mask": torch.ones(1, 0, n_tokens).float(),
        "template_backbone_frame_mask": torch.ones(1, 0, n_tokens).float(),
        "template_distogram": torch.ones(1, 0, n_tokens, n_tokens, 39).float(),
        "template_unit_vector": torch.ones(1, 0, n_tokens, n_tokens, 3).float(),
        "token_bonds": torch.zeros(1, n_tokens, n_tokens).int(),
        "token_mask": torch.ones(1, n_tokens).float(),
        "atom_mask": torch.ones(1, n_atom).float(),
        "start_atom_index": start_atom_index,
        "num_atoms_per_token": num_atoms_per_token,
        "atom_to_token_index": atom_to_token,
        "msa_mask": torch.ones(1, n_msa, n_tokens).float(),
        "num_paired_seqs": torch.tensor([n_msa // 4]),
    }

    def _to_device(t):
        if isinstance(t, torch.Tensor):
            if t.is_floating_point():
                return t.to(device=device, dtype=dtype)
            return t.to(device=device)
        return t

    features = {k: _to_device(v) for k, v in features.items()}
    return features


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def _benchmark_module(module, run_fn, num_warmup, num_iters, label):
    """Run warmup + timed iterations for a module, return latency stats."""
    for i in tqdm(range(num_warmup), desc=f"  {label} warmup", leave=False):
        with torch.no_grad():
            run_fn()
        torch.cuda.synchronize()

    latencies = []
    for i in tqdm(range(num_iters), desc=f"  {label} bench", leave=False):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            run_fn()
        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - t0)

    median = float(np.median(latencies))
    mean = float(np.mean(latencies))
    min_t = float(np.min(latencies))
    peak = torch.cuda.max_memory_allocated() / 1e6
    params = sum(p.numel() for p in module.parameters())

    _print(f"  {label}: median={median:.4f}s  mean={mean:.4f}s  "
           f"min={min_t:.4f}s  params={params:,}  peak_mem={peak:.0f}MB")

    return {
        "median_s": median, "mean_s": mean, "min_s": min_t,
        "latencies": latencies, "params": params, "peak_mem_mb": peak,
    }


# ---------------------------------------------------------------------------
# Correctness test (PairFormerStack + Heads, shared weights)
# ---------------------------------------------------------------------------

def run_correctness_test(cfg: dict, device: str, dtype: torch.dtype) -> dict:
    """Run end-to-end correctness comparison with shared weights."""
    n_tokens = cfg["n_tokens"]
    _print(f"\n  Building PairFormerStack ({cfg['pf_blocks']} blocks) + heads ...")

    kb_pf = build_kb_pairformer(cfg, device, dtype)
    _print(f"    KB PairFormer built ({sum(p.numel() for p in kb_pf.parameters()):,} params)")
    randomize_weights(kb_pf)

    _print(f"    Building reference PairFormerStack ...")
    ref_pf = build_ref_pairformer(cfg, device, dtype)
    n_pf = transfer_matching_weights(kb_pf, ref_pf)

    kb_heads = build_kb_heads(device, dtype)
    randomize_weights(kb_heads)
    ref_heads = build_ref_heads(device, dtype)
    n_heads = transfer_matching_weights(kb_heads, ref_heads)

    _print(f"    Shared PF weights: {n_pf} tensors, Head weights: {n_heads} tensors")

    s = torch.randn(1, n_tokens, 384, device=device, dtype=dtype)
    z = torch.randn(1, n_tokens, n_tokens, 128, device=device, dtype=dtype)
    single_mask = torch.ones(1, n_tokens, device=device, dtype=dtype)
    pair_mask = torch.ones(1, n_tokens, n_tokens, device=device, dtype=dtype)

    _print(f"    Running forward pass ({n_tokens} tokens) ...")
    with torch.no_grad():
        kb_s, kb_z = kb_pf(s=s.clone(), z=z.clone(),
                           single_mask=single_mask, pair_mask=pair_mask)
        kb_head_out = kb_heads(s=kb_s, z=kb_z)

        ref_s, ref_z = ref_pf(s=s.clone(), z=z.clone(),
                              single_mask=single_mask, pair_mask=pair_mask)
        ref_head_out = ref_heads(s=ref_s, z=ref_z)

    results = {}

    for name, kb_t, ref_t in [
        ("trunk_s", kb_s, ref_s),
        ("trunk_z", kb_z, ref_z),
    ]:
        cos = cosine_sim(kb_t, ref_t)
        mad = max_abs_diff(kb_t, ref_t)
        threshold = CORRECTNESS_THRESHOLDS[name]
        passed = cos >= threshold
        status = "PASS" if passed else "FAIL"
        results[name] = {"cosine": cos, "max_abs_diff": mad,
                         "threshold": threshold, "passed": passed}
        _print(f"    [{status}] {name}: cosine={cos:.8f}  "
               f"max_abs_diff={mad:.6e}  (threshold={threshold})")

    for head_name in ["plddt_logits", "distogram_logits", "pae_logits", "pde_logits"]:
        kb_h = kb_head_out[head_name]
        ref_h = ref_head_out[head_name]
        cos = cosine_sim(kb_h, ref_h)
        mad = max_abs_diff(kb_h, ref_h)
        threshold = CORRECTNESS_THRESHOLDS[head_name]
        passed = cos >= threshold
        status = "PASS" if passed else "FAIL"
        results[head_name] = {"cosine": cos, "max_abs_diff": mad,
                              "threshold": threshold, "passed": passed}
        _print(f"    [{status}] {head_name}: cosine={cos:.8f}  "
               f"max_abs_diff={mad:.6e}  (threshold={threshold})")

    del kb_pf, ref_pf, kb_heads, ref_heads
    torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Latency benchmark — full end-to-end pipeline
# ---------------------------------------------------------------------------

def run_latency_benchmark(
    cfg: dict, device: str, dtype: torch.dtype,
    skip_reference: bool = False,
) -> dict:
    """Benchmark latency for the FULL pipeline (forward()), not just trunk."""
    n_tokens = cfg["n_tokens"]
    num_warmup = cfg["num_warmup"]
    num_iters = cfg["num_iters"]

    _print(f"\n  --- E2E Latency benchmark ({n_tokens} tokens, "
           f"{num_warmup} warmup + {num_iters} iters) ---")

    result = {}

    # ---- KB-nano full model ----
    _print(f"\n  Building KB-nano full L4 model ({cfg['pf_blocks']} PF blocks, "
           f"{cfg['msa_blocks']} MSA blocks, "
           f"{cfg.get('no_rollout_steps', 5)} rollout steps) ...")
    kb_model, kb_config = build_kb_full_model(cfg, device, dtype)
    randomize_weights(kb_model)

    kb_batch = _make_kb_batch(cfg, kb_config, device, dtype)

    torch.cuda.reset_peak_memory_stats()
    kb_e2e_stats = _benchmark_module(
        kb_model,
        lambda: kb_model(copy.deepcopy(kb_batch)),
        num_warmup, num_iters, "KB  E2E",
    )
    kb_e2e_stats["throughput_tok_s"] = n_tokens / kb_e2e_stats["median_s"]

    result["kb_e2e"] = kb_e2e_stats

    # Also measure trunk-only for comparison
    torch.cuda.reset_peak_memory_stats()
    kb_trunk_stats = _benchmark_module(
        kb_model,
        lambda: kb_model.run_trunk(kb_batch),
        num_warmup, num_iters, "KB  Trunk",
    )
    kb_trunk_stats["throughput_tok_s"] = n_tokens / kb_trunk_stats["median_s"]
    result["kb_trunk"] = kb_trunk_stats

    del kb_model
    torch.cuda.empty_cache()

    # ---- Reference full model ----
    if not skip_reference:
        _print(f"\n  Building reference OpenFold3 full model ...")
        try:
            ref_model, ref_config = build_ref_full_model(cfg, device, dtype)
            randomize_weights(ref_model)
            _print(f"    Ref model built ({sum(p.numel() for p in ref_model.parameters()):,} params)")

            ref_batch = _make_ref_batch(cfg, device, dtype)

            torch.cuda.reset_peak_memory_stats()
            ref_e2e_stats = _benchmark_module(
                ref_model,
                lambda: ref_model(batch=copy.deepcopy(ref_batch)),
                num_warmup, num_iters, "Ref E2E",
            )
            ref_e2e_stats["throughput_tok_s"] = n_tokens / ref_e2e_stats["median_s"]

            speedup = ref_e2e_stats["median_s"] / kb_e2e_stats["median_s"]
            _print(f"  E2E speedup (kb-nano / ref): {speedup:.2f}x")

            result["ref_e2e"] = ref_e2e_stats
            result["e2e_speedup"] = speedup

            del ref_model
        except Exception as e:
            import traceback
            traceback.print_exc()
            _print(f"  [WARNING] Reference E2E benchmark failed: {e}")
            _print(f"  Falling back to PairFormer-only comparison ...")
            result["ref_e2e"] = {"error": str(e)}

        torch.cuda.empty_cache()

    # ---- PairFormer apples-to-apples (matching exactly) ----
    _print(f"\n  --- PairFormer apples-to-apples comparison ---")

    s = torch.randn(1, n_tokens, 384, device=device, dtype=dtype)
    z = torch.randn(1, n_tokens, n_tokens, 128, device=device, dtype=dtype)
    single_mask = torch.ones(1, n_tokens, device=device, dtype=dtype)
    pair_mask = torch.ones(1, n_tokens, n_tokens, device=device, dtype=dtype)

    _print(f"  Building KB-nano PairFormerStack ({cfg['pf_blocks']} blocks) ...")
    kb_pf = build_kb_pairformer(cfg, device, dtype)
    randomize_weights(kb_pf)

    torch.cuda.reset_peak_memory_stats()
    kb_pf_stats = _benchmark_module(
        kb_pf,
        lambda: kb_pf(s=s.clone(), z=z.clone(),
                      single_mask=single_mask, pair_mask=pair_mask),
        num_warmup, num_iters, "KB  PairFormer",
    )
    kb_pf_stats["throughput_tok_s"] = n_tokens / kb_pf_stats["median_s"]

    del kb_pf
    torch.cuda.empty_cache()

    result["kb_pairformer"] = kb_pf_stats

    if not skip_reference:
        _print(f"  Building reference PairFormerStack ({cfg['pf_blocks']} blocks) ...")
        ref_pf = build_ref_pairformer(cfg, device, dtype)
        randomize_weights(ref_pf)

        torch.cuda.reset_peak_memory_stats()
        ref_pf_stats = _benchmark_module(
            ref_pf,
            lambda: ref_pf(s=s.clone(), z=z.clone(),
                           single_mask=single_mask, pair_mask=pair_mask),
            num_warmup, num_iters, "Ref PairFormer",
        )
        ref_pf_stats["throughput_tok_s"] = n_tokens / ref_pf_stats["median_s"]

        pf_speedup = ref_pf_stats["median_s"] / kb_pf_stats["median_s"]
        _print(f"  PairFormer speedup (kb-nano / ref): {pf_speedup:.2f}x")

        result["ref_pairformer"] = ref_pf_stats
        result["pairformer_speedup"] = pf_speedup

        del ref_pf
        torch.cuda.empty_cache()

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="End-to-end benchmark: kb-nano OpenFold3 vs reference openfold3"
    )
    parser.add_argument("--workload", type=str, default=None,
                        choices=list(WORKLOADS.keys()),
                        help="Run specific workload (default: run all)")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--skip-reference", action="store_true",
                        help="Skip reference openfold3 comparison")
    parser.add_argument("--skip-correctness", action="store_true",
                        help="Skip correctness tests (latency only)")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype)
    gpu = detect_gpu()
    torch.manual_seed(42)

    _print(f"{'='*80}")
    _print(f"  OpenFold3 End-to-End Benchmark: kb-nano vs reference openfold3")
    _print(f"{'='*80}")
    _print(f"  GPU     : {gpu} ({torch.cuda.get_device_name() if device == 'cuda' else 'cpu'})")
    _print(f"  Dtype   : {args.dtype}")
    _print(f"  Device  : {device}")
    _print(f"{'='*80}")

    workloads = [WORKLOADS[args.workload]] if args.workload else list(WORKLOADS.values())

    output_dir = args.output_dir or str(
        _THIS_DIR / "results" / "openfold3" / f"{gpu}_{args.dtype}"
    )
    os.makedirs(output_dir, exist_ok=True)

    all_results = {
        "gpu": gpu, "dtype": args.dtype, "device": device, "workloads": {},
    }

    for wl in workloads:
        wl_name = wl["name"]
        _print(f"\n{'#'*80}")
        _print(f"  WORKLOAD: {wl_name} — {wl['description']}")
        _print(f"  Tokens: {wl['n_tokens']}  PF blocks: {wl['pf_blocks']}  "
               f"MSA blocks: {wl['msa_blocks']}  MSA seqs: {wl['n_msa_seqs']}  "
               f"Rollout steps: {wl.get('no_rollout_steps', 5)}")
        pair_mem_mb = wl['n_tokens'] ** 2 * 128 * 2 / 1e6
        _print(f"  Pair tensor: {wl['n_tokens']}×{wl['n_tokens']}×128 = "
               f"{wl['n_tokens']**2 * 128 / 1e6:.1f}M elements ({pair_mem_mb:.0f} MB bf16)")
        _print(f"{'#'*80}")

        wl_results = {}

        # Correctness
        if not args.skip_correctness and not args.skip_reference:
            try:
                wl_results["correctness"] = run_correctness_test(wl, device, dtype)
            except Exception as e:
                import traceback
                traceback.print_exc()
                _print(f"  [ERROR] Correctness test failed: {e}")
                wl_results["correctness"] = {"error": str(e)}

        # Latency / throughput
        try:
            torch.cuda.reset_peak_memory_stats()
            wl_results["latency"] = run_latency_benchmark(
                wl, device, dtype, skip_reference=args.skip_reference
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            _print(f"  [ERROR] Latency benchmark failed: {e}")
            wl_results["latency"] = {"error": str(e)}

        all_results["workloads"][wl_name] = wl_results
        torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    W = 18

    _print(f"\n\n{'='*100}")
    _print("  SUMMARY")
    _print(f"{'='*100}")

    # Correctness summary
    _print(f"\n  --- Correctness (cosine similarity, PairFormer + Heads shared weights) ---")
    _print(f"  {'Workload':<{W}} {'trunk_s':>10} {'trunk_z':>10} {'plddt':>10} "
           f"{'distogram':>12} {'pae':>10} {'pde':>10} {'Status':>8}")
    _print(f"  {'-'*90}")

    all_correct = True
    for wl_name, wl_res in all_results["workloads"].items():
        corr = wl_res.get("correctness", {})
        if "error" in corr or not corr:
            _print(f"  {wl_name:<{W}} {'N/A':>10} {'N/A':>10} {'N/A':>10} "
                   f"{'N/A':>12} {'N/A':>10} {'N/A':>10} {'SKIP':>8}")
            continue

        row_pass = True
        vals = []
        for key in ["trunk_s", "trunk_z", "plddt_logits", "distogram_logits",
                     "pae_logits", "pde_logits"]:
            entry = corr.get(key, {})
            cos = entry.get("cosine")
            if cos is not None:
                vals.append(f"{cos:.6f}")
                if not entry.get("passed", False):
                    row_pass = False
                    all_correct = False
            else:
                vals.append("N/A")

        status = "PASS" if row_pass else "FAIL"
        _print(f"  {wl_name:<{W}} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10} "
               f"{vals[3]:>12} {vals[4]:>10} {vals[5]:>10} {status:>8}")

    # E2E Latency summary
    _print(f"\n  --- End-to-End Latency (full forward(), median seconds) ---")
    _print(f"  {'Workload':<{W}} {'Tokens':>7} {'PF blks':>8} {'KB E2E':>10} "
           f"{'Ref E2E':>10} {'Speedup':>9} {'KB tok/s':>10} {'Ref tok/s':>10} "
           f"{'KB Peak':>10}")
    _print(f"  {'-'*94}")

    for wl_name, wl_res in all_results["workloads"].items():
        lat = wl_res.get("latency", {})
        if "error" in lat or not lat:
            wl_cfg = WORKLOADS.get(wl_name, {})
            _print(f"  {wl_name:<{W}} {wl_cfg.get('n_tokens', '?'):>7} "
                   f"{wl_cfg.get('pf_blocks', '?'):>8} {'ERROR':>10}")
            continue

        wl_cfg = WORKLOADS[wl_name]
        kb = lat.get("kb_e2e", {})
        ref = lat.get("ref_e2e", {})
        kb_med = kb.get("median_s", 0)
        ref_med = ref.get("median_s", 0) if not isinstance(ref, dict) or "error" not in ref else 0
        speedup = lat.get("e2e_speedup", 0)
        kb_tps = kb.get("throughput_tok_s", 0)
        ref_tps = ref.get("throughput_tok_s", 0) if not isinstance(ref, dict) or "error" not in ref else 0
        kb_peak = kb.get("peak_mem_mb", 0)

        ref_str = f"{ref_med:.4f}s" if ref_med else "N/A"
        speedup_str = f"{speedup:.2f}x" if speedup else "N/A"
        ref_tps_str = f"{ref_tps:.0f}" if ref_tps else "N/A"

        _print(f"  {wl_name:<{W}} {wl_cfg['n_tokens']:>7} {wl_cfg['pf_blocks']:>8} "
               f"{kb_med:.4f}s{'':<3} {ref_str:>10} {speedup_str:>9} "
               f"{kb_tps:>10.0f} {ref_tps_str:>10} {kb_peak:>9.0f}MB")

    # PairFormer-only summary
    _print(f"\n  --- PairFormer Latency (apples-to-apples, median seconds) ---")
    _print(f"  {'Workload':<{W}} {'Tokens':>7} {'PF blks':>8} {'KB PF':>10} "
           f"{'Ref PF':>10} {'Speedup':>9} {'KB tok/s':>10} {'Ref tok/s':>10}")
    _print(f"  {'-'*84}")

    for wl_name, wl_res in all_results["workloads"].items():
        lat = wl_res.get("latency", {})
        if "error" in lat or not lat:
            continue

        wl_cfg = WORKLOADS[wl_name]
        kb = lat.get("kb_pairformer", {})
        ref = lat.get("ref_pairformer", {})
        kb_med = kb.get("median_s", 0)
        ref_med = ref.get("median_s", 0)
        speedup = lat.get("pairformer_speedup", 0)
        kb_tps = kb.get("throughput_tok_s", 0)
        ref_tps = ref.get("throughput_tok_s", 0)

        ref_str = f"{ref_med:.4f}s" if ref_med else "N/A"
        speedup_str = f"{speedup:.2f}x" if speedup else "N/A"
        ref_tps_str = f"{ref_tps:.0f}" if ref_tps else "N/A"

        _print(f"  {wl_name:<{W}} {wl_cfg['n_tokens']:>7} {wl_cfg['pf_blocks']:>8} "
               f"{kb_med:.4f}s{'':<3} {ref_str:>10} {speedup_str:>9} "
               f"{kb_tps:>10.0f} {ref_tps_str:>10}")

    # Trunk vs E2E breakdown
    _print(f"\n  --- KB-nano Pipeline Breakdown (trunk vs full E2E) ---")
    _print(f"  {'Workload':<{W}} {'Tokens':>7} {'Trunk':>10} {'E2E':>10} "
           f"{'Diff%':>8} {'E2E tok/s':>10} {'Peak MB':>10}")
    _print(f"  {'-'*75}")
    for wl_name, wl_res in all_results["workloads"].items():
        lat = wl_res.get("latency", {})
        trunk = lat.get("kb_trunk", {})
        e2e = lat.get("kb_e2e", {})
        if not trunk or not e2e or "error" in lat:
            continue
        wl_cfg = WORKLOADS[wl_name]
        trunk_med = trunk["median_s"]
        e2e_med = e2e["median_s"]
        diff_pct = (e2e_med - trunk_med) / trunk_med * 100 if trunk_med > 0 else 0
        _print(f"  {wl_name:<{W}} {wl_cfg['n_tokens']:>7} {trunk_med:.4f}s"
               f"{'':<3} {e2e_med:.4f}s{'':<3} {diff_pct:>+6.1f}% "
               f"{e2e['throughput_tok_s']:>10.0f} {e2e['peak_mem_mb']:>9.0f}")

    overall = "PASS" if all_correct else "FAIL"
    _print(f"\n  Overall correctness: {overall}")
    _print(f"{'='*100}")

    # Save results
    results_path = os.path.join(output_dir, "benchmark_results.json")

    serializable = json.loads(json.dumps(all_results, default=str))
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    _print(f"\n  Results saved to: {results_path}")


if __name__ == "__main__":
    main()
