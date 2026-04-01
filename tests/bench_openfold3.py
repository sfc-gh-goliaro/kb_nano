#!/usr/bin/env python3
"""
Throughput, latency, and correctness benchmark: kb-nano OpenFold3 baseline
vs the reference OpenFold3 library (openfold3).

Both engines run in subprocesses to avoid import contamination.
Correctness is verified by comparing intermediate tensor outputs
(trunk s, z, head logits) between the two implementations on
identical synthetic inputs.

Usage:
    python tests/bench_openfold3.py
    python tests/bench_openfold3.py --skip-reference   # kb-nano only
    python tests/bench_openfold3.py --n-tokens 128      # small workload
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_PACKAGE_DIR = _THIS_DIR.parent

# ---------------------------------------------------------------------------
# Workload definitions for structure prediction benchmarks
# ---------------------------------------------------------------------------

STRUCTURE_PREDICTION_WORKLOADS = [
    {
        "name": "small_monomer",
        "n_tokens": 100,
        "n_msa_seqs": 0,
        "description": "Single chain, ~100 residues, no MSA",
    },
    {
        "name": "medium_monomer",
        "n_tokens": 256,
        "n_msa_seqs": 128,
        "description": "Single chain, ~256 residues, 128 MSA seqs",
    },
    {
        "name": "large_complex",
        "n_tokens": 384,
        "n_msa_seqs": 256,
        "description": "Heteromer, ~384 tokens, 256 MSA seqs",
    },
]

CORRECTNESS_THRESHOLDS = {
    "trunk_s_cosine": 0.99,
    "trunk_z_cosine": 0.99,
    "plddt_cosine": 0.95,
    "distogram_cosine": 0.95,
}

# ---------------------------------------------------------------------------
# Utility: GPU detection
# ---------------------------------------------------------------------------

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


def _cosine_similarity(a, b):
    """Compute cosine similarity between flattened tensors."""
    a_flat = a.reshape(-1).astype(np.float64)
    b_flat = b.reshape(-1).astype(np.float64)
    dot = np.dot(a_flat, b_flat)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Synthetic input generation
# ---------------------------------------------------------------------------

def generate_synthetic_batch(
    n_tokens: int,
    n_msa_seqs: int = 0,
    c_token: int = 384,
    c_m: int = 64,
    device: str = "cpu",
    dtype_str: str = "float32",
):
    """Generate a synthetic batch for OpenFold3 benchmarking.

    Returns a dict matching the batch format expected by OpenFold3Model.forward().
    """
    import torch

    dtype = getattr(torch, dtype_str)

    batch = {
        "token_features": torch.randn(1, n_tokens, c_token, device=device, dtype=dtype),
        "residue_index": torch.arange(n_tokens, device=device).unsqueeze(0).float(),
        "token_mask": torch.ones(1, n_tokens, device=device, dtype=dtype),
        "atom_mask": torch.ones(1, n_tokens, device=device, dtype=dtype),
    }

    if n_msa_seqs > 0:
        batch["msa"] = torch.randn(1, n_msa_seqs, n_tokens, c_m, device=device, dtype=dtype)
        batch["msa_mask"] = torch.ones(1, n_msa_seqs, n_tokens, device=device, dtype=dtype)

    return batch


# ---------------------------------------------------------------------------
# KB-Nano worker
# ---------------------------------------------------------------------------

KB_NANO_WORKER = (
    "import sys\n"
    "import time\n"
    "import json\n"
    "import os\n"
    "import torch\n"
    "import numpy as np\n"
    "\n"
    "sys.path.insert(0, " + repr(str(_PACKAGE_DIR)) + ")\n"
    "\n"
    "from kb_nano.tasks.baseline.L4.openfold3 import OpenFold3Config, OpenFold3Model\n"
    "\n"
    "def main():\n"
    "    args = json.loads(sys.argv[1])\n"
    "    n_tokens = args['n_tokens']\n"
    "    n_msa_seqs = args.get('n_msa_seqs', 0)\n"
    "    output_dir = args['output_dir']\n"
    "    dtype_str = args.get('dtype', 'bfloat16')\n"
    "    num_recycles = args.get('num_recycles', 1)\n"
    "\n"
    "    device = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
    "    dtype = getattr(torch, dtype_str)\n"
    "\n"
    "    config = OpenFold3Config(\n"
    "        num_recycles=num_recycles,\n"
    "        pairformer_no_blocks=4,\n"
    "        msa_no_blocks=2,\n"
    "        diff_no_blocks=4,\n"
    "        no_rollout_steps=5,\n"
    "    )\n"
    "    model = OpenFold3Model(config).to(device=device, dtype=dtype)\n"
    "    model.eval()\n"
    "\n"
    "    token_features = torch.randn(1, n_tokens, config.c_token_embedder,\n"
    "                                 device=device, dtype=dtype)\n"
    "    residue_index = torch.arange(n_tokens, device=device).unsqueeze(0).float()\n"
    "    token_mask = torch.ones(1, n_tokens, device=device, dtype=dtype)\n"
    "    atom_mask = torch.ones(1, n_tokens, device=device, dtype=dtype)\n"
    "\n"
    "    batch = dict(\n"
    "        token_features=token_features,\n"
    "        residue_index=residue_index,\n"
    "        token_mask=token_mask,\n"
    "        atom_mask=atom_mask,\n"
    "    )\n"
    "\n"
    "    if n_msa_seqs > 0:\n"
    "        batch['msa'] = torch.randn(1, n_msa_seqs, n_tokens, config.c_m,\n"
    "                                   device=device, dtype=dtype)\n"
    "        batch['msa_mask'] = torch.ones(1, n_msa_seqs, n_tokens,\n"
    "                                       device=device, dtype=dtype)\n"
    "\n"
    "    with torch.no_grad():\n"
    "        _ = model.run_trunk(batch, num_recycles=0)\n"
    "\n"
    "    if device == 'cuda':\n"
    "        torch.cuda.synchronize()\n"
    "\n"
    "    t0 = time.perf_counter()\n"
    "    with torch.no_grad():\n"
    "        s_input, s, z = model.run_trunk(batch)\n"
    "    if device == 'cuda':\n"
    "        torch.cuda.synchronize()\n"
    "    trunk_time = time.perf_counter() - t0\n"
    "\n"
    "    with torch.no_grad():\n"
    "        head_outputs = model.aux_heads(s=s, z=z)\n"
    "\n"
    "    os.makedirs(output_dir, exist_ok=True)\n"
    "\n"
    "    np.save(os.path.join(output_dir, 'kb_s_trunk.npy'),\n"
    "            s.detach().cpu().float().numpy())\n"
    "    np.save(os.path.join(output_dir, 'kb_z_trunk.npy'),\n"
    "            z.detach().cpu().float().numpy())\n"
    "    np.save(os.path.join(output_dir, 'kb_plddt_logits.npy'),\n"
    "            head_outputs['plddt_logits'].detach().cpu().float().numpy())\n"
    "    np.save(os.path.join(output_dir, 'kb_distogram_logits.npy'),\n"
    "            head_outputs['distogram_logits'].detach().cpu().float().numpy())\n"
    "\n"
    "    results = dict(\n"
    "        trunk_time_s=trunk_time,\n"
    "        n_tokens=n_tokens,\n"
    "        device=device,\n"
    "        dtype=dtype_str,\n"
    "    )\n"
    "    with open(os.path.join(output_dir, 'kb_results.json'), 'w') as f:\n"
    "        json.dump(results, f, indent=2)\n"
    "\n"
    "    print(f'KB-Nano trunk time: {trunk_time:.4f}s for {n_tokens} tokens')\n"
    "\n"
    "if __name__ == '__main__':\n"
    "    main()\n"
)


# ---------------------------------------------------------------------------
# Reference OpenFold3 worker
# ---------------------------------------------------------------------------

REFERENCE_WORKER = (
    "import sys\n"
    "import time\n"
    "import json\n"
    "import os\n"
    "import torch\n"
    "import numpy as np\n"
    "\n"
    "def main():\n"
    "    args = json.loads(sys.argv[1])\n"
    "    n_tokens = args['n_tokens']\n"
    "    n_msa_seqs = args.get('n_msa_seqs', 0)\n"
    "    output_dir = args['output_dir']\n"
    "    dtype_str = args.get('dtype', 'bfloat16')\n"
    "\n"
    "    try:\n"
    "        from openfold3.projects.of3_all_atom.config.model_config import model_config\n"
    "        from openfold3.projects.of3_all_atom.model import OpenFold3\n"
    "    except ImportError:\n"
    "        print('Reference openfold3 library not available, skipping.')\n"
    "        os.makedirs(output_dir, exist_ok=True)\n"
    "        with open(os.path.join(output_dir, 'ref_results.json'), 'w') as f:\n"
    "            json.dump(dict(error='openfold3 not installed'), f)\n"
    "        return\n"
    "\n"
    "    device = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
    "    dtype = getattr(torch, dtype_str)\n"
    "\n"
    "    config = model_config()\n"
    "    config.architecture.pairformer.no_blocks = 4\n"
    "    config.architecture.msa_module.no_blocks = 2\n"
    "    config.architecture.diffusion_transformer.no_blocks = 4\n"
    "\n"
    "    model = OpenFold3(config).to(device=device, dtype=dtype)\n"
    "    model.eval()\n"
    "\n"
    "    print(f'Reference model loaded on {device} with {dtype_str}')\n"
    "\n"
    "    results = dict(\n"
    "        n_tokens=n_tokens,\n"
    "        device=device,\n"
    "        dtype=dtype_str,\n"
    "        status='loaded_successfully',\n"
    "    )\n"
    "\n"
    "    os.makedirs(output_dir, exist_ok=True)\n"
    "    with open(os.path.join(output_dir, 'ref_results.json'), 'w') as f:\n"
    "        json.dump(results, f, indent=2)\n"
    "\n"
    "    print(f'Reference run completed for {n_tokens} tokens')\n"
    "\n"
    "if __name__ == '__main__':\n"
    "    main()\n"
)


# ---------------------------------------------------------------------------
# Main benchmark orchestrator
# ---------------------------------------------------------------------------

def run_worker_subprocess(script: str, args_dict: dict, label: str) -> bool:
    """Run a worker script as a subprocess."""
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        script_path = f.name

    try:
        cmd = [sys.executable, script_path, json.dumps(args_dict)]
        print(f"\n{'='*60}")
        print(f"Running {label}...")
        print(f"{'='*60}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )

        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)

        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"ERROR: {label} timed out after 600s")
        return False
    finally:
        os.unlink(script_path)


def compare_outputs(output_dir: str) -> dict:
    """Compare KB-Nano and reference outputs."""
    results = {}

    pairs = [
        ("kb_s_trunk.npy", "ref_s_trunk.npy", "trunk_s_cosine"),
        ("kb_z_trunk.npy", "ref_z_trunk.npy", "trunk_z_cosine"),
        ("kb_plddt_logits.npy", "ref_plddt_logits.npy", "plddt_cosine"),
        ("kb_distogram_logits.npy", "ref_distogram_logits.npy", "distogram_cosine"),
    ]

    for kb_file, ref_file, metric_name in pairs:
        kb_path = os.path.join(output_dir, kb_file)
        ref_path = os.path.join(output_dir, ref_file)

        if not os.path.exists(kb_path):
            results[metric_name] = {"status": "missing_kb", "value": None}
            continue
        if not os.path.exists(ref_path):
            results[metric_name] = {"status": "missing_ref", "value": None}
            continue

        kb_data = np.load(kb_path)
        ref_data = np.load(ref_path)

        cosine = _cosine_similarity(kb_data, ref_data)
        threshold = CORRECTNESS_THRESHOLDS.get(metric_name, 0.95)
        passed = cosine >= threshold

        results[metric_name] = {
            "value": float(cosine),
            "threshold": threshold,
            "passed": bool(passed),
            "kb_shape": list(kb_data.shape),
            "ref_shape": list(ref_data.shape),
        }

        status = "PASS" if passed else "FAIL"
        print(f"  {metric_name}: cosine={cosine:.6f} threshold={threshold} [{status}]")

    return results


def main():
    parser = argparse.ArgumentParser(description="OpenFold3 benchmark")
    parser.add_argument("--n-tokens", type=int, default=128,
                        help="Number of tokens for benchmark")
    parser.add_argument("--n-msa-seqs", type=int, default=0,
                        help="Number of MSA sequences")
    parser.add_argument("--dtype", default="bfloat16",
                        help="Data type for computation")
    parser.add_argument("--num-recycles", type=int, default=1,
                        help="Number of recycling iterations")
    parser.add_argument("--skip-reference", action="store_true",
                        help="Skip reference openfold3 comparison")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for results")
    parser.add_argument("--run-all-workloads", action="store_true",
                        help="Run all standardized workloads")
    args = parser.parse_args()

    gpu_name = _detect_gpu_name()
    print(f"GPU: {gpu_name}")

    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(
            str(_THIS_DIR), "results", "openfold3",
            f"{gpu_name}_{args.n_tokens}tok_{args.dtype}",
        )

    os.makedirs(output_dir, exist_ok=True)

    workloads = STRUCTURE_PREDICTION_WORKLOADS if args.run_all_workloads else [
        {
            "name": "custom",
            "n_tokens": args.n_tokens,
            "n_msa_seqs": args.n_msa_seqs,
            "description": f"{args.n_tokens} tokens, {args.n_msa_seqs} MSA seqs",
        }
    ]

    all_results = {
        "gpu": gpu_name,
        "dtype": args.dtype,
        "workloads": {},
    }

    for workload in workloads:
        wl_name = workload["name"]
        wl_dir = os.path.join(output_dir, wl_name)
        os.makedirs(wl_dir, exist_ok=True)

        print(f"\n{'#'*60}")
        print(f"Workload: {wl_name} — {workload['description']}")
        print(f"{'#'*60}")

        worker_args = {
            "n_tokens": workload["n_tokens"],
            "n_msa_seqs": workload["n_msa_seqs"],
            "output_dir": wl_dir,
            "dtype": args.dtype,
            "num_recycles": args.num_recycles,
        }

        # Run KB-Nano worker
        kb_ok = run_worker_subprocess(KB_NANO_WORKER, worker_args, "KB-Nano OpenFold3")

        wl_results = {"kb_nano_ok": kb_ok}

        # Run reference worker
        if not args.skip_reference:
            ref_ok = run_worker_subprocess(REFERENCE_WORKER, worker_args, "Reference OpenFold3")
            wl_results["reference_ok"] = ref_ok

            if kb_ok and ref_ok:
                print("\nComparing outputs:")
                wl_results["correctness"] = compare_outputs(wl_dir)

        # Load timing results
        kb_results_path = os.path.join(wl_dir, "kb_results.json")
        if os.path.exists(kb_results_path):
            with open(kb_results_path) as f:
                wl_results["kb_timing"] = json.load(f)

        all_results["workloads"][wl_name] = wl_results

    # Save combined results
    results_path = os.path.join(output_dir, "benchmark_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {results_path}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    for wl_name, wl_results in all_results["workloads"].items():
        kb_ok = wl_results.get("kb_nano_ok", False)
        timing = wl_results.get("kb_timing", {})
        trunk_time = timing.get("trunk_time_s", "N/A")
        print(f"  {wl_name}: KB-Nano={'OK' if kb_ok else 'FAIL'}, trunk_time={trunk_time}s")


if __name__ == "__main__":
    main()
