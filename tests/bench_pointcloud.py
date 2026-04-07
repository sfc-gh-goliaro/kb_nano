"""Benchmark PointTransformerV3 vs official detached implementation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_KB_ROOT = Path(__file__).resolve().parents[1]
if str(_KB_ROOT) not in sys.path:
    sys.path.insert(0, str(_KB_ROOT))

import torch

from bench.utils.worker import run_worker

POINTCLOUD_WORKER = r'''
import json, os, random, sys, time
import numpy as np
import torch

with open(sys.argv[1]) as f:
    cfg = json.load(f)

sys.path.insert(0, cfg["kb_root"])

from infra.pointcloud_loader import (
    default_ptv3_kwargs,
    load_ours_point_model,
    load_reference_point_model,
)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_batch(points_per_cloud: int, batch_size: int, feat_dim: int, device: str, dtype: torch.dtype, grid_size: float):
    total = points_per_cloud * batch_size
    coord = torch.rand(total, 3, device=device, dtype=dtype) * 10.0
    feat = torch.rand(total, feat_dim, device=device, dtype=dtype)
    counts = torch.full((batch_size,), points_per_cloud, device=device, dtype=torch.long)
    offset = torch.cumsum(counts, dim=0)
    return {"coord": coord, "feat": feat, "grid_size": grid_size, "offset": offset}

def forward_model(model, batch):
    out = model(batch)
    return out.feat if hasattr(out, "feat") else out["feat"]

def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)
    device = cfg["device"]
    dtype = torch.float16 if cfg["use_fp16"] else torch.float32
    enable_flash = bool(cfg.get("enable_flash", False))
    model_kwargs = default_ptv3_kwargs(enable_flash=enable_flash)

    set_seed(cfg["seed"])
    ours = load_ours_point_model(cfg["model"], device=device, dtype=dtype, **model_kwargs)
    set_seed(cfg["seed"])
    ref = None
    if not cfg.get("skip_reference", False):
        ref = load_reference_point_model(cfg["model"], device=device, dtype=dtype, **model_kwargs)
        missing, unexpected = ref.load_state_dict(ours.state_dict(), strict=True)
        if missing or unexpected:
            raise RuntimeError(f"PTv3 state mismatch missing={missing} unexpected={unexpected}")

    throughput_batch = make_batch(
        cfg["points_per_cloud"],
        cfg["batch_size"],
        cfg["feat_dim"],
        device,
        dtype,
        cfg["grid_size"],
    )
    align_batch = make_batch(
        cfg["alignment_points_per_cloud"],
        cfg["alignment_batch_size"],
        cfg["feat_dim"],
        device,
        dtype,
        cfg["grid_size"],
    )

    def measure(model):
        for _ in range(cfg["warmup_iters"]):
            forward_model(model, throughput_batch)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(cfg["measure_iters"]):
            forward_model(model, throughput_batch)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        total_points = cfg["points_per_cloud"] * cfg["batch_size"] * cfg["measure_iters"]
        return total_points / elapsed

    result = {
        "model": cfg["model"],
        "points_per_cloud": cfg["points_per_cloud"],
        "batch_size": cfg["batch_size"],
        "enable_flash": enable_flash,
    }

    ours_tps = measure(ours)
    result["ours"] = {"baseline_name": "kb-nano", "points_per_second": ours_tps}

    if ref is not None:
        ref_tps = measure(ref)
        with torch.no_grad():
            ours_feat = forward_model(ours, align_batch).float()
            ref_feat = forward_model(ref, align_batch).float()
        feat_cos = torch.nn.functional.cosine_similarity(
            ours_feat.reshape(1, -1), ref_feat.reshape(1, -1)
        ).item()
        feat_mae = torch.mean(torch.abs(ours_feat - ref_feat)).item()
        result["reference"] = {"baseline_name": "official-detached", "points_per_second": ref_tps}
        result["comparison"] = {
            "throughput_ratio": ours_tps / ref_tps if ref_tps > 0 else float("inf"),
            "feat_cosine": feat_cos,
            "feat_mae": feat_mae,
            "feat_shape": list(ours_feat.shape),
        }

    with open(cfg["output_file"], "w") as f:
        json.dump(result, f)

if __name__ == "__main__":
    main()
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark PointTransformerV3")
    parser.add_argument("--model", default="Pointcept/PointTransformerV3")
    parser.add_argument("--points-per-cloud", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--alignment-points-per-cloud", type=int, default=2048)
    parser.add_argument("--alignment-batch-size", type=int, default=1)
    parser.add_argument("--feat-dim", type=int, default=6)
    parser.add_argument("--grid-size", type=float, default=0.05)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--measure-iters", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--use-fp16", action="store_true")
    parser.add_argument("--enable-flash", action="store_true")
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument("--output-dir", default="/tmp/pointtransv3_bench")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = {
        "kb_root": str(_KB_ROOT),
        "model": args.model,
        "device": device,
        "points_per_cloud": args.points_per_cloud,
        "batch_size": args.batch_size,
        "alignment_points_per_cloud": args.alignment_points_per_cloud,
        "alignment_batch_size": args.alignment_batch_size,
        "feat_dim": args.feat_dim,
        "grid_size": args.grid_size,
        "warmup_iters": args.warmup_iters,
        "measure_iters": args.measure_iters,
        "seed": args.seed,
        "use_fp16": bool(args.use_fp16 and device == "cuda"),
        "enable_flash": args.enable_flash,
        "skip_reference": args.skip_reference,
    }
    data = run_worker(POINTCLOUD_WORKER, cfg, "PointTransformerV3 benchmark", timeout=7200)
    if data is None:
        raise SystemExit(1)
    output_path = os.path.join(args.output_dir, "results.json")
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(json.dumps(data, indent=2))
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
