"""Benchmark 3D Gaussian Splatting render throughput and alignment."""

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

THREEDGS_WORKER = r'''
import json, sys, time
import torch

with open(sys.argv[1]) as f:
    cfg = json.load(f)

sys.path.insert(0, cfg["kb_root"])

from infra.graphics_loader import load_poster_scene, load_ours_3dgs, load_reference_3dgs

def render(model):
    with torch.inference_mode():
        out = model(return_meta=False)
    if isinstance(out, tuple) and len(out) == 3:
        return out
    raise RuntimeError(f"Unexpected 3DGS output format: {type(out)}")

def measure(model, warmup_iters, measure_iters):
    for _ in range(warmup_iters):
        render(model)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(measure_iters):
        render(model)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return measure_iters / elapsed

def main():
    device = cfg["device"]
    dtype = torch.float32
    scene = load_poster_scene(
        scene_name=cfg["scene"],
        num_cameras=cfg["num_cameras"],
        max_points=cfg["max_points"],
        device=device,
        dtype=dtype,
    )
    ours = load_ours_3dgs(scene)
    if hasattr(ours, "prepare_graph"):
        try:
            ours.prepare_graph()
        except Exception:
            pass
    ref = None if cfg["skip_reference"] else load_reference_3dgs(scene)

    result = {
        "scene": scene.scene_name,
        "num_cameras": cfg["num_cameras"],
        "max_points": cfg["max_points"],
        "resolution": [scene.height, scene.width],
        "dtype": str(dtype).replace("torch.", ""),
    }

    ours_ips = measure(ours, cfg["warmup_iters"], cfg["measure_iters"])
    result["ours"] = {"baseline_name": "kb-nano", "images_per_second": ours_ips}

    if ref is not None:
        ref_ips = measure(ref, cfg["warmup_iters"], cfg["measure_iters"])
        with torch.inference_mode():
            ours_rgb, ours_alpha, _ = render(ours)
            ref_rgb, ref_alpha, _ = render(ref)
            ours_rgb = ours_rgb.float()
            ref_rgb = ref_rgb.float()
            ours_alpha = ours_alpha.float()
            ref_alpha = ref_alpha.float()
        rgb_cos = torch.nn.functional.cosine_similarity(ours_rgb.reshape(1, -1), ref_rgb.reshape(1, -1)).item()
        rgb_mae = torch.mean(torch.abs(ours_rgb - ref_rgb)).item()
        alpha_cos = torch.nn.functional.cosine_similarity(ours_alpha.reshape(1, -1), ref_alpha.reshape(1, -1)).item()
        alpha_mae = torch.mean(torch.abs(ours_alpha - ref_alpha)).item()
        result["reference"] = {"baseline_name": "gsplat", "images_per_second": ref_ips}
        result["comparison"] = {
            "throughput_ratio": ours_ips / ref_ips if ref_ips > 0 else float("inf"),
            "rgb_cosine": rgb_cos,
            "rgb_mae": rgb_mae,
            "alpha_cosine": alpha_cos,
            "alpha_mae": alpha_mae,
            "image_shape": list(ours_rgb.shape),
        }

    with open(cfg["output_file"], "w") as f:
        json.dump(result, f)

if __name__ == "__main__":
    main()
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark 3D Gaussian Splatting")
    parser.add_argument("--scene", default="poster")
    parser.add_argument("--num-cameras", type=int, default=2)
    parser.add_argument("--max-points", type=int, default=8000)
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--measure-iters", type=int, default=20)
    parser.add_argument("--use-fp16", action="store_true")
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument("--output-dir", default="/tmp/3dgs_bench")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = {
        "kb_root": str(_KB_ROOT),
        "device": device,
        "scene": args.scene,
        "num_cameras": args.num_cameras,
        "max_points": args.max_points,
        "warmup_iters": args.warmup_iters,
        "measure_iters": args.measure_iters,
        "use_fp16": bool(args.use_fp16 and device == "cuda"),
        "skip_reference": args.skip_reference,
    }
    data = run_worker(THREEDGS_WORKER, cfg, "3DGS benchmark", timeout=7200)
    if data is None:
        raise SystemExit(1)
    output_path = os.path.join(args.output_dir, "results.json")
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(json.dumps(data, indent=2))
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
