#!/usr/bin/env python3
"""Benchmark repo-native detection models against official baselines."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PACKAGE_DIR = _THIS_DIR.parent
sys.path.insert(0, str(_PACKAGE_DIR))

from bench.utils.worker import run_worker
from infra.detection_loader import infer_image_size


def _detect_gpu_name() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
        ).strip().splitlines()[0]
        for tag in ("B200", "B100", "H200", "H100", "A100", "L40S", "L40", "L4"):
            if tag in out:
                return tag
        return out.split()[-1]
    except Exception:
        return "unknown"


DETECTION_WORKER = r'''
import json, os, sys, time
import importlib.util
from pathlib import Path
import torch

with open(sys.argv[1]) as f:
    cfg = json.load(f)
sys.path.insert(0, cfg["project_root"])

pkg_root = Path(cfg["project_root"])
spec = importlib.util.spec_from_file_location(
    "kb_nano", pkg_root / "__init__.py",
    submodule_search_locations=[str(pkg_root)],
)
kb_nano = importlib.util.module_from_spec(spec)
sys.modules["kb_nano"] = kb_nano
spec.loader.exec_module(kb_nano)

from kb_nano.infra.detection_loader import (
    load_ours_detector,
    load_reference_detector,
    run_ours_detector,
    run_reference_detector,
)


def _run_model(model, backend, model_name, pixel_values, image_size, max_detections):
    if backend == "ours":
        return run_ours_detector(model, model_name, pixel_values, image_size, max_detections=max_detections)
    return run_reference_detector(model, model_name, pixel_values, image_size, max_detections=max_detections)


def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)

    device = cfg.get("device", "cuda")
    dtype = torch.float16 if cfg.get("use_fp16", True) else torch.float32
    model_name = cfg["model"]
    backend = cfg["backend"]
    image_size = cfg["image_size"]
    batch_size = cfg["batch_size"]
    num_images = cfg["num_images"]
    max_detections = cfg["max_detections"]

    g = torch.Generator(device=device)
    g.manual_seed(cfg["seed"])
    pixel_values = torch.rand(
        num_images, 3, image_size, image_size, device=device, dtype=dtype, generator=g
    )

    if backend == "ours":
        model = load_ours_detector(model_name, device=device, dtype=dtype)
        baseline_name = "kb-nano"
    else:
        model, baseline_name = load_reference_detector(model_name, device=device, dtype=dtype)

    det = _run_model(model, backend, model_name, pixel_values[:1], image_size, max_detections)

    for _ in range(cfg["warmup_iters"]):
        for start in range(0, num_images, batch_size):
            _ = _run_model(model, backend, model_name, pixel_values[start:start + batch_size], image_size, max_detections)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(cfg["measure_iters"]):
        for start in range(0, num_images, batch_size):
            _ = _run_model(model, backend, model_name, pixel_values[start:start + batch_size], image_size, max_detections)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    result = {
        "backend": backend,
        "baseline_name": baseline_name,
        "elapsed": elapsed / cfg["measure_iters"],
        "images_per_second": num_images / (elapsed / cfg["measure_iters"]),
        "boxes_shape": list(det["boxes"].shape),
        "scores_shape": list(det["scores"].shape),
        "labels_shape": list(det["labels"].shape),
    }

    latencies = {}
    for bs in cfg.get("latency_batch_sizes", []):
        batch = pixel_values[:bs]
        samples = []
        for _ in range(cfg["latency_iters"]):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = _run_model(model, backend, model_name, batch, image_size, max_detections)
            torch.cuda.synchronize()
            samples.append(time.perf_counter() - t0)
        latencies[str(bs)] = {
            "median": sorted(samples)[len(samples) // 2],
            "mean": sum(samples) / len(samples),
        }
    result["latency"] = latencies

    align_batch = pixel_values[:cfg["alignment_images"]]
    align_det = _run_model(model, backend, model_name, align_batch, image_size, max_detections)
    result["alignment_boxes"] = align_det["boxes"].float().cpu().tolist()
    result["alignment_scores"] = align_det["scores"].float().cpu().tolist()
    result["alignment_labels"] = align_det["labels"].cpu().tolist()

    with open(cfg["output_file"], "w") as f:
        json.dump(result, f)
        f.flush()
        os.fsync(f.fileno())
    os._exit(0)


if __name__ == "__main__":
    main()
'''


def _cosine(a, b):
    import numpy as np

    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 and nb == 0.0:
        return 1.0
    denom = (na * nb) or 1.0
    return float(a @ b / denom)


def _mae(a, b):
    import numpy as np

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.mean(np.abs(a - b)))


def _label_match_rate(a, b):
    total = 0
    matched = 0
    for row_a, row_b in zip(a, b):
        for x, y in zip(row_a, row_b):
            total += 1
            matched += int(x == y)
    return matched / max(total, 1)


def _compare_outputs(ours, ref):
    latency_ratio = {}
    for bs, stats in ours.get("latency", {}).items():
        if bs in ref.get("latency", {}):
            latency_ratio[bs] = ref["latency"][bs]["median"] / stats["median"]
    return {
        "throughput_ratio": ours["images_per_second"] / ref["images_per_second"],
        "boxes_cosine": _cosine(ours["alignment_boxes"], ref["alignment_boxes"]),
        "boxes_mae": _mae(ours["alignment_boxes"], ref["alignment_boxes"]),
        "scores_cosine": _cosine(ours["alignment_scores"], ref["alignment_scores"]),
        "scores_mae": _mae(ours["alignment_scores"], ref["alignment_scores"]),
        "labels_match_rate": _label_match_rate(ours["alignment_labels"], ref["alignment_labels"]),
        "latency_ratio": latency_ratio,
    }


def _summarize_backend_result(result):
    if not result:
        return None
    return {
        "baseline_name": result.get("baseline_name"),
        "images_per_second": result.get("images_per_second"),
        "latency": result.get("latency", {}),
        "boxes_shape": result.get("boxes_shape"),
        "scores_shape": result.get("scores_shape"),
        "labels_shape": result.get("labels_shape"),
    }


def main():
    ap = argparse.ArgumentParser(description="Benchmark repo-native detection models vs official baselines")
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--image-size", type=int, default=0)
    ap.add_argument("--num-images", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--alignment-images", type=int, default=4)
    ap.add_argument("--max-detections", type=int, default=100)
    ap.add_argument("--latency-batch-sizes", type=str, default="1,4")
    ap.add_argument("--latency-iters", type=int, default=5)
    ap.add_argument("--warmup-iters", type=int, default=1)
    ap.add_argument("--measure-iters", type=int, default=3)
    ap.add_argument("--use-fp16", action="store_true")
    ap.add_argument("--skip-reference", action="store_true")
    ap.add_argument("--output-dir", type=str, default=None)
    args = ap.parse_args()

    image_size = args.image_size or infer_image_size(args.model)
    gpu = _detect_gpu_name()
    out_dir = Path(args.output_dir or f"tests/results/{gpu}/{Path(args.model).name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    common = {
        "project_root": str(_PACKAGE_DIR),
        "model": args.model,
        "image_size": image_size,
        "num_images": args.num_images,
        "batch_size": args.batch_size,
        "alignment_images": args.alignment_images,
        "max_detections": args.max_detections,
        "latency_batch_sizes": [int(x) for x in args.latency_batch_sizes.split(",") if x],
        "latency_iters": args.latency_iters,
        "warmup_iters": args.warmup_iters,
        "measure_iters": args.measure_iters,
        "use_fp16": args.use_fp16,
        "device": "cuda",
        "seed": 1234,
    }

    ours = None
    references = {}
    comparisons = {}
    try:
        ours = run_worker(DETECTION_WORKER, {**common, "backend": "ours"}, "kb-nano detection")
    except Exception:
        ours = None

    if not args.skip_reference:
        ref = run_worker(DETECTION_WORKER, {**common, "backend": "reference"}, "official detection baseline")
        references["reference"] = ref
        if ours and ref:
            comparisons["reference"] = _compare_outputs(ours, ref)

    results = {"model": args.model, "image_size": image_size, "ours": ours, "references": references, "comparisons": comparisons}
    out_file = out_dir / "results.json"
    out_file.write_text(json.dumps(results, indent=2))
    summary = {
        "model": args.model,
        "image_size": image_size,
        "ours": _summarize_backend_result(ours),
        "references": {name: _summarize_backend_result(result) for name, result in references.items()},
        "comparisons": comparisons,
    }
    print(json.dumps(summary, indent=2))
    print(f"\nSaved results to {out_file}")


if __name__ == "__main__":
    main()
