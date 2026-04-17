"""Benchmark kernel-level InstantNGP field primitives vs direct tinycudann."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_KB_ROOT = Path(__file__).resolve().parents[1]
if str(_KB_ROOT) not in sys.path:
    sys.path.insert(0, str(_KB_ROOT))

from bench.utils.worker import run_worker


INSTANTNGP_KERNEL_WORKER = r'''
import json, sys, time
import torch
import torch.nn.functional as F

with open(sys.argv[1]) as f:
    cfg = json.load(f)

sys.path.insert(0, cfg["kb_root"])

import tinycudann as tcnn
from tasks.baseline.L2.instantngp_field import InstantNGPField

HASHGRID_CONFIG = {
    "otype": "HashGrid",
    "n_levels": 8,
    "n_features_per_level": 4,
    "log2_hashmap_size": 19,
    "base_resolution": 16,
}

DIR_CONFIG = {
    "otype": "Composite",
    "nested": [
        {
            "n_dims_to_encode": 3,
            "otype": "SphericalHarmonics",
            "degree": 4,
        },
        {
            "otype": "Identity",
        },
    ],
}

DENSITY_CONFIG = {
    "otype": "FullyFusedMLP",
    "activation": "ReLU",
    "output_activation": "None",
    "n_neurons": 64,
    "n_hidden_layers": 1,
}

RGB_CONFIG = {
    "otype": "FullyFusedMLP",
    "activation": "ReLU",
    "output_activation": "None",
    "n_neurons": 64,
    "n_hidden_layers": 2,
}

class ReferenceField(torch.nn.Module):
    def __init__(self, seed: int):
        super().__init__()
        self.position_encoding = tcnn.Encoding(3, HASHGRID_CONFIG, seed=seed, dtype=torch.float16)
        self.density_mlp = tcnn.Network(self.position_encoding.n_output_dims, 16, DENSITY_CONFIG, seed=seed)
        self.direction_encoding = tcnn.Encoding(3, DIR_CONFIG, seed=seed, dtype=torch.float16)
        self.rgb_mlp = tcnn.Network(self.direction_encoding.n_output_dims + 16, 3, RGB_CONFIG, seed=seed)

    def forward(self, positions: torch.Tensor, directions: torch.Tensor):
        density_features = self.density_mlp(self.position_encoding(positions))
        sigma = density_features[:, :1]
        geo_feat = density_features[:, 1:]
        rgb = self.rgb_mlp(torch.cat([density_features, self.direction_encoding(directions)], dim=-1))
        return sigma, geo_feat, rgb

def sample_inputs(num_samples: int, device: str):
    g = torch.Generator(device=device)
    g.manual_seed(1234)
    positions = torch.rand(num_samples, 3, device=device, dtype=torch.float32, generator=g)
    directions = torch.randn(num_samples, 3, device=device, dtype=torch.float32, generator=g)
    directions = F.normalize(directions, dim=-1)
    return positions, directions

def measure(fn, positions, directions, warmup_iters, measure_iters):
    for _ in range(warmup_iters):
        fn(positions, directions)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(measure_iters):
        fn(positions, directions)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return (measure_iters * positions.shape[0]) / elapsed

def main():
    device = "cuda"
    positions, directions = sample_inputs(cfg["num_samples"], device=device)

    ours = InstantNGPField(seed=cfg["seed"]).to(device).eval()
    ref = ReferenceField(seed=cfg["seed"]).to(device).eval()

    ref.position_encoding.load_state_dict(ours.position_encoding.encoding.state_dict())
    ref.density_mlp.load_state_dict(ours.density_mlp.network.state_dict())
    ref.direction_encoding.load_state_dict(ours.direction_encoding.encoding.state_dict())
    ref.rgb_mlp.load_state_dict(ours.rgb_mlp.network.state_dict())

    with torch.inference_mode():
        ours_out = ours(positions, directions)
        ref_sigma, ref_geo, ref_rgb = ref(positions, directions)

    ours_ips = measure(lambda p, d: ours(p, d), positions, directions, cfg["warmup_iters"], cfg["measure_iters"])
    ref_ips = measure(lambda p, d: ref(p, d), positions, directions, cfg["warmup_iters"], cfg["measure_iters"])

    def cosine(a, b):
        return F.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).item()

    def mae(a, b):
        return torch.mean(torch.abs(a - b)).item()

    result = {
        "num_samples": cfg["num_samples"],
        "ours": {
            "baseline_name": "kb-nano-kernel",
            "samples_per_second": ours_ips,
        },
        "reference": {
            "baseline_name": "tinycudann-direct",
            "samples_per_second": ref_ips,
        },
        "comparison": {
            "throughput_ratio": ours_ips / ref_ips if ref_ips > 0 else float("inf"),
            "sigma_cosine": cosine(ours_out.sigma.float(), ref_sigma.float()),
            "sigma_mae": mae(ours_out.sigma.float(), ref_sigma.float()),
            "geo_feat_cosine": cosine(ours_out.geo_feat.float(), ref_geo.float()),
            "geo_feat_mae": mae(ours_out.geo_feat.float(), ref_geo.float()),
            "rgb_cosine": cosine(ours_out.rgb.float(), ref_rgb.float()),
            "rgb_mae": mae(ours_out.rgb.float(), ref_rgb.float()),
        },
    }

    with open(cfg["output_file"], "w") as f:
        json.dump(result, f)

if __name__ == "__main__":
    main()
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark InstantNGP kernel-level field")
    parser.add_argument("--num-samples", type=int, default=131072)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--measure-iters", type=int, default=50)
    parser.add_argument("--output-dir", default="/tmp/instantngp_kernel_bench")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    cfg = {
        "kb_root": str(_KB_ROOT),
        "num_samples": args.num_samples,
        "seed": args.seed,
        "warmup_iters": args.warmup_iters,
        "measure_iters": args.measure_iters,
    }
    data = run_worker(
        INSTANTNGP_KERNEL_WORKER,
        cfg,
        "InstantNGP kernel benchmark",
        timeout=7200,
    )
    if data is None:
        raise SystemExit(1)
    output_path = os.path.join(args.output_dir, "results.json")
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(json.dumps(data, indent=2))
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
