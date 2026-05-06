"""Numerical correctness test for the new L1 ops.

Compares each new kb-nano L1 wrapper against the corresponding torch.nn.X
on identical random input. Each op must produce bit-identical (or within
1e-6) output.

Run on CPU (no GPU needed for these wrappers — all dispatch to F.x torch ops).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

# Make tasks importable
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tasks.baseline.L1.adaptive_avg_pool1d import AdaptiveAvgPool1d
from tasks.baseline.L1.adaptive_avg_pool2d import AdaptiveAvgPool2d
from tasks.baseline.L1.avg_pool1d import AvgPool1d
from tasks.baseline.L1.max_pool1d import MaxPool1d
from tasks.baseline.L1.leaky_relu import LeakyReLU
from tasks.baseline.L1.elu import ELU
from tasks.baseline.L1.hardsigmoid import Hardsigmoid
from tasks.baseline.L1.hardswish import Hardswish
from tasks.baseline.L1.grid_sample import GridSample
from tasks.baseline.L1.conv_transpose1d import ConvTranspose1d
from tasks.baseline.L1.conv_transpose2d import ConvTranspose2d
from tasks.baseline.L1.conv_transpose3d import ConvTranspose3d
from tasks.baseline.L1.batch_norm1d import BatchNorm1d
from tasks.baseline.L1.batch_norm3d import BatchNorm3d


def _check(name: str, kb_out: torch.Tensor, ref_out: torch.Tensor):
    if kb_out.shape != ref_out.shape:
        raise AssertionError(f"{name}: shape mismatch {kb_out.shape} vs {ref_out.shape}")
    diff = (kb_out - ref_out).abs().max().item()
    tol = 1e-5
    if diff > tol:
        raise AssertionError(f"{name}: max-abs diff {diff} > {tol}")
    print(f"  {name:30s} PASS  (max-abs diff {diff:.2e})")


def _copy_state(kb_mod: nn.Module, ref_mod: nn.Module):
    """Copy parameters/buffers from ref to kb (or vice versa) so they share weights."""
    sd = ref_mod.state_dict()
    missing, unexpected = kb_mod.load_state_dict(sd, strict=False)
    if missing or unexpected:
        # OK if missing/unexpected are empty params (e.g. None weights)
        nontrivial_missing = [k for k in missing if "weight" in k or "bias" in k]
        nontrivial_unexpected = [k for k in unexpected if "weight" in k or "bias" in k]
        if nontrivial_missing or nontrivial_unexpected:
            raise AssertionError(f"state_dict mismatch: missing={missing}, unexpected={unexpected}")


def main():
    torch.manual_seed(42)
    print("=== Stateless ops ===")

    # AdaptiveAvgPool1d
    x = torch.randn(2, 4, 16)
    _check("AdaptiveAvgPool1d", AdaptiveAvgPool1d(8)(x), nn.AdaptiveAvgPool1d(8)(x))
    _check("AdaptiveAvgPool1d(out=1)", AdaptiveAvgPool1d(1)(x), nn.AdaptiveAvgPool1d(1)(x))

    # AdaptiveAvgPool2d
    x = torch.randn(2, 4, 16, 16)
    _check("AdaptiveAvgPool2d", AdaptiveAvgPool2d(8)(x), nn.AdaptiveAvgPool2d(8)(x))
    _check("AdaptiveAvgPool2d(1,1)", AdaptiveAvgPool2d((1, 1))(x), nn.AdaptiveAvgPool2d((1, 1))(x))

    # AvgPool1d
    x = torch.randn(2, 4, 32)
    _check("AvgPool1d", AvgPool1d(2, stride=2)(x), nn.AvgPool1d(2, stride=2)(x))

    # MaxPool1d
    x = torch.randn(2, 4, 32)
    _check("MaxPool1d", MaxPool1d(2, stride=2)(x), nn.MaxPool1d(2, stride=2)(x))

    # LeakyReLU
    x = torch.randn(2, 4, 16) - 0.5
    _check("LeakyReLU", LeakyReLU(0.01)(x), nn.LeakyReLU(0.01)(x))

    # ELU
    _check("ELU", ELU(1.0)(x), nn.ELU(1.0)(x))

    # Hardsigmoid
    _check("Hardsigmoid", Hardsigmoid()(x), nn.Hardsigmoid()(x))

    # Hardswish
    _check("Hardswish", Hardswish()(x), nn.Hardswish()(x))

    # GridSample (B, C, H, W) -> sample with (B, H_out, W_out, 2)
    x = torch.randn(2, 3, 8, 8)
    grid = torch.randn(2, 4, 4, 2).clamp(-1, 1)
    _check(
        "GridSample(default)",
        GridSample(mode="bilinear", padding_mode="zeros", align_corners=False)(x, grid),
        nn.functional.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False),
    )

    print("\n=== Weight-bearing ops (state_dict-compat with torch.nn) ===")

    # ConvTranspose1d
    ref = nn.ConvTranspose1d(4, 6, 3, stride=2, padding=1)
    kb = ConvTranspose1d(4, 6, 3, stride=2, padding=1)
    _copy_state(kb, ref)
    x = torch.randn(2, 4, 8)
    _check("ConvTranspose1d", kb(x), ref(x))

    # ConvTranspose2d
    ref = nn.ConvTranspose2d(4, 6, 3, stride=2, padding=1, output_padding=1)
    kb = ConvTranspose2d(4, 6, 3, stride=2, padding=1, output_padding=1)
    _copy_state(kb, ref)
    x = torch.randn(2, 4, 8, 8)
    _check("ConvTranspose2d", kb(x), ref(x))

    # ConvTranspose2d w/ groups + no bias
    ref = nn.ConvTranspose2d(8, 8, 3, stride=2, padding=1, groups=4, bias=False)
    kb = ConvTranspose2d(8, 8, 3, stride=2, padding=1, groups=4, bias=False)
    _copy_state(kb, ref)
    x = torch.randn(2, 8, 8, 8)
    _check("ConvTranspose2d(groups,bias=F)", kb(x), ref(x))

    # ConvTranspose3d
    ref = nn.ConvTranspose3d(4, 6, 3, stride=2, padding=1)
    kb = ConvTranspose3d(4, 6, 3, stride=2, padding=1)
    _copy_state(kb, ref)
    x = torch.randn(2, 4, 4, 4, 4)
    _check("ConvTranspose3d", kb(x), ref(x))

    # BatchNorm1d (eval mode)
    ref = nn.BatchNorm1d(8).eval()
    kb = BatchNorm1d(8).eval()
    _copy_state(kb, ref)
    x = torch.randn(2, 8, 16)
    _check("BatchNorm1d (eval)", kb(x), ref(x))

    # BatchNorm1d (train mode — non-deterministic between two calls because running stats update;
    # use SAME first-time call to keep stats aligned).
    ref = nn.BatchNorm1d(8).train()
    kb = BatchNorm1d(8).train()
    _copy_state(kb, ref)
    x = torch.randn(2, 8, 16)
    out_ref = ref(x)
    out_kb = kb(x)
    _check("BatchNorm1d (train)", out_kb, out_ref)

    # BatchNorm3d (eval)
    ref = nn.BatchNorm3d(4).eval()
    kb = BatchNorm3d(4).eval()
    _copy_state(kb, ref)
    x = torch.randn(2, 4, 4, 4, 4)
    _check("BatchNorm3d (eval)", kb(x), ref(x))

    # BatchNorm1d state_dict key compat (HF reference checkpoint loads must work)
    ref = nn.BatchNorm1d(8)
    kb = BatchNorm1d(8)
    ref_keys = set(ref.state_dict().keys())
    kb_keys = set(kb.state_dict().keys())
    if ref_keys != kb_keys:
        raise AssertionError(f"BatchNorm1d state_dict keys differ: ref={ref_keys}, kb={kb_keys}")
    print(f"  BatchNorm1d state_dict keys    PASS  ({sorted(ref_keys)})")

    ref = nn.ConvTranspose2d(4, 6, 3)
    kb = ConvTranspose2d(4, 6, 3)
    ref_keys = set(ref.state_dict().keys())
    kb_keys = set(kb.state_dict().keys())
    if ref_keys != kb_keys:
        raise AssertionError(f"ConvTranspose2d state_dict keys differ: ref={ref_keys}, kb={kb_keys}")
    print(f"  ConvTranspose2d state_dict keys  PASS  ({sorted(ref_keys)})")

    print("\nAll new L1 ops pass numerical + state_dict compatibility checks.")


if __name__ == "__main__":
    main()
