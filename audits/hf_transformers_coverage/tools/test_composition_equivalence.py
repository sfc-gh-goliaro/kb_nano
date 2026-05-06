"""Composition-equivalence tests for the 8 new L1 ops that COULD have been
treated as 'composable from existing kb-nano primitives' (analogous to how
nn.MultiheadAttention is treated).

These tests prove that the 8 ops are mathematically equivalent to compositions
of pre-existing kb-nano L1 ops. The new L1 files are therefore stylistic
wrappers, not new compute primitives — which means the audit's `composable`
classification for those rows is justifiable EITHER WAY (with or without the
new L1 file).

The remaining 8 new L1 ops (ConvTranspose 1d/2d/3d, AdaptiveAvgPool 1d/2d,
GridSample, LSTM, ChunkGatedDeltaRule) are genuinely new primitives that
cannot be trivially composed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# Pre-existing kb-nano L1 ops (the "composition basis"):
from tasks.baseline.L1.batch_norm2d import BatchNorm2d as KBNanoBatchNorm2d
from tasks.baseline.L1.max_pool2d import MaxPool2d as KBNanoMaxPool2d
from tasks.baseline.L1.avg_pool2d import AvgPool2d as KBNanoAvgPool2d
from tasks.baseline.L1.relu import ReLU as KBNanoReLU
from tasks.baseline.L1.sigmoid import Sigmoid as KBNanoSigmoid

# The new L1 ops (which we claim are equivalent to compositions of the above):
from tasks.baseline.L1.batch_norm1d import BatchNorm1d
from tasks.baseline.L1.batch_norm3d import BatchNorm3d
from tasks.baseline.L1.max_pool1d import MaxPool1d
from tasks.baseline.L1.avg_pool1d import AvgPool1d
from tasks.baseline.L1.leaky_relu import LeakyReLU
from tasks.baseline.L1.elu import ELU
from tasks.baseline.L1.hardsigmoid import Hardsigmoid
from tasks.baseline.L1.hardswish import Hardswish


def _check(name: str, kb_out: torch.Tensor, comp_out: torch.Tensor, tol: float = 1e-5):
    if kb_out.shape != comp_out.shape:
        raise AssertionError(f"{name}: shape mismatch {kb_out.shape} vs {comp_out.shape}")
    diff = (kb_out - comp_out).abs().max().item()
    status = "PASS" if diff <= tol else "FAIL"
    print(f"  {name:60s} {status}  (max-abs diff {diff:.2e})")
    if diff > tol:
        raise AssertionError(f"{name}: diff {diff} > tol {tol}")


def main():
    torch.manual_seed(0)
    print("=== 8 'composable' ops: equivalence between new L1 wrapper and composition ===\n")

    # 1. BatchNorm1d via reshape + BatchNorm2d
    bn1 = BatchNorm1d(8).eval()
    bn2 = KBNanoBatchNorm2d(8).eval()
    bn2.load_state_dict(bn1.state_dict())  # share weights
    x = torch.randn(2, 8, 16)
    new = bn1(x)
    composed = bn2(x.unsqueeze(-1)).squeeze(-1)  # B,C,L -> B,C,L,1 -> bn2d -> B,C,L
    _check("BatchNorm1d == reshape + BatchNorm2d + reshape", new, composed)

    # 2. BatchNorm3d via reshape + BatchNorm2d
    bn3 = BatchNorm3d(4).eval()
    bn2 = KBNanoBatchNorm2d(4).eval()
    bn2.load_state_dict(bn3.state_dict())
    x = torch.randn(2, 4, 4, 8, 8)  # B,C,D,H,W
    new = bn3(x)
    # BN3d normalizes per-channel across (B*D*H*W). BN2d on view (B,C,D*H*W,1) does the same per-channel.
    B, C, D, H, W = x.shape
    composed = bn2(x.reshape(B, C, D * H * W, 1)).reshape(B, C, D, H, W)
    _check("BatchNorm3d == reshape + BatchNorm2d + reshape", new, composed)

    # 3. MaxPool1d via MaxPool2d
    mp1 = MaxPool1d(3, stride=2)
    mp2 = KBNanoMaxPool2d((1, 3), stride=(1, 2))
    x = torch.randn(2, 4, 32)
    new = mp1(x)
    composed = mp2(x.unsqueeze(-2)).squeeze(-2)
    _check("MaxPool1d == reshape + MaxPool2d (1, k) + reshape", new, composed)

    # 4. AvgPool1d via AvgPool2d
    ap1 = AvgPool1d(3, stride=2)
    ap2 = KBNanoAvgPool2d((1, 3), stride=(1, 2))
    x = torch.randn(2, 4, 32)
    new = ap1(x)
    composed = ap2(x.unsqueeze(-2)).squeeze(-2)
    _check("AvgPool1d == reshape + AvgPool2d (1, k) + reshape", new, composed)

    # 5. LeakyReLU via where + arithmetic
    lr = LeakyReLU(0.05)
    x = torch.randn(8, 16) - 0.2
    new = lr(x)
    composed = torch.where(x > 0, x, 0.05 * x)
    _check("LeakyReLU == where(x > 0, x, alpha * x)", new, composed)

    # 6. ELU via where + exp + arithmetic
    elu = ELU(1.0)
    x = torch.randn(8, 16) - 0.2
    new = elu(x)
    composed = torch.where(x >= 0, x, 1.0 * (torch.exp(x) - 1))
    _check("ELU == where(x >= 0, x, alpha * (exp(x) - 1))", new, composed)

    # 7. Hardsigmoid via clamp
    hs = Hardsigmoid()
    x = torch.randn(8, 16) * 4
    new = hs(x)
    composed = torch.clamp((x + 3) / 6, 0, 1)
    _check("Hardsigmoid == clamp((x + 3) / 6, 0, 1)", new, composed)

    # 8. Hardswish via x * hardsigmoid(x)
    hsw = Hardswish()
    x = torch.randn(8, 16) * 4
    new = hsw(x)
    composed = x * torch.clamp((x + 3) / 6, 0, 1)
    _check("Hardswish == x * hardsigmoid(x)", new, composed)

    print("\nConclusion: the 8 new L1 wrappers are equivalent to compositions")
    print("of pre-existing kb-nano L1 ops + standard torch arithmetic.")
    print("Adding them as L1 files is a stylistic choice (consistent with the")
    print("existing pattern of one L1 file per torch.nn op), not a new primitive.")


if __name__ == "__main__":
    main()
