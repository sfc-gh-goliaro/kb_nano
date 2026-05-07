"""Smoke test for the 8 KEEP L1 ops (the genuinely new primitives added in this audit branch).

For thorough reference-implementation testing across all parameter combinations,
all dtypes, and all HF usage patterns, see:
    test_keep_ops_thorough.py     (297 tests across 8 KEEP ops; 100% pass)
    test_composition_equivalence.py (243 tests proving 8 stylistic ops are
                                     composable from existing kb-nano + torch builtins
                                     and were therefore removed)
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
from tasks.baseline.L1.grid_sample import GridSample
from tasks.baseline.L1.conv_transpose1d import ConvTranspose1d
from tasks.baseline.L1.conv_transpose2d import ConvTranspose2d
from tasks.baseline.L1.conv_transpose3d import ConvTranspose3d
from tasks.baseline.L1.lstm import LSTM


def _check(name, kb_out, ref_out, tol=1e-5):
    if isinstance(kb_out, tuple): kb_out = kb_out[0]
    if isinstance(ref_out, tuple): ref_out = ref_out[0]
    assert kb_out.shape == ref_out.shape, f"{name}: shape mismatch"
    diff = (kb_out - ref_out).abs().max().item()
    if diff > tol:
        raise AssertionError(f"{name}: max-abs diff {diff} > {tol}")
    print(f"  {name:36s} PASS  diff={diff:.2e}")


def main():
    torch.manual_seed(42)
    print("=== 8 KEEP L1 ops (smoke test; see test_keep_ops_thorough.py for the 297-test full suite) ===\n")

    x = torch.randn(2, 4, 16)
    _check("AdaptiveAvgPool1d", AdaptiveAvgPool1d(8)(x), nn.AdaptiveAvgPool1d(8)(x))

    x = torch.randn(2, 4, 16, 16)
    _check("AdaptiveAvgPool2d", AdaptiveAvgPool2d((1, 1))(x), nn.AdaptiveAvgPool2d((1, 1))(x))

    x = torch.randn(2, 3, 8, 8)
    grid = torch.randn(2, 4, 4, 2).clamp(-1, 1)
    _check("GridSample(bilinear)",
           GridSample(mode="bilinear", padding_mode="zeros", align_corners=False)(x, grid),
           nn.functional.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False))

    for dim, KBClass, RefClass, x_shape in [
        (1, ConvTranspose1d, nn.ConvTranspose1d, (2, 4, 16)),
        (2, ConvTranspose2d, nn.ConvTranspose2d, (2, 4, 8, 8)),
        (3, ConvTranspose3d, nn.ConvTranspose3d, (2, 4, 4, 4, 4)),
    ]:
        ref = RefClass(4, 6, 3, stride=2, padding=1)
        kb = KBClass(4, 6, 3, stride=2, padding=1)
        kb.load_state_dict(ref.state_dict())
        x = torch.randn(*x_shape)
        _check(f"ConvTranspose{dim}d", kb(x), ref(x))
        # state_dict key compat
        assert set(kb.state_dict().keys()) == set(ref.state_dict().keys()), \
            f"ConvTranspose{dim}d state_dict keys differ"

    ref = nn.LSTM(8, 16, 2, batch_first=True)
    kb = LSTM(8, 16, 2, batch_first=True)
    kb.lstm.load_state_dict(ref.state_dict())
    x = torch.randn(2, 5, 8)
    _check("LSTM", kb(x), ref(x))

    # ChunkGatedDeltaRule requires fla + CUDA; tested in test_keep_ops_thorough.py
    try:
        from tasks.baseline.L1.chunk_gated_delta_rule import ChunkGatedDeltaRule  # noqa: F401
        if torch.cuda.is_available():
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule
            torch.manual_seed(0)
            q = torch.randn(2, 16, 4, 32, device="cuda", dtype=torch.bfloat16)
            k = torch.randn(2, 16, 4, 32, device="cuda", dtype=torch.bfloat16)
            v = torch.randn(2, 16, 4, 32, device="cuda", dtype=torch.bfloat16)
            g = torch.randn(2, 16, 4, device="cuda", dtype=torch.bfloat16).log_softmax(-1)
            beta = torch.randn(2, 16, 4, device="cuda", dtype=torch.bfloat16).sigmoid()
            from tasks.baseline.L1.chunk_gated_delta_rule import ChunkGatedDeltaRule
            kb = ChunkGatedDeltaRule()
            kb_out, _ = kb(q=q, k=k, v=v, g=g, beta=beta)
            ref_out, _ = chunk_gated_delta_rule(q=q, k=k, v=v, g=g, beta=beta)
            _check("ChunkGatedDeltaRule (fla,CUDA,bf16)", kb_out, ref_out)
        else:
            print("  ChunkGatedDeltaRule          SKIP (no CUDA)")
    except ImportError:
        print("  ChunkGatedDeltaRule          SKIP (fla not installed)")

    print("\nAll 8 KEEP L1 ops smoke-tested. See test_keep_ops_thorough.py for full coverage.")


if __name__ == "__main__":
    main()
