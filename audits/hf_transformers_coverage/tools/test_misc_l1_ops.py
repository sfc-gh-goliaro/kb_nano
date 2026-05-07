"""Tests for L1 wrappers added in this audit pass not covered by test_v3_ops.py
(which focuses on Conv1d/Pool1d/activations) or test_rg_lru.py.

Coverage:
- AdaptiveAvgPool1d, AdaptiveAvgPool2d
- ConvTranspose1d, ConvTranspose2d, ConvTranspose3d  (state_dict-compat with nn.ConvTranspose*)
- LSTM  (state_dict-compat with nn.LSTM)
- GridSample
- ChunkGatedDeltaRule, FusedRecurrentGatedDeltaRule  (pass-through to fla.ops.gated_delta_rule)

Run with venv activated. ChunkGatedDeltaRule tests require CUDA + fla installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tasks.baseline.L1.adaptive_avg_pool1d import AdaptiveAvgPool1d
from tasks.baseline.L1.adaptive_avg_pool2d import AdaptiveAvgPool2d
from tasks.baseline.L1.conv_transpose1d import ConvTranspose1d
from tasks.baseline.L1.conv_transpose2d import ConvTranspose2d
from tasks.baseline.L1.conv_transpose3d import ConvTranspose3d
from tasks.baseline.L1.lstm import LSTM
from tasks.baseline.L1.grid_sample import GridSample


PASS: list[tuple[str, float]] = []
FAIL: list[tuple[str, float]] = []


def check(name: str, kb, ref, tol: float = 0.0):
    if isinstance(kb, tuple):
        kb = kb[0]
    if isinstance(ref, tuple):
        ref = ref[0]
    if kb.shape != ref.shape:
        FAIL.append((name, -1.0))
        print(f"  FAIL  {name}: shape mismatch {kb.shape} vs {ref.shape}")
        return
    diff = (kb.float() - ref.float()).abs().max().item()
    if diff <= tol:
        PASS.append((name, diff))
        print(f"  PASS  {name:65s} diff={diff:.2e}")
    else:
        FAIL.append((name, diff))
        print(f"  FAIL  {name:65s} diff={diff:.2e} > {tol:.0e}")


def test_adaptive_avg_pool():
    for shape, out in [((4, 8, 32), 16), ((1, 4, 17), 5), ((2, 16, 9), 3)]:
        x = torch.randn(*shape)
        check(f"AdaptiveAvgPool1d/{shape}->{out}",
              AdaptiveAvgPool1d(out)(x), nn.AdaptiveAvgPool1d(out)(x), tol=1e-6)
    for shape, out in [((4, 8, 32, 32), (16, 16)), ((1, 4, 9, 7), (3, 2)),
                       ((2, 16, 17, 17), 5)]:
        x = torch.randn(*shape)
        check(f"AdaptiveAvgPool2d/{shape}->{out}",
              AdaptiveAvgPool2d(out)(x), nn.AdaptiveAvgPool2d(out)(x), tol=1e-6)


def test_conv_transpose():
    for kw in [dict(stride=2, padding=1),
               dict(stride=2, padding=1, output_padding=1),
               dict(stride=1, dilation=2, padding=2),
               dict(groups=2, bias=False)]:
        in_c, out_c, k = 4, 8, 3
        if kw.get("groups", 1) > 1:
            in_c, out_c = 8, 4
        ref = nn.ConvTranspose1d(in_c, out_c, k, **kw)
        kb = ConvTranspose1d(in_c, out_c, k, **kw)
        kb.load_state_dict(ref.state_dict())
        x = torch.randn(2, in_c, 16)
        check(f"ConvTranspose1d/{kw}", kb(x), ref(x), tol=1e-5)
    for kw in [dict(stride=2, padding=1),
               dict(stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),
               dict(groups=2, bias=False)]:
        in_c, out_c, k = 4, 8, 3
        if kw.get("groups", 1) > 1:
            in_c, out_c = 8, 4
        ref = nn.ConvTranspose2d(in_c, out_c, k, **kw)
        kb = ConvTranspose2d(in_c, out_c, k, **kw)
        kb.load_state_dict(ref.state_dict())
        x = torch.randn(2, in_c, 16, 16)
        check(f"ConvTranspose2d/{kw}", kb(x), ref(x), tol=1e-5)
    for kw in [dict(stride=2, padding=1),
               dict(stride=(2, 2, 2), padding=(1, 1, 1), output_padding=(1, 1, 1))]:
        ref = nn.ConvTranspose3d(2, 4, 3, **kw)
        kb = ConvTranspose3d(2, 4, 3, **kw)
        kb.load_state_dict(ref.state_dict())
        x = torch.randn(1, 2, 8, 8, 8)
        check(f"ConvTranspose3d/{kw}", kb(x), ref(x), tol=1e-5)


def test_lstm():
    """LSTM is an nn.Module wrapper around nn.LSTM. State_dict keys nested under 'lstm.';
    inner keys match nn.LSTM exactly. Numerical output identical to nn.LSTM."""
    for kw in [dict(num_layers=1), dict(num_layers=2), dict(bidirectional=True),
               dict(num_layers=2, bidirectional=True), dict(batch_first=True),
               dict(num_layers=1, bias=False)]:
        ref = nn.LSTM(8, 16, **kw)
        kb = LSTM(8, 16, **kw)
        # Inner state_dict (kb.lstm.state_dict()) must match nn.LSTM exactly.
        ref_keys = set(ref.state_dict().keys())
        inner_keys = set(kb.lstm.state_dict().keys())
        if ref_keys != inner_keys:
            FAIL.append((f"LSTM/{kw}/inner_keys", -1.0))
            print(f"  FAIL  LSTM/{kw}: inner keys differ ref={sorted(ref_keys)}, kb.lstm={sorted(inner_keys)}")
            continue
        kb.lstm.load_state_dict(ref.state_dict())
        x = torch.randn(2, 5, 8) if kw.get("batch_first") else torch.randn(5, 2, 8)
        check(f"LSTM/{kw}", kb(x)[0], ref(x)[0], tol=1e-5)


def test_grid_sample():
    for mode in ["bilinear", "nearest"]:
        for pad in ["zeros", "border", "reflection"]:
            x = torch.randn(2, 3, 8, 8)
            grid = torch.randn(2, 4, 4, 2) * 1.5
            kb = GridSample(mode=mode, padding_mode=pad, align_corners=False)(x, grid)
            ref = F.grid_sample(x, grid, mode=mode, padding_mode=pad, align_corners=False)
            check(f"GridSample/mode={mode}/pad={pad}", kb, ref, tol=0.0)


def test_chunk_gated_delta_rule():
    """ChunkGatedDeltaRule + FusedRecurrentGatedDeltaRule pass-through to fla.ops."""
    if not torch.cuda.is_available():
        print("  SKIP  ChunkGatedDeltaRule (requires CUDA)")
        return
    try:
        from fla.ops.gated_delta_rule import (
            chunk_gated_delta_rule as ref_chunk,
            fused_recurrent_gated_delta_rule as ref_fused,
        )
        from tasks.baseline.L1.chunk_gated_delta_rule import (
            ChunkGatedDeltaRule, FusedRecurrentGatedDeltaRule,
        )
    except ImportError as e:
        print(f"  SKIP  fla not installed: {e}")
        return

    B, T, H, K, V = 2, 64, 4, 32, 32
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(0)
    q = torch.randn(B, T, H, K, device=device, dtype=dtype)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    g = torch.randn(B, T, H, device=device, dtype=torch.float32)
    beta = torch.rand(B, T, H, device=device, dtype=dtype)
    o_ref, _ = ref_chunk(q=q, k=k, v=v, g=g, beta=beta, output_final_state=False)
    o_kb, _ = ChunkGatedDeltaRule()(q=q, k=k, v=v, g=g, beta=beta, output_final_state=False)
    check("ChunkGatedDeltaRule/B2T64H4D32", o_kb, o_ref, tol=0.0)

    T1 = 1
    q1 = torch.randn(B, T1, H, K, device=device, dtype=dtype)
    k1 = torch.randn(B, T1, H, K, device=device, dtype=dtype)
    v1 = torch.randn(B, T1, H, V, device=device, dtype=dtype)
    g1 = torch.randn(B, T1, H, device=device, dtype=torch.float32)
    beta1 = torch.rand(B, T1, H, device=device, dtype=dtype)
    init_state = torch.zeros(B, H, K, V, device=device, dtype=torch.float32)
    o_ref, _ = ref_fused(q=q1, k=k1, v=v1, g=g1, beta=beta1,
                         initial_state=init_state, output_final_state=True)
    o_kb, _ = FusedRecurrentGatedDeltaRule()(
        q=q1, k=k1, v=v1, g=g1, beta=beta1,
        initial_state=init_state, output_final_state=True,
    )
    check("FusedRecurrentGatedDeltaRule/B2T1H4D32", o_kb, o_ref, tol=0.0)


def main():
    torch.manual_seed(0)
    print("=" * 95)
    print("Misc L1 wrappers added in audit pass (adaptive pool, conv-transpose, LSTM, grid_sample, gated-delta)")
    print("=" * 95)
    print("\n--- Adaptive avg pool ---")
    test_adaptive_avg_pool()
    print("\n--- ConvTranspose1d/2d/3d (state_dict-compat with nn.ConvTranspose*) ---")
    test_conv_transpose()
    print("\n--- LSTM (state_dict-compat with nn.LSTM, bare keys) ---")
    test_lstm()
    print("\n--- GridSample ---")
    test_grid_sample()
    print("\n--- ChunkGatedDeltaRule (CUDA only) ---")
    test_chunk_gated_delta_rule()
    print()
    print("=" * 95)
    print(f"Total: {len(PASS)} PASS, {len(FAIL)} FAIL")
    print("=" * 95)
    if FAIL:
        for name, diff in FAIL:
            print(f"  FAIL  {name}  (diff={diff:.2e})")
        sys.exit(1)


if __name__ == "__main__":
    main()
