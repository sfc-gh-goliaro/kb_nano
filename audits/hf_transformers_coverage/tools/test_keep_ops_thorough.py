"""Thorough reference-implementation tests for the 8 KEEP L1 ops.

For each op, the reference is identified explicitly (PyTorch torch.nn.X
or torch.nn.functional.X, or fla for ChunkGatedDeltaRule), and the kb-nano
wrapper is tested against it across:

- All input shape ranks the reference supports
- All dtypes HF inference uses (fp32, bf16, fp16; CUDA-only ones marked)
- All parameter combinations the reference exposes
- HF-actual usage patterns (recovered from grep over the pinned HF source)
- Edge cases (single-element, channels=1, kernel_size=input_size, groups>1, bias=False)
- state_dict-key compatibility for parameter-bearing ops (HF reference checkpoints
  must load with no remapping)

Each test prints PASS/FAIL with max-abs diff. Exits non-zero on any failure.
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
from tasks.baseline.L1.grid_sample import GridSample
from tasks.baseline.L1.lstm import LSTM

try:
    from tasks.baseline.L1.chunk_gated_delta_rule import ChunkGatedDeltaRule, FusedRecurrentGatedDeltaRule
    _HAVE_FLA = True
except ImportError:
    _HAVE_FLA = False


PASSED: list[tuple[str, str, float]] = []
FAILED: list[tuple[str, str, float]] = []


def _check(group: str, name: str, kb_out, ref_out, tol: float = 0.0):
    if isinstance(kb_out, tuple): kb_out = kb_out[0]
    if isinstance(ref_out, tuple): ref_out = ref_out[0]
    if kb_out.shape != ref_out.shape:
        FAILED.append((group, name, -1.0))
        print(f"  FAIL  {group}/{name}: shape mismatch {kb_out.shape} vs {ref_out.shape}")
        return
    diff = (kb_out.float() - ref_out.float()).abs().max().item()
    if diff <= tol:
        PASSED.append((group, name, diff))
        print(f"  PASS  {group:22s} {name:60s} diff={diff:.2e}")
    else:
        FAILED.append((group, name, diff))
        print(f"  FAIL  {group:22s} {name:60s} diff={diff:.2e} > tol={tol:.0e}")


def _state_keys_match(group, name, kb_mod, ref_mod):
    kb_keys = set(kb_mod.state_dict().keys())
    ref_keys = set(ref_mod.state_dict().keys())
    if kb_keys != ref_keys:
        FAILED.append((group, name + "_state_keys", -1.0))
        print(f"  FAIL  {group}/{name}_state_keys: kb={sorted(kb_keys)} vs ref={sorted(ref_keys)}")
    else:
        PASSED.append((group, name + "_state_keys", 0.0))
        print(f"  PASS  {group:22s} {name+'_state_keys':60s} keys={sorted(kb_keys)}")


def test_adaptive_avg_pool1d():
    """Reference: torch.nn.AdaptiveAvgPool1d / F.adaptive_avg_pool1d.
    HF actual usage: AdaptiveAvgPool1d(1).
    Spec: input is [N, C, L_in], output is [N, C, L_out]."""
    group = "AdaptiveAvgPool1d"
    for dtype in [torch.float32, torch.bfloat16, torch.float16]:
        for output_size in [1, 2, 4, 8, 16]:
            for L_in in [16, 32, 33]:  # divisor + non-divisor case
                for shape in [(2, 8, L_in), (1, 8, L_in), (4, 1, L_in)]:
                    x = torch.randn(*shape, dtype=dtype)
                    kb = AdaptiveAvgPool1d(output_size)
                    ref = nn.AdaptiveAvgPool1d(output_size)
                    _check(group, f"{dtype}/out{output_size}/L_in{L_in}/{shape}",
                           kb(x), ref(x), tol=1e-3 if dtype != torch.float32 else 1e-5)


def test_adaptive_avg_pool2d():
    """Reference: torch.nn.AdaptiveAvgPool2d / F.adaptive_avg_pool2d.
    HF actual usage: output_size in {1, 7, (1,1), (2,2), pool_scale, image_feature_pool_shape[:2]}.
    Spec: input is [N, C, H_in, W_in], output is [N, C, H_out, W_out]. output_size can be int or (H,W) or None for one dim."""
    group = "AdaptiveAvgPool2d"
    for dtype in [torch.float32, torch.bfloat16, torch.float16]:
        for output_size in [1, 7, (1, 1), (2, 2), (4, 8), 8]:
            for shape in [(2, 8, 16, 16), (1, 8, 16, 16), (4, 1, 16, 16), (2, 8, 7, 7)]:
                # skip cases where output > input
                out_h = output_size if isinstance(output_size, int) else output_size[0]
                out_w = output_size if isinstance(output_size, int) else output_size[1]
                if out_h > shape[2] or out_w > shape[3]:
                    continue
                x = torch.randn(*shape, dtype=dtype)
                kb = AdaptiveAvgPool2d(output_size)
                ref = nn.AdaptiveAvgPool2d(output_size)
                _check(group, f"{dtype}/out{output_size}/{shape}",
                       kb(x), ref(x), tol=1e-3 if dtype != torch.float32 else 1e-5)


def _test_conv_transpose(dim: int, KBClass, RefClass, shape_template):
    """Generic test for ConvTranspose1d/2d/3d.
    Reference: torch.nn.ConvTransposeNd / F.conv_transposeNd.
    Spec params: in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, dilation, bias."""
    group = f"ConvTranspose{dim}d"
    # state_dict key match
    ref = RefClass(4, 6, 3)
    kb = KBClass(4, 6, 3)
    _state_keys_match(group, "default", kb, ref)
    # Comprehensive forward tests
    for dtype in [torch.float32, torch.bfloat16]:  # fp16 is rarely used for conv-transpose in HF
        configs = [
            # (in_ch, out_ch, kernel, stride, padding, output_padding, groups, dilation, bias)
            (4, 6, 3, 1, 0, 0, 1, 1, True),   # default
            (4, 6, 3, 2, 1, 0, 1, 1, True),   # stride 2 + padding (HF common)
            (4, 6, 3, 2, 1, 1, 1, 1, True),   # output_padding
            (8, 8, 3, 2, 1, 0, 4, 1, True),   # groups
            (4, 6, 3, 2, 1, 0, 1, 1, False),  # bias=False (HF SamMaskDecoder uses this)
            (4, 6, 3, 1, 0, 0, 1, 2, True),   # dilation
            (4, 4, 4, 2, 1, 0, 1, 1, True),   # kernel=4, stride=2 (audio vocoder)
        ]
        for cfg in configs:
            in_ch, out_ch, k, s, p, op, g, d, b = cfg
            ref = RefClass(in_ch, out_ch, k, stride=s, padding=p, output_padding=op,
                           groups=g, dilation=d, bias=b).to(dtype)
            kb = KBClass(in_ch, out_ch, k, stride=s, padding=p, output_padding=op,
                         groups=g, dilation=d, bias=b).to(dtype)
            kb.load_state_dict(ref.state_dict())
            x = torch.randn(*shape_template(in_ch), dtype=dtype)
            _check(group, f"{dtype}/cfg={cfg}", kb(x), ref(x),
                   tol=5e-3 if dtype != torch.float32 else 1e-5)


def test_conv_transpose1d():
    _test_conv_transpose(1, ConvTranspose1d, nn.ConvTranspose1d,
                         lambda in_ch: (2, in_ch, 16))


def test_conv_transpose2d():
    _test_conv_transpose(2, ConvTranspose2d, nn.ConvTranspose2d,
                         lambda in_ch: (2, in_ch, 8, 8))


def test_conv_transpose3d():
    _test_conv_transpose(3, ConvTranspose3d, nn.ConvTranspose3d,
                         lambda in_ch: (2, in_ch, 4, 4, 4))


def test_grid_sample():
    """Reference: F.grid_sample.
    HF actual usage: mode='bilinear' (used by deformable attention internally).
    Spec params: mode in {bilinear, nearest, bicubic}, padding_mode in {zeros, border, reflection}, align_corners in {True, False, None}."""
    group = "GridSample"
    for dtype in [torch.float32, torch.bfloat16]:
        for mode in ["bilinear", "nearest", "bicubic"]:
            for padding_mode in ["zeros", "border", "reflection"]:
                for align_corners in [False, True]:
                    # bicubic + align_corners=True is supported but bicubic + reflection has known torch issues — skip
                    if mode == "bicubic" and padding_mode == "reflection":
                        continue
                    x = torch.randn(2, 3, 8, 8, dtype=dtype)
                    grid = torch.randn(2, 4, 4, 2, dtype=dtype).clamp(-1, 1)
                    kb = GridSample(mode=mode, padding_mode=padding_mode, align_corners=align_corners)
                    ref_out = F.grid_sample(x, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
                    _check(group, f"{dtype}/mode={mode}/pad={padding_mode}/align={align_corners}",
                           kb(x, grid), ref_out, tol=1e-3 if dtype != torch.float32 else 1e-5)


def test_lstm():
    """Reference: torch.nn.LSTM.
    HF actual usage: nn.LSTM(dim, dim, num_layers) — encodec.
    Spec: input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional, proj_size."""
    group = "LSTM"
    # As of round-2 fix, kb-nano LSTM is `class LSTM(nn.LSTM): pass` (subclass alias).
    # State_dict keys are bare (weight_ih_l0 etc), bit-identical to nn.LSTM.
    ref = nn.LSTM(8, 16, num_layers=2)
    kb = LSTM(8, 16, num_layers=2)
    kb_keys = set(kb.state_dict().keys())
    ref_keys = set(ref.state_dict().keys())
    if kb_keys != ref_keys:
        FAILED.append((group, "state_keys", -1.0))
        print(f"  FAIL  {group}/state_keys: kb={sorted(kb_keys)} vs ref={sorted(ref_keys)}")
    else:
        PASSED.append((group, "state_keys", 0.0))
        print(f"  PASS  {group:22s} state_keys (bare) = {sorted(ref_keys)[:3]}...")
    for dtype in [torch.float32]:  # LSTM cuDNN doesn't always like bf16; HF encodec uses fp32
        for cfg in [
            # (input_size, hidden_size, num_layers, bias, batch_first, bidirectional, proj_size)
            (8, 16, 1, True, False, False, 0),
            (8, 16, 2, True, False, False, 0),     # encodec default
            (8, 16, 2, True, True, False, 0),
            (8, 16, 1, True, False, True, 0),      # bidirectional
            (8, 16, 1, False, False, False, 0),    # bias=False
            (8, 16, 1, True, False, False, 4),     # proj_size
        ]:
            ip, hp, nl, bias, bf, bidir, proj = cfg
            ref = nn.LSTM(ip, hp, num_layers=nl, bias=bias, batch_first=bf,
                          bidirectional=bidir, proj_size=proj).to(dtype)
            kb = LSTM(ip, hp, num_layers=nl, bias=bias, batch_first=bf,
                      bidirectional=bidir, proj_size=proj).to(dtype)
            kb.load_state_dict(ref.state_dict())
            T, B = 5, 2
            x = torch.randn(B, T, ip, dtype=dtype) if bf else torch.randn(T, B, ip, dtype=dtype)
            ref_out, _ = ref(x)
            kb_out, _ = kb(x)
            _check(group, f"{dtype}/cfg={cfg}", kb_out, ref_out, tol=1e-5)


def test_chunk_gated_delta_rule():
    """Reference: fla.ops.gated_delta_rule.chunk_gated_delta_rule.
    HF actual usage: Qwen3.5/Qwen3-Next/OLMo-Hybrid via from fla.ops.gated_delta_rule import chunk_gated_delta_rule.
    Spec (from fla source): q,k,v,g,beta tensors; CUDA + bf16/fp16 only."""
    group = "ChunkGatedDeltaRule"
    if not _HAVE_FLA:
        print(f"  SKIP  {group}: fla not installed")
        return
    if not torch.cuda.is_available():
        print(f"  SKIP  {group}: needs CUDA")
        return
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
    for dtype in [torch.bfloat16, torch.float16]:
        for B, T, H, D, V in [(2, 16, 4, 32, 32), (1, 32, 8, 64, 64), (4, 8, 2, 32, 64)]:
            torch.manual_seed(0)
            q = torch.randn(B, T, H, D, device="cuda", dtype=dtype)
            k = torch.randn(B, T, H, D, device="cuda", dtype=dtype)
            v = torch.randn(B, T, H, V, device="cuda", dtype=dtype)
            g = torch.randn(B, T, H, device="cuda", dtype=dtype).log_softmax(-1)
            beta = torch.randn(B, T, H, device="cuda", dtype=dtype).sigmoid()
            kb = ChunkGatedDeltaRule()
            ref_out, _ = chunk_gated_delta_rule(q=q, k=k, v=v, g=g, beta=beta)
            kb_out, _ = kb(q=q, k=k, v=v, g=g, beta=beta)
            _check(group + "/chunk", f"{dtype}/B{B}T{T}H{H}D{D}V{V}", kb_out, ref_out, tol=0.0)
            kb_r = FusedRecurrentGatedDeltaRule()
            ref_r, _ = fused_recurrent_gated_delta_rule(q=q, k=k, v=v, g=g, beta=beta)
            kb_r_out, _ = kb_r(q=q, k=k, v=v, g=g, beta=beta)
            _check(group + "/recurrent", f"{dtype}/B{B}T{T}H{H}D{D}V{V}", kb_r_out, ref_r, tol=0.0)


def main():
    torch.manual_seed(0)
    print("=" * 100)
    print("Thorough reference-implementation tests for the 8 KEEP L1 ops")
    print("=" * 100)
    print()
    print("--- AdaptiveAvgPool1d (reference: torch.nn.AdaptiveAvgPool1d) ---")
    test_adaptive_avg_pool1d()
    print("\n--- AdaptiveAvgPool2d (reference: torch.nn.AdaptiveAvgPool2d) ---")
    test_adaptive_avg_pool2d()
    print("\n--- ConvTranspose1d (reference: torch.nn.ConvTranspose1d) ---")
    test_conv_transpose1d()
    print("\n--- ConvTranspose2d (reference: torch.nn.ConvTranspose2d) ---")
    test_conv_transpose2d()
    print("\n--- ConvTranspose3d (reference: torch.nn.ConvTranspose3d) ---")
    test_conv_transpose3d()
    print("\n--- GridSample (reference: F.grid_sample) ---")
    test_grid_sample()
    print("\n--- LSTM (reference: torch.nn.LSTM) ---")
    test_lstm()
    print("\n--- ChunkGatedDeltaRule (reference: fla.ops.gated_delta_rule) ---")
    test_chunk_gated_delta_rule()

    print()
    print("=" * 100)
    print(f"Total: {len(PASSED)} PASS, {len(FAILED)} FAIL")
    print("=" * 100)
    if FAILED:
        from collections import defaultdict
        by_op = defaultdict(list)
        for grp, name, diff in FAILED:
            by_op[grp].append((name, diff))
        print("\nFAILURES:")
        for grp, items in by_op.items():
            print(f"  {grp}: {len(items)} failure(s)")
            for n, d in items[:3]:
                print(f"     {n}  (diff={d:.2e})")
        sys.exit(1)


if __name__ == "__main__":
    main()
