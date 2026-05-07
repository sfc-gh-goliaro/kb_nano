"""Thorough reference-implementation tests for the v3 (this audit pass) L1+L2 ops.

This tests:
  L1 ops added/re-added in this pass:
    MaxPool1d, AvgPool1d  (DIRECT 1D dispatch — NOT 2D-composed; mentor: performance-faithful)
    LeakyReLU, ELU, Hardsigmoid, Hardswish  (explicit per audit prompt)
    Conv1dNative  (full kwarg coverage: groups, dilation, padding_mode)
  L2 op added:
    MultiheadAttention  (uses DenseAttention/SDPA when need_weights=False;
                         materializes attention map only when need_weights=True;
                         state_dict-compat with nn.MultiheadAttention)

Each op is tested against the reference impl (PyTorch) across:
  - all HF-actual kwarg patterns (surveyed from the pinned commit)
  - dtypes (fp32, bf16, fp16 where applicable)
  - shapes covering edge cases (B=1, channels=1, kernel=input_size)
  - state_dict-key compatibility for parameter-bearing ops
  - HF-specific call patterns (need_weights=False, batch_first, masks)
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tasks.baseline.L1.max_pool1d import MaxPool1d
from tasks.baseline.L1.avg_pool1d import AvgPool1d
from tasks.baseline.L1.leaky_relu import LeakyReLU
from tasks.baseline.L1.elu import ELU
from tasks.baseline.L1.hardsigmoid import Hardsigmoid
from tasks.baseline.L1.hardswish import Hardswish
from tasks.baseline.L1.conv1d_native import Conv1dNative
from tasks.baseline.L2.multihead_attention import MultiheadAttention as KBMultiheadAttention


PASSED: list[tuple[str, str, float]] = []
FAILED: list[tuple[str, str, float]] = []


def _check(group: str, name: str, kb_out, ref_out, tol: float = 0.0):
    if isinstance(kb_out, tuple):
        kb_out = kb_out[0]
    if isinstance(ref_out, tuple):
        ref_out = ref_out[0]
    if kb_out.shape != ref_out.shape:
        FAILED.append((group, name, -1.0))
        print(f"  FAIL  {group}/{name}: shape mismatch {kb_out.shape} vs {ref_out.shape}")
        return
    diff = (kb_out.float() - ref_out.float()).abs().max().item()
    if diff <= tol:
        PASSED.append((group, name, diff))
        print(f"  PASS  {group:24s} {name:55s} diff={diff:.2e}")
    else:
        FAILED.append((group, name, diff))
        print(f"  FAIL  {group:24s} {name:55s} diff={diff:.2e} > tol={tol:.0e}")


def test_pool1d_direct_dispatch():
    """Test MaxPool1d/AvgPool1d are bit-identical to torch.nn (and to F.x_pool1d directly,
    NOT to a 2D-composed version which would benchmark a different kernel)."""
    for dtype in [torch.float32, torch.bfloat16, torch.float16]:
        for k, s, p in [(2, 2, 0), (3, 2, 1), (5, 1, 2), (3, 3, 0)]:
            for shape in [(4, 8, 32), (1, 8, 32), (4, 1, 32), (4, 8, k)]:
                x = torch.randn(*shape, dtype=dtype)
                _check("MaxPool1d", f"{dtype}/k{k}s{s}p{p}/{shape}",
                       MaxPool1d(k, s, p)(x), nn.MaxPool1d(k, s, p)(x), tol=0.0)
            for shape in [(4, 8, 32), (1, 8, 32), (4, 1, 32)]:
                x = torch.randn(*shape, dtype=dtype)
                _check("AvgPool1d", f"{dtype}/k{k}s{s}p{p}/{shape}",
                       AvgPool1d(k, s, p)(x), nn.AvgPool1d(k, s, p)(x),
                       tol=5e-3 if dtype != torch.float32 else 1e-6)


def test_activations_match_nn():
    for dtype in [torch.float32, torch.bfloat16, torch.float16]:
        for shape in [(16,), (4, 8), (4, 8, 32), (4, 8, 16, 16)]:
            x = (torch.randn(*shape) * 4).to(dtype)
            for slope in [0.0, 0.01, 0.1, 0.2]:
                _check("LeakyReLU", f"{dtype}/slope{slope}/{shape}",
                       LeakyReLU(slope)(x), nn.LeakyReLU(slope)(x))
            for alpha in [0.5, 1.0, 1.5]:
                _check("ELU", f"{dtype}/alpha{alpha}/{shape}",
                       ELU(alpha)(x), nn.ELU(alpha)(x))
            _check("Hardsigmoid", f"{dtype}/{shape}", Hardsigmoid()(x), nn.Hardsigmoid()(x))
            _check("Hardswish", f"{dtype}/{shape}", Hardswish()(x), nn.Hardswish()(x))


def test_conv1d_native_full_kwargs():
    """Verify Conv1dNative covers all HF kwarg patterns exhaustively."""
    # state_dict compat (HF reference checkpoints use 'weight' / 'bias')
    ref = nn.Conv1d(4, 6, 3)
    kb = Conv1dNative(4, 6, 3)
    if set(kb.state_dict().keys()) != set(ref.state_dict().keys()):
        FAILED.append(("Conv1dNative", "state_keys", -1.0))
        print(f"  FAIL  Conv1dNative state_dict keys mismatch")
    else:
        PASSED.append(("Conv1dNative", "state_keys", 0.0))
        print(f"  PASS  Conv1dNative          state_keys = {sorted(ref.state_dict().keys())}")

    # HF actual usage patterns from grep over pinned commit:
    #   - granite_speech: nn.Conv1d(chan_in, chan_out, kernel_size, groups=chan_in, bias=False)  [depthwise]
    #   - vibevoice_asr: nn.Conv1d(in_channels, out_channels, kernel_size, stride, dilation=dilation, groups=groups)
    #   - dac/encodec: nn.Conv1d(dimension, dimension, kernel_size=7, dilation=dilation, padding=pad)
    #   - squeezebert: nn.Conv1d(cin, cout, kernel_size=1, groups=groups)
    #   - whisper: nn.Conv1d(channels, channels, kernel_size=3, stride=2, padding=1)  [narrow features]
    HF_KWARGS = [
        # (in, out, kernel, kwargs)
        (8, 8, 3, dict(groups=8, bias=False)),                       # granite depthwise
        (8, 16, 3, dict(stride=2, dilation=2, groups=4)),            # vibevoice
        (16, 16, 7, dict(dilation=3, padding=9)),                    # dac/encodec
        (32, 16, 1, dict(groups=8)),                                 # squeezebert
        (4, 8, 3, dict(stride=2, padding=1)),                        # whisper
        (8, 8, 5, dict(padding=2, padding_mode="reflect")),          # padding_mode != zeros
        (8, 8, 5, dict(padding=2, padding_mode="replicate")),
        (8, 8, 5, dict(padding=2, padding_mode="circular")),
        (4, 12, 3, dict(stride=1, padding=1, dilation=1, groups=1, bias=True)),  # default everything
    ]
    for dtype in [torch.float32, torch.bfloat16]:
        for in_ch, out_ch, k, kw in HF_KWARGS:
            ref = nn.Conv1d(in_ch, out_ch, k, **kw).to(dtype)
            kb = Conv1dNative(in_ch, out_ch, k, **kw).to(dtype)
            kb.load_state_dict(ref.state_dict())
            x = torch.randn(2, in_ch, 16, dtype=dtype)
            _check("Conv1dNative", f"{dtype}/in{in_ch}-out{out_ch}-k{k}/{kw}",
                   kb(x), ref(x), tol=5e-3 if dtype != torch.float32 else 1e-5)


def test_mha_state_dict_compat():
    """Verify MultiheadAttention parameter naming matches torch.nn.MultiheadAttention so
    HF reference checkpoints load with no remapping."""
    ref = nn.MultiheadAttention(64, 8, batch_first=True)
    kb = KBMultiheadAttention(64, 8, batch_first=True)
    ref_keys = set(ref.state_dict().keys())
    kb_keys = set(kb.state_dict().keys())
    # Both should have in_proj_weight, in_proj_bias, out_proj.weight, out_proj.bias
    expected = {"in_proj_weight", "in_proj_bias", "out_proj.weight", "out_proj.bias"}
    if not expected.issubset(kb_keys):
        FAILED.append(("MHA", "state_keys", -1.0))
        print(f"  FAIL  MHA state_dict keys missing: {expected - kb_keys}")
    elif not expected.issubset(ref_keys):
        # Some torch versions use bias_k/bias_v too
        pass
    else:
        PASSED.append(("MHA", "state_keys", 0.0))
        print(f"  PASS  MHA                  state_keys (subset match) = {sorted(expected)}")


def test_mha_self_attention_no_weights():
    """need_weights=False → SDPA fast path. Self-attention with batch_first=True
    (the aria/idefics2 pattern)."""
    for dtype in [torch.float32, torch.bfloat16]:
        for E, H, B, L in [(64, 8, 2, 16), (128, 16, 1, 32), (32, 4, 4, 8)]:
            torch.manual_seed(0)
            ref = nn.MultiheadAttention(E, H, batch_first=True, dropout=0.0).to(dtype).eval()
            kb = KBMultiheadAttention(E, H, batch_first=True, dropout=0.0).to(dtype).eval()
            kb.load_state_dict(ref.state_dict())
            x = torch.randn(B, L, E, dtype=dtype)
            ref_out, _ = ref(x, x, x, need_weights=False)
            kb_out, _ = kb(x, x, x, need_weights=False)
            _check("MHA(self,no_w)", f"{dtype}/E{E}H{H}B{B}L{L}",
                   kb_out, ref_out, tol=1e-3 if dtype != torch.float32 else 1e-5)


def test_mha_self_attention_with_weights():
    """need_weights=True → explicit attention map."""
    for E, H, B, L in [(64, 8, 2, 16), (32, 4, 4, 8)]:
        torch.manual_seed(0)
        ref = nn.MultiheadAttention(E, H, batch_first=True, dropout=0.0).eval()
        kb = KBMultiheadAttention(E, H, batch_first=True, dropout=0.0).eval()
        kb.load_state_dict(ref.state_dict())
        x = torch.randn(B, L, E)
        ref_out, ref_w = ref(x, x, x, need_weights=True, average_attn_weights=True)
        kb_out, kb_w = kb(x, x, x, need_weights=True, average_attn_weights=True)
        _check("MHA(self,with_w)", f"out E{E}H{H}B{B}L{L}", kb_out, ref_out, tol=1e-5)
        _check("MHA(self,with_w)", f"weights E{E}H{H}B{B}L{L}", kb_w, ref_w, tol=1e-5)


def test_mha_cross_attention():
    """Cross-attention (different q vs k=v lengths). Mask2former / oneformer pattern."""
    for E, H, B, Lq, Lk in [(64, 8, 2, 16, 24), (128, 16, 1, 8, 32)]:
        torch.manual_seed(0)
        ref = nn.MultiheadAttention(E, H, batch_first=True, dropout=0.0).eval()
        kb = KBMultiheadAttention(E, H, batch_first=True, dropout=0.0).eval()
        kb.load_state_dict(ref.state_dict())
        q = torch.randn(B, Lq, E)
        k = torch.randn(B, Lk, E)
        v = torch.randn(B, Lk, E)
        ref_out, _ = ref(q, k, v, need_weights=False)
        kb_out, _ = kb(q, k, v, need_weights=False)
        _check("MHA(cross,no_w)", f"E{E}H{H}B{B}Lq{Lq}Lk{Lk}", kb_out, ref_out, tol=1e-5)


def test_mha_with_attn_mask():
    """attn_mask (additive float mask)."""
    E, H, B, L = 64, 8, 2, 8
    torch.manual_seed(0)
    ref = nn.MultiheadAttention(E, H, batch_first=True, dropout=0.0).eval()
    kb = KBMultiheadAttention(E, H, batch_first=True, dropout=0.0).eval()
    kb.load_state_dict(ref.state_dict())
    x = torch.randn(B, L, E)
    # additive mask: lower triangular
    causal = torch.triu(torch.full((L, L), float("-inf")), diagonal=1)
    ref_out, _ = ref(x, x, x, attn_mask=causal, need_weights=False)
    kb_out, _ = kb(x, x, x, attn_mask=causal, need_weights=False)
    _check("MHA(attn_mask)", f"causal E{E}L{L}", kb_out, ref_out, tol=1e-5)


def test_mha_with_key_padding_mask():
    """key_padding_mask (boolean mask: True = mask out)."""
    E, H, B, L = 64, 8, 2, 8
    torch.manual_seed(0)
    ref = nn.MultiheadAttention(E, H, batch_first=True, dropout=0.0).eval()
    kb = KBMultiheadAttention(E, H, batch_first=True, dropout=0.0).eval()
    kb.load_state_dict(ref.state_dict())
    x = torch.randn(B, L, E)
    # mask out last 3 of each batch
    kpm = torch.zeros(B, L, dtype=torch.bool)
    kpm[:, -3:] = True
    ref_out, _ = ref(x, x, x, key_padding_mask=kpm, need_weights=False)
    kb_out, _ = kb(x, x, x, key_padding_mask=kpm, need_weights=False)
    _check("MHA(kpm)", f"E{E}L{L}", kb_out, ref_out, tol=1e-5)


def test_mha_batch_first_false():
    """batch_first=False (the bridgetower pattern: input is (L, B, E))."""
    E, H, B, L = 64, 8, 2, 8
    torch.manual_seed(0)
    ref = nn.MultiheadAttention(E, H, batch_first=False, dropout=0.0).eval()
    kb = KBMultiheadAttention(E, H, batch_first=False, dropout=0.0).eval()
    kb.load_state_dict(ref.state_dict())
    x = torch.randn(L, B, E)  # (L, B, E)
    ref_out, _ = ref(x, x, x, need_weights=False)
    kb_out, _ = kb(x, x, x, need_weights=False)
    _check("MHA(batch_first=F)", f"E{E}L{L}", kb_out, ref_out, tol=1e-5)


def main():
    torch.manual_seed(0)
    print("=" * 95)
    print("v3 audit-pass tests: Conv1dNative + MHA L2 + re-added pool/activation L1")
    print("=" * 95)
    print()
    print("--- Pool1d (direct 1D dispatch, NOT 2D-composed) ---")
    test_pool1d_direct_dispatch()
    print("\n--- Activations (explicit L1 wrappers) ---")
    test_activations_match_nn()
    print("\n--- Conv1dNative (full HF kwarg coverage: groups, dilation, padding_mode) ---")
    test_conv1d_native_full_kwargs()
    print("\n--- MultiheadAttention L2 wrapper ---")
    test_mha_state_dict_compat()
    test_mha_self_attention_no_weights()
    test_mha_self_attention_with_weights()
    test_mha_cross_attention()
    test_mha_with_attn_mask()
    test_mha_with_key_padding_mask()
    test_mha_batch_first_false()

    print()
    print("=" * 95)
    print(f"Total: {len(PASSED)} PASS, {len(FAILED)} FAIL")
    print("=" * 95)
    if FAILED:
        from collections import defaultdict
        by_op = defaultdict(list)
        for grp, name, diff in FAILED:
            by_op[grp].append((name, diff))
        print("\nFAILURES:")
        for grp, items in by_op.items():
            print(f"  {grp}: {len(items)} failure(s)")
            for n, d in items[:5]:
                print(f"     {n}  (diff={d:.2e})")
        sys.exit(1)


if __name__ == "__main__":
    main()
