"""Round-2 audit tests: actually-on-GPU correctness + state-output + real-config sizes.

The original `test_rg_lru.py` and `test_misc_l1_ops.py` ran tensors on CPU even when
`CUDA_VISIBLE_DEVICES` was set. This file explicitly moves all tensors to CUDA so
the cuDNN / Triton / device-transfer paths are actually exercised.

Coverage added beyond the CPU tests:
- RG-LRU on GPU at the real recurrent_gemma 2B config (lru_width=2560, num_attention_heads=10).
- RG-LRU state output (`recurrent_states`) compared in fp32 + bf16, T=1 + T=16.
- RG-LRU mid-sequence reset in bf16 (was fp32-only).
- RG-LRU autoregressive chain in bf16 (was fp32-only).
- LSTM on cuDNN path (state_dict load after .to(device)).
- ChunkGatedDeltaRule across multiple shapes + dtypes (was 1 shape only).
- Conv1d/Conv3d on GPU with HF kwarg patterns.

Requires CUDA. Skips gracefully if not available.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


PASS: list[str] = []
FAIL: list[tuple[str, str]] = []


def check(name: str, kb, ref, tol: float = 0.0, allow_nan: bool = False):
    if isinstance(kb, tuple):
        kb = kb[0]
    if isinstance(ref, tuple):
        ref = ref[0]
    if kb.shape != ref.shape:
        FAIL.append((name, f"shape mismatch {kb.shape} vs {ref.shape}"))
        print(f"  FAIL  {name}: shape mismatch")
        return
    if allow_nan:
        nan_kb, nan_ref = torch.isnan(kb), torch.isnan(ref)
        if not torch.equal(nan_kb, nan_ref):
            FAIL.append((name, "NaN positions differ"))
            print(f"  FAIL  {name}: NaN positions differ")
            return
        diff = (kb.float() - ref.float()).where(~nan_kb, torch.zeros_like(kb.float())).abs().max().item()
    else:
        diff = (kb.float() - ref.float()).abs().max().item()
    if diff <= tol:
        PASS.append(name)
        print(f"  PASS  {name:78s} diff={diff:.2e}")
    else:
        FAIL.append((name, f"diff={diff:.2e} > {tol:.0e}"))
        print(f"  FAIL  {name:78s} diff={diff:.2e} > {tol:.0e}")


def test_rg_lru_gpu_real_config():
    """RG-LRU on CUDA at the real recurrent_gemma 2B config."""
    from tasks.baseline.L1.rg_lru import RGLRU as KBRGLRU
    from transformers.models.recurrent_gemma.modeling_recurrent_gemma import (
        RecurrentGemmaRglru,
    )

    class _Cfg:
        def __init__(self, h, w):
            self.num_attention_heads = h
            self.lru_width = w

    device = torch.device("cuda")

    def make_pair(h, w, dtype):
        torch.manual_seed(0)
        ref = RecurrentGemmaRglru(_Cfg(h, w)).to(dtype).to(device)
        kb = KBRGLRU(h, w).to(dtype).to(device)
        for _, p in ref.named_parameters():
            torch.nn.init.normal_(p, std=0.1)
        kb.load_state_dict(ref.state_dict())
        return kb, ref

    # Real config: recurrent_gemma 2B uses lru_width=2560, num_attention_heads=10
    for h, w in [(10, 2560), (4, 32)]:
        for dtype in [torch.float32, torch.bfloat16]:
            for B, T in [(2, 16), (1, 1)]:
                kb, ref = make_pair(h, w, dtype)
                ref.recurrent_states = None
                kb.recurrent_states = None
                torch.manual_seed(7)
                x = torch.randn(B, T, w, dtype=dtype, device=device)
                pos = torch.arange(T, device=device).unsqueeze(0).expand(B, -1).contiguous()
                out_kb = kb(x, pos)
                out_ref = ref(x, pos)
                tol = 5e-3 if dtype != torch.float32 else 1e-5
                dtype_name = str(dtype).split(".")[-1]
                check(f"RGLRU-GPU/h{h}w{w}/{dtype_name}/B{B}T{T}", out_kb, out_ref, tol)
                # State output too
                if ref.recurrent_states is not None and kb.recurrent_states is not None:
                    diff = (ref.recurrent_states.float() - kb.recurrent_states.float()).abs().max().item()
                    if diff <= tol:
                        PASS.append(f"RGLRU-GPU-state/h{h}w{w}/{dtype_name}/B{B}T{T}")
                        print(f"  PASS  RGLRU-GPU-state/h{h}w{w}/{dtype_name}/B{B}T{T:<3d}                          diff={diff:.2e}")
                    else:
                        FAIL.append((f"RGLRU-GPU-state/h{h}w{w}/{dtype_name}/B{B}T{T}", f"diff={diff:.2e}"))
                        print(f"  FAIL  RGLRU-GPU-state/h{h}w{w}/{dtype_name}/B{B}T{T}                              diff={diff:.2e}")


def test_rg_lru_bf16_paths():
    """Mid-sequence reset + autoregressive chain in bf16 (only fp32 was in CPU tests)."""
    from tasks.baseline.L1.rg_lru import RGLRU as KBRGLRU
    from transformers.models.recurrent_gemma.modeling_recurrent_gemma import (
        RecurrentGemmaRglru,
    )

    class _Cfg:
        def __init__(self, h, w):
            self.num_attention_heads = h
            self.lru_width = w

    device = torch.device("cuda")

    def make_pair(h, w, dtype):
        torch.manual_seed(0)
        ref = RecurrentGemmaRglru(_Cfg(h, w)).to(dtype).to(device)
        kb = KBRGLRU(h, w).to(dtype).to(device)
        for _, p in ref.named_parameters():
            torch.nn.init.normal_(p, std=0.1)
        kb.load_state_dict(ref.state_dict())
        return kb, ref

    # Mid-seq reset in bf16
    kb, ref = make_pair(4, 32, torch.bfloat16)
    torch.manual_seed(0)
    x = torch.randn(2, 16, 32, device=device, dtype=torch.bfloat16)
    pos = torch.arange(16, device=device).unsqueeze(0).expand(2, -1).clone().contiguous()
    pos[:, 8] = 0
    pos[:, 9:] = torch.arange(7, device=device).unsqueeze(0).expand(2, -1) + 1
    ref.recurrent_states = None
    kb.recurrent_states = None
    check("RGLRU-bf16/mid-reset", kb(x, pos), ref(x, pos), tol=5e-3)

    # Autoregressive in bf16
    kb, ref = make_pair(4, 32, torch.bfloat16)
    ref.recurrent_states = None
    kb.recurrent_states = None
    torch.manual_seed(0)
    x_pre = torch.randn(2, 8, 32, device=device, dtype=torch.bfloat16)
    pos_pre = torch.arange(8, device=device).unsqueeze(0).expand(2, -1).contiguous()
    check("RGLRU-bf16/AR-prefill", kb(x_pre, pos_pre), ref(x_pre, pos_pre), tol=5e-3)
    for step in range(4):
        x_step = torch.randn(2, 1, 32, device=device, dtype=torch.bfloat16)
        pos_step = torch.tensor([[8 + step], [8 + step]], device=device)
        check(f"RGLRU-bf16/AR-decode{step}", kb(x_step, pos_step), ref(x_step, pos_step), tol=5e-3)


def test_lstm_cudnn():
    """LSTM on the cuDNN path. Was tested only on CPU (which uses pure-pytorch)."""
    from tasks.baseline.L1.lstm import LSTM as KBLSTM

    device = torch.device("cuda")
    for kw in [dict(num_layers=1), dict(num_layers=2, bidirectional=True),
               dict(batch_first=True), dict(num_layers=1, bias=False)]:
        ref = nn.LSTM(8, 16, **kw).to(device)
        kb = KBLSTM(8, 16, **kw).to(device)
        kb.lstm.load_state_dict(ref.state_dict())
        if kw.get("batch_first"):
            x = torch.randn(2, 5, 8, device=device)
        else:
            x = torch.randn(5, 2, 8, device=device)
        check(f"LSTM-cuDNN/{kw}", kb(x)[0], ref(x)[0], tol=1e-5)


def test_chunk_gdr_multi_shape():
    """ChunkGatedDeltaRule across multiple shapes + dtypes, with bounded log-gates.

    Original test was 1 shape (B=2,T=64,H=4,K=V=32) with bf16 only and unbounded gates
    (which produces NaN at large T due to numeric overflow). This re-tests with
    realistic bounded gates."""
    try:
        from fla.ops.gated_delta_rule import (
            chunk_gated_delta_rule as ref_chunk,
            fused_recurrent_gated_delta_rule as ref_fused,
        )
        from tasks.baseline.L1.chunk_gated_delta_rule import (
            ChunkGatedDeltaRule, FusedRecurrentGatedDeltaRule,
        )
    except ImportError as e:
        print(f"  SKIP  chunk_gated_delta_rule: {e}")
        return

    device = torch.device("cuda")
    for B, T, H, K, V in [(2, 64, 4, 32, 32), (1, 128, 8, 64, 64),
                          (4, 32, 4, 16, 16), (1, 256, 2, 32, 32)]:
        for dtype in [torch.bfloat16, torch.float16]:
            torch.manual_seed(B * T + H + K + V)
            q = torch.randn(B, T, H, K, device=device, dtype=dtype) * 0.5
            k = torch.randn(B, T, H, K, device=device, dtype=dtype) * 0.5
            v = torch.randn(B, T, H, V, device=device, dtype=dtype) * 0.5
            g = torch.nn.functional.logsigmoid(
                torch.randn(B, T, H, device=device, dtype=torch.float32)
            )  # bounded log-gates, like real use
            beta = torch.rand(B, T, H, device=device, dtype=dtype) * 0.1
            o_ref, _ = ref_chunk(q=q, k=k, v=v, g=g, beta=beta, output_final_state=False)
            o_kb, _ = ChunkGatedDeltaRule()(q=q, k=k, v=v, g=g, beta=beta, output_final_state=False)
            dtype_name = str(dtype).split(".")[-1]
            check(f"ChunkGDR/B{B}T{T}H{H}K{K}V{V}/{dtype_name}", o_kb, o_ref, tol=0.0)


def test_conv_gpu():
    """Conv1d (extended) and Conv3d (extended) on GPU."""
    from tasks.baseline.L1.conv1d import Conv1d as KBConv1d
    from tasks.baseline.L1.conv3d import Conv3d as KBConv3d

    device = torch.device("cuda")
    # Conv1d
    for in_c, out_c, k, kw in [(8, 8, 3, dict(groups=8, bias=False)),
                                (8, 16, 3, dict(stride=2, dilation=2, groups=4)),
                                (16, 16, 7, dict(dilation=3, padding=9)),
                                (8, 8, 5, dict(padding=2, padding_mode="reflect"))]:
        ref = nn.Conv1d(in_c, out_c, k, **kw).to(device)
        kb = KBConv1d(in_c, out_c, k, **kw).to(device)
        kb.conv.load_state_dict(ref.state_dict())
        x = torch.randn(2, in_c, 16, device=device)
        check(f"Conv1d-GPU/in{in_c}-out{out_c}-k{k}/{kw}", kb(x), ref(x), tol=1e-5)
    # Conv3d
    for kw in [dict(stride=1, padding=0), dict(stride=2, padding=1, dilation=1),
               dict(stride=1, padding=1, groups=2)]:
        in_c, out_c, k = 4, 8, 3
        if kw.get("groups", 1) > 1:
            in_c = out_c = 8
        ref = nn.Conv3d(in_c, out_c, k, bias=False, **kw).to(device)
        kb = KBConv3d(in_c, out_c, k, **kw).to(device)
        kb.conv.load_state_dict(ref.state_dict())
        x = torch.randn(1, in_c, 8, 8, 8, device=device)
        check(f"Conv3d-GPU/{kw}", kb(x), ref(x), tol=1e-5)


def main():
    if not torch.cuda.is_available():
        print("CUDA not available; skipping round-2 GPU tests.")
        sys.exit(0)
    print("=" * 95)
    print(f"Round-2 GPU audit tests on {torch.cuda.get_device_name(0)}")
    print("=" * 95)
    print("\n--- RG-LRU on CUDA (incl. real recurrent_gemma 2B config) ---")
    test_rg_lru_gpu_real_config()
    print("\n--- RG-LRU bf16 paths (mid-reset + autoregressive) ---")
    test_rg_lru_bf16_paths()
    print("\n--- LSTM cuDNN path ---")
    test_lstm_cudnn()
    print("\n--- ChunkGatedDeltaRule multi-shape + multi-dtype with bounded gates ---")
    test_chunk_gdr_multi_shape()
    print("\n--- Conv1d / Conv3d on GPU ---")
    test_conv_gpu()
    print()
    print("=" * 95)
    print(f"Round-2 GPU audit tests: {len(PASS)} PASS, {len(FAIL)} FAIL")
    print("=" * 95)
    if FAIL:
        for name, why in FAIL:
            print(f"  FAIL  {name}: {why}")
        sys.exit(1)


if __name__ == "__main__":
    main()
