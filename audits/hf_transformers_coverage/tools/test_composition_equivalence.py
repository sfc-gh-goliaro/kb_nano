"""Rigorous composition-equivalence tests.

Proves that 8 of the originally-added 16 L1 wrappers are unnecessary because
each op is either:
- a direct torch builtin (F.x) — covered by the audit's passthrough mechanism, or
- the same as kb-nano's existing rank-agnostic BatchNorm2d / MaxPool2d / AvgPool2d
  with a trivial reshape.

Tested across:
- multiple input shapes (including the actual ranks HF uses, e.g. BatchNorm1d
  in `[B, C]` rank-2 mode used by groupvit/levit, AND `[B, C, L]` rank-3 mode)
- multiple dtypes (fp32, bf16, fp16 — what HF inference actually uses)
- multiple parameter configurations
- edge cases (batch_size=1, channels=1)
- eval AND train mode for stateful ops (BatchNorm running stats)

Conclusion drawn at end:
- Operations that pass these tests can be removed (no new L1 file needed).
- Operations that fail are genuine new primitives requiring a wrapper.

Note: the elementwise activations (LeakyReLU, ELU, Hardsigmoid, Hardswish)
are tested via direct F.x calls — those are bit-identical to nn.X by
construction (since nn.X just wraps F.x). The result is documentation of
that fact, not a numerical claim.
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


PASSED: list[tuple[str, str, float]] = []
FAILED: list[tuple[str, str, float]] = []


def _check(group: str, name: str, kb_out: torch.Tensor, comp_out: torch.Tensor, tol: float = 1e-4):
    if kb_out.shape != comp_out.shape:
        FAILED.append((group, name, -1.0))
        print(f"  FAIL  {group}/{name}: shape mismatch {kb_out.shape} vs {comp_out.shape}")
        return
    diff = (kb_out.float() - comp_out.float()).abs().max().item()
    if diff <= tol:
        PASSED.append((group, name, diff))
        print(f"  PASS  {group:24s} {name:50s} diff={diff:.2e}")
    else:
        FAILED.append((group, name, diff))
        print(f"  FAIL  {group:24s} {name:50s} diff={diff:.2e} > tol={tol:.0e}")


def test_batch_norm1d():
    """BatchNorm1d == kb-nano BatchNorm2d (which calls F.batch_norm rank-agnostically)."""
    for dtype in [torch.float32, torch.bfloat16, torch.float16]:
        for mode in ["eval", "train"]:
            for shape_name, shape in [
                ("[B,C]_rank2", (8, 16)),       # groupvit/levit pattern
                ("[B,C,L]_rank3", (4, 8, 32)),  # fastspeech/hubert pattern
                ("[B=1,C]", (1, 16)),
                ("[B,C=1]", (4, 1)),
                ("[B=1,C,L]", (1, 8, 32)),
            ]:
                # nn.BatchNorm in train requires >1 element/channel
                if mode == "train" and shape[0] == 1 and (len(shape) == 2 or shape[2] == 1):
                    continue
                ref = nn.BatchNorm1d(shape[1])
                kb = KBNanoBatchNorm2d(shape[1])
                kb.load_state_dict(ref.state_dict())
                if mode == "eval":
                    ref.eval(); kb.eval()
                else:
                    ref.train(); kb.train()
                ref = ref.to(dtype); kb = kb.to(dtype)
                x = torch.randn(*shape, dtype=dtype)
                _check("BatchNorm1d", f"{dtype}/{mode}/{shape_name}", kb(x), ref(x),
                       tol=5e-3 if dtype != torch.float32 else 1e-5)


def test_batch_norm3d():
    """BatchNorm3d == kb-nano BatchNorm2d (rank-agnostic)."""
    for dtype in [torch.float32, torch.bfloat16, torch.float16]:
        for mode in ["eval", "train"]:
            for shape_name, shape in [
                ("[B,C,D,H,W]", (2, 4, 4, 8, 8)),
                ("[B=1,C,D,H,W]", (1, 4, 4, 8, 8)),
                ("[B,C=1,D,H,W]", (2, 1, 4, 8, 8)),
                ("[B,C,D=1,H,W]", (2, 4, 1, 8, 8)),
            ]:
                if mode == "train" and shape[0] == 1 and shape[2] == 1:
                    continue
                ref = nn.BatchNorm3d(shape[1])
                kb = KBNanoBatchNorm2d(shape[1])
                kb.load_state_dict(ref.state_dict())
                if mode == "eval": ref.eval(); kb.eval()
                else: ref.train(); kb.train()
                ref = ref.to(dtype); kb = kb.to(dtype)
                x = torch.randn(*shape, dtype=dtype)
                _check("BatchNorm3d", f"{dtype}/{mode}/{shape_name}", kb(x), ref(x),
                       tol=5e-3 if dtype != torch.float32 else 1e-5)


def test_maxpool1d():
    """MaxPool1d == kb-nano MaxPool2d with kernel (1, k) + reshape."""
    for dtype in [torch.float32, torch.bfloat16, torch.float16]:
        for k, s, p in [(2, 2, 0), (3, 2, 1), (5, 1, 2), (3, 3, 0)]:
            for shape_name, shape in [
                ("[B,C,L]", (4, 8, 32)),
                ("[B=1,C,L]", (1, 8, 32)),
                ("[B,C=1,L]", (4, 1, 32)),
                ("[B,C,L=k]", (4, 8, k)),
            ]:
                ref = nn.MaxPool1d(kernel_size=k, stride=s, padding=p)
                kb_2d = KBNanoMaxPool2d(kernel_size=(1, k), stride=(1, s), padding=(0, p))
                x = torch.randn(*shape, dtype=dtype)
                _check("MaxPool1d", f"{dtype}/k{k}s{s}p{p}/{shape_name}",
                       kb_2d(x.unsqueeze(-2)).squeeze(-2), ref(x),
                       tol=5e-3 if dtype != torch.float32 else 0.0)


def test_avgpool1d():
    """AvgPool1d == kb-nano AvgPool2d with kernel (1, k) + reshape."""
    for dtype in [torch.float32, torch.bfloat16, torch.float16]:
        for k, s, p in [(2, 2, 0), (3, 1, 1), (5, 1, 2), (3, 3, 0)]:
            for shape_name, shape in [
                ("[B,C,L]", (4, 8, 32)),
                ("[B=1,C,L]", (1, 8, 32)),
                ("[B,C=1,L]", (4, 1, 32)),
            ]:
                ref = nn.AvgPool1d(kernel_size=k, stride=s, padding=p)
                kb_2d = KBNanoAvgPool2d(kernel_size=(1, k), stride=(1, s), padding=(0, p))
                x = torch.randn(*shape, dtype=dtype)
                _check("AvgPool1d", f"{dtype}/k{k}s{s}p{p}/{shape_name}",
                       kb_2d(x.unsqueeze(-2)).squeeze(-2), ref(x),
                       tol=5e-3 if dtype != torch.float32 else 1e-6)


def test_elementwise_activations_via_F_x():
    """LeakyReLU, ELU, Hardsigmoid, Hardswish — torch builtins (F.x).

    These are bit-identical to nn.X by construction (nn.X just wraps F.x).
    No new L1 file needed; the audit's `passthrough` mechanism for torch
    builtins covers them.
    """
    for dtype in [torch.float32, torch.bfloat16, torch.float16]:
        for shape_name, shape in [
            ("vec", (16,)),
            ("[B,C]", (4, 8)),
            ("[B,C,L]", (4, 8, 32)),
            ("[B,C,H,W]", (4, 8, 16, 16)),
        ]:
            x = (torch.randn(*shape) * 4).to(dtype)
            for slope in [0.0, 0.01, 0.1, 0.2]:
                _check("LeakyReLU(F)", f"{dtype}/slope{slope}/{shape_name}",
                       F.leaky_relu(x, negative_slope=slope),
                       nn.LeakyReLU(slope)(x), tol=0.0)
            for alpha in [0.5, 1.0, 1.5]:
                _check("ELU(F)", f"{dtype}/alpha{alpha}/{shape_name}",
                       F.elu(x, alpha=alpha), nn.ELU(alpha)(x), tol=0.0)
            _check("Hardsigmoid(F)", f"{dtype}/{shape_name}",
                   F.hardsigmoid(x), nn.Hardsigmoid()(x), tol=0.0)
            _check("Hardswish(F)", f"{dtype}/{shape_name}",
                   F.hardswish(x), nn.Hardswish()(x), tol=0.0)


def main():
    torch.manual_seed(0)
    print("=" * 80)
    print("Rigorous composition-equivalence tests")
    print("=" * 80)
    print()
    print("--- BatchNorm1d (compose via kb-nano BatchNorm2d, rank-agnostic) ---")
    test_batch_norm1d()
    print("\n--- BatchNorm3d (compose via kb-nano BatchNorm2d, rank-agnostic) ---")
    test_batch_norm3d()
    print("\n--- MaxPool1d (compose via kb-nano MaxPool2d (1, k) + reshape) ---")
    test_maxpool1d()
    print("\n--- AvgPool1d (compose via kb-nano AvgPool2d (1, k) + reshape) ---")
    test_avgpool1d()
    print("\n--- LeakyReLU/ELU/Hardsigmoid/Hardswish (torch builtins F.x; passthrough) ---")
    test_elementwise_activations_via_F_x()

    print()
    print("=" * 80)
    print(f"Total: {len(PASSED)} PASS, {len(FAILED)} FAIL")
    print("=" * 80)
    if FAILED:
        print("\nFAILURES (these ops cannot be safely treated as compositions; KEEP the L1 wrapper):")
        from collections import defaultdict
        by_op = defaultdict(list)
        for grp, name, diff in FAILED:
            by_op[grp].append((name, diff))
        for grp, items in by_op.items():
            print(f"  {grp}: {len(items)} failure(s)")
            for n, d in items[:3]:
                print(f"     {n}  (diff={d:.2e})")
        sys.exit(1)
    else:
        print("\nAll 8 ops are SAFELY composable. The corresponding L1 wrappers can be removed.")


if __name__ == "__main__":
    main()
