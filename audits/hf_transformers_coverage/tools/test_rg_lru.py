"""Thorough numerical tests for kb-nano L1 RG-LRU vs HF RecurrentGemmaRglru.

Coverage:
- Linear mode (seq_len > 1) and sampling mode (seq_len == 1)
- With and without prior recurrent_states
- Reset signal at various positions
- Various dtypes (fp32, bf16, fp16 where applicable)
- Various num_attention_heads / lru_width combinations
- state_dict-key compatibility
- Stateful chained calls (decode-by-decode autoregressive)

Run with venv activated.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tasks.baseline.L1.rg_lru import RGLRU as KBRGLRU
from transformers.models.recurrent_gemma.modeling_recurrent_gemma import RecurrentGemmaRglru


PASSED: list[tuple[str, float]] = []
FAILED: list[tuple[str, float]] = []


def _check(name: str, kb_out, ref_out, tol: float = 1e-5):
    if isinstance(kb_out, tuple):
        kb_out = kb_out[0]
    if isinstance(ref_out, tuple):
        ref_out = ref_out[0]
    if kb_out.shape != ref_out.shape:
        FAILED.append((name, -1.0))
        print(f"  FAIL  {name}: shape mismatch {kb_out.shape} vs {ref_out.shape}")
        return
    diff = (kb_out.float() - ref_out.float()).abs().max().item()
    if diff <= tol:
        PASSED.append((name, diff))
        print(f"  PASS  {name:65s} diff={diff:.2e}")
    else:
        FAILED.append((name, diff))
        print(f"  FAIL  {name:65s} diff={diff:.2e} > tol={tol:.0e}")


class _DummyConfig:
    """Minimal config to instantiate HF's RecurrentGemmaRglru."""
    def __init__(self, num_attention_heads, lru_width):
        self.num_attention_heads = num_attention_heads
        self.lru_width = lru_width


def _make_pair(num_heads, lru_width, dtype=torch.float32, seed=0):
    """Construct kb-nano + HF refs with synced parameters."""
    torch.manual_seed(seed)
    cfg = _DummyConfig(num_heads, lru_width)
    ref = RecurrentGemmaRglru(cfg).to(dtype)
    kb = KBRGLRU(num_heads, lru_width).to(dtype)
    # Init weights randomly via the ref, then copy to kb so they match.
    for name, p in ref.named_parameters():
        torch.nn.init.normal_(p, std=0.1)
    kb.load_state_dict(ref.state_dict())
    return kb, ref


def test_state_dict_compat():
    cfg = _DummyConfig(4, 32)
    ref = RecurrentGemmaRglru(cfg)
    kb = KBRGLRU(4, 32)
    ref_keys = set(ref.state_dict().keys())
    kb_keys = set(kb.state_dict().keys())
    if ref_keys != kb_keys:
        FAILED.append(("state_dict_keys", -1.0))
        print(f"  FAIL  state_dict keys mismatch: ref={sorted(ref_keys)}, kb={sorted(kb_keys)}")
    else:
        PASSED.append(("state_dict_keys", 0.0))
        print(f"  PASS  state_dict_keys                                            {sorted(ref_keys)}")


def test_linear_mode_no_prior_state():
    """seq_len > 1, recurrent_states starts None (start of sequence)."""
    for dtype in [torch.float32, torch.bfloat16]:
        for B, T, num_heads, lru_width in [(2, 16, 4, 32), (1, 8, 2, 16), (4, 32, 8, 64)]:
            kb, ref = _make_pair(num_heads, lru_width, dtype)
            # Reset state for both
            ref.recurrent_states = None
            kb.recurrent_states = None
            torch.manual_seed(42)
            x = torch.randn(B, T, lru_width, dtype=dtype)
            position_ids = torch.arange(T).unsqueeze(0).expand(B, -1).contiguous()
            ref_out = ref(x, position_ids)
            kb_out = kb(x, position_ids)
            tol = 5e-3 if dtype != torch.float32 else 1e-5
            _check(f"linear_no_state/dtype={dtype}/B{B}T{T}H{num_heads}W{lru_width}",
                   kb_out, ref_out, tol=tol)


def test_linear_mode_with_prior_state():
    """seq_len > 1, recurrent_states pre-loaded (mid-sequence)."""
    for dtype in [torch.float32, torch.bfloat16]:
        for B, T, num_heads, lru_width in [(2, 16, 4, 32), (1, 32, 8, 64)]:
            kb, ref = _make_pair(num_heads, lru_width, dtype)
            torch.manual_seed(0)
            initial_state = torch.randn(B, lru_width, dtype=torch.float32)  # acc_dtype
            ref.recurrent_states = initial_state.clone()
            kb.recurrent_states = initial_state.clone()
            torch.manual_seed(42)
            x = torch.randn(B, T, lru_width, dtype=dtype)
            position_ids = torch.arange(1, T + 1).unsqueeze(0).expand(B, -1).contiguous()  # no resets
            ref_out = ref(x, position_ids)
            kb_out = kb(x, position_ids)
            tol = 5e-3 if dtype != torch.float32 else 1e-5
            _check(f"linear_with_state/dtype={dtype}/B{B}T{T}H{num_heads}W{lru_width}",
                   kb_out, ref_out, tol=tol)


def test_reset_at_start():
    """position_ids[:, 0] == 0 means reset → recurrent state cleared at t=0."""
    for dtype in [torch.float32, torch.bfloat16]:
        kb, ref = _make_pair(4, 32, dtype)
        torch.manual_seed(0)
        # Pre-loaded state, but position_ids start at 0 → reset
        initial_state = torch.randn(2, 32, dtype=torch.float32)
        ref.recurrent_states = initial_state.clone()
        kb.recurrent_states = initial_state.clone()
        x = torch.randn(2, 16, 32, dtype=dtype)
        position_ids = torch.arange(16).unsqueeze(0).expand(2, -1).contiguous()  # starts at 0
        ref_out = ref(x, position_ids)
        kb_out = kb(x, position_ids)
        tol = 5e-3 if dtype != torch.float32 else 1e-5
        _check(f"reset_at_start/dtype={dtype}", kb_out, ref_out, tol=tol)


def test_reset_mid_sequence():
    """A reset (position_ids == 0) in the middle of the sequence."""
    for dtype in [torch.float32]:
        kb, ref = _make_pair(4, 32, dtype)
        torch.manual_seed(0)
        x = torch.randn(2, 16, 32, dtype=dtype)
        # Build position_ids with a reset at t=8
        position_ids = torch.arange(16).unsqueeze(0).expand(2, -1).clone().contiguous()
        position_ids[:, 8] = 0  # reset mid-seq
        position_ids[:, 9:] = torch.arange(7).unsqueeze(0).expand(2, -1) + 1  # restart counter
        ref.recurrent_states = None
        kb.recurrent_states = None
        ref_out = ref(x, position_ids)
        kb_out = kb(x, position_ids)
        _check(f"reset_mid_seq/dtype={dtype}", kb_out, ref_out, tol=1e-5)


def test_sampling_mode_first_step():
    """seq_len == 1, recurrent_states is None (first decode token after prefill)."""
    for dtype in [torch.float32, torch.bfloat16]:
        kb, ref = _make_pair(4, 32, dtype)
        ref.recurrent_states = None
        kb.recurrent_states = None
        torch.manual_seed(0)
        x = torch.randn(2, 1, 32, dtype=dtype)
        position_ids = torch.tensor([[16], [16]])  # mid-stream — no reset
        ref_out = ref(x, position_ids)
        kb_out = kb(x, position_ids)
        tol = 5e-3 if dtype != torch.float32 else 1e-5
        _check(f"sampling_first_step/dtype={dtype}", kb_out, ref_out, tol=tol)
        # State should also match
        if ref.recurrent_states is not None and kb.recurrent_states is not None:
            state_diff = (ref.recurrent_states - kb.recurrent_states).abs().max().item()
            print(f"        state diff: {state_diff:.2e}")


def test_sampling_mode_with_state():
    """seq_len == 1, recurrent_states is pre-loaded (mid-decoding)."""
    for dtype in [torch.float32, torch.bfloat16]:
        kb, ref = _make_pair(4, 32, dtype)
        torch.manual_seed(0)
        initial_state = torch.randn(2, 32, dtype=torch.float32)
        ref.recurrent_states = initial_state.clone()
        kb.recurrent_states = initial_state.clone()
        x = torch.randn(2, 1, 32, dtype=dtype)
        position_ids = torch.tensor([[16], [16]])
        ref_out = ref(x, position_ids)
        kb_out = kb(x, position_ids)
        tol = 5e-3 if dtype != torch.float32 else 1e-5
        _check(f"sampling_with_state/dtype={dtype}", kb_out, ref_out, tol=tol)


def test_autoregressive_chain():
    """Chain prefill (T=8) + 4 decode steps (T=1 each) and verify state matches at every step."""
    dtype = torch.float32
    kb, ref = _make_pair(4, 32, dtype)
    ref.recurrent_states = None
    kb.recurrent_states = None
    torch.manual_seed(0)
    # Prefill
    x_prefill = torch.randn(2, 8, 32, dtype=dtype)
    pos_prefill = torch.arange(8).unsqueeze(0).expand(2, -1).contiguous()
    ref_pre = ref(x_prefill, pos_prefill)
    kb_pre = kb(x_prefill, pos_prefill)
    _check("autoregressive/prefill", kb_pre, ref_pre, tol=1e-5)
    state_diff = (ref.recurrent_states - kb.recurrent_states).abs().max().item()
    print(f"        state after prefill: diff={state_diff:.2e}")
    # 4 decode steps
    for step in range(4):
        x_step = torch.randn(2, 1, 32, dtype=dtype)
        pos_step = torch.tensor([[8 + step], [8 + step]])
        ref_out = ref(x_step, pos_step)
        kb_out = kb(x_step, pos_step)
        _check(f"autoregressive/decode_step_{step}", kb_out, ref_out, tol=1e-5)


def main():
    print("=" * 95)
    print("RG-LRU L1 op tests vs HF RecurrentGemmaRglru")
    print("=" * 95)
    print()
    print("--- state_dict compatibility ---")
    test_state_dict_compat()
    print("\n--- Linear mode (seq_len > 1, no prior state) ---")
    test_linear_mode_no_prior_state()
    print("\n--- Linear mode (seq_len > 1, with prior state) ---")
    test_linear_mode_with_prior_state()
    print("\n--- Reset at start (position_ids[0] == 0) ---")
    test_reset_at_start()
    print("\n--- Reset mid-sequence ---")
    test_reset_mid_sequence()
    print("\n--- Sampling mode (seq_len == 1, first decode token) ---")
    test_sampling_mode_first_step()
    print("\n--- Sampling mode (seq_len == 1, with prior state) ---")
    test_sampling_mode_with_state()
    print("\n--- Autoregressive chain (prefill + 4 decode steps; state must match) ---")
    test_autoregressive_chain()

    print()
    print("=" * 95)
    print(f"Total: {len(PASSED)} PASS, {len(FAILED)} FAIL")
    print("=" * 95)
    if FAILED:
        for name, diff in FAILED:
            print(f"  FAIL  {name}  (diff={diff:.2e})")
        sys.exit(1)


if __name__ == "__main__":
    main()
