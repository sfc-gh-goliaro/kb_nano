#!/usr/bin/env python3
"""Micro-benchmark of kb-nano's per-step decode CPU prep loop.

Goal: figure out where the ~3 ms/step CPU overhead in DeepSeek-V3.2 TP=8 BS=128
comes from, without booting the whole model.

We synthesize ``n_dc`` fake ``Sequence`` objects (just enough state for the
``_update_decode_arrays_incremental``/``_write_decode_shm`` hot path) and time
each component of the per-step CPU loop:

  prep_arrays    : _update_decode_arrays_incremental
  shm_write      : _write_decode_shm
  shm_signal     : _signal_workers (shared-memory seq-counter bump)
  post           : per-token append + EOS check (mirrors engine loop)

Run::

    PYTHONPATH=/home/yak python -m kb_nano.tests.debug.bench_perstep_cpu \
        --bs 128 --seq-len 4032 --steps 1000

Compare against vLLM's claimed CPU budget (~1 ms/step for DeepSeek TP=8 BS=128).
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

from kb_nano.infra.engine import BLOCK_SIZE  # type: ignore  # noqa: E402


def make_fake_seqs(n: int, seq_len: int, max_blocks: int):
    """Build minimal Sequence-like objects.

    The hot path only reads ``token_ids[-1]``, ``_num_tokens``, ``_last_token``,
    ``block_table``, and ``mrope_position_delta``.  We use ``list`` for
    ``block_table`` (matches engine).  Each seq starts at ``seq_len``.
    """
    seqs = []
    for i in range(n):
        s = type("S", (), {})()
        s.token_ids = None  # use _num_tokens / _last_token fast path
        s._num_tokens = seq_len
        s._last_token = 1234 + i
        # block_table = list of block ids
        nb = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        s.block_table = list(range(i * max_blocks, i * max_blocks + nb))
        s.mrope_position_delta = 0
        seqs.append(s)
    return seqs


# Strip down a ModelRunner to just the prep/SHM helpers we want to time.
class FakeRunner:
    SHM_SIZE = 32 * 1024 * 1024
    _SHM_FLAG_OFFSET = SHM_SIZE - 4
    _SHM_SEQ_OFFSET = SHM_SIZE - 8

    def __init__(self, max_bs: int, max_blocks: int, *, is_deepseek_mla: bool):
        self.is_qwen_vl = False
        self.is_deepseek_mla = is_deepseek_mla
        sm_np_dtype = np.int64 if is_deepseek_mla else np.int32
        self._np_ids = np.empty(max_bs, dtype=np.int64)
        self._np_pos = np.empty(max_bs, dtype=np.int64)
        self._np_sm = np.empty(max_bs, dtype=sm_np_dtype)
        self._np_cl = np.empty(max_bs, dtype=np.int32)
        self._np_bt = np.full((max_bs, max_blocks), -1, dtype=np.int32)
        self._prev_max_bt = 0
        self.shm = shared_memory.SharedMemory(create=True, size=self.SHM_SIZE)

    def cleanup(self):
        self.shm.close()
        self.shm.unlink()

    # Copied verbatim from engine.py to avoid import cycles
    def _update_decode_arrays_incremental(self, n, token_ids, decode_seqs):
        ids_np = self._np_ids
        pos_np = self._np_pos
        sm_np = self._np_sm
        cl_np = self._np_cl
        bt_np = self._np_bt
        bs = BLOCK_SIZE

        ids_np[:n] = token_ids
        pos_np[:n] += 1
        cl_np[:n] += 1
        sm_np[:n] += 1

        boundary_mask = cl_np[:n] % bs == 1
        if boundary_mask.any():
            max_bt = 0
            for i in np.where(boundary_mask)[0]:
                seq = decode_seqs[i]
                bt = seq.block_table
                blen = len(bt)
                sm_np[i] = bt[-1] * bs
                bt_np[i, :blen] = bt
                if blen > max_bt:
                    max_bt = blen
            if max_bt == 0:
                max_bt = self._prev_max_bt
            else:
                for i in np.where(~boundary_mask)[0]:
                    blen = len(decode_seqs[i].block_table)
                    if blen > max_bt:
                        max_bt = blen
                self._prev_max_bt = max_bt
        else:
            max_bt = self._prev_max_bt
        return (n, ids_np[:n], pos_np[:n], sm_np[:n], cl_np[:n],
                bt_np[:n, :max_bt])

    def _prepare_decode_arrays(self, seqs):
        n = len(seqs)
        ids_np = self._np_ids
        pos_np = self._np_pos
        sm_np = self._np_sm
        cl_np = self._np_cl
        max_bt = 0
        bs = BLOCK_SIZE
        for i, seq in enumerate(seqs):
            tids = seq.token_ids
            if tids is not None:
                slen = len(tids)
                ids_np[i] = tids[-1]
            else:
                slen = seq._num_tokens
                ids_np[i] = seq._last_token
            pos_np[i] = slen - 1
            cl_np[i] = slen
            bt = seq.block_table
            blen = len(bt)
            r = slen % bs
            sm_np[i] = bt[-1] * bs + (r - 1 if r else bs - 1)
            if blen > max_bt:
                max_bt = blen
        bt_np = self._np_bt
        for i, seq in enumerate(seqs):
            b = seq.block_table
            blen = len(b)
            bt_np[i, :blen] = b
            if blen < max_bt:
                bt_np[i, blen:max_bt] = -1
        self._prev_max_bt = max_bt
        return (n, ids_np[:n], pos_np[:n], sm_np[:n], cl_np[:n], bt_np[:n, :max_bt])

    def _write_decode_shm(self, n, ids_np, pos_np, sm_np, cl_np, bt_np):
        max_bt = bt_np.shape[1]
        buf = self.shm.buf
        buf[0:2] = n.to_bytes(2, "little")
        buf[2:4] = max_bt.to_bytes(2, "little")
        off = 4
        for arr in (ids_np, pos_np.ravel(), sm_np, cl_np, bt_np):
            nb = arr.nbytes
            buf[off:off+nb] = arr.tobytes()
            off += nb

    def _signal_workers(self):
        buf = self.shm.buf
        seq_off = self._SHM_SEQ_OFFSET
        cur = int.from_bytes(buf[seq_off:seq_off+4], "little")
        nxt = (cur + 1) & 0xFFFFFFFF
        buf[seq_off:seq_off+4] = nxt.to_bytes(4, "little")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--seq-len", type=int, default=4032,
                    help="initial seq length (each step appends 1 token)")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--max-blocks", type=int, default=128)
    ap.add_argument("--is-deepseek-mla", action="store_true",
                    help="use int64 slot_mapping")
    args = ap.parse_args()

    bs = args.bs
    seqs = make_fake_seqs(bs, args.seq_len, args.max_blocks)
    runner = FakeRunner(max_bs=bs, max_blocks=args.max_blocks,
                        is_deepseek_mla=args.is_deepseek_mla)

    runner._prepare_decode_arrays(seqs)
    token_ids = list(range(bs))
    eos = -1

    n_warmup = 50
    n = bs

    # ----- Time prep_arrays -----
    for _ in range(n_warmup):
        runner._update_decode_arrays_incremental(n, token_ids, seqs)

    t0 = time.perf_counter()
    for s in range(args.steps):
        runner._update_decode_arrays_incremental(n, token_ids, seqs)
    t_prep = (time.perf_counter() - t0) / args.steps * 1e6

    decode_data = runner._update_decode_arrays_incremental(n, token_ids, seqs)

    # ----- Time shm_write -----
    for _ in range(n_warmup):
        runner._write_decode_shm(*decode_data)
    t0 = time.perf_counter()
    for _ in range(args.steps):
        runner._write_decode_shm(*decode_data)
    t_shm = (time.perf_counter() - t0) / args.steps * 1e6

    # ----- Time _signal_workers -----
    for _ in range(n_warmup):
        runner._signal_workers()
    t0 = time.perf_counter()
    for _ in range(args.steps):
        runner._signal_workers()
    t_signal = (time.perf_counter() - t0) / args.steps * 1e6

    # ----- Time SHM flag byte (mr.shm.buf[OFF] = 1) -----
    flag_off = runner._SHM_FLAG_OFFSET
    for _ in range(n_warmup):
        runner.shm.buf[flag_off] = 1
    t0 = time.perf_counter()
    for _ in range(args.steps):
        runner.shm.buf[flag_off] = 1
    t_flag = (time.perf_counter() - t0) / args.steps * 1e6

    # ----- Time post-loop (append + EOS check) -----
    class FakeSeq2:
        __slots__ = ("generated_ids", "max_tokens", "ignore_eos")

        def __init__(self):
            self.generated_ids = []
            self.max_tokens = 1_000_000
            self.ignore_eos = True

        def append_token(self, tid):
            self.generated_ids.append(tid)
    decode_seqs = [FakeSeq2() for _ in range(bs)]

    for _ in range(n_warmup):
        any_finished = False
        for seq, tid in zip(decode_seqs, token_ids):
            seq.append_token(tid)
            done = len(seq.generated_ids) >= seq.max_tokens
            if not seq.ignore_eos:
                done = done or tid == eos
            if done:
                any_finished = True

    t0 = time.perf_counter()
    for _ in range(args.steps):
        any_finished = False
        for seq, tid in zip(decode_seqs, token_ids):
            seq.append_token(tid)
            done = len(seq.generated_ids) >= seq.max_tokens
            if not seq.ignore_eos:
                done = done or tid == eos
            if done:
                any_finished = True
    t_post = (time.perf_counter() - t0) / args.steps * 1e6

    # ----- Time block_table.append on boundary check -----
    for s in seqs:
        s._num_tokens = args.seq_len
    t0 = time.perf_counter()
    for _ in range(args.steps):
        for seq in seqs:
            if (seq._num_tokens % BLOCK_SIZE) == 1:
                seq.block_table.append(0)
            seq._num_tokens += 1  # roll forward
    t_btappend = (time.perf_counter() - t0) / args.steps * 1e6

    print(f"\n=== Per-step CPU micro-benchmark (BS={bs}, seq_len={args.seq_len}, steps={args.steps}) ===")
    print(f"  block_table boundary check (loop)  : {t_btappend:7.1f} us  -- bm.free_block_ids loop")
    print(f"  _update_decode_arrays_incremental  : {t_prep:7.1f} us  -- numpy vectorized prep")
    print(f"  _write_decode_shm (n*max_bt={n*args.max_blocks}) : {t_shm:7.1f} us  -- 5 SHM memcpy")
    print(f"  shm.buf[FLAG_OFF] = 1              : {t_flag:7.1f} us")
    print(f"  _signal_workers (seq counter bump) : {t_signal:7.1f} us")
    print(f"  post-loop (append + EOS)           : {t_post:7.1f} us")
    total = t_btappend + t_prep + t_shm + t_flag + t_signal + t_post
    print(f"  ---")
    print(f"  TOTAL (excludes graph launch + GPU wait): {total:7.1f} us  ({total/1000:.2f} ms)")
    print(f"\nReference: kb-nano DeepSeek-V3.2 TP=8 BS=128 wall-time gap = "
          f"~99 ms / 32 steps = ~3100 us per step")

    runner.cleanup()


if __name__ == "__main__":
    main()
