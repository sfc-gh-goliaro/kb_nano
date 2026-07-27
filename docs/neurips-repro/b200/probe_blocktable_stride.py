"""Does trtllm-gen decode honour a column-sliced block table's stride?

The eager decode path passes

    block_tables = self._eager_block_tables[:n, :bt_cols]      # infra/engine.py

which is a *column* slice of a wider allocation: its rows are bt_cols long but its
stride(0) is the full allocated width. The captured-graph path instead passes
``block_tables[:bs]`` -- full width, so stride(0) == size(-1).

If flashinfer's launcher computes the row stride from size(-1) rather than stride(0), the
eager path reads the wrong pages for every row except row 0, which would explain the
measured signature exactly: exact_matches 0-1 of 1000, and agreement collapsing to ~23
tokens on precisely the runs that reach the eager path.

This compares one decode against the identical table copied into a contiguous tensor.
"""
from __future__ import annotations

import torch
from flashinfer.decode import trtllm_batch_decode_with_kv_cache

HQ, HKV, D = 32, 8, 128
PAGE = 16
B = 8
SEQ = 600
ALLOC_COLS = 128                      # what the engine preallocates (max_model_len/page)


def main() -> int:
    torch.manual_seed(0)
    dev = "cuda"
    scale = D ** -0.5
    npages = (SEQ + PAGE - 1) // PAGE          # 38 used columns
    nblocks = B * npages + 8

    kv = torch.randn(nblocks, 2, HKV, PAGE, D, dtype=torch.bfloat16, device=dev) * 0.3
    k_cache, v_cache = kv[:, 0].contiguous(), kv[:, 1].contiguous()
    q = torch.randn(B, HQ, D, dtype=torch.bfloat16, device=dev) * 0.3
    seqlens = torch.full((B,), SEQ, dtype=torch.int32, device=dev)

    # Wide allocation, as the engine does; fill the used columns with real page ids and
    # the tail with a DIFFERENT value so a stride bug shows up as wrong data, not zeros.
    wide = torch.full((B, ALLOC_COLS), 12345, dtype=torch.int32, device=dev)
    wide[:, :npages] = torch.arange(B * npages, dtype=torch.int32,
                                    device=dev).view(B, npages)
    sliced = wide[:, :npages]                       # non-contiguous: stride(0)=128
    packed = sliced.contiguous()                    # stride(0)=38

    print(f"sliced : shape {tuple(sliced.shape)} stride {sliced.stride()} "
          f"contiguous={sliced.is_contiguous()}")
    print(f"packed : shape {tuple(packed.shape)} stride {packed.stride()} "
          f"contiguous={packed.is_contiguous()}")

    ws = torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device=dev)

    def run(bt):
        out = trtllm_batch_decode_with_kv_cache(
            query=q, kv_cache=(k_cache, v_cache), workspace_buffer=ws,
            block_tables=bt, seq_lens=seqlens, max_seq_len=SEQ,
            bmm1_scale=scale, bmm2_scale=1.0, kv_layout="HND")
        torch.cuda.synchronize()
        return out.reshape(B, HQ, D).float()

    a = run(sliced)
    b = run(packed)

    print("\nper-row difference, sliced vs contiguous (identical page ids either way):")
    bad = 0
    for i in range(B):
        d = (a[i] - b[i]).abs().max().item()
        cos = torch.nn.functional.cosine_similarity(
            a[i].reshape(1, -1), b[i].reshape(1, -1), dim=-1).item()
        flag = "" if d == 0.0 else "   <-- WRONG"
        if d != 0.0:
            bad += 1
        print(f"  row {i}: max abs diff {d:.6g}  cos {cos:.8f}{flag}")
    print(f"\n{bad}/{B} rows differ. A stride-aware kernel would give 0/{B}.")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
