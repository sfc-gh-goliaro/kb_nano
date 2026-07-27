"""Check TRTLLMPrefill's non-causal paged path against a PyTorch reference.

This is the path added for Whisper cross-attention on Blackwell (the TRTLLM-gen
context kernel is causal-only, so non-causal paged prefill goes through
FlashInfer's BatchPrefillWithPagedKVCacheWrapper). Whisper runs but only matches
~32 of 444 tokens per request against vLLM, versus ~390 on H200, so the numerics
of this path are suspect and were never verified directly.

The cache is HND: [num_blocks, num_kv_heads, page_size, head_dim].
"""
from __future__ import annotations

import torch

from fastkernels.tasks.baseline.L1.flashinfer_prefill import TRTLLMPrefill

PAGE = 16


def reference(q, k_cache, v_cache, cu_q, cu_k, block_table, scale):
    """Per-request non-causal attention over the gathered paged K/V."""
    outs = []
    for i in range(cu_q.numel() - 1):
        qs, qe = int(cu_q[i]), int(cu_q[i + 1])
        klen = int(cu_k[i + 1] - cu_k[i])
        npages = (klen + PAGE - 1) // PAGE
        ks, vs = [], []
        for p in range(npages):
            blk = int(block_table[i, p])
            take = min(PAGE, klen - p * PAGE)
            # HND -> [page_size, heads, dim]
            ks.append(k_cache[blk, :, :take, :].permute(1, 0, 2))
            vs.append(v_cache[blk, :, :take, :].permute(1, 0, 2))
        k = torch.cat(ks, dim=0).float()          # [klen, H, D]
        v = torch.cat(vs, dim=0).float()
        qi = q[qs:qe].float()                      # [qlen, H, D]
        scores = torch.einsum("qhd,khd->hqk", qi, k) * scale
        probs = scores.softmax(dim=-1)
        outs.append(torch.einsum("hqk,khd->qhd", probs, v))
    return torch.cat(outs, dim=0)


def main() -> int:
    torch.manual_seed(0)
    dev = "cuda"
    H, D = 20, 64                  # whisper-large-v3: 20 heads, head_dim 64
    q_lens = [4, 1, 3, 4]          # decoder prompt lengths
    k_lens = [1500, 1500, 800, 37] # encoder lengths incl. a non-multiple of 16
    B = len(q_lens)
    ppr = (max(k_lens) + PAGE - 1) // PAGE
    nblocks = B * ppr + 8

    k_cache = torch.randn(nblocks, H, PAGE, D, dtype=torch.bfloat16, device=dev) * 0.5
    v_cache = torch.randn(nblocks, H, PAGE, D, dtype=torch.bfloat16, device=dev) * 0.5
    block_table = torch.arange(B * ppr, dtype=torch.int32, device=dev).view(B, ppr)
    q = torch.randn(sum(q_lens), H, D, dtype=torch.bfloat16, device=dev) * 0.5

    cu_q = torch.tensor([0] + list(torch.tensor(q_lens).cumsum(0)),
                        dtype=torch.int32, device=dev)
    cu_k = torch.tensor([0] + list(torch.tensor(k_lens).cumsum(0)),
                        dtype=torch.int32, device=dev)
    scale = D ** -0.5

    op = TRTLLMPrefill(H, H, D)
    got = op(q, k_cache, v_cache, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
             max_seqlen_q=max(q_lens), max_seqlen_k=max(k_lens),
             softmax_scale=scale, causal=False, block_table=block_table)
    torch.cuda.synchronize()
    exp = reference(q, k_cache, v_cache, cu_q, cu_k, block_table, scale)

    got_f, exp_f = got.float(), exp.float()
    cos = torch.nn.functional.cosine_similarity(
        got_f.reshape(1, -1), exp_f.reshape(1, -1), dim=-1).item()
    err = (got_f - exp_f).abs().max().item()
    print(f"shape {tuple(got.shape)} expected {tuple(exp.shape)}")
    print(f"overall cosine {cos:.8f}  max abs err {err:.4g}  mean|exp| {exp_f.abs().mean():.4g}")
    # Per-request, so a single bad sequence cannot hide in the aggregate.
    off = 0
    ok = True
    for i, ql in enumerate(q_lens):
        a, b = got_f[off:off + ql], exp_f[off:off + ql]
        c = torch.nn.functional.cosine_similarity(
            a.reshape(1, -1), b.reshape(1, -1), dim=-1).item()
        print(f"  seq{i} qlen={ql:<2} klen={k_lens[i]:<5} cosine {c:.8f} "
              f"maxerr {(a - b).abs().max().item():.4g}")
        ok &= c > 0.999
        off += ql
    print("PASS" if ok and err < 5e-2 else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
