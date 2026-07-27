"""Does a sequence's decode output depend on who else is in the batch?

Everything else is eliminated: TRTLLM and flash_attn prefill are both 1.95e-3 from
fp32 (1.3e-3 apart), TRTLLM and the FlashInfer wrapper decode are both ~2e-3 from
fp32, and forcing the reference onto TRTLLM decode at every batch size did not move
full-scale Llama agreement at all (27/21/21, same as auto).

But probe_decode_accuracy.py showed TRTLLM's error against fp32 *moving* with batch
size (2.77e-3 at batch 1 -> 2.20e-3 at batch >= 32) where flash_attn's stayed flat
(2.14e-3 -> 2.19e-3). If the TRTLLM-gen kernel splits work across the batch, one
sequence's result depends on its neighbours -- and then agreement with a reference
that batches differently must decay as occupancy grows, which is exactly the
observed monotone decline with queue position.

This holds one sequence fixed and changes only the rest of the batch.
"""
from __future__ import annotations

import torch

from fastkernels.tasks.baseline.L1.flashinfer_decode import TRTLLMDecode
from fastkernels.tasks.baseline.L1.flash_attn_decode import FlashAttnDecode

HQ, HKV, D = 32, 8, 128
TARGET_LEN = 1024
PAGE_HND, PAGE_NHD = 16, 256
BATCHES = (1, 8, 64, 256, 512, 1000)


def build(k, v, lens, page, layout):
    """Paged cache + block table for a ragged batch."""
    ppr = max((n + page - 1) // page for n in lens)
    nblk = len(lens) * ppr
    bt = torch.arange(nblk, dtype=torch.int32, device=k[0].device).view(len(lens), ppr)
    if layout == "HND":
        kc = torch.zeros(nblk, HKV, page, D, dtype=k[0].dtype, device=k[0].device)
    else:
        kc = torch.zeros(nblk, page, HKV, D, dtype=k[0].dtype, device=k[0].device)
    vc = torch.zeros_like(kc)
    for b, n in enumerate(lens):
        for p in range((n + page - 1) // page):
            lo, hi = p * page, min((p + 1) * page, n)
            if layout == "HND":
                kc[bt[b, p], :, : hi - lo] = k[b][lo:hi].permute(1, 0, 2)
                vc[bt[b, p], :, : hi - lo] = v[b][lo:hi].permute(1, 0, 2)
            else:
                kc[bt[b, p], : hi - lo] = k[b][lo:hi]
                vc[bt[b, p], : hi - lo] = v[b][lo:hi]
    return kc, vc, bt


def main() -> int:
    torch.manual_seed(0)
    dev = "cuda"
    scale = D ** -0.5
    # The one sequence we track. Index 0 of every batch.
    q0 = torch.randn(1, HQ, D, dtype=torch.bfloat16, device=dev) * 0.3
    k0 = torch.randn(TARGET_LEN, HKV, D, dtype=torch.bfloat16, device=dev) * 0.3
    v0 = torch.randn(TARGET_LEN, HKV, D, dtype=torch.bfloat16, device=dev) * 0.3

    ops = (("TRTLLM/HND", TRTLLMDecode(HQ, HKV, D), PAGE_HND, "HND"),
           ("flash_attn/NHD", FlashAttnDecode(HQ, HKV, D), PAGE_NHD, "NHD"))
    base = {}
    print(f"same sequence (len {TARGET_LEN}), only its batch neighbours change")
    print(f"{'batch':>6}  " + "  ".join(f"{n:>34}" for n, _, _, _ in ops))
    for B in BATCHES:
        gen = torch.Generator(device=dev).manual_seed(1234 + B)
        # Ragged neighbours, so the batch is realistic rather than uniform.
        lens = [TARGET_LEN] + [int(torch.randint(64, 2048, (1,), generator=gen,
                                                 device=dev).item())
                               for _ in range(B - 1)]
        ks = [k0] + [torch.randn(n, HKV, D, dtype=torch.bfloat16, device=dev,
                                 generator=gen) * 0.3 for n in lens[1:]]
        vs = [v0] + [torch.randn(n, HKV, D, dtype=torch.bfloat16, device=dev,
                                 generator=gen) * 0.3 for n in lens[1:]]
        q = torch.cat([q0] + [torch.randn(1, HQ, D, dtype=torch.bfloat16,
                                          device=dev, generator=gen) * 0.3
                              for _ in range(B - 1)], 0)
        seqlens = torch.tensor(lens, dtype=torch.int32, device=dev)

        cells = []
        for name, op, page, layout in ops:
            kc, vc, bt = build(ks, vs, lens, page, layout)
            out = op(q, kc, vc, cache_seqlens=seqlens, block_table=bt,
                     softmax_scale=scale, max_seq_len=max(lens))
            tgt = out.reshape(B, HQ, D)[0].float()
            torch.cuda.synchronize()
            if name not in base:
                base[name] = tgt
                cells.append("(reference, batch 1)")
            else:
                d = ((tgt - base[name]).norm() / base[name].norm()).item()
                bits = (tgt != base[name]).float().mean().item()
                cells.append(f"rel {d:.3e}  differing elems {bits:.1%}")
        print(f"{B:>6}  " + "  ".join(f"{c:>34}" for c in cells))
    print("\nA nonzero row means the kernel's answer for one sequence depends on\n"
          "what else is being decoded alongside it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
