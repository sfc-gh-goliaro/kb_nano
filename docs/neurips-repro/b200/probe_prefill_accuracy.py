"""Is the Blackwell TRTLLM *prefill* kernel the inaccurate one?

Chain of elimination so far, on full-scale Llama-3.1 (1000 prompts):
  * default TRTLLM path              -> 27 / 21 / 21 matched tokens
  * NHD pin (flash_attn prefill+decode) -> 205 / 256 / 254
  * reference forced onto TRTLLM decode at every batch size -> 27 / 21 / 21

So making the *decode* kernels match changed nothing, while the NHD pin -- which
swaps prefill *and* decode -- recovered an order of magnitude. That leaves prefill.
probe_decode_accuracy.py showed both decode kernels sit 2.2e-3 from fp32; this asks
the same question of the two prefill kernels, on a ragged batch with real per-request
lengths.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from fastkernels.tasks.baseline.L1.flashinfer_prefill import TRTLLMPrefill
from fastkernels.tasks.baseline.L1.flash_attn_varlen import FlashAttnVarlen

HQ, HKV, D = 32, 8, 128
PAGE = 16
LENS = [953, 512, 480, 1148, 406, 842, 137, 2001]   # wildchat-like prompt lengths


def reference(q, k, v, lens, scale):
    """fp32 causal attention per request."""
    outs = []
    off = 0
    g = HQ // HKV
    for n in lens:
        qi = q[off:off + n].float()                                # [n, HQ, D]
        ki = k[off:off + n].float().repeat_interleave(g, dim=1)
        vi = v[off:off + n].float().repeat_interleave(g, dim=1)
        s = torch.einsum("qhd,khd->hqk", qi, ki) * scale
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=q.device), 1)
        s = s.masked_fill(mask, float("-inf"))
        outs.append(torch.einsum("hqk,khd->qhd", s.softmax(-1), vi))
        off += n
    return torch.cat(outs, 0)


def build_hnd(k, v, lens):
    """Paged HND cache holding each request's K/V, plus its block table."""
    ppr = max((n + PAGE - 1) // PAGE for n in lens)
    nblk = len(lens) * ppr
    bt = torch.arange(nblk, dtype=torch.int32, device=k.device).view(len(lens), ppr)
    kc = torch.zeros(nblk, HKV, PAGE, D, dtype=k.dtype, device=k.device)
    vc = torch.zeros_like(kc)
    off = 0
    for b, n in enumerate(lens):
        for p in range((n + PAGE - 1) // PAGE):
            lo = p * PAGE
            hi = min(lo + PAGE, n)
            kc[bt[b, p], :, : hi - lo] = k[off + lo:off + hi].permute(1, 0, 2)
            vc[bt[b, p], :, : hi - lo] = v[off + lo:off + hi].permute(1, 0, 2)
        off += n
    return kc, vc, bt


def main() -> int:
    torch.manual_seed(0)
    dev = "cuda"
    scale = D ** -0.5
    total = sum(LENS)
    q = torch.randn(total, HQ, D, dtype=torch.bfloat16, device=dev) * 0.3
    k = torch.randn(total, HKV, D, dtype=torch.bfloat16, device=dev) * 0.3
    v = torch.randn(total, HKV, D, dtype=torch.bfloat16, device=dev) * 0.3
    cu = torch.tensor([0] + list(torch.tensor(LENS).cumsum(0)),
                      dtype=torch.int32, device=dev)
    exp = reference(q, k, v, LENS, scale)

    kc, vc, bt = build_hnd(k, v, LENS)
    trt = TRTLLMPrefill(HQ, HKV, D)
    got_t = trt(q, kc, vc, cu_seqlens_q=cu, cu_seqlens_k=cu,
                max_seqlen_q=max(LENS), max_seqlen_k=max(LENS),
                softmax_scale=scale, causal=True, block_table=bt)
    fa = FlashAttnVarlen()
    got_f = fa(q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu,
               max_seqlen_q=max(LENS), max_seqlen_k=max(LENS),
               softmax_scale=scale, causal=True)
    torch.cuda.synchronize()

    print(f"ragged causal prefill, lens={LENS}")
    print(f"{'kernel':<28} {'rel err vs fp32':>16}  {'min per-req cosine':>20}")
    for name, got in (("TRTLLM-gen paged (HND/16)", got_t),
                      ("flash_attn varlen", got_f)):
        g = got.reshape(total, HQ, D).float()
        rel = ((g - exp).norm() / exp.norm()).item()
        cs = []
        off = 0
        for n in LENS:
            cs.append(F.cosine_similarity(g[off:off + n].reshape(1, -1),
                                          exp[off:off + n].reshape(1, -1),
                                          dim=-1).item())
            off += n
        print(f"{name:<28} {rel:>16.3e}  {min(cs):>20.7f}")

    gt, gf = got_t.reshape(total, HQ, D).float(), got_f.reshape(total, HQ, D).float()
    print(f"{'trtllm vs flash_attn':<28} {((gt - gf).norm() / gf.norm()).item():>16.3e}")

    # Per-request, so one bad length cannot hide in the aggregate.
    print("\nper-request rel err (trtllm | flash_attn):")
    off = 0
    for n in LENS:
        a = gt[off:off + n]
        b = gf[off:off + n]
        e = exp[off:off + n]
        print(f"  len {n:>5}: {((a - e).norm() / e.norm()).item():.3e} | "
              f"{((b - e).norm() / e.norm()).item():.3e}")
        off += n
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
