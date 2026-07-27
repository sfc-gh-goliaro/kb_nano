"""Does the TRTLLM sinks/window decode path survive CUDA graph capture?

gpt-oss dies with cudaErrorIllegalInstruction / illegal memory access during
"[5/6] Compiling + capturing CUDA graphs", which is where the decode op runs
under ``torch.cuda.graph``. The earlier probe only exercised eager launches, so
capture is the untested half. Each case runs in its own subprocess because a
poisoned CUDA context would mask everything after it.
"""
from __future__ import annotations

import os
import subprocess
import sys

CASE = r'''
import sys
import torch
from fastkernels.tasks.baseline.L1.flashinfer_decode import TRTLLMDecode

use_sinks = sys.argv[1] == "1"
wl = int(sys.argv[2])
capture = sys.argv[3] == "1"

torch.manual_seed(0)
dev = "cuda"
# gpt-oss-120b at TP=2: 64 q heads / 8 kv heads / head_dim 64, sliding window 128.
NH, NKV, D, PAGE = 64, 8, 64, 16
B, CTX = 32, 1024
ppr = CTX // PAGE
nblocks = B * ppr + 16

k = torch.randn(nblocks, NKV, PAGE, D, dtype=torch.bfloat16, device=dev) * 0.1
v = torch.randn(nblocks, NKV, PAGE, D, dtype=torch.bfloat16, device=dev) * 0.1
bt = torch.arange(B * ppr, dtype=torch.int32, device=dev).view(B, ppr)
seq = torch.full((B,), CTX, dtype=torch.int32, device=dev)
q = torch.randn(B, NH, D, dtype=torch.bfloat16, device=dev) * 0.1
# gpt-oss stores sinks as a bf16 parameter, one per query head.
sinks = torch.randn(NH, dtype=torch.bfloat16, device=dev) if use_sinks else None

op = TRTLLMDecode(NH, NKV, D)
kw = dict(cache_seqlens=seq, block_table=bt, softmax_scale=D ** -0.5,
          max_seq_len=CTX, s_aux=sinks,
          window_size=(wl, 0) if wl >= 0 else None)

eager = op(q, k, v, **kw)
torch.cuda.synchronize()

if capture:
    g = torch.cuda.CUDAGraph()
    # Warm up on a side stream the way torch.cuda.graph docs require.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            op(q, k, v, **kw)
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        out = op(q, k, v, **kw)
    g.replay()
    torch.cuda.synchronize()
    diff = (out.float() - eager.float()).abs().max().item()
    print(f"OK capture sinks={use_sinks} wl={wl} replay_vs_eager_maxdiff={diff:.3g}")
else:
    print(f"OK eager sinks={use_sinks} wl={wl} mean={eager.float().abs().mean().item():.5f}")
'''


def main() -> int:
    path = "/tmp/_trtllm_graph_case.py"
    with open(path, "w") as f:
        f.write(CASE)
    env = dict(os.environ)
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    for capture in ("0", "1"):
        for use_sinks in ("0", "1"):
            for wl in ("-1", "127"):
                r = subprocess.run([sys.executable, path, use_sinks, wl, capture],
                                   capture_output=True, text=True, env=env)
                tag = f"capture={capture} sinks={use_sinks} wl={wl:>4}"
                if r.returncode == 0:
                    print(f"  PASS  {tag}  {r.stdout.strip().splitlines()[-1]}")
                else:
                    lines = [l for l in (r.stdout + r.stderr).splitlines() if l.strip()]
                    print(f"  FAIL  {tag}  rc={r.returncode}  {lines[-1][:150] if lines else '?'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
