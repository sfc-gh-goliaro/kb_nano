"""Isolate which TRTLLM argument faults on B200: sinks, window_left, or both.

gpt-oss-20b started dying with an async "illegal memory access" during CUDA
graph capture right after attention sinks and the sliding window were forwarded
to the TRTLLM-gen kernels. Exercise each combination on a small HND paged cache
in its own subprocess so a poisoned CUDA context cannot mask the next case.
"""
from __future__ import annotations

import os
import subprocess
import sys

CASE = r"""
import torch
from flashinfer.decode import trtllm_batch_decode_with_kv_cache
from flashinfer.prefill import trtllm_batch_context_with_kv_cache

mode, use_sinks, wl = sys.argv[1], sys.argv[2] == "1", int(sys.argv[3])

torch.manual_seed(0)
dev = "cuda"
NH, NKV, D, PAGE = 64, 8, 64, 16
B, CTX = 4, 512
pages_per_seq = CTX // PAGE
nblocks = B * pages_per_seq + 8

k = torch.randn(nblocks, NKV, PAGE, D, dtype=torch.bfloat16, device=dev)
v = torch.randn(nblocks, NKV, PAGE, D, dtype=torch.bfloat16, device=dev)
bt = torch.arange(B * pages_per_seq, dtype=torch.int32, device=dev).view(B, pages_per_seq)
ws = torch.zeros(512 * 1024 * 1024, dtype=torch.uint8, device=dev)
sinks = torch.randn(NH, dtype=torch.float32, device=dev) if use_sinks else None

if mode == "decode":
    q = torch.randn(B, NH, D, dtype=torch.bfloat16, device=dev)
    seq_lens = torch.full((B,), CTX, dtype=torch.int32, device=dev)
    out = trtllm_batch_decode_with_kv_cache(
        query=q, kv_cache=(k, v), workspace_buffer=ws, block_tables=bt,
        seq_lens=seq_lens, max_seq_len=CTX, bmm1_scale=D ** -0.5, bmm2_scale=1.0,
        window_left=wl, sinks=sinks, kv_layout="HND",
    )
else:
    QL = 32
    q = torch.randn(B * QL, NH, D, dtype=torch.bfloat16, device=dev)
    seq_lens = torch.full((B,), CTX, dtype=torch.int32, device=dev)
    cu_q = torch.arange(0, B * QL + 1, QL, dtype=torch.int32, device=dev)
    cu_k = torch.arange(0, B * CTX + 1, CTX, dtype=torch.int32, device=dev)
    out = trtllm_batch_context_with_kv_cache(
        query=q, kv_cache=(k, v), workspace_buffer=ws, block_tables=bt,
        seq_lens=seq_lens, max_q_len=QL, max_kv_len=CTX,
        bmm1_scale=D ** -0.5, bmm2_scale=1.0, batch_size=B,
        cum_seq_lens_q=cu_q, cum_seq_lens_kv=cu_k,
        window_left=wl, sinks=sinks, kv_layout="HND",
    )

torch.cuda.synchronize()
print(f"OK mode={mode} sinks={use_sinks} window_left={wl} "
      f"mean={out.float().abs().mean().item():.5f}")
"""


def main() -> int:
    script = "import sys\n" + CASE
    path = "/tmp/_trtllm_case.py"
    with open(path, "w") as f:
        f.write(script)

    env = dict(os.environ)
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    for mode in ("decode", "prefill"):
        for use_sinks in ("0", "1"):
            for wl in ("-1", "127"):
                r = subprocess.run([sys.executable, path, mode, use_sinks, wl],
                                   capture_output=True, text=True, env=env)
                tag = f"{mode:<8} sinks={use_sinks} window_left={wl:>4}"
                if r.returncode == 0:
                    print(f"  PASS  {tag}  {r.stdout.strip().splitlines()[-1]}")
                else:
                    last = [ln for ln in (r.stdout + r.stderr).splitlines() if ln.strip()]
                    print(f"  FAIL  {tag}  {last[-1][:160] if last else '?'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
