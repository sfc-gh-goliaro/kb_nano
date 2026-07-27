"""Does our GDN decode path lose precision relative to vLLM's fused one?

Qwen3-Next reports a good speedup on B200 but poor token agreement with vLLM.
One structural difference between the two decode paths:

  ours  : _fused_gdn_gating writes g (fp32) and beta (**bf16**) to memory, then
          fused_recurrent_gated_delta_rule consumes them.
  vLLM  : fused_sigmoid_gating_delta_rule_update takes A_log/a/b/dt_bias and
          keeps g and beta in fp32 registers inside the recurrent kernel.

vLLM's own unit test only checks a single step (atol 1e-2). The recurrent state is
multiplied by exp(g) every step, so a per-step gate error compounds over a
512-token decode. This probe iterates the real kernels for many steps and measures
drift against an fp32 run of the same kernel, which isolates the gate precision
from everything else.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from vllm.model_executor.layers.fla.ops import (
    fused_recurrent_gated_delta_rule,
    fused_sigmoid_gating_delta_rule_update,
)

from fastkernels.tasks.baseline.L2.qwen3_next_gdn_attention import _fused_gdn_gating

STEPS = 512          # one decode-heavy scenario's worth of tokens
B = 4                # requests
HK, HV = 16, 32      # qwen3-next: 16 k heads, 32 v heads
DK, DV = 128, 128


def cos(a, b):
    return F.cosine_similarity(a.reshape(1, -1).float(),
                               b.reshape(1, -1).float(), dim=-1).item()


def run(mode: str, dtype, inputs, state):
    """Iterate the recurrent update for STEPS steps, returning the last output."""
    state = state.clone().to(dtype)
    cu = torch.arange(0, B + 1, dtype=torch.int32, device="cuda")
    idx = torch.arange(B, dtype=torch.int32, device="cuda")
    out = None
    for q, k, v, a, b, A_log, dt_bias in inputs:
        q, k, v = (t.to(dtype) for t in (q, k, v))
        a, b = a.to(dtype), b.to(dtype)
        A_log, dt_bias = A_log.to(dtype), dt_bias.to(dtype)
        if mode == "ours":
            # exactly what qwen3_next_gdn_attention.forward does on decode
            g, beta = _fused_gdn_gating(A_log, a, b, dt_bias)
            out, _ = fused_recurrent_gated_delta_rule(
                q=q, k=k, v=v, g=g, beta=beta,
                initial_state=state, inplace_final_state=True,
                cu_seqlens=cu.to(torch.long), ssm_state_indices=idx,
                use_qk_l2norm_in_kernel=True)
        else:
            out, _ = fused_sigmoid_gating_delta_rule_update(
                A_log=A_log, a=a, b=b, dt_bias=dt_bias,
                q=q, k=k, v=v,
                initial_state=state, inplace_final_state=True,
                cu_seqlens=cu, ssm_state_indices=idx,
                use_qk_l2norm_in_kernel=True)
    return out.float(), state.float()


def main() -> int:
    torch.manual_seed(0)
    dev = "cuda"
    inputs = []
    for _ in range(STEPS):
        inputs.append((
            torch.randn(1, B, HK, DK, dtype=torch.float32, device=dev) * 0.5,
            torch.randn(1, B, HK, DK, dtype=torch.float32, device=dev) * 0.5,
            torch.randn(1, B, HV, DV, dtype=torch.float32, device=dev) * 0.5,
            torch.randn(B, HV, dtype=torch.float32, device=dev),
            torch.randn(B, HV, dtype=torch.float32, device=dev),
            torch.randn(HV, dtype=torch.float32, device=dev) * 0.5,
            torch.randn(HV, dtype=torch.float32, device=dev) * 0.5,
        ))
    state0 = torch.randn(B, HV, DK, DV, dtype=torch.float32, device=dev) * 0.1

    ref_out, ref_state = run("vllm", torch.float32, inputs, state0)
    ours_out, ours_state = run("ours", torch.bfloat16, inputs, state0)
    vllm_out, vllm_state = run("vllm", torch.bfloat16, inputs, state0)
    torch.cuda.synchronize()

    print(f"after {STEPS} decode steps, vs the same kernels in fp32:")
    for name, o, s in (("ours (g fp32, beta bf16, separate kernel)", ours_out, ours_state),
                       ("vllm (gating fused, fp32 internally)", vllm_out, vllm_state)):
        print(f"  {name:<44} out cos {cos(o, ref_out):.6f}  "
              f"state cos {cos(s, ref_state):.6f}  "
              f"out maxerr {(o - ref_out).abs().max().item():.4g}")
    print(f"  ours vs vllm (both bf16)                      out cos "
          f"{cos(ours_out, vllm_out):.6f}  state cos {cos(ours_state, vllm_state):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
