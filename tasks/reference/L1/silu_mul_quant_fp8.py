"""Fused SiLU-mul + per-token-group FP8 quantization with column-major scales.

Pure-PyTorch semantic reference.  The baseline is a single Triton kernel that:
1. Reads the gate/up pair from the first MoE GEMM output [M, 2*N]
2. Computes SiLU(gate) * up -> [M, N]
3. Quantizes each group of 128 elements to FP8 with UE8M0 power-of-two scales
4. Writes FP8 output and column-major scales

Two details of the kernel are load-bearing and reproduced here:

* The SiLU result is cast back to the **input dtype** before being multiplied
  by ``up`` (the kernel does ``.to(y_ptr.dtype.element_ty)`` on the activation,
  then promotes the product to fp32).  Skipping that round-trip changes the
  low bits of every expert activation.
* ``output_scales`` is allocated as ``(N_2 // 128, M)`` and transposed, i.e.
  column-major with the group index on the outer axis.  DeepGEMM's grouped
  GEMM reads the scales in that layout, so this is not merely a view detail.

Matches vLLM's ``silu_mul_per_token_group_quant_fp8_colmajor``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

_FP8_INFO = torch.finfo(torch.float8_e4m3fn)
_GROUP_SIZE = 128


class SiluMulQuantFp8(nn.Module):
    """Fused SiLU-mul + per-token-group FP8 quantization (colmajor scales)."""

    def forward(
        self,
        input: torch.Tensor,
        output: torch.Tensor | None = None,
        use_ue8m0: bool = True,
        eps: float = 1e-10,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fused SiLU-mul + per-token-group FP8 quantization.

        Args:
            input: [M, N] where N = 2 * intermediate_size (gate/up concatenated)
            output: Optional pre-allocated [M, N//2] FP8 output buffer
            use_ue8m0: Use power-of-two (UE8M0) scales for DeepGEMM
            eps: Minimum absmax to avoid division by zero

        Returns:
            (output_fp8, output_scales) where output_fp8 is [M, N//2] in
            float8_e4m3fn and output_scales is [M, (N//2)//128] in float32
            (column-major layout)
        """
        assert input.ndim == 2
        M, N = input.size()
        N_2 = N // 2

        assert M % _GROUP_SIZE == 0, f"M={M} must be divisible by {_GROUP_SIZE}"
        assert N_2 % _GROUP_SIZE == 0, f"N//2={N_2} must be divisible by {_GROUP_SIZE}"

        if output is None:
            output = torch.empty(
                (M, N_2), dtype=torch.float8_e4m3fn, device=input.device,
            )

        output_scales = torch.empty(
            (N_2 // _GROUP_SIZE, M), dtype=torch.float32, device=input.device,
        ).transpose(0, 1)

        fp8_min = _FP8_INFO.min
        fp8_max = _FP8_INFO.max

        gate = input[:, :N_2]
        up = input[:, N_2:]

        # SiLU in fp32, then back to the input dtype before the multiply --
        # mirrors the kernel's cast.
        gate_f32 = gate.to(torch.float32)
        silu = (gate_f32 / (1.0 + torch.exp(-gate_f32))).to(input.dtype)
        y = (silu * up).to(torch.float32)                        # [M, N_2]

        groups = y.view(M, N_2 // _GROUP_SIZE, _GROUP_SIZE)
        absmax = groups.abs().amax(dim=-1).clamp_min(eps)         # [M, G]
        scale = absmax / fp8_max
        if use_ue8m0:
            scale = torch.exp2(torch.ceil(torch.log2(scale)))

        y_q = torch.clamp(
            groups / scale.unsqueeze(-1), fp8_min, fp8_max,
        ).to(output.dtype)

        output.copy_(y_q.view(M, N_2))
        output_scales.copy_(scale)
        return output, output_scales
