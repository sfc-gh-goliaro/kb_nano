"""MSA row attention with pair bias for AlphaFold3.

Row-wise attention over the MSA representation with additive pair bias.

Reference: openfold3/core/model/layers/msa.py MSARowAttentionWithPairBias
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from ..L1.openfold3_attention import OF3Attention


def _permute_final_dims(tensor: torch.Tensor, inds: tuple[int, ...]) -> torch.Tensor:
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


class MSARowAttentionWithPairBias(nn.Module):
    """AF3 MSA row attention with pair bias.

    Args:
        c_m: MSA input channel dimension
        c_z: Pair embedding channel dimension
        c_hidden: Per-head hidden channel dimension
        no_heads: Number of attention heads
        inf: Large constant for masking
    """

    def __init__(
        self,
        c_m: int,
        c_z: int,
        c_hidden: int,
        no_heads: int,
        inf: float = 1e9,
    ):
        super().__init__()
        self.c_m = c_m
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.inf = inf

        self.layer_norm_m = LayerNorm(c_m)
        self.layer_norm_z = LayerNorm(c_z)
        self.linear_z = Linear(c_z, no_heads, bias=False)

        self.mha = OF3Attention(
            c_q=c_m, c_k=c_m, c_v=c_m,
            c_hidden=c_hidden, no_heads=no_heads,
        )

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        chunk_size: int | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            m:    [*, N_seq, N_res, C_m] MSA embedding
            z:    [*, N_res, N_res, C_z] pair embedding
            mask: [*, N_seq, N_res] MSA mask

        Returns:
            [*, N_seq, N_res, C_m] updated MSA embedding
        """
        n_seq, n_res = m.shape[-3:-1]

        if mask is None:
            mask = m.new_ones(m.shape[:-3] + (n_seq, n_res))

        # [*, N_seq, 1, 1, N_res]
        mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]

        biases = [mask_bias]

        if z is not None:
            z_norm = self.layer_norm_z(z)
            z_proj = self.linear_z(z_norm)
            # [*, 1, no_heads, N_res, N_res]
            z_bias = _permute_final_dims(z_proj, (2, 0, 1)).unsqueeze(-4)
            biases.append(z_bias)

        m = self.layer_norm_m(m)
        m = self.mha(q_x=m, kv_x=m, biases=biases)

        return m
