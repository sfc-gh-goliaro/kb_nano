"""Input embedder for AlphaFold3.

Produces initial single (s) and pair (z) representations from token features.

Reference: openfold3/core/model/feature_embedders/input_embedders.py
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear


class InputEmbedder(nn.Module):
    """Produces initial single and pair representations from token features.

    Args:
        c_token: Token feature dimension (input)
        c_s: Single representation dimension
        c_z: Pair representation dimension
        relpos_k: Maximum relative position for pair bias (default 32)
    """

    def __init__(self, c_token: int, c_s: int, c_z: int, relpos_k: int = 32):
        super().__init__()
        self.c_token = c_token
        self.c_s = c_s
        self.c_z = c_z
        self.relpos_k = relpos_k

        self.linear_s = Linear(c_token, c_s, bias=False)

        self.linear_z_i = Linear(c_token, c_z, bias=False)
        self.linear_z_j = Linear(c_token, c_z, bias=False)

        n_relpos_bins = 2 * relpos_k + 1
        self.linear_relpos = Linear(n_relpos_bins, c_z, bias=False)

    def _relpos_encoding(
        self, residue_index: torch.Tensor,
    ) -> torch.Tensor:
        """Compute relative position encoding.

        Args:
            residue_index: [*, N_token] residue indices

        Returns:
            [*, N_token, N_token, C_z] relative position embedding
        """
        d = residue_index[..., :, None] - residue_index[..., None, :]
        d = d.clamp(-self.relpos_k, self.relpos_k) + self.relpos_k

        n_bins = 2 * self.relpos_k + 1
        one_hot = torch.nn.functional.one_hot(d.long(), n_bins).to(
            dtype=self.linear_relpos.weight.dtype
        )
        return self.linear_relpos(one_hot)

    def forward(
        self,
        token_features: torch.Tensor,
        residue_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            token_features: [*, N_token, C_token] per-token features
            residue_index:  [*, N_token] residue indices

        Returns:
            s: [*, N_token, C_s] single representation
            z: [*, N_token, N_token, C_z] pair representation
        """
        s = self.linear_s(token_features)

        z_i = self.linear_z_i(token_features)[..., :, None, :]
        z_j = self.linear_z_j(token_features)[..., None, :, :]
        z = z_i + z_j

        z = z + self._relpos_encoding(residue_index)

        return s, z
