"""Auxiliary prediction heads for AlphaFold3.

Distogram, pLDDT, PAE, PDE confidence heads that produce binned logits
from single and pair representations.

Reference: openfold3/core/model/heads/head_modules.py AuxiliaryHeadsAllAtom
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear


class DistogramHead(nn.Module):
    """Predicts inter-residue distance distribution.

    Args:
        c_z: Pair embedding channel dimension
        no_bins: Number of distance bins
    """

    def __init__(self, c_z: int, no_bins: int = 64):
        super().__init__()
        self.linear = Linear(c_z, no_bins, bias=True)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [*, N_token, N_token, C_z] pair embedding

        Returns:
            [*, N_token, N_token, no_bins] distogram logits
        """
        logits = self.linear(z)
        logits = logits + logits.transpose(-2, -3)
        return logits


class PLDDTHead(nn.Module):
    """Predicts per-atom pLDDT confidence.

    Args:
        c_s: Single embedding channel dimension
        no_bins: Number of pLDDT bins
    """

    def __init__(self, c_s: int, no_bins: int = 50):
        super().__init__()
        self.layer_norm = LayerNorm(c_s)
        self.linear = Linear(c_s, no_bins, bias=True)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s: [*, N_token, C_s] single embedding

        Returns:
            [*, N_token, no_bins] pLDDT logits
        """
        return self.linear(self.layer_norm(s))


class PAEHead(nn.Module):
    """Predicts Predicted Aligned Error (PAE).

    Args:
        c_z: Pair embedding channel dimension
        no_bins: Number of PAE bins
    """

    def __init__(self, c_z: int, no_bins: int = 64):
        super().__init__()
        self.linear = Linear(c_z, no_bins, bias=True)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [*, N_token, N_token, C_z] pair embedding

        Returns:
            [*, N_token, N_token, no_bins] PAE logits
        """
        return self.linear(z)


class PDEHead(nn.Module):
    """Predicts Predicted Distance Error (PDE).

    Args:
        c_z: Pair embedding channel dimension
        no_bins: Number of PDE bins
    """

    def __init__(self, c_z: int, no_bins: int = 64):
        super().__init__()
        self.linear = Linear(c_z, no_bins, bias=True)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [*, N_token, N_token, C_z] pair embedding

        Returns:
            [*, N_token, N_token, no_bins] PDE logits
        """
        logits = self.linear(z)
        logits = logits + logits.transpose(-2, -3)
        return logits


class AuxiliaryHeads(nn.Module):
    """All auxiliary prediction heads for AF3.

    Args:
        c_s: Single embedding channel dimension
        c_z: Pair embedding channel dimension
    """

    def __init__(self, c_s: int = 384, c_z: int = 128):
        super().__init__()
        self.distogram = DistogramHead(c_z, no_bins=64)
        self.plddt = PLDDTHead(c_s, no_bins=50)
        self.pae = PAEHead(c_z, no_bins=64)
        self.pde = PDEHead(c_z, no_bins=64)

    def forward(
        self, s: torch.Tensor, z: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            s: [*, N_token, C_s] single embedding
            z: [*, N_token, N_token, C_z] pair embedding

        Returns:
            Dict with distogram_logits, plddt_logits, pae_logits, pde_logits
        """
        return {
            "distogram_logits": self.distogram(z),
            "plddt_logits": self.plddt(s),
            "pae_logits": self.pae(z),
            "pde_logits": self.pde(z),
        }
