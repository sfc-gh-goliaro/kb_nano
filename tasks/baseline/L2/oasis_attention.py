"""Compatibility exports for Oasis L2 attention modules.

The concrete operators live in one file per module. ``OasisVAEAttentionBlock``
was moved to ``tasks.baseline.L3.oasis_vae_attention_block`` because it
composes attention, MLP, norms, and residuals.
"""

from .oasis_spatial_axial_attention import OasisSpatialAxialAttention
from .oasis_temporal_axial_attention import OasisTemporalAxialAttention
from .oasis_vae_attention import OasisVAEAttention

__all__ = [
    "OasisSpatialAxialAttention",
    "OasisTemporalAxialAttention",
    "OasisVAEAttention",
]
