"""Semantic PyTorch reference for chunk_retention.

This file is used for specification/prompting and optional validation only.
It is not the production baseline and should not be used for reported speed.
"""

from __future__ import annotations

from kb_nano.tasks.reference.L1.fused_recurrent_retention import FusedRecurrentRetention


class ChunkRetention(FusedRecurrentRetention):
    pass
