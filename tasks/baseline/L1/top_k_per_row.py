"""Top-K per row selection for DSA indexer."""

from __future__ import annotations

import torch
import torch.nn as nn


class TopKPerRow(nn.Module):
    """Top-K per row selection for sparse attention indexer."""

    def forward_prefill(
        self,
        logits: torch.Tensor,
        cu_seqlen_ks: torch.Tensor,
        cu_seqlen_ke: torch.Tensor,
        topk: int,
    ) -> torch.Tensor:
        """Top-K within variable row boundaries.

        Args:
            logits: ``[M, max_seq_len]`` float32 logits.
            cu_seqlen_ks: ``[M]`` row start offsets (inclusive).
            cu_seqlen_ke: ``[M]`` row end offsets (exclusive).
            topk: number of indices to keep per row.

        Returns:
            ``indices``: ``[M, topk]`` int32 — global column indices; unused
            entries are ``-1``.
        """
        M = logits.shape[0]
        indices = torch.full((M, topk), -1, dtype=torch.int32, device=logits.device)

        for i in range(M):
            start = int(cu_seqlen_ks[i].item())
            end = int(cu_seqlen_ke[i].item())
            row_len = end - start

            if row_len <= 0:
                continue

            row = logits[i, start:end]
            k = min(topk, row_len)
            _, top_ids = row.topk(k, dim=-1)
            indices[i, :k] = (top_ids + start).to(torch.int32)

        return indices

    def forward_decode(
        self,
        logits: torch.Tensor,
        seq_lens: torch.Tensor,
        next_n: int,
        topk: int,
    ) -> torch.Tensor:
        """Top-K for decode rows with per-sequence length caps.

        Args:
            logits: ``[B * next_n, max_seq_len]`` float32 logits.
            seq_lens: ``[B]`` current sequence lengths.
            next_n: speculative tokens per sequence (row stride).
            topk: number of indices per row.

        Returns:
            ``indices``: ``[B * next_n, topk]`` int32; padding is ``-1``.
        """
        total_rows = logits.shape[0]
        indices = torch.full(
            (total_rows, topk), -1, dtype=torch.int32, device=logits.device,
        )
        B = seq_lens.shape[0]

        for b in range(B):
            sl = int(seq_lens[b].item())
            for n in range(next_n):
                row_idx = b * next_n + n
                if row_idx >= total_rows:
                    break

                effective_len = sl + n
                if effective_len <= 0:
                    continue

                row = logits[row_idx, :effective_len]
                k = min(topk, effective_len)
                _, top_ids = row.topk(k, dim=-1)
                indices[row_idx, :k] = top_ids.to(torch.int32)

        return indices
