"""Tensor-parallel aware layers adapted from nano-vllm."""

from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn

from .context import get_context
from ..tasks.L1.linear import Linear
from ..tasks.L1.embedding import Embedding


# ---------------------------------------------------------------------------
# TP helpers
# ---------------------------------------------------------------------------
def _tp_size():
    return dist.get_world_size() if dist.is_initialized() else 1

def _tp_rank():
    return dist.get_rank() if dist.is_initialized() else 0


# ---------------------------------------------------------------------------
# Global custom allreduce communicator (set by engine, used by TP layers)
# ---------------------------------------------------------------------------
_CUSTOM_AR: Optional["CustomAllreduce"] = None  # noqa: F821


def set_custom_ar(ar):
    global _CUSTOM_AR
    _CUSTOM_AR = ar


def get_custom_ar():
    return _CUSTOM_AR


# ---------------------------------------------------------------------------
# TP-aware linear layers
# ---------------------------------------------------------------------------
class ColumnParallelLinear(nn.Module):
    """Splits output dim across TP ranks."""

    def __init__(self, input_size: int, output_size: int, bias: bool = False):
        super().__init__()
        tp = _tp_size()
        assert output_size % tp == 0
        self.output_size_per_partition = output_size // tp
        self.weight = nn.Parameter(torch.empty(self.output_size_per_partition, input_size))
        self.weight.weight_loader = self._weight_loader
        self.bias = nn.Parameter(torch.empty(self.output_size_per_partition)) if bias else None
        if self.bias is not None:
            self.bias.weight_loader = self._weight_loader
        self.linear_op = Linear()

    def _weight_loader(self, param, loaded_weight):
        tp, rank = _tp_size(), _tp_rank()
        shard = param.data.size(0)
        loaded_weight = loaded_weight.narrow(0, rank * shard, shard)
        param.data.copy_(loaded_weight)

    def forward(self, x):
        return self.linear_op(x, self.weight, self.bias)


class MergedColumnParallelLinear(nn.Module):
    """gate_proj + up_proj merged into one linear, sharded across TP."""

    def __init__(self, input_size: int, output_sizes: list[int], bias: bool = False):
        super().__init__()
        tp = _tp_size()
        self.output_sizes = output_sizes
        total = sum(output_sizes)
        assert all(s % tp == 0 for s in output_sizes)
        self.weight = nn.Parameter(torch.empty(total // tp, input_size))
        self.weight.weight_loader = self._weight_loader
        self.bias = None
        self.linear_op = Linear()

    def _weight_loader(self, param, loaded_weight, shard_id: int):
        tp, rank = _tp_size(), _tp_rank()
        shard_offset = sum(self.output_sizes[:shard_id]) // tp
        shard_size = self.output_sizes[shard_id] // tp
        dst = param.data.narrow(0, shard_offset, shard_size)
        src = loaded_weight.chunk(tp, 0)[rank]
        dst.copy_(src)

    def forward(self, x):
        return self.linear_op(x, self.weight, self.bias)


class QKVParallelLinear(nn.Module):
    """Q, K, V projections merged and sharded across TP."""

    def __init__(self, hidden_size: int, head_size: int,
                 total_num_heads: int, total_num_kv_heads: int,
                 bias: bool = False):
        super().__init__()
        tp = _tp_size()
        self.head_size = head_size
        self.num_heads = total_num_heads // tp
        self.num_kv_heads = total_num_kv_heads // tp
        output_size = (self.num_heads + 2 * self.num_kv_heads) * head_size
        self.weight = nn.Parameter(torch.empty(output_size, hidden_size))
        self.weight.weight_loader = self._weight_loader
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self._weight_loader
        self.linear_op = Linear()

    def _weight_loader(self, param, loaded_weight, shard_id: str):
        tp, rank = _tp_size(), _tp_rank()
        if shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        dst = param.data.narrow(0, shard_offset, shard_size)
        src = loaded_weight.chunk(tp, 0)[rank]
        dst.copy_(src)

    def forward(self, x):
        return self.linear_op(x, self.weight, self.bias)


class RowParallelLinear(nn.Module):
    """Splits input dim across TP ranks, all-reduces output."""

    def __init__(self, input_size: int, output_size: int, bias: bool = False):
        super().__init__()
        tp = _tp_size()
        assert input_size % tp == 0
        self.input_size_per_partition = input_size // tp
        self.weight = nn.Parameter(torch.empty(output_size, self.input_size_per_partition))
        self.weight.weight_loader = self._weight_loader
        self.tp_size = tp
        self.tp_rank = _tp_rank()
        self.bias = nn.Parameter(torch.empty(output_size)) if bias else None
        if self.bias is not None:
            self.bias.weight_loader = lambda p, w: p.data.copy_(w)
        self.linear_op = Linear()

    def _weight_loader(self, param, loaded_weight):
        tp, rank = _tp_size(), _tp_rank()
        shard = param.data.size(1)
        loaded_weight = loaded_weight.narrow(1, rank * shard, shard)
        param.data.copy_(loaded_weight)

    def forward(self, x):
        y = self.linear_op(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            ar = _CUSTOM_AR
            if ar is not None:
                out = ar.custom_all_reduce(y)
                if out is not None:
                    return out
            dist.all_reduce(y)
        return y


# ---------------------------------------------------------------------------
# TP-aware embedding and LM head
# ---------------------------------------------------------------------------
class VocabParallelEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        tp, rank = _tp_size(), _tp_rank()
        assert num_embeddings % tp == 0
        self.num_embeddings = num_embeddings
        self.per_partition = num_embeddings // tp
        self.vocab_start = self.per_partition * rank
        self.vocab_end = self.vocab_start + self.per_partition
        self.tp_size = tp
        self.weight = nn.Parameter(torch.empty(self.per_partition, embedding_dim))
        self.weight.weight_loader = self._weight_loader
        self.embedding_op = Embedding()

    def _weight_loader(self, param, loaded_weight):
        tp, rank = _tp_size(), _tp_rank()
        shard = param.data.size(0)
        param.data.copy_(loaded_weight.narrow(0, rank * shard, shard))

    def forward(self, x):
        if self.tp_size > 1:
            mask = (x >= self.vocab_start) & (x < self.vocab_end)
            x = mask * (x - self.vocab_start)
        y = self.embedding_op(x, self.weight)
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y
            ar = _CUSTOM_AR
            if ar is not None:
                out = ar.custom_all_reduce(y)
                if out is not None:
                    return out
            dist.all_reduce(y)
        return y


class ParallelLMHead(VocabParallelEmbedding):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim)
        self.linear_op = Linear()

    def project(self, x):
        """Linear projection only (no gather). Used inside CUDA graph."""
        ctx = get_context()
        if ctx.is_prefill or ctx.is_mixed:
            last_indices = ctx.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        return self.linear_op(x, self.weight)

    def gather_logits(self, logits):
        """Gather partial logits from all ranks. Used outside CUDA graph."""
        if self.tp_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if _tp_rank() == 0 else None
            dist.gather(logits, all_logits, 0)
            logits = torch.cat(all_logits, -1) if _tp_rank() == 0 else logits
        return logits

    def gather_greedy(self, logits):
        """Fast path for greedy: local argmax + small allgather.

        Instead of gathering full vocab logits (~31MB/rank), gather only
        the (max_val, max_idx) per sequence (~2KB/rank).
        Returns token IDs directly on rank 0, None on other ranks.
        """
        if self.tp_size <= 1:
            return None

        rank = _tp_rank()
        local_max_vals, local_max_idxs = logits.max(dim=-1)
        local_max_idxs = local_max_idxs + self.vocab_start

        info = torch.stack([local_max_vals, local_max_idxs.float()], dim=-1)
        gathered = [torch.empty_like(info) for _ in range(self.tp_size)]
        dist.all_gather(gathered, info)
        if rank == 0:
            all_info = torch.stack(gathered, dim=0)
            all_vals = all_info[:, :, 0]
            all_idxs = all_info[:, :, 1].long()
            best_rank = all_vals.argmax(dim=0)
            bs = logits.size(0)
            token_ids = all_idxs[best_rank, torch.arange(bs, device=logits.device)]
            return token_ids
        return None

    def forward(self, x):
        logits = self.project(x)
        return self.gather_logits(logits)
