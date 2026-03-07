"""TP-aware linear layers (L2 operators)."""

from __future__ import annotations

import torch
import torch.nn as nn

from ....infra.tp import _tp_size, _tp_rank
from ..L1.linear import Linear
from ..L1.allreduce import AllReduce


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
        output = self.linear_op(x, self.weight, self.bias)
        return output, None


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
        output = self.linear_op(x, self.weight, self.bias)
        return output, None


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
        output = self.linear_op(x, self.weight, self.bias)
        return output, None


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
        self.allreduce = AllReduce()

    def _weight_loader(self, param, loaded_weight):
        tp, rank = _tp_size(), _tp_rank()
        shard = param.data.size(1)
        loaded_weight = loaded_weight.narrow(1, rank * shard, shard)
        param.data.copy_(loaded_weight)

    def forward(self, x):
        y = self.linear_op(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            y = self.allreduce(y)
        return y, None
