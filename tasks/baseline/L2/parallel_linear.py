"""TP-aware linear layers (L2 operators)."""

from __future__ import annotations

import torch
import torch.nn as nn

from ....infra.tp import _tp_size, _tp_rank
from ..L1.linear import Linear
from ..L1.allreduce import AllReduce


def _ceildiv(a: int, b: int) -> int:
    return -(-a // b)


class ColumnParallelLinear(nn.Module):
    """Splits output dim across TP ranks."""

    def __init__(self, input_size: int, output_size: int, bias: bool = False,
                 fp8_block_size: tuple[int, int] | None = None):
        super().__init__()
        tp = _tp_size()
        assert output_size % tp == 0
        self.output_size_per_partition = output_size // tp
        out, inp = self.output_size_per_partition, input_size

        if fp8_block_size is not None:
            from ..L1.fp8_linear import FP8Linear
            self.weight = nn.Parameter(
                torch.empty(out, inp, dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            bs_n, bs_k = fp8_block_size
            self.weight_scale_inv = nn.Parameter(
                torch.ones(_ceildiv(out, bs_n), _ceildiv(inp, bs_k), dtype=torch.float32),
                requires_grad=False,
            )
            self.weight_scale_inv.weight_loader = self._scale_weight_loader
            self.linear_op = FP8Linear(fp8_block_size)
        else:
            self.weight = nn.Parameter(torch.empty(out, inp))
            self.linear_op = Linear()

        self.weight.weight_loader = self._weight_loader
        self.bias = nn.Parameter(torch.empty(self.output_size_per_partition)) if bias else None
        if self.bias is not None:
            self.bias.weight_loader = self._weight_loader

    def _weight_loader(self, param, loaded_weight):
        tp, rank = _tp_size(), _tp_rank()
        shard = param.data.size(0)
        loaded_weight = loaded_weight.narrow(0, rank * shard, shard)
        param.data.copy_(loaded_weight)

    def _scale_weight_loader(self, param, loaded_weight):
        tp, rank = _tp_size(), _tp_rank()
        shard = param.data.size(0)
        loaded_weight = loaded_weight.narrow(0, rank * shard, shard)
        param.data.copy_(loaded_weight)

    def forward(self, x):
        if hasattr(self, "weight_scale_inv"):
            return self.linear_op(x, self.weight, self.weight_scale_inv, self.bias)
        return self.linear_op(x, self.weight, self.bias)


class MergedColumnParallelLinear(nn.Module):
    """gate_proj + up_proj merged into one linear, sharded across TP."""

    def __init__(self, input_size: int, output_sizes: list[int], bias: bool = False,
                 fp8_block_size: tuple[int, int] | None = None):
        super().__init__()
        tp = _tp_size()
        self.output_sizes = output_sizes
        total = sum(output_sizes)
        assert all(s % tp == 0 for s in output_sizes)
        out, inp = total // tp, input_size

        if fp8_block_size is not None:
            from ..L1.fp8_linear import FP8Linear
            self._fp8_block_size = fp8_block_size
            self.weight = nn.Parameter(
                torch.empty(out, inp, dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            bs_n, bs_k = fp8_block_size
            self.weight_scale_inv = nn.Parameter(
                torch.ones(_ceildiv(out, bs_n), _ceildiv(inp, bs_k), dtype=torch.float32),
                requires_grad=False,
            )
            self.weight_scale_inv.weight_loader = self._scale_weight_loader
            self.linear_op = FP8Linear(fp8_block_size)
        else:
            self._fp8_block_size = None
            self.weight = nn.Parameter(torch.empty(out, inp))
            self.linear_op = Linear()

        self.weight.weight_loader = self._weight_loader
        self.bias = None

    def _weight_loader(self, param, loaded_weight, shard_id: int):
        tp, rank = _tp_size(), _tp_rank()
        shard_offset = sum(self.output_sizes[:shard_id]) // tp
        shard_size = self.output_sizes[shard_id] // tp
        dst = param.data.narrow(0, shard_offset, shard_size)
        src = loaded_weight.chunk(tp, 0)[rank]
        dst.copy_(src)

    def _scale_weight_loader(self, param, loaded_weight, shard_id: int):
        tp, rank = _tp_size(), _tp_rank()
        bs_n = self._fp8_block_size[0]
        src = loaded_weight.chunk(tp, 0)[rank]
        scale_rows_per_shard = [_ceildiv(s // tp, bs_n) for s in self.output_sizes]
        shard_offset = sum(scale_rows_per_shard[:shard_id])
        dst = param.data.narrow(0, shard_offset, src.shape[0])
        dst.copy_(src)

    def forward(self, x):
        if hasattr(self, "weight_scale_inv"):
            return self.linear_op(x, self.weight, self.weight_scale_inv, self.bias)
        return self.linear_op(x, self.weight, self.bias)

    def forward_fp8(self, input_fp8, input_scale):
        """Forward with pre-quantized FP8 input."""
        return self.linear_op(input_fp8, self.weight, self.weight_scale_inv,
                              self.bias, input_scale=input_scale)


class QKVParallelLinear(nn.Module):
    """Q, K, V projections merged and sharded across TP."""

    def __init__(self, hidden_size: int, head_size: int,
                 total_num_heads: int, total_num_kv_heads: int,
                 bias: bool = False,
                 fp8_block_size: tuple[int, int] | None = None):
        super().__init__()
        tp = _tp_size()
        self.head_size = head_size
        self.num_heads = total_num_heads // tp
        self.num_kv_heads = total_num_kv_heads // tp
        output_size = (self.num_heads + 2 * self.num_kv_heads) * head_size

        if fp8_block_size is not None:
            from ..L1.fp8_linear import FP8Linear
            self._fp8_block_size = fp8_block_size
            self.weight = nn.Parameter(
                torch.empty(output_size, hidden_size, dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            bs_n, bs_k = fp8_block_size
            self.weight_scale_inv = nn.Parameter(
                torch.ones(_ceildiv(output_size, bs_n), _ceildiv(hidden_size, bs_k),
                           dtype=torch.float32),
                requires_grad=False,
            )
            self.weight_scale_inv.weight_loader = self._scale_weight_loader
            self.linear_op = FP8Linear(fp8_block_size)
        else:
            self._fp8_block_size = None
            self.weight = nn.Parameter(torch.empty(output_size, hidden_size))
            self.linear_op = Linear()

        self.weight.weight_loader = self._weight_loader
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self._weight_loader

    def _shard_info(self, shard_id: str):
        q_size = self.num_heads * self.head_size
        kv_size = self.num_kv_heads * self.head_size
        if shard_id == "q":
            return 0, q_size
        elif shard_id == "k":
            return q_size, kv_size
        else:
            return q_size + kv_size, kv_size

    def _weight_loader(self, param, loaded_weight, shard_id: str):
        tp, rank = _tp_size(), _tp_rank()
        shard_offset, shard_size = self._shard_info(shard_id)
        dst = param.data.narrow(0, shard_offset, shard_size)
        src = loaded_weight.chunk(tp, 0)[rank]
        dst.copy_(src)

    def _scale_weight_loader(self, param, loaded_weight, shard_id: str):
        tp, rank = _tp_size(), _tp_rank()
        bs_n = self._fp8_block_size[0]
        q_size = self.num_heads * self.head_size
        kv_size = self.num_kv_heads * self.head_size
        shard_sizes = {"q": q_size, "k": kv_size, "v": kv_size}
        shard_order = ["q", "k", "v"]
        scale_offset = sum(_ceildiv(shard_sizes[s], bs_n) for s in shard_order[:shard_order.index(shard_id)])
        src = loaded_weight.chunk(tp, 0)[rank]
        dst = param.data.narrow(0, scale_offset, src.shape[0])
        dst.copy_(src)

    def forward(self, x):
        if hasattr(self, "weight_scale_inv"):
            return self.linear_op(x, self.weight, self.weight_scale_inv, self.bias)
        return self.linear_op(x, self.weight, self.bias)

    def forward_fp8(self, input_fp8, input_scale):
        """Forward with pre-quantized FP8 input."""
        return self.linear_op(input_fp8, self.weight, self.weight_scale_inv,
                              self.bias, input_scale=input_scale)


class RowParallelLinear(nn.Module):
    """Splits input dim across TP ranks, all-reduces output."""

    def __init__(self, input_size: int, output_size: int, bias: bool = False,
                 fp8_block_size: tuple[int, int] | None = None):
        super().__init__()
        tp = _tp_size()
        assert input_size % tp == 0
        self.input_size_per_partition = input_size // tp
        out, inp = output_size, self.input_size_per_partition

        if fp8_block_size is not None:
            from ..L1.fp8_linear import FP8Linear
            self.weight = nn.Parameter(
                torch.empty(out, inp, dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            bs_n, bs_k = fp8_block_size
            self.weight_scale_inv = nn.Parameter(
                torch.ones(_ceildiv(out, bs_n), _ceildiv(inp, bs_k), dtype=torch.float32),
                requires_grad=False,
            )
            self.weight_scale_inv.weight_loader = self._scale_weight_loader
            self.linear_op = FP8Linear(fp8_block_size)
        else:
            self.weight = nn.Parameter(torch.empty(out, inp))
            self.linear_op = Linear()

        self.weight.weight_loader = self._weight_loader
        self.tp_size = tp
        self.tp_rank = _tp_rank()
        self.bias = nn.Parameter(torch.empty(output_size)) if bias else None
        if self.bias is not None:
            self.bias.weight_loader = lambda p, w: p.data.copy_(w)
        self.allreduce = AllReduce()

    def _weight_loader(self, param, loaded_weight):
        tp, rank = _tp_size(), _tp_rank()
        shard = param.data.size(1)
        loaded_weight = loaded_weight.narrow(1, rank * shard, shard)
        param.data.copy_(loaded_weight)

    def _scale_weight_loader(self, param, loaded_weight):
        tp, rank = _tp_size(), _tp_rank()
        shard = param.data.size(1)
        loaded_weight = loaded_weight.narrow(1, rank * shard, shard)
        param.data.copy_(loaded_weight)

    def forward(self, x):
        if hasattr(self, "weight_scale_inv"):
            y = self.linear_op(x, self.weight, self.weight_scale_inv,
                               self.bias if self.tp_rank == 0 else None)
        else:
            y = self.linear_op(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            y = self.allreduce(y)
        return y
