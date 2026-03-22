"""Tensor-parallel, expert-parallel, and data-parallel utilities.

Process group layout for DP×TP (with EP folding DP×TP for MoE):

  World size = dp_size × tp_size
  EP size    = dp_size × tp_size  (all ranks participate in expert dispatch)

Groups:
  - TP group:  ranks sharing the same DP shard (size = tp_size)
  - DP group:  ranks at the same TP position   (size = dp_size)
  - EP group:  all ranks (size = dp_size × tp_size) — used for MoE dispatch/combine
"""

from __future__ import annotations

import torch.distributed as dist
from torch.distributed import ProcessGroup


_TP_GROUP: ProcessGroup | None = None
_DP_GROUP: ProcessGroup | None = None
_EP_GROUP: ProcessGroup | None = None

_TP_SIZE: int = 1
_TP_RANK: int = 0
_DP_SIZE: int = 1
_DP_RANK: int = 0
_EP_SIZE: int = 1
_EP_RANK: int = 0


def _tp_size():
    return _TP_SIZE

def _tp_rank():
    return _TP_RANK

def _dp_size():
    return _DP_SIZE

def _dp_rank():
    return _DP_RANK

def _ep_size():
    return _EP_SIZE

def _ep_rank():
    return _EP_RANK

def get_tp_group() -> ProcessGroup | None:
    return _TP_GROUP

def get_dp_group() -> ProcessGroup | None:
    return _DP_GROUP

def get_ep_group() -> ProcessGroup | None:
    return _EP_GROUP


def init_parallel_groups(
    tp_size: int = 1,
    dp_size: int = 1,
    enable_expert_parallel: bool = False,
) -> None:
    """Initialize TP, DP, and EP process groups.

    Must be called after dist.init_process_group.
    Rank layout: global_rank = dp_rank * tp_size + tp_rank
    """
    global _TP_GROUP, _DP_GROUP, _EP_GROUP
    global _TP_SIZE, _TP_RANK, _DP_SIZE, _DP_RANK, _EP_SIZE, _EP_RANK

    if not dist.is_initialized():
        _TP_SIZE = 1
        _TP_RANK = 0
        _DP_SIZE = 1
        _DP_RANK = 0
        _EP_SIZE = 1
        _EP_RANK = 0
        return

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    assert world_size == dp_size * tp_size, (
        f"world_size ({world_size}) != dp_size ({dp_size}) * tp_size ({tp_size})"
    )

    _TP_SIZE = tp_size
    _DP_SIZE = dp_size
    _TP_RANK = rank % tp_size
    _DP_RANK = rank // tp_size

    # TP groups: ranks within each DP shard
    for dp in range(dp_size):
        tp_ranks = list(range(dp * tp_size, (dp + 1) * tp_size))
        group = dist.new_group(tp_ranks)
        if rank in tp_ranks:
            _TP_GROUP = group

    # DP groups: ranks at the same TP position across DP shards
    for tp in range(tp_size):
        dp_ranks = [tp + dp * tp_size for dp in range(dp_size)]
        group = dist.new_group(dp_ranks)
        if rank in dp_ranks:
            _DP_GROUP = group

    # EP group: all ranks (for MoE dispatch/combine when EP is enabled)
    if enable_expert_parallel and world_size > 1:
        all_ranks = list(range(world_size))
        _EP_GROUP = dist.new_group(all_ranks)
        _EP_SIZE = world_size
        _EP_RANK = rank
    else:
        _EP_SIZE = 1
        _EP_RANK = 0
