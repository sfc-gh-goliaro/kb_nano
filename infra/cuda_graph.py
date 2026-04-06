"""CUDA graph capture and replay wrapper adapted from vLLM.

``CUDAGraphWrapper`` wraps a callable (typically a compiled subgraph) and
transparently captures / replays CUDA graphs keyed by batch size.  Dispatch
is controlled by ``Context.cudagraph_runtime_mode`` and the wrapper's own
``runtime_mode``:

  - If context mode is ``NONE`` → fall through (no graph).
  - If context mode **does not match** ``self.runtime_mode`` → fall through.
  - If context mode **matches** → capture (first time) or replay (cached).

This mode-matching is critical for FULL_AND_PIECEWISE operation (vLLM's
default for decode): piecewise wrappers only activate for PIECEWISE mode,
while the engine's full-model graph only activates for FULL mode.
"""

from __future__ import annotations

import dataclasses
import logging
from contextlib import nullcontext

import torch

from .context import CUDAGraphMode, get_context

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class _CUDAGraphEntry:
    cudagraph: torch.cuda.CUDAGraph
    output: torch.Tensor


class CUDAGraphWrapper(torch.nn.Module):
    """Wrap a callable with per-batch-size CUDA graph capture/replay.

    Inherits ``nn.Module`` so it can be assigned as a submodule of the
    FX split graph (``setattr(split_gm, submod_name, wrapper)``).

    Parameters
    ----------
    runnable : callable
        The function to capture (e.g. compiled model forward).
    runtime_mode : CUDAGraphMode
        Which mode this wrapper responds to.  A PIECEWISE wrapper only
        captures/replays when context says PIECEWISE; a FULL wrapper only
        when context says FULL.  Defaults to PIECEWISE (for subgraph wrapping).
    graph_pool : optional
        Shared ``torch.cuda.graph_pool_handle()`` for memory reuse.
    capture_context : optional
        Context manager to enter during capture (e.g. ``custom_ar.capture()``).
    """

    def __init__(
        self,
        runnable,
        runtime_mode: CUDAGraphMode = CUDAGraphMode.PIECEWISE,
        graph_pool=None,
        capture_context=None,
    ):
        super().__init__()
        self.runnable = runnable
        self.runtime_mode = runtime_mode
        self.graph_pool = graph_pool
        self.capture_context = capture_context
        self._cache: dict[int, _CUDAGraphEntry] = {}

    @property
    def captured_sizes(self) -> list[int]:
        return sorted(self._cache.keys())

    def forward(self, *args, **kwargs) -> torch.Tensor:
        ctx = get_context()
        mode = ctx.cudagraph_runtime_mode

        # Only capture/replay when context mode matches this wrapper's mode.
        # This enables FULL_AND_PIECEWISE: piecewise wrappers ignore FULL
        # mode calls, and vice versa.
        if mode == CUDAGraphMode.NONE or mode != self.runtime_mode:
            return self.runnable(*args, **kwargs)

        bs = ctx.batch_size_for_graph
        entry = self._cache.get(bs)

        if entry is None:
            entry = self._capture(bs, *args, **kwargs)
            return entry.output

        entry.cudagraph.replay()
        return entry.output

    def _capture(self, bs: int, *args, **kwargs) -> _CUDAGraphEntry:
        logger.debug("Capturing %s CUDA graph for batch_size=%d",
                     self.runtime_mode.name, bs)
        cap_ctx = self.capture_context or nullcontext()

        with cap_ctx:
            self.runnable(*args, **kwargs)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.graph_pool):
                output = self.runnable(*args, **kwargs)

            if self.graph_pool is None:
                self.graph_pool = graph.pool()

        entry = _CUDAGraphEntry(cudagraph=graph, output=output)
        self._cache[bs] = entry
        torch.cuda.synchronize()
        return entry
