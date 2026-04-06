"""Inductor post-grad passes for kb_nano.

With the dual-dispatch architecture (forward_native for compiled path,
forward_cuda for eager), Inductor sees pure PyTorch code for norms and
activations.  This means Inductor's own fusion engine handles most
optimizations automatically (e.g., fusing RMSNorm with adjacent quant,
eliminating intermediate tensors).

The pass manager provides:
  - ``NoopEliminationPass`` — removes identity reshape/view/expand/slice ops
    that can block Inductor's automatic fusion
  - Infrastructure for custom passes when needed
"""

from __future__ import annotations

import logging

import torch
import torch._inductor.custom_graph_pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class InductorPass:
    """Base class for post-grad Inductor passes."""

    name: str = "base"

    def __call__(self, graph: torch.fx.Graph) -> None:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Noop elimination
# ---------------------------------------------------------------------------

class NoopEliminationPass(InductorPass):
    """Remove redundant reshape/view/expand/slice ops.

    These identity ops are inserted by functionalization and prevent
    Inductor's built-in fusion from matching adjacent ops.  Mirrors
    vLLM's ``NoOpEliminationPass``.
    """

    name = "noop_elimination"

    _IDENTITY_TARGETS = frozenset({
        torch.ops.aten.reshape.default,
        torch.ops.aten.view.default,
        torch.ops.aten.expand.default,
        torch.ops.aten.slice.Tensor,
    })

    def __call__(self, graph: torch.fx.Graph) -> None:
        count = 0
        for node in list(graph.nodes):
            if node.op != "call_function":
                continue
            if node.target not in self._IDENTITY_TARGETS:
                continue
            if self._is_identity(node):
                node.replace_all_uses_with(node.args[0])
                graph.erase_node(node)
                count += 1
        if count > 0:
            logger.debug("NoopElimination: removed %d identity ops", count)

    @staticmethod
    def _is_identity(node: torch.fx.Node) -> bool:
        inp = node.args[0]
        if not isinstance(inp, torch.fx.Node):
            return False
        out_val = node.meta.get("val")
        inp_val = inp.meta.get("val")
        if out_val is None or inp_val is None:
            return False
        if hasattr(out_val, "shape") and hasattr(inp_val, "shape"):
            return (list(out_val.shape) == list(inp_val.shape)
                    and out_val.dtype == inp_val.dtype)
        return False


# ---------------------------------------------------------------------------
# Pass manager
# ---------------------------------------------------------------------------

class PostGradPassManager(torch._inductor.custom_graph_pass.CustomGraphPass):
    """Orchestrates post-grad Inductor passes.

    Wired into ``torch._inductor.config.post_grad_custom_post_pass`` to run
    after Inductor's own optimizations.

    Inherits ``CustomGraphPass`` so Inductor's cache machinery can type-check
    and hash the pass correctly.
    """

    def __init__(self) -> None:
        self.passes: list[InductorPass] = [
            NoopEliminationPass(),
        ]

    def __call__(self, graph: torch.fx.Graph) -> None:
        for p in self.passes:
            p(graph)

    def uuid(self):
        return None

    def add(self, pass_: InductorPass) -> None:
        self.passes.append(pass_)


def configure_post_grad_passes() -> None:
    """Install the kb_nano post-grad pass manager into Inductor config."""
    pm = PostGradPassManager()
    torch._inductor.config.post_grad_custom_post_pass = pm
    logger.info("Installed PostGradPassManager with passes: %s",
                [p.name for p in pm.passes])


def remove_post_grad_passes() -> None:
    """Remove kb_nano post-grad passes from Inductor config."""
    torch._inductor.config.post_grad_custom_post_pass = None
