"""Standalone torch.compile integration adapted from vLLM's compilation stack.

Implements the same compilation flow as vLLM's ``VllmBackend`` +
``PiecewiseBackend``:

1. ``mark_dynamic`` marks batch dimensions as symbolic before the first
   ``torch.compile`` call.
2. Dynamo traces the model **once** (guards dropped via
   ``skip_all_guards_unsafe``).
3. ``KBNanoBackend`` receives the FX graph, splits it at attention
   custom-op boundaries.
4. Each non-splitting subgraph is compiled **once** with ``compile_fx``
   using fake/symbolic args extracted from graph placeholder metadata.
5. Deduplication: structurally identical subgraphs (repeated transformer
   layers) reuse the same compiled artifact via ``autograd_cache_key``.
6. Each compiled subgraph is wrapped with ``CUDAGraphWrapper`` for
   per-batch-size CUDA graph capture/replay during decode.
7. At runtime, no recompilation occurs — Dynamo guards are dropped and
   the compiled subgraphs handle any batch size.
"""

from __future__ import annotations

import copy
import dataclasses
import logging
import operator
from collections import defaultdict
from collections.abc import Callable
from contextlib import ExitStack
from typing import Any
from unittest.mock import patch

import torch
import torch.fx as fx

from .context import CUDAGraphMode, enable_custom_ops
from .cuda_graph import CUDAGraphWrapper
from .custom_ops import SPLITTING_OPS, ensure_custom_ops_registered

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FX graph splitting (adapted from vllm/compilation/backends.py)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class SplitItem:
    submod_name: str
    graph_id: int
    is_splitting_graph: bool
    graph: fx.GraphModule


def _should_split(node: fx.Node, splitting_ops: list[str]) -> bool:
    if node.op != "call_function":
        return False
    target = node.target
    if isinstance(target, torch._ops.OpOverloadPacket):
        return target._qualified_op_name in splitting_ops
    if isinstance(target, torch._ops.OpOverload):
        packet_name = target.name()
        overload_name = f"{packet_name}.{target._overloadname}"
        return overload_name in splitting_ops or packet_name in splitting_ops
    return False


def _is_empty_allocation_node(node: fx.Node) -> bool:
    if node.op == "call_method":
        return node.target == "new_empty"
    if node.op != "call_function":
        return False
    target = node.target
    if target in (torch.empty, torch.empty_like, torch.empty_strided):
        return True
    if isinstance(target, torch._ops.OpOverloadPacket):
        pname = target._qualified_op_name
    elif isinstance(target, torch._ops.OpOverload):
        pname = target.name()
    else:
        return False
    return pname.startswith("aten::empty") or pname.startswith("aten::new_empty")


def _merge_empty_only_subgraphs(
    node_to_subgraph_id: dict[fx.Node, int],
    split_op_graphs: list[int],
) -> None:
    nodes_by_sgid: dict[int, list[fx.Node]] = defaultdict(list)
    for node, sgid in node_to_subgraph_id.items():
        nodes_by_sgid[sgid].append(node)

    splitting_set = set(split_op_graphs)
    prev_ns: int | None = None
    max_sgid = max(node_to_subgraph_id.values(), default=-1)

    for sgid in range(max_sgid + 1):
        nodes = nodes_by_sgid.get(sgid, [])
        if not nodes:
            continue
        is_ns = sgid not in splitting_set
        is_eo = len(nodes) == 1 and _is_empty_allocation_node(nodes[0])
        merged = False
        if is_eo and prev_ns is not None:
            empty_node = nodes[0]
            if all(
                inp.op == "placeholder"
                or node_to_subgraph_id[inp] <= prev_ns
                for inp in empty_node.all_input_nodes
            ):
                node_to_subgraph_id[empty_node] = prev_ns
                merged = True
        if not merged and is_ns:
            prev_ns = sgid


def split_graph(
    graph: fx.GraphModule,
    splitting_ops: list[str],
) -> tuple[fx.GraphModule, list[SplitItem]]:
    """Split an FX graph at custom-op boundaries."""
    subgraph_id = 0
    node_to_subgraph_id: dict[fx.Node, int] = {}
    split_op_graphs: list[int] = []

    for node in graph.graph.nodes:
        if node.op in ("output", "placeholder"):
            continue

        if node.op == "call_function" and node.target == operator.getitem:
            input_node = node.args[0]
            if input_node.op != "placeholder":
                assert input_node in node_to_subgraph_id
                node_to_subgraph_id[node] = node_to_subgraph_id[input_node]
                continue

        if _should_split(node, splitting_ops):
            subgraph_id += 1
            node_to_subgraph_id[node] = subgraph_id
            split_op_graphs.append(subgraph_id)
            if _should_split(node.next, splitting_ops):
                subgraph_id -= 1
            else:
                subgraph_id += 1
        else:
            node_to_subgraph_id[node] = subgraph_id

    _merge_empty_only_subgraphs(node_to_subgraph_id, split_op_graphs)

    split_gm = torch.fx.passes.split_module.split_module(
        graph, None,
        lambda node: node_to_subgraph_id[node],
        keep_original_order=True,
    )

    outputs: list[SplitItem] = []
    for name, module in split_gm.named_modules():
        if "." in name or name == "":
            continue
        gid = int(name.replace("submod_", ""))
        outputs.append(SplitItem(name, gid, gid in split_op_graphs, module))
    outputs.sort(key=lambda x: x.graph_id)

    return split_gm, outputs


# ---------------------------------------------------------------------------
# AlwaysHitShapeEnv (from vLLM compiler_interface.py)
# ---------------------------------------------------------------------------

class AlwaysHitShapeEnv:
    """Dummy shape environment that makes Inductor cache lookups always hit.

    When compiling subgraphs outside of Dynamo's tracing context, there's
    no ShapeEnv to provide. This dummy makes the cache work anyway.
    """

    def __init__(self) -> None:
        self.guards: list[Any] = []

    def evaluate_guards_expression(self, *args: Any, **kwargs: Any) -> bool:
        return True

    def get_pruned_guards(self, *args: Any, **kwargs: Any) -> list[Any]:
        return []

    def produce_guards_expression(self, *args: Any, **kwargs: Any) -> str:
        return ""


# ---------------------------------------------------------------------------
# Helpers for extracting fake args from graph (mirrors vLLM)
# ---------------------------------------------------------------------------

def _get_fake_args(graph: fx.GraphModule) -> list:
    """Get fake/symbolic args from placeholder metadata.

    This is the key mechanism: Inductor receives fake tensors with symbolic
    shapes, so it generates code that works for ANY concrete batch size —
    no per-shape recompilation needed.
    """
    fake_args = []
    for node in graph.graph.nodes:
        if node.op == "placeholder":
            val = node.meta.get("example_value")
            if val is None:
                val = node.meta.get("val")
            if val is not None:
                fake_args.append(val)
        else:
            break
    return fake_args


# ---------------------------------------------------------------------------
# PiecewiseBackend (mirrors vLLM's PiecewiseBackend)
# ---------------------------------------------------------------------------

class _StopCompiling(BaseException):
    pass


class PiecewiseBackend:
    """Compiled backend for a single non-splitting subgraph.

    Compiles the subgraph once with symbolic/fake args from the graph,
    then dispatches at runtime. Identical layers are deduplicated via
    autograd_cache_key normalization.
    """

    # Class-level cache: autograd_cache_key -> compiled callable.
    # Shared across all PiecewiseBackend instances to enable dedup.
    _loaded_artifacts: dict[str, Any] = {}

    def __init__(
        self,
        graph: fx.GraphModule,
        piecewise_compile_index: int,
        total_piecewise_compiles: int,
        sym_shape_indices: list[int],
        returns_tuple: bool,
        fake_args: list[Any] | None = None,
    ):
        self.graph = graph
        self.piecewise_compile_index = piecewise_compile_index
        self.total_piecewise_compiles = total_piecewise_compiles
        self.sym_shape_indices = sym_shape_indices
        self.returns_tuple = returns_tuple

        self.is_first_graph = piecewise_compile_index == 0
        self.is_last_graph = (
            piecewise_compile_index == total_piecewise_compiles - 1
        )

        self.runnable = self._compile(fake_args)

    def _compile(self, fake_args: list[Any] | None = None) -> Callable[..., Any]:
        """Compile this subgraph once with symbolic args."""
        from torch._inductor.compile_fx import compile_fx

        if fake_args is None:
            fake_args = _get_fake_args(self.graph)

        # Inductor can inplace-modify the graph, so deep-copy it.
        graph_copy = copy.deepcopy(self.graph)

        # Detect the FakeTensorMode from our fake args. Dynamo creates
        # a new `backend_fake_mode` right before calling the backend and
        # sets it on the tracing context. But our fake args have the
        # ORIGINAL FakeTensorMode from tracing. We need to patch the
        # tracing context to use our fake args' mode so detect_fake_mode
        # inside compile_fx doesn't see a mismatch.
        from torch._subclasses.fake_tensor import FakeTensor
        input_fake_mode = None
        for x in fake_args:
            if isinstance(x, FakeTensor):
                input_fake_mode = x.fake_mode
                break

        cache_key = None
        orig_cache_key_fn = (
            torch._functorch._aot_autograd.autograd_cache.autograd_cache_key
        )

        def patched_autograd_cache_key(*args, **kwargs):
            result = orig_cache_key_fn(*args, **kwargs)
            if result is None:
                return None
            nonlocal cache_key
            cache_key = result[0]
            if cache_key in PiecewiseBackend._loaded_artifacts:
                raise _StopCompiling()
            return result

        def _get_shape_env() -> AlwaysHitShapeEnv:
            return AlwaysHitShapeEnv()

        def _check_can_cache(*args, **kwargs) -> None:
            return

        with ExitStack() as stack:
            stack.enter_context(
                torch._functorch.config.patch(
                    autograd_cache_normalize_inputs=True
                )
            )
            stack.enter_context(
                patch(
                    "torch._functorch._aot_autograd.autograd_cache"
                    ".autograd_cache_key",
                    patched_autograd_cache_key,
                )
            )
            stack.enter_context(
                patch(
                    "torch._inductor.codecache.FxGraphCache._get_shape_env",
                    _get_shape_env,
                )
            )
            from torch._functorch._aot_autograd.autograd_cache import (
                AOTAutogradCache,
            )
            if hasattr(AOTAutogradCache, "_get_shape_env"):
                stack.enter_context(
                    patch(
                        "torch._functorch._aot_autograd.autograd_cache"
                        ".AOTAutogradCache._get_shape_env",
                        _get_shape_env,
                    )
                )
            stack.enter_context(
                patch(
                    "torch._inductor.codecache.FxGraphCache._check_can_cache",
                    _check_can_cache,
                )
            )
            stack.enter_context(
                torch._inductor.config.patch(fx_graph_remote_cache=False)
            )
            stack.enter_context(
                torch._functorch.config.patch(enable_autograd_cache=False)
            )
            stack.enter_context(
                torch._functorch.config.patch(
                    enable_remote_autograd_cache=False
                )
            )

            if hasattr(torch._dynamo, "utils"):
                ctx = torch._dynamo.utils.get_metrics_context()
                stack.enter_context(ctx)

            # Patch the tracing context's fake_mode to match our fake args.
            # Dynamo creates a new backend_fake_mode before calling the
            # backend, but our FakeTensors were created with the original
            # tracing FakeTensorMode. compile_fx calls detect_fake_mode
            # which asserts they match.
            tracing_ctx = torch._guards.TracingContext.try_get()
            old_tracing_fake_mode = None
            if tracing_ctx is not None and input_fake_mode is not None:
                old_tracing_fake_mode = tracing_ctx.fake_mode
                tracing_ctx.fake_mode = input_fake_mode

            try:
                compiled = compile_fx(
                    graph_copy,
                    fake_args,
                    config_patches={
                        "fx_graph_cache": True,
                        "fx_graph_remote_cache": False,
                    },
                )
            except _StopCompiling:
                assert cache_key is not None
                logger.debug(
                    "Subgraph %d/%d deduplicated (cache_key hit)",
                    self.piecewise_compile_index,
                    self.total_piecewise_compiles,
                )
                return PiecewiseBackend._loaded_artifacts[cache_key]
            finally:
                if tracing_ctx is not None and old_tracing_fake_mode is not None:
                    tracing_ctx.fake_mode = old_tracing_fake_mode

        if cache_key is not None and compiled is not None:
            PiecewiseBackend._loaded_artifacts[cache_key] = compiled

        logger.debug(
            "Compiled subgraph %d/%d",
            self.piecewise_compile_index,
            self.total_piecewise_compiles,
        )
        return compiled

    def __call__(self, *args: Any) -> Any:
        graph_output = self.runnable(*args)
        if self.returns_tuple or not isinstance(graph_output, (tuple, list)):
            return graph_output
        return graph_output[0]


# ---------------------------------------------------------------------------
# PiecewiseCompileInterpreter (mirrors vLLM)
# ---------------------------------------------------------------------------

class PiecewiseCompileInterpreter(torch.fx.Interpreter):
    """Interpreter that replaces compilable submodules with PiecewiseBackend
    instances, optionally wrapped with CUDAGraphWrapper.

    Runs the split graph with fake/symbolic args to drive compilation of
    each subgraph. After this, the split graph's submodules are compiled
    callables.
    """

    def __init__(
        self,
        module: fx.GraphModule,
        compile_submod_names: list[str],
        cudagraph_enabled: bool = True,
    ):
        super().__init__(module)
        self.compile_submod_names = compile_submod_names
        self.cudagraph_enabled = cudagraph_enabled
        self.extra_traceback = False

    def call_module(
        self,
        target: torch.fx.node.Target,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        assert isinstance(target, str)

        gm = getattr(self.module, target)
        outputs = gm.graph.output_node().args[0]
        output = fx.map_arg(outputs, lambda node: node.meta["example_value"])

        if target in self.compile_submod_names:
            index = self.compile_submod_names.index(target)
            submod = self.fetch_attr(target)

            sym_shape_indices = [
                i for i, x in enumerate(args) if isinstance(x, torch.SymInt)
            ]

            from torch._inductor.compile_fx import graph_returns_tuple

            # Pass the interpreter's args as fake_args. These are in the
            # same FakeTensorMode as the tracing context (they were
            # propagated through the interpreter from the original graph's
            # fake args). This avoids FakeTensorMode mismatches.
            piecewise = PiecewiseBackend(
                submod,
                index,
                len(self.compile_submod_names),
                sym_shape_indices,
                graph_returns_tuple(submod),
                fake_args=list(args),
            )

            if self.cudagraph_enabled:
                wrapper = CUDAGraphWrapper(
                    runnable=piecewise,
                    runtime_mode=CUDAGraphMode.PIECEWISE,
                )
                self.module.__dict__[target] = wrapper
            else:
                self.module.__dict__[target] = piecewise

        return output


# ---------------------------------------------------------------------------
# KBNano Dynamo backend (mirrors vLLM's VllmBackend)
# ---------------------------------------------------------------------------

class KBNanoBackend:
    """Custom Dynamo backend that mirrors vLLM's VllmBackend.

    Called **exactly once** by Dynamo (guards are dropped). It:
    1. Splits the FX graph at attention custom-op boundaries
    2. Creates PiecewiseBackend for each non-splitting subgraph
    3. Compiles each with ``compile_fx`` using symbolic/fake args
    4. Deduplicates identical layers via ``autograd_cache_key``
    5. Wraps with CUDAGraphWrapper for PIECEWISE capture/replay
    """

    def __init__(
        self,
        splitting_ops: list[str] | None = None,
        cudagraph_enabled: bool = True,
    ):
        self.splitting_ops = splitting_ops or SPLITTING_OPS
        self.cudagraph_enabled = cudagraph_enabled
        self._called = False

    def __call__(
        self,
        graph: fx.GraphModule,
        example_inputs: list[torch.Tensor],
    ) -> Any:
        assert not self._called, "KBNanoBackend should only be called once"
        self._called = True

        logger.info("KBNanoBackend: splitting graph at %s", self.splitting_ops)

        split_gm, piecewise_graphs = split_graph(graph, self.splitting_ops)

        compile_submod_names = [
            item.submod_name
            for item in piecewise_graphs
            if not item.is_splitting_graph
        ]

        logger.info(
            "KBNanoBackend: %d subgraphs (%d compilable, %d splitting ops)",
            len(piecewise_graphs),
            len(compile_submod_names),
            len(piecewise_graphs) - len(compile_submod_names),
        )

        all_fake_values = []
        for node in graph.graph.find_nodes(op="placeholder"):
            all_fake_values.append(node.meta["example_value"])

        fake_args = [
            all_fake_values[i]
            if isinstance(t, torch.Tensor)
            else t
            for i, t in enumerate(example_inputs)
        ]

        PiecewiseCompileInterpreter(
            split_gm,
            compile_submod_names,
            cudagraph_enabled=self.cudagraph_enabled,
        ).run(*fake_args)

        logger.info("KBNanoBackend: compilation complete")

        return split_gm


# ---------------------------------------------------------------------------
# Model compilation entry point
# ---------------------------------------------------------------------------

def compile_model(
    model: torch.nn.Module,
    cudagraph_enabled: bool = True,
) -> torch.nn.Module:
    """Apply torch.compile with the KBNano backend.

    Mirrors vLLM's compilation flow:
    1. Register and enable custom ops for attention/MoE
    2. ``mark_dynamic`` on batch dimensions so Dynamo traces with symbolic
       shapes
    3. ``fullgraph=True`` — single graph, no graph breaks
    4. ``skip_all_guards_unsafe`` — Dynamo never re-traces
    5. ``KBNanoBackend`` — splits, compiles with symbolic shapes, deduplicates

    The model is compiled once, then reused for all batch sizes.
    """
    ensure_custom_ops_registered()
    enable_custom_ops()

    PiecewiseBackend._loaded_artifacts.clear()

    backend = KBNanoBackend(
        cudagraph_enabled=cudagraph_enabled,
    )

    options: dict[str, Any] = {}
    if hasattr(torch.compiler, "skip_all_guards_unsafe"):
        options["guard_filter_fn"] = torch.compiler.skip_all_guards_unsafe
    else:
        options["guard_filter_fn"] = lambda x: [False for _ in x]

    compiled = torch.compile(
        model,
        fullgraph=True,
        dynamic=False,
        backend=backend,
        options=options,
    )

    original_cache_size = torch._dynamo.config.cache_size_limit
    original_accumulated = torch._dynamo.config.accumulated_cache_size_limit
    torch._dynamo.config.cache_size_limit = 2048
    torch._dynamo.config.accumulated_cache_size_limit = 8192

    model._kb_nano_compiled = compiled  # type: ignore[attr-defined]
    model._kb_nano_cache_restore = (  # type: ignore[attr-defined]
        original_cache_size, original_accumulated
    )
    model._kb_nano_first_call = True  # type: ignore[attr-defined]

    logger.info(
        "Model wrapped with KBNanoBackend (piecewise compile, "
        "symbolic shapes, autograd_cache_key dedup)"
    )
    return compiled
