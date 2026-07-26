#!/usr/bin/env python3
"""AKO4X -> fastkernels benchmark entrypoint (single-operator, subprocess CLI).

Usage
-----
    agent_entrypoint.py --op rms_norm --candidate /path/to/kernel.py
    agent_entrypoint.py --op rms_norm --baseline-identity
    agent_entrypoint.py --op rms_norm --candidate k.py --scenarios tokens-1,tokens-4

Contract
--------
Writes EXACTLY one JSON object to stdout (the AKO4X normalized result dict, see
``AKO4X/scripts/benchmark_adapter.py``)::

    {definition_name: {workload_uuid: {"status", "solution", "axes",
        "latency_ms", "reference_latency_ms", "speedup_factor",
        "max_abs_error", "max_rel_error", "error_log"}}}

definition_name = f"kb_{op}"; workload_uuid = the scenario name. All logging goes
to stderr (fd 1 is redirected to fd 2 for the whole run so that library banners
cannot corrupt the JSON; the real stdout fd is held aside and written at exit).

Exit codes: 0 when the benchmark ran (scenario failures are *data*, not errors);
2 on infrastructure errors (bad args, unknown op, tree/candidate import failure,
no CUDA, empty scenario selection).

Correctness / timing semantics are the release runner's
(``fastkernels/bench/kernels/runner.py``) -- its comparison helpers, its
tolerances, its median timing are imported and reused, not reimplemented. Ten
deliberate differences from ``run_kernel_benchmark``:

1. **Strict weight transfer.** The runner wraps ``load_state_dict`` in a bare
   ``try/except pass`` (runner.py:496-500), so a candidate whose parameters do
   not line up with the baseline's silently runs on its own initialisation. Here
   the transfer is unconditional and any raise / non-empty ``missing_keys`` /
   non-empty ``unexpected_keys`` fails the scenario with RUNTIME_ERROR.
2. **Targeted baseline discovery.** ``kernel_swapper.get()`` calls
   ``discover_targets()``, which imports *every* baseline module in the tree and
   dies in this environment on an unrelated optional dependency
   (``tasks/baseline/L2/pointtransformerv3_layers.py`` -> ``import spconv``).
   ``_resolve_target`` below reproduces ``discover_targets``'s per-op logic
   (same module path, same ``kernel_swapper._find_module_class``) for one op only.
3. **Local instantiation** (``_instantiate_module`` here, not the runner's): no
   key mangling of the traced ``init_args``, no silent ``cls()`` fallback,
   ``config``-shaped dicts wrapped into attribute-accessible namespaces, traced
   activation names resolved to callables. See the block comment above
   ``_instantiate_module``.
4. **Deterministic fixtures.** torch's CPU+CUDA RNG is seeded from a stable hash
   of (operator, scenario) before each scenario's inputs are materialised, so a
   verdict is reproducible across processes and machines.
5. **Input preparation.** Registry fixtures record *shapes*, not values, so
   index-like arguments (``cu_seqlens``, ``block_table``, ``cache_seqlens``,
   ``slot_mapping``, expert routing) arrive as uniform random integers that
   violate the kernel's contract. ``_prepare_inputs_for_target`` repairs them
   once, before either module runs, so both sides see identical valid inputs; it
   also materialises the container input kinds the registry cannot build.
6. **Non-finite baseline outputs are a fixture fault, not a verdict.** The
   runner turns them into a numerical failure (runner.py:302-303); here they
   become RUNTIME_ERROR, because nothing about the candidate was measured.
7. **Uninitialised baseline parameters are repaired** (seeded, deterministic)
   before the perturbation, because several baselines allocate parameters with
   ``torch.empty`` and never fill them -- see ``_repair_degenerate_parameters``.
8. **``moe_align`` outputs are canonicalized before comparison** (both sides,
   identically). The op's output is a token->expert grouping built by parallel
   atomic appends; order *within* one expert's block-group is not part of the
   contract -- its only consumer (``tasks/baseline/L2/fused_experts.py:309-354``)
   uses ``sorted_token_ids``/``expert_ids`` purely as gather/scatter index
   metadata. Baseline-vs-itself legitimately produces different-but-equivalent
   permutations run to run, so both sides are sorted with the same
   (expert, token) key before the runner's comparison. Scoped to this one op;
   tolerances are untouched. See ``_canonicalize_moe_align_output``.
9. **``chunk_gla`` is graded at its computation precision.** The FLA kernel
   assembles its *float32* final state out of bf16-rounded decayed keys whose
   decay factor comes from Triton's ``ex2.approx``; measured, the BASELINE sits
   ~1.1e-4 from the fp64-exact value at elements where the fp32 tolerance band
   is ~1.1e-5. Grading that output under fp32 tolerances therefore demands
   bit-reproduction of ``ex2.approx`` -- a Triton-only artefact -- rather than
   semantic correctness. Every tensor of this one op is compared with the
   runner's low-precision (bf16-class) tolerances on the raw values; see
   ``_low_precision_tolerances`` for why the values are not cast instead.
10. **Three cancellation-amplified ops get a second, fp64-oracle pass arm.**
   ``gpt_oss_moe``, ``chunk_gla`` (its o-path) and ``vision_block`` all
   subtract nearly-equal large quantities (top-k slot sums, chunk-state vs
   intra-chunk contributions, pre-norm residuals). A ~2-ULP kernel-level
   difference -- comfortably inside the band where it is produced -- is
   amplified to 1.1x-3.1x of the band at the output, so a correct
   implementation that is not bit-identical to the kernel cannot pass. For
   gpt_oss_moe the direction was proven: at the offending elements the
   bf16-rounded fp64-exact slot values reproduce the naive reference exactly,
   i.e. it is the BASELINE that is one ULP off the truth. Scoped to these three
   ops, a candidate passes if EITHER (a) it is inside the standard band of the
   baseline (unchanged behaviour), OR (b) elementwise
   ``|cand - truth| <= |base - truth| + (atol_low + rtol_low * |truth|)``,
   where ``truth`` is a float64 evaluation of the operator computed by this
   harness from the same fixture inputs and the same transferred weights. Arm
   (b) is evaluated only after arm (a) has already failed, so it can rescue a
   verdict but never condemn one. Both arms' numbers are reported in the result
   entry, together with the FlashAttention-style ratio.
   **Arm (b) currently reports but does not enforce** (``_ORACLE_ARM_ENFORCED``
   is False): measured, neither it nor the FlashAttention criterion clears the
   acceptance matrix on all three ops -- the margin criterion fails the very
   references it exists to rescue, and the FlashAttention criterion is blind on
   ``gpt_oss_moe``, where the baseline's own worst error against the truth is
   larger than a semantic bug's. The constant carries the measured numbers.
   With it False, every op's verdict -- including these three -- is exactly
   what the pre-M9 harness produced. See ``_FP64_ORACLE_OPS``.

Tolerances are NOT settable from the CLI: they are read from the runner's module
constants. A calling agent must not be able to loosen its own correctness gate.
"""

from __future__ import annotations

import argparse
import hashlib
import contextlib
import importlib
import importlib.util
import inspect
import json
import math
import os
import shutil
import sys
import traceback
from types import SimpleNamespace
from typing import Any

# --- stdout quarantine -------------------------------------------------------
# Hold the real stdout fd aside and point fd 1 at stderr, so that anything the
# imported stack prints (vLLM/flashinfer banners, the runner's own print()s)
# lands on stderr instead of corrupting the single JSON object we emit.
_REAL_STDOUT_FD = os.dup(1)
os.dup2(2, 1)


def _log(msg: str) -> None:
    print(f"[agent_entrypoint] {msg}", file=sys.stderr, flush=True)


def _emit_json(obj: Any) -> None:
    payload = json.dumps(obj, allow_nan=False) + "\n"
    sys.stdout.flush()
    os.write(_REAL_STDOUT_FD, payload.encode())


class InfraError(RuntimeError):
    """Whole-run failure: nothing meaningful to report, exit nonzero."""


# --- AKO4X status constants (mirrored from benchmark_adapter.py) --------------
STATUS_PASSED = "PASSED"
STATUS_COMPILE_ERROR = "COMPILE_ERROR"
STATUS_INCORRECT_NUMERICAL = "INCORRECT_NUMERICAL"
STATUS_RUNTIME_ERROR = "RUNTIME_ERROR"
STATUS_TIMEOUT = "TIMEOUT"

# Runner defaults, pinned here so the CLI exposes no timing knob either.
NUM_WARMUP = 10
NUM_RUNS = 100


# --- fastkernels tree bootstrap ---------------------------------------------

def _tree_candidates() -> list[str]:
    roots: list[str] = []
    env_tree = os.environ.get("FASTKERNELS_TREE")
    if env_tree:
        roots.append(env_tree)
    roots.extend(p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p)
    roots.append(os.getcwd())
    seen, out = set(), []
    for r in roots:
        a = os.path.abspath(r)
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out


def _looks_like_tree(root: str) -> bool:
    return all(os.path.isfile(os.path.join(root, *p)) for p in (
        ("__init__.py",),
        ("bench", "kernels", "runner.py"),
        ("infra", "kernel_swapper.py"),
    ))


def _bootstrap_fastkernels() -> str:
    """Register the on-disk tree as the ``fastkernels`` package and return its root.

    The tree declares ``[tool.setuptools.package-dir] "fastkernels" = "."``, i.e.
    the repo root *is* the package. Rather than requiring a symlink named
    ``fastkernels`` on sys.path, bind the name explicitly here -- this also wins
    over any installed/editable copy, because the name is in sys.modules before
    anything imports it.
    """
    for root in _tree_candidates():
        if not _looks_like_tree(root):
            continue
        spec = importlib.util.spec_from_file_location(
            "fastkernels", os.path.join(root, "__init__.py"),
            submodule_search_locations=[root],
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["fastkernels"] = module
        spec.loader.exec_module(module)
        return root
    raise InfraError(
        "could not locate a fastkernels tree. Set FASTKERNELS_TREE or put the "
        f"repo root on PYTHONPATH. Looked in: {_tree_candidates()}"
    )


def _ensure_ninja_on_path() -> None:
    """L1 baselines JIT-build a CUDA extension; torch needs ninja on PATH."""
    if shutil.which("ninja") is None:
        bindir = os.path.dirname(os.path.abspath(sys.executable))
        os.environ["PATH"] = bindir + os.pathsep + os.environ.get("PATH", "")
        _log(f"ninja not on PATH; prepended {bindir}")


# --- target / candidate resolution ------------------------------------------

def _resolve_target(op: str):
    """BenchTarget for one op without importing the whole baseline corpus."""
    from fastkernels import KB_ROOT
    from fastkernels.infra.kernel_swapper import (
        BenchTarget, _find_module_class, registry_class_pin,
    )

    for level in (1, 2, 3, 4):
        path = KB_ROOT / "tasks" / "baseline" / f"L{level}" / f"{op}.py"
        if not path.is_file():
            continue
        module_path = f"tasks.baseline.L{level}.{op}"
        mod = importlib.import_module(f"fastkernels.{module_path}")
        cls = _find_module_class(mod, pin=registry_class_pin(op))
        if cls is None:
            raise InfraError(f"no nn.Module class found in {path}")
        _log(f"baseline module: {mod.__file__}")
        _log(f"baseline class : {cls.__name__} (L{level})")
        return BenchTarget(
            name=op, level=level, module_path=module_path, models=[],
            target_cls=cls, requires_recompile=(level == 1),
        )
    raise InfraError(f"no baseline file tasks/baseline/L*/{op}.py under {KB_ROOT}")


def _load_candidate_from_path(path: str, baseline_cls: type) -> type:
    """Path-taking port of ``kernel_swapper.load_candidate``.

    Copied (not imported) because the upstream function derives its path from
    ``CANDIDATE_DIR`` and takes no file argument. Selection logic is identical:
    exec the file, prefer the class named like the baseline, else the first
    nn.Module subclass.
    """
    import torch.nn as nn

    if not os.path.isfile(path):
        raise InfraError(f"candidate file not found: {path}")
    module_name = "_agent_candidate_impl"
    spec = importlib.util.spec_from_file_location(module_name, os.path.abspath(path))
    if spec is None or spec.loader is None:
        raise InfraError(f"cannot build an import spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        sys.modules.pop(module_name, None)
        raise InfraError(
            f"candidate import failed: {type(exc).__name__}: {exc}\n"
            + traceback.format_exc()
        ) from exc
    cls = getattr(mod, baseline_cls.__name__, None)
    if cls is None:
        for v in vars(mod).values():
            if isinstance(v, type) and issubclass(v, nn.Module) and v is not nn.Module:
                cls = v
                break
    if cls is None:
        raise InfraError(f"no nn.Module subclass found in {path}")
    return cls


# --- module instantiation -----------------------------------------------------
#
# The release runner's ``_instantiate_module`` is deliberately NOT reused; two of
# its behaviours are wrong for a correctness harness.
#
# 1. **Key mangling** (runner.py:72-77) rewrites the traced ``init_args`` before
#    construction: ``head_size`` -> ``head_dim``, ``base`` -> ``rope_theta``, and
#    unconditional ``pop("rotary_dim")`` / ``pop("is_neox_style")``. The renamed
#    or dropped key is then removed by the signature filter (the class never
#    declared the new name), so the class is constructed *without* a parameter it
#    declares as required. Measured on this registry: ``attention_impl``
#    (``Attention(num_heads, head_size, scale)``) and ``mrope``
#    (``MRotaryEmbedding(..., rotary_dim, ...)``) are unconstructible for that
#    reason alone -- 20 + 5 scenarios that can never report a verdict. Nothing is
#    mangled here: traced key names are passed through unchanged and only the
#    signature filter (copied verbatim from the runner) applies.
# 2. **The ``except TypeError: cls()`` fallback** silently default-constructs a
#    module when the real call fails, so an unbuildable scenario either reports a
#    numerical verdict computed on the wrong module, or -- as in the census --
#    reports the *fallback's* exception ("Attention.__init__() missing 3 required
#    positional arguments") instead of the real one. There is no fallback here:
#    a construction failure propagates and becomes RUNTIME_ERROR with the real
#    message.
#
# Added on top of the runner's logic: dict-valued ``init_args`` that stand in for
# a model config object are converted recursively to attribute-accessible
# namespaces, because HF-style modules read ``config.hidden_size``, not
# ``config["hidden_size"]``. Nested dicts (e.g. a ``vision`` sub-config) and
# lists of dicts are wrapped elementwise. The trigger is narrow on purpose --
# key literally named ``config`` or a parameter annotated ``*Config`` -- so that
# dict arguments which really are dicts (``quant_config={"weight_block_size":
# ...}``) keep their mapping interface.


def _wrap_namespaces(value: Any) -> Any:
    """dict -> SimpleNamespace, recursively (lists/tuples elementwise)."""
    if isinstance(value, dict):
        return SimpleNamespace(**{
            str(k): _wrap_namespaces(v) for k, v in value.items()
        })
    if isinstance(value, list):
        return [_wrap_namespaces(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_wrap_namespaces(v) for v in value)
    return value


def _callable_init_arg(value: str):
    """Resolve a traced activation name to a fresh callable, or None.

    A YAML trace cannot carry a ``Callable`` ctor argument, so an op like
    ``vision_mlp`` (``act_fn: Callable = QuickGELU()``) silently benchmarks its
    *default* activation while the real model uses another one -- Qwen3-VL's
    vision config says ``gelu_pytorch_tanh``. Proposals therefore carry the
    activation as a string and it is resolved here. A fresh instance per
    construction keeps the baseline and the candidate from sharing a submodule;
    none of these carry parameters, so the state_dict transfer is unaffected.
    """
    import torch.nn as nn

    key = str(value).lower()
    if key in ("gelu_tanh", "gelu_pytorch_tanh"):
        return nn.GELU(approximate="tanh")
    if key in ("quick_gelu", "quickgelu"):
        from fastkernels.tasks.baseline.L1.quickgelu import QuickGELU

        return QuickGELU()
    if key == "silu":
        return nn.SiLU()
    return None


def _wants_callable(name: str, param: Any) -> bool:
    """True if a string passed as ``name`` names a callable, not a mode flag.

    Deliberately narrow: HF-style configs carry ``hidden_act="silu"`` as a
    *string* that modules compare against, so only arguments that are declared
    callable (or named like one) are resolved.
    """
    # Name-based triggers are limited to slots that are callables by
    # convention. Note ``hidden_act`` is deliberately NOT one of them: HF
    # configs carry it as a mode string that modules compare against, and it
    # would match a naive ``*_act`` suffix rule.
    if name in ("act_fn", "act_layer", "norm_layer") or name.endswith(
            ("_fn", "_layer")):
        return True
    if param is None:
        return False
    annotation = getattr(param, "annotation", inspect.Parameter.empty)
    if annotation is inspect.Parameter.empty:
        return False
    text = annotation if isinstance(annotation, str) else str(annotation)
    return "Callable" in text or "Module" in text


def _wants_config_object(name: str, param: Any) -> bool:
    """True if a dict passed as ``name`` should be namespace-wrapped."""
    if name == "config":
        return True
    if param is None:
        return False
    annotation = getattr(param, "annotation", inspect.Parameter.empty)
    if annotation is inspect.Parameter.empty:
        return False
    text = annotation if isinstance(annotation, str) else getattr(
        annotation, "__name__", "")
    return isinstance(text, str) and text.split(".")[-1].endswith("Config")


def _build_oasis_rotary(kind: str, head_dim: int):
    """Fresh OasisRotaryEmbedding, mirroring the model-level construction.

    ``tasks/baseline/L3/oasis_dit.py:41-42`` is the in-repo ground truth for how
    Oasis wires its rotary modules: spatial attention gets
    ``OasisRotaryEmbedding(dim=head_dim // 2, freqs_for="pixel", max_freq=256)``,
    temporal gets ``OasisRotaryEmbedding(dim=head_dim, freqs_for="lang")``.
    (The codex runner used ``dim_head // 4`` with the default ``max_freq=10``
    for the spatial module; the model file wins.) Construction is RNG-free, so
    building a fresh instance per module keeps baseline and candidate from
    sharing a submodule while guaranteeing identical values.
    """
    from fastkernels.tasks.baseline.L1.oasis_rotary import OasisRotaryEmbedding

    if kind == "spatial":
        return OasisRotaryEmbedding(dim=head_dim // 2, freqs_for="pixel",
                                    max_freq=256)
    return OasisRotaryEmbedding(dim=head_dim, freqs_for="lang")


@contextlib.contextmanager
def _low_precision_tolerances(runner_mod):
    """Difference 9 helper: grade every tensor with the runner's
    low-precision (bf16-class) tolerances regardless of container dtype.

    Casting fp32 to bf16 instead was tried and rejected: values adjacent to
    a bf16 rounding boundary amplify sub-tolerance fp32 noise into full-ULP
    flips (measured: max_abs_error exactly one bf16 quantum). Patching the
    tolerance selection keeps the raw values and the runner's traversal.
    """
    original = runner_mod._tolerances_for_dtype
    low = (runner_mod._LOW_PRECISION_ATOL, runner_mod._LOW_PRECISION_RTOL)
    runner_mod._tolerances_for_dtype = lambda dtype: low
    try:
        yield
    finally:
        runner_mod._tolerances_for_dtype = original


def _boost_yolo_cls_bias(module: Any, scenario_name: str) -> None:
    """Make YOLO detection fixtures produce non-empty detections.

    The detect head's confidence selection reduces over the anchors that
    clear the score threshold; on repaired random weights (whose production
    ``bias_init`` prior is log(~few detections) -- very negative) NO anchor
    clears it, and the reduction over an empty set raises
    ``max(): Expected reduction dim ... numel() == 0``. Comparing hidden
    states instead would over-constrain candidates (a fused kernel need not
    materialise the baseline's intermediates), so the fix is fixture-side:
    set the classification branches' final-conv biases to a positive
    constant so sigmoid confidences (~0.98) put every anchor above any
    threshold, deterministically. Runs before the weight transfer, so both
    sides see identical values; the decode/select path is then genuinely
    exercised and compared.
    """
    import torch.nn as nn

    boosted = 0
    for sub in module.modules():
        if not (hasattr(sub, "cv3") and hasattr(sub, "one2one_cv3")
                and hasattr(sub, "nc")):
            continue
        for branch_list in (sub.cv3, sub.one2one_cv3):
            for branch in branch_list:
                last = branch[-1] if isinstance(branch, nn.Sequential) else branch
                bias = getattr(last, "bias", None)
                if bias is not None:
                    bias.data[: sub.nc] = 4.0
                    boosted += 1
    if boosted:
        _log(f"{scenario_name}: boosted {boosted} YOLO cls-branch biases "
             f"(non-empty detections fixture)")


def _augment_oasis_init_args(class_name: str, kwargs: dict[str, Any],
                             inputs: dict[str, Any]) -> None:
    """Fill the Oasis constructor arguments the YAML trace cannot express.

    The Oasis attention/block constructors require a rotary-embedding *module*
    argument (and dims derivable only from the activation shape); the registry
    records neither. The harness builds them here, in trusted code, identically
    for both sides -- the same pattern as ``_callable_init_arg``.

    ``x`` is ``[batch, time, height, width, dim]`` for all three classes.
    ``num_heads`` for the DiT block is not traced; 16 is the model default
    (``oasis_dit.py:26``), consistent with the ``heads: 16`` the attention
    scenarios do record.
    """
    import torch

    x = inputs.get("x")
    if not isinstance(x, torch.Tensor) or x.ndim < 2:
        return
    dim = int(x.shape[-1])
    if class_name in ("OasisSpatialAxialAttention", "OasisTemporalAxialAttention"):
        heads = int(kwargs.get("heads", 16))
        head_dim = dim // max(1, heads)
        kwargs.setdefault("dim", dim)
        kwargs.setdefault("dim_head", head_dim)
        if "rotary_emb" not in kwargs:
            kind = "spatial" if class_name == "OasisSpatialAxialAttention" else "temporal"
            kwargs["rotary_emb"] = _build_oasis_rotary(kind, head_dim)
    elif class_name == "SpatioTemporalDiTBlock":
        num_heads = int(kwargs.get("num_heads", 16))
        head_dim = dim // max(1, num_heads)
        kwargs.setdefault("hidden_size", dim)
        kwargs.setdefault("num_heads", num_heads)
        kwargs.setdefault("is_causal", True)  # oasis_dit.py:50
        if "spatial_rotary_emb" not in kwargs:
            kwargs["spatial_rotary_emb"] = _build_oasis_rotary("spatial", head_dim)
        if "temporal_rotary_emb" not in kwargs:
            kwargs["temporal_rotary_emb"] = _build_oasis_rotary("temporal", head_dim)


def _instantiate_module(cls: type, init_args: dict[str, Any], device: str = "cuda",
                        dtype: Any = None, inputs: dict[str, Any] | None = None):
    """Construct ``cls`` from traced ``init_args``. See the block comment above.

    ``inputs`` (the already-prepared fixture dict) is consulted only by the
    narrow per-class augmentations that need an activation shape or a module
    argument the registry cannot express (currently the Oasis family).
    """
    import torch
    import torch.nn as nn

    kwargs = dict(init_args)
    params: dict[str, Any] = {}
    if cls.__init__ in (nn.Module.__init__, object.__init__):
        # The class declares no constructor of its own. ``nn.Module.__init__``
        # is ``(*args, **kwargs)``, so the signature filter below would see a
        # VAR_KEYWORD parameter and forward every traced key into a constructor
        # that accepts none of them (``FlashAttnVarlen(training=False)`` ->
        # TypeError). The runner never noticed because its ``except TypeError:
        # cls()`` fallback quietly produced the right module for the wrong
        # reason; with the fallback gone the filter has to be correct.
        kwargs = {}
    else:
        try:
            params = dict(inspect.signature(cls.__init__).parameters)
            accepts_kwargs = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
            )
            if not accepts_kwargs:
                kwargs = {
                    k: v for k, v in kwargs.items()
                    if k in params and k != "self"
                }
        except (TypeError, ValueError):
            params = {}

    for key, value in list(kwargs.items()):
        if isinstance(value, dict) and _wants_config_object(key, params.get(key)):
            kwargs[key] = _wrap_namespaces(value)
        elif isinstance(value, str) and _wants_callable(key, params.get(key)):
            resolved = _callable_init_arg(value)
            if resolved is not None:
                kwargs[key] = resolved

    if inputs is not None and cls.__name__ in (
            "OasisSpatialAxialAttention", "OasisTemporalAxialAttention",
            "SpatioTemporalDiTBlock"):
        _augment_oasis_init_args(cls.__name__, kwargs, inputs)

    module = cls(**kwargs)  # no fallback: a TypeError here is the verdict

    module = module.to(device)
    if dtype is not None:
        # Cast learnable parameters to the scenario dtype without changing
        # precision-sensitive buffers such as RoPE/YARN cos/sin caches (from the
        # runner), and without touching FP8 parameters: ``torch.float8_e4m3fn``
        # answers True to ``is_floating_point()``, so the runner's version
        # silently rewrites quantized expert weights to bf16 while the module's
        # ``use_fp8`` flag stays set -- the FP8 Triton kernel then fails to
        # compile (verified on qwen3_moe). A quantized parameter's dtype is part
        # of the module's contract, not a scenario knob.
        #
        # Scale parameters are exempt for the same reason (codex runner
        # precedent, runner.py:370-375): FP8 block-scale tensors such as
        # ``parallel_linear``'s ``weight_scale_inv`` are float32 by kernel
        # contract -- DeepGEMM asserts ``sfb_dtype == torch::kFloat or
        # torch::kInt``, so casting them to bf16 makes every fp8-quantized
        # module (qwen3_moe_decoder) unrunnable.
        with torch.no_grad():
            for _name, param in module.named_parameters(recurse=True):
                if (param.is_floating_point()
                        and "float8" not in str(param.dtype)
                        and "scale" not in _name):
                    param.data = param.data.to(dtype=dtype)
    module.eval()
    return module


# --- deterministic fixtures ---------------------------------------------------

def _stable_seed(*parts: str) -> int:
    """Process-independent seed from a name (``hash()`` is salted per process)."""
    digest = hashlib.sha256("/".join(parts).encode()).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF


# Repaired values are a pure function of (op, tensor name, shape, dtype): the
# per-tensor generator is seeded from (op, name) and the draw depends on
# nothing else. Recomputing them for every scenario of a 32-layer model burns
# tens of CPU-seconds per scenario (measured: ~30 s/scenario fixture prep on
# gla, ~350 repaired tensors, ~2.7e9 elements), so the finished on-device
# tensor is cached and re-copied on later scenarios -- byte-identical to a
# fresh draw. Only the *draw* is cached; the degeneracy *decision* still runs
# per scenario, exactly as before.
_REPAIR_VALUE_CACHE: dict[tuple, Any] = {}


def _repair_degenerate_parameters(module: Any, op: str) -> list[str]:
    """Give uninitialised baseline parameters finite, non-degenerate values.

    Several kb baselines allocate parameters with ``torch.empty`` and rely on a
    later weight load that a kernel-level benchmark never performs, so the
    "weights" are whatever the CUDA allocator handed back. Measured over 81
    enumerated (op, shape) tuples: 62 came up NaN/Inf from stale device memory
    and 11 came up all-zero. The perturbation below cannot rescue either --
    ``NaN * s + shift`` is NaN (the forward then returns NaN and the scenario is
    unmeasurable), and ``0 * s + shift`` leaves a weight matrix that is pure
    noise of amplitude 0.05 with no structure.

    Only degenerate parameters are touched: a parameter that is finite and not
    identically zero is a real initialisation (``ones`` for norms, an embedding
    table, ...) and is left for the perturbation step.

    FP8 parameters are included: e.g. ``parallel_linear`` allocates
    ``torch.empty(..., dtype=float8_e4m3fn)`` weights whose stale bytes can
    encode e4m3 NaN (0x7f/0xff), which no downstream step can rescue. Their
    block scales are separate float32 parameters (repaired/perturbed on their
    own), so a plain seeded fp32 draw quantized to fp8 is a coherent fixture.
    The ``|max| > 1e4`` stale-memory heuristic is skipped for fp8 (e4m3 tops
    out at 448, so it can never fire and the fp32 upcast is exact).

    PACKED-QUANTIZED uint8 state is also included: ``GptOssMoE`` holds its
    MXFP4 expert weights as ``torch.zeros`` uint8 parameters (packed FP4
    pairs) with uint8 E8M0 block scales (``gpt_oss_moe.py:54-78``), relying
    on a checkpoint load that never happens here. All-zero packed weights
    make the MoE output weight-independent (bias-only) -- the same vacuity
    class as the historical weight=ones blind spot the perturbation step
    closes for float parameters. A uint8 tensor is recognised as
    packed-quantized payload when a sibling ``<name>_scale`` tensor exists
    (and as an E8M0 scale when it is the uint8 ``*_scale`` of a uint8
    payload); it is repaired only when ALL-ZERO (uint8 has no NaN; any
    nonzero content is a real load). Draws use the mxfp4 fixture's bounds:
    payload bytes are unconstrained (every byte is a valid e2m1/int2 pair --
    neither format has NaN/Inf encodings), scale exponents are bounded to
    [121, 127] (decoded 2^-6 .. 2^0) so activations stay finite. BitNet's
    ``BitLinear`` (uint8 packed int2 ``weight`` + float ``weight_scale``)
    matches the same payload rule; its float scale takes the float path.

    Values are drawn from a CPU generator seeded by (op, parameter name), so the
    fixture is identical on every machine and in every process.
    """
    import torch

    repaired: list[str] = []

    def _named_tensors():
        # Buffers need the same treatment as parameters: BatchNorm running
        # stats and similar non-parameter state are also torch.empty-allocated
        # in several kb baselines (measured: the residual nonfinite failures in
        # vision_block / yolov10* / attention were all buffer-borne).
        yield from module.named_parameters(recurse=True)
        yield from module.named_buffers(recurse=True)

    named = list(_named_tensors())
    by_name = {n: t for n, t in named}

    with torch.no_grad():
        for name, param in named:
            if param.dtype == torch.uint8 and param.numel() > 0:
                sibling_scale = by_name.get(name + "_scale")
                base = name[: -len("_scale")] if name.endswith("_scale") else None
                payload = base is not None and by_name.get(base) is not None \
                    and by_name[base].dtype == torch.uint8
                if sibling_scale is None and not payload:
                    continue  # unpaired uint8 state: not quantized weights
                is_scale = payload  # this tensor is the E8M0 scale of a payload
                if bool((param != 0).any()):
                    continue  # nonzero = actually loaded; leave it alone
                cache_key = (op, name, tuple(param.shape), str(param.dtype),
                             str(param.device))
                prepared = _REPAIR_VALUE_CACHE.get(cache_key)
                if prepared is None:
                    generator = torch.Generator(device="cpu").manual_seed(
                        _stable_seed(op, name))
                    low, high = (121, 128) if is_scale else (0, 256)
                    prepared = torch.randint(
                        low, high, tuple(param.shape), generator=generator,
                        dtype=torch.uint8).to(param.device)
                    _REPAIR_VALUE_CACHE[cache_key] = prepared
                param.copy_(prepared)
                repaired.append(name)
                continue
            if param.numel() == 0:
                # zero-element state (e.g. the YOLO head's lazily-filled
                # anchor/stride buffers): nothing to repair, and reductions
                # like .max() would raise on the empty tensor.
                continue
            if not param.is_floating_point():
                continue
            is_fp8 = "float8" in str(param.dtype)
            values = param.detach().float()
            finite = bool(torch.isfinite(values).all())
            absmax = float(values.abs().max().item()) if finite else 0.0
            degenerate = (
                not finite
                or absmax == 0.0
                # finite allocator garbage: torch.empty residue that happens to
                # be finite but astronomically scaled. Real initialisations are
                # O(1); anything with |max| > 1e4 is stale memory (measured:
                # yolov10_c2f flipped verdicts across runs because finite
                # residue overflowed the forward on some draws only).
                #
                # float16 gets a tighter bound: its max is 65504, so garbage
                # the generic rule tolerates (|max| <= 1e4) is one wide GEMM
                # away from inf -- measured: oasis_block tokens-8/09bba215
                # flipped between 10/10 PASSED and baseline_output_nonfinite
                # across runs on identical code, the same lottery the 1e4 rule
                # closed for bf16. 1024 clears the largest legitimate fp16
                # initialisation in the corpus (the Oasis pixel rotary
                # frequency table, |max| ~ 402) with a 2.5x margin.
                or (not is_fp8 and absmax > (
                    1024.0 if param.dtype == torch.float16 else 1e4))
                # quantization scale tensors are strictly positive by contract
                # (reciprocals of quantization step sizes), so any nonpositive
                # entry is stale memory even when finite and O(1). This
                # matters on SM100: DeepGEMM re-casts weight SFs to UE8M0
                # (log2 of the scale), so one signed garbage entry in
                # ``weight_scale_inv`` turns the whole projection output NaN
                # (measured on qwen3_moe_decoder, where the repair lottery on
                # ``o_proj.weight_scale_inv`` decided each run's fate).
                or ("scale" in name and finite and bool((values <= 0).any()))
            )
            if not degenerate:
                continue
            cache_key = (op, name, tuple(param.shape), str(param.dtype),
                         str(param.device))
            prepared = _REPAIR_VALUE_CACHE.get(cache_key)
            if prepared is None:
                generator = torch.Generator(device="cpu").manual_seed(
                    _stable_seed(op, name))
                if param.ndim >= 2:
                    # PyTorch's own default for Linear/Conv weights: uniform
                    # over +-1/sqrt(fan_in), which keeps activations O(1) at
                    # any width. For fp8 the fan-in is the last (reduction)
                    # dim, not the whole trailing slice: a 3D expert weight
                    # [E, N, K] would otherwise get bound = 1/sqrt(N*K), far
                    # below e4m3's minimum subnormal, and quantize to
                    # all-zero (measured on qwen3_moe's w13/w2).
                    if is_fp8:
                        fan_in = max(1, int(param.shape[-1]))
                    else:
                        fan_in = max(1, int(param[0].numel()))
                    bound = 1.0 / math.sqrt(fan_in)
                    fresh = torch.empty(param.shape, dtype=torch.float32)
                    fresh.uniform_(-bound, bound, generator=generator)
                else:
                    fresh = torch.randn(
                        param.shape, generator=generator,
                        dtype=torch.float32) * 0.02
                if "running_var" in name or name.endswith("_var"):
                    # variance-like buffers must stay positive: BatchNorm
                    # computes sqrt(var + eps), so a negative repair value
                    # would emit NaN.
                    fresh = fresh.abs() + 0.5
                elif "scale" in name:
                    # quantization scales must be positive: they are
                    # reciprocals of quantization steps, and the SM100 fp8
                    # GEMMs pack them as UE8M0 exponents (log2 of the
                    # scale), so a negative repair value turns every output
                    # element NaN (measured on qwen3_moe_decoder: all-NaN
                    # qkv projection while a dequantized reference GEMM on
                    # the same tensors was finite).
                    fresh = fresh.abs() + 0.5
                prepared = fresh.to(device=param.device, dtype=param.dtype)
                _REPAIR_VALUE_CACHE[cache_key] = prepared
            param.copy_(prepared)
            repaired.append(name)
    return repaired


def _postprocess_fp8_module_weights(module: Any) -> int:
    """Apply the engine's post-load FP8 weight processing to ``module``.

    The kb engine does not run FP8 modules on their checkpoint-format weights:
    ``infra/weight_loader.py:1127`` (``_postprocess_fp8_weights``) rewrites
    every ``Fp8Linear``'s ``(weight, weight_scale_inv)`` through
    ``postprocess_fp8_weights`` (UE8M0 re-quantization + DeepGEMM scale-layout
    transform) and gives every fp8 ``Qwen3MoE`` its DeepGEMM-layout
    ``w13_scale_dg`` / ``w2_scale_dg`` buffers. That step is part of the
    module's forward contract on SM100 -- DeepGEMM reads the scale tensor as
    UE8M0 exponent data, so raw (repaired/perturbed) fp32 scales produce
    all-NaN projections (measured on qwen3_moe_decoder: the qkv output was
    entirely NaN while a dequantized reference GEMM on the same tensors was
    finite).

    Called on BOTH modules AFTER the strict weight transfer: the transfer
    stays in checkpoint space (raw shapes on both sides), and the
    post-processing is a pure function of the transferred values, so both
    sides end up bit-identical -- exactly the engine's load-then-postprocess
    order. Returns the number of processed weight groups (0 for modules
    without fp8 state, making this a no-op for every non-fp8 op). A candidate
    that reuses the baseline's ``Fp8Linear`` / ``Qwen3MoE`` building blocks is
    processed identically; one that rolls its own fp8 layout owns that layout.
    """
    import torch

    if not any("float8" in str(p.dtype) for p in module.parameters(recurse=True)):
        return 0

    from fastkernels.tasks.baseline.L1.fp8_linear import (
        Fp8Linear, postprocess_fp8_weights)

    count = 0
    for sub in module.modules():
        if (isinstance(getattr(sub, "linear_op", None), Fp8Linear)
                and isinstance(getattr(sub, "weight", None), torch.Tensor)
                and "float8" in str(sub.weight.dtype)
                and isinstance(getattr(sub, "weight_scale_inv", None), torch.Tensor)):
            w_new, s_new = postprocess_fp8_weights(
                sub.weight.data, sub.weight_scale_inv.data)
            sub.weight = torch.nn.Parameter(w_new, requires_grad=False)
            sub.weight_scale_inv = torch.nn.Parameter(s_new, requires_grad=False)
            count += 1

    moe_candidates = [
        sub for sub in module.modules()
        if getattr(sub, "use_fp8", False)
        and isinstance(getattr(sub, "w13", None), torch.Tensor)
        and "float8" in str(sub.w13.dtype)
        and isinstance(getattr(sub, "w13_scale", None), torch.Tensor)
    ]
    if moe_candidates:
        # The engine's own routine (keeps checkpoint scales, adds the
        # DeepGEMM-layout ``*_scale_dg`` buffers the forward reads).
        from fastkernels.infra.weight_loader import _postprocess_moe_fp8_weights

        for sub in moe_candidates:
            count += 1 if _postprocess_moe_fp8_weights(sub) else 0
    return count


def _seed_scenario(op: str, scenario_name: str) -> int:
    """Seed torch (CPU + CUDA) from a stable hash of (operator, scenario).

    The registry materialises every shape-only fixture with ``torch.randn`` /
    ``torch.randint``, and the preparation layer below draws as well
    (``randperm``). Unseeded, the fixture differs run to run, so any scenario
    sitting near the tolerance edge flips verdict between runs -- measured:
    repeated ``flashinfer_decode`` censuses failed *different* scenario sets.
    ``hash()`` is salted per process, so the name is hashed explicitly.
    """
    import torch

    seed = _stable_seed(op, scenario_name)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


# --- input preparation --------------------------------------------------------
#
# Ported from the codex kernel runner (validated on H200: 483/483 scenarios) and
# extended where this harness's own failures pinned down a further root cause.
# The registry records *shapes*; index-like arguments therefore arrive as uniform
# random integers, which for most kernels is not a hard fixture but an invalid
# one: ``cu_seqlens`` must be a monotonic partition, ``block_table`` entries must
# be distinct in-range pages, ``slot_mapping`` entries must not collide,
# ``cache_seqlens`` must be non-degenerate, GLA gates must be log-space. An
# invalid fixture shows up as a *candidate* verdict (INCORRECT / RUNTIME_ERROR),
# which is exactly the confusion this harness exists to avoid.
#
# Everything here runs ONCE per scenario, before either module is invoked, and
# both sides are cloned from the same prepared dict -- so preparation can never
# advantage one implementation over the other.

# FlashInfer/TRTLLM paged caches use 16-token pages on this stack; used only as a
# *lower bound* when converting a block-table width into a token capacity, so an
# NHD cache with larger pages is merely under-filled, never read out of range.
_PAGE_SIZE_LOWER_BOUND = 16
# Confirmed: a decode row whose cache_seqlens is 0 leaves that output row
# untouched (undefined memory), so the two sides disagree on garbage.
_MIN_CACHE_SEQLEN = 64


def _balanced_cu_seqlens(total_tokens: int, batch: int, max_seqlen: int | None,
                         *, dtype: Any, device: Any):
    """Monotonic cu_seqlens over ``total_tokens`` (codex port, verbatim).

    Preserves the recorded tensor shape: ``cu[0] == 0``, ``cu[-1] ==
    total_tokens``, ``batch`` segments, no segment wider than ``max_seqlen``.
    """
    import torch

    batch = max(0, int(batch))
    max_seqlen = int(max_seqlen) if max_seqlen is not None else total_tokens
    max_seqlen = max(1, max_seqlen)
    if batch == 0:
        return torch.zeros(1, dtype=dtype, device=device), []
    if total_tokens > batch * max_seqlen:
        max_seqlen = math.ceil(total_tokens / batch)

    base, extra = divmod(int(total_tokens), batch)
    lengths = [base + (1 if i < extra else 0) for i in range(batch)]
    if any(length > max_seqlen for length in lengths):
        lengths = []
        remaining = int(total_tokens)
        for i in range(batch):
            slots_left = batch - i
            max_after = (slots_left - 1) * max_seqlen
            length = min(max_seqlen, max(0, remaining - max_after))
            lengths.append(length)
            remaining -= length
        if remaining > 0:
            lengths[-1] += remaining

    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + int(length))
    return torch.tensor(cu, dtype=dtype, device=device), lengths


def _prepare_cu_seqlens(inputs: dict[str, Any]) -> None:
    """Rebuild ``cu_seqlens_q`` / ``cu_seqlens_k`` (codex port, verbatim)."""
    import torch

    q_lengths: list[int] | None = None
    if isinstance(inputs.get("cu_seqlens_q"), torch.Tensor):
        cu = inputs["cu_seqlens_q"]
        q = inputs.get("q")
        if isinstance(q, torch.Tensor):
            total_q = int(q.shape[0])
            batch = max(0, int(cu.numel()) - 1)
            inputs["cu_seqlens_q"], q_lengths = _balanced_cu_seqlens(
                total_q, batch,
                int(inputs["max_seqlen_q"]) if "max_seqlen_q" in inputs else None,
                dtype=cu.dtype, device=cu.device,
            )

    if isinstance(inputs.get("cu_seqlens_k"), torch.Tensor):
        cu = inputs["cu_seqlens_k"]
        k = inputs.get("k")
        batch = max(0, int(cu.numel()) - 1)
        if (
            isinstance(k, torch.Tensor)
            and k.ndim == 4
            and isinstance(inputs.get("block_table"), torch.Tensor)
            and q_lengths is not None
            and len(q_lengths) == batch
        ):
            # Paged K: ``k`` is the cache, not a token stream -- the per-request
            # K length is the Q length (self-attention over the same tokens).
            total_k = sum(q_lengths)
        elif isinstance(k, torch.Tensor):
            total_k = int(k.shape[0])
        elif q_lengths is not None:
            total_k = sum(q_lengths)
        else:
            total_k = batch
        inputs["cu_seqlens_k"], _ = _balanced_cu_seqlens(
            total_k, batch,
            int(inputs["max_seqlen_k"]) if "max_seqlen_k" in inputs else None,
            dtype=cu.dtype, device=cu.device,
        )


def _token_total(tensor: Any) -> int:
    """Token count of a varlen-style activation ([1, T, ...] or [T, B, ...])."""
    if tensor.ndim >= 3 and int(tensor.shape[0]) == 1:
        return int(tensor.shape[1])
    if tensor.ndim >= 2:
        return int(tensor.shape[0]) * int(tensor.shape[1])
    return int(tensor.shape[0])


def _prepare_generic_cu_seqlens(inputs: dict[str, Any]) -> None:
    """Rebuild any other ``cu_seqlens*`` argument (extends the codex port).

    codex handled ``cu_seqlens`` per operator (chunk_gla, gla_attention); the
    same contract holds for every operator that takes one, so the rule is
    applied by name here. Random values are not merely a bad partition: FLA's
    chunk kernels index with ``cu[i+1]-cu[i]`` and crash outright (2 of 5
    ``chunk_gla`` scenarios were RUNTIME_ERROR for this reason).

    ``input_ids`` / ``inputs_embeds`` are pairing fallbacks (after the
    activation names) for L4 causal-LM ops whose varlen scenarios carry only a
    token stream: ``gla``'s ``cu_seqlens`` stayed raw ``randint(0, 100)``
    because none of the activation names exist in its fixture, and FLA's
    ``prepare_chunk_indices`` does ``torch.arange(cdiv(cu[i+1]-cu[i], 64))`` --
    any segment of length <= -64 raises "upper bound and lower bound
    inconsistent with step sign" (measured: gla tokens-70/2f8947b2, whose
    seeded draw contains 39-89 = -50 ... 26-89 = -63 near-misses and several
    <= -64 segments; tokens-2/-3 only survived because ``cdiv`` truncates
    their small negative lengths to zero chunks). Registry census: ``gla`` is
    the only op with an unpaired generic ``cu_seqlens``, so the fallback
    changes exactly its 3 varlen scenarios.

    When the fixture also records ``logits_indices`` with one entry per
    segment, it is rebuilt as ``cu[1:] - 1`` (each sequence's last token), the
    consistent pair the model contract expects -- the raw fixture's uniform
    ``randint(0, 100)`` happens to stay in range but samples arbitrary
    positions.
    """
    import torch

    for name in sorted(inputs):
        if not name.startswith("cu_seqlens") or name in ("cu_seqlens_q", "cu_seqlens_k"):
            continue
        cu = inputs.get(name)
        if not isinstance(cu, torch.Tensor):
            continue
        paired = None
        for candidate in ("q", "query", "hidden_states", "x", "k", "key", "v",
                          "input_ids", "inputs_embeds"):
            value = inputs.get(candidate)
            if isinstance(value, torch.Tensor) and value.ndim >= 2:
                paired = value
                break
        if paired is None:
            continue
        inputs[name], _ = _balanced_cu_seqlens(
            _token_total(paired), max(0, int(cu.numel()) - 1), None,
            dtype=cu.dtype, device=cu.device,
        )
        rebuilt = inputs[name]
        logits_indices = inputs.get("logits_indices")
        if (isinstance(logits_indices, torch.Tensor)
                and not logits_indices.is_floating_point()
                and logits_indices.numel() == rebuilt.numel() - 1):
            inputs["logits_indices"] = (
                (rebuilt[1:] - 1).to(dtype=logits_indices.dtype)
                .reshape(logits_indices.shape)
            )


def _cache_token_capacity(inputs: dict[str, Any]) -> int | None:
    """Largest per-request KV length the fixture's cache can actually serve."""
    import torch

    capacity = None
    block_table = inputs.get("block_table")
    if isinstance(block_table, torch.Tensor) and block_table.ndim == 2:
        capacity = int(block_table.shape[1]) * _PAGE_SIZE_LOWER_BOUND
    else:
        for name in ("k_cache", "kv_cache", "k"):
            cache = inputs.get(name)
            if isinstance(cache, torch.Tensor) and cache.ndim == 4:
                # Unpaged layout [batch, seqlen, heads, dim].
                capacity = int(cache.shape[1])
                break
    declared = inputs.get("max_seq_len")
    if isinstance(declared, int) and declared > 0:
        capacity = declared if capacity is None else min(capacity, declared)
    return capacity


def _prepare_paged_attention_inputs(inputs: dict[str, Any]) -> None:
    """Valid ``block_table`` + non-degenerate ``cache_seqlens`` (codex port + fix).

    ``block_table``: codex's arange-over-pages, verbatim -- every row gets
    distinct in-range pages, so no two requests alias the same KV.

    ``cache_seqlens``: codex filled with ``max_seq_len - i % 7`` clamped to >= 1.
    Clamped to >= ``_MIN_CACHE_SEQLEN`` here (bounded by the cache's real
    capacity), because the registry's uniform ``randint(0, 100)`` produces rows
    of length 0 whose output rows are never written -- the confirmed cause of
    ``flashinfer_decode`` failing a *different* ~9 of 56 scenarios per run.
    """
    import torch

    block_table = inputs.get("block_table")
    if isinstance(block_table, torch.Tensor):
        num_blocks = 1
        for cache_name in ("k_cache", "k"):
            cache = inputs.get(cache_name)
            if isinstance(cache, torch.Tensor) and cache.ndim == 4:
                num_blocks = int(cache.shape[0])
                break
        inputs["block_table"] = (
            torch.arange(block_table.numel(), dtype=block_table.dtype,
                         device=block_table.device)
            .reshape_as(block_table)
            .remainder(max(1, num_blocks))
        )

    cache_seqlens = inputs.get("cache_seqlens")
    if isinstance(cache_seqlens, torch.Tensor):
        capacity = _cache_token_capacity(inputs) or 1
        values = torch.full_like(cache_seqlens, capacity)
        if cache_seqlens.numel() > 1:
            values -= torch.arange(
                cache_seqlens.numel(), dtype=cache_seqlens.dtype,
                device=cache_seqlens.device,
            ).remainder(min(capacity, 7))
        values.clamp_(min=min(_MIN_CACHE_SEQLEN, capacity), max=capacity)
        inputs["cache_seqlens"] = values


def _prepare_slot_mapping(inputs: dict[str, Any], init_args: dict[str, Any]) -> None:
    """Collision-free ``slot_mapping`` (replaces codex's ``arange``).

    The registry draws slots i.i.d. from ``[0, num_slots)``; two tokens landing
    on one slot means two thread blocks race for the same cache line, and the
    winner differs between the baseline and candidate runs. At n=16384 over
    111253*16 slots the expected number of colliding pairs is ~75, which is
    exactly the one ``store_kvcache`` scenario that failed. Sampling without
    replacement removes the race while keeping the writes spread across the
    cache (codex's ``arange`` only ever touches slots 0..n-1, so the page
    arithmetic of the HND kernel is never exercised).
    """
    import torch

    slot_mapping = inputs.get("slot_mapping")
    if not isinstance(slot_mapping, torch.Tensor) or slot_mapping.numel() == 0:
        return
    n = int(slot_mapping.numel())

    upper = 0
    page_size = init_args.get("page_size")
    for name in ("k_cache", "kv_cache"):
        cache = inputs.get(name)
        if isinstance(cache, torch.Tensor) and cache.ndim >= 2 and isinstance(page_size, int):
            upper = int(cache.shape[0]) * int(page_size)
            break
    if upper < n:
        upper = max(int(slot_mapping.max().item()) + 1, n)

    if upper < n:  # cache smaller than the token count: keep it in range
        slots = torch.arange(n, dtype=torch.int64, device=slot_mapping.device)
    else:
        slots = torch.randperm(upper, device=slot_mapping.device)[:n]
    inputs["slot_mapping"] = slots.to(dtype=slot_mapping.dtype).reshape(slot_mapping.shape)


def _materialize_leaf_spec(spec: dict[str, Any], device: str):
    """Build one tensor from the registry's ``{shape, dtype}`` leaf spec."""
    import torch

    from fastkernels.bench.kernels.scenario_registry import _parse_dtype

    dtype = _parse_dtype(spec["dtype"])
    shape = list(spec["shape"])
    if dtype in (torch.int32, torch.int64):
        return torch.randint(0, 100, shape, dtype=dtype, device=device)
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, dtype=torch.uint8, device=device).bool()
    if "float8" in str(dtype):
        source = _parse_dtype(spec.get("source_dtype", "bfloat16"))
        return torch.randn(shape, dtype=source, device=device).to(dtype)
    return torch.randn(shape, dtype=dtype, device=device)


def _prepare_structured_inputs(inputs: dict[str, Any], device: str) -> None:
    """Materialize ``{kind: list|dict, items: ...}`` specs into real tensors.

    Ops such as ``yolov10_concat(xs=[Tensor, ...])`` and
    ``yolov10_neck(feats={name: Tensor})`` take a *container* of tensors.
    ``scenario_registry._materialize_shape_inputs`` only understands a leaf
    ``{shape, dtype}`` dict and passes anything else through verbatim, so the op
    would receive the YAML dict itself. Building the container here keeps the
    registry untouched (it is shared with the release runner).
    """
    for name, spec in list(inputs.items()):
        if not isinstance(spec, dict) or "shape" in spec:
            continue
        kind, items = spec.get("kind"), spec.get("items")
        if kind == "list" and isinstance(items, list):
            inputs[name] = [_materialize_leaf_spec(i, device) for i in items]
        elif kind == "dict" and isinstance(items, dict):
            inputs[name] = {
                k: _materialize_leaf_spec(v, device) for k, v in items.items()
            }


# Traced names for "how many rows does the table have"; first hit wins.
_INDEX_LIMIT_KEYS = ("num_embeddings", "vocab_size", "org_num_embeddings",
                     "org_vocab_size")


def _prepare_index_inputs(op: str, inputs: dict[str, Any],
                          init_args: dict[str, Any]) -> None:
    """Fold token-index arguments into the table they index (extends the port).

    The registry has no constrained path for ``input_ids``: it emits
    ``randint(0, 100)``. Against an embedding table with fewer rows than that,
    ``nn.Embedding`` raises a *device-side* assert, which does not just fail the
    scenario -- it poisons the CUDA context, so every later scenario in the same
    process dies too. Modulo keeps every id in range without changing the shape
    or the dtype the trace recorded.
    """
    import torch

    limit = None
    for key in _INDEX_LIMIT_KEYS:
        value = init_args.get(key)
        if isinstance(value, int) and value > 0:
            limit = value
            break

    names = ["input_ids"]
    if "embed" in op:
        # Position/segment ids index their own tables in embedding layers.
        names += ["token_type_ids", "position_ids", "positions"]
    for name in names:
        tensor = inputs.get(name)
        if not isinstance(tensor, torch.Tensor) or tensor.is_floating_point():
            continue
        bound = limit
        if name == "token_type_ids":
            declared = init_args.get("type_vocab_size")
            bound = declared if isinstance(declared, int) and declared > 0 else limit
        if name in ("position_ids", "positions"):
            declared = init_args.get("max_position_embeddings")
            bound = declared if isinstance(declared, int) and declared > 0 else limit
        if isinstance(bound, int) and bound > 0:
            inputs[name] = tensor.remainder(bound)


def _prepare_log_gates(inputs: dict[str, Any]) -> None:
    """Map GLA-family gates into log space (extends the codex port).

    ``tasks/baseline/L1/chunk_gla.py`` documents ``g`` as a *log-space* forget
    gate, and ``fused_recurrent_gla.py`` documents ``gk`` the same way, so the
    kernel exponentiates a running sum of these values. The registry's
    ``randn`` fixture is half positive, and over 1084 timesteps the cumulative
    sum overflows to inf and then NaN. codex substituted ``-rand * 0.01``, which
    discards the fixture; ``logsigmoid`` keeps it, is the convention FLA's own
    tests use, and guarantees the required ``g <= 0``.
    """
    import torch

    for name in ("g", "gk"):
        gate = inputs.get(name)
        if isinstance(gate, torch.Tensor) and gate.is_floating_point():
            inputs[name] = torch.nn.functional.logsigmoid(gate.float()).to(gate.dtype)


_FP8_WEIGHT_CACHE: dict[Any, Any] = {}


def _prepare_fp8_linear_inputs(inputs: dict[str, Any]) -> None:
    """Coherent FP8 weight + DeepGEMM scale layout (codex port, verbatim)."""
    import torch

    weight = inputs.get("weight_fp8")
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        return
    if str(weight.dtype) != "torch.float8_e4m3fn":
        return
    # A shape-only fixture should not require runtime FlashInfer cubin
    # compilation for tiny M; DeepGEMM covers the same contract here.
    os.environ.setdefault("VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER", "0")

    key = (str(weight.dtype), int(weight.shape[0]), int(weight.shape[1]), weight.device)
    cached = _FP8_WEIGHT_CACHE.get(key)
    if cached is None:
        from fastkernels.tasks.baseline.L1.fp8_linear import postprocess_fp8_weights

        n, k = int(weight.shape[0]), int(weight.shape[1])
        block = 128
        raw_scale = torch.ones(
            math.ceil(n / block), math.ceil(k / block),
            dtype=torch.float32, device=weight.device,
        ) * 0.02
        raw_weight = torch.randn(
            n, k, dtype=torch.bfloat16, device=weight.device,
        ).to(torch.float8_e4m3fn)
        cached = postprocess_fp8_weights(raw_weight, raw_scale)
        _FP8_WEIGHT_CACHE[key] = cached
    inputs["weight_fp8"], inputs["weight_scale_inv"] = cached


# One prepared (w1, w2, quant_config) triple per distinct problem shape; the
# raw draws use a dedicated generator seeded from the shape (not the
# per-scenario RNG), so the fixture is the same no matter which scenario runs
# first or whether --scenarios filtered the selection.
_MXFP4_FIXTURE_CACHE: dict[tuple, tuple] = {}


def _prepare_mxfp4_moe_inputs(inputs: dict[str, Any], device: str) -> None:
    """Materialise the MXFP4 expert-weight objects ``mxfp4_moe`` needs.

    ``tasks/baseline/L1/mxfp4_moe.py::Mxfp4MoE`` is stateless: expert weights
    arrive as *forward* arguments (``w1``/``w2`` are swizzled
    ``triton_kernels`` tensors, ``quant_config`` an ``Mxfp4MoEQuantConfig``).
    The tracer could not record those objects, so the registry carries only
    ``hidden_states`` / ``gating_output`` / scalars and every scenario died
    with missing forward arguments.

    The harness generates seeded RAW MXFP4 tensors in the exact checkpoint
    layout ``GptOssMoE`` declares (``gpt_oss_moe.py:54-78``): packed uint8
    FP4 pairs ``[E, 2*I_pad, H//2]`` / ``[E, H, I_pad//2]`` with E8M0 block
    scales over 32-element groups, biases float32 (the Triton kernel asserts
    this). It then builds the forward objects through the baseline's own
    trusted statics -- ``Mxfp4MoE.prepare_weight`` and ``make_quant_config``,
    the same calls ``GptOssMoE.process_weights_after_loading`` makes -- ONCE,
    and hands the identical objects to both sides. No per-side preparation.

    Value ranges: every uint8 byte is a valid pair of e2m1 values (the format
    has no NaN/Inf encodings), so the packed weights are unconstrained draws.
    The E8M0 scale exponents ARE constrained, to [121, 127] (decoded 2^-6 ..
    2^0): an unconstrained exponent can reach 2^+-127 and overflows the bf16
    activations; a bounded random scale is still a fully valid, non-degenerate
    MXFP4 fixture.

    ``E`` and ``H`` come from the recorded activations; the intermediate size
    is not recorded for this op, so it comes from the registry's
    ``gpt_oss_moe`` config (``intermediate_size: 1440`` per rank), padded to
    64 exactly like ``GptOssMoE`` (``_round_up`` -> 1472).
    """
    import torch

    hidden = inputs.get("hidden_states")
    gating = inputs.get("gating_output")
    if not (isinstance(hidden, torch.Tensor) and isinstance(gating, torch.Tensor)):
        return
    if "w1" in inputs and "w2" in inputs and "quant_config" in inputs:
        return

    num_experts = int(gating.shape[-1])
    hidden_size = int(hidden.shape[-1])
    intermediate = 1440  # registry gpt_oss_moe config, per rank
    i_pad = (intermediate + 63) // 64 * 64  # GptOssMoE._round_up(..., 64)
    block = 32  # GptOssMoE.MXFP4_BLOCK

    key = (num_experts, hidden_size, i_pad, str(device))
    cached = _MXFP4_FIXTURE_CACHE.get(key)
    if cached is None:
        from fastkernels.tasks.baseline.L1.mxfp4_moe import Mxfp4MoE

        generator = torch.Generator(device="cpu").manual_seed(_stable_seed(
            "mxfp4_moe", "expert_weights",
            f"{num_experts}x{hidden_size}x{i_pad}"))

        def _u8(*shape: int, low: int = 0, high: int = 256):
            return torch.randint(
                low, high, shape, generator=generator, dtype=torch.uint8,
            ).to(device)

        w1_raw = _u8(num_experts, 2 * i_pad, hidden_size // 2)
        w1_scale = _u8(num_experts, 2 * i_pad, hidden_size // block,
                       low=121, high=128)
        w2_raw = _u8(num_experts, hidden_size, i_pad // 2)
        w2_scale = _u8(num_experts, hidden_size, i_pad // block,
                       low=121, high=128)
        w1_bias = (torch.randn(num_experts, 2 * i_pad, generator=generator,
                               dtype=torch.float32) * 0.02).to(device)
        w2_bias = (torch.randn(num_experts, hidden_size, generator=generator,
                               dtype=torch.float32) * 0.02).to(device)

        w1, w1_precision = Mxfp4MoE.prepare_weight(w1_raw, w1_scale)
        w2, w2_precision = Mxfp4MoE.prepare_weight(w2_raw, w2_scale)
        quant_config = Mxfp4MoE.make_quant_config(
            w1_precision, w2_precision, w1_bias=w1_bias, w2_bias=w2_bias,
        )
        cached = (w1, w2, quant_config)
        _MXFP4_FIXTURE_CACHE[key] = cached

    inputs["w1"], inputs["w2"], inputs["quant_config"] = cached


# One (model, vae) fixture pair per distinct problem geometry, built ONCE per
# process; the draws happen inside a fork_rng'd CPU stream seeded from the op
# name (not the per-scenario RNG), so the fixture is bit-identical no matter
# which scenario runs first or whether --scenarios filtered the selection.
_OASIS_ROLLOUT_MODULE_CACHE: dict[tuple, tuple] = {}


def _prepare_oasis_rollout_inputs(inputs: dict[str, Any], device: str) -> None:
    """Materialise the DiT + VAE sub-networks ``oasis_rollout`` needs.

    ``tasks/baseline/L3/oasis_rollout.py:69-81`` -- ``OasisRollout.forward(
    model, vae, prompt, actions, ...)`` -- takes the ENTIRE Oasis diffusion
    transformer and VAE decoder as *forward* arguments (the L4 pipeline owns
    them and passes its own, ``tasks/baseline/L4/oasis.py:74-81,103-113``).
    The registry records only tensor shapes, so every scenario died with
    "missing 2 required positional arguments: 'model' and 'vae'". The harness
    builds both here, in trusted code, exactly as the L4 pipeline does --
    ``DiT_S_2`` (oasis.py:19-20: patch_size=2, hidden_size=1024, depth=16,
    num_heads=16, max_frames=32) and ``ViT_L_20_Shallow_Encoder``
    (oasis.py:23-35: latent_dim=16, patch_size=20, enc 1024x6x16, dec
    1024x12x16) -- and hands the IDENTICAL instances to both sides as
    read-only forward inputs (the ``gpt_oss_decoder`` rotary_emb pattern; the
    runner's ``_clone_inputs`` passes non-tensor values through by
    reference). Geometry the fixture *does* record is derived from it rather
    than assumed: ``external_cond_dim`` from ``actions.shape[-1]``, the frame
    size from ``prompt.shape[-2:]``, and the DiT's latent grid from the frame
    size over the VAE patch size.

    Construction is wrapped in a CPU ``fork_rng`` seeded from the op name:
    both nets initialise through the global RNG (``oasis_dit.py:60-83``,
    ``oasis_autoencoder_kl.py:97-111``), so this keeps the modules
    reproducible across processes and scenario orders while leaving the
    per-scenario fixture stream untouched.

    The fresh DiT is deliberately degenerate: ``initialize_weights`` zeroes
    every adaLN modulation head AND the final projection
    (``oasis_dit.py:74-83``), so an untouched instance predicts exactly v=0
    and the rollout output would not depend on the model calls at all -- a
    candidate that never invokes the DiT would pass vacuously. Running the
    harness's own ``_repair_degenerate_parameters`` over both sub-networks
    replaces exactly those all-zero tensors (and the zero-init biases) with
    seeded non-degenerate values, so the fixture discriminates how the
    candidate drives the model; the xavier-initialised weights are finite,
    non-zero, and left untouched. No perturbation: these are fixtures, not
    the module under test, and they never pass through the weight transfer.

    Kept at construction dtype (fp32): the rollout itself autocasts the DiT
    calls and the prompt encode to the scenario's fp16
    (oasis_rollout.py:28-32,54,116) and ``decode_latents`` reads its compute
    dtype from the VAE's own weights (oasis_rollout.py:63), so the modules'
    storage dtype is not a scenario knob.
    """
    import torch

    prompt = inputs.get("prompt")
    actions = inputs.get("actions")
    if not (isinstance(prompt, torch.Tensor) and prompt.ndim == 5
            and isinstance(actions, torch.Tensor)):
        return
    if isinstance(inputs.get("model"), torch.nn.Module) and isinstance(
            inputs.get("vae"), torch.nn.Module):
        return

    vae_patch = 20  # ViT-L/20 shallow encoder (oasis.py:23-35)
    height, width = int(prompt.shape[-2]), int(prompt.shape[-1])
    external_cond_dim = int(actions.shape[-1])
    key = (external_cond_dim, height, width, str(device))
    cached = _OASIS_ROLLOUT_MODULE_CACHE.get(key)
    if cached is None:
        from fastkernels.tasks.baseline.L3.oasis_autoencoder_kl import (
            OasisAutoencoderKL)
        from fastkernels.tasks.baseline.L3.oasis_dit import OasisDiT

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(_stable_seed(
                "oasis_rollout", "fixture_modules",
                f"{external_cond_dim}x{height}x{width}"))
            model = OasisDiT(
                input_h=height // vae_patch, input_w=width // vae_patch,
                patch_size=2, hidden_size=1024, depth=16, num_heads=16,
                external_cond_dim=external_cond_dim, max_frames=32,
            )
            vae = OasisAutoencoderKL(
                latent_dim=16, patch_size=vae_patch,
                enc_dim=1024, enc_depth=6, enc_heads=16,
                dec_dim=1024, dec_depth=12, dec_heads=16,
                input_height=height, input_width=width,
            )
        model = model.to(device).eval()
        vae = vae.to(device).eval()
        repaired = _repair_degenerate_parameters(model, "oasis_rollout.model")
        repaired += _repair_degenerate_parameters(vae, "oasis_rollout.vae")
        _log(f"oasis_rollout: built DiT+VAE fixture modules (latent grid "
             f"{height // vae_patch}x{width // vae_patch}, external_cond_dim "
             f"{external_cond_dim}); re-initialised {len(repaired)} zero-init/"
             f"degenerate tensor(s)")
        cached = (model, vae)
        _OASIS_ROLLOUT_MODULE_CACHE[key] = cached
    inputs["model"], inputs["vae"] = cached


def _prepare_moe_grouped_gemm_inputs(inputs: dict[str, Any]) -> None:
    """Block-aligned MoE routing metadata (codex port, verbatim).

    ``sorted_token_ids`` becomes ``arange(valid)`` padded with the sentinel
    ``valid`` (the kernel masks ``id < num_valid``), ``expert_ids`` one entry per
    BLOCK_SIZE_M block in ascending order, and ``num_tokens_post_padded`` the
    honest padded length rather than the registry's full-buffer value.
    """
    import torch

    a = inputs.get("A")
    b = inputs.get("B")
    sorted_token_ids = inputs.get("sorted_token_ids")
    expert_ids = inputs.get("expert_ids")
    num_tokens_post_padded = inputs.get("num_tokens_post_padded")
    if not (
        isinstance(a, torch.Tensor)
        and isinstance(b, torch.Tensor)
        and isinstance(sorted_token_ids, torch.Tensor)
        and isinstance(expert_ids, torch.Tensor)
        and isinstance(num_tokens_post_padded, torch.Tensor)
    ):
        return

    top_k = int(inputs.get("top_k", 1))
    valid = int(a.shape[0]) * max(1, top_k)
    inputs["config"] = {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 16,
        "num_warps": 4,
        "num_stages": 5,
    }
    block_size = int(inputs["config"]["BLOCK_SIZE_M"])
    padded = math.ceil(int(sorted_token_ids.numel()) / block_size) * block_size
    values = torch.full(
        (padded,), valid,
        dtype=sorted_token_ids.dtype, device=sorted_token_ids.device,
    )
    values[: min(valid, padded)] = torch.arange(
        min(valid, padded),
        dtype=sorted_token_ids.dtype, device=sorted_token_ids.device,
    )
    inputs["sorted_token_ids"] = values

    used_blocks = math.ceil(padded / block_size)
    inputs["expert_ids"] = torch.arange(
        used_blocks, dtype=expert_ids.dtype, device=expert_ids.device,
    ).remainder(max(1, int(b.shape[0])))
    inputs["num_tokens_post_padded"] = torch.full_like(num_tokens_post_padded, padded)


def _prepare_fused_experts_inputs(inputs: dict[str, Any]) -> None:
    """In-range expert ids + normalised routing weights (codex port, verbatim)."""
    import torch

    topk_ids = inputs.get("topk_ids")
    topk_weights = inputs.get("topk_weights")
    num_experts = int(inputs.get("num_experts", 0))
    if isinstance(topk_ids, torch.Tensor) and num_experts > 0:
        inputs["topk_ids"] = topk_ids.remainder(num_experts).to(torch.int32)
    if isinstance(topk_weights, torch.Tensor):
        inputs["topk_weights"] = torch.softmax(topk_weights.float(), dim=-1)

    if bool(inputs.get("use_fp8_w8a8", False)):
        for key in ("w13_scale", "w13_scale_dg", "w2_scale", "w2_scale_dg"):
            scale = inputs.get(key)
            if isinstance(scale, torch.Tensor):
                inputs[key] = torch.full_like(scale.float(), 0.005)


def _prepare_inputs_for_target(op: str, inputs: dict[str, Any], device: str,
                               init_args: dict[str, Any]) -> dict[str, Any]:
    """Repair a materialised fixture in place (codex port + the fixes above).

    Not ported from codex, deliberately:
      * ``_prepare_mxfp4_moe_inputs`` / ``_prepare_inputs_for_module`` -- they
        pull weights out of a HuggingFace snapshot (``safetensors``,
        ``huggingface_hub``) and then call *module* methods
        (``module.prepare_weight``) to build per-implementation inputs, which
        both breaks this file's stdlib+torch+repo-only rule and hands each side
        differently-derived tensors, defeating the strict weight transfer.
      * ``_canonicalize_output_for_target`` -- it re-sorts ``moe_align`` outputs
        before comparison, i.e. it changes the comparator, not the fixture.
    """
    import torch

    try:
        from fastkernels.infra.context import set_context

        set_context(False)
    except Exception:
        pass

    _prepare_structured_inputs(inputs, device)
    _prepare_cu_seqlens(inputs)
    _prepare_generic_cu_seqlens(inputs)
    _prepare_paged_attention_inputs(inputs)
    _prepare_slot_mapping(inputs, init_args)
    _prepare_index_inputs(op, inputs, init_args)
    _prepare_log_gates(inputs)

    if op == "mrope_input_positions" and "input_tokens" not in inputs:
        offsets = []
        for key in ("image_offsets", "video_offsets"):
            value = inputs.get(key)
            if isinstance(value, list):
                offsets.extend(int(v) for v in value)
        seq_len = (max(offsets) + 16) if offsets else 16
        inputs["input_tokens"] = [0] * seq_len

    if op == "vision_rotary_emb":
        sms = int(inputs.get("spatial_merge_size", 2))
        inputs.setdefault("grid_thw_list", [[1, sms, sms]])
        inputs.setdefault("dtype", torch.bfloat16)
        inputs.setdefault("device", torch.device(device))

    if op == "vision_pos_embed_interpolate":
        inputs.setdefault("grid_thw_list", [[1, 2, 2]])
        inputs.setdefault("dtype", torch.bfloat16)
        inputs.setdefault("device", torch.device(device))

    if op == "yolov10_concat":
        inputs.setdefault("xs", [
            torch.randn((1, 8, 4, 4), dtype=torch.bfloat16, device=device),
            torch.randn((1, 8, 4, 4), dtype=torch.bfloat16, device=device),
        ])

    if op == "oasis_patch_embed" and isinstance(inputs.get("x"), torch.Tensor):
        inputs["x"] = inputs["x"].to(torch.bfloat16)

    if op == "oasis_rollout":
        _prepare_oasis_rollout_inputs(inputs, device)

    if op in ("attention", "attention_impl",
              "llama_decoder", "qwen3_moe_decoder", "gpt_oss_decoder"):
        # These layers read their KV metadata from the global Context, the way
        # vLLM reads ``get_forward_context()``; the fixture carries none. The
        # L3 decoders wrap the same L2 ``Attention``: with only the default
        # ``set_context(False)`` above, it takes the decode path and calls the
        # TRTLLM decode kernel with no KV cache and no block table
        # ("Mismatched type on argument #7 ... Expected DLTensor* but got
        # None"). A single-sequence prefill context over the recorded token
        # count is the valid fixture, exactly as for the L2 attention ops.
        from fastkernels.infra.context import set_context

        tensor = inputs.get("hidden_states", inputs.get("query"))
        if isinstance(tensor, torch.Tensor):
            n_tokens = int(tensor.shape[0])
            cu = torch.tensor([0, n_tokens], dtype=torch.int32, device=tensor.device)
            set_context(True, cu_seqlens_q=cu, cu_seqlens_k=cu,
                        max_seqlen_q=n_tokens, max_seqlen_k=n_tokens)

    if op == "gpt_oss_decoder" and "rotary_emb" not in inputs:
        # ``GptOssDecoderLayer.forward(positions, hidden_states, residual,
        # rotary_emb)`` takes the rotary embedding as a *forward* argument
        # (shared across layers at the model level); the registry cannot
        # express a module input, so every scenario died with "missing 1
        # required positional argument: 'rotary_emb'". Build it exactly as the
        # model does (``tasks/baseline/L4/gpt_oss.py:157-166``) with the
        # ``GptOssConfig`` defaults; construction is deterministic (no RNG)
        # and the single instance is shared by both sides as a read-only
        # forward input.
        from fastkernels.tasks.baseline.L1.yarn_rotary_emb import YaRNRotaryEmbedding

        cfg = init_args.get("config") or {}
        head_dim = int(cfg.get("head_dim", 64)) if isinstance(cfg, dict) else 64
        rotary = YaRNRotaryEmbedding(
            head_dim,
            131072,      # max_position_embeddings (GptOssConfig default)
            150000.0,    # rope_theta
            scaling_factor=32.0,
            original_max_position_embeddings=4096,
            beta_fast=32.0,
            beta_slow=1.0,
            truncate=False,
        )
        inputs["rotary_emb"] = rotary.to(device).eval()

    if op == "mxfp4_moe":
        _prepare_mxfp4_moe_inputs(inputs, device)

    if op == "fp8_linear":
        _prepare_fp8_linear_inputs(inputs)

    if op == "parallel_linear":
        os.environ["VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER"] = "0"

    if op == "moe_grouped_gemm":
        _prepare_moe_grouped_gemm_inputs(inputs)

    if op == "fused_experts":
        _prepare_fused_experts_inputs(inputs)

    return inputs


# --- reporting helpers -------------------------------------------------------

def _scenario_axes(scenario) -> dict[str, Any]:
    """Flat, JSON-safe axes from the scenario's shape dict.

    AKO4X consumers format axes as ``k=v`` pairs and filter on scalar values, so
    each declared shape is flattened to ``<arg>_dim<i>`` plus ``<arg>_dtype``.
    """
    axes: dict[str, Any] = {}
    for arg, spec in scenario.inputs.items():
        if isinstance(spec, dict) and "shape" in spec:
            for i, dim in enumerate(spec["shape"]):
                axes[f"{arg}_dim{i}"] = int(dim)
            if spec.get("dtype") is not None:
                axes[f"{arg}_dtype"] = str(spec["dtype"])
        elif spec is None or isinstance(spec, (int, float, str, bool)):
            axes[arg] = spec
    return axes


def _num(x: Any) -> Any:
    """JSON-safe float: non-finite -> "NaN" (the contract's sentinel)."""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return "NaN"
    return f if math.isfinite(f) else "NaN"


def _has_nonfinite(value: Any) -> bool:
    """True if any floating tensor in the tree holds a NaN or an infinity."""
    import torch

    if isinstance(value, torch.Tensor):
        if not value.is_floating_point():
            return False
        try:
            return bool((~torch.isfinite(value.float())).any().item())
        except Exception:
            return False
    if isinstance(value, (tuple, list)):
        return any(_has_nonfinite(v) for v in value)
    if isinstance(value, dict):
        return any(_has_nonfinite(v) for v in value.values())
    return False


def _canonicalize_moe_align_output(output: Any, inputs: dict[str, Any]) -> Any:
    """Order-canonical form of ``moe_align``'s output triple (op-scoped).

    ``MoeAlign`` returns ``(sorted_token_ids, expert_ids,
    num_tokens_post_padded)`` built by parallel atomic appends, so the order of
    token ids *within* one expert's block-group differs run to run. That order
    is not part of the contract: the only consumer,
    ``tasks/baseline/L2/fused_experts.py:309-354``, uses both arrays purely as
    gather/scatter index metadata. The buffers are also pre-allocated
    (``torch.empty``) and only the first ``num_tokens_post_padded`` entries are
    written, so the tail beyond it is garbage on both sides.

    Canonical form (ported from the codex runner's
    ``_canonicalize_output_for_target``, runner.py:955): truncate both arrays
    to the valid padded length, expand ``expert_ids`` from per-block to
    per-token, and sort by (expert, token id). Expert grouping and the
    padding sentinel (token id == numel, larger than every real id, so it
    sorts to the end of its own expert's group) are preserved; a candidate
    that assigns any token to a different expert, drops a token, or reports a
    different padded length still mismatches. Applied identically to both
    sides, before the runner's unmodified comparison; tolerances untouched.
    """
    import torch

    if not (isinstance(output, (tuple, list)) and len(output) == 3):
        return output
    sorted_token_ids, expert_ids, num_tokens_post_padded = output
    if not isinstance(num_tokens_post_padded, torch.Tensor):
        return output
    valid = int(num_tokens_post_padded.reshape(-1)[0].item())
    block_size = max(1, int(inputs.get("block_size", 1) or 1))
    if isinstance(sorted_token_ids, torch.Tensor) and isinstance(expert_ids, torch.Tensor):
        sorted_token_ids = sorted_token_ids[:valid]
        expert_ids = expert_ids[:math.ceil(valid / block_size)]
        if sorted_token_ids.numel() > 0:
            expanded_experts = expert_ids.repeat_interleave(
                block_size)[:sorted_token_ids.numel()]
            token_range = max(int(sorted_token_ids.max().item()) + 1, 1)
            order = torch.argsort(
                expanded_experts.to(torch.int64) * token_range
                + sorted_token_ids.to(torch.int64))
            sorted_token_ids = sorted_token_ids[order]
            expert_ids = expanded_experts[order]
    return type(output)((sorted_token_ids, expert_ids, num_tokens_post_padded))


# --- Difference 10: the fp64 oracle arm ---------------------------------------
#
# WHY these three ops need one. Each subtracts nearly-equal large quantities on
# the way to its output -- gpt_oss_moe sums signed top-k slot contributions,
# chunk_gla adds an inter-chunk state term to an intra-chunk attention term of
# opposite sign, vision_block adds an attention/MLP update to a residual it
# nearly cancels. Cancellation does not create error, it *reveals* it: the
# absolute error carried by the large operands survives while the result
# shrinks, so a difference that is 2 ULP where it is produced is 1.1x-3.1x of
# the tolerance band by the time it reaches the output. Every one of the ten
# independently-written pure-torch vision_block variants lands on the same
# deviation, which is the signature of "the kernel rounds here and torch does
# not", not of a bug in any of them.
#
# WHY AN ORACLE IS THE RIGHT INSTRUMENT. The standard arm asks "is the candidate
# close to the baseline". At a cancellation element that question is unanswerable
# without knowing which side is right -- and for gpt_oss_moe the measurement says
# the baseline is the wrong one (bf16-rounding the fp64-exact slot values
# reproduces the naive reference bit-for-bit). Arm (b) asks the answerable
# question instead: "is the candidate at least as accurate as the production
# baseline, up to the same tolerance band". That never fails an implementation
# that is at least as good as what we ship, and it cannot be gamed by being
# wrong, because being wrong moves a candidate away from the truth broadly while
# the extra allowance |base - truth| is only large at the isolated elements where
# the baseline itself is inaccurate.
#
# HOW THE TRUTH IS BUILT, AND WHY IT IS TRUSTWORTHY.
#   * It is computed HERE, in the harness, from ``inputs`` (the prepared fixture
#     dict, which neither module ever sees -- both sides run on clones) and from
#     ``baseline_sd`` (the state_dict snapshot taken before either forward, i.e.
#     exactly the tensors the strict transfer copied into the candidate). The
#     candidate contributes nothing to it, and the same truth would be produced
#     if the candidate did not exist.
#   * It is float64 end to end, with no intermediate rounding: every stored
#     value (bf16 weights, uint8 MXFP4 payloads, fp32 states) is upcast exactly,
#     so the only error left is fp64 roundoff, ~1e-13 of a bf16 ULP even after
#     the cancellation amplification that motivates this whole difference.
#   * It is deterministic and side-independent: pure arithmetic on tensors that
#     are already fixed before either module runs, no RNG, no kernel autotuning.
#
# WHY NOT JUST RUN ``tasks/reference/**`` IN FLOAT64 (the obvious construction).
# Those files are *kernel mirrors*, not ideal-math references: they deliberately
# hard-wire fp32 compute and reproduce the production kernel's rounding points,
# which is what makes them good specification prose and useless as an oracle.
# Feeding them float64 inputs silently computes in float32 rather than raising:
#   * ``tasks/reference/L2/gpt_oss_moe.py:230-231`` (``output = zeros_like(...,
#     dtype=torch.float32)``, ``x_all = hidden_states.float()``) and :244/:261
#     (``w1[expert].float()``) pin the matmuls to fp32, while :256 and :267
#     round the SwiGLU output and each slot's contribution back to the input
#     dtype on purpose ("mirror that rounding"); ``prepare_weight`` (:181)
#     dequantizes MXFP4 to bfloat16.
#   * ``tasks/reference/L1/chunk_gla.py:80-90`` rounds the per-chunk state and
#     the decayed q/k to the input dtype and calls ``.float()`` on every
#     operand; its own docstring (:8-12) states that mirroring those rounding
#     points is the point of the file.
#   * ``tasks/reference/L3/vision_block.py:58`` (LayerNorm ``x.float()``),
#     :260-266 (rotary in fp32, rounded once) do the same, and :147-151 dispatches
#     to ``aten._scaled_dot_product_flash_attention``, which has no fp64 kernel
#     at all.
# So for all three ops the oracle is written out below as the same naive math in
# float64. Each is a transcription of the reference's semantics (routing, SwiGLU
# with the OAI clamps, the GLA recurrence, the pre-norm block wiring) with the
# fp32/bf16 pins removed -- and it is checked against the baseline on every use:
# ``oracle_baseline_ratio`` in the result entry is how far the BASELINE sits from
# the oracle in tolerance units, which would be enormous if the oracle's
# semantics were wrong, and is ~1 for these ops.
#
# FAILURE MODE. Anything unexpected (missing state_dict key, unknown activation,
# structure mismatch) returns None / False and leaves the standard verdict
# standing. The oracle never invents a pass it cannot justify.

_FP64_ORACLE_OPS = ("gpt_oss_moe", "chunk_gla", "vision_block")

# Whether arm (b) may CHANGE a verdict. False = the oracle is computed and its
# numbers are reported, but the verdict is the standard arm's, i.e. grading is
# byte-identical to the pre-M9 harness for every op including these three.
#
# It ships false because the acceptance matrix does not support enforcing
# either candidate criterion (measured; full tables in the M9 stream report):
#   * the M9 margin criterion fails the in-tree references it exists to rescue
#     -- vision_block 5/5 scenarios (1.23-2.20), chunk_gla 2/5 (1.07, 1.50),
#     gpt_oss_moe 3/5 (1.48, 2.73, 48.9) -- and where it passes gpt_oss_moe it
#     passes at 0.9995 and 0.997 while a uniform +1% error on the same
#     candidate scores 1.009, i.e. there is no discrimination left at the
#     decision point;
#   * FlashAttention's published 2x criterion (``oracle_flashattn_ratio``)
#     does pass all three references (0.500 everywhere) and does reject both
#     negative controls on vision_block (1.67-7.94) and chunk_gla (11.1-127),
#     but it is BLIND on gpt_oss_moe: the SwiGLU-alpha control scores 0.500 on
#     all 5 scenarios and the x1.10 gross control scores 0.533-0.546 on 3 of 5.
#     Its threshold is 2x the baseline's own worst error, and on this fixture
#     that error reaches 2.5e3 absolute (1.2e5 tolerance bands), which is
#     larger than the entire signal a semantic bug produces.
# Flipping this to True enables the M9 criterion exactly as specified.
_ORACLE_ARM_ENFORCED = False

# e2m1 code -> value (MXFP4). Every entry has at most 2 significant bits, so the
# dequantized value ``LUT[code] * 2**(e8m0 - 127)`` is exact in fp64 (and, as it
# happens, already exact in bf16 -- the packed format loses nothing on upcast).
_FP4_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def _oracle_linear(x: Any, weight: Any, bias: Any) -> Any:
    """``F.linear`` in fp64 (weight/bias upcast from their stored dtype)."""
    import torch.nn.functional as F

    return F.linear(x, weight.double(), None if bias is None else bias.double())


def _oracle_layer_norm(x: Any, weight: Any, bias: Any, eps: float) -> Any:
    """LayerNorm over the last axis in fp64 (biased variance, as ``F.layer_norm``)."""
    import torch

    mean = x.mean(-1, keepdim=True)
    centered = x - mean
    var = centered.pow(2).mean(-1, keepdim=True)
    out = centered * torch.rsqrt(var + eps)
    if weight is not None:
        out = out * weight.double()
    if bias is not None:
        out = out + bias.double()
    return out


def _oracle_apply_rotary(x: Any, cos: Any, sin: Any) -> Any:
    """Non-interleaved (NeoX) rotary in fp64 over ``x`` = [batch, seq, heads, dim]."""
    import torch

    rotary_dim = 2 * int(cos.shape[-1])
    seqlen = int(x.shape[1])
    c = torch.cat([cos[:seqlen].double()] * 2, dim=-1)[None, :, None, :]
    s = torch.cat([sin[:seqlen].double()] * 2, dim=-1)[None, :, None, :]
    rot = x[..., :rotary_dim]
    half = rotary_dim // 2
    rotated = torch.cat((-rot[..., half:], rot[..., :half]), dim=-1)
    out = rot * c + rotated * s
    if rotary_dim < int(x.shape[-1]):
        out = torch.cat([out, x[..., rotary_dim:]], dim=-1)
    return out


def _oracle_vision_block(module: Any, sd: dict[str, Any],
                         inputs: dict[str, Any]) -> Any:
    """fp64 truth for ``vision_block`` (tasks/baseline/L3/vision_block.py).

    Pre-norm residual block: ``x + proj(attn(norm1(x)))`` then ``x + mlp(norm2(x))``,
    with non-causal full attention inside every ``cu_seqlens`` segment and the
    rotary applied to q and k exactly as ``VisionAttention.forward`` does
    (concatenate over the batch axis, one cos/sin table for both). Attention is
    walked per (segment, head) so the fp64 score matrix stays small.
    """
    import torch

    x = inputs.get("x")
    cu = inputs.get("cu_seqlens")
    cos = inputs.get("rotary_pos_emb_cos")
    sin = inputs.get("rotary_pos_emb_sin")
    if not (isinstance(x, torch.Tensor) and isinstance(cu, torch.Tensor)):
        return None
    act_name = type(getattr(module.mlp, "act_fn", None)).__name__
    if act_name == "QuickGELU":
        def activation(t):
            return t * torch.sigmoid(1.702 * t)
    elif act_name == "SiLU":
        def activation(t):
            return t * torch.sigmoid(t)
    else:
        _log(f"fp64 oracle: unsupported vision_block activation {act_name!r}")
        return None

    heads = int(module.attn.num_heads)
    head_dim = int(module.attn.head_dim)

    x64 = x.double()
    seq_len, batch, _ = x64.shape
    hidden = _oracle_layer_norm(x64, sd.get("norm1.weight"), sd.get("norm1.bias"),
                                float(module.norm1.eps))
    qkv = _oracle_linear(hidden, sd["attn.qkv.weight"], sd.get("attn.qkv.bias"))
    q_size = heads * head_dim
    q, k, v = qkv.split([q_size, q_size, q_size], dim=-1)
    q = q.view(seq_len, batch, heads, head_dim).transpose(0, 1)
    k = k.view(seq_len, batch, heads, head_dim).transpose(0, 1)
    v = v.view(seq_len, batch, heads, head_dim).transpose(0, 1)
    if isinstance(cos, torch.Tensor) and isinstance(sin, torch.Tensor):
        qk = _oracle_apply_rotary(torch.cat([q, k], dim=0), cos, sin)
        q, k = qk.chunk(2, dim=0)
    q = q.reshape(-1, heads, head_dim)
    k = k.reshape(-1, heads, head_dim)
    v = v.reshape(-1, heads, head_dim)

    scale = float(head_dim) ** -0.5
    attn = torch.zeros_like(q)
    bounds = [int(b) for b in cu.to(torch.int64).tolist()]
    for i in range(len(bounds) - 1):
        s0, s1 = bounds[i], bounds[i + 1]
        if s1 <= s0:
            continue
        for h in range(heads):
            scores = (q[s0:s1, h] @ k[s0:s1, h].transpose(0, 1)) * scale
            attn[s0:s1, h] = torch.softmax(scores, dim=-1) @ v[s0:s1, h]

    x64 = x64 + _oracle_linear(attn.reshape(seq_len, batch, -1),
                               sd["attn.proj.weight"], sd.get("attn.proj.bias"))
    hidden = _oracle_layer_norm(x64, sd.get("norm2.weight"), sd.get("norm2.bias"),
                                float(module.norm2.eps))
    hidden = activation(
        _oracle_linear(hidden, sd["mlp.fc1.weight"], sd.get("mlp.fc1.bias")))
    return x64 + _oracle_linear(hidden, sd["mlp.fc2.weight"], sd.get("mlp.fc2.bias"))


def _oracle_gla_intra_A(q_c: Any, k_c: Any, gc: Any, scale: float,
                        sub_block: int = 16) -> Any:
    """fp64 intra-chunk attention matrix (see tasks/reference/L1/chunk_gla.py).

    ``A[l, j] = scale * sum_k q[l,k] k[j,k] exp(gc[l,k] - gc[j,k])`` for j <= l.
    Blocks strictly below the diagonal are factored through the sub-block's
    first row so the tensors stay matmul-shaped; the diagonal block is masked
    before the exponential. Both exponents are <= 0 because ``gc`` is a cumsum
    of a log-space (non-positive) gate.
    """
    import torch

    N, L, H, K = q_c.shape
    A = q_c.new_zeros((N, H, L, L))
    for i0 in range(0, L, sub_block):
        i1 = min(i0 + sub_block, L)
        gn = gc[:, i0]
        qi = q_c[:, i0:i1]
        gi = gc[:, i0:i1]
        if i0 > 0:
            qg = qi * torch.exp(gi - gn[:, None]) * scale
            kg = k_c[:, :i0] * torch.exp(gn[:, None] - gc[:, :i0])
            A[:, :, i0:i1, :i0] = torch.einsum("nlhk,njhk->nhlj", qg, kg)
        kj = k_c[:, i0:i1]
        gj = gc[:, i0:i1]
        li = i1 - i0
        diff = gi[:, :, None] - gj[:, None, :]
        tri = (torch.arange(li, device=q_c.device)[:, None]
               >= torch.arange(li, device=q_c.device)[None, :])
        diff = torch.where(tri[None, :, :, None, None], diff,
                           torch.full_like(diff, float("-inf")))
        Ad = (qi[:, :, None] * kj[:, None] * torch.exp(diff)).sum(-1) * scale
        A[:, :, i0:i1, i0:i1] = Ad.permute(0, 3, 1, 2)
    return A


def _oracle_gla_run(q: Any, k: Any, v: Any, g: Any, scale: float, h: Any,
                    o_out: Any, chunk_size: int = 64) -> Any:
    """fp64 GLA recurrence over one packed slab; writes ``o_out``, returns state.

    ``S[t] = diag(exp(g[t])) S[t-1] + k[t] (x) v[t]``, ``o[t] = scale q[t]^T S[t]``
    -- evaluated chunk-wise so the work is matmul-shaped. In fp64 the chunk size
    is a performance knob only: unlike the production kernel there is no rounding
    at the chunk boundary, so the result is the exact recurrence to ~1e-16.
    Slices are upcast per chunk to keep the fp64 footprint bounded.
    """
    import torch

    T = int(q.shape[1])
    for s in range(0, T, chunk_size):
        e = min(s + chunk_size, T)
        q_c = q[:, s:e].double()
        k_c = k[:, s:e].double()
        v_c = v[:, s:e].double()
        gc = g[:, s:e].double().cumsum(1)
        o_c = torch.einsum("nlhk,nhkv->nlhv", q_c * torch.exp(gc), h) * scale
        o_c = o_c + torch.einsum(
            "nhlj,njhv->nlhv", _oracle_gla_intra_A(q_c, k_c, gc, scale), v_c)
        o_out[:, s:e] = o_c
        g_last = gc[:, -1]
        kg = k_c * torch.exp(g_last[:, None] - gc)
        h = h * torch.exp(g_last)[..., None] + torch.einsum(
            "nlhk,nlhv->nhkv", kg, v_c)
        del q_c, k_c, v_c, gc, o_c, kg
    return h


def _oracle_chunk_gla(module: Any, sd: dict[str, Any],
                      inputs: dict[str, Any]) -> Any:
    """fp64 truth for ``chunk_gla`` (parameter-free: everything is a forward arg)."""
    import torch

    q, k, v, g = (inputs.get(n) for n in ("q", "k", "v", "g"))
    if not all(isinstance(t, torch.Tensor) for t in (q, k, v, g)):
        return None
    scale = inputs.get("scale")
    scale = float(q.shape[-1]) ** -0.5 if scale is None else float(scale)
    initial_state = inputs.get("initial_state")
    output_final_state = bool(inputs.get("output_final_state", False))
    cu_seqlens = inputs.get("cu_seqlens")

    B, T, H, K = q.shape
    V = int(v.shape[-1])
    o = torch.empty((B, T, H, V), dtype=torch.float64, device=q.device)

    if not isinstance(cu_seqlens, torch.Tensor):
        h = (initial_state.double().clone() if isinstance(initial_state, torch.Tensor)
             else torch.zeros((B, H, K, V), dtype=torch.float64, device=q.device))
        h = _oracle_gla_run(q, k, v, g, scale, h, o)
        return o, (h if output_final_state else None)

    num_seqs = int(cu_seqlens.numel()) - 1
    ht = torch.zeros((num_seqs, H, K, V), dtype=torch.float64, device=q.device)
    bounds = [int(b) for b in cu_seqlens.to(torch.int64).tolist()]
    for n in range(num_seqs):
        s0, s1 = bounds[n], bounds[n + 1]
        if s1 <= s0:
            continue
        h = (initial_state[n:n + 1].double().clone()
             if isinstance(initial_state, torch.Tensor)
             else torch.zeros((1, H, K, V), dtype=torch.float64, device=q.device))
        h = _oracle_gla_run(q[:, s0:s1], k[:, s0:s1], v[:, s0:s1], g[:, s0:s1],
                            scale, h, o[:, s0:s1])
        ht[n] = h[0]
    return o, (ht if output_final_state else None)


def _oracle_dequant_mxfp4(blocks: Any, scales: Any) -> Any:
    """Unpack one expert's MXFP4 payload + E8M0 block scales to a dense fp64 tensor.

    ``blocks``: (..., cols // 2) uint8, low nibble = even element (checkpoint
    convention, verified bitwise against ``triton_kernels...upcast_from_mxfp``
    by tasks/reference/L1/mxfp4_moe.py). ``scales``: (..., cols // 32) uint8
    exponents, value = ``2 ** (byte - 127)``. Both factors are exact in fp64,
    so this loses nothing relative to what the kernel reads.
    """
    import torch

    lut = torch.tensor(_FP4_E2M1_VALUES, dtype=torch.float64, device=blocks.device)
    low = (blocks & 0x0F).long()
    high = ((blocks >> 4) & 0x0F).long()
    values = lut[torch.stack([low, high], dim=-1).reshape(*blocks.shape[:-1], -1)]
    scale = torch.exp2(scales.double() - 127.0)
    values = values.view(*values.shape[:-1], int(scales.shape[-1]), 32)
    values = values * scale.unsqueeze(-1)
    return values.reshape(*values.shape[:-2], -1)


def _oracle_gpt_oss_moe(module: Any, sd: dict[str, Any],
                        inputs: dict[str, Any]) -> Any:
    """fp64 truth for ``gpt_oss_moe`` (tasks/baseline/L2/gpt_oss_moe.py).

    Router GEMM -> top-k on the raw logits -> softmax over the selected logits
    (``renormalize=True``) -> per-expert OAI SwiGLU
    (``s = gate*sigmoid(alpha*gate)``, ``out = s*up + s``, alpha 1.702, gate
    clamped above at 7 and up clamped to +-7 -- ``triton_kernels/swiglu_details/
    _swiglu.py::compute_swiglu``, gate/up interleaved even/odd as ``tl.split``
    reads them) -> gamma-weighted sum over the slots. Tokens are grouped by
    expert so each expert costs two fp64 GEMMs instead of one per token; the
    packed weights are dequantized one expert at a time.
    """
    import torch

    x = inputs.get("hidden_states")
    if not isinstance(x, torch.Tensor):
        return None
    for key in ("router.weight", "router.bias", "w13_weight", "w13_weight_scale",
                "w13_bias", "w2_weight", "w2_weight_scale", "w2_bias"):
        if key not in sd:
            _log(f"fp64 oracle: gpt_oss_moe state_dict is missing {key!r}")
            return None

    top_k = int(module.top_k)
    xf = x.reshape(-1, int(module.hidden_size)).double()
    logits = _oracle_linear(xf, sd["router.weight"], sd["router.bias"])
    top_vals, top_ids = torch.topk(logits, top_k, dim=-1)
    gammas = torch.softmax(top_vals, dim=-1)

    w13, w13_scale = sd["w13_weight"], sd["w13_weight_scale"]
    w2, w2_scale = sd["w2_weight"], sd["w2_weight_scale"]
    b13, b2 = sd["w13_bias"].double(), sd["w2_bias"].double()

    out = torch.zeros_like(xf)
    for expert in sorted(int(e) for e in torch.unique(top_ids).tolist()):
        rows, slots = (top_ids == expert).nonzero(as_tuple=True)
        if rows.numel() == 0:
            continue
        gate_up = _oracle_linear(
            xf[rows], _oracle_dequant_mxfp4(w13[expert], w13_scale[expert]),
            b13[expert])
        gate = gate_up[:, 0::2].clamp(max=7.0)
        up = gate_up[:, 1::2].clamp(min=-7.0, max=7.0)
        s = gate / (1.0 + torch.exp(-1.702 * gate))
        y = _oracle_linear(
            s * up + s, _oracle_dequant_mxfp4(w2[expert], w2_scale[expert]),
            b2[expert])
        out.index_add_(0, rows, y * gammas[rows, slots].unsqueeze(-1))
        del gate_up, gate, up, s, y
    return out.reshape(x.shape)


_FP64_ORACLE_BUILDERS = {
    "gpt_oss_moe": _oracle_gpt_oss_moe,
    "chunk_gla": _oracle_chunk_gla,
    "vision_block": _oracle_vision_block,
}


_ORACLE_FAIL = {"margin_ratio": float("inf"), "baseline_ratio": float("inf"),
                "flashattn_ratio": float("inf"), "max_abs_base_err": float("inf"),
                "max_abs_cand_err": float("inf"), "structural": True}


def _oracle_metrics(baseline_out: Any, candidate_out: Any, truth: Any,
                    atol: float, rtol: float) -> dict[str, Any]:
    """Both arm-(b) criteria against the fp64 truth. Fails closed on surprises.

    ``margin_ratio`` (the M9 criterion) =
        max_e |cand - truth| / (|base - truth| + atol + rtol|truth|)
    -- elementwise; <= 1.0 means "nowhere farther from the truth than the
    baseline is, plus one tolerance band".

    ``flashattn_ratio`` (FlashAttention's published correctness test, which
    compares GLOBAL maxima rather than elements) =
        max_e |cand - truth| / (2 * max_e |base - truth|)
    -- <= 1.0 means "the candidate's worst error is at most 2x the worst error
    of the in-tree implementation". Reported for comparison; see the note in
    the module docstring for why neither number is currently enforced.

    ``baseline_ratio`` = max_e |base - truth| / (atol + rtol|truth|) is the
    audit number: how far the production baseline itself sits from the truth,
    in tolerance bands. It is what would explode if the oracle's semantics were
    wrong, and it is what makes ``flashattn_ratio`` blind on the MoE.
    """
    import torch

    if isinstance(truth, torch.Tensor):
        if not (isinstance(baseline_out, torch.Tensor)
                and isinstance(candidate_out, torch.Tensor)):
            return dict(_ORACLE_FAIL)
        if baseline_out.shape != truth.shape or candidate_out.shape != truth.shape:
            return dict(_ORACLE_FAIL)
        t = truth.double()
        b = baseline_out.double()
        c = candidate_out.double()
        if not (bool(torch.isfinite(t).all()) and bool(torch.isfinite(b).all())
                and bool(torch.isfinite(c).all())):
            return dict(_ORACLE_FAIL)
        band = atol + rtol * t.abs()
        base_err = (b - t).abs()
        cand_err = (c - t).abs()
        return {
            "margin_ratio": float((cand_err / (base_err + band)).max().item()),
            "baseline_ratio": float((base_err / band).max().item()),
            "max_abs_base_err": float(base_err.max().item()),
            "max_abs_cand_err": float(cand_err.max().item()),
            "structural": False,
        }

    if isinstance(truth, (tuple, list)):
        if not (isinstance(baseline_out, (tuple, list))
                and isinstance(candidate_out, (tuple, list))):
            return dict(_ORACLE_FAIL)
        if not len(truth) == len(baseline_out) == len(candidate_out):
            return dict(_ORACLE_FAIL)
        merged = {"margin_ratio": 0.0, "baseline_ratio": 0.0,
                  "max_abs_base_err": 0.0, "max_abs_cand_err": 0.0,
                  "structural": False}
        for t, b, c in zip(truth, baseline_out, candidate_out):
            part = _oracle_metrics(b, c, t, atol, rtol)
            for key in ("margin_ratio", "baseline_ratio", "max_abs_base_err",
                        "max_abs_cand_err"):
                merged[key] = max(merged[key], part[key])
            merged["structural"] = merged["structural"] or part["structural"]
        return merged

    if truth is None:
        # Nothing to grade at this slot (e.g. ``output_final_state=False``);
        # a tensor on either side is a structure mismatch, not a free pass.
        if isinstance(baseline_out, torch.Tensor) or isinstance(candidate_out, torch.Tensor):
            return dict(_ORACLE_FAIL)
        return {"margin_ratio": 0.0, "baseline_ratio": 0.0,
                "max_abs_base_err": 0.0, "max_abs_cand_err": 0.0,
                "structural": False}

    return dict(_ORACLE_FAIL)


def _apply_fp64_oracle_arm(op: str, entry: dict[str, Any], correct: bool,
                           inputs_correct: bool, module: Any,
                           state_dict: dict[str, Any], inputs: dict[str, Any],
                           baseline_out: Any, candidate_out: Any,
                           standard_ratio: float, runner_mod: Any) -> bool:
    """Evaluate arm (b) when arm (a) failed; annotate ``entry``; return the verdict.

    With ``_ORACLE_ARM_ENFORCED`` false (the measured default -- see the module
    docstring) the numbers are reported and the verdict is exactly the standard
    arm's, so this function is observational for every op.
    """
    import torch

    entry["standard_arm_ratio"] = _num(standard_ratio)
    if correct:
        entry["oracle_arm"] = "not_evaluated:standard_arm_passed"
        return True

    truth = None
    try:
        truth = _FP64_ORACLE_BUILDERS[op](module, state_dict, inputs)
    except Exception as exc:  # an unbuildable oracle must not change a verdict
        _log(f"fp64 oracle unavailable for {op}: {type(exc).__name__}: {exc}")
    if truth is None:
        entry["oracle_arm"] = "unavailable"
        return correct

    m = _oracle_metrics(baseline_out, candidate_out, truth,
                        runner_mod._LOW_PRECISION_ATOL,
                        runner_mod._LOW_PRECISION_RTOL)
    del truth
    torch.cuda.empty_cache()

    base_err = m["max_abs_base_err"]
    fa_ratio = (m["max_abs_cand_err"] / (2.0 * base_err)) if base_err > 0 \
        else float("inf")
    ok = (not m["structural"]) and m["margin_ratio"] <= 1.0 and inputs_correct
    entry["oracle_margin_ratio"] = _num(m["margin_ratio"])
    entry["oracle_baseline_ratio"] = _num(m["baseline_ratio"])
    entry["oracle_flashattn_ratio"] = _num(fa_ratio)
    entry["oracle_arm_enforced"] = bool(_ORACLE_ARM_ENFORCED)
    entry["oracle_arm"] = ("would_pass" if ok else "would_fail") if not \
        _ORACLE_ARM_ENFORCED else ("passed" if ok else "failed")
    _log(f"fp64 oracle arm {entry['oracle_arm']}: margin ratio "
         f"{m['margin_ratio']:.4g}, FlashAttention-style ratio {fa_ratio:.4g} "
         f"(standard arm {standard_ratio:.4g}; the baseline itself is "
         f"{m['baseline_ratio']:.4g} tolerance bands / {base_err:.4g} absolute "
         f"from the truth)")
    return (correct or ok) if _ORACLE_ARM_ENFORCED else correct


def _abs_rel_errors(baseline_out: Any, candidate_out: Any) -> tuple[float, float]:
    """max |b-c| and max |b-c|/|b| over a matching output tree.

    Reported only; the pass/fail decision belongs entirely to the runner's
    tolerance-normalized ``_compare_outputs``.
    """
    import torch

    if isinstance(baseline_out, torch.Tensor) and isinstance(candidate_out, torch.Tensor):
        if baseline_out.shape != candidate_out.shape:
            return float("inf"), float("inf")
        b = baseline_out.float()
        c = candidate_out.float()
        diff = (b - c).abs()
        denom = b.abs()
        rel = torch.where(
            denom > 0, diff / denom.clamp_min(torch.finfo(torch.float32).tiny),
            torch.where(diff > 0, torch.full_like(diff, float("inf")),
                        torch.zeros_like(diff)),
        )
        return diff.max().item(), rel.max().item()

    if isinstance(baseline_out, (tuple, list)) and isinstance(candidate_out, (tuple, list)):
        if len(baseline_out) != len(candidate_out):
            return float("inf"), float("inf")
        pairs = zip(baseline_out, candidate_out)
    elif isinstance(baseline_out, dict) and isinstance(candidate_out, dict):
        if set(baseline_out) != set(candidate_out):
            return float("inf"), float("inf")
        pairs = ((baseline_out[k], candidate_out[k]) for k in sorted(baseline_out))
    else:
        return 0.0, 0.0

    max_abs = max_rel = 0.0
    for b, c in pairs:
        a, r = _abs_rel_errors(b, c)
        max_abs = max(max_abs, a)
        max_rel = max(max_rel, r)
    return max_abs, max_rel


# --- the benchmark ------------------------------------------------------------

def run(op: str, candidate_path: str | None, scenario_filters: list[str] | None,
        baseline_identity: bool) -> dict[str, Any]:
    import torch

    from fastkernels.bench.kernels import runner as R
    from fastkernels.bench.kernels.scenario_registry import InputRegistry

    _log(f"runner.__file__ = {R.__file__}")
    _log(f"scenario_registry -> {InputRegistry.__module__}")
    _log(f"tolerances (from runner constants): fp32 atol={R._FP32_ATOL} rtol={R._FP32_RTOL} "
         f"| low-precision atol={R._LOW_PRECISION_ATOL} rtol={R._LOW_PRECISION_RTOL} "
         f"| fp8 atol={R._FP8_ATOL} rtol={R._FP8_RTOL}")

    if not torch.cuda.is_available():
        raise InfraError("no CUDA device visible")
    _log(f"device: {torch.cuda.get_device_name(0)}")

    target = _resolve_target(op)

    if baseline_identity:
        user_impl = target.target_cls
        solution = "baseline_identity"
    else:
        user_impl = _load_candidate_from_path(candidate_path, target.target_cls)
        solution = os.path.abspath(candidate_path)
        _log(f"candidate class: {user_impl.__name__} from {candidate_path}")

    registry = InputRegistry()
    scenarios = registry.scenarios(op)
    if not scenarios:
        raise InfraError(f"no scenarios registered for operator {op!r}")
    if scenario_filters:
        names = {s.name for s in scenarios}
        selected, unmatched = [], []
        for pat in scenario_filters:
            hits = [s for s in scenarios if s.name == pat] or \
                   [s for s in scenarios if pat in s.name]
            if not hits:
                unmatched.append(pat)
            selected.extend(hits)
        if unmatched:
            raise InfraError(
                f"--scenarios patterns matched nothing for {op!r}: {unmatched}. "
                f"{len(names)} scenarios available."
            )
        seen: set[str] = set()
        deduped = []
        for s in selected:
            if s.name not in seen:
                seen.add(s.name)
                deduped.append(s)
        scenarios = deduped
    _log(f"operator {op!r}: {len(scenarios)} scenario(s) selected")

    definition = f"kb_{op}"
    results: dict[str, Any] = {}

    for scenario in scenarios:
        entry: dict[str, Any] = {
            "status": STATUS_RUNTIME_ERROR,
            "solution": solution,
            "axes": _scenario_axes(scenario),
        }
        try:
            _seed_scenario(op, scenario.name)
            inputs = registry.get_inputs(op, scenario.name, device="cuda")
            inputs = _prepare_inputs_for_target(
                op, inputs, "cuda", scenario.init_args)
            input_dtype = R._first_floating_dtype(inputs)

            baseline_mod = _instantiate_module(
                target.target_cls, scenario.init_args, "cuda",
                dtype=input_dtype, inputs=inputs)
            candidate_mod = _instantiate_module(
                user_impl, scenario.init_args, "cuda",
                dtype=input_dtype, inputs=inputs)

            # --- uninitialised baseline parameters (torch.empty) ---
            # Runs BEFORE the perturbation: a NaN parameter survives
            # ``NaN * scale + shift`` and makes the scenario unmeasurable.
            repaired = _repair_degenerate_parameters(baseline_mod, op)
            if repaired:
                _log(f"{scenario.name}: re-initialised uninitialised "
                     f"parameter(s) {repaired}")
            if op in ("yolov10_head", "yolov10"):
                _boost_yolo_cls_bias(baseline_mod, scenario.name)

            # --- strict weight transfer (tightens runner.py:496-500) ---
            # Perturb the baseline's floating-point parameters (deterministic,
            # seeded) BEFORE the transfer. Several kb baselines initialise
            # parameters to identity values (rms_norm: weight = ones), which
            # makes "candidate forgot to apply the weight" invisible to the
            # comparison. Both modules see the SAME perturbed values, so
            # correct candidates are unaffected; degenerate-parameter blind
            # spots are closed. Measured before this fix: a no-weight-multiply
            # candidate PASSED every scenario with max_abs_error 0.0.
            _pgen = torch.Generator(device="cpu").manual_seed(0x5EED)
            with torch.no_grad():
                for _pname, _p in baseline_mod.named_parameters():
                    # float8 exempt: mul_/add_ are unimplemented for float8 on
                    # CUDA, and quantized expert weights are not identity-valued
                    # (the blind spot this perturbation closes); both sides
                    # still share the identical copied fp8 tensors.
                    if _p.is_floating_point() and "float8" not in str(_p.dtype):
                        _scale = torch.empty(_p.shape, dtype=torch.float32)
                        _scale.uniform_(0.75, 1.25, generator=_pgen)
                        _shift = torch.empty(_p.shape, dtype=torch.float32)
                        _shift.uniform_(-0.05, 0.05, generator=_pgen)
                        _p.mul_(_scale.to(device=_p.device, dtype=_p.dtype))
                        _p.add_(_shift.to(device=_p.device, dtype=_p.dtype))
            baseline_sd = baseline_mod.state_dict()
            try:
                incompatible = candidate_mod.load_state_dict(baseline_sd, strict=False)
            except Exception as exc:
                entry["error_log"] = (
                    "weight_transfer_failed: load_state_dict raised "
                    f"{R._short_exception(exc)}. baseline keys="
                    f"{sorted(baseline_sd)} candidate keys="
                    f"{sorted(candidate_mod.state_dict())}"
                )
                results[scenario.name] = entry
                continue
            missing = list(getattr(incompatible, "missing_keys", []))
            unexpected = list(getattr(incompatible, "unexpected_keys", []))
            if missing or unexpected:
                entry["error_log"] = (
                    "weight_transfer_incomplete: load_state_dict(strict=False) "
                    f"reported missing_keys={missing} unexpected_keys={unexpected}. "
                    "The candidate would have run on its own initialisation, so "
                    "any correctness result would be meaningless."
                )
                results[scenario.name] = entry
                continue

            # --- engine-parity FP8 post-processing (after the transfer, so
            # the transfer itself stays in checkpoint space; see the helper's
            # docstring). Both sides transform identical transferred values.
            fp8_groups = _postprocess_fp8_module_weights(baseline_mod)
            if fp8_groups:
                _postprocess_fp8_module_weights(candidate_mod)
                _log(f"{scenario.name}: applied engine FP8 weight "
                     f"post-processing to {fp8_groups} weight group(s) per side")

            # --- correctness: the runner's own comparison, unmodified ---
            baseline_check_inputs = R._clone_inputs(inputs)
            candidate_check_inputs = R._clone_inputs(inputs)
            baseline_out = R._run_forward_once(baseline_mod, baseline_check_inputs)

            # --- non-finite baseline output = fixture fault, not a verdict ---
            # runner.py:302-303 turns "either side is non-finite" into a
            # numerical failure. When it is the *baseline* that produced the
            # NaN/inf, nothing about the candidate has been measured: the
            # comparison is inf-vs-inf regardless of what the candidate did.
            # Confirmed on flux_attention, where bitwise-identical outputs
            # reported max_error_ratio=inf. Report the fixture instead. The
            # candidate is NOT given the same excuse -- a candidate-only
            # non-finite output still fails the comparison below.
            if _has_nonfinite(baseline_out):
                entry["status"] = STATUS_RUNTIME_ERROR
                entry["error_log"] = (
                    "baseline_output_nonfinite: degenerate fixture (uninitialized parameters or out-of-distribution inputs)"
                )
                results[scenario.name] = entry
                continue

            candidate_out = R._run_forward_once(candidate_mod, candidate_check_inputs)

            if op == "moe_align":
                # Difference 8 (see module docstring): order within an expert
                # group is not contractual; canonicalize BOTH sides with the
                # same key before the runner's unmodified comparison.
                baseline_out = _canonicalize_moe_align_output(
                    baseline_out, baseline_check_inputs)
                candidate_out = _canonicalize_moe_align_output(
                    candidate_out, candidate_check_inputs)

            if op == "chunk_gla":
                # Difference 9: the kernel assembles its fp32 final state
                # from bf16-rounded decayed keys whose decay uses Triton's
                # ex2.approx; measured (E3 stream report, 2026-07-26), the
                # BASELINE itself sits ~1.1e-4 from the fp64-exact value at
                # elements where the fp32 tolerance band is ~1.1e-5 -- so
                # fp32-dtype grading demands bit-reproduction of ex2.approx
                # (Triton-only, forbidden to seeds), not semantic
                # correctness. Grade this op at the computation precision:
                # low-precision tolerances on the raw values (see
                # _low_precision_tolerances for why not a bf16 cast).
                with _low_precision_tolerances(R):
                    output_check = R._compare_outputs(baseline_out, candidate_out)
                    input_check = R._compare_outputs(
                        baseline_check_inputs, candidate_check_inputs)
            else:
                output_check = R._compare_outputs(baseline_out, candidate_out)
                input_check = R._compare_outputs(
                    baseline_check_inputs, candidate_check_inputs)
            correct, max_error_ratio, mean_diff = R._merge_correctness(
                output_check, input_check)

            if op in _FP64_ORACLE_OPS:
                # Difference 10: second (fp64-oracle) pass arm, consulted only
                # now that the standard arm has produced its verdict. ``inputs``
                # is the pristine prepared fixture -- both modules ran on clones
                # of it -- and ``baseline_sd`` was snapshotted before either
                # forward, so the truth is independent of the candidate.
                correct = _apply_fp64_oracle_arm(
                    op, entry, correct, input_check[0], baseline_mod,
                    baseline_sd, inputs, baseline_out, candidate_out,
                    max_error_ratio, R)

            out_abs, out_rel = _abs_rel_errors(baseline_out, candidate_out)
            in_abs, in_rel = _abs_rel_errors(baseline_check_inputs, candidate_check_inputs)
            max_abs, max_rel = max(out_abs, in_abs), max(out_rel, in_rel)

            # --- timing: the runner's median-of-N ---
            # Identity mode needs correctness, not benchmark-grade timing:
            # baseline-vs-itself latency is ~1.0x by construction, and the
            # full median-of-100 protocol costs ~220 forward executions per
            # scenario (30 min/run on generative-loop ops like
            # oasis_rollout). Candidate grading keeps the full protocol.
            n_warm, n_runs = (1, 3) if baseline_identity else (NUM_WARMUP, NUM_RUNS)
            _, baseline_ms = R._time_forward(
                baseline_mod, R._clone_inputs(inputs), n_warm, n_runs)
            _, candidate_ms = R._time_forward(
                candidate_mod, R._clone_inputs(inputs), n_warm, n_runs)

            entry["status"] = STATUS_PASSED if correct else STATUS_INCORRECT_NUMERICAL
            entry["latency_ms"] = _num(candidate_ms)
            entry["reference_latency_ms"] = _num(baseline_ms)
            entry["speedup_factor"] = _num(
                baseline_ms / candidate_ms if candidate_ms > 0 else float("inf"))
            entry["max_abs_error"] = _num(max_abs)
            entry["max_rel_error"] = _num(max_rel)
            if not correct:
                atol, rtol = R._tolerances_for_dtype(
                    input_dtype or torch.float32)
                entry["error_log"] = (
                    "output_mismatch: max_error_ratio="
                    f"{max_error_ratio:.6g} > 1.0 (tolerance = atol {atol} + rtol "
                    f"{rtol} * |baseline|); mean_abs_diff={mean_diff:.6g}, "
                    f"max_abs_error={max_abs:.6g}, max_rel_error={max_rel:.6g}"
                )

            del baseline_mod, candidate_mod, baseline_out, candidate_out
            del baseline_check_inputs, candidate_check_inputs, inputs

        except Exception as exc:
            entry["status"] = STATUS_RUNTIME_ERROR
            entry["error_log"] = (
                f"scenario raised {type(exc).__name__}: {exc}\n"
                + traceback.format_exc()
            )
        results[scenario.name] = entry

    tally: dict[str, int] = {}
    for e in results.values():
        tally[e["status"]] = tally.get(e["status"], 0) + 1
    _log(f"status tally: {json.dumps(tally, sort_keys=True)}")
    _audit_loaded_trees()
    return {definition: results}


def _audit_loaded_trees() -> None:
    """Prove which on-disk tree the fastkernels modules actually came from.

    A second checkout can be installed (editable) in the same interpreter under a
    different distribution name, so 'the import worked' is not evidence that the
    intended tree was used. Count every loaded module by tree root instead.
    """
    root = os.path.abspath(str(sys.modules["fastkernels"].KB_ROOT))
    from_tree = [n for n, m in list(sys.modules.items())
                 if getattr(m, "__file__", None)
                 and os.path.abspath(m.__file__).startswith(root + os.sep)]
    _log(f"modules loaded from {root}: {len(from_tree)}")
    foreign = sorted(
        f"{n} <- {m.__file__}" for n, m in list(sys.modules.items())
        if (n == "fastkernels" or n.startswith("fastkernels."))
        and getattr(m, "__file__", None)
        and not os.path.abspath(m.__file__).startswith(root)
    )
    if foreign:
        _log(f"WARNING: fastkernels.* modules resolved outside {root}: {foreign}")
    else:
        _log(f"all fastkernels.* modules resolve under {root}: OK")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agent_entrypoint.py",
        description="Run one fastkernels operator's scenarios and emit the "
                    "AKO4X normalized result dict on stdout.",
    )
    parser.add_argument("--op", required=True, help="operator name, e.g. rms_norm")
    parser.add_argument("--candidate", help="path to the candidate kernel .py")
    parser.add_argument("--scenarios", help="comma-separated scenario names/substrings")
    parser.add_argument("--baseline-identity", action="store_true",
                        help="self-test: candidate := a second baseline instance")
    args = parser.parse_args(argv)

    # The GPT-OSS MXFP4 family runs triton_kernels' persistent matmul_ogs,
    # whose ragged-TMA load computes addresses outside the activation buffer
    # (compute-sanitizer: "Warp illegal address ... _p_matmul_ogs_NNT_
    # bf16xbf16xmxfp4 ... ragged_tma.py:74"; reproduced on BOTH mxfp4_moe and
    # gpt_oss_decoder, every launch). The stray read is harmless when it lands
    # on mapped pages -- vLLM production never faults because its VA space is
    # dense -- but this small per-op process has sparse VA, and the read
    # faulted reproducibly on gpt_oss_decoder's tokens-1 -> tokens-4 scenario
    # transition (allocator-layout dependent; tokens-4 alone, or the reversed
    # order, ran clean). The non-persistent kernel is not an alternative on
    # SM100 ("Must use persistent kernel and be TMA-compliant for native
    # MXFP4"). Expandable segments give the allocator one dense mapped arena,
    # which removes the faulting case (measured: the failing pair goes
    # 2/2 PASSED). Scoped to the affected ops so every other op keeps the
    # allocator (and therefore its torch.empty garbage distribution, which
    # the repair step's degeneracy DECISIONS depend on) byte-for-byte
    # unchanged.
    _MXFP4_MATMUL_OGS_OPS = ("mxfp4_moe", "gpt_oss_moe", "gpt_oss_decoder",
                             "gpt_oss")
    if args.op in _MXFP4_MATMUL_OGS_OPS:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                              "expandable_segments:True")

    try:
        if args.baseline_identity and args.candidate:
            raise InfraError("--baseline-identity and --candidate are mutually "
                             "exclusive; pass exactly one")
        if not args.baseline_identity and not args.candidate:
            raise InfraError("pass --candidate <path> or --baseline-identity")

        _ensure_ninja_on_path()
        root = _bootstrap_fastkernels()
        _log(f"fastkernels tree: {root}")
        _log(f"fastkernels.__file__ = {sys.modules['fastkernels'].__file__}")

        filters = [s for s in (args.scenarios or "").split(",") if s.strip()] or None
        payload = run(args.op, args.candidate, filters, args.baseline_identity)
    except InfraError as exc:
        _log(f"INFRASTRUCTURE ERROR: {exc}")
        return 2
    except Exception as exc:  # unexpected -> still an infrastructure error
        _log(f"INFRASTRUCTURE ERROR: {type(exc).__name__}: {exc}")
        traceback.print_exc(file=sys.stderr)
        return 2

    _emit_json(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
