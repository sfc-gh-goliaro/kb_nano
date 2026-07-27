"""Synthesize InputRegistry scenarios for targets the tracer never visited.

``shape_registry.yaml`` only covers operators exercised while tracing the nine
models in ``benchmark_scenarios/small/config.yaml``.  Architectures that were
never traced end up with zero scenarios, and Tier 1 reports "No scenarios found"
-- indistinguishable from a broken target, and impossible to benchmark at all.

For a target whose ``forward`` takes plain activations and whose constructor
takes only a HuggingFace ``config``, the scenario is fully determined by the
config plus a choice of token count: the activation is ``[tokens, hidden_size]``
(or ``[tokens]`` for token ids).  Those can be generated without running the
model, which is what this module does.

Scenarios produced here are **synthetic**, not captured from a real workload.
They exercise the same shapes production would, but they do not reproduce the
data-dependent behaviour (MoE load skew, hot-expert identity) that captured
tensors preserve, so results derived from them should be reported separately
from the traced set.
"""

from __future__ import annotations

from typing import Any

from fastkernels.bench.kernels.init_resolver import config_candidates

# Token counts mirroring the traced scenarios' batch regimes.
DEFAULT_TOKENS = (1, 4, 32, 256)

# Argument name -> how to shape it, given (tokens, hidden).
# Encoder blocks (BERT/XLM-RoBERTa family) take batched
# ``[batch, seq, hidden]`` activations and unpack three dims, so a flat
# ``[tokens, hidden]`` activation fails with "not enough values to unpack
# (expected 3)".  Decoder-style blocks take the flat form.  ``rank`` selects
# which convention a target uses.
_TENSOR_SPECS: dict[str, Any] = {
    "hidden_states": lambda tok, hid, rank: (
        {"shape": [1, tok, hid], "dtype": "bfloat16"} if rank == 3
        else {"shape": [tok, hid], "dtype": "bfloat16"}
    ),
    "x": lambda tok, hid, rank: (
        {"shape": [1, tok, hid], "dtype": "bfloat16"} if rank == 3
        else {"shape": [tok, hid], "dtype": "bfloat16"}
    ),
    "inputs_embeds": lambda tok, hid, rank: (
        {"shape": [1, tok, hid], "dtype": "bfloat16"} if rank == 3
        else {"shape": [tok, hid], "dtype": "bfloat16"}
    ),
    "input_ids": lambda tok, hid, rank: (
        {"shape": [1, tok], "dtype": "int64"} if rank == 3
        else {"shape": [tok], "dtype": "int64"}
    ),
    "positions": lambda tok, hid, rank: (
        {"shape": [1, tok], "dtype": "int64"} if rank == 3
        else {"shape": [tok], "dtype": "int64"}
    ),
    "residual": lambda tok, hid, rank: None,
}

# Families whose blocks consume batched 3-D activations.
_RANK3_HINTS = ("roberta", "bert", "encoder", "bge")


def _input_rank(target: Any) -> int:
    name = f"{target.name} {target.target_cls.__name__}".lower()
    return 3 if any(h in name for h in _RANK3_HINTS) else 2


def _hidden_size(models: tuple[str, ...]) -> int | None:
    for cfg in config_candidates(models):
        for attr in ("hidden_size", "d_model", "embed_dim"):
            val = getattr(cfg, attr, None)
            if isinstance(val, int) and val > 0:
                return val
    return None


def synthesizable(target: Any) -> bool:
    """True if every required forward argument has a known shape recipe."""
    import inspect

    try:
        sig = inspect.signature(target.target_cls.forward)
    except (TypeError, ValueError):
        return False
    required = [
        name for name, p in sig.parameters.items()
        if name != "self"
        and p.default is inspect.Parameter.empty
        and p.kind not in (
            inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD,
        )
    ]
    if not required:
        return False
    return all(name in _TENSOR_SPECS for name in required)


def synthesize(target: Any, tokens: tuple[int, ...] = DEFAULT_TOKENS) -> dict:
    """Build a ``shape_registry``-shaped entry for ``target``.

    Returns ``{"scenarios": [...]}``, or ``{}`` when the target's forward takes
    an argument this module cannot shape (a module, a list of feature maps).
    """
    import inspect

    if not synthesizable(target):
        return {}
    hidden = _hidden_size(tuple(target.models or ()))
    if not hidden:
        return {}

    sig = inspect.signature(target.target_cls.forward)
    names = [
        name for name, p in sig.parameters.items()
        if name != "self"
        and p.default is inspect.Parameter.empty
        and p.kind not in (
            inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD,
        )
    ]

    rank = _input_rank(target)
    scenarios = []
    for tok in tokens:
        inputs: dict[str, Any] = {}
        for name in names:
            spec = _TENSOR_SPECS[name](tok, hidden, rank)
            inputs[name] = spec
        scenarios.append({
            "name": f"tokens-{tok}/synthetic",
            "init_args": {"training": False},
            "inputs": inputs,
        })
    return {"scenarios": scenarios}


def synthesize_for_models(model_names: set[str]) -> dict[str, dict]:
    """Entries for every zero-scenario L3/L4 target of the given models."""
    from fastkernels.bench.kernels.scenario_registry import InputRegistry
    from fastkernels.infra.kernel_swapper import discover_targets

    reg = InputRegistry()
    out: dict[str, dict] = {}
    for target in discover_targets():
        if target.level not in (3, 4):
            continue
        if not (set(target.models or ()) & model_names):
            continue
        try:
            if reg.scenarios(target.name):
                continue  # already covered by the traced registry
        except Exception:
            pass
        entry = synthesize(target)
        if entry:
            out[target.name] = entry
    return out
