"""Recover constructor arguments the input tracer could not serialize.

``shape_registry.yaml`` records a scenario's ``init_args`` as captured by the
tracer, but the tracer only sees arguments that were passed **by keyword** and
that are YAML-serializable.  Two classes of argument are therefore lost:

* HuggingFace ``config`` objects (and other ``nn.Module`` arguments), which have
  no YAML form -- e.g. ``llama_decoder`` records only ``{"training": false}``.
* Arguments the caller passed **positionally** -- e.g. ``oasis_block`` loses
  ``hidden_size`` / ``num_heads`` while ``oasis_dit``, whose caller uses
  keywords, keeps all of its scalars.

Without recovery, ``_instantiate_module`` cannot build 52 of the 106 targets
that have registry scenarios, including nearly every L3 block and L4 pipeline.
The Tier-1 harness then reports a failure for *every* scenario -- even in
``baseline_identity`` mode, where the baseline is compared against itself.

This module reconstructs those arguments at benchmark time:

1. ``config`` is loaded with ``AutoConfig.from_pretrained`` for the HF
   checkpoint the target's L4 pipeline tracks (config.json only -- no weights),
   with sub-configs (``text_config`` / ``vision_config``) offered as fallbacks
   because L3 blocks of multimodal models take the text sub-config.
2. ``layer_idx``-style indices default to 0.
3. Plain scalars are read off the resolved config through an alias table, and
   fall back to the trailing dimension of the scenario's own input tensors.

Arguments that remain unresolvable (a required ``nn.Module``, e.g.
``oasis_block``'s ``spatial_rotary_emb``) are left out, so instantiation fails
with the original, informative ``TypeError`` rather than a fabricated value.
"""

from __future__ import annotations

import functools
import inspect
import os
from pathlib import Path
from typing import Any

import torch

_HERE = Path(__file__).resolve()
KB_ROOT = _HERE.parents[2]

# L4 module name -> HF checkpoint the pipeline tracks.  Seeded from the traced
# models in benchmark_scenarios/small/config.yaml, then extended with the
# aliases whose L4 module name does not match the scenario key.
_EXPLICIT_MODEL_TO_HF: dict[str, str] = {
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "llama_eagle3": "meta-llama/Llama-3.1-8B-Instruct",
    "gla": "fla-hub/gla-2.7B-100B",
    "gpt_oss": "openai/gpt-oss-120b",
    "qwen3_vl": "Qwen/Qwen3-VL-235B-A22B-Instruct",
    "qwen2_vl": "Qwen/Qwen2-VL-7B-Instruct",
    "yolov10": "jameslahm/yolov10n",
    "oasis": "Etched/oasis-500m",
    "flux": "black-forest-labs/FLUX.1-dev",
    "bge_m3": "BAAI/bge-m3",
    "openfold3": "OpenFold/OpenFold3",
    "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "retnet": "fla-hub/retnet-2.7B-100B",
    "rwkv7": "fla-hub/rwkv7-1.5B-world",
    "mamba": "state-spaces/mamba-2.8b-hf",
    "mamba2": "mistralai/Mamba-Codestral-7B-v0.1",
    "deepseek": "deepseek-ai/DeepSeek-V3.2-Exp",
    "gemma4": "google/gemma-4-E4B-it",
    "bitnet": "1bitLLM/bitnet_b1_58-3B",
}


@functools.cache
def _scenario_model_map() -> dict[str, str]:
    """key -> hf_name from the traced-scenario config, if readable."""
    out: dict[str, str] = {}
    cfg = (KB_ROOT / "bench" / "kernels" / "benchmark_scenarios"
           / "small" / "config.yaml")
    try:
        import yaml
        data = yaml.safe_load(cfg.read_text()) or {}
        for entry in data.get("models") or []:
            key, hf = entry.get("key"), entry.get("hf_name")
            if key and hf:
                out[key] = hf
                # 'llama31-8b' -> 'llama31', 'llama'
                stem = key.split("-")[0]
                out.setdefault(stem, hf)
    except Exception:
        pass
    return out


def hf_id_for_model(model_name: str) -> str | None:
    if model_name in _EXPLICIT_MODEL_TO_HF:
        return _EXPLICIT_MODEL_TO_HF[model_name]
    scen = _scenario_model_map()
    if model_name in scen:
        return scen[model_name]
    for key, hf in scen.items():
        if key.startswith(model_name) or model_name.startswith(key):
            return hf
    return None


def _register_extra_architectures() -> None:
    """Import packages that register their own config classes with transformers.

    ``fla-hub`` checkpoints declare ``model_type: gla`` / ``retnet`` / ``rwkv7``,
    which ``AutoConfig`` only recognizes once ``flash-linear-attention`` has been
    imported.  Without this, GLA/RetNet/RWKV configs fail to load and every
    linear-attention L3/L4 target stays unresolvable.
    """
    # fla performs its ``AutoConfig.register`` calls in ``fla.models``, so
    # importing the top-level package alone does not register anything.
    for pkg in ("fla", "fla.models"):
        try:
            __import__(pkg)
        except Exception:
            pass


@functools.cache
def _load_hf_config(hf_id: str) -> Any | None:
    """Load a config from the local HF cache (no weights, no network needed)."""
    try:
        from transformers import AutoConfig
    except Exception:
        return None
    # Register up front rather than only after a failure: a failed
    # ``from_pretrained`` can leave transformers' auto-mapping in a state where
    # the retry still misses the architecture.
    _register_extra_architectures()
    for kwargs in ({"trust_remote_code": True}, {}):
        try:
            return AutoConfig.from_pretrained(hf_id, **kwargs)
        except Exception:
            continue
    return None


# Attributes FastKernels' own config plumbing supplies but a raw HF config may
# not carry.  Reconstructing a config with ``AutoConfig`` therefore has to fill
# them in, or the model code dies on ``config.rope_theta`` / ``config.is_moe``
# before a single kernel runs.  Values are either a constant or a callable that
# derives the value from the config.
_CONFIG_DEFAULTS: dict[str, Any] = {
    "rope_theta": lambda c: (
        getattr(c, "rope_theta", None)
        or getattr(getattr(c, "text_config", None), "rope_theta", None)
        or getattr(c, "rotary_emb_base", None)
        or 10000.0
    ),
    "is_moe": lambda c: (
        getattr(c, "num_local_experts", None) is not None
        or getattr(c, "num_experts", None) is not None
        or getattr(c, "n_routed_experts", None) is not None
    ),
    "head_dim": lambda c: (
        getattr(c, "head_dim", None)
        or (
            getattr(c, "hidden_size", 0) // getattr(c, "num_attention_heads", 1)
            if getattr(c, "num_attention_heads", 0) else None
        )
    ),
    "rms_norm_eps": lambda c: (
        getattr(c, "rms_norm_eps", None)
        or getattr(c, "layer_norm_eps", None)
        or 1e-6
    ),
}


def _flatten_rope_scaling(cfg: Any) -> None:
    """Expose ``rope_scaling`` sub-keys as flat ``rope_<key>`` attributes.

    HF nests RoPE parameters in a dict; the task modules read them flat
    (``config.rope_scaling_factor``, ``rope_low_freq_factor``,
    ``rope_beta_fast``, ...).  Flattening the whole dict covers Llama-3.1's
    piecewise scaling and YaRN in one rule instead of chasing one
    AttributeError at a time.
    """
    rs = getattr(cfg, "rope_scaling", None)
    if not isinstance(rs, dict):
        # Some config classes expose ``rope_scaling`` as a dataclass Field
        # descriptor rather than a value; treat anything non-dict as absent.
        return
    for key, value in rs.items():
        if not isinstance(key, str):
            continue
        # Expose both spellings: some modules read the prefixed name
        # (``rope_scaling_factor``), others the bare key (Qwen-VL's
        # ``mrope_section``).
        names = {key, key if key.startswith("rope_") else f"rope_{key}"}
        for name in names:
            if getattr(cfg, name, None) is None:
                try:
                    setattr(cfg, name, value)
                except Exception:
                    pass


def _promote_text_config(cfg: Any) -> None:
    """Copy scalars up from ``text_config`` onto a multimodal wrapper.

    A wrapper such as ``Qwen3VLMoeConfig`` keeps ``vocab_size`` /
    ``hidden_size`` only on its text sub-config, while the L4 pipeline reads
    them off the top-level object.
    """
    sub = getattr(cfg, "text_config", None)
    if sub is None:
        return
    for name in dir(sub):
        if name.startswith("_"):
            continue
        try:
            value = getattr(sub, name)
        except Exception:
            continue
        if isinstance(value, type) or callable(value):
            continue
        if not isinstance(value, (int, float, str, bool, list, tuple)):
            continue
        if getattr(cfg, name, None) is None:
            try:
                setattr(cfg, name, value)
            except Exception:
                pass


def _fla_intermediate_size(cfg: Any) -> int | None:
    """FLA derives FFN width from ``hidden_ratio``, leaving ``intermediate_size``
    null in the checkpoint config (GLA/RetNet/RWKV).

    Verbatim from ``fla/modules/mlp.py:40-44``: the multiple of 256 closest to
    ``2/3 * hidden_size * hidden_ratio``.  Without it ``GLAMLP`` is built as
    ``Linear(2560, None)`` and every one of gla_decoder's 320 scenarios dies.
    """
    if getattr(cfg, "intermediate_size", None) is not None:
        return None
    hidden = getattr(cfg, "hidden_size", None)
    if not hidden:
        return None
    ratio = getattr(cfg, "hidden_ratio", None)
    if ratio is None:
        ratio = 4
    size = int(hidden * ratio * 2 / 3)
    return 256 * ((size + 256 - 1) // 256)


_CONFIG_DEFAULTS["intermediate_size"] = _fla_intermediate_size


def _patch_config_defaults(cfg: Any) -> Any:
    """Fill attributes the model code reads but the HF config does not define."""
    if cfg is None:
        return cfg
    _flatten_rope_scaling(cfg)
    # Qwen-VL keeps mrope_section in text_config.rope_scaling, so the nested
    # config has to be flattened before its scalars are promoted upward.
    sub = getattr(cfg, "text_config", None)
    if sub is not None:
        _flatten_rope_scaling(sub)
    _promote_text_config(cfg)
    for attr, source in _CONFIG_DEFAULTS.items():
        if getattr(cfg, attr, None) is not None:
            continue
        try:
            value = source(cfg) if callable(source) else source
        except Exception:
            continue
        if value is not None:
            try:
                setattr(cfg, attr, value)
            except Exception:
                pass
    return cfg


def config_candidates(models: tuple[str, ...]) -> list[Any]:
    """Configs to try for a target, most-specific sub-configs included."""
    out: list[Any] = []
    for model_name in models:
        hf_id = hf_id_for_model(model_name)
        if hf_id is None:
            continue
        cfg = _patch_config_defaults(_load_hf_config(hf_id))
        if cfg is None:
            continue
        # The wrapper goes first: an L4 pipeline is built from the top-level
        # config, and a sub-config lacks the fields it reads (Qwen3VLMoe's
        # vision tower has no ``vision`` attribute of its own).  Sub-configs
        # follow for the L3 blocks that genuinely take them; the activation
        # width ranking below still promotes an exact match.
        if cfg not in out:
            out.append(cfg)
        for attr in ("text_config", "llm_config", "vision_config"):
            sub = getattr(cfg, attr, None)
            if sub is not None and sub not in out:
                out.append(_patch_config_defaults(sub))
    return out


# Constructor parameter -> config attributes that can supply it, in order.
_SCALAR_ALIASES: dict[str, tuple[str, ...]] = {
    "hidden_size": ("hidden_size", "d_model", "n_embd", "dim"),
    "embed_dim": ("hidden_size", "embed_dim", "d_model"),
    "dim": ("hidden_size", "dim", "d_model"),
    "in_features": ("hidden_size", "d_model"),
    "input_size": ("hidden_size", "d_model"),
    "output_size": ("hidden_size", "d_model"),
    "embedding_dim": ("hidden_size", "embedding_dim", "d_model"),
    "num_heads": ("num_attention_heads", "num_heads", "n_head"),
    "num_attention_heads": ("num_attention_heads", "num_heads", "n_head"),
    "num_key_value_heads": (
        "num_key_value_heads", "num_kv_heads", "num_attention_heads",
    ),
    "head_dim": ("head_dim", "attention_head_dim"),
    "attention_head_dim": ("head_dim", "attention_head_dim"),
    "intermediate_size": ("intermediate_size", "ffn_dim", "n_inner"),
    "mlp_hidden_dim": ("intermediate_size", "ffn_dim"),
    "pooled_projection_dim": ("pooled_projection_dim", "hidden_size"),
    "conditioning_embedding_dim": ("hidden_size",),
    "rms_norm_eps": ("rms_norm_eps", "layer_norm_eps"),
    "num_experts": ("num_local_experts", "num_experts", "n_routed_experts"),
    "vocab_size": ("vocab_size",),
}


def _rope_scaling(cfg: Any) -> dict:
    rs = getattr(cfg, "rope_scaling", None)
    return rs if isinstance(rs, dict) else {}


# HF exposes RoPE scaling as a nested ``rope_scaling`` dict, while the task
# modules read flat ``config.rope_scaling_factor`` /
# ``config.rope_original_max_position_embeddings`` (see
# tasks/baseline/L1/rotary_emb.py).  Multimodal wrappers are likewise addressed
# as ``config.vision`` / ``config.text`` rather than ``*_config``.  These are
# name-schema mappings, not invented values.
_CONFIG_DEFAULTS.update({
    "rope_scaling_factor": lambda c: float(
        _rope_scaling(c).get("factor", 1.0) or 1.0
    ),
    "rope_original_max_position_embeddings": lambda c: (
        _rope_scaling(c).get("original_max_position_embeddings")
        or getattr(c, "max_position_embeddings", None)
    ),
    # HF calls it ``attention_bias``; the task modules read ``qkv_bias``.
    "qkv_bias": lambda c: bool(getattr(c, "attention_bias", False)),
    "attention_bias": lambda c: bool(getattr(c, "qkv_bias", False)),
    "vision": lambda c: getattr(c, "vision_config", None),
    "text": lambda c: getattr(c, "text_config", None),
})

# Parameters that are plain indices: 0 is always a valid choice.
_INDEX_PARAMS = frozenset({"layer_idx", "layer_id", "block_idx", "idx"})


def _required_params(cls: type) -> list[str]:
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return []
    return [
        name for name, p in sig.parameters.items()
        if name != "self"
        and p.default is inspect.Parameter.empty
        and p.kind not in (
            inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD,
        )
    ]


# Input names that carry the activation whose width is the module's model
# dimension.  Checked before any other tensor: a scenario also carries rotary
# caches and masks, and ``vision_block``'s ``rotary_pos_emb_cos`` (1320, 36)
# would otherwise be mistaken for a 36-wide hidden size.
_ACTIVATION_KEYS = (
    "hidden_states", "x", "query", "input", "inputs_embeds", "hidden_state",
)


def _trailing_dim(inputs: Any) -> int | None:
    """Width of the scenario's activation: its last dimension."""
    if isinstance(inputs, torch.Tensor):
        if inputs.is_floating_point() and inputs.ndim >= 1:
            return int(inputs.shape[-1])
        return None
    if isinstance(inputs, dict):
        for key in _ACTIVATION_KEYS:
            d = _trailing_dim(inputs.get(key))
            if d is not None:
                return d
        for v in inputs.values():
            d = _trailing_dim(v)
            if d is not None:
                return d
    if isinstance(inputs, (list, tuple)):
        for v in inputs:
            d = _trailing_dim(v)
            if d is not None:
                return d
    return None


def _dim_from_inputs(param: str, inputs: Any, cfg: Any) -> Any:
    """Derive a constructor dim from the scenario's tensors when config lacks it."""
    if inputs is None:
        return None
    tensors = [v for v in (inputs.values() if isinstance(inputs, dict) else [])
               if isinstance(v, torch.Tensor)]
    if not tensors:
        return None
    width = _trailing_dim(inputs)

    if param in ("dim", "embed_dim", "hidden_size", "in_features",
                 "input_size", "output_size", "embedding_dim"):
        return width
    if param == "attention_head_dim":
        heads = getattr(cfg, "num_attention_heads", None) or 24
        return width // heads if width else None
    if param in ("num_heads", "num_attention_heads"):
        # Prefer the config; with no config at all (Oasis ships no HF config)
        # fall back to the near-universal 64-wide head.
        n = (getattr(cfg, "num_attention_heads", None)
             or getattr(cfg, "num_heads", None)) if cfg is not None else None
        if n:
            return n
        return max(1, width // 64) if width else None
    if param == "mlp_hidden_dim":
        return width * 4 if width else None
    if param in ("frame_height", "frame_width"):
        # A VAE attention block flattens an H*W grid into the token axis.
        for t in tensors:
            if t.ndim >= 2:
                tokens = int(t.shape[-2])
                root = int(round(tokens ** 0.5))
                if root * root == tokens:
                    return root
        return None
    return None


def _build_module_arg(param: str, cfg: Any, inputs: Any) -> Any:
    """Rebuild an ``nn.Module`` constructor argument the tracer could not store.

    ``oasis_block`` takes the spatial/temporal rotary modules its parent
    ``oasis_dit`` owns; they are constructed there from ``head_dim`` alone
    (tasks/baseline/L3/oasis_dit.py:41-42), so they can be rebuilt exactly.
    """
    if param not in ("spatial_rotary_emb", "temporal_rotary_emb"):
        return None
    width = _trailing_dim(inputs)
    heads = head_dim = None
    if cfg is not None:
        heads = (getattr(cfg, "num_heads", None)
                 or getattr(cfg, "num_attention_heads", None))
        head_dim = getattr(cfg, "head_dim", None)
    if heads is None:
        # No HF config at all (Oasis ships none): recover the head count the
        # same way the scalar path does.
        heads = _dim_from_inputs("num_heads", inputs, cfg)
    if head_dim is None and width and heads:
        head_dim = width // heads
    if not head_dim:
        return None
    try:
        from fastkernels.tasks.baseline.L1.oasis_rotary import OasisRotaryEmbedding
    except Exception:
        return None
    try:
        if param == "spatial_rotary_emb":
            return OasisRotaryEmbedding(
                dim=head_dim // 2, freqs_for="pixel", max_freq=256,
            )
        return OasisRotaryEmbedding(dim=head_dim, freqs_for="lang")
    except Exception:
        return None


# Constructor arguments the tracer drops even though the class declares a
# default, where that default belongs to a different model variant.  Only the
# owning pipeline knows the real value, so it is recorded here with its source.
#
# ``YOLOv10DetectHead``'s ``ch`` defaults to the (256, 512, 1024) widths of a
# larger YOLO; ``tasks/baseline/L4/yolov10.py:45`` builds it as
# ``YOLOv10DetectHead(nc=80, ch=(64, 128, 256))``, matching the neck this
# checkpoint actually ships.  The tracer records only YAML-serializable kwargs,
# so the tuple never reached the registry, and the head is built expecting 256
# input channels where the neck emits 64 -- "weight of size [64, 256, 3, 3],
# expected input[32, 64, 80, 80]".
_CTOR_OVERRIDES: dict[str, dict[str, Any]] = {
    "YOLOv10DetectHead": {"ch": (64, 128, 256)},
}


def candidate_kwargs(
    cls: type,
    init_args: dict[str, Any],
    models: tuple[str, ...] = (),
    inputs: Any = None,
    level: int | None = None,
) -> list[dict[str, Any]]:
    """Kwarg dicts to try, in order, when constructing ``cls``.

    The first entry is the recorded ``init_args`` (plus any ``_CTOR_OVERRIDES``
    for arguments the tracer is known to drop), so a target whose scenario is
    already complete behaves exactly as before this module existed.
    """
    init_args = dict(init_args)
    for key, value in _CTOR_OVERRIDES.get(cls.__name__, {}).items():
        init_args.setdefault(key, value)
    attempts: list[dict[str, Any]] = [dict(init_args)]

    missing = [p for p in _required_params(cls) if p not in init_args]
    if not missing:
        return attempts

    configs = config_candidates(models) or [None]
    # An L4 pipeline is constructed from the top-level config; a sub-config
    # lacks the fields it reads (Qwen3VLMoe's vision tower has no ``vision``
    # attribute of its own).  Drop sub-configs so the width ranking below
    # cannot promote one over the wrapper.
    if level == 4 and len(configs) > 1:
        configs = configs[:1]
    fallback_dim = _trailing_dim(inputs)

    # A task shared by several pipelines (or a multimodal model's text vs vision
    # towers) offers several configs.  The scenario's own activation width is
    # the decisive evidence for which one produced it: `vision_block` fed a
    # 1152-wide SigLIP activation must not be built from a 4096-wide text
    # config.  Rank exact hidden-size matches first, preserving order otherwise.
    if fallback_dim and len(configs) > 1:
        def _width(cfg: Any) -> int | None:
            for attr in ("hidden_size", "d_model", "embed_dim", "dim"):
                val = getattr(cfg, attr, None)
                if isinstance(val, int):
                    return val
            return None

        configs = sorted(
            configs,
            key=lambda c: (0 if _width(c) == fallback_dim else 1),
        )

    for cfg in configs:
        kwargs = dict(init_args)
        for param in missing:
            if param == "config":
                if cfg is not None:
                    kwargs["config"] = cfg
                continue
            if param in _INDEX_PARAMS:
                kwargs[param] = 0
                continue
            built = _build_module_arg(param, cfg, inputs)
            if built is not None:
                kwargs[param] = built
                continue
            value = None
            for attr in _SCALAR_ALIASES.get(param, ()):
                if cfg is not None and getattr(cfg, attr, None) is not None:
                    value = getattr(cfg, attr)
                    break
            # Several blocks are parameterised by dims that appear directly in
            # the scenario's own tensors: flux_transformer_block's ``dim`` is
            # the hidden width (3072), oasis_vae_attention_block's
            # ``frame_height``/``frame_width`` factor its 576-token grid.
            if value is None:
                value = _dim_from_inputs(param, inputs, cfg)
            if value is None and param in _SCALAR_ALIASES and fallback_dim:
                # Dimension-like parameter with no config source: the trailing
                # dim of the scenario's own activation is the best evidence.
                if param in (
                    "hidden_size", "embed_dim", "dim", "in_features",
                    "input_size", "output_size", "embedding_dim",
                ):
                    value = fallback_dim
            if value is not None:
                kwargs[param] = value
        if kwargs not in attempts:
            attempts.append(kwargs)

    return attempts


def describe_unresolved(
    cls: type,
    init_args: dict[str, Any],
    models: tuple[str, ...] = (),
    inputs: Any = None,
) -> list[str]:
    """Required parameters no attempt could supply (for diagnostics)."""
    attempts = candidate_kwargs(cls, init_args, models, inputs)
    required = set(_required_params(cls))
    best = max((set(a) for a in attempts), key=lambda s: len(required & s),
               default=set())
    return sorted(required - best)


def preferred_dtype(models: tuple[str, ...]) -> Any:
    """The dtype the checkpoint declares, for scenarios with no float input.

    An L4 pipeline's scenario carries only integer tensors (input_ids,
    positions), so no dtype can be inferred from the inputs and the module would
    be built in fp32 -- which FlashAttention rejects.  The checkpoint's own
    ``torch_dtype`` is the right answer, and unlike a blanket bf16 default it
    leaves fp32 vision models (yolov10) alone.
    """
    for cfg in config_candidates(models):
        # transformers renamed the field; accept both spellings.
        for attr in ("torch_dtype", "dtype"):
            dt = getattr(cfg, attr, None)
            if isinstance(dt, torch.dtype) and dt.is_floating_point:
                return dt
            if isinstance(dt, str):
                resolved = getattr(torch, dt, None)
                if isinstance(resolved, torch.dtype) and resolved.is_floating_point:
                    return resolved

    # A quantized checkpoint (gpt-oss is MXFP4) records no float dtype at all,
    # and an L4 scenario carries only int64 input_ids, so nothing infers one.
    # Leaving it unset builds the model in fp32, which FlashAttention rejects
    # outright ("only supports fp16, bf16, and fp8_e4m3").  BF16 is the compute
    # dtype these models actually run in.
    if models:
        return torch.bfloat16
    return None
