"""Synthesize forward() arguments the input tracer could not serialize.

The tracer records a scenario's inputs as tensors in YAML, so any forward
argument that is not a plain tensor is dropped:

* ``nn.Module`` arguments -- ``gpt_oss_decoder`` takes the shared
  ``rotary_emb`` module its L4 pipeline owns, ``oasis_rollout`` takes the
  diffusion ``model`` and ``vae``.
* ``list[Tensor]`` arguments -- ``yolov10_neck`` takes ``feats`` (multi-scale
  feature maps) and ``yolov10_head`` takes ``x``; both scenarios record *zero*
  inputs as a result.

Tier 1 then fails with "missing 1 required positional argument" before any
kernel runs, which is indistinguishable from a broken candidate.

What can be rebuilt is rebuilt here, from the same config the module was
constructed with; what cannot (a full VAE, a trained detector's pyramid) is
left alone so the failure stays honest.
"""

from __future__ import annotations

import functools
import inspect
from typing import Any

import torch
import torch.nn as nn

from fastkernels.bench.kernels.init_resolver import config_candidates


def _required_forward_params(cls: type) -> list[str]:
    try:
        sig = inspect.signature(cls.forward)
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


def _build_rotary_emb(cfg: Any, device: str, dtype: torch.dtype | None):
    """Rebuild the rotary-embedding module an L4 pipeline shares with its layers.

    Mirrors the construction in the L4 model files (e.g.
    ``tasks/baseline/L4/gpt_oss.py``): YaRN when the config carries YaRN
    parameters, plain RoPE otherwise.
    """
    head_dim = getattr(cfg, "head_dim", None)
    if not head_dim:
        hidden = getattr(cfg, "hidden_size", 0)
        heads = getattr(cfg, "num_attention_heads", 0)
        head_dim = hidden // heads if heads else None
    if not head_dim:
        return None

    max_pos = getattr(cfg, "max_position_embeddings", 4096)
    theta = getattr(cfg, "rope_theta", 10000.0)

    has_yarn = getattr(cfg, "rope_beta_fast", None) is not None
    try:
        if has_yarn:
            from fastkernels.tasks.baseline.L1.yarn_rotary_emb import (
                YaRNRotaryEmbedding,
            )
            mod = YaRNRotaryEmbedding(
                head_dim,
                max_pos,
                theta,
                scaling_factor=getattr(cfg, "rope_scaling_factor", 1.0),
                original_max_position_embeddings=getattr(
                    cfg, "rope_original_max_position_embeddings", max_pos,
                ),
                beta_fast=getattr(cfg, "rope_beta_fast", 32),
                beta_slow=getattr(cfg, "rope_beta_slow", 1),
                truncate=getattr(cfg, "rope_truncate", False),
            )
        else:
            from fastkernels.tasks.baseline.L1.rotary_emb import RotaryEmbedding
            mod = RotaryEmbedding(head_dim, max_pos, theta)
    except Exception:
        return None

    mod = mod.to(device)
    if dtype is not None:
        with torch.no_grad():
            for p in mod.parameters(recurse=True):
                if p.is_floating_point():
                    p.data = p.data.to(dtype=dtype)
    mod.eval()
    return mod


# Forward-argument name -> builder.  Only arguments that can be faithfully
# reconstructed from the config appear here.
def _build_oasis_model(cfg, device: str, dtype):
    """The DiT that ``oasis_rollout`` denoises with (factory in L4/oasis.py:19)."""
    try:
        from fastkernels.tasks.baseline.L4.oasis import DiT_S_2
    except Exception:
        return None
    try:
        max_frames = getattr(cfg, "max_frames", None) or 32
        m = DiT_S_2(max_frames=max_frames).to(device)
        _load_oasis_shard(m, "oasis500m.safetensors")
        if dtype is not None:
            m = m.to(dtype=dtype)
        m.eval()
        return m
    except Exception:
        return None


def _build_oasis_vae(cfg, device: str, dtype):
    """The VAE encoder ``oasis_rollout`` needs (factory in L4/oasis.py:23)."""
    del cfg
    try:
        from fastkernels.tasks.baseline.L4.oasis import (
            ViT_L_20_Shallow_Encoder,
        )
    except Exception:
        return None
    try:
        m = ViT_L_20_Shallow_Encoder().to(device)
        _load_oasis_shard(m, "vit-l-20.safetensors")
        if dtype is not None:
            m = m.to(dtype=dtype)
        m.eval()
        return m
    except Exception:
        return None


def _load_oasis_shard(module: nn.Module, filename: str) -> int:
    """Load an Oasis checkpoint into a synthesized module.

    Oasis ships two plain shards (``oasis500m.safetensors``,
    ``vit-l-20.safetensors``) rather than the sharded/indexed layout
    ``real_weights`` scans, so they are loaded directly here.  With random
    weights instead, DDIM sampling diverges: at ``ddim_steps=4`` both the
    baseline and the reference return non-finite frames and the comparison
    says nothing about the candidate.
    """
    import os
    from fastkernels.bench.kernels.real_weights import _local_snapshot
    from fastkernels.bench.kernels.init_resolver import hf_id_for_model

    hf_id = hf_id_for_model("oasis")
    root = _local_snapshot(hf_id) if hf_id else None
    if not root:
        return 0
    path = os.path.join(root, filename)
    if not os.path.exists(path):
        return 0
    try:
        from safetensors.torch import load_file
        state = load_file(path)
    except Exception:
        return 0
    own = dict(module.named_parameters())
    filled = 0
    with torch.no_grad():
        for name, tensor in state.items():
            param = own.get(name)
            if param is None:
                # Checkpoints often carry a module prefix the bare block lacks.
                tail = name.split(".", 1)[-1]
                param = own.get(tail)
            if param is None or tuple(param.shape) != tuple(tensor.shape):
                continue
            param.data.copy_(tensor.to(device=param.device, dtype=param.dtype))
            filled += 1
    return filled


def _build_stage_module(name: str, device: str, dtype):
    """Construct another benchmark target exactly the way the runner does.

    Reuses ``runner._instantiate_module`` rather than re-deriving the kwargs:
    that helper drops keys the constructor does not accept and falls back to a
    bare ``cls()``, which is what ``YOLOv10Backbone`` needs -- its scenario
    records ``training=False``, an argument it does not take, so building
    straight from ``candidate_kwargs`` fails with "unexpected keyword argument
    'training'" and the whole reconstruction silently returns None.

    Imported lazily: ``runner`` imports this module at load time.
    """
    from fastkernels.bench.kernels.runner import _instantiate_module
    from fastkernels.bench.kernels.scenario_registry import InputRegistry
    from fastkernels.infra.kernel_swapper import discover_targets, get

    discover_targets()
    target = get(name)
    scenarios = InputRegistry().scenarios(name) or []
    init_args = scenarios[0].init_args if scenarios else {}
    inputs = scenarios[0].inputs if scenarios else None

    try:
        module = _instantiate_module(
            target.target_cls, init_args, device, dtype,
            tuple(target.models or ()), inputs, target.level,
        )
    except Exception:
        return None
    module.eval()
    return module


@functools.lru_cache(maxsize=4)
def _yolo_stages(device: str, dtype_str: str):
    """Run YOLOv10's backbone and neck to recover the inputs of later stages.

    ``YOLOv10Neck.forward`` takes ``feats: dict[str, Tensor]`` and
    ``YOLOv10DetectHead.forward`` takes ``x: list[Tensor]``.  Neither is a plain
    tensor, so the tracer recorded no inputs at all (both scenarios are
    ``tokens-0``) and Tier 1 fails with "missing 1 required positional argument"
    before any kernel runs.

    The shapes could be guessed from the head's ``init_args`` (``nl=3``,
    ``shape=[32, 144, 80, 80]``), but the channel counts entering the head are
    the *neck's* output widths, which that hint does not give.  Running the two
    earlier stages instead reproduces them exactly, on the same 640x640 image
    the registry recorded for ``yolov10_backbone``.
    """
    from fastkernels.bench.kernels.scenario_registry import InputRegistry

    dtype = getattr(torch, dtype_str) if dtype_str else torch.bfloat16
    backbone = _build_stage_module("yolov10_backbone", device, dtype)
    neck = _build_stage_module("yolov10_neck", device, dtype)
    if backbone is None:
        return None, None

    shape = [32, 3, 640, 640]
    for scenario in InputRegistry().scenarios("yolov10_backbone") or []:
        spec = (scenario.inputs or {}).get("x")
        recorded = getattr(spec, "shape", None) or (
            spec.get("shape") if isinstance(spec, dict) else None
        )
        if recorded:
            shape = list(recorded)
            break

    image = torch.randn(*shape, device=device, dtype=dtype)
    with torch.no_grad():
        feats = backbone(image)
        neck_out = neck(feats) if neck is not None else None
    return feats, neck_out


def _build_yolo_feats(cfg, device: str, dtype):
    del cfg
    feats, _ = _yolo_stages(device, dtype.__str__().split(".")[-1] if dtype else "")
    return feats


def _build_yolo_head_inputs(cfg, device: str, dtype):
    del cfg
    _, neck_out = _yolo_stages(device, dtype.__str__().split(".")[-1] if dtype else "")
    return neck_out


_BUILDERS = {
    "rotary_emb": _build_rotary_emb,
    # oasis_rollout takes the DiT and VAE its L4 pipeline owns; both are
    # constructed from config alone, so they can be rebuilt exactly.
    "model": _build_oasis_model,
    "vae": _build_oasis_vae,
}

# Builders that only apply to one target.  ``x`` is far too generic a name to
# resolve globally, so these are keyed by (class name, argument name).
_CLASS_BUILDERS = {
    ("YOLOv10Neck", "feats"): _build_yolo_feats,
    ("YOLOv10DetectHead", "x"): _build_yolo_head_inputs,
}


def synthesize_forward_args(
    cls: type,
    inputs: dict[str, Any],
    models: tuple[str, ...] = (),
    device: str = "cuda",
    dtype: torch.dtype | None = None,
) -> dict[str, Any]:
    """Return extra forward kwargs for arguments the registry does not supply.

    Only fills names this module knows how to rebuild; unknown gaps are left so
    the original error surfaces.
    """
    missing = [p for p in _required_forward_params(cls) if p not in inputs]
    if not missing:
        return {}

    extra: dict[str, Any] = {}
    # ``or [None]``: Oasis ships no HF config, so config_candidates is empty and
    # the loop below would never run.  The builders already tolerate cfg=None.
    configs = config_candidates(models) or [None]
    for name in missing:
        builder = (
            _CLASS_BUILDERS.get((cls.__name__, name)) or _BUILDERS.get(name)
        )
        if builder is None:
            continue
        for cfg in configs:
            built = builder(cfg, device, dtype)
            if built is not None:
                extra[name] = built
                break
    return extra
