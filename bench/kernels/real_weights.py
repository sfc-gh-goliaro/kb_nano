"""Load real checkpoint weights into a Tier-1 benchmark module.

Tier 1 constructs modules with ``torch.empty`` and never loads a checkpoint, so
every parameter starts as uninitialized memory -- in practice a zero page.  For
most operators that is merely unrealistic, but for anything with a log-space
gate or a deep chain of matmuls it is fatal: GLA computes ``log(sigmoid(0))``
and returns NaN, and an fp16 attention block overflows once random weights are
substituted instead.  Either way *both* the baseline and the candidate produce
NaN, so the correctness comparison reports a failure that says nothing about
the candidate.

This module fills a benchmark module from the architecture's real safetensors
shards, matching by parameter-name suffix so that an L3 block (whose parameters
are named ``self_attn.q_proj.weight``) picks up the corresponding tensors from
any layer of the full checkpoint (``model.layers.0.self_attn.q_proj.weight``).

Tensors are read lazily and only the shards that contain a needed name are
touched, so loading one decoder layer does not pull a 120B checkpoint into
memory.  When no local checkpoint is available the module is left untouched and
the caller falls back to synthetic initialization.
"""

from __future__ import annotations

import functools
import glob
import os
from typing import Any

import torch
import torch.nn as nn

from fastkernels.bench.kernels.init_resolver import hf_id_for_model


@functools.cache
def _local_snapshot(hf_id: str) -> str | None:
    """Path to an already-downloaded HF snapshot, or None (never downloads)."""
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(hf_id, local_files_only=True)
    except Exception:
        return None


@functools.cache
def _shard_index(model_path: str) -> tuple[tuple[str, ...], ...]:
    """(file, *tensor_names) for each weight shard under ``model_path``.

    Two wrinkles in a real HF cache: a repo can have several snapshots where
    only one carries the weights (bge-m3 keeps ``model.safetensors`` under a
    different revision than the one ``snapshot_download`` resolves to), and some
    repos ship only ``pytorch_model.bin``.  Search sibling snapshots and accept
    both formats, otherwise the loader silently finds zero tensors and every
    parameter falls back to synthetic values.
    """
    from safetensors import safe_open

    roots = [model_path]
    snapshots = os.path.dirname(model_path)
    if os.path.basename(snapshots) == "snapshots":
        for sib in sorted(os.listdir(snapshots)):
            cand = os.path.join(snapshots, sib)
            if cand != model_path and os.path.isdir(cand):
                roots.append(cand)

    out = []
    seen_files: set[str] = set()
    for root in roots:
        for pattern in ("*.safetensors", "*.bin"):
            for f in sorted(glob.glob(os.path.join(root, "**", pattern),
                                      recursive=True)):
                real = os.path.realpath(f)
                if real in seen_files:
                    continue
                seen_files.add(real)
                try:
                    if f.endswith(".safetensors"):
                        with safe_open(f, "pt", "cpu") as handle:
                            out.append((f, *handle.keys()))
                    else:
                        state = torch.load(f, map_location="cpu",
                                           mmap=True, weights_only=True)
                        out.append((f, *state.keys()))
                except Exception:
                    continue
    return tuple(out)


# Checkpoints whose layout shares no suffix with the module tree, so
# suffix matching alone finds nothing.  YOLOv10 ships ultralytics' positional
# naming (``model.model.0.conv.weight``) while the modules use semantic names
# (``stem1.conv.weight``); with no translation the backbone loads 0/108
# parameters, falls back to synthetic values, and a 100-layer conv stack
# saturates bf16 -- every feature map comes back NaN.
#
# Transcribed from the pipeline that owns the mapping,
# ``tasks/baseline/L4/yolov10.py:18`` (``_PREFIX_MAP``).
_PREFIX_REMAP: dict[str, tuple[tuple[str, str], ...]] = {
    "yolov10": (
        ("model.model.0.", "backbone.stem1."),
        ("model.model.1.", "backbone.stem2."),
        ("model.model.2.", "backbone.stage2."),
        ("model.model.3.", "backbone.down3."),
        ("model.model.4.", "backbone.stage3."),
        ("model.model.5.", "backbone.down4."),
        ("model.model.6.", "backbone.stage4."),
        ("model.model.7.", "backbone.down5."),
        ("model.model.8.", "backbone.stage5."),
        ("model.model.9.", "backbone.sppf."),
        ("model.model.10.", "backbone.psa."),
        ("model.model.13.", "neck.c2f_p4."),
        ("model.model.16.", "neck.c2f_p3."),
        ("model.model.17.", "neck.down_p3."),
        ("model.model.19.", "neck.c2f_n4."),
        ("model.model.20.", "neck.down_n4."),
        ("model.model.22.", "neck.c2fcib_n5."),
        ("model.model.23.", "detect."),
    ),
}


def _prefix_rules(models: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    rules: list[tuple[str, str]] = []
    for name in models:
        rules.extend(_PREFIX_REMAP.get(name, ()))
    return tuple(rules)


def _suffix_map(
    names: list[str],
    prefix_rules: tuple[tuple[str, str], ...] = (),
) -> dict[str, str]:
    """Map a checkpoint name's meaningful suffix back to the full name.

    ``model.layers.0.self_attn.q_proj.weight`` is indexed under
    ``self_attn.q_proj.weight``, ``q_proj.weight`` and ``weight`` so a block
    whose parameters carry only the tail of the path still resolves.  Earlier
    layers win, keeping the choice deterministic.

    ``prefix_rules`` additionally indexes each name under its translated form,
    so a checkpoint that shares no suffix with the module tree still resolves.
    """
    out: dict[str, str] = {}

    def _index(key: str, full: str) -> None:
        parts = key.split(".")
        for i in range(len(parts)):
            out.setdefault(".".join(parts[i:]), full)

    for full in names:
        _index(full, full)
        for old, new in prefix_rules:
            if full.startswith(old):
                _index(new + full[len(old):], full)
                break
    return out


# Leaf-name differences between FastKernels' modules and the checkpoints they
# track.  Qwen-VL's vision MLP is ``mlp.linear_fc1``/``linear_fc2`` upstream but
# ``mlp.fc1``/``fc2`` here; without the alias those four tensors keep synthetic
# values while everything else is real, and the block's activations blow past
# bf16 range (observed scale 8.2e36, output non-finite).
_NAME_ALIASES: dict[str, tuple[str, ...]] = {
    "fc1": ("linear_fc1", "gate_proj", "up_proj"),
    "fc2": ("linear_fc2", "down_proj"),
    "linear_fc1": ("fc1",),
    "linear_fc2": ("fc2",),
}


# MXFP4 MoE: FastKernels names the packed expert tensors ``w13_*``/``w2_*``
# while gpt-oss ships them as ``experts.gate_up_proj_*``/``experts.down_proj_*``
# with ``_blocks``/``_scales`` suffixes.  Mapping them keeps the expert weights
# real instead of synthetic -- with synthetic ones the packed layout no longer
# matches and the block dies on "shape '[128, 5760, 32]' is invalid".
# Only the *bias* tensors map cleanly.  The ``_blocks``/``_scales`` pair is a
# packed MXFP4 layout whose shape differs from the module parameter it feeds;
# copying it in makes the module's own dequant misread the block structure
# ("shape '[128, 5760, 32]' is invalid for input of size 2123366400").  Those
# stay untouched so the module keeps a self-consistent packed value.
_MXFP4_LEAF = {
    "w13_bias": "experts.gate_up_proj_bias",
    "w2_bias": "experts.down_proj_bias",
}


def _module_name_tails(pname: str) -> list[str]:
    """Progressively shorter tails of a module parameter path, longest first.

    Stops before the path gets short enough to be ambiguous: a bare ``weight``
    or ``0.weight`` would bind to whichever tensor the suffix map happened to
    index first.
    """
    parts = pname.split(".")
    return [".".join(parts[i:]) for i in range(1, max(1, len(parts) - 1))]


def _alias_candidates(pname: str) -> list[str]:
    """Alternative checkpoint spellings for a parameter path."""
    parts = pname.split(".")
    out = []

    # Whole-leaf rename (mlp.w13_weight -> mlp.experts.gate_up_proj_blocks)
    if parts and parts[-1] in _MXFP4_LEAF:
        out.append(".".join(parts[:-1] + [_MXFP4_LEAF[parts[-1]]]))

    if len(parts) < 2:
        return out
    leaf, suffix = parts[-2], parts[-1]
    for alt in _NAME_ALIASES.get(leaf, ()):  # type: ignore[arg-type]
        out.append(".".join(parts[:-2] + [alt, suffix]))
    return out


def load_real_weights(
    module: nn.Module,
    models: tuple[str, ...],
    device: str = "cuda",
    dtype: torch.dtype | None = None,
) -> int:
    """Fill ``module``'s parameters from a local checkpoint.

    Returns the number of parameters filled; 0 means nothing was available and
    the caller should fall back to synthetic initialization.
    """
    from safetensors import safe_open

    paths = []
    for name in models:
        hf_id = hf_id_for_model(name)
        if hf_id is None:
            continue
        p = _local_snapshot(hf_id)
        if p:
            paths.append(p)
    if not paths:
        return 0

    wanted = {
        name: param for name, param in module.named_parameters(recurse=True)
        if param.is_floating_point()
    }
    if not wanted:
        return 0

    # Production modules fuse projections the checkpoint stores separately
    # (``to_qkv`` <- to_q/to_k/to_v, ``gate_up_proj`` <- gate_proj/up_proj).
    # Without this the fused parameter keeps its synthetic values while every
    # other weight is real, and the block's output drifts far enough to fail
    # the tolerance check.
    FUSIONS = {
        "to_qkv": ("to_q", "to_k", "to_v"),
        "qkv_proj": ("q_proj", "k_proj", "v_proj"),
        "gate_up_proj": ("gate_proj", "up_proj"),
    }

    filled = 0
    for model_path in paths:
        shards = _shard_index(model_path)
        if not shards:
            continue
        for entry in shards:
            fname, keys = entry[0], list(entry[1:])
            smap = _suffix_map(keys, _prefix_rules(models))
            # Which of the still-missing parameters live in this shard?
            todo = {}
            for pname in list(wanted):
                if pname in smap:
                    todo[pname] = smap[pname]
                    continue
                for alias in _alias_candidates(pname):
                    if alias in smap:
                        todo[pname] = smap[alias]
                        break
                if pname in todo:
                    continue
                # The module path can be *longer* than the checkpoint's: a
                # wrapper adds a level the checkpoint never had.  BGE-M3 stores
                # ``embeddings.word_embeddings.weight`` while
                # ``BgeM3EmbeddingModel`` holds it at
                # ``model.embeddings.word_embeddings.weight``, so indexing only
                # the checkpoint side finds 0/393 parameters and the embedding
                # model returns non-finite scores.  Strip leading components
                # from the module name too, longest match first, and keep at
                # least two so a bare ``weight`` cannot bind to an unrelated
                # tensor.  The shape check below is the final guard.
                for match in _module_name_tails(pname):
                    if match in smap:
                        todo[pname] = smap[match]
                        break
            if not todo:
                continue
            try:
                if not fname.endswith(".safetensors"):
                    state = torch.load(fname, map_location="cpu",
                                       mmap=True, weights_only=True)

                    class _Handle:
                        def get_tensor(self, k):
                            return state[k]

                        def __enter__(self):
                            return self

                        def __exit__(self, *a):
                            return False

                    handle_ctx = _Handle()
                else:
                    handle_ctx = safe_open(fname, "pt", "cpu")
                with handle_ctx as handle:
                    for pname, ckpt_name in todo.items():
                        param = wanted.get(pname)
                        if param is None:
                            continue
                        try:
                            tensor = handle.get_tensor(ckpt_name)
                        except Exception:
                            continue
                        if tuple(tensor.shape) != tuple(param.shape):
                            continue  # sharded/fused layout: skip, not guess
                        with torch.no_grad():
                            param.data.copy_(
                                tensor.to(device=param.device,
                                          dtype=param.dtype)
                            )
                        wanted.pop(pname, None)
                        filled += 1
            except Exception:
                continue
        if not wanted:
            break

    # Second pass: assemble fused parameters from their split counterparts.
    if wanted:
        for model_path in paths:
            shards = _shard_index(model_path)
            if not shards:
                continue
            index: dict[str, str] = {}
            for entry in shards:
                for k in entry[1:]:
                    index.setdefault(k, entry[0])
            smap_all = _suffix_map(list(index), _prefix_rules(models))
            for pname in list(wanted):
                leaf = pname.split(".")[-2] if "." in pname else ""
                parts = FUSIONS.get(leaf)
                if not parts:
                    continue
                suffix = pname.split(".")[-1]          # weight / bias
                prefix = ".".join(pname.split(".")[:-2])
                pieces = []
                for part in parts:
                    key = f"{prefix}.{part}.{suffix}" if prefix else f"{part}.{suffix}"
                    full = smap_all.get(key)
                    if full is None:
                        pieces = []
                        break
                    pieces.append(full)
                if not pieces:
                    continue
                try:
                    from safetensors import safe_open as _open
                    tensors = []
                    for full in pieces:
                        with _open(index[full], "pt", "cpu") as h:
                            tensors.append(h.get_tensor(full))
                    fused = torch.cat(tensors, dim=0)
                except Exception:
                    continue
                param = wanted.get(pname)
                if param is None or tuple(fused.shape) != tuple(param.shape):
                    continue
                with torch.no_grad():
                    param.data.copy_(
                        fused.to(device=param.device, dtype=param.dtype)
                    )
                wanted.pop(pname, None)
                filled += 1

    return filled
