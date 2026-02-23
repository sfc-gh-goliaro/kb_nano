"""Convention-based target discovery via static import analysis.

Scans tasks/baseline/L4/ files as model entry points and recursively walks
their imports (using ast.parse) to determine which L1-L3 operators each model
uses.  No _META annotations needed -- the import graph is the single source of
truth.
"""

from __future__ import annotations

import ast
import importlib
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch.nn as nn

_KB_ROOT = Path(__file__).resolve().parent.parent

_L4_MODEL_KEYS: dict[str, str] = {
    "llama": "llama31",
    "mixtral": "mixtral",
}


@dataclass
class BenchTarget:
    name: str
    level: int
    module_path: str
    models: list[str]
    target_cls: type


_TARGETS: list[BenchTarget] | None = None


def _resolve_internal_imports(filepath: Path, visited: set[Path] | None = None) -> set[Path]:
    """Recursively collect all internal .py files imported by *filepath*.

    Only follows imports that resolve to files under _KB_ROOT and belong to
    tasks/baseline/L1/, tasks/baseline/L2/, tasks/baseline/L3/,
    tasks/baseline/L4/, or infra/ directories.
    """
    if visited is None:
        visited = set()
    if filepath in visited:
        return visited
    visited.add(filepath)

    try:
        source = filepath.read_text()
    except OSError:
        return visited

    tree = ast.parse(source, filename=str(filepath))
    file_dir = filepath.parent

    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        level = node.level or 0
        if level == 0:
            continue
        base = file_dir
        for _ in range(level - 1):
            base = base.parent

        parts = node.module.split(".")
        candidate = base / "/".join(parts)
        candidate_file = candidate.with_suffix(".py")
        if candidate_file.is_file() and _KB_ROOT in candidate_file.parents:
            _resolve_internal_imports(candidate_file, visited)

    return visited


def _find_module_class(mod) -> type | None:
    """Find the primary nn.Module subclass defined in a target module.

    Returns the last nn.Module class defined in the file, since the
    top-level class is conventionally the final definition.
    """
    result = None
    for v in vars(mod).values():
        if (isinstance(v, type)
                and issubclass(v, nn.Module)
                and v is not nn.Module
                and v.__module__ == mod.__name__):
            result = v
    return result


def _build_model_map() -> dict[str, set[str]]:
    """Build a mapping from relative module path -> set of model keys.

    Walks tasks/baseline/L4/ entry points and traces their imports to build the reverse map.
    """
    l4_dir = _KB_ROOT / "tasks" / "baseline" / "L4"
    if not l4_dir.is_dir():
        return {}

    file_to_models: dict[Path, set[str]] = {}

    for fname in sorted(os.listdir(l4_dir)):
        if fname.startswith("_") or not fname.endswith(".py"):
            continue
        stem = fname[:-3]
        model_key = _L4_MODEL_KEYS.get(stem)
        if model_key is None:
            continue
        l4_file = l4_dir / fname
        deps = _resolve_internal_imports(l4_file)
        for dep in deps:
            file_to_models.setdefault(dep, set()).add(model_key)

    path_to_models: dict[str, set[str]] = {}
    for fpath, keys in file_to_models.items():
        try:
            rel = fpath.relative_to(_KB_ROOT)
        except ValueError:
            continue
        mod_path = str(rel.with_suffix("")).replace("/", ".")
        path_to_models[mod_path] = keys

    return path_to_models


def discover_targets() -> list[BenchTarget]:
    global _TARGETS
    if _TARGETS is not None:
        return _TARGETS

    pkg_root = __package__.rsplit(".", 1)[0]
    model_map = _build_model_map()
    targets = []

    for level_num in (1, 2, 3, 4):
        level_dir = _KB_ROOT / "tasks" / "baseline" / f"L{level_num}"
        if not level_dir.is_dir():
            continue
        for fname in sorted(os.listdir(level_dir)):
            if fname.startswith("_") or not fname.endswith(".py"):
                continue
            name = fname[:-3]
            module_path = f"tasks.baseline.L{level_num}.{name}"
            models = sorted(model_map.get(module_path, []))
            if not models:
                continue

            mod = importlib.import_module(f"{pkg_root}.{module_path}")
            target_cls = _find_module_class(mod)
            if target_cls is None:
                continue

            targets.append(BenchTarget(
                name=name,
                level=level_num,
                module_path=module_path,
                models=models,
                target_cls=target_cls,
            ))

    _TARGETS = targets
    return _TARGETS


def get(name: str) -> BenchTarget:
    targets = discover_targets()
    for t in targets:
        if t.name == name:
            return t
    available = ", ".join(sorted(t.name for t in targets))
    raise KeyError(f"Unknown bench target {name!r}. Available: {available}")


def list_targets(level: int | None = None) -> list[BenchTarget]:
    targets = discover_targets()
    if level is not None:
        targets = [t for t in targets if t.level == level]
    return sorted(targets, key=lambda t: (t.level, t.name))


def models_for_target(name: str) -> list[str]:
    """Return model keys that use a given target operator."""
    return get(name).models


def targets_for_model(model_key: str) -> list[BenchTarget]:
    """Return all targets used by a given model key (e.g. 'llama31', 'mixtral')."""
    return [t for t in discover_targets() if model_key in t.models]


def print_model_operator_map() -> None:
    """Print which operators each model uses, and which models each operator belongs to."""
    targets = discover_targets()

    model_to_targets: dict[str, list[BenchTarget]] = {}
    for t in targets:
        for m in t.models:
            model_to_targets.setdefault(m, []).append(t)

    print(f"\n{'=' * 70}")
    print("  OPERATORS BY MODEL")
    print(f"{'=' * 70}")
    for model_key in sorted(model_to_targets):
        ops = sorted(model_to_targets[model_key], key=lambda t: (t.level, t.name))
        print(f"\n  {model_key}:")
        for t in ops:
            print(f"    L{t.level}  {t.name:<25} {t.target_cls.__name__}")

    print(f"\n{'=' * 70}")
    print("  MODELS BY OPERATOR")
    print(f"{'=' * 70}")
    for t in sorted(targets, key=lambda t: (t.level, t.name)):
        print(f"  L{t.level}  {t.name:<25} {','.join(t.models)}")
    print()
