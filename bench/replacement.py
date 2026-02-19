"""
Class monkey-patching for benchmarking.

Patches the original nn.Module class in its source module AND in every
other loaded module that imported it, so that subsequent model construction
uses the replacement regardless of how the class was imported.
"""

from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager

from .discovery import BenchTarget


def _get_package_root() -> str:
    return __package__.rsplit(".", 1)[0]


def _find_all_references(original_cls: type) -> list[tuple[object, str]]:
    """Find all (module, attr_name) pairs where attr is original_cls."""
    pkg_root = _get_package_root()
    refs = []
    for mod_name, mod in list(sys.modules.items()):
        if mod is None or not mod_name.startswith(pkg_root):
            continue
        for attr_name in list(vars(mod)):
            try:
                if getattr(mod, attr_name) is original_cls:
                    refs.append((mod, attr_name))
            except Exception:
                continue
    return refs


def patch_class(target: BenchTarget, user_cls: type) -> list[tuple]:
    """Monkey-patch the target class with user_cls everywhere it's referenced.

    Returns undo info: a list of (module, attr_name, original_cls) tuples.
    """
    original_cls = target.target_cls
    refs = _find_all_references(original_cls)
    undo_info = []
    for mod, attr_name in refs:
        undo_info.append((mod, attr_name, original_cls))
        setattr(mod, attr_name, user_cls)
    return undo_info


def restore(undo_info: list[tuple]) -> None:
    """Restore all original classes from undo info."""
    for mod, attr_name, original_cls in undo_info:
        setattr(mod, attr_name, original_cls)


@contextmanager
def replacement_context(target: BenchTarget, user_cls: type):
    """Context manager that patches the class and restores on exit."""
    undo = patch_class(target, user_cls)
    try:
        yield
    finally:
        restore(undo)
