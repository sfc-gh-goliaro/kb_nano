"""
Simple class monkey-patching for benchmarking.

Patches the original nn.Module class in its source module with the user's
subclass, so that subsequent model construction uses the replacement.
"""

from __future__ import annotations

import importlib
from contextlib import contextmanager

from .discovery import BenchTarget


def _get_package_root() -> str:
    return __package__.rsplit(".", 1)[0]


def patch_class(target: BenchTarget, user_cls: type) -> tuple:
    """Monkey-patch the target class with user_cls. Returns undo info."""
    pkg_root = _get_package_root()
    mod = importlib.import_module(f"{pkg_root}.{target.module_path}")
    original_cls = target.target_cls
    cls_name = original_cls.__name__
    setattr(mod, cls_name, user_cls)
    return (mod, cls_name, original_cls)


def restore(undo_info: tuple) -> None:
    """Restore the original class from undo info."""
    mod, cls_name, original_cls = undo_info
    setattr(mod, cls_name, original_cls)


@contextmanager
def replacement_context(target: BenchTarget, user_cls: type):
    """Context manager that patches the class and restores on exit."""
    undo = patch_class(target, user_cls)
    try:
        yield
    finally:
        restore(undo)
