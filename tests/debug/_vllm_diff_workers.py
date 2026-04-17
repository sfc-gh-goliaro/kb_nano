"""Pickle-friendly trampolines for the diff diagnostic.

vLLM's ``apply_model`` ships the user-supplied callback to its EngineCore
worker process via ``pickle.dumps`` (see
``vllm/distributed/device_communicators/shm_broadcast.py``).  Functions
defined inside ``diff_deepseek_layers.py`` live in the parent's
``__main__`` module, which the EngineCore subprocess has no way to
re-import — so stdlib pickle errors out with::

    Can't pickle <function _vllm_install_hooks_worker ...>:
      it's not the same object as __main__._vllm_install_hooks_worker

To avoid this we route ``apply_model`` through this module instead, which
*is* importable by its fully-qualified package path
(``kb_nano.tests.debug._vllm_diff_workers``) on both the parent and the
EngineCore subprocess.  The trampolines below dynamically load the
diagnostic script as a regular module on first call and forward to the
real implementation.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys


_SCRIPT_PATH_ENV = "KB_DIFF_SCRIPT_PATH"


def _load_diff_module():
    """Import ``diff_deepseek_layers.py`` as a regular module.

    The diagnostic is normally launched as ``python diff_deepseek_layers.py``
    so it lives in ``sys.modules['__main__']``.  Inside the EngineCore
    worker (a fresh process spawned by vLLM) we load the file from disk
    using its absolute path stashed in ``KB_DIFF_SCRIPT_PATH``.
    """
    if "diff_deepseek_layers" in sys.modules:
        return sys.modules["diff_deepseek_layers"]
    path = os.environ.get(_SCRIPT_PATH_ENV)
    if not path or not os.path.isfile(path):
        raise RuntimeError(
            f"{_SCRIPT_PATH_ENV} must point to diff_deepseek_layers.py "
            f"(got {path!r})"
        )
    script_dir = os.path.dirname(path)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    spec = importlib.util.spec_from_file_location("diff_deepseek_layers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["diff_deepseek_layers"] = mod
    spec.loader.exec_module(mod)
    return mod


def install_hooks(model):
    return _load_diff_module()._vllm_install_hooks_worker(model)


def save_dump(model):
    return _load_diff_module()._vllm_save_dump_worker(model)
