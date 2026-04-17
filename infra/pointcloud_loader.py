"""Loader helpers for PointTransformerV3 benchmarking."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch


def is_pointtransv3_model(model_name: str) -> bool:
    name = model_name.lower()
    return "pointtransformerv3" in name or "pointtransv3" in name or "ptv3" in name


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _ptv3_repo_root() -> Path:
    return _project_root() / "third_party" / "PointTransformerV3"


def _install_torch_scatter_shim() -> None:
    if "torch_scatter" in sys.modules:
        return
    mod = types.ModuleType("torch_scatter")

    def segment_csr(src: torch.Tensor, indptr: torch.Tensor, reduce: str = "sum") -> torch.Tensor:
        reduce_map = {"sum": "sum", "mean": "mean", "min": "amin", "max": "amax"}
        return torch.segment_reduce(src, reduce=reduce_map[reduce], offsets=indptr, axis=0)

    mod.segment_csr = segment_csr
    sys.modules["torch_scatter"] = mod


def _load_official_ptv3_class():
    _install_torch_scatter_shim()
    repo_root = _ptv3_repo_root()
    if not repo_root.exists():
        raise FileNotFoundError(
            f"Missing PointTransformerV3 repo at {repo_root}. Clone https://github.com/Pointcept/PointTransformerV3 there."
        )
    package_name = "ptv3_ref"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(repo_root)]
        sys.modules[package_name] = package
    module_name = f"{package_name}.model"
    if module_name in sys.modules:
        return sys.modules[module_name].PointTransformerV3
    spec = importlib.util.spec_from_file_location(
        module_name,
        repo_root / "model.py",
        submodule_search_locations=[str(repo_root)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.PointTransformerV3


def default_ptv3_kwargs(enable_flash: bool = False) -> dict:
    patch = (128, 128, 128, 128, 128)
    dec_patch = (128, 128, 128, 128)
    return {
        "enable_flash": enable_flash,
        "enc_patch_size": patch,
        "dec_patch_size": dec_patch,
        "shuffle_orders": False,
    }


def load_reference_point_model(model_name: str, device: str = "cuda", dtype: torch.dtype = torch.float16, **model_kwargs):
    if not is_pointtransv3_model(model_name):
        raise ValueError(f"Unsupported point model: {model_name}")
    model_cls = _load_official_ptv3_class()
    model = model_cls(**model_kwargs).to(device=device, dtype=dtype).eval()
    return model


def load_ours_point_model(model_name: str, device: str = "cuda", dtype: torch.dtype = torch.float16, **model_kwargs):
    if not is_pointtransv3_model(model_name):
        raise ValueError(f"Unsupported point model: {model_name}")
    from tasks.baseline.L4.pointtransformerv3 import PointTransformerV3

    model = PointTransformerV3(**model_kwargs).to(device=device, dtype=dtype).eval()
    return model
