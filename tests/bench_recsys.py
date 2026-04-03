#!/usr/bin/env python3
"""Alignment and throughput benchmark for recsys baselines.

Current reference backends:
- `lightgcn`: `torch_geometric.nn.models.LightGCN`
- `dlrmv2`: `torchrec.models.dlrm.DLRM`

Usage:
    python tests/bench_recsys.py --model lightgcn
    python tests/bench_recsys.py --model dlrmv2
    python tests/bench_recsys.py --model all
    python tests/bench_recsys.py --model lightgcn --skip-throughput
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _bootstrap_local_package() -> None:
    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "kb_nano",
        root / "__init__.py",
        submodule_search_locations=[str(root)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["kb_nano"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)


_bootstrap_local_package()

from kb_nano.tasks.baseline.L4.dlrmv2 import DLRMv2, DLRMv2Config
from kb_nano.tasks.baseline.L4.lightgcn import LightGCN, LightGCNConfig


def _load_torchrec_dlrm():
    from torchrec.models.dlrm import DLRM as TorchRecDLRM
    from torchrec.modules.embedding_configs import EmbeddingBagConfig, PoolingType
    from torchrec.modules.embedding_modules import EmbeddingBagCollection as TorchRecEmbeddingBagCollection
    from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

    return TorchRecDLRM, EmbeddingBagConfig, PoolingType, TorchRecEmbeddingBagCollection, KeyedJaggedTensor


def _build_torchrec_dlrm_reference(config: DLRMv2Config, device: torch.device):
    TorchRecDLRM, EmbeddingBagConfig, PoolingType, TorchRecEmbeddingBagCollection, _ = _load_torchrec_dlrm()
    pooling_by_mode = {
        "sum": PoolingType.SUM,
        "mean": PoolingType.MEAN,
    }
    if config.embedding_bag_mode not in pooling_by_mode:
        raise ValueError(f"unsupported TorchRec pooling mode: {config.embedding_bag_mode}")

    tables = [
        EmbeddingBagConfig(
            name=f"t{index}",
            embedding_dim=config.embedding_dim,
            num_embeddings=num_embeddings,
            feature_names=[f"f{index}"],
            pooling=pooling_by_mode[config.embedding_bag_mode],
        )
        for index, num_embeddings in enumerate(config.num_embeddings_per_feature)
    ]
    ebc = TorchRecEmbeddingBagCollection(tables=tables, device=device)
    return TorchRecDLRM(
        embedding_bag_collection=ebc,
        dense_in_features=config.num_dense_features,
        dense_arch_layer_sizes=config.bottom_mlp_dims,
        over_arch_layer_sizes=config.top_mlp_dims,
        dense_device=device,
    ).to(device).eval()


def _build_torchrec_kjt(sparse_indices: list[torch.Tensor]):
    _, _, _, _, KeyedJaggedTensor = _load_torchrec_dlrm()
    batch_size = sparse_indices[0].shape[0]
    keys = [f"f{index}" for index in range(len(sparse_indices))]
    flat_values = []
    offsets = [0]
    for indices in sparse_indices:
        if indices.ndim != 2:
            raise ValueError("TorchRec reference expects 2D sparse indices")
        bag_size = indices.shape[1]
        flat_values.append(indices.reshape(-1))
        for _ in range(batch_size):
            offsets.append(offsets[-1] + bag_size)
    values = torch.cat(flat_values, dim=0)
    offsets_tensor = torch.tensor(offsets, device=values.device, dtype=torch.long)
    return KeyedJaggedTensor.from_offsets_sync(
        keys=keys,
        values=values,
        offsets=offsets_tensor,
        stride=batch_size,
    )


def _safe_cosine_similarity(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    lhs_flat = lhs.reshape(-1).float()
    rhs_flat = rhs.reshape(-1).float()
    lhs_norm = torch.linalg.vector_norm(lhs_flat)
    rhs_norm = torch.linalg.vector_norm(rhs_flat)
    if lhs_norm.item() == 0.0 and rhs_norm.item() == 0.0:
        return 1.0
    if lhs_norm.item() == 0.0 or rhs_norm.item() == 0.0:
        return 0.0
    return float(F.cosine_similarity(lhs_flat.unsqueeze(0), rhs_flat.unsqueeze(0)).item())


def _tensor_metrics(lhs: torch.Tensor, rhs: torch.Tensor) -> dict[str, float]:
    lhs = lhs.detach()
    rhs = rhs.detach()
    diff = (lhs - rhs).abs().float()
    return {
        "cosine": _safe_cosine_similarity(lhs, rhs),
        "mean_abs_diff": float(diff.mean().item()),
        "max_abs_diff": float(diff.max().item()),
    }


def _maybe_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _benchmark_forward(
    fn,
    *,
    device: torch.device,
    warmup_iters: int,
    measure_iters: int,
    items_per_iter: int,
    metric_name: str,
) -> dict[str, float]:
    for _ in range(warmup_iters):
        fn()
    _maybe_sync(device)

    latencies = []
    for _ in range(measure_iters):
        _maybe_sync(device)
        start = time.perf_counter()
        fn()
        _maybe_sync(device)
        latencies.append(time.perf_counter() - start)

    total_elapsed = sum(latencies)
    total_items = items_per_iter * measure_iters
    return {
        metric_name: total_items / total_elapsed,
        "latency_ms_avg": (total_elapsed / measure_iters) * 1000.0,
        "latency_ms_p50": float(torch.tensor(latencies).median().item() * 1000.0),
    }


def _detect_gpu_name() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
        ).strip().splitlines()[0]
        for tag in ("B200", "B100", "H200", "H100", "A100", "A10G", "L40S", "L40", "L4"):
            if tag in out:
                return tag
        return out.split()[-1]
    except Exception:
        return "unknown"


def _auto_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _copy_dlrm_weights_from_torchrec(ref, ours: DLRMv2) -> None:
    with torch.no_grad():
        for index, num_embeddings in enumerate(ours.config.num_embeddings_per_feature):
            start = int(ours.embedding_bag_collection.table_offsets[index].item())
            end = start + num_embeddings
            ours.embedding_bag_collection.embedding_bag.emb.weight[start:end].copy_(
                ref.sparse_arch.embedding_bag_collection.embedding_bags[f"t{index}"].weight,
            )
        for ref_layer, ours_layer in zip(
            ref.dense_arch.model._mlp,
            ours.bottom_mlp.layers,
            strict=True,
        ):
            ours_layer.weight.copy_(ref_layer._linear.weight)
            ours_layer.bias.copy_(ref_layer._linear.bias)
        for ref_layer, ours_layer in zip(
            ref.over_arch.model[0]._mlp,
            ours.top_mlp.layers[:-1],
            strict=True,
        ):
            ours_layer.weight.copy_(ref_layer._linear.weight)
            ours_layer.bias.copy_(ref_layer._linear.bias)
        ours.top_mlp.layers[-1].weight.copy_(ref.over_arch.model[1].weight)
        ours.top_mlp.layers[-1].bias.copy_(ref.over_arch.model[1].bias)


def _generate_dlrm_inputs(
    config: DLRMv2Config,
    *,
    batch_size: int,
    bag_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    dense_features = torch.randn(batch_size, config.num_dense_features, device=device)
    sparse_indices = [
        torch.randint(0, table_size, (batch_size, bag_size), device=device, dtype=torch.long)
        for table_size in config.num_embeddings_per_feature
    ]
    return dense_features, sparse_indices


def _run_dlrm_alignment(
    *,
    device: torch.device,
    batch_size: int,
    bag_size: int,
) -> dict[str, Any]:
    config = DLRMv2Config()
    ref = _build_torchrec_dlrm_reference(config, device)
    ours = DLRMv2(config).to(device).eval()
    _copy_dlrm_weights_from_torchrec(ref, ours)
    dense_features, sparse_indices = _generate_dlrm_inputs(
        config,
        batch_size=batch_size,
        bag_size=bag_size,
        device=device,
    )
    kjt = _build_torchrec_kjt(sparse_indices)

    with torch.inference_mode():
        ours_dense = ours.bottom_mlp(dense_features)
        ref_dense = ref.dense_arch(dense_features)
        ours_sparse = ours.embedding_bag_collection(sparse_indices)
        ref_sparse = ref.sparse_arch(kjt)
        ours_interacted = ours.interaction(ours_dense, ours_sparse)
        ref_interacted = ref.inter_arch(ref_dense, ref_sparse)
        ours_logits = ours.top_mlp(ours_interacted)
        ref_logits = ref.over_arch(ref_interacted)

    return {
        "reference": "torchrec.models.dlrm.DLRM",
        "dense_embedding": _tensor_metrics(ours_dense, ref_dense),
        "sparse_embeddings": _tensor_metrics(ours_sparse, ref_sparse),
        "interaction": _tensor_metrics(ours_interacted, ref_interacted),
        "logits": _tensor_metrics(ours_logits, ref_logits),
    }


def _run_dlrm_throughput(
    *,
    device: torch.device,
    batch_size: int,
    bag_size: int,
    warmup_iters: int,
    measure_iters: int,
) -> dict[str, Any]:
    config = DLRMv2Config()
    ref = _build_torchrec_dlrm_reference(config, device)
    ours = DLRMv2(config).to(device).eval()
    _copy_dlrm_weights_from_torchrec(ref, ours)
    dense_features, sparse_indices = _generate_dlrm_inputs(
        config,
        batch_size=batch_size,
        bag_size=bag_size,
        device=device,
    )
    kjt = _build_torchrec_kjt(sparse_indices)

    with torch.inference_mode():
        ours_metrics = _benchmark_forward(
            lambda: ours(dense_features, sparse_indices),
            device=device,
            warmup_iters=warmup_iters,
            measure_iters=measure_iters,
            items_per_iter=batch_size,
            metric_name="samples_per_second",
        )
        ref_metrics = _benchmark_forward(
            lambda: ref(dense_features, kjt),
            device=device,
            warmup_iters=warmup_iters,
            measure_iters=measure_iters,
            items_per_iter=batch_size,
            metric_name="samples_per_second",
        )

    ours_sps = ours_metrics["samples_per_second"]
    ref_sps = ref_metrics["samples_per_second"]
    return {
        "reference": "torchrec.models.dlrm.DLRM",
        "ours": ours_metrics,
        "reference_metrics": ref_metrics,
        "ratio_vs_reference": ours_sps / ref_sps if ref_sps > 0 else math.nan,
    }


def _load_pyg_lightgcn():
    from torch_geometric.nn.models import LightGCN as PyGLightGCN
    return PyGLightGCN


def _copy_lightgcn_weights(ours: LightGCN, ref) -> None:
    with torch.no_grad():
        num_users = ours.config.num_users
        ref.embedding.weight[:num_users].copy_(ours.user_embedding.emb.weight)
        ref.embedding.weight[num_users:].copy_(ours.item_embedding.emb.weight)


def _generate_lightgcn_inputs(
    config: LightGCNConfig,
    *,
    num_edges: int,
    num_pairs: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    edge_users = torch.randint(0, config.num_users, (num_edges,), device=device, dtype=torch.long)
    edge_items = torch.randint(0, config.num_items, (num_edges,), device=device, dtype=torch.long)
    user_ids = torch.randint(0, config.num_users, (num_pairs,), device=device, dtype=torch.long)
    item_ids = torch.randint(0, config.num_items, (num_pairs,), device=device, dtype=torch.long)
    adjacency = LightGCN.build_adjacency(
        edge_users,
        edge_items,
        config.num_users,
        config.num_items,
        device=device,
    )
    return edge_users, edge_items, user_ids, item_ids, adjacency


def _run_lightgcn_alignment(
    *,
    device: torch.device,
    num_users: int,
    num_items: int,
    embedding_dim: int,
    num_layers: int,
    num_edges: int,
    num_pairs: int,
) -> dict[str, Any]:
    PyGLightGCN = _load_pyg_lightgcn()
    config = LightGCNConfig(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
    )
    ours = LightGCN(config).to(device).eval()
    ref = PyGLightGCN(
        num_nodes=config.num_users + config.num_items,
        embedding_dim=config.embedding_dim,
        num_layers=config.num_layers,
        alpha=1.0 / (config.num_layers + 1),
        normalize=False,
    ).to(device).eval()
    _copy_lightgcn_weights(ours, ref)

    _, _, user_ids, item_ids, adjacency = _generate_lightgcn_inputs(
        config,
        num_edges=num_edges,
        num_pairs=num_pairs,
        device=device,
    )
    adjacency_csr = adjacency
    edge_label_index = torch.stack([user_ids, item_ids + config.num_users], dim=0)

    with torch.inference_mode():
        ours_user, ours_item = ours.get_user_item_embeddings(adjacency)
        ref_all = ref.get_embedding(adjacency_csr)
        ref_user = ref_all[:config.num_users]
        ref_item = ref_all[config.num_users:]
        ours_scores = ours(user_ids, item_ids, adjacency)
        ref_scores = ref(adjacency_csr, edge_label_index=edge_label_index)

    return {
        "reference": "torch_geometric.nn.models.LightGCN",
        "user_embeddings": _tensor_metrics(ours_user, ref_user),
        "item_embeddings": _tensor_metrics(ours_item, ref_item),
        "scores": _tensor_metrics(ours_scores, ref_scores),
    }


def _run_lightgcn_throughput(
    *,
    device: torch.device,
    num_users: int,
    num_items: int,
    embedding_dim: int,
    num_layers: int,
    num_edges: int,
    num_pairs: int,
    warmup_iters: int,
    measure_iters: int,
) -> dict[str, Any]:
    PyGLightGCN = _load_pyg_lightgcn()
    config = LightGCNConfig(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
    )
    ours = LightGCN(config).to(device).eval()
    ref = PyGLightGCN(
        num_nodes=config.num_users + config.num_items,
        embedding_dim=config.embedding_dim,
        num_layers=config.num_layers,
        alpha=1.0 / (config.num_layers + 1),
        normalize=False,
    ).to(device).eval()
    _copy_lightgcn_weights(ours, ref)

    _, _, user_ids, item_ids, adjacency = _generate_lightgcn_inputs(
        config,
        num_edges=num_edges,
        num_pairs=num_pairs,
        device=device,
    )
    adjacency_csr = adjacency
    edge_label_index = torch.stack([user_ids, item_ids + config.num_users], dim=0)

    with torch.inference_mode():
        ours_metrics = _benchmark_forward(
            lambda: ours(user_ids, item_ids, adjacency),
            device=device,
            warmup_iters=warmup_iters,
            measure_iters=measure_iters,
            items_per_iter=num_pairs,
            metric_name="pairs_per_second",
        )
        ref_metrics = _benchmark_forward(
            lambda: ref(adjacency_csr, edge_label_index=edge_label_index),
            device=device,
            warmup_iters=warmup_iters,
            measure_iters=measure_iters,
            items_per_iter=num_pairs,
            metric_name="pairs_per_second",
        )

    ours_pps = ours_metrics["pairs_per_second"]
    ref_pps = ref_metrics["pairs_per_second"]
    return {
        "reference": "torch_geometric.nn.models.LightGCN",
        "ours": ours_metrics,
        "reference_metrics": ref_metrics,
        "ratio_vs_reference": ours_pps / ref_pps if ref_pps > 0 else math.nan,
    }


def _summarize_model_result(name: str, result: dict[str, Any]) -> None:
    print(f"\n== {name} ==")
    alignment = result.get("alignment")
    if alignment:
        print(f"reference: {alignment['reference']}")
        for key, value in alignment.items():
            if isinstance(value, dict) and "cosine" in value:
                print(
                    f"  {key}: cosine={value['cosine']:.6f}, "
                    f"mae={value['mean_abs_diff']:.6e}, "
                    f"max={value['max_abs_diff']:.6e}"
                )
        if "note" in alignment:
            print(f"  note: {alignment['note']}")
    throughput = result.get("throughput")
    if throughput:
        metric_name = "samples_per_second" if name == "dlrmv2" else "pairs_per_second"
        ours = throughput["ours"][metric_name]
        ref = throughput["reference_metrics"][metric_name]
        print(
            f"  throughput: ours={ours:.2f}, reference={ref:.2f}, "
            f"ratio={throughput['ratio_vs_reference']:.2f}x"
        )


def _run_model(args: argparse.Namespace, model_name: str, device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {
        "model": model_name,
        "device": str(device),
    }
    if model_name == "dlrmv2":
        if not args.skip_alignment:
            result["alignment"] = _run_dlrm_alignment(
                device=device,
                batch_size=args.dlrm_batch_size,
                bag_size=args.dlrm_bag_size,
            )
        if not args.skip_throughput:
            result["throughput"] = _run_dlrm_throughput(
                device=device,
                batch_size=args.dlrm_batch_size,
                bag_size=args.dlrm_bag_size,
                warmup_iters=args.warmup_iters,
                measure_iters=args.measure_iters,
            )
    elif model_name == "lightgcn":
        if not args.skip_alignment:
            result["alignment"] = _run_lightgcn_alignment(
                device=device,
                num_users=args.lightgcn_num_users,
                num_items=args.lightgcn_num_items,
                embedding_dim=args.lightgcn_embedding_dim,
                num_layers=args.lightgcn_num_layers,
                num_edges=args.lightgcn_num_edges,
                num_pairs=args.lightgcn_num_pairs,
            )
        if not args.skip_throughput:
            result["throughput"] = _run_lightgcn_throughput(
                device=device,
                num_users=args.lightgcn_num_users,
                num_items=args.lightgcn_num_items,
                embedding_dim=args.lightgcn_embedding_dim,
                num_layers=args.lightgcn_num_layers,
                num_edges=args.lightgcn_num_edges,
                num_pairs=args.lightgcn_num_pairs,
                warmup_iters=args.warmup_iters,
                measure_iters=args.measure_iters,
            )
    else:
        raise ValueError(f"unsupported model: {model_name}")
    return result


def _default_output_dir(model: str) -> Path:
    gpu = _detect_gpu_name()
    return Path("tests") / "results" / gpu / f"{model}_recsys"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=["dlrmv2", "lightgcn", "all"],
        default="all",
        help="Which model baseline to benchmark.",
    )
    parser.add_argument("--device", default="auto", help="Device to use (default: auto)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--warmup-iters", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--measure-iters", type=int, default=20, help="Measured iterations")
    parser.add_argument("--skip-alignment", action="store_true", help="Skip numerical alignment")
    parser.add_argument("--skip-throughput", action="store_true", help="Skip throughput benchmark")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write results.json into. Defaults to tests/results/<GPU>/<model>_recsys "
             "for single-model runs.",
    )

    parser.add_argument("--dlrm-batch-size", type=int, default=1024)
    parser.add_argument("--dlrm-bag-size", type=int, default=4)

    parser.add_argument("--lightgcn-num-users", type=int, default=4096)
    parser.add_argument("--lightgcn-num-items", type=int, default=8192)
    parser.add_argument("--lightgcn-embedding-dim", type=int, default=64)
    parser.add_argument("--lightgcn-num-layers", type=int, default=3)
    parser.add_argument("--lightgcn-num-edges", type=int, default=32768)
    parser.add_argument("--lightgcn-num-pairs", type=int, default=8192)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)

    device = _auto_device() if args.device == "auto" else torch.device(args.device)
    models = ["dlrmv2", "lightgcn"] if args.model == "all" else [args.model]

    results = {
        "seed": args.seed,
        "device": str(device),
        "gpu": _detect_gpu_name() if device.type == "cuda" else None,
        "models": {},
    }

    for model_name in models:
        model_result = _run_model(args, model_name, device)
        results["models"][model_name] = model_result
        _summarize_model_result(model_name, model_result)

    output_dir = args.output_dir
    if output_dir is None and len(models) == 1:
        output_dir = _default_output_dir(models[0])
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "results.json"
        with output_path.open("w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote results to {output_path}")


if __name__ == "__main__":
    main()
