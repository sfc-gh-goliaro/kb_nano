#!/usr/bin/env python3
"""Alignment and throughput benchmark for recsys baselines.

Real-data defaults:
- `dlrmv2`: Hugging Face `scikit-learn/adult-census-income`
- `lightgcn`: official GroupLens MovieLens 1M ratings

Current reference backends:
- `lightgcn`: `torch_geometric.nn.models.LightGCN`
- `dlrmv2`: `torchrec.models.dlrm.DLRM`

Usage:
    python tests/bench_recsys.py --model dlrmv2
    python tests/bench_recsys.py --model lightgcn
    python tests/bench_recsys.py --model all
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


ADULT_DATASET_ID = "scikit-learn/adult-census-income"
MOVIELENS_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
DEFAULT_DATASET_ROOT = Path("tests") / "data" / "recsys"

ADULT_NUMERIC_COLUMNS = [
    "age",
    "fnlwgt",
    "education.num",
    "capital.gain",
    "capital.loss",
    "hours.per.week",
]
ADULT_CATEGORICAL_COLUMNS = [
    "workclass",
    "education",
    "marital.status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native.country",
]


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
    try:
        from torchrec.models.dlrm import DLRM as TorchRecDLRM
        from torchrec.modules.embedding_configs import EmbeddingBagConfig, PoolingType
        from torchrec.modules.embedding_modules import EmbeddingBagCollection as TorchRecEmbeddingBagCollection
        from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "TorchRec reference benchmark requires optional dependency 'torchrec' "
            "(and its matching fbgemm_gpu build). Install the recsys benchmark "
            "dependencies from README before running alignment/throughput.",
        ) from exc

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


def _load_adult_train_split(dataset_root: Path):
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Real DLRMv2 benchmark requires optional dependency 'datasets'. "
            "Install it with `pip install datasets`.",
        ) from exc

    cache_dir = dataset_root / "adult_census_hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return load_dataset(ADULT_DATASET_ID, split="train", cache_dir=str(cache_dir))


def _normalize_adult_category(value: Any) -> str:
    if value is None:
        return "<missing>"
    normalized = str(value).strip()
    return normalized if normalized else "<missing>"


def _transform_adult_dense_value(value: Any) -> float:
    if value is None:
        return 0.0
    return math.log1p(max(float(value), 0.0))


def _build_adult_categorical_mappings(train_split) -> dict[str, dict[str, int]]:
    mappings: dict[str, dict[str, int]] = {}
    for column in ADULT_CATEGORICAL_COLUMNS:
        mapping = {"<unk>": 0}
        values = sorted({_normalize_adult_category(value) for value in train_split[column]})
        for index, value in enumerate(values, start=1):
            mapping[value] = index
        mappings[column] = mapping
    return mappings


def _adult_rows_to_tensors(
    rows: list[dict[str, Any]],
    *,
    mappings: dict[str, dict[str, int]],
    device: torch.device,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    dense_rows = []
    sparse_columns = [[] for _ in ADULT_CATEGORICAL_COLUMNS]

    for row in rows:
        dense_rows.append([
            _transform_adult_dense_value(row[column])
            for column in ADULT_NUMERIC_COLUMNS
        ])
        for column_index, column in enumerate(ADULT_CATEGORICAL_COLUMNS):
            token = _normalize_adult_category(row[column])
            sparse_columns[column_index].append(mappings[column].get(token, 0))

    dense_features = torch.tensor(dense_rows, dtype=torch.float32, device=device)
    sparse_indices = [
        torch.tensor(values, dtype=torch.long, device=device).unsqueeze(1)
        for values in sparse_columns
    ]
    return dense_features, sparse_indices


def _take_dataset_rows(dataset, *, start: int, count: int) -> list[dict[str, Any]]:
    dataset_size = len(dataset)
    if dataset_size == 0:
        raise ValueError("dataset is empty")
    return [
        dataset[(start + index) % dataset_size]
        for index in range(count)
    ]


def _prepare_dlrm_inputs(
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    if args.dlrm_dataset != "adult":
        raise ValueError(f"unsupported real DLRMv2 dataset: {args.dlrm_dataset}")

    train_split = _load_adult_train_split(args.dataset_root)
    shuffled = train_split.shuffle(seed=args.seed)
    mappings = _build_adult_categorical_mappings(train_split)
    config = DLRMv2Config(
        num_dense_features=len(ADULT_NUMERIC_COLUMNS),
        num_embeddings_per_feature=[
            len(mappings[column])
            for column in ADULT_CATEGORICAL_COLUMNS
        ],
        embedding_dim=64,
        bottom_mlp_dims=[128, 64],
        top_mlp_dims=[128, 64, 1],
        embedding_bag_mode="sum",
    )

    alignment_rows = _take_dataset_rows(
        shuffled,
        start=0,
        count=args.dlrm_batch_size,
    )
    throughput_rows = _take_dataset_rows(
        shuffled,
        start=args.dlrm_batch_size,
        count=args.dlrm_batch_size,
    )

    return {
        "config": config,
        "alignment_batch": _adult_rows_to_tensors(
            alignment_rows,
            mappings=mappings,
            device=device,
        ),
        "throughput_batch": _adult_rows_to_tensors(
            throughput_rows,
            mappings=mappings,
            device=device,
        ),
        "metadata": {
            "dataset": ADULT_DATASET_ID,
            "split": "train",
            "rows": len(train_split),
            "batch_size": args.dlrm_batch_size,
            "bag_size": 1,
            "numeric_features": len(ADULT_NUMERIC_COLUMNS),
            "categorical_features": len(ADULT_CATEGORICAL_COLUMNS),
        },
    }


def _run_dlrm_alignment(
    *,
    device: torch.device,
    prepared_inputs: dict[str, Any],
) -> dict[str, Any]:
    config: DLRMv2Config = prepared_inputs["config"]
    ref = _build_torchrec_dlrm_reference(config, device)
    ours = DLRMv2(config).to(device).eval()
    _copy_dlrm_weights_from_torchrec(ref, ours)
    dense_features, sparse_indices = prepared_inputs["alignment_batch"]
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
    prepared_inputs: dict[str, Any],
    warmup_iters: int,
    measure_iters: int,
) -> dict[str, Any]:
    config: DLRMv2Config = prepared_inputs["config"]
    ref = _build_torchrec_dlrm_reference(config, device)
    ours = DLRMv2(config).to(device).eval()
    _copy_dlrm_weights_from_torchrec(ref, ours)
    dense_features, sparse_indices = prepared_inputs["throughput_batch"]
    kjt = _build_torchrec_kjt(sparse_indices)
    batch_size = dense_features.shape[0]

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
    try:
        from torch_geometric.nn.models import LightGCN as PyGLightGCN
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "LightGCN reference benchmark requires optional dependency "
            "'torch-geometric'. Install the recsys benchmark dependencies "
            "from README before running alignment/throughput.",
        ) from exc
    return PyGLightGCN


def _copy_lightgcn_weights(ours: LightGCN, ref) -> None:
    with torch.no_grad():
        num_users = ours.config.num_users
        ref.embedding.weight[:num_users].copy_(ours.user_embedding.emb.weight)
        ref.embedding.weight[num_users:].copy_(ours.item_embedding.emb.weight)


def _download_file(url: str, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".tmp")
    urllib.request.urlretrieve(url, tmp_path)
    tmp_path.replace(destination)


def _load_movielens_1m(
    dataset_root: Path,
    *,
    min_rating: float,
) -> dict[str, Any]:
    work_dir = dataset_root / "movielens_1m"
    raw_zip = work_dir / "ml-1m.zip"
    extracted_root = work_dir / "raw"
    ratings_path = extracted_root / "ml-1m" / "ratings.dat"
    cache_key = str(min_rating).replace(".", "_")
    processed_path = work_dir / f"processed_min_rating_{cache_key}.pt"

    if processed_path.exists():
        return torch.load(processed_path, map_location="cpu")

    _download_file(MOVIELENS_1M_URL, raw_zip)
    if not ratings_path.exists():
        extracted_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(raw_zip) as archive:
            archive.extractall(extracted_root)

    edge_users = []
    edge_items = []
    total_rows = 0
    with ratings_path.open("r", encoding="latin-1") as handle:
        for line in handle:
            total_rows += 1
            user_id_str, item_id_str, rating_str, _timestamp = line.rstrip().split("::")
            if float(rating_str) < min_rating:
                continue
            edge_users.append(int(user_id_str) - 1)
            edge_items.append(int(item_id_str) - 1)

    if not edge_users:
        raise ValueError(f"MovieLens 1M produced no edges at min_rating={min_rating}")

    payload = {
        "dataset": "MovieLens 1M",
        "source_url": MOVIELENS_1M_URL,
        "num_users": max(edge_users) + 1,
        "num_items": max(edge_items) + 1,
        "num_edges": len(edge_users),
        "num_ratings_total": total_rows,
        "min_rating": min_rating,
        "edge_users": torch.tensor(edge_users, dtype=torch.long),
        "edge_items": torch.tensor(edge_items, dtype=torch.long),
    }
    work_dir.mkdir(parents=True, exist_ok=True)
    torch.save(payload, processed_path)
    return payload


def _sample_lightgcn_pairs(
    edge_users: torch.Tensor,
    edge_items: torch.Tensor,
    *,
    num_pairs: int,
    seed: int,
    offset: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if edge_users.numel() < offset + num_pairs:
        raise ValueError(
            f"requested {offset + num_pairs} positive pairs but dataset only has {edge_users.numel()} edges",
        )
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(edge_users.numel(), generator=generator)
    selection = permutation[offset:offset + num_pairs]
    return edge_users[selection], edge_items[selection]


def _prepare_lightgcn_inputs(
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    if args.lightgcn_dataset != "movielens-1m":
        raise ValueError(f"unsupported real LightGCN dataset: {args.lightgcn_dataset}")

    payload = _load_movielens_1m(
        args.dataset_root,
        min_rating=args.lightgcn_min_rating,
    )
    edge_users = payload["edge_users"]
    edge_items = payload["edge_items"]
    alignment_users, alignment_items = _sample_lightgcn_pairs(
        edge_users,
        edge_items,
        num_pairs=args.lightgcn_num_pairs,
        seed=args.seed,
        offset=0,
    )
    throughput_users, throughput_items = _sample_lightgcn_pairs(
        edge_users,
        edge_items,
        num_pairs=args.lightgcn_num_pairs,
        seed=args.seed,
        offset=args.lightgcn_num_pairs,
    )

    config = LightGCNConfig(
        num_users=payload["num_users"],
        num_items=payload["num_items"],
        embedding_dim=args.lightgcn_embedding_dim,
        num_layers=args.lightgcn_num_layers,
    )
    edge_users_device = edge_users.to(device)
    edge_items_device = edge_items.to(device)
    adjacency = LightGCN.build_adjacency(
        edge_users_device,
        edge_items_device,
        config.num_users,
        config.num_items,
        device=device,
    )

    return {
        "config": config,
        "alignment_batch": (
            edge_users_device,
            edge_items_device,
            alignment_users.to(device),
            alignment_items.to(device),
            adjacency,
        ),
        "throughput_batch": (
            edge_users_device,
            edge_items_device,
            throughput_users.to(device),
            throughput_items.to(device),
            adjacency,
        ),
        "metadata": {
            "dataset": "movielens-1m",
            "source": MOVIELENS_1M_URL,
            "users": payload["num_users"],
            "items": payload["num_items"],
            "edges": payload["num_edges"],
            "pairs": args.lightgcn_num_pairs,
            "min_rating": args.lightgcn_min_rating,
        },
    }


def _run_lightgcn_alignment(
    *,
    device: torch.device,
    prepared_inputs: dict[str, Any],
) -> dict[str, Any]:
    PyGLightGCN = _load_pyg_lightgcn()
    config: LightGCNConfig = prepared_inputs["config"]
    ours = LightGCN(config).to(device).eval()
    ref = PyGLightGCN(
        num_nodes=config.num_users + config.num_items,
        embedding_dim=config.embedding_dim,
        num_layers=config.num_layers,
        alpha=1.0 / (config.num_layers + 1),
        normalize=False,
    ).to(device).eval()
    _copy_lightgcn_weights(ours, ref)

    _edge_users, _edge_items, user_ids, item_ids, adjacency = prepared_inputs["alignment_batch"]
    edge_label_index = torch.stack([user_ids, item_ids + config.num_users], dim=0)

    with torch.inference_mode():
        ours_user, ours_item = ours.get_user_item_embeddings(adjacency)
        ref_all = ref.get_embedding(adjacency)
        ref_user = ref_all[:config.num_users]
        ref_item = ref_all[config.num_users:]
        ours_scores = ours(user_ids, item_ids, adjacency)
        ref_scores = ref(adjacency, edge_label_index=edge_label_index)

    return {
        "reference": "torch_geometric.nn.models.LightGCN",
        "user_embeddings": _tensor_metrics(ours_user, ref_user),
        "item_embeddings": _tensor_metrics(ours_item, ref_item),
        "scores": _tensor_metrics(ours_scores, ref_scores),
    }


def _run_lightgcn_throughput(
    *,
    device: torch.device,
    prepared_inputs: dict[str, Any],
    warmup_iters: int,
    measure_iters: int,
) -> dict[str, Any]:
    PyGLightGCN = _load_pyg_lightgcn()
    config: LightGCNConfig = prepared_inputs["config"]
    ours = LightGCN(config).to(device).eval()
    ref = PyGLightGCN(
        num_nodes=config.num_users + config.num_items,
        embedding_dim=config.embedding_dim,
        num_layers=config.num_layers,
        alpha=1.0 / (config.num_layers + 1),
        normalize=False,
    ).to(device).eval()
    _copy_lightgcn_weights(ours, ref)

    _edge_users, _edge_items, user_ids, item_ids, adjacency = prepared_inputs["throughput_batch"]
    edge_label_index = torch.stack([user_ids, item_ids + config.num_users], dim=0)

    with torch.inference_mode():
        ours_metrics = _benchmark_forward(
            lambda: ours(user_ids, item_ids, adjacency),
            device=device,
            warmup_iters=warmup_iters,
            measure_iters=measure_iters,
            items_per_iter=user_ids.numel(),
            metric_name="pairs_per_second",
        )
        ref_metrics = _benchmark_forward(
            lambda: ref(adjacency, edge_label_index=edge_label_index),
            device=device,
            warmup_iters=warmup_iters,
            measure_iters=measure_iters,
            items_per_iter=user_ids.numel(),
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
    metadata = result.get("data")
    if metadata:
        parts = [f"dataset={metadata['dataset']}"]
        if "split" in metadata:
            parts.append(f"split={metadata['split']}")
        if "batch_size" in metadata:
            parts.append(f"batch={metadata['batch_size']}")
        if "bag_size" in metadata:
            parts.append(f"bag={metadata['bag_size']}")
        if "edges" in metadata:
            parts.append(f"edges={metadata['edges']}")
        if "pairs" in metadata:
            parts.append(f"pairs={metadata['pairs']}")
        print(f"  data: {', '.join(parts)}")

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
        prepared_inputs = _prepare_dlrm_inputs(args, device)
        result["data"] = prepared_inputs["metadata"]
        if not args.skip_alignment:
            result["alignment"] = _run_dlrm_alignment(
                device=device,
                prepared_inputs=prepared_inputs,
            )
        if not args.skip_throughput:
            result["throughput"] = _run_dlrm_throughput(
                device=device,
                prepared_inputs=prepared_inputs,
                warmup_iters=args.warmup_iters,
                measure_iters=args.measure_iters,
            )
    elif model_name == "lightgcn":
        prepared_inputs = _prepare_lightgcn_inputs(args, device)
        result["data"] = prepared_inputs["metadata"]
        if not args.skip_alignment:
            result["alignment"] = _run_lightgcn_alignment(
                device=device,
                prepared_inputs=prepared_inputs,
            )
        if not args.skip_throughput:
            result["throughput"] = _run_lightgcn_throughput(
                device=device,
                prepared_inputs=prepared_inputs,
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
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Local cache root for real benchmark datasets.",
    )
    parser.add_argument("--device", default="auto", help="Device to use (default: auto)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--warmup-iters", type=int, default=100, help="Warmup iterations")
    parser.add_argument("--measure-iters", type=int, default=11000, help="Measured iterations")
    parser.add_argument("--skip-alignment", action="store_true", help="Skip numerical alignment")
    parser.add_argument("--skip-throughput", action="store_true", help="Skip throughput benchmark")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write results.json into. Defaults to tests/results/<GPU>/<model>_recsys "
             "for single-model runs.",
    )

    parser.add_argument(
        "--dlrm-dataset",
        choices=["adult"],
        default="adult",
        help="Real dataset used by DLRMv2 benchmark.",
    )
    parser.add_argument(
        "--dlrm-batch-size",
        type=int,
        default=16384,
        help="Per-iteration batch size for the real Adult benchmark.",
    )

    parser.add_argument(
        "--lightgcn-dataset",
        choices=["movielens-1m"],
        default="movielens-1m",
        help="Real dataset used by LightGCN benchmark.",
    )
    parser.add_argument("--lightgcn-min-rating", type=float, default=4.0)
    parser.add_argument("--lightgcn-embedding-dim", type=int, default=64)
    parser.add_argument("--lightgcn-num-layers", type=int, default=3)
    parser.add_argument(
        "--lightgcn-num-pairs",
        type=int,
        default=131072,
        help="Number of real positive (user, item) pairs scored per iteration.",
    )
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
        "dataset_root": str(args.dataset_root),
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
