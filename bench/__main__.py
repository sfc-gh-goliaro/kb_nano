"""
CLI entry point for the kb-nano benchmark suite.

Usage:
    python -m kb_nano.bench --list
    python -m kb_nano.bench --list --level 1

    python -m kb_nano.bench \\
        --target rms_norm \\
        --user-impl path/to/my_kernel.py:MyRMSNorm \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --max-tokens 50

    # Auto-discover from tasks/candidate/:
    python -m kb_nano.bench --target rms_norm
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch.nn as nn

from . import benchmark, list_targets, print_model_operator_map
from .discovery import get as get_target

_CANDIDATE_DIR = Path(__file__).resolve().parent.parent / "tasks" / "candidate"


def _import_from_path(spec_str: str):
    """Import a callable from 'path/to/file.py:name' or 'module.path:name'."""
    if ":" not in spec_str:
        raise ValueError(
            f"user-impl must be in 'path/to/file.py:ClassName' or "
            f"'module.path:ClassName' format. Got: {spec_str!r}"
        )
    path_or_module, name = spec_str.rsplit(":", 1)

    if path_or_module.endswith(".py"):
        spec = importlib.util.spec_from_file_location("_user_impl", path_or_module)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from {path_or_module}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    else:
        mod = importlib.import_module(path_or_module)

    if not hasattr(mod, name):
        raise AttributeError(f"{path_or_module} has no attribute {name!r}")
    return getattr(mod, name)


def _load_candidate(target_name: str):
    """Load the candidate kernel from tasks/candidate/L{level}/{target_name}.py."""
    target = get_target(target_name)
    candidate_file = _CANDIDATE_DIR / f"L{target.level}" / f"{target_name}.py"
    if not candidate_file.is_file():
        return None
    class_name = target.target_cls.__name__
    spec = importlib.util.spec_from_file_location("_candidate_impl", str(candidate_file))
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cls = getattr(mod, class_name, None)
    if cls is None:
        for v in vars(mod).values():
            if isinstance(v, type) and issubclass(v, nn.Module) and v is not nn.Module:
                cls = v
                break
    return cls


def main():
    parser = argparse.ArgumentParser(
        description="kb-nano CUDA kernel benchmark suite",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available benchmark targets and exit",
    )
    parser.add_argument(
        "--map", action="store_true",
        help="Print operators-by-model and models-by-operator mappings",
    )
    parser.add_argument(
        "--level", type=int, default=None,
        help="Filter targets by level (1-4) when listing",
    )
    parser.add_argument(
        "--target", type=str, default=None,
        help="Benchmark target name (e.g. 'rms_norm', 'attention')",
    )
    parser.add_argument(
        "--user-impl", type=str, default=None,
        help="Path to user implementation: 'path/to/file.py:ClassName'",
    )
    parser.add_argument(
        "--model", nargs="+", default=None,
        help="HuggingFace model name(s) to benchmark against",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=50,
        help="Max tokens to generate per prompt (default: 50)",
    )
    parser.add_argument(
        "--tp", type=int, default=1,
        help="Tensor parallelism degree (default: 1)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--num-warmup", type=int, default=1,
        help="Number of warmup iterations (default: 1)",
    )
    parser.add_argument(
        "--num-runs", type=int, default=3,
        help="Number of timed runs to average (default: 3)",
    )
    args = parser.parse_args()

    if args.map:
        print_model_operator_map()
        return

    if args.list:
        targets = list_targets(level=args.level)
        if not targets:
            print("No targets found.")
            return
        print(f"\n{'Level':<7} {'Name':<25} {'Class':<25} Models")
        print("-" * 80)
        for t in targets:
            print(
                f"  L{t.level:<5} {t.name:<25} {t.target_cls.__name__:<25} "
                f"{','.join(t.models)}"
            )
        print(f"\n{len(targets)} targets total.")
        return

    if args.target is None:
        parser.error("--target is required (or use --list to see available targets)")

    if args.user_impl is not None:
        user_impl = _import_from_path(args.user_impl)
    else:
        user_impl = _load_candidate(args.target)
        if user_impl is None:
            parser.error(
                f"No candidate kernel found for {args.target!r} in "
                f"{_CANDIDATE_DIR}. Provide --user-impl or place the kernel "
                f"in tasks/candidate/L<level>/{args.target}.py"
            )

    results = benchmark(
        target_name=args.target,
        user_impl=user_impl,
        models=args.model,
        max_tokens=args.max_tokens,
        tp=args.tp,
        seed=args.seed,
        num_warmup=args.num_warmup,
        num_runs=args.num_runs,
    )

    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")
    for r in results:
        print(f"\n{r.report()}")

    any_fail = any(r.kl_mean > 0.1 for r in results)
    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()
