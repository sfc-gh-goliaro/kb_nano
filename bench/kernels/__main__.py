"""CLI entry point for the kb-nano kernel benchmark suite.

Usage:
    # Test all operators that have candidates
    python -m kb_nano.bench.kernels

    # Test a specific operator
    python -m kb_nano.bench.kernels --target rms_norm

    # Filter to a specific model family
    python -m kb_nano.bench.kernels --target rms_norm --model llama

    # Filter to TP=4 scenarios only
    python -m kb_nano.bench.kernels --target rms_norm --tp 4

    # Restrict to LLM category
    python -m kb_nano.bench.kernels --target rms_norm --category llm

    # List available targets
    python -m kb_nano.bench.kernels --list
    python -m kb_nano.bench.kernels --list --level 1
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

from kb_nano.infra.kernel_swapper import (
    _CANDIDATE_DIR,
    list_targets,
    load_candidate,
    print_model_operator_map,
)

from .result import KernelBenchResult
from .runner import run_all_kernel_benchmarks, run_kernel_benchmark

_DEFAULT_OUTPUT = "bench/results/kernels.json"


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


def main():
    parser = argparse.ArgumentParser(
        description="kb-nano CUDA kernel benchmark suite (isolated forward() testing)",
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
        help="Benchmark target name (e.g. 'rms_norm', 'attention'). "
             "Omit to test all operators with candidates.",
    )
    parser.add_argument(
        "--user-impl", type=str, default=None,
        help="Path to user implementation: 'path/to/file.py:ClassName'",
    )
    parser.add_argument(
        "--model", nargs="+", default=None,
        help="Filter scenarios by model key prefix (e.g. 'llama31' 'mixtral')",
    )
    parser.add_argument(
        "--tp", type=int, nargs="+", default=None,
        help="Filter scenarios by TP degree(s) (e.g. 1 4)",
    )
    parser.add_argument(
        "--category", type=str, default=None,
        help="Filter scenarios by category (e.g. 'llm', 'vision')",
    )
    parser.add_argument(
        "--num-warmup", type=int, default=10,
        help="Number of warmup iterations (default: 10)",
    )
    parser.add_argument(
        "--num-runs", type=int, default=100,
        help="Number of timed runs for median (default: 100)",
    )
    parser.add_argument(
        "--output-json", type=str, default=None,
        help=f"Path to save JSON results (default: {_DEFAULT_OUTPUT})",
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

    output_path = args.output_json or _DEFAULT_OUTPUT

    if args.target is not None:
        if args.user_impl is not None:
            user_impl = _import_from_path(args.user_impl)
        else:
            user_impl = load_candidate(args.target)
            if user_impl is None:
                parser.error(
                    f"No candidate kernel found for {args.target!r} in "
                    f"{_CANDIDATE_DIR}. Provide --user-impl or place the kernel "
                    f"in tasks/candidate/L<level>/{args.target}.py"
                )

        op_result = run_kernel_benchmark(
            target_name=args.target,
            user_impl=user_impl,
            models=args.model,
            tp=args.tp,
            category=args.category,
            num_warmup=args.num_warmup,
            num_runs=args.num_runs,
        )

        result = KernelBenchResult(operators=[op_result])
        result.compute_aggregates()
    else:
        result = run_all_kernel_benchmarks(
            models=args.model,
            tp=args.tp,
            category=args.category,
            num_warmup=args.num_warmup,
            num_runs=args.num_runs,
        )

    result.print_table(single_target=(args.target is not None))
    result.save_json(output_path)
    print(f"\n  Results saved to: {output_path}")

    sys.exit(0 if result.all_passed() else 1)


if __name__ == "__main__":
    main()
