"""CLI dispatcher for kb-nano e2e benchmarks.

Usage:
    python -m kb_nano.bench.e2e throughput [args...]
    python -m kb_nano.bench.e2e latency [args...]
    python -m kb_nano.bench.e2e serve [args...]
    python -m kb_nano.bench.e2e eval [args...]
"""

from __future__ import annotations

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="python -m kb_nano.bench.e2e",
        description="kb-nano end-to-end benchmarks",
    )
    subparsers = parser.add_subparsers(dest="command", help="Benchmark type")

    from kb_nano.bench.e2e import eval as eval_mod
    from kb_nano.bench.e2e import latency, serve, throughput

    tp = subparsers.add_parser("throughput", help="Offline throughput benchmark")
    throughput.add_cli_args(tp)

    lp = subparsers.add_parser("latency", help="Single-batch latency benchmark")
    latency.add_cli_args(lp)

    sp = subparsers.add_parser("serve", help="Online serving benchmark")
    serve.add_cli_args(sp)

    ep = subparsers.add_parser("eval", help="Candidate kernels vs baseline evaluation")
    eval_mod.add_cli_args(ep)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "throughput":
        throughput.main(args)
    elif args.command == "latency":
        latency.main(args)
    elif args.command == "serve":
        serve.main(args)
    elif args.command == "eval":
        eval_mod.main(args)


if __name__ == "__main__":
    main()
