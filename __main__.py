"""CLI entry point for fastkernels.

Dispatches to the appropriate subcommand:

    fastkernels bench [args...]
    fastkernels capture [args...]
    fastkernels validate [args...]
    fastkernels list [args...]
    fastkernels create-stubs [args...]
"""

from __future__ import annotations

import sys


def main():
    if len(sys.argv) < 2:
        _print_usage()
        sys.exit(1)

    command = sys.argv[1]
    # Remove the subcommand from argv so the downstream parsers see clean args
    sys.argv = [f"fastkernels {command}"] + sys.argv[2:]

    if command == "bench":
        from fastkernels.bench_cli import main as bench_main
        raise SystemExit(bench_main(sys.argv[1:]))
    elif command == "capture":
        from fastkernels.capture import main as capture_main
        raise SystemExit(capture_main(sys.argv[1:]))
    elif command == "validate":
        from fastkernels.validate import main as validate_main
        raise SystemExit(validate_main(sys.argv[1:]))
    elif command == "list":
        from fastkernels.list import main as list_main
        list_main(sys.argv[1:])
    elif command == "create-stubs":
        from fastkernels.agent.create_stubs import main as stubs_main
        stubs_main()
    elif command in ("-h", "--help", "help"):
        _print_usage()
    else:
        print(f"Unknown command: {command}")
        _print_usage()
        sys.exit(1)


def _print_usage():
    print("Usage: fastkernels <command> [args...]")
    print()
    print("Commands:")
    print("  list             List architectures, workloads, and model/operator maps")
    print("  capture          Capture runtime operator init/forward metadata")
    print("  bench            Benchmark candidate kernels from captured metadata")
    print("  validate         Run per-model SOTA validation harnesses")
    print("  create-stubs     Create skeleton replacement modules")
    print()
    print("Run 'fastkernels <command> --help' for command-specific options.")


if __name__ == "__main__":
    main()
