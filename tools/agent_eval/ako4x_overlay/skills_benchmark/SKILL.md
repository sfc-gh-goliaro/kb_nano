---
name: benchmark
description: Reference for the active benchmark harness — what it IS and how it behaves (the active benchmark is kb-nano/fastkernels, driven through a subprocess entrypoint). Covers the nn.Module task model and its frozen interface contract (class name, forward signature, parameter names, in-place mutation), config.toml structure (`[solution]`/`[build]`/`[benchmark]` tables) and which `[benchmark]` keys are honored vs ignored, the status enum, correctness (kb's own per-dtype tolerances — NOT settable from config.toml), expert-baseline + scoring mechanics, and the no-delegation rule. Invoke whenever you decode a bench status string, hit an unfamiliar config.toml field, see `weight_transfer_incomplete`, suspect "correctness passed but the headline is implausible", or need to know what's frozen before proposing a bench edit — do NOT guess field or status semantics. Bench commands and noise methodology (A/B compare, variance check, drift cancellation) live in the `bench` skill.
---

# Benchmark

The active benchmark harness — what this project's `bench` skill is a frontend for. Reference for **what the benchmark is and how it behaves**, distinct from the `bench` skill which covers **how to drive it from the harness shim**. The active benchmark is **kb-nano / fastkernels**: your solution is an `nn.Module` swapped in for a kb baseline module and scored on kb's own scenarios. Detailed body: `benchmark.md`.

When this skill applies:
- Writing or editing `solution/kernel.py` — the class-name / signature / parameter-name contract is frozen and violating it fails every workload before any numerics run.
- Looking up a `config.toml` field you don't recognize, or wondering why editing `atol` changed nothing.
- Decoding a status string from bench output (`COMPILE_ERROR`, `INCORRECT_NUMERICAL`, `RUNTIME_ERROR`, `TIMEOUT`).
- Reasoning about why correctness "passed" but the headline looks too good.
- Checking what's frozen for bench comparability (scoring / baseline / tolerances) before proposing a harness edit.
- Deciding whether a library call is an allowed building block or banned delegation.
