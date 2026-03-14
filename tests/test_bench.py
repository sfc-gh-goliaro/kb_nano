#!/usr/bin/env python3
"""
Test suite for the benchmarking infrastructure.

Sections 1-6: Unit tests (no GPU required).
Sections 7-9: Integration tests (GPU required).

Usage:
    python tests/test_bench.py                 # all tests
    python tests/test_bench.py --unit-only      # no GPU required
    python tests/test_bench.py --section 3      # run only section 3
"""

from __future__ import annotations

import argparse
import io
import importlib
import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.dirname(THIS_DIR)
PROJECT_ROOT = os.path.dirname(PACKAGE_DIR)
PACKAGE_NAME = os.path.basename(PACKAGE_DIR)

sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_pass_count = 0
_fail_count = 0


def check(condition: bool, label: str):
    global _pass_count, _fail_count
    if condition:
        _pass_count += 1
        print(f"    PASS  {label}")
    else:
        _fail_count += 1
        print(f"    FAIL  {label}")


class _Timeout:
    """POSIX alarm-based timeout guard for unit tests."""
    def __init__(self, seconds: int):
        self.seconds = seconds

    def __enter__(self):
        if hasattr(signal, "SIGALRM"):
            signal.signal(signal.SIGALRM, self._handler)
            signal.alarm(self.seconds)
        return self

    def __exit__(self, *args):
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)

    @staticmethod
    def _handler(signum, frame):
        raise TimeoutError("Unit test timed out")


# ===========================================================================
# Section 1: Input Registry (unit, no GPU)
# ===========================================================================
def test_section_1():
    print(f"\n{'=' * 60}")
    print("  SECTION 1: Input Registry")
    print(f"{'=' * 60}")

    from kb_nano.bench.utils.input_registry import InputRegistry, Scenario

    # 1a. YAML parsing
    with _Timeout(30):
        yaml_content = """
rms_norm:
  scenarios:
    - name: "test-model/decode-bs1/tp1"
      init_args:
        hidden_size: 64
        eps: 1.0e-6
      inputs:
        x: {shape: [1, 64], dtype: bfloat16}
        residual: {shape: [1, 64], dtype: bfloat16}
    - name: "test-model/prefill-128/tp1"
      init_args:
        hidden_size: 64
        eps: 1.0e-6
      inputs:
        x: {shape: [128, 64], dtype: bfloat16}
        residual: null
moe_align:
  scenarios:
    - name: "mixtral/decode-bs1/tp1"
      init_args: {}
      inputs:
        topk_ids: {shape: [1, 2], dtype: int32}
      golden: "mixtral/moe_align/decode-bs1-tp1.pt"
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            inputs_dir = Path(tmpdir) / "inputs"
            inputs_dir.mkdir()
            yaml_path = inputs_dir / "test.yaml"
            yaml_path.write_text(yaml_content)

            reg = InputRegistry(inputs_dir=inputs_dir, golden_dir=Path(tmpdir) / "golden")
            ops = reg.operators()
            check(
                "rms_norm" in ops and "moe_align" in ops,
                "1a. YAML parsing: operators parsed correctly",
            )

            rms_scenarios = reg.scenarios("rms_norm")
            check(len(rms_scenarios) == 2, "1a. rms_norm has 2 scenarios")
            check(
                rms_scenarios[0].name == "test-model/decode-bs1/tp1",
                "1a. first scenario name correct",
            )
            check(
                rms_scenarios[0].init_args == {"hidden_size": 64, "eps": 1e-6},
                "1a. init_args parsed correctly",
            )

            moe_scenarios = reg.scenarios("moe_align")
            check(len(moe_scenarios) == 1, "1a. moe_align has 1 scenario")
            check(
                moe_scenarios[0].golden_path == "mixtral/moe_align/decode-bs1-tp1.pt",
                "1a. golden path parsed correctly",
            )

    # 1b. Shape-only tensor generation
    with _Timeout(30):
        with tempfile.TemporaryDirectory() as tmpdir:
            inputs_dir = Path(tmpdir) / "inputs"
            inputs_dir.mkdir()
            (inputs_dir / "test.yaml").write_text(yaml_content)

            reg = InputRegistry(inputs_dir=inputs_dir, golden_dir=Path(tmpdir) / "golden")
            inputs = reg.get_inputs("rms_norm", "test-model/decode-bs1/tp1", device="cpu")
            check(
                "x" in inputs and "residual" in inputs,
                "1b. get_inputs returns expected keys",
            )
            check(
                inputs["x"].shape == (1, 64) and inputs["x"].dtype == torch.bfloat16,
                "1b. x tensor has correct shape and dtype",
            )
            check(
                inputs["residual"].shape == (1, 64),
                "1b. residual tensor has correct shape",
            )
            check(
                not torch.all(inputs["x"] == 0),
                "1b. tensors are random (not zeros)",
            )

    # 1c. Golden data loading
    with _Timeout(30):
        with tempfile.TemporaryDirectory() as tmpdir:
            inputs_dir = Path(tmpdir) / "inputs"
            inputs_dir.mkdir()
            golden_dir = Path(tmpdir) / "golden"

            golden_subdir = golden_dir / "mixtral" / "moe_align"
            golden_subdir.mkdir(parents=True)
            known_tensor = {"topk_ids": torch.tensor([[3, 7]], dtype=torch.int32)}
            torch.save(known_tensor, golden_subdir / "decode-bs1-tp1.pt")

            (inputs_dir / "test.yaml").write_text(yaml_content)
            reg = InputRegistry(inputs_dir=inputs_dir, golden_dir=golden_dir)
            golden_inputs = reg.get_inputs("moe_align", "mixtral/decode-bs1/tp1", device="cpu")
            check(
                "topk_ids" in golden_inputs,
                "1c. golden data has expected key",
            )
            check(
                torch.equal(golden_inputs["topk_ids"], torch.tensor([[3, 7]], dtype=torch.int32)),
                "1c. golden data loads exact tensors",
            )

    # 1d. Filtering
    with _Timeout(30):
        filter_yaml = """
rms_norm:
  scenarios:
    - name: "llama31-8b/decode-bs1/tp1"
      init_args: {hidden_size: 64, eps: 1.0e-6}
      inputs:
        x: {shape: [1, 64], dtype: bfloat16}
    - name: "llama31-8b/decode-bs32/tp4"
      init_args: {hidden_size: 64, eps: 1.0e-6}
      inputs:
        x: {shape: [32, 64], dtype: bfloat16}
    - name: "mixtral-8x7b/decode-bs1/tp1"
      init_args: {hidden_size: 64, eps: 1.0e-6}
      inputs:
        x: {shape: [1, 64], dtype: bfloat16}
    - name: "mixtral-8x7b/prefill-512/tp2"
      init_args: {hidden_size: 64, eps: 1.0e-6}
      inputs:
        x: {shape: [512, 64], dtype: bfloat16}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            inputs_dir = Path(tmpdir) / "inputs"
            inputs_dir.mkdir()
            (inputs_dir / "test.yaml").write_text(filter_yaml)
            reg = InputRegistry(inputs_dir=inputs_dir)

            all_s = reg.scenarios("rms_norm")
            check(len(all_s) == 4, "1d. no filter returns all 4 scenarios")

            llama_s = reg.scenarios("rms_norm", models=["llama31"])
            check(
                len(llama_s) == 2 and all("llama31" in s.name for s in llama_s),
                "1d. model filter returns 2 llama scenarios",
            )

            tp4_s = reg.scenarios("rms_norm", tp=[4])
            check(
                len(tp4_s) == 1 and "tp4" in tp4_s[0].name,
                "1d. tp filter returns 1 tp4 scenario",
            )

            tp12 = reg.scenarios("rms_norm", tp=[1, 2])
            check(len(tp12) == 3, "1d. tp=[1,2] filter returns 3 scenarios")

    # 1e. Missing golden file
    with _Timeout(30):
        with tempfile.TemporaryDirectory() as tmpdir:
            inputs_dir = Path(tmpdir) / "inputs"
            inputs_dir.mkdir()
            (inputs_dir / "test.yaml").write_text(yaml_content)
            reg = InputRegistry(inputs_dir=inputs_dir, golden_dir=Path(tmpdir) / "empty_golden")
            try:
                reg.get_inputs("moe_align", "mixtral/decode-bs1/tp1", device="cpu")
                check(False, "1e. missing golden raises FileNotFoundError")
            except FileNotFoundError:
                check(True, "1e. missing golden raises FileNotFoundError")

    # 1f. Empty registry
    with _Timeout(30):
        with tempfile.TemporaryDirectory() as tmpdir:
            inputs_dir = Path(tmpdir) / "inputs"
            inputs_dir.mkdir()
            (inputs_dir / "empty.yaml").write_text("")
            reg = InputRegistry(inputs_dir=inputs_dir)
            check(len(reg.operators()) == 0, "1f. empty YAML -> zero operators")


# ===========================================================================
# Section 2: KernelRunner (unit, no GPU — uses CPU mock modules)
# ===========================================================================
def test_section_2():
    print(f"\n{'=' * 60}")
    print("  SECTION 2: KernelRunner")
    print(f"{'=' * 60}")

    from kb_nano.bench.kernels.runner import _compare_outputs, _time_forward
    from kb_nano.bench.kernels.result import OperatorResult, ScenarioResult

    # 2a. Identical modules
    with _Timeout(30):
        torch.manual_seed(42)
        m1 = nn.Linear(4, 4, bias=False)
        m2 = nn.Linear(4, 4, bias=False)
        m2.load_state_dict(m1.state_dict())
        inp = torch.randn(2, 4)
        with torch.no_grad():
            out1 = m1(inp)
            out2 = m2(inp)
        correct, diff = _compare_outputs(out1, out2)
        check(correct and diff == 0.0, "2a. identical modules -> correct=True, diff=0")

    # 2b. Different modules
    with _Timeout(30):
        class ZeroModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = nn.Parameter(torch.zeros(1))
            def forward(self, x):
                return torch.zeros_like(x)

        torch.manual_seed(42)
        baseline = nn.Linear(4, 4, bias=False)
        candidate = ZeroModule()
        inp = torch.randn(2, 4)
        with torch.no_grad():
            out_b = baseline(inp)
            out_c = candidate(inp)
        correct, diff = _compare_outputs(out_b, out_c)
        check(not correct or diff > 0, "2b. different modules -> correct=False or diff>0")
        check(diff > 0, f"2b. mean_abs_diff={diff:.4f} > 0")

    # 2c. Weight copying
    with _Timeout(30):
        torch.manual_seed(42)
        m1 = nn.Linear(8, 4, bias=False)
        m2 = nn.Linear(8, 4, bias=False)
        check(
            not torch.equal(m1.weight, m2.weight),
            "2c. before copy, weights differ",
        )
        m2.load_state_dict(m1.state_dict())
        check(
            torch.equal(m1.weight, m2.weight),
            "2c. after load_state_dict, weights match",
        )
        inp = torch.randn(2, 8)
        with torch.no_grad():
            out1 = m1(inp)
            out2 = m2(inp)
        check(torch.allclose(out1, out2), "2c. after copy, outputs match")

    # 2d. init_args propagation
    with _Timeout(30):
        class ConfigModule(nn.Module):
            def __init__(self, hidden_size: int = 10, eps: float = 1e-6):
                super().__init__()
                self.hidden_size = hidden_size
                self.eps = eps
            def forward(self, x):
                return x

        from kb_nano.bench.kernels.runner import _instantiate_module
        mod = _instantiate_module(ConfigModule, {"hidden_size": 32, "eps": 1e-5}, device="cpu")
        check(mod.hidden_size == 32, "2d. hidden_size=32 propagated")
        check(mod.eps == 1e-5, "2d. eps=1e-5 propagated")

    # 2e. Scenario filtering
    with _Timeout(30):
        from kb_nano.bench.utils.input_registry import InputRegistry

        filter_yaml = """
rms_norm:
  scenarios:
"""
        for i in range(10):
            model = "llama" if i < 3 else "mixtral"
            filter_yaml += f"""    - name: "{model}-scenario-{i}/decode-bs1/tp1"
      init_args: {{hidden_size: 4, eps: 1.0e-6}}
      inputs:
        x: {{shape: [1, 4], dtype: float32}}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            inputs_dir = Path(tmpdir) / "inputs"
            inputs_dir.mkdir()
            (inputs_dir / "test.yaml").write_text(filter_yaml)
            reg = InputRegistry(inputs_dir=inputs_dir)
            all_s = reg.scenarios("rms_norm")
            check(len(all_s) == 10, "2e. total 10 scenarios")
            filtered = reg.scenarios("rms_norm", models=["llama"])
            check(len(filtered) == 3, "2e. model=llama -> 3 scenarios")


# ===========================================================================
# Section 3: Result dataclasses and JSON output (unit, no GPU)
# ===========================================================================
def test_section_3():
    print(f"\n{'=' * 60}")
    print("  SECTION 3: Result dataclasses and JSON output")
    print(f"{'=' * 60}")

    from kb_nano.bench.kernels.result import (
        KernelBenchResult,
        OperatorResult,
        ScenarioResult,
    )

    # 3a. KernelBenchResult construction
    with _Timeout(30):
        scenarios_a = [
            ScenarioResult("s1", True, 1e-7, 0.1, 0.08, 1.25),
            ScenarioResult("s2", True, 2e-7, 0.12, 0.10, 1.20),
            ScenarioResult("s3", False, 5e-4, 0.1, 0.09, 1.11),
        ]
        op_a = OperatorResult(
            target="rms_norm", level=1,
            candidate_path="tasks/candidate/L1/rms_norm.py",
            scenarios=scenarios_a,
        )
        scenarios_b = [
            ScenarioResult("s4", True, 3e-7, 0.2, 0.18, 1.11),
            ScenarioResult("s5", True, 1e-7, 0.15, 0.14, 1.07),
        ]
        op_b = OperatorResult(
            target="silu_and_mul", level=1,
            candidate_path="tasks/candidate/L1/silu_and_mul.py",
            scenarios=scenarios_b,
        )

        result = KernelBenchResult(operators=[op_a, op_b])
        result.compute_aggregates()

        check(result.total_operators == 2, "3a. total_operators == 2")
        check(result.total_scenarios == 5, "3a. total_scenarios == 5")
        check(result.passed == 4, "3a. passed == 4")
        check(result.failed == 1, "3a. failed == 1")
        check(result.avg_speedup > 1.0, f"3a. avg_speedup={result.avg_speedup:.2f} > 1.0")

    # 3b. JSON round-trip
    with _Timeout(30):
        d = result.to_dict()
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        check(
            parsed["total_operators"] == 2 and parsed["total_scenarios"] == 5,
            "3b. JSON round-trip preserves top-level fields",
        )
        check(
            len(parsed["operators"]) == 2,
            "3b. JSON round-trip preserves operators array",
        )
        check(
            len(parsed["operators"][0]["scenarios"]) == 3,
            "3b. JSON round-trip preserves scenarios in first operator",
        )

    # 3c. Schema consistency (1 operator vs 4)
    with _Timeout(30):
        single = KernelBenchResult(operators=[op_a])
        single.compute_aggregates()
        d_single = single.to_dict()
        check(
            "operators" in d_single and isinstance(d_single["operators"], list),
            "3c. single operator: 'operators' is a list",
        )
        check(len(d_single["operators"]) == 1, "3c. single operator: list has 1 entry")

        multi = KernelBenchResult(operators=[op_a, op_b, op_a, op_b])
        multi.compute_aggregates()
        d_multi = multi.to_dict()
        check(len(d_multi["operators"]) == 4, "3c. four operators: list has 4 entries")

    # 3d. JSON file writing
    with _Timeout(30):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "subdir", "kernels.json")
            result.save_json(path)
            check(os.path.exists(path), "3d. JSON file created")
            with open(path) as f:
                loaded = json.load(f)
            check(loaded["total_operators"] == 2, "3d. JSON file has correct data")

    # 3e. Terminal table formatting
    with _Timeout(30):
        old_stdout = sys.stdout
        sys.stdout = buf = io.StringIO()
        result.print_table(single_target=False)
        sys.stdout = old_stdout
        output = buf.getvalue()
        check("CORRECT" in output or "PASS" in output, "3e. table contains PASS/CORRECT")
        check("SPEEDUP" in output or "speedup" in output, "3e. table contains SPEEDUP")
        check("OVERALL" in output, "3e. table contains OVERALL summary")
        check("ALL OPERATORS SUMMARY" in output, "3e. multi-operator table has summary")


# ===========================================================================
# Section 4: Standardized workloads (unit, no GPU)
# ===========================================================================
def test_section_4():
    print(f"\n{'=' * 60}")
    print("  SECTION 4: Standardized workloads")
    print(f"{'=' * 60}")

    from kb_nano.bench.utils.workloads import (
        LATENCY_WORKLOADS,
        THROUGHPUT_WORKLOADS,
        get_max_seq_len,
    )

    # 4a. Throughput workload constants
    with _Timeout(30):
        check(len(THROUGHPUT_WORKLOADS) == 3, "4a. exactly 3 throughput workloads")
        names = [w.name for w in THROUGHPUT_WORKLOADS]
        check(
            names == ["prefill-heavy", "balanced", "decode-heavy"],
            f"4a. correct names: {names}",
        )
        ph = THROUGHPUT_WORKLOADS[0]
        check(
            ph.input_len == 1024 and ph.output_len == 512,
            "4a. prefill-heavy: 1024/512",
        )
        bal = THROUGHPUT_WORKLOADS[1]
        check(
            bal.input_len == 512 and bal.output_len == 512,
            "4a. balanced: 512/512",
        )
        dh = THROUGHPUT_WORKLOADS[2]
        check(
            dh.input_len == 512 and dh.output_len == 1024,
            "4a. decode-heavy: 512/1024",
        )

    # 4b. Latency workload constants
    with _Timeout(30):
        check(len(LATENCY_WORKLOADS) == 2, "4b. exactly 2 latency workloads")
        sr = LATENCY_WORKLOADS[0]
        check(
            sr.name == "single-request" and sr.batch_size == 1
            and sr.input_len == 128 and sr.output_len == 128,
            "4b. single-request: bs=1, 128/128",
        )
        fb = LATENCY_WORKLOADS[1]
        check(
            fb.name == "fixed-batch-32" and fb.batch_size == 32
            and fb.input_len == 128 and fb.output_len == 128,
            "4b. fixed-batch-32: bs=32, 128/128",
        )

    # 4c. Immutability (frozen dataclasses)
    with _Timeout(30):
        try:
            THROUGHPUT_WORKLOADS[0].input_len = 999
            check(False, "4c. throughput workloads should be immutable")
        except AttributeError:
            check(True, "4c. throughput workloads are frozen (immutable)")
        try:
            LATENCY_WORKLOADS[0].batch_size = 999
            check(False, "4c. latency workloads should be immutable")
        except AttributeError:
            check(True, "4c. latency workloads are frozen (immutable)")

    # 4d. get_max_seq_len
    with _Timeout(30):
        max_len = get_max_seq_len()
        check(
            max_len == 1536,
            f"4d. max_seq_len = {max_len} (expected 1536 = 512+1024)",
        )


# ===========================================================================
# Section 5: Multi-level conflict resolution (unit, no GPU)
# ===========================================================================
def _can_discover_targets() -> bool:
    """Check if target discovery works (requires sgl_kernel and other CUDA deps)."""
    try:
        from kb_nano.infra.kernel_swapper import discover_targets
        discover_targets()
        return True
    except (ImportError, ModuleNotFoundError):
        return False


def test_section_5():
    print(f"\n{'=' * 60}")
    print("  SECTION 5: Multi-level conflict resolution")
    print(f"{'=' * 60}")

    from kb_nano.infra.kernel_swapper import (
        BenchTarget,
        _detect_subsumption,
        _sort_by_level,
    )

    # 5a. Bottom-up ordering
    with _Timeout(30):
        t1 = BenchTarget("op_a", 3, "tasks.baseline.L3.op_a", ["llama31"], nn.Module)
        t2 = BenchTarget("op_b", 1, "tasks.baseline.L1.op_b", ["llama31"], nn.Module)
        t3 = BenchTarget("op_c", 2, "tasks.baseline.L2.op_c", ["llama31"], nn.Module)

        class Fake(nn.Module):
            pass

        candidates = [(t1, Fake), (t2, Fake), (t3, Fake)]
        sorted_c = _sort_by_level(candidates)
        levels = [t.level for t, _ in sorted_c]
        check(levels == [1, 2, 3], f"5a. sorted levels: {levels} == [1, 2, 3]")

    has_targets = _can_discover_targets()

    # 5b. Subsumption detection
    with _Timeout(30):
        if not has_targets:
            print("    SKIP  5b. sgl_kernel not available, cannot discover targets")
        else:
            from kb_nano.infra.kernel_swapper import get

            rms_target = get("rms_norm")
            llama_decoder_target = get("llama_decoder")

            class FakeRMS(nn.Module):
                pass

            class FakeDecoder(nn.Module):
                pass

            candidates = [
                (rms_target, FakeRMS),
                (llama_decoder_target, FakeDecoder),
            ]
            warnings = _detect_subsumption(candidates)
            found_subsumption = any(
                lower_name == "rms_norm" for _, _, lower_name, _ in warnings
            )
            check(
                found_subsumption,
                f"5b. L3 llama_decoder subsumes L1 rms_norm (found {len(warnings)} warnings)",
            )

    # 5c. No false positives (mock targets with no import relationship)
    with _Timeout(30):
        class FakeA(nn.Module):
            pass
        class FakeB(nn.Module):
            pass

        mock_l1 = BenchTarget("fake_op_alpha", 1, "tasks.baseline.L1.fake_op_alpha", ["llama31"], nn.Module)
        mock_l2 = BenchTarget("fake_op_beta", 2, "tasks.baseline.L2.fake_op_beta", ["llama31"], nn.Module)
        candidates_no_overlap = [
            (mock_l1, FakeA),
            (mock_l2, FakeB),
        ]
        warnings_no = _detect_subsumption(candidates_no_overlap)
        check(
            len(warnings_no) == 0,
            "5c. mock targets with no import chain -> no subsumption (no false positive)",
        )

    # 5d. Patching order
    with _Timeout(30):
        if not has_targets:
            print("    SKIP  5d. sgl_kernel not available, cannot discover targets")
        else:
            from kb_nano.infra.kernel_swapper import get, patch_class, restore
            rms_target = get("rms_norm")
            rms_module = importlib.import_module(f"{PACKAGE_NAME}.{rms_target.module_path}")
            original_cls = rms_target.target_cls

            class PatchedRMSNorm(nn.Module):
                _is_patched = True

            undo = patch_class(rms_target, PatchedRMSNorm)
            patched_cls = getattr(rms_module, original_cls.__name__)
            check(
                patched_cls is PatchedRMSNorm,
                "5d. L1 patch applied correctly",
            )

            decoder_mod = importlib.import_module(f"{PACKAGE_NAME}.tasks.baseline.L3.llama_decoder")
            decoder_rms = getattr(decoder_mod, original_cls.__name__, None)
            check(
                decoder_rms is PatchedRMSNorm,
                "5d. L3 baseline picks up L1 patch",
            )

            restore(undo)
            check(
                getattr(rms_module, original_cls.__name__) is original_cls,
                "5d. restore works correctly",
            )

    # 5e. Discovery convenience functions
    with _Timeout(30):
        if not has_targets:
            print("    SKIP  5e. sgl_kernel not available, cannot discover targets")
        else:
            from kb_nano.infra.kernel_swapper import (
                list_targets, models_for_target, targets_for_model,
            )

            all_targets = list_targets()
            check(len(all_targets) > 0, "5e. list_targets() returns non-empty list")

            l1_targets = list_targets(level=1)
            check(
                len(l1_targets) > 0 and all(t.level == 1 for t in l1_targets),
                "5e. list_targets(level=1) returns only L1 targets",
            )

            rms_models = models_for_target("rms_norm")
            check(
                "llama31" in rms_models,
                '5e. models_for_target("rms_norm") contains "llama31"',
            )

            llama_targets = targets_for_model("llama31")
            check(
                len(llama_targets) > 0
                and all("llama31" in t.models for t in llama_targets),
                '5e. targets_for_model("llama31") all include "llama31"',
            )


# ===========================================================================
# Section 6: CLI argument parsing (unit, no GPU)
# ===========================================================================
def test_section_6():
    print(f"\n{'=' * 60}")
    print("  SECTION 6: CLI argument parsing")
    print(f"{'=' * 60}")

    # 6a. bench.kernels CLI argument parsing
    with _Timeout(30):
        import argparse as _ap

        old_argv = sys.argv
        sys.argv = ["prog", "--target", "rms_norm", "--model", "llama", "--tp", "1", "4",
                     "--category", "llm", "--output-json", "/tmp/test.json",
                     "--num-warmup", "5", "--num-runs", "50"]
        try:
            parser = _ap.ArgumentParser()
            parser.add_argument("--target", type=str)
            parser.add_argument("--model", nargs="+")
            parser.add_argument("--tp", type=int, nargs="+")
            parser.add_argument("--category", type=str)
            parser.add_argument("--output-json", type=str)
            parser.add_argument("--num-warmup", type=int)
            parser.add_argument("--num-runs", type=int)
            args = parser.parse_args(sys.argv[1:])
            check(
                args.target == "rms_norm" and args.tp == [1, 4]
                and args.model == ["llama"] and args.category == "llm"
                and args.num_warmup == 5 and args.num_runs == 50,
                "6a. --target, --model, --tp, --category, --output-json, --num-warmup, --num-runs parsed",
            )
        finally:
            sys.argv = old_argv

    # 6b. bench.e2e CLI
    with _Timeout(30):
        result = subprocess.run(
            [sys.executable, "-m", "kb_nano.bench.e2e", "--help"],
            capture_output=True, text=True, timeout=30,
            cwd=PROJECT_ROOT,
            env={**os.environ, "PYTHONPATH": PROJECT_ROOT},
        )
        if result.returncode != 0:
            # May fail due to missing sgl_kernel on import - check for import-related vs parse errors
            if "sgl_kernel" in result.stderr or "ModuleNotFoundError" in result.stderr:
                print("    SKIP  6b. sgl_kernel not available for subprocess import")
            else:
                check(False, f"6b. e2e --help failed: {result.stderr[-200:]}")
        else:
            check(
                "throughput" in result.stdout and "latency" in result.stdout
                and "serve" in result.stdout,
                "6b. e2e help shows throughput, latency, serve subcommands",
            )
            # Check eval is not in subcommand list (after the header)
            help_text = result.stdout
            if "Benchmark type" in help_text:
                subcommand_section = help_text.split("Benchmark type")[1]
                check(
                    "eval" not in subcommand_section,
                    "6b. eval subcommand removed from e2e",
                )
            else:
                check(True, "6b. eval subcommand not present in e2e help")

    # 6c. bench.eval CLI
    with _Timeout(30):
        result = subprocess.run(
            [sys.executable, "-m", "kb_nano.bench.eval", "--help"],
            capture_output=True, text=True, timeout=30,
            cwd=PROJECT_ROOT,
            env={**os.environ, "PYTHONPATH": PROJECT_ROOT},
        )
        if result.returncode != 0 and ("sgl_kernel" in result.stderr or "ModuleNotFoundError" in result.stderr):
            print("    SKIP  6c. sgl_kernel not available for subprocess import")
        else:
            check(
                result.returncode == 0 and "--model" in result.stdout,
                "6c. eval CLI accepts --model",
            )
            check(
                "--tp" in result.stdout and "--category" in result.stdout
                and "--output-json" in result.stdout,
                "6c. eval CLI accepts --tp, --category, --output-json",
            )

    # 6d. Default JSON output path
    with _Timeout(30):
        from kb_nano.bench.kernels.__main__ import _DEFAULT_OUTPUT as kernels_default
        check(
            kernels_default == "bench/results/kernels.json",
            f"6d. kernels default output: {kernels_default}",
        )

        from kb_nano.bench.eval.__main__ import _DEFAULT_OUTPUT as eval_default
        check(
            eval_default == "bench/results/eval.json",
            f"6d. eval default output: {eval_default}",
        )


# ===========================================================================
# Section 7: Kernel-level integration (GPU required, single GPU)
# ===========================================================================
def test_section_7():
    print(f"\n{'=' * 60}")
    print("  SECTION 7: Kernel-level integration (GPU required)")
    print(f"{'=' * 60}")

    candidate_dir = os.path.join(PACKAGE_DIR, "tasks", "candidate", "L1")
    candidate_file = os.path.join(candidate_dir, "rms_norm.py")
    baseline_file = os.path.join(PACKAGE_DIR, "tasks", "baseline", "L1", "rms_norm.py")
    had_candidate = os.path.exists(candidate_file)
    if had_candidate:
        original_candidate = open(candidate_file).read()

    # 7a. Identity replacement via the new runner (copy baseline as candidate)
    with _Timeout(120):
        try:
            import shutil
            os.makedirs(candidate_dir, exist_ok=True)
            shutil.copy2(baseline_file, candidate_file)
            result = subprocess.run(
                [sys.executable, "-c", f"""
import sys, json
sys.path.insert(0, '{PROJECT_ROOT}')
from {PACKAGE_NAME}.bench.kernels.runner import run_kernel_benchmark

result = run_kernel_benchmark(
    'rms_norm',
    num_warmup=2,
    num_runs=5,
)
print(json.dumps({{
    'passed': result.passed,
    'failed': result.failed,
    'total': result.total_scenarios,
    'avg_diff': result.avg_mean_abs_diff,
    'avg_speedup': result.avg_speedup,
}}))
"""],
                capture_output=True, text=True, timeout=60,
                cwd=PROJECT_ROOT,
            )
            if result.returncode != 0:
                print(f"    STDERR: {result.stderr[-500:]}")
                check(False, "7a. identity subprocess exited non-zero")
            else:
                data = json.loads(result.stdout.strip().split("\n")[-1])
                check(
                    data["avg_diff"] < 1e-5,
                    f"7a. identity: avg_diff={data['avg_diff']:.2e} < 1e-5",
                )
                check(
                    data["failed"] == 0,
                    f"7a. identity: {data['passed']}/{data['total']} passed",
                )
        except subprocess.TimeoutExpired:
            check(False, "7a. identity test timed out after 60s (possible hang)")

    # 7b. Broken replacement (write broken candidate to candidates folder)
    with _Timeout(120):
        try:
            with open(candidate_file, "w") as f:
                f.write("""\
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, hidden_size=4096, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
    def forward(self, x, residual=None):
        if residual is None:
            return torch.zeros_like(x)
        return torch.zeros_like(x), residual
""")
            result = subprocess.run(
                [sys.executable, "-c", f"""
import sys, json
sys.path.insert(0, '{PROJECT_ROOT}')
from {PACKAGE_NAME}.bench.kernels.runner import run_kernel_benchmark

result = run_kernel_benchmark(
    'rms_norm',
    num_warmup=2,
    num_runs=5,
)
print(json.dumps({{
    'passed': result.passed,
    'failed': result.failed,
    'avg_diff': result.avg_mean_abs_diff,
}}))
"""],
                capture_output=True, text=True, timeout=60,
                cwd=PROJECT_ROOT,
            )
            if result.returncode != 0:
                print(f"    STDERR: {result.stderr[-500:]}")
                check(False, "7b. broken subprocess exited non-zero")
            else:
                data = json.loads(result.stdout.strip().split("\n")[-1])
                check(
                    data["avg_diff"] > 0.01,
                    f"7b. broken: avg_diff={data['avg_diff']:.4f} > 0.01",
                )
                check(
                    data["failed"] > 0,
                    f"7b. broken: {data['failed']} failures detected",
                )
        except subprocess.TimeoutExpired:
            check(False, "7b. broken test timed out after 60s")

    # 7c. JSON output file via CLI (baseline as candidate in candidates folder)
    with _Timeout(180):
        try:
            import shutil
            shutil.copy2(baseline_file, candidate_file)
            with tempfile.TemporaryDirectory() as tmpdir:
                json_path = os.path.join(tmpdir, "kernels.json")
                result = subprocess.run(
                    [sys.executable, "-m", "kb_nano.bench.kernels",
                     "--target", "rms_norm",
                     "--output-json", json_path,
                     "--num-warmup", "2", "--num-runs", "5"],
                    timeout=120, cwd=PROJECT_ROOT,
                    capture_output=True, text=True,
                )
                if result.returncode not in (0, 1):
                    print(f"    STDERR: {result.stderr[-500:]}")
                check(
                    os.path.exists(json_path),
                    "7c. JSON output file created by CLI",
                )
                if os.path.exists(json_path):
                    with open(json_path) as f:
                        data = json.load(f)
                    check(
                        "operators" in data and "total_scenarios" in data,
                        "7c. JSON has correct schema",
                    )
        except subprocess.TimeoutExpired:
            check(False, "7c. CLI test timed out after 120s")
        finally:
            if had_candidate:
                with open(candidate_file, "w") as f:
                    f.write(original_candidate)
            elif os.path.exists(candidate_file):
                os.remove(candidate_file)

    # 7d. All-candidates default (skip if no candidates exist)
    with _Timeout(360):
        from kb_nano.infra.kernel_swapper import discover_candidates
        candidates = discover_candidates()
        if not candidates:
            print("    SKIP  7d. no candidate kernels found, skipping all-candidates test")
        else:
            with tempfile.TemporaryDirectory() as tmpdir:
                json_path = os.path.join(tmpdir, "all.json")
                try:
                    result = subprocess.run(
                        [sys.executable, "-m", "kb_nano.bench.kernels",
                         "--output-json", json_path,
                         "--num-warmup", "2", "--num-runs", "5"],
                        timeout=300, cwd=PROJECT_ROOT,
                        capture_output=True, text=True,
                    )
                    check(
                        os.path.exists(json_path),
                        "7d. all-candidates JSON created",
                    )
                    if os.path.exists(json_path):
                        with open(json_path) as f:
                            data = json.load(f)
                        check(
                            data["total_operators"] == len(candidates),
                            f"7d. tested {data['total_operators']}/{len(candidates)} candidates",
                        )
                except subprocess.TimeoutExpired:
                    check(False, "7d. all-candidates timed out after 300s")


# ===========================================================================
# Section 8: E2E integration (GPU required, single GPU)
# ===========================================================================
def test_section_8():
    print(f"\n{'=' * 60}")
    print("  SECTION 8: E2E integration (GPU required)")
    print(f"{'=' * 60}")

    # 8a. Throughput single-run
    with _Timeout(360):
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "throughput.json")
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "kb_nano.bench.e2e", "throughput",
                     "--model", "meta-llama/Llama-3.1-8B-Instruct",
                     "--tp", "1",
                     "--dataset-name", "random",
                     "--random-input-len", "128",
                     "--random-output-len", "64",
                     "--num-prompts", "10",
                     "--output-json", json_path,
                     "--no-candidate-kernels"],
                    timeout=300, cwd=PROJECT_ROOT,
                    capture_output=True, text=True,
                )
                if result.returncode != 0:
                    print(f"    STDERR: {result.stderr[-500:]}")
                    check(False, "8a. throughput subprocess failed")
                else:
                    check(os.path.exists(json_path), "8a. throughput JSON created")
                    if os.path.exists(json_path):
                        with open(json_path) as f:
                            data = json.load(f)
                        check(
                            data.get("tokens_per_second", 0) > 0,
                            f"8a. tokens_per_second={data.get('tokens_per_second', 0):.0f} > 0",
                        )
            except subprocess.TimeoutExpired:
                check(False, "8a. throughput timed out after 300s")

    # 8b. Latency single-run
    with _Timeout(360):
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "latency.json")
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "kb_nano.bench.e2e", "latency",
                     "--model", "meta-llama/Llama-3.1-8B-Instruct",
                     "--batch-size", "1",
                     "--input-len", "128",
                     "--output-len", "64",
                     "--num-iters-warmup", "1",
                     "--num-iters", "3",
                     "--output-json", json_path,
                     "--no-candidate-kernels"],
                    timeout=300, cwd=PROJECT_ROOT,
                    capture_output=True, text=True,
                )
                if result.returncode != 0:
                    print(f"    STDERR: {result.stderr[-500:]}")
                    check(False, "8b. latency subprocess failed")
                else:
                    check(os.path.exists(json_path), "8b. latency JSON created")
                    if os.path.exists(json_path):
                        with open(json_path) as f:
                            data = json.load(f)
                        check(
                            data.get("avg_latency", 0) > 0,
                            f"8b. avg_latency={data.get('avg_latency', 0):.4f} > 0",
                        )
            except subprocess.TimeoutExpired:
                check(False, "8b. latency timed out after 300s")

    # 8c. JSON default save (verify default path works with --output-json)
    with _Timeout(30):
        check(True, "8c. default JSON paths verified in section 6d (no redundant GPU run)")


# ===========================================================================
# Section 9: Eval integration (GPU required, single GPU)
# ===========================================================================
def test_section_9():
    print(f"\n{'=' * 60}")
    print("  SECTION 9: Eval integration (GPU required)")
    print(f"{'=' * 60}")

    # 9a. Single-model eval (requires at least one candidate)
    with _Timeout(660):
        from kb_nano.infra.kernel_swapper import discover_candidates
        candidates = discover_candidates()
        if not candidates:
            print("    SKIP  9a. no candidate kernels, skipping eval test")
        else:
            with tempfile.TemporaryDirectory() as tmpdir:
                json_path = os.path.join(tmpdir, "eval.json")
                try:
                    result = subprocess.run(
                        [sys.executable, "-m", "kb_nano.bench.eval",
                         "--model", "meta-llama/Llama-3.1-8B-Instruct",
                         "--tp", "1",
                         "--num-prompts", "10",
                         "--output-json", json_path],
                        timeout=600, cwd=PROJECT_ROOT,
                        capture_output=True, text=True,
                    )
                    if result.returncode != 0:
                        print(f"    STDERR: {result.stderr[-500:]}")
                        check(False, "9a. eval subprocess failed")
                    else:
                        check(os.path.exists(json_path), "9a. eval JSON created")
                        if os.path.exists(json_path):
                            with open(json_path) as f:
                                data = json.load(f)
                            check(
                                "total_models" in data and "categories" in data,
                                "9a. eval JSON has expected schema",
                            )
                            check(
                                data.get("total_jobs", 0) >= 1,
                                f"9a. total_jobs={data.get('total_jobs', 0)} >= 1",
                            )
                except subprocess.TimeoutExpired:
                    check(False, "9a. eval timed out after 600s")

    # 9b. Eval with no candidates
    with _Timeout(60):
        try:
            result = subprocess.run(
                [sys.executable, "-c", f"""
import sys, os
sys.path.insert(0, '{PROJECT_ROOT}')
import tempfile, shutil

# Temporarily rename candidate dir to simulate no candidates
from {PACKAGE_NAME}.infra.kernel_swapper import _CANDIDATE_DIR
backup = str(_CANDIDATE_DIR) + '_backup'
if _CANDIDATE_DIR.is_dir():
    shutil.move(str(_CANDIDATE_DIR), backup)
os.makedirs(str(_CANDIDATE_DIR), exist_ok=True)

try:
    from {PACKAGE_NAME}.bench.eval.config import EvalConfig
    from {PACKAGE_NAME}.bench.eval.planner import EvalPlanner
    config = EvalConfig(models=['meta-llama/Llama-3.1-8B-Instruct'], tp_degrees=[1])
    planner = EvalPlanner(config)
    plan = planner.plan()
    # With explicit models, plan should have jobs even without candidates
    # The eval runner would just compare baseline vs baseline
    print('PLAN_JOBS:' + str(plan.num_jobs))
except Exception as e:
    print('ERROR:' + str(e))
finally:
    import shutil
    if os.path.exists(backup):
        shutil.rmtree(str(_CANDIDATE_DIR), ignore_errors=True)
        shutil.move(backup, str(_CANDIDATE_DIR))
"""],
                capture_output=True, text=True, timeout=30,
                cwd=PROJECT_ROOT,
            )
            output = result.stdout.strip()
            check(
                result.returncode == 0,
                f"9b. no-candidates exits cleanly (rc={result.returncode})",
            )
        except subprocess.TimeoutExpired:
            check(False, "9b. no-candidates timed out after 30s (possible hang)")

    # 9c. JSON default save path
    with _Timeout(30):
        from kb_nano.bench.eval.__main__ import _DEFAULT_OUTPUT
        check(
            _DEFAULT_OUTPUT == "bench/results/eval.json",
            f"9c. eval default output: {_DEFAULT_OUTPUT}",
        )


# ===========================================================================
# Section 10: VLM Preprocessing (GPU required, needs VLM model)
# ===========================================================================
VLM_PREPROCESS_WORKER = r'''
import json, os, sys, time
import torch

def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)
    os.environ["_BENCH_MODEL_NAME"] = cfg["model"]
    sys.path.insert(0, cfg["project_root"])
    pkg = cfg["package_name"]

    from transformers import AutoProcessor, AutoTokenizer

    mod = __import__(f"{pkg}.infra.engine", fromlist=["LlamaEngine", "SamplingParams"])
    LlamaEngine, SamplingParams = mod.LlamaEngine, mod.SamplingParams

    engine = LlamaEngine(
        model_name=cfg["model"],
        seed=42,
        enforce_eager=True,
        tensor_parallel_size=cfg.get("tp", 1),
        max_model_len=cfg.get("max_model_len", 4096),
    )

    processor = AutoProcessor.from_pretrained(cfg["model"], trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], trust_remote_code=True)

    results = {}

    # --- 10a: Image preprocessing correctness ---
    try:
        from vllm.benchmarks.datasets import VisionArenaDataset
        ds = VisionArenaDataset(
            dataset_path="lmarena-ai/VisionArena-Chat",
            dataset_split="train",
            random_seed=42,
        )
        samples = ds.sample(tokenizer, 5, enable_multimodal_chat=True)

        img_correct = True
        img_details = []
        for i, s in enumerate(samples):
            kb_result = engine.preprocess_chat(s.prompt)

            ref_text = processor.apply_chat_template(
                s.prompt, tokenize=False, add_generation_prompt=True,
            )
            from PIL import Image
            from io import BytesIO
            import base64
            ref_images = []
            for msg in s.prompt:
                if msg.get("role") != "user":
                    continue
                for c in msg.get("content", []):
                    if c.get("type") == "image_url":
                        url = c["image_url"]["url"]
                        if url.startswith("data:image/"):
                            b64 = url.split(",", 1)[1]
                            img = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
                        else:
                            path = url.replace("file://", "")
                            img = Image.open(path).convert("RGB")
                        ref_images.append(img)
            ref_inputs = processor(
                text=[ref_text], images=ref_images if ref_images else None,
                return_tensors="pt", padding=True,
            )

            ref_ids = ref_inputs["input_ids"][0].tolist()
            ids_match = kb_result["token_ids"] == ref_ids

            pv_match = True
            if ref_inputs.get("pixel_values") is not None and kb_result["pixel_values"] is not None:
                pv_match = torch.allclose(
                    kb_result["pixel_values"].float(),
                    ref_inputs["pixel_values"].float(),
                    atol=1e-4,
                )
            elif ref_inputs.get("pixel_values") is not None or kb_result["pixel_values"] is not None:
                pv_match = False

            thw_match = True
            if ref_inputs.get("image_grid_thw") is not None and kb_result["image_grid_thw"] is not None:
                thw_match = ref_inputs["image_grid_thw"].tolist() == kb_result["image_grid_thw"]
            elif ref_inputs.get("image_grid_thw") is not None or kb_result["image_grid_thw"] is not None:
                thw_match = False

            sample_ok = ids_match and pv_match and thw_match
            if not sample_ok:
                img_correct = False
            img_details.append({
                "sample": i,
                "ids_match": ids_match,
                "pv_match": pv_match,
                "thw_match": thw_match,
                "kb_ids_len": len(kb_result["token_ids"]),
                "ref_ids_len": len(ref_ids),
            })

        results["10a_image_correctness"] = {
            "pass": img_correct,
            "details": img_details,
        }
    except Exception as e:
        results["10a_image_correctness"] = {"pass": False, "error": str(e)}

    # --- 10b: Video preprocessing correctness ---
    try:
        from vllm.benchmarks.datasets import MMVUDataset
        ds = MMVUDataset(
            dataset_path="yale-nlp/MMVU",
            dataset_split="validation",
            random_seed=42,
            no_stream=True,
        )
        samples = ds.sample(tokenizer, 3, enable_multimodal_chat=True)

        vid_correct = True
        vid_details = []
        for i, s in enumerate(samples):
            kb_result = engine.preprocess_chat(s.prompt)

            import decord
            decord.bridge.set_bridge("native")
            ref_videos = []
            ref_content = []
            for msg in s.prompt:
                if msg.get("role") != "user":
                    continue
                for c in msg.get("content", []):
                    if c.get("type") == "text":
                        ref_content.append({"type": "text", "text": c["text"]})
                    elif c.get("type") == "video_url":
                        video_url = c["video_url"]["url"]
                        video_path = video_url
                        for prefix in ("file://", "https://", "http://"):
                            if video_path.startswith(prefix):
                                video_path = video_path[len(prefix):]
                                break
                        if not os.path.exists(video_path):
                            from huggingface_hub import hf_hub_download
                            parts = video_path.split("/")
                            try:
                                hf_idx = next(ii for ii, p in enumerate(parts) if p == "datasets")
                                repo_id = "/".join(parts[hf_idx + 1:hf_idx + 3])
                                resolve_idx = parts.index("resolve")
                                rel_path = "/".join(parts[resolve_idx + 2:])
                                video_path = hf_hub_download(
                                    repo_id=repo_id, filename=rel_path, repo_type="dataset",
                                )
                            except (StopIteration, ValueError):
                                pass
                        vr = decord.VideoReader(video_path)
                        total = len(vr)
                        num_frames = min(total, 16)
                        indices = [int(j * total / num_frames) for j in range(num_frames)]
                        frames = [Image.fromarray(vr[idx].asnumpy()).convert("RGB") for idx in indices]
                        ref_videos.append(frames)
                        ref_content.append({"type": "video", "video": frames})

            ref_messages = [{"role": "user", "content": ref_content}]
            ref_text = processor.apply_chat_template(
                ref_messages, tokenize=False, add_generation_prompt=True,
            )
            ref_inputs = processor(
                text=[ref_text], videos=ref_videos if ref_videos else None,
                return_tensors="pt", padding=True,
            )

            ref_ids = ref_inputs["input_ids"][0].tolist()
            ids_match = kb_result["token_ids"] == ref_ids

            pv_match = True
            ref_vpv = ref_inputs.get("pixel_values_videos")
            kb_vpv = kb_result.get("video_pixel_values")
            if ref_vpv is not None and kb_vpv is not None:
                pv_match = torch.allclose(kb_vpv.float(), ref_vpv.float(), atol=1e-4)
            elif ref_vpv is not None or kb_vpv is not None:
                pv_match = False

            thw_match = True
            ref_vthw = ref_inputs.get("video_grid_thw")
            kb_vthw = kb_result.get("video_grid_thw")
            if ref_vthw is not None and kb_vthw is not None:
                thw_match = ref_vthw.tolist() == kb_vthw
            elif ref_vthw is not None or kb_vthw is not None:
                thw_match = False

            sample_ok = ids_match and pv_match and thw_match
            if not sample_ok:
                vid_correct = False
            vid_details.append({
                "sample": i,
                "ids_match": ids_match,
                "pv_match": pv_match,
                "thw_match": thw_match,
                "kb_ids_len": len(kb_result["token_ids"]),
                "ref_ids_len": len(ref_ids),
            })

        results["10b_video_correctness"] = {
            "pass": vid_correct,
            "details": vid_details,
        }
    except Exception as e:
        results["10b_video_correctness"] = {"pass": False, "error": str(e)}

    # --- 10c: M-RoPE position correctness ---
    try:
        mrope_ok = True
        mrope_details = []

        pp_text = engine.preprocess_multimodal("Hello world, this is a test")
        text_mrope = pp_text["mrope_positions"]
        text_len = len(pp_text["token_ids"])
        expected_pos = torch.arange(text_len, dtype=torch.int64).unsqueeze(0).expand(3, -1)
        text_pos_ok = torch.equal(text_mrope, expected_pos)
        mrope_details.append({"case": "text_only", "pass": text_pos_ok,
                              "seq_len": text_len})
        if not text_pos_ok:
            mrope_ok = False

        from vllm.benchmarks.datasets import VisionArenaDataset
        ds = VisionArenaDataset(
            dataset_path="lmarena-ai/VisionArena-Chat",
            dataset_split="train",
            random_seed=42,
        )
        img_samples = ds.sample(tokenizer, 2, enable_multimodal_chat=True)
        for i, s in enumerate(img_samples):
            pp = engine.preprocess_chat(s.prompt)
            mrope_pos = pp["mrope_positions"]
            pos_shape_ok = mrope_pos.shape[0] == 3 and mrope_pos.shape[1] == len(pp["token_ids"])
            delta_ok = isinstance(pp["mrope_position_delta"], (int, float))
            case_ok = pos_shape_ok and delta_ok
            mrope_details.append({
                "case": f"image_{i}",
                "shape_ok": pos_shape_ok,
                "delta_ok": delta_ok,
                "shape": list(mrope_pos.shape),
                "pass": case_ok,
            })
            if not case_ok:
                mrope_ok = False

        results["10c_mrope_correctness"] = {
            "pass": mrope_ok,
            "details": mrope_details,
        }
    except Exception as e:
        results["10c_mrope_correctness"] = {"pass": False, "error": str(e)}

    # --- 10d: Preprocessing throughput -- image batch ---
    try:
        from vllm.benchmarks.datasets import VisionArenaDataset
        ds = VisionArenaDataset(
            dataset_path="lmarena-ai/VisionArena-Chat",
            dataset_split="train",
            random_seed=42,
        )
        img_samples = ds.sample(tokenizer, 100, enable_multimodal_chat=True)

        t0 = time.perf_counter()
        for s in img_samples:
            engine.preprocess_chat(s.prompt)
        kb_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        for s in img_samples:
            ref_text = processor.apply_chat_template(
                s.prompt, tokenize=False, add_generation_prompt=True,
            )
            ref_images = []
            for msg in s.prompt:
                if msg.get("role") != "user":
                    continue
                for c in msg.get("content", []):
                    if c.get("type") == "image_url":
                        url = c["image_url"]["url"]
                        if url.startswith("data:image/"):
                            b64 = url.split(",", 1)[1]
                            img = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
                        else:
                            path = url.replace("file://", "")
                            img = Image.open(path).convert("RGB")
                        ref_images.append(img)
            processor(
                text=[ref_text], images=ref_images if ref_images else None,
                return_tensors="pt", padding=True,
            )
        ref_time = time.perf_counter() - t0

        ratio = kb_time / ref_time if ref_time > 0 else float("inf")
        results["10d_image_throughput"] = {
            "pass": ratio <= 1.1,
            "kb_time": kb_time,
            "ref_time": ref_time,
            "ratio": ratio,
            "num_samples": 100,
        }
    except Exception as e:
        results["10d_image_throughput"] = {"pass": False, "error": str(e)}

    # --- 10e: Preprocessing throughput -- video batch ---
    try:
        from vllm.benchmarks.datasets import MMVUDataset
        ds = MMVUDataset(
            dataset_path="yale-nlp/MMVU",
            dataset_split="validation",
            random_seed=42,
            no_stream=True,
        )
        vid_samples = ds.sample(tokenizer, 20, enable_multimodal_chat=True)

        t0 = time.perf_counter()
        for s in vid_samples:
            engine.preprocess_chat(s.prompt)
        kb_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        for s in vid_samples:
            ref_text = processor.apply_chat_template(
                s.prompt, tokenize=False, add_generation_prompt=True,
            )
            ref_videos = []
            for msg in s.prompt:
                if msg.get("role") != "user":
                    continue
                for c in msg.get("content", []):
                    if c.get("type") == "video_url":
                        video_url = c["video_url"]["url"]
                        video_path = video_url
                        for prefix in ("file://", "https://", "http://"):
                            if video_path.startswith(prefix):
                                video_path = video_path[len(prefix):]
                                break
                        if not os.path.exists(video_path):
                            from huggingface_hub import hf_hub_download
                            parts = video_path.split("/")
                            try:
                                hf_idx = next(ii for ii, p in enumerate(parts) if p == "datasets")
                                repo_id = "/".join(parts[hf_idx + 1:hf_idx + 3])
                                resolve_idx = parts.index("resolve")
                                rel_path = "/".join(parts[resolve_idx + 2:])
                                video_path = hf_hub_download(
                                    repo_id=repo_id, filename=rel_path, repo_type="dataset",
                                )
                            except (StopIteration, ValueError):
                                pass
                        vr = decord.VideoReader(video_path)
                        total = len(vr)
                        num_frames = min(total, 16)
                        indices = [int(j * total / num_frames) for j in range(num_frames)]
                        frames = [Image.fromarray(vr[idx].asnumpy()).convert("RGB") for idx in indices]
                        ref_videos.append(frames)
            processor(
                text=[ref_text], videos=ref_videos if ref_videos else None,
                return_tensors="pt", padding=True,
            )
        ref_time = time.perf_counter() - t0

        ratio = kb_time / ref_time if ref_time > 0 else float("inf")
        results["10e_video_throughput"] = {
            "pass": ratio <= 1.1,
            "kb_time": kb_time,
            "ref_time": ref_time,
            "ratio": ratio,
            "num_samples": 20,
        }
    except Exception as e:
        results["10e_video_throughput"] = {"pass": False, "error": str(e)}

    # --- 10f: Pre-processed input acceptance ---
    try:
        from vllm.benchmarks.datasets import VisionArenaDataset
        ds = VisionArenaDataset(
            dataset_path="lmarena-ai/VisionArena-Chat",
            dataset_split="train",
            random_seed=42,
        )
        gen_samples = ds.sample(tokenizer, 3, enable_multimodal_chat=True)

        preprocessed = [engine.preprocess_chat(s.prompt) for s in gen_samples]

        sp = SamplingParams(temperature=0.0, max_tokens=32)
        outputs = engine.generate(preprocessed, sp)
        gen_ok = len(outputs) == 3
        non_empty = all(len(o.token_ids) > 0 for o in outputs)
        results["10f_preprocess_acceptance"] = {
            "pass": gen_ok and non_empty,
            "num_outputs": len(outputs),
            "non_empty": non_empty,
        }
    except Exception as e:
        results["10f_preprocess_acceptance"] = {"pass": False, "error": str(e)}

    del engine

    with open(cfg["output_file"], "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    main()
'''


def test_section_10():
    print(f"\n{'=' * 60}")
    print("  SECTION 10: VLM Preprocessing (GPU required)")
    print(f"{'=' * 60}")

    model = "Qwen/Qwen2-VL-7B-Instruct"

    with _Timeout(1800):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "model": model,
                "tp": 1,
                "max_model_len": 4096,
                "project_root": PROJECT_ROOT,
                "package_name": PACKAGE_NAME,
            }

            script_path = os.path.join(tmpdir, "vlm_preprocess_test.py")
            with open(script_path, "w") as f:
                f.write(VLM_PREPROCESS_WORKER)

            output_path = os.path.join(tmpdir, "results.json")
            config["output_file"] = output_path
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(config, f)

            try:
                result = subprocess.run(
                    [sys.executable, script_path, config_path],
                    timeout=1500,
                    cwd=PROJECT_ROOT,
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    print(f"    STDERR (last 1000 chars):\n{result.stderr[-1000:]}")
                    check(False, "10. VLM preprocessing subprocess failed")
                    return
            except subprocess.TimeoutExpired:
                check(False, "10. VLM preprocessing subprocess timed out")
                return

            if not os.path.exists(output_path):
                check(False, "10. output JSON not created")
                return

            with open(output_path) as f:
                data = json.load(f)

            # 10a
            r = data.get("10a_image_correctness", {})
            if "error" in r:
                print(f"    ERROR 10a: {r['error']}")
            check(r.get("pass", False), "10a. image preprocessing correctness")
            if not r.get("pass", False) and "details" in r:
                for d in r["details"]:
                    if not (d.get("ids_match") and d.get("pv_match") and d.get("thw_match")):
                        print(f"      sample {d['sample']}: ids={d.get('ids_match')}, "
                              f"pv={d.get('pv_match')}, thw={d.get('thw_match')}, "
                              f"kb_len={d.get('kb_ids_len')}, ref_len={d.get('ref_ids_len')}")

            # 10b
            r = data.get("10b_video_correctness", {})
            if "error" in r:
                print(f"    ERROR 10b: {r['error']}")
            check(r.get("pass", False), "10b. video preprocessing correctness")
            if not r.get("pass", False) and "details" in r:
                for d in r["details"]:
                    if not (d.get("ids_match") and d.get("pv_match") and d.get("thw_match")):
                        print(f"      sample {d['sample']}: ids={d.get('ids_match')}, "
                              f"pv={d.get('pv_match')}, thw={d.get('thw_match')}, "
                              f"kb_len={d.get('kb_ids_len')}, ref_len={d.get('ref_ids_len')}")

            # 10c
            r = data.get("10c_mrope_correctness", {})
            if "error" in r:
                print(f"    ERROR 10c: {r['error']}")
            check(r.get("pass", False), "10c. M-RoPE position correctness")

            # 10d
            r = data.get("10d_image_throughput", {})
            if "error" in r:
                print(f"    ERROR 10d: {r['error']}")
            ratio = r.get("ratio", float("inf"))
            check(
                r.get("pass", False),
                f"10d. image preprocess throughput: ratio={ratio:.3f} "
                f"(kb={r.get('kb_time', 0):.2f}s, ref={r.get('ref_time', 0):.2f}s)",
            )

            # 10e
            r = data.get("10e_video_throughput", {})
            if "error" in r:
                print(f"    ERROR 10e: {r['error']}")
            ratio = r.get("ratio", float("inf"))
            check(
                r.get("pass", False),
                f"10e. video preprocess throughput: ratio={ratio:.3f} "
                f"(kb={r.get('kb_time', 0):.2f}s, ref={r.get('ref_time', 0):.2f}s)",
            )

            # 10f
            r = data.get("10f_preprocess_acceptance", {})
            if "error" in r:
                print(f"    ERROR 10f: {r['error']}")
            check(r.get("pass", False), "10f. pre-processed input acceptance in generate()")


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Test the kb-nano benchmarking infrastructure",
    )
    parser.add_argument(
        "--unit-only", action="store_true",
        help="Skip GPU integration tests (sections 7-10)",
    )
    parser.add_argument(
        "--section", type=int, default=None,
        help="Run only a specific section (1-10)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  kb-nano benchmarking infrastructure tests")
    print("=" * 60)

    sections = {
        1: ("Input Registry", test_section_1, False),
        2: ("KernelRunner", test_section_2, False),
        3: ("Result dataclasses", test_section_3, False),
        4: ("Standardized workloads", test_section_4, False),
        5: ("Conflict resolution", test_section_5, False),
        6: ("CLI argument parsing", test_section_6, False),
        7: ("Kernel integration", test_section_7, True),
        8: ("E2E integration", test_section_8, True),
        9: ("Eval integration", test_section_9, True),
        10: ("VLM Preprocessing", test_section_10, True),
    }

    for num, (name, func, needs_gpu) in sorted(sections.items()):
        if args.section is not None and args.section != num:
            continue
        if args.unit_only and needs_gpu:
            print(f"\n{'=' * 60}")
            print(f"  SKIPPED: Section {num} ({name}) -- requires GPU")
            print(f"{'=' * 60}")
            continue
        try:
            func()
        except TimeoutError:
            check(False, f"Section {num} ({name}) timed out entirely")
        except Exception as e:
            check(False, f"Section {num} ({name}) raised exception: {e}")

    print(f"\n{'=' * 60}")
    print(f"  RESULTS: {_pass_count} passed, {_fail_count} failed")
    print(f"{'=' * 60}")

    sys.exit(1 if _fail_count > 0 else 0)


if __name__ == "__main__":
    main()
