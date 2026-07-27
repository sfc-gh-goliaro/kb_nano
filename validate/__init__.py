"""``fastkernels validate`` — run the proper reference-library harness per model.

Resolves a scenario table (a path, or a packaged name like ``full`` / ``default``
/ ``minimal`` under ``fastkernels/scenarios/``) and, for each scenario, runs the
``bench_*.py`` harness that validates that model's family against its SOTA
reference library (vLLM, SGLang, FLA, diffusers, timm, …). The model→harness
choice mirrors ``capture.py``'s per-model dispatch: name-pattern predicates first,
then ``workloads.module_for`` → an explicit module→harness table.

A single ``--max-requests`` is translated to each current harness's own flag
where one exists (harnesses that lack an analog ignore it, with a note).
``--max-layers`` is accepted for CLI compatibility but not forwarded in this
port because the current engines and validation harnesses are preserved.
Scenarios are packed across the available GPUs by their TP degree; each harness
runs as its own subprocess (harnesses are script-shaped and already isolate
their own process).

Usage::

    fastkernels validate minimal                       # all-GPU, full workload
    fastkernels validate full --max-requests 8
    fastkernels validate minimal --dry-run             # print the plan, run nothing
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from math import floor
from pathlib import Path

from fastkernels import RESULTS_DIR

_VALIDATE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _VALIDATE_DIR.parent
_SCENARIO_DIR = _REPO_ROOT / "scenarios"
_DEFAULT_VALIDATE_ROOT = RESULTS_DIR / "validate"
_RAY_DASHBOARD_HOST = "127.0.0.1"
_RAY_PROGRESS_INTERVAL_SEC = 30.0


@dataclass(frozen=True)
class ValidateScenario:
    """Validation-only scenario supporting release-branch legacy workloads."""

    hf_name: str
    tp: int
    dtype: str
    legacy_workloads: tuple[str, ...]
    enforce_eager: bool = False
    max_num_seqs: int | None = None

# --- model -> harness dispatch (mirrors capture.py's per-model predicates) ---
# module stem (from workloads.module_for) -> harness filename (no .py).
_MODULE_TO_HARNESS: dict[str, str] = {
    # LLMs / VLMs / ASR — bench_vllm routes VLM/omni/whisper internally by --model.
    "llama": "bench_vllm", "deepseek": "bench_vllm", "mixtral": "bench_vllm",
    "gpt_oss": "bench_vllm", "gemma4": "bench_vllm", "mamba": "bench_vllm",
    "mamba2": "bench_vllm", "qwen3_next": "bench_vllm",
    "kimi_linear": "bench_vllm", "qwen2_vl": "bench_vllm", "qwen3_vl": "bench_vllm",
    "qwen2_5_omni": "bench_vllm", "whisper": "bench_vllm",
    # diffusion / video / TTS
    "flux": "bench_vllm_omni", "hunyuan_video": "bench_vllm_omni",
    "cosyvoice3": "bench_vllm_omni", "sdxl": "bench_diffusers",
    # vision encoders / classification / detection / segmentation
    "sam3": "bench_sam", "siglip2": "bench_timm", "dinov3": "bench_timm",
    "swinv2": "bench_timm", "mobilenetv4": "bench_timm",
    "convnextv2": "bench_image_cls", "efficientnetv2": "bench_image_cls",
    "yolov10": "bench_detection", "rtdetrv2": "bench_detection",
    # embeddings / recsys
    "bge_m3": "bench_embedding", "colbertv2": "bench_embedding",
    "dlrmv2": "bench_recsys", "lightgcn": "bench_recsys",
    # 3D / robotics / science / world models
    "gaussian_splatting": "bench_3dgs", "instant_ngp": "bench_instantngp",
    "pointtransformerv3": "bench_pointcloud", "openfold3": "bench_openfold3",
    "pi0": "bench_openpi", "dp3": "bench_dp3", "oasis": "bench_oasis",
    "vjepa2": "bench_vjepa2", "ttt_e2e": "bench_ttt_e2e", "llada": "bench_dllm",
}


def _harness_for(hf_name: str) -> str | None:
    """Harness filename (no ``.py``) for a model, or ``None`` if unmapped."""
    n = hf_name.lower()
    if "eagle3" in n:
        return "bench_sglang"
    if n.startswith("fla-hub/"):
        return "bench_fla"
    if "jamba" in n:
        return "bench_jamba"
    if "bitnet" in n:
        return "bench_microsoft_bitnet"
    if "llada" in n:
        return "bench_dllm"
    if "stable-diffusion" in n or "sdxl" in n:
        return "bench_diffusers"
    from fastkernels.workloads import module_for
    module = module_for(hf_name)
    return _MODULE_TO_HARNESS.get(module) if module else None


# --- per-harness flag adapter ------------------------------------------------
# How each harness names "number of requests/items". Harnesses not listed take no
# request-count flag, so --max-requests is ignored for them (with a printed note).
_REQUESTS_FLAG = {
    "bench_vllm": "--num-seqs",
    "bench_fla": "--num-seqs",
    "bench_jamba": "--num-seqs",
    "bench_sglang": "--num-seqs",
    "bench_microsoft_bitnet": "--num-prompts",
    "bench_detection": "--num-images",
    "bench_timm": "--num-images",
    "bench_image_cls": "--num-images",
    "bench_sam": "--num-items",
    "bench_dllm": "--max-samples",
    "bench_dp3": "--num-requests",
    "bench_openpi": "--num-requests",
    "bench_openfold3": "--num-seqs",
    "bench_pointcloud": "--max-samples",
    "bench_vjepa2": "--num-videos",
    "bench_ttt_e2e": "--n-sequences",
}
_TP_OK = {"bench_vllm", "bench_embedding"}          # accept --tp
_MAXLAYERS_OK: set[str] = set()                      # preserved harnesses do not accept --max-layers
_EAGER_OK = {"bench_vllm", "bench_sglang", "bench_diffusers", "bench_embedding"}

_HF_MODEL_ARG = {
    "bench_vllm",
    "bench_sglang",
    "bench_fla",
    "bench_jamba",
    "bench_microsoft_bitnet",
    "bench_vllm_omni",
    "bench_diffusers",
    "bench_sam",
    "bench_timm",
    "bench_image_cls",
    "bench_detection",
    "bench_embedding",
    "bench_dllm",
    "bench_pointcloud",
    "bench_oasis",
    "bench_openpi",
    "bench_vjepa2",
}
_MODULE_MODEL_ARG = {"bench_recsys"}


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _safe_slug(text: str, *, max_len: int | None = 96) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")
    slug = re.sub(r"_+", "_", slug)
    slug = slug or "scenario"
    return slug[:max_len] if max_len is not None else slug


def _scenario_label(name_or_path: str | Path) -> str:
    text = str(name_or_path)
    stem = Path(text).stem if any(sep in text for sep in ("/", "\\")) else text
    if stem.endswith((".yaml", ".yml")):
        stem = Path(stem).stem
    return _safe_slug(stem, max_len=None)


def _scenario_workloads(scenario) -> tuple[str, ...]:
    legacy = getattr(scenario, "legacy_workloads", None)
    if legacy:
        return tuple(str(w) for w in legacy)
    return tuple(getattr(w, "value", str(w)) for w in getattr(scenario, "workloads", ()))


def _module_for_scenario(scenario) -> str | None:
    from fastkernels.workloads import module_for
    return module_for(scenario.hf_name)


def _model_args(scenario, harness: str) -> list[str]:
    if harness == "bench_sglang":
        raw = scenario.hf_name
        if " + " in raw:
            target, draft = raw.split(" + ", 1)
            draft = draft.replace("(draft)", "").strip()
            return ["--model", target.strip(), "--draft-model", draft]
        return ["--model", raw]
    if harness in _HF_MODEL_ARG:
        return ["--model", scenario.hf_name]
    if harness in _MODULE_MODEL_ARG:
        module = _module_for_scenario(scenario)
        return ["--model", module] if module else []
    return []


def _build_cmd(
    scenario,
    harness: str,
    args,
    *,
    output_dir: Path | None = None,
) -> list[str]:
    cmd = [
        sys.executable,
        "-u",
        str(_VALIDATE_DIR / f"{harness}.py"),
        *_model_args(scenario, harness),
    ]
    if harness in _TP_OK and scenario.tp:
        cmd += ["--tp", str(scenario.tp)]
    if args.max_layers is not None and harness in _MAXLAYERS_OK:
        cmd += ["--max-layers", str(args.max_layers)]
    if args.max_requests is not None:
        flag = _REQUESTS_FLAG.get(harness)
        if flag:
            cmd += [flag, str(args.max_requests)]
        else:
            print(f"    note: {harness} has no request-count flag; "
                f"--max-requests ignored")
    if scenario.enforce_eager and harness in _EAGER_OK:
        cmd += ["--enforce-eager"]
    if output_dir is not None:
        cmd += ["--output-dir", str(output_dir)]
    return cmd


def _job_paths(root: Path, index: int, scenario, harness: str) -> tuple[Path, Path]:
    name = _safe_slug(f"{index:03d}_{harness}_{scenario.hf_name}")
    run_dir = root / name
    return run_dir, run_dir / "run.log"


def _gpu_ids_from_env() -> list[str]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    return [t.strip() for t in raw.split(",") if t.strip()]


def _visible_to_physical_gpu_ids(gpu_ids: list[str], parent_visible: list[str]) -> list[str]:
    """Map Ray logical GPU ids back to parent-visible physical ids when possible."""
    out: list[str] = []
    for gid in gpu_ids:
        try:
            idx = int(gid)
        except ValueError:
            out.append(gid)
            continue
        if parent_visible and 0 <= idx < len(parent_visible):
            out.append(parent_visible[idx])
        else:
            out.append(gid)
    return out


def _numa_nodes_for_gpus(gpu_ids: list[str]) -> list[str]:
    nodes: set[str] = set()
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return []
    for gid in gpu_ids:
        if not gid.isdigit():
            continue
        try:
            out = subprocess.run(
                [smi, "topo", "-C", "-i", gid],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            ).stdout
        except Exception:
            continue
        m = re.search(r"NUMA IDs of closest CPU:\s*([0-9, -]+)", out)
        if m:
            for token in re.split(r"[, ]+", m.group(1).strip()):
                if token:
                    nodes.add(token)
    return sorted(nodes, key=int)


def _numactl_prefix(gpu_ids: list[str], mode: str) -> list[str]:
    if mode == "off":
        return []
    numactl = shutil.which("numactl")
    if numactl is None:
        return []
    nodes = _numa_nodes_for_gpus(gpu_ids)
    if not nodes:
        return []
    node_list = ",".join(nodes)
    prefix = [numactl, f"--cpunodebind={node_list}"]
    if mode == "strict":
        prefix.append(f"--membind={node_list}")
    return prefix


def _total_memory_bytes() -> int:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages) * int(page_size)
    except (ValueError, OSError, AttributeError):
        return 0


def _reserved_cpus(total_cpus: int) -> int:
    # Keep Ray's driver, dashboard, GCS, log monitor, and OS services responsive.
    return min(max(4, total_cpus // 16), max(1, total_cpus - 1))


def _reserved_memory_bytes(total_memory: int) -> int:
    if total_memory <= 0:
        return 0
    sixteen_gib = 16 * 1024 ** 3
    return min(max(sixteen_gib, total_memory // 10), max(0, total_memory - 1024 ** 3))


def _ray_resource_options(tp: int, total_gpus: int, cluster_resources: dict) -> dict:
    total_cpus = int(cluster_resources.get("CPU") or os.cpu_count() or 1)
    reserved_cpus = _reserved_cpus(total_cpus)
    alloc_cpus = max(1, total_cpus - reserved_cpus)
    fraction = tp / max(1, total_gpus)
    num_cpus = max(1, floor(alloc_cpus * fraction))
    opts = {
        "num_gpus": tp,
        "num_cpus": num_cpus,
    }

    total_mem = int(cluster_resources.get("memory") or _total_memory_bytes())
    reserved_mem = _reserved_memory_bytes(total_mem)
    alloc_mem = max(0, total_mem - reserved_mem)
    if alloc_mem > 0:
        # Keep a little slack because Ray's object store/system reserve is not
        # identical to total system memory.
        opts["memory"] = max(256 * 1024 ** 2, int(alloc_mem * fraction * 0.85))
    return opts


def _scenario_path(name_or_path: str | Path) -> Path:
    path = Path(name_or_path)
    if path.is_file():
        return path
    stem = str(name_or_path)
    if not stem.endswith((".yaml", ".yml")):
        stem = f"{stem}.yaml"
    packaged = _SCENARIO_DIR / stem
    if packaged.is_file():
        return packaged
    raise FileNotFoundError(
        f"scenario table {name_or_path!r} not found as a path or in {_SCENARIO_DIR}"
    )


def _load_legacy_validate_scenarios(path: Path) -> list[ValidateScenario] | None:
    """Load validation tables that use release-branch legacy workload names.

    Returns ``None`` when the YAML is a new workload table and should be parsed
    by ``fastkernels.workloads.resolve_benchmark`` instead.
    """
    import yaml

    data = yaml.safe_load(path.read_text()) or {}
    entries = data.get("scenarios")
    if not isinstance(entries, list):
        raise ValueError(f"{path.name}: expected top-level 'scenarios' list")
    if not any("legacy_workloads" in e for e in entries if isinstance(e, dict)):
        return None

    allowed_dtypes = {"bfloat16", "float16", "float32", "fp8", "mxfp4"}
    scenarios: list[ValidateScenario] = []
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"{path.name}: scenario #{idx} is not a mapping")
        try:
            model = str(entry["model"])
            tp = int(entry["tp"])
            dtype = str(entry["dtype"])
            workloads = entry["legacy_workloads"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path.name}: bad scenario #{idx}: {exc}") from exc
        if dtype not in allowed_dtypes:
            raise ValueError(
                f"{path.name}: {model}: dtype {dtype!r} not in {sorted(allowed_dtypes)}"
            )
        if not isinstance(workloads, list) or not workloads:
            raise ValueError(f"{path.name}: {model}: legacy_workloads must be a non-empty list")
        scenarios.append(
            ValidateScenario(
                hf_name=model,
                tp=tp,
                dtype=dtype,
                legacy_workloads=tuple(str(w) for w in workloads),
                enforce_eager=bool(entry.get("enforce_eager", False)),
                max_num_seqs=entry.get("max_num_seqs"),
            )
        )
    return scenarios


def _resolve_validate_scenarios(name_or_path: str | Path):
    path = _scenario_path(name_or_path)
    legacy = _load_legacy_validate_scenarios(path)
    if legacy is not None:
        return legacy
    from fastkernels.workloads import resolve_benchmark
    return resolve_benchmark(path)


# --- GPU pool ----------------------------------------------------------------
def _detect_gpus(explicit: str | None) -> list[str]:
    if explicit:
        return [t.strip() for t in explicit.split(",") if t.strip()]
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env:
        return [t.strip() for t in env.split(",") if t.strip()]
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            out = subprocess.run(
                [smi, "--query-gpu=index", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=30, check=True).stdout
            ids = [ln.strip() for ln in out.splitlines() if ln.strip()]
            if ids:
                return ids
        except Exception:  # noqa: BLE001
            pass
    return ["0"]


def _kill_group(proc: subprocess.Popen) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            return
        for _ in range(40):
            if proc.poll() is not None:
                return
            time.sleep(0.2)


# --- output styling ----------------------------------------------------------
def _c(text: str, code: str) -> str:
    """ANSI-wrap when stdout is a TTY, else return plain."""
    return f"\033[{code}m{text}\033[0m" if sys.stdout.isatty() else text


def _tail(path: Path, n: int) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return ""


def _append_run_event(root: Path, event: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    payload = {"ts": time.time(), **event}
    with (root / "run.jsonl").open("a") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def _make_job(index: int, scenario, harness: str, args, root: Path) -> dict:
    run_dir, log_path = _job_paths(root, index, scenario, harness)
    return {
        "index": index,
        "name": scenario.hf_name,
        "tp": int(scenario.tp),
        "dtype": scenario.dtype,
        "harness": harness,
        "legacy_workloads": list(_scenario_workloads(scenario)),
        "cmd": _build_cmd(scenario, harness, args, output_dir=run_dir),
        "run_dir": str(run_dir),
        "log_path": str(log_path),
    }


def _plan_jobs(scenarios, args, total_gpus: int, root: Path) -> tuple[list[dict], dict[int, str]]:
    jobs: list[dict] = []
    results: dict[int, str] = {}
    for i, s in enumerate(scenarios):
        harness = _harness_for(s.hf_name)
        if harness is None:
            print(f"  {_c('-', '2')} skip {s.hf_name}: no harness mapped for this model")
            results[i] = "SKIP(no-harness)"
        elif s.tp > total_gpus:
            print(f"  {_c('-', '2')} skip {s.hf_name}: needs tp={s.tp} > {total_gpus} GPU(s)")
            results[i] = "SKIP(tp>gpus)"
        else:
            jobs.append(_make_job(i, s, harness, args, root))
    return jobs, results


def _kill_process_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    for _ in range(40):
        if proc.poll() is not None:
            return
        time.sleep(0.2)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        return


def _run_job_subprocess(
    job: dict,
    *,
    timeout: int,
    repo_root: str,
    env_updates: dict[str, str] | None = None,
    explicit_gpus: list[str] | None = None,
    parent_visible_gpus: list[str] | None = None,
    numactl_mode: str = "off",
) -> dict:
    """Run one validate/bench_*.py harness in a subprocess."""
    start = time.monotonic()
    run_dir = Path(job["run_dir"])
    log_path = Path(job["log_path"])
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env["FASTKERNELS_VALIDATE_JOB_INDEX"] = str(job["index"])
    env["FASTKERNELS_VALIDATE_JOB_NAME"] = _safe_slug(job["name"], max_len=120)
    if env_updates:
        env.update(env_updates)
    if explicit_gpus is not None:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(explicit_gpus)

    visible_gpus = [t.strip() for t in env.get("CUDA_VISIBLE_DEVICES", "").split(",") if t.strip()]
    physical_gpus = _visible_to_physical_gpu_ids(visible_gpus, parent_visible_gpus or [])
    prefix = _numactl_prefix(physical_gpus or visible_gpus, numactl_mode)
    cmd = [*prefix, *job["cmd"]]

    rc = -1
    timed_out = False
    with log_path.open("w", buffering=1) as log:
        log.write(f"job_index: {job['index']}\n")
        log.write(f"name: {job['name']}\n")
        log.write(f"harness: {job['harness']}\n")
        log.write(f"tp: {job['tp']}\n")
        log.write(f"legacy_workloads: {', '.join(job.get('legacy_workloads') or [])}\n")
        log.write(f"CUDA_VISIBLE_DEVICES: {env.get('CUDA_VISIBLE_DEVICES', '')}\n")
        log.write(f"physical_gpus_for_numa: {','.join(physical_gpus)}\n")
        log.write(f"numactl_prefix: {' '.join(prefix)}\n")
        log.write(f"run_dir: {run_dir}\n")
        log.write("command: " + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
            cwd=repo_root,
            start_new_session=True,
        )
        old_handlers = {}

        def _pump_stdout() -> None:
            if proc.stdout is None:
                return
            for line in proc.stdout:
                log.write(line)
                print(line, end="", flush=True)

        pump = threading.Thread(target=_pump_stdout, daemon=True)
        pump.start()

        def _forward_signal(signum, _frame):
            _kill_process_group(proc)
            raise SystemExit(128 + int(signum))

        for sig in (signal.SIGTERM, signal.SIGINT):
            old_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, _forward_signal)
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_process_group(proc)
            rc = proc.poll() if proc.poll() is not None else -9
        finally:
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
            pump.join(timeout=5.0)

    elapsed = time.monotonic() - start
    return {
        "index": job["index"],
        "name": job["name"],
        "harness": job["harness"],
        "tp": job["tp"],
        "returncode": rc,
        "timed_out": timed_out,
        "elapsed_s": elapsed,
        "log_path": str(log_path),
        "run_dir": str(run_dir),
        "visible_gpus": visible_gpus,
        "physical_gpus": physical_gpus,
        "status": "PASS" if rc == 0 else f"FAIL(rc={rc})",
    }


def _ray_run_job(
    job: dict,
    timeout: int,
    repo_root: str,
    parent_visible_gpus: list[str],
    numactl_mode: str,
) -> dict:
    """Ray task entry point. Ray sets CUDA_VISIBLE_DEVICES for this process."""
    return _run_job_subprocess(
        job,
        timeout=timeout,
        repo_root=repo_root,
        parent_visible_gpus=parent_visible_gpus,
        numactl_mode=numactl_mode,
    )


def _ray_dashboard_url(ray) -> str | None:
    try:
        get_url = getattr(ray, "get_dashboard_url", None)
        if get_url is not None:
            return get_url()
    except Exception:
        pass
    try:
        return ray._private.worker._global_node.webui_url  # noqa: SLF001
    except Exception:
        return None


def _ray_job_id(scenario_name: str, job: dict) -> str:
    model = _safe_slug(job["name"], max_len=None)
    datasets = _safe_slug("_".join(job.get("legacy_workloads") or ["workloads"]), max_len=None)
    return (
        f"validate_{scenario_name}_{job['index']:03d}_{model}_"
        f"tp{job['tp']}_{datasets}"
    )


def _dashboard_http_url(ray) -> str | None:
    dash = _ray_dashboard_url(ray)
    if not dash:
        return None
    return dash if dash.startswith("http") else f"http://{dash}"


def _run_ray(scenarios, args, gpus: list[str], root: Path) -> int:
    try:
        import ray
    except ModuleNotFoundError:
        print("error: Ray is not installed. Install with `pip install 'ray[default]'`.", file=sys.stderr)
        return 2

    timeout = int(os.environ.get("FASTKERNELS_VALIDATE_TIMEOUT_SEC", str(args.timeout)))
    jobs, results = _plan_jobs(scenarios, args, len(gpus), root)
    root.mkdir(parents=True, exist_ok=True)
    if not jobs:
        return _summary(scenarios, results)

    previous_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)

    try:
        ray.init(
            include_dashboard=True,
            dashboard_host=_RAY_DASHBOARD_HOST,
            ignore_reinit_error=True,
            log_to_driver=False,
        )
    finally:
        if args.gpus:
            if previous_cvd is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = previous_cvd

    dashboard_url = _dashboard_http_url(ray)
    if dashboard_url is None:
        print("error: Ray dashboard URL is unavailable.", file=sys.stderr)
        return 2
    cluster_resources = ray.cluster_resources()

    print(
        f"\n{_c('▶', '36')} submitting {_c(str(len(jobs)), '1')} named Ray task(s) "
        f"across {len(gpus)} visible GPU(s)"
    )
    print(f"  Ray dashboard: {dashboard_url}")
    print(f"  output root: {root}")
    print(f"  run log: {root / 'run.jsonl'}\n", flush=True)
    _append_run_event(root, {
        "event": "run_start",
        "scenario_table": str(args.scenarios),
        "visible_gpus": gpus,
        "dashboard_url": dashboard_url,
        "job_count": len(jobs),
        "ray_mode": "named_tasks",
    })

    remote_fn = ray.remote(_ray_run_job)
    refs = []
    ref_to_job: dict[object, dict] = {}
    scenario_name = _scenario_label(args.scenarios)
    for job in jobs:
        task_name = _ray_job_id(scenario_name, job)
        opts = _ray_resource_options(job["tp"], len(gpus), cluster_resources)
        ref = remote_fn.options(name=task_name, **opts).remote(
            job,
            timeout,
            str(_REPO_ROOT),
            gpus,
            args.numactl_mode,
        )
        refs.append(ref)
        ref_to_job[ref] = job
        _append_run_event(root, {
            "event": "task_submitted",
            "task_name": task_name,
            "job": job,
            "resources": opts,
        })
        print(
            f"{_c('▷', '36')} {task_name}  [{job['index']}] {job['name']} "
            f"{_c('→ ' + job['harness'], '2')}  tp={job['tp']}  "
            f"log={job['log_path']}",
            flush=True,
        )

    pending = list(refs)
    last_heartbeat = time.monotonic()
    while pending:
        ready, pending = ray.wait(pending, num_returns=1, timeout=2.0)
        if not ready:
            now = time.monotonic()
            if now - last_heartbeat >= _RAY_PROGRESS_INTERVAL_SEC:
                print(f"  running/pending tasks: {len(pending)}", flush=True)
                last_heartbeat = now
            continue
        for ref in ready:
            job = ref_to_job[ref]
            task_name = _ray_job_id(scenario_name, job)
            try:
                res = ray.get(ref)
                result_status = res.get("status", "PASS")
            except Exception as exc:  # noqa: BLE001
                result_status = f"FAIL(ray:{type(exc).__name__})"
                res = {
                    "elapsed_s": 0.0,
                    "log_path": job["log_path"],
                    "run_dir": job["run_dir"],
                    "error": repr(exc),
                }
            results[job["index"]] = result_status
            _append_run_event(root, {
                "event": "task_finished",
                "task_name": task_name,
                "status": result_status,
                "result": res,
            })
            ok = result_status == "PASS"
            mark = _c("✓", "32") if ok else _c("✗", "31")
            print(
                f"{mark} {task_name}  [{job['index']}] {job['name']}  ·  "
                f"{job['harness']}  ·  {result_status} "
                f"({int(res.get('elapsed_s', 0))}s)  ·  {res.get('log_path', job['log_path'])}",
                flush=True,
            )
            if not ok:
                for ln in _tail(Path(res.get("log_path", job["log_path"])), 12).splitlines():
                    print(f"    {_c(ln, '2')}")
    rc = _summary(scenarios, results)
    _append_run_event(root, {
        "event": "run_finished",
        "status": "PASS" if rc == 0 else "FAIL",
        "results": results,
    })
    _write_and_print_validate_summary(root, scenarios, results)
    return rc


def _validation_root(args) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    name = _safe_slug(Path(str(args.scenarios)).stem)
    return _DEFAULT_VALIDATE_ROOT / f"{name}_{_timestamp()}"


# --- scheduler ---------------------------------------------------------------
def _summary(scenarios, results: dict[int, str]) -> int:
    print(_c("\nvalidate summary", "1"))
    for i, s in enumerate(scenarios):
        r = results.get(i, "?")
        mark = (_c("✓", "32") if r == "PASS"
                else _c("-", "2") if r.startswith("SKIP") else _c("✗", "31"))
        print(f"  {mark} {r:16} {s.hf_name}")
    return 0 if all(v == "PASS" for v in results.values()) else 1


def _load_task_results(root: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    path = root / "run.jsonl"
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("event") != "task_finished":
            continue
        result = event.get("result") or {}
        idx = result.get("index")
        if isinstance(idx, int):
            out[idx] = {**result, "task_name": event.get("task_name")}
    return out


def _fmt_speedup(value) -> str:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return "-"
    return f"{value:.2f}x"


def _alignment_summary(alignment: dict | None) -> str:
    if not isinstance(alignment, dict):
        return "-"
    avg = alignment.get("avg_matching_tokens_per_request")
    exact = alignment.get("exact_matches")
    total = alignment.get("total_seqs")
    parts: list[str] = []
    if isinstance(avg, (int, float)):
        parts.append(f"avg prefix match len: {avg:.1f}")
    if isinstance(exact, int) and isinstance(total, int) and total:
        parts.append(f"exact match: {exact}/{total} ({exact / total * 100:.1f}%)")
    return "; ".join(parts) if parts else "-"


def _cosine_summary(items: list[float]) -> str:
    vals = [v for v in items if isinstance(v, (int, float))]
    return f"min_cos={min(vals):.4f}" if vals else "-"


def _reference_name(harness: str) -> str:
    return {
        "bench_vllm": "vLLM",
        "bench_fla": "FLA",
        "bench_vllm_omni": "vllm-omni",
        "bench_detection": "reference",
        "bench_openfold3": "reference",
        "bench_embedding": "vLLM",
        "bench_oasis": "open-oasis",
    }.get(harness, "reference")


def _throughput_rows_for_result(model: str, harness: str, data: dict) -> list[dict]:
    rows: list[dict] = []
    ref = _reference_name(harness)
    if harness in {"bench_vllm", "bench_fla"}:
        for item in data.get("scenarios", []):
            rows.append({
                "model": model,
                "workload": item.get("scenario") or item.get("name"),
                "reference": ref,
                "speedup": item.get("speedup"),
                "correctness": _alignment_summary(item.get("alignment")),
            })
    elif harness == "bench_vllm_omni":
        correctness = data.get("correctness") or {}
        for item in data.get("fastkernels", {}).get("throughput", []):
            name = item.get("scenario") or item.get("name")
            ref_item = next(
                (
                    x for x in data.get("vllm_omni", {}).get("throughput", [])
                    if (x.get("scenario") or x.get("name")) == name
                ),
                {},
            )
            fk_rate = (
                item.get("images_per_second")
                or item.get("items_per_second")
                or item.get("samples_per_second")
            )
            ref_rate = (
                ref_item.get("images_per_second")
                or ref_item.get("items_per_second")
                or ref_item.get("samples_per_second")
            )
            corr = correctness.get(name, {}) if isinstance(correctness, dict) else {}
            cosines: list[float] = []
            if isinstance(corr, dict):
                for key, value in corr.items():
                    if isinstance(value, dict) and isinstance(value.get("cosine"), (int, float)):
                        cosines.append(value["cosine"])
                    elif key.endswith("cosine") and isinstance(value, (int, float)):
                        cosines.append(value)
                    elif key.endswith("cosine_sim") and isinstance(value, (int, float)):
                        cosines.append(value)
            rows.append({
                "model": model,
                "workload": name,
                "reference": ref,
                "speedup": (fk_rate / ref_rate) if fk_rate and ref_rate else item.get("speedup"),
                "correctness": _cosine_summary(cosines),
            })
    elif harness == "bench_detection":
        corr = data.get("correctness") or {}
        ref_items = data.get("reference", {}).get("throughput", [])
        for idx, item in enumerate(data.get("fastkernels", {}).get("throughput", [])):
            ref_item = ref_items[idx] if idx < len(ref_items) else {}
            fk_rate = item.get("images_per_second")
            ref_rate = ref_item.get("images_per_second")
            rows.append({
                "model": model,
                "workload": item.get("name"),
                "reference": ref,
                "speedup": (fk_rate / ref_rate) if fk_rate and ref_rate else item.get("speedup"),
                "correctness": (
                    f"boxes={corr.get('boxes_cosine', 0):.4f}; "
                    f"scores={corr.get('scores_cosine', 0):.4f}; "
                    f"labels={corr.get('labels_match_rate', 0) * 100:.1f}%"
                ),
            })
    elif harness == "bench_openfold3":
        for item in data.get("throughput_scenarios", []):
            align = item.get("alignment") or {}
            rows.append({
                "model": model,
                "workload": item.get("scenario"),
                "reference": ref,
                "speedup": item.get("speedup"),
                "correctness": (
                    f"align={align.get('pass_rate', 0) * 100:.1f}%"
                    if align else "-"
                ),
            })
    elif harness == "bench_embedding":
        for item in data.get("throughput_scenarios", []):
            corr = item.get("correctness") or {}
            rows.append({
                "model": model,
                "workload": item.get("scenario"),
                "reference": ref,
                "speedup": item.get("speedup"),
                "correctness": (
                    f"pass={corr.get('pass')}; min_cos={corr.get('min_cosine', 0):.6f}"
                    if corr else "-"
                ),
            })
    elif harness == "bench_oasis":
        for item in data.get("performance", []):
            scenario = item.get("scenario")
            name = scenario.get("name") if isinstance(scenario, dict) else scenario
            corr = item.get("correctness") or {}
            cosines: list[float] = []
            passes: list[bool] = []
            if isinstance(corr, dict):
                for value in corr.values():
                    if isinstance(value, dict):
                        if isinstance(value.get("pass"), bool):
                            passes.append(value["pass"])
                        if isinstance(value.get("cosine"), (int, float)):
                            cosines.append(value["cosine"])
            correctness = _cosine_summary(cosines)
            if passes:
                correctness = f"pass={all(passes)}; {correctness}"
            rows.append({
                "model": model,
                "workload": name,
                "reference": ref,
                "speedup": item.get("speedup"),
                "correctness": correctness,
            })
    return rows


def _latency_rows_for_result(model: str, harness: str, data: dict) -> list[dict]:
    rows: list[dict] = []
    ref = _reference_name(harness)
    if harness in {"bench_vllm", "bench_fla", "bench_embedding"}:
        for item in data.get("latency_scenarios", []):
            rows.append({
                "model": model,
                "workload": item.get("scenario") or item.get("name"),
                "reference": ref,
                "speedup": item.get("speedup"),
            })
    elif harness == "bench_vllm_omni":
        ref_items = data.get("vllm_omni", {}).get("latency", [])
        for idx, item in enumerate(data.get("fastkernels", {}).get("latency", [])):
            ref_item = ref_items[idx] if idx < len(ref_items) else {}
            fk_med = item.get("median_s")
            ref_med = ref_item.get("median_s")
            rows.append({
                "model": model,
                "workload": item.get("scenario") or item.get("name"),
                "reference": ref,
                "speedup": (ref_med / fk_med) if fk_med and ref_med else item.get("speedup"),
            })
    elif harness == "bench_detection":
        ref_items = data.get("reference", {}).get("latency", [])
        for idx, item in enumerate(data.get("fastkernels", {}).get("latency", [])):
            ref_item = ref_items[idx] if idx < len(ref_items) else {}
            fk_med = item.get("median_s")
            ref_med = ref_item.get("median_s")
            rows.append({
                "model": model,
                "workload": item.get("name"),
                "reference": ref,
                "speedup": (ref_med / fk_med) if fk_med and ref_med else item.get("speedup"),
            })
    elif harness == "bench_openfold3":
        for item in data.get("latency_scenarios", []):
            rows.append({
                "model": model,
                "workload": item.get("scenario"),
                "reference": ref,
                "speedup": item.get("speedup"),
            })
    return rows


def _format_table(rows: list[dict], headers: list[tuple[str, str]]) -> str:
    widths = [len(label) for _key, label in headers]
    rendered: list[list[str]] = []
    for row in rows:
        values = []
        for key, _label in headers:
            value = row.get(key)
            if key == "speedup":
                value = _fmt_speedup(value)
            else:
                value = "-" if value is None else str(value)
            values.append(value)
        rendered.append(values)
        for idx, value in enumerate(values):
            widths[idx] = max(widths[idx], len(value))
    lines = [
        "  ".join(label.ljust(widths[idx]) for idx, (_key, label) in enumerate(headers)),
        "  ".join("-" * width for width in widths),
    ]
    for values in rendered:
        lines.append("  ".join(values[idx].ljust(widths[idx]) for idx in range(len(headers))))
    return "\n".join(lines)


def _build_validate_summary(root: Path, scenarios, results: dict[int, str]) -> dict:
    task_results = _load_task_results(root)
    throughput: list[dict] = []
    latency: list[dict] = []
    models: list[dict] = []
    for idx, scenario in enumerate(scenarios):
        task = task_results.get(idx, {})
        run_dir = Path(task.get("run_dir", ""))
        results_path = run_dir / "results.json" if run_dir else Path()
        harness = task.get("harness") or _harness_for(scenario.hf_name)
        data = {}
        if results_path.is_file():
            try:
                data = json.loads(results_path.read_text())
            except json.JSONDecodeError:
                data = {}
        model = data.get("model") or task.get("name") or scenario.hf_name
        model_entry = {
            "index": idx,
            "model": model,
            "harness": harness,
            "reference": _reference_name(harness or ""),
            "tp": data.get("tp") or task.get("tp") or scenario.tp,
            "status": results.get(idx, task.get("status", "?")),
            "paths": {
                "run_log": str(run_dir / "run.log") if run_dir else None,
                "results_json": str(results_path) if results_path else None,
            },
        }
        models.append(model_entry)
        if data and harness:
            throughput.extend(_throughput_rows_for_result(model, harness, data))
            latency.extend(_latency_rows_for_result(model, harness, data))
    return {
        "run": {
            "status": "PASS" if all(v == "PASS" for v in results.values()) else "FAIL",
            "root": str(root),
            "models_total": len(models),
            "models_passed": sum(1 for m in models if m["status"] == "PASS"),
        },
        "models": models,
        "throughput": throughput,
        "latency": latency,
    }


def _write_and_print_validate_summary(root: Path, scenarios, results: dict[int, str]) -> None:
    summary = _build_validate_summary(root, scenarios, results)
    path = root / "summary.json"
    path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary saved to: {path}")
    if summary["throughput"]:
        print("\nTHROUGHPUT / E2E WORKLOADS")
        print(_format_table(
            summary["throughput"],
            [
                ("model", "MODEL"),
                ("workload", "WORKLOAD"),
                ("reference", "REFERENCE"),
                ("speedup", "SPEEDUP"),
                ("correctness", "CORRECTNESS"),
            ],
        ))
    if summary["latency"]:
        print("\nLATENCY WORKLOADS")
        print(_format_table(
            summary["latency"],
            [
                ("model", "MODEL"),
                ("workload", "WORKLOAD"),
                ("reference", "REFERENCE"),
                ("speedup", "SPEEDUP"),
            ],
        ))



def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="fastkernels validate",
        description="Run the proper reference-library bench harness for each model "
                    "in a scenario table (fastkernels vs SOTA reference).")
    p.add_argument("scenarios",
                   help="Scenario table: a path, or a packaged name resolved "
                        "against fastkernels/scenarios/. Validation tables may "
                        "use release-branch legacy_workloads.")
    p.add_argument("--max-requests", type=int, default=None,
                   help="Cap requests/items per harness (translated to its own flag, "
                        "e.g. --num-seqs / --num-images; ignored where unsupported).")
    p.add_argument("--max-layers", type=int, default=None,
                   help="Accepted for compatibility but not forwarded in this "
                        "port; the current engines and harnesses are preserved.")
    p.add_argument("--gpus", default=None,
                   help="Comma-separated GPU ids to pack across (default: all visible).")
    p.add_argument("--output-dir", default=None,
                   help="Root directory containing run.jsonl plus one directory per scenario "
                        "(default: ~/.fastkernels/results/validate/"
                        "<scenario>_<timestamp>).")
    p.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Per-scenario wall-clock timeout in seconds. Can also be set with "
             "FASTKERNELS_VALIDATE_TIMEOUT_SEC.",
    )
    p.add_argument(
        "--numactl-mode",
        choices=("off", "cpu", "strict"),
        default="cpu",
        help="NUMA binding for each harness subprocess. 'cpu' binds CPU nodes "
             "closest to assigned GPUs; 'strict' also binds memory.",
    )
    p.add_argument("--dry-run", action="store_true",
                   help="Print the chosen harness + command per scenario; run nothing.")
    args = p.parse_args(argv)
    if args.max_layers is not None and not _MAXLAYERS_OK:
        print(
            "note: --max-layers is not forwarded because this port preserves "
            "the current validation harnesses and engines."
        )

    try:
        scenarios = _resolve_validate_scenarios(args.scenarios)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: could not load scenarios {args.scenarios!r}: {exc}")
        return 2

    if args.dry_run:
        for i, s in enumerate(scenarios):
            harness = _harness_for(s.hf_name)
            if harness is None:
                print(f"[{i}] {s.hf_name}: NO HARNESS MAPPED")
                continue
            cmd = _build_cmd(s, harness, args)
            print(f"[{i}] {s.hf_name}  (tp={s.tp}, dtype={s.dtype}) -> {harness}")
            legacy_workloads = getattr(s, "legacy_workloads", None)
            if legacy_workloads:
                print("      legacy workloads: " + ", ".join(legacy_workloads))
            print("      " + " ".join(cmd))
        return 0

    gpus = _detect_gpus(args.gpus)
    root = _validation_root(args)
    return _run_ray(scenarios, args, gpus, root)


if __name__ == "__main__":
    raise SystemExit(main())
