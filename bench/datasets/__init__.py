"""Real-prompt benchmark dataset assets, builders, and publishers.

Workflow:

  1. :mod:`build_real_prompts` fetches three source datasets from Hugging Face
     (LongBench, WildChat-1M, OpenThoughts-114k), tokenises them with
     Llama-3.1-8B-Instruct's chat template, applies the per-scenario caps,
     and writes one JSON artefact per scenario under ``real_prompts/``.

  2. :mod:`publish_real_prompts` converts those JSON artefacts into parquet
     and uploads them as public Hugging Face datasets at
     :data:`HF_REPO_IDS` (one repo per scenario).

  3. :func:`kb_nano.bench.utils.datasets.load_real_prompt_workload` downloads
     each scenario from the Hub at benchmark time via :mod:`datasets` and
     :mod:`huggingface_hub`; it never reads the local JSON artefacts.
"""

from __future__ import annotations

SCENARIO_NAMES: tuple[str, ...] = (
    "prefill-heavy",
    "balanced",
    "decode-heavy",
)

HF_REPO_OWNER = "sfc-gh-goliaro"

HF_REPO_IDS: dict[str, str] = {
    name: f"{HF_REPO_OWNER}/kb-nano-{name}" for name in SCENARIO_NAMES
}

__all__ = ["SCENARIO_NAMES", "HF_REPO_OWNER", "HF_REPO_IDS"]
