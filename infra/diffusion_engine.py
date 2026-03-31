"""
Diffusion inference engine for FLUX-style DiT models.

Unlike the autoregressive LlamaEngine (paged KV, continuous batching),
this engine runs iterative denoising loops with no KV cache.

Mirrors vllm-omni's DiffusionEngine / DiffusionModelRunner.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from glob import glob
from typing import Any

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

from ..tasks.baseline.L4.flux import (
    DiffusionOutput,
    DiffusionSamplingParams,
    FluxConfig,
    FluxPipeline,
)

logger = logging.getLogger(__name__)


def _download_flux_model(model_name: str) -> str:
    """Download FLUX model weights from HuggingFace."""
    return snapshot_download(
        model_name,
        allow_patterns=[
            "*.safetensors", "*.json", "*.txt", "*.model",
            "tokenizer*", "scheduler/*", "text_encoder/*",
            "text_encoder_2/*", "vae/*", "transformer/*",
        ],
    )


def _load_safetensors_from_dir(directory: str) -> list[tuple[str, torch.Tensor]]:
    """Load all safetensors from a directory."""
    safetensor_files = sorted(glob(os.path.join(directory, "*.safetensors")))
    weights = []
    for sf_file in safetensor_files:
        with safe_open(sf_file, "pt", "cpu") as f:
            for key in f.keys():
                weights.append((key, f.get_tensor(key)))
    return weights


def _load_flux_weights(
    pipeline: FluxPipeline,
    model_path: str,
) -> None:
    """Load transformer and T5 encoder weights from safetensors.

    CLIP weights are loaded by ``from_pretrained`` in the pipeline
    constructor.  The transformer and T5 encoder use custom weight
    loaders for TP-aware stacked-param mapping.
    """
    transformer_dir = os.path.join(model_path, "transformer")
    weights = _load_safetensors_from_dir(transformer_dir)
    if not weights:
        weights = _load_safetensors_from_dir(model_path)
    if not weights:
        raise FileNotFoundError(
            f"No .safetensors files found in {transformer_dir} or {model_path}"
        )
    loaded = pipeline.transformer.load_weights(weights)
    logger.info("Loaded %d transformer weight entries", len(loaded))

    t5_weights = _load_safetensors_from_dir(
        os.path.join(model_path, "text_encoder_2"),
    )
    if t5_weights:
        loaded_t5 = pipeline.text_encoder_2.load_weights(t5_weights)
        logger.info("Loaded %d T5 encoder weight entries", len(loaded_t5))


class DiffusionEngine:
    """Engine for running FLUX diffusion inference.

    Handles model loading, device placement, optional torch.compile,
    and deterministic generation via seeded generators.
    """

    def __init__(
        self,
        model_name: str,
        seed: int = 42,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        enforce_eager: bool = False,
    ):
        self.model_name = model_name
        self.seed = seed
        self.dtype = dtype
        self.device_str = device
        self.device = torch.device(device)
        self.enforce_eager = enforce_eager
        self._pipeline: FluxPipeline | None = None

    def _get_pipeline(self) -> FluxPipeline:
        if self._pipeline is not None:
            return self._pipeline

        logger.info("Loading FLUX model: %s", self.model_name)
        model_path = _download_flux_model(self.model_name)

        config = FluxConfig.from_pretrained(model_path)
        pipeline = FluxPipeline(config, model_path)

        _load_flux_weights(pipeline, model_path)

        pipeline.transformer.to(device=self.device, dtype=self.dtype)
        pipeline.text_encoder.to(device=self.device)
        pipeline.text_encoder_2.to(device=self.device, dtype=self.dtype)
        pipeline.vae.to(device=self.device)

        # FLUX: torch.compile disabled — iterative denoising (28 steps)
        # compounds non-deterministic FP rounding, causing ~8-12% cosine
        # divergence vs reference implementations.
        is_flux = "flux" in self.model_name.lower()
        if not self.enforce_eager and not is_flux:
            try:
                pipeline.transformer = torch.compile(
                    pipeline.transformer, mode="default",
                )
                logger.info("torch.compile applied to transformer")
            except Exception as e:
                logger.warning("torch.compile failed, using eager: %s", e)
        else:
            logger.info("Running transformer in eager mode")

        pipeline.eval()
        self._pipeline = pipeline
        return pipeline

    def generate(
        self,
        prompts: str | list[str],
        params: DiffusionSamplingParams | None = None,
    ) -> DiffusionOutput:
        """Generate images from text prompts.

        Parameters
        ----------
        prompts : str or list[str]
            Text prompt(s) for image generation.
        params : DiffusionSamplingParams, optional
            Sampling parameters (height, width, steps, guidance, etc.).

        Returns
        -------
        DiffusionOutput
            Contains generated images and/or latents.
        """
        pipeline = self._get_pipeline()
        params = params or DiffusionSamplingParams()

        seed = params.seed if params.seed is not None else self.seed
        generator = torch.Generator(device=self.device).manual_seed(seed)

        return pipeline.forward(prompts, params, generator=generator)

    def warmup(self, num_steps: int = 2) -> None:
        """Run a small warmup generation to prime CUDA graphs / compile."""
        params = DiffusionSamplingParams(
            height=256, width=256,
            num_inference_steps=num_steps,
            output_type="latent",
        )
        self.generate("warmup", params)

    def _cleanup(self) -> None:
        """Release GPU memory."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
        torch.cuda.empty_cache()
