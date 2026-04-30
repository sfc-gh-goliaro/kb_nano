"""Standardized workload definitions for eval (Tier 3).

These workloads are constants that ensure reproducible, comparable results
across runs and users. They are not configurable by design.

LLM workloads (text-only, real WildChat-derived requests):
  Throughput: 3 scenarios (prefill-heavy, balanced, decode-heavy), 1000 reqs each.
  Latency: 2 scenarios (single-request, fixed-batch-32).

VLM workloads (multi-modal):
  Throughput: 3 scenarios (text-only, image, video), 1000 reqs each.
  Latency: 2 scenarios (single-image, single-video), batch_size=1.
"""

from __future__ import annotations

from dataclasses import dataclass

from kb_nano.bench.utils.real_prompts import DEFAULT_WORKLOAD_DATASETS


@dataclass(frozen=True)
class ThroughputWorkload:
    name: str
    num_requests: int = 1000
    dataset_name: str = ""


@dataclass(frozen=True)
class LatencyWorkload:
    name: str
    batch_size: int
    input_len: int
    output_len: int
    num_warmup: int = 3
    num_iters: int = 5


THROUGHPUT_WORKLOADS: list[ThroughputWorkload] = [
    ThroughputWorkload(
        name="prefill-heavy",
        dataset_name=DEFAULT_WORKLOAD_DATASETS["prefill-heavy"],
    ),
    ThroughputWorkload(
        name="balanced",
        dataset_name=DEFAULT_WORKLOAD_DATASETS["balanced"],
    ),
    ThroughputWorkload(
        name="decode-heavy",
        dataset_name=DEFAULT_WORKLOAD_DATASETS["decode-heavy"],
    ),
]

LATENCY_WORKLOADS: list[LatencyWorkload] = [
    LatencyWorkload(name="single-request",  batch_size=1,  input_len=128, output_len=128),
    LatencyWorkload(name="fixed-batch-32",  batch_size=32, input_len=128, output_len=128),
]

ALL_WORKLOADS = {
    "throughput": THROUGHPUT_WORKLOADS,
    "latency": LATENCY_WORKLOADS,
}


# ---------------------------------------------------------------------------
# VLM workloads (multi-modal)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VLMThroughputWorkload:
    name: str
    modality: str  # "text", "image", "video"
    input_len: int | None  # fixed input token length (text only)
    output_len: int
    dataset_name: str | None = None  # HF dataset (image/video only)
    num_requests: int = 1000


VLM_THROUGHPUT_WORKLOADS: list[VLMThroughputWorkload] = [
    VLMThroughputWorkload(
        "text-only", "text", input_len=512, output_len=1024),
    VLMThroughputWorkload(
        "image", "image", input_len=None, output_len=512,
        dataset_name="lmarena-ai/VisionArena-Chat"),
    VLMThroughputWorkload(
        "video", "video", input_len=None, output_len=512,
        dataset_name="yale-nlp/MMVU"),
]


@dataclass(frozen=True)
class VLMLatencyWorkload:
    name: str
    modality: str  # "image", "video"
    output_len: int
    batch_size: int = 1
    dataset_name: str | None = None
    num_warmup: int = 3
    num_iters: int = 5


VLM_LATENCY_WORKLOADS: list[VLMLatencyWorkload] = [
    VLMLatencyWorkload(
        "single-image", "image", output_len=128,
        dataset_name="lmarena-ai/VisionArena-Chat"),
    VLMLatencyWorkload(
        "single-video", "video", output_len=128,
        dataset_name="yale-nlp/MMVU"),
]

ALL_VLM_WORKLOADS = {
    "throughput": VLM_THROUGHPUT_WORKLOADS,
    "latency": VLM_LATENCY_WORKLOADS,
}


# ---------------------------------------------------------------------------
# Diffusion workloads (image generation)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DiffusionModelConfig:
    num_inference_steps: int
    guidance_scale: float

FLUX_CONFIG = DiffusionModelConfig(num_inference_steps=28, guidance_scale=3.5)
SDXL_CONFIG = DiffusionModelConfig(num_inference_steps=50, guidance_scale=5.0)


@dataclass(frozen=True)
class DiffusionThroughputWorkload:
    name: str
    height: int
    width: int
    batch_size: int
    num_requests: int = 10

@dataclass(frozen=True)
class DiffusionLatencyWorkload:
    name: str
    height: int
    width: int
    batch_size: int = 1
    num_warmup: int = 2
    num_iters: int = 5

DIFFUSION_THROUGHPUT_WORKLOADS: list[DiffusionThroughputWorkload] = [
    DiffusionThroughputWorkload("1024x1024", height=1024, width=1024, batch_size=4, num_requests=10),
    DiffusionThroughputWorkload("512x512",   height=512,  width=512,  batch_size=8, num_requests=10),
]

DIFFUSION_LATENCY_WORKLOADS: list[DiffusionLatencyWorkload] = [
    DiffusionLatencyWorkload("single-1024x1024", height=1024, width=1024),
    DiffusionLatencyWorkload("single-512x512",   height=512,  width=512),
]

ALL_DIFFUSION_WORKLOADS = {
    "throughput": DIFFUSION_THROUGHPUT_WORKLOADS,
    "latency": DIFFUSION_LATENCY_WORKLOADS,
}


# ---------------------------------------------------------------------------
# Segmentation workloads (promptable concept segmentation)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SegmentationThroughputWorkload:
    """Workload for segmentation throughput measurement."""
    name: str
    resolution: int
    num_requests: int
    dataset_name: str
    dataset_subset: str = ""
    modality: str = "image"  # "image" or "video"

@dataclass(frozen=True)
class SegmentationLatencyWorkload:
    """Workload for segmentation latency measurement."""
    name: str
    resolution: int
    batch_size: int
    dataset_name: str
    dataset_subset: str = ""
    modality: str = "image"
    num_warmup: int = 3
    num_iters: int = 10

SEGMENTATION_THROUGHPUT_WORKLOADS: list[SegmentationThroughputWorkload] = [
    SegmentationThroughputWorkload(
        "gold-metaclip-nps", resolution=1008, num_requests=500,
        dataset_name="facebook/SACo-Gold", dataset_subset="metaclip_nps",
    ),
    SegmentationThroughputWorkload(
        "gold-wiki-common", resolution=1008, num_requests=500,
        dataset_name="facebook/SACo-Gold", dataset_subset="wiki_common",
    ),
    SegmentationThroughputWorkload(
        "gold-crowded", resolution=1008, num_requests=500,
        dataset_name="facebook/SACo-Gold", dataset_subset="crowded",
    ),
    SegmentationThroughputWorkload(
        "veval-sav-val", resolution=1008, num_requests=100,
        dataset_name="facebook/SACo-VEval", dataset_subset="sav_val",
        modality="video",
    ),
    SegmentationThroughputWorkload(
        "veval-yt1b-val", resolution=1008, num_requests=100,
        dataset_name="facebook/SACo-VEval", dataset_subset="yt1b_val",
        modality="video",
    ),
]

SEGMENTATION_LATENCY_WORKLOADS: list[SegmentationLatencyWorkload] = [
    SegmentationLatencyWorkload(
        "single-image-1008", resolution=1008, batch_size=1,
        dataset_name="facebook/SACo-Gold", dataset_subset="metaclip_nps",
    ),
    SegmentationLatencyWorkload(
        "batch-4-image-1008", resolution=1008, batch_size=4,
        dataset_name="facebook/SACo-Gold", dataset_subset="metaclip_nps",
    ),
    SegmentationLatencyWorkload(
        "single-video-frame-1008", resolution=1008, batch_size=1,
        dataset_name="facebook/SACo-VEval", dataset_subset="smartglasses_val",
        modality="video",
    ),
]

@dataclass(frozen=True)
class SegmentationVideoWorkload:
    """Workload for multi-frame video segmentation benchmark."""
    name: str
    resolution: int
    num_clips: int
    frames_per_clip: int
    dataset_name: str
    dataset_subset: str = ""
    text_prompt: str = "objects"

SEGMENTATION_VIDEO_WORKLOADS: list[SegmentationVideoWorkload] = [
    SegmentationVideoWorkload(
        "sav-val-video", resolution=1008, num_clips=10, frames_per_clip=16,
        dataset_name="facebook/SACo-VEval", dataset_subset="sav_val",
    ),
    SegmentationVideoWorkload(
        "smartglasses-val-video", resolution=1008, num_clips=10, frames_per_clip=16,
        dataset_name="facebook/SACo-VEval", dataset_subset="smartglasses_val",
    ),
]

ALL_SEGMENTATION_WORKLOADS = {
    "throughput": SEGMENTATION_THROUGHPUT_WORKLOADS,
    "latency": SEGMENTATION_LATENCY_WORKLOADS,
    "video": SEGMENTATION_VIDEO_WORKLOADS,
}


# ---------------------------------------------------------------------------
# TTS workloads (text-to-speech, e.g. CosyVoice3)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TTSModelConfig:
    sample_rate: int
    n_timesteps: int

COSYVOICE3_CONFIG = TTSModelConfig(sample_rate=24000, n_timesteps=10)


@dataclass(frozen=True)
class TTSThroughputWorkload:
    """TTS throughput workload definition.

    Uses the SEED-TTS-Eval dataset for realistic TTS benchmarking.
    Each request has a text prompt and a reference audio for voice cloning.
    """
    name: str
    num_requests: int = 100
    max_text_len: int = 200
    dataset_name: str = "zhaochenyang20/seed-tts-eval"
    dataset_split: str = "train"


@dataclass(frozen=True)
class TTSLatencyWorkload:
    name: str
    batch_size: int = 1
    max_text_len: int = 200
    dataset_name: str = "zhaochenyang20/seed-tts-eval"
    dataset_split: str = "train"
    num_warmup: int = 2
    num_iters: int = 5


TTS_THROUGHPUT_WORKLOADS: list[TTSThroughputWorkload] = [
    TTSThroughputWorkload("tts-short", num_requests=100, max_text_len=50),
    TTSThroughputWorkload("tts-medium", num_requests=100, max_text_len=200),
    TTSThroughputWorkload("tts-long", num_requests=50, max_text_len=500),
]

TTS_LATENCY_WORKLOADS: list[TTSLatencyWorkload] = [
    TTSLatencyWorkload("single-utterance", batch_size=1, max_text_len=100),
]

ALL_TTS_WORKLOADS = {
    "throughput": TTS_THROUGHPUT_WORKLOADS,
    "latency": TTS_LATENCY_WORKLOADS,
}


# ---------------------------------------------------------------------------
# Object detection workloads (COCO val2017)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DetectionThroughputWorkload:
    name: str
    image_size: int
    num_images: int
    batch_size: int
    dataset_name: str = "detection-datasets/coco"
    dataset_split: str = "val"

@dataclass(frozen=True)
class DetectionLatencyWorkload:
    name: str
    image_size: int
    batch_size: int
    dataset_name: str = "detection-datasets/coco"
    dataset_split: str = "val"
    num_warmup: int = 3
    num_iters: int = 20

DETECTION_THROUGHPUT_WORKLOADS: list[DetectionThroughputWorkload] = [
    DetectionThroughputWorkload("coco-val", image_size=640, num_images=5000, batch_size=32),
]

DETECTION_LATENCY_WORKLOADS: list[DetectionLatencyWorkload] = [
    DetectionLatencyWorkload("single-image", image_size=640, batch_size=1),
    DetectionLatencyWorkload("batch-4", image_size=640, batch_size=4),
]

ALL_DETECTION_WORKLOADS = {
    "throughput": DETECTION_THROUGHPUT_WORKLOADS,
    "latency": DETECTION_LATENCY_WORKLOADS,
}


def get_max_seq_len() -> int:
    """Return the maximum static sequence length for standardized LLM workloads.

    Throughput decode lengths are data-dependent for real-prompt workloads, so
    eval computes their max sequence length after loading the dataset.
    """
    max_len = 0
    for w in LATENCY_WORKLOADS:
        max_len = max(max_len, w.input_len + w.output_len)
    return max_len


# ---------------------------------------------------------------------------
# Video diffusion workloads (text-to-video generation)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VideoDiffusionModelConfig:
    num_inference_steps: int
    guidance_scale: float

HUNYUAN_VIDEO_CONFIG = VideoDiffusionModelConfig(
    num_inference_steps=30, guidance_scale=6.0,
)

@dataclass(frozen=True)
class VideoDiffusionThroughputWorkload:
    name: str
    height: int
    width: int
    num_frames: int
    num_prompts: int

@dataclass(frozen=True)
class VideoDiffusionLatencyWorkload:
    name: str
    height: int
    width: int
    num_frames: int
    num_warmup: int = 2
    num_iters: int = 5

VIDEO_DIFFUSION_THROUGHPUT_WORKLOADS: list[VideoDiffusionThroughputWorkload] = [
    VideoDiffusionThroughputWorkload(
        "480p-short", height=480, width=832, num_frames=25, num_prompts=16,
    ),
    VideoDiffusionThroughputWorkload(
        "480p-medium", height=480, width=832, num_frames=49, num_prompts=8,
    ),
]

VIDEO_DIFFUSION_LATENCY_WORKLOADS: list[VideoDiffusionLatencyWorkload] = [
    VideoDiffusionLatencyWorkload(
        "single-480p-short", height=480, width=832, num_frames=25,
    ),
    VideoDiffusionLatencyWorkload(
        "single-480p-medium", height=480, width=832, num_frames=49,
    ),
]

ALL_VIDEO_DIFFUSION_WORKLOADS = {
    "throughput": VIDEO_DIFFUSION_THROUGHPUT_WORKLOADS,
    "latency": VIDEO_DIFFUSION_LATENCY_WORKLOADS,
}


# ---------------------------------------------------------------------------
# Oasis workloads (autoregressive video diffusion)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OasisWorkload:
    name: str
    batch_clips: int
    num_frames: int
    ddim_steps: int
    n_prompt_frames: int = 1
    kind: str = "throughput"
    dataset_name: str = "TESS-Computer/minecraft-vla-stage1"
    dataset_split: str = "train"


OASIS_THROUGHPUT_WORKLOADS: list[OasisWorkload] = [
    OasisWorkload(
        name="short-bs4-16f-4ddim",
        batch_clips=4,
        num_frames=16,
        ddim_steps=4,
    ),
    OasisWorkload(
        name="medium-bs8-24f-4ddim",
        batch_clips=8,
        num_frames=24,
        ddim_steps=4,
    ),
    OasisWorkload(
        name="long-bs8-32f-4ddim",
        batch_clips=8,
        num_frames=32,
        ddim_steps=4,
    ),
    OasisWorkload(
        name="denoise-bs4-16f-8ddim",
        batch_clips=4,
        num_frames=16,
        ddim_steps=8,
    ),
]

OASIS_LATENCY_WORKLOADS: list[OasisWorkload] = [
    OasisWorkload(
        name="latency-bs1-8f-4ddim",
        batch_clips=1,
        num_frames=8,
        ddim_steps=4,
        kind="latency",
    ),
]

ALL_OASIS_WORKLOADS = {
    "throughput": OASIS_THROUGHPUT_WORKLOADS,
    "latency": OASIS_LATENCY_WORKLOADS,
}


# ---------------------------------------------------------------------------
# Vision encoder workloads (pure image feature extraction, e.g. SigLIP-2, DINOv3)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VisionEncoderThroughputWorkload:
    """Throughput workload for vision encoders.

    Processes num_images real images from dataset_name at the given resolution
    in fixed batch_size batches, measuring images/sec.
    """
    name: str
    resolution: int
    num_images: int
    batch_size: int
    dataset_name: str = "ILSVRC/imagenet-1k"
    dataset_split: str = "validation"

@dataclass(frozen=True)
class VisionEncoderLatencyWorkload:
    """Latency workload for vision encoders.

    Repeated inference on real images from dataset_name at the model's default
    resolution, measuring median and P99 latency.
    """
    name: str
    resolution: int
    batch_size: int
    dataset_name: str = "ILSVRC/imagenet-1k"
    dataset_split: str = "validation"
    num_warmup: int = 3
    num_iters: int = 10


VISION_ENCODER_THROUGHPUT_WORKLOADS: list[VisionEncoderThroughputWorkload] = [
    VisionEncoderThroughputWorkload("default-res", resolution=0, num_images=5000, batch_size=32),
    VisionEncoderThroughputWorkload("high-res",    resolution=512, num_images=2500, batch_size=16),
]

VISION_ENCODER_LATENCY_WORKLOADS: list[VisionEncoderLatencyWorkload] = [
    VisionEncoderLatencyWorkload("single-image", resolution=0, batch_size=1, num_warmup=5, num_iters=30),
    VisionEncoderLatencyWorkload("batch-8",      resolution=0, batch_size=8, num_warmup=5, num_iters=30),
]

ALL_VISION_ENCODER_WORKLOADS = {
    "throughput": VISION_ENCODER_THROUGHPUT_WORKLOADS,
    "latency": VISION_ENCODER_LATENCY_WORKLOADS,
}
