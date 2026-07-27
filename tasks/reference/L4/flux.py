"""FLUX.1-dev diffusion pipeline (L4 pipeline).

Contains:
- FluxTransformer2DModel: the DiT backbone (dual + single stream blocks).
- FluxPipeline: full text-to-image pipeline (encode, diffuse, decode).

L4 wiring/configuration; computation lives in L1-L3 tasks.
"""


from __future__ import annotations


# Inlined helper (no baseline task): pure-torch AutoencoderKL.
# The baseline imports this from diffusers, so there is no baseline file
# to inline this from.
import json
import os
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


# The baseline imports this from diffusers, so there is no baseline file
# to inline this from.
@dataclass
class AutoencoderKLConfig:
    """The subset of ``diffusers`` VAE config this reference reads."""

    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 16
    block_out_channels: tuple[int, ...] = (128, 256, 512, 512)
    layers_per_block: int = 2
    norm_num_groups: int = 32
    act_fn: str = "silu"
    scaling_factor: float = 0.3611
    shift_factor: float | None = 0.1159
    mid_block_add_attention: bool = True
    use_quant_conv: bool = False
    use_post_quant_conv: bool = False
    force_upcast: bool = True
    sample_size: int = 1024
    extra: dict = field(default_factory=dict)

    def get(self, key: str, default=None):
        return getattr(self, key, self.extra.get(key, default))

    @classmethod
    def from_dict(cls, data: dict) -> "AutoencoderKLConfig":
        known = {f for f in cls.__dataclass_fields__ if f != "extra"}
        kwargs = {k: v for k, v in data.items() if k in known}
        if "block_out_channels" in kwargs:
            kwargs["block_out_channels"] = tuple(kwargs["block_out_channels"])
        # A VAE without an explicit quant conv flag has one (pre-FLUX default).
        for flag in ("use_quant_conv", "use_post_quant_conv"):
            if flag not in data:
                kwargs[flag] = True
        return cls(extra={k: v for k, v in data.items() if k not in known}, **kwargs)


def _group_norm(channels: int, num_groups: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=num_groups, num_channels=channels, eps=1e-6, affine=True)


class ResnetBlock2D(nn.Module):
    """``diffusers.models.resnet.ResnetBlock2D`` with ``temb=None``."""

    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 32):
        super().__init__()
        self.norm1 = _group_norm(in_channels, num_groups)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        self.norm2 = _group_norm(out_channels, num_groups)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        self.nonlinearity = nn.SiLU()
        self.use_in_shortcut = in_channels != out_channels
        self.conv_shortcut = (
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=True)
            if self.use_in_shortcut else None
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.nonlinearity(self.norm1(hidden_states))
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.nonlinearity(self.norm2(hidden_states))
        hidden_states = self.conv2(hidden_states)
        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)
        return residual + hidden_states


class Attention(nn.Module):
    """Single-head spatial self-attention as used in the VAE mid block.

    Mirrors ``diffusers`` ``Attention`` + ``AttnProcessor2_0`` for the VAE
    settings: ``heads=1``, ``group_norm`` over channels, residual connection,
    and ``to_out`` as a ModuleList so the checkpoint key is ``to_out.0.*``.
    """

    def __init__(self, channels: int, num_groups: int = 32):
        super().__init__()
        self.channels = channels
        self.heads = 1
        self.group_norm = _group_norm(channels, num_groups)
        self.to_q = nn.Linear(channels, channels, bias=True)
        self.to_k = nn.Linear(channels, channels, bias=True)
        self.to_v = nn.Linear(channels, channels, bias=True)
        self.to_out = nn.ModuleList([nn.Linear(channels, channels, bias=True),
                                     nn.Dropout(0.0)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        batch, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch, channel, height * width).transpose(1, 2)
        hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        head_dim = key.shape[-1] // self.heads
        query = query.view(batch, -1, self.heads, head_dim).transpose(1, 2)
        key = key.view(batch, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch, -1, self.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch, -1, self.heads * head_dim).to(query.dtype)
        hidden_states = self.to_out[1](self.to_out[0](hidden_states))
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch, channel, height, width)
        return hidden_states + residual


class UNetMidBlock2D(nn.Module):
    def __init__(self, channels: int, num_groups: int = 32, add_attention: bool = True):
        super().__init__()
        self.resnets = nn.ModuleList([
            ResnetBlock2D(channels, channels, num_groups),
            ResnetBlock2D(channels, channels, num_groups),
        ])
        self.attentions = nn.ModuleList(
            [Attention(channels, num_groups) if add_attention else None])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.resnets[0](hidden_states)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                hidden_states = attn(hidden_states)
            hidden_states = resnet(hidden_states)
        return hidden_states


class Downsample2D(nn.Module):
    """``diffusers.Downsample2D``: asymmetric (0,1,0,1) pad then stride-2 conv."""

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=0)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = F.pad(hidden_states, (0, 1, 0, 1), mode="constant", value=0)
        return self.conv(hidden_states)


class Upsample2D(nn.Module):
    """``diffusers.Upsample2D``: nearest 2x interpolation then a 3x3 conv."""

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.conv = nn.Conv2d(channels, channels, 3, stride=1, padding=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()
        hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        return self.conv(hidden_states)


class DownEncoderBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_layers: int,
                 num_groups: int, add_downsample: bool):
        super().__init__()
        self.resnets = nn.ModuleList([
            ResnetBlock2D(in_channels if i == 0 else out_channels, out_channels, num_groups)
            for i in range(num_layers)
        ])
        self.downsamplers = nn.ModuleList([Downsample2D(out_channels)]) if add_downsample else None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)
        if self.downsamplers is not None:
            for down in self.downsamplers:
                hidden_states = down(hidden_states)
        return hidden_states


class UpDecoderBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_layers: int,
                 num_groups: int, add_upsample: bool):
        super().__init__()
        self.resnets = nn.ModuleList([
            ResnetBlock2D(in_channels if i == 0 else out_channels, out_channels, num_groups)
            for i in range(num_layers)
        ])
        self.upsamplers = nn.ModuleList([Upsample2D(out_channels)]) if add_upsample else None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)
        if self.upsamplers is not None:
            for up in self.upsamplers:
                hidden_states = up(hidden_states)
        return hidden_states


class Encoder(nn.Module):
    def __init__(self, config: AutoencoderKLConfig):
        super().__init__()
        boc = config.block_out_channels
        g = config.norm_num_groups
        self.conv_in = nn.Conv2d(config.in_channels, boc[0], 3, stride=1, padding=1)
        blocks = []
        out_ch = boc[0]
        for i, ch in enumerate(boc):
            in_ch, out_ch = out_ch, ch
            blocks.append(DownEncoderBlock2D(
                in_ch, out_ch, config.layers_per_block, g,
                add_downsample=i != len(boc) - 1))
        self.down_blocks = nn.ModuleList(blocks)
        self.mid_block = UNetMidBlock2D(boc[-1], g, config.mid_block_add_attention)
        self.conv_norm_out = _group_norm(boc[-1], g)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(boc[-1], 2 * config.latent_channels, 3, padding=1)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        sample = self.conv_in(sample)
        for block in self.down_blocks:
            sample = block(sample)
        sample = self.mid_block(sample)
        sample = self.conv_out(self.conv_act(self.conv_norm_out(sample)))
        return sample


class Decoder(nn.Module):
    def __init__(self, config: AutoencoderKLConfig):
        super().__init__()
        boc = config.block_out_channels
        g = config.norm_num_groups
        reversed_boc = list(reversed(boc))
        self.conv_in = nn.Conv2d(config.latent_channels, reversed_boc[0], 3, stride=1, padding=1)
        self.mid_block = UNetMidBlock2D(reversed_boc[0], g, config.mid_block_add_attention)
        blocks = []
        out_ch = reversed_boc[0]
        for i, ch in enumerate(reversed_boc):
            in_ch, out_ch = out_ch, ch
            blocks.append(UpDecoderBlock2D(
                in_ch, out_ch, config.layers_per_block + 1, g,
                add_upsample=i != len(boc) - 1))
        self.up_blocks = nn.ModuleList(blocks)
        self.conv_norm_out = _group_norm(boc[0], g)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(boc[0], config.out_channels, 3, padding=1)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        sample = self.conv_in(sample)
        sample = self.mid_block(sample)
        for block in self.up_blocks:
            sample = block(sample)
        sample = self.conv_out(self.conv_act(self.conv_norm_out(sample)))
        return sample


class DiagonalGaussianDistribution:
    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if deterministic:
            self.var = self.std = torch.zeros_like(self.mean)

    def sample(self, generator: torch.Generator | None = None) -> torch.Tensor:
        noise = torch.randn(self.mean.shape, generator=generator,
                            device=self.mean.device, dtype=self.mean.dtype)
        return self.mean + self.std * noise

    def mode(self) -> torch.Tensor:
        return self.mean


class DecoderOutput:
    def __init__(self, sample: torch.Tensor):
        self.sample = sample

    def __getitem__(self, idx):
        return (self.sample,)[idx]


class AutoencoderKLOutput:
    def __init__(self, latent_dist: DiagonalGaussianDistribution):
        self.latent_dist = latent_dist

    def __getitem__(self, idx):
        return (self.latent_dist,)[idx]


class AutoencoderKL(nn.Module):
    """Pure-torch ``diffusers.AutoencoderKL`` replacement (encode + decode)."""

    def __init__(self, config: AutoencoderKLConfig | None = None, **kwargs):
        super().__init__()
        if config is None:
            config = AutoencoderKLConfig.from_dict(kwargs) if kwargs else AutoencoderKLConfig()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.quant_conv = (
            nn.Conv2d(2 * config.latent_channels, 2 * config.latent_channels, 1)
            if config.use_quant_conv else None)
        self.post_quant_conv = (
            nn.Conv2d(config.latent_channels, config.latent_channels, 1)
            if config.use_post_quant_conv else None)

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def encode(self, x: torch.Tensor, return_dict: bool = True):
        h = self.encoder(x)
        if self.quant_conv is not None:
            h = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(h)
        return AutoencoderKLOutput(posterior) if return_dict else (posterior,)

    def decode(self, z: torch.Tensor, return_dict: bool = True):
        if self.post_quant_conv is not None:
            z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return DecoderOutput(dec) if return_dict else (dec,)

    def forward(self, sample: torch.Tensor, sample_posterior: bool = False,
                generator: torch.Generator | None = None, return_dict: bool = True):
        posterior = self.encode(sample).latent_dist
        z = posterior.sample(generator=generator) if sample_posterior else posterior.mode()
        return self.decode(z, return_dict=return_dict)

    @classmethod
    def from_pretrained(cls, model_name: str, subfolder: str | None = None,
                        local_files_only: bool = False, **kwargs) -> "AutoencoderKL":
        """Load ``config.json`` + ``*.safetensors`` from a local dir or the Hub.

        The safetensors/hub readers are imported lazily: they are checkpoint
        plumbing, so the reference stays importable with torch alone.
        """
        from glob import glob
        from safetensors.torch import load_file

        path = os.path.join(model_name, subfolder) if subfolder else model_name
        if not os.path.isdir(path):
            from huggingface_hub import snapshot_download
            repo = snapshot_download(model_name, local_files_only=local_files_only)
            path = os.path.join(repo, subfolder) if subfolder else repo

        with open(os.path.join(path, "config.json")) as fh:
            config = AutoencoderKLConfig.from_dict(json.load(fh))

        model = cls(config)
        state: dict[str, torch.Tensor] = {}
        for fpath in sorted(glob(os.path.join(path, "*.safetensors"))):
            state.update(load_file(fpath))
        model.load_state_dict(state, strict=True)
        return model


# Inlined helper (no baseline task): pure-torch flow-match Euler scheduler.
# The baseline imports this from diffusers, so there is no baseline file
# to inline this from.
import math

import numpy as np


# The baseline imports this from diffusers, so there is no baseline file
# to inline this from.
@dataclass
class FlowMatchEulerDiscreteSchedulerConfig:
    """Config with both attribute and ``.get()`` access, as the pipeline uses both."""

    num_train_timesteps: int = 1000
    shift: float = 1.0
    use_dynamic_shifting: bool = False
    base_shift: float | None = 0.5
    max_shift: float | None = 1.15
    base_image_seq_len: int = 256
    max_image_seq_len: int = 4096
    invert_sigmas: bool = False
    shift_terminal: float | None = None
    use_karras_sigmas: bool = False
    use_exponential_sigmas: bool = False
    use_beta_sigmas: bool = False
    time_shift_type: str = "exponential"
    stochastic_sampling: bool = False
    extra: dict = field(default_factory=dict)

    def get(self, key: str, default=None):
        if key in self.__dataclass_fields__ and key != "extra":
            return getattr(self, key)
        return self.extra.get(key, default)

    def __getitem__(self, key):
        return self.get(key)

    @classmethod
    def from_dict(cls, data: dict) -> "FlowMatchEulerDiscreteSchedulerConfig":
        known = {f for f in cls.__dataclass_fields__ if f != "extra"}
        return cls(extra={k: v for k, v in data.items() if k not in known},
                   **{k: v for k, v in data.items() if k in known})


@dataclass
class FlowMatchEulerDiscreteSchedulerOutput:
    prev_sample: torch.Tensor

    def __getitem__(self, idx):
        return (self.prev_sample,)[idx]


class FlowMatchEulerDiscreteScheduler:
    """Flow-matching Euler scheduler (``diffusers`` semantics, torch only)."""

    def __init__(self, config: FlowMatchEulerDiscreteSchedulerConfig | None = None, **kwargs):
        if config is None:
            config = (FlowMatchEulerDiscreteSchedulerConfig.from_dict(kwargs)
                      if kwargs else FlowMatchEulerDiscreteSchedulerConfig())
        self.config = config

        timesteps = np.linspace(
            1, config.num_train_timesteps, config.num_train_timesteps, dtype=np.float32,
        )[::-1].copy()
        timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32)
        sigmas = timesteps / config.num_train_timesteps
        if not config.use_dynamic_shifting:
            sigmas = config.shift * sigmas / (1 + (config.shift - 1) * sigmas)

        self.timesteps = sigmas * config.num_train_timesteps
        self.sigmas = sigmas.to("cpu")
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()
        self._shift = config.shift
        self._step_index: int | None = None
        self._begin_index: int | None = None
        self.num_inference_steps: int | None = None

    # -- properties ---------------------------------------------------------
    @property
    def shift(self) -> float:
        return self._shift

    @property
    def step_index(self) -> int | None:
        return self._step_index

    @property
    def begin_index(self) -> int | None:
        return self._begin_index

    def set_shift(self, shift: float) -> None:
        self._shift = shift

    def set_begin_index(self, begin_index: int = 0) -> None:
        self._begin_index = begin_index

    # -- schedule -----------------------------------------------------------
    def time_shift(self, mu: float, sigma: float, t):
        if self.config.time_shift_type == "exponential":
            return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)
        if self.config.time_shift_type == "linear":
            return mu / (mu + (1 / t - 1) ** sigma)
        raise ValueError(f"Unknown time_shift_type {self.config.time_shift_type!r}")

    def stretch_shift_to_terminal(self, t: np.ndarray) -> np.ndarray:
        one_minus_z = 1 - t
        scale_factor = one_minus_z[-1] / (1 - self.config.shift_terminal)
        return 1 - (one_minus_z / scale_factor)

    def _sigma_to_t(self, sigma: float) -> float:
        return sigma * self.config.num_train_timesteps

    def set_timesteps(
        self,
        num_inference_steps: int | None = None,
        device: str | torch.device | None = None,
        sigmas: list[float] | np.ndarray | None = None,
        mu: float | None = None,
        timesteps: list[float] | np.ndarray | None = None,
    ) -> None:
        cfg = self.config
        for unsupported in ("use_karras_sigmas", "use_exponential_sigmas", "use_beta_sigmas"):
            if getattr(cfg, unsupported):
                raise NotImplementedError(
                    f"{unsupported} is not implemented by this reference scheduler")
        if cfg.invert_sigmas:
            raise NotImplementedError(
                "invert_sigmas is not implemented by this reference scheduler")
        if cfg.use_dynamic_shifting and mu is None:
            raise ValueError("`mu` must be passed when `use_dynamic_shifting` is True")
        if sigmas is not None and timesteps is not None and len(sigmas) != len(timesteps):
            raise ValueError("`sigmas` and `timesteps` should have the same length")

        if num_inference_steps is None:
            num_inference_steps = len(sigmas) if sigmas is not None else len(timesteps)
        self.num_inference_steps = num_inference_steps

        is_timesteps_provided = timesteps is not None
        if is_timesteps_provided:
            timesteps = np.array(timesteps).astype(np.float32)

        if sigmas is None:
            if timesteps is None:
                timesteps = np.linspace(
                    self._sigma_to_t(self.sigma_max),
                    self._sigma_to_t(self.sigma_min),
                    num_inference_steps,
                )
            sigmas = timesteps / cfg.num_train_timesteps
        else:
            sigmas = np.array(sigmas).astype(np.float32)
            num_inference_steps = len(sigmas)

        if cfg.use_dynamic_shifting:
            sigmas = self.time_shift(mu, 1.0, sigmas)
        else:
            sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)

        if cfg.shift_terminal:
            sigmas = self.stretch_shift_to_terminal(sigmas)

        sigmas = torch.from_numpy(np.asarray(sigmas)).to(dtype=torch.float32, device=device)
        if not is_timesteps_provided:
            timesteps = sigmas * cfg.num_train_timesteps
        else:
            timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32, device=device)

        self.timesteps = timesteps
        self.sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])
        self._step_index = None
        self._begin_index = None

    # -- stepping -----------------------------------------------------------
    def index_for_timestep(self, timestep, schedule_timesteps: torch.Tensor | None = None) -> int:
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps
        indices = (schedule_timesteps == timestep).nonzero()
        pos = 1 if len(indices) > 1 else 0
        return indices[pos].item()

    def _init_step_index(self, timestep) -> None:
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def scale_noise(self, sample: torch.Tensor, timestep, noise: torch.Tensor) -> torch.Tensor:
        sigma = (timestep / self.config.num_train_timesteps).to(sample.device)
        while sigma.ndim < sample.ndim:
            sigma = sigma.unsqueeze(-1)
        return sigma * noise + (1.0 - sigma) * sample

    def step(
        self,
        model_output: torch.Tensor,
        timestep: float | torch.Tensor,
        sample: torch.Tensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: torch.Generator | None = None,
        per_token_timesteps: torch.Tensor | None = None,
        return_dict: bool = True,
    ):
        if isinstance(timestep, int) or (
            isinstance(timestep, torch.Tensor)
            and timestep.dtype in (torch.int16, torch.int32, torch.int64)
        ):
            raise ValueError(
                "Passing integer indices as timesteps to step() is not supported; "
                "pass one of scheduler.timesteps.")

        if self.step_index is None:
            self._init_step_index(timestep)

        # Upcast to avoid precision issues when computing prev_sample.
        sample = sample.to(torch.float32)

        if per_token_timesteps is not None:
            per_token_sigmas = per_token_timesteps / self.config.num_train_timesteps
            sigmas = self.sigmas[:, None, None]
            lower_mask = sigmas < per_token_sigmas[None] - 1e-6
            lower_sigmas, _ = (lower_mask * sigmas).max(dim=0)
            current_sigma = per_token_sigmas[..., None]
            next_sigma = lower_sigmas[..., None]
            dt = current_sigma - next_sigma
        else:
            current_sigma = self.sigmas[self.step_index]
            next_sigma = self.sigmas[self.step_index + 1]
            dt = next_sigma - current_sigma

        if self.config.stochastic_sampling:
            x0 = sample - current_sigma * model_output
            noise = torch.randn(sample.shape, generator=generator,
                                device=sample.device, dtype=sample.dtype)
            prev_sample = (1.0 - next_sigma) * x0 + next_sigma * noise
        else:
            prev_sample = sample + dt * model_output

        self._step_index += 1
        if per_token_timesteps is None:
            prev_sample = prev_sample.to(model_output.dtype)

        if not return_dict:
            return (prev_sample,)
        return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)

    def __len__(self) -> int:
        return self.config.num_train_timesteps

    @classmethod
    def from_pretrained(cls, model_name: str, subfolder: str | None = None,
                        local_files_only: bool = False, **kwargs):
        """Read ``scheduler_config.json`` from a local dir or the Hub."""
        path = os.path.join(model_name, subfolder) if subfolder else model_name
        if not os.path.isdir(path):
            from huggingface_hub import snapshot_download
            repo = snapshot_download(model_name, local_files_only=local_files_only)
            path = os.path.join(repo, subfolder) if subfolder else repo
        with open(os.path.join(path, "scheduler_config.json")) as fh:
            data = json.load(fh)
        data.pop("_class_name", None)
        data.pop("_diffusers_version", None)
        return cls(FlowMatchEulerDiscreteSchedulerConfig.from_dict(data))


# Inlined from tasks/reference/L1/layer_norm.py
class LayerNorm(nn.Module):
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        create_scale: bool = True,
        create_offset: bool = True,
    ):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine and create_scale:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
        else:
            self.register_parameter("weight", None)
        if elementwise_affine and create_offset:
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        weight = self.weight.float() if self.weight is not None else None
        bias = self.bias.float() if self.bias is not None else None
        return F.layer_norm(
            x.float(), self.normalized_shape, weight, bias, self.eps,
        ).to(orig_dtype)


# Inlined from tasks/reference/L1/linear.py
class Matmul(nn.Module):
    """Pure functional linear: takes input, weight, and optional bias as forward args."""

    def forward(self, input, weight, bias=None):
        return F.linear(input, weight, bias)


class BMM(nn.Module):
    """Batch matrix multiply: torch.matmul(a, b)."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.matmul(a, b)


class Linear(nn.Module):
    """Parametric linear: stores weight and bias internally."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.matmul = Matmul()

    def forward(self, input):
        return self.matmul(input, self.weight, self.bias)


# Inlined from tasks/reference/L1/silu.py
class SiLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x)


# Inlined from tasks/reference/L2/ada_layer_norm_continuous.py
class AdaLayerNormContinuous(nn.Module):
    r"""
    Adaptive normalization layer with a norm layer (layer_norm or rms_norm).

    Args:
        embedding_dim (`int`): Embedding dimension to use during projection.
        conditioning_embedding_dim (`int`): Dimension of the input condition.
        elementwise_affine (`bool`, defaults to `True`):
            Boolean flag to denote if affine transformation should be applied.
        eps (`float`, defaults to 1e-5): Epsilon factor.
        bias (`bool`, defaults to `True`): Whether to use bias in the linear layer.
        norm_type (`str`, defaults to `"layer_norm"`):
            Normalization layer to use. Values supported: "layer_norm", "rms_norm".
    """

    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        elementwise_affine=True,
        eps=1e-5,
        bias=True,
        norm_type="layer_norm",
    ):
        super().__init__()
        self.silu = SiLU()
        self.linear = Linear(conditioning_embedding_dim, embedding_dim * 2, bias=bias)
        if norm_type == "layer_norm":
            self.norm = LayerNorm(embedding_dim, eps=eps, elementwise_affine=elementwise_affine)
        else:
            raise ValueError(f"unknown norm_type {norm_type}")

    def forward(self, x: torch.Tensor, conditioning_embedding: torch.Tensor) -> torch.Tensor:
        emb = self.linear(self.silu(conditioning_embedding).to(x.dtype))
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x


# Inlined from tasks/reference/L2/timestep_embedding.py
def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> torch.Tensor:
    """Sinusoidal timestep embedding (DDPM-style)."""
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device,
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = timesteps[:, None].float() * torch.exp(exponent)[None, :]
    emb = scale * emb
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class Timesteps(nn.Module):
    """Wraps get_timestep_embedding as an nn.Module."""

    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float, scale: int = 1):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return get_timestep_embedding(
            timesteps, self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )


class TimestepEmbedding(nn.Module):
    """Two-layer MLP that projects sinusoidal timestep encodings."""

    def __init__(self, in_channels: int, time_embed_dim: int, act_fn: str = "silu"):
        super().__init__()
        self.linear_1 = Linear(in_channels, time_embed_dim, bias=True)
        self.act = SiLU()
        self.linear_2 = Linear(time_embed_dim, time_embed_dim, bias=True)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


class CombinedTimestepTextProjEmbeddings(nn.Module):
    """Combines sinusoidal timestep encoding with pooled text projection.

    Produces ``timestep_embedder`` + ``text_embedder`` weight names matching
    the diffusers checkpoint layout.
    """

    def __init__(self, embedding_dim: int, pooled_projection_dim: int):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.text_embedder = TimestepEmbedding(in_channels=pooled_projection_dim, time_embed_dim=embedding_dim)

    def forward(self, timestep: torch.Tensor, pooled_projection: torch.Tensor) -> torch.Tensor:
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=pooled_projection.dtype))
        pooled_projections = self.text_embedder(pooled_projection)
        return timesteps_emb + pooled_projections


class CombinedTimestepGuidanceTextProjEmbeddings(nn.Module):
    """Combines sinusoidal timestep + guidance encoding with pooled text projection.

    Adds a ``guidance_embedder`` on top of
    :class:`CombinedTimestepTextProjEmbeddings`.  Weight names match the
    diffusers checkpoint layout.
    """

    def __init__(self, embedding_dim: int, pooled_projection_dim: int):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.guidance_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.text_embedder = TimestepEmbedding(in_channels=pooled_projection_dim, time_embed_dim=embedding_dim)

    def forward(self, timestep: torch.Tensor, guidance: torch.Tensor, pooled_projection: torch.Tensor) -> torch.Tensor:
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=pooled_projection.dtype))
        guidance_proj = self.time_proj(guidance)
        guidance_emb = self.guidance_embedder(guidance_proj.to(dtype=pooled_projection.dtype))
        pooled_projections = self.text_embedder(pooled_projection)
        return timesteps_emb + guidance_emb + pooled_projections


# Inlined from tasks/reference/L1/video_processor.py
class VideoProcessor(nn.Module):
    """Post-processor for VAE-decoded image and video tensors."""

    def __init__(self, vae_scale_factor: int = 8, do_normalize: bool = True):
        super().__init__()
        self.vae_scale_factor = vae_scale_factor
        self.do_normalize = do_normalize

    def postprocess(
        self,
        image: torch.Tensor,
        output_type: str = "pil",
    ) -> list | np.ndarray | torch.Tensor:
        """Post-process a 4D ``(B, C, H, W)`` image tensor."""
        if output_type == "latent" or output_type == "pt":
            return image

        if self.do_normalize:
            image = (image * 0.5 + 0.5).clamp(0, 1)

        image = image.cpu().permute(0, 2, 3, 1).float().numpy()

        if output_type == "np":
            return image

        images = (image * 255).round().astype("uint8")
        # Container conversion only — no arithmetic happens past this point.
        from PIL import Image
        return [Image.fromarray(img) for img in images]

    def postprocess_video(
        self,
        video: torch.Tensor,
        output_type: str = "np",
    ) -> np.ndarray | torch.Tensor | list:
        """Post-process a 5D ``(B, C, T, H, W)`` video tensor."""
        batch_size = video.shape[0]
        outputs = []
        for batch_idx in range(batch_size):
            batch_vid = video[batch_idx].permute(1, 0, 2, 3)
            batch_output = self.postprocess(batch_vid, output_type)
            outputs.append(batch_output)

        if output_type == "np":
            return np.stack(outputs)
        elif output_type == "pt":
            return torch.stack(outputs)
        elif output_type == "pil":
            return outputs
        else:
            raise ValueError(f"Unsupported output_type: {output_type}")


# Inlined helper (no baseline task): HuggingFace config dataclasses.
# The baselines get these from transformers, so there is no baseline file
# to inline this from.
# The baselines get these from transformers, so there is no baseline file
# to inline this from.


def _read_config_json(model_name: str, subfolder: str | None = None,
                      local_files_only: bool = False) -> dict:
    """Load ``config.json`` from a local directory or the Hub."""
    path = os.path.join(model_name, subfolder) if subfolder else model_name
    if not os.path.isdir(path):
        from huggingface_hub import snapshot_download
        repo = snapshot_download(model_name, local_files_only=local_files_only)
        path = os.path.join(repo, subfolder) if subfolder else repo
    with open(os.path.join(path, "config.json")) as fh:
        return json.load(fh)


@dataclass
class CLIPTextConfig:
    """Fields of ``transformers.CLIPTextConfig`` read by the CLIP references."""

    vocab_size: int = 49408
    hidden_size: int = 512
    intermediate_size: int = 2048
    num_hidden_layers: int = 12
    num_attention_heads: int = 8
    max_position_embeddings: int = 77
    layer_norm_eps: float = 1e-5
    hidden_act: str = "quick_gelu"
    eos_token_id: int = 49407
    extra: dict = field(default_factory=dict)

    def get(self, key: str, default=None):
        if key in self.__dataclass_fields__ and key != "extra":
            return getattr(self, key)
        return self.extra.get(key, default)

    @classmethod
    def from_dict(cls, data: dict) -> "CLIPTextConfig":
        # A full CLIP model config nests the text tower under "text_config".
        if "text_config" in data and isinstance(data["text_config"], dict):
            data = data["text_config"]
        known = {f for f in cls.__dataclass_fields__ if f != "extra"}
        return cls(extra={k: v for k, v in data.items() if k not in known},
                   **{k: v for k, v in data.items() if k in known})

    @classmethod
    def from_pretrained(cls, model_name: str, subfolder: str | None = None,
                        local_files_only: bool = False, **kwargs) -> "CLIPTextConfig":
        return cls.from_dict(_read_config_json(model_name, subfolder, local_files_only))


@dataclass
class T5Config:
    """Fields of ``transformers.T5Config`` read by the T5 references.

    ``dense_act_fn`` / ``is_gated_act`` are derived from ``feed_forward_proj``
    exactly as transformers does, including the ``gated-gelu -> gelu_new``
    backwards-compatibility remap.
    """

    vocab_size: int = 32128
    d_model: int = 512
    d_kv: int = 64
    d_ff: int = 2048
    num_layers: int = 6
    num_heads: int = 8
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    layer_norm_epsilon: float = 1e-6
    feed_forward_proj: str = "relu"
    dense_act_fn: str = "relu"
    is_gated_act: bool = False
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        act_info = self.feed_forward_proj.split("-")
        self.dense_act_fn = act_info[-1]
        self.is_gated_act = act_info[0] == "gated"
        if len(act_info) > 1 and act_info[0] != "gated" or len(act_info) > 2:
            raise ValueError(
                f"`feed_forward_proj`: {self.feed_forward_proj} is not a valid activation "
                "function of the dense layer. Expected `gated-{ACT_FN}` or `{ACT_FN}`.")
        if self.feed_forward_proj == "gated-gelu":
            self.dense_act_fn = "gelu_new"

    def get(self, key: str, default=None):
        if key in self.__dataclass_fields__ and key != "extra":
            return getattr(self, key)
        return self.extra.get(key, default)

    @classmethod
    def from_dict(cls, data: dict) -> "T5Config":
        known = {f for f in cls.__dataclass_fields__
                 if f not in ("extra", "dense_act_fn", "is_gated_act")}
        return cls(extra={k: v for k, v in data.items() if k not in known},
                   **{k: v for k, v in data.items() if k in known})

    @classmethod
    def from_pretrained(cls, model_name: str, subfolder: str | None = None,
                        local_files_only: bool = False, **kwargs) -> "T5Config":
        return cls.from_dict(_read_config_json(model_name, subfolder, local_files_only))


# Inlined from tasks/reference/L1/embedding.py
class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int,
                 padding_idx: int | None = None):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim,
                                padding_idx=padding_idx)

    def forward(self, input_ids):
        return self.emb(input_ids)


# Inlined from tasks/reference/L1/quickgelu.py
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


# Inlined from tasks/reference/L2/clip_mlp.py
class CLIPMLP(nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.fc1 = Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.fc2 = Linear(config.intermediate_size, config.hidden_size, bias=True)
        self.activation_fn = QuickGELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class CLIPTextEmbeddings(nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.token_embedding = Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = Embedding(config.max_position_embeddings, config.hidden_size)
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=False,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_length = input_ids.shape[-1]
        position_ids = self.position_ids[:, :seq_length]
        return self.token_embedding(input_ids) + self.position_embedding(position_ids)


# Inlined from tasks/reference/L1/softmax.py
class Softmax(nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(x, dim=self.dim)


# Inlined from tasks/reference/L2/clip_attention.py
class CLIPAttention(nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = Linear(self.embed_dim, self.embed_dim, bias=True)
        self.k_proj = Linear(self.embed_dim, self.embed_dim, bias=True)
        self.v_proj = Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = Linear(self.embed_dim, self.embed_dim, bias=True)

        self.bmm = BMM()
        self.softmax = Softmax(dim=-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_length, _ = hidden_states.shape

        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        queries = queries.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = self.bmm(queries, keys.transpose(-1, -2)) * self.scale
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = self.softmax(attn_weights.float()).to(queries.dtype)

        attn_output = self.bmm(attn_weights, values)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_length, self.embed_dim)
        return self.out_proj(attn_output)


# Inlined from tasks/reference/L3/clip_encoder_layer.py
class CLIPEncoderLayer(nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


# Inlined from tasks/reference/L4/clip_text_model.py
from dataclasses import dataclass


@dataclass
class CLIPTextModelOutput:
    last_hidden_state: torch.Tensor
    pooler_output: torch.Tensor


class CLIPEncoder(nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            CLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)
        ])

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)
        return hidden_states


class CLIPTextModel(nn.Module):
    """CLIP text encoder with pooled output.

    Weight names match ``transformers.CLIPTextModel`` under the ``text_model.``
    prefix so checkpoints are directly loadable.
    """

    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.config = config
        self.text_model = _CLIPTextTransformer(config)

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(
        self,
        input_ids: torch.Tensor,
        **kwargs,
    ) -> CLIPTextModelOutput:
        return self.text_model(input_ids)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        subfolder: str | None = None,
        local_files_only: bool = False,
        **kwargs,
    ) -> "CLIPTextModel":
        import os
        from glob import glob
        from safetensors.torch import load_file

        local_path = model_name
        if subfolder:
            local_path = os.path.join(model_name, subfolder)

        if os.path.isdir(local_path):
            config = CLIPTextConfig.from_pretrained(
                local_path, local_files_only=True,
            )
            weight_dir = local_path
        else:
            config = CLIPTextConfig.from_pretrained(
                model_name, subfolder=subfolder,
                local_files_only=local_files_only,
            )
            from huggingface_hub import snapshot_download
            repo_dir = snapshot_download(
                model_name, local_files_only=local_files_only,
            )
            weight_dir = os.path.join(repo_dir, subfolder) if subfolder else repo_dir

        model = cls(config)

        safetensor_files = sorted(glob(os.path.join(weight_dir, "*.safetensors")))
        state_dict: dict[str, torch.Tensor] = {}
        for fpath in safetensor_files:
            state_dict.update(load_file(fpath))

        remapped: dict[str, torch.Tensor] = {}
        for name, tensor in state_dict.items():
            new_name = name
            if "token_embedding.weight" in name:
                new_name = name.replace("token_embedding.weight", "token_embedding.emb.weight")
            elif "position_embedding.weight" in name:
                new_name = name.replace("position_embedding.weight", "position_embedding.emb.weight")
            remapped[new_name] = tensor

        model.load_state_dict(remapped, strict=False)
        return model


class _CLIPTextTransformer(nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.config = config
        self.embeddings = CLIPTextEmbeddings(config)
        self.encoder = CLIPEncoder(config)
        self.final_layer_norm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.eos_token_id = config.eos_token_id

    def forward(self, input_ids: torch.Tensor) -> CLIPTextModelOutput:
        hidden_states = self.embeddings(input_ids)

        causal_mask = self._make_causal_mask(input_ids.shape, hidden_states.dtype, hidden_states.device)

        hidden_states = self.encoder(hidden_states, attention_mask=causal_mask)
        last_hidden_state = self.final_layer_norm(hidden_states)

        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
        ]

        return CLIPTextModelOutput(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
        )

    @staticmethod
    def _make_causal_mask(
        input_shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        batch_size, seq_length = input_shape
        mask = torch.full((seq_length, seq_length), torch.finfo(dtype).min, dtype=dtype, device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)


# Inlined from tasks/reference/L1/t5_layer_norm.py
class T5LayerNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


# Inlined from infra/tp.py
import torch.distributed as dist


def _tp_size():
    return dist.get_world_size() if dist.is_initialized() else 1

def _tp_rank():
    return dist.get_rank() if dist.is_initialized() else 0


# Inlined from tasks/reference/L1/fp8_linear.py
_GROUP_SIZE = 128


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _expand_weight_scale(weight_fp8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    rows, cols = weight_fp8.shape[-2], weight_fp8.shape[-1]
    row_blocks = _ceil_div(rows, _GROUP_SIZE)
    col_blocks = _ceil_div(cols, _GROUP_SIZE)
    scale_f = scale.float()
    if scale_f.shape[-2:] == (row_blocks, col_blocks):
        expanded = scale_f.repeat_interleave(_GROUP_SIZE, dim=-2)
        expanded = expanded.repeat_interleave(_GROUP_SIZE, dim=-1)
        return expanded[..., :rows, :cols]
    if scale_f.shape[-1] == col_blocks:
        expanded = scale_f.repeat_interleave(_GROUP_SIZE, dim=-1)
        return expanded[..., :cols].unsqueeze(-2).expand_as(weight_fp8.float())
    return scale_f.expand_as(weight_fp8.float())


def _quantize_fp8_per_token_group(
    source: torch.Tensor,
    out_fp8: torch.Tensor,
    out_scale: torch.Tensor,
    *,
    use_ue8m0: bool = True,
    eps: float = 1e-10,
) -> None:
    info = torch.finfo(torch.float8_e4m3fn)
    flat = source.reshape(-1, source.shape[-1]).float()
    groups = _ceil_div(flat.shape[-1], _GROUP_SIZE)
    padded_cols = groups * _GROUP_SIZE
    if padded_cols != flat.shape[-1]:
        padded = flat.new_zeros(flat.shape[0], padded_cols)
        padded[:, :flat.shape[-1]] = flat
    else:
        padded = flat
    grouped = padded.view(flat.shape[0], groups, _GROUP_SIZE)
    scale = grouped.abs().amax(dim=-1).clamp_min(eps) / info.max
    if use_ue8m0:
        scale = torch.pow(2.0, torch.ceil(torch.log2(scale)))
    expanded = scale.repeat_interleave(_GROUP_SIZE, dim=-1)[:, :flat.shape[-1]]
    out_fp8.copy_(torch.clamp(flat / expanded, info.min, info.max).to(out_fp8.dtype).view_as(out_fp8))
    out_scale.copy_(scale.view_as(out_scale))


class _Fp8PrefillBufs:
    def __init__(self):
        self.input_fp8 = None
        self.input_scale = None
        self.output = None


class PerTokenGroupQuantFp8(nn.Module):
    def forward(self, x: torch.Tensor, out_fp8: torch.Tensor,
                out_scale: torch.Tensor) -> None:
        _quantize_fp8_per_token_group(x, out_fp8, out_scale)


class Fp8Linear(nn.Module):
    BLOCK_SIZE = _GROUP_SIZE
    _FLASHINFER_M_THRESHOLD = 32

    def __init__(self):
        super().__init__()
        self._a_buf = None
        self._s_buf = None
        self._o_buf = None
        self._pf = None

    def _ensure_buffers(self, max_tokens: int, K: int, N: int, device: torch.device):
        self._a_buf = torch.empty(max_tokens, K, dtype=torch.float8_e4m3fn, device=device)
        self._s_buf = torch.empty(max_tokens, math.ceil(K / _GROUP_SIZE), dtype=torch.float32, device=device)
        self._o_buf = torch.empty(max_tokens, N, dtype=torch.bfloat16, device=device)

    def forward(self, input_bf16: torch.Tensor,
                weight_fp8: torch.Tensor,
                weight_scale_inv: torch.Tensor,
                bias: torch.Tensor | None = None) -> torch.Tensor:
        n, k = weight_fp8.shape
        input_2d = input_bf16.reshape(-1, k)
        weight = weight_fp8.float() * _expand_weight_scale(weight_fp8, weight_scale_inv)
        output = F.linear(input_2d.float(), weight.float(), bias.float() if bias is not None else None)
        return output.to(input_bf16.dtype).view(*input_bf16.shape[:-1], n)


def postprocess_fp8_weights(
    weight_fp8: torch.Tensor,
    scale_inv: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return weight_fp8, scale_inv


# Inlined from tasks/reference/L1/allreduce.py
from contextlib import nullcontext
from typing import Optional

from torch.distributed import ProcessGroup


_CUSTOM_AR: Optional["CustomAllreduce"] = None


def set_custom_ar(ar):
    global _CUSTOM_AR
    _CUSTOM_AR = ar


def get_custom_ar():
    return _CUSTOM_AR


class AllReduce(nn.Module):
    def forward(self, tensor):
        dist.all_reduce(tensor)
        return tensor


class CustomAllreduce:
    """Compatibility shim for callers expecting the baseline custom AR API."""

    disabled = True

    def __init__(
        self,
        group: ProcessGroup,
        device: int | str | torch.device,
        max_size: int = 8192 * 1024,
    ) -> None:
        del group, device, max_size

    def capture(self):
        return nullcontext()

    def custom_all_reduce(self, input: torch.Tensor) -> None:
        del input
        return None

    def close(self) -> None:
        pass

__all__ = ["AllReduce", "CustomAllreduce", "get_custom_ar", "set_custom_ar"]


# Inlined from tasks/reference/L2/parallel_linear.py
def _get_fp8_linear_cls():
    return Fp8Linear

_FP8_BLOCK = 128


def _scale_shape(out_dim: int, in_dim: int) -> tuple[int, int]:
    return (math.ceil(out_dim / _FP8_BLOCK), math.ceil(in_dim / _FP8_BLOCK))


class ColumnParallelLinear(nn.Module):
    """Splits output dim across TP ranks."""

    def __init__(self, input_size: int, output_size: int, bias: bool = False,
                 quant_config: dict | None = None):
        super().__init__()
        tp = _tp_size()
        assert output_size % tp == 0
        self.output_size_per_partition = output_size // tp
        self.use_fp8 = quant_config is not None

        if self.use_fp8:
            self.weight = nn.Parameter(
                torch.empty(self.output_size_per_partition, input_size,
                            dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            self.weight_scale_inv = nn.Parameter(
                torch.empty(*_scale_shape(self.output_size_per_partition, input_size),
                            dtype=torch.float32),
                requires_grad=False,
            )
            self.weight.weight_loader = self._weight_loader
            self.weight_scale_inv.weight_loader = self._scale_loader
            self.linear_op = _get_fp8_linear_cls()()
        else:
            self.weight = nn.Parameter(torch.empty(self.output_size_per_partition, input_size))
            self.weight.weight_loader = self._weight_loader

        self.bias = nn.Parameter(torch.empty(self.output_size_per_partition)) if bias else None
        if self.bias is not None:
            self.bias.weight_loader = self._weight_loader

    def _weight_loader(self, param, loaded_weight):
        tp, rank = _tp_size(), _tp_rank()
        shard = param.data.size(0)
        loaded_weight = loaded_weight.narrow(0, rank * shard, shard)
        param.data.copy_(loaded_weight)

    def _scale_loader(self, param, loaded_weight):
        tp, rank = _tp_size(), _tp_rank()
        rows_per_shard = param.data.size(0)
        loaded_weight = loaded_weight.narrow(0, rank * rows_per_shard, rows_per_shard)
        param.data.copy_(loaded_weight)

    def forward(self, x):
        if self.use_fp8:
            return self.linear_op(x, self.weight, self.weight_scale_inv, self.bias)
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(nn.Module):
    """gate_proj + up_proj merged into one linear, sharded across TP."""

    def __init__(self, input_size: int, output_sizes: list[int], bias: bool = False,
                 quant_config: dict | None = None, disable_tp: bool = False):
        super().__init__()
        tp = _tp_size()
        self.disable_tp = disable_tp
        self.output_sizes = output_sizes
        total = sum(output_sizes)
        if not disable_tp:
            assert all(s % tp == 0 for s in output_sizes)
        self.use_fp8 = quant_config is not None

        effective_tp = 1 if disable_tp else tp
        if self.use_fp8:
            self.weight = nn.Parameter(
                torch.empty(total // effective_tp, input_size, dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            self.weight_scale_inv = nn.Parameter(
                torch.empty(*_scale_shape(total // effective_tp, input_size), dtype=torch.float32),
                requires_grad=False,
            )
            self.weight.weight_loader = self._weight_loader
            self.weight_scale_inv.weight_loader = self._scale_loader
            self.linear_op = _get_fp8_linear_cls()()
        else:
            self.weight = nn.Parameter(torch.empty(total // effective_tp, input_size))
            self.weight.weight_loader = self._weight_loader

        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.empty(total // tp))
            self.bias.weight_loader = self._weight_loader

    def _weight_loader(self, param, loaded_weight, shard_id: int | None = None):
        tp, rank = _tp_size(), _tp_rank()
        if shard_id is None:
            # Fused weight: ``loaded_weight`` is the full ``[sum(output_sizes), in]``
            # tensor.  Recurse per-shard so each output block is sharded across
            # TP ranks independently (mirrors vLLM's ``MergedColumnParallelLinear``
            # weight loader when called without an explicit shard id).
            offset = 0
            for sid, sz in enumerate(self.output_sizes):
                self._weight_loader(
                    param, loaded_weight.narrow(0, offset, sz), sid,
                )
                offset += sz
            return
        effective_tp = 1 if self.disable_tp else tp
        shard_offset = sum(self.output_sizes[:shard_id]) // effective_tp
        shard_size = self.output_sizes[shard_id] // effective_tp
        dst = param.data.narrow(0, shard_offset, shard_size)
        if self.disable_tp:
            dst.copy_(loaded_weight)
        else:
            src = loaded_weight.chunk(tp, 0)[rank]
            dst.copy_(src)

    def _scale_loader(self, param, loaded_weight, shard_id: int):
        tp, rank = _tp_size(), _tp_rank()
        effective_tp = 1 if self.disable_tp else tp
        shard_size_out = self.output_sizes[shard_id] // effective_tp
        scale_rows = math.ceil(shard_size_out / _FP8_BLOCK)
        shard_offset_out = sum(self.output_sizes[:shard_id]) // effective_tp
        scale_offset = math.ceil(shard_offset_out / _FP8_BLOCK)
        if self.disable_tp:
            param.data.narrow(0, scale_offset, scale_rows).copy_(loaded_weight)
        else:
            src = loaded_weight.chunk(tp, 0)[rank]
            param.data.narrow(0, scale_offset, scale_rows).copy_(src)

    def forward(self, x):
        if self.use_fp8:
            return self.linear_op(x, self.weight, self.weight_scale_inv, self.bias)
        return F.linear(x, self.weight, self.bias)


class QKVParallelLinear(nn.Module):
    """Q, K, V projections merged and sharded across TP."""

    def __init__(self, hidden_size: int, head_size: int,
                 total_num_heads: int, total_num_kv_heads: int,
                 bias: bool = False, quant_config: dict | None = None):
        super().__init__()
        tp = _tp_size()
        self.head_size = head_size
        self.num_heads = total_num_heads // tp
        # Replicate KV heads when not evenly divisible by TP
        if total_num_kv_heads % tp == 0:
            self.num_kv_heads = total_num_kv_heads // tp
            self._replicate_kv = False
        else:
            self.num_kv_heads = total_num_kv_heads
            self._replicate_kv = True
        output_size = (self.num_heads + 2 * self.num_kv_heads) * head_size
        self.use_fp8 = quant_config is not None

        if self.use_fp8:
            self.weight = nn.Parameter(
                torch.empty(output_size, hidden_size, dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            self.weight_scale_inv = nn.Parameter(
                torch.empty(*_scale_shape(output_size, hidden_size), dtype=torch.float32),
                requires_grad=False,
            )
            self.weight.weight_loader = self._weight_loader
            self.weight_scale_inv.weight_loader = self._scale_loader
            self.linear_op = _get_fp8_linear_cls()()
        else:
            self.weight = nn.Parameter(torch.empty(output_size, hidden_size))
            self.weight.weight_loader = self._weight_loader

        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self._weight_loader

    def _weight_loader(self, param, loaded_weight, shard_id: str):
        tp, rank = _tp_size(), _tp_rank()
        if shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
            src = loaded_weight.chunk(tp, 0)[rank]
        elif shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
            src = loaded_weight if self._replicate_kv else loaded_weight.chunk(tp, 0)[rank]
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
            src = loaded_weight if self._replicate_kv else loaded_weight.chunk(tp, 0)[rank]
        dst = param.data.narrow(0, shard_offset, shard_size)
        dst.copy_(src)

    def _scale_loader(self, param, loaded_weight, shard_id: str):
        tp, rank = _tp_size(), _tp_rank()
        if shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        scale_rows = math.ceil(shard_size / _FP8_BLOCK)
        scale_offset = math.ceil(shard_offset / _FP8_BLOCK)
        src = loaded_weight.chunk(tp, 0)[rank]
        param.data.narrow(0, scale_offset, scale_rows).copy_(src)

    def forward(self, x):
        if self.use_fp8:
            return self.linear_op(x, self.weight, self.weight_scale_inv, self.bias)
        return F.linear(x, self.weight, self.bias)


class ReplicatedLinear(nn.Module):
    """Full weight replicated on every TP rank (no sharding, no all-reduce)."""

    def __init__(self, input_size: int, output_size: int, bias: bool = True,
                 quant_config: dict | None = None):
        super().__init__()
        self.use_fp8 = quant_config is not None

        if self.use_fp8:
            self.weight = nn.Parameter(
                torch.empty(output_size, input_size, dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            self.weight_scale_inv = nn.Parameter(
                torch.empty(*_scale_shape(output_size, input_size),
                            dtype=torch.float32),
                requires_grad=False,
            )
            self.weight.weight_loader = lambda p, w: p.data.copy_(w)
            self.weight_scale_inv.weight_loader = lambda p, w: p.data.copy_(w)
            self.linear_op = _get_fp8_linear_cls()()
        else:
            self.weight = nn.Parameter(torch.empty(output_size, input_size))
            self.weight.weight_loader = lambda p, w: p.data.copy_(w)

        self.bias = nn.Parameter(torch.empty(output_size)) if bias else None
        if self.bias is not None:
            self.bias.weight_loader = lambda p, w: p.data.copy_(w)

    def forward(self, x):
        if self.use_fp8:
            return self.linear_op(x, self.weight, self.weight_scale_inv, self.bias)
        return F.linear(x, self.weight, self.bias)


class RowParallelLinear(nn.Module):
    """Splits input dim across TP ranks, all-reduces output."""

    def __init__(self, input_size: int, output_size: int, bias: bool = False,
                 quant_config: dict | None = None, reduce_results: bool = True):
        super().__init__()
        tp = _tp_size()
        assert input_size % tp == 0
        self.input_size_per_partition = input_size // tp
        self.tp_size = tp
        self.tp_rank = _tp_rank()
        self.reduce_results = reduce_results
        self.use_fp8 = quant_config is not None

        if self.use_fp8:
            self.weight = nn.Parameter(
                torch.empty(output_size, self.input_size_per_partition,
                            dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            self.weight_scale_inv = nn.Parameter(
                torch.empty(*_scale_shape(output_size, self.input_size_per_partition),
                            dtype=torch.float32),
                requires_grad=False,
            )
            self.weight.weight_loader = self._weight_loader
            self.weight_scale_inv.weight_loader = self._scale_loader
            self.linear_op = _get_fp8_linear_cls()()
        else:
            self.weight = nn.Parameter(torch.empty(output_size, self.input_size_per_partition))
            self.weight.weight_loader = self._weight_loader

        self.bias = nn.Parameter(torch.empty(output_size)) if bias else None
        if self.bias is not None:
            self.bias.weight_loader = lambda p, w: p.data.copy_(w)
        self.allreduce = AllReduce()

    def _weight_loader(self, param, loaded_weight):
        tp, rank = _tp_size(), _tp_rank()
        shard = param.data.size(1)
        loaded_weight = loaded_weight.narrow(1, rank * shard, shard)
        param.data.copy_(loaded_weight)

    def _scale_loader(self, param, loaded_weight):
        tp, rank = _tp_size(), _tp_rank()
        cols_per_shard = param.data.size(1)
        loaded_weight = loaded_weight.narrow(1, rank * cols_per_shard, cols_per_shard)
        param.data.copy_(loaded_weight)

    def forward(self, x):
        if self.use_fp8:
            y = self.linear_op(x, self.weight, self.weight_scale_inv,
                               self.bias if self.tp_rank == 0 else None)
        else:
            y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.reduce_results and self.tp_size > 1:
            y = self.allreduce(y)
        return y


# Inlined from tasks/reference/L2/t5_attention.py
class T5SelfAttention(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias: bool = False):
        super().__init__()
        self.d_model = config.d_model
        self.d_kv = config.d_kv
        self.n_heads = config.num_heads
        self.inner_dim = self.n_heads * self.d_kv
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance

        tp_size = _tp_size()
        assert self.n_heads % tp_size == 0
        self.n_heads_per_partition = self.n_heads // tp_size

        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.d_model,
            head_size=self.d_kv,
            total_num_heads=self.n_heads,
            total_num_kv_heads=self.n_heads,
            bias=False,
        )

        self.o = RowParallelLinear(self.inner_dim, self.d_model, bias=False)

        self.bmm = BMM()
        self.softmax = Softmax(dim=-1)

        if has_relative_attention_bias:
            self.relative_attention_bias = Embedding(
                self.relative_attention_num_buckets, self.n_heads,
            )

    @staticmethod
    def _relative_position_bucket(
        relative_position: torch.Tensor,
        bidirectional: bool = True,
        num_buckets: int = 32,
        max_distance: int = 128,
    ) -> torch.Tensor:
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(
                relative_position, torch.zeros_like(relative_position),
            )
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, num_buckets - 1),
        )
        relative_buckets += torch.where(
            is_small, relative_position, relative_position_if_large,
        )
        return relative_buckets

    def compute_bias(self, query_length: int, key_length: int, device: torch.device) -> torch.Tensor:
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_position_bucket(
            relative_position, bidirectional=True,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(relative_position_bucket)
        tp_rank = _tp_rank()
        head_start = tp_rank * self.n_heads_per_partition
        head_end = head_start + self.n_heads_per_partition
        values = values[:, :, head_start:head_end]
        values = values.permute(2, 0, 1).unsqueeze(0)
        return values

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor | None = None,
        position_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_length = hidden_states.shape[:2]

        qkv = self.qkv_proj(hidden_states)
        q_size = self.n_heads_per_partition * self.d_kv
        kv_size = self.n_heads_per_partition * self.d_kv
        query_states, key_states, value_states = qkv.split(
            [q_size, kv_size, kv_size], dim=-1,
        )

        query_states = query_states.view(
            batch_size, seq_length, self.n_heads_per_partition, self.d_kv,
        ).transpose(1, 2)
        key_states = key_states.view(
            batch_size, seq_length, self.n_heads_per_partition, self.d_kv,
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, seq_length, self.n_heads_per_partition, self.d_kv,
        ).transpose(1, 2)

        scores = self.bmm(query_states, key_states.transpose(3, 2))

        if position_bias is None:
            if self.has_relative_attention_bias:
                position_bias = self.compute_bias(
                    seq_length, seq_length, device=scores.device,
                )
            else:
                position_bias = torch.zeros(
                    (1, self.n_heads_per_partition, seq_length, seq_length),
                    device=scores.device, dtype=scores.dtype,
                )
            if mask is not None:
                position_bias = position_bias + mask

        scores += position_bias
        attn_weights = self.softmax(scores.float()).type_as(scores)
        attn_output = self.bmm(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_length, -1)
        attn_output = self.o(attn_output)

        return attn_output, position_bias


# Inlined from tasks/reference/L1/gelu.py
class GELU(nn.Module):
    def __init__(self, approximate: str = "none"):
        super().__init__()
        self.approximate = approximate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x, approximate=self.approximate)


# Inlined from tasks/reference/L2/t5_dense.py
class NewGELUActivation(nn.Module):
    """GELU approximation matching HuggingFace's NewGELUActivation exactly."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


def _get_act_fn(name: str) -> nn.Module:
    act_fns = {
        "relu": nn.ReLU(),
        "gelu": GELU(),
        "gelu_new": NewGELUActivation(),
        "silu": SiLU(),
    }
    if name in act_fns:
        return act_fns[name]
    raise ValueError(f"Unknown activation function: {name}")


class T5DenseGatedActDense(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.wi = MergedColumnParallelLinear(
            config.d_model, [config.d_ff, config.d_ff], bias=False,
        )
        self.wo = RowParallelLinear(config.d_ff, config.d_model, bias=False)
        self.act = _get_act_fn(config.dense_act_fn)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up = self.wi(hidden_states)
        gate, up = gate_up.chunk(2, dim=-1)
        hidden_states = self.act(gate) * up
        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5DenseActDense(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.wi = ColumnParallelLinear(config.d_model, config.d_ff, bias=False)
        self.wo = RowParallelLinear(config.d_ff, config.d_model, bias=False)
        self.act = _get_act_fn(config.dense_act_fn)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


# Inlined from tasks/reference/L3/t5_block.py
class T5LayerSelfAttention(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias: bool = False):
        super().__init__()
        self.SelfAttention = T5SelfAttention(config, has_relative_attention_bias)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor | None = None,
        position_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normed = self.layer_norm(hidden_states)
        attn_output, position_bias = self.SelfAttention(
            normed, mask=mask, position_bias=position_bias,
        )
        hidden_states = hidden_states + attn_output
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        return hidden_states, position_bias


class T5LayerFF(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        if config.is_gated_act:
            self.DenseReluDense = T5DenseGatedActDense(config)
        else:
            self.DenseReluDense = T5DenseActDense(config)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normed = self.layer_norm(hidden_states)
        ff_output = self.DenseReluDense(normed)
        hidden_states = hidden_states + ff_output
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        return hidden_states


class T5Block(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias: bool = False):
        super().__init__()
        self.layer = nn.ModuleList([
            T5LayerSelfAttention(config, has_relative_attention_bias),
            T5LayerFF(config),
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor | None = None,
        position_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states, position_bias = self.layer[0](
            hidden_states, mask=mask, position_bias=position_bias,
        )
        hidden_states = self.layer[1](hidden_states)
        return hidden_states, position_bias


# Inlined from tasks/reference/L4/t5_encoder.py
from collections.abc import Iterable


class T5Stack(nn.Module):
    def __init__(self, config: T5Config, shared: nn.Embedding):
        super().__init__()
        self.embed_tokens = shared
        self.block = nn.ModuleList([
            T5Block(config, has_relative_attention_bias=(i == 0))
            for i in range(config.num_layers)
        ])
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)

        if attention_mask is not None:
            extended_mask = attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
            extended_mask = (1.0 - extended_mask) * torch.finfo(hidden_states.dtype).min
        else:
            extended_mask = None

        position_bias = None
        for block in self.block:
            hidden_states, position_bias = block(
                hidden_states, mask=extended_mask, position_bias=position_bias,
            )

        hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states


class T5EncoderModel(nn.Module):
    """TP-aware T5 encoder model for diffusion pipelines."""

    def __init__(self, config: T5Config):
        super().__init__()
        self.config = config
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = T5Stack(config, self.shared)

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, ...]:
        hidden_states = self.encoder(input_ids, attention_mask=attention_mask)
        return (hidden_states,)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ("qkv_proj", "q", "q"),
            ("qkv_proj", "k", "k"),
            ("qkv_proj", "v", "v"),
            ("wi", "wi_0", 0),
            ("wi", "wi_1", 1),
        ]

        def _default_weight_loader(param, loaded_weight):
            param.data.copy_(loaded_weight)

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            original_name = name
            lookup_name = name

            if "relative_attention_bias.weight" in name:
                lookup_name = name.replace(
                    "relative_attention_bias.weight",
                    "relative_attention_bias.emb.weight",
                )

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if f".{weight_name}." not in name:
                    continue
                lookup_name = name.replace(f".{weight_name}.", f".{param_name}.")
                if lookup_name not in params_dict:
                    continue
                param = params_dict[lookup_name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if lookup_name not in params_dict:
                    continue
                param = params_dict[lookup_name]
                weight_loader = getattr(param, "weight_loader", _default_weight_loader)
                weight_loader(param, loaded_weight)

            loaded_params.add(original_name)
            loaded_params.add(lookup_name)

        return loaded_params


# Inlined from tasks/reference/L1/flux_pos_embed.py
def _get_1d_rotary_pos_embed(
    dim: int,
    pos: np.ndarray | int | torch.Tensor,
    theta: float = 10000.0,
    use_real: bool = False,
    linear_factor: float = 1.0,
    ntk_factor: float = 1.0,
    repeat_interleave_real: bool = True,
    freqs_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Precompute the frequency tensor for complex exponentials (cis).

    Copied from ``diffusers.models.embeddings.get_1d_rotary_pos_embed``.
    Returns complex64 tensor of shape [S, dim/2] when ``use_real=False``.
    """
    assert dim % 2 == 0

    if isinstance(pos, int):
        pos = torch.arange(pos)
    if isinstance(pos, np.ndarray):
        pos = torch.from_numpy(pos)

    theta = theta * ntk_factor
    freqs = (
        1.0 / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=pos.device) / dim)) / linear_factor
    )
    freqs = torch.outer(pos, freqs)

    if use_real and repeat_interleave_real:
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()
        return freqs_cos, freqs_sin
    elif use_real:
        freqs_cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).float()
        freqs_sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).float()
        return freqs_cos, freqs_sin
    else:
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis


class FluxPosEmbed(nn.Module):
    """2D rotary position embeddings for FLUX."""

    def __init__(self, theta: int, axes_dim: list[int] | tuple[int, ...]):
        super().__init__()
        self.theta = theta
        self.axes_dim = list(axes_dim)

    def forward(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n_axes = ids.shape[-1]
        cos_out = []
        sin_out = []
        pos = ids.float()
        freqs_dtype = torch.float32 if ids.device.type in ("mps", "npu") else torch.float64
        for i in range(n_axes):
            freqs_cis = _get_1d_rotary_pos_embed(
                self.axes_dim[i], pos[:, i],
                theta=self.theta, use_real=False,
                freqs_dtype=freqs_dtype,
            )
            cos_out.append(freqs_cis.real)
            sin_out.append(freqs_cis.imag)
        freqs_cos = torch.cat(cos_out, dim=-1).to(ids.device)
        freqs_sin = torch.cat(sin_out, dim=-1).to(ids.device)
        return freqs_cos, freqs_sin


# Inlined from tasks/reference/L2/ada_layer_norm.py
class AdaLayerNormZero(nn.Module):
    r"""
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim: int, num_embeddings: int | None = None,
                 norm_type="layer_norm", bias=True):
        super().__init__()
        self.emb = None

        self.silu = SiLU()
        self.linear = Linear(embedding_dim, 6 * embedding_dim, bias=bias)
        if norm_type == "layer_norm":
            self.norm = LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)
        else:
            raise ValueError(
                f"Unsupported `norm_type` ({norm_type}) provided. Supported ones are: 'layer_norm'."
            )

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor | None = None,
        class_labels: torch.LongTensor | None = None,
        hidden_dtype: torch.dtype | None = None,
        emb: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.emb is not None:
            emb = self.emb(timestep, class_labels, hidden_dtype=hidden_dtype)
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormZeroSingle(nn.Module):
    r"""
    Norm layer adaptive layer norm zero (adaLN-Zero) for single-stream blocks.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
    """

    def __init__(self, embedding_dim: int, norm_type="layer_norm", bias=True):
        super().__init__()

        self.silu = SiLU()
        self.linear = Linear(embedding_dim, 3 * embedding_dim, bias=bias)
        if norm_type == "layer_norm":
            self.norm = LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)
        else:
            raise ValueError(
                f"Unsupported `norm_type` ({norm_type}) provided. Supported ones are: 'layer_norm'."
            )

    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa = emb.chunk(3, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa


# Inlined from tasks/reference/L1/diffusion_rope.py
from typing import Optional, Union


import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel  (from flash_attn / vllm_flash_attn, Copyright (c) 2023 Tri Dao)
# ---------------------------------------------------------------------------

@triton.jit
def _rotary_kernel(
    OUT, X, COS, SIN, CU_SEQLENS, SEQLEN_OFFSETS,
    seqlen, rotary_dim, seqlen_ro,
    stride_out_batch, stride_out_seqlen, stride_out_nheads, stride_out_headdim,
    stride_x_batch, stride_x_seqlen, stride_x_nheads, stride_x_headdim,
    BLOCK_K: tl.constexpr,
    IS_SEQLEN_OFFSETS_TENSOR: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    CONJUGATE: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_head = tl.program_id(axis=1)
    pid_batch = tl.program_id(axis=2)
    rotary_dim_half = rotary_dim // 2

    if not IS_VARLEN:
        X = X + pid_batch * stride_x_batch + pid_head * stride_x_nheads
        OUT = OUT + pid_batch * stride_out_batch + pid_head * stride_out_nheads
    else:
        start_idx = tl.load(CU_SEQLENS + pid_batch)
        seqlen = tl.load(CU_SEQLENS + pid_batch + 1) - start_idx
        X = X + start_idx * stride_x_seqlen + pid_head * stride_x_nheads
        OUT = OUT + start_idx * stride_out_seqlen + pid_head * stride_out_nheads

    if pid_m * BLOCK_M >= seqlen:
        return
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    if not IS_SEQLEN_OFFSETS_TENSOR:
        rm_cs = rm + SEQLEN_OFFSETS
    else:
        rm_cs = rm + tl.load(SEQLEN_OFFSETS + pid_batch)
    rk = tl.arange(0, BLOCK_K)
    rk_half = tl.arange(0, BLOCK_K // 2)

    if not INTERLEAVED:
        X = X + (rm[:, None] * stride_x_seqlen + rk_half[None, :] * stride_x_headdim)
        COS = COS + (rm_cs[:, None] * rotary_dim_half + rk_half[None, :])
        SIN = SIN + (rm_cs[:, None] * rotary_dim_half + rk_half[None, :])
        cos = tl.load(
            COS, mask=(rm_cs[:, None] < seqlen_ro) & (rk_half[None, :] < rotary_dim_half), other=1.0
        ).to(tl.float32)
        sin = tl.load(
            SIN, mask=(rm_cs[:, None] < seqlen_ro) & (rk_half[None, :] < rotary_dim_half), other=0.0
        ).to(tl.float32)
        x0 = tl.load(
            X, mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half), other=0.0
        ).to(tl.float32)
        x1 = tl.load(
            X + rotary_dim_half * stride_x_headdim,
            mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half),
            other=0.0,
        ).to(tl.float32)
        if CONJUGATE:
            sin = -sin
        o0 = x0 * cos - x1 * sin
        o1 = x0 * sin + x1 * cos
        OUT = OUT + (rm[:, None] * stride_out_seqlen + rk_half[None, :] * stride_out_headdim)
        tl.store(OUT, o0, mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half))
        tl.store(
            OUT + rotary_dim_half * stride_out_headdim,
            o1,
            mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half),
        )
    else:
        rk_swap = rk + ((rk + 1) % 2) * 2 - 1  # 1, 0, 3, 2, 5, 4, ...
        rk_repeat = tl.arange(0, BLOCK_K) // 2
        X0 = X + (rm[:, None] * stride_x_seqlen + rk[None, :] * stride_x_headdim)
        X1 = X + (rm[:, None] * stride_x_seqlen + rk_swap[None, :] * stride_x_headdim)
        COS = COS + (rm_cs[:, None] * rotary_dim_half + rk_repeat[None, :])
        SIN = SIN + (rm_cs[:, None] * rotary_dim_half + rk_repeat[None, :])
        cos = tl.load(
            COS,
            mask=(rm_cs[:, None] < seqlen_ro) & (rk_repeat[None, :] < rotary_dim_half),
            other=1.0,
        ).to(tl.float32)
        sin = tl.load(
            SIN,
            mask=(rm_cs[:, None] < seqlen_ro) & (rk_repeat[None, :] < rotary_dim_half),
            other=0.0,
        ).to(tl.float32)
        x0 = tl.load(X0, mask=(rm[:, None] < seqlen) & (rk[None, :] < rotary_dim), other=0.0).to(
            tl.float32
        )
        x1 = tl.load(
            X1, mask=(rm[:, None] < seqlen) & (rk_swap[None, :] < rotary_dim), other=0.0
        ).to(tl.float32)
        if CONJUGATE:
            sin = -sin
        x0_cos = x0 * cos
        x1_sin = x1 * sin
        out = tl.where(rk[None, :] % 2 == 0, x0_cos - x1_sin, x0_cos + x1_sin)
        OUT = OUT + (rm[:, None] * stride_out_seqlen + rk[None, :] * stride_out_headdim)
        tl.store(OUT, out, mask=(rm[:, None] < seqlen) & (rk[None, :] < rotary_dim))


def _apply_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    interleaved: bool = False,
    inplace: bool = False,
    conjugate: bool = False,
) -> torch.Tensor:
    """Launch the Triton rotary-embedding kernel.

    Args:
        x: (batch, seqlen, nheads, headdim) or (total_seqlen, nheads, headdim)
            if ``cu_seqlens`` is provided.
        cos, sin: (seqlen_ro, rotary_dim / 2)
    """
    is_varlen = cu_seqlens is not None
    if not is_varlen:
        batch, seqlen, nheads, headdim = x.shape
    else:
        assert max_seqlen is not None
        total_seqlen, nheads, headdim = x.shape
        batch = cu_seqlens.shape[0] - 1
        seqlen = max_seqlen
    seqlen_ro, rotary_dim = cos.shape
    rotary_dim *= 2
    assert rotary_dim <= headdim
    assert headdim <= 256
    assert seqlen_ro >= seqlen

    cos, sin = cos.contiguous(), sin.contiguous()
    if isinstance(seqlen_offsets, torch.Tensor):
        seqlen_offsets = seqlen_offsets.contiguous()

    output = torch.empty_like(x) if not inplace else x
    if rotary_dim < headdim and not inplace:
        output[..., rotary_dim:].copy_(x[..., rotary_dim:])

    BLOCK_K = (
        32 if rotary_dim <= 32
        else (64 if rotary_dim <= 64
              else (128 if rotary_dim <= 128 else 256))
    )
    BLOCK_M = 4 if interleaved else (8 if rotary_dim <= 128 else 4)
    grid = lambda META: (triton.cdiv(seqlen, META["BLOCK_M"]), nheads, batch)  # noqa

    with torch.cuda.device(x.device.index):
        _rotary_kernel[grid](
            output, x, cos, sin, cu_seqlens, seqlen_offsets,
            seqlen, rotary_dim, seqlen_ro,
            output.stride(0) if not is_varlen else 0,
            output.stride(-3), output.stride(-2), output.stride(-1),
            x.stride(0) if not is_varlen else 0,
            x.stride(-3), x.stride(-2), x.stride(-1),
            BLOCK_K,
            isinstance(seqlen_offsets, torch.Tensor),
            is_varlen, interleaved, conjugate, BLOCK_M,
            num_warps=2 if rotary_dim <= 64 else 4,
        )
    return output


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------

class DiffusionRoPE(nn.Module):
    """Apply rotary embeddings given pre-computed (cos, sin) tensors.

    Parameters
    ----------
    is_neox_style : bool
        If True, use the GPT-NeoX (half-split) layout.
        If False (default for FLUX), use the interleaved (GPT-J) layout.
    """

    def __init__(self, is_neox_style: bool = False) -> None:
        super().__init__()
        self.interleaved = not is_neox_style

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        if cos.dim() == 3:
            cos = cos[0]
            sin = sin[0]
        return _apply_rotary(x, cos, sin, interleaved=self.interleaved)


# Inlined from tasks/reference/L1/dense_attention.py
from typing import Literal


class DenseAttention(nn.Module):
    """Dense multi-head attention with ``(batch, seq, heads, dim)`` layout."""

    def __init__(self, backend: Literal["auto", "sdpa", "flash_attn"] = "auto"):
        super().__init__()
        del backend

    def forward(
        self,
        query,
        key,
        value,
        softmax_scale=None,
        causal=False,
        attn_mask=None,
    ):
        q = query.permute(0, 2, 1, 3)
        k = key.permute(0, 2, 1, 3)
        v = value.permute(0, 2, 1, 3)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=causal,
            scale=softmax_scale,
        )
        return out.permute(0, 2, 1, 3)


# Inlined from tasks/reference/L2/flux_attention.py
def _tensor_model_parallel_all_gather(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Gather tensor across TP ranks along the given dimension."""
    import torch.distributed as dist
    tp = _tp_size()
    if tp <= 1:
        return tensor
    gather_list = [torch.empty_like(tensor) for _ in range(tp)]
    dist.all_gather(gather_list, tensor)
    return torch.cat(gather_list, dim=dim)


# The baseline spells this ``from ..L1.t5_layer_norm import T5LayerNorm as
# FP32RMSNorm``; inlining drops the alias, so rebind it here.  It has to live
# in this module's own section rather than inside the inlined t5_layer_norm
# block: a composite reference emits each inlined block once by path, so an
# alias parked in a shared block is lost whenever another module contributes
# that same block first.
FP32RMSNorm = T5LayerNorm


class FluxAttention(nn.Module):
    """Multi-head attention for FLUX diffusion transformer.

    Supports two modes controlled by constructor args:
    - Dual-stream (``added_kv_proj_dim is not None``): separate QKV for image
      and text streams, concatenated before attention, split after.
    - Single-stream / pre-only (``pre_only=True``): standard self-attention,
      no output projection (caller handles it).
    """

    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        added_kv_proj_dim: int | None = None,
        added_proj_bias: bool | None = True,
        out_bias: bool = True,
        eps: float = 1e-5,
        out_dim: int | None = None,
        context_pre_only: bool | None = None,
        pre_only: bool = False,
        quant_config: dict | None = None,
    ):
        super().__init__()
        self.head_dim = dim_head
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.use_bias = bias
        self.dropout = dropout
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.context_pre_only = context_pre_only
        self.pre_only = pre_only
        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.added_kv_proj_dim = added_kv_proj_dim

        self.norm_q = FP32RMSNorm(dim_head, eps=eps)
        self.norm_k = FP32RMSNorm(dim_head, eps=eps)

        self.to_qkv = QKVParallelLinear(
            hidden_size=query_dim,
            head_size=self.head_dim,
            total_num_heads=self.heads,
            total_num_kv_heads=self.heads,
            bias=bias,
            quant_config=quant_config,
        )

        if not self.pre_only:
            self.to_out = nn.ModuleList([
                RowParallelLinear(self.inner_dim, self.out_dim, bias=out_bias,
                                  quant_config=quant_config),
                nn.Dropout(dropout),
            ])

        if added_kv_proj_dim is not None:
            self.norm_added_q = FP32RMSNorm(dim_head, eps=eps)
            self.norm_added_k = FP32RMSNorm(dim_head, eps=eps)

            self.add_kv_proj = QKVParallelLinear(
                hidden_size=added_kv_proj_dim,
                head_size=self.head_dim,
                total_num_heads=self.heads,
                total_num_kv_heads=self.heads,
                bias=added_proj_bias if added_proj_bias is not None else True,
                quant_config=quant_config,
            )

            self.to_add_out = RowParallelLinear(
                self.inner_dim, query_dim, bias=out_bias,
                quant_config=quant_config,
            )

        self.rope = DiffusionRoPE(is_neox_style=False)
        self.attn = DenseAttention()

    def _apply_rope(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if image_rotary_emb is not None:
            cos, sin = image_rotary_emb
            cos = cos.to(query.dtype)
            sin = sin.to(query.dtype)
            query = self.rope(query, cos, sin)
            key = self.rope(key, cos, sin)
        return query, key

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        num_heads = self.to_qkv.num_heads
        num_kv_heads = self.to_qkv.num_kv_heads

        qkv = self.to_qkv(hidden_states)
        q_size = num_heads * self.head_dim
        kv_size = num_kv_heads * self.head_dim
        query, key, value = qkv.split([q_size, kv_size, kv_size], dim=-1)

        query = query.unflatten(-1, (num_heads, -1))
        key = key.unflatten(-1, (num_kv_heads, -1))
        value = value.unflatten(-1, (num_kv_heads, -1))

        query = self.norm_q(query)
        key = self.norm_k(key)

        if self.added_kv_proj_dim is not None:
            add_num_heads = self.add_kv_proj.num_heads
            add_num_kv_heads = self.add_kv_proj.num_kv_heads

            encoder_qkv = self.add_kv_proj(encoder_hidden_states)
            add_q_size = add_num_heads * self.head_dim
            add_kv_size = add_num_kv_heads * self.head_dim
            encoder_query, encoder_key, encoder_value = encoder_qkv.split(
                [add_q_size, add_kv_size, add_kv_size], dim=-1
            )

            encoder_query = encoder_query.unflatten(-1, (add_num_heads, -1))
            encoder_key = encoder_key.unflatten(-1, (add_num_kv_heads, -1))
            encoder_value = encoder_value.unflatten(-1, (add_num_kv_heads, -1))

            encoder_query = self.norm_added_q(encoder_query)
            encoder_key = self.norm_added_k(encoder_key)

            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        query, key = self._apply_rope(query, key, image_rotary_emb)

        softmax_scale = 1.0 / (self.head_dim ** 0.5)
        hidden_states = self.attn(query, key, value, softmax_scale=softmax_scale, causal=False)
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]],
                dim=1,
            )
            hidden_states = self.to_out[0](hidden_states.contiguous())
            hidden_states = self.to_out[1](hidden_states)
            encoder_hidden_states = self.to_add_out(encoder_hidden_states.contiguous())
            return hidden_states, encoder_hidden_states
        else:
            if _tp_size() > 1:
                hidden_states = _tensor_model_parallel_all_gather(hidden_states, dim=-1)
            return hidden_states


# Inlined from tasks/reference/L2/flux_feedforward.py
class ColumnParallelApproxGELU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, *, approximate: str, bias: bool = True,
                 quant_config: dict | None = None):
        super().__init__()
        self.proj = ColumnParallelLinear(dim_in, dim_out, bias=bias, quant_config=quant_config)
        self.gelu = GELU(approximate=approximate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return self.gelu(x)


class FeedForward(nn.Module):
    """FLUX FFN: GELU(tanh) linear -> linear with TP sharding."""

    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        mult: int = 4,
        inner_dim: int | None = None,
        bias: bool = True,
        quant_config: dict | None = None,
    ) -> None:
        super().__init__()
        inner_dim = inner_dim or int(dim * mult)
        dim_out = dim_out or dim

        layers: list[nn.Module] = [
            ColumnParallelApproxGELU(dim, inner_dim, approximate="tanh", bias=bias,
                                      quant_config=quant_config),
            nn.Identity(),
            RowParallelLinear(inner_dim, dim_out, bias=bias, quant_config=quant_config),
        ]
        self.net = nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


# Inlined from tasks/reference/L3/flux_transformer_block.py
from typing import Any


class FluxTransformerBlock(nn.Module):
    """Dual-stream DiT block: joint attention over text+image, then separate FFNs."""

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        qk_norm: str = "rms_norm",
        eps: float = 1e-6,
        quant_config: dict | None = None,
    ):
        super().__init__()
        self.norm1 = AdaLayerNormZero(dim)
        self.norm1_context = AdaLayerNormZero(dim)

        self.attn = FluxAttention(
            query_dim=dim,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            eps=eps,
            quant_config=quant_config,
        )

        self.norm2 = LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, dim_out=dim, quant_config=quant_config)

        self.norm2_context = LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_context = FeedForward(dim=dim, dim_out=dim, quant_config=quant_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states, emb=temb
        )
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )
        joint_attention_kwargs = joint_attention_kwargs or {}

        attention_outputs = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        if len(attention_outputs) == 2:
            attn_output, context_attn_output = attention_outputs
        elif len(attention_outputs) == 3:
            attn_output, context_attn_output, ip_attn_output = attention_outputs

        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output
        hidden_states = hidden_states + ff_output

        if len(attention_outputs) == 3:
            hidden_states = hidden_states + ip_attn_output

        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = (
            norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        )

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


class FluxSingleTransformerBlock(nn.Module):
    """Single-stream DiT block: text+image concatenated, self-attention + MLP in parallel."""

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 4.0,
        quant_config: dict | None = None,
    ):
        super().__init__()
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm = AdaLayerNormZeroSingle(dim)
        self.proj_mlp = ReplicatedLinear(dim, self.mlp_hidden_dim, bias=True,
                                         quant_config=quant_config)
        self.act_mlp = GELU(approximate="tanh")
        self.proj_out = ReplicatedLinear(dim + self.mlp_hidden_dim, dim, bias=True,
                                         quant_config=quant_config)

        self.attn = FluxAttention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            eps=1e-6,
            pre_only=True,
            quant_config=quant_config,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text_seq_len = encoder_hidden_states.shape[1]
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))

        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        gate = gate.unsqueeze(1)
        hidden_states = gate * self.proj_out(hidden_states)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        encoder_hidden_states, hidden_states = (
            hidden_states[:, :text_seq_len],
            hidden_states[:, text_seq_len:],
        )
        return encoder_hidden_states, hidden_states


VaeImageProcessor = VideoProcessor


import logging


from torch import nn


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class FluxConfig:
    """Configuration for the FLUX transformer model."""
    num_layers: int = 19
    num_single_layers: int = 38
    attention_head_dim: int = 128
    num_attention_heads: int = 24
    in_channels: int = 64
    out_channels: int | None = None
    joint_attention_dim: int = 4096
    pooled_projection_dim: int = 768
    guidance_embeds: bool = True
    axes_dims_rope: tuple[int, int, int] = (16, 56, 56)
    patch_size: int = 1

    @classmethod
    def from_pretrained(cls, model_name: str) -> "FluxConfig":
        """Load config from transformer/config.json in a FLUX model repo."""
        from huggingface_hub import hf_hub_download
        if os.path.isdir(model_name):
            config_path = os.path.join(model_name, "transformer", "config.json")
        else:
            config_path = hf_hub_download(model_name, "transformer/config.json")
        with open(config_path) as f:
            data = json.load(f)
        return cls(
            num_layers=data.get("num_layers", 19),
            num_single_layers=data.get("num_single_layers", 38),
            attention_head_dim=data.get("attention_head_dim", 128),
            num_attention_heads=data.get("num_attention_heads", 24),
            in_channels=data.get("in_channels", 64),
            out_channels=data.get("out_channels", None),
            joint_attention_dim=data.get("joint_attention_dim", 4096),
            pooled_projection_dim=data.get("pooled_projection_dim", 768),
            guidance_embeds=data.get("guidance_embeds", True),
            axes_dims_rope=tuple(data.get("axes_dims_rope", [16, 56, 56])),
            patch_size=data.get("patch_size", 1),
        )


@dataclass
class DiffusionSamplingParams:
    """Sampling parameters for diffusion generation."""
    height: int | None = None
    width: int | None = None
    num_inference_steps: int = 28
    guidance_scale: float = 3.5
    true_cfg_scale: float = 1.0
    num_outputs_per_prompt: int = 1
    seed: int | None = None
    sigmas: list[float] | None = None
    output_type: str = "pil"
    max_sequence_length: int = 512


@dataclass
class DiffusionOutput:
    """Output of the diffusion pipeline."""
    images: list[Any] | None = None
    latents: torch.Tensor | None = None


# ---------------------------------------------------------------------------
# Transformer backbone
# ---------------------------------------------------------------------------

class FluxTransformer2DModel(nn.Module):
    """FLUX DiT backbone: dual-stream + single-stream transformer blocks.

    Takes packed latent patches + text embeddings + timestep conditioning
    and produces noise predictions.
    """

    def __init__(self, config: FluxConfig, quant_config: dict | None = None):
        super().__init__()
        self.config = config
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels or config.in_channels
        inner_dim = config.num_attention_heads * config.attention_head_dim
        self.inner_dim = inner_dim
        self.guidance_embeds = config.guidance_embeds

        self.pos_embed = FluxPosEmbed(
            theta=10000, axes_dim=config.axes_dims_rope,
        )

        text_time_guidance_cls = (
            CombinedTimestepGuidanceTextProjEmbeddings
            if config.guidance_embeds
            else CombinedTimestepTextProjEmbeddings
        )
        self.time_text_embed = text_time_guidance_cls(
            embedding_dim=inner_dim,
            pooled_projection_dim=config.pooled_projection_dim,
        )

        self.context_embedder = Linear(config.joint_attention_dim, inner_dim)
        self.x_embedder = Linear(config.in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList([
            FluxTransformerBlock(
                dim=inner_dim,
                num_attention_heads=config.num_attention_heads,
                attention_head_dim=config.attention_head_dim,
                quant_config=quant_config,
            )
            for _ in range(config.num_layers)
        ])

        self.single_transformer_blocks = nn.ModuleList([
            FluxSingleTransformerBlock(
                dim=inner_dim,
                num_attention_heads=config.num_attention_heads,
                attention_head_dim=config.attention_head_dim,
                quant_config=quant_config,
            )
            for _ in range(config.num_single_layers)
        ])

        self.norm_out = AdaLayerNormContinuous(
            inner_dim, inner_dim, elementwise_affine=False, eps=1e-6,
        )
        self.proj_out = Linear(
            inner_dim,
            config.patch_size * config.patch_size * self.out_channels,
            bias=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor]:
        hidden_states = self.x_embedder(hidden_states)
        timestep = timestep.to(
            device=hidden_states.device, dtype=hidden_states.dtype
        ) * 1000

        if guidance is not None:
            guidance = guidance.to(
                device=hidden_states.device, dtype=hidden_states.dtype
            ) * 1000

        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            img_ids = img_ids[0]

        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)

        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        for block in self.single_transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if not return_dict:
            return (output,)
        return output

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            (".to_qkv", ".to_q", "q"),
            (".to_qkv", ".to_k", "k"),
            (".to_qkv", ".to_v", "v"),
            (".add_kv_proj", ".add_q_proj", "q"),
            (".add_kv_proj", ".add_k_proj", "k"),
            (".add_kv_proj", ".add_v_proj", "v"),
        ]

        def _default_weight_loader(param, loaded_weight):
            param.data.copy_(loaded_weight)

        params_dict = dict(self.named_parameters())
        for name, buffer in self.named_buffers():
            if name.endswith(".beta") or name.endswith(".eps"):
                params_dict[name] = buffer

        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            original_name = name
            lookup_name = name
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in original_name:
                    continue
                lookup_name = original_name.replace(weight_name, param_name)
                param = params_dict[lookup_name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if lookup_name not in params_dict and ".to_out.0." in lookup_name:
                    lookup_name = lookup_name.replace(".to_out.0.", ".to_out.")
                param = params_dict[lookup_name]
                weight_loader = getattr(param, "weight_loader", _default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(original_name)
            loaded_params.add(lookup_name)
        return loaded_params


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def _retrieve_timesteps(
    scheduler,
    num_inference_steps: int | None = None,
    device: str | torch.device | None = None,
    sigmas: list[float] | None = None,
    **kwargs,
) -> tuple[torch.Tensor, int]:
    if sigmas is not None:
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
    return scheduler.timesteps, len(scheduler.timesteps)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

class FluxPipeline(nn.Module):
    """Full FLUX text-to-image pipeline.

    Composes: CLIP + T5 text encoding -> latent preparation -> denoising
    loop (flow-match Euler) -> VAE decode.
    """

    def __init__(self, config: FluxConfig, model_name: str,
                 quant_config: dict | None = None):
        from transformers import AutoConfig, CLIPTokenizer, T5TokenizerFast
        super().__init__()
        self.config = config
        self.model_name = model_name

        local_files_only = os.path.isdir(model_name)

        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_name, subfolder="scheduler", local_files_only=local_files_only,
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_name, subfolder="text_encoder", local_files_only=local_files_only,
        )
        t5_config = AutoConfig.from_pretrained(
            model_name, subfolder="text_encoder_2", local_files_only=local_files_only,
        )
        self.text_encoder_2 = T5EncoderModel(t5_config)
        self.vae = AutoencoderKL.from_pretrained(
            model_name, subfolder="vae", local_files_only=local_files_only,
        )
        self.transformer = FluxTransformer2DModel(config, quant_config=quant_config)

        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_name, subfolder="tokenizer", local_files_only=local_files_only,
        )
        self.tokenizer_2 = T5TokenizerFast.from_pretrained(
            model_name, subfolder="tokenizer_2", local_files_only=local_files_only,
        )

        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1)
            if hasattr(self.vae, "config") and hasattr(self.vae.config, "block_out_channels")
            else 8
        )
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length
            if self.tokenizer is not None
            else 77
        )
        self.default_sample_size = 128

    # -----------------------------------------------------------------------
    # Text encoding
    # -----------------------------------------------------------------------

    def _get_clip_prompt_embeds(
        self, prompt: str | list[str], num_images_per_prompt: int = 1,
    ) -> torch.Tensor:
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt, padding="max_length", max_length=self.tokenizer_max_length,
            truncation=True, return_tensors="pt",
        )
        prompt_embeds = self.text_encoder(
            text_inputs.input_ids.to(self.vae.device), output_hidden_states=False,
        )
        prompt_embeds = prompt_embeds.pooler_output
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=self.vae.device)
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)
        return prompt_embeds

    def _get_t5_prompt_embeds(
        self,
        prompt: str | list[str],
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 512,
    ) -> torch.Tensor:
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer_2(
            prompt, padding="max_length", max_length=max_sequence_length,
            truncation=True, return_tensors="pt",
        )
        prompt_embeds = self.text_encoder_2(
            text_inputs.input_ids.to(self.vae.device), output_hidden_states=False,
        )[0]
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=self.vae.device)
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        return prompt_embeds

    def encode_prompt(
        self,
        prompt: str | list[str],
        prompt_2: str | list[str] | None = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 512,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt_2 = prompt_2 or prompt
        prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

        pooled_prompt_embeds = self._get_clip_prompt_embeds(
            prompt=prompt, num_images_per_prompt=num_images_per_prompt,
        )
        prompt_embeds = self._get_t5_prompt_embeds(
            prompt=prompt_2, num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )

        t5_dtype = self.text_encoder_2.dtype
        pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=t5_dtype)
        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(
            device=self.vae.device, dtype=t5_dtype,
        )
        return prompt_embeds, pooled_prompt_embeds, text_ids

    # -----------------------------------------------------------------------
    # Latent preparation
    # -----------------------------------------------------------------------

    @staticmethod
    def _prepare_latent_image_ids(
        batch_size: int, height: int, width: int,
        device: torch.device, dtype: torch.dtype,
    ) -> torch.Tensor:
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]
        latent_image_ids = latent_image_ids.reshape(height * width, 3)
        return latent_image_ids.to(device=device, dtype=dtype)

    @staticmethod
    def _pack_latents(
        latents: torch.Tensor, batch_size: int,
        num_channels: int, height: int, width: int,
    ) -> torch.Tensor:
        latents = latents.view(batch_size, num_channels, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels * 4)
        return latents

    @staticmethod
    def _unpack_latents(
        latents: torch.Tensor, height: int, width: int, vae_scale_factor: int,
    ) -> torch.Tensor:
        batch_size, num_patches, channels = latents.shape
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))
        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(batch_size, channels // (2 * 2), height, width)
        return latents

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator | list[torch.Generator] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = 2 * (int(height) // (self.vae_scale_factor * 2))
        w = 2 * (int(width) // (self.vae_scale_factor * 2))
        shape = (batch_size, num_channels_latents, h, w)
        latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        latents = self._pack_latents(latents, batch_size, num_channels_latents, h, w)
        latent_image_ids = self._prepare_latent_image_ids(
            batch_size, h // 2, w // 2, device, dtype,
        )
        return latents, latent_image_ids

    def prepare_timesteps(
        self, num_inference_steps: int, sigmas: list[float] | None, image_seq_len: int,
    ) -> tuple[torch.Tensor, int]:
        sigmas_arr = (
            np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            if sigmas is None
            else sigmas
        )
        mu = _calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = _retrieve_timesteps(
            self.scheduler, num_inference_steps, sigmas=sigmas_arr, mu=mu,
        )
        return timesteps, num_inference_steps

    # -----------------------------------------------------------------------
    # Denoising
    # -----------------------------------------------------------------------

    def diffuse(
        self,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        latents: torch.Tensor,
        latent_image_ids: torch.Tensor,
        text_ids: torch.Tensor,
        timesteps: torch.Tensor,
        guidance: torch.Tensor | None,
    ) -> torch.Tensor:
        self.scheduler.set_begin_index(0)
        for t in timesteps:
            timestep = t.expand(latents.shape[0]).to(
                device=latents.device, dtype=latents.dtype,
            )
            noise_pred = self.transformer(
                hidden_states=latents,
                timestep=timestep / 1000,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                joint_attention_kwargs={},
                return_dict=False,
            )
            if isinstance(noise_pred, tuple):
                noise_pred = noise_pred[0]
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        return latents

    # -----------------------------------------------------------------------
    # Forward (full pipeline)
    # -----------------------------------------------------------------------

    @torch.inference_mode()
    def forward(
        self,
        prompts: str | list[str],
        params: DiffusionSamplingParams | None = None,
        generator: torch.Generator | list[torch.Generator] | None = None,
    ) -> DiffusionOutput:
        params = params or DiffusionSamplingParams()
        if isinstance(prompts, str):
            prompts = [prompts]

        height = params.height or self.default_sample_size * self.vae_scale_factor
        width = params.width or self.default_sample_size * self.vae_scale_factor
        batch_size = len(prompts)
        num_images = params.num_outputs_per_prompt
        device = self.vae.device

        prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompt(
            prompt=prompts,
            num_images_per_prompt=num_images,
            max_sequence_length=params.max_sequence_length,
        )

        num_channels_latents = self.transformer.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images, num_channels_latents,
            height, width,
            prompt_embeds.dtype, device, generator,
        )

        timesteps, _ = self.prepare_timesteps(
            params.num_inference_steps, params.sigmas, latents.shape[1],
        )

        if self.transformer.guidance_embeds:
            guidance = torch.full(
                [1], params.guidance_scale, dtype=prompt_embeds.dtype, device=device,
            ).expand(latents.shape[0])
        else:
            guidance = None

        latents = self.diffuse(
            prompt_embeds, pooled_prompt_embeds,
            latents, latent_image_ids, text_ids,
            timesteps, guidance,
        )

        if params.output_type == "latent":
            return DiffusionOutput(latents=latents)

        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        latents = latents.to(dtype=self.vae.dtype)
        images = self.vae.decode(latents, return_dict=False)[0]

        image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        pil_images = image_processor.postprocess(images)

        return DiffusionOutput(images=pil_images, latents=latents)
