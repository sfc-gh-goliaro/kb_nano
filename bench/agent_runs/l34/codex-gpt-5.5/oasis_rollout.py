from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn as nn


class OasisRollout(nn.Module):
    def __init__(
        self,
        scaling_factor: float,
        max_noise_level: int,
        stabilization_level: int,
        noise_abs_max: float,
    ):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.max_noise_level = max_noise_level
        self.stabilization_level = stabilization_level
        self.noise_abs_max = noise_abs_max
        self._alpha_cache: dict[tuple[str, torch.dtype, int], torch.Tensor] = {}

    @staticmethod
    def _autocast(device: torch.device, dtype: torch.dtype):
        if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
            return torch.autocast("cuda", dtype=dtype)
        return nullcontext()

    @staticmethod
    def sigmoid_beta_schedule(
        timesteps: int,
        start: float = -3,
        end: float = 3,
        tau: float = 1,
        clamp_min: float = 0.0,
    ) -> torch.Tensor:
        steps = timesteps + 1
        t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
        v_start = torch.tensor(start / tau).sigmoid()
        v_end = torch.tensor(end / tau).sigmoid()
        alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, clamp_min, 0.999)

    def _alphas_cumprod(self, device: torch.device) -> torch.Tensor:
        key = (device.type if device.index is None else f"{device.type}:{device.index}", torch.float32, self.max_noise_level)
        cached = self._alpha_cache.get(key)
        if cached is not None and cached.device == device:
            return cached
        betas = self.sigmoid_beta_schedule(self.max_noise_level).float().to(device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0).reshape(-1, 1, 1, 1)
        self._alpha_cache[key] = alphas_cumprod
        return alphas_cumprod

    def encode_prompt(self, vae: OasisAutoencoderKL, prompt: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        bsz, frames, channels, height, width = prompt.shape
        prompt = prompt.reshape(bsz * frames, channels, height, width)
        with torch.inference_mode(), self._autocast(prompt.device, dtype):
            posterior = vae.encode(prompt * 2 - 1)
            latents = posterior.mean * self.scaling_factor
        h = height // vae.patch_size
        w = width // vae.patch_size
        return latents.reshape(bsz, frames, h, w, latents.shape[-1]).permute(0, 1, 4, 2, 3)

    def decode_latents(self, vae: OasisAutoencoderKL, latents: torch.Tensor) -> torch.Tensor:
        bsz, frames, channels, height, width = latents.shape
        target_dtype = vae.post_quant_conv.weight.dtype
        latents = latents.permute(0, 1, 3, 4, 2).reshape(bsz * frames, height * width, channels).to(target_dtype)
        with torch.inference_mode():
            decoded = (vae.decode(latents / self.scaling_factor) + 1) / 2
        return decoded.reshape(bsz, frames, decoded.shape[1], decoded.shape[2], decoded.shape[3])

    def forward(
        self,
        model: OasisDiT,
        vae: OasisAutoencoderKL,
        prompt: torch.Tensor,
        actions: torch.Tensor,
        *,
        num_frames: int,
        ddim_steps: int,
        n_prompt_frames: int,
        seed: int | None,
        dtype: torch.dtype = torch.float16,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = prompt.device
        batch = prompt.shape[0]

        prompt_latents = self.encode_prompt(vae, prompt, dtype=dtype)[:, :n_prompt_frames]
        if num_frames <= n_prompt_frames:
            x = prompt_latents
            video = self.decode_latents(vae, x)
            return video, x, prompt_latents

        latent_shape = prompt_latents.shape[-3:]
        x_dtype = torch.promote_types(prompt_latents.dtype, torch.get_default_dtype())
        x = torch.empty((batch, num_frames, *latent_shape), device=device, dtype=x_dtype)
        if n_prompt_frames:
            x[:, :n_prompt_frames].copy_(prompt_latents)

        alphas_cumprod = self._alphas_cumprod(device)
        noise_levels = torch.linspace(-1, self.max_noise_level - 1, ddim_steps + 1, device=device).to(torch.long)
        generator = torch.Generator(device=device).manual_seed(seed if seed is not None else 0)
        stab = self.stabilization_level - 1

        for index in range(n_prompt_frames, num_frames):
            chunk = torch.randn((batch, 1, *latent_shape), generator=generator, device=device)
            chunk.clamp_(-self.noise_abs_max, self.noise_abs_max)
            x[:, index : index + 1].copy_(chunk)
            start_frame = max(0, index + 1 - model.max_frames)
            curr_frames = index + 1 - start_frame
            actions_curr = actions[:, start_frame : index + 1]

            x_curr = torch.empty((batch, curr_frames, *latent_shape), device=device, dtype=x.dtype)
            t_curr = torch.empty((batch, curr_frames), dtype=torch.long, device=device)
            if curr_frames > 1:
                t_curr[:, :-1].fill_(stab)

            for noise_idx in reversed(range(1, ddim_steps + 1)):
                t_level = noise_levels[noise_idx]
                next_level = noise_levels[noise_idx - 1]
                t_curr[:, -1] = t_level
                next_level = torch.where(next_level < 0, t_level, next_level)

                x_curr.copy_(x[:, start_frame : index + 1])
                with torch.inference_mode(), self._autocast(device, dtype):
                    v = model(x_curr, t_curr, actions_curr)

                x_last = x_curr[:, -1:]
                v_last = v[:, -1:]
                alpha = alphas_cumprod[t_level]

                x_start = alpha.sqrt() * x_last - (1 - alpha).sqrt() * v_last
                x_noise = ((1 / alpha).sqrt() * x_last - x_start) / (1 / alpha - 1).sqrt()

                if noise_idx == 1:
                    x[:, index : index + 1].copy_(x_start)
                else:
                    alpha_next = alphas_cumprod[next_level]
                    x_pred = alpha_next.sqrt() * x_start + x_noise * (1 - alpha_next).sqrt()
                    x[:, index : index + 1].copy_(x_pred)

        video = self.decode_latents(vae, x)
        return video, x, prompt_latents
