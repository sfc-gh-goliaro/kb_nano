from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}),
        triton.Config({"BLOCK_SIZE": 512}),
        triton.Config({"BLOCK_SIZE": 1024}),
        triton.Config({"BLOCK_SIZE": 2048}),
    ],
    key=["n_elements"],
)
@triton.jit
def _fused_ddim_update_kernel(
    x_ptr,
    v_ptr,
    out_ptr,
    coeff_x,
    coeff_v,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x_val = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    v_val = tl.load(v_ptr + offsets, mask=mask).to(tl.float32)
    result = x_val * coeff_x + v_val * coeff_v
    tl.store(out_ptr + offsets, result, mask=mask)


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
        alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (
            v_end - v_start
        )
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, clamp_min, 0.999)

    def encode_prompt(
        self, vae, prompt: torch.Tensor, *, dtype: torch.dtype
    ) -> torch.Tensor:
        bsz, frames, channels, height, width = prompt.shape
        prompt = prompt.reshape(bsz * frames, channels, height, width)
        with torch.inference_mode(), self._autocast(prompt.device, dtype):
            posterior = vae.encode(prompt * 2 - 1)
            latents = posterior.mean * self.scaling_factor
        h = height // vae.patch_size
        w = width // vae.patch_size
        return latents.reshape(bsz, frames, h, w, latents.shape[-1]).permute(
            0, 1, 4, 2, 3
        )

    def decode_latents(self, vae, latents: torch.Tensor) -> torch.Tensor:
        bsz, frames, channels, height, width = latents.shape
        target_dtype = vae.post_quant_conv.weight.dtype
        latents = (
            latents.permute(0, 1, 3, 4, 2)
            .reshape(bsz * frames, height * width, channels)
            .to(target_dtype)
        )
        with torch.inference_mode():
            decoded = (vae.decode(latents / self.scaling_factor) + 1) / 2
        return decoded.reshape(
            bsz, frames, decoded.shape[1], decoded.shape[2], decoded.shape[3]
        )

    @torch.no_grad()
    def forward(
        self,
        model,
        vae,
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
        prompt_latents = self.encode_prompt(vae, prompt, dtype=dtype)[
            :, :n_prompt_frames
        ]

        bsz = prompt.shape[0]
        latent_c = prompt_latents.shape[2]
        latent_h = prompt_latents.shape[3]
        latent_w = prompt_latents.shape[4]

        x = torch.empty(
            bsz,
            num_frames,
            latent_c,
            latent_h,
            latent_w,
            device=device,
            dtype=prompt_latents.dtype,
        )
        x[:, :n_prompt_frames] = prompt_latents

        noise_range = torch.linspace(
            -1, self.max_noise_level - 1, ddim_steps + 1
        )

        betas = self.sigmoid_beta_schedule(self.max_noise_level).float().to(device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        ddim_step_data = []
        for noise_idx in reversed(range(1, ddim_steps + 1)):
            t_val = int(noise_range[noise_idx].long().item())
            t_next_raw = int(noise_range[noise_idx - 1].long().item())
            t_next_val = t_val if t_next_raw < 0 else t_next_raw

            a = float(alphas_cumprod[t_val].item())
            a_next = (
                1.0
                if noise_idx == 1
                else float(alphas_cumprod[t_next_val].item())
            )

            sa = a**0.5
            s1a = (1.0 - a) ** 0.5
            san = a_next**0.5
            s1an = (1.0 - a_next) ** 0.5

            ddim_step_data.append(
                (
                    t_val,
                    float(sa * san + s1a * s1an),
                    float(sa * s1an - san * s1a),
                )
            )

        stab_level = self.stabilization_level - 1
        generator = torch.Generator(device=device).manual_seed(
            seed if seed is not None else 0
        )
        total_last_numel = bsz * latent_c * latent_h * latent_w

        for index in range(n_prompt_frames, num_frames):
            chunk = torch.randn(
                bsz,
                1,
                latent_c,
                latent_h,
                latent_w,
                generator=generator,
                device=device,
            )
            chunk.clamp_(-self.noise_abs_max, self.noise_abs_max)
            x[:, index : index + 1] = chunk

            start_frame = max(0, index + 1 - model.max_frames)
            window = index + 1 - start_frame

            t_curr = torch.full(
                (bsz, window), stab_level, dtype=torch.long, device=device
            )

            for t_val, coeff_x, coeff_v in ddim_step_data:
                t_curr[:, -1] = t_val

                x_curr = x[:, start_frame : index + 1].contiguous()

                with torch.inference_mode(), self._autocast(device, dtype):
                    v = model(
                        x_curr, t_curr, actions[:, start_frame : index + 1]
                    )

                x_last = x_curr[:, -1].contiguous()
                v_last = v[:, -1].contiguous()

                out = torch.empty(
                    bsz,
                    latent_c,
                    latent_h,
                    latent_w,
                    device=device,
                    dtype=torch.float32,
                )
                grid = ((total_last_numel + 2047) // 2048,)
                _fused_ddim_update_kernel[grid](
                    x_last,
                    v_last,
                    out,
                    coeff_x,
                    coeff_v,
                    total_last_numel,
                )
                x[:, index] = out

        video = self.decode_latents(vae, x)
        return video, x, prompt_latents
