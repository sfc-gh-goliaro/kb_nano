"""Oasis 500M world model and rollout pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.nn as nn
from einops import rearrange
from safetensors.torch import load_file

from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from ..L1.oasis_rotary import OasisRotaryEmbedding
from ..L1.silu import SiLU
from ..L2.oasis_final_layer import OasisFinalLayer
from ..L2.oasis_patch_embed import OasisPatchEmbed
from ..L2.oasis_timestep_embedder import OasisTimestepEmbedder
from ..L3.oasis_block import SpatioTemporalDiTBlock
from ..L3.oasis_vae_attention_block import OasisVAEAttentionBlock


def sigmoid_beta_schedule(timesteps: int, start: float = -3, end: float = 3, tau: float = 1, clamp_min: float = 1e-5):
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, clamp_min, 0.999)


class DiagonalGaussianDistribution:
    def __init__(self, parameters: torch.Tensor, deterministic: bool = False, dim: int = 1):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=dim)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean, device=self.parameters.device)

    def sample(self) -> torch.Tensor:
        return self.mean + self.std * torch.randn(self.mean.shape, device=self.parameters.device)

    def mode(self) -> torch.Tensor:
        return self.mean


class AutoencoderKL(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        *,
        input_height: int = 360,
        input_width: int = 640,
        patch_size: int = 20,
        enc_dim: int = 1024,
        enc_depth: int = 6,
        enc_heads: int = 16,
        dec_dim: int = 1024,
        dec_depth: int = 12,
        dec_heads: int = 16,
        mlp_ratio: float = 4.0,
        use_variational: bool = True,
    ):
        super().__init__()
        self.input_height = input_height
        self.input_width = input_width
        self.patch_size = patch_size
        self.seq_h = input_height // patch_size
        self.seq_w = input_width // patch_size
        self.seq_len = self.seq_h * self.seq_w
        self.patch_dim = 3 * patch_size ** 2
        self.latent_dim = latent_dim
        self.use_variational = use_variational

        self.patch_embed = OasisPatchEmbed(input_height, input_width, patch_size, 3, enc_dim)
        self.encoder = nn.ModuleList(
            [
                OasisVAEAttentionBlock(
                    enc_dim,
                    enc_heads,
                    self.seq_h,
                    self.seq_w,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                )
                for _ in range(enc_depth)
            ]
        )
        self.enc_norm = LayerNorm(enc_dim, eps=1e-6)

        mult = 2 if self.use_variational else 1
        self.quant_conv = Linear(enc_dim, mult * latent_dim, bias=True)
        self.post_quant_conv = Linear(latent_dim, dec_dim, bias=True)

        self.decoder = nn.ModuleList(
            [
                OasisVAEAttentionBlock(
                    dec_dim,
                    dec_heads,
                    self.seq_h,
                    self.seq_w,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                )
                for _ in range(dec_depth)
            ]
        )
        self.dec_norm = LayerNorm(dec_dim, eps=1e-6)
        self.predictor = Linear(dec_dim, self.patch_dim, bias=True)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        def _init_weights(module):
            if isinstance(module, Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, LayerNorm):
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1.0)

        self.apply(_init_weights)
        weight = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(weight.view(weight.shape[0], -1))

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        x = x.reshape(bsz, self.seq_h, self.seq_w, self.patch_dim).permute(0, 3, 1, 2)
        x = x.reshape(bsz, 3, self.patch_size, self.patch_size, self.seq_h, self.seq_w)
        x = x.permute(0, 1, 4, 2, 5, 3)
        return x.reshape(bsz, 3, self.input_height, self.input_width)

    def encode(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        x = self.patch_embed(x)
        for block in self.encoder:
            x = block(x)
        x = self.enc_norm(x)
        moments = self.quant_conv(x)
        if not self.use_variational:
            moments = torch.cat((moments, torch.zeros_like(moments)), dim=2)
        return DiagonalGaussianDistribution(moments, deterministic=not self.use_variational, dim=2)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = self.post_quant_conv(z)
        for block in self.decoder:
            z = block(z)
        z = self.dec_norm(z)
        z = self.predictor(z)
        return self.unpatchify(z)


class DiT(nn.Module):
    def __init__(
        self,
        *,
        input_h: int = 18,
        input_w: int = 32,
        patch_size: int = 2,
        in_channels: int = 16,
        hidden_size: int = 1024,
        depth: int = 16,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        external_cond_dim: int = 25,
        max_frames: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.max_frames = max_frames

        self.x_embedder = OasisPatchEmbed(input_h, input_w, patch_size, in_channels, hidden_size, flatten=False)
        self.t_embedder = OasisTimestepEmbedder(hidden_size)
        head_dim = hidden_size // num_heads
        self.spatial_rotary_emb = OasisRotaryEmbedding(dim=head_dim // 2, freqs_for="pixel", max_freq=256)
        self.temporal_rotary_emb = OasisRotaryEmbedding(dim=head_dim, freqs_for="lang")
        self.external_cond = Linear(external_cond_dim, hidden_size, bias=True) if external_cond_dim > 0 else nn.Identity()
        self.blocks = nn.ModuleList(
            [
                SpatioTemporalDiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    is_causal=True,
                    spatial_rotary_emb=self.spatial_rotary_emb,
                    temporal_rotary_emb=self.temporal_rotary_emb,
                )
                for _ in range(depth)
            ]
        )
        self.final_layer = OasisFinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        def _basic_init(module):
            if isinstance(module, Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        weight = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(weight.view(weight.shape[0], -1))
        if self.x_embedder.proj.bias is not None:
            nn.init.constant_(self.x_embedder.proj.bias, 0)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        for block in self.blocks:
            nn.init.constant_(block.s_adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.s_adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.t_adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.t_adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        if self.final_layer.linear.bias is not None:
            nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = x.shape[1]
        w = x.shape[2]
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], c, h * p, w * p)

    def forward(self, x: torch.Tensor, t: torch.Tensor, external_cond: torch.Tensor | None = None) -> torch.Tensor:
        _, time, _, _, _ = x.shape
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.x_embedder(x)
        x = rearrange(x, "(b t) h w d -> b t h w d", t=time)
        t = rearrange(t, "b t -> (b t)")
        c = self.t_embedder(t)
        c = rearrange(c, "(b t) d -> b t d", t=time)
        if torch.is_tensor(external_cond):
            c = c + self.external_cond(external_cond)
        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x, c)
        x = rearrange(x, "b t h w d -> (b t) h w d")
        x = self.unpatchify(x)
        return rearrange(x, "(b t) c h w -> b t c h w", t=time)


def DiT_S_2() -> DiT:
    return DiT(patch_size=2, hidden_size=1024, depth=16, num_heads=16)


def ViT_L_20_Shallow_Encoder() -> AutoencoderKL:
    return AutoencoderKL(
        latent_dim=16,
        patch_size=20,
        enc_dim=1024,
        enc_depth=6,
        enc_heads=16,
        dec_dim=1024,
        dec_depth=12,
        dec_heads=16,
        input_height=360,
        input_width=640,
    )


@dataclass
class OasisConfig:
    dit_variant: str = "DiT-S/2"
    vae_variant: str = "vit-l-20-shallow-encoder"
    scaling_factor: float = 0.07843137255
    max_noise_level: int = 1000
    max_frames: int = 32
    stabilization_level: int = 15
    noise_abs_max: float = 20.0


@dataclass
class OasisSamplingParams:
    num_frames: int = 8
    ddim_steps: int = 4
    n_prompt_frames: int = 1
    video_offset: int | None = None
    seed: int | None = None
    output_type: str = "video"


@dataclass
class OasisOutput:
    video: torch.Tensor
    latents: torch.Tensor
    prompt_latents: torch.Tensor


class OasisPipeline(nn.Module):
    def __init__(self, config: OasisConfig):
        super().__init__()
        self.config = config
        self.model = DiT_S_2()
        self.vae = ViT_L_20_Shallow_Encoder()

    def load_weights(self, model_dir: str) -> None:
        dit_path = os.path.join(model_dir, "oasis500m.safetensors")
        vae_path = os.path.join(model_dir, "vit-l-20.safetensors")
        self.model.load_state_dict(load_file(dit_path), strict=False)
        self.vae.load_state_dict(load_file(vae_path), strict=False)

    def encode_prompt(self, prompt: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        bsz, frames, _, height, width = prompt.shape
        prompt = rearrange(prompt, "b t c h w -> (b t) c h w")
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
            posterior = self.vae.encode(prompt * 2 - 1)
            latents = posterior.mean * self.config.scaling_factor
        return rearrange(
            latents,
            "(b t) (h w) c -> b t c h w",
            b=bsz,
            t=frames,
            h=height // self.vae.patch_size,
            w=width // self.vae.patch_size,
        )

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        bsz, frames, _, _, _ = latents.shape
        target_dtype = self.vae.post_quant_conv.weight.dtype
        latents = rearrange(latents, "b t c h w -> (b t) (h w) c").to(target_dtype)
        with torch.inference_mode():
            decoded = (self.vae.decode(latents / self.config.scaling_factor) + 1) / 2
        return rearrange(decoded, "(b t) c h w -> b t c h w", b=bsz, t=frames)

    def rollout(
        self,
        prompt: torch.Tensor,
        actions: torch.Tensor,
        params: OasisSamplingParams,
        *,
        dtype: torch.dtype = torch.float16,
    ) -> OasisOutput:
        device = prompt.device
        prompt_latents = self.encode_prompt(prompt, dtype=dtype)[:, :params.n_prompt_frames]
        x = prompt_latents
        total_frames = params.num_frames
        max_noise_level = self.config.max_noise_level
        noise_range = torch.linspace(-1, max_noise_level - 1, params.ddim_steps + 1, device=device)

        betas = sigmoid_beta_schedule(max_noise_level).float().to(device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod = rearrange(alphas_cumprod, "t -> t 1 1 1")

        seed = params.seed if params.seed is not None else 0
        generator = torch.Generator(device=device).manual_seed(seed)

        for index in range(params.n_prompt_frames, total_frames):
            chunk = torch.randn((prompt.shape[0], 1, *x.shape[-3:]), generator=generator, device=device)
            chunk = torch.clamp(chunk, -self.config.noise_abs_max, self.config.noise_abs_max)
            x = torch.cat([x, chunk], dim=1)
            start_frame = max(0, index + 1 - self.model.max_frames)

            for noise_idx in reversed(range(1, params.ddim_steps + 1)):
                t_ctx = torch.full(
                    (prompt.shape[0], index),
                    self.config.stabilization_level - 1,
                    dtype=torch.long,
                    device=device,
                )
                t = torch.full((prompt.shape[0], 1), noise_range[noise_idx], dtype=torch.long, device=device)
                t_next = torch.full((prompt.shape[0], 1), noise_range[noise_idx - 1], dtype=torch.long, device=device)
                t_next = torch.where(t_next < 0, t, t_next)
                t = torch.cat([t_ctx, t], dim=1)
                t_next = torch.cat([t_ctx, t_next], dim=1)

                x_curr = x[:, start_frame:].clone()
                t_curr = t[:, start_frame:]
                t_next_curr = t_next[:, start_frame:]

                with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
                    v = self.model(x_curr, t_curr, actions[:, start_frame:index + 1])

                x_start = alphas_cumprod[t_curr].sqrt() * x_curr - (1 - alphas_cumprod[t_curr]).sqrt() * v
                x_noise = ((1 / alphas_cumprod[t_curr]).sqrt() * x_curr - x_start) / (
                    1 / alphas_cumprod[t_curr] - 1
                ).sqrt()

                alpha_next = alphas_cumprod[t_next_curr]
                alpha_next[:, :-1] = torch.ones_like(alpha_next[:, :-1])
                if noise_idx == 1:
                    alpha_next[:, -1:] = torch.ones_like(alpha_next[:, -1:])
                x_pred = alpha_next.sqrt() * x_start + x_noise * (1 - alpha_next).sqrt()
                x[:, -1:] = x_pred[:, -1:]

        video = self.decode_latents(x)
        return OasisOutput(video=video, latents=x, prompt_latents=prompt_latents)
