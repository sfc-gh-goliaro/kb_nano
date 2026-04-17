"""Native InstantNGP renderer backed by repo field kernels and nerfacc traversal."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
_cuda_bin = "/usr/local/cuda/bin"
if _cuda_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"{_cuda_bin}:{os.environ.get('PATH', '')}"

import nerfacc
import torch
import torch.nn as nn

from infra.nerf_loader import InstantNGPSnapshotAssets
from tasks.baseline.L2.instantngp_field import InstantNGPField


NERF_GRID_SIZE = 128
NERF_STEPS = 1024
NERF_MIN_OPTICAL_THICKNESS = 0.01
SQRT3 = 1.73205080757
MIN_CONE_STEPSIZE = SQRT3 / NERF_STEPS
MAX_DEPTH = 16384.0

_MORTON_TO_DENSE_128: torch.Tensor | None = None
_DENSE_TO_MORTON_128: torch.Tensor | None = None

_SOBOL_DIRS_2D = (
    (
        0x80000000, 0x40000000, 0x20000000, 0x10000000,
        0x08000000, 0x04000000, 0x02000000, 0x01000000,
        0x00800000, 0x00400000, 0x00200000, 0x00100000,
        0x00080000, 0x00040000, 0x00020000, 0x00010000,
        0x00008000, 0x00004000, 0x00002000, 0x00001000,
        0x00000800, 0x00000400, 0x00000200, 0x00000100,
        0x00000080, 0x00000040, 0x00000020, 0x00000010,
        0x00000008, 0x00000004, 0x00000002, 0x00000001,
    ),
    (
        0x80000000, 0xC0000000, 0xA0000000, 0xF0000000,
        0x88000000, 0xCC000000, 0xAA000000, 0xFF000000,
        0x80800000, 0xC0C00000, 0xA0A00000, 0xF0F00000,
        0x88880000, 0xCCCC0000, 0xAAAA0000, 0xFFFF0000,
        0x80008000, 0xC000C000, 0xA000A000, 0xF000F000,
        0x88008800, 0xCC00CC00, 0xAA00AA00, 0xFF00FF00,
        0x80808080, 0xC0C0C0C0, 0xA0A0A0A0, 0xF0F0F0F0,
        0x88888888, 0xCCCCCCCC, 0xAAAAAAAA, 0xFFFFFFFF,
    ),
)


@dataclass(frozen=True)
class InstantNGPRenderOutput:
    rgba: torch.Tensor


@dataclass(frozen=True)
class InstantNGPExactSampleCache:
    ray_origins: torch.Tensor
    ray_directions: torch.Tensor
    ray_indices: torch.Tensor
    t_starts: torch.Tensor
    t_ends: torch.Tensor
    n_rays: int


def _opencv_lens_distortion_delta(params: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    k1, k2, p1, p2 = [params[i] for i in range(4)]
    u = uv[..., 0]
    v = uv[..., 1]
    u2 = u * u
    uv_prod = u * v
    v2 = v * v
    r2 = u2 + v2
    radial = k1 * r2 + k2 * r2 * r2
    du = u * radial + 2.0 * p1 * uv_prod + p2 * (r2 + 2.0 * u2)
    dv = v * radial + 2.0 * p2 * uv_prod + p1 * (r2 + 2.0 * v2)
    return torch.stack([du, dv], dim=-1)


def _iterative_opencv_lens_undistortion(
    params: torch.Tensor,
    uv: torch.Tensor,
    iterations: int = 100,
) -> torch.Tensor:
    if torch.max(torch.abs(params)).item() == 0.0:
        return uv

    x0 = uv
    x = uv.clone()
    eps = torch.finfo(x.dtype).eps
    for _ in range(iterations):
        step0 = torch.clamp(1e-6 * torch.abs(x[..., 0]), min=eps)
        step1 = torch.clamp(1e-6 * torch.abs(x[..., 1]), min=eps)

        dx = _opencv_lens_distortion_delta(params, x)
        dx_0b = _opencv_lens_distortion_delta(
            params, torch.stack([x[..., 0] - step0, x[..., 1]], dim=-1)
        )
        dx_0f = _opencv_lens_distortion_delta(
            params, torch.stack([x[..., 0] + step0, x[..., 1]], dim=-1)
        )
        dx_1b = _opencv_lens_distortion_delta(
            params, torch.stack([x[..., 0], x[..., 1] - step1], dim=-1)
        )
        dx_1f = _opencv_lens_distortion_delta(
            params, torch.stack([x[..., 0], x[..., 1] + step1], dim=-1)
        )

        j00 = 1.0 + (dx_0f[..., 0] - dx_0b[..., 0]) / (2.0 * step0)
        j10 = (dx_1f[..., 0] - dx_1b[..., 0]) / (2.0 * step1)
        j01 = (dx_0f[..., 1] - dx_0b[..., 1]) / (2.0 * step0)
        j11 = 1.0 + (dx_1f[..., 1] - dx_1b[..., 1]) / (2.0 * step1)

        residual = x + dx - x0
        det = j00 * j11 - j01 * j10
        det = torch.where(det.abs() < eps, torch.full_like(det, eps), det)
        step_x = torch.stack(
            [
                (j11 * residual[..., 0] - j10 * residual[..., 1]) / det,
                (-j01 * residual[..., 0] + j00 * residual[..., 1]) / det,
            ],
            dim=-1,
        )
        x = x - step_x
    return x


def _srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    threshold = 0.04045
    return torch.where(
        x <= threshold,
        x / 12.92,
        torch.pow((torch.clamp(x, min=0.0) + 0.055) / 1.055, 2.4),
    )


def _fractf(x: float) -> float:
    return x - math.floor(x)


def _reverse_bits(x: int) -> int:
    x &= 0xFFFFFFFF
    x = (((x & 0xAAAAAAAA) >> 1) | ((x & 0x55555555) << 1)) & 0xFFFFFFFF
    x = (((x & 0xCCCCCCCC) >> 2) | ((x & 0x33333333) << 2)) & 0xFFFFFFFF
    x = (((x & 0xF0F0F0F0) >> 4) | ((x & 0x0F0F0F0F) << 4)) & 0xFFFFFFFF
    x = (((x & 0xFF00FF00) >> 8) | ((x & 0x00FF00FF) << 8)) & 0xFFFFFFFF
    return ((x >> 16) | (x << 16)) & 0xFFFFFFFF


def _laine_karras_permutation(x: int, seed: int) -> int:
    x = (x + seed) & 0xFFFFFFFF
    x ^= (x * 0x6C50B47C) & 0xFFFFFFFF
    x &= 0xFFFFFFFF
    x ^= (x * 0xB82F1E52) & 0xFFFFFFFF
    x &= 0xFFFFFFFF
    x ^= (x * 0xC7AFE638) & 0xFFFFFFFF
    x &= 0xFFFFFFFF
    x ^= (x * 0x8D22F6E6) & 0xFFFFFFFF
    return x & 0xFFFFFFFF


def _nested_uniform_scramble_base2(x: int, seed: int) -> int:
    x = _reverse_bits(x)
    x = _laine_karras_permutation(x, seed & 0xFFFFFFFF)
    x = _reverse_bits(x)
    return x & 0xFFFFFFFF


def _sobol(index: int, dim: int) -> int:
    x = 0
    dirs = _SOBOL_DIRS_2D[dim]
    for bit in range(32):
        mask = (index >> bit) & 1
        x ^= mask * dirs[bit]
    return x & 0xFFFFFFFF


def _hash_combine(seed: int, value: int) -> int:
    seed &= 0xFFFFFFFF
    value &= 0xFFFFFFFF
    return (seed ^ (value + ((seed << 6) & 0xFFFFFFFF) + (seed >> 2))) & 0xFFFFFFFF


def _reverse_bits_tensor(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.int64) & 0xFFFFFFFF
    x = (((x & 0xAAAAAAAA) >> 1) | ((x & 0x55555555) << 1)) & 0xFFFFFFFF
    x = (((x & 0xCCCCCCCC) >> 2) | ((x & 0x33333333) << 2)) & 0xFFFFFFFF
    x = (((x & 0xF0F0F0F0) >> 4) | ((x & 0x0F0F0F0F) << 4)) & 0xFFFFFFFF
    x = (((x & 0xFF00FF00) >> 8) | ((x & 0x00FF00FF) << 8)) & 0xFFFFFFFF
    return (((x >> 16) | (x << 16)) & 0xFFFFFFFF).to(torch.int64)


def _laine_karras_permutation_tensor(x: torch.Tensor, seed: torch.Tensor) -> torch.Tensor:
    x = (x.to(torch.int64) + seed.to(torch.int64)) & 0xFFFFFFFF
    x = (x ^ ((x * 0x6C50B47C) & 0xFFFFFFFF)) & 0xFFFFFFFF
    x = (x ^ ((x * 0xB82F1E52) & 0xFFFFFFFF)) & 0xFFFFFFFF
    x = (x ^ ((x * 0xC7AFE638) & 0xFFFFFFFF)) & 0xFFFFFFFF
    x = (x ^ ((x * 0x8D22F6E6) & 0xFFFFFFFF)) & 0xFFFFFFFF
    return x


def _nested_uniform_scramble_base2_tensor(x: torch.Tensor, seed: torch.Tensor) -> torch.Tensor:
    x = _reverse_bits_tensor(x)
    x = _laine_karras_permutation_tensor(x, seed & 0xFFFFFFFF)
    x = _reverse_bits_tensor(x)
    return x & 0xFFFFFFFF


def _hash_combine_tensor(seed: torch.Tensor, value: int) -> torch.Tensor:
    seed = seed.to(torch.int64) & 0xFFFFFFFF
    value_t = torch.full_like(seed, value & 0xFFFFFFFF)
    return (seed ^ (value_t + ((seed << 6) & 0xFFFFFFFF) + (seed >> 2))) & 0xFFFFFFFF


def _ld_random_val_tensor(index: torch.Tensor, seed: torch.Tensor, dim: int = 0) -> torch.Tensor:
    index = _nested_uniform_scramble_base2_tensor(index, seed)
    dirs = torch.tensor(_SOBOL_DIRS_2D[dim], device=index.device, dtype=torch.int64)
    x = torch.zeros_like(index, dtype=torch.int64)
    for bit in range(32):
        mask = (index >> bit) & 1
        x ^= mask * dirs[bit]
    x = _nested_uniform_scramble_base2_tensor(x, _hash_combine_tensor(seed, dim))
    return x.to(torch.float32) * (1.0 / float(1 << 32))


def _ld_random_val_2d(index: int, seed: int) -> tuple[float, float]:
    index = _nested_uniform_scramble_base2(index, seed)
    out = []
    for dim in range(2):
        x = _sobol(index, dim)
        x = _nested_uniform_scramble_base2(x, _hash_combine(seed, dim))
        out.append(float(x) / float(1 << 32))
    return out[0], out[1]


def _ld_random_pixel_offset(spp: int) -> tuple[float, float]:
    x0, y0 = _ld_random_val_2d(0, 0xDEADBEEF)
    x1, y1 = _ld_random_val_2d(spp, 0xDEADBEEF)
    return _fractf(0.5 - x0 + x1), _fractf(0.5 - y0 + y1)


def _morton3d_invert(x: torch.Tensor) -> torch.Tensor:
    x = x & 0x49249249
    x = (x | (x >> 2)) & 0xC30C30C3
    x = (x | (x >> 4)) & 0x0F00F00F
    x = (x | (x >> 8)) & 0xFF0000FF
    x = (x | (x >> 16)) & 0x0000FFFF
    return x


def _morton3d(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    def _expand_bits(v: torch.Tensor) -> torch.Tensor:
        v = v.to(torch.int64) & 0x3FF
        v = (v | (v << 16)) & 0x030000FF
        v = (v | (v << 8)) & 0x0300F00F
        v = (v | (v << 4)) & 0x030C30C3
        v = (v | (v << 2)) & 0x09249249
        return v

    return _expand_bits(x) | (_expand_bits(y) << 1) | (_expand_bits(z) << 2)


def _morton_to_dense_128() -> torch.Tensor:
    global _MORTON_TO_DENSE_128
    if _MORTON_TO_DENSE_128 is None:
        n = NERF_GRID_SIZE ** 3
        morton = torch.arange(n, dtype=torch.int64)
        x = _morton3d_invert(morton >> 0)
        y = _morton3d_invert(morton >> 1)
        z = _morton3d_invert(morton >> 2)
        _MORTON_TO_DENSE_128 = (x + NERF_GRID_SIZE * (y + NERF_GRID_SIZE * z)).to(torch.long)
    return _MORTON_TO_DENSE_128


def _dense_to_morton_128() -> torch.Tensor:
    global _DENSE_TO_MORTON_128
    if _DENSE_TO_MORTON_128 is None:
        morton_to_dense = _morton_to_dense_128()
        dense_to_morton = torch.empty_like(morton_to_dense)
        dense_to_morton[morton_to_dense] = torch.arange(morton_to_dense.numel(), dtype=torch.long)
        _DENSE_TO_MORTON_128 = dense_to_morton
    return _DENSE_TO_MORTON_128


def _density_grid_to_bitfield_bytes(density_grid: torch.Tensor, n_levels: int) -> torch.Tensor:
    n_cells = NERF_GRID_SIZE ** 3
    n_bytes = n_cells // 8
    density_grid = density_grid[: n_levels * n_cells].reshape(n_levels, n_cells).to(torch.float32).cpu()
    mean_density = torch.clamp_min(density_grid[0], 0.0).mean()
    threshold = min(NERF_MIN_OPTICAL_THICKNESS, float(mean_density.item()))

    occ = (density_grid > threshold).reshape(n_levels, n_bytes, 8).to(torch.uint8)
    bit_shifts = (1 << torch.arange(8, dtype=torch.uint8)).view(1, 1, 8)
    bitfield = torch.sum(occ * bit_shifts, dim=-1, dtype=torch.uint8)

    pooled_bytes = n_cells // 64
    for level in range(1, n_levels):
        prev = bitfield[level - 1]
        next_level = bitfield[level].clone()
        pooled = torch.zeros((pooled_bytes,), dtype=torch.uint8)
        prev_groups = prev[: pooled_bytes * 8].reshape(pooled_bytes, 8)
        for bit in range(8):
            pooled |= ((prev_groups[:, bit] > 0).to(torch.uint8) << bit)
        pooled_index = torch.arange(pooled_bytes, dtype=torch.int64)
        x = _morton3d_invert(pooled_index >> 0) + NERF_GRID_SIZE // 8
        y = _morton3d_invert(pooled_index >> 1) + NERF_GRID_SIZE // 8
        z = _morton3d_invert(pooled_index >> 2) + NERF_GRID_SIZE // 8
        next_level[_morton3d(x, y, z).to(torch.long)] = pooled
        bitfield[level] = next_level

    return bitfield


def _bitfield_bytes_to_dense_binaries(bitfield: torch.Tensor) -> torch.Tensor:
    n_levels, n_bytes = bitfield.shape
    n_cells = n_bytes * 8
    morton_occ = ((bitfield.unsqueeze(-1) >> torch.arange(8, dtype=torch.uint8)) & 1).reshape(n_levels, n_cells)
    dense = torch.zeros((n_levels, n_cells), dtype=torch.bool)
    dense[:, _morton_to_dense_128()] = morton_occ.bool()
    return dense.view(n_levels, NERF_GRID_SIZE, NERF_GRID_SIZE, NERF_GRID_SIZE)


class InstantNGPRenderer(nn.Module):
    def __init__(
        self,
        *,
        field: InstantNGPField,
        assets: InstantNGPSnapshotAssets,
        width: int,
        height: int,
        spp: int = 1,
        ray_chunk_size: int = 131072,
        use_exact_marcher: bool = True,
        render_step_scale: float = 0.5,
        fast_alpha_thre: float = 0.0,
        fast_early_stop_eps: float = 0.0,
        fast_use_sigma_pruning: bool = False,
    ):
        super().__init__()
        self.field = field
        self.assets = assets
        self.width = width
        self.height = height
        self.spp = spp
        self.ray_chunk_size = ray_chunk_size
        self.use_exact_marcher = use_exact_marcher
        self.render_step_scale = render_step_scale
        self.fast_alpha_thre = fast_alpha_thre
        self.fast_early_stop_eps = fast_early_stop_eps
        self.fast_use_sigma_pruning = fast_use_sigma_pruning
        self._direction_cache: dict[tuple[int, int, torch.device], torch.Tensor] = {}
        self._exact_sample_cache: dict[tuple[int, int, int], InstantNGPExactSampleCache] = {}

        n_levels = assets.max_cascade + 1
        bitfield = _density_grid_to_bitfield_bytes(assets.density_grid, n_levels)
        binaries = _bitfield_bytes_to_dense_binaries(bitfield)

        base_aabb = torch.cat(
            [
                assets.aabb_min.to(torch.float32),
                assets.aabb_max.to(torch.float32),
            ],
            dim=0,
        )
        estimator = nerfacc.OccGridEstimator(
            roi_aabb=base_aabb,
            resolution=NERF_GRID_SIZE,
            levels=n_levels,
        )
        estimator.binaries.copy_(binaries)
        estimator.occs.copy_(binaries.reshape(-1).to(torch.float32))
        self.estimator = estimator
        self.register_buffer("occupancy_binaries", binaries.clone(), persistent=False)
        self.register_buffer("occupancy_binaries_flat", binaries.reshape(n_levels, -1).clone(), persistent=False)
        self.register_buffer("occupancy_bitfield", bitfield.clone(), persistent=False)

        self.register_buffer("aabb_min", assets.aabb_min.clone(), persistent=False)
        self.register_buffer("aabb_max", assets.aabb_max.clone(), persistent=False)
        self.register_buffer("render_aabb_min", assets.render_aabb_min.clone(), persistent=False)
        self.register_buffer("render_aabb_max", assets.render_aabb_max.clone(), persistent=False)
        self.register_buffer(
            "render_aabb_to_local",
            assets.render_aabb_to_local.clone(),
            persistent=False,
        )
        self.register_buffer(
            "background_color",
            assets.background_color.clone(),
            persistent=False,
        )

    def _get_camera_rays(
        self,
        *,
        width: int,
        height: int,
        focal_length: torch.Tensor,
        principal_point: torch.Tensor,
        lens_params: torch.Tensor,
        sample_index: int,
        device: torch.device,
    ) -> torch.Tensor:
        key = (
            width,
            height,
            device,
            tuple(round(float(x), 6) for x in focal_length.detach().cpu().tolist()),
            tuple(round(float(x), 6) for x in principal_point.detach().cpu().tolist()),
            tuple(round(float(x), 6) for x in lens_params.detach().cpu().tolist()),
            sample_index,
            self.assets.snap_to_pixel_centers,
        )
        cached = self._direction_cache.get(key)
        if cached is not None:
            return cached

        offset_x, offset_y = (
            (0.5, 0.5)
            if self.assets.snap_to_pixel_centers
            else _ld_random_pixel_offset(sample_index)
        )
        xs = (torch.arange(width, device=device, dtype=torch.float32) + float(offset_x)) / float(width)
        ys = (torch.arange(height, device=device, dtype=torch.float32) + float(offset_y)) / float(height)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        uv = torch.stack([grid_x, grid_y], dim=-1)

        original_resolution = torch.tensor([width, height], device=device, dtype=torch.float32)
        dir_xy = torch.empty_like(uv)
        # `pyngp.render()` uses the training-view principal point directly in
        # `uv_to_ray` for this path; using `1 - principal_point` introduces a
        # consistent image-space shift.
        screen_center = principal_point
        dir_xy[..., 0] = (uv[..., 0] - screen_center[0]) * original_resolution[0] / focal_length[0]
        dir_xy[..., 1] = (uv[..., 1] - screen_center[1]) * original_resolution[1] / focal_length[1]
        dir_xy = _iterative_opencv_lens_undistortion(lens_params, dir_xy)

        dirs = torch.stack(
            [dir_xy[..., 0], dir_xy[..., 1], torch.ones_like(dir_xy[..., 0])],
            dim=-1,
        )
        cached = dirs.reshape(-1, 3)
        self._direction_cache[key] = cached
        return cached

    @staticmethod
    def _safe_reciprocal(x: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        x = torch.where(x.abs() < eps, torch.where(x >= 0, eps, -eps), x)
        return 1.0 / x

    @staticmethod
    def _warp_position(pos: torch.Tensor, aabb_min: torch.Tensor, aabb_max: torch.Tensor) -> torch.Tensor:
        return (pos - aabb_min) / (aabb_max - aabb_min)

    @staticmethod
    def _warp_direction(direction: torch.Tensor) -> torch.Tensor:
        return (direction + 1.0) * 0.5

    @staticmethod
    def _to_stepping_space(t: torch.Tensor, cone_angle: float) -> torch.Tensor:
        if cone_angle <= 1e-5:
            return t / MIN_CONE_STEPSIZE

        log1p_c = math.log1p(cone_angle)
        a = (math.log(MIN_CONE_STEPSIZE) - math.log(log1p_c)) / log1p_c
        b = (math.log((MIN_CONE_STEPSIZE * (1 << 7))) - math.log(log1p_c)) / log1p_c
        at = math.exp(a * log1p_c)
        bt = math.exp(b * log1p_c)

        result = torch.empty_like(t)
        mask_a = t <= at
        mask_b = (~mask_a) & (t <= bt)
        result[mask_a] = (t[mask_a] - at) / MIN_CONE_STEPSIZE + a
        result[mask_b] = torch.log(t[mask_b]) / log1p_c
        result[~(mask_a | mask_b)] = (t[~(mask_a | mask_b)] - bt) / (MIN_CONE_STEPSIZE * (1 << 7)) + b
        return result

    @staticmethod
    def _from_stepping_space(n: torch.Tensor, cone_angle: float) -> torch.Tensor:
        if cone_angle <= 1e-5:
            return n * MIN_CONE_STEPSIZE

        log1p_c = math.log1p(cone_angle)
        a = (math.log(MIN_CONE_STEPSIZE) - math.log(log1p_c)) / log1p_c
        b = (math.log((MIN_CONE_STEPSIZE * (1 << 7))) - math.log(log1p_c)) / log1p_c
        at = math.exp(a * log1p_c)
        bt = math.exp(b * log1p_c)

        result = torch.empty_like(n)
        mask_a = n <= a
        mask_b = (~mask_a) & (n <= b)
        result[mask_a] = (n[mask_a] - a) * MIN_CONE_STEPSIZE + at
        result[mask_b] = torch.exp(n[mask_b] * log1p_c)
        result[~(mask_a | mask_b)] = (n[~(mask_a | mask_b)] - b) * (MIN_CONE_STEPSIZE * (1 << 7)) + bt
        return result

    @classmethod
    def _advance_n_steps(cls, t: torch.Tensor, cone_angle: float, n: torch.Tensor | float) -> torch.Tensor:
        if not torch.is_tensor(n):
            n = torch.full_like(t, float(n))
        return cls._from_stepping_space(cls._to_stepping_space(t, cone_angle) + n, cone_angle)

    @classmethod
    def _calc_dt(cls, t: torch.Tensor, cone_angle: float) -> torch.Tensor:
        return cls._advance_n_steps(t, cone_angle, 1.0) - t

    @staticmethod
    def _mip_from_pos(pos: torch.Tensor, max_cascade: int) -> torch.Tensor:
        maxval = torch.amax(torch.abs(pos - 0.5), dim=-1)
        _, exponent = torch.frexp(maxval)
        mip = torch.where(
            maxval > 0,
            exponent.to(torch.int64) + 1,
            torch.zeros_like(maxval, dtype=torch.int64),
        )
        return torch.clamp(mip, 0, max_cascade)

    @staticmethod
    def _advance_to_next_voxel(
        t: torch.Tensor,
        cone_angle: float,
        pos: torch.Tensor,
        direction: torch.Tensor,
        inv_direction: torch.Tensor,
        mip: torch.Tensor,
    ) -> torch.Tensor:
        res = torch.ldexp(
            torch.full_like(t, float(NERF_GRID_SIZE)),
            -mip.to(torch.int32),
        )
        p = res.unsqueeze(-1) * (pos - 0.5)
        sign_dir = torch.sign(direction)
        target = torch.floor(p + 0.5 + 0.5 * sign_dir)
        tx = (target[..., 0] - p[..., 0]) * inv_direction[..., 0]
        ty = (target[..., 1] - p[..., 1]) * inv_direction[..., 1]
        tz = (target[..., 2] - p[..., 2]) * inv_direction[..., 2]
        t_dist = torch.clamp(torch.minimum(torch.minimum(tx, ty), tz) / res, min=0.0)
        t_target = t + t_dist
        t_step = InstantNGPRenderer._to_stepping_space(t, cone_angle)
        t_target_step = InstantNGPRenderer._to_stepping_space(t_target, cone_angle)
        return InstantNGPRenderer._from_stepping_space(
            t_step + torch.ceil(torch.clamp(t_target_step - t_step, min=0.5)),
            cone_angle,
        )

    def _occupancy_at(self, pos: torch.Tensor, mip: torch.Tensor) -> torch.Tensor:
        mip_scale = torch.ldexp(
            torch.ones_like(pos[..., 0], dtype=torch.float32, device=pos.device),
            -mip.to(torch.int32),
        )
        grid_pos = (pos - 0.5) * mip_scale.unsqueeze(-1) + 0.5
        idx = torch.floor(grid_pos * float(NERF_GRID_SIZE)).to(torch.long)
        valid = (
            (idx[..., 0] >= 0)
            & (idx[..., 0] < NERF_GRID_SIZE)
            & (idx[..., 1] >= 0)
            & (idx[..., 1] < NERF_GRID_SIZE)
            & (idx[..., 2] >= 0)
            & (idx[..., 2] < NERF_GRID_SIZE)
        )
        occ = torch.zeros(pos.shape[0], dtype=torch.bool, device=pos.device)
        if valid.any():
            v = valid.nonzero(as_tuple=False).squeeze(-1)
            dense_index = idx[v, 0] + NERF_GRID_SIZE * (idx[v, 1] + NERF_GRID_SIZE * idx[v, 2])
            occ[v] = self.occupancy_binaries_flat[mip[v], dense_index]
        return occ

    def _render_chunk_exact(
        self,
        *,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        ray_t_min: torch.Tensor,
        ray_t_max: torch.Tensor,
        pixel_index_offset: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_rays = ray_origins.shape[0]
        colors = torch.zeros((n_rays, 3), device=device, dtype=torch.float32)
        opacities = torch.zeros((n_rays,), device=device, dtype=torch.float32)
        max_weights = torch.zeros((n_rays,), device=device, dtype=torch.float32)
        best_depth = torch.full((n_rays,), MAX_DEPTH, device=device, dtype=torch.float32)
        t = ray_t_min.clone()
        inv_dir = self._safe_reciprocal(ray_directions)
        active = ray_t_max > ray_t_min
        render_aabb_to_local = self.render_aabb_to_local
        render_aabb_min = self.render_aabb_min
        render_aabb_max = self.render_aabb_max
        aabb_min = self.aabb_min
        aabb_max = self.aabb_max
        max_cascade = self.assets.max_cascade
        cone_angle = self.assets.cone_angle_constant

        sample_index = max(self.spp - 1, 0)
        if active.any():
            ray_ids = torch.arange(pixel_index_offset, pixel_index_offset + n_rays, device=device, dtype=torch.int64)
            sample_ids = torch.full_like(ray_ids, sample_index)
            initial_n = _ld_random_val_tensor(sample_ids, (ray_ids * 786433) & 0xFFFFFFFF)
            t = torch.where(active, self._advance_n_steps(t, cone_angle, initial_n), t)

        for _ in range(1024):
            if not active.any():
                break
            active_idx = active.nonzero(as_tuple=False).squeeze(-1)
            t_active = t[active_idx]
            pos = ray_origins[active_idx] + ray_directions[active_idx] * t_active.unsqueeze(-1)
            local_pos = pos @ render_aabb_to_local.T

            inside = (
                (t_active < ray_t_max[active_idx])
                & (t_active < MAX_DEPTH)
                & (
                    torch.all(
                        (render_aabb_min <= local_pos)
                        & (local_pos <= render_aabb_max),
                        dim=-1,
                    )
                )
            )
            if (~inside).any():
                active[active_idx[~inside]] = False
            if not inside.any():
                continue

            active_idx = active_idx[inside]
            t_active = t[active_idx]
            pos = pos[inside]
            dirs = ray_directions[active_idx]
            inv_dirs = inv_dir[active_idx]

            mip = self._mip_from_pos(pos, self.assets.max_cascade)
            occ = self._occupancy_at(pos, mip)
            if max_cascade > 0:
                for _ in range(max_cascade):
                    can_inc = mip < max_cascade
                    if not can_inc.any():
                        break
                    occ_next = self._occupancy_at(pos[can_inc], mip[can_inc] + 1)
                    inc_idx = can_inc.nonzero(as_tuple=False).squeeze(-1)[~occ_next]
                    if inc_idx.numel() == 0:
                        break
                    mip[inc_idx] += 1
                occ = self._occupancy_at(pos, mip)

            if (~occ).any():
                march_idx = active_idx[~occ]
                t[march_idx] = self._advance_to_next_voxel(
                    t[march_idx],
                    cone_angle,
                    pos[~occ],
                    dirs[~occ],
                    inv_dirs[~occ],
                    mip[~occ],
                )

            if occ.any():
                hit_idx = active_idx[occ]
                t_hit = t[hit_idx]
                pos_hit = pos[occ]
                dir_hit = dirs[occ]
                dt = self._calc_dt(t_hit, cone_angle)
                outputs = self.field(
                    self._warp_position(pos_hit, aabb_min, aabb_max),
                    self._warp_direction(dir_hit),
                )
                sigma = torch.exp(outputs.sigma.squeeze(-1).to(torch.float32))
                rgb = torch.sigmoid(outputs.rgb.to(torch.float32))
                alpha = 1.0 - torch.exp(-sigma * dt)
                weight = alpha * (1.0 - opacities[hit_idx])
                colors[hit_idx] += rgb * weight.unsqueeze(-1)
                stronger = weight > max_weights[hit_idx]
                if stronger.any():
                    best_depth[hit_idx[stronger]] = t_hit[stronger]
                    max_weights[hit_idx[stronger]] = weight[stronger]
                opacities[hit_idx] += weight
                t[hit_idx] = t_hit + dt

                terminate = opacities[hit_idx] > (1.0 - self.assets.min_transmittance)
                if terminate.any():
                    term_idx = hit_idx[terminate]
                    colors[term_idx] = colors[term_idx] / opacities[term_idx].unsqueeze(-1).clamp_min(1e-6)
                    opacities[term_idx] = 1.0
                    active[term_idx] = False

        return colors, opacities.unsqueeze(-1)

    def _build_exact_chunk_samples(
        self,
        *,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        ray_t_min: torch.Tensor,
        ray_t_max: torch.Tensor,
        pixel_index_offset: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_rays = ray_origins.shape[0]
        opacities = torch.zeros((n_rays,), device=device, dtype=torch.float32)
        t = ray_t_min.clone()
        inv_dir = self._safe_reciprocal(ray_directions)
        active = ray_t_max > ray_t_min
        render_aabb_to_local = self.render_aabb_to_local
        render_aabb_min = self.render_aabb_min
        render_aabb_max = self.render_aabb_max
        aabb_min = self.aabb_min
        aabb_max = self.aabb_max
        max_cascade = self.assets.max_cascade
        cone_angle = self.assets.cone_angle_constant

        sample_index = max(self.spp - 1, 0)
        if active.any():
            ray_ids = torch.arange(
                pixel_index_offset,
                pixel_index_offset + n_rays,
                device=device,
                dtype=torch.int64,
            )
            sample_ids = torch.full_like(ray_ids, sample_index)
            initial_n = _ld_random_val_tensor(sample_ids, (ray_ids * 786433) & 0xFFFFFFFF)
            t = torch.where(active, self._advance_n_steps(t, cone_angle, initial_n), t)

        sample_ray_indices: list[torch.Tensor] = []
        sample_t_starts: list[torch.Tensor] = []
        sample_t_ends: list[torch.Tensor] = []

        for _ in range(1024):
            if not active.any():
                break
            active_idx = active.nonzero(as_tuple=False).squeeze(-1)
            t_active = t[active_idx]
            pos = ray_origins[active_idx] + ray_directions[active_idx] * t_active.unsqueeze(-1)
            local_pos = pos @ render_aabb_to_local.T

            inside = (
                (t_active < ray_t_max[active_idx])
                & (t_active < MAX_DEPTH)
                & (
                    torch.all(
                        (render_aabb_min <= local_pos) & (local_pos <= render_aabb_max),
                        dim=-1,
                    )
                )
            )
            if (~inside).any():
                active[active_idx[~inside]] = False
            if not inside.any():
                continue

            active_idx = active_idx[inside]
            t_active = t[active_idx]
            pos = pos[inside]
            dirs = ray_directions[active_idx]
            inv_dirs = inv_dir[active_idx]

            mip = self._mip_from_pos(pos, max_cascade)
            occ = self._occupancy_at(pos, mip)
            if max_cascade > 0:
                for _ in range(max_cascade):
                    can_inc = mip < max_cascade
                    if not can_inc.any():
                        break
                    occ_next = self._occupancy_at(pos[can_inc], mip[can_inc] + 1)
                    inc_idx = can_inc.nonzero(as_tuple=False).squeeze(-1)[~occ_next]
                    if inc_idx.numel() == 0:
                        break
                    mip[inc_idx] += 1
                occ = self._occupancy_at(pos, mip)

            if (~occ).any():
                march_idx = active_idx[~occ]
                t[march_idx] = self._advance_to_next_voxel(
                    t[march_idx],
                    cone_angle,
                    pos[~occ],
                    dirs[~occ],
                    inv_dirs[~occ],
                    mip[~occ],
                )

            if occ.any():
                hit_idx = active_idx[occ]
                t_hit = t[hit_idx]
                pos_hit = pos[occ]
                dir_hit = dirs[occ]
                dt = self._calc_dt(t_hit, cone_angle)
                sample_ray_indices.append(hit_idx + pixel_index_offset)
                sample_t_starts.append(t_hit)
                sample_t_ends.append(t_hit + dt)
                sigma = self.field.query_density(
                    self._warp_position(pos_hit, aabb_min, aabb_max)
                ).squeeze(-1)
                alpha = 1.0 - torch.exp(-sigma * dt)
                weight = alpha * (1.0 - opacities[hit_idx])
                opacities[hit_idx] += weight
                t[hit_idx] = t_hit + dt

                terminate = opacities[hit_idx] > (1.0 - self.assets.min_transmittance)
                if terminate.any():
                    active[hit_idx[terminate]] = False

        if not sample_ray_indices:
            empty_long = torch.empty((0,), dtype=torch.int64, device=device)
            empty_float = torch.empty((0,), dtype=torch.float32, device=device)
            return empty_long, empty_float, empty_float
        return (
            torch.cat(sample_ray_indices, dim=0),
            torch.cat(sample_t_starts, dim=0),
            torch.cat(sample_t_ends, dim=0),
        )

    def _build_exact_sample_cache(
        self,
        *,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        ray_t_min: torch.Tensor,
        ray_t_max: torch.Tensor,
        device: torch.device,
    ) -> InstantNGPExactSampleCache:
        sample_ray_indices: list[torch.Tensor] = []
        sample_t_starts: list[torch.Tensor] = []
        sample_t_ends: list[torch.Tensor] = []
        for start in range(0, ray_directions.shape[0], self.ray_chunk_size):
            end = min(start + self.ray_chunk_size, ray_directions.shape[0])
            ray_indices, t_starts, t_ends = self._build_exact_chunk_samples(
                ray_origins=ray_origins[start:end],
                ray_directions=ray_directions[start:end],
                ray_t_min=ray_t_min[start:end],
                ray_t_max=ray_t_max[start:end],
                pixel_index_offset=start,
                device=device,
            )
            if ray_indices.numel() > 0:
                sample_ray_indices.append(ray_indices)
                sample_t_starts.append(t_starts)
                sample_t_ends.append(t_ends)

        if sample_ray_indices:
            ray_indices = torch.cat(sample_ray_indices, dim=0).to(torch.int32)
            t_starts = torch.cat(sample_t_starts, dim=0)
            t_ends = torch.cat(sample_t_ends, dim=0)
            order = torch.argsort(ray_indices.to(torch.int64), stable=True)
            ray_indices = ray_indices[order]
            t_starts = t_starts[order]
            t_ends = t_ends[order]
        else:
            ray_indices = torch.empty((0,), dtype=torch.int32, device=device)
            t_starts = torch.empty((0,), dtype=torch.float32, device=device)
            t_ends = torch.empty((0,), dtype=torch.float32, device=device)

        return InstantNGPExactSampleCache(
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            ray_indices=ray_indices,
            t_starts=t_starts,
            t_ends=t_ends,
            n_rays=ray_directions.shape[0],
        )

    def _render_exact_from_cache(
        self,
        cache: InstantNGPExactSampleCache,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if cache.ray_indices.numel() == 0:
            colors = torch.zeros((cache.n_rays, 3), device=device, dtype=torch.float32)
            opacities = torch.zeros((cache.n_rays, 1), device=device, dtype=torch.float32)
            return colors, opacities

        ray_indices = cache.ray_indices.to(torch.long)
        sample_origins = cache.ray_origins[ray_indices]
        sample_directions = cache.ray_directions[ray_indices]
        positions = sample_origins + sample_directions * cache.t_starts.unsqueeze(-1)
        outputs = self.field(
            self._warp_position(positions, self.aabb_min, self.aabb_max),
            self._warp_direction(sample_directions),
        )
        rgbs = torch.sigmoid(outputs.rgb.to(torch.float32))
        sigmas = torch.exp(outputs.sigma.squeeze(-1).to(torch.float32))
        weights, _, _ = nerfacc.render_weight_from_density(
            cache.t_starts,
            cache.t_ends,
            sigmas,
            ray_indices=ray_indices,
            n_rays=cache.n_rays,
        )
        colors = nerfacc.accumulate_along_rays(
            weights,
            values=rgbs,
            ray_indices=ray_indices,
            n_rays=cache.n_rays,
        )
        opacities = nerfacc.accumulate_along_rays(
            weights,
            values=None,
            ray_indices=ray_indices,
            n_rays=cache.n_rays,
        )
        terminate = opacities.squeeze(-1) > (1.0 - self.assets.min_transmittance)
        if terminate.any():
            colors[terminate] = colors[terminate] / opacities[terminate].clamp_min(1e-6)
            opacities[terminate] = 1.0
        return colors, opacities

    def _ray_box_intersection(
        self,
        origins: torch.Tensor,
        directions: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        local_origins = origins @ self.render_aabb_to_local.to(device).T
        local_directions = directions @ self.render_aabb_to_local.to(device).T
        inv_local_dirs = self._safe_reciprocal(local_directions)
        t0 = (self.render_aabb_min.to(device) - local_origins) * inv_local_dirs
        t1 = (self.render_aabb_max.to(device) - local_origins) * inv_local_dirs
        t_near = torch.max(torch.minimum(t0, t1), dim=-1).values
        t_far = torch.min(torch.maximum(t0, t1), dim=-1).values
        t_min = torch.clamp(t_near, min=0.0) + 1e-6
        t_max = t_far
        valid = t_max > t_min
        t_min = torch.where(valid, t_min, torch.zeros_like(t_min))
        t_max = torch.where(valid, t_max, torch.zeros_like(t_max))
        return t_min, t_max

    @torch.inference_mode()
    def render(
        self,
        view_index: int,
        *,
        width: int | None = None,
        height: int | None = None,
    ) -> InstantNGPRenderOutput:
        width = width or self.width
        height = height or self.height
        device = next(self.field.parameters()).device

        view = self.assets.views[view_index]
        fov_axis = 1
        relative_focal_length = view.focal_length.to(device) / float(view.resolution[fov_axis])
        focal_length = relative_focal_length * float((width, height)[fov_axis])
        principal_point = view.principal_point.to(device)
        # The pyngp benchmark path renders training views through the plain
        # perspective camera path here rather than re-applying per-image lens
        # distortion, so keep the native path pinhole as well.
        lens_params = torch.zeros_like(view.lens_params, device=device)
        camera_dirs = self._get_camera_rays(
            width=width,
            height=height,
            focal_length=focal_length,
            principal_point=principal_point,
            lens_params=lens_params,
            sample_index=max(self.spp - 1, 0),
            device=device,
        )

        camera_matrix = view.camera_matrix.to(device)
        rotation = camera_matrix[:, :3]
        translation = camera_matrix[:, 3]
        world_dirs = camera_dirs @ rotation.T
        origins = (
            translation.unsqueeze(0).expand_as(world_dirs)
            + world_dirs * float(self.assets.render_near_distance)
        )
        directions = torch.nn.functional.normalize(world_dirs, dim=-1)
        t_min, t_max = self._ray_box_intersection(origins, directions, device)

        rgba = torch.zeros((directions.shape[0], 4), device=device, dtype=torch.float32)
        background = self.background_color[:3].to(device=device, dtype=torch.float32)
        estimator = self.estimator.to(device)
        estimator.eval()

        if self.use_exact_marcher:
            cache_key = (view_index, width, height)
            cache = self._exact_sample_cache.get(cache_key)
            if cache is None:
                cache = self._build_exact_sample_cache(
                    ray_origins=origins,
                    ray_directions=directions,
                    ray_t_min=t_min,
                    ray_t_max=t_max,
                    device=device,
                )
                self._exact_sample_cache[cache_key] = cache
            colors, opacities = self._render_exact_from_cache(cache, device)
            colors = colors + background * (1.0 - opacities)
            if not self.assets.train_in_linear_colors:
                colors = _srgb_to_linear(colors)
            rgba[:, :3] = colors
            rgba[:, 3] = 1.0
            return InstantNGPRenderOutput(rgba=rgba.reshape(height, width, 4))

        for start in range(0, directions.shape[0], self.ray_chunk_size):
            end = min(start + self.ray_chunk_size, directions.shape[0])
            ray_origins = origins[start:end]
            ray_directions = directions[start:end]
            ray_t_min = t_min[start:end]
            ray_t_max = t_max[start:end]
            valid = ray_t_max > ray_t_min
            if not valid.any():
                continue

            ray_ids = torch.arange(start, end, device=device, dtype=torch.int64)
            sample_ids = torch.full_like(ray_ids, max(self.spp - 1, 0))
            ray_jitter = _ld_random_val_tensor(sample_ids, (ray_ids * 786433) & 0xFFFFFFFF)
            ray_t_min = torch.where(
                valid,
                self._advance_n_steps(ray_t_min, self.assets.cone_angle_constant, ray_jitter),
                ray_t_min,
            )

            if self.use_exact_marcher:
                colors, opacities = self._render_chunk_exact(
                    ray_origins=ray_origins,
                    ray_directions=ray_directions,
                    ray_t_min=ray_t_min,
                    ray_t_max=ray_t_max,
                    pixel_index_offset=start,
                    device=device,
                )
            else:
                def sample_midpoints(
                    sample_t_starts: torch.Tensor,
                    sample_t_ends: torch.Tensor,
                    sample_ray_indices: torch.Tensor,
                ) -> tuple[torch.Tensor, torch.Tensor]:
                    sample_origins = ray_origins[sample_ray_indices]
                    sample_directions = ray_directions[sample_ray_indices]
                    sample_positions = sample_origins + sample_directions * (
                        ((sample_t_starts + sample_t_ends) * 0.5).unsqueeze(-1)
                    )
                    return sample_positions, sample_directions

                sigma_fn = None
                if self.fast_use_sigma_pruning:
                    def sigma_fn(
                        sample_t_starts: torch.Tensor,
                        sample_t_ends: torch.Tensor,
                        sample_ray_indices: torch.Tensor,
                    ) -> torch.Tensor:
                        sample_positions, _ = sample_midpoints(
                            sample_t_starts, sample_t_ends, sample_ray_indices
                        )
                        return self.field.query_density(
                            self._warp_position(
                                sample_positions,
                                self.aabb_min.to(device),
                                self.aabb_max.to(device),
                            )
                        ).squeeze(-1)

                ray_indices, t_starts, t_ends = estimator.sampling(
                    ray_origins,
                    ray_directions,
                    sigma_fn=sigma_fn,
                    t_min=ray_t_min,
                    t_max=ray_t_max,
                    render_step_size=MIN_CONE_STEPSIZE * self.render_step_scale,
                    cone_angle=self.assets.cone_angle_constant,
                    alpha_thre=self.fast_alpha_thre,
                    early_stop_eps=self.fast_early_stop_eps,
                )

                def rgb_sigma_fn(
                    sample_t_starts: torch.Tensor,
                    sample_t_ends: torch.Tensor,
                    sample_ray_indices: torch.Tensor,
                ) -> tuple[torch.Tensor, torch.Tensor]:
                    positions, sample_directions = sample_midpoints(
                        sample_t_starts, sample_t_ends, sample_ray_indices
                    )
                    outputs = self.field(
                        self._warp_position(
                            positions,
                            self.aabb_min.to(device),
                            self.aabb_max.to(device),
                        ),
                        self._warp_direction(sample_directions),
                    )
                    rgbs = torch.sigmoid(outputs.rgb.to(torch.float32))
                    sigmas = torch.exp(outputs.sigma.squeeze(-1).to(torch.float32))
                    return rgbs, sigmas

                colors, opacities, _, _ = nerfacc.rendering(
                    t_starts,
                    t_ends,
                    ray_indices=ray_indices,
                    n_rays=end - start,
                    rgb_sigma_fn=rgb_sigma_fn,
                )
                terminate = opacities.squeeze(-1) > (1.0 - self.assets.min_transmittance)
                if terminate.any():
                    colors[terminate] = colors[terminate] / opacities[terminate].clamp_min(1e-6)
                    opacities[terminate] = 1.0

            colors = colors + background * (1.0 - opacities)
            if not self.assets.train_in_linear_colors:
                colors = _srgb_to_linear(colors)

            rgba[start:end, :3] = colors
            rgba[start:end, 3] = 1.0

        return InstantNGPRenderOutput(rgba=rgba.reshape(height, width, 4))
