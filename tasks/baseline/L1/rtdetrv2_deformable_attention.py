"""RTDetrV2 multi-scale deformable attention primitive."""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


def multi_scale_deformable_attention_v2(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    num_points_list: list[int],
    method: str = "default",
) -> torch.Tensor:
    batch_size, _, num_heads, hidden_dim = value.shape
    _, num_queries, _, _, _ = sampling_locations.shape
    value_list = (
        value.permute(0, 2, 3, 1)
        .flatten(0, 1)
        .split([height * width for height, width in value_spatial_shapes], dim=-1)
    )
    if method == "default":
        sampling_grids = 2 * sampling_locations - 1
    elif method == "discrete":
        sampling_grids = sampling_locations
    else:
        raise ValueError(f"Unsupported decoder method: {method}")
    sampling_grids = sampling_grids.permute(0, 2, 1, 3, 4).flatten(0, 1)
    sampling_grids = sampling_grids.split(num_points_list, dim=-2)

    sampling_value_list = []
    for level_id, (height, width) in enumerate(value_spatial_shapes):
        value_l = value_list[level_id].reshape(batch_size * num_heads, hidden_dim, height, width)
        sampling_grid_l = sampling_grids[level_id]
        if method == "default":
            sampling_value_l = F.grid_sample(
                value_l,
                sampling_grid_l,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
        else:
            sampling_coord = (sampling_grid_l * torch.tensor([[width, height]], device=value.device) + 0.5).to(
                torch.int64
            )
            sampling_coord_x = sampling_coord[..., 0].clamp(0, width - 1)
            sampling_coord_y = sampling_coord[..., 1].clamp(0, height - 1)
            sampling_coord = torch.stack([sampling_coord_x, sampling_coord_y], dim=-1)
            sampling_coord = sampling_coord.reshape(batch_size * num_heads, num_queries * num_points_list[level_id], 2)
            sampling_idx = torch.arange(sampling_coord.shape[0], device=value.device).unsqueeze(-1)
            sampling_idx = sampling_idx.repeat(1, sampling_coord.shape[1])
            sampling_value_l = value_l[sampling_idx, :, sampling_coord[..., 1], sampling_coord[..., 0]]
            sampling_value_l = sampling_value_l.permute(0, 2, 1).reshape(
                batch_size * num_heads, hidden_dim, num_queries, num_points_list[level_id]
            )
        sampling_value_list.append(sampling_value_l)

    attention_weights = attention_weights.permute(0, 2, 1, 3).reshape(
        batch_size * num_heads, 1, num_queries, sum(num_points_list)
    )
    output = (
        (torch.concat(sampling_value_list, dim=-1) * attention_weights)
        .sum(-1)
        .view(batch_size, num_heads * hidden_dim, num_queries)
    )
    return output.transpose(1, 2).contiguous()


class RTDetrV2MultiscaleDeformableAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        num_heads = config.decoder_attention_heads
        n_points = config.decoder_n_points

        if config.d_model % num_heads != 0:
            raise ValueError(
                f"embed_dim (d_model) must be divisible by num_heads, but got {config.d_model} and {num_heads}"
            )
        dim_per_head = config.d_model // num_heads
        if not ((dim_per_head & (dim_per_head - 1) == 0) and dim_per_head != 0):
            warnings.warn(
                "RTDetrV2MultiscaleDeformableAttention is most efficient when dim per head is a power of 2."
            )

        self.d_model = config.d_model
        self.n_levels = config.decoder_n_levels
        self.n_heads = num_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(config.d_model, num_heads * self.n_levels * n_points * 2)
        self.attention_weights = nn.Linear(config.d_model, num_heads * self.n_levels * n_points)
        self.value_proj = nn.Linear(config.d_model, config.d_model)
        self.output_proj = nn.Linear(config.d_model, config.d_model)

        self.offset_scale = config.decoder_offset_scale
        self.method = config.decoder_method

        self.n_points_list = [self.n_points for _ in range(self.n_levels)]
        n_points_scale = [1 / n for n in self.n_points_list for _ in range(n)]
        self.register_buffer("n_points_scale", torch.tensor(n_points_scale, dtype=torch.float32))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        position_embeddings: torch.Tensor | None = None,
        reference_points: torch.Tensor | None = None,
        spatial_shapes: torch.Tensor | None = None,
        spatial_shapes_list: list[tuple[int, int]] | None = None,
        level_start_index: torch.Tensor | None = None,
        output_attentions: bool = False,
    ):
        del level_start_index
        if position_embeddings is not None:
            hidden_states = hidden_states + position_embeddings

        batch_size, num_queries, _ = hidden_states.shape
        _, sequence_length, _ = encoder_hidden_states.shape
        if (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() != sequence_length:
            raise ValueError("Spatial shapes must match encoder sequence length.")

        value = self.value_proj(encoder_hidden_states)
        if attention_mask is not None:
            value = value.masked_fill(~attention_mask[..., None], 0.0)
        value = value.view(batch_size, sequence_length, self.n_heads, self.d_model // self.n_heads)

        sampling_offsets = self.sampling_offsets(hidden_states).view(
            batch_size, num_queries, self.n_heads, self.n_levels * self.n_points, 2
        )
        attention_weights = self.attention_weights(hidden_states).view(
            batch_size, num_queries, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = F.softmax(attention_weights, dim=-1)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], dim=-1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            n_points_scale = self.n_points_scale.to(dtype=hidden_states.dtype).unsqueeze(-1)
            offset = sampling_offsets * n_points_scale * reference_points[:, :, None, :, 2:] * self.offset_scale
            sampling_locations = reference_points[:, :, None, :, :2] + offset
        else:
            raise ValueError(f"Last dim of reference_points must be 2 or 4, but got {reference_points.shape[-1]}")

        output = multi_scale_deformable_attention_v2(
            value,
            spatial_shapes_list,
            sampling_locations,
            attention_weights,
            self.n_points_list,
            self.method,
        )
        output = self.output_proj(output)
        return output, attention_weights
