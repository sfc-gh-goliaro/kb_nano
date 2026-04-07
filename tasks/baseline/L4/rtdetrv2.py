"""Repo-native RTDetrV2 object detector."""

from __future__ import annotations

import os
from types import SimpleNamespace

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import RTDetrV2Config

from ..L1.batch_norm2d import BatchNorm2d
from ..L1.conv2d import Conv2d
from ..L2.rtdetrv2_layers import RTDetrV2MLPPredictionHead
from ..L3.rtdetrv2_backbone import RTDetrV2ConvEncoder
from ..L3.rtdetrv2_decoder import RTDetrV2Decoder
from ..L3.rtdetrv2_hybrid_encoder import RTDetrV2HybridEncoder


def _boxes_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


class RTDetrV2Model(nn.Module):
    def __init__(self, config: RTDetrV2Config):
        super().__init__()
        self.config = config
        self.backbone = RTDetrV2ConvEncoder(config)
        intermediate_channel_sizes = self.backbone.intermediate_channel_sizes

        self.encoder_input_proj = nn.ModuleList()
        for in_channels in intermediate_channel_sizes:
            self.encoder_input_proj.append(
                nn.Sequential(
                    Conv2d(in_channels, config.encoder_hidden_dim, kernel_size=1, bias=False),
                    BatchNorm2d(config.encoder_hidden_dim),
                )
            )

        self.encoder = RTDetrV2HybridEncoder(config)

        if config.num_denoising > 0:
            self.denoising_class_embed = nn.Embedding(config.num_labels + 1, config.d_model, padding_idx=config.num_labels)

        if config.learn_initial_query:
            self.weight_embedding = nn.Embedding(config.num_queries, config.d_model)

        self.enc_output = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model, eps=config.layer_norm_eps),
        )
        self.enc_score_head = nn.Linear(config.d_model, config.num_labels)
        self.enc_bbox_head = RTDetrV2MLPPredictionHead(config, config.d_model, config.d_model, 4, num_layers=3)

        self.decoder_input_proj = nn.ModuleList()
        in_channels = config.decoder_in_channels[-1]
        for decoder_in in config.decoder_in_channels:
            self.decoder_input_proj.append(
                nn.Sequential(
                    Conv2d(decoder_in, config.d_model, kernel_size=1, bias=False),
                    BatchNorm2d(config.d_model, eps=config.batch_norm_eps),
                )
            )
        for _ in range(config.num_feature_levels - len(config.decoder_in_channels)):
            self.decoder_input_proj.append(
                nn.Sequential(
                    Conv2d(in_channels, config.d_model, kernel_size=3, stride=2, padding=1, bias=False),
                    BatchNorm2d(config.d_model, eps=config.batch_norm_eps),
                )
            )
            in_channels = config.d_model

        self.decoder = RTDetrV2Decoder(config)

    def generate_anchors(self, spatial_shapes, grid_size=0.05, device="cpu", dtype=torch.float32):
        anchors = []
        for level, (height, width) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(
                torch.arange(end=height, device=device).to(dtype),
                torch.arange(end=width, device=device).to(dtype),
                indexing="ij",
            )
            grid_xy = torch.stack([grid_x, grid_y], -1)
            grid_xy = grid_xy.unsqueeze(0) + 0.5
            grid_xy[..., 0] /= width
            grid_xy[..., 1] /= height
            wh = torch.ones_like(grid_xy) * grid_size * (2.0**level)
            anchors.append(torch.concat([grid_xy, wh], -1).reshape(-1, height * width, 4))
        eps = 1e-2
        anchors = torch.concat(anchors, 1)
        valid_mask = ((anchors > eps) * (anchors < 1 - eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.tensor(torch.finfo(dtype).max, dtype=dtype, device=device))
        return anchors, valid_mask

    def forward(self, pixel_values, pixel_mask=None, output_attentions=False, output_hidden_states=False, return_dict=True):
        batch_size, _, height, width = pixel_values.shape
        device = pixel_values.device
        if pixel_mask is None:
            pixel_mask = torch.ones((batch_size, height, width), device=device)

        features = self.backbone(pixel_values, pixel_mask)
        proj_feats = [self.encoder_input_proj[level](source) for level, (source, mask) in enumerate(features)]
        encoder_outputs = self.encoder(
            proj_feats,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        sources = []
        for level, source in enumerate(encoder_outputs.last_hidden_state):
            sources.append(self.decoder_input_proj[level](source))
        if self.config.num_feature_levels > len(sources):
            source = encoder_outputs.last_hidden_state[-1]
            for i in range(len(sources), self.config.num_feature_levels):
                source = self.decoder_input_proj[i](source)
                sources.append(source)

        source_flatten = []
        spatial_shapes_list = []
        spatial_shapes = torch.empty((len(sources), 2), device=device, dtype=torch.long)
        for level, source in enumerate(sources):
            h, w = source.shape[-2:]
            spatial_shapes[level, 0] = h
            spatial_shapes[level, 1] = w
            spatial_shapes_list.append((h, w))
            source_flatten.append(source.flatten(2).transpose(1, 2))
        source_flatten = torch.cat(source_flatten, 1)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        anchors, valid_mask = self.generate_anchors(spatial_shapes_list, device=device, dtype=source_flatten.dtype)
        memory = valid_mask.to(source_flatten.dtype) * source_flatten
        output_memory = self.enc_output(memory)
        enc_outputs_class = self.enc_score_head(output_memory)
        enc_outputs_coord_logits = self.enc_bbox_head(output_memory) + anchors
        _, topk_ind = torch.topk(enc_outputs_class.max(-1).values, self.config.num_queries, dim=1)
        reference_points_unact = enc_outputs_coord_logits.gather(
            dim=1,
            index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_coord_logits.shape[-1]),
        )
        target = output_memory.gather(dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1])).detach()
        decoder_outputs = self.decoder(
            inputs_embeds=target,
            encoder_hidden_states=source_flatten,
            encoder_attention_mask=None,
            reference_points=reference_points_unact.detach(),
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            level_start_index=level_start_index,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        if return_dict:
            return SimpleNamespace(
                last_hidden_state=decoder_outputs.last_hidden_state,
                intermediate_hidden_states=decoder_outputs.intermediate_hidden_states,
                intermediate_logits=decoder_outputs.intermediate_logits,
                intermediate_reference_points=decoder_outputs.intermediate_reference_points,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            )
        return (
            decoder_outputs.last_hidden_state,
            decoder_outputs.intermediate_hidden_states,
            decoder_outputs.intermediate_logits,
            decoder_outputs.intermediate_reference_points,
        )


class RTDetrV2ForObjectDetection(nn.Module):
    def __init__(self, config: RTDetrV2Config):
        super().__init__()
        self.config = config
        self.model = RTDetrV2Model(config)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ) -> "RTDetrV2ForObjectDetection":
        config = RTDetrV2Config.from_pretrained(model_name)
        model = cls(config)
        model_dir = snapshot_download(model_name, allow_patterns=["*.safetensors", "*.bin", "*.json"])
        state_dict = {}
        safetensor_path = os.path.join(model_dir, "model.safetensors")
        bin_path = os.path.join(model_dir, "pytorch_model.bin")
        if os.path.exists(safetensor_path):
            state_dict.update(load_file(safetensor_path))
        elif os.path.exists(bin_path):
            loaded = torch.load(bin_path, map_location="cpu")
            state_dict.update(loaded.get("state_dict", loaded))
        else:
            raise FileNotFoundError(f"No RTDetrV2 checkpoint found in {model_dir}")

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        unexpected = [k for k in unexpected if not (k.startswith("class_embed.") or k.startswith("bbox_embed."))]
        if missing or unexpected:
            raise RuntimeError(f"RTDetrV2 weight load mismatch: missing={missing}, unexpected={unexpected}")
        return model.to(device=device, dtype=dtype).eval()

    def forward(self, pixel_values: torch.Tensor):
        outputs = self.model(pixel_values=pixel_values, return_dict=True)
        logits = outputs.intermediate_logits[:, -1]
        pred_boxes = outputs.intermediate_reference_points[:, -1]
        return SimpleNamespace(logits=logits, pred_boxes=pred_boxes)

    def predict(self, pixel_values: torch.Tensor, image_size: int, max_detections: int = 100):
        outputs = self.forward(pixel_values)
        logits = outputs.logits.float().sigmoid()
        pred_boxes = outputs.pred_boxes.float()
        scores, labels = logits.max(dim=-1)
        boxes = _boxes_cxcywh_to_xyxy(pred_boxes) * float(image_size)
        topk = min(max_detections, scores.shape[1])
        top_scores, top_indices = scores.topk(topk, dim=1)
        top_labels = labels.gather(1, top_indices)
        top_boxes = boxes.gather(1, top_indices.unsqueeze(-1).expand(-1, -1, 4))

        batch = scores.shape[0]
        padded_boxes = torch.zeros(batch, max_detections, 4, device=boxes.device)
        padded_scores = torch.zeros(batch, max_detections, device=boxes.device)
        padded_labels = torch.full((batch, max_detections), -1, device=boxes.device, dtype=torch.long)
        padded_boxes[:, :topk] = top_boxes
        padded_scores[:, :topk] = top_scores
        padded_labels[:, :topk] = top_labels
        return {"boxes": padded_boxes, "scores": padded_scores, "labels": padded_labels}


RTDetrV2ForObjectDetectionWrapper = RTDetrV2ForObjectDetection
