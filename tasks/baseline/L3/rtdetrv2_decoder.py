"""RTDetrV2 decoder stack."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..L1.rtdetrv2_deformable_attention import RTDetrV2MultiscaleDeformableAttention
from ..L2.rtdetrv2_layers import RTDetrV2MLPPredictionHead, RTDetrV2MultiheadAttention, _activation_fn


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


class RTDetrV2DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = RTDetrV2MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.dropout = config.dropout
        self.activation_fn = _activation_fn(config.decoder_activation_function)
        self.activation_dropout = config.activation_dropout
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.encoder_attn = RTDetrV2MultiscaleDeformableAttention(config)
        self.encoder_attn_layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.fc1 = nn.Linear(config.d_model, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, config.d_model)
        self.final_layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states,
        position_embeddings=None,
        reference_points=None,
        spatial_shapes=None,
        spatial_shapes_list=None,
        level_start_index=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        residual = hidden_states
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=encoder_attention_mask,
            position_embeddings=position_embeddings,
            output_attentions=output_attentions,
        )
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        second_residual = hidden_states
        hidden_states, cross_attn_weights = self.encoder_attn(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            position_embeddings=position_embeddings,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            level_start_index=level_start_index,
            output_attentions=output_attentions,
        )
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = second_residual + hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)
        return outputs


class RTDetrV2Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dropout = config.dropout
        self.layers = nn.ModuleList([RTDetrV2DecoderLayer(config) for _ in range(config.decoder_layers)])
        self.query_pos_head = RTDetrV2MLPPredictionHead(config, 4, 2 * config.d_model, config.d_model, num_layers=2)
        self.class_embed = nn.ModuleList([nn.Linear(config.d_model, config.num_labels) for _ in range(config.decoder_layers)])
        self.bbox_embed = nn.ModuleList(
            [RTDetrV2MLPPredictionHead(config, config.d_model, config.d_model, 4, num_layers=3) for _ in range(config.decoder_layers)]
        )

    def forward(
        self,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        position_embeddings=None,
        reference_points=None,
        spatial_shapes=None,
        spatial_shapes_list=None,
        level_start_index=None,
        valid_ratios=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        del position_embeddings, valid_ratios
        output_attentions = bool(output_attentions)
        output_hidden_states = bool(output_hidden_states)
        hidden_states = inputs_embeds

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if output_attentions else None
        intermediate = ()
        intermediate_reference_points = ()
        intermediate_logits = ()
        reference_points = F.sigmoid(reference_points)

        for idx, decoder_layer in enumerate(self.layers):
            reference_points_input = reference_points.unsqueeze(2)
            query_position_embeddings = self.query_pos_head(reference_points)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                position_embeddings=query_position_embeddings,
                encoder_hidden_states=encoder_hidden_states,
                reference_points=reference_points_input,
                spatial_shapes=spatial_shapes,
                spatial_shapes_list=spatial_shapes_list,
                level_start_index=level_start_index,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]
            predicted_corners = self.bbox_embed[idx](hidden_states)
            new_reference_points = F.sigmoid(predicted_corners + inverse_sigmoid(reference_points))
            reference_points = new_reference_points.detach()
            intermediate += (hidden_states,)
            intermediate_reference_points += (new_reference_points,)
            intermediate_logits += (self.class_embed[idx](hidden_states),)
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
                all_cross_attentions += (layer_outputs[2],)

        intermediate = torch.stack(intermediate, dim=1)
        intermediate_reference_points = torch.stack(intermediate_reference_points, dim=1)
        intermediate_logits = torch.stack(intermediate_logits, dim=1)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        if return_dict:
            return SimpleNamespace(
                last_hidden_state=hidden_states,
                intermediate_hidden_states=intermediate,
                intermediate_logits=intermediate_logits,
                intermediate_reference_points=intermediate_reference_points,
                hidden_states=all_hidden_states,
                attentions=all_self_attns,
                cross_attentions=all_cross_attentions,
            )
        return (
            hidden_states,
            intermediate,
            intermediate_logits,
            intermediate_reference_points,
            all_hidden_states,
            all_self_attns,
            all_cross_attentions,
        )
