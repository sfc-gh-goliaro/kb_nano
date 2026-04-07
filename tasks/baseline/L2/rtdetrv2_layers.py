"""RTDetrV2 shared layers."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..L1.batch_norm2d import BatchNorm2d
from ..L1.conv2d import Conv2d


def _activation_module(name: str | None) -> nn.Module:
    if name is None:
        return nn.Identity()
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")


def _activation_fn(name: str):
    name = name.lower()
    if name == "relu":
        return F.relu
    if name == "gelu":
        return F.gelu
    if name == "silu":
        return F.silu
    raise ValueError(f"Unsupported activation: {name}")


class RTDetrV2ConvNormLayer(nn.Module):
    def __init__(self, config, in_channels, out_channels, kernel_size, stride, padding=None, activation=None):
        super().__init__()
        self.conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2 if padding is None else padding,
            bias=False,
        )
        self.norm = BatchNorm2d(out_channels, eps=config.batch_norm_eps)
        self.activation = _activation_module(activation)

    def forward(self, hidden_state):
        hidden_state = self.conv(hidden_state)
        hidden_state = self.norm(hidden_state)
        hidden_state = self.activation(hidden_state)
        return hidden_state


class RTDetrV2MultiheadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(f"embed_dim must be divisible by num_heads, got {embed_dim} and {num_heads}")
        self.scaling = self.head_dim**-0.5
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _reshape(self, tensor: torch.Tensor, seq_len: int, batch_size: int):
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    @staticmethod
    def with_pos_embed(tensor: torch.Tensor, position_embeddings: torch.Tensor | None):
        return tensor if position_embeddings is None else tensor + position_embeddings

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: torch.Tensor | None = None,
        output_attentions: bool = False,
    ):
        batch_size, target_len, embed_dim = hidden_states.size()
        if position_embeddings is not None:
            hidden_states_original = hidden_states
            hidden_states = self.with_pos_embed(hidden_states, position_embeddings)
        else:
            hidden_states_original = hidden_states

        query_states = self.q_proj(hidden_states) * self.scaling
        key_states = self._reshape(self.k_proj(hidden_states), -1, batch_size)
        value_states = self._reshape(self.v_proj(hidden_states_original), -1, batch_size)

        proj_shape = (batch_size * self.num_heads, -1, self.head_dim)
        query_states = self._reshape(query_states, target_len, batch_size).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        source_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attention_mask is not None:
            attention_mask = attention_mask.expand(batch_size, 1, *attention_mask.size())
            if attention_mask.dtype == torch.bool:
                attention_mask = torch.zeros_like(attention_mask, dtype=attn_weights.dtype).masked_fill_(
                    attention_mask, -torch.inf
                )
            attn_weights = attn_weights.view(batch_size, self.num_heads, target_len, source_len) + attention_mask
            attn_weights = attn_weights.view(batch_size * self.num_heads, target_len, source_len)

        attn_weights = F.softmax(attn_weights, dim=-1)
        if output_attentions:
            attn_weights_reshaped = attn_weights.view(batch_size, self.num_heads, target_len, source_len)
            attn_weights = attn_weights_reshaped.view(batch_size * self.num_heads, target_len, source_len)
        else:
            attn_weights_reshaped = None

        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_probs, value_states)
        attn_output = attn_output.view(batch_size, self.num_heads, target_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, target_len, embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights_reshaped


class RTDetrV2EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.normalize_before = config.normalize_before
        self.self_attn = RTDetrV2MultiheadAttention(
            embed_dim=config.encoder_hidden_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.dropout,
        )
        self.self_attn_layer_norm = nn.LayerNorm(config.encoder_hidden_dim, eps=config.layer_norm_eps)
        self.dropout = config.dropout
        self.activation_fn = _activation_fn(config.encoder_activation_function)
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(config.encoder_hidden_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, config.encoder_hidden_dim)
        self.final_layer_norm = nn.LayerNorm(config.encoder_hidden_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None, position_embeddings=None, output_attentions=False, **kwargs):
        del kwargs
        residual = hidden_states
        if self.normalize_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            output_attentions=output_attentions,
        )
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        if not self.normalize_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        if self.normalize_before:
            hidden_states = self.final_layer_norm(hidden_states)
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        if not self.normalize_before:
            hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs


class RTDetrV2RepVggBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        activation = config.activation_function
        hidden_channels = int(config.encoder_hidden_dim * config.hidden_expansion)
        self.conv1 = RTDetrV2ConvNormLayer(config, hidden_channels, hidden_channels, 3, 1, padding=1)
        self.conv2 = RTDetrV2ConvNormLayer(config, hidden_channels, hidden_channels, 1, 1, padding=0)
        self.activation = _activation_module(activation)

    def forward(self, x):
        return self.activation(self.conv1(x) + self.conv2(x))


class RTDetrV2CSPRepLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        in_channels = config.encoder_hidden_dim * 2
        out_channels = config.encoder_hidden_dim
        hidden_channels = int(out_channels * config.hidden_expansion)
        activation = config.activation_function
        self.conv1 = RTDetrV2ConvNormLayer(config, in_channels, hidden_channels, 1, 1, activation=activation)
        self.conv2 = RTDetrV2ConvNormLayer(config, in_channels, hidden_channels, 1, 1, activation=activation)
        self.bottlenecks = nn.Sequential(*[RTDetrV2RepVggBlock(config) for _ in range(3)])
        if hidden_channels != out_channels:
            self.conv3 = RTDetrV2ConvNormLayer(config, hidden_channels, out_channels, 1, 1, activation=activation)
        else:
            self.conv3 = nn.Identity()

    def forward(self, hidden_state):
        hidden_state_1 = self.conv1(hidden_state)
        hidden_state_1 = self.bottlenecks(hidden_state_1)
        hidden_state_2 = self.conv2(hidden_state)
        return self.conv3(hidden_state_1 + hidden_state_2)


class RTDetrV2MLPPredictionHead(nn.Module):
    def __init__(self, config, input_dim, d_model, output_dim, num_layers):
        super().__init__()
        del config
        self.num_layers = num_layers
        h = [d_model] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
