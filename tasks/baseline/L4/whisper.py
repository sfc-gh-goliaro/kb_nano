"""Standalone Whisper encoder-decoder model for speech-to-text.

Implements the full WhisperForConditionalGeneration architecture:
  - Audio encoder: Conv1d feature extraction + sinusoidal pos embeddings + transformer layers
  - Text decoder: token embedding + learned pos embeddings + transformer layers + cross-attention
  - LM head (tied to decoder embedding)

Matches vLLM's whisper.py WhisperForConditionalGeneration interface.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig

from ..L1.conv1d import Conv1d
from ..L1.gelu import GELU
from ..L1.layer_norm import LayerNorm
from ..L1.embedding import Embedding
from ..L1.linear import Linear
from ..L2.parallel_embedding import ParallelLMHead
from ..L2.parallel_linear import ColumnParallelLinear
from ..L3.whisper_encoder_layer import WhisperEncoderLayer
from ..L3.whisper_decoder_layer import WhisperDecoderLayer


def _sinusoids(length: int, channels: int) -> torch.Tensor:
    """Sinusoidal positional embeddings matching HF's whisper implementation."""
    assert channels % 2 == 0
    log_timescale_increment = math.log(10000) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, None] * inv_timescales[None, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


@dataclass
class WhisperConfig:
    d_model: int = 1280
    encoder_layers: int = 32
    encoder_attention_heads: int = 20
    encoder_ffn_dim: int = 5120
    decoder_layers: int = 32
    decoder_attention_heads: int = 20
    decoder_ffn_dim: int = 5120
    vocab_size: int = 51866
    num_mel_bins: int = 128
    max_source_positions: int = 1500
    max_target_positions: int = 448
    scale_embedding: bool = False
    activation_function: str = "gelu"
    pad_token_id: int = 50256
    bos_token_id: int = 50257
    eos_token_id: int = 50257
    decoder_start_token_id: int = 50258
    is_encoder_decoder: bool = True
    dtype: torch.dtype = torch.bfloat16
    # For compatibility with engine's KV cache allocation
    num_hidden_layers: int = 32
    num_key_value_heads: int = 20
    num_attention_heads: int = 20
    head_dim: int = 64
    hidden_size: int = 1280

    @classmethod
    def from_pretrained(cls, model_name: str) -> "WhisperConfig":
        hf = AutoConfig.from_pretrained(model_name)
        head_dim = hf.d_model // hf.decoder_attention_heads
        return cls(
            d_model=hf.d_model,
            encoder_layers=hf.encoder_layers,
            encoder_attention_heads=hf.encoder_attention_heads,
            encoder_ffn_dim=hf.encoder_ffn_dim,
            decoder_layers=hf.decoder_layers,
            decoder_attention_heads=hf.decoder_attention_heads,
            decoder_ffn_dim=hf.decoder_ffn_dim,
            vocab_size=hf.vocab_size,
            num_mel_bins=hf.num_mel_bins,
            max_source_positions=hf.max_source_positions,
            max_target_positions=hf.max_target_positions,
            scale_embedding=hf.scale_embedding,
            activation_function=hf.activation_function,
            pad_token_id=hf.pad_token_id,
            bos_token_id=hf.bos_token_id,
            eos_token_id=hf.eos_token_id,
            decoder_start_token_id=hf.decoder_start_token_id,
            is_encoder_decoder=hf.is_encoder_decoder,
            num_hidden_layers=hf.decoder_layers,
            num_key_value_heads=hf.decoder_attention_heads,
            num_attention_heads=hf.decoder_attention_heads,
            head_dim=head_dim,
            hidden_size=hf.d_model,
        )


class WhisperEncoder(nn.Module):
    def __init__(self, config: WhisperConfig):
        super().__init__()
        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.max_source_positions = config.max_source_positions

        self.conv1 = Conv1d(config.num_mel_bins, embed_dim,
                            kernel_size=3, padding=1)
        self.conv2 = Conv1d(embed_dim, embed_dim,
                            kernel_size=3, stride=2, padding=1)
        self.gelu = GELU()

        self.layers = nn.ModuleList([
            WhisperEncoderLayer(config)
            for _ in range(config.encoder_layers)
        ])
        self.layer_norm = LayerNorm(embed_dim)

        self.register_buffer(
            "embed_positions",
            _sinusoids(config.max_source_positions, embed_dim),
        )

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_features: [B, num_mel_bins, T] log-mel spectrograms
        Returns:
            [B, T_enc, D] encoder hidden states
        """
        hidden_states = self.gelu(self.conv1(input_features))
        hidden_states = self.gelu(self.conv2(hidden_states))
        hidden_states = hidden_states.transpose(-1, -2)

        T_enc = hidden_states.shape[1]
        hidden_states = hidden_states + self.embed_positions[:T_enc].to(hidden_states.dtype)

        for layer in self.layers:
            hidden_states = layer(hidden_states)

        B, T, D = hidden_states.shape
        hidden_states = self.layer_norm(hidden_states.reshape(B * T, D))
        hidden_states = hidden_states.view(B, T, D)
        return hidden_states


class WhisperDecoder(nn.Module):
    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.max_target_positions = config.max_target_positions
        self.d_model = config.d_model

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model,
                                         padding_idx=config.pad_token_id)
        self.embed_positions = nn.Embedding(config.max_target_positions, config.d_model)

        self.layers = nn.ModuleList([
            WhisperDecoderLayer(config)
            for _ in range(config.decoder_layers)
        ])
        self.layer_norm = LayerNorm(config.d_model)
        self.embedding_op = Embedding()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        num_decoder_seqs: int | None = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [N] flat token IDs
            positions: [N] position indices
            encoder_hidden_states: [B, T_enc, D] or None
            num_decoder_seqs: number of decoder sequences
        """
        inputs_embeds = self.embedding_op(input_ids, self.embed_tokens.weight)
        pos_embeds = self.embedding_op(positions, self.embed_positions.weight)
        hidden_states = inputs_embeds + pos_embeds

        for layer in self.layers:
            hidden_states = layer(
                hidden_states, encoder_hidden_states,
                num_decoder_seqs=num_decoder_seqs,
            )

        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class WhisperForConditionalGeneration(nn.Module):
    packed_modules_mapping = {
        "self_attn.q_proj": ("self_attn.qkv_proj", "q"),
        "self_attn.k_proj": ("self_attn.qkv_proj", "k"),
        "self_attn.v_proj": ("self_attn.qkv_proj", "v"),
    }

    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.config = config
        self.encoder = WhisperEncoder(config)
        self.decoder = WhisperDecoder(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.d_model)

        self._encoder_output: torch.Tensor | None = None
        self._linear_op = Linear()

    def encode(self, input_features: torch.Tensor) -> torch.Tensor:
        """Run encoder and cache output for cross-attention."""
        self._encoder_output = self.encoder(input_features)
        return self._encoder_output

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decoder forward pass.

        Args:
            input_ids: [N] flat token IDs
            positions: [N] position indices
            encoder_hidden_states: [B, T_enc, D] or None (uses cached if None)
        """
        if encoder_hidden_states is None:
            encoder_hidden_states = self._encoder_output

        num_decoder_seqs = None
        if encoder_hidden_states is not None:
            num_decoder_seqs = encoder_hidden_states.shape[0]

        return self.decoder(
            input_ids, positions, encoder_hidden_states,
            num_decoder_seqs=num_decoder_seqs,
        )

    def forward_with_lm_proj(self, input_ids, positions,
                             encoder_hidden_states=None):
        hidden_states = self.forward(input_ids, positions, encoder_hidden_states)
        return self.lm_head.project(hidden_states)

    def compute_logits(self, hidden_states):
        logits = self.lm_head(hidden_states)
        if logits is not None:
            logits = logits.float()
        return logits

    def compute_logits_decode(self, partial_logits):
        logits = self.lm_head.gather_logits(partial_logits)
        if logits is not None:
            logits = logits.float()
        return logits

    def greedy_sample_decode(self, partial_logits):
        result = self.lm_head.gather_greedy(partial_logits.float())
        return result

    def clear_cross_attention_cache(self):
        """Clear cached encoder K/V in all cross-attention layers."""
        for layer in self.decoder.layers:
            layer.encoder_attn.clear_cache()
        self._encoder_output = None
