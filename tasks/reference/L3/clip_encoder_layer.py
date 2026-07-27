"""CLIP encoder layer: pre-norm attention + MLP with residual connections (L3).

Standard transformer encoder layer used in CLIP text models.
"""


from __future__ import annotations


# Inlined helper (no baseline task): HuggingFace config dataclasses.
# The baselines get these from transformers, so there is no baseline file
# to inline this from.
# The baselines get these from transformers, so there is no baseline file
# to inline this from.
import json
import os
from dataclasses import dataclass, field


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


# Inlined from tasks/reference/L1/layer_norm.py
import torch
import torch.nn as nn
import torch.nn.functional as F


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
