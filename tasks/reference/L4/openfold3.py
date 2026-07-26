"""OpenFold3 / AlphaFold3 structure prediction pipeline (L4 pipeline).

Contains:
- OpenFold3Config: model configuration dataclass.
- OpenFold3Model: full AF3 model wiring input embedder, MSA module,
  PairFormer, diffusion module, and auxiliary heads.
- load_openfold3_checkpoint: weight remapping from OpenFold3 HF checkpoint.

L4 wiring/configuration; computation lives in L1-L3 tasks.

Reference: openfold3/projects/of3_all_atom/model.py OpenFold3
           openfold3/projects/of3_all_atom/config/model_config.py
"""


from __future__ import annotations


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


# Inlined from tasks/reference/L1/tensor_ops.py
class Pad(nn.Module):
    """Functional padding op."""

    def forward(
        self, x: torch.Tensor, pad: tuple[int, ...], value: float = 0.0,
    ) -> torch.Tensor:
        return F.pad(x, pad, value=value)


class OneHot(nn.Module):
    """Functional one-hot encoding op."""

    def forward(self, x: torch.Tensor, num_classes: int) -> torch.Tensor:
        return F.one_hot(x, num_classes)


# Inlined from tasks/reference/L1/relu.py
class ReLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x)


# Inlined from tasks/reference/L1/sigmoid.py
class Sigmoid(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)


# Inlined from tasks/reference/L1/silu.py
class SiLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x)


# Inlined from tasks/baseline/L2/alphafold3_swiglu.py
class SwiGLU(nn.Module):
    """SwiGLU activation: SiLU(Wa x) * Wb x.

    Args:
        c_in: Number of input channels
        c_out: Number of output channels
    """

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.silu = SiLU()
        self.linear_a = Linear(c_in, c_out, bias=False)
        self.linear_b = Linear(c_in, c_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.silu(self.linear_a(x)) * self.linear_b(x)


class AdaLN(nn.Module):
    """Adaptive Layer Normalization matching the reference AdaLN.

    Submodule structure matches checkpoint keys:
    - layer_norm_s: LayerNorm(c_s), weight-only
    - linear_g: Linear(c_s, c_a, bias=True) — gating
    - linear_s: Linear(c_s, c_a, bias=False) — additive conditioning

    Reference: openfold3/core/model/primitives/normalization.py AdaLN

    Args:
        c_a: Activation channel dimension
        c_s: Conditioning channel dimension
    """

    def __init__(self, c_a: int, c_s: int):
        super().__init__()
        self.c_a = c_a
        self.c_s = c_s

        self.layer_norm_a = LayerNorm(c_a, create_scale=False, create_offset=False)
        self.layer_norm_s = LayerNorm(c_s, create_offset=False)
        self.sigmoid = Sigmoid()
        self.linear_g = Linear(c_s, c_a, bias=True)
        self.linear_s = Linear(c_s, c_a, bias=False)

    def forward(self, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        s_norm = self.layer_norm_s(s)
        g = self.sigmoid(self.linear_g(s_norm))
        a_norm = self.layer_norm_a(a)
        return g * (a_norm + self.linear_s(s_norm))


# Inlined from tasks/reference/L1/softmax.py
class Softmax(nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(x, dim=self.dim)


# Inlined from tasks/baseline/L2/alphafold3_of3_attention.py
import math


def _attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    biases: list[torch.Tensor],
) -> torch.Tensor:
    """Core SDPA: scores = softmax(Q K^T + biases) V.

    Args:
        query: [*, H, Q, C_hidden]
        key:   [*, H, K, C_hidden]
        value: [*, H, V, C_hidden]
        biases: list of tensors broadcastable to [*, H, Q, K]

    Returns:
        [*, H, Q, C_hidden]
    """
    scores = torch.einsum("...qc,...kc->...qk", query, key)

    for b in biases:
        scores = scores + b

    scores = F.softmax(scores, dim=-1)

    return torch.einsum("...qk,...kc->...qc", scores.to(dtype=value.dtype), value)


class OF3Attention(nn.Module):
    """Standard multi-head attention with gating and bias list support.

    Reference: openfold3/core/model/primitives/attention.py Attention

    Args:
        c_q: Input dimension of query data
        c_k: Input dimension of key data
        c_v: Input dimension of value data
        c_hidden: Per-head hidden dimension
        no_heads: Number of attention heads
        gating: Whether to gate output using query data
    """

    def __init__(
        self,
        c_q: int,
        c_k: int,
        c_v: int,
        c_hidden: int,
        no_heads: int,
        gating: bool = True,
        q_bias: bool = False,
    ):
        super().__init__()
        self.c_q = c_q
        self.c_k = c_k
        self.c_v = c_v
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.gating = gating

        self.linear_q = Linear(c_q, c_hidden * no_heads, bias=q_bias)
        self.linear_k = Linear(c_k, c_hidden * no_heads, bias=False)
        self.linear_v = Linear(c_v, c_hidden * no_heads, bias=False)
        self.linear_o = Linear(c_hidden * no_heads, c_q, bias=False)

        self.linear_g = None
        if gating:
            self.linear_g = Linear(c_q, c_hidden * no_heads, bias=False)

    def _prep_qkv(
        self, q_x: torch.Tensor, kv_x: torch.Tensor, apply_scale: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.linear_q(q_x)
        k = self.linear_k(kv_x)
        v = self.linear_v(kv_x)

        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        k = k.view(k.shape[:-1] + (self.no_heads, -1))
        v = v.view(v.shape[:-1] + (self.no_heads, -1))

        q = q.transpose(-2, -3)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)

        if apply_scale:
            q = q / math.sqrt(self.c_hidden)

        return q, k, v

    def _wrap_up(self, o: torch.Tensor, q_x: torch.Tensor) -> torch.Tensor:
        if self.linear_g is not None:
            g = torch.sigmoid(self.linear_g(q_x))
            g = g.view(g.shape[:-1] + (self.no_heads, -1))
            o = o * g

        o = o.reshape(o.shape[:-2] + (-1,))
        return self.linear_o(o)

    def forward(
        self,
        q_x: torch.Tensor,
        kv_x: torch.Tensor,
        biases: list[torch.Tensor] | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        lma_q_chunk_size: int = 1024,
        lma_kv_chunk_size: int = 4096,
        use_high_precision: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            q_x:  [*, Q, C_q] query data
            kv_x: [*, K, C_k] key data
            biases: List of biases that broadcast to [*, H, Q, K]

        Returns:
            [*, Q, C_q] attention update
        """
        if biases is None:
            biases = []

        q, k, v = self._prep_qkv(q_x, kv_x)

        o = _attention(q, k, v, biases)
        o = o.transpose(-2, -3)

        return self._wrap_up(o, q_x)


# Inlined from tasks/baseline/L2/alphafold3_attention_pair_bias.py
def _permute_final_dims(tensor: torch.Tensor, inds: list[int]) -> torch.Tensor:
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


class AttentionPairBias(nn.Module):
    """AF3 Algorithm 24: Attention with pair bias.

    When use_ada_layer_norm is True, uses two separate AdaLN instances
    (layer_norm_a_q, layer_norm_a_k) for query and key normalization,
    plus a linear_ada_out for output gating.

    Reference: openfold3/core/model/layers/attention_pair_bias.py

    Args:
        c_q: Input dimension of query/key/value
        c_s: Single activation channel dimension (for AdaLN)
        c_z: Pair activation channel dimension
        c_hidden: Per-head hidden dimension
        no_heads: Number of attention heads
        use_ada_layer_norm: Whether to use AdaLN-Zero conditioning
        gating: Whether to gate output
        inf: Large constant for masking
    """

    def __init__(
        self,
        c_q: int,
        c_k: int = 0,
        c_v: int = 0,
        c_s: int = 0,
        c_z: int = 128,
        c_hidden: int = 32,
        no_heads: int = 4,
        use_ada_layer_norm: bool = False,
        gating: bool = True,
        inf: float = 1e9,
    ):
        super().__init__()
        c_k = c_k or c_q
        c_v = c_v or c_q

        self.c_q = c_q
        self.c_s = c_s
        self.c_z = c_z
        self.inf = inf
        self.use_ada_layer_norm = use_ada_layer_norm

        if use_ada_layer_norm:
            self.layer_norm_a = AdaLN(c_a=c_q, c_s=c_s)
            self.linear_ada_out = Linear(c_s, c_q, bias=True)
        else:
            self.layer_norm_a = LayerNorm(c_q)

        self.layer_norm_z = LayerNorm(
            c_z, create_offset=not use_ada_layer_norm,
        )
        self.linear_z = Linear(c_z, no_heads, bias=False)

        self.sigmoid = Sigmoid()

        self.mha = OF3Attention(
            c_q=c_q, c_k=c_k, c_v=c_v,
            c_hidden=c_hidden, no_heads=no_heads, gating=gating,
            q_bias=True,
        )

    def _prep_bias(
        self, a: torch.Tensor, z: torch.Tensor, mask: torch.Tensor | None,
    ) -> list[torch.Tensor]:
        if mask is None:
            mask = a.new_ones(a.shape[:-1])

        batch_dims = a.shape[:-2]
        mask = mask.expand((*batch_dims, -1))

        mask_bias = (self.inf * (mask - 1))[..., None, None, :]
        biases = [mask_bias]

        z = self.layer_norm_z(z)
        z = self.linear_z(z)
        z = _permute_final_dims(z, [2, 0, 1])
        biases.append(z)

        return biases

    def forward(
        self,
        a: torch.Tensor,
        z: torch.Tensor,
        s: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        use_high_precision_attention: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            a:    [*, N, C_q] token/atom-level embedding
            z:    [*, N, N, C_z] pair embedding
            s:    [*, N, C_s] single embedding (for AdaLN)
            mask: [*, N] mask

        Returns:
            [*, N, C_q] attention update
        """
        biases = self._prep_bias(a=a, z=z, mask=mask)

        a = self.layer_norm_a(a, s) if self.use_ada_layer_norm else self.layer_norm_a(a)

        a = self.mha(q_x=a, kv_x=a, biases=biases)

        if self.use_ada_layer_norm:
            a = self.sigmoid(self.linear_ada_out(s)) * a

        return a


class CrossAttentionPairBias(nn.Module):
    """AF3 Algorithm 24: Cross-attention with pair bias for atom attention.

    Uses separate layer_norm_a_q and layer_norm_a_k for query/key, and
    does NOT apply layer_norm_z (pair bias goes through linear_z directly).
    Handles sequence-local blocked inputs.

    Reference: openfold3/core/model/layers/attention_pair_bias.py CrossAttentionPairBias

    Args:
        c_q: Input dimension of query/key/value
        c_s: Single activation channel dimension (for AdaLN)
        c_z: Pair activation channel dimension
        c_hidden: Per-head hidden dimension
        no_heads: Number of attention heads
        use_ada_layer_norm: Whether to use AdaLN-Zero conditioning
        n_query: Block size for queries
        n_key: Block size for keys
        gating: Whether to gate output
        inf: Large constant for masking
    """

    def __init__(
        self,
        c_q: int,
        c_k: int = 0,
        c_v: int = 0,
        c_s: int = 0,
        c_z: int = 16,
        c_hidden: int = 16,
        no_heads: int = 4,
        use_ada_layer_norm: bool = True,
        n_query: int | None = None,
        n_key: int | None = None,
        gating: bool = True,
        inf: float = 1e9,
    ):
        super().__init__()
        c_k = c_k or c_q
        c_v = c_v or c_q

        self.c_q = c_q
        self.c_s = c_s
        self.c_z = c_z
        self.inf = inf
        self.use_ada_layer_norm = use_ada_layer_norm
        self.n_query = n_query
        self.n_key = n_key

        if use_ada_layer_norm:
            self.layer_norm_a_q = AdaLN(c_a=c_q, c_s=c_s)
            self.layer_norm_a_k = AdaLN(c_a=c_q, c_s=c_s)
            self.linear_ada_out = Linear(c_s, c_q, bias=True)
        else:
            self.layer_norm_a_q = LayerNorm(c_q)
            self.layer_norm_a_k = LayerNorm(c_q)

        self.linear_z = Linear(c_z, no_heads, bias=False)

        self.sigmoid = Sigmoid()

        self.mha = OF3Attention(
            c_q=c_q, c_k=c_k, c_v=c_v,
            c_hidden=c_hidden, no_heads=no_heads, gating=gating,
            q_bias=True,
        )

    def forward(
        self,
        a: torch.Tensor,
        z: torch.Tensor,
        s: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            a:    [*, N_atom, C_q] atom-level embedding
            z:    [*, N_blocks, n_key, n_key, C_z] blocked pair embedding
            s:    [*, N_atom, C_s] single embedding (for AdaLN)
            mask: [*, N_atom] mask

        Returns:
            [*, N_atom, C_q] attention update
        """

        batch_dims = a.shape[:-2]
        n_atom, n_dim = a.shape[-2:]

        if mask is None:
            mask = a.new_ones(a.shape[:-1])

        a_query, a_key, block_mask = _convert_single_rep_to_blocks(
            ql=a, n_query=self.n_query, n_key=self.n_key, atom_mask=mask,
        )

        mask_bias = (self.inf * (block_mask - 1))[..., None, :, :]
        biases = [mask_bias]

        z_bias = self.linear_z(z)
        z_bias = _permute_final_dims(z_bias, [2, 0, 1])
        biases.append(z_bias)

        if self.use_ada_layer_norm:
            s_q, s_k, _ = _apply_block_indices(
                ql=s, n_query=self.n_query, n_key=self.n_key, atom_mask=mask,
            )
            a_q = self.layer_norm_a_q(a_query, s_q)
            a_k = self.layer_norm_a_k(a_key, s_k)
        else:
            a_q = self.layer_norm_a_q(a_query)
            a_k = self.layer_norm_a_k(a_key)

        a_out = self.mha(q_x=a_q, kv_x=a_k, biases=biases)

        a_out = a_out.reshape((*batch_dims, -1, n_dim))[..., :n_atom, :]

        if self.use_ada_layer_norm:
            a_out = self.sigmoid(self.linear_ada_out(s)) * a_out

        return a_out


# Inlined from tasks/baseline/L2/alphafold3_swiglu_transition.py
class SwiGLUTransition(nn.Module):
    """AF3 Algorithm 11: SwiGLU-based transition.

    Args:
        c_in: Input channel dimension
        n: Factor multiplied to c_in for hidden dimension
    """

    def __init__(self, c_in: int, n: int):
        super().__init__()
        self.c_in = c_in
        self.n = n

        self.layer_norm = LayerNorm(c_in)
        self.swiglu = SwiGLU(c_in, n * c_in)
        self.linear_out = Linear(n * c_in, c_in, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        chunk_size: int | None = None,
        ckpt_chunk_size: int | None = None,
    ) -> torch.Tensor:
        if mask is None:
            mask = x.new_ones(x.shape[:-1])

        mask = mask.unsqueeze(-1)

        x = self.layer_norm(x)
        x = self.swiglu(x)
        x = self.linear_out(x)
        x = x * mask

        return x


class ConditionedTransitionBlock(nn.Module):
    """AF3 Algorithm 25: SwiGLU transition with AdaLN-Zero conditioning.

    Submodule names match the reference checkpoint:
    - layer_norm: AdaLN
    - swiglu: SwiGLU
    - linear_g: output gate Linear(c_s, c_a, bias=True)
    - linear_out: Linear(n*c_a, c_a, bias=False)

    Reference: openfold3/core/model/layers/transition.py ConditionedTransitionBlock

    Args:
        c_a: Activation channel dimension
        c_s: Conditioning channel dimension
        n: Factor for hidden dimension
    """

    def __init__(self, c_a: int, c_s: int, n: int):
        super().__init__()
        self.layer_norm = AdaLN(c_a=c_a, c_s=c_s)
        self.swiglu = SwiGLU(c_a, n * c_a)
        self.sigmoid = Sigmoid()
        self.linear_g = Linear(c_s, c_a, bias=True)
        self.linear_out = Linear(n * c_a, c_a, bias=False)

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        mask: torch.Tensor | None = None,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        if mask is None:
            mask = a.new_ones(a.shape[:-1])

        mask = mask.unsqueeze(-1)

        a = self.layer_norm(a, s)
        b = self.swiglu(a)
        a = self.sigmoid(self.linear_g(s)) * self.linear_out(b)
        a = a * mask

        return a


# Inlined from tasks/baseline/L3/alphafold3_diffusion_transformer.py
class DiffusionTransformerBlock(nn.Module):
    """AF3 Algorithm 23: Diffusion transformer block.

    Args:
        c_a: Token activation channel dimension
        c_s: Single activation channel dimension
        c_z: Pair activation channel dimension
        c_hidden: Per-head hidden dimension
        no_heads: Number of attention heads
        n_transition: Transition layer scale
        use_ada_layer_norm: Whether to use AdaLN-Zero
        inf: Large masking constant
    """

    def __init__(
        self,
        c_a: int,
        c_s: int,
        c_z: int,
        c_hidden: int,
        no_heads: int,
        n_transition: int,
        use_ada_layer_norm: bool = True,
        n_query: int | None = None,
        n_key: int | None = None,
        inf: float = 1e9,
    ):
        super().__init__()
        self.use_cross_attention = n_query is not None

        if not self.use_cross_attention:
            self.attention_pair_bias = AttentionPairBias(
                c_q=c_a, c_k=c_a, c_v=c_a,
                c_s=c_s, c_z=c_z,
                c_hidden=c_hidden,
                no_heads=no_heads,
                use_ada_layer_norm=use_ada_layer_norm,
                gating=True,
                inf=inf,
            )
        else:
            self.attention_pair_bias = CrossAttentionPairBias(
                c_q=c_a, c_k=c_a, c_v=c_a,
                c_s=c_s, c_z=c_z,
                c_hidden=c_hidden,
                no_heads=no_heads,
                use_ada_layer_norm=use_ada_layer_norm,
                n_query=n_query,
                n_key=n_key,
                gating=True,
                inf=inf,
            )

        self.conditioned_transition = ConditionedTransitionBlock(
            c_a=c_a, c_s=c_s, n=n_transition,
        )

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        use_high_precision_attention: bool = False,
        _mask_trans: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            a:    [*, N, C_token] token-level embedding
            s:    [*, N, C_s] single embedding
            z:    [*, N, N, C_z] pair embedding
            mask: [*, N] mask

        Returns:
            [*, N, C_token] updated token embedding
        """
        a = a + self.attention_pair_bias(a=a, z=z, s=s, mask=mask)

        trans_mask = mask if _mask_trans else None
        a = a + self.conditioned_transition(a=a, s=s, mask=trans_mask)

        return a


class DiffusionTransformer(nn.Module):
    """AF3 Algorithm 23: Diffusion transformer stack.

    Args:
        c_a: Token activation channel dimension
        c_s: Single activation channel dimension
        c_z: Pair activation channel dimension
        c_hidden: Per-head hidden dimension
        no_heads: Number of attention heads
        no_blocks: Number of transformer blocks
        n_transition: Transition layer scale
        use_ada_layer_norm: Whether to use AdaLN-Zero
        inf: Large masking constant
    """

    def __init__(
        self,
        c_a: int,
        c_s: int,
        c_z: int,
        c_hidden: int,
        no_heads: int,
        no_blocks: int,
        n_transition: int,
        use_ada_layer_norm: bool = True,
        n_query: int | None = None,
        n_key: int | None = None,
        inf: float = 1e9,
        blocks_per_ckpt: int | None = None,
        **kwargs,
    ):
        super().__init__()

        self.use_cross_attention = n_query is not None
        if self.use_cross_attention:
            self.layer_norm_z = LayerNorm(c_z, create_offset=False)

        self.blocks = nn.ModuleList([
            DiffusionTransformerBlock(
                c_a=c_a, c_s=c_s, c_z=c_z,
                c_hidden=c_hidden, no_heads=no_heads,
                n_transition=n_transition,
                use_ada_layer_norm=use_ada_layer_norm,
                n_query=n_query,
                n_key=n_key,
                inf=inf,
            )
            for _ in range(no_blocks)
        ])

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        use_high_precision_attention: bool = False,
        _mask_trans: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            a:    [*, N, C_token] token-level embedding
            s:    [*, N, C_s] single embedding
            z:    [*, N, N, C_z] pair embedding
            mask: [*, N] mask

        Returns:
            [*, N, C_token] updated token embedding
        """
        if self.use_cross_attention:
            z = self.layer_norm_z(z)

        for block in self.blocks:
            a = block(a=a, s=s, z=z, mask=mask)

        return a


# Inlined from tasks/baseline/L2/alphafold3_atom_attention.py
def _get_block_key_indices(
    atom_mask: torch.Tensor, n_query: int, n_key: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized computation of key-block gather indices.

    Returns:
        safe_indices: [*, N_blocks, n_key] clamped indices
        invalid_mask: [*, N_blocks, n_key] True where index is out of range
    """
    batch_dims = atom_mask.shape[:-1]
    n_atom = atom_mask.shape[-1]
    num_blocks = math.ceil(n_atom / n_query)
    device = atom_mask.device
    offset = n_query // 2

    subset_centers = offset + torch.arange(num_blocks, device=device) * n_query
    subset_centers = subset_centers.reshape(*(1,) * len(batch_dims), num_blocks)
    subset_centers = subset_centers.expand(*batch_dims, num_blocks)

    n_real = atom_mask.sum(dim=-1, keepdim=True).expand(*batch_dims, num_blocks)

    initial = (
        subset_centers.unsqueeze(-1)
        + torch.arange(-n_key // 2, n_key // 2, device=device)
    ).int()

    underflow = torch.relu(-initial[..., 0])
    overflow = torch.relu(initial[..., -1] - (n_real - 1))
    total_shift = torch.where(underflow > 0, underflow, -overflow)
    final = initial + total_shift.unsqueeze(-1)

    n_real_exp = n_real.unsqueeze(-1)
    invalid = (final < 0) | (final >= n_real_exp)
    safe = torch.clamp(final, torch.zeros_like(n_real_exp), (n_real_exp - 1).clamp(min=0))

    return safe.long(), invalid


def _convert_single_rep_to_blocks(
    ql: torch.Tensor,
    n_query: int,
    n_key: int,
    atom_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Convert flat atom representation to windowed block format (vectorized).

    Args:
        ql: [*, N_atom, C] atom features
        n_query: block height
        n_key: block width
        atom_mask: [*, N_atom] mask

    Returns:
        ql_query: [*, N_blocks, n_query, C]
        ql_key:   [*, N_blocks, n_key, C]
        mask_blocks: [*, N_blocks, n_query, n_key] or None
    """
    batch_dims = ql.shape[:-2]
    n_atom, c = ql.shape[-2], ql.shape[-1]
    num_blocks = math.ceil(n_atom / n_query)
    pad_q = (-n_atom) % n_query

    if pad_q > 0:
        ql = Pad()(ql, (0, 0, 0, pad_q))
        if atom_mask is not None:
            atom_mask = Pad()(atom_mask, (0, pad_q))

    ql_query = ql.reshape(*batch_dims, num_blocks, n_query, c)

    if atom_mask is None:
        atom_mask = ql.new_ones(*batch_dims, n_atom + pad_q)

    atom_mask = atom_mask.expand(*batch_dims, -1)
    key_indices, invalid_mask = _get_block_key_indices(atom_mask, n_query, n_key)

    flat_batch = int(math.prod(batch_dims)) if batch_dims else 1
    ql_flat = ql.reshape(flat_batch, n_atom + pad_q, c)
    idx_flat = key_indices.reshape(flat_batch, num_blocks * n_key)
    idx_expanded = idx_flat.unsqueeze(-1).expand(-1, -1, c)

    ql_key_flat = torch.gather(ql_flat, 1, idx_expanded)
    mask_flat = invalid_mask.reshape(flat_batch, num_blocks * n_key).unsqueeze(-1).expand(-1, -1, c)
    ql_key_flat.masked_fill_(mask_flat, 0.0)
    ql_key = ql_key_flat.reshape(*batch_dims, num_blocks, n_key, c)

    mask_q = atom_mask.reshape(*batch_dims, num_blocks, n_query)
    mask_k_valid = (~invalid_mask).to(atom_mask.dtype)
    atom_mask_at_keys = torch.gather(
        atom_mask.reshape(flat_batch, -1), 1,
        idx_flat,
    ).reshape(*batch_dims, num_blocks, n_key)
    mask_k_valid = mask_k_valid * atom_mask_at_keys
    mask_blocks = mask_q.unsqueeze(-1) * mask_k_valid.unsqueeze(-2)

    return ql_query, ql_key, mask_blocks


_apply_block_indices = _convert_single_rep_to_blocks


def _convert_pair_rep_to_blocks(
    batch: dict,
    zij_trunk: torch.Tensor,
    n_query: int,
    n_key: int,
) -> torch.Tensor:
    """Convert pair representation to block format for atom attention (vectorized).

    Args:
        batch: needs atom_mask, atom_to_token_index
        zij_trunk: [*, N_token, N_token, C_z]
        n_query: block height
        n_key: block width

    Returns:
        [*, N_blocks, n_query, n_key, C_z]
    """
    atom_mask = batch["atom_mask"]
    n_atoms = atom_mask.shape[-1]
    batch_dims = zij_trunk.shape[:-3]
    c_z = zij_trunk.shape[-1]

    if "atom_to_token_index" in batch:
        atom_to_token = batch["atom_to_token_index"]
        if atom_to_token.dim() > 1:
            atom_to_token = atom_to_token[0]
    else:
        n_token = zij_trunk.shape[-2]
        atom_to_token = torch.arange(n_token, device=zij_trunk.device)
        if n_atoms > n_token:
            atom_to_token = atom_to_token.repeat_interleave(
                (n_atoms + n_token - 1) // n_token
            )[:n_atoms]

    num_blocks = math.ceil(n_atoms / n_query)
    pad_q = (-n_atoms) % n_query

    atk_padded = Pad()(atom_to_token, (0, pad_q))
    q_indices = atk_padded.reshape(num_blocks, n_query)

    atom_mask_exp = atom_mask.expand(*batch_dims, -1)
    key_indices, invalid_mask = _get_block_key_indices(atom_mask_exp, n_query, n_key)

    flat_batch = int(math.prod(batch_dims)) if batch_dims else 1

    atk_flat = atom_to_token.expand(flat_batch, -1)
    key_idx_flat = key_indices.reshape(flat_batch, num_blocks * n_key)
    k_token_flat = torch.gather(atk_flat, 1, key_idx_flat.clamp(min=0, max=n_atoms - 1))
    k_indices = k_token_flat.reshape(flat_batch, num_blocks, n_key)

    zij_flat = zij_trunk.reshape(flat_batch, *zij_trunk.shape[-3:])
    batch_idx = torch.arange(flat_batch, device=zij_trunk.device).view(-1, 1, 1, 1)
    q_idx = q_indices.long().unsqueeze(0).expand(flat_batch, -1, -1)

    plm = zij_flat[batch_idx, q_idx.unsqueeze(-1), k_indices.unsqueeze(-2)]

    inv_expanded = invalid_mask.reshape(flat_batch, num_blocks, n_key)
    plm.masked_fill_(inv_expanded[:, :, None, :, None].expand_as(plm), 0.0)

    pair_mask = _get_pair_atom_block_mask(
        atom_mask=atom_mask_exp, num_blocks=num_blocks,
        n_query=n_query, n_key=n_key, pad_q=pad_q,
        key_indices=key_indices, invalid_mask=invalid_mask,
    )
    plm = plm * pair_mask.reshape(flat_batch, num_blocks, n_query, n_key, 1)
    plm = plm.reshape(*batch_dims, num_blocks, n_query, n_key, c_z)

    return plm


def _get_pair_atom_block_mask(
    atom_mask: torch.Tensor,
    num_blocks: int,
    n_query: int,
    n_key: int,
    pad_q: int,
    key_indices: torch.Tensor,
    invalid_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute pair atom block mask."""
    batch_dims = atom_mask.shape[:-1]
    flat_batch = int(math.prod(batch_dims)) if batch_dims else 1
    mask_flat = atom_mask.reshape(flat_batch, -1)

    mask_padded = Pad()(mask_flat, (0, pad_q))
    mask_q = mask_padded.reshape(flat_batch, num_blocks, n_query)

    idx_flat = key_indices.reshape(flat_batch, num_blocks * n_key)
    mask_k_vals = torch.gather(mask_flat, 1, idx_flat.clamp(min=0, max=mask_flat.shape[-1] - 1))
    mask_k = mask_k_vals.reshape(flat_batch, num_blocks, n_key)
    inv_flat = invalid_mask.reshape(flat_batch, num_blocks, n_key)
    mask_k = mask_k * (~inv_flat).to(mask_k.dtype)

    pair_mask = mask_q.unsqueeze(-1) * mask_k.unsqueeze(-2)
    return pair_mask.reshape(*batch_dims, num_blocks, n_query, n_key)


def _broadcast_token_feat_to_atoms(
    token_mask: torch.Tensor,
    num_atoms_per_token: torch.Tensor | None,
    token_feat: torch.Tensor,
    atom_to_token_index: torch.Tensor | None = None,
    n_atoms: int | None = None,
) -> torch.Tensor:
    """Broadcast token-level features to atom-level.

    Args:
        token_mask: [*, N_token]
        num_atoms_per_token: [*, N_token] or None
        token_feat: [*, N_token, C]
        atom_to_token_index: [*, N_atom] optional direct mapping
        n_atoms: total number of atoms if atom_to_token_index not provided

    Returns:
        [*, N_atom, C]
    """
    if atom_to_token_index is not None:
        idx = atom_to_token_index.long()
        while idx.dim() < token_feat.dim() - 1:
            idx = idx.unsqueeze(1)
        idx = idx.expand(*token_feat.shape[:-2], idx.shape[-1])
        return torch.gather(
            token_feat, -2,
            idx.unsqueeze(-1).expand(*idx.shape, token_feat.shape[-1]),
        )

    if num_atoms_per_token is not None:
        return torch.repeat_interleave(
            token_feat, num_atoms_per_token.long(), dim=-2,
        )

    return token_feat


def _aggregate_atom_feat_to_tokens(
    token_mask: torch.Tensor,
    atom_to_token_index: torch.Tensor,
    atom_mask: torch.Tensor,
    atom_feat: torch.Tensor,
    mode: str = "mean",
) -> torch.Tensor:
    """Aggregate atom-level features to token-level.

    Args:
        token_mask: [*, N_token]
        atom_to_token_index: [N_atom]
        atom_mask: [*, N_atom]
        atom_feat: [*, N_atom, C]
        mode: "mean" or "sum"

    Returns:
        [*, N_token, C]
    """
    n_token = token_mask.shape[-1]
    c = atom_feat.shape[-1]
    batch_shape = atom_feat.shape[:-2]

    atom_mask_expanded = atom_mask.expand(*batch_shape, -1)

    result = atom_feat.new_zeros(*batch_shape, n_token, c)
    masked_feat = atom_feat * atom_mask_expanded[..., None]

    idx = atom_to_token_index.long().expand(*batch_shape, -1)
    result.scatter_add_(-2, idx.unsqueeze(-1).expand_as(masked_feat), masked_feat)

    if mode == "mean":
        counts = torch.zeros(*batch_shape, n_token, dtype=result.dtype, device=result.device)
        counts.scatter_add_(-1, idx, atom_mask_expanded.to(dtype=result.dtype))
        counts = counts.clamp(min=1.0)
        result = result / counts.unsqueeze(-1)

    return result


class RefAtomFeatureEmbedder(nn.Module):
    """Embeds reference atom features (Algorithm 5, lines 1-6).

    Args:
        c_atom_ref_element: Reference element one-hot dim (119)
        c_atom_ref_name_chars: Reference atom name chars dim (256 = 4*64)
        c_atom: Atom single conditioning dim
        c_atom_pair: Atom pair conditioning dim
    """

    def __init__(
        self,
        c_atom_ref_element: int = 119,
        c_atom_ref_name_chars: int = 256,
        c_atom: int = 128,
        c_atom_pair: int = 16,
    ):
        super().__init__()
        self.linear_ref_pos = Linear(3, c_atom, bias=False)
        self.linear_ref_charge = Linear(1, c_atom, bias=False)
        self.linear_ref_mask = Linear(1, c_atom, bias=False)
        self.linear_ref_element = Linear(c_atom_ref_element, c_atom, bias=False)
        self.linear_ref_atom_chars = Linear(c_atom_ref_name_chars, c_atom, bias=False)
        self.linear_ref_offset = Linear(3, c_atom_pair, bias=False)
        self.linear_inv_sq_dists = Linear(1, c_atom_pair, bias=False)
        self.linear_valid_mask = Linear(1, c_atom_pair, bias=False)

    def forward(
        self,
        batch: dict,
        n_query: int,
        n_key: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dtype = batch["ref_pos"].dtype

        cl = self.linear_ref_pos(batch["ref_pos"])
        cl = cl + self.linear_ref_charge(
            torch.arcsinh(batch["ref_charge"].unsqueeze(-1))
        )
        cl = cl + self.linear_ref_mask(batch["ref_mask"].unsqueeze(-1).to(dtype=dtype))
        cl = cl + self.linear_ref_element(batch["ref_element"].to(dtype=dtype))
        cl = cl + self.linear_ref_atom_chars(
            batch["ref_atom_name_chars"].flatten(start_dim=-2).to(dtype=dtype)
        )

        d_l, d_m, atom_mask = _convert_single_rep_to_blocks(
            ql=batch["ref_pos"],
            n_query=n_query, n_key=n_key,
            atom_mask=batch["atom_mask"],
        )
        v_l, v_m, _ = _convert_single_rep_to_blocks(
            ql=batch["ref_space_uid"].unsqueeze(-1),
            n_query=n_query, n_key=n_key,
            atom_mask=batch["atom_mask"],
        )

        dlm = (d_l.unsqueeze(-2) - d_m.unsqueeze(-3)) * atom_mask.unsqueeze(-1)
        vlm = (v_l.unsqueeze(-2) == v_m.unsqueeze(-3)).to(
            dtype=dlm.dtype
        ) * atom_mask.unsqueeze(-1)

        plm = self.linear_ref_offset(dlm) * vlm

        inv_sq_dists = 1.0 / (1 + torch.sum(dlm ** 2, dim=-1, keepdim=True))
        plm = plm + self.linear_inv_sq_dists(inv_sq_dists) * vlm
        plm = plm + self.linear_valid_mask(vlm) * vlm

        return cl, plm


class NoisyPositionEmbedder(nn.Module):
    """Embeds noisy positions and trunk embeddings (Algorithm 5, lines 8-12).

    Args:
        c_s: Single representation channel dimension
        c_z: Pair representation channel dimension
        c_atom: Atom single conditioning channel dimension
        c_atom_pair: Atom pair conditioning channel dimension
    """

    def __init__(self, c_s: int, c_z: int, c_atom: int, c_atom_pair: int):
        super().__init__()
        self.layer_norm_s = LayerNorm(c_s, create_offset=False)
        self.linear_s = Linear(c_s, c_atom, bias=False)
        self.layer_norm_z = LayerNorm(c_z, create_offset=False)
        self.linear_z = Linear(c_z, c_atom_pair, bias=False)
        self.linear_r = Linear(3, c_atom, bias=False)

    def forward(
        self,
        batch: dict,
        cl: torch.Tensor,
        plm: torch.Tensor,
        si_trunk: torch.Tensor,
        zij_trunk: torch.Tensor,
        rl: torch.Tensor,
        n_query: int,
        n_key: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        si_trunk_proj = self.linear_s(self.layer_norm_s(si_trunk))
        si_trunk_proj = _broadcast_token_feat_to_atoms(
            token_mask=batch["token_mask"],
            num_atoms_per_token=batch.get("num_atoms_per_token"),
            token_feat=si_trunk_proj,
            atom_to_token_index=batch.get("atom_to_token_index"),
        )
        cl = cl + si_trunk_proj

        zij_trunk_proj = self.linear_z(self.layer_norm_z(zij_trunk))
        zij_trunk_block = _convert_pair_rep_to_blocks(
            batch=batch, zij_trunk=zij_trunk_proj,
            n_query=n_query, n_key=n_key,
        )
        plm = plm + zij_trunk_block

        ql = cl + self.linear_r(rl)

        return cl, plm, ql


class AtomAttentionEncoder(nn.Module):
    """AF3 Algorithm 5: Atom attention encoder.

    Args:
        c_atom: Atom single representation channel dimension
        c_atom_pair: Atom pair representation channel dimension
        c_token: Token single representation output channel dimension
        c_atom_ref_element: Reference element one-hot dim
        c_atom_ref_name_chars: Reference atom name chars dim
        add_noisy_pos: Whether to embed noisy positions and trunk reps
        c_s: Single representation dim (optional, needed if add_noisy_pos)
        c_z: Pair representation dim (optional, needed if add_noisy_pos)
        c_hidden: Per-head hidden dim for atom transformer
        no_heads: Number of attention heads
        no_blocks: Number of transformer blocks
        n_transition: Transition blocks per transformer block
        n_query: Block height for sequence-local attention
        n_key: Block width for sequence-local attention
        use_ada_layer_norm: Whether to use AdaLN
    """

    def __init__(
        self,
        c_atom: int = 128,
        c_atom_pair: int = 16,
        c_token: int = 384,
        c_atom_ref_element: int = 119,
        c_atom_ref_name_chars: int = 256,
        add_noisy_pos: bool = False,
        c_s: int | None = None,
        c_z: int | None = None,
        c_hidden: int = 32,
        no_heads: int = 4,
        no_blocks: int = 3,
        n_transition: int = 2,
        n_query: int = 32,
        n_key: int = 128,
        use_ada_layer_norm: bool = True,
        transformer_cls=None,
    ):
        super().__init__()
        if transformer_cls is None:
            transformer_cls = DiffusionTransformer

        self.n_query = n_query
        self.n_key = n_key

        self.ref_atom_feature_embedder = RefAtomFeatureEmbedder(
            c_atom_ref_element=c_atom_ref_element,
            c_atom_ref_name_chars=c_atom_ref_name_chars,
            c_atom=c_atom,
            c_atom_pair=c_atom_pair,
        )

        self.noisy_position_embedder: NoisyPositionEmbedder | None = None
        if add_noisy_pos:
            assert c_s is not None and c_z is not None
            self.noisy_position_embedder = NoisyPositionEmbedder(
                c_s=c_s, c_z=c_z, c_atom=c_atom, c_atom_pair=c_atom_pair,
            )

        self.relu = ReLU()
        self.linear_l = Linear(c_atom, c_atom_pair, bias=False)
        self.linear_m = Linear(c_atom, c_atom_pair, bias=False)

        self.pair_mlp = nn.Sequential(
            ReLU(),
            Linear(c_atom_pair, c_atom_pair, bias=False),
            ReLU(),
            Linear(c_atom_pair, c_atom_pair, bias=False),
            ReLU(),
            Linear(c_atom_pair, c_atom_pair, bias=False),
        )

        self.atom_transformer = transformer_cls(
            c_a=c_atom, c_s=c_atom, c_z=c_atom_pair,
            c_hidden=c_hidden, no_heads=no_heads,
            no_blocks=no_blocks, n_transition=n_transition,
            use_ada_layer_norm=use_ada_layer_norm,
            n_query=n_query, n_key=n_key,
        )

        self.linear_q = nn.Sequential(
            Linear(c_atom, c_token, bias=False),
            ReLU(),
        )

    def forward(
        self,
        batch: dict,
        rl: torch.Tensor | None = None,
        si_trunk: torch.Tensor | None = None,
        zij_trunk: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            ai: [*, N_token, c_token] token representation
            ql: [*, N_atom, c_atom] atom single representation
            cl: [*, N_atom, c_atom] atom single conditioning
            plm: [*, N_blocks, n_query, n_key, c_atom_pair] atom pair rep
        """
        atom_mask = batch["atom_mask"]

        cl, plm = self.ref_atom_feature_embedder(
            batch=batch, n_query=self.n_query, n_key=self.n_key,
        )

        if rl is not None and self.noisy_position_embedder is not None:
            cl, plm, ql = self.noisy_position_embedder(
                batch=batch, cl=cl, plm=plm,
                si_trunk=si_trunk, zij_trunk=zij_trunk, rl=rl,
                n_query=self.n_query, n_key=self.n_key,
            )
        else:
            ql = cl.clone()

        cl_l, cl_m, block_mask = _convert_single_rep_to_blocks(
            ql=cl, n_query=self.n_query, n_key=self.n_key, atom_mask=atom_mask,
        )

        cl_lm = (
            self.linear_l(self.relu(cl_l.unsqueeze(-2)))
            + self.linear_m(self.relu(cl_m.unsqueeze(-3)))
        )
        if block_mask is not None:
            cl_lm = cl_lm * block_mask.unsqueeze(-1)

        plm = plm + cl_lm
        plm = plm + self.pair_mlp(plm)
        if block_mask is not None:
            plm = plm * block_mask.unsqueeze(-1)

        ql = self.atom_transformer(
            a=ql, s=cl, z=plm, mask=atom_mask,
        )

        ql = ql * atom_mask.unsqueeze(-1)

        atom_proj = self.linear_q(ql)

        if "atom_to_token_index" in batch:
            ai = _aggregate_atom_feat_to_tokens(
                token_mask=batch["token_mask"],
                atom_to_token_index=batch["atom_to_token_index"],
                atom_mask=atom_mask,
                atom_feat=atom_proj,
                mode="mean",
            )
        else:
            ai = atom_proj

        return ai, ql, cl, plm


class AtomAttentionDecoder(nn.Module):
    """AF3 Algorithm 6: Atom attention decoder.

    Args:
        c_atom: Atom single representation channel dimension
        c_atom_pair: Atom pair representation channel dimension
        c_token: Token diffusion channel dimension
        c_hidden: Per-head hidden dim
        no_heads: Number of attention heads
        no_blocks: Number of transformer blocks
        n_transition: Transition blocks per transformer block
        n_query: Block height
        n_key: Block width
        use_ada_layer_norm: Whether to use AdaLN
    """

    def __init__(
        self,
        c_atom: int = 128,
        c_atom_pair: int = 16,
        c_token: int = 768,
        c_hidden: int = 32,
        no_heads: int = 4,
        no_blocks: int = 3,
        n_transition: int = 2,
        n_query: int = 32,
        n_key: int = 128,
        use_ada_layer_norm: bool = True,
        transformer_cls=None,
    ):
        super().__init__()
        if transformer_cls is None:
            transformer_cls = DiffusionTransformer

        self.linear_q_in = Linear(c_token, c_atom, bias=False)

        self.atom_transformer = transformer_cls(
            c_a=c_atom, c_s=c_atom, c_z=c_atom_pair,
            c_hidden=c_hidden, no_heads=no_heads,
            no_blocks=no_blocks, n_transition=n_transition,
            use_ada_layer_norm=use_ada_layer_norm,
            n_query=n_query, n_key=n_key,
        )

        self.layer_norm = LayerNorm(c_atom, create_offset=False)
        self.linear_q_out = Linear(c_atom, 3, bias=False)

    def forward(
        self,
        batch: dict,
        ai: torch.Tensor,
        ql: torch.Tensor,
        cl: torch.Tensor,
        plm: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns:
            rl_update: [*, N_atom, 3] atom position updates
        """
        ai_broadcast = _broadcast_token_feat_to_atoms(
            token_mask=batch["token_mask"],
            num_atoms_per_token=batch.get("num_atoms_per_token"),
            token_feat=self.linear_q_in(ai),
            atom_to_token_index=batch.get("atom_to_token_index"),
        )
        ql = ql + ai_broadcast

        ql = self.atom_transformer(
            a=ql, s=cl, z=plm, mask=batch["atom_mask"],
        )

        rl_update = self.linear_q_out(self.layer_norm(ql))

        return rl_update


# Inlined from tasks/baseline/L2/alphafold3_input_embedder.py
def _binned_one_hot(
    x: torch.Tensor, boundaries: torch.Tensor,
) -> torch.Tensor:
    """One-hot encoding with bin boundaries (matches reference binned_one_hot)."""
    return (x[..., None] > boundaries).to(dtype=x.dtype)


def relpos_complex(
    batch: dict,
    max_relative_idx: int,
    max_relative_chain: int,
) -> torch.Tensor:
    """Build relative position features matching the reference implementation.

    Produces 139 features when max_relative_idx=32, max_relative_chain=2:
      66 (rel_pos) + 66 (rel_token) + 1 (same_entity) + 6 (rel_chain)

    Reference: openfold3/core/utils/relpos.py relpos_complex
    """
    res_idx = batch["residue_index"]
    asym_id = batch["asym_id"]
    entity_id = batch["entity_id"]
    same_chain = asym_id[..., None] == asym_id[..., None, :]
    same_res = res_idx[..., None] == res_idx[..., None, :]
    same_entity = entity_id[..., None] == entity_id[..., None, :]

    def _relpos(
        pos: torch.Tensor, condition: torch.BoolTensor, rel_clip_idx: int,
    ) -> torch.Tensor:
        offset = pos[..., None] - pos[..., None, :]
        clipped_offset = torch.clamp(offset + rel_clip_idx, min=0, max=2 * rel_clip_idx)
        final_offset = torch.where(
            condition,
            clipped_offset,
            (2 * rel_clip_idx + 1) * torch.ones_like(clipped_offset),
        )
        boundaries = torch.arange(
            start=0, end=2 * rel_clip_idx + 2, device=final_offset.device,
        ).to(dtype=final_offset.dtype)
        return _binned_one_hot(final_offset, boundaries)

    rel_pos = _relpos(pos=res_idx, condition=same_chain, rel_clip_idx=max_relative_idx)
    rel_token = _relpos(
        pos=batch["token_index"],
        condition=same_chain & same_res,
        rel_clip_idx=max_relative_idx,
    )
    rel_chain = _relpos(
        pos=batch["sym_id"],
        condition=same_entity,
        rel_clip_idx=max_relative_chain,
    )

    same_entity_feat = same_entity[..., None].to(dtype=rel_pos.dtype)

    return torch.cat([rel_pos, rel_token, same_entity_feat, rel_chain], dim=-1)


class InputEmbedder(nn.Module):
    """Produces initial single and pair representations from token features.

    Matches InputEmbedderAllAtom: runs AtomAttentionEncoder to get a
    token-level representation, concatenates with restype/profile/deletion_mean
    to form s_input (449 dims), then projects to s and z.

    Args:
        c_s_input: Input single representation dimension (449 for all-atom)
        c_s: Single representation dimension
        c_z: Pair representation dimension
        relpos_k: Maximum relative residue position
        max_relative_chain: Maximum relative chain index
        c_atom: Atom single representation dim
        c_atom_pair: Atom pair representation dim
        c_token: Token dim for atom attention encoder output
    """

    def __init__(
        self,
        c_s_input: int,
        c_s: int,
        c_z: int,
        relpos_k: int = 32,
        max_relative_chain: int = 2,
        c_atom: int = 128,
        c_atom_pair: int = 16,
        c_token: int | None = None,
    ):
        super().__init__()
        self.c_s_input = c_s_input
        self.c_s = c_s
        self.c_z = c_z
        self.relpos_k = relpos_k
        self.max_relative_chain = max_relative_chain
        self._one_hot = OneHot()
        self._pad = Pad()

        if c_token is None:
            c_token = c_s

        self.atom_attn_enc = AtomAttentionEncoder(
            c_atom=c_atom,
            c_atom_pair=c_atom_pair,
            c_token=c_token,
            add_noisy_pos=False,
        )

        self.linear_s = Linear(c_s_input, c_s, bias=False)
        self.linear_z_i = Linear(c_s_input, c_z, bias=False)
        self.linear_z_j = Linear(c_s_input, c_z, bias=False)

        num_rel_pos_bins = 2 * relpos_k + 2
        num_rel_token_bins = 2 * relpos_k + 2
        num_rel_chain_bins = 2 * max_relative_chain + 2
        num_same_entity_features = 1
        n_relpos_features = (
            num_rel_pos_bins + num_rel_token_bins
            + num_rel_chain_bins + num_same_entity_features
        )
        self.linear_relpos = Linear(n_relpos_features, c_z, bias=False)

        self.linear_token_bonds = Linear(1, c_z, bias=False)

    def forward(
        self,
        token_features: torch.Tensor,
        residue_index: torch.Tensor,
        batch: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            token_features: [*, N_token, c_s_input] per-token features.
                If batch contains ref_pos (atom features), only restype/profile/deletion_mean
                are expected here and atom_attn_enc produces the remaining features.
                Otherwise, treated as pre-built s_input.
            residue_index:  [*, N_token] residue indices
            batch: Feature dict for relpos and atom attention.

        Returns:
            s_input: [*, N_token, c_s_input] input single representation
            s: [*, N_token, C_s] single representation
            z: [*, N_token, N_token, C_z] pair representation
        """
        if batch is not None and "ref_pos" in batch:
            a, _, _, _ = self.atom_attn_enc(batch=batch)
            s_input = torch.cat(
                [
                    a,
                    batch.get("restype", token_features[..., :32]),
                    batch.get("profile", token_features[..., 32:64]),
                    batch.get("deletion_mean", token_features[..., -1:]).unsqueeze(-1)
                    if batch.get("deletion_mean") is not None and batch["deletion_mean"].dim() == token_features.dim() - 1
                    else batch.get("deletion_mean", token_features[..., -1:]),
                ],
                dim=-1,
            )
        else:
            s_input = token_features

        s = self.linear_s(s_input)

        z_i = self.linear_z_i(s_input)[..., :, None, :]
        z_j = self.linear_z_j(s_input)[..., None, :, :]
        z = z_i + z_j

        if batch is not None and "asym_id" in batch:
            relpos_feats = relpos_complex(
                batch=batch,
                max_relative_idx=self.relpos_k,
                max_relative_chain=self.max_relative_chain,
            ).to(dtype=z.dtype)
        else:
            d = residue_index[..., :, None] - residue_index[..., None, :]
            d = d.clamp(-self.relpos_k, self.relpos_k) + self.relpos_k
            n_bins = 2 * self.relpos_k + 2
            relpos_feats = self._one_hot(d.long(), n_bins).to(
                dtype=z.dtype,
            )
            n_relpos_in = self.linear_relpos.weight.shape[-1]
            if relpos_feats.shape[-1] < n_relpos_in:
                pad_size = n_relpos_in - relpos_feats.shape[-1]
                relpos_feats = self._pad(relpos_feats, (0, pad_size))

        z = z + self.linear_relpos(relpos_feats)

        if batch is not None and "token_bonds" in batch:
            token_bonds_emb = self.linear_token_bonds(
                batch["token_bonds"].unsqueeze(-1).to(dtype=s.dtype)
            )
            z = z + token_bonds_emb

        return s_input, s, z


# Inlined from tasks/baseline/L2/alphafold3_msa_module_embedder.py
class MSAModuleEmbedder(nn.Module):
    """AF3 Algorithm 8, lines 1-4: MSA feature embedding.

    Args:
        c_m_feats: MSA input features channel dimension (34 = 32 msa + has_deletion + deletion_value)
        c_m: MSA channel dimension
        c_s_input: Single (s_input) channel dimension
    """

    def __init__(
        self,
        c_m_feats: int = 34,
        c_m: int = 64,
        c_s_input: int = 449,
    ):
        super().__init__()
        self.linear_m = Linear(c_m_feats, c_m, bias=False)
        self.linear_s_input = Linear(c_s_input, c_m, bias=False)

    def forward(
        self,
        batch: dict,
        s_input: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            batch: needs msa [*, N_msa, N_token, 32],
                   has_deletion [*, N_msa, N_token],
                   deletion_value [*, N_msa, N_token],
                   msa_mask [*, N_msa, N_token]
            s_input: [*, N_token, c_s_input]

        Returns:
            m: [*, N_seq, N_token, c_m]
            msa_mask: [*, N_seq, N_token]
        """
        msa_feat = torch.cat(
            [
                batch["msa"],
                batch["has_deletion"].unsqueeze(-1),
                batch["deletion_value"].unsqueeze(-1),
            ],
            dim=-1,
        )
        msa_mask = batch["msa_mask"]

        m = self.linear_m(msa_feat)
        m = m + self.linear_s_input(s_input).unsqueeze(-3)

        return m, msa_mask


# Inlined from tasks/baseline/L2/alphafold3_triangle_multiplication.py
def _permute_final_dims(tensor: torch.Tensor, inds: tuple[int, ...]) -> torch.Tensor:
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


class TriangleMultiplicativeUpdate(nn.Module):
    """AF3 Algorithms 12/13: Triangle multiplicative update.

    Args:
        c_z: Input channel dimension
        c_hidden: Hidden channel dimension
        _outgoing: If True, outgoing (Alg 12); else incoming (Alg 13)
    """

    def __init__(self, c_z: int, c_hidden: int, _outgoing: bool = True):
        super().__init__()
        self.c_z = c_z
        self.c_hidden = c_hidden
        self._outgoing = _outgoing

        self.linear_a_p = Linear(c_z, c_hidden, bias=False)
        self.linear_a_g = Linear(c_z, c_hidden, bias=False)
        self.linear_b_p = Linear(c_z, c_hidden, bias=False)
        self.linear_b_g = Linear(c_z, c_hidden, bias=False)

        self.linear_g = Linear(c_z, c_z, bias=False)
        self.linear_z = Linear(c_hidden, c_z, bias=False)

        self.layer_norm_in = LayerNorm(c_z)
        self.layer_norm_out = LayerNorm(c_hidden)

    def _combine_projections(
        self, a: torch.Tensor, b: torch.Tensor,
    ) -> torch.Tensor:
        if self._outgoing:
            a = _permute_final_dims(a, (2, 0, 1))
            b = _permute_final_dims(b, (2, 1, 0))
        else:
            a = _permute_final_dims(a, (2, 1, 0))
            b = _permute_final_dims(b, (2, 0, 1))

        p = torch.einsum("...ij,...jk->...ik", a, b)
        return _permute_final_dims(p, (1, 2, 0))

    def forward(
        self,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
        inplace_safe: bool = False,
        use_cueq_triangle_kernels: bool = False,
        _add_with_inplace: bool = False,
        _inplace_chunk_size: int | None = 256,
    ) -> torch.Tensor:
        """
        Args:
            z:    [*, N_res, N_res, C_z] input tensor
            mask: [*, N_res, N_res] input mask

        Returns:
            [*, N_res, N_res, C_z] output tensor
        """
        if mask is None:
            mask = z.new_ones(z.shape[:-1])

        mask = mask.unsqueeze(-1)

        z_ln = self.layer_norm_in(z)

        a = mask * torch.sigmoid(self.linear_a_g(z_ln)) * self.linear_a_p(z_ln)
        b = mask * torch.sigmoid(self.linear_b_g(z_ln)) * self.linear_b_p(z_ln)

        x = self._combine_projections(a, b)

        del a, b
        x = self.layer_norm_out(x)
        x = self.linear_z(x)
        x = x * torch.sigmoid(self.linear_g(z_ln))

        return x


class TriangleMultiplicationOutgoing(TriangleMultiplicativeUpdate):
    """AF3 Algorithm 12."""

    def __init__(self, c_z: int, c_hidden: int):
        super().__init__(c_z=c_z, c_hidden=c_hidden, _outgoing=True)


class TriangleMultiplicationIncoming(TriangleMultiplicativeUpdate):
    """AF3 Algorithm 13."""

    def __init__(self, c_z: int, c_hidden: int):
        super().__init__(c_z=c_z, c_hidden=c_hidden, _outgoing=False)


# Inlined from tasks/baseline/L2/alphafold3_triangle_attention.py
def _permute_final_dims(tensor: torch.Tensor, inds: tuple[int, ...]) -> torch.Tensor:
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


class TriangleAttention(nn.Module):
    """AF3 Algorithms 14/15: Triangle attention.

    Args:
        c_in: Input channel dimension
        c_hidden: Overall hidden channel dimension (not per-head)
        no_heads: Number of attention heads
        starting: If True, starting node (Alg 14); else ending node (Alg 15)
        inf: Large constant for masking
    """

    def __init__(
        self,
        c_in: int,
        c_hidden: int,
        no_heads: int,
        starting: bool = True,
        inf: float = 1e9,
    ):
        super().__init__()
        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.starting = starting
        self.inf = inf

        self.layer_norm = LayerNorm(c_in)
        self.linear_z = Linear(c_in, no_heads, bias=False)

        self.mha = OF3Attention(
            c_q=c_in,
            c_k=c_in,
            c_v=c_in,
            c_hidden=c_hidden,
            no_heads=no_heads,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        chunk_size: int | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: [*, I, J, C_in] input tensor (pair representation)

        Returns:
            [*, I, J, C_in] output tensor
        """
        if mask is None:
            mask = x.new_ones(x.shape[:-1])

        if not self.starting:
            x = x.transpose(-2, -3)
            mask = mask.transpose(-1, -2)

        x = self.layer_norm(x)

        # [*, I, 1, 1, J]
        mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]

        # [*, H, I, J] -> [*, 1, H, I, J]
        triangle_bias = _permute_final_dims(self.linear_z(x), (2, 0, 1))
        triangle_bias = triangle_bias.unsqueeze(-4)

        biases = [mask_bias, triangle_bias]

        x = self.mha(q_x=x, kv_x=x, biases=biases)

        if not self.starting:
            x = x.transpose(-2, -3)

        return x


TriangleAttentionStartingNode = TriangleAttention


class TriangleAttentionEndingNode(TriangleAttention):
    """AF3 Algorithm 15."""

    def __init__(self, c_in: int, c_hidden: int, no_heads: int, inf: float = 1e9):
        super().__init__(c_in=c_in, c_hidden=c_hidden, no_heads=no_heads,
                         starting=False, inf=inf)


# Inlined from tasks/baseline/L2/alphafold3_pair_block.py
class PairBlock(nn.Module):
    """Shared pair stack block for AF3 PairFormer / MSA module / template.

    Args:
        c_z: Pair embedding channel dimension
        c_hidden_mul: Hidden dim for triangle multiplication
        c_hidden_pair_att: Per-head hidden dim for triangle attention
        no_heads_pair: Number of heads in triangle attention
        transition_n: Scale of pair transition hidden dimension
        pair_dropout: Dropout rate (unused in inference baseline)
        inf: Large constant for masking
    """

    def __init__(
        self,
        c_z: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_pair: int,
        transition_n: int,
        pair_dropout: float = 0.0,
        fuse_projection_weights: bool = False,
        inf: float = 1e9,
    ):
        super().__init__()

        self.tri_mul_out = TriangleMultiplicationOutgoing(c_z, c_hidden_mul)
        self.tri_mul_in = TriangleMultiplicationIncoming(c_z, c_hidden_mul)

        self.tri_att_start = TriangleAttention(
            c_z, c_hidden_pair_att, no_heads_pair, starting=True, inf=inf,
        )
        self.tri_att_end = TriangleAttention(
            c_z, c_hidden_pair_att, no_heads_pair, starting=False, inf=inf,
        )

        self.pair_transition = SwiGLUTransition(c_in=c_z, n=transition_n)

    def forward(
        self,
        z: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: int | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
        _attn_chunk_size: int | None = None,
    ) -> torch.Tensor:
        """
        Args:
            z:         [*, N, N, C_z] pair embedding
            pair_mask: [*, N, N] pair mask

        Returns:
            [*, N, N, C_z] updated pair embedding
        """
        pair_trans_mask = pair_mask if _mask_trans else None

        # Triangle multiplicative updates
        z = z + self.tri_mul_out(z, mask=pair_mask)
        z = z + self.tri_mul_in(z, mask=pair_mask)

        # Triangle attention (start)
        z = z + self.tri_att_start(z, mask=pair_mask)

        # Triangle attention (end) -- uses transposed mask internally
        z = z + self.tri_att_end(z, mask=pair_mask)

        # Pair transition
        z = z + self.pair_transition(z, mask=pair_trans_mask)

        return z


# Inlined from tasks/baseline/L2/alphafold3_template_embedder.py
class TemplatePairEmbedder(nn.Module):
    """Embeds template pair features (Algorithm 16, lines 1-5).

    Args:
        c_in: Pair embedding input dimension (c_z)
        c_dgram: Distogram feature dim
        c_aatype: Template aatype feature dim
        c_out: Output template embedding dim (c_t)
    """

    def __init__(
        self,
        c_in: int = 128,
        c_dgram: int = 39,
        c_aatype: int = 32,
        c_out: int = 64,
    ):
        super().__init__()
        self.dgram_linear = Linear(c_dgram, c_out, bias=False)
        self.aatype_linear_1 = Linear(c_aatype, c_out, bias=False)
        self.aatype_linear_2 = Linear(c_aatype, c_out, bias=False)
        self.pseudo_beta_mask_linear = Linear(1, c_out, bias=False)
        self.x_linear = Linear(1, c_out, bias=False)
        self.y_linear = Linear(1, c_out, bias=False)
        self.z_linear = Linear(1, c_out, bias=False)
        self.backbone_mask_linear = Linear(1, c_out, bias=False)

        self.layer_norm_z = LayerNorm(c_in)
        self.linear_z = Linear(c_in, c_out, bias=False)

    def forward(
        self, batch: dict, z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            batch: needs template_distogram, template_restype,
                   template_pseudo_beta_mask, template_unit_vector,
                   template_backbone_frame_mask, asym_id
            z: [*, N_token, N_token, C_z]

        Returns:
            [*, N_templ, N_token, N_token, C_t]
        """
        dtype = batch["template_unit_vector"].dtype

        multichain_pair_mask = (
            batch["asym_id"][..., None] == batch["asym_id"][..., None, :]
        )
        multichain_pair_mask = multichain_pair_mask[..., None, :, :, None]

        pseudo_beta_pair_mask = (
            batch["template_pseudo_beta_mask"][..., None]
            * batch["template_pseudo_beta_mask"][..., None, :]
        )[..., None] * multichain_pair_mask

        backbone_frame_pair_mask = (
            batch["template_backbone_frame_mask"][..., None]
            * batch["template_backbone_frame_mask"][..., None, :]
        )[..., None] * multichain_pair_mask

        template_unit_vector = batch["template_unit_vector"]
        x, y, z_coord = template_unit_vector.unbind(dim=-1)

        template_restype = batch["template_restype"]
        n_token = template_restype.shape[-2]
        template_restype_ti = template_restype[..., None, :].expand(
            *template_restype.shape[:-2], -1, n_token, -1
        )
        template_restype_tj = template_restype[..., None, :, :].expand(
            *template_restype.shape[:-2], n_token, -1, -1
        )

        t = (
            self.dgram_linear(batch["template_distogram"].to(dtype=dtype))
            * pseudo_beta_pair_mask
        )
        t = t + self.aatype_linear_1(template_restype_ti.to(dtype=dtype))
        t = t + self.aatype_linear_2(template_restype_tj.to(dtype=dtype))
        t = t + (
            self.pseudo_beta_mask_linear(
                batch["template_pseudo_beta_mask"][..., None, None].to(dtype=dtype).expand(
                    *batch["template_pseudo_beta_mask"].shape, n_token, 1
                )
            )
            * multichain_pair_mask
        )
        t = t + self.x_linear(x[..., None]) * backbone_frame_pair_mask
        t = t + self.y_linear(y[..., None]) * backbone_frame_pair_mask
        t = t + self.z_linear(z_coord[..., None]) * backbone_frame_pair_mask
        t = t + (
            self.backbone_mask_linear(
                batch["template_backbone_frame_mask"][..., None, None].to(dtype=dtype).expand(
                    *batch["template_backbone_frame_mask"].shape, n_token, 1
                )
            )
            * multichain_pair_mask
        )

        z_emb = self.linear_z(self.layer_norm_z(z))
        t = t + z_emb[..., None, :, :, :]

        return t


class TemplatePairStackBlock(PairBlock):
    """Single block of the template pair stack.

    Inherits directly from PairBlock to avoid extra key nesting.
    """

    def __init__(
        self,
        c_t: int = 64,
        c_hidden_tri_att: int = 16,
        c_hidden_tri_mul: int = 64,
        no_heads: int = 4,
        transition_n: int = 2,
        dropout_rate: float = 0.25,
        tri_mul_first: bool = True,
        inf: float = 1e9,
    ):
        super().__init__(
            c_z=c_t,
            c_hidden_mul=c_hidden_tri_mul,
            c_hidden_pair_att=c_hidden_tri_att,
            no_heads_pair=no_heads,
            transition_n=transition_n,
            pair_dropout=dropout_rate,
            inf=inf,
        )


class TemplatePairStack(nn.Module):
    """Template pair stack.

    Args:
        c_t: Template pair dim
        c_hidden_tri_att: Triangle attention hidden dim
        c_hidden_tri_mul: Triangle multiplication hidden dim
        no_blocks: Number of blocks
        no_heads: Number of attention heads
        transition_n: Transition block scale
        dropout_rate: Dropout rate
        tri_mul_first: Triangle multiplication before attention
    """

    def __init__(
        self,
        c_t: int = 64,
        c_hidden_tri_att: int = 16,
        c_hidden_tri_mul: int = 64,
        no_blocks: int = 2,
        no_heads: int = 4,
        transition_n: int = 2,
        dropout_rate: float = 0.25,
        tri_mul_first: bool = True,
        inf: float = 1e9,
        **kwargs,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            TemplatePairStackBlock(
                c_t=c_t,
                c_hidden_tri_att=c_hidden_tri_att,
                c_hidden_tri_mul=c_hidden_tri_mul,
                no_heads=no_heads,
                transition_n=transition_n,
                dropout_rate=dropout_rate,
                tri_mul_first=tri_mul_first,
                inf=inf,
            )
            for _ in range(no_blocks)
        ])
        self.layer_norm = LayerNorm(c_t)

    def forward(
        self,
        t: torch.Tensor,
        pair_mask: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        n_templ = t.shape[-4]
        for block in self.blocks:
            for i in range(n_templ):
                t[..., i, :, :, :] = block(
                    z=t[..., i, :, :, :], pair_mask=pair_mask[..., 0, :, :],
                )
        return self.layer_norm(t)


class TemplateEmbedder(nn.Module):
    """AF3 Algorithm 16: Template embedder.

    Args:
        c_t: Template pair embedding dim
        c_z: Pair representation dim
        c_dgram: Distogram feature dim
        c_aatype: Template aatype feature dim
        c_hidden_tri_att: Triangle attention hidden dim
        c_hidden_tri_mul: Triangle multiplication hidden dim
        no_blocks: Number of pair stack blocks
        no_heads: Number of attention heads
        transition_n: Transition scale
        dropout_rate: Dropout rate
    """

    def __init__(
        self,
        c_t: int = 64,
        c_z: int = 128,
        c_dgram: int = 39,
        c_aatype: int = 32,
        c_hidden_tri_att: int = 16,
        c_hidden_tri_mul: int = 64,
        no_blocks: int = 2,
        no_heads: int = 4,
        transition_n: int = 2,
        dropout_rate: float = 0.25,
        tri_mul_first: bool = True,
        inf: float = 1e9,
    ):
        super().__init__()
        self.relu = ReLU()
        self.template_pair_embedder = TemplatePairEmbedder(
            c_in=c_z, c_dgram=c_dgram, c_aatype=c_aatype, c_out=c_t,
        )
        self.template_pair_stack = TemplatePairStack(
            c_t=c_t,
            c_hidden_tri_att=c_hidden_tri_att,
            c_hidden_tri_mul=c_hidden_tri_mul,
            no_blocks=no_blocks,
            no_heads=no_heads,
            transition_n=transition_n,
            dropout_rate=dropout_rate,
            tri_mul_first=tri_mul_first,
            inf=inf,
        )
        self.linear_t = Linear(c_t, c_z, bias=False)

    def forward(
        self,
        batch: dict,
        z: torch.Tensor,
        pair_mask: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            batch: Template features
            z: [*, N_token, N_token, C_z] pair embedding
            pair_mask: [*, N_token, N_token] pair mask

        Returns:
            [*, N_token, N_token, C_z] template embedding added to pair rep
        """
        template_embeds = self.template_pair_embedder(batch, z)
        n_templ = template_embeds.shape[-4]

        pair_mask_4d = pair_mask[..., None, :, :].to(dtype=z.dtype)

        t = self.template_pair_stack(
            template_embeds, pair_mask_4d,
        )

        t = torch.sum(t, dim=-4) / n_templ
        t = self.relu(t)
        t = self.linear_t(t)

        return t


# Inlined from tasks/baseline/L2/alphafold3_diffusion_conditioning.py
class FourierEmbedding(nn.Module):
    """Fourier time embedding for diffusion conditioning.

    Uses random Fourier features (matching the reference's seeded initialization).

    Args:
        c: Embedding dimension (256 in the reference)
        seed: Random seed for weight initialization
    """

    def __init__(self, c: int = 256, seed: int = 42):
        super().__init__()
        self.c = c
        generator = torch.Generator()
        generator.manual_seed(seed)
        self.register_buffer(
            "w", torch.randn(c, generator=generator),
        )
        self.register_buffer(
            "b", torch.randn(c, generator=generator),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        x = t * self.w + self.b
        return torch.cos(2 * math.pi * x)


class DiffusionConditioning(nn.Module):
    """Conditioning for diffusion module.

    Matches the reference:
    - Pair: concat([zij_trunk, relpos], dim=-1) -> LayerNorm -> Linear -> 2x SwiGLU transition
    - Single: concat([si_trunk, si_input], dim=-1) -> LayerNorm -> Linear + fourier -> 2x SwiGLU transition

    Reference: openfold3/core/model/layers/diffusion_conditioning.py

    Args:
        c_s: Single representation channel dimension
        c_z: Pair representation channel dimension
        c_s_input: Input single representation dimension (449)
        sigma_data: Noise level scaling for Fourier embedding
        relpos_k: Maximum relative position for pair bias
        max_relative_chain: Maximum relative chain index
        c_fourier_emb: Fourier embedding dimension (256)
        seed_fourier_emb: Fourier embedding random seed
    """

    def __init__(
        self,
        c_s: int = 384,
        c_z: int = 128,
        c_s_input: int = 449,
        sigma_data: float = 16.0,
        relpos_k: int = 32,
        max_relative_chain: int = 2,
        c_fourier_emb: int = 256,
        seed_fourier_emb: int = 42,
    ):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.c_s_input = c_s_input
        self.c_fourier_emb = c_fourier_emb
        self.sigma_data = sigma_data
        self.relpos_k = relpos_k
        self.max_relative_chain = max_relative_chain

        num_rel_pos_bins = 2 * relpos_k + 2
        num_rel_token_bins = 2 * relpos_k + 2
        num_rel_chain_bins = 2 * max_relative_chain + 2
        num_same_entity_features = 1
        num_relpos_dims = (
            num_rel_pos_bins + num_rel_token_bins
            + num_rel_chain_bins + num_same_entity_features
        )

        self.layer_norm_z = LayerNorm(num_relpos_dims + c_z, create_offset=False)
        self.linear_z = Linear(num_relpos_dims + c_z, c_z, bias=False)

        self.transition_z = nn.ModuleList([
            SwiGLUTransition(c_in=c_z, n=2)
            for _ in range(2)
        ])

        self.layer_norm_s = LayerNorm(c_s + c_s_input, create_offset=False)
        self.linear_s = Linear(c_s + c_s_input, c_s, bias=False)

        self.fourier_emb = FourierEmbedding(c=c_fourier_emb, seed=seed_fourier_emb)
        self.layer_norm_n = LayerNorm(c_fourier_emb, create_offset=False)
        self.linear_n = Linear(c_fourier_emb, c_s, bias=False)

        self.transition_s = nn.ModuleList([
            SwiGLUTransition(c_in=c_s, n=2)
            for _ in range(2)
        ])

    def forward(
        self,
        batch: dict,
        t: torch.Tensor,
        si_input: torch.Tensor,
        si_trunk: torch.Tensor,
        zij_trunk: torch.Tensor,
        use_conditioning: bool,
        chunk_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            batch:     Feature dictionary (needs asym_id, entity_id etc. for relpos)
            t:         [*] noise level
            si_input:  [*, N_token, c_s_input] input embedding
            si_trunk:  [*, N_token, c_s] trunk single rep
            zij_trunk: [*, N_token, N_token, c_z] trunk pair rep
            use_conditioning: Whether to condition with trunk reps

        Returns:
            si:  [*, N_token, c_s] conditioned single rep
            zij: [*, N_token, N_token, c_z] conditioned pair rep
        """
        if use_conditioning:
            # Pair conditioning: concat trunk pair with relpos features
            if "asym_id" in batch:
                relpos_zij = relpos_complex(
                    batch=batch,
                    max_relative_idx=self.relpos_k,
                    max_relative_chain=self.max_relative_chain,
                ).to(dtype=zij_trunk.dtype)
            else:
                relpos_dim = self.linear_z.weight.shape[-1] - self.c_z
                relpos_zij = zij_trunk.new_zeros(
                    zij_trunk.shape[:-1] + (relpos_dim,),
                )

            zij = torch.cat([zij_trunk, relpos_zij], dim=-1)
            zij = self.linear_z(self.layer_norm_z(zij))

            # Single conditioning: concat trunk single with input
            si = torch.cat([si_trunk, si_input], dim=-1)
            si = self.linear_s(self.layer_norm_s(si))
        else:
            zij = zij_trunk.new_zeros(zij_trunk.shape)
            si = si_trunk.new_zeros(si_trunk.shape[:-1] + (self.c_s,))

        # Fourier noise embedding
        n = 0.25 * torch.log(t / self.sigma_data)
        n_emb = self.fourier_emb(n.unsqueeze(-1) if n.dim() == 0 else n)
        si = si + self.linear_n(self.layer_norm_n(n_emb)).unsqueeze(-2)

        # Apply transition layers
        token_mask = batch.get("token_mask")
        if token_mask is not None:
            pair_mask = token_mask[..., :, None] * token_mask[..., None, :]
        else:
            pair_mask = None

        for layer in self.transition_z:
            zij = zij + layer(zij, mask=pair_mask)

        for layer in self.transition_s:
            si = si + layer(si, mask=token_mask)

        return si, zij


# Inlined from tasks/baseline/L3/alphafold3_diffusion_module.py
def _quat_to_rot(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternion [*, 4] to rotation matrix [*, 3, 3]."""
    q = quat[..., None] * quat[..., None, :]
    r00 = q[..., 0, 0] + q[..., 1, 1] - q[..., 2, 2] - q[..., 3, 3]
    r01 = 2 * (q[..., 1, 2] - q[..., 0, 3])
    r02 = 2 * (q[..., 1, 3] + q[..., 0, 2])
    r10 = 2 * (q[..., 1, 2] + q[..., 0, 3])
    r11 = q[..., 0, 0] - q[..., 1, 1] + q[..., 2, 2] - q[..., 3, 3]
    r12 = 2 * (q[..., 2, 3] - q[..., 0, 1])
    r20 = 2 * (q[..., 1, 3] - q[..., 0, 2])
    r21 = 2 * (q[..., 2, 3] + q[..., 0, 1])
    r22 = q[..., 0, 0] - q[..., 1, 1] - q[..., 2, 2] + q[..., 3, 3]
    return torch.stack([
        torch.stack([r00, r01, r02], dim=-1),
        torch.stack([r10, r11, r12], dim=-1),
        torch.stack([r20, r21, r22], dim=-1),
    ], dim=-2)


def centre_random_augmentation(
    xl: torch.Tensor, atom_mask: torch.Tensor, scale_trans: float = 1.0,
) -> torch.Tensor:
    """AF3 Algorithm 19: random rotation + translation + centering."""
    q = torch.randn(*xl.shape[:-2], 4, dtype=xl.dtype, device=xl.device)
    q = q / torch.linalg.norm(q, dim=-1, keepdim=True)
    rots = _quat_to_rot(q)

    trans = scale_trans * torch.randn(
        (*xl.shape[:-2], 3), dtype=xl.dtype, device=xl.device,
    )

    mask_sum = torch.sum(atom_mask[..., None], dim=-2, keepdim=True).clamp(min=1.0)
    mean_xl = torch.sum(xl * atom_mask[..., None], dim=-2, keepdim=True) / mask_sum
    pos_centered = xl - mean_xl
    pos_out = pos_centered @ rots.transpose(-1, -2) + trans[..., None, :]
    return pos_out * atom_mask[..., None]


class DiffusionModule(nn.Module):
    """AF3 Algorithm 20: Diffusion module.

    Full version with atom-level attention encoder/decoder matching the
    reference OpenFold3 DiffusionModule.

    Args:
        c_s: Single representation channel dimension
        c_z: Pair representation channel dimension
        c_token: Token diffusion channel dimension
        c_s_input: Input single representation dimension
        sigma_data: Data variance constant
        no_diff_blocks: Number of diffusion transformer blocks
        no_diff_heads: Number of diffusion attention heads
        c_diff_hidden: Per-head hidden dim for diffusion attention
        n_diff_transition: Transition scale
        relpos_k: Relative position max index
        max_relative_chain: Max relative chain index
        c_atom: Atom single representation dim
        c_atom_pair: Atom pair representation dim
        atom_attn_n_query: Block height for atom attention
        atom_attn_n_key: Block width for atom attention
    """

    def __init__(
        self,
        c_s: int = 384,
        c_z: int = 128,
        c_token: int = 768,
        c_s_input: int = 449,
        sigma_data: float = 16.0,
        no_diff_blocks: int = 24,
        no_diff_heads: int = 16,
        c_diff_hidden: int = 48,
        n_diff_transition: int = 2,
        relpos_k: int = 32,
        max_relative_chain: int = 2,
        c_atom: int = 128,
        c_atom_pair: int = 16,
        atom_attn_n_query: int = 32,
        atom_attn_n_key: int = 128,
    ):
        super().__init__()
        self.c_s = c_s
        self.c_token = c_token
        self.sigma_data = sigma_data

        self.diffusion_conditioning = DiffusionConditioning(
            c_s=c_s, c_z=c_z, c_s_input=c_s_input,
            sigma_data=sigma_data,
            relpos_k=relpos_k, max_relative_chain=max_relative_chain,
        )

        self.atom_attn_enc = AtomAttentionEncoder(
            c_atom=c_atom,
            c_atom_pair=c_atom_pair,
            c_token=c_token,
            add_noisy_pos=True,
            c_s=c_s,
            c_z=c_z,
            c_hidden=32,
            no_heads=4,
            no_blocks=3,
            n_transition=2,
            n_query=atom_attn_n_query,
            n_key=atom_attn_n_key,
            use_ada_layer_norm=True,
        )

        self.diffusion_transformer = DiffusionTransformer(
            c_a=c_token, c_s=c_s, c_z=c_z,
            c_hidden=c_diff_hidden, no_heads=no_diff_heads,
            no_blocks=no_diff_blocks,
            n_transition=n_diff_transition,
            use_ada_layer_norm=True,
        )

        self.atom_attn_dec = AtomAttentionDecoder(
            c_atom=c_atom,
            c_atom_pair=c_atom_pair,
            c_token=c_token,
            c_hidden=32,
            no_heads=4,
            no_blocks=3,
            n_transition=2,
            n_query=atom_attn_n_query,
            n_key=atom_attn_n_key,
            use_ada_layer_norm=True,
        )

        self.layer_norm_s = LayerNorm(c_s, create_offset=False)
        self.linear_s = Linear(c_s, c_token, bias=False)

        self.layer_norm_a = LayerNorm(c_token, create_offset=False)

    def forward(
        self,
        batch: dict,
        xl_noisy: torch.Tensor,
        token_mask: torch.Tensor,
        atom_mask: torch.Tensor,
        t: torch.Tensor,
        si_input: torch.Tensor,
        si_trunk: torch.Tensor,
        zij_trunk: torch.Tensor,
        use_conditioning: bool,
        chunk_size: int | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        use_high_precision_attention: bool = False,
        _mask_trans: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            batch:     Feature dictionary
            xl_noisy:  [*, N_atom, 3] noisy atom positions
            token_mask:[*, N_token] token mask
            atom_mask: [*, N_atom] atom mask
            t:         [*] noise level
            si_input:  [*, N_token, c_s_input] input embedding
            si_trunk:  [*, N_token, c_s] trunk single rep
            zij_trunk: [*, N_token, N_token, c_z] trunk pair rep
            use_conditioning: Whether to condition with trunk reps

        Returns:
            [*, N_atom, 3] denoised atom positions
        """
        si, zij = self.diffusion_conditioning(
            batch=batch, t=t,
            si_input=si_input, si_trunk=si_trunk, zij_trunk=zij_trunk,
            use_conditioning=use_conditioning,
        )

        xl_noisy = xl_noisy * atom_mask[..., None]

        rl_noisy = xl_noisy / torch.sqrt(t[..., None, None] ** 2 + self.sigma_data ** 2)

        ai, ql, cl, plm = self.atom_attn_enc(
            batch=batch,
            rl=rl_noisy,
            si_trunk=si,
            zij_trunk=zij,
        )

        ai = ai + self.linear_s(self.layer_norm_s(si))

        ai = self.diffusion_transformer(
            a=ai, s=si, z=zij, mask=token_mask,
        )

        ai = self.layer_norm_a(ai)

        rl_update = self.atom_attn_dec(
            batch=batch, ai=ai, ql=ql, cl=cl, plm=plm,
        )

        # EDM-style combination
        xl_out = (
            self.sigma_data ** 2
            / (self.sigma_data ** 2 + t[..., None, None] ** 2)
            * xl_noisy
            + self.sigma_data
            * t[..., None, None]
            / torch.sqrt(self.sigma_data ** 2 + t[..., None, None] ** 2)
            * rl_update
        )

        xl_out = xl_out * atom_mask[..., None]

        return xl_out


class SampleDiffusion(nn.Module):
    """AF3 Algorithm 18: Diffusion sampling.

    Args:
        gamma_0: Schedule controlling factor
        gamma_min: Minimum schedule threshold
        noise_scale: Noise scaling factor
        step_scale: Step scaling factor
        diffusion_module: Instantiated DiffusionModule
    """

    def __init__(
        self,
        gamma_0: float,
        gamma_min: float,
        noise_scale: float,
        step_scale: float,
        diffusion_module: DiffusionModule,
    ):
        super().__init__()
        self.gamma_0 = gamma_0
        self.gamma_min = gamma_min
        self.noise_scale = noise_scale
        self.step_scale = step_scale
        self.diffusion_module = diffusion_module

    def forward(
        self,
        batch: dict,
        si_input: torch.Tensor,
        si_trunk: torch.Tensor,
        zij_trunk: torch.Tensor,
        noise_schedule: torch.Tensor,
        no_rollout_samples: int,
        use_conditioning: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            batch:           Feature dictionary
            si_input:        [*, N_token, c_s_input] input embedding
            si_trunk:        [*, N_token, c_s] trunk single rep
            zij_trunk:       [*, N_token, N_token, c_z] trunk pair rep
            noise_schedule:  [no_rollout_steps + 1] noise schedule
            no_rollout_samples: Number of diffusion samples

        Returns:
            [*, N_samples, N_atom, 3] sampled atom positions
        """
        atom_mask = batch["atom_mask"]
        batch_dim = atom_mask.shape[0]
        num_atoms = atom_mask.shape[-1]

        xl = noise_schedule[0] * torch.randn(
            (batch_dim, no_rollout_samples, num_atoms, 3),
            device=atom_mask.device, dtype=atom_mask.dtype,
        )

        for tau, c_tau in enumerate(noise_schedule[1:]):
            xl = centre_random_augmentation(xl=xl, atom_mask=atom_mask)

            gamma = self.gamma_0 if c_tau > self.gamma_min else 0
            t = noise_schedule[tau] * (gamma + 1)

            noise = (
                self.noise_scale
                * torch.sqrt(t ** 2 - noise_schedule[tau] ** 2)
                * torch.randn_like(xl)
            )
            xl_noisy = xl + noise

            xl_denoised = self.diffusion_module(
                batch=batch,
                xl_noisy=xl_noisy,
                token_mask=batch["token_mask"],
                atom_mask=atom_mask,
                t=t.to(xl_noisy.device),
                si_input=si_input,
                si_trunk=si_trunk,
                zij_trunk=zij_trunk,
                use_conditioning=use_conditioning,
            )

            delta = (xl_noisy - xl_denoised) / t
            dt = c_tau - t
            xl = xl_noisy + self.step_scale * dt * delta

        return xl


# Inlined from tasks/baseline/L3/alphafold3_pairformer.py
class PairFormerBlock(nn.Module):
    """Single block of AF3 Algorithm 17.

    Args:
        c_s: Single embedding channel dimension
        c_z: Pair embedding channel dimension
        c_hidden_pair_bias: Hidden dim for AttentionPairBias
        no_heads_pair_bias: Heads for AttentionPairBias
        c_hidden_mul: Hidden dim for triangle multiplication
        c_hidden_pair_att: Hidden dim for triangle attention
        no_heads_pair: Heads for triangle attention
        transition_n: Scale for transition hidden dim
        pair_dropout: Dropout rate
        inf: Large masking constant
    """

    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_hidden_pair_bias: int,
        no_heads_pair_bias: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_pair: int,
        transition_n: int,
        pair_dropout: float = 0.0,
        fuse_projection_weights: bool = False,
        inf: float = 1e9,
    ):
        super().__init__()

        self.pair_stack = PairBlock(
            c_z=c_z,
            c_hidden_mul=c_hidden_mul,
            c_hidden_pair_att=c_hidden_pair_att,
            no_heads_pair=no_heads_pair,
            transition_n=transition_n,
            pair_dropout=pair_dropout,
            inf=inf,
        )

        self.attn_pair_bias = AttentionPairBias(
            c_q=c_s, c_k=c_s, c_v=c_s,
            c_s=c_s, c_z=c_z,
            c_hidden=c_hidden_pair_bias,
            no_heads=no_heads_pair_bias,
            use_ada_layer_norm=False,
            gating=True,
            inf=inf,
        )

        self.single_transition = SwiGLUTransition(c_in=c_s, n=transition_n)

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        single_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: int | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
        _attn_chunk_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s:           [*, N_token, C_s] single embedding
            z:           [*, N_token, N_token, C_z] pair embedding
            single_mask: [*, N_token] single mask
            pair_mask:   [*, N_token, N_token] pair mask

        Returns:
            (s, z): updated single and pair embeddings
        """
        single_trans_mask = single_mask if _mask_trans else None

        z = self.pair_stack(z=z, pair_mask=pair_mask)

        s = s + self.attn_pair_bias(a=s, z=z, s=None, mask=single_mask)

        s = s + self.single_transition(s, mask=single_trans_mask)

        return s, z


class PairFormerStack(nn.Module):
    """AF3 Algorithm 17: PairFormer stack.

    Args:
        c_s: Single embedding channel dimension
        c_z: Pair embedding channel dimension
        c_hidden_pair_bias: Hidden dim for AttentionPairBias
        no_heads_pair_bias: Heads for AttentionPairBias
        c_hidden_mul: Hidden dim for triangle multiplication
        c_hidden_pair_att: Hidden dim for triangle attention
        no_heads_pair: Heads for triangle attention
        no_blocks: Number of PairFormer blocks
        transition_n: Scale for transition hidden dim
        pair_dropout: Dropout rate
        inf: Large masking constant
    """

    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_hidden_pair_bias: int,
        no_heads_pair_bias: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_pair: int,
        no_blocks: int,
        transition_n: int,
        pair_dropout: float = 0.0,
        fuse_projection_weights: bool = False,
        blocks_per_ckpt: int | None = None,
        inf: float = 1e9,
        **kwargs,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            PairFormerBlock(
                c_s=c_s, c_z=c_z,
                c_hidden_pair_bias=c_hidden_pair_bias,
                no_heads_pair_bias=no_heads_pair_bias,
                c_hidden_mul=c_hidden_mul,
                c_hidden_pair_att=c_hidden_pair_att,
                no_heads_pair=no_heads_pair,
                transition_n=transition_n,
                pair_dropout=pair_dropout,
                inf=inf,
            )
            for _ in range(no_blocks)
        ])

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        single_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: int | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s:           [*, N_token, C_s] single embedding
            z:           [*, N_token, N_token, C_z] pair embedding
            single_mask: [*, N_token] single mask
            pair_mask:   [*, N_token, N_token] pair mask

        Returns:
            (s, z): updated single and pair embeddings
        """
        for block in self.blocks:
            s, z = block(
                s=s, z=z,
                single_mask=single_mask,
                pair_mask=pair_mask,
            )

        return s, z


# Inlined from tasks/baseline/L3/alphafold3_heads.py
class DistogramHead(nn.Module):
    """Predicts inter-residue distance distribution.

    Args:
        c_z: Pair embedding channel dimension
        no_bins: Number of distance bins
    """

    def __init__(self, c_z: int, no_bins: int = 64):
        super().__init__()
        self.linear = Linear(c_z, no_bins, bias=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        logits = self.linear(z)
        logits = logits + logits.transpose(-2, -3)
        return logits


class PLDDTHead(nn.Module):
    """Predicts per-atom pLDDT confidence (PerResidueLDDTAllAtom).

    Outputs max_atoms_per_token * no_bins logits per token.

    Args:
        c_s: Single embedding channel dimension
        no_bins: Number of pLDDT bins
        max_atoms_per_token: Maximum atoms per token (23 for all-atom)
    """

    def __init__(self, c_s: int, no_bins: int = 50, max_atoms_per_token: int = 23):
        super().__init__()
        self.no_bins = no_bins
        self.max_atoms_per_token = max_atoms_per_token
        self.layer_norm = LayerNorm(c_s)
        self.linear = Linear(c_s, max_atoms_per_token * no_bins, bias=False)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.linear(self.layer_norm(s))


class PAEHead(nn.Module):
    """Predicts Predicted Aligned Error (PAE).

    Args:
        c_z: Pair embedding channel dimension
        no_bins: Number of PAE bins
    """

    def __init__(self, c_z: int, no_bins: int = 64):
        super().__init__()
        self.layer_norm = LayerNorm(c_z)
        self.linear = Linear(c_z, no_bins, bias=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.linear(self.layer_norm(z))


class PDEHead(nn.Module):
    """Predicts Predicted Distance Error (PDE).

    Args:
        c_z: Pair embedding channel dimension
        no_bins: Number of PDE bins
    """

    def __init__(self, c_z: int, no_bins: int = 64):
        super().__init__()
        self.layer_norm = LayerNorm(c_z)
        self.linear = Linear(c_z, no_bins, bias=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        logits = self.linear(self.layer_norm(z))
        logits = logits + logits.transpose(-2, -3)
        return logits


class ExperimentallyResolvedHead(nn.Module):
    """Predicts per-atom experimental resolution confidence.

    Args:
        c_s: Single embedding channel dimension
        no_bins: Number of bins (2 for resolved/not resolved)
        max_atoms_per_token: Maximum atoms per token (23 for all-atom)
    """

    def __init__(self, c_s: int, no_bins: int = 2, max_atoms_per_token: int = 23):
        super().__init__()
        self.no_bins = no_bins
        self.max_atoms_per_token = max_atoms_per_token
        self.layer_norm = LayerNorm(c_s)
        self.linear = Linear(c_s, max_atoms_per_token * no_bins, bias=False)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.linear(self.layer_norm(s))


class PairformerEmbedding(nn.Module):
    """Confidence head PairformerEmbedding.

    Refines pair representation using predicted atom positions before
    confidence heads (PAE, PDE, pLDDT, experimentally resolved).

    Reference: openfold3/core/model/heads/prediction_heads.py PairformerEmbedding

    Args:
        c_s_input: Input single rep dimension
        c_z: Pair rep dimension
        c_s: Single rep dimension
        no_distance_bins: Number of distance bins
        pairformer_kwargs: Config for pairformer stack
    """

    def __init__(
        self,
        c_s_input: int = 449,
        c_z: int = 128,
        c_s: int = 384,
        no_distance_bins: int = 39,
        pairformer_no_blocks: int = 4,
        pairformer_c_hidden_pair_bias: int = 24,
        pairformer_no_heads_pair_bias: int = 16,
        pairformer_c_hidden_mul: int = 128,
        pairformer_c_hidden_pair_att: int = 32,
        pairformer_no_heads_pair: int = 4,
        pairformer_transition_n: int = 4,
        pairformer_pair_dropout: float = 0.0,
    ):
        super().__init__()

        self.linear_i = Linear(c_s_input, c_z, bias=False)
        self.linear_j = Linear(c_s_input, c_z, bias=False)
        self.linear_distance = Linear(no_distance_bins, c_z, bias=False)

        self.pairformer_stack = PairFormerStack(
            c_s=c_s,
            c_z=c_z,
            c_hidden_pair_bias=pairformer_c_hidden_pair_bias,
            no_heads_pair_bias=pairformer_no_heads_pair_bias,
            c_hidden_mul=pairformer_c_hidden_mul,
            c_hidden_pair_att=pairformer_c_hidden_pair_att,
            no_heads_pair=pairformer_no_heads_pair,
            no_blocks=pairformer_no_blocks,
            transition_n=pairformer_transition_n,
            pair_dropout=pairformer_pair_dropout,
        )

    def forward(
        self,
        si_input: torch.Tensor,
        zij: torch.Tensor,
        s: torch.Tensor,
        single_mask: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        zij = (
            zij
            + self.linear_i(si_input)[..., :, None, :]
            + self.linear_j(si_input)[..., None, :, :]
        )

        s, zij = self.pairformer_stack(
            s=s, z=zij, single_mask=single_mask, pair_mask=pair_mask,
        )
        return s, zij


class AuxiliaryHeads(nn.Module):
    """All auxiliary prediction heads for AF3.

    Args:
        c_s: Single embedding channel dimension
        c_z: Pair embedding channel dimension
        c_s_input: Input single rep dimension (for PairformerEmbedding)
        max_atoms_per_token: Max atoms per token (23 for all-atom)
    """

    def __init__(
        self,
        c_s: int = 384,
        c_z: int = 128,
        c_s_input: int = 449,
        max_atoms_per_token: int = 23,
    ):
        super().__init__()
        self.pairformer_embedding = PairformerEmbedding(
            c_s_input=c_s_input,
            c_z=c_z,
            c_s=c_s,
        )
        self.distogram = DistogramHead(c_z, no_bins=64)
        self.plddt = PLDDTHead(c_s, no_bins=50, max_atoms_per_token=max_atoms_per_token)
        self.pae = PAEHead(c_z, no_bins=64)
        self.pde = PDEHead(c_z, no_bins=64)
        self.experimentally_resolved = ExperimentallyResolvedHead(
            c_s, no_bins=2, max_atoms_per_token=max_atoms_per_token,
        )

    def forward(
        self, s: torch.Tensor, z: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return {
            "distogram_logits": self.distogram(z),
            "plddt_logits": self.plddt(s),
            "pae_logits": self.pae(z),
            "pde_logits": self.pde(z),
            "experimentally_resolved_logits": self.experimentally_resolved(s),
        }


# Inlined from tasks/baseline/L2/alphafold3_msa_attention.py
def _permute_final_dims(tensor: torch.Tensor, inds: tuple[int, ...]) -> torch.Tensor:
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


class MSARowAttentionWithPairBias(nn.Module):
    """AF3 MSA Pair-Weighted Averaging (Algorithm 10).

    Uses pair activations as weights (softmax over token dim) instead of
    key-query attention.  Parameter names match the checkpoint layout:
    linear_v, linear_g, linear_o (no nested mha).

    Args:
        c_m: MSA input channel dimension
        c_z: Pair embedding channel dimension
        c_hidden: Per-head hidden channel dimension
        no_heads: Number of attention heads
        inf: Large constant for masking
    """

    def __init__(
        self,
        c_m: int,
        c_z: int,
        c_hidden: int,
        no_heads: int,
        inf: float = 1e9,
    ):
        super().__init__()
        self.c_m = c_m
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.inf = inf

        self.layer_norm_m = LayerNorm(c_m)
        self.layer_norm_z = LayerNorm(c_z)
        self.linear_z = Linear(c_z, no_heads, bias=False)

        self.linear_v = Linear(c_m, c_hidden * no_heads, bias=False)
        self.linear_g = Linear(c_m, c_hidden * no_heads, bias=False)
        self.linear_o = Linear(c_hidden * no_heads, c_m, bias=False)

        self.sigmoid = Sigmoid()
        self.softmax = Softmax(dim=-1)

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        chunk_size: int | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            m:    [*, N_seq, N_res, C_m] MSA embedding
            z:    [*, N_res, N_res, C_z] pair embedding
            mask: [*, N_res, N_res] pair mask

        Returns:
            [*, N_seq, N_res, C_m] updated MSA embedding
        """
        if z is None:
            return m

        n_res = z.shape[-2]

        if mask is None:
            mask = z.new_ones(z.shape[:-1])

        # Pair bias: [*, 1, no_heads, N_res, N_res]
        mask_bias = (self.inf * (mask - 1))[..., None, None, :, :]
        z_norm = self.layer_norm_z(z)
        z_proj = self.linear_z(z_norm)
        z_weights = _permute_final_dims(z_proj, (2, 0, 1)).unsqueeze(-4)
        z_weights = z_weights + mask_bias
        z_weights = self.softmax(z_weights)

        m = self.layer_norm_m(m)

        # Value projection
        v = self.linear_v(m)
        v = v.view(v.shape[:-1] + (self.no_heads, -1))
        v = v.transpose(-2, -3)  # [*, N_seq, H, N_res, C_hidden]

        # Weighted average: [*, N_seq, H, N_res, C_hidden]
        o = torch.einsum("...hqk,...hkc->...qhc", z_weights, v)

        # Gating
        g = self.sigmoid(self.linear_g(m))
        g = g.view(g.shape[:-1] + (self.no_heads, -1))

        o = o * g

        # Flatten heads and project
        o = o.reshape(o.shape[:-2] + (-1,))
        o = self.linear_o(o)

        return o


# Inlined from tasks/baseline/L2/alphafold3_outer_product_mean.py
class OuterProductMean(nn.Module):
    """AF3 Algorithm 9: Outer product mean.

    Args:
        c_m: MSA embedding channel dimension
        c_z: Pair embedding channel dimension
        c_hidden: Hidden channel dimension
        eps: Epsilon for numerical stability
    """

    def __init__(self, c_m: int, c_z: int, c_hidden: int, eps: float = 1e-3):
        super().__init__()
        self.c_m = c_m
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.eps = eps

        self.layer_norm = LayerNorm(c_m)
        self.linear_1 = Linear(c_m, c_hidden, bias=False)
        self.linear_2 = Linear(c_m, c_hidden, bias=False)
        self.linear_out = Linear(c_hidden ** 2, c_z, bias=True)

    def forward(
        self,
        m: torch.Tensor,
        mask: torch.Tensor | None = None,
        chunk_size: int | None = None,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            m:    [*, N_seq, N_res, C_m] MSA embedding
            mask: [*, N_seq, N_res] MSA mask

        Returns:
            [*, N_res, N_res, C_z] pair embedding update
        """
        if mask is None:
            mask = m.new_ones(m.shape[:-1])

        ln = self.layer_norm(m)

        mask = mask.unsqueeze(-1)
        a = self.linear_1(ln) * mask
        b = self.linear_2(ln) * mask

        del ln

        # [*, N_res, N_seq, C]
        a = a.transpose(-2, -3)
        b = b.transpose(-2, -3)

        # [*, N_res, N_res, C, C]
        outer = torch.einsum("...bac,...dae->...bdce", a, b)

        # [*, N_res, N_res, C * C]
        outer = outer.reshape(outer.shape[:-2] + (-1,))

        # [*, N_res, N_res, C_z]
        outer = self.linear_out(outer)

        # Normalization: count valid sequence pairs per residue pair
        norm = torch.einsum("...abc,...adc->...bdc", mask, mask)
        norm = norm + self.eps

        outer = outer / norm

        return outer


# Inlined from tasks/baseline/L3/alphafold3_msa_module.py
class MSAModuleBlock(nn.Module):
    """Single block of AF3 Algorithm 8.

    Args:
        c_m: MSA channel dimension
        c_z: Pair channel dimension
        c_hidden_msa_att: Hidden dim in MSA attention
        c_hidden_opm: Hidden dim in outer product mean
        c_hidden_mul: Hidden dim in triangle multiplication
        c_hidden_pair_att: Hidden dim in triangle attention
        no_heads_msa: Heads for MSA attention
        no_heads_pair: Heads for triangle attention
        transition_n: Transition layer scale
        msa_dropout: MSA dropout rate
        pair_dropout: Pair dropout rate
        opm_first: Whether OPM comes before MSA attention
    """

    def __init__(
        self,
        c_m: int,
        c_z: int,
        c_hidden_msa_att: int,
        c_hidden_opm: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_msa: int,
        no_heads_pair: int,
        transition_n: int,
        msa_dropout: float = 0.0,
        pair_dropout: float = 0.0,
        opm_first: bool = True,
        fuse_projection_weights: bool = False,
        inf: float = 1e9,
        eps: float = 1e-3,
        last_block: bool = False,
    ):
        super().__init__()
        self.opm_first = opm_first
        self.skip_msa_update = last_block and opm_first

        if not self.skip_msa_update:
            self.msa_att_row = MSARowAttentionWithPairBias(
                c_m=c_m, c_z=c_z,
                c_hidden=c_hidden_msa_att,
                no_heads=no_heads_msa,
                inf=inf,
            )

            self.msa_transition = SwiGLUTransition(c_in=c_m, n=transition_n)

        self.outer_product_mean = OuterProductMean(
            c_m=c_m, c_z=c_z, c_hidden=c_hidden_opm, eps=eps,
        )

        self.pair_stack = PairBlock(
            c_z=c_z,
            c_hidden_mul=c_hidden_mul,
            c_hidden_pair_att=c_hidden_pair_att,
            no_heads_pair=no_heads_pair,
            transition_n=transition_n,
            pair_dropout=pair_dropout,
            inf=inf,
        )

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: int | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            m:        [*, N_seq, N_res, C_m] MSA embedding
            z:        [*, N_res, N_res, C_z] pair embedding
            msa_mask: [*, N_seq, N_res] MSA mask
            pair_mask:[*, N_res, N_res] pair mask

        Returns:
            (m, z): updated MSA and pair embeddings
        """
        if self.opm_first:
            z = z + self.outer_product_mean(m, mask=msa_mask)

        if not self.skip_msa_update:
            m = m + self.msa_att_row(m, z=z, mask=pair_mask)
            m = m + self.msa_transition(m)

        if not self.opm_first:
            z = z + self.outer_product_mean(m, mask=msa_mask)

        z = self.pair_stack(z=z, pair_mask=pair_mask)

        return m, z


class MSAModuleStack(nn.Module):
    """AF3 Algorithm 8: MSA module stack.

    Args:
        c_m: MSA channel dimension
        c_z: Pair channel dimension
        c_hidden_msa_att: Hidden dim in MSA attention
        c_hidden_opm: Hidden dim in outer product mean
        c_hidden_mul: Hidden dim in triangle multiplication
        c_hidden_pair_att: Hidden dim in triangle attention
        no_heads_msa: Heads for MSA attention
        no_heads_pair: Heads for triangle attention
        no_blocks: Number of MSA module blocks
        transition_n: Transition scale
        opm_first: Whether OPM comes before MSA attention
    """

    def __init__(
        self,
        c_m: int,
        c_z: int,
        c_hidden_msa_att: int,
        c_hidden_opm: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_msa: int,
        no_heads_pair: int,
        no_blocks: int,
        transition_n: int,
        msa_dropout: float = 0.0,
        pair_dropout: float = 0.0,
        opm_first: bool = True,
        fuse_projection_weights: bool = False,
        blocks_per_ckpt: int | None = None,
        inf: float = 1e9,
        eps: float = 1e-3,
        **kwargs,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            MSAModuleBlock(
                c_m=c_m, c_z=c_z,
                c_hidden_msa_att=c_hidden_msa_att,
                c_hidden_opm=c_hidden_opm,
                c_hidden_mul=c_hidden_mul,
                c_hidden_pair_att=c_hidden_pair_att,
                no_heads_msa=no_heads_msa,
                no_heads_pair=no_heads_pair,
                transition_n=transition_n,
                msa_dropout=msa_dropout,
                pair_dropout=pair_dropout,
                opm_first=opm_first,
                inf=inf,
                eps=eps,
                last_block=(i == no_blocks - 1),
            )
            for i in range(no_blocks)
        ])

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            m:        [*, N_seq, N_res, C_m] MSA embedding
            z:        [*, N_res, N_res, C_z] pair embedding
            msa_mask: [*, N_seq, N_res] MSA mask
            pair_mask:[*, N_res, N_res] pair mask

        Returns:
            (m, z): updated MSA and pair embeddings
        """
        for block in self.blocks:
            m, z = block(m=m, z=z, msa_mask=msa_mask, pair_mask=pair_mask)

        return m, z


from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class OpenFold3Config:
    """Configuration for AlphaFold3 model.

    Default values match openfold3/projects/of3_all_atom/config/model_config.py.
    """

    # Channel dimensions
    c_s: int = 384
    c_z: int = 128
    c_m: int = 64
    c_atom: int = 128
    c_atom_pair: int = 16
    c_token_embedder: int = 384
    c_token_diffusion: int = 768
    c_s_input: int = 449
    c_t: int = 64
    max_atoms_per_token: int = 23

    # PairFormer
    pairformer_no_blocks: int = 48
    pairformer_c_hidden_mul: int = 128
    pairformer_c_hidden_pair_att: int = 32
    pairformer_no_heads_pair: int = 4
    pairformer_c_hidden_pair_bias: int = 24
    pairformer_no_heads_pair_bias: int = 16
    pairformer_transition_n: int = 4
    pairformer_pair_dropout: float = 0.0

    # MSA module
    msa_no_blocks: int = 4
    msa_c_hidden_msa_att: int = 8
    msa_c_hidden_opm: int = 32
    msa_c_hidden_mul: int = 128
    msa_c_hidden_pair_att: int = 32
    msa_no_heads_msa: int = 8
    msa_no_heads_pair: int = 4
    msa_transition_n: int = 4
    msa_opm_first: bool = True
    msa_dropout: float = 0.0
    msa_pair_dropout: float = 0.0

    # Diffusion module
    diff_no_blocks: int = 24
    diff_no_heads: int = 16
    diff_c_hidden: int = 48
    diff_n_transition: int = 2
    sigma_data: float = 16.0

    # Noise schedule
    no_rollout_steps: int = 200
    s_max: float = 160.0
    s_min: float = 4e-4
    noise_schedule_p: int = 7

    # Sampling
    gamma_0: float = 0.8
    gamma_min: float = 1.0
    noise_scale: float = 1.003
    step_scale: float = 1.5

    # Recycling
    num_recycles: int = 3

    # Input
    relpos_k: int = 32
    max_relative_chain: int = 2

    # Atom attention
    atom_attn_n_query: int = 32
    atom_attn_n_key: int = 128

    @classmethod
    def from_pretrained(cls, model_name: str) -> "OpenFold3Config":
        """Create config for known OpenFold3 models."""
        return cls()


def _create_noise_schedule(
    no_rollout_steps: int,
    sigma_data: float,
    s_max: float,
    s_min: float,
    p: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """AF3 noise schedule (Page 24 of supplement)."""
    t = torch.arange(0, 1 + no_rollout_steps, dtype=dtype, device=device) / no_rollout_steps
    return sigma_data * (s_max ** (1 / p) + t * (s_min ** (1 / p) - s_max ** (1 / p))) ** p


class OpenFold3Model(nn.Module):
    """Full AlphaFold3 / OpenFold3 model.

    Wires input embedder -> template embedder -> MSA module -> PairFormer
    -> diffusion module -> auxiliary heads.

    Reference: openfold3/projects/of3_all_atom/model.py OpenFold3

    Args:
        config: OpenFold3Config with all hyperparameters.
    """

    def __init__(self, config: OpenFold3Config):
        super().__init__()
        self.config = config

        # Input embedding (includes AtomAttentionEncoder)
        self.input_embedder = InputEmbedder(
            c_s_input=config.c_s_input,
            c_s=config.c_s,
            c_z=config.c_z,
            relpos_k=config.relpos_k,
            max_relative_chain=config.max_relative_chain,
            c_atom=config.c_atom,
            c_atom_pair=config.c_atom_pair,
            c_token=config.c_token_embedder,
        )

        # Recycle projections (z first, then s, matching reference ordering)
        self.layer_norm_z = LayerNorm(config.c_z)
        self.linear_z = Linear(config.c_z, config.c_z, bias=False)

        # Template embedder
        self.template_embedder = TemplateEmbedder(
            c_t=config.c_t,
            c_z=config.c_z,
        )

        # MSA module embedder
        self.msa_module_embedder = MSAModuleEmbedder(
            c_m_feats=34,
            c_m=config.c_m,
            c_s_input=config.c_s_input,
        )

        # MSA module
        self.msa_module = MSAModuleStack(
            c_m=config.c_m,
            c_z=config.c_z,
            c_hidden_msa_att=config.msa_c_hidden_msa_att,
            c_hidden_opm=config.msa_c_hidden_opm,
            c_hidden_mul=config.msa_c_hidden_mul,
            c_hidden_pair_att=config.msa_c_hidden_pair_att,
            no_heads_msa=config.msa_no_heads_msa,
            no_heads_pair=config.msa_no_heads_pair,
            no_blocks=config.msa_no_blocks,
            transition_n=config.msa_transition_n,
            msa_dropout=config.msa_dropout,
            pair_dropout=config.msa_pair_dropout,
            opm_first=config.msa_opm_first,
        )

        # Recycle single projection (after template/MSA, before pairformer)
        self.layer_norm_s = LayerNorm(config.c_s)
        self.linear_s = Linear(config.c_s, config.c_s, bias=False)

        # PairFormer stack
        self.pairformer_stack = PairFormerStack(
            c_s=config.c_s,
            c_z=config.c_z,
            c_hidden_pair_bias=config.pairformer_c_hidden_pair_bias,
            no_heads_pair_bias=config.pairformer_no_heads_pair_bias,
            c_hidden_mul=config.pairformer_c_hidden_mul,
            c_hidden_pair_att=config.pairformer_c_hidden_pair_att,
            no_heads_pair=config.pairformer_no_heads_pair,
            no_blocks=config.pairformer_no_blocks,
            transition_n=config.pairformer_transition_n,
            pair_dropout=config.pairformer_pair_dropout,
        )

        # Diffusion module (includes AtomAttentionEncoder/Decoder)
        self.diffusion_module = DiffusionModule(
            c_s=config.c_s,
            c_z=config.c_z,
            c_token=config.c_token_diffusion,
            c_s_input=config.c_s_input,
            sigma_data=config.sigma_data,
            no_diff_blocks=config.diff_no_blocks,
            no_diff_heads=config.diff_no_heads,
            c_diff_hidden=config.diff_c_hidden,
            n_diff_transition=config.diff_n_transition,
            relpos_k=config.relpos_k,
            max_relative_chain=config.max_relative_chain,
            c_atom=config.c_atom,
            c_atom_pair=config.c_atom_pair,
            atom_attn_n_query=config.atom_attn_n_query,
            atom_attn_n_key=config.atom_attn_n_key,
        )

        self.sample_diffusion = SampleDiffusion(
            gamma_0=config.gamma_0,
            gamma_min=config.gamma_min,
            noise_scale=config.noise_scale,
            step_scale=config.step_scale,
            diffusion_module=self.diffusion_module,
        )

        # Auxiliary heads (includes PairformerEmbedding + ExperimentallyResolved)
        self.aux_heads = AuxiliaryHeads(
            c_s=config.c_s,
            c_z=config.c_z,
            c_s_input=config.c_s_input,
            max_atoms_per_token=config.max_atoms_per_token,
        )

    def run_trunk(
        self,
        batch: dict,
        num_recycles: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the trunk: input embedding -> MSA module -> PairFormer.

        Args:
            batch: Feature dictionary with token_features, residue_index,
                   msa (optional), token_mask, pair_mask, msa_mask.
            num_recycles: Number of recycling iterations.

        Returns:
            (s_input, s, z):
                s_input: [*, N_token, C_s_input]
                s: [*, N_token, C_s]
                z: [*, N_token, N_token, C_z]
        """
        if num_recycles is None:
            num_recycles = self.config.num_recycles

        token_features = batch["token_features"]
        residue_index = batch["residue_index"]
        token_mask = batch["token_mask"]
        pair_mask = batch.get("pair_mask", token_mask[..., :, None] * token_mask[..., None, :])

        s_input, s_init, z_init = self.input_embedder(
            token_features, residue_index, batch=batch,
        )

        s = s_init
        z = z_init

        for cycle in range(num_recycles + 1):
            if cycle > 0:
                z = z_init + self.linear_z(self.layer_norm_z(z))
                s = s_init + self.linear_s(self.layer_norm_s(s))

            # Template embedder
            if "template_distogram" in batch:
                t_emb = self.template_embedder(
                    batch=batch, z=z, pair_mask=pair_mask,
                )
                z = z + t_emb

            # MSA module
            if "msa" in batch and batch["msa"] is not None:
                m, msa_mask = self.msa_module_embedder(batch=batch, s_input=s_input)
                _, z = self.msa_module(
                    m=m, z=z, msa_mask=msa_mask, pair_mask=pair_mask,
                )

            # PairFormer
            s, z = self.pairformer_stack(
                s=s, z=z,
                single_mask=token_mask,
                pair_mask=pair_mask,
            )

        return s_input, s, z

    def forward(self, batch: dict) -> tuple[dict, dict]:
        """Full forward pass.

        Args:
            batch: Feature dictionary. Required keys:
                - token_features: [*, N_token, C_token] per-token features
                - residue_index:  [*, N_token] residue indices
                - token_mask:     [*, N_token] token mask
                - atom_mask:      [*, N_atom] atom mask
                Optional:
                - msa:       [*, N_seq, N_res, C_m] MSA features
                - msa_mask:  [*, N_seq, N_res] MSA mask
                - pair_mask: [*, N_token, N_token] pair mask

        Returns:
            (outputs, aux_outputs):
                outputs: dict with atom_positions_predicted, head logits
                aux_outputs: dict with trunk representations
        """
        s_input, s, z = self.run_trunk(batch)

        # Auxiliary heads
        head_outputs = self.aux_heads(s=s, z=z)

        # Diffusion sampling
        noise_schedule = _create_noise_schedule(
            no_rollout_steps=self.config.no_rollout_steps,
            sigma_data=self.config.sigma_data,
            s_max=self.config.s_max,
            s_min=self.config.s_min,
            p=self.config.noise_schedule_p,
            dtype=s.dtype,
            device=s.device,
        )

        with torch.no_grad():
            atom_positions = self.sample_diffusion(
                batch=batch,
                si_input=s_input,
                si_trunk=s,
                zij_trunk=z,
                noise_schedule=noise_schedule,
                no_rollout_samples=1,
                use_conditioning=True,
            )

        outputs = {
            "atom_positions_predicted": atom_positions,
            **head_outputs,
        }

        aux_outputs = {
            "s_input": s_input,
            "s_trunk": s,
            "z_trunk": z,
        }

        return outputs, aux_outputs


def load_openfold3_checkpoint(
    model: OpenFold3Model, checkpoint_path: str,
) -> None:
    """Load an OpenFold3 checkpoint into a fastkernels OpenFold3Model.

    The fastkernels model architecture is a 1:1 match with the reference
    OpenFold/OpenFold3 checkpoint, so strict loading is used.

    Args:
        model: fastkernels OpenFold3Model instance.
        checkpoint_path: Path to the checkpoint file.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif "model" in ckpt and isinstance(ckpt["model"], dict):
        sd = ckpt["model"]
    else:
        sd = ckpt

    cleaned = {}
    for k, v in sd.items():
        if k.startswith("model."):
            k = k[len("model."):]
        cleaned[k] = v

    model.load_state_dict(cleaned, strict=True)
