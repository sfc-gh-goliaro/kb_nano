"""Sequence-local atom attention for AlphaFold3.

AtomAttentionEncoder (Algorithm 5) and AtomAttentionDecoder (Algorithm 6).

Reference: openfold3/core/model/layers/sequence_local_atom_attention.py
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear


def _convert_single_rep_to_blocks(
    ql: torch.Tensor,
    n_query: int,
    n_key: int,
    atom_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Convert flat atom representation to windowed block format.

    Args:
        ql: [*, N_atom, C] atom features
        n_query: block height
        n_key: block width
        atom_mask: [*, N_atom] mask

    Returns:
        ql_blocks: [*, N_blocks, n_query, C]
        qk_blocks: [*, N_blocks, n_key, C]
        mask_blocks: [*, N_blocks, n_query, n_key] or None
    """
    n_atoms = ql.shape[-2]
    c = ql.shape[-1]
    pad_to = ((n_atoms + n_query - 1) // n_query) * n_query
    if pad_to > n_atoms:
        ql = torch.nn.functional.pad(ql, (0, 0, 0, pad_to - n_atoms))
        if atom_mask is not None:
            atom_mask = torch.nn.functional.pad(atom_mask, (0, pad_to - n_atoms))

    n_blocks = pad_to // n_query

    batch_shape = ql.shape[:-2]
    ql_blocks = ql.reshape(*batch_shape, n_blocks, n_query, c)

    half_key = (n_key - n_query) // 2
    ql_padded = torch.nn.functional.pad(ql, (0, 0, half_key, half_key))
    qk_list = []
    for i in range(n_blocks):
        start = i * n_query
        qk_list.append(ql_padded[..., start:start + n_key, :])
    qk_blocks = torch.stack(qk_list, dim=-3)

    mask_blocks = None
    if atom_mask is not None:
        mask_padded = torch.nn.functional.pad(atom_mask, (half_key, half_key))
        mask_q = atom_mask.reshape(*batch_shape, n_blocks, n_query)
        mask_list = []
        for i in range(n_blocks):
            start = i * n_query
            mask_list.append(mask_padded[..., start:start + n_key])
        mask_k = torch.stack(mask_list, dim=-2)
        mask_blocks = mask_q[..., :, None] * mask_k[..., None, :]

    return ql_blocks, qk_blocks, mask_blocks


def _convert_pair_rep_to_blocks(
    batch: dict,
    zij_trunk: torch.Tensor,
    n_query: int,
    n_key: int,
) -> torch.Tensor:
    """Convert pair representation to block format for atom attention.

    Broadcasts token-level pair rep to atom-level blocks.

    Args:
        batch: needs token_mask, num_atoms_per_token or atom_to_token_index
        zij_trunk: [*, N_token, N_token, C_z]
        n_query: block height
        n_key: block width

    Returns:
        [*, N_blocks, n_query, n_key, C_z]
    """
    n_atoms = batch["atom_mask"].shape[-1]
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

    pad_to = ((n_atoms + n_query - 1) // n_query) * n_query
    n_blocks = pad_to // n_query

    if pad_to > n_atoms:
        pad_idx = atom_to_token.new_zeros(pad_to - n_atoms)
        atom_to_token_padded = torch.cat([atom_to_token, pad_idx], dim=-1)
    else:
        atom_to_token_padded = atom_to_token

    half_key = (n_key - n_query) // 2
    atk_padded = torch.nn.functional.pad(atom_to_token_padded, (half_key, half_key))

    result = zij_trunk.new_zeros(
        zij_trunk.shape[:-3] + (n_blocks, n_query, n_key, c_z)
    )

    for b in range(n_blocks):
        q_start = b * n_query
        q_indices = atom_to_token_padded[q_start:q_start + n_query]

        k_start = b * n_query
        k_indices = atk_padded[k_start:k_start + n_key]

        result[..., b, :, :, :] = zij_trunk[..., q_indices[:, None], k_indices[None, :], :]

    return result


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
    ):
        super().__init__()
        from ..L3.openfold3_diffusion_transformer import DiffusionTransformer

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

        self.relu = nn.ReLU()
        self.linear_l = Linear(c_atom, c_atom_pair, bias=False)
        self.linear_m = Linear(c_atom, c_atom_pair, bias=False)

        self.pair_mlp = nn.Sequential(
            nn.ReLU(),
            Linear(c_atom_pair, c_atom_pair, bias=False),
            nn.ReLU(),
            Linear(c_atom_pair, c_atom_pair, bias=False),
            nn.ReLU(),
            Linear(c_atom_pair, c_atom_pair, bias=False),
        )

        self.atom_transformer = DiffusionTransformer(
            c_a=c_atom, c_s=c_atom, c_z=c_atom_pair,
            c_hidden=c_hidden, no_heads=no_heads,
            no_blocks=no_blocks, n_transition=n_transition,
            use_ada_layer_norm=use_ada_layer_norm,
            n_query=n_query, n_key=n_key,
        )

        self.linear_q = nn.Sequential(
            Linear(c_atom, c_token, bias=False),
            nn.ReLU(),
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
    ):
        super().__init__()
        from ..L3.openfold3_diffusion_transformer import DiffusionTransformer

        self.linear_q_in = Linear(c_token, c_atom, bias=False)

        self.atom_transformer = DiffusionTransformer(
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
