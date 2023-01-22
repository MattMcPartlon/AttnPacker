from typing import Optional, Tuple, NamedTuple

import torch
from einops import rearrange
from torch import nn, einsum, broadcast_tensors, Tensor

from protein_learning.common.helpers import batched_index_select, exists, default, safe_normalize
from protein_learning.networks.common.equivariant.linear import VNLinear  # noqa
from protein_learning.networks.common.helpers.neighbor_utils import get_neighbor_info
from protein_learning.networks.common.net_utils import FeedForward, Residual


# classes

class EGNNLayer(nn.Module):
    """Single EGNN Layer"""

    def __init__(
            self,
            node_dim: int,
            pair_dim: int,
            coord_dim: int,
            m_dim: int = 16,
            dropout: float = 0.0,
            norm_rel_coords: bool = False,
            use_nearest: bool = False,
            max_radius: float = 20,
            top_k: int = 32,
            lin_proj_coords: bool = False,
            use_rezero: bool = False,
    ):
        super().__init__()
        self.use_nearest = use_nearest
        self.to_nbr_info = lambda x: get_neighbor_info(x, max_radius=max_radius, top_k=top_k, exclude_self=False)

        # feature norms
        self.feat_pre_norm = nn.LayerNorm(node_dim)
        self.node_norm = nn.LayerNorm(node_dim + m_dim)
        self.coord_norm = lambda x: safe_normalize(x) if norm_rel_coords else x
        # Feature FFs
        pair_dim_in = pair_dim + 2 * node_dim + coord_dim
        self.pair_mlp = FeedForward(dim_in=pair_dim_in, dim_out=m_dim, mult=2, dropout=dropout)
        self.node_mlp = FeedForward(dim_in=node_dim + m_dim, dim_out=node_dim, mult=2, num_layers=2, dropout=dropout)
        self.coord_weight_mlp = FeedForward(dim_in=m_dim, dim_out=1, mult=4, num_layers=2, dropout=dropout)
        # coord proj.
        self.coord_proj = VNLinear(dim_in=coord_dim) if lin_proj_coords else nn.Identity()
        # residuals
        self.node_residual = Residual(use_rezero=use_rezero)
        self.coord_residual = Residual(use_rezero=use_rezero)

    def forward(
            self,
            node_feats: Tensor,
            coords: Tensor,
            pair_feats: Tensor,
            mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, ...]:
        # Gather Input Features
        proj_coords, normed_feats = self.coord_proj(coords), self.feat_pre_norm(node_feats)
        rel_coords = proj_coords.unsqueeze(2) - proj_coords.unsqueeze(1)  # (b,n,n,a,c)
        rel_coords = self.coord_norm(rel_coords)
        rel_dists = torch.sum(torch.square(rel_coords), dim=-1)

        if self.use_nearest:
            nbr_info = self.to_nbr_info(torch.mean(coords, dim=-2))
            feats_j = batched_index_select(normed_feats, nbr_info.indices, dim=1)
            mask = torch.logical_and(default(mask, nbr_info.full_mask), nbr_info.full_mask)
            pair_feats, rel_dists, rel_coords, mask = map(
                lambda x: batched_index_select(x, nbr_info.indices, dim=2),
                (pair_feats, rel_dists, rel_coords, mask)
            )
        else:
            feats_j = rearrange(normed_feats, 'b j d -> b () j d')

        feats_i = rearrange(normed_feats, 'b i d -> b i () d')
        feats_i, feats_j = broadcast_tensors(feats_i, feats_j)

        edge_input = torch.cat((feats_i, feats_j, rel_dists, pair_feats), dim=-1)
        m_ij = self.pair_mlp(edge_input)

        coord_weights = self.coord_weight_mlp(m_ij).squeeze(-1)  # b,n,N,1

        if exists(mask):
            m_ij = m_ij.masked_fill(~mask.unsqueeze(-1), 0.)
            coord_weights.masked_fill_(~mask, 0.)

        if exists(mask):
            # masked mean
            mask_sum = torch.clamp_min(mask.sum(dim=-1, keepdim=True), 1)  # noqa
            m_i = torch.sum(m_ij, dim=-2) / mask_sum
        else:
            m_i = torch.mean(m_ij, dim=-2)

        node_mlp_input = torch.cat((node_feats, m_i), dim=-1)
        node_out = self.node_residual(self.node_mlp(self.node_norm(node_mlp_input)), res=node_feats)
        coord_update = einsum('b i j, b i j a c -> b i a c', coord_weights, rel_coords)
        coord_out = self.coord_residual(coord_update, res=coords)
        return node_out, coord_out


class EGNN(nn.Module):
    def __init__(
            self,
            depth: int,
            node_dim: int,
            pair_dim: int,
            coord_dim_in: int,
            coord_dim_hidden: int = None,
            coord_dim_out: int = None,
            coord_scale: float = 0.1,
            **kwargs
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for ind in range(depth):
            self.layers.append(
                EGNNLayer(
                    node_dim=node_dim,
                    coord_dim=default(coord_dim_hidden, coord_dim_in),
                    pair_dim=pair_dim,
                    **kwargs
                )
            )
        coord_dim_hidden = None if default(coord_dim_hidden, coord_dim_in) <= coord_dim_in else coord_dim_hidden
        self.coord_in = VNLinear(dim_in=coord_dim_in, dim_out=coord_dim_hidden - coord_dim_in) \
            if exists(coord_dim_hidden) else None

        self.coord_out = VNLinear(dim_in=default(coord_dim_hidden, coord_dim_in), dim_out=coord_dim_out) \
            if exists(coord_dim_out) else None

        self.coord_scale = coord_scale

    def forward(
            self,
            node_feats: Tensor,
            coords: Tensor,
            pair_feats: Tensor,
            mask: Optional[Tensor] = None,
            return_coord_changes: bool = False,
    ):
        coor_changes = []
        coords = coords * self.coord_scale
        if exists(self.coord_in):
            coords = torch.cat((coords, self.coord_in(coords)), dim=-2)

        for egnn in self.layers:
            node_feats, coords = egnn(
                node_feats=node_feats,
                pair_feats=pair_feats,
                coords=coords,
                mask=mask,
            )

            if return_coord_changes:
                coor_changes.append(coords.detach())
        if exists(self.coord_out):
            coords = self.coord_out(coords)

        if return_coord_changes:
            return node_feats, coords * (1 / self.coord_scale), coor_changes
        return node_feats, coords * (1 / self.coord_scale)


class EGNNConfig(NamedTuple):
    """Config for EGNN"""
    depth: int
    node_dim: int
    pair_dim: int
    coord_dim_in: int
    coord_dim_hidden: int = None
    coord_dim_out: int = None
    coord_scale: float = 0.1
    m_dim: int = 16
    dropout: float = 0.0
    norm_rel_coords: bool = False
    use_nearest: bool = False
    max_radius: float = 20
    top_k: int = 32
    lin_proj_coords: bool = False
    use_rezero: bool = False

    def kwargs(self):
        """Get attributes of config as dict."""
        return self._asdict()  # noqa


"""
if __name__ == "__main__":
    b, n, a = 1, 20, 3
    node_dim, pair_dim = 10, 5
    net = EGNN(
        depth=2,
        node_dim=node_dim,
        pair_dim=pair_dim,
        coord_dim_in=a,
        coord_dim_hidden=10,
        coord_dim_out=6,
        coord_scale=0.1,
        m_dim=16,
        dropout=0,
        norm_rel_coords=True,
        use_nearest=True,
        top_k=40,
        max_radius=20,
        use_rezero=False,
        lin_proj_coords=True,
    )
    node_feats = torch.randn(b, n, node_dim)
    pair_feats = torch.randn(b, n, n, pair_dim)
    coords = torch.randn(b, n, a, 3)
    mask = None

    node_feats, coords = net(node_feats=node_feats, pair_feats=pair_feats, coords=coords, mask=mask)

    print(node_feats.shape)
    print(coords.shape)
"""
