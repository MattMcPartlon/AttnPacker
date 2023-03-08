import time
from abc import abstractmethod
from typing import Union, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F  # noqa
from einops import rearrange, repeat  # noqa
from torch import Tensor
from torch import nn

from protein_learning.common.global_constants import get_logger
from protein_learning.common.helpers import time_fn
from protein_learning.networks.common.equivariant.fiber_units import (
    FiberNorm,
    FiberFeedForwardResidualBlock,
    FiberFeedForward,
    FiberResidual,
    FiberDropout,
)
from protein_learning.networks.common.helpers.neighbor_utils import NeighborInfo, get_neighbor_info
from protein_learning.common.helpers import maybe_add_batch
from protein_learning.networks.common.helpers.torch_utils import batched_index_select
from protein_learning.networks.common.helpers.torch_utils import fused_gelu as GELU  # noqa
from protein_learning.networks.tfn.repr.basis import get_basis
from protein_learning.networks.common.utils import exists
from protein_learning.networks.se3_transformer.attention.tfn_attention import TFNAttention
from protein_learning.networks.se3_transformer.se3_transformer_config import SE3TransformerConfig

logger = get_logger(__name__)


def get_attention_layer(config: SE3TransformerConfig) -> nn.Module:
    if config.attn_ty.lower() == "tfn":
        return TFNAttention(
            fiber_in=config.fiber_hidden,
            config=config.attn_config(),
            tfn_config=config.tfn_config(),
            share_keys_and_values=config.share_keys_and_values,
        )
    raise Exception(f"Attention not implemented for: {config.attn_ty}")


class AttentionBlock(nn.Module):
    def __init__(
        self,
        config: SE3TransformerConfig,
    ):
        super().__init__()
        self.attn = get_attention_layer(config=config)
        self.prenorm = FiberNorm(
            fiber=config.fiber_hidden,
            nonlin=config.nonlin,  # noqa
            use_layernorm=config.use_coord_layernorm,
        )
        self.residual = FiberResidual(use_re_zero=config.use_re_zero)
        self.dropout = FiberDropout(fiber=config.fiber_hidden, p=config.dropout)

    def forward(
        self,
        features: Dict[str, Tensor],
        edge_info: NeighborInfo,
        basis: Dict[str, Tensor],
        global_feats: Optional[Union[Tensor, Dict[str, Tensor]]] = None,
    ) -> Dict[str, Tensor]:
        """Attention Block

        :param features: Dict mapping feature types to feature values
        :param edge_info: edge features
        :param basis: equivariant basis mapping feats of type i to type j.
        :param global_feats: global features
        :return: dict mapping feature types to hidden features
        """
        res = features
        outputs = self.prenorm(features)
        outputs = self.attn(features=outputs, edge_info=edge_info, basis=basis, global_feats=global_feats)
        return self.residual(self.dropout(outputs), res)


class AttentionLayer(nn.Module):
    def __init__(self, config: SE3TransformerConfig):
        super().__init__()
        self.attn_block = AttentionBlock(config)

        self.ff_residual = FiberFeedForwardResidualBlock(
            feedforward=FiberFeedForward(
                fiber_in=config.fiber_hidden,
                hidden_mult=config.hidden_mult,
                n_hidden=1,
            ),
            pre_norm=FiberNorm(
                fiber=config.fiber_hidden,
                nonlin=config.nonlin,  # noqa
                use_layernorm=config.use_coord_layernorm,
            ),
            use_re_zero=config.use_re_zero,
        )

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        edge_info: Tuple[torch.Tensor, NeighborInfo],
        basis,
        global_feats=None,
    ) -> Dict[str, torch.Tensor]:
        """Attention Layer

        norm(feats) -> x =AttentionBlock(feats) -> x = residual(x,feats)
        -> x = norm(x) -> residual(ff(x), x)

        :param features: Dict mapping feature types to feature values
        :param edge_info: edge features
        :param basis: equivariant basis mapping feats of type i to type j.
        :param global_feats: global features
        :return: dict mapping feature types to hidden features
        """
        attn_feats = self.attn_block(
            features=features,
            edge_info=edge_info,
            basis=basis,
            global_feats=global_feats,
        )
        return self.ff_residual(attn_feats)


def get_input(res_feats, pair_feats, coords, top_k=16, include_sc: bool = False):
    """Get Transformer input"""
    N, CA, C, O = map(lambda x: maybe_add_batch(x.unsqueeze(-2), 3), torch.unbind(coords[..., :4, :], -2))
    if not include_sc:
        crd_feats = torch.cat((N, C, O), dim=-2)
    else:
        crd_feats = torch.cat((coords[..., :1, :], coords[..., 2:36, :]), dim=-2)
        crd_feats = crd_feats if crd_feats.ndim == 4 else crd_feats.unsqueeze(0)

    nbr_info = get_neighbor_info(CA.squeeze(-2), max_radius=16, top_k=top_k)
    feats = {"0": res_feats, "1": crd_feats - CA}
    return feats, pair_feats, nbr_info


class SE3Transformer(nn.Module):
    def __init__(
        self,
        config: SE3TransformerConfig,
        freeze: bool = False,
        pre_norm_edges: bool = True,
        include_initial_sc: bool = False,
    ):
        super().__init__()
        self.config = config
        self.include_initial_sc = include_initial_sc

        # global features
        self.accept_global_feats = exists(config.global_feats_dim)

        # Attention layers
        self.attn_layers = nn.ModuleList([AttentionLayer(config=config) for _ in range(config.depth)])

        # freeze layers
        if freeze:
            self.freeze()

        self.edge_norm = (
            nn.LayerNorm(config.edge_dim) if (exists(config.edge_dim) and pre_norm_edges) else nn.Identity()
        )

    def freeze(self):
        n_layers = len(self.attn_layers)
        print(f"[INFO] Freezing SE(3)-Transformer layers " f"1-{n_layers - 1} (of {n_layers})")
        for module in self.attn_layers[:-1]:
            for param in module.parameters():
                param.requires_grad_(False)

    def forward(
        self,
        node_feats: torch.Tensor,
        pair_feats: torch.Tensor,
        coords: torch.Tensor,
        global_feats: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,
        basis_dir=None,
        **kwargs,  # noqa
    ) -> Dict[str, torch.Tensor]:
        """SE(3)-Equivariant Transforer

        :return: Dict of updated features.
        """
        config = self.config

        feats, edges, neighbor_info = get_input(node_feats, pair_feats, coords, include_sc=self.include_initial_sc)

        assert not (
            self.accept_global_feats ^ exists(global_feats)
        ), "you cannot pass in global features unless you init the class correctly"

        # convert features to dictionary representation
        feats = {"0": feats} if torch.is_tensor(feats) else feats
        feats["0"] = feats["0"] if len(feats["0"].shape) == 4 else feats["0"].unsqueeze(-1)
        global_feats = {"0": global_feats[..., None]} if torch.is_tensor(global_feats) else global_feats

        # check that input degrees and dimensions are as expected
        for deg, dim in config.fiber_in:
            feat_dim, feat_deg = feats[str(deg)].shape[-2:]
            assert dim == feat_dim, f" expected dim {dim} for input degree {deg}, got {feat_dim}"
            assert deg * 2 + 1 == feat_deg, (
                f"wrong degree for feature {deg}, expected " f": {deg * 2 + 1}, got : {feat_deg}"
            )

        if exists(edges):
            if edges.shape[1] == edges.shape[2]:
                edges = batched_index_select(edges, neighbor_info.indices, dim=2)
                edges = self.edge_norm(edges)

        # get basis
        basis_time, basis = time_fn(
            self.compute_basis,
            neighbor_info=neighbor_info,
            max_degree=config.max_degrees - 1,
            differentiable=config.differentiable_coords,
            dirname=basis_dir,
        )
        logger.info(f"computed basis in time : {np.round(basis_time, 3)}")

        # main logic
        x, edge_info = feats, (edges, neighbor_info)

        x = self.project_in(features=x, edge_info=edge_info, basis=basis)

        attn_start = time.time()
        for attn_layer in self.attn_layers:
            x = attn_layer(
                x,
                edge_info=edge_info,
                basis=basis,
                global_feats=global_feats,
            )
        logger.info(f"TFN-Transformer time (forward) : {np.round(time.time() - attn_start, 3)}")
        return self.project_out(features=x, edge_info=edge_info, basis=basis)

    @staticmethod
    def compute_basis(neighbor_info: NeighborInfo, max_degree: int, differentiable: bool, dirname=None):
        basis = get_basis(
            neighbor_info.rel_pos.detach(),
            max_degree=max_degree,
            differentiable=differentiable,
            dirname=dirname,
        )
        # reshape basis for faster / more memory efficient kernel computation
        for key in basis:
            b = basis[key].shape[0]
            i, _ = key.split(",")
            basis[key] = rearrange(basis[key], "... a b c d e -> ... a c d e b")
            n, top_k = neighbor_info.coords.shape[1], neighbor_info.top_k
            basis[key] = basis[key].reshape(b * n * top_k, 2 * int(i) + 1, -1)
        return basis

    @abstractmethod
    def project_in(
        self, features: Dict[str, Tensor], edge_info: Tuple[Optional[Tensor], NeighborInfo], basis: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        """Equivariant input projection

        :param features: Dict mapping feature types to feature values
        :param edge_info: edge features and neighbor info
        :param basis: equivariant basis mapping feats of type i to type j.
        :return: dict mapping feature types to hidden shapes
        """
        pass

    @abstractmethod
    def project_out(
        self, features: Dict[str, Tensor], edge_info: Tuple[Optional[Tensor], NeighborInfo], basis: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        """Equivariant output projection

        :param features: Dict mapping feature types to feature values
        :param edge_info: edge features and neighbor info
        :param basis: equivariant basis mapping feats of type i to type j.
        :return: dict mapping feature types to output shapes
        """
        pass

    def _to_output(self, features: Dict[str, Tensor], nbr_info: NeighborInfo):
        CA = nbr_info.coords
        node_feats = features["0"].squeeze(-1)
        coords = CA.unsqueeze(-2) + features["1"]
        return node_feats, coords
