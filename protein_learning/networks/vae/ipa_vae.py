"""Graph Transformer Encoder/Decoder implementations"""

from typing import Optional, Tuple

import torch
from einops.layers.torch import Rearrange  # noqa
from torch import Tensor, nn

from protein_learning.common.global_constants import get_logger
from protein_learning.common.rigids import Rigids
from protein_learning.networks.geometric_gt.geom_gt_config import GeomGTConfig
from protein_learning.networks.geometric_gt.geometric_graph_transformer import GraphTransformer
from protein_learning.networks.vae.vae_abc import (
    EncoderABC,
    DecoderABC,
)

logger = get_logger(__name__)


class GTEncoder(EncoderABC):
    """Encoder Graph Transformer"""

    def __init__(
            self,
            config: GeomGTConfig,

    ):
        super(GTEncoder, self).__init__()
        self.gt_encoder = GraphTransformer(config)

    def encode(
            self,
            node_feats: Tensor,
            pair_feats: Tensor,
            mask: Tensor,
            rigids: Optional[Rigids] = None,
            true_rigids: Optional[Rigids] = None,
            res_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Encode input"""
        # example forward pass
        node_feats, pair_feats, *_ = self.gt_encoder.forward(
            node_feats=node_feats,
            pair_feats=pair_feats,
            rigids=rigids,
            pair_mask=mask,
        )
        return node_feats, pair_feats


class GTDecoder(DecoderABC):  # noqa
    """Graph Transformer Decoder with IPA"""

    def __init__(
            self,
            config: GeomGTConfig,
            coord_dim_out: int,
    ):
        super(GTDecoder, self).__init__()
        self.gt_decoder = GraphTransformer(config)

        # predict points in local frame of each residue
        self.to_points = nn.Sequential(
            nn.LayerNorm(config.node_dim),
            nn.Linear(config.node_dim, 3 * coord_dim_out, bias=False),
            Rearrange("b n (d c) -> b n d c", c=3),
        ) if coord_dim_out > 0 else None

    def decode(
            self,
            node_feats: Tensor,
            pair_feats: Tensor,
            mask: Tensor,
            rigids: Optional[Rigids] = None,
            true_rigids: Optional[Rigids] = None,
            res_mask: Optional[Tensor] = None,
            **kwargs
    ) -> Tuple[Tensor, Tensor, Tensor, Rigids, Tensor]:
        """Decode"""

        out = self.gt_decoder.forward(
            node_feats=node_feats,
            pair_feats=pair_feats,
            rigids=rigids,
            true_rigids=true_rigids,
            res_mask=res_mask,
            pair_mask=mask,
            **kwargs
        )
        node_feats, pair_feats, rigids, aux_loss = out
        coords = self.get_coords(node_feats, rigids=rigids)
        return node_feats, pair_feats, coords, rigids, aux_loss

    def get_coords(
            self,
            node_feats: Tensor,
            rigids: Optional[Rigids] = None,
            CA_posn: int = 1
    ) -> Tensor:
        """Get coordinates from output"""
        assert rigids is not None
        # predict from node features
        local_points = self.to_points(node_feats)
        # replace predicted CA with rigid translation
        # (helps empirically)
        CA = torch.zeros_like(local_points[:, :, 1])
        local_points[:, :, CA_posn] = CA
        # place points in global frame by applying rigids
        return rigids.apply(local_points)
