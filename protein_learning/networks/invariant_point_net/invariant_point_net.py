"""Invariant Coordinate Prediction"""
from typing import Optional

import torch.nn.functional as F  # noqa
from einops.layers.torch import Rearrange # noqa
from torch import nn, Tensor

from protein_learning.common.global_constants import get_logger
from protein_learning.common.helpers import exists, default
from protein_learning.common.rigids import Rigids
from protein_learning.networks.common.net_utils import Residual, LearnedOuterProd, FeedForward, PreNorm
from protein_learning.networks.evoformer.triangle_updates import TriangleMul
from protein_learning.networks.invariant_point_net.geometric_graph_attention import GeometricGraphAttentionBlock
from protein_learning.networks.invariant_point_net.ipn_config import IPNConfig
import torch.utils.checkpoint as checkpoint

logger = get_logger(__name__)


class PairUpdate(nn.Module):
    """Pair Update Module"""

    def __init__(self,
                 pair_dim,
                 scalar_dim,
                 ff_mult: float = 2,
                 num_ff_layers: int = 2,
                 tri_mul_hidden: Optional[int] = None,
                 do_checkpoint: bool = True,
                 ):
        super(PairUpdate, self).__init__()
        self.do_checkpoint = do_checkpoint
        self.pair_dim, self.scalar_dim = pair_dim, scalar_dim
        # triangle multiplicative updates
        tri_mul_hidden = default(tri_mul_hidden, min(128, pair_dim))
        self.tri_incoming = TriangleMul(
            dim_in=pair_dim, incoming=True, residual=Residual(), dim_hidden=tri_mul_hidden
        )
        self.tri_outgoing = TriangleMul(
            dim_in=pair_dim, incoming=False, residual=Residual(), dim_hidden=tri_mul_hidden
        )
        # scalar -> pair communication (outer product of scalar feats)
        self.outer = LearnedOuterProd(dim_in=scalar_dim, dim_out=pair_dim, pre_norm=True)
        self.outer_residual = Residual()
        # transition
        self.transition = PreNorm(pair_dim, FeedForward(pair_dim, mult=ff_mult, num_layers=num_ff_layers))
        self.transition_residual = Residual()

    def forward(self, scalar_feats: Tensor, pair_feats: Tensor, mask: Tensor):
        """Update pair features"""
        pair_feats = self.outer_residual(self.outer(scalar_feats), pair_feats)
        if self.do_checkpoint:
            pair_feats = checkpoint.checkpoint(self.tri_outgoing, pair_feats, mask)
            pair_feats = checkpoint.checkpoint(self.tri_incoming, pair_feats, mask)
        else:
            pair_feats = self.tri_outgoing(pair_feats, mask)
            pair_feats = self.tri_incoming(pair_feats, mask)
        return self.transition_residual(self.transition(pair_feats), pair_feats)


class InvariantPointNet(nn.Module):
    """Invariant Point Network"""

    def __init__(self,
                 config: IPNConfig,
                 ):
        super(InvariantPointNet, self).__init__()
        self.config, c = config, config

        self.layers = nn.ModuleList([
            nn.ModuleList(
                [
                    PairUpdate(
                        pair_dim=c.pair_dim,
                        scalar_dim=c.scalar_dim_in,
                        ff_mult=c.ff_mult,
                        num_ff_layers=c.num_ff_layers
                    ) if c.update_edges else nn.Identity(),
                    GeometricGraphAttentionBlock(
                        dim=c.scalar_dim_in,
                        attn_kwargs=c.attn_kwargs,
                        ff_mult=c.ff_mult,
                        ff_num_layers=c.num_ff_layers
                    ),
                    nn.LayerNorm(c.pair_dim),
                    # pair attn update residual
                    Residual() if c.augment_edges else None
                ]
            )
            for _ in range(config.depth)]
        )

        # shared rigid update accross all layers
        self.to_rigid_update = nn.Sequential(
            nn.LayerNorm(config.scalar_dim_in),
            nn.Linear(config.scalar_dim_in, 6)
        ) if c.use_rigids else None

        # point prediction net
        self.to_points = nn.Sequential(
            nn.LayerNorm(c.scalar_dim_in),
            nn.Linear(c.scalar_dim_in, 3 * c.dim_out(coord=True), bias=False),
            Rearrange("b n (d c) -> b n d c", c=3)
        ) if c.compute_coords else None

    def forward(
            self,
            scalar_feats: Tensor,
            pair_feats: Tensor,
            rigids: Optional[Rigids] = None,
            mask: Optional[Tensor] = None,
            scale_factor: float = 10.,
            **kwargs  # noqa
    ):
        """Update scalar and pair features + predict coordinates"""
        config, device = self.config, scalar_feats.device
        b, n, *_ = scalar_feats.shape
        # if no initial quaternions passed in, start from identity
        rigids = rigids if exists(rigids) else \
            Rigids.IdentityRigid(leading_shape=(b, n), device=device)  # noqa
        rigids = rigids.scale(1 / scale_factor)

        for idx in range(config.depth):
            # detach the rotation from rigids (allegedly provides more stable training)
            rigids = rigids.detach_rot()
            pair_update, node_update, pair_attn_norm, pair_residual = self.layers[idx]
            # update pair and scalar features
            if config.update_edges:
                logger.info("updating pair feats")
                pair_feats = pair_update(scalar_feats=scalar_feats, pair_feats=pair_feats, mask=mask)
            scalar_feats, pair_attn_update = node_update(
                scalar_feats=scalar_feats,
                pair_feats=pair_attn_norm(pair_feats),
                rigids=rigids,
                mask=mask,
            )
            if config.use_rigids:
                logger.info("updating rigids")
                quaternion_update, translation_update = self.to_rigid_update(scalar_feats).chunk(2, dim=-1)
                quaternion_update = F.pad(quaternion_update, (1, 0), value=1.)
                rigids = rigids.compose(Rigids(quaternion_update, translation_update))

            if config.augment_edges:
                logger.info("augmenting pair feats")
                pair_feats = pair_residual(pair_attn_update, pair_feats)

        # return points
        rigids = rigids.scale(scale_factor)
        points = rigids.apply(scale_factor * self.to_points(scalar_feats))
        return scalar_feats, pair_feats, rigids, points
