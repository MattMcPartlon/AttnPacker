"""Simplified Evoformer (no MSA)"""
from typing import Optional, Tuple

import torch
import torch.nn.functional as F  # noqa
from einops import rearrange, repeat  # noqa
from torch import nn, Tensor

from protein_learning.common.global_constants import get_logger
from protein_learning.networks.attention.node_attention import NodeUpdateBlock
from protein_learning.networks.attention.pair_attention import PairUpdateBlock
from protein_learning.networks.evoformer.evoformer_config import EvoformerConfig

logger = get_logger(__name__)
TorchList = nn.ModuleList  # noqa
Proj = lambda dim_in, dim_out: nn.Sequential(nn.LayerNorm(dim_in), nn.Linear(dim_in, dim_out))


class Evoformer(nn.Module):  # noqa
    """Evoformer"""

    def __init__(self, config: EvoformerConfig, freeze: bool = False):
        super().__init__()
        self.config = config
        node_in, node_hidden, node_out = config.scalar_dims
        edge_in, edge_hidden, edge_out = config.pair_dims

        # Input/Output projections
        self.node_project_out = Proj(node_hidden, node_out) if config.project_out else nn.Identity()
        self.edge_project_out = Proj(edge_hidden, edge_out) if config.project_out else nn.Identity()
        self.layers = get_transformer_layers(config)
        _freeze(self.layers, freeze)

    def forward(
            self,
            node_feats: torch.Tensor,
            edge_feats: torch.Tensor,
            adj_mask: Optional[Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run EvoFormer Forward Pass"""

        for layer, (node_block, edge_block) in enumerate(self.layers):
            # node to node attention (with edge bias)
            node_feats = node_block(node_feats=node_feats, pair_feats=edge_feats, mask=adj_mask)
            edge_feats = edge_block(node_feats=node_feats, pair_feats=edge_feats, mask=adj_mask)
        return self.node_project_out(node_feats), self.edge_project_out(edge_feats)


def get_transformer_layers(config: EvoformerConfig):
    """Helper method for getting Evoformer layers"""
    # set up transformer blocks
    pair_hidden, node_hidden = config.edge_dim, config.node_dim
    layers = TorchList()
    for i in range(config.depth):
        layers.append(
            TorchList(
                [
                    NodeUpdateBlock(
                        node_dim=node_hidden,
                        pair_dim=pair_hidden,
                        ff_mult=config.node_ff_mult,
                        use_ipa=False,
                        dropout=config.node_dropout,
                        dim_out=node_hidden,
                        heads=config.node_attn_heads,
                        dim_head=config.node_dim_head,

                    ),
                    PairUpdateBlock(
                        pair_dim=pair_hidden,
                        node_dim=node_hidden,
                        heads=config.edge_attn_heads,
                        dim_head=config.edge_dim_head,
                        dropout=config.edge_dropout,
                        tri_mul_dim=config.triangle_mul_dim,
                        do_checkpoint=config.checkpoint,
                        ff_mult=config.edge_ff_mult,
                        do_tri_mul=config.do_tri_mul,
                        do_tri_attn=config.do_tri_attn,
                    )
                ]
            )
        )

    return layers


def _freeze(layers: TorchList, freeze: bool):
    """Freeze parameters in all but final layer"""
    if not freeze:
        return
    print(f"[INFO] Freezing evoformer layers"
          f"1-{len(layers) - 1} (of {len(layers)})")
    for layer in layers[:-1]:  # noqa
        for module in layer:
            for param in module.parameters():
                param.requires_grad_(False)
