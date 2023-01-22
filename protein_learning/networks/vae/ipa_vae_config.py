"""Config class for GT-VAE"""
from typing import Tuple, Union

from protein_learning.networks.geometric_gt.geom_gt_config import GeomGTConfig
from protein_learning.networks.vae.ipa_vae import GTEncoder, GTDecoder
from protein_learning.networks.vae.vae_abc import VAE


class IPAVAEConfig:
    """VAE config"""

    def __init__(
            self,
            node_dim: int,
            pair_dim: int,
            coord_dim_out: int,
            latent_dim: int,

            # encoder/ decoder args
            use_ipa: Tuple[Union[int, bool], Union[int, bool]],
            depth: Tuple[int, int],
            share_weights: Tuple[Union[int, bool], Union[int, bool]],
            node_dropout: Tuple[float, float],

            # encoder/ decoder node kwargs
            node_heads: Tuple[int, int],
            node_dim_query: Tuple[int, int],
            node_dim_value: Tuple[int, int],
            node_dim_query_point: Tuple[int, int],
            node_dim_value_point: Tuple[int, int],
            use_dist_attn: Tuple[Union[int, bool], Union[int, bool]],

            # encoder/ decoder pair_kwargs
            pair_heads: Tuple[int, int],
            pair_dim_head: Tuple[int, int],
            pair_dropout: Tuple[float, float],
            do_tri_mul: Tuple[Union[int, bool], Union[int, bool]],
            do_tri_attn: Tuple[Union[int, bool], Union[int, bool]],

            extra_node_feat_dim: int = 0,
            extra_pair_feat_dim: int = 0,

            scale_rigid_update: bool = False,
    ):
        self.coord_dim_out = coord_dim_out
        self.latent_dim = latent_dim
        self.node_dim = node_dim
        self.pair_dim = pair_dim
        self.depth = depth
        self.share_weights = list(map(bool, share_weights))
        self.use_ipa = list(map(bool, use_ipa))

        node_kwargs = lambda idx: dict(
            dropout=node_dropout[idx],
            heads=node_heads[idx],
            dim_query=node_dim_query[idx],
            dim_head=node_dim_query[idx],
            dim_value=node_dim_value[idx],
            dim_query_point=node_dim_query_point[idx],
            dim_value_point=node_dim_value_point[idx],
            use_dist_attn=bool(use_dist_attn[idx])
        )

        pair_kwargs = lambda idx: dict(
            dropout=pair_dropout[idx],
            heads=pair_heads[idx],
            dim_head=pair_dim_head[idx],
            do_tri_mul=bool(do_tri_mul[idx]),
            do_tri_attn=bool(do_tri_attn[idx]),
        )

        self.extra_node_feat_dim = extra_node_feat_dim
        self.extra_pair_feat_dim = extra_pair_feat_dim

        _config = lambda i: GeomGTConfig(
            node_dim=self.node_dim,
            pair_dim=self.pair_dim,
            depth=self.depth[i],
            use_ipa=self.use_ipa[i],
            share_weights=self.share_weights[i],
            node_update_kwargs=node_kwargs(i),
            pair_update_kwargs=pair_kwargs(i),
            scale_rigid_update=scale_rigid_update,
        )

        self.encoder_config = _config(0)
        self.decoder_config = _config(1)

    def get_vae(self) -> VAE:
        """Get VAE Model"""
        return VAE(
            encoder=GTEncoder(self.encoder_config),
            decoder=GTDecoder(
                self.decoder_config,
                coord_dim_out=self.coord_dim_out
            ),
            node_dim_hidden=self.node_dim,
            pair_dim_hidden=self.pair_dim,
            latent_dim=self.latent_dim,
            extra_node_dim=self.extra_node_feat_dim,
            extra_pair_dim=self.extra_pair_feat_dim,
        )


def add_vae_options(parser):
    """Add feature options to parser"""
    vae_options = parser.add_argument_group("vae_args")
    vae_options.add_argument("--node_dropout", type=float, default=[0, 0], nargs="+")
    vae_options.add_argument("--latent_dim", type=int, default=32)
    vae_options.add_argument("--scale_rigid_update", action="store_true")

    # encoder/ decoder node kwargs
    kwargs = "use_ipa depth share_weights use_dist_attn".split()
    node_kwargs = "heads dim_query dim_value dim_query_point dim_value_point".split()
    node_kwargs = ["node_" + x for x in node_kwargs]
    kwargs += node_kwargs
    pair_kwargs = "pair_heads pair_dim_head pair_dropout do_tri_mul do_tri_attn".split()
    kwargs += pair_kwargs
    for opt in kwargs:
        vae_options.add_argument(f"--{opt}", type=int, nargs="+", default=None)

    return parser, vae_options
