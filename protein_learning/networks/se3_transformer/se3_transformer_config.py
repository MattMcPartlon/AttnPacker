"""SE3-Transformer configuration"""
from __future__ import annotations

from typing import Union, Tuple, Dict, Optional, Any, Callable
from protein_learning.networks.common.utils import default
from protein_learning.networks.common.helpers.torch_utils import fused_gelu as GELU  # noqa
from protein_learning.networks.tfn.repr.fiber import Fiber, cast_fiber
from protein_learning.networks.tfn.tfn_config import TFNConfig
from protein_learning.networks.se3_transformer.se3_attention_config import SE3AttentionConfig
from protein_learning.networks.config.net_config import NetConfig
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from protein_learning.common.helpers import parse_bool

NONLIN = GELU


class SE3TransformerConfig(NetConfig):
    """Configuration for se3-transformer"""

    def __init__(
            self,
            fiber_in: Union[int, Tuple, Fiber, Dict],
            fiber_out: Union[int, Tuple, Fiber, Dict],
            fiber_hidden: Union[int, Tuple, Fiber, Dict] = (128, 16),
            heads: Union[int, Tuple, Dict] = (10, 10),
            dim_heads: Union[int, Tuple, Dict] = (20, 4),
            edge_dim: int = None,
            depth: int = 6,
            conv_in_layers: int = 1,
            conv_out_layers: int = 1,
            differentiable_coords: bool = False,
            global_feats_dim: Optional[int] = None,
            attend_self: bool = True,
            use_null_kv: bool = True,
            linear_proj_keys: bool = False,
            fourier_encode_rel_dist: bool = False,
            fourier_rel_dist_feats: int = 4,
            share_keys_and_values: bool = False,
            hidden_mult=2,
            project_out: bool = True,
            norm_out: bool = True,
            normalize_radial_dists=True,
            use_re_zero: bool = True,
            share_attn_weights: bool = True,
            use_dist_sim: bool = False,
            learn_head_weights: bool = False,
            use_coord_attn: bool = True,
            append_rel_dist: bool = False,
            append_edge_attn: bool = False,
            dropout: float = 0,
            use_dist_conv: bool = False,
            pairwise_dist_conv: bool = False,
            num_dist_conv_filters: int = 16,
            radial_dropout: float = 0.2,
            radial_compress: bool = False,
            radial_mult: float = 2,
            pair_bias: bool = True,
            append_norm: bool = False,
            checkpoint_tfn: bool = False,
            attn_ty: str = "tfn",
            nonlin: Callable = GELU,
            use_coord_layernorm: bool = False,
    ):
        super(SE3TransformerConfig, self).__init__()
        # se3 transformer input fiber
        self.fiber_in = cast_fiber(fiber_in)
        self.fiber_hidden = default(cast_fiber(fiber_hidden), self.fiber_in)
        # se3 transformer output fiber
        self.fiber_out = default(cast_fiber(fiber_out), self.fiber_hidden)
        self.global_feats_dim = global_feats_dim
        self.max_degrees = max([self.fiber_in.n_degrees, self.fiber_hidden.n_degrees, self.fiber_out.n_degrees])
        self.edge_dim = edge_dim
        self.depth = depth
        self.conv_in_layers = conv_in_layers
        self.conv_out_layers = conv_out_layers
        self.project_out = project_out
        self.norm_out = norm_out
        self.normalize_radial_dists = normalize_radial_dists
        self.append_norm = append_norm
        self.pair_bias = pair_bias

        # options
        self.dropout = dropout
        self.differentiable_coords = differentiable_coords
        self.append_rel_dist = append_rel_dist
        self.append_edge_attn = append_edge_attn
        self.use_re_zero = use_re_zero
        self.radial_dropout = radial_dropout
        self.radial_compress = radial_compress
        self.radial_mult = radial_mult
        self.checkpoint_tfn = checkpoint_tfn

        # attention specific args
        self.heads = cast_fiber(heads, degrees=self.fiber_hidden.n_degrees)
        self.dim_head = cast_fiber(dim_heads, degrees=self.fiber_hidden.n_degrees)
        self.attend_self = attend_self
        self.use_null_kv = use_null_kv
        self.linear_proj_keys = linear_proj_keys
        self.fourier_encode_rel_dist = fourier_encode_rel_dist
        self.fourier_rel_dist_feats = fourier_rel_dist_feats
        self.share_keys_and_values = share_keys_and_values
        self.hidden_mult = hidden_mult
        self.share_attn_weights = share_attn_weights
        self.use_dist_sim = use_dist_sim
        self.learn_head_weights = learn_head_weights
        self.use_coord_attn = use_coord_attn
        self.use_dist_conv = use_dist_conv
        self.pairwise_dist_conv = pairwise_dist_conv
        self.num_dist_conv_filters = num_dist_conv_filters

        self.attn_ty = attn_ty
        self.nonlin = nonlin
        self.use_coord_layernorm = use_coord_layernorm

    @property
    def coord_dims(self) -> Tuple[int, int, int]:
        """Coordinate feature dimensions (in, hidden, out)"""
        return self.fiber_in[1], self.fiber_hidden[1], self.fiber_out[1]

    def scalar_dims(self) -> Tuple[int, int, int]:
        """Scalar feature dimensions (in, hidden, out)"""
        return self.fiber_in[0], self.fiber_hidden[0], self.fiber_out[0]

    def pair_dims(self) -> Tuple[int, int, int]:
        """Pair feature dimensions (in, hidden, out)"""
        return tuple([self.edge_dim] * 3)

    def override(self, attr: str, val: Any, clone: bool = True) -> SE3TransformerConfig:
        new_config = SE3TransformerConfig(**vars(self)) if clone else self
        new_config.__setattr__(attr, val)
        return new_config

    def tfn_config(
            self,
            fiber_in: Optional[Fiber] = None,
            fiber_out: Optional[Fiber] = None,
            pool: bool = False,
            self_interaction: bool = False,
            **override
    ) -> TFNConfig:
        return TFNConfig(
            fiber_in=default(fiber_in, self.fiber_hidden),
            fiber_out=default(fiber_out, self.fiber_hidden),
            self_interaction=self_interaction,
            pool=pool,
            edge_dim=self.edge_dim,
            fourier_encode_dist=self.fourier_encode_rel_dist,
            num_fourier_features=self.fourier_rel_dist_feats,
            radial_dropout=self.radial_dropout,
            radial_mult=self.radial_mult,
            checkpoint=self.checkpoint_tfn,
        ).override(**override)

    def attn_config(self, **override):
        return SE3AttentionConfig(
            heads=self.heads,
            dim_heads=self.dim_head,
            edge_dim=self.edge_dim,
            global_feats_dim=self.global_feats_dim,
            attend_self=self.attend_self,
            use_null_kv=self.use_null_kv,
            share_attn_weights=self.share_attn_weights,
            use_dist_sim=self.use_dist_sim,
            learn_head_weights=self.learn_head_weights,
            use_coord_attn=self.use_coord_attn,
            append_edge_attn=self.append_edge_attn,
            use_dist_conv=self.use_dist_conv,
            pairwise_dist_conv=self.pairwise_dist_conv,
            num_dist_conv_filters=self.num_dist_conv_filters,
            pair_bias=self.pair_bias,
            append_norm=self.append_norm,
        ).override(**override)


def add_se3_options(_parser):

    parser = _parser.add_argument_group("se3_args")
    parser.add_argument(
        '--fiber_in',
        help='',
        type=int,
        nargs="+",
        default=(-1, -1)
    )

    parser.add_argument(
        '--fiber_out',
        help='',
        type=int,
        nargs="+",
        default=(-1, -1)
    )

    parser.add_argument(
        '--fiber_hidden',
        help='',
        type=int,
        nargs="+",
        default=(128, 16)
    )

    parser.add_argument(
        '--se3_heads',
        help='',
        type=int,
        nargs="+",
        default=(10, 10)
    )

    parser.add_argument(
        '--se3_dim_heads',
        help='',
        type=int,
        nargs="+",
        default=(20, 4)
    )

    parser.add_argument(
        '--se3_edge_dim',
        help='',
        type=int,
        default=None
    )

    parser.add_argument(
        '--se3_depth',
        help='',
        type=int,
        default=6
    )

    parser.add_argument("--append_norm", action="store_true")
    parser.add_argument("--append_rel_dist",action="store_true")
    parser.add_argument("--append_edge_attn",action="store_true")
    parser.add_argument("--learn_head_weights",action="store_true")

    return _parser
