from typing import Dict, Tuple, Optional

from einops import rearrange  # noqa
from torch import nn

from protein_learning.networks.common.helpers.neighbor_utils import NeighborInfo
from protein_learning.networks.common.helpers.torch_utils import batched_index_select, masked_mean
from protein_learning.networks.common.utils import exists, default
from protein_learning.networks.tfn.repr.fiber import default_tymap, to_order
from protein_learning.networks.common.equivariant.fiber_units import FiberLinear, FiberResidual
import torch.nn.functional as F  # noqa
from torch import Tensor
import torch.utils.checkpoint as checkpoint
from protein_learning.networks.common.helpers.torch_utils import fused_gelu as GELU  # noqa
from protein_learning.networks.tfn.tfn_config import TFNConfig
from protein_learning.networks.common.net_utils import SplitLinear
from functools import partial


class ConvSE3(nn.Module):
    def __init__(self, config: TFNConfig):
        super().__init__()
        self.config = config
        self.fiber_in, self.fiber_out = config.fiber_in, config.fiber_out

        # Neighbor -> center weights
        self.kernel_unary = nn.ModuleDict()
        if not config.fuse_tfn_kernels:
            for (di, mi), (do, mo) in self.fiber_in * self.fiber_out:
                self.kernel_unary[f"({di},{do})"] = PairwiseConv(
                    di,
                    mi,
                    do,
                    mo,
                    edge_dim=config.get_edge_dim(),
                    radial_dropout=config.radial_dropout,
                    radial_mult=config.radial_mult,
                )
        else:
            kernel_feats = self.fiber_in * self.fiber_out
            self.kernel_unary = FusedConv(
                kernel_feats,
                edge_dim=config.get_edge_dim(),
                radial_dropout=config.radial_dropout,
                radial_mult=config.radial_mult,
            )

        # Center -> center weights
        if config.self_interaction:
            assert config.pool, "must pool edges if followed with self interaction"
            self.self_interact = FiberLinear(config.fiber_in, config.fiber_out)
            self.self_interact_sum = FiberResidual(safe_mode=True)

        self.ty_map = default(config.ty_map, default_tymap(config.fiber_in, config.fiber_out))

    def forward(
        self,
        features: Dict[str, Tensor],
        edge_info: Tuple[Optional[Tensor], NeighborInfo],
        basis: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """TFN-based convolution

        :param features: Dict mapping feature types to feature values
        :param edge_info: edge features and neighbor info
        :param basis: equivariant basis mapping feats of type i to type j.
        :return: dict mapping feature types to output shapes
        """
        config = self.config
        edges, neighbor_info = edge_info
        neighbor_indices, neighbor_masks = neighbor_info.indices, neighbor_info.mask
        # select neighbors for graph convolution
        selected_input = {deg: batched_index_select(x, neighbor_indices, dim=1) for deg, x in features.items()}
        edge_features = config.get_edge_features(edges=edges, nbr_info=neighbor_info)

        # go through every permutation of input degree type to output degree type
        outputs = {}
        for degree_out, m_out in config.fiber_out:
            output = 0
            degree_out_key = str(degree_out)
            for degree_in, m_in in config.fiber_in:
                # short circuit if mapping between types not specified
                if not self.ty_map[degree_in][degree_out]:
                    continue

                etype = f"({degree_in},{degree_out})"
                kernel_fn = self.kernel_unary[etype]

                x = selected_input[str(degree_in)]
                # process input, edges, and basis in chunks along the sequence dimension
                _basis = basis[f"{degree_in},{degree_out}"]
                if config.checkpoint and self.training:
                    kernel_out = checkpoint.checkpoint(kernel_fn.forward, edge_features, x, _basis)
                else:
                    kernel_out = kernel_fn(edge_features, x, _basis)
                output = output + kernel_out

            output = output.view(*x.shape[:3], -1, to_order(degree_out))  # noqa added reshape before mean
            if config.pool:
                output = masked_mean(output, neighbor_masks, dim=2) if exists(neighbor_masks) else output.mean(dim=2)

            leading_shape = x.shape[:2] if config.pool else x.shape[:3]
            output = output.view(*leading_shape, -1, to_order(degree_out))
            outputs[degree_out_key] = output

        if config.self_interaction:
            self_interact_out = self.self_interact(features)
            outputs = self.self_interact_sum(outputs, self_interact_out)

        if config.fuse_tfn_kernels:
            self.kernel_unary.clear()

        return outputs


class PairwiseConv(nn.Module):
    """SE(3)-equivariant convolution between two single-type features"""

    def __init__(
        self,
        degree_in,
        nc_in,
        degree_out,
        nc_out,
        edge_dim=0,
        radial_dropout=0.0,
        radial_mult: float = 2,
    ):
        super().__init__()
        self.degree_in = degree_in
        self.degree_out = degree_out
        self.nc_in = nc_in
        self.nc_out = nc_out

        self.num_freq = to_order(min(degree_in, degree_out))
        self.d_out = to_order(degree_out)
        self.edge_dim = edge_dim
        self.rp = RadialFunc(
            self.num_freq, nc_in, nc_out, edge_dim, dropout=radial_dropout, mid_dim=int(edge_dim * radial_mult)
        )

    def forward(self, edges, feats, basis) -> Tensor:
        out_shape = (*feats.shape[:3], -1, to_order(self.degree_out))
        feats = feats.view(-1, self.nc_in, to_order(self.degree_in))
        num_edges, in_dim = feats.shape[0], to_order(self.degree_in)
        radial_weights = self.rp(edges).view(-1, self.nc_out, self.nc_in * self.num_freq)
        tmp = (feats @ basis).view(num_edges, -1, self.d_out)
        return (radial_weights @ tmp).view(out_shape)


class FusedConv(nn.Module):
    def __init__(
        self,
        feat_dims,
        edge_dim=0,
        radial_dropout=0.0,
        radial_mult: float = 2,
    ):
        super(FusedConv, self).__init__()
        self.dims = dict()
        split_sizes = []
        for (di, mi), (do, mo) in feat_dims:
            key = f"({di},{do})"
            num_freq = to_order(min(di, do))
            self.dims[key] = (di, mi, do, mo)
            split_sizes.append(num_freq * mi * mo)
        self._radial_weights = None
        self.edge_dim = edge_dim
        mid_dim = edge_dim * radial_mult * len(self.dims)
        self.rp = nn.Sequential(
            nn.Linear(edge_dim, mid_dim),
            GELU,
            nn.Dropout(radial_dropout) if radial_dropout > 0 else nn.Identity(),
            nn.Linear(mid_dim, mid_dim),
            GELU,
            nn.Dropout(radial_dropout) if radial_dropout > 0 else nn.Identity(),
            SplitLinear(mid_dim, dim_out=sum(split_sizes), sizes=split_sizes, bias=False),
        )

    def radial_weights(self, edges, key):
        if self._radial_weights is None:
            self._radial_weights = {}
            rp = self.rp(edges)
            for i, k in enumerate(self.dims):
                di, mi, do, mo = self.dims[k]
                self._radial_weights[k] = rearrange(rp[i], "... (o i f) -> ... o () i () f", i=mi, o=mo)
        return self._radial_weights[key]

    def forward(self, edges, feats, basis, key) -> Tensor:
        di, mi, do, mo = self.dims[key]
        rp = self.radial_weights(edges, key)
        out_shape = (*feats.shape[:3], -1, to_order(do))
        feats = feats.view(-1, mi, to_order(di))
        num_edges, in_dim = feats.shape[0], to_order(di)
        radial_weights = rp.view(-1, mo, mi * to_order(min(di, do)))
        tmp = (feats @ basis).view(num_edges, -1, to_order(do))
        return (radial_weights @ tmp).view(out_shape)

    def __getitem__(self, key):
        return partial(self.forward, key=key)

    def clear(self):  # noqa
        self._radial_weights = None


class RadialFunc(nn.Module):
    """NN parameterized radial profile function."""

    def __init__(
        self,
        num_freq: int,
        in_dim: int,
        out_dim: int,
        edge_dim=None,
        mid_dim=None,
        nonlin=GELU,
        hidden_layer: bool = True,
        compress=False,
        dropout=0.0,
    ):
        super().__init__()
        self.num_freq = num_freq
        self.in_dim = in_dim
        self.edge_dim = default(edge_dim, 0)
        mid_dim = default(mid_dim, edge_dim)
        self.out_dim = out_dim
        bias = dropout > 0

        layer = lambda i, o, norm=True: nn.ModuleList(
            [
                nn.Linear(i, o),
                nn.LayerNorm(o) if norm else nn.Identity,
                nonlin,
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            ]
        )
        if not compress:
            self.net = nn.Sequential(
                *layer(edge_dim, mid_dim),
                *layer(mid_dim, mid_dim) if hidden_layer else nn.Identity(),
                nn.Linear(mid_dim, num_freq * in_dim * out_dim, bias=bias),
            )
        else:
            mid_dim, code_dim = edge_dim // 2, edge_dim // 4
            self.net = nn.Sequential(
                *layer(edge_dim, mid_dim, norm=True),
                *layer(mid_dim, code_dim, norm=True),
                *layer(code_dim, mid_dim, norm=True),
                nn.Linear(mid_dim, num_freq * in_dim * out_dim, bias=bias),
            )

    def forward(self, x) -> Tensor:
        return rearrange(self.net(x), "... (o i f) -> ... o () i () f", i=self.in_dim, o=self.out_dim)
