from typing import NamedTuple, Union, Tuple, Dict, Optional
from protein_learning.networks.tfn.repr.fiber import Fiber
from protein_learning.networks.common.utils import update_named_tuple
from torch import Tensor
from protein_learning.networks.common.helpers.neighbor_utils import NeighborInfo
from protein_learning.networks.common.helpers.torch_utils import (
    safe_cat,
    fourier_encode,
)
from einops import rearrange
from protein_learning.networks.common.constants import DIST_SCALE
import torch


class TFNConfig(NamedTuple):
    fiber_in: Union[int, Tuple, Fiber, Dict] = None
    fiber_out: Union[int, Tuple, Fiber, Dict] = None
    edge_dim: int = 0
    self_interaction: bool = True
    pool: bool = True
    radial_dropout: float = 0.0
    radial_mult: float = 2.0
    checkpoint: bool = False
    ty_map: Optional[Tensor] = None
    fourier_encode_dist: bool = False
    num_fourier_features: int = 4
    fuse_tfn_kernels: bool = False

    def override(self, **kwargs):
        return update_named_tuple(TFNConfig, self, **kwargs)

    def get_edge_dim(self):
        pad = 0 if not self.fourier_encode_dist else (self.num_fourier_features * 2)
        return self.edge_dim + pad + 1

    def get_edge_features(self, edges: Optional[Tensor], nbr_info: NeighborInfo) -> Tensor:
        rel_dist = rearrange(nbr_info.rel_dists, 'b m n -> b m n ()')
        assert not rel_dist.requires_grad
        edge_features = safe_cat(edges, rel_dist * DIST_SCALE, dim=-1)

        if self.fourier_encode_dist:
            fourier_rel_dist = fourier_encode(rel_dist,
                                              num_encodings=self.num_fourier_features,
                                              include_self=False
                                              )
            edge_features = torch.cat((fourier_rel_dist.detach(), edge_features), dim=-1)
        return edge_features
