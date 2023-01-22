"""Invariant Point Network Config"""
from typing import Optional
from protein_learning.networks.config.net_config import NetConfig
from typing import Union, Tuple, List
from protein_learning.networks.common.attention_utils import SimilarityType
from protein_learning.common.helpers import default, exists


class IPNConfig(NetConfig):
    """Invariant Point Network Config"""

    def __init__(
            self,
            scalar_dim_in: int,
            pair_dim: Optional[int],
            coord_dim_out: Optional[int] = 4,
            heads: int = 8,
            scalar_kv_dims: Union[List[int], Tuple[int, int], int] = 16,
            point_kv_dims: Union[List[int], Tuple[int, int], int] = 4,
            depth: int = 1,
            dropout: float = 0,
            ff_mult: float = 4,
            num_ff_layers: int = 2,
            use_pairwise_kernel: bool = False,
            use_dist_sim: bool = True,
            use_rigids: bool = True,
            update_edges: bool = True,
            augment_edges: bool = True,
            use_pair_bias: bool = True,
    ):
        super(IPNConfig, self).__init__()
        to_dims = lambda x: x if (isinstance(x, list) or isinstance(x, tuple)) else [x] * 2
        scalar_kv_dims, point_kv_dims = map(to_dims, (scalar_kv_dims, point_kv_dims))
        self.scalar_kv_dims, self.point_kv_dims = scalar_kv_dims, point_kv_dims
        self.scalar_dim_in = scalar_dim_in
        self.pair_dim = pair_dim
        self.depth = depth
        self.dropout = dropout
        self.coord_dim_out = coord_dim_out
        self.heads = heads
        self.ff_mult = ff_mult
        self.num_ff_layers = num_ff_layers
        self.use_dist_sim = use_dist_sim
        self.use_pairwise_kernel = use_pairwise_kernel
        self.use_rigids = use_rigids
        self.update_edges = update_edges
        self.augment_edges = augment_edges
        self.sim_ty = SimilarityType.DISTANCE if self.use_dist_sim else SimilarityType.DOT_PROD
        self.use_pair_bias = use_pair_bias

    @property
    def require_pair_repr(self):
        return self.pair_dim is not None

    @property
    def compute_coords(self):
        return self.coord_dim_out is not None

    @property
    def scalar_dims(self) -> Tuple[int, int, int]:
        """pair dimensions (input, hidden, output)"""
        return [self.scalar_dim_in] * 3

    @property
    def pair_dims(self) -> Tuple[int, int, int]:
        """pair dimensions (input, hidden, output)"""
        return [self.pair_dim] * 3

    @property
    def coord_dims(self) -> Tuple[int, int, int]:
        """coordinate dimensions (input, hidden, output)"""
        return -1, -1, self.coord_dim_out

    @property
    def attn_kwargs(self):
        """Key word arguments for self attention"""
        return dict(
            scalar_dim=self.scalar_dim_in,
            pair_dim=self.pair_dim,
            heads=self.heads,
            scalar_kv_dims=self.scalar_kv_dims,
            point_kv_dims=self.point_kv_dims,
            sim_ty=self.sim_ty,
            return_pair_update=self.augment_edges,
            use_pair_bias=self.use_pair_bias,
        )
