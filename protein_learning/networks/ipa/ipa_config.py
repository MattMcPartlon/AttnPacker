"""Configuration for Invariant Point Attention"""
from typing import Optional
from protein_learning.networks.config.net_config import NetConfig
from typing import Tuple, Union, List


class IPAConfig(NetConfig):
    """Evoformer configuration"""

    def __init__(
            self,
            scalar_dim_in: int,
            pair_dim: Optional[int],
            coord_dim_out: Optional[int] = 4,
            heads: int = 8,
            scalar_kv_dims: Union[int, List[int]] = 16,
            point_kv_dims: Union[int, List[int]] = 4,
            depth: int = 1,
            dropout: float = 0,
            ff_mult: float = 4,
            num_ff_layers: int = 2,
            share_weights: bool = False,
            use_dist_sim: bool = True,
            use_rigids: bool = True,
    ):
        super(IPAConfig, self).__init__()
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
        self.share_weights = share_weights
        self.use_dist_sim = use_dist_sim
        self.use_rigids = use_rigids

    @property
    def require_pair_repr(self):
        """whether to require pair features"""
        return self.pair_dim is not None

    @property
    def compute_coords(self):
        """whether to compute output coordinates"""
        return self.coord_dim_out is not None

    @property
    def scalar_dims(self) -> Tuple[int, int, int]:
        """pair dimensions (input, hidden, output)"""
        return self.scalar_dim_in, -1, self.scalar_dim_in

    @property
    def pair_dims(self) -> Tuple[int, int, int]:
        """pair dimensions (input, hidden, output)"""
        return [self.pair_dim] * 3

    @property
    def coord_dims(self) -> Tuple[int, int, int]:
        """Coord dimensions (in, hidden, out)"""
        return -1, -1, self.coord_dim_out

    @property
    def attn_kwargs(self):
        """Key word arguments for IPA"""
        return dict(
            dim=self.scalar_dim_in,
            heads=self.heads,
            scalar_kv_dims=self.scalar_kv_dims,
            point_kv_dims=self.point_kv_dims,
            pairwise_repr_dim=self.pair_dim,
            require_pairwise_repr=self.require_pair_repr,
        )
