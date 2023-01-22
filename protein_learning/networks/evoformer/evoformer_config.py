"""Evoformer Config"""
from typing import Optional
from protein_learning.networks.common.utils import default
from protein_learning.networks.config.net_config import NetConfig
from typing import Tuple


class EvoformerConfig(NetConfig):
    """Evoformer configuration"""

    def __init__(
            self,
            node_dim: int,
            edge_dim: int,
            node_dim_out: Optional[int] = None,
            edge_dim_out: Optional[int] = None,
            do_tri_attn: bool = True,
            do_tri_mul: bool = True,
            depth: int = 10,
            node_dropout: float = 0,
            edge_dropout: float = 0,
            edge_attn_heads: int = 4,
            edge_dim_head: int = 32,
            triangle_mul_dim: Optional[int] = None,
            outer_prod_dim: int = 16,
            node_attn_heads: int = 12,
            use_nbr_attn: bool = False,
            node_dim_head: int = 20,
            checkpoint: bool = True,
            project_out: bool = False,
            node_ff_mult: int = 4,
            edge_ff_mult: int = 2,
    ):
        super(EvoformerConfig, self).__init__()
        self.node_dim = node_dim
        self.node_dim_out = default(node_dim_out, node_dim)
        self.edge_dim = edge_dim
        self.edge_dim_out = default(edge_dim_out, edge_dim)
        self.do_tri_attn, self.do_tri_mul = do_tri_attn, do_tri_mul
        self.depth = depth
        self.node_dropout, self.edge_dropout = node_dropout, edge_dropout
        self.node_attn_heads, self.edge_attn_heads = node_attn_heads, edge_attn_heads
        self.node_dim_head, self.edge_dim_head = node_dim_head, edge_dim_head
        self.triangle_mul_dim = default(triangle_mul_dim, edge_dim)
        self.outer_prod_dim = outer_prod_dim
        self.use_nbr_attn = use_nbr_attn
        self.checkpoint = checkpoint
        self.project_out = project_out
        self.node_ff_mult, self.edge_ff_mult = node_ff_mult, edge_ff_mult

    @property
    def scalar_dims(self) -> Tuple[int, int, int]:
        """pair dimensions (input, hidden, output)"""
        return self.node_dim, self.node_dim, self.node_dim_out

    @property
    def pair_dims(self) -> Tuple[int, int, int]:
        """pair dimensions (input, hidden, output)"""
        return self.edge_dim, self.edge_dim, self.edge_dim_out
