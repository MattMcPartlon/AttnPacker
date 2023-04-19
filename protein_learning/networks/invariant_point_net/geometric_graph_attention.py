"""Invariant Point Attention"""

import torch
import torch.nn.functional as F  # noqa
from torch import nn, einsum, Tensor

from einops.layers.torch import Rearrange  # noqa
from einops import rearrange, repeat  # noqa
from protein_learning.common.helpers import exists, default, get_min_val as max_neg_value, disable_tf32
from protein_learning.networks.common.net_utils import Residual, SplitLinear, FeedForward, PreNorm
from protein_learning.common.rigids import Rigids
from typing import Optional, Tuple
from protein_learning.networks.common.attention_utils import (
    SimilarityType,
)
from protein_learning.common.helpers import safe_norm
from protein_learning.common.global_constants import get_logger

logger = get_logger(__name__)


def coords_to_pw_angles(coords: Tensor) -> Tensor:
    """convert coordinates to (cosine of) pairwise angles"""
    assert coords.ndim == 4
    normed_coords = coords / safe_norm(coords, dim=-1, keepdim=True)
    dots = rearrange(normed_coords, "b n d c -> b () n d c") * rearrange(normed_coords, "b n d c -> b n () d c")  # noqa
    return torch.sum(dots, dim=-1, keepdim=True)


def coords_to_rel_coords(coords: Tensor, other: Optional[Tensor] = None) -> Tensor:
    """Convert coordinates to relative coordinates"""
    return rearrange(coords, "b n ... c -> b () n ... c") - rearrange(
        default(other, coords), "b n ... c -> b n () ... c"
    )  # noqa


class GeometricGraphAttention(nn.Module):
    """Geometric Graph Attention"""

    def __init__(
        self,
        scalar_dim: int,
        pair_dim: int,
        scalar_kv_dims: Tuple[int, int],
        point_kv_dims: Tuple[int, int],
        heads: int = 8,
        sim_ty: SimilarityType = SimilarityType.DOT_PROD,
        return_pair_update: bool = True,
        use_pair_bias: bool = True,
    ):
        super().__init__()
        self.heads, self.sim_ty = heads, sim_ty
        scalar_key_dim, scalar_value_dim = scalar_kv_dims
        point_key_dim, point_value_dim = point_kv_dims
        logger.info(f"scalar key and value dims {scalar_key_dim} {scalar_value_dim}")
        logger.info(f"point key and value dims {point_key_dim} {point_value_dim}")

        # set up q,k,v nets for scalar and point features
        scalar_sizes = [scalar_key_dim * heads] * 2 + [scalar_value_dim * heads]
        point_sizes = [point_key_dim * heads * 3] * 2 + [point_value_dim * heads * 3]
        self.to_qkv = SplitLinear(
            dim_in=scalar_dim, dim_out=sum(scalar_sizes + point_sizes), bias=False, sizes=scalar_sizes + point_sizes
        )

        # rearrangement of heads and points
        self.rearrange_heads = Rearrange("b ... (h d) -> b h ... d ", h=heads)
        self.rearrange_points = Rearrange("... (d c) -> ... d c", c=3)

        # pair bias
        # point and scalar attention weights
        self.use_pair_bias = use_pair_bias
        self.to_pair_bias = nn.Linear(pair_dim, heads, bias=False) if use_pair_bias else None

        # head weights for points
        self.point_weights = nn.Parameter(torch.log(torch.exp(torch.ones(1, heads, 1, 1)) - 1.0))

        n_attn_logits = 3 if use_pair_bias else 2
        self.point_attn_logits_scale = ((n_attn_logits * point_key_dim) * (9 / 2)) ** -0.5
        self.scalar_attn_logits_scale = (n_attn_logits * scalar_key_dim) ** -0.5
        self.pair_attn_logits_scale = n_attn_logits**-0.5

        # project from hidden dimension back to input dimension after performing attention
        self.dim_out = heads * (pair_dim + scalar_value_dim + (4 * point_value_dim))
        self.to_scalar_out = nn.Linear(self.dim_out, scalar_dim)
        self.to_pair_out = (
            nn.Sequential(nn.Linear(3 * heads, pair_dim), nn.GELU(), nn.Linear(pair_dim, pair_dim))
            if return_pair_update
            else None
        )

    def forward(
        self, scalar_feats: Tensor, pair_feats: Tensor, rigids: Optional[Rigids], mask: Optional[Tensor] = None
    ):
        """Compute multi-head attention"""
        b, h, device = scalar_feats.shape[0], self.heads, scalar_feats.device

        # queries, keys and values for scalar and point features.
        q_scalar, k_scalar, v_scalar, q_point, k_point, v_point = self.to_qkv(scalar_feats)

        # Derive attention for scalar features
        q_scalar, k_scalar, v_scalar = map(
            self.rearrange_heads, (q_scalar, k_scalar, v_scalar)
        )  # shapes (b i (h d) -> (b h i d)

        # Scalar attn logits
        scalar_attn_logits = einsum("b h i d, b h j d -> b h i j", q_scalar, k_scalar)

        # Derive attention bias form pairwise features
        pair_bias = self.rearrange_heads(self.to_pair_bias(pair_feats)).squeeze(-1) if self.use_pair_bias else 0

        # Derive attention for point features
        # Add trailing dimension (3) to point features.
        q_point, k_point, v_point = map(self.rearrange_points, (q_point, k_point, v_point))
        # Place points in global frame
        q_point, k_point, v_point = map(lambda x: rigids.apply(x), (q_point, k_point, v_point))
        # Add head dimension
        q_point, k_point, v_point = map(
            lambda x: rearrange(x, "b i (h d) c -> b h i d c", h=h), (q_point, k_point, v_point)
        )

        # derive attn logits for point attention
        point_weights = F.softplus(self.point_weights)
        if self.sim_ty == SimilarityType.DISTANCE:
            point_qk_diff = rearrange(q_point, "b h i d c -> b h i () (d c)") - rearrange(
                k_point, "b h j d c -> b h () j (d c)"
            )
            point_attn_logits = -(point_qk_diff**2).sum(dim=-1)

        else:
            point_attn_logits = torch.einsum("b h i d c, b h j d c -> b h i j", q_point, k_point)
        point_attn_logits = point_attn_logits * point_weights

        # mask attn. logits
        point_attn_logits, scalar_attn_logits, pair_bias = self.mask_attn_logits(
            mask, point_attn_logits, scalar_attn_logits, pair_bias
        )

        # weight and combine attn logits
        attn_logits = (
            self.scalar_attn_logits_scale * scalar_attn_logits
            + self.point_attn_logits_scale * point_attn_logits
            + self.pair_attn_logits_scale * pair_bias
        )  # noqa

        # compute attention weights
        attn = attn_logits.softmax(dim=-1)

        # compute attention features for scalar, coord, pair
        with disable_tf32(), torch.autocast(enabled=False):
            # disable TF32 for precision and aggregate values
            results_scalar = einsum("b h i j, b h j d -> b h i d", attn, v_scalar)
            results_pairwise = einsum("b h i j, b i j d -> b h i d", attn, pair_feats)
            results_points = einsum("b h i j, b h j d c -> b h i d c", attn, v_point)
            # map back to local frames
            results_points = rigids.apply_inverse(rearrange(results_points, "b h i d c -> (b h) i d c"))
            results_points = rearrange(results_points, "(b h) i d c -> b h i d c", h=h)
            results_points_norm = safe_norm(results_points, dim=-1)
            results_points = rearrange(results_points, "b h i d c -> b h i (d c)")

        # merge back heads
        results = map(
            lambda x: rearrange(x, "b h ... d -> b ... (h d)"),
            (results_scalar, results_pairwise, results_points, results_points_norm),
        )

        pair_out = self.get_pair_update(scalar_logits=scalar_attn_logits, point_logits=point_attn_logits, attn=attn)
        # concat results and project out
        return self.to_scalar_out(torch.cat([x for x in results], dim=-1)), pair_out

    def mask_attn_logits(self, mask: Optional[Tensor], *logits):
        """Masks attention logits"""
        if not exists(mask):
            return logits if len(logits) > 1 else logits[0]
        mask = ~(repeat(mask, "b i -> b h i ()", h=self.heads) * repeat(mask, "b j -> b h () j", h=self.heads))
        fill = lambda x: x.masked_fill(mask, max_neg_value(x)) if torch.is_tensor(x) else x
        out = [fill(mat) for mat in logits]
        return out if len(out) > 1 else out[0]

    def get_pair_update(self, scalar_logits, point_logits, attn) -> Optional[Tensor]:
        """Get pairwise update matrix (optional)"""
        if exists(self.to_pair_out):
            logger.info("getting pair update in geometric graph attn.")
            scalar_attn_wts, point_attn_wts = map(lambda x: torch.softmax(x, dim=-1), (scalar_logits, point_logits))
            edge_update_mat = torch.cat((attn, point_attn_wts, scalar_attn_wts), dim=1)
            return self.to_pair_out(rearrange(edge_update_mat, "b h i j -> b i j h"))
        return None


class GeometricGraphAttentionBlock(nn.Module):
    """Geometric Graph Attention block"""

    def __init__(
        self,
        dim,
        attn_kwargs,
        ff_mult: float = 2,
        ff_num_layers: int = 2,
    ):
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = GeometricGraphAttention(**attn_kwargs)
        self.ff = PreNorm(dim, FeedForward(dim, mult=ff_mult, num_layers=ff_num_layers))
        self.attn_residual, self.ff_residual = Residual(), Residual()

    def forward(self, scalar_feats, **kwargs):
        """apply attention and ff + residuals"""
        attn_feats, pair_update = self.attn(self.attn_norm(scalar_feats), **kwargs)
        scalar_feats = self.attn_residual(attn_feats, res=scalar_feats)
        return self.ff_residual(self.ff(scalar_feats), res=scalar_feats), pair_update
