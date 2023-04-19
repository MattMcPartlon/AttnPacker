"""Functions for performing Scalar Feature Attention Updates"""
from typing import Optional

import torch
import torch.nn.functional as F  # noqa
from einops import rearrange, repeat  # noqa
from einops.layers.torch import Rearrange  # noqa
from torch import nn, einsum, Tensor
from protein_learning.common.rigids import Rigids
from protein_learning.networks.common.net_utils import (
    exists,
    default,
    ReZero as Residual,
    Transition,
    SplitLinear,
    disable_tf32,
    get_min_val,
)
from protein_learning.networks.common.jit_scripts import Fuser

max_neg_value = lambda x: torch.finfo(x.dtype).min  # noqa


class NodeAttention(nn.Module):  # noqa
    """Scalar Feature masked attention with pair bias and gating"""

    def __init__(
        self,
        node_dim: int,
        dim_head: int = 32,
        heads: int = 8,
        pair_dim: Optional[int] = None,
        bias: bool = False,
        dim_out: Optional[int] = None,
        **kawrgs,  # noqa
    ):
        """dim"""
        super().__init__()
        inner_dim = dim_head * heads
        self.node_dim, self.pair_dim = node_dim, pair_dim
        self.heads, self.scale = heads, dim_head**-0.5
        self.to_qkv = nn.Linear(node_dim, inner_dim * 3, bias=bias)
        self.to_g = nn.Linear(node_dim, inner_dim)
        self.to_out_node = nn.Linear(inner_dim, default(dim_out, node_dim))
        self.node_norm = nn.LayerNorm(node_dim)
        if exists(pair_dim):
            self.to_bias = nn.Linear(pair_dim, heads, bias=False)
            self.pair_norm = nn.LayerNorm(pair_dim)
        else:
            self.to_bias, self.pair_norm = None, None
        self.bias_softmax, self.sigmoid_gate = Fuser().bias_softmax, Fuser().sigmoid_gate

    def forward(
        self,
        node_feats: Tensor,
        pair_feats: Optional[Tensor],
        mask: Optional[Tensor],
    ) -> Tensor:
        """Multi-head scalar Attention Layer

        :param node_feats: scalar features of shape (b,n,d_s)
        :param pair_feats: pair features of shape (b,n,n,d_e)
        :param mask: boolean tensor of node adjacencies
        :return:
        """
        assert exists(self.to_bias) or not exists(pair_feats)
        node_feats, h = self.node_norm(node_feats), self.heads
        pair_feats = self.pair_norm(pair_feats) if exists(pair_feats) else None
        q, k, v = self.to_qkv(node_feats).chunk(3, dim=-1)
        g = self.to_g(node_feats)
        b = rearrange(self.to_bias(pair_feats), "b ... h -> b h ...") if exists(pair_feats) else 0
        q, k, v, g = map(lambda t: rearrange(t, "b ... (h d) -> b h ... d", h=h), (q, k, v, g))
        attn_feats = self._attn(q, k, v, b, mask)
        attn_feats = rearrange(self.sigmoid_gate(attn_feats, gate=g), "b h n d -> b n (h d)", h=h)
        return self.to_out_node(attn_feats)

    def _attn(self, q, k, v, b, mask: Optional[Tensor]) -> Tensor:
        """Perform attention update"""
        sim = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        if exists(mask):
            mask = rearrange(mask, "b i j -> b () i j")
            sim = sim.masked_fill(~mask, max_neg_value(sim))
        attn = self.bias_softmax(sim, bias=b)
        return einsum("b h i j, b h j d -> b h i d", attn, v)


def dist_attn(q: Tensor, k: Tensor) -> Tensor:
    """Get (distance-based) attention logits"""
    q = rearrange(q, "b h i d c -> b h i () d c")
    k = rearrange(k, "b h j d c -> b h () j d c")
    return -torch.sum(torch.square(q - k), dim=(-1, -2))


def dot_attn(q: Tensor, k: Tensor, wts=Tensor) -> Tensor:
    """Get (angle-based) attention logits"""
    return torch.einsum("b h i d c, b h j d c -> b h i j", q, k) * wts


class IPA(nn.Module):  # noqa
    """Invariant Point Attention"""

    def __init__(
        self,
        node_dim: int,
        pair_dim: Optional[int] = None,
        heads: int = 12,
        dim_query: int = 16,
        dim_query_point: int = 4,
        dim_value: int = 16,
        dim_value_point: int = 8,
        use_dist_attn: bool = True,
        dim_out: Optional[int] = None,
        **kwargs,  # noqa
    ):
        super().__init__()
        self.heads = heads

        # point and scalar attention weights
        # head weights for points
        self.point_weights = nn.Parameter(torch.log(torch.exp(torch.ones(1, heads, 1, 1)) - 1.0))
        scale = 3 if exists(pair_dim) else 2
        self.point_attn_logits_scale = ((scale * dim_query_point) * (9 * (2**0.5))) ** -0.5
        self.scalar_attn_logits_scale = (scale * dim_query) ** -0.5
        self.pair_attn_logits_scale = 3**-0.5

        # pair related nets
        if exists(pair_dim):
            self.to_pair_bias = nn.Linear(pair_dim, heads, bias=False)
            self.pair_norm = nn.LayerNorm(pair_dim)
        else:
            self.pair_bias, self.pair_norm = None, None

        # set up q,k,v nets for scalar and point features
        sizes = (
            [heads * dim_query] * 2
            + [heads * dim_value]
            + [heads * dim_query_point * 3] * 2
            + [heads * dim_value_point * 3]
        )  # noqa
        self.to_qkv = SplitLinear(
            dim_in=node_dim,
            dim_out=sum(sizes),
            bias=False,
            sizes=sizes,
        )
        self.scalar_norm = nn.LayerNorm(node_dim)
        # rearrangement of heads and points
        self.rearrange_heads = Rearrange("b ... (h d) -> b h ... d ", h=heads)
        self.rearrange_points = Rearrange("... (d c) -> ... d c", c=3)

        # project from hidden dimension back to input dimension after performing attention
        self.dim_out = heads * (default(pair_dim, 0) + dim_value + (4 * dim_value_point))
        self.to_scalar_out = nn.Linear(self.dim_out, default(dim_out, node_dim))
        self.point_attn_fn = Fuser().dist_attn if use_dist_attn else dot_attn
        self.weight_n_add = Fuser().weight_n_add
        self.safe_norm = Fuser().safe_norm

    def forward(
        self,
        node_feats: Tensor,
        rigids: Rigids,
        pair_feats: Optional[Tensor],
        mask: Optional[Tensor] = None,
    ):
        """Invariant Point Attention"""
        assert exists(self.to_pair_bias) or not exists(pair_feats)
        b, h, device = node_feats.shape[0], self.heads, node_feats.device
        node_feats = self.scalar_norm(node_feats)

        # map pair feats to bias and rearrange
        pair_feats = self.pair_norm(pair_feats) if exists(pair_feats) else None
        pair_bias = self.rearrange_heads(self.to_pair_bias(pair_feats)).squeeze(-1) if exists(pair_feats) else 0

        # queries, keys and values for scalar and point features.
        q_scalar, k_scalar, v_scalar, q_point, k_point, v_point = self.to_qkv(node_feats)

        # Derive attention for scalar features
        q_scalar, k_scalar, v_scalar = map(
            self.rearrange_heads, (q_scalar, k_scalar, v_scalar)
        )  # shapes (b i (h d)) -> (b h i d)

        # Scalar attn logits
        scalar_attn_logits = einsum("b h i d, b h j d -> b h i j", q_scalar, k_scalar)

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
        point_attn_logits = self.point_attn_fn(q=q_point, k=k_point, wts=point_weights)

        # mask attn. logits
        point_attn_logits, scalar_attn_logits, pair_bias = self.mask_attn_logits(
            mask, point_attn_logits, scalar_attn_logits, pair_bias
        )

        w1, x1, w2, x2, w3, x3 = (
            self.scalar_attn_logits_scale,
            scalar_attn_logits,
            self.point_attn_logits_scale,
            point_attn_logits,
            self.pair_attn_logits_scale,
            pair_bias,
        )

        # weight and combine attn logits
        attn_logits = self.weight_n_add(x1, x2, x3, w1, w2, w3)

        # compute attention weights
        attn = attn_logits.softmax(dim=-1)  # noqa

        # compute attention features for scalar, coord, pair
        device_type = "cpu" if attn.device.type == "cpu" else "cuda"
        with disable_tf32(), torch.autocast(device_type=device_type, enabled=False):
            # disable TF32 for precision and aggregate values
            results_scalar = einsum("b h i j, b h j d -> b h i d", attn, v_scalar)
            results_pairwise = einsum("b h i j, b i j d -> b h i d", attn, pair_feats) if exists(pair_feats) else None
            results_points = einsum("b h i j, b h j d c -> b h i d c", attn, v_point)
            # map back to local frames
            results_points = rigids.apply_inverse(rearrange(results_points, "b h i d c -> b i (h d) c"))
            results_points = rearrange(results_points, "b i (h d) c -> b h i d c", h=h)
            results_points_norm = self.safe_norm(results_points)
            results_points = rearrange(results_points, "b h i d c -> b h i (d c)")

        # merge back heads
        results = [results_scalar, results_points, results_points_norm]
        results = map(
            lambda x: rearrange(x, "b h ... d -> b ... (h d)"),
            results + [results_pairwise] if exists(pair_feats) else results,
        )

        # concat results and project out
        return self.to_scalar_out(torch.cat([x for x in results], dim=-1))

    def mask_attn_logits(self, mask: Optional[Tensor], *logits):  # noqa
        """Masks attention logits"""
        if not exists(mask):
            return logits if len(logits) > 1 else logits[0]
        mask = rearrange(mask, "b i j -> b () i j")
        fill = lambda x: x.masked_fill(~mask, get_min_val(x)) if torch.is_tensor(x) else x
        out = [fill(mat) for mat in logits]
        return out if len(out) > 1 else out[0]


class NodeUpdateBlock(nn.Module):  # noqa
    """Node Feature Update Layer
    Input scalar_feats
    (1) Attention
    (2) Residual
    (3) Dropout
    (4) Norm + Feed Forward
    (5) Residual
    """

    def __init__(
        self,
        node_dim: int,
        pair_dim: Optional[int] = None,
        ff_mult: int = None,
        use_ipa: bool = True,
        dropout: float = 0,
        dim_out: Optional[int] = None,
        **net_kwargs,
    ):
        super().__init__()
        kls = IPA if use_ipa else NodeAttention
        self.use_ipa = use_ipa
        self.attn = kls(node_dim=node_dim, pair_dim=pair_dim, dim_out=dim_out, **net_kwargs)
        self.attn_residual = Residual()
        self.transition = Transition(dim_in=node_dim, mult=default(ff_mult, 2 if use_ipa else 4), dim_out=dim_out)
        self.transition_residual = Residual()
        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.transn_dropout = nn.Dropout(dropout) if (dropout > 0 and use_ipa) else nn.Identity()

    def forward(
        self,
        node_feats: Tensor,
        pair_feats: Optional[Tensor],
        mask: Optional[Tensor] = None,
        rigids: Optional[Rigids] = None,
    ) -> Tensor:
        """Perform Scalar Updates"""
        forward_kwargs = dict(node_feats=node_feats, pair_feats=pair_feats, mask=mask)
        if self.use_ipa:
            forward_kwargs["rigids"] = rigids
        attn_feats = self.attn(**forward_kwargs)
        attn_feats = self.attn_dropout(attn_feats)
        node_feats = self.attn_residual(attn_feats, node_feats)
        transn_feats = self.transn_dropout(self.transition(node_feats))
        return self.transition_residual(transn_feats, node_feats)
