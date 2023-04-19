"""Invariant Point Attention"""
import torch
import torch.nn.functional as F
from torch import nn, einsum, Tensor

from einops.layers.torch import Rearrange
from einops import rearrange, repeat  # noqa
from protein_learning.common.helpers import exists, default, get_min_val as max_neg_value, disable_tf32
from protein_learning.networks.common.net_utils import SplitLinear, Residual
from protein_learning.networks.ipa.ipa_config import IPAConfig
from protein_learning.common.rigids import Rigids
from protein_learning.networks.loss.coord_loss import FAPELoss
from typing import Optional, List


class InvariantPointAttention(nn.Module):
    """Invariant Point Attention"""

    def __init__(
        self,
        dim,
        scalar_kv_dims: List[int],
        point_kv_dims: List[int],
        heads=8,
        pairwise_repr_dim=None,
        require_pairwise_repr=True,
        eps=1e-8,
    ):
        super().__init__()
        self.eps = eps
        self.heads = heads
        self.require_pairwise_repr = require_pairwise_repr
        scalar_key_dim, scalar_value_dim = scalar_kv_dims
        point_key_dim, point_value_dim = point_kv_dims

        # num attention contributions
        num_attn_logits = 3 if require_pairwise_repr else 2

        # qkv projection for scalar attention (normal)
        self.scalar_attn_logits_scale = (num_attn_logits * scalar_key_dim) ** -0.5

        # SplitLinear with 6 splits is approximately twice as fast as using 6 separate
        # linear projections to produce q,k,v for scalars and points
        scalar_qkv_sizes = [scalar_key_dim * heads] * 2 + [scalar_value_dim * heads]
        point_qkv_sizes = [point_key_dim * heads * 3] * 2 + [point_value_dim * heads * 3]
        split_sizes = scalar_qkv_sizes + point_qkv_sizes
        self.to_qkv = SplitLinear(dim, sum(split_sizes), bias=False, sizes=split_sizes)

        # qkv projection for point attention (coordinate and orientation aware)

        point_weight_init_value = torch.log(torch.exp(torch.full((heads,), 1.0)) - 1.0)
        self.point_weights = nn.Parameter(point_weight_init_value)

        self.point_attn_logits_scale = ((num_attn_logits * point_key_dim) * (9 / 2)) ** -0.5

        # pairwise representation projection to attention bias
        pairwise_repr_dim = default(pairwise_repr_dim, dim) if require_pairwise_repr else 0
        if require_pairwise_repr:
            self.pairwise_attn_logits_scale = num_attn_logits**-0.5
            self.to_pairwise_attn_bias = nn.Sequential(
                nn.Linear(pairwise_repr_dim, heads), Rearrange("b ... h -> (b h) ...")
            )

        # combine out - scalar dim + pairwise dim + point dim * (3 for coordinates in R3 and then 1 for norm)
        self.to_out = nn.Linear(heads * (scalar_value_dim + pairwise_repr_dim + point_value_dim * (3 + 1)), dim)

    def forward(self, single_repr: Tensor, rigids: Rigids, pairwise_repr=None, mask=None):
        x, b, h, eps, require_pairwise_repr = (
            single_repr,
            single_repr.shape[0],
            self.heads,
            self.eps,
            self.require_pairwise_repr,
        )
        assert not (
            require_pairwise_repr and not exists(pairwise_repr)
        ), "pairwise representation must be given as second argument"

        # get queries, keys, values for scalar and point (coordinate-aware) attention pathways
        q_scalar, k_scalar, v_scalar, q_point, k_point, v_point = self.to_qkv(x)
        # split out heads
        q_scalar, k_scalar, v_scalar = map(
            lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q_scalar, k_scalar, v_scalar)
        )
        q_point, k_point, v_point = map(
            lambda t: rearrange(t, "b n (h d c) -> (b h) n d c", h=h, c=3), (q_point, k_point, v_point)
        )

        # rotate qkv points into global frame
        q_point, k_point, v_point = map(lambda ps: rigids.apply(ps), (q_point, k_point, v_point))

        # derive attn logits for scalar and pairwise

        attn_logits_scalar = einsum("b i d, b j d -> b i j", q_scalar, k_scalar) * self.scalar_attn_logits_scale

        if require_pairwise_repr:
            attn_logits_pairwise = self.to_pairwise_attn_bias(pairwise_repr) * self.pairwise_attn_logits_scale

        # derive attn logits for point attention

        point_qk_diff = rearrange(q_point, "b i d c -> b i () d c") - rearrange(k_point, "b j d c -> b () j d c")
        point_dist = (point_qk_diff**2).sum(dim=(-1, -2))

        point_weights = F.softplus(self.point_weights)
        point_weights = repeat(point_weights, "h -> (b h) () ()", b=b)

        attn_logits_points = -0.5 * (point_dist * point_weights * self.point_attn_logits_scale)

        # combine attn logits

        attn_logits = attn_logits_scalar + attn_logits_points

        if require_pairwise_repr:
            attn_logits = attn_logits + attn_logits_pairwise  # noqa

        # mask

        if exists(mask):
            mask = rearrange(mask, "b i -> b i ()") * rearrange(mask, "b j -> b () j")
            mask = repeat(mask, "b i j -> (b h) i j", h=h)
            attn_logits = attn_logits.masked_fill(~mask, max_neg_value(attn_logits))

        # attention

        attn = attn_logits.softmax(dim=-1)

        with disable_tf32(), torch.autocast(enabled=False):
            # disable TF32 for precision
            # aggregate values
            results_scalar = einsum("b i j, b j d -> b i d", attn, v_scalar)
            attn_with_heads = rearrange(attn, "(b h) i j -> b h i j", h=h)
            results_pairwise = None
            if require_pairwise_repr:
                results_pairwise = einsum("b h i j, b i j d -> b h i d", attn_with_heads, pairwise_repr)
                results_pairwise = rearrange(results_pairwise, "b h n d -> b n (h d)", h=h)

            # aggregate point values
            results_points = einsum("b i j, b j d c -> b i d c", attn, v_point)

            # rotate aggregated point values back into local frame
            results_points = rigids.apply_inverse(results_points)
            results_points_norm = torch.sqrt(torch.square(results_points).sum(dim=-1) + eps)

        # merge back heads
        results_scalar = rearrange(results_scalar, "(b h) n d -> b n (h d)", h=h)
        results_points = rearrange(results_points, "(b h) n d c -> b n (h d c)", h=h)
        results_points_norm = rearrange(results_points_norm, "(b h) n d -> b n (h d)", h=h)

        results = (results_scalar, results_points, results_points_norm, results_pairwise)

        # concat results and project out
        results = torch.cat([x for x in results if x is not None], dim=-1)
        return self.to_out(results)


# one transformer block based on IPA


def FeedForward(dim, mult=1.0, num_layers=2, act=nn.ReLU):
    """FeedForward (Transition) Layer"""
    layers = []
    dim_hidden = int(dim * mult)

    for ind in range(num_layers):
        is_first = ind == 0
        is_last = ind == (num_layers - 1)
        dim_in = dim if is_first else dim_hidden
        dim_out = dim if is_last else dim_hidden
        layers.append(nn.Linear(dim_in, dim_out))
        if is_last:
            continue
        layers.append(act())
    return nn.Sequential(*layers)


class IPABlock(nn.Module):
    def __init__(
        self,
        dim,
        attn_kwargs,
        ff_mult=1.0,
        ff_num_layers=2,
    ):
        super().__init__()
        self.norm_in = nn.LayerNorm(dim)
        self.attn_norm = nn.LayerNorm(dim)
        self.ff_norm = nn.LayerNorm(dim)
        self.attn = InvariantPointAttention(**attn_kwargs)
        self.ff = FeedForward(dim, mult=ff_mult, num_layers=ff_num_layers)
        self.residual = Residual()

    def forward(self, x, **kwargs):
        x = self.residual(self.attn(self.norm_in(x), **kwargs), res=x)
        return self.ff(self.ff_norm(x))


class IPATransformer(nn.Module):
    def __init__(
        self,
        config: IPAConfig,
    ):
        super(IPATransformer, self).__init__()
        self.config = config
        self.layers = nn.ModuleList([])
        self.pair_pre_norm = nn.LayerNorm(config.pair_dim)
        scalar_dim = config.dim_in(scalar=True)
        for _ in range(config.depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        IPABlock(
                            dim=scalar_dim,
                            ff_mult=config.ff_mult,
                            ff_num_layers=config.num_ff_layers,
                            attn_kwargs=config.attn_kwargs,
                        ),
                        nn.LayerNorm(scalar_dim),
                        nn.Linear(scalar_dim, 6),
                    ]
                )
            )
            if config.share_weights:
                break
        self.residuals = nn.ModuleList([Residual() for _ in range(config.depth)])
        # output points
        self.to_points = None
        if config.compute_coords:
            self.to_points = nn.Linear(scalar_dim, 3 * config.dim_out(coord=True), bias=False)
            nn.init.uniform_(self.to_points.weight, a=0, b=1e-4)

    def forward(
        self,
        scalar_feats,
        pair_feats=None,
        rigids=None,
        mask=None,
        detach_rot: bool = True,
        true_rigids: Optional[Rigids] = None,
        scale_factor: float = 10,
    ):
        config = self.config
        x, device, normed_x = scalar_feats, scalar_feats.device, scalar_feats
        b, n, *_ = x.shape
        # if no initial quaternions passed in, start from identity
        if not exists(rigids):
            rigids = Rigids.IdentityRigid(leading_shape=x.shape[:2], device=x.device)
        # go through the layers and apply invariant point attention and feedforward
        get_layer = lambda i: self.layers[i] if not config.share_weights else self.layers[0]
        loss = 0
        normed_pair_feats = self.pair_pre_norm(pair_feats) if exists(pair_feats) else None
        for idx in range(config.depth):
            block, norm, to_update = get_layer(idx)
            block_out = block(
                x, pairwise_repr=normed_pair_feats, rigids=rigids.detach_rot() if detach_rot else rigids, mask=mask
            )
            # update quaternion and translation
            x = self.residuals[idx](block_out, res=x)
            normed_x = norm(x)
            quaternion_update, translation_update = to_update(normed_x).chunk(2, dim=-1)
            quaternion_update = F.pad(quaternion_update, (1, 0), value=1.0)
            rigids = rigids.compose(Rigids(quaternion_update, translation_update))
            # auxilliary loss
            if config.share_weights:
                loss = (
                    loss
                    + self.auxilliary_loss(rigids, true_rigids, scale_factor=scale_factor, mask=mask) / config.depth
                )

        rigids = rigids.scale(scale_factor)
        out = (x, pair_feats, rigids, loss) if config.share_weights else (x, pair_feats, rigids)
        if not exists(self.to_points):
            return out
        points_local = rearrange(self.to_points(normed_x) * scale_factor, "b n (a c) -> b n a c", c=3)
        return *out, rigids.apply(points_local)

    @staticmethod
    def auxilliary_loss(
        pred_rigids: Rigids,
        true_rigids: Rigids,
        scale_factor: float = 10,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """FAPE auxillary loss on CA-coordinates"""
        pred_rigids = pred_rigids.scale(scale_factor)
        true_ca = true_rigids.translations.unsqueeze(-2)
        pred_ca = pred_rigids.translations.unsqueeze(-2)
        fape = FAPELoss()
        return fape.forward(
            pred_coords=pred_ca,
            true_coords=true_ca,
            pred_rigids=pred_rigids,
            true_rigids=true_rigids,
            coord_mask=rearrange(mask, "b n -> b n ()") if exists(mask) else None,
        )
