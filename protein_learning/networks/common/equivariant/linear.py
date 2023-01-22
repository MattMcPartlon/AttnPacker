import math
from enum import Enum
from functools import partial
from typing import Optional, Callable, Tuple

import torch
from torch import nn, einsum, Tensor

from protein_learning.common.helpers import default, safe_norm
from protein_learning.networks.common.net_utils import FeedForward


class LinearInitTy(Enum):
    UNIFORM = 'uniform'
    NON_NEG_UNIFORM = "non_neg_uniform"
    DEFAULT = 'default'
    CONSTANT = 'constant'
    IDENTITY = 'identity'
    RELU = 'relu'


class LinearInit:
    def __init__(self,
                 weight_init_ty: LinearInitTy = LinearInitTy.DEFAULT,
                 weight_init_val: Optional[float] = None,
                 bias_init_ty: LinearInitTy = LinearInitTy.DEFAULT,
                 bias_init_val: Optional[float] = None,
                 use_bias: bool = False
                 ):
        self.weight_init_ty = weight_init_ty
        self.weight_init_val = weight_init_val
        self.bias_init_ty = bias_init_ty
        self.bias_init_val = bias_init_val
        self.use_bias = use_bias

    def init_func(self, override=False) -> Callable:
        return partial(
            _linear_init,
            wt_init_ty=self.weight_init_ty,
            wt_init_val=self.weight_init_val,
            bias_init_ty=self.bias_init_ty,
            bias_init_val=self.bias_init_val,
            override=override
        )


def _linear_init(
        param,
        wt_init_ty: LinearInitTy,
        bias_init_ty: LinearInitTy,
        wt_init_val: Optional[float] = None,
        bias_init_val: Optional[float] = None,
        override: bool = False,
):
    if wt_init_ty == LinearInitTy.IDENTITY:
        nn.init.eye_(param.weight)
        if param.bias is not None:
            nn.init.constant_(param.bias, 0)
        return

    if not override and (not isinstance(param, nn.Linear) or not isinstance(param, VNLinear)):
        return
    items = zip([wt_init_ty, bias_init_ty], ["weight", "bias"], [wt_init_val, bias_init_val])

    for (ty, key, val) in items:
        if not hasattr(param, key) or getattr(param, key) is None:
            continue

        if ty == LinearInitTy.DEFAULT:
            # from pytorch documentation
            # nn.init.kaiming_uniform_(getattr(param, key), a=math.sqrt(5))
            return

        elif ty == LinearInitTy.CONSTANT:
            assert wt_init_val is not None
            nn.init.constant_(getattr(param, key), val)

        elif ty == LinearInitTy.UNIFORM:
            nn.init.xavier_uniform_(getattr(param, key), gain=1.0)

        elif ty == LinearInitTy.RELU:
            nn.init.kaiming_uniform(getattr(param, key))

        elif ty == LinearInitTy.NON_NEG_UNIFORM:
            assert wt_init_val is not None
            nn.init.uniform_(getattr(param, key), a=0, b=val)

        else:
            raise Exception(f"could not find linear init ty {ty}")


class LinearKernel(nn.Module):
    """Pairwise Linear Kernel Function"""

    def __init__(self, coord_dim, feature_dim, coord_dim_out=None, mult=2):
        super().__init__()
        coord_dim_out = default(coord_dim_out, coord_dim)
        self.transform = FeedForward(dim_in=feature_dim, mult=mult,
                                     dim_out=coord_dim * coord_dim_out)
        self.coord_dim_out = coord_dim_out

    def forward(self, rel_coords, pair_feats):
        """
        Given atom features h_i and h_j, relative coordinate diffs d_ij, and edge
        features e_ij, learns a matrix W_ij to transform the relative coordinates
        (x_i-x_j).
        """
        view = pair_feats.shape[:-1]
        # can think of as a learned per-point kernel
        kernel = self.transform(pair_feats).reshape(*view, self.coord_dim_out, rel_coords.shape[-2])
        return einsum('...ij,...jk->...ik', kernel, rel_coords)


class VNLinear(nn.Module):
    """SE(k) equivariant Vector Neuron Linear Layer

    Any linear operator acting on points in R^{k} is SE(k) equivariant, since for any R in SO(k)
    W((v+t)R) = W(vR + tR) = (Wv)R+(Wt)R by associativity

    Note:
        we intentionally omit a bias term, as the addition of this term would  interfere with equivariance

    Reference:
        https://arxiv.org/pdf/2104.12229.pdf
        "Vector Neurons: A General Framework for SO(3)-Equivariant Networks"
    """

    def __init__(
            self,
            dim_in: int,
            dim_out: int = None,
            init: LinearInit = None,
    ):
        super().__init__()
        init: LinearInit = default(init, LinearInit())
        dim_out = default(dim_out, dim_in)
        self.weight, self.bias = nn.Parameter(torch.randn(dim_in, dim_out) / math.sqrt(dim_in)), None
        self.apply(init.init_func(override=True))

    def forward(self, x):
        return einsum('b n d c, d e -> b n e c', x, self.weight)


class GVPLinear(nn.Module):
    """Linear Unit from "LEARNING FROM PROTEIN STRUCTURE WITH
    GEOMETRIC VECTOR PERCEPTRONS"
    https://openreview.net/pdf?id=1YLJDvSx6J4 (see fig. 1, Algorithm 1.).
    """

    def __init__(self,
                 dim_in: Tuple[int, int],
                 dim_out: Optional[Tuple[int, int]] = None,
                 norm_scale: float = 0.1,
                 use_scalar_bias: bool = True
                 ):
        """
        :param dim_in: scalar and coordinate input dimension
        :param dim_out: scalar and coordinate output dimension
        :param norm_scale : scale coordinate norms by this amount before concatenating
        to scalar features.
        :param scalar_bias: whether to use a bias term in linear projection for scalar
        features (not applicable for coordinate projections).
        """
        super(GVPLinear, self).__init__()
        self.dim_in = dim_in
        self.norm_scale = norm_scale
        self.dim_out = default(dim_out, dim_in)
        coord_dim_hidden = max(dim_in[1], dim_out[1])
        scalar_dim_in = self.dim_out[1] + self.dim_in[0]
        self.coord_message = VNLinear(self.dim_in[1], coord_dim_hidden)
        self.coord_proj = VNLinear(coord_dim_hidden, self.dim_out[1])
        self.scalar_proj = nn.Linear(scalar_dim_in, self.dim_out[0], bias=use_scalar_bias)

    def forward(self, scalar_feats: Tensor, coord_feats: Tensor) -> Tuple[Tensor, Tensor]:
        """Apply GVP Linear Layer.

        This layer is equivalent to the GVP feedforward block minus the
        non-linearities applied on the output. FOr full block, see ../feed_forward.py

        :param scalar_feats: scalar features of shape (b,n,d_scalar,1)
        :param coord_feats: coordinate features of shape (b,n,d_coord,3)
        :return: updated scalar and coordinte feature tuple.
        """
        assert scalar_feats.ndim == 3 and coord_feats.ndim == 4
        coord_message = self.coord_message(coord_feats)
        coord_out = self.coord_proj(coord_message)
        message_norms = safe_norm(coord_message, dim=-1, keepdim=True) * self.norm_scale
        scalar_in = torch.cat((scalar_feats, message_norms), dim=-1)
        return self.scalar_proj(scalar_in), coord_out
