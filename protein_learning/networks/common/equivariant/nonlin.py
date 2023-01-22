import torch
from torch import einsum
from torch import nn

from protein_learning.networks.common.equivariant.norm import CoordNorm
from protein_learning.networks.common.constants import EPS
from protein_learning.networks.common.utils import default


class PhaseNorm(nn.Module):
    """Norm-based SE(k)-equivariant nonlinearity.

    > for feature type in features:
    >    norm, phase <- feature
    >    output = fnc(norm) * phase

    where fnc: {R+}^m -> R^m is a learnable map from m norms to m scalars.

    For more details see: https://arxiv.org/abs/1802.08219
    "Tensor field networks: Rotation- and translation-equivariant neural networks for 3D point clouds"

    """

    def __init__(
            self,
            dim_in,
            nonlin=nn.GELU(),
            eps=1e-5,
    ):
        """
        :param dim_in: input dimension of point features
        :param nonlin: Nonlinearity to apply on the norms
        :param eps: value to clamp norm at (for stability)
        """
        super().__init__()
        self.eps = eps
        self.transform = nn.Sequential(nn.LayerNorm([dim_in, 1]), nonlin)

    def forward(self, x):
        """
        :param x:

        :return:
        """
        # Compute the norms and normalized features
        norm = x.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        phase = x / norm
        transformed = self.transform(norm)
        return (transformed * phase).view(*x.shape)


class VNReLUABC(nn.Module):
    """SE(k) Equivariant ReLU

    Given an input vector list V = [V_1...V_n], V_i in R^(d x k), maps each
    V_i to a learned feature q in R^(1 x k) and a learned direction k in R^(1 x k) for
    each channel 1..d.

    Essentially learns 2d linear combinations of the input coordinate features for each
    input coordinate (d for the learned directions, and d for the learned features)

    For more details, see section 3.2 https://arxiv.org/pdf/2104.12229.pdf
    "Vector Neurons: A General Framework for SO(3)-Equivariant Networks"
    """

    def __init__(self, dim_in, share_nonlinearity, map_feats,
                 negative_slope, eps, dim_out=None, use_norm=True, norm=None,
                 n_dims=4, init_eps=EPS):  # noqa
        super().__init__()
        self.dim_in = dim_in
        dim_out = default(dim_out, self.dim_in)
        self.eps = eps
        self.negative_slope = negative_slope
        self.U, self.W = self.get_U_and_W(dim_in, dim_out, map_feats, share_nonlinearity, scale=init_eps)
        norm = default(norm, CoordNorm) if use_norm else nn.Identity
        self.norm = norm(dim_out, nonlin=nn.Identity(), beta_init=1e-1)  # noqa
        self.map_feats = map_feats

    @staticmethod
    def get_U_and_W(dim_in, dim_out, map_feats, share_nonlinearity, scale=1e-2):
        W = nn.Parameter(torch.randn((dim_out, dim_in)) * scale) if map_feats else None
        dim_out_dir = 1 if share_nonlinearity else dim_out
        U = nn.Parameter(torch.randn((dim_out_dir, dim_in)) * scale)
        return U, W

    def forward(self, x):
        slope = self.negative_slope
        directions, proj = self.U, self.W
        # get inner product of featuers and learned directions
        x_proj = x if not self.map_feats else einsum('i j , ... j k -> ... i k', proj, x)
        Q = self.norm(x_proj)
        K = einsum('i j , ... j k -> ... i k', directions, x)
        # normalize and project entries with inner product <0
        QK = torch.sum(Q * K, dim=-1, keepdim=True)
        K_normed = K / (torch.norm(K, dim=-1, keepdim=True) + self.eps)
        proj_mask = (QK >= 0).float()
        resid = 0 if slope == 0 else (slope * Q)
        return resid + (1 - slope) * (Q * proj_mask + (1 - proj_mask) * (Q - (QK * K_normed)))


class VNLinearReLU(VNReLUABC):
    """SE(k) Equivariant ReLU

    Given an input vector list V = [V_1...V_n], V_i in R^(d x k), maps each
    V_i to a learned feature q in R^(1 x k) and a learned direction k in R^(1 x k) for
    each channel 1..d.

    Essentially learns 2d linear combinations of the input coordinate features for each
    input coordinate (d for the learned directions, and d for the learned features)

    For more details, see section 3.2 https://arxiv.org/pdf/2104.12229.pdf
    "Vector Neurons: A General Framework for SO(3)-Equivariant Networks"
    """

    def __init__(self, dim_in: int, dim_out: int = None, eps=1e-5,
                 share_nonlinearity=False, n_dims=4, init_eps=EPS, use_norm=True,
                 norm=None):
        super().__init__(dim_in, share_nonlinearity=share_nonlinearity, map_feats=True,
                         negative_slope=0, eps=eps, dim_out=dim_out, n_dims=n_dims,
                         init_eps=init_eps, use_norm=use_norm, norm=norm)


class VNReLU(VNReLUABC):
    """SE(k) Equivariant ReLU

    Given an input vector list V = [V_1...V_n], V_i in R^(d x k), maps each list to a
    sequence od d learned directions, and projects the ith input coordinate in the
    ith corresponding learned direction.

    Equivalent to VNLinearReLU with W set to the identity matrix

    For more details, see section 3.2 https://arxiv.org/pdf/2104.12229.pdf
    "Vector Neurons: A General Framework for SO(3)-Equivariant Networks"
    """

    def __init__(self, dim_in: int, eps=1e-5, share_nonlinearity=False, n_dims=4,
                 init_eps=EPS, use_norm=True, norm=None):
        super().__init__(dim_in, share_nonlinearity=share_nonlinearity,
                         map_feats=False, negative_slope=0, eps=eps, n_dims=n_dims,
                         init_eps=init_eps, use_norm=use_norm, norm=norm)


class VNLeakyReLU(VNReLUABC):
    """Equivariant Leaky Relu.

    The "Leaky" version of VNReLU
    """

    def __init__(self, dim_in: int, negative_slope=0.2, share_nonlinearity=False, eps=1e-5,
                 n_dims=4, init_eps=EPS, use_norm=True, norm=None):
        super().__init__(dim_in, share_nonlinearity=share_nonlinearity,
                         map_feats=False, negative_slope=negative_slope, eps=eps,
                         n_dims=n_dims, init_eps=init_eps, use_norm=use_norm, norm=norm)


class VNLinearLeakyReLU(VNReLUABC):
    """Equivariant Leaky Relu.

    The "Leaky" version of VNLinearReLU
    """

    def __init__(self, dim_in: int, fiber_out=None, negative_slope=0.2, share_nonlinearity=False, eps=1e-5,
                 n_dims=4, init_eps=EPS, use_norm=True, norm=None):
        super().__init__(dim_in, share_nonlinearity=share_nonlinearity,
                         map_feats=True, negative_slope=negative_slope, eps=eps, dim_out=fiber_out,
                         n_dims=n_dims, init_eps=init_eps, use_norm=use_norm, norm=norm)
