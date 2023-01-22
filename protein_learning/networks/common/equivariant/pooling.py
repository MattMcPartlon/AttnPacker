import torch
from einops import rearrange
from torch import nn, einsum

from protein_learning.networks.common.constants import EPS
from protein_learning.networks.common.utils import default
from protein_learning.networks.common.invariant.units import FeedForward


class MeanPool(nn.Module):
    """Mean pooling
    """

    def __init__(self, dim=-3):
        super().__init__()
        self.dim = dim

    def forward(self, x, dim=None):
        """Neighborhood-wise Mean Pooling.

        :return: average of each feature type over specified dimensions
        """
        return torch.mean(x, dim=default(dim, self.dim))


class MaxPool(nn.Module):
    """VN-Max Pooling

    For global pooling, we are given a set of vector-lists V in R^(n x dim x deg).
    We learn an element-wise signal of data dependent directions K in R^(n x dim x deg).

    The max is given by taking the element that best aligns with K and selecting it
    as a global feature - for each channel 1..dim

    Reference:
        For more details, see section 3.3 https://arxiv.org/pdf/2104.12229.pdf
        "Vector Neurons: A General Framework for SO(3)-Equivariant Networks"
    """

    def __init__(self, dim_in, dim_out=None, init_eps=EPS, ):
        super().__init__()
        dim_out = default(dim_out, dim_in)
        self.W = nn.Parameter(torch.randn(dim_out, dim_in) * init_eps)
        self.dim_out = dim_out

    def forward(self, x):
        """Neighborhood-wise Max Pooling.

        If neighbor indices are specified, then the max is taken for each
        for each input node wrt the given neighbors. Otherwise, a global
        max operation is performed.

        :param x: feature dict where each feature has shape (b,n,N,dim_k,k),
        where k is the feature dimension, and N (optional) is the number of neighbors
        per-point.

        :return: A tensor of shape (b,n,dim_k,k) if neighbor dimension (N) is present,
        otherwise a tensor of shape (b,dim_k,k).

        """

        n_dims = len(x.shape)
        # if neighbors not given, treat input as
        feats = rearrange(x, 'b n c d -> b () n c d') if n_dims == 4 else x
        b, n, N, c, d = feats.shape
        # c = self.dim_out
        out_shape = (b, c, d) if n == 1 else (b, n, c, d)
        # determine directions for max pooling
        directions = einsum('ij, ...jk->...ik', self.W, feats)
        sim = torch.sum(directions * feats, dim=-1)
        nbr_max_idxs = torch.argmax(sim, dim=-2)
        batch_indices = torch.arange(b).reshape(b, 1, 1).expand(b, n, c)
        channel_idxs = torch.arange(c).expand(b, n, c)
        coord_idxs = torch.arange(n).reshape(1, n, 1).expand(b, n, c)
        return feats[batch_indices, coord_idxs, nbr_max_idxs, channel_idxs].reshape(out_shape)


class WeightedPool(nn.Module):

    def __init__(self, dim_in, feat_dim, mult=2, use_norm=True, norm=None, nonlin=None):
        """Performs a weighted pooling.
        """
        super().__init__()
        self.transform = FeedForward(feat_dim, feat_dim * mult, dim_in, use_norm=use_norm,
                                     norm=norm, nonlin=nonlin)
        self.pool = MeanPool()

    def forward(self, x, invariant_feats):
        weights = self.transform(invariant_feats).unsqueeze(-1)
        return self.pool(x * weights)
