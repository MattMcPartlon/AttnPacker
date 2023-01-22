import torch
from einops import rearrange
from torch import nn

from protein_learning.networks.common.helpers.torch_utils import batched_index_select
from protein_learning.networks.common.utils import exists, default


class PairwiseKernel(nn.Module):
    def __init__(self, c_in, c_out, edge_dim, mid_dim=128):
        super().__init__()
        self.kernel = nn.Sequential(nn.Linear(edge_dim, mid_dim),
                                    nn.LayerNorm(mid_dim),
                                    nn.GELU(),
                                    nn.Linear(mid_dim, mid_dim),
                                    nn.LayerNorm(mid_dim),
                                    nn.GELU(),
                                    nn.Linear(mid_dim, c_in * c_out))

    def forward(self, edges):
        return self.kernel(edges)


class Zero2Zero(nn.Module):
    """
    For each node i, and each neighbor j, we use e_ij to produce
    c_out many lists of c_in scalars. Each list of scalars is dotted with
    feature j to create a new output feature.
    """

    def __init__(self, c_in, c_out, edge_dim, mid_dim=128):
        super().__init__()
        self.c_in, self.c_out = c_in, c_out
        self.kernel = PairwiseKernel(c_in, c_out, edge_dim, mid_dim=mid_dim)

    def forward(self, x, edge_info):
        # edge info should have shape b x n x N x d_e
        # x should have shape b x n x N x i x 1
        kernels = rearrange(self.kernel(edge_info), '... (o i) -> ... o i', o=self.c_out, i=self.c_in)
        # kernel shape should be b x n x N x o x i
        return torch.sum(rearrange(x, '... i f-> ... f i') * kernels, dim=-1, keepdim=True)


class One2One(nn.Module):
    """
    For each node i, and each neighbor j, we use e_ij to produce
    a c_out x c_in matrix and use this matrix to map the c_in many
    relative coordinates to c_out many transformed relative coordinates.
    """

    def __init__(self, c_in, c_out, edge_dim, mid_dim=128):
        super().__init__()
        self.c_in, self.c_out = c_in, c_out
        self.kernel = PairwiseKernel(c_in, c_out, edge_dim, mid_dim=mid_dim)

    def forward(self, x, edge_info):
        # edge info should have shape b x n x N x d_e
        # x should have shape b x n x N x i x 1
        kernels = rearrange(self.kernel(edge_info), '... (o i) -> ... o i', o=self.c_out, i=self.c_in)
        # kernel shape should be b x n x N x o x i
        return torch.matmul(kernels, x)


class One2Zero(nn.Module):
    """
    Using a dot product with learned transformation
    """

    def __init__(self, c_in, c_out, edge_dim, mid_dim=128, init_eps=1e-3):
        super().__init__()
        self.c_in, self.c_out = c_in, c_out
        self.conv = One2One(c_in, c_out, edge_dim, mid_dim=mid_dim)
        self.trans = nn.Parameter(torch.randn((c_out, c_in)) * init_eps)

    def forward(self, x, edge_info):
        edge_transformed = self.conv(x, edge_info)
        lin_transformed = torch.matmul(self.trans, x)
        return torch.sum(edge_transformed * lin_transformed, dim=-1, keepdim=True)


class Zero2One(nn.Module):
    # use the EGNN update here

    def __init__(self, feat_dim_in, coord_dim_in, dim_out, edge_dim,
                 mid_dim=128, use_self_feats=True):
        super().__init__()
        self.dim_out = dim_out
        self.coord_dim_in = coord_dim_in
        self.use_self_feats = use_self_feats
        f = 2 if use_self_feats else 1
        dim_message = f * feat_dim_in + edge_dim
        self.kernel = PairwiseKernel(coord_dim_in, dim_out, dim_message,
                                     mid_dim=max(mid_dim, dim_message))

    def forward(self, feats, coords, edges, self_feats=None):
        assert feats.shape[-1] == 1
        if self.use_self_feats:
            assert self_feats is not None
            edge_info = torch.cat((feats.squeeze(-1), self_feats.squeeze(-1), edges), dim=-1)
        else:
            edge_info = torch.cat((feats.squeeze(-1), edges), dim=-1)
        weights = self.kernel(edge_info)
        #
        weights = rearrange(weights, '... (o m) -> ... o m ()', o=self.dim_out, m=self.coord_dim_in)
        crds = rearrange(coords, '... m d -> ... () m d')
        return torch.sum(weights * crds, dim=-2)


class TFNProx(nn.Module):

    def __init__(self, fiber_in, fiber_out, edge_dim, use_self_feats=True, mid_dim=128,
                 residual=False, reduce=False, **kwargs):

        super().__init__()
        feat_dim_in, feat_dim_out = fiber_in[0], fiber_out[0]
        coord_dim_in, coord_dim_out = fiber_in[1] + 1, fiber_out[1]
        edge_add = coord_dim_in + 1
        feat_dim_out = default(feat_dim_out, feat_dim_in)
        coord_dim_out = default(coord_dim_out, coord_dim_in)
        self.zero2zero = Zero2Zero(feat_dim_in, feat_dim_out, edge_dim + edge_add, mid_dim=mid_dim)
        self.zero2one = Zero2One(feat_dim_in, coord_dim_in, coord_dim_out,
                                 edge_dim + edge_add, mid_dim=mid_dim, use_self_feats=use_self_feats)
        self.one2zero = One2Zero(coord_dim_in, feat_dim_out, edge_dim + edge_add, mid_dim=mid_dim)
        self.one2one = One2One(coord_dim_in, coord_dim_out, edge_dim + edge_add, mid_dim=mid_dim)
        self.use_self_feats = use_self_feats
        self.residual = residual
        self.reduce_out = reduce

    def forward(self, features, edge_info, rel_dist, basis):
        reduce_out = self.reduce_out
        feats, coords = features['0'], features['1']
        (nbrs, mask, edges) = edge_info
        if exists(rel_dist):
            edges = torch.cat((edges, rel_dist.unsqueeze(-1)), dim=-1)
        if feats.shape[-1] != 1:
            feats = feats.unsqueeze(-1)
        pairwise_feats = batched_index_select(feats.squeeze(-1), nbrs, dim=1)
        if edges.shape[:-1] != pairwise_feats.shape[:-1]:
            edges = batched_index_select(edges, nbrs, dim=2)
        self_feats = None
        if self.use_self_feats:
            self_feats = rearrange(feats.squeeze(-1), 'b n m -> b n () m').expand_as(pairwise_feats)
            self_feats = self_feats.unsqueeze(-1)
        pairwise_feats = pairwise_feats.unsqueeze(-1)
        rel_coords = rearrange(coords, 'b n m d -> b n () m d') - \
                     rearrange(coords, 'b n m d -> b () n m d')

        rel_coords = batched_index_select(rel_coords, nbrs, dim=2)
        edges = torch.cat((edges, rel_coords.norm(dim=-1)), -1)

        zero2zero = self.zero2zero(pairwise_feats, edges)
        one2zero = self.one2zero(rel_coords, edges)
        one2one = self.one2one(rel_coords, edges)
        zero2one = self.zero2one(pairwise_feats, rel_coords, edges, self_feats=self_feats)

        updated_feats, updated_coords = zero2zero + one2zero, zero2one + one2one
        if reduce_out:
            updated_feats = torch.mean(updated_feats, dim=-3, keepdim=True)
            updated_coords = torch.mean(updated_coords, dim=-3, keepdim=True)
            if self.residual:
                updated_feats, updated_coords = updated_feats + feats, updated_coords + coords
        else:
            assert not self.residual
        return {'0': updated_feats, '1': updated_coords}
