import torch
from einops import rearrange
from torch import nn

from protein_learning.networks.common.helpers.torch_utils import fused_gelu as NONLIN
from protein_learning.networks.common.utils import default
from protein_learning.networks.common.invariant.units import FeedForward


class FiberWeightedOut(nn.Module):

    def __init__(
            self,
            fiber,
            nonlin=NONLIN,
            eps=1e-6,  # TODO: changed from 1e-12
            include_order_stats=False,
    ):
        super().__init__()
        self.fiber = fiber
        self.nonlin = nonlin
        self.eps = eps
        self.transform = nn.ModuleDict()
        self.include_order_stats = include_order_stats
        dim0, dim1 = fiber.dims[0] + fiber.dims[1], fiber.dims[1]
        if include_order_stats:
            dim0 += 2 * fiber.dims[1]
        mid = (dim0 + dim1) // 2
        self.weight_net = nn.Sequential(
            nn.Linear(dim0, mid),
            nonlin(),
            nn.LayerNorm(mid),
            nn.Linear(mid, mid),
            nonlin(),
            nn.LayerNorm(mid),
            nn.Linear(mid, dim1),
        )

    def forward(self, features):
        output = {}
        norm = torch.norm(features['1'], dim=-1)
        std, mean = torch.std_mean(norm, dim=1)
        rel_norm = (norm - mean) / (std + self.eps)
        if self.include_order_stats:
            m, s = mean.unsqueeze(1), std.unsqueeze(1)
            m, s = m.expand_as(rel_norm), s.expand_as(rel_norm)
            inp = torch.cat((features['0'].squeeze(-1), rel_norm, m, s), dim=-1)
        else:
            inp = torch.cat((features['0'].squeeze(-1), rel_norm), dim=-1)
        weights = self.weight_net(inp).unsqueeze(-1)
        output['0'] = features['0']
        output['1'] = torch.sum(features['1'] * weights, dim=-2, keepdim=True)
        return output


class WeightedOut(nn.Module):
    def __init__(
            self,
            coord_dim,
            feat_dim,
            coord_dim_out=1,
            nonlin=NONLIN,
            eps=1e-5,
            include_norms=True,
            n_hidden=1,
    ):
        super().__init__()
        self.nonlin = nonlin
        self.eps = eps
        self.transform = nn.ModuleDict()
        self.coord_dim_out = coord_dim_out
        self.coord_dim = coord_dim
        dim_in = coord_dim + feat_dim if include_norms else feat_dim
        mid, dim_out = max(128, dim_in), coord_dim * coord_dim_out
        self.weight_net = FeedForward(dim_in, mid, dim_out, n_hidden=n_hidden)

    def forward(self, coords, feats, return_feats=True):
        norm = torch.norm(coords, dim=-1)
        std, mean = torch.std_mean(norm, dim=1)
        rel_norm = (norm - mean) / (std + self.eps)
        inp = torch.cat((feats.squeeze(-1), rel_norm), dim=-1)
        weights = self.weight_net(inp)
        weight_shape = (*weights.shape[:-1], self.coord_dim, self.coord_dim_out, 1)
        weights = weights.view(weight_shape)
        coords = coords.unsqueeze(-2)
        transformed_coords = torch.sum(coords * weights, dim=-3)

        if return_feats:
            return transformed_coords, feats
        else:
            return transformed_coords


class RadialFunc(nn.Module):
    """NN parameterized radial profile function."""

    def __init__(self, num_freq, in_dim, out_dim, edge_dim=None, mid_dim=None, nonlin=NONLIN,
                 hidden_layer: bool = True, compress=False, dropout=0.0):
        super().__init__()
        self.num_freq = num_freq
        self.in_dim = in_dim
        self.edge_dim = default(edge_dim, 0)
        mid_dim = default(mid_dim, edge_dim)
        self.out_dim = out_dim
        bias = dropout > 0

        layer = lambda i, o, norm=True: nn.ModuleList([
            nn.Linear(i, o),
            nn.LayerNorm(o) if norm else nn.Identity,
            nonlin(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        ])
        if not compress:
            self.net = nn.Sequential(
                *layer(edge_dim, mid_dim),
                *layer(mid_dim, mid_dim) if hidden_layer else nn.Identity(),
                nn.Linear(mid_dim, num_freq * in_dim * out_dim, bias=bias)
            )
        else:
            mid_dim, code_dim = edge_dim // 2, edge_dim // 4
            self.net = nn.Sequential(
                *layer(edge_dim, mid_dim, norm=True),
                *layer(mid_dim, code_dim, norm=True),
                *layer(code_dim, mid_dim, norm=True),
                nn.Linear(mid_dim, num_freq * in_dim * out_dim, bias=bias)
            )

    def forward(self, x):
        y = self.net(x)
        return rearrange(y, '... (o i f) -> ... o () i () f', i=self.in_dim, o=self.out_dim)


class RadialKernel(nn.Module):
    """NN parameterized radial profile function."""

    def __init__(self, num_freq, in_dim, out_dim, edge_dim=None, mid_dim=128):
        super().__init__()
        self.num_freq = num_freq
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.bin_embedding = nn.Embedding(34, num_freq * in_dim * out_dim)
        self._dist_bins = torch.arange(34)

    def dist_bins(self, device):
        if self._dist_bins.device != device:
            self._dist_bins = self._dist_bins.to(device)
        return self._dist_bins

    def forward(self, dists):
        print('in radial kernel')
        kernels = self.bin_embedding(self.dist_bins(dists.device))
        actual_bins = torch.round(torch.clamp((dists - 2.4) / 0.4, 0, 33)).long()
        kernels = kernels[actual_bins].squeeze(-2)
        return rearrange(kernels, '... (o i f) -> ... o () i () f', i=self.in_dim, o=self.out_dim)
