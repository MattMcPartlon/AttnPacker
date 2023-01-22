import torch
from torch import nn, Tensor

from protein_learning.common.helpers import safe_normalize
from protein_learning.networks.common.helpers.torch_utils import rand_uniform


class CoordNorm(nn.Module):
    def __init__(
            self,
            dim: int,
            nonlin: nn.Module = nn.ReLU,
            eps: float = 1e-8,
            use_layernorm: bool = False,
    ):
        super().__init__()
        self.nonlin = nonlin
        self.eps = eps
        self.use_layernorm = use_layernorm
        self.scale = nn.Parameter(torch.ones(1, 1, dim, dtype=torch.float32))
        self.bias = nn.Parameter(rand_uniform(shape=(1, 1, dim), min_val=-1e-3, max_val=1e-3))

    def forward(self, features):
        # Compute the norms and normalized features
        norm = features.norm(dim=-1).clamp(min=self.eps)
        phase = features / norm.unsqueeze(-1)
        if self.use_layernorm:
            std, mean = torch.std_mean(norm, dim=-1, keepdim=True)
            norm = (norm - mean) / (std + self.eps)
        transformed = self.nonlin(norm * self.scale + self.bias).unsqueeze(-1)

        # Nonlinearity on norm
        return (transformed * phase).view(*features.shape)


class CoordUnitNorm(nn.Module):
    """Make coordinates have unit norm"""

    def __init__(self):
        super(CoordUnitNorm, self).__init__()
        pass

    def forward(self, coords: Tensor)->Tensor:
        return safe_normalize(coords, eps=1e-5, dim=(-1, -2))


class SeqNormSE3(nn.Module):
    def __init__(self, dim:int, nonlin=nn.ReLU, gamma_init=1e-5, beta_init=1, eps=1e-4, seq_dim=1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones((dim, 1)) * gamma_init)
        self.beta = nn.Parameter(torch.ones((dim, 1)) * beta_init)
        self.nonlin = nonlin
        self.eps = eps
        self.seq_dim = seq_dim

    def forward(self, feats):
        feat_norms = torch.norm(feats, dim=-1, keepdim=True)
        normed_feats = feats / (feat_norms + self.eps)
        std = torch.std(feat_norms, dim=self.seq_dim, keepdim=True)  # std value for coord norm
        feat_norms = feat_norms / (std + self.eps)
        transformed_norms = (feat_norms * self.beta + self.gamma)
        return self.nonlin(transformed_norms) * normed_feats
