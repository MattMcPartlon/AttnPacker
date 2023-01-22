import torch
from torch import Tensor, nn
import torch.nn.functional as F  # noqa
from protein_learning.networks.common.constants import FUSE
from einops import rearrange, repeat  # noqa


@torch.jit.script
def fused_gelu(x: Tensor) -> Tensor:
    return x * 0.5 * (1.0 + torch.erf(x / 1.41421))


class FusedGELUModule(nn.Module):

    def __init__(self):
        super(FusedGELUModule, self).__init__()

    def forward(self, x: Tensor) -> Tensor:
        return fused_gelu(x)


@torch.jit.script
def fused_gate_n_mul_sigmoid(g: Tensor, x: Tensor) -> Tensor:
    return torch.sigmoid(g) * x


@torch.jit.script
def fused_gate_n_mul_gelu(g: Tensor, x: Tensor) -> Tensor:
    return (g * 0.5 * (1.0 + torch.erf(g / 1.41421))) * x


@torch.jit.script
def fused_sigmoid_gate(logits: Tensor, gate: Tensor):
    return torch.sigmoid(gate) * logits


def sigmoid_gate(logits: Tensor, gate: Tensor):
    return torch.sigmoid(gate) * logits


@torch.jit.script
def fused_sigmoid(x: Tensor) -> Tensor:
    return torch.sigmoid(x)


@torch.jit.script
def fused_scale_mul_bias(x: Tensor, scale, bias: Tensor) -> Tensor:
    return x * scale + bias


def scale_mul_bias(x: Tensor, scale, bias: Tensor) -> Tensor:
    return x * scale + bias


@torch.jit.script
def fused_geglu(x: Tensor, g: Tensor) -> Tensor:
    return (g * 0.5 * (1.0 + torch.erf(g / 1.41421))) * x


def geglu(x: Tensor, g: Tensor) -> Tensor:
    return (g * 0.5 * (1.0 + torch.erf(g / 1.41421))) * x


@torch.jit.script
def fused_bias_softmax(x: Tensor, bias: Tensor):
    return F.softmax(x + bias, dim=-1)


def bias_softmax(x: Tensor, bias: Tensor):
    return F.softmax(x + bias, dim=-1)


@torch.jit.script
def fused_dist_attn(q, k, wts):
    q, k = q.unsqueeze(-3), k.unsqueeze(-4)
    logits = -torch.sum(torch.square(q - k), dim=(-1, -2))
    return logits * wts


def dist_attn(q, k, wts):
    q = rearrange(q, "b h i d c -> b h i () d c")
    k = rearrange(k, "b h j d c -> b h () j d c")
    logits = -torch.sum(torch.square(q - k), dim=(-1, -2))
    return logits * wts


@torch.jit.script
def weight_n_add(x1, x2, x3, w1: float, w2: float, w3: float) -> Tensor:
    return x1 * w1 + x2 * w2 + x3 * w3


@torch.jit.script
def safe_norm(x: Tensor):
    """Safe norm of a vector"""
    return torch.sqrt(torch.sum(torch.square(x), dim=-1, keepdim=False) + 1e-12)


class Fuser:

    def __init__(self):
        self.sigmoid = fused_sigmoid if FUSE else torch.sigmoid
        self.sigmoid_gate = fused_sigmoid_gate if FUSE else sigmoid_gate
        self.scale_mul_bias = fused_scale_mul_bias if FUSE else scale_mul_bias
        self.geglu = fused_geglu if FUSE else geglu
        self.softplus = F.softplus
        self.bias_softmax = fused_bias_softmax if FUSE else bias_softmax
        self.dist_attn = fused_dist_attn if FUSE else dist_attn
        self.weight_n_add = weight_n_add
        self.safe_norm = safe_norm

    def GELU(self) -> nn.Module:
        return FusedGELUModule if FUSE else nn.GELU  # noqa
