"""Helper functions for graph transformer"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Tuple, Union, Optional, List, Any

import torch
import torch.utils.checkpoint as checkpoint
from einops import rearrange  # noqa
from torch import nn, Tensor, tensor, tensor_split  # noqa
from protein_learning.networks.common.jit_scripts import Fuser

"""Helper Functions"""
rot_mul_vec = lambda x, y: torch.einsum("... i j, ... j -> ... i", x, y)
get_min_val = lambda x: torch.finfo(x.dtype).min  # noqa
get_max_val = lambda x: torch.finfo(x.dtype).max  # noqa

REZERO_INIT = 0.025


def exists(val: Any) -> bool:
    """returns whether val is not none"""
    return val is not None


def default(x: Any, y: Any) -> Any:
    """returns x if it exists, otherwise y"""
    return x if exists(x) else y


def safe_norm(x: Tensor, dim: int, keepdim: bool = False, eps: float = 1e-12) -> Tensor:
    """Safe norm of a vector"""
    return torch.sqrt(torch.sum(torch.square(x), dim=dim, keepdim=keepdim) + eps)


def safe_normalize(x: Tensor, eps: float = 1e-12) -> Tensor:
    """Safe normalization of a vector"""
    return x / safe_norm(x, dim=-1, keepdim=True, eps=eps)


@contextmanager
def disable_tf32():
    """temporarily disable 32-bit float ops"""
    if torch.cuda.is_available():
        orig_value = torch.backends.cuda.matmul.allow_tf32  # noqa
        torch.backends.cuda.matmul.allow_tf32 = False  # noqa
        yield
        torch.backends.cuda.matmul.allow_tf32 = orig_value  # noqa
    else:
        yield


def coords_to_rel_coords(coords: Tensor, other: Optional[Tensor] = None) -> Tensor:
    """Convert coordinates to relative coordinates"""
    return rearrange(coords, "b n ... c -> b () n ... c") - rearrange(
        default(other, coords), "b n ... c -> b n () ... c"
    )  # noqa


def masked_mean(x: Tensor, mask: Optional[Tensor], dim: Union[int, Tuple[int, ...]] = -1, keepdim=False) -> Tensor:
    """Performs a masked mean over a given dimension
    :param x: tensor to apply mean to
    :param mask: mask to use for calculating mean
    :param dim: dimension to extract mean over
    :param keepdim: keep the dimension where mean is taken
    :return: masked mean of tensor according to mask along dimension dim.
    """
    if not exists(mask):
        return torch.mean(x, dim=dim, keepdim=keepdim)
    assert x.ndim == mask.ndim
    assert x.shape[dim] == mask.shape[dim] if isinstance(dim, int) else True
    x = torch.masked_fill(x, ~mask, 0)
    total_el = mask.sum(dim=dim, keepdim=keepdim)
    mean = x.sum(dim=dim, keepdim=keepdim) / total_el.clamp(min=1.0)
    return mean.masked_fill(total_el == 0, 0.0)


class PreNorm(nn.Module):  # noqa
    """Apply norm before applying fn"""

    def __init__(self, dim, fn, norm=None):
        super().__init__()
        self.fn = fn
        self.norm = default(norm, nn.LayerNorm(dim))

    def forward(self, x, *args, **kwargs):
        return self.fn(self.norm(x), *args, **kwargs)


class _Identity(nn.Module):  # noqa
    def __init__(self, *args, **kwargs):  # noqa
        super(_Identity, self).__init__()

    def forward(self, *args):  # noqa
        return args


class LearnedOuter(nn.Module):  # noqa
    """Learned Outer Product"""

    def __init__(
        self, dim_in: int, dim_out: int, dim_hidden: int = 16, pre_norm: bool = True, do_checkpoint: bool = True
    ):
        super(LearnedOuter, self).__init__()
        self.pre_norm = nn.LayerNorm(dim_in) if pre_norm else nn.Identity()
        self.project_in = nn.Linear(dim_in, 2 * dim_hidden)
        self.project_out = nn.Linear(dim_hidden**2, dim_out)
        self.do_checkpoint = do_checkpoint

    def forward(self, feats: Tensor) -> Tensor:
        """Apply learned outer product"""
        if self.do_checkpoint and self.training:
            return checkpoint.checkpoint(self._forward, feats)
        return self._forward(feats)

    def _forward(self, feats: Tensor) -> Tensor:
        a, b = self.project_in(self.pre_norm(feats)).chunk(2, dim=-1)
        mul = torch.einsum("b i x, b j y -> b i j x y", a, b)
        return self.project_out(rearrange(mul, "b i j x y -> b i j (x y)"))


def FeedForwardTransition(dim: int, ff_mult: int = 4, pre_norm: bool = True):
    """FeedForward with two layers"""
    return nn.Sequential(
        nn.LayerNorm(dim) if pre_norm else nn.Identity(),
        nn.Linear(dim, dim * ff_mult),
        Fuser().GELU(),
        nn.Linear(dim * ff_mult, dim),
    )


class IdentityResidual(nn.Module):  # noqa
    """Identity Residual"""

    def __init__(self):
        super().__init__()

    def forward(self, out, res):  # noqa
        return res


class ReZero(nn.Module):  # noqa
    """ReZero Residual"""

    def __init__(self):
        super(ReZero, self).__init__()
        self.alpha = nn.Parameter(torch.zeros(1).float() + REZERO_INIT)
        self.res_fn = Fuser().scale_mul_bias

    def forward(self, out, res):
        if out.shape != res.shape:
            print(f"WARNING: rezero sizes don't match! out : {out.shape}, res : {res.shape}")
            return out
        return self.res_fn(x=out, scale=self.alpha, bias=res)


class Residual(nn.Module):  # noqa
    """Residual"""

    def __init__(self, use_rezero: bool = True, use_identity: bool = False):
        super(Residual, self).__init__()
        self.residual = ReZero() if use_rezero else None
        self.residual = IdentityResidual() if use_identity else self.residual
        assert exists(self.residual)

    def forward(self, out, res):
        """Perform residual"""
        return self.residual(out, res)


def identity(x, *args, **kwargs):
    return x


class FeedForward(nn.Module):  # noqa
    """FeedForward network"""

    def __init__(
        self,
        dim_in,
        mult: Optional[Union[int, float]] = 2,
        dim_hidden: Optional[int] = None,
        dim_out: Optional[int] = None,
        nonlin: nn.Module = None,
        num_layers: int = 2,
        dropout: float = 0,
    ):
        super(FeedForward, self).__init__()
        assert num_layers > 0
        layers = []
        nonlin = default(nonlin, nn.GELU())
        dim_out = default(dim_out, dim_in)
        dim_hidden = default(dim_hidden, int(dim_in * mult))
        for idx in range(num_layers):
            d_in = dim_in if (idx == 0) else dim_hidden
            d_out = dim_out if (idx == num_layers - 1) else dim_hidden
            if dropout > 0 and idx < num_layers - 1:
                layers += [nn.Linear(d_in, d_out), nn.Dropout(dropout), nonlin]
            else:
                layers += [nn.Linear(d_in, d_out), nonlin]
        # remove final activation
        self.net = nn.Sequential(*layers[:-1])

    def forward(self, x: Tensor) -> Tensor:
        """Apply FF"""
        return self.net(x)


class SplitLinear(nn.Module):  # noqa
    """Linear projection from one input to several outputs of varying sizes"""

    def __init__(
        self, dim_in: int, dim_out: int, bias: bool = True, chunks: int = 1, sizes: Optional[List[int]] = None
    ):
        super(SplitLinear, self).__init__()
        self.dim_in, self.dim_out = dim_in, dim_out
        self.linear = nn.Linear(dim_in, dim_out, bias=bias)
        if not exists(sizes) and chunks == 1:
            self.to_out = lambda x: x
        else:
            sizes = tensor(default(sizes, [dim_out // chunks] * chunks))
            assert sum(sizes) == dim_out
            self.sizes = torch.cumsum(sizes, dim=0).long()[:-1]
            self.to_out = lambda x: tensor_split(x, self.sizes, dim=-1)

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, ...]]:
        """Compute linear projections"""
        return self.to_out(self.linear(x))


class GLU(nn.Module):  # noqa
    """Gated Linear Unit"""

    def __init__(self, dim_in: int, dim_out: int, nonlin: nn.Module, bias: bool = True):
        super(GLU, self).__init__()
        self.dim_out = dim_out
        self.nonlin = nonlin  # TODO: no longer used
        self.proj = nn.Linear(dim_in, 2 * dim_out, bias=bias)
        self.gate_fn = Fuser().geglu

    def forward(self, x: Tensor) -> Tensor:
        """Apply function to x"""
        feats = self.proj(x)
        return self.gate_fn(x=feats[..., : self.dim_out], g=feats[..., self.dim_out :])


class Transition(nn.Module):  # noqa
    """FeedForward Transition"""

    def __init__(
        self,
        dim_in: int,
        dim_out: Optional[int] = None,
        mult: int = 2,
        pre_norm: bool = True,
        nonlin: Optional[nn.Module] = None,
        use_glu: bool = True,
        bias: bool = False,
        residual: Optional[nn.Module] = None,
        mid_norm: bool = True,
    ):
        super().__init__()
        nonlin = default(nonlin, Fuser().GELU())
        dim_out = default(dim_out, dim_in)
        glu = lambda: GLU(dim_in=dim_in, dim_out=mult * dim_in, nonlin=nonlin, bias=bias)
        self.net = nn.Sequential(
            nn.LayerNorm(dim_in) if pre_norm else nn.Identity(),
            glu() if use_glu else nn.Linear(dim_in, mult * dim_in),
            nn.Identity() if use_glu else nonlin,
            nn.LayerNorm(mult * dim_in, elementwise_affine=False) if mid_norm else nn.Identity(),
            nn.Linear(mult * dim_in, dim_out, bias=bias),
        )
        self.residual = default(residual, identity)

    def forward(self, x, res=None):
        """apply net to x"""
        return self.residual(self.net(x), res=default(x, res))
