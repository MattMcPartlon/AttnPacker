import contextlib
from functools import wraps
from typing import Any

import torch
from torch import Tensor
from einops import rearrange # noqa
from torch import nn

from protein_learning.networks.common.utils import exists

get_max_neg_value = lambda x: torch.finfo(x.dtype).min  # noqa
get_eps = lambda x: torch.finfo(x.dtype).eps  # noqa
get_max_pos_value = lambda x: torch.finfo(x.dtype).max  # noqa


@contextlib.contextmanager
def torch_default_dtype(dtype):
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(prev_dtype)


def cast_torch_tensor(fn):
    @wraps(fn)
    def inner(t):
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=torch.get_default_dtype())
        return fn(t)

    return inner


class FusedGELU(nn.Module):
    def __init__(self):
        super(FusedGELU, self).__init__()

    def forward(self, x):  # noqa
        return x * 0.5 * (1.0 + torch.erf(x / 1.41421))


fused_gelu = torch.jit.script(FusedGELU())


def safe_norm(x: Tensor, dim: int, keepdim=False):
    x = torch.sum(torch.square(x), dim=dim, keepdim=keepdim)
    return torch.sqrt(x + get_eps(x))


def to_rel_pos(coords):
    return rearrange(coords, 'b n ... c -> b () n ... c') - \
           rearrange(coords, 'b n ... c -> b n () ... c')


def rand_uniform(shape, min_val, max_val):
    return torch.rand(shape) * (max_val - min_val) + min_val


def masked_mean(tensor, mask, dim=-1, keepdim=False):
    if not exists(mask):
        return torch.mean(tensor, dim=dim, keepdim=keepdim)
    diff_len = len(tensor.shape) - len(mask.shape)
    mask = mask[(..., *((None,) * diff_len))]
    tensor.masked_fill_(~mask, 0.)
    total_el = mask.sum(dim=dim, keepdim=keepdim)
    mean = tensor.sum(dim=dim, keepdim=keepdim) / total_el.clamp(min=1.)
    mean.masked_fill_(total_el == 0, 0.)
    return mean


def fast_split(arr, splits, dim=0):
    axis_len = arr.shape[dim]
    splits = min(axis_len, max(splits, 1))
    chunk_size = axis_len // splits
    remainder = axis_len - chunk_size * splits
    s = 0
    for i in range(splits):
        adjust, remainder = 1 if remainder > 0 else 0, remainder - 1
        yield torch.narrow(arr, dim, s, chunk_size + adjust)
        s += chunk_size + adjust


def batched_index_select(values, indices, dim=1):
    value_dims = values.shape[(dim + 1):]
    values_shape, indices_shape = map(lambda t: list(t.shape), (values, indices))
    indices = indices[(..., *((None,) * len(value_dims)))]
    indices = indices.expand(*((-1,) * len(indices_shape)), *value_dims)
    value_expand_len = len(indices_shape) - (dim + 1)
    values = values[(*((slice(None),) * dim), *((None,) * value_expand_len), ...)]

    value_expand_shape = [-1] * len(values.shape)
    expand_slice = slice(dim, (dim + value_expand_len))
    value_expand_shape[expand_slice] = indices.shape[expand_slice]
    values = values.expand(*value_expand_shape)

    dim += value_expand_len
    return values.gather(dim, indices)


def fourier_encode(x, num_encodings=4, include_self=True, flatten=True):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x
    scales = 2 ** torch.arange(num_encodings, device=device, dtype=dtype)
    x = x / scales
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim=-1) if include_self else x
    x = rearrange(x, 'b m n ... -> b m n (...)') if flatten else x
    return x


class VerboseExecution(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

        # Register a hook for each layer
        for name, layer in self.model.named_children():
            layer.__name__ = name
            layer.register_forward_hook(
                lambda _layer, _, output: print(f"{_layer.__name__}: {output.shape}")
            )

    def forward(self, x):
        return self.model(x)


class GradientChecker(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.register_hook(model, parents=[])

    def register_hook(self, module: nn.Module, parents):
        for name, child_module in module.named_children():
            self.register_hook(child_module, parents + [name])
        for name, par in module.named_parameters():
            s = f" parameter name :{name}, parent modules {parents}"
            par.register_hook(lambda grad, _s=s: print(_s, f'num gradient'
                                                           f' nans {torch.sum(torch.isnan(grad))}, has'
                                                           f' nan  {torch.sum(torch.isnan(grad)) > 0}, '
                                                           f' shape {grad.shape}') if torch.sum(
                torch.isnan(grad)).item() > 0 else None)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


def augment_with_batch_dim(t: torch.Tensor, batch_dim=0):
    if t.shape[batch_dim] != 1:
        return t.unsqueeze(0)
    return t


def nan_replace_hook(model, val=0):
    for parameter in model.parameters():
        parameter.register_hook(lambda grad, _val=val: nan_hook(grad, _val))
    return model


def nan_hook(grad, val=0):
    safe_tensor = torch.where(torch.isnan(grad), torch.ones_like(grad, device=grad.device) * val, grad)
    safe_tensor = safe_tensor.to(grad.device)
    return safe_tensor


def gradient_clipper(model: nn.Module, val: float) -> nn.Module:
    for parameter in model.parameters():
        parameter.register_hook(lambda grad: grad.clamp_(-val, val))

    return model


def safe_cat(arr, el, dim):
    if not exists(arr):
        return el
    return torch.cat((arr, el), dim=dim)


def get_tensor_device_and_dtype(features):
    first_tensor = next(iter(features.items()))[1]
    return first_tensor.device, first_tensor.dtype


def _to_device(*args, device: Any = 'cpu'):
    ret = [a.to(device) if isinstance(a, torch.Tensor) else a for a in args]
    return ret[0] if len(ret) == 1 else ret


def _detach_n_clone(*args):
    ret = [a.detach().clone() if isinstance(a, torch.Tensor) else a for a in args]
    return ret[0] if len(ret) == 1 else ret


def _detach(*args):
    ret = [a.detach() if isinstance(a, torch.Tensor) else a for a in args]
    return ret[0] if len(ret) == 1 else ret


def ndim(x: torch.Tensor) -> int:
    return len(x.shape)
