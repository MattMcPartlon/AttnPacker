"""Helper functions for computing model loss
"""
from typing import Optional, Dict, List

import torch
import torch.nn.functional as F  # noqa
from einops import rearrange  # noqa
from torch import einsum
from torch import nn

from protein_learning.common.helpers import default

MAX_FLOAT = 1e6
to_rel_coord = lambda x: rearrange(x, "... n c -> ... n () c") - rearrange(x, "... n c-> ... () n c")
outer_sum = lambda x, y=None: rearrange(x, "... i -> ... i ()") + rearrange(default(y, x), "... i -> ... () i")
outer_prod = lambda x, y=None: einsum("... i, ... j -> ... i j", x, default(y, x))


def FeedForward(
        dim_in: int,
        dim_out: int,
        pre_norm: bool = False,
        dim_hidden: Optional[int] = None,
        n_hidden_layers: int = 1,
        nonlin=nn.GELU
):
    """FeedForward Network"""
    dim_hidden = default(dim_hidden, dim_in) if n_hidden_layers > 0 else dim_out
    fst = nn.Linear(dim_in, dim_hidden)
    lst = nn.Linear(dim_hidden, dim_out) if n_hidden_layers > 0 else nn.Identity()
    nrm = nn.LayerNorm(dim_in) if pre_norm else nn.Identity()
    hidden_layers = [nonlin() if n_hidden_layers > 0 else nn.Identity()]
    for layer_idx in range(max(0, n_hidden_layers - 1)):
        hidden_layers.append(nn.Linear(dim_in, dim_hidden))
        hidden_layers.append(nonlin())
    layers = [nrm] + [fst] + hidden_layers + [lst]
    return nn.Sequential(*layers)


def partition(lst: List, chunk: int):
    """Parittions a list into chunks of size chunk"""
    for i in range(0, len(lst), chunk):  # noqa
        yield lst[i:i + chunk]


def get_tm_scale(n: int) -> float:
    """Gets scale value applied to normalize TM-score"""
    return 0.5 if n <= 15 else 1.24 * ((n - 15.0) ** (1. / 3.)) - 1.8


def get_loss_func(p: int, *args, **kwargs):
    """Returns l_p loss function"""
    if p == 1:
        return ClampedSmoothL1Loss(*args, **kwargs)
    elif p == 2:
        return ClampedMSELoss(*args, **kwargs)
    else:
        return lambda x, y: torch.mean(torch.pow(x - y, p) ** (1 / p))


class ClampedSmoothL1Loss(torch.nn.Module):
    """Clamped smooth l1-loss"""

    def __init__(self, beta: float = 1, reduction: str = 'mean', min_clamp: float = 0, max_clamp: float = MAX_FLOAT):
        super().__init__()
        self.beta = beta
        self.reduction = reduction
        self.clamp = (min_clamp, max_clamp)

    def forward(self, pred: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
        """Apply function to predicted and ground truth tensor"""
        if self.beta < 1e-5:
            # avoid nan in gradients
            loss = torch.abs(pred - actual)
        else:
            n = torch.abs(pred - actual)
            cond = n < self.beta
            loss = torch.where(cond, 0.5 * n ** 2 / self.beta, n - 0.5 * self.beta)
        loss = loss.clamp(*self.clamp)
        if self.reduction == "mean":
            return loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class ClampedMSELoss(torch.nn.Module):
    """Clamped MSE loss"""

    def __init__(
            self,
            beta=None,  # noqa
            reduction: str = 'mean',
            min_clamp: float = 0,
            max_clamp: float = 10,
            normalize=True):
        super().__init__()
        self.reduction = reduction
        self.clamp = (min_clamp, max_clamp)
        self.normalize = normalize

    def forward(self, pred: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
        """Apply function to predicted and ground truth tensor"""
        loss = (pred - actual) ** 2
        loss = torch.clamp(loss, self.clamp[0] ** 2, self.clamp[1] ** 2)
        # loss = loss / self.clamp[1] if self.normalize else loss
        if self.reduction == "mean":
            loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss


def softmax_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Computes softmax cross entropy of logits given ground truth labels"""
    return -torch.sum(labels * nn.functional.log_softmax(logits, dim=-1), dim=-1)


def sigmoid_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Computes sigmoid cross entropy of logits given ground truth labels"""
    log_p = nn.functional.logsigmoid(logits)
    log_not_p = nn.functional.logsigmoid(-logits)
    return -labels * log_p - (1. - labels) * log_not_p


def get_centers(bin_edges: torch.Tensor):
    """Gets bin centers from the bin edges.

    Args:
      bin_edges: tensor of shape (num_bins + 1) defining bin edges

    Returns:
      bin_centers: [num_bins] the error bin centers.
    """
    centers = [(s + e) / 2 for s, e in zip(bin_edges[:-1], bin_edges[1:])]
    return torch.tensor(centers)


def compute_predicted_distance_error(
        logits: torch.Tensor,
        dist_bins: torch.Tensor) -> torch.Tensor:
    """Computes aligned confidence metrics from logits.

    Args:
      logits: tensor of shape (n_res, n_res, d).
      dist_bins: tensor of shape (d)

    Returns:
      tensor of shape (n_res, n_res) containing expected
      (signed) distance error.

    """
    dist_probs = F.softmax(logits, dim=-1)
    return torch.sum(dist_probs * dist_bins.reshape(1, 1, dist_bins.shape[-1]), dim=-1)


def safe_detach_item(x) -> float:
    if torch.is_tensor(x):
        assert x.numel() == 1
        return x.detach().cpu().item()
    assert isinstance(x, int) or isinstance(x, float)
    return x


def to_info(baseline, actual, pred, loss_val: Optional[float] = None) -> Dict[str, float]:
    return dict(
        baseline=safe_detach_item(baseline),
        actual=safe_detach_item(actual),
        predicted=safe_detach_item(pred),
        loss_val=safe_detach_item(loss_val)
    )
