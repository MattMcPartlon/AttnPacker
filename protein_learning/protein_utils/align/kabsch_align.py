from typing import Optional, Tuple

import torch
from einops import rearrange # noqa
from torch import Tensor

from protein_learning.common.helpers import masked_mean, exists

to_rel_pos = lambda x: rearrange(x, "a b c -> () a b c") - rearrange(x, "a b c -> a () b c")
matmul = lambda a, b: torch.einsum('b i j, b j k -> b i k', a, b)


def kabsch_align(
        align_to: Tensor,
        align_from: Tensor,
        apply_translation: bool = True,
        mask: Optional[Tensor] = None,
        device='cpu'
) -> Tuple[Tensor, Tensor]:
    """ Kabsch alignment of "align_from" into "align_to".
    The coordinates specified in "align_to" will not be modified.

    align_to: tensor of shape (b,n,3)
    align_from: tensor of shape (b,n,3)
    mask: tensor of shape (b,n)
    device: device to run alignment on (cpu tends to be fastest)
    return:
        align_to, alignment(aligned_from)
    """
    assert align_to.ndim == 3 and align_to.shape == align_from.shape
    if exists(mask):
        assert align_to.shape[:2] == mask.shape, f"{align_to.shape},{mask.shape}"
    b, n, _device = *align_to.shape[:2], align_to.device  # noqa
    with torch.no_grad():
        rot_to, mean_to, mean_from = _calc_kabsch_rot_n_trans(align_to=align_to,
                                                              align_from=align_from,
                                                              mask=mask,
                                                              device=device)
        if not apply_translation:
            mean_to, mean_from = 0, 0
        aligned_from = torch.matmul(align_from - mean_from, rot_to) + mean_to
        return align_to, aligned_from.detach()


def _calc_kabsch_rot_n_trans(
        align_to: torch.Tensor,
        align_from: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        device='cpu'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculate RMSD-minimizing rotation and translation mapping align_from to align_to

    :param align_to: tensor of shape (b,n,3) (batch,seq,coords)
    :param align_from: tensor of shape (b,n,3)
    :param mask : mask tensor of shape (b,n) indicating which coordinates to compute alignment for.
    """
    _device = align_to.device
    #  center X and Y to the origin
    if exists(mask):
        mask = rearrange(mask, "b n -> b n ()")
        assert mask.shape[0] == align_to.shape[0] and mask.shape[1] == align_to.shape[1]
    mean_to, mean_from = map(lambda x: masked_mean(x, mask, dim=1, keepdim=True), (align_to, align_from))
    align_to, align_from = align_to - mean_to, align_from - mean_from
    if exists(mask):
        align_to, align_from = map(lambda x: torch.masked_fill(x, ~mask, 0), (align_to, align_from))

    # calculate covariance matrix (for each prot in the batch)
    C = matmul(rearrange(align_from, "b n c -> b c n"), align_to).to(device)
    V, S, W = torch.svd(C)
    # correct for right-handed coord. system
    d = (torch.det(V) * torch.det(W)) < 0.0
    S[d, -1] = S[d, -1] * (-1)
    V[d, :, -1] = V[d, :, -1] * (-1)
    # Create Rotation matrix U
    U = matmul(V, rearrange(W, "b i j-> b j i")).to(_device)
    return U, mean_to.detach(), mean_from.detach()
