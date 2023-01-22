from math import sqrt
from typing import Tuple, Optional
from torch import Tensor
import torch
from einops import rearrange # noqa
from protein_learning.protein_utils.align.kabsch_align import _calc_kabsch_rot_n_trans
from protein_learning.common.helpers import default


def get_per_res_alignment(
        align_to: Tensor,
        align_from: Tensor,
        to_align: Optional[Tensor] = None,
) -> Tensor:
    """aligns atom coordinates on a per-residue basis

    This function generates a rotation and translation for each residue in the
    input mapping the coordinates in align_from to the coordinates in align_to

    The rotations/translations are applied to to_align (if it is given), otherwise
    applied to 'align_from' and the output is returned.

    :param align_to: per-residue atom coordinates to align to (n,a,3)
    :param align_from: per-residue atom coordinates to align from (n,a,3)
    :param to_align: the coordinates to align (optional), align_from is used as default
    :return: per-residue rotation and translation applied to 'to_align'
    """
    rot, trans_to, trans_from = get_rot_n_trans_for_per_res_align(align_to, align_from)
    to_align = default(to_align, align_from)
    aligned_coords = apply_rot_n_trans_for_per_res_align(
        coords=to_align,
        rot=rot,
        pre_rot_trans=trans_from,
        post_rot_trans=trans_to
    )
    return aligned_coords


def impute_beta_carbon(bb_coords: Tensor) -> Tensor:
    """Imputes coordinates of beta carbon from tensor of residue coordinates
    :param bb_coords: shape (n,4,3) where dim=1 is N,CA,C coordinates.
    :return: imputed CB coordinates for each residue
    """
    assert bb_coords.shape[1] == 3 and bb_coords.ndim == 3, f"{bb_coords.shape}"
    bb_coords = rearrange(bb_coords, "n a c -> a n c")
    N, CA, C = bb_coords  # noqa
    n, c = N - CA, C - CA
    n_cross_c = torch.cross(n, c)
    t1 = sqrt(1 / 3) * (n_cross_c / torch.norm(n_cross_c, dim=-1, keepdim=True))
    n_plus_c = n + c
    t2 = sqrt(2 / 3) * (n_plus_c / torch.norm(n_plus_c, dim=-1, keepdim=True))
    return CA + (t1 + t2)


def get_rot_n_trans_for_per_res_align(align_to: Tensor, align_from: Tensor) -> Tuple[Tensor, ...]:
    """Gets rotation and translation mapping per-residue native coordinate frames into
    respective decoy frames
    :param align_to: coordinates to align to (n,a,3) - second axis is atom type
    :param native_bb: coordinates to align from (n,a,3) - second axis is atom type
    :return: tensor of rotations (n,3,3) and tensor of translations (n,1,3)
    """
    assert align_from.ndim == align_to.ndim == 3
    rot, trans_to, trans_from = _calc_kabsch_rot_n_trans(align_to=align_to, align_from=align_from)
    rot = rearrange(rot, "a b c -> a c b")
    return rot, trans_to, trans_from


def apply_rot_n_trans_for_per_res_align(
        coords: Tensor,
        rot: Tensor,
        pre_rot_trans: Tensor,
        post_rot_trans: Tensor,
        enable_grad: bool = False
) -> Tensor:
    """Applies rotation and translation on a per-residue basis to the input coords
    :param coords: per-residue coordinates (n,a,3) - (sequence, atom_ty, coord)
    :param rot: tensor of rotations to apply to each residue - (n,3,3)
    :param pre_rot_trans: tensor of translations to apply to each residue - (n,1,3)
    before applying rotation
    :param post_rot_trans: tensor of translations to apply to each residue - (n,1,3)
    after applying rotation
    :param enable_grad: whether to enable gradients during computation
    :return: native coordinates with rotation and translation applied - (n,a,c)
    """
    device = coords.device
    rot, pre_rot_trans, post_rot_trans = rot.to(device), pre_rot_trans.to(device), post_rot_trans.to(device)
    with torch.set_grad_enabled(enable_grad):
        # first rotate, then translate
        coords = coords - pre_rot_trans
        return torch.einsum("nij,naj->nai", rot, coords) + post_rot_trans
