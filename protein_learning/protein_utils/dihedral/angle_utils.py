import torch
from torch import Tensor
from typing import Tuple, Union, List

cos_max, cos_min = (1 - 1e-9), -(1 - 1e-9)
min_norm_clamp = 1e-9
from protein_learning.common.helpers import disable_tf32


def signed_dihedral_4(
    ps: Union[Tensor, List[Tensor]],
    return_mask=False,
    return_unit_vec: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """computes (signed) dihedral angle of input points.

     works for batched and unbatched point lists

    :param ps: a list of four tensors of points. dihedral angle between
    ps[0,i],ps[1,i],ps[2,i], and ps[3,i] will be ith entry of output.
    :param return_mask: whether to return a mask indicating where dihedral
    computation may have had precision errors.

    :returns : list of dihedral angles
    """
    # switch to higher precision dtype
    p0, p1, p2, p3 = ps
    device_type = "cpu" if p0.device.type == "cpu" else "cuda"
    with disable_tf32(), torch.autocast(device_type=device_type, enabled=False):
        b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
        mask = torch.norm(b1, dim=-1) > 1e-7
        b1 = torch.clamp_min(b1, 1e-6)
        b1 = b1 / torch.norm(b1, dim=-1, keepdim=True)
        v = b0 - torch.sum(b0 * b1, dim=-1, keepdim=True) * b1
        w = b2 - torch.sum(b2 * b1, dim=-1, keepdim=True) * b1
        x = torch.sum(v * w, dim=-1)
        y = torch.sum(torch.cross(b1, v) * w, dim=-1)
    if not return_unit_vec:
        res = torch.atan2(y, x)
    else:
        res = torch.cat((y.unsqueeze(-1), x.unsqueeze(-1)), dim=-1)
    return res if not return_mask else (res, mask)


def signed_dihedral_all_12(ps: List[Tensor]) -> Tensor:
    """
    Computes signed dihedral of points taking p2-p1 as dihedral axis
    :param ps:
    :return:
    """
    device_type = "cpu" if ps[0].device.type == "cpu" else "cuda"
    with disable_tf32(), torch.autocast(device_type=device_type, enabled=False):
        p0, p1, p2, p3 = ps
        b0, b1, b2 = p0 - p1, p2.unsqueeze(-3) - p1.unsqueeze(-2), p3 - p2
        b1 = b1 / torch.norm(b1, dim=-1, keepdim=True).clamp_min(min_norm_clamp)
        v = b0.unsqueeze(-2) - torch.sum(b0.unsqueeze(-2) * b1, dim=-1, keepdim=True) * b1
        w = b2.unsqueeze(-3) - torch.sum(b2.unsqueeze(-3) * b1, dim=-1, keepdim=True) * b1
        x = torch.sum(v * w, dim=-1)
        y = torch.sum(torch.cross(b1, v) * w, dim=-1)
    return torch.atan2(y, x)


def signed_dihedral_all_123(ps) -> Tensor:
    """

    :param ps:
    :return:
    """
    device_type = "cpu" if ps[0].device.type == "cpu" else "cuda"
    with disable_tf32(), torch.autocast(device_type=device_type, enabled=False):
        p0, p1, p2, p3 = ps
        b0, b1, b2 = p0 - p1, p2 - p1, p3.unsqueeze(-3) - p2.unsqueeze(-2)
        b1 = b1 / torch.norm(b1, dim=-1, keepdim=True).clamp_min(min_norm_clamp)
        v = b0 - torch.sum(b0 * b1, dim=-1, keepdim=True) * b1
        w = b2 - torch.sum(b2 * b1.unsqueeze(-2), dim=-1, keepdim=True) * b1.unsqueeze(-2)
        x = torch.sum(v.unsqueeze(-2) * w, dim=-1)
        y = torch.sum(torch.cross(b1, v).unsqueeze(-2) * w, dim=-1)
        ret = torch.atan2(y, x)
    return ret


def unsigned_angle_all(ps: List[Tensor]) -> Tensor:
    """Retrieves a matrix of (unsigned) angles between input points

    returns: a matrix M where M[i,j] is the angle btwn the lines formed
    by ps0[i],ps1[i] and ps[1,i],ps[2,j].
    """
    device_type = "cpu" if ps[0].device.type == "cpu" else "cuda"
    with disable_tf32(), torch.autocast(device_type=device_type, enabled=False):
        p0, p1, p2 = ps[0], ps[1], ps[2]
        b01, b12 = p0 - p1, p2.unsqueeze(-3) - p1.unsqueeze(-2)
        M = b01.unsqueeze(-2) * b12
        n01, n12 = torch.norm(b01, dim=-1, keepdim=True), torch.norm(b12, dim=-1)
        prods = torch.clamp_min(n01 * n12, min_norm_clamp)
        cos_theta = torch.sum(M, dim=-1) / prods
        cos_theta[cos_theta < cos_min] = cos_min
        cos_theta[cos_theta > cos_max] = cos_max
    return torch.acos(cos_theta)
