import torch
from torch import Tensor
from typing import Tuple, List
from protein_learning.protein_utils.dihedral.angle_utils import (
    signed_dihedral_all_12,
    signed_dihedral_4,
    signed_dihedral_all_123,
    unsigned_angle_all
)
from enum import Enum


class TrRosettaOrientationType(Enum):
    """Type enum for trRosetta dihedral
    """
    PHI = ['N', 'CA', 'CB', 'CB']
    PSI = ['CA', 'CB', 'CB']
    OMEGA = ['CA', 'CB', 'CB', 'CA']


def get_atom_tys_for_ori_key(key: TrRosettaOrientationType) -> List[str]:
    """Returns a list of atom types for the given trRosetta orienation type
    """
    return key.value


def get_tr_rosetta_orientation_mat(
        N: Tensor,
        CA: Tensor,
        CB: Tensor,
        ori_type: TrRosettaOrientationType) -> Tensor:
    """Gets trRosetta dihedral matrix for the given coordinates
    :param N: backbone Nitrogen coordinates - shape (b,n,3) or (n,3)
    :param CA: backbone Nitrogen coordinates - shape (b,n,3) or (n,3)
    :param CB: backbone Nitrogen coordinates - shape (b,n,3) or (n,3)
    :param ori_type: trRosetta dihedral type to compute
    :return: dihedral matrix with shape (b,n,n,3) or (n,n,3)
    """
    if ori_type == TrRosettaOrientationType.PSI:
        mat = unsigned_angle_all([CA, CB, CB])
    elif ori_type == TrRosettaOrientationType.OMEGA:
        mat = signed_dihedral_all_12([CA, CB, CB, CA])
    elif ori_type == TrRosettaOrientationType.PHI:
        mat = signed_dihedral_all_123([N, CA, CB, CB])
    else:
        raise Exception(f'dihedral type {ori_type} not accepted')
    # expand back to full size
    return mat


def get_tr_rosetta_orientation_mats(
        N: Tensor,
        CA: Tensor,
        CB: Tensor) -> Tuple[Tensor, ...]:
    phi = get_tr_rosetta_orientation_mat(N, CA, CB, TrRosettaOrientationType.PHI)
    psi = get_tr_rosetta_orientation_mat(N, CA, CB, TrRosettaOrientationType.PSI)
    omega = get_tr_rosetta_orientation_mat(N, CA, CB, TrRosettaOrientationType.OMEGA)
    return phi, psi, omega


def get_bb_dihedral(N: Tensor, CA: Tensor, C: Tensor) -> Tuple[Tensor, ...]:
    """
    Gets backbone dihedrals for
    :param N: (n,3) or (b,n,3) tensor of backbone Nitrogen coordinates
    :param CA: (n,3) or (b,n,3) tensor of backbone C-alpha coordinates
    :param C: (n,3) or (b,n,3) tensor of backbone Carbon coordinates
    :return: phi, psi, and omega dihedrals angles (each of shape (n,) or (b,n))
    """
    assert all([len(N.shape) == len(x.shape) for x in (CA, C)])
    squeeze = len(N.shape) == 2
    N, CA, C = map(lambda x: x.unsqueeze(0), (N, CA, C)) if squeeze else (N, CA, C)
    b, n = N.shape[:2]
    phi, psi, omega = [torch.zeros(b, n, device=N.device) for _ in range(3)]
    phi[:, 1:] = signed_dihedral_4([C[:, :-1], N[:, 1:], CA[:, 1:], C[:, 1:]])
    psi[:, :-1] = signed_dihedral_4([N[:, :-1], CA[:, :-1], C[:, :-1], N[:, 1:]])
    omega[:, :-1] = signed_dihedral_4([CA[:, :-1], C[:, :-1], N[:, 1:], CA[:, 1:]])
    return map(lambda x: x.squeeze(0), (phi, psi, omega)) if squeeze else (phi, psi, omega)
