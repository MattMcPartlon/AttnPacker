"""Utility functions for working with protein side-chains"""
from typing import Tuple, Union, List

import torch
from einops import rearrange, repeat  # noqa
from torch import Tensor
from scp_postprocess.helpers import masked_mean, batched_index_select, disable_tf32
import scp_postprocess.sidechain_rigid_utils as scru
import scp_postprocess.protein_constants as pc
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
from torch.cuda.amp import autocast
cos_max, cos_min = (1 - 1e-9), -(1 - 1e-9)
min_norm_clamp = 1e-9

patch_typeguard()

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
    with disable_tf32(), autocast(enabled=False):
        p0, p1, p2, p3 = ps
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

@typechecked
def chi_mask_n_indices(
    seq: TensorType["batch","seq"],
    atom_mask: TensorType["batch","seq",37],
) -> Tuple[Tensor, Tensor]:
    """Gets chi-dihedral mask and atom indices"""
    # print("[chi_mask_n_indices] Start")
    res_to_mask = scru.res_to_chi_angle_mask_tensor.to(seq.device)
    res_to_posns = scru.res_to_chi_atom_groups_tensor.to(seq.device)
    chi_masks = res_to_mask[seq.squeeze()]  # n,4
    chi_indices = res_to_posns[seq.squeeze()]  # n,4,4

    # res_index, group_index, atom_index
    reshaped_indices = rearrange(chi_indices, "n g a -> g n a")
    reshaped_atom_mask = repeat(atom_mask, "... n a -> (... g) n a", g=4)
    atom_exists_mask = batched_index_select(reshaped_atom_mask, reshaped_indices, dim=2)
    atom_exists_mask = rearrange(atom_exists_mask, "g n a -> n g a").all(dim=-1)
    mask, indices = chi_masks & atom_exists_mask, chi_indices
    # print("     mask and index shapes: ", mask.shape, indices.shape)
    # print("[chi_mask_n_indices] End")
    return mask, indices


def get_sc_dihedral(
    coords: TensorType["batch","seq",37,3],
    chi_mask: TensorType["batch","seq",4],
    chi_indices: TensorType["batch","seq",4,4],
    return_unit_vec: bool = True,
) -> Tensor:
    """Get side-chain dihedral angles"""
    reshape_chi_mask = rearrange(chi_mask, "n g -> g n")  # nx4 -> 4xn
    assert reshape_chi_mask.shape[0] == 4, f"{reshape_chi_mask.shape}"
    reshaped_indices = rearrange(chi_indices, "n g a -> g n a")
    assert reshaped_indices.shape[0] == reshaped_indices.shape[-1] == 4, f"{reshaped_indices.shape}"
    m = reshape_chi_mask.shape[0]
    assert m <= 4, f"{reshape_chi_mask.shape}"
    coords = coords if coords.ndim == 4 else coords.unsqueeze(0)
    rep_coords = repeat(coords, "b n a c -> (m b) n a c", m=m)
    assert rep_coords.shape[:2] == reshaped_indices.shape[:2]
    select_coords = batched_index_select(rep_coords, reshaped_indices, dim=2)
    assert select_coords.shape[:3] == reshaped_indices.shape
    dihedrals = signed_dihedral_4(
        ps=rearrange(select_coords, "b n a c -> a b n c"),
        return_unit_vec=return_unit_vec,
    )
    dihedrals = rearrange(dihedrals, "(b g) n d -> b n g d", g=chi_mask.shape[-1])
    return dihedrals


# data entries -> (pdb, res_ty, res_idx, rmsd, num_neighbors, chis)
@typechecked
def _per_residue_rmsd(
    predicted: TensorType["batch","seq","atoms",3],
    native: TensorType["batch","seq","atoms",3],
    mask: TensorType["batch","seq","atoms"],
    fn=lambda x: torch.square(x), 
    reduce=False
    ):
    tmp = torch.sum(fn(predicted - native), dim=-1)
    if not reduce:
        return masked_mean(tmp, mask, dim=-1)
    else:
        return torch.sum(masked_mean(tmp, mask, dim=-1)) / max(1, torch.sum(mask.any(dim=-1)))

@typechecked
def swap_symmetric_atoms(
    atom_coords: TensorType["batch","seq","atoms",3], 
    seq_encoding: TensorType["batch","seq"]
    ) -> TensorType["batch","seq","atoms",3]:
    """swap symmetric side chain atom coordinates"""
    left_mask = pc.RES_TO_LEFT_SYMM_SC_ATOM_MASK.to(seq_encoding.device)
    right_mask = pc.RES_TO_RIGHT_SYMM_SC_ATOM_MASK.to(seq_encoding.device)
    left_mask, right_mask = left_mask[seq_encoding], right_mask[seq_encoding]
    swapped_coords = atom_coords.detach().clone()
    assert left_mask.shape == (*atom_coords.shape[:3],2)
    for i in range(left_mask.shape[-1]):
        swapped_coords[left_mask[...,i]] = atom_coords[right_mask[...,i]]
        swapped_coords[right_mask[...,i]] = atom_coords[left_mask[...,i]]
    return swapped_coords

def align_symmetric_sidechains(
    native_coords:TensorType["batch","seq",37,3],
    predicted_coords:TensorType["batch","seq",37,3],
    atom_mask:TensorType["batch","seq",37],
    native_seq:TensorType["batch","seq"],
):
    """Align side chain atom coordinates"""
    assert native_coords.shape == predicted_coords.shape
    assert native_coords.shape[-2] == 37 == atom_mask.shape[-1]
    with torch.no_grad():
        initial_rmsds = _per_residue_rmsd(predicted_coords, native_coords, atom_mask)
        # swap atoms in symmetric sidechains and get swapped rmsds
        swapped_native = swap_symmetric_atoms(native_coords, native_seq)
        swapped_rmsds = _per_residue_rmsd(predicted_coords, swapped_native, atom_mask)
        swap_mask = initial_rmsds > swapped_rmsds
        assert swap_mask.ndim == 2
        native_coords[swap_mask] = swapped_native[swap_mask]
        return native_coords


