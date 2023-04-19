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


@typechecked
def chi_mask_n_indices(
    seq: TensorType["batch", "seq"],
    atom_mask: TensorType["batch", "seq", 37],
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
    coords: TensorType["batch", "seq", 37, 3],
    seq: TensorType["batch", "seq"],
    atom_mask: TensorType["batch", "seq", 37],
    return_unit_vec: bool = False,
    return_mask: bool = False,
) -> Tensor:
    """Get side-chain dihedral angles"""
    # print("[get_sc_dihedral] Start")
    chi_mask, chi_indices = chi_mask_n_indices(seq, atom_mask)
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
    # shape is [b*n_dihedral,n,(1 or 2)]
    dihedrals = rearrange(dihedrals, "(b g) n d -> b n g d", g=chi_mask.shape[-1])
    # #print("     dihedral shape:", dihedrals.shape)
    # #print("[get_sc_dihedral] End")
    return dihedrals, chi_mask if return_mask else dihedrals


# data entries -> (pdb, res_ty, res_idx, rmsd, num_neighbors, chis)
@typechecked
def _per_residue_rmsd(
    predicted: TensorType["batch", "seq", "atoms", 3],
    native: TensorType["batch", "seq", "atoms", 3],
    mask: TensorType["batch", "seq", "atoms"],
    fn=lambda x: torch.square(x),
    reduce=False,
):
    tmp = torch.sum(fn(predicted - native), dim=-1)
    if not reduce:
        return masked_mean(tmp, mask, dim=-1)
    else:
        return torch.sum(masked_mean(tmp, mask, dim=-1)) / max(1, torch.sum(mask.any(dim=-1)))


@typechecked
def swap_symmetric_atoms(
    atom_coords: TensorType["batch", "seq", "atoms", 3], seq_encoding: TensorType["batch", "seq"]
) -> TensorType["batch", "seq", "atoms", 3]:
    """swap symmetric side chain atom coordinates"""
    left_mask = pc.RES_TO_LEFT_SYMM_SC_ATOM_MASK.to(seq_encoding.device)
    right_mask = pc.RES_TO_RIGHT_SYMM_SC_ATOM_MASK.to(seq_encoding.device)
    left_mask, right_mask = left_mask[seq_encoding], right_mask[seq_encoding]
    swapped_coords = atom_coords.detach().clone()
    assert left_mask.shape == (*atom_coords.shape[:3], 2)
    for i in range(left_mask.shape[-1]):
        swapped_coords[left_mask[..., i]] = atom_coords[right_mask[..., i]]
        swapped_coords[right_mask[..., i]] = atom_coords[left_mask[..., i]]
    return swapped_coords


def ca_rmsd(x: TensorType["batch", "seq", 37, 3], y: TensorType["batch", "seq", 37, 3]):
    ca_x, ca_y = map(lambda c: c[..., 1, :], (x, y))
    eps = torch.randn_like(ca_x) * 1e-8
    return torch.sqrt(torch.mean(torch.sum(torch.square(ca_x - (ca_y + eps)), dim=-1)))


def align_symmetric_sidechains(
    native_coords: TensorType["batch", "seq", 37, 3],
    predicted_coords: TensorType["batch", "seq", 37, 3],
    atom_mask: TensorType["batch", "seq", 37],
    native_seq: TensorType["batch", "seq"],
    align_tol: float = 0.25,
):
    """Align side chain atom coordinates"""
    assert native_coords.shape == predicted_coords.shape, f"{native_coords.shape},{predicted_coords.shape}"
    assert native_coords.shape[-2] == 37 == atom_mask.shape[-1]

    aligned_native, aligned_pred = native_coords.detach().clone(), predicted_coords.detach().clone()
    target_frames = SimpleRigids.IdentityRigid(aligned_native.shape[:2], aligned_native.device)
    bb_rmsd = ca_rmsd(native_coords, predicted_coords)
    if bb_rmsd > align_tol:
        target_frames, frames = map(
            lambda x: SimpleRigids.RigidFromBackbone(x[..., :3, :]), (aligned_native, aligned_pred)
        )
        aligned_native = target_frames.apply_inverse(aligned_native)
        aligned_pred = frames.apply_inverse(aligned_pred)

    with torch.no_grad():
        initial_rmsds = _per_residue_rmsd(aligned_pred, aligned_native, atom_mask)
        # swap atoms in symmetric sidechains and get swapped rmsds
        swapped_native = swap_symmetric_atoms(aligned_native, native_seq)
        swapped_rmsds = _per_residue_rmsd(aligned_pred, swapped_native, atom_mask)
        swapped_native = target_frames.apply(swapped_native)
        swap_mask = initial_rmsds > swapped_rmsds
        native_coords[swap_mask] = swapped_native[swap_mask]
        return native_coords
