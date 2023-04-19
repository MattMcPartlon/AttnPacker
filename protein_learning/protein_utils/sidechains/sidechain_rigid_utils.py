"""Functions adapted from OpenFold:

https://github.com/aqlaboratory/openfold/blob/59277de16825cfdafe37033012d0530595b9ad6d/openfold/np/residue_constants.py
https://github.com/aqlaboratory/openfold/blob/59277de16825cfdafe37033012d0530595b9ad6d/openfold/data/data_transforms.py

Note that there is not a one-to-one correspondence between these 
functions and the corresponding functions in the openfold implementation.
Function names often differ, and all but a few helped functions 
require major refactors to fit our framework. 
"""
import torch
from protein_learning.networks.common.of_rigid_utils import Rigid, Rotation
from torch import nn
from typing import Dict
import protein_learning.common.protein_constants as pc
from protein_learning.common.helpers import disable_tf32
from typing import Dict
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()

# A list of atoms (excluding hydrogen) for each AA type. PDB naming convention.
# Format: The list for each AA type contains chi1, chi2, chi3, chi4 in
# this order (or a relevant subset from chi1 onwards). ALA and GLY don't have
# chi angles so their chi angle lists are empty.
res_to_chi_atom_groups = {
    "ALA": [],
    # Chi5 in arginine is always 0 +- 5 degrees, so ignore it.
    "ARG": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "NE"],
        ["CG", "CD", "NE", "CZ"],
    ],
    "ASN": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "OD1"]],
    "ASP": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "OD1"]],
    "CYS": [["N", "CA", "CB", "SG"]],
    "GLN": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "OE1"],
    ],
    "GLU": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "OE1"],
    ],
    "GLY": [],
    "HIS": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "ND1"]],
    "ILE": [["N", "CA", "CB", "CG1"], ["CA", "CB", "CG1", "CD1"]],
    "LEU": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "LYS": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "CE"],
        ["CG", "CD", "CE", "NZ"],
    ],
    "MET": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "SD"],
        ["CB", "CG", "SD", "CE"],
    ],
    "PHE": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "PRO": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"]],
    "SER": [["N", "CA", "CB", "OG"]],
    "THR": [["N", "CA", "CB", "OG1"]],
    "TRP": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "TYR": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "VAL": [["N", "CA", "CB", "CG1"]],
}

res_to_chi_atom_groups_tensor = torch.zeros(21, 4, 4)
for res_ty, atom_groups in res_to_chi_atom_groups.items():
    res_posn = pc.AA_TO_INDEX[res_ty]
    for group_idx, group in enumerate(atom_groups):
        for atom_idx, atom in enumerate(group):
            atom_posn = pc.ALL_ATOM_POSNS[atom]
            res_to_chi_atom_groups_tensor[res_posn, group_idx, atom_idx] = atom_posn
res_to_chi_atom_groups_tensor = res_to_chi_atom_groups_tensor.long()

# If chi angles given in fixed-length array, this matrix determines how to mask
# them for each AA type. The order is as per restype_order (see below).
res_to_chi_angle_mask = {}
for residue, chi_groups in res_to_chi_atom_groups.items():
    res_to_chi_angle_mask[residue] = torch.zeros(4)
    res_to_chi_angle_mask[residue][: len(chi_groups)] = 1

res_to_chi_angle_mask_tensor = torch.stack(
    [res_to_chi_angle_mask[pc.ONE_TO_THREE[res]] for res in pc.RES_TYPES],
).bool()
residue_to_atoms = {k: pc.BB_ATOMS + pc.AA_TO_SC_ATOMS[k] for k in pc.AA_TO_SC_ATOMS}


# create an array with (restype, atomtype) --> rigid_group_idx
# and an array with (restype, atomtype, coord) for the atom positions
# and compute affine transformation matrices (4,4) from one rigid group to the
# previous group
restype_atom37_to_rigid_group = torch.zeros(21, 37, dtype=torch.long)
restype_atom37_mask = torch.zeros(21, 37, dtype=torch.float32)
restype_atom37_rigid_group_positions = torch.zeros(21, 37, 3, dtype=torch.float32)
restype_rigid_group_default_frame = torch.zeros(21, 8, 4, 4, dtype=torch.float32)


def _make_rigid_transformation_4x4(ex, ey, translation):
    """Create a rigid 4x4 transformation matrix from two axes and transl."""
    # Normalize ex.

    ex_normalized = ex / torch.norm(ex + 1e-12)

    # make ey perpendicular to ex
    ey_normalized = ey - torch.dot(ey, ex_normalized) * ex_normalized
    ey_normalized /= torch.norm(ey_normalized + 1e-12)

    # compute ez as cross product
    eznorm = torch.cross(ex_normalized, ey_normalized, dim=-1)
    m = torch.stack([ex_normalized, ey_normalized, eznorm, translation]).T
    m = torch.cat((m, torch.tensor([[0.0, 0.0, 0.0, 1.0]])), dim=0)
    return m


def _make_rigid_group_constants():
    """Fill the arrays above."""
    for res_num, restype_letter in enumerate(pc.RES_TYPES):
        resname = pc.ONE_TO_THREE[restype_letter]
        for atom_name, group_idx, atom_position in pc.RIGID_GROUP_ATOM_POSITIONS[resname]:
            atom_num = pc.ALL_ATOM_POSNS[atom_name]
            restype_atom37_to_rigid_group[res_num, atom_num] = group_idx
            restype_atom37_mask[res_num, atom_num] = 1
            posn = torch.tensor(atom_position)
            restype_atom37_rigid_group_positions[res_num, atom_num, :] = posn

    # all atom positions for given residue
    for res_num, restype_letter in enumerate(pc.RES_TYPES):
        resname = pc.ONE_TO_THREE[restype_letter]
        atom_positions = {name: torch.tensor(pos) for name, _, pos in pc.RIGID_GROUP_ATOM_POSITIONS[resname]}

        # backbone to backbone is the identity transform
        restype_rigid_group_default_frame[res_num, 0, :, :] = torch.eye(4)

        # pre-omega-frame to backbone (currently dummy identity matrix)
        restype_rigid_group_default_frame[res_num, 1, :, :] = torch.eye(4)

        # phi-frame to backbone
        mat = _make_rigid_transformation_4x4(
            ex=atom_positions["N"] - atom_positions["CA"],
            ey=torch.tensor([1.0, 0.0, 0.0]),
            translation=atom_positions["N"],
        )
        restype_rigid_group_default_frame[res_num, 2, :, :] = mat

        # psi-frame to backbone
        mat = _make_rigid_transformation_4x4(
            ex=atom_positions["C"] - atom_positions["CA"],
            ey=atom_positions["CA"] - atom_positions["N"],
            translation=atom_positions["C"],
        )
        restype_rigid_group_default_frame[res_num, 3, :, :] = mat

        # chi1-frame to backbone
        if res_to_chi_angle_mask[resname][0]:
            base_atom_names = res_to_chi_atom_groups[resname][0]
            base_atom_positions = [atom_positions[name] for name in base_atom_names]
            mat = _make_rigid_transformation_4x4(
                ex=base_atom_positions[2] - base_atom_positions[1],
                ey=base_atom_positions[0] - base_atom_positions[1],
                translation=base_atom_positions[2],
            )
            restype_rigid_group_default_frame[res_num, 4, :, :] = mat

        # chi2-frame to chi1-frame
        # chi3-frame to chi2-frame
        # chi4-frame to chi3-frame
        # luckily all rotation axes for the next frame start at (0,0,0) of the
        # previous frame
        for chi_idx in range(1, 4):
            if res_to_chi_angle_mask[resname][chi_idx] == 1:
                axis_end_atom_name = res_to_chi_atom_groups[resname][chi_idx][2]
                axis_end_atom_position = atom_positions[axis_end_atom_name]
                mat = _make_rigid_transformation_4x4(
                    ex=axis_end_atom_position,
                    ey=torch.tensor([-1.0, 0.0, 0.0]),
                    translation=axis_end_atom_position,
                )
                restype_rigid_group_default_frame[res_num, 4 + chi_idx, :, :] = mat


_make_rigid_group_constants()


def frames_and_literature_positions_to_atom37_pos(
    r: Rigid,
    aatype: torch.Tensor,
    default_frames,
    group_idx,
    atom_mask,
    lit_positions,
):
    # [*, N, 37]
    group_mask = group_idx[aatype, ...]

    # [*, N, 37, 8]
    group_mask = nn.functional.one_hot(
        group_mask,
        num_classes=default_frames.shape[-3],
    )
    # print("group_mask", group_mask.shape)
    # print(group_mask[0,30])

    # [*, N, 37, 8]
    t_atoms_to_global = r[..., None, :] * group_mask

    # [*, N, 37]
    t_atoms_to_global = t_atoms_to_global.map_tensor_fn(lambda x: torch.sum(x, dim=-1))

    # [*, N, 37, 1]
    atom_mask = atom_mask[aatype, ...].unsqueeze(-1)

    # [*, N, 37, 3]
    lit_positions = lit_positions[aatype, ...]
    pred_positions = t_atoms_to_global.apply(lit_positions)
    pred_positions = pred_positions * atom_mask

    return pred_positions


def torsion_angles_to_frames(
    r: Rigid,
    alpha: torch.Tensor,
    aatype: torch.Tensor,
    rrgdf: torch.Tensor,
):
    # print("[Torsion to Frames] Start")
    # print("     r:", r.shape)
    # print("     alpha:", alpha.shape)
    # print("     aatype:", aatype.shape)
    # print("     rrgdf:", rrgdf.shape)

    # [*, N, 8, 4, 4]
    default_4x4 = rrgdf[aatype, ...]
    # print("     default_4x4:", default_4x4.shape, "[*, N, 8, 4, 4]")

    # [*, N, 8] transformations, i.e.
    #   One [*, N, 8, 3, 3] rotation matrix and
    #   One [*, N, 8, 3]    translation matrix
    default_r = r.from_tensor_4x4(default_4x4)
    # print("     default_r:", default_r.shape, "[*, N, 8, ???]")

    bb_rot = alpha.new_zeros((*((1,) * len(alpha.shape[:-1])), 2))
    bb_rot[..., 1] = 1
    # print("     bb_rot:", bb_rot.shape)

    # [*, N, 8, 2]
    alpha = torch.cat([bb_rot.expand(*alpha.shape[:-2], -1, -1), alpha], dim=-2)
    ##print("     alpha reshaped :", alpha.shape)

    # [*, N, 8, 3, 3]
    # Produces rotation matrices of the form:
    # [
    #   [1, 0  , 0  ],
    #   [0, a_2,-a_1],
    #   [0, a_1, a_2]
    # ]
    # This follows the original code rather than the supplement, which uses
    # different indices.
    # print("     default_r rot:", default_r.get_rots().get_rot_mats().shape)
    # print("     default_r trans:", default_r.get_trans().shape)

    all_rots = alpha.new_zeros(default_r.get_rots().get_rot_mats().shape)
    all_rots[..., 0, 0] = 1
    all_rots[..., 1, 1] = alpha[..., 1]
    all_rots[..., 1, 2] = -alpha[..., 0]
    all_rots[..., 2, 1:] = alpha

    all_rots = Rigid(Rotation(rot_mats=all_rots), None)

    all_frames = default_r.compose(all_rots)

    chi2_frame_to_frame = all_frames[..., 5]
    chi3_frame_to_frame = all_frames[..., 6]
    chi4_frame_to_frame = all_frames[..., 7]

    chi1_frame_to_bb = all_frames[..., 4]
    chi2_frame_to_bb = chi1_frame_to_bb.compose(chi2_frame_to_frame)
    chi3_frame_to_bb = chi2_frame_to_bb.compose(chi3_frame_to_frame)
    chi4_frame_to_bb = chi3_frame_to_bb.compose(chi4_frame_to_frame)

    all_frames_to_bb = Rigid.cat(
        [
            all_frames[..., :5],
            chi2_frame_to_bb.unsqueeze(-1),
            chi3_frame_to_bb.unsqueeze(-1),
            chi4_frame_to_bb.unsqueeze(-1),
        ],
        dim=-1,
    )

    all_frames_to_global = r[..., None].compose(all_frames_to_bb)
    # print("[Torsion to Frames] End")

    return all_frames_to_global


def atom37_to_torsion_angles(
    protein,
    prefix="",
):
    """
    Convert coordinates to torsion angles.
    This function is extremely sensitive to floating point imprecisions
    and should be run with double precision whenever possible.
    Args:
        Dict containing:
            * (prefix)aatype:
                [*, N_res] residue indices
            * (prefix)all_atom_positions:
                [*, N_res, 37, 3] atom positions (in atom37
                format)
            * (prefix)all_atom_mask:
                [*, N_res, 37] atom position mask
    Returns:
        The same dictionary updated with the following features:
        "(prefix)torsion_angles_sin_cos" ([*, N_res, 7, 2])
            Torsion angles
        "(prefix)alt_torsion_angles_sin_cos" ([*, N_res, 7, 2])
            Alternate torsion angles (accounting for 180-degree symmetry)
        "(prefix)torsion_angles_mask" ([*, N_res, 7])
            Torsion angles mask
    """
    device = "cpu" if protein[prefix + "aatype"].device.type == "cpu" else "cuda"
    with disable_tf32(), torch.autocast(device_type=device, enabled=False):
        N, CA, C, O = [pc.ALL_ATOM_POSNS[x] for x in "N,CA,C,O".split(",")]
        aatype = protein[prefix + "aatype"]
        assert torch.max(aatype) < 20, f"{torch.max(aatype)}"
        all_atom_positions = protein[prefix + "all_atom_positions"]
        all_atom_mask = protein[prefix + "all_atom_mask"]

        aatype = torch.clamp(aatype, max=20)

        pad = all_atom_positions.new_zeros([*all_atom_positions.shape[:-3], 1, 37, 3])
        prev_all_atom_positions = torch.cat([pad, all_atom_positions[..., :-1, :, :]], dim=-3)

        pad = all_atom_mask.new_zeros([*all_atom_mask.shape[:-2], 1, 37])
        prev_all_atom_mask = torch.cat([pad, all_atom_mask[..., :-1, :]], dim=-2)

        pre_omega_atom_pos = torch.cat(
            [prev_all_atom_positions[..., [CA, C], :], all_atom_positions[..., [N, CA], :]],
            dim=-2,
        )
        phi_atom_pos = torch.cat(
            [prev_all_atom_positions[..., [C], :], all_atom_positions[..., [N, CA, C], :]],
            dim=-2,
        )
        psi_atom_pos = torch.cat(
            [all_atom_positions[..., [N, CA, C], :], all_atom_positions[..., [O], :]],
            dim=-2,
        )

        pre_omega_mask = torch.prod(prev_all_atom_mask[..., [CA, C]], dim=-1) * torch.prod(
            all_atom_mask[..., [N, CA]], dim=-1
        )
        phi_mask = prev_all_atom_mask[..., C] * torch.prod(
            all_atom_mask[..., [N, CA, C]], dim=-1, dtype=all_atom_mask.dtype
        )
        psi_mask = torch.prod(all_atom_mask[..., [N, CA, C]], dim=-1, dtype=all_atom_mask.dtype) * all_atom_mask[..., O]

        chi_atom_indices = torch.as_tensor(get_chi_atom_indices(), device=aatype.device)

        atom_indices = chi_atom_indices[..., aatype, :, :]
        chis_atom_pos = batched_gather(all_atom_positions, atom_indices, -2, len(atom_indices.shape[:-2]))

        chi_angles_mask = pc.CHI_ANGLES_MASK_LIST
        chi_angles_mask.append([0.0, 0.0, 0.0, 0.0])
        chi_angles_mask = all_atom_mask.new_tensor(chi_angles_mask)

        chis_mask = chi_angles_mask[aatype, :]

        chi_angle_atoms_mask = batched_gather(
            all_atom_mask,
            atom_indices,
            dim=-1,
            no_batch_dims=len(atom_indices.shape[:-2]),
        )
        chi_angle_atoms_mask = torch.prod(chi_angle_atoms_mask, dim=-1, dtype=chi_angle_atoms_mask.dtype)
        chis_mask = chis_mask * chi_angle_atoms_mask

        torsions_atom_pos = torch.cat(
            [
                pre_omega_atom_pos[..., None, :, :],
                phi_atom_pos[..., None, :, :],
                psi_atom_pos[..., None, :, :],
                chis_atom_pos,
            ],
            dim=-3,
        )

        torsion_angles_mask = torch.cat(
            [
                pre_omega_mask[..., None],
                phi_mask[..., None],
                psi_mask[..., None],
                chis_mask,
            ],
            dim=-1,
        )
        torsion_frames = Rigid.from_3_points(
            torsions_atom_pos[..., 1, :],
            torsions_atom_pos[..., 2, :],
            torsions_atom_pos[..., 0, :],
            eps=1e-8,
        )
        fourth_atom_rel_pos = torsion_frames.invert().apply(torsions_atom_pos[..., 3, :])

        torsion_angles_sin_cos = torch.stack([fourth_atom_rel_pos[..., 2], fourth_atom_rel_pos[..., 1]], dim=-1)

        denom = torch.sqrt(
            torch.sum(
                torch.square(torsion_angles_sin_cos),
                dim=-1,
                dtype=torsion_angles_sin_cos.dtype,
                keepdims=True,
            )
            + 1e-10
        )
        torsion_angles_sin_cos = torsion_angles_sin_cos / denom

        # torsion_angles_sin_cos = (
        #    torsion_angles_sin_cos
        #    * all_atom_mask.new_tensor(
        #        [1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0],
        #    )[((None,) * len(torsion_angles_sin_cos.shape[:-2])) + (slice(None), None)]
        # )
        # print(torsion_angles_sin_cos.shape)
        # rigid group for BB-Oxygen must be flipped. Above has a bug
        torsion_angles_sin_cos[..., 2, :] = torsion_angles_sin_cos[..., 2, :] * -1

        chi_is_ambiguous = torsion_angles_sin_cos.new_tensor(
            pc.CHI_PI_PERIODIC_LIST,
        )[aatype, ...]

        mirror_torsion_angles = torch.cat(
            [
                all_atom_mask.new_ones(*aatype.shape, 3),
                1.0 - 2.0 * chi_is_ambiguous,
            ],
            dim=-1,
        )

        alt_torsion_angles_sin_cos = torsion_angles_sin_cos * mirror_torsion_angles[..., None]

    torsion_angles_sin_cos[..., 2, :] = torsion_angles_sin_cos[..., 2, :] * -1
    protein[prefix + "torsion_angles_sin_cos"] = torsion_angles_sin_cos
    protein[prefix + "alt_torsion_angles_sin_cos"] = alt_torsion_angles_sin_cos
    protein[prefix + "torsion_angles_mask"] = torsion_angles_mask
    return protein


def get_chi_atom_indices():
    """Returns atom indices needed to compute chi angles for all residue types.
    Returns:
      A tensor of shape [residue_types=21, chis=4, atoms=4]. The residue types are
      in the order specified in rc.restypes + unknown residue type
      at the end. For chi angles which are not defined on the residue, the
      positions indices are by default set to 0.
    """
    chi_atom_indices = []
    for residue_name in pc.RES_TYPES:
        residue_name = pc.ONE_TO_THREE[residue_name]
        residue_chi_groups = res_to_chi_atom_groups[residue_name]
        atom_indices = []
        for chi_angle in residue_chi_groups:
            atom_indices.append([pc.ALL_ATOM_POSNS[atom] for atom in chi_angle])
        for _ in range(4 - len(atom_indices)):
            atom_indices.append([0, 0, 0, 0])  # For chi angles not defined on the AA.
        chi_atom_indices.append(atom_indices)
    chi_atom_indices.append([[0, 0, 0, 0]] * 4)  # For UNKNOWN residue.
    return chi_atom_indices


def get_chi_angles_and_mask(protein: Dict):
    dtype = protein["all_atom_mask"].dtype
    protein["chi_angles_sin_cos"] = (protein["torsion_angles_sin_cos"][..., 3:, :]).to(dtype)
    protein["chi_mask"] = protein["torsion_angles_mask"][..., 3:].to(dtype)
    return protein["chi_angles_sin_cos"], protein["chi_mask"]


def batched_gather(data, inds, dim=0, no_batch_dims=0):
    ranges = []
    for i, s in enumerate(data.shape[:no_batch_dims]):
        r = torch.arange(s)
        r = r.view(*(*((1,) * i), -1, *((1,) * (len(inds.shape) - i - 1))))
        ranges.append(r)

    remaining_dims = [slice(None) for _ in range(len(data.shape) - no_batch_dims)]
    remaining_dims[dim - no_batch_dims if dim >= 0 else dim] = inds
    ranges.extend(remaining_dims)
    return data[ranges]
