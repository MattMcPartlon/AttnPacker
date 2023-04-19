"""Utility classes and functions for feature representations
"""
from __future__ import annotations

from typing import Optional, List, Tuple, Dict, Any, Union

import torch
from einops import rearrange  # noqa
from protein_learning.common.transforms import (  # noqa
    quaternion_multiply,  # noqa
    quaternion_to_matrix,  # noqa
    matrix_to_quaternion,  # noqa
    quaternion_invert,  # noqa
    quaternion_apply,  # noqa
)
from torch import Tensor

from protein_learning.common.helpers import exists, coords_to_rel_coords, safe_normalize
from protein_learning.common.protein_constants import AA_INDEX_MAP
from protein_learning.features.feature import Feature
from protein_learning.features.feature_config import FeatureTy, FeatureName
from protein_learning.features.input_features import PI
from protein_learning.protein_utils.dihedral.orientation_utils import get_bb_dihedral, get_tr_rosetta_orientation_mats
from protein_learning.protein_utils.sidechains.sidechain_rigid_utils import atom37_to_torsion_angles


def string_encode(mapping: Dict[str, int], *x, device: Any = "cpu") -> Tensor:
    """Encodes a string (or list of strings) according to the given mapping
    :param x: string(s) to encode
    :param mapping: map from string to integer defining encoding
    :param device: device to place output tensor on
    :return: encoded strings accordint to given mapping
    """
    assert all([len(el) == len(x[0]) for el in x]), "ERROR: all strings must have same length"
    out = torch.tensor([[mapping[pos] for pos in el] for el in x], device=device)
    return out[0] if len(x) == 1 else out


def fourier_encode(x: Tensor, num_encodings=4, include_self=True) -> Tensor:
    """Applies fourier encoding (sin + cos scaled by freq.) to input x

    :param x: tensor to apply encoding to
    :param num_encodings: number of frequencies to encode for (1,...1/2**(num_encodings-1))
    :param include_self: whether to append x[...-1] to encodings
    :return: fourier encoding of x
    """
    trailing_one = x.shape[-1] == 1
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x
    scales = 2 ** torch.arange(num_encodings, device=device, dtype=dtype)
    x = x / scales
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim=-1) if include_self else x
    return x.squeeze(-2) if trailing_one else x


def bin_encode(data: Tensor, bins: Tensor):
    """Assigns each value in data to
    :param data: the data to apply bin encoding to
    :param bins: description of bin positions to encode into
        [(bins[i],bins[i+1])] is used to define each position.
    :return: bin index of each value in input data
    """
    assert torch.min(data) >= bins[0] and torch.max(data) < bins[-1], (
        f"incorrect bins, got min/max of data: ({torch.min(data)},{torch.max(data)})\n"
        f"but bin min/max = ({bins[0]},{bins[-1]}])"
    )
    binned_data = -torch.ones_like(data)
    for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
        mask = torch.logical_and(data >= low, data < high)  # noqa
        binned_data[mask] = i
    return binned_data.long()


def res_ty_encoding(
    seq: str,
    corrupt_prob: float = 0,
) -> Feature:
    """Encodes sequence either as a Tensor of ints.
    :param seq: sequence to encode
    :param corrupt_prob: probability with which to corrups sequence labels
    :return: encoded sequence.
    """
    seq_emb = string_encode(AA_INDEX_MAP, seq)
    if corrupt_prob > 0:
        corrupt_mask = torch.rand(len(seq)) < corrupt_prob
        corrupt_aas = torch.randint(0, 20, size=(len(corrupt_mask[corrupt_mask]),))  # noqa
        seq_emb[corrupt_mask] = corrupt_aas

    return Feature(
        raw_data=seq,
        encoded_data=seq_emb.unsqueeze(-1),
        name=FeatureName.RES_TY.value,
        dtype=torch.long,
        ty=FeatureTy.RESIDUE,
        n_classes=len(AA_INDEX_MAP),
    )


def rel_pos_encoding(res_ids: Union[Tensor, List[Tensor]], n_classes: int = 10) -> Feature:
    """Encodes each residue position based on the relative position in the sequence."""
    res_ids = [res_ids] if torch.is_tensor(res_ids) else res_ids
    assert all([torch.all(res_ids[i] >= 0) for i in range(len(res_ids))]), f"{res_ids}"  # noqa
    encs = []
    for ids in res_ids:
        max_posn, _ = torch.max(ids, dim=-1, keepdim=True)
        encs.append(torch.floor((ids.float() * n_classes) / (max_posn + 1)).long())
    rel_pos_enc = torch.cat(encs, dim=-1)
    assert torch.all(rel_pos_enc >= 0)  # noqa
    return Feature(
        raw_data=torch.cat(res_ids, dim=-1).unsqueeze(-1),
        encoded_data=rel_pos_enc.unsqueeze(-1),
        name=FeatureName.REL_POS.value,
        dtype=torch.long,
        ty=FeatureTy.RESIDUE,
        n_classes=n_classes,
    )


def bb_dihedral_encoding(
    bb_coords: Optional[List[Tensor]] = None,
    n_classes: int = 36,
    encode: bool = True,
    bb_dihedrals: Optional[Tuple[Tensor, ...]] = None,
) -> Feature:
    """BB DIhedral Features (encoded or raw)"""
    assert exists(bb_dihedrals) or exists(bb_coords)
    phi, psi, omega = bb_dihedrals if exists(bb_dihedrals) else get_bb_dihedral(*bb_coords)
    bb_dihedrals = torch.cat([x.unsqueeze(-1) for x in (phi, psi, omega)], dim=-1)
    encoded_bb_dihedrals = None
    if encode:
        encoded_bb_dihedrals = torch.clamp(((bb_dihedrals / PI) + 1) / 2, 0, 1) * (n_classes - 1)
    return Feature(
        raw_data=bb_dihedrals,
        encoded_data=encoded_bb_dihedrals,
        name=FeatureName.BB_DIHEDRAL.value,
        dtype=torch.long,
        ty=FeatureTy.RESIDUE,
        n_classes=n_classes,
    )


def degree_centrality_encoding(
    coords: Tensor,
    chain_indices: List[Tensor],
    n_classes: int = 6,
    max_radius: float = 12,
    bounds: Tuple[int, int] = (6, 30),
) -> Feature:
    """Residue degree centrality features"""
    assert coords.ndim == 2
    cens, normed_cens = torch.zeros(coords.shape[0]), torch.zeros(coords.shape[0])
    for idxs in chain_indices:
        dists = torch.cdist(coords[idxs], coords[idxs])
        cmin, cmax = bounds
        res_centrality = torch.sum(dists <= max_radius, dim=-1) - 1  # noqa
        clamped_centrality = torch.clamp(res_centrality, cmin + 1, cmax) - cmin
        norm_res_centrality = clamped_centrality / (cmax - cmin)
        # fill in for chain
        cens[idxs] = res_centrality.float()
        normed_cens[idxs] = norm_res_centrality.float()

    binned_res_centrality = normed_cens * (n_classes - 1)
    assert torch.all(binned_res_centrality >= 0)
    return Feature(
        raw_data=cens.unsqueeze(-1),
        encoded_data=binned_res_centrality.unsqueeze(-1).long(),
        name=FeatureName.CENTRALITY.value,
        dtype=torch.long,
        ty=FeatureTy.RESIDUE,
        n_classes=n_classes,
    )


def rel_sep_encoding(res_ids: Union[Tensor, List[Tensor]], sep_bins: List) -> Feature:
    """Relative Separation Encoding"""
    res_ids = [res_ids] if torch.is_tensor(res_ids) else res_ids
    res_posns = torch.cat(res_ids, dim=-1)
    sep_mat = rearrange(res_posns, "n -> () n ()") - rearrange(res_posns, "n -> n () ()")  # noqa
    enc_sep_mat = bin_encode(sep_mat, bins=torch.tensor(sep_bins))
    assert torch.all(enc_sep_mat >= 0)
    return Feature(
        encoded_data=enc_sep_mat,
        raw_data=sep_mat,
        name=FeatureName.REL_SEP.value,
        dtype=torch.long,
        ty=FeatureTy.PAIR,
        n_classes=len(sep_bins),
    )


def rel_dist_encoding(
    rel_dists: Tensor,
    dist_bounds=(2.5, 16.5),
    n_classes=32,
) -> Feature:
    """Relative Distance Encoding"""
    min_dist, max_dist = dist_bounds
    normed_dists = (rel_dists - min_dist) / (max_dist - min_dist)
    dist_bins = torch.clamp(normed_dists, 0, 1) * (n_classes - 1)
    return Feature(
        raw_data=rel_dists,
        encoded_data=dist_bins,
        name=FeatureName.REL_DIST.value,
        dtype=torch.long,
        ty=FeatureTy.PAIR,
        n_classes=n_classes,
    )


def tr_rosetta_ori_encoding(
    bb_coords: List[Tensor] = None,
    n_classes: int = 36,
    encode: bool = True,
    tr_angles: Optional[Tuple[Tensor, ...]] = None,
) -> Feature:
    """trRosetta dihedral features"""
    phi, psi, omega = get_tr_rosetta_orientation_mats(*bb_coords) if not exists(tr_angles) else tr_angles
    ori_feats = torch.cat([x.unsqueeze(-1) for x in (phi, psi, omega)], dim=-1)
    encoded_ori_feats = None
    if encode:
        normed_ori = torch.clamp(((ori_feats / PI) + 1) / 2, 0, 1)
        encoded_ori_feats = normed_ori * (n_classes - 1)
    return Feature(
        raw_data=ori_feats,
        encoded_data=encoded_ori_feats,
        name=FeatureName.TR_ORI.value,
        dtype=torch.long if encode else torch.float32,
        ty=FeatureTy.PAIR,
        n_classes=n_classes,
    )


def rel_ori_encoding(quats: Tensor) -> Feature:
    """Invariant relative orientation encoding (as quaternions)"""
    rquats = rearrange(quats, "n i -> () n i")
    rquats_inv = quaternion_invert(rearrange(quats, "n i -> n () i"))

    return Feature(
        raw_data=quaternion_multiply(rquats_inv, rquats),
        encoded_data=None,
        name=FeatureName.REL_ORI.value,
        ty=FeatureTy.PAIR,
        n_classes=None,
        raw_mask_value=torch.tensor([1, 0, 0, 0]).float(),
    )


def local_rel_coords(ca_coords: Tensor, quats: Tensor) -> Feature:
    """invariant encoding of residue relative coordinates"""
    quat_inv = rearrange(quaternion_invert(quats), "n i -> n () i")
    rel_coords = coords_to_rel_coords(ca_coords)
    inv_rel_coords = quaternion_apply(quat_inv, rel_coords)
    return Feature(
        raw_data=inv_rel_coords,
        encoded_data=None,
        name=FeatureName.REL_COORD.value,
        ty=FeatureTy.PAIR,
        n_classes=None,
        raw_mask_value=torch.zeros(3).float(),
    )


def rel_chain_encoding(chain_ids) -> Feature:
    """Encode relative chain"""
    diffs = rearrange(chain_ids, "i -> i () ()") - rearrange(chain_ids, "i -> () i ()")  # noqa
    clamped_diffs = (2 + torch.clamp(diffs, min=-1, max=1)).long()
    return Feature(
        raw_data=diffs,
        encoded_data=clamped_diffs,
        name=FeatureName.REL_CHAIN.value,
        ty=FeatureTy.PAIR,
        n_classes=5,
    )


def extra_encoding(extra: Tensor, ty: FeatureTy) -> Feature:
    """Encode flags"""
    return Feature(
        raw_data=extra.float(),
        encoded_data=extra.float(),
        name=FeatureName.EXTRA_RES.value if ty == FeatureTy.RESIDUE else FeatureName.EXTRA_PAIR.value,
        ty=ty,
        n_classes=extra.shape[-1],
    )


def sc_dihedral_encoding(coords, mask, sequence_enc):
    coords, mask, sequence_enc = map(lambda x: x.unsqueeze(0) if x.shape[0] > 1 else x, (coords, mask, sequence_enc))
    ptn = dict(aatype=sequence_enc, all_atom_positions=coords, all_atom_mask=mask)
    torsion_info = atom37_to_torsion_angles(ptn)
    sin_cos = safe_normalize(torsion_info["torsion_angles_sin_cos"].detach()).squeeze(0)
    theta = torch.atan2(*sin_cos.unbind(-1))
    sin_cos = rearrange(sin_cos, "n d a -> n (d a)")
    data = torch.cat((sin_cos, theta), dim=-1).detach()
    assert data.shape[-1] == 21
    return Feature(
        raw_data=data.float(),
        encoded_data=data.float(),
        name=FeatureName.SC_DIHEDRAL.value,
        ty=FeatureTy.RESIDUE,
        n_classes=data.shape[-1],
        raw_mask_value=torch.zeros(21),
    )
