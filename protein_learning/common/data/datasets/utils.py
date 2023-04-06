import random
from typing import Tuple, List

import numpy as np
import torch
import torch.nn.functional as F  # noqa
from einops import rearrange  # noqa
from torch import Tensor

from protein_learning.common.data.data_types.protein import Protein
from protein_learning.networks.loss.coord_loss import CoordDeviationLoss
from protein_learning.networks.loss.coord_loss import TMLoss
from protein_learning.common.helpers import exists
from protein_learning.common.protein_constants import AA_TO_SC_ATOMS, BB_ATOMS, NAT_AA_SET


def fill_atom_masks(protein: Protein, overwrite: bool = False) -> Protein:
    seq, atom_tys = protein.seq, protein.atom_tys
    bb_atom_set = set(BB_ATOMS)
    mask = torch.ones(len(seq), len(atom_tys))
    for i, a in enumerate(atom_tys):
        if a in bb_atom_set:
            continue
        for idx, s in enumerate(seq):
            if s not in NAT_AA_SET or a not in AA_TO_SC_ATOMS[s]:
                mask[idx, i] = 0
    if not overwrite:
        protein.atom_masks = protein.atom_masks & mask.bool()
    else:
        protein.atom_masks = mask.bool()
    return protein


def fill_missing_coords(protein: Protein) -> Protein:
    ca_coords = protein["CA"]
    atom_masks = protein.atom_masks
    for i in range(len(protein.atom_tys)):
        assert atom_masks[:, i].ndim == 1
        msk = ~atom_masks[:, i]
        protein.atom_coords[msk, i] = ca_coords[msk]
    return protein


def set_canonical_coords_n_masks(protein: Protein, overwrite: bool = False):
    return fill_missing_coords(fill_atom_masks(protein, overwrite=overwrite))


def window_sum(x, w):
    assert x.shape[-1] >= w, f"shape:{x.shape}, w:{w}"
    return x.unfold(0, w, 1).sum(dim=-1)


def get_contiguous_crop(crop_len: int, n_res: int) -> Tuple[int, int]:
    """Get a contiguous interval to crop"""
    if crop_len < 0:
        return 0, n_res
    start, end = 0, n_res
    start = random.randint(0, (end - crop_len)) if end > crop_len else start
    return start, min(end, start + crop_len)


def get_dimer_crop_lens(partition: List[Tensor], crop_len: int, min_len=100) -> Tuple[int, int]:
    assert len(partition) == 2, f"{len(partition)}"
    l1, l2 = len(partition[0]), len(partition[1])
    if l1 + l2 <= crop_len or crop_len < 0:
        c1, c2 = l1, l2
    elif random.randint(0, 1) == 1:
        end = min(l1, max(crop_len - l2, min_len))
        c1 = random.randint(max(crop_len - l2, min(l1, min_len)), end)
        c2 = min(max(crop_len - c1, min_len), l2)
    else:
        end = min(l2, max(crop_len - l1, min_len))
        c2 = random.randint(max(crop_len - l1, min(l2, min_len)), end)
        c1 = min(max(crop_len - c2, min_len), l1)
    return c1, c2


def get_dimer_crop(partition: List[Tensor], crop_len: int, min_len=100) -> List[Tensor]:
    c1, c2 = get_dimer_crop_lens(partition, crop_len, min_len=min_len)
    mn1, mx1 = get_contiguous_crop(c1, len(partition[0]))
    mn2, mx2 = get_contiguous_crop(c2, len(partition[2]))
    return [partition[0][mn1:mx1], partition[1][mn2:mx2]]


def get_dimer_spatial_crop(partition, coords: List[Tensor], crop_len, min_len: int = 100, sigma=4) -> List[Tensor]:
    assert len(coords) == 2 and coords[0].ndim == 2
    l1, l2 = get_dimer_crop_lens(partition, crop_len=crop_len, min_len=min_len)
    scores = 1 / (1 + torch.square(torch.cdist(coords[0], coords[1]) / sigma))
    s1 = scores.sum(dim=1)
    s1 = window_sum(torch.exp(sigma * s1 / torch.max(s1)), w=l1).numpy()
    c1 = np.random.choice(len(s1), p=s1 / np.sum(s1))
    s2 = torch.sum(scores[c1 : c1 + l1, :], dim=0)
    s2 = window_sum(torch.exp(sigma * s2 / torch.max(s2)), w=l2).numpy()
    c2 = np.random.choice(len(s2), p=s2 / np.sum(s2))
    return [partition[0][c1 : c1 + l1], partition[1][c2 : c2 + l2]]


def get_ab_ag_spatial_crop(ag_ab: Protein, crop_len: int):
    crop_len = min(crop_len, len(ag_ab))
    ca_coords = ag_ab["CA"]
    ab_coords = ca_coords[ag_ab.chain_indices[0]]
    ag_coords = ca_coords[ag_ab.chain_indices[1]]
    ab_len, ag_len = map(len, ag_ab.chain_indices)
    ag_cropped_size = max(10, crop_len - ab_len)
    nearest_res, _ = torch.min(torch.cdist(ab_coords, ag_coords), dim=0)
    nearest_res[nearest_res < 8] = 1
    nearest_res[nearest_res >= 8] = 0
    scores = window_sum(nearest_res, w=ag_cropped_size).numpy()
    scores = np.maximum(1, scores) * (len(scores) ** (-1 / 2))
    scores = np.exp(scores)
    c2 = np.random.choice(len(scores), p=scores / np.sum(scores))
    return [ag_ab.chain_indices[0], ag_ab.chain_indices[1][c2 : c2 + ag_cropped_size]]


def is_homodimer(chain_1: Protein, chain_2: Protein, tol=2) -> bool:
    if len(chain_1) != len(chain_2):
        return False
    return get_rmsd(chain_1, chain_2) < tol


def get_tm(a: Protein, b: Protein) -> float:
    mask = a.valid_residue_mask & b.valid_residue_mask
    return (
        -TMLoss()
        .forward(
            a["CA"][mask].clone().unsqueeze(0),
            b["CA"][mask].clone().unsqueeze(0),
            align=True,
            reduce=True,
        )
        .item()
    )


def get_rmsd(a: Protein, b: Protein) -> float:
    mask = a.valid_residue_mask & b.valid_residue_mask
    ca1, ca2 = map(lambda x: rearrange(x["CA"][mask], "n c -> () n () c"), (a, b))
    loss = CoordDeviationLoss()(ca1, ca2, coord_mask=None, align=True)
    return loss.mean().item()


def restrict_protein_to_aligned_residues(a: Protein, b: Protein):
    valid_res_mask_a = torch.any(a.atom_masks, dim=-1)
    valid_res_mask_b = torch.any(b.atom_masks, dim=-1)
    valid_mask = valid_res_mask_a & valid_res_mask_b
    valid_indices = torch.arange(valid_mask.numel())[valid_mask]
    a, b = map(lambda x: _restrict_to_indices(x, valid_indices), (a, b))
    return a, b


def _restrict_to_indices(a: Protein, idxs: Tensor) -> Protein:
    sec_struct = None
    if exists(a.sec_struct):
        sec_struct = "".join([a.sec_struct[i] for i in idxs])
    return Protein(
        atom_coords=a.atom_coords[idxs],
        atom_masks=a.atom_masks[idxs],
        atom_tys=a.atom_tys,
        seq="".join([a.seq[int(i)] for i in idxs]),
        name=a._name,
        res_ids=[a.res_ids[0][idxs]],
        sec_struct=sec_struct,
    )
