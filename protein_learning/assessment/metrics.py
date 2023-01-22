"""Functions for computing scoring metrics on proteins
"""
import math
from itertools import combinations
from math import pi as PI  # noqa
from typing import Optional, List, Tuple, Any

import torch
from einops import repeat, rearrange  # noqa
from torch import Tensor

from protein_learning.common.helpers import (
    get_eps,
    calc_tm_torch,
    default,
    masked_mean,
    exists,
    safe_norm

)
from protein_learning.common.protein_constants import (
    AA_INDEX_MAP,
    ALL_ATOM_POSNS,
)
from protein_learning.common.rigids import Rigids
from protein_learning.features.masking.masking_utils import get_chain_masks, get_partition_mask
from protein_learning.protein_utils.align.kabsch_align import kabsch_align


def count_true(x: Tensor) -> int:
    """count the number of entries in x that are equal to True"""
    return x[x].numel()


def tensor_to_list(x: Tensor) -> List:
    """Convert torch tensor to python list"""
    return x.detach().cpu().numpy().tolist()


def tensor_to_array(x: Tensor) -> List:
    """Convert torch tensor to numpy array"""
    return x.detach().cpu().numpy()


def get_sep_mask(n: int, min_sep: int, max_sep: int, device: Any) -> Tensor:
    """Get separation mask"""
    rel_sep = torch.abs(repeat(torch.arange(n, device=device), "i -> () i ()") -
                        repeat(torch.arange(n, device=device), "i -> () () i"))
    # compute separation mask
    max_sep = max_sep if max_sep > 0 else n
    return torch.logical_and(rel_sep >= min_sep, rel_sep <= max_sep)  # noqa


def batch_coords(predicted_coords: Tensor, actual_coords: Tensor, batched_len: int):
    """(potentially) adds batch dimension to coordinates and returns whether
    coordinates already had a batch dimension
    """
    batched = predicted_coords.ndim == batched_len
    actual = actual_coords if actual_coords.ndim == batched_len else actual_coords.unsqueeze(0)
    pred = predicted_coords if predicted_coords.ndim == batched_len else predicted_coords.unsqueeze(0)
    assert actual.ndim == pred.ndim == batched_len, f"{actual.shape}, {pred.shape}, {batched_len}"
    return batched, pred, actual


def compute_coord_lddt(
        predicted_coords: Tensor,
        actual_coords: Tensor,
        cutoff: float = 15.,
        per_residue: bool = True,
        pred_rigids: Optional[Rigids] = None,
        pair_mask: Optional[Tensor] = None,
        thresholds = None,

) -> Tensor:
    """Computes LDDT of predicted and actual coords.

    If rigids are provided, the pLDDT will be taken w.r.t the local frame
    of the input.

    :param pred_rigids:
    :param predicted_coords: tensor of shape (b, n, 3) or (n,3)
    :param actual_coords: tensor of shape (b, n, 3) or (n,3)
    :param cutoff: LDDT cutoff value
    :param per_residue: whether to compute LDDT per-residue or for all coords.
    :return: LDDT or pLDDT tensor
    """
    thresholds = default(thresholds,[0.5,1,2,4]) # plddt cutoff thresholds
    # reshape so that each set of coords has batch dimension
    batched, actual_coords, predicted_coords = batch_coords(
        predicted_coords=predicted_coords,
        actual_coords=actual_coords,
        batched_len=3
    )
    n = predicted_coords.shape[1]
    actual_dists = torch.cdist(actual_coords, actual_coords)
    if not exists(pred_rigids):
        pred_dists = torch.cdist(predicted_coords, predicted_coords)
    else:
        rel_coords = pred_rigids.apply_inverse(rearrange(predicted_coords, "b n c -> b () n c"))
        pred_dists = safe_norm(rel_coords, dim=-1)

    not_self = (1 - torch.eye(n, device=pred_dists.device)).bool()
    mask = torch.logical_and(pred_dists < cutoff, not_self).float()  # noqa
    if exists(pair_mask):
        mask = pair_mask.float() * mask
    l1_dists = torch.abs(pred_dists - actual_dists).detach()

    scores = (1/len(thresholds))*sum([(l1_dists < t).float() for t in thresholds])

    dims = (1, 2) if not per_residue else (2,)
    eps = get_eps(l1_dists)
    scale = 1 / (eps + torch.sum(mask, dim=dims))
    scores = eps + torch.sum(scores * mask, dim=dims)
    return scale * scores if batched else (scale * scores)[0]


def compute_interface_tm(
        predicted_coords: Tensor,
        actual_coords: Tensor,
        pred_rigids: Rigids,
        actual_rigids: Rigids,
        chain_indices: List[Tensor],
        normalize: bool = False,
        reduce: bool = False,
):
    """Interface TM score (Unnormalized and not reduced)"""
    tm_scale = lambda n: 0.5 if n <= 15 else 1.24 * ((n - 15.0) ** (1. / 3.)) - 1.8

    assert predicted_coords.ndim == actual_coords[0].ndim
    shape = predicted_coords.shape
    assert predicted_coords.ndim == 3, f"input shape should be (b,n,3), got {shape}"
    _, pair_masks, inter_pair_mask = get_chain_masks(n_res=actual_coords.shape[0], chain_indices=chain_indices)

    predicted_coords, actual_coords = map(lambda x: rearrange(x, "b n c -> b () n c"),
                                          (predicted_coords, actual_coords))
    pred_rel_coords = pred_rigids.apply_inverse(predicted_coords)
    actual_rel_coords = actual_rigids.apply_inverse(actual_coords)

    raise Exception("Not implemented")


def get_inter_chain_contacts(
        coords: Tensor,
        partition: List[Tensor],
        atom_mask: Optional[Tensor],
        contact_thresh: float = 12,

) -> Tensor:
    """Get contact flags for pair features"""
    contacts = torch.cdist(coords.squeeze(), coords.squeeze())
    part_mask = get_partition_mask(n_res=sum([len(p) for p in partition]), partition=partition)
    part_mask = part_mask.to(contacts.device)
    contacts[~part_mask] = contact_thresh + 1
    pair_mask = torch.einsum("i,j->ij", atom_mask, atom_mask)
    contacts[~pair_mask] = contact_thresh + 1
    return contacts < contact_thresh  # noqa


def compute_interface_rmsd(
        predicted_coords: Tensor,
        actual_coords: Tensor,
        chain_indices: List[Tensor],
        atom_mask: Optional[Tensor],
        contact_thresh: float = 10,
        align: bool = True,
):
    assert len(chain_indices) == 2
    assert predicted_coords.ndim == actual_coords.ndim == 2
    contacts = get_inter_chain_contacts(
        actual_coords,
        chain_indices,
        atom_mask=atom_mask,
        contact_thresh=contact_thresh
    )
    interface_mask = torch.any(contacts, dim=-1)
    assert interface_mask.shape == actual_coords.shape[:1]
    if exists(atom_mask):
        assert atom_mask.shape == interface_mask.shape
        interface_mask = interface_mask & atom_mask

    interface_coords_actual = actual_coords[interface_mask]
    interface_coords_pred = predicted_coords[interface_mask]
    atom_mask = torch.ones(interface_coords_actual.shape[0])
    return compute_coord_rmsd(
        predicted_coords=interface_coords_pred,
        actual_coords=interface_coords_actual,
        atom_mask=atom_mask.to(interface_mask.device).bool(),
        per_res=False,
        align=align,
    )


def compute_coord_tm(
        predicted_coords: Tensor,
        actual_coords: Tensor,
        norm_len: Optional[int] = None,
        align: bool = True
) -> Tensor:
    """Compute TM-Score of predicted and actual coordinates

    shape should be (b,n,3) or (n,3)
    """
    assert predicted_coords.ndim <= 3
    # reshape so that each set of coords has batch dimension
    batched, actual_coords, predicted_coords = batch_coords(
        predicted_coords=predicted_coords,
        actual_coords=actual_coords,
        batched_len=3
    )
    if align:
        _, actual_coords = kabsch_align(align_to=predicted_coords, align_from=actual_coords)
    deviations = torch.norm(predicted_coords - actual_coords, dim=-1)
    norm_len = default(norm_len, predicted_coords.shape[1])
    tm = calc_tm_torch(deviations, norm_len=norm_len)
    return tm


def mean_aligned_error(
        predicted_coords: Tensor,
        actual_coords: Tensor,
        mask: Optional[Tensor],
        per_residue: bool,
        fn=lambda x: torch.square(x),
        align: bool = True,
):
    """mean per-residue error w.r.t given function"""
    # reshape so that each set of coords has batch dimension
    batched, actual_coords, predicted_coords = batch_coords(
        predicted_coords=predicted_coords,
        actual_coords=actual_coords,
        batched_len=4 if per_residue else 3
    )
    if exists(mask):
        mask = mask if batched else mask.unsqueeze(0)
        assert mask.ndim == actual_coords.ndim - 1

    if align:
        _, actual_coords = kabsch_align(align_to=predicted_coords, align_from=actual_coords, mask=mask)

    tmp = torch.sum(fn(predicted_coords - actual_coords), dim=-1)
    mean_error = masked_mean(tmp, mask, dim=-1)
    return mean_error


def compute_coord_rmsd(
        predicted_coords: Tensor,
        actual_coords: Tensor,
        atom_mask: Optional[Tensor] = None,
        per_res: bool = False,
        align: bool = True,
) -> Tensor:
    """Computes RMSD between predicted and actual coordinates

    :param predicted_coords: tensor of shape (...,n,a,3) if per_res
    is specified, otherwise (...,n,3) - where a is number of atom types
    :param actual_coords: tensor of shape (...,n,a,3)  if per_res
    is specified, otherwise (...,n,3) - where a is number of atom types
    :param atom_mask: mask tensor of shape (...,n,a) if per_res is specified
    otherwise (...,n)
    :param per_res: whether to return deviation for each residue,
    or for the entire structure.
    :param align: whether to kabsch align coordinates before computing rmsd
    :return: RMSD
    """
    mse = mean_aligned_error(
        predicted_coords=predicted_coords,
        actual_coords=actual_coords,
        mask=atom_mask,
        fn=torch.square,
        per_residue=per_res,
        align=align,
    )
    return torch.sqrt(mse)


def compute_coord_mae(
        predicted_coords: Tensor,
        actual_coords: Tensor,
        atom_mask: Optional[Tensor],
        per_res: bool = True
) -> Tensor:
    """Computes mean l1 deviatoin between predicted and actual coordinates

    :param predicted_coords: tensor of shape (...,n,a,3) if per_res
    is specified, otherwise (...,n,3) - where a is number of atom types
    :param actual_coords: tensor of shape (...,n,a,3)  if per_res
    is specified, otherwise (...,n,3) - where a is number of atom types
    :param atom_mask: mask tensor of shape (...,n,a) if per_res is specified
    otherwise (...,n)
    :param per_res: whether to return deviation for each residue,
    or for the entire structure.
    :return: Mean l1 coordinate deviation
    """
    # reshape so that each set of coords has batch dimension
    return mean_aligned_error(
        predicted_coords=predicted_coords,
        actual_coords=actual_coords,
        mask=atom_mask,
        fn=lambda x: torch.sqrt(torch.square(x) + get_eps(x)),
        per_residue=per_res
    )


def per_residue_neighbor_counts(atom_coords: Tensor, mask: Optional[Tensor] = None, dist_cutoff=10):
    """Computes the number of coordinates within a dist_cutoff radius of each input coordinate
    :param atom_coords: Tensor of shape (...,n,3).
    :param mask: ignore coordinate i if mask[...,i] is False (optional)
    :param dist_cutoff: cutoff distance for two coordinates to be considered neighbors
    :return: number of neighbors per atom
    """
    batched = atom_coords.ndim == 3
    rel_dists = torch.cdist(atom_coords, atom_coords)
    dist_mask = torch.einsum("... n, ... m-> ... nm", mask, mask) if exists(mask) else \
        torch.ones(1, device=atom_coords.device).bool()
    exclude_self_mask = torch.eye(atom_coords.shape[-2], device=atom_coords.device)
    mask = torch.logical_and(dist_mask, exclude_self_mask.unsqueeze(0) if batched else exclude_self_mask)
    rel_dists[mask] = dist_cutoff + 1
    return torch.sum((rel_dists < dist_cutoff), dim=-1)  # noqa


def compute_angle_mae(source: Tensor, target: Tensor) -> Tensor:
    """computes absolute error between two lists of angles"""
    a = source - target
    a[a > PI] -= 2 * PI
    a[a < -PI] += 2 * PI
    return torch.abs(a)


def calculate_sequence_identity(pred_seq: Tensor, target_seq: Tensor) -> Tensor:
    """Calculate average sequence identity between pred_seq and target_seq"""
    return torch.mean((pred_seq == target_seq).float())


def detect_disulfide_bond_pairs(target_seq: Tensor, target_coords: Tensor) -> List[Tuple[int, int]]:
    """Returns Cysteine pairs forming disulfide bonds"""
    # calculate cystine positions
    cys_posns = torch.arange(target_seq.numel())[target_seq == AA_INDEX_MAP["CYS"]]
    SG = ALL_ATOM_POSNS["SG"]
    is_bond = lambda p1, p2: torch.norm(target_coords[p1, SG] - target_coords[p2, SG]) < 2.5
    return list(filter(lambda x: is_bond(*x), combinations(tensor_to_list(cys_posns), 2)))


def calculate_average_entropy(pred_aa_logits: Tensor):
    """Calculate Average Entropy"""
    log_probs = torch.log_softmax(pred_aa_logits, dim=-1)
    probs = torch.exp(log_probs)
    return torch.mean(torch.sum(-probs * (log_probs * math.log2(math.e)), dim=-1))  # entropy


def calculate_perplexity(pred_aa_logits: Tensor, true_labels: Tensor):
    """Calculate Perplexity"""
    ce = torch.nn.CrossEntropyLoss()
    return torch.exp(ce(pred_aa_logits, true_labels))


def calculate_unnormalized_confusion(pred_labels: Tensor, true_labels: Tensor):
    """Calculate (un-normalized) confusion"""
    pred_one_hot, target_one_hot = map(lambda x: torch.nn.functional.one_hot(x, 21), (pred_labels, true_labels))
    return torch.einsum("n i, n j -> i j", pred_one_hot.float(), target_one_hot.float())


def get_percentage_contacts(
        a: Tensor,
        b: Tensor,
        min_sep: int,
        max_sep: int,
        threshold: float = 8,
        pair_mask: Optional[Tensor] = None,
):
    """Get percentage of contacts in a that are common to b

    :param a: tensor of coordinates having shape (n,3) or (b,n,3)
    :param b: tensor of coordinates having shape (n,3) or (b,n,3)
    :param min_sep: minimum sequence separation to consider
    :param max_sep: maximum sequence separation to consider (use -1 to exclude cutoff)
    :param threshold: distance cutoff threshold for which two residues are considered
    to be in contact (i.e. if d_ij<threshold the pair i,j is in contact).
    :return: The percentage of pairs (i,j) making contacts in structure a
    that also make contact in structure b.
    """
    assert a.shape == b.shape
    batched, a, b = batch_coords(
        predicted_coords=a,
        actual_coords=b,
        batched_len=3,
    )
    n = a.shape[1]
    a_contacts, b_contacts = map(lambda x: torch.cdist(x, x) < threshold, (a, b))
    if exists(pair_mask):
        a_contacts[~pair_mask] = False
        b_contacts[~pair_mask] = False
    sep_mask = get_sep_mask(n, min_sep=min_sep, max_sep=max_sep, device=a.device)
    valid_contacts_a, valid_contacts_b = a_contacts[sep_mask], b_contacts[sep_mask]
    num_contacts_b = count_true(valid_contacts_b)
    num_contacts_common = count_true(valid_contacts_b & valid_contacts_a)
    return num_contacts_common / max(1, num_contacts_b)


def get_contact_recall(
        pred_coords: Tensor,
        actual_coords: Tensor,
        min_sep: int,
        max_sep: int,
        threshold: float = 8,
        pair_mask: Optional[Tensor] = None,
        partition: Optional[List[Tensor]] = None,
):
    """Contact recall (see get_percentage_contacts)

    :param pred_coords: tensor of coordinates having shape (n,3) or (b,n,3)
    :param actual_coords:
    :param min_sep:
    :param max_sep:
    :param threshold:
    :return: contact recall between predicted and native structure
    """
    part_mask = None
    if exists(partition):
        part_mask = get_partition_mask(n_res=sum([len(p) for p in partition]), partition=partition)
        part_mask = part_mask.to(actual_coords.device)
    if exists(pair_mask):
        pair_mask = pair_mask & part_mask if exists(part_mask) else pair_mask

    return get_percentage_contacts(
        pred_coords,
        actual_coords,
        min_sep=min_sep,
        max_sep=max_sep,
        threshold=threshold,
        pair_mask=pair_mask,
    )


def get_contact_precision(
        pred_coords: Tensor,
        actual_coords: Tensor,
        min_sep: int,
        max_sep: int,
        threshold: float = 8,
        pair_mask: Optional[Tensor] = None,
        partition: Optional[List[Tensor]] = None,
):
    """Contact precision (see get_percentage_contacts)

    :param pred_coords: tensor of coordinates having shape (n,3) or (b,n,3)
    :param actual_coords:
    :param min_sep:
    :param max_sep:
    :param threshold:
    :return: contact precision between predicted and native structure
    """
    part_mask = None
    if exists(partition):
        part_mask = get_partition_mask(n_res=sum([len(p) for p in partition]), partition=partition)
        part_mask = part_mask.to(actual_coords.device)
    if exists(pair_mask):
        pair_mask = pair_mask & part_mask if exists(part_mask) else pair_mask

    return get_percentage_contacts(
        actual_coords,
        pred_coords,
        min_sep=min_sep,
        max_sep=max_sep,
        threshold=threshold,
        pair_mask=pair_mask
    )


def calculate_average_l1_dist_difference(
        predicted_coords: Tensor,
        actual_coords: Tensor,
        min_sep: int = 0,
        max_sep: int = -1
) -> Tensor:
    """Average L1-difference between predicted and actual coordinates"""
    _, actual_coords, predicted_coords = batch_coords(
        predicted_coords=predicted_coords,
        actual_coords=actual_coords,
        batched_len=3,
    )
    a, b = map(lambda x: torch.cdist(x, x), (actual_coords, predicted_coords))
    sep_mask = get_sep_mask(a.shape[1], min_sep=min_sep, max_sep=max_sep, device=a.device)

    return torch.mean(torch.abs(a - b)[sep_mask])
