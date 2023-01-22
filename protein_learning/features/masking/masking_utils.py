"""Masked Feature Generation"""
from __future__ import annotations

import random
from typing import Optional, Tuple, Any, List, Callable, Dict

import numpy as np
import torch
from einops import repeat, rearrange  # noqa
from torch import Tensor

from protein_learning.common.helpers import exists, default
from protein_learning.features.feature_config import FeatureName, FeatureTy
from protein_learning.features.input_features import Feature

max_value = lambda x: torch.finfo(x.dtype).max  # noqa

INTER_IGNORE_FEATS = {
    FeatureName.REL_CHAIN,
    FeatureName.EXTRA_PAIR,
    FeatureName.EXTRA_RES
}

INTRA_IGNORE_FEAT_LIST = [
    FeatureName.REL_CHAIN,
    FeatureName.REL_SEP,
    FeatureName.REL_POS,
    FeatureName.EXTRA_RES,
    FeatureName.EXTRA_PAIR
]
INTRA_IGNORE_FEATS = set(INTRA_IGNORE_FEAT_LIST)

INTER_IGNORE_FEAT_NAMES = set([x.value for x in INTER_IGNORE_FEATS])
INTRA_IGNORE_FEAT_NAMES = set([x.value for x in INTRA_IGNORE_FEATS])

norm_weights = lambda x: np.array(x) / sum(np.array(x)) if exists(x) else None
TRUE, FALSE = torch.ones(1).bool(), torch.zeros(1).bool()


def count_true(x: Optional[Tensor]) -> int:
    """counts the number of entries in x[x]"""
    return x[x].numel() if exists(x) else 0


def cast_list(x: Any) -> Optional[List]:
    """Cast x to a list"""
    if isinstance(x, list):
        return x if len(x) > 0 else None
    else:
        return x if exists(x) else None


def cast_tuple(x: Any) -> Optional[Tuple]:
    """Cast x to a tuple"""
    if isinstance(x, tuple):
        return x if len(x) > 0 else None
    else:
        return (x,) if exists(x) else None


def bool_tensor(n, fill=True, posns: Optional[Tensor] = None):
    """Bool tensor of length n, initialized to 'fill' in all given positions
    and ~fill in other positions.
    """
    # create bool tensor initialized to ~fill
    mask = torch.zeros(n).bool() if fill else torch.ones(n).bool()
    mask[posns if exists(posns) else torch.arange(n)] = fill
    return mask


def apply_mask_to_seq(seq: str, mask: Tensor):
    """Modifies the input sequence so that s[i]="-" iff mask[i]"""
    assert mask.ndim == 1
    return "".join(["-" if mask[i] else s for i, s in enumerate(seq)])


def get_mask_len(
        n_res: int,
        min_n_max_p: Tuple[float, float],
        min_n_max_res: Optional[Tuple[int, int]] = None,
) -> int:
    """gets number of residues to mask given
    (1) number of residues in sequence
    (2) minimum/maximum percentage of sequence to mask (min_n_max_p)
    (3) [Optional] upper and lower bounds on number of residues to mask (min_n_max_res)
    """
    min_frac, max_frac = min_n_max_p
    min_res, max_res = default(min_n_max_res, [0, n_res])
    min_res = int(max(min_res, n_res * min_frac, 0))
    max_res = int(max(min_res, min(max_res, n_res * max_frac, n_res)))
    mask_len = random.randint(min_res, max_res)
    return min(n_res, mask_len)


def bb_dihedral_mask(feat_mask: Tensor) -> Tensor:
    """Get backbone dihedral feature mask"""
    mask = feat_mask.clone()
    mask[:-1] = torch.logical_or(feat_mask[1:], mask[:-1])
    mask[1:] = torch.logical_or(feat_mask[:-1], mask[1:])
    mask[0] = mask[-1] = True
    return repeat(mask, "i -> i a", a=3)


def pair_feat_mask(feat_mask) -> Tensor:
    """Generate mask for pairwise features"""
    return ~torch.einsum("i,j->ij", ~feat_mask, ~feat_mask)


def chain_mask_to_full_mask(n_res: int, mask: Tensor, indices: Tensor) -> Tensor:
    """Convert mask for chain to mask for full protein"""
    full_mask = torch.zeros(n_res, device=mask.device).bool()
    full_mask[indices] = mask
    return full_mask


"""Masks for creating pseudo-complexes from chains"""


def get_bipartite_mask(
        mask: Tensor,
        S_indices: Tensor,
        T_indices: Optional[Tensor] = None,
        fill: Any = True,
) -> Tensor:
    """Mask all (i,j) such that (i in S and j in T) or (j in S and i in T)
    """
    T_indices = default(T_indices, S_indices)
    rep_T = repeat(T_indices, "i -> i m", m=S_indices.numel())
    rep_S = repeat(S_indices, "i -> i m", m=T_indices.numel())
    mask[S_indices, rep_T] = fill
    mask[T_indices, rep_S] = fill
    return mask


def get_partition_mask(
        n_res: int, partition: List[Tensor], part_adjs: Optional[Tensor] = None,
) -> Tensor:
    """Mask edges crossing between components of partition.

    e.g. if partition = [[1,2],[6], [8]], then edges according to the
    adjacency lists :
        1 : (6,7,8)
        2:  (6,7,8)
        6:  (1,2,8)
        8 : (1,2,6)
    will be masked (in the undirected sense).

    :param n_res: size of underlying graph
    :param partition: partition of the vertices (NOTE: does not strictly
    need to be a partition, can consist of any subsets of vertices)
    :param part_adjs: adjacency matrix for subsets of partition
    :return: adjacency mask with partition edges set to "True", and
    all other edges set to "False".
    """
    assert len(partition) > 0
    n, device = len(partition), partition[0].device
    part_adjs = default(part_adjs, torch.ones(n, n, device=device).bool())
    mask = torch.zeros(n_res, n_res, device=device).bool()
    for i in range(len(partition)):
        X = partition[i]
        for j in range(i + 1, len(partition)):  # noqa
            if part_adjs[i, j]:
                Y = partition[j]
                mask = get_bipartite_mask(mask, X, Y, fill=TRUE.to(device))
    return mask


def normalize_residue_indices(res_indices: Tensor) -> Tensor:
    """Normalize residue indices to begin from 0"""
    return res_indices - torch.min(res_indices)


def sample_strategy(strats: List[Callable], wts: List[float]) -> Callable:
    """Sample a strategy"""
    strategy_idx = np.random.choice(len(strats), p=wts)
    return strats[strategy_idx]


def get_chain_masks(n_res: int, chain_indices: List[Tensor]
                    ) -> Tuple[List[Tensor], List[Tensor], Tensor]:
    """Single and pairwise masks for each chain"""
    chain_masks, chain_pair_masks, device = [], [], chain_indices[0].device
    for indices in chain_indices:
        chain_masks.append(
            chain_mask_to_full_mask(
                n_res,
                mask=torch.ones(len(indices), device=device).bool(),
                indices=indices
            )
        )
        chain_pair_masks.append(pair_feat_mask(chain_masks[-1]))
    inter_chain_mask = get_partition_mask(n_res=n_res, partition=chain_indices)
    return chain_masks, chain_pair_masks, inter_chain_mask


def mask_intra_chain_features(
        feats: Dict[str, Feature],
        feat_mask: Tensor,
        mask_seq: bool = True
) -> Dict[str, Feature]:
    """Mask intra-chain features"""
    assert exists(feat_mask)
    seq_mask = {FeatureName.RES_TY.value if not mask_seq else ""}
    ignore_feats = set(INTRA_IGNORE_FEAT_NAMES).union(seq_mask)
    pair_mask = pair_feat_mask(feat_mask)
    dihedral_mask = bb_dihedral_mask(feat_mask)
    for feature_name, feature in feats.items():
        if feature.name in ignore_feats:
            # print("skipping mask for", feature_name)
            continue

        if feature.ty == FeatureTy.RESIDUE:
            if feature_name == FeatureName.BB_DIHEDRAL.value:
                feature.apply_mask(dihedral_mask)
            else:
                feature.apply_mask(feat_mask)

        elif feature.ty == FeatureTy.PAIR:
            feature.apply_mask(pair_mask)

        else:
            raise Exception(f"Error masking {feature.name}, feature type "
                            f": {feature.ty.value} not supported")

    # print("intra chain mask ",feat_mask[feat_mask].numel()/feat_mask.numel())
    # print("intra chain pair mask ", pair_mask[pair_mask].numel()/pair_mask.numel())
    return feats


def mask_inter_chain_pair_features(feats: Dict[str, Feature], pair_mask: Tensor
                                   ) -> Dict[str, Feature]:
    """Mask inter-chain pair features"""
    for feature_name, feature in feats.items():
        if feature.name in INTER_IGNORE_FEAT_NAMES or feature.ty == FeatureTy.RESIDUE:
            continue
        if feature.ty == FeatureTy.PAIR:
            feature.apply_mask(pair_mask)
    # print("inter chain pair mask ", pair_mask[pair_mask].numel() / pair_mask.numel())
    return feats
