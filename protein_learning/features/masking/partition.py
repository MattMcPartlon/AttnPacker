"""Masked Feature Generation"""
from __future__ import annotations

import random
from functools import partial
from typing import List, Optional, Tuple, Callable, Dict

import numpy as np
import torch
from einops import repeat, rearrange  # noqa
from torch import Tensor

from protein_learning.common.data.data_types.protein import Protein
from protein_learning.common.global_constants import get_logger
from protein_learning.common.helpers import exists
from protein_learning.features.masking.masking_utils import (
    get_mask_len,
    sample_strategy,
    cast_list,
    norm_weights
)

logger = get_logger(__name__)


def bilinear_partition(
        n_res: int,
        min_frac: float = 0,
        max_frac: float = 1,
        min_len: int = 5,
        max_len: int = 60,
        *args, **kwargs,  # noqa
) -> List[Tensor]:
    """Partition n_res elements into two subsets by selecting
    a random contiguous (linear) segment of the elements and
    setting that segment as the first subset in the partition

    e.g. If n_res = 10, then a possible partition may be
    {[4,5,6], [1,2,3,7,8,9,10]}

    The first subset in the partition will have size bounded by
    min/max frac and len.
    """
    min_len = min(max(min_len, int(min_frac * n_res)), n_res // 2)
    mask_len = get_mask_len(
        n_res=n_res - min_len,
        min_n_max_p=(min_frac, max_frac),
        min_n_max_res=(min_len, max_len)
    )
    start = random.randint(0, n_res - mask_len)
    S = torch.arange(start=start, end=start + mask_len)
    T = torch.cat((torch.arange(0, start), torch.arange(start + mask_len, n_res)), dim=0)
    return [S, T]


def random_linear_partition(
        n_res: int,
        n_classes: Tuple[int, int] = (2, 2),
        min_len: int = 10,
        min_frac: int = 0.0,
        *args, **kwargs,  # noqa
) -> List[Tensor]:
    """Similar to bilinear partition, this method will partition n_res
    elements into n_classes sets by selecting random contiguous (linear) segments

    Unlike the bilinear mask, the leading/trailing elements are placed in
    separate subsets.

    e.g. If n_res = 10, and n_classes=3, then a possible partition may be
    {[1,2,3],[4,5,6],[7,8,9,10]}

    e.g. If n_res = 10, and n_classes=2, then a possible partition may be
    {[1,2,3,4,5,6],[7,8,9,10]}

    all subsets in the partition will have size bounded below by min_len
    """
    min_len = max(min_len, n_res * min_frac)
    min_diff = lambda x: min([a - b for a, b in zip(x[1:], x[:-1])])
    split_idxs, attempts = [0, 0], 0
    n_classes = random.randint(*n_classes)
    min_len = min(min_len, n_res // (2 * n_classes))
    assert min_len >= 5, "minimum length should not be smaller than 5!"
    best_diff, best_split = 0, None
    while best_diff < min_len and attempts < 30:
        split_idxs = np.random.choice(np.arange(n_res - 2 * min_len), n_classes - 1, replace=False) + min_len
        split_idxs = np.sort([0] + split_idxs.tolist() + [n_res])
        diff = min_diff(split_idxs)
        if diff > best_diff:
            best_diff, best_split = diff, split_idxs
        attempts += 1
    return [torch.arange(start=s, end=e) for s, e in zip(best_split[:-1], best_split[1:])]


def identity_partition(n_res: int) -> List[Tensor]:
    """Identity partition - all indices in same set"""
    return [torch.arange(n_res)]


def get_partition_strategies_n_weights(
        linear_partition_weight: float = 0,
        bilinear_partition_weight: float = 0,
        identity_partition_weight: float = 0,
        linear_n_classes: List[int, int] = (2, 2),
        bilinear_max_len: int = 60,
        partition_min_frac: float = 0,
        partition_min_len: int = 10,
) -> Tuple[List[Callable], List[float]]:
    """Get list of (callable) partition strategies and corresponding weights"""
    strats, wts = [], []

    def register(strat, wt):
        """register strategy and weight"""
        if wt > 0:
            wts.append(wt)
            strats.append(strat)

    register(
        partial(
            random_linear_partition,
            n_classes=linear_n_classes,
            min_len=partition_min_len,
            min_frac=partition_min_frac,
        ),
        linear_partition_weight
    )

    register(
        partial(
            bilinear_partition,
            min_len=partition_min_len,
            max_len=bilinear_max_len,
            min_frac=partition_min_frac,
        ),
        bilinear_partition_weight
    )

    register(identity_partition, identity_partition_weight)

    return strats, wts


class ChainPartitionGenerator:
    """Partition a protein chain to create a pseudo-complex"""

    def __init__(
            self,
            strats: Optional[List[Callable]] = None,
            weights: Optional[List[float]] = None,
            strat_n_weight_kwargs: Optional[Dict] = None,
    ):
        if not exists(strats):
            assert exists(strat_n_weight_kwargs)
            strats, weights = get_partition_strategies_n_weights(**strat_n_weight_kwargs)
        self.strats = cast_list(strats)
        self.weights = norm_weights(cast_list(weights))

    def get_chain_partition_info(self, protein: Protein) -> Optional[Tuple[Tensor, Tensor, List]]:
        """Generate masks, indices, and IDs for protein chains

        :param protein: the protein to generate chain info for.

        :return:
            chain_mask: (n, n) mask with inter-chain edges sampled according to partition strategies
            chain_indices: (n,) re-mapped indices for intra chain residues
            chain_ids: (n, ) Mapping from residue to chain id (integer from 0...n_chains).
        """
        if exists(self.strats):
            assert not protein.is_complex, "should not be partitioning a protein complex!"
            partition_strat = sample_strategy(self.strats, self.weights)
            partition = partition_strat(len(protein))
            residue_indices = protein.res_ids[0]
            chain_ids = torch.zeros_like(residue_indices)
            # remap the residue indices and create chain ids
            for chain_idx, subset in enumerate(partition):
                residue_indices[subset] -= torch.min(residue_indices[subset])
                chain_ids[subset] = chain_idx
            # logger.info(f"CHAIN-PARTITION: partition sizes {[len(x) for x in partition]}, n_res : {len(protein)}")
            return residue_indices, chain_ids, partition
        else:
            raise Exception("Not yet implemented!")
