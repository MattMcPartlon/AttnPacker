"""Masked Feature Generation"""
from __future__ import annotations

import random
from typing import List, Optional, Callable, Dict

import torch
from einops import repeat, rearrange  # noqa
from einops import repeat, rearrange  # noqa
from torch import Tensor

from protein_learning.common.global_constants import get_logger
from protein_learning.common.helpers import exists, default
from protein_learning.features.masking.masking_utils import (
    sample_strategy,
    cast_list,
    norm_weights,
    get_partition_mask,
)

logger = get_logger(__name__)


def inter_chain_random_mask(n_chains) -> Tensor:
    """Mask inter-chain information randomly"""
    return torch.randint(0, 2, size=(n_chains, n_chains)).bool()


def inter_chain_full_mask(n_chains) -> Tensor:
    """Mask all inter-chain information"""
    return torch.ones(n_chains, n_chains).bool()


def inter_chain_one_to_all_mask(n_chains) -> Tensor:
    """Mask inter-chain information between a single chain,
    and all other chains.
    """
    idx = random.randint(0, n_chains - 1)
    mask = torch.zeros(n_chains, n_chains)
    mask[idx, :], mask[:, idx] = 1, 1
    return mask.bool()


def get_inter_chain_mask_strats_n_weights(
        inter_random_mask_weight: float = 0,
        inter_one_to_all_mask_weight: float = 0,
        inter_full_mask_weight: float = 0,
        inter_no_mask_weight: float = 0,
):
    """Retrieve list of strategies and weights"""
    strats, weights = [], []

    def register(strat, weight):
        """register strategy and weight"""
        if weight > 0:
            weights.append(weight)
            strats.append(strat)

    register(inter_chain_random_mask, inter_random_mask_weight)
    register(inter_chain_one_to_all_mask, inter_one_to_all_mask_weight)
    register(inter_chain_full_mask, inter_full_mask_weight)
    register(lambda x: ~inter_chain_full_mask(x), inter_no_mask_weight)
    return strats, weights


class InterChainMaskGenerator:
    """Generates and applies masks for inter-chain features"""

    def __init__(
            self,
            strats: Optional[List[Callable]] = None,
            weights: Optional[List[float]] = None,
            mask_transform: Optional[Callable] = None,
            strat_n_weight_kwargs: Optional[Dict] = None
    ):
        if not exists(strats):
            assert exists(strat_n_weight_kwargs)
            strats, weights = get_inter_chain_mask_strats_n_weights(**strat_n_weight_kwargs)
        self.strats = cast_list(strats) if exists(strats) else None
        self.weights = norm_weights(cast_list(weights)) if exists(weights) else None
        self.mask_transform = default(mask_transform, lambda *args: args[0])

    def get_mask(self, n_res, partition: List[Tensor]) -> Tensor:
        """n_res x n_res adjacency mask with "True" stored at edges crossing between chains
        (*the actual edges that are masked depend on the inter-chain masking strategy).
        """
        assert len(self.strats) > 0
        adjs = sample_strategy(self.strats, self.weights)(len(partition))
        mask = get_partition_mask(n_res, partition=partition, part_adjs=adjs)
        return self.mask_transform(mask)
