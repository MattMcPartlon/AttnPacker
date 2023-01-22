"""Masked Feature Generation"""
from __future__ import annotations

from functools import partial
from typing import List, Tuple, Union, Optional, Callable, Dict

import numpy as np
import torch
from einops import repeat, rearrange  # noqa
from torch import Tensor

from protein_learning.assessment.metrics import get_inter_chain_contacts
from protein_learning.common.data.data_types.protein import Protein
from protein_learning.common.global_constants import get_logger
from protein_learning.common.helpers import exists
from protein_learning.common.helpers import k_spatial_nearest_neighbors
from protein_learning.features.masking.masking_utils import (
    bool_tensor,
    get_mask_len,
    sample_strategy,
    cast_list,
    norm_weights,
)

logger = get_logger(__name__)

max_value = lambda x: torch.finfo(x.dtype).max  # noqa

"""Intra-Chain Masks"""


def point_mask(num_residue: int, posn: int, **kwargs):
    """Mask a single residue at given position"""
    mask = torch.zeros(num_residue)
    mask[posn] = 1
    return mask.bool()


def contiguous_mask(
        num_residue: int,
        coords: Tensor,  # noqa
        min_n_max_res: Tuple[int, int],
        min_n_max_p: Tuple[float, float],
        **kwargs,
) -> Tensor:
    """Masks a contiguous segment of a sequence"""
    mask_len = get_mask_len(
        n_res=num_residue - 1,
        min_n_max_p=min_n_max_p,
        min_n_max_res=min_n_max_res,
    )
    mask_start = np.random.randint(0, max(0, num_residue - mask_len))
    mask_posns = torch.arange(start=mask_start, end=mask_start + mask_len)
    return bool_tensor(num_residue, posns=mask_posns, fill=True)


def spatial_mask(
        num_residue: int,
        coords: Tensor,
        min_n_max_p: Tuple[float, float],
        top_k: Union[List[int, int], List[int], int] = 30,
        max_radius=12,
        mask_self: bool = False,
        atom_pos: int = 1,
        **kwargs,  # noqa

) -> Tensor:
    """Masks positions in a sequence based on spatial proximity to a random query residue"""
    coords = coords.squeeze(0) if coords.ndim == 4 else coords
    top_k = get_mask_len(
        n_res=num_residue,
        min_n_max_p=min_n_max_p,
        min_n_max_res=(1, top_k)
    )
    mask_posns = k_spatial_nearest_neighbors(
        points=coords[:, atom_pos],
        idx=np.random.choice(num_residue),
        top_k=min(num_residue, top_k - int(not mask_self)),
        max_dist=max_radius,
        include_self=not mask_self
    )
    return bool_tensor(num_residue, posns=mask_posns, fill=True)


def random_mask(num_residue: int, coords: Tensor, min_p: float, max_p: float, **kwargs) -> Tensor:  # noqa
    """Randomly masks each sequence position w.p. in range (min_p, max_p)"""
    mask_prob = np.random.uniform(min_p, max_p)
    mask_posns = torch.arange(num_residue)[torch.rand(num_residue) < mask_prob]
    return bool_tensor(num_residue, posns=mask_posns, fill=True)


def no_mask(num_residue: int, *args, **kwargs) -> Tensor:  # noqa
    """Does not mask any sequence positions"""
    return bool_tensor(num_residue, fill=False)


def full_mask(num_residue: int, *args, **kwargs) -> Tensor:  # noqa
    """Does not mask any sequence positions"""
    return bool_tensor(num_residue, fill=True)


def interface_mask(
        num_residue: int,
        scores: Tensor,
        min_frac: float = 0,
        max_frac: float = 0.5,
        **kwargs
) -> Tensor:
    """Mask residues according to interface score"""
    n_to_mask = max(1, int(np.random.uniform(min_frac, max_frac) * num_residue))
    scores = scores.detach().cpu().numpy()
    p = scores / np.sum(scores)
    mask_posns = np.random.choice(len(scores), size=n_to_mask, replace=False, p=p)
    return bool_tensor(num_residue, posns=torch.tensor(mask_posns).long(), fill=True)


def inverse_interface_mask(
        num_residue: int,
        scores: Tensor,
        min_frac: float = 0,
        max_frac: float = 0.5,
        **kwargs
) -> Tensor:
    """Mask residues according to interface score"""
    return ~interface_mask(
        num_residue=num_residue,
        scores=scores,
        min_frac=1 - max_frac,
        max_frac=1 - min_frac,
        **kwargs,
    )


def interface_full_mask(
        num_residue: int,
        scores: Tensor,
        min_frac: float = 0,
        max_frac: float = 0.5,
        **kwargs
) -> Tensor:
    """Mask residues according to interface"""
    n_to_mask = max(1, int(np.random.uniform(min_frac, max_frac) * num_residue))
    scores = scores.detach().cpu().numpy()
    mask_posns = np.argsort(-scores)[:n_to_mask]
    return bool_tensor(num_residue, posns=torch.tensor(mask_posns).long(), fill=True)


def inverse_interface_full_mask(
        num_residue: int,
        scores: Tensor,
        min_frac: float = 0,
        max_frac: float = 0.5,
        **kwargs
) -> Tensor:
    """Mask residues according to interface scaffold"""
    return ~interface_full_mask(
        num_residue=num_residue,
        scores=scores,
        min_frac=1 - max_frac,
        max_frac=1 - min_frac,
        **kwargs
    )


def select_chain_to_mask(
        protein: Protein,
) -> Tensor:
    """Selects ids of chain to mask"""
    partition, idx = protein.chain_indices, 0
    if protein.is_complex:
        idx = np.argmin([len(p) for p in partition])
    return partition[idx]


def true_interface_mask(
        num_residue: int,
        scores: Tensor,
        native: Protein,
        **kwargs
) -> Tensor:
    """Mask residues according to interface scaffold"""
    chain = select_chain_to_mask(native)
    contacts = get_inter_chain_contacts(native["CA"], partition=native.chain_indices, contact_thresh=12)
    scores = (torch.sum(contacts.float(), dim=-1)[chain] > 0).float()
    mask_posns = torch.arange(len(scores))[scores == 1]
    return bool_tensor(num_residue, posns=mask_posns.long(), fill=True)


def true_inverse_interface_mask(
        num_residue: int,
        scores: Tensor,
        native: Protein,
        **kwargs
) -> Tensor:
    """Mask residues according to interface scaffold"""
    return ~true_interface_mask(num_residue, scores, native, **kwargs)


def cdr_mask(
        num_residue: int,
        native: Protein,
        **kwargs
) -> Tensor:
    """Mask residues according to interface scaffold"""
    assert native.cdrs is not None
    cdrs = torch.cat([torch.arange(s, e + 1) for (s, e) in native.cdrs["heavy"]])
    return bool_tensor(num_residue, posns=cdrs.long(), fill=True)


def get_intra_chain_mask_strats_n_weights(
        no_mask_weight: float = 0,
        random_mask_weight: float = 0,
        contiguous_mask_weight: float = 0,
        spatial_mask_weight: float = 0,
        full_mask_weight: float = 0,
        interface_mask_weight: float = 0,
        inverse_interface_mask_weight: float = 0,
        interface_full_mask_weight: float = 0,
        inverse_interface_full_mask_weight: float = 0,
        true_interface_mask_weight: float = 0,
        true_inverse_interface_mask_weight: float = 0,
        cdr_mask_weight: float = 0,
        random_mask_min_p: float = 0,
        random_mask_max_p: float = 0,
        spatial_mask_top_k: int = 30,
        spatial_mask_max_radius: float = 12,
        spatial_mask_mask_self: bool = False,
        spatial_mask_atom_pos: int = 1,
        contiguous_mask_max_len: int = 60,
        contiguous_mask_min_len: int = 5,
        max_mask_frac: float = 0.3,
        min_mask_frac: float = 0.0,
        interface_mask_min_frac: float = 0.0,
        interface_mask_max_frac: float = 0.5,
        inverse_interface_mask_min_frac: float = 0.2,
        inverse_interface_mask_max_frac: float = 0.6,
):
    """Get mask strategy functions and strategy weights"""
    # mask options
    mask_strategies = [
        no_mask,
        full_mask,
        partial(
            random_mask,
            min_p=random_mask_min_p,
            max_p=random_mask_max_p
        ),
        partial(
            spatial_mask,
            top_k=spatial_mask_top_k,
            max_radius=spatial_mask_max_radius,
            min_n_max_p=(min_mask_frac, max_mask_frac),
            mask_self=spatial_mask_mask_self,
            atom_pos=spatial_mask_atom_pos,
        ),
        partial(
            contiguous_mask,
            min_n_max_res=(contiguous_mask_min_len, contiguous_mask_max_len),
            min_n_max_p=(min_mask_frac, max_mask_frac),
        ),
        partial(
            interface_mask,
            min_frac=interface_mask_min_frac,
            max_frac=interface_mask_max_frac,
        ),
        partial(
            inverse_interface_mask,
            min_frac=inverse_interface_mask_min_frac,
            max_frac=inverse_interface_mask_max_frac,
        ),
        partial(interface_full_mask,
                min_frac=interface_mask_min_frac,
                max_frac=interface_mask_max_frac,
                ),
        partial(inverse_interface_full_mask,
                min_frac=inverse_interface_mask_min_frac,
                max_frac=inverse_interface_mask_max_frac,
                ),
        partial(true_interface_mask),
        partial(cdr_mask),
        partial(true_inverse_interface_mask),
    ]

    strategy_weights = [
        no_mask_weight,
        full_mask_weight,
        random_mask_weight,
        spatial_mask_weight,
        contiguous_mask_weight,
        interface_mask_weight,
        inverse_interface_mask_weight,
        interface_full_mask_weight,
        inverse_interface_full_mask_weight,
        true_interface_mask_weight,
        cdr_mask_weight,
        true_inverse_interface_mask_weight
    ]

    strats_n_weights = list(zip(mask_strategies, strategy_weights))
    fltr = lambda idx: [x[idx] for x in strats_n_weights if x[1] > 0]
    return fltr(0), fltr(1)


class IntraChainMaskGenerator:
    """Generate and apply intra-chain sequence and feature masks"""

    def __init__(
            self,
            strats: Optional[List[Callable]] = None,
            weights: Optional[List[float]] = None,
            strat_n_weight_kwargs: Optional[Dict] = None,

    ):
        if not exists(strats) or not exists(weights):
            assert exists(strat_n_weight_kwargs)
            strats, weights = get_intra_chain_mask_strats_n_weights(**strat_n_weight_kwargs)
        self.strats = cast_list(strats) if exists(strats) else None
        self.weights = norm_weights(cast_list(weights)) if exists(weights) else None

    def get_mask(
            self,
            n_res: int,
            coords: Tensor,
            scores: Optional[Tensor] = None,
            native: Optional[Protein] = None,
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """Gets masks to apply to sequence and coordinate features"""
        assert len(self.strats), f"no strategies passed in init!"
        return sample_strategy(
            self.strats,
            self.weights,
        )(num_residue=n_res, coords=coords, scores=scores, native=native)
