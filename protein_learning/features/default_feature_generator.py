"""Default Feature Generation"""
from __future__ import annotations

import random
from typing import Callable, Optional, Dict

import numpy as np
import torch
from einops import repeat, rearrange  # noqa
from torch import Tensor

from protein_learning.common.data.data_types.model_input import ExtraInput
from protein_learning.common.data.data_types.protein import Protein
from protein_learning.common.global_constants import get_logger
from protein_learning.common.helpers import exists, default
from protein_learning.features.feature_config import FeatureName
from protein_learning.features.feature_config import InputFeatureConfig
from protein_learning.features.feature_generator import FeatureGenerator, get_input_features
from protein_learning.features.input_features import InputFeatures
from protein_learning.features.masking.inter_chain import InterChainMaskGenerator
from protein_learning.features.masking.intra_chain import IntraChainMaskGenerator
from protein_learning.features.masking.masking_utils import (
    chain_mask_to_full_mask,
    mask_inter_chain_pair_features,
    mask_intra_chain_features,
)

logger = get_logger(__name__)

max_value = lambda x: torch.finfo(x.dtype).max  # noqa


def select_chain_to_mask(
    protein: Protein,
) -> Tensor:
    """Selects ids of chain to mask"""
    partition, idx = protein.chain_indices, 0
    if protein.is_antibody:
        return partition[0]
    if protein.is_complex:
        idx = np.argmin([len(p) for p in partition])
    return partition[idx]


class DefaultFeatureGenerator(FeatureGenerator):
    """Feature Generator with masking functionality"""

    def __init__(
        self,
        config: InputFeatureConfig,
        intra_chain_mask_kwargs: Optional[Dict],
        inter_chain_mask_kwargs: Optional[Dict],
        mask_feats: bool = False,
        mask_seq: bool = False,
        mask_feat_n_seq_indep_prob: float = 0,
        select_ids_to_mask_fn: Optional[Callable] = None,
        coord_noise: float = 0,
        apply_masks: bool = True,
    ):
        super(DefaultFeatureGenerator, self).__init__(config=config)

        self.intra_mask = (
            IntraChainMaskGenerator(
                strat_n_weight_kwargs=intra_chain_mask_kwargs,
            )
            if exists(intra_chain_mask_kwargs)
            else None
        )
        logger.info(f"got intra-mask args : {intra_chain_mask_kwargs}")
        self.inter_mask = (
            InterChainMaskGenerator(
                strat_n_weight_kwargs=inter_chain_mask_kwargs,
            )
            if exists(inter_chain_mask_kwargs)
            else None
        )
        logger.info(f"got inter-mask args : {intra_chain_mask_kwargs}")
        # what to mask
        self.mask_seq, self.mask_feats = mask_seq, mask_feats
        self.mask_seq_n_feat_indep_prob = mask_feat_n_seq_indep_prob
        logger.info(f"mask seq/feat : {mask_seq},{mask_feats}")

        # how to select the chain to mask
        self.select_chain_mask_fn = default(select_ids_to_mask_fn, select_chain_to_mask)

        # noise to add to coordinates
        self.coord_noise = coord_noise

        # whether to apply masks to input or store in model options
        self.apply_mask = apply_masks

        # sanity check
        if apply_masks:
            assert config.pad_embeddings and (exists(intra_chain_mask_kwargs) or exists(inter_chain_mask_kwargs))

    def generate_masks(self, protein: Protein, extra: ExtraInput, native: Optional[Protein] = None):
        """Generate seq/feature/pair/chain masks"""

        ids_to_mask = self.select_chain_mask_fn(protein=protein)

        # Feature Masks
        seq_mask, feat_mask, inter_chain_pair_mask = None, None, None
        native = default(native, protein)
        _mask = lambda: self.intra_mask.get_mask(
            n_res=len(ids_to_mask),
            coords=protein.atom_coords[ids_to_mask],
            native=native,
            **extra.intra_chain_mask_kwargs(ids_to_mask),
        )
        if self.mask_seq:
            seq_mask = _mask()
        if self.mask_feats:
            if random.random() < self.mask_seq_n_feat_indep_prob:
                feat_mask = _mask()
            else:
                feat_mask = seq_mask if exists(seq_mask) else _mask()
                if exists(feat_mask) and not torch.any(feat_mask):  # noqa
                    feat_mask = None

        # Expand sequence and feature mask to full protein
        if exists(seq_mask):
            seq_mask = chain_mask_to_full_mask(len(protein), seq_mask, ids_to_mask)  # noqa
        if exists(feat_mask):
            feat_mask = chain_mask_to_full_mask(len(protein), feat_mask, ids_to_mask)  # noqa

        # Mask for inter-chain pair features
        if exists(self.inter_mask) and protein.is_complex:
            inter_chain_pair_mask = self.inter_mask.get_mask(
                len(protein),
                protein.chain_indices,
            )

        return seq_mask, feat_mask, inter_chain_pair_mask

    def generate_features(
        self,
        protein: Protein,
        extra: Optional[ExtraInput] = None,
        feat_mask: Tensor = None,
        seq_mask: Tensor = None,
        inter_chain_pair_mask: Tensor = None,
        dihedral_mask: Tensor = None,
    ) -> InputFeatures:
        """Generate ProteinModel input features"""
        native = getattr(extra, "native", None)
        if native is None:
            print("[Warning]: missing native protein in extra input")
        seq, coords = protein.seq, protein.atom_coords
        mask_info = self.generate_masks(protein, extra=extra, native=native)
        _seq_mask, _feat_mask, _inter_chain_pair_mask = mask_info
        seq_mask = default(seq_mask, _seq_mask)
        feat_mask = default(feat_mask, _feat_mask)
        inter_chain_pair_mask = default(inter_chain_pair_mask, _inter_chain_pair_mask)
        res_extra, pair_extra = extra.extra_feats(protein)

        # Generate the (unmasked) features
        feats = get_input_features(
            protein=protein,
            config=self.config,
            extra_residue=res_extra,
            extra_pair=pair_extra,
        )
        # Mask Sequence
        if exists(seq_mask) and self.apply_mask and self.mask_seq:
            feats[FeatureName.RES_TY.value].apply_mask(seq_mask)
            logger.info(f"[seq] masked : {seq_mask[seq_mask].numel()}/{seq_mask.numel()}")
        if FeatureName.SC_DIHEDRAL.value in feats:
            _dihedral_mask = default(dihedral_mask, torch.ones(len(protein), device=protein.device).bool())
            dihedral_mask = torch.logical_or(default(seq_mask, _dihedral_mask), _dihedral_mask)
            feats[FeatureName.SC_DIHEDRAL.value].apply_mask(dihedral_mask)
        # Apply intra-chain masks
        if exists(feat_mask) and self.apply_mask and self.mask_feats:
            feats = mask_intra_chain_features(feats, feat_mask=feat_mask, mask_seq=False)
            logger.info(f"[feat] masked : {feat_mask[feat_mask].numel()}/{feat_mask.numel()}")
        # Apply inter-chain pair mask
        if exists(inter_chain_pair_mask) and self.apply_mask and self.mask_feats:
            feats = mask_inter_chain_pair_features(feats=feats, pair_mask=inter_chain_pair_mask)
            logger.info(
                f"[inter-chain] masked : "
                f"{inter_chain_pair_mask[inter_chain_pair_mask].numel()}"
                f"/{inter_chain_pair_mask.numel()}"
            )

        input_feats = InputFeatures(
            features=feats,
            batch_size=1,
            length=len(seq),
            inter_chain_mask=inter_chain_pair_mask,
            feat_mask=feat_mask,
            seq_mask=seq_mask,
        ).maybe_add_batch()
        # TODO: hack!
        if res_extra is not None:
            res_extra_desc, pair_extra_desc = extra.extra_feat_descs(protein)
            res_flags = input_feats["extra_res"]
            res_flags.kwargs = res_extra_desc
            input_feats["extra_res"] = res_flags
        # optionally load extra features where mask info is required

        extra.load_extra_mask_inputs(
            seq_mask=seq_mask,
            feat_mask=feat_mask,
            inter_chain_mask=inter_chain_pair_mask,
        )

        return input_feats
