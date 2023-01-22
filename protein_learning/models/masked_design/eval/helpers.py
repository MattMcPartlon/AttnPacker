"""
Flags
    (1) Contact Flags
    (2) Pocket Flags

Masks
    (1) Intra-chain masks
    (2) Inter-chain masks

"""

from typing import Optional, List, Dict

import torch
from torch import Tensor

from protein_learning.common.data.data_types.protein import Protein
from protein_learning.features.feature_config import InputFeatureConfig
from protein_learning.features.input_features import InputFeatures
from protein_learning.features.feature_generator import FeatureGenerator, get_input_features
from protein_learning.features.masking.masking_utils import (
    get_partition_mask,
    apply_mask_to_seq,
    mask_intra_chain_features,
    mask_inter_chain_pair_features,
)
from protein_learning.common.data.data_types.model_input import ExtraInput


class ExtraComplexDesignEval(ExtraInput):
    def __init__(self,
                 feature_config: InputFeatureConfig,
                 partition: List[Tensor],
                 chain_ids: Tensor,
                 seq_mask: Tensor,
                 feat_mask: Tensor,
                 pair_mask: Tensor,
                 res_flags: Optional[Tensor],
                 pair_flags: Optional[Tensor],
                 ):
        self.feat_config = feature_config
        self.partition = partition


class ComplexEvalFeatGen(FeatureGenerator):
    def __init__(self, data_dict: Dict):
        self.data_dict = data_dict

    def get_masks_n_data_for_target(self, target: Protein):
        chain_names = tuple(target.chain_names)
        if chain_names in self.data_dict:
            print(f"[WARNING]: {chain_names} not in data dict!")

    def generate_features(self, protein: Protein, extra: ExtraComplexDesignEval = None) -> InputFeatures:
        # Generate the (unmasked) features
        p0, p1 = extra.partition[0], extra.partition[1]
        res_ids = torch.cat((torch.arange(len(p0)), torch.arange(len(p1))), dim=0)

        feats = get_input_features(
            seq=apply_mask_to_seq(protein.seq, seq_mask),
            coords=protein.atom_coords,
            res_ids=res_ids,
            atom_ty_to_coord_idx=protein.atom_positions,
            config=feature_config,
            chain_ids=chain_ids,
            extra_residue=res_flags,
            extra_pair=pair_flags,
        )

        pair_mask = get_partition_mask(n_res=len(p0) + len(p1), partition=partition)
        # mask features
        feats = mask_intra_chain_features(feats, feat_mask=feat_mask)
        feats = mask_inter_chain_pair_features(feats, pair_mask=pair_mask)
        return InputFeatures(
            features=feats,
            batch_size=1,
            length=len(protein.seq),
            masks=(seq_mask, feat_mask),
            chain_indices=partition,
            chain_ids=chain_ids,
        ).maybe_add_batch()
