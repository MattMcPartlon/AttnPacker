from __future__ import annotations

from typing import Any, Optional, Tuple, Dict, Callable

import torch
from einops import repeat  # noqa
from torch import Tensor

from protein_learning.common.data.data_types.model_input import ExtraInput
from protein_learning.common.data.data_types.protein import Protein
from protein_learning.common.helpers import exists, default
from protein_learning.common.protein_constants import AA_TO_INDEX
from protein_learning.common.rigids import Rigids
from protein_learning.models.utils.feature_flags import (
    get_interface_scores,
    FeatureFlagGen,
)


class ExtraComplexDesign(ExtraInput):
    """Store Encoded Native Sequence"""

    def __init__(self,
                 native_seq_enc: Tensor,
                 true_rigids: Rigids,
                 decoy_rigids: Rigids,
                 native_protein: Protein,
                 decoy_protein: Protein,
                 flag_gen: FeatureFlagGen,
                 use_esm: bool,
                 esm_chain_selector: Optional[Callable],
                 ):
        super(ExtraComplexDesign, self).__init__()
        self.native_seq_enc = native_seq_enc if native_seq_enc.ndim == 2 \
            else native_seq_enc.unsqueeze(0)
        self.true_rigids = true_rigids
        self.decoy_rigids = decoy_rigids
        self.native = native_protein
        self.decoy = decoy_protein
        self.flag_gen = flag_gen
        self.use_esm = use_esm
        self.esm_feats = None
        self.esm_chain_selector = esm_chain_selector

    def crop(self, start, end) -> ExtraComplexDesign:
        """Crop native seq. encoding"""
        self.native_seq_enc = self.native_seq_enc[:, start:end]
        self.true_rigids = self.true_rigids.crop(start, end)
        return self

    def to(self, device: Any) -> ExtraComplexDesign:
        """Send native sequence encoding to device"""
        self.native_seq_enc = self.native_seq_enc.to(device)
        self.true_rigids = self.true_rigids.to(device)
        self.decoy_rigids = self.decoy_rigids.to(device) if exists(self.decoy_rigids) else None
        if exists(self.esm_feats):
            self.esm_feats = {k: v.to(device) if torch.is_tensor(v) else v for k, v in self.esm_feats.items()}
        return self

    def intra_chain_mask_kwargs(self, ids_to_mask: Tensor) -> Dict:
        """Get key word mask arguments"""
        return dict(
            scores=get_interface_scores(
                self.decoy.chain_indices,
                self.native["CA"],
                sigma=3,
            )[ids_to_mask]
        )

    def extra_feats(self, protein: Protein) -> Optional[Tuple[Optional[Tensor], Optional[Tensor]]]:
        """Flag features"""
        return self.flag_gen.gen_flags(self.native, self.decoy)

    def extra_feat_descs(self, protein: Protein):
        return self.flag_gen.res_feat_descs, self.flag_gen.pair_feat_descs

    def load_extra_mask_inputs(
            self,
            seq_mask: Optional[Tensor],
            feat_mask: Optional[Tensor],
            inter_chain_mask: Optional[Tensor]
    ):
        if not self.use_esm:
            return

        use_chains = self.esm_chain_selector(self.decoy, self.native.is_complex)
        chain_indices = self.decoy.chain_indices
        seqs, seq_masks, msa_files, msa_crops = [], [], [], []
        for cidx in range(self.decoy.n_chains):
            if use_chains[cidx]:
                seqs.append("".join([self.decoy.seq[sidx] for sidx in chain_indices[cidx]]))
                if exists(seq_mask):
                    seq_masks.append(torch.tensor([seq_mask[sidx] for sidx in chain_indices[cidx]]))
                else:
                    seq_masks.append(None)
                msa_crops.append(self.decoy.input_partition[cidx])
                msa_files.append(self.decoy.chain_names[cidx] + ".a3m")
            else:
                seqs.append(None)
                seq_masks.append(None)
                msa_crops.append(None)
                msa_files.append(None)
        self.esm_feats = dict(
            chain_indices=self.decoy.chain_indices,
            seqs=seqs,
            seq_masks=seq_masks,
            msa_files=msa_files,
            msa_crops=msa_crops
        )


def augment(
        decoy_protein: Protein,
        native_protein: Protein,
        precompute_rigids: bool,
        flag_gen: Optional[FeatureFlagGen],
        use_esm: bool,
        esm_chain_selector: Optional[Callable],
        include_all_rigids: bool = False,
) -> ExtraComplexDesign:
    """Augment function for storing native seq. encoding in ModelInput object"""
    seq = native_protein.seq
    native_seq_enc = [AA_TO_INDEX[r] for r in seq]
    return ExtraComplexDesign(
        torch.tensor(native_seq_enc).long(),
        true_rigids=native_protein.rigids,
        decoy_protein=decoy_protein,
        native_protein=native_protein,
        decoy_rigids=get_decoy_rigids(decoy_protein, precompute_rigids, include_all_rigids),
        flag_gen=flag_gen,
        use_esm=use_esm,
        esm_chain_selector=esm_chain_selector,
    )


def get_decoy_rigids(decoy: Protein, precompute: bool, include_all_rigids: bool = False) -> Optional[Rigids]:
    """Get rigids"""
    if not decoy.is_complex or include_all_rigids:
        return decoy.rigids if precompute else None
    if precompute:
        chain_indices = decoy.chain_indices
        chains = [decoy.atom_coords[idxs] for idxs in chain_indices]
        assert chains[0].ndim == 3, f"{chains[0].shape}"
        chain_cas = [decoy["CA"][idxs].unsqueeze(1) for idxs in chain_indices]
        chains = [chains[i] - chain_cas[i].mean(dim=(0, 1), keepdim=True) for i in range(len(chains))]
        chain_lens = list(map(len, decoy.chain_indices))
        if chain_lens[0] > chain_lens[1]:
            rigids1 = Rigids.RigidFromBackbone(chains[0].unsqueeze(0))
            rigids2 = Rigids.IdentityRigid(leading_shape=(1, chain_lens[1]), device=decoy.device)
        else:
            rigids2 = Rigids.RigidFromBackbone(chains[1].unsqueeze(0))
            rigids1 = Rigids.IdentityRigid(leading_shape=(1, chain_lens[0]), device=decoy.device)
        return rigids1.concat(rigids2)
    return None
