"""Generator for design features"""
from __future__ import annotations

import _pickle as cPickle
import bz2
import os
from typing import List, Callable, Any, Optional, Dict, Tuple, Set

import esm  # noqa
import numpy as np
import torch
from einops import rearrange  # noqa
from torch import Tensor

from protein_learning.common.data.data_types.model_input import ExtraInput
from protein_learning.common.data.data_types.protein import Protein
from protein_learning.common.helpers import exists, default
from protein_learning.common.protein_constants import AA_ALPHABET
from protein_learning.common.protein_constants import AA_TO_INDEX
from protein_learning.features.feature_config import InputFeatureConfig
from protein_learning.features.feature_generator import FeatureGenerator, get_input_features
from protein_learning.features.feature_utils import bin_encode
from protein_learning.features.input_features import InputFeatures
from protein_learning.features.masking.masking_utils import apply_mask_to_seq, bool_tensor


def get_mask_from_mask_dict(n_res: int, entry: Any) -> Tensor:
    """Get sequence mask from mask dict entry"""
    if isinstance(entry, int):
        entry = [entry]
    if isinstance(entry, list):
        if isinstance(entry[0], int):
            assert min(entry) > 0 and max(entry) <= n_res, f"min/max entry :{min(entry)}/{max(entry)}, n_res : {n_res}"
            return bool_tensor(n=n_res, fill=True, posns=torch.tensor(entry) - 1)
        else:
            assert isinstance(entry[0], bool), f" mask entry should be list of " \
                                               f"ints or bools, got {type(entry[0])}\n{entry}"
            return torch.tensor(entry).bool()

    raise Exception(f"[ERROR] Unsupported mask type {type(entry)}, expected List, or int")


def corrupt_seq(seq: str, corrupt_prob) -> str:
    """Corrupt seq by randomly replacing residue labels"""
    seq_arr = np.array([x for x in seq])
    if corrupt_prob > 0:
        corrupt_mask = np.random.uniform(0, 1, size=len(seq)) <= corrupt_prob
        n_corruptions = len(corrupt_mask[corrupt_mask])
        corrupt_aas = np.random.choice(AA_ALPHABET[:-1], n_corruptions)
        seq_arr[corrupt_mask] = corrupt_aas
    return "".join(seq_arr)


class DesignFeatureGenerator(FeatureGenerator):
    """Generator for Design Features"""

    def __init__(
            self,
            config: InputFeatureConfig,
            mask_strategies: List[Callable[..., str]],
            strategy_weights: List[float],
            mask_dict: Optional[Dict] = None,
            coord_noise: float = 0,
            use_esm: bool = False,
            esm_gpu: int = 0,
            corrupt_seq_prob: float = 0,

    ):
        super(DesignFeatureGenerator, self).__init__(config=config)
        assert len(strategy_weights) == len(mask_strategies)
        assert sum(strategy_weights) > 0, f"got strategy weights {strategy_weights}"
        self.strategy_weights = np.array(strategy_weights) / sum(strategy_weights)
        self.mask_strategies = mask_strategies
        self.mask_dict = mask_dict
        self.coord_noise = coord_noise
        self.use_esm, self.esm_gpu = use_esm, esm_gpu
        self.corrupt_seq_prob = corrupt_seq_prob
        if use_esm:
            self.esm, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
            self.esm_converter = alphabet.get_batch_converter()
            self.esm = self.esm.eval().to(f"cuda:{esm_gpu}")

    def mask_sequence(self, protein: Protein, pdb_id: Optional[str] = None) -> Tuple[Tensor, str]:
        """Masks sequence"""
        seq, coords = protein.seq, protein.atom_coords
        mask = None
        if exists(self.mask_dict):
            pdb_id = pdb_id[:-4] if pdb_id.endswith(".pdb") else pdb_id
            if pdb_id in self.mask_dict:
                mask = get_mask_from_mask_dict(len(seq), entry=self.mask_dict[pdb_id])
            else:
                print(f"[WARNING] No entry for {pdb_id} in mask dict!")
        if not exists(mask):
            strategy_idx = np.random.choice(len(self.strategy_weights), p=self.strategy_weights)
            mask = self.mask_strategies[strategy_idx](len(seq), coords)
        masked_seq = apply_mask_to_seq(seq, mask)
        return mask, masked_seq

    def get_esm_embeddings(self, seq, seq_mask) -> Tuple[Tensor, Tensor]:
        masked_seq = " ".join(["<mask>" if seq_mask[i] else seq[i] for i in range(len(seq))])
        _, _, batch_tokens = self.esm_converter([("tmp", masked_seq)])
        with torch.no_grad():
            results = self.esm(
                batch_tokens.to(f"cuda:{self.esm_gpu}"),
                repr_layers=[33],
                return_contacts=True
            )
        # extract residue features (1,n+2,1280)
        res_feats = results["representations"][33][:, 1:-1, :]

        # extract pair features
        # shape is (1,33,20,n+2,n+2) -> (batch,layer,head,...)
        pair_feats = results["attentions"][33].squeeze()[1:-1, :]
        pair_feats = rearrange(pair_feats, "b l h n m -> b l n m h")

        return res_feats, pair_feats

    def generate_features(self, protein: Protein, extra: Optional[ExtraInput] = None) -> InputFeatures:
        """Generate input features for ProteinModel"""
        seq, coords = protein.seq, protein.atom_coords
        seq_mask, seq = self.mask_sequence(protein, pdb_id=protein.name)
        coords = coords + torch.randn_like(coords) * self.coord_noise
        protein.atom_coords = coords
        feats = get_input_features(
            seq=seq,
            coords=coords,
            res_ids=protein.res_ids,
            atom_ty_to_coord_idx=protein.atom_positions,
            config=self.config,
            extra_residue=extra.residue_flags(),
            extra_pair=extra.pair_flags(),
        )
        return InputFeatures(
            features=feats, batch_size=1, length=len(seq), masks=(seq_mask, None)
        ).maybe_add_batch()


class ExtraDesign(ExtraInput):
    """Store Encoded Native Sequence"""

    def __init__(self,
                 native_seq_enc: Tensor,
                 exp_resolved: bool,
                 plddts: Optional[Tensor]
                 ):
        super(ExtraDesign, self).__init__()
        self.native_seq_enc = native_seq_enc if native_seq_enc.ndim == 2 \
            else native_seq_enc.unsqueeze(0)
        self.exp_resolved = exp_resolved
        self.plddts = plddts

    def crop(self, start, end) -> ExtraDesign:
        """Crop native seq. encoding"""
        self.native_seq_enc = self.native_seq_enc[:, start:end]
        return self

    def to(self, device: Any) -> ExtraDesign:
        """Send native sequence encoding to device"""
        self.native_seq_enc = self.native_seq_enc.to(device)
        return self

    def residue_flags(self, **kwargs) -> Optional[Tensor]:
        """Flags for residue features"""
        n = self.native_seq_enc.numel()
        flags = torch.zeros(n, 1) if self.exp_resolved else torch.ones(n, 1)
        if exists(self.plddts):
            plddts = bin_encode(self.plddts, torch.arange(12) * 10)
            flags = torch.cat((flags, torch.nn.functional.one_hot(plddts, 11)), dim=-1)
        return flags

    def pair_flags(self, **kwargs) -> Optional[Tensor]:
        """Flags for pair features"""
        n = self.native_seq_enc.numel()
        flags = torch.zeros(n, n, 1) if self.exp_resolved else torch.ones(n, n, 1)
        return flags


def augment(
        decoy_protein: Protein,
        native_protein: Protein,
        exp_res_list: Optional[Set] = None,
        plddt_dir: Optional[str] = None,
) -> ExtraDesign:  # noqa
    """Augment function for storing native seq. encoding in ModelInput object"""
    exp_res_list = default(exp_res_list, set())
    seq = native_protein.seq
    native_seq_enc = [AA_TO_INDEX[r] for r in seq]
    exp_res = native_protein.name in exp_res_list
    plddts = torch.ones(len(native_protein)) * 102 if exists(plddt_dir) else None
    if exists(plddt_dir) and not exp_res:
        pkl_dir = os.path.join(plddt_dir, native_protein.name)
        try:
            pkl_file = os.path.join(pkl_dir, [x for x in os.listdir(pkl_dir) if x.endswith("bz2")][0])
            plddts = torch.tensor(decompress_pickle(pkl_file)["plddt"])
        except:
            print(f"[WARNING] : Unable to load plddt for : {pkl_dir}")
        plddts = plddts[native_protein.crops[0]:native_protein.crops[1]] if \
            exists(native_protein.crops) else plddts

    return ExtraDesign(torch.tensor(native_seq_enc).long(), exp_resolved=exp_res, plddts=plddts)


def decompress_pickle(file):
    """decompress and load .pkl.bz2 file"""
    data = bz2.BZ2File(file, 'rb')
    data = cPickle.load(data)  # noqa
    return data


def load_pdb_list(path) -> List:
    with open(path, "r") as f:
        pdb_list = f.readlines()
    get_name = lambda x: x[:-4] if x.endswith(".pdb") else x
    return [get_name(x.strip()) for x in pdb_list if len(x.strip()) > 1]
