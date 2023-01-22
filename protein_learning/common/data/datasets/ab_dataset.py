"""Protein Complex dataset"""
import random
from typing import Optional, List, Callable, Tuple, Union

from protein_learning.common.data.data_types.model_input import ExtraInput, ModelInput
from protein_learning.common.data.data_types.protein import Protein
from protein_learning.common.data.datasets.protein_dataset import ProteinDataset
from protein_learning.common.data.datasets.utils import (
    get_ab_ag_spatial_crop
)
from protein_learning.common.global_constants import get_logger
from protein_learning.features.feature_generator import FeatureGenerator
import os

logger = get_logger(__name__)

cast_list = lambda x: x if isinstance(x, list) else [x]


class AbDataset(ProteinDataset):
    """Generic protein dataset"""

    def __init__(
            self,
            model_list: str,
            decoy_folder: str,
            native_folder: str,
            seq_folder: str,
            raise_exceptions: bool,
            feat_gen: FeatureGenerator,
            shuffle: bool = True,
            atom_tys: Optional[List[str]] = None,
            crop_len: int = -1,
            augment_fn: Callable[[Protein, Protein], ExtraInput] = lambda *args, **kwargs: None,
            filter_fn: Callable[[Protein, Protein], bool] = lambda native, decoy: True,
            transform_fn: Callable[[Protein, Protein], Tuple[Protein, Protein]] = lambda native, decoy: (native, decoy),
            load_sec_structure: bool = True,
            ignore_res_ids: bool = False,
            **kwargs,

    ):
        super(AbDataset, self).__init__(
            model_list=model_list,
            decoy_folder=decoy_folder,
            native_folder=native_folder,
            seq_folder=seq_folder,
            raise_exceptions=raise_exceptions,
            shuffle=shuffle,
            crop_len=crop_len,
            augment_fn=augment_fn,
            filter_fn=filter_fn,
            transform_fn=transform_fn,
            load_sec_structure=load_sec_structure,
            feat_gen=feat_gen,
            atom_tys=atom_tys,
        )
        print(f"raise exceptions? {raise_exceptions}")
        self.ignore_res_ids = ignore_res_ids
        self.replica = 0

    def get_item_from_pdbs_n_seq(
            self,
            seq_path: Optional[Union[str, List[str]]],
            decoy_pdb_path: Optional[Union[str, List[str]]],
            native_pdb_path: Optional[Union[str, List[str]]],
    ) -> Optional[ModelInput]:
        """Load data given native and decoy pdb paths and sequence path"""
        assert isinstance(native_pdb_path, list), f"[ERROR] dataloader - native pdbs is not a list"
        include_light = True
        try:
            heavy, light, ag = decoy_pdb_path
            if ag is None or (not os.path.exists(ag)):
                ag = native_pdb_path[2]  # unrelaxed ag sometimes
            if heavy is None or not os.path.exists(heavy):
                heavy = native_pdb_path[0]
                if (native_pdb_path[1] is not None) and os.path.exists(native_pdb_path[1]):
                    light = native_pdb_path[1]
            light = light if include_light else None
            ab_ag_decoy = Protein.make_antibody(
                heavy=heavy,
                light=light,
                antigen=ag,
                atom_tys=self.atom_tys,
            )
            if self.native_folder != self.decoy_folder:
                chain_lens = ab_ag_decoy.chain_lens
                seq = ab_ag_decoy.seq
                seqs = [seq[int(sum(chain_lens[:i])):int(sum(chain_lens[:i + 1]))] for i in range(3)]
                heavy, light, ag = native_pdb_path
                light = light if include_light else None
                ab_ag = Protein.make_antibody(
                    heavy=heavy,
                    light=light,
                    antigen=ag,
                    atom_tys=self.atom_tys,
                    seqs=seqs,
                    load_ss=False
                )
            else:
                ab_ag = ab_ag_decoy.clone()

        except Exception as e:
            print(f"[ERROR] caught exception {e} loading proteins {native_pdb_path}")
            if self.raise_exceptions:
                print("raising exception")
                raise e
            return None
        try:
            if self.crop_len > 0:
                if ab_ag.n_chains > 1:
                    # has antigen, may need to crop it
                    if self.crop_len < len(ab_ag):
                        crop_posns = get_ab_ag_spatial_crop(ab_ag, crop_len=self.crop_len)
                        ab_ag, ab_ag_decoy = map(lambda x: x.restrict_to(crop_posns), (ab_ag, ab_ag_decoy))
                        ab_ag.set_input_partition(crop_posns)
                        ab_ag_decoy.set_input_partition(crop_posns)
                    else:
                        ab_ag_decoy.set_input_partition(ab_ag_decoy.chain_indices)
        except:
            print("caught exception cropping!")
            return None


        native, decoy = ab_ag, ab_ag_decoy

        if not self.filter_fn(decoy, native):
            print(f"filtering out : {native_pdb_path}")
            return None

        # HAck, change later! # TODO
        decoy.replica = self.replica
        native.replica = self.replica

        decoy, native = self.transform_fn(decoy, native)
        extra = self.augment(decoy, native)
        return ModelInput(
            decoy=decoy,
            native=native,
            input_features=self.feat_gen.generate_features(
                decoy,
                extra=extra,
            ),
            extra=extra,
        )
