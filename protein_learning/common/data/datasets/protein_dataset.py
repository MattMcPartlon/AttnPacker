"""Protein Complex dataset"""
from typing import Optional, List, Any, Callable, Tuple, Union

import torch

from protein_learning.common.data.data_types.model_input import ExtraInput, ModelInput
from protein_learning.common.data.data_types.protein import Protein
from protein_learning.common.data.datasets.dataset import ProteinDatasetABC
from protein_learning.common.data.datasets.utils import (
    get_contiguous_crop,
    get_dimer_spatial_crop,
    restrict_protein_to_aligned_residues,
    set_canonical_coords_n_masks,
)
from protein_learning.common.global_constants import get_logger
from protein_learning.common.helpers import default, exists
#from protein_learning.common.io.dips_utils import load_proteins_from_dill
from protein_learning.features.feature_generator import FeatureGenerator

logger = get_logger(__name__)

cast_list = lambda x: x if isinstance(x, list) else [x] # noqa

def load_proteins_from_dill(*args,**kwargs):
    raise Exception("Removed")


class ProteinDataset(ProteinDatasetABC):
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
        load_native_ss: bool = False,
        ignore_seqs: bool = False,
    ):
        super(ProteinDataset, self).__init__(
            model_list=model_list,
            decoy_folder=decoy_folder,
            native_folder=native_folder,
            seq_folder=seq_folder,
            max_samples=-1,
            raise_exceptions=raise_exceptions,
            shuffle=shuffle,
        )
        self.crop_len = crop_len
        self.feat_gen = feat_gen
        self.atom_tys = default(atom_tys, self.default_atom_tys())
        self.augment_fn = augment_fn
        self.filter_fn, self.transform_fn = filter_fn, transform_fn
        self.get_ptn = lambda pdb, seq, lss=True: Protein.FromPDBAndSeq(
            pdb,
            seq,
            self.atom_tys,
            load_ss=lss,
            ignore_res_ids=ignore_res_ids,
        )
        self.load_native_ss = load_native_ss
        self.load_sec_structure = load_sec_structure
        self.ignore_seqs = ignore_seqs
        self.replica = 0

    def get_item_from_pdbs_n_seq(
        self,
        seq_path: Optional[Union[str, List[str]]],
        decoy_pdb_path: Optional[Union[str, List[str]]],
        native_pdb_path: Optional[Union[str, List[str]]],
    ) -> Optional[ModelInput]:
        """Load data given native and decoy pdb paths and sequence path"""
        seq_paths, decoy_pdbs, native_pdbs = map(cast_list, (seq_path, decoy_pdb_path, native_pdb_path))
        if exists(decoy_pdbs[0]):
            assert len(decoy_pdbs) == len(native_pdbs), f"{decoy_pdbs}\n{native_pdbs}"
        seq_paths = seq_paths if exists(seq_paths[0]) else seq_paths * len(native_pdbs)
        decoy_pdbs = decoy_pdbs if exists(decoy_pdbs[0]) else native_pdbs

        decoy_seqs, native_seqs, decoy_proteins, native_proteins = [], [], [], []
        if not native_pdbs[0].endswith(".dill"):
            for seq_path, decoy_pdb, native_pdb in zip(seq_paths, decoy_pdbs, native_pdbs):
                try:
                    decoy_seqs.append(self.safe_load_sequence(seq_path, decoy_pdb))
                    if self.ignore_seqs:
                        native_seqs.append(self.safe_load_sequence(seq_path, native_pdb))
                    else:
                        native_seqs.append(decoy_seqs[-1])
                except Exception as e:
                    print(f"[ERROR]: caught exception {e} loading sequences")
                    if self.raise_exceptions:
                        raise e
                    return None

            for native_pdb, decoy_pdb, decoy_seq, native_seq in zip(native_pdbs, decoy_pdbs, decoy_seqs, native_seqs):
                try:
                    _nat = self.get_ptn(native_pdb, native_seq, lss=self.load_native_ss)
                    _decoy = self.get_ptn(decoy_pdb, decoy_seq, lss=self.load_sec_structure)
                    if not self.ignore_seqs:
                        _nat, _decoy = restrict_protein_to_aligned_residues(_nat, _decoy)
                    native_proteins.append(_nat)
                    decoy_proteins.append(_decoy)
                except Exception as e:
                    print(f"[ERROR] caught exception {e} loading proteins")
                    if self.raise_exceptions:
                        raise e
                    return None
        else:
            try:
                native_pdb, decoy_pdb = native_pdbs[0], decoy_pdbs[0]
                native_proteins = load_proteins_from_dill(native_pdb, self.atom_tys)
                if native_pdb == decoy_pdb:
                    decoy_proteins = [x.clone() for x in native_proteins]
                else:
                    decoy_proteins = load_proteins_from_dill(decoy_pdb, self.atom_tys)
            except Exception as e:
                print(f"[ERROR] caught exception {e} loading proteins")
                if self.raise_exceptions:
                    raise e
                return None

        if len(native_proteins) == 2:
            # build complex
            n1, n2 = map(len, native_proteins)
            partition = [torch.arange(n1), torch.arange(n1, n1 + n2)]
            native_cas = list(map(lambda x: x["CA"], native_proteins))
            p1, p2 = get_dimer_spatial_crop(partition, native_cas, crop_len=self.crop_len)
            native = native_proteins[0].add_chain(native_proteins[1])
            decoy = decoy_proteins[0].add_chain(decoy_proteins[1])
            if len(native) != len(decoy):
                print(f"[WARNING] length mismatch (native,decoy) {len(native)},{len(decoy)}")
                return None
            native, decoy = map(lambda x: x.restrict_to([p1, p2]), (native, decoy))
            input_partition = [p1, p2]
        else:
            native, decoy = native_proteins[0], decoy_proteins[0]
            start, end = get_contiguous_crop(self.crop_len, len(native))
            decoy = decoy.crop(start, end)
            native = native.crop(start, end)
            input_partition = [torch.arange(start, end)]
        native.set_input_partition(input_partition)
        decoy.set_input_partition(input_partition)

        # HAck, change later! # TODO
        decoy.replica = self.replica
        native.replica = self.replica

        if not self.filter_fn(decoy, native):
            return None

        decoy, native = self.transform_fn(decoy, native)
        extra = self.augment(decoy, native)
        decoy, native = map(set_canonical_coords_n_masks, (decoy, native))

        return ModelInput(
            decoy=decoy,
            native=native,
            input_features=self.feat_gen.generate_features(
                decoy,
                extra=extra,
            ),
            extra=extra,
        )

    def augment(self, decoy_protein: Protein, native_protein: Protein) -> Any:  # noqa
        """Override in subclass to augment model input with extra informaiton"""
        return self.augment_fn(decoy_protein, native_protein)  # noqa
