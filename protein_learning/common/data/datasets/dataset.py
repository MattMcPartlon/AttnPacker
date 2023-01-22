"""Baseline protein dataset"""
import torch

torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.data import Dataset, DataLoader
from abc import abstractmethod
from protein_learning.common.data.data_types.model_input import ModelInput
from typing import Optional, Tuple, List, Union
import os
import numpy as np
from protein_learning.common.helpers import exists, default
from protein_learning.common.io.pdb_utils import (
    extract_atom_coords_n_mask_tensors,
    extract_pdb_seq_from_pdb_file,
)
from protein_learning.common.io.sequence_utils import load_fasta_file
from torch import Tensor
from protein_learning.common.protein_constants import (
    ALL_ATOMS, BB_ATOMS, SC_ATOMS
)
from protein_learning.common.global_constants import get_logger
import numpy
import random

g = torch.Generator()
g.manual_seed(0)  # noqa

logger = get_logger(__name__)


def seed_worker(worker_id):  # noqa
    worker_seed = torch.initial_seed() % 2 ** 32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


def load_model_list(list_path, max_to_load=-1) -> List[List[str]]:
    """Loads a model list"""
    logger.info(f"Loading model list {list_path}")
    all_data = []
    with open(list_path, 'r') as f:
        for i, line in enumerate(f):
            targets = line.strip().split()
            if len(targets) > 0 and len(targets[0]) > 1:
                dat = [t[:-4] if t.endswith(".pdb") else t for t in targets]
                all_data.append(dat)
            if i > max_to_load > 0:
                break
    logger.info(f"Finished loading model list {list_path}")
    return all_data


def item(x: Union[str, List[str]]) -> Union[str, List[str]]:
    """Return first item in list if len==1 else whole list."""
    if isinstance(x, str):
        return x
    return x if len(x) > 1 else x[0]


class ModelList:
    """Model List"""

    def __init__(
            self,
            model_list_path: str,
            native_folder: str,
            decoy_folder: str,
            seq_folder: str
    ):
        self.model_list = load_model_list(model_list_path)
        self.native_fldr, self.decoy_fldr, self.seq_fldr = \
            native_folder, decoy_folder, seq_folder
        self.idxs = np.arange(len(self.model_list))

    def shuffle(self):
        """Shuffles the model list"""
        np.random.shuffle(self.idxs)

    @staticmethod
    def _get_path(entry: List[str], fldr: str, exts: Optional[List]
                  ) -> Union[str, List[str]]:
        """ge pdb path for list entries with source folder fldr"""
        exts = default(exts, [])
        target_paths = []
        for target in entry:
            added = False
            if target.lower() == "none":
                added = True
                target_paths.append(None)
            for tgt in [target] + [f"{target}.{ext}" for ext in exts]:
                path = os.path.join(fldr, tgt)
                if os.path.exists(path):
                    added = True
                    target_paths.append(path)
            if not added:
                raise Exception(f"no pdb path found for\n"
                                f"entry : {entry}\ndir: {fldr}")
        return item(target_paths)

    def _get_native_pdb_path(self, entry) -> Union[str, List[str]]:
        """get native pdb path for list entry"""
        return self._get_path(entry, self.native_fldr, ["pdb"])

    def _get_decoy_pdb_path(self, entry) -> str:
        """get decoy pdb path for list entry"""
        return self._get_path(entry, self.decoy_fldr, ["pdb"])

    def _get_seq_path(self, entry) -> Optional[str]:
        """get sequence path for list entry"""
        if not exists(self.seq_fldr):
            return None
        return self._get_path(entry, self.seq_fldr, ["fasta"])

    def get_entry(self, idx):
        """Gets given list entry"""
        return self.model_list[self.idxs[idx]]

    def __getitem__(self, idx: int
                    ) -> Tuple[
        Optional[Union[str, List[str]]],
        Optional[str],
        Optional[str],
        List[Exception]
    ]:
        entry = self.model_list[self.idxs[idx]]
        native_pdb, decoy_pdb, seq_path = None, None, None
        exceptions = []

        try:
            native_pdb = self._get_native_pdb_path(entry)
        except Exception as e:
            exceptions.append(e)
        try:
            decoy_pdb = self._get_decoy_pdb_path(entry)
        except Exception as e:
            exceptions.append(e)
        try:
            seq_path = self._get_seq_path(entry)
        except Exception as e:  # noqa
            exceptions.append(e)
        return native_pdb, decoy_pdb, seq_path, exceptions

    def __len__(self):
        return len(self.model_list)


class ProteinDatasetABC(Dataset):
    """Dataset base class"""

    def __init__(
            self,
            model_list: str,
            decoy_folder: str,
            native_folder: str,
            seq_folder: str,
            max_samples: int,
            raise_exceptions: bool,
            crop_len: int = -1,
            shuffle: bool = True,
    ):
        super(ProteinDatasetABC, self).__init__()
        self.max_samples = max_samples
        self.model_list = ModelList(
            model_list,
            native_folder=native_folder,
            decoy_folder=decoy_folder,
            seq_folder=seq_folder
        )
        if shuffle:
            self.shuffle()
        self.raise_exceptions = raise_exceptions
        self.crop_len = crop_len
        self.native_folder, self.decoy_folder = native_folder, decoy_folder

    def __len__(self):
        return len(self.model_list)

    def __getitem__(self, idx) -> Optional[ModelInput]:
        native_pdb, decoy_pdb, seq, exceptions = self.model_list[idx]

        # optionally raise exceptions
        if len(exceptions) > 0:
            msgs = [str(e) for e in exceptions]
            logger.warn(f"WARNING: caught exceptions {msgs} loading data")
            print(f"[WARNING]: caught exceptions {msgs} loading data")
            if self.raise_exceptions:
                raise exceptions[0]

        try:
            return self.get_item_from_pdbs_n_seq(
                seq_path=seq,
                decoy_pdb_path=decoy_pdb,
                native_pdb_path=native_pdb,
            )
        except Exception as e:  # noqa
            if self.raise_exceptions:
                raise e
            print(f"[WARNING]: Got exception : {e} in dataloader abc")
            return None

    def shuffle(self) -> None:
        """Shuffle the dataset"""
        self.model_list.shuffle()

    @abstractmethod
    def get_item_from_pdbs_n_seq(
            self,
            seq_path: Optional[Union[str, List[str]]],
            decoy_pdb_path: Optional[Union[str, List[str]]],
            native_pdb_path: Optional[Union[str, List[str]]],
    ) -> ModelInput:
        """Load data given native and decoy pdb paths and sequence path"""
        pass

    @staticmethod
    def extract_atom_coords_n_mask(
            seq: str,
            pdb_path: str,
            atom_tys: Optional[List[str]] = None,
            return_res_ids: bool = True,
            remove_invalid_residues: bool = True,
            ignore_non_standard: bool = True,
    ) -> Union[Tuple[Tensor, Tensor, Tensor, str], Tuple[Tensor, Tensor, str]]:
        """Extract atom coordinates and mask from pdb file"""
        return extract_atom_coords_n_mask_tensors(
            seq=seq,
            pdb_path=pdb_path,
            atom_tys=atom_tys,
            remove_invalid_residues=remove_invalid_residues,
            return_res_ids=return_res_ids,
            ignore_non_standard=ignore_non_standard
        )

    @staticmethod
    def safe_load_sequence(
            seq_path: Optional[str],
            pdb_path: str,
            ignore_non_standard: bool = True
    ) -> str:
        """Loads sequence, either from fasta or given pdb file"""
        if exists(seq_path):
            return load_fasta_file(seq_path)
        pdbseqs, residueLists, chains = extract_pdb_seq_from_pdb_file(
            pdb_path,
            ignore_non_standard=ignore_non_standard
        )
        if len(pdbseqs) > 1:
            print(f"[WARNING] Multiple chains found for {pdb_path} ")
        return pdbseqs[0]

    @staticmethod
    def default_atom_tys(sc=True, bb=True):
        """default atom types"""
        if sc and bb:
            return ALL_ATOMS
        if sc:
            return SC_ATOMS
        if bb:
            return BB_ATOMS

    def get_dataloader(
            self,
            batch_size: int,
            shuffle: bool = True,
            num_workers: int = 2,
            **kwargs) -> DataLoader:
        """Get a data loader for this dataset"""
        dl_kwargs = dict(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=lambda samples: [x for x in samples if x is not None],
            drop_last=False,
            worker_init_fn=seed_worker,
        )
        dl_kwargs.update(kwargs)
        return DataLoader(**dl_kwargs)
