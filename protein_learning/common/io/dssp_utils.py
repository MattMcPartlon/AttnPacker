import os
from typing import List, Tuple

import Bio  # noqa
import torch
from Bio.PDB.DSSP import DSSP  # noqa
from torch import Tensor

from protein_learning.common.global_constants import get_logger
from protein_learning.common.helpers import default
from protein_learning.common.io.pdb_utils import map_seq_to_pdb, get_structure

logger = get_logger(__name__)

SS_REPLACE_KEYS = [('-', 'C'), ('I', 'C'), ('T', 'C'), ('S', 'C'), ('G', 'H'), ('B', 'E')]
SS_KEY_MAP = {"C": 0, "H": 1, "E": 2}


def get_dssp(pdbfile: str, name: str = None) -> Bio.PDB.DSSP:
    """Extract dssp from pdb file"""
    name = default(name, os.path.basename(pdbfile)[:-4])
    structure = get_structure(pdbfile=pdbfile, name=name)
    dssp = DSSP(structure[0], pdbfile)
    return dssp


def get_aligned_dssp_keys(
        pdbfile,
        seq: str
) -> List[Tuple]:
    _, residue_list, *_ = map_seq_to_pdb(sequence=seq, pdbfile=pdbfile)
    return [res.get_full_id()[-2:] for res in residue_list]


def to_ss3(sec_structure):
    for (a, b) in SS_REPLACE_KEYS:
        sec_structure = sec_structure.replace(a, b)
    return sec_structure


def get_ss_from_pdb_and_seq(pdbfile: str, seq: str, return_loop_on_error: bool= True) -> str:

    try:
        dssp = get_dssp(pdbfile=pdbfile)
        dssp_keys = get_aligned_dssp_keys(pdbfile, seq)
        sec_structure, n_missed = "", 0
        for res_key in dssp_keys:
            if res_key in dssp:
                sec_structure += dssp[res_key][2]
            else:
                n_missed += 1
                sec_structure += "C"
        if n_missed > max(10, int(len(seq) * 0.1)):
            raise Exception(f"missed {n_missed}/{len(seq)} residues loading sec. structure")
        return to_ss3(sec_structure)

    except Exception as e:
        #act_set, exp_set = set(dssp.keys()), set(dssp_keys)
        #print("[ERROR]: got error {e} loading secondary structure")
        #print(f"actual/expected key set lens : {len(act_set)}/{len(act_set)}")
        #print(f"elements in act not in exp : {act_set - exp_set}")
        #print(f"elements in exp not in act : {exp_set - act_set}")
        if not return_loop_on_error:
            raise e
        return "".join(["C"]*len(seq))


def encode_sec_structure(sec_structure: str) -> Tensor:
    return torch.tensor([SS_KEY_MAP[s] for s in sec_structure])
