"""Utility functions for working with pdbs

Adapted from Raptorx3DModelling/Common/PDBUtils.py
"""
import os
from typing import Dict, List, Tuple, Union, Optional

import numpy as np
import torch
from Bio import pairwise2  # noqa
from Bio.PDB import PDBParser, MMCIFParser  # noqa
from Bio.PDB.Polypeptide import three_to_one, is_aa  # noqa
from Bio.PDB.Residue import Residue  # noqa
from Bio.PDB.Structure import Structure  # noqa
from Bio.SubsMat.MatrixInfo import blosum80 as BLOSUM80  # noqa
from torch import Tensor

from protein_learning.common.helpers import default
from protein_learning.common.io.select_atoms import SelectCG, SelectCB
from protein_learning.common.protein_constants import VALID_AA_3_LETTER, VALID_AA_1_LETTER

VALID_AA_1_LETTER_SET = set(VALID_AA_1_LETTER)
VALID_AA_3_LETTER_SET = set(VALID_AA_3_LETTER)
VALID_AA_3_LETTER_SET_EXTENDED = set(list(VALID_AA_3_LETTER) + ["C3Y", "ASX", "GLX", "UNK"])


# getting errors for non-standard AAs
class SubMat:
    """Wrapper Around BLOSUM80 subst. matrix for handling
    non-standard residue types
    """

    @staticmethod
    def subst_matrix_get(aa1, aa2):
        """Heavy penalty for non-standard aas"""
        if (aa1, aa2) in BLOSUM80:
            val = BLOSUM80[(aa1, aa2)]
        elif (aa2, aa1) in BLOSUM80:
            val = BLOSUM80[(aa2, aa1)]
        else:
            val = -1000
        return val

    def __contains__(self, *item):
        return True

    def __getitem__(self, item):
        return self.subst_matrix_get(*item)


def replace_non_standard(x):
    x = x.upper()
    if x in VALID_AA_3_LETTER_SET:
        return x
    elif x == "GLX":
        return "GLU"
    elif x == "ASX":
        return "ASP"
    elif x == "C3Y":
        return "CYS"
    return "ALA"


class PDBExtractException(Exception):
    pass


def get_structure_parser(pdb_file: str) -> Union[PDBParser, MMCIFParser]:
    """gets a parser for the underlying pdb structure

    :param pdb_file: the file to obtain a structure parser for
    :return: structure parser for pdb input
    """
    is_pdb, is_cif = [pdb_file.endswith(x) for x in (".pdb", ".cif")]
    assert is_pdb or is_cif, f"ERROR: pdb file must have .cif or .pdb type, got {pdb_file}"
    return MMCIFParser(QUIET=True) if is_cif else PDBParser(QUIET=True)


# @lru_cache(maxsize=4)
def get_structure(pdbfile: str, name: str = None):
    """Get BIO.Structure object"""
    parser = get_structure_parser(pdbfile)
    name = default(name, os.path.basename(pdbfile))
    return parser.get_structure(name, pdbfile)


def extract_pdb_seq_from_residues(
    residues: List[Residue],
    ignore_non_standard=True,
) -> Tuple[str, List[Residue]]:
    """
    extract a list of residues with valid 3D coordinates excluding non-standard amino acids
    returns the amino acid sequence as well as a list of residues with standard amino acids
    """
    valid_list = VALID_AA_3_LETTER_SET if ignore_non_standard else VALID_AA_3_LETTER_SET_EXTENDED
    fltr = lambda r: is_aa(r, standard=ignore_non_standard) and r.get_resname().upper() in valid_list
    residueList = list(filter(fltr, residues))
    res_names = list(map(lambda x: replace_non_standard(x.get_resname()), residueList))
    pdbseq = "".join(list(map(lambda x: three_to_one(x), res_names)))
    return pdbseq, residueList


def extract_pdb_seq_by_chain(structure: Structure, ignore_non_standard=True) -> Tuple[List, ...]:
    """extract sequences and residue lists for each chain
    :return: pdbseqs, residue lists and also the chain objects
    """
    model = structure[0]
    pdbseqs, residueLists, chains = [], [], []
    for chain in model:
        residues = list(chain.get_residues())
        pdbseq, residueList = extract_pdb_seq_from_residues(residues, ignore_non_standard=ignore_non_standard)
        pdbseqs.append(pdbseq)
        residueLists.append(residueList)
        chains.append(chain)
    return pdbseqs, residueLists, chains


def extract_pdb_seq_from_pdb_file(
    pdbfile: str, name: Optional[str] = None, ignore_non_standard=True
) -> Tuple[List, ...]:
    """Extract sequences and residue lists from pdbfile for all the chains
    :param pdbfile: pdb file to extract from
    :param name: name for bio.pdb structure
    :param ignore_non_standard: ignore non-standard residue types
    :return: lists of : pdbseqs, residueLists, chains from each chain in input pdb file
    """
    name = default(name, os.path.basename(pdbfile)[:-4])
    structure = get_structure(pdbfile=pdbfile, name=name)
    return extract_pdb_seq_by_chain(structure, ignore_non_standard)


def calc_num_mismatches(alignment) -> Tuple[int, int]:
    """Calculate number of mismatches in sequence alignment

    :param alignment: sequence alignment(s)
    :return: number of mismatches in sequence alignment
    """
    S1, S2 = alignment[:2]
    numMisMatches = np.sum([a != b for a, b in zip(S1, S2) if a != "-" and b != "-" and a != "X" and b != "X"])
    numMatches = np.sum([a == b for a, b in zip(S1, S2) if a != "-" and a != "X"])
    return int(numMisMatches), int(numMatches)


def alignment_to_mapping(alignment) -> List:
    """Convert sequence alignment to residue-wise mapping

    :param alignment: sequence alignment
    :return: mapping
    """
    S1, S2 = alignment[:2]
    # convert an aligned seq to a binary vector with 1 indicates aligned and 0 gap
    y = np.array([1 if a != "-" else 0 for a in S2])
    # get the position of each residue in the original sequence, starting from 0.
    ycs = np.cumsum(y) - 1
    np.putmask(ycs, y == 0, -1)
    # map from the 1st seq to the 2nd one. set -1 for an unaligned residue in the 1st sequence
    mapping = [y0 for a, y0 in zip(S1, ycs) if a != "-"]
    return mapping


def map_seq_to_residue_list(
    sequence: str, pdbseq: str, residueList: List[Residue]
) -> Tuple[Optional[List], Optional[int], Optional[int]]:
    """map one query sequence to a list of PDB residues by sequence alignment
    pdbseq and residueList are generated by ExtractPDBSeq or ExtractPDBSeqByChain from a PDB file
    :param sequence:
    :param pdbseq:
    :param residueList:
    :return: seq2pdb mapping, numMisMatches and numMatches
    """
    # here we align PDB residues to query sequence instead of query to PDB residues
    alignments = pairwise2.align.localds(pdbseq, sequence, SubMat(), -5, -0.2)
    if not bool(alignments):
        return None, None, None

    # find the alignment with the minimum difference
    diffs = []
    for alignment in alignments:
        mapping_pdb2seq, diff = alignment_to_mapping(alignment), 0
        for current_map, prev_map, current_residue, prev_residue in zip(
            mapping_pdb2seq[1:], mapping_pdb2seq[:-1], residueList[1:], residueList[:-1]
        ):
            # in principle, every PDB residue with valid 3D coordinates shall appear in the query sequence.
            # otherwise, apply a big penalty
            if current_map < 0:
                diff += 10
                continue

            if prev_map < 0:
                continue

            # calculate the difference of sequence separation in both the PDB seq and the query seq
            # the smaller, the better
            current_id = current_residue.get_id()[1]
            prev_id = prev_residue.get_id()[1]
            id_diff = max(1, current_id - prev_id)
            map_diff = current_map - prev_map
            diff += abs(id_diff - map_diff)

        numMisMatches, numMatches = calc_num_mismatches(alignment)
        diffs.append(diff - numMatches)

    diffs = np.array(diffs)
    alignment = alignments[diffs.argmin()]

    numMisMatches, numMatches = calc_num_mismatches(alignment)

    # map from the query seq to pdb
    mapping_seq2pdb = alignment_to_mapping((alignment[1], alignment[0]))

    return mapping_seq2pdb, numMisMatches, numMatches


def map_seq_to_pdb(
    sequence,
    pdbfile,
    maxMisMatches=None,
    minMatchRatio=0.5,
    ignore_non_standard=True,
):
    """Maps sequence to a pdb file, selecting the sequence from chain with best match.
    :param sequence: sequence (string)
    :param pdbfile: pdb file to map to
    :param maxMisMatches: max allowed number of mismatches
    :param minMatchRatio: the minimum ratio of matches on the query sequence
    :return: eq2pdb mapping, the pdb residue list, the pdb seq, the pdb chain,
     the number of mismtaches and matches
    """
    maxMisMatches = max(5, default(maxMisMatches, int(0.05 * len(sequence))))
    if not os.path.isfile(pdbfile):
        raise Exception("ERROR: the pdb file does not exist: ", pdbfile)

    # extract PDB sequences by chains
    pdbseqs, residueLists, chains = extract_pdb_seq_from_pdb_file(pdbfile, ignore_non_standard=ignore_non_standard)

    bestPDBSeq = None
    bestMapping = None
    bestResidueList = None
    bestChain = None
    minMisMatches = np.iinfo(np.int32).max
    maxMatches = np.iinfo(np.int32).min

    for pdbseq, residueList, chain in zip(pdbseqs, residueLists, chains):
        seq2pdb_mapping, numMisMatches, numMatches = map_seq_to_residue_list(sequence, pdbseq, residueList)
        if seq2pdb_mapping is None:
            continue
        if maxMatches < numMatches:
            # if numMisMatches < minMisMatches:
            bestMapping = seq2pdb_mapping
            minMisMatches = numMisMatches
            maxMatches = numMatches
            bestResidueList = residueList
            bestPDBSeq = pdbseq
            bestChain = chain

    if minMisMatches > maxMisMatches:
        print(
            f"ERROR: there are  {minMisMatches} mismatches between"
            f" the query sequence and PDB file: {pdbfile}\n"
            f"num residue : {len(sequence)}"
        )
        return None, None, None, None, None, None

    if maxMatches < min(30.0, minMatchRatio * len(sequence)):
        print(
            "ERROR: there are only  {maxMatches} matches on query sequence,"
            f" less than  {minMatchRatio} of its length from PDB file: {pdbfile}"
        )
        return None, None, None, None, None, None

    return bestMapping, bestResidueList, bestPDBSeq, bestChain, minMisMatches, maxMatches


def extract_coords_by_mapping(sequence, seq2pdb_mapping, residueList, atoms, bUseAlternativeAtoms=True):
    """Extract coordinates from residue list by sequence mapping
    :param sequence:
    :param seq2pdb_mapping:
    :param residueList:
    :param atoms:
    :param bUseAlternativeAtoms:
    :return:
    """
    neededAtoms = [a.upper() for a in atoms]
    numInvalidAtoms = dict()
    for atom in atoms:
        numInvalidAtoms[atom] = 0

    atomCoordinates = []
    for i, j in zip(list(range(len(sequence))), seq2pdb_mapping):
        coordinates = dict()
        if j < 0:
            for atom in neededAtoms:
                coordinates[atom] = None
            # coordinates['valid'] = False
            atomCoordinates.append(coordinates)
            continue

        res = residueList[j]
        try:
            AAname = three_to_one(res.get_resname())
        except:
            AAname = three_to_one(replace_non_standard(res.get_resname()))

        for atom in neededAtoms:
            coordinates[atom] = None
            if atom == "CG":
                a = SelectCG(AAname, bUseAlternativeCG=bUseAlternativeAtoms)
            elif atom == "CB":
                a = SelectCB(AAname, bUseAlternativeCB=bUseAlternativeAtoms)
            else:
                a = atom

            for atom2 in res:
                if atom2.get_id().upper() == a:
                    coordinates[atom] = atom2.get_vector()
                    break

            if coordinates[atom] is None:
                numInvalidAtoms[atom] += 1

        atomCoordinates.append(coordinates)

    return atomCoordinates, numInvalidAtoms


def extract_seq_from_pdb_n_chain_id(pdbfile: str, chain_id: str, name: str = None, ignore_non_std=True) -> str:
    """Extract the sequence of a specific pdb chain"""

    name = default(name, os.path.basename(pdbfile)[:-4])
    structure = get_structure(pdbfile=pdbfile, name=name)
    model = structure[0]
    chain_ids = []
    for chain in model:
        chain_ids.append(chain.get_id())
        residues = chain.get_residues()
        if chain.get_id() == chain_id:
            pdbseq, _ = extract_pdb_seq_from_residues(residues, ignore_non_standard=ignore_non_std)
            return pdbseq
    raise PDBExtractException(f"No chain with id {chain_id}, found chains: {[chain_ids]}")


def extract_coords_from_seq_n_pdb(
    sequence,
    pdbfile,
    atoms,
    maxMisMatches=5,
    minMatchRatio=0.5,
    bUseAlternativeAtoms=True,
    ignore_non_standard=True,
):
    """
    :param sequence: sequence to map from
    :param pdbfile: pdb file to extract from
    :param atoms: atom types to extract coords for.
    :param maxMisMatches: maximum number of allowed mismatches in seq to pdb_seq alignment
    :param minMatchRatio: minimum allowed match ratio
    :param bUseAlternativeAtoms: use alternative atoms (e.g. CA for glycine CB)
    :return: tuple containining:
        (1) atom_coordinates : atom coordinates for each residue in the input sequence
        (2) pdb_seq : pdb sequence for chain which was mapped to
        (3) num_mismatches: number of mismatches in alignment
        (4) num_matches: number of matches in alignment
    """
    out = map_seq_to_pdb(
        sequence=sequence,
        pdbfile=pdbfile,
        maxMisMatches=maxMisMatches,
        minMatchRatio=minMatchRatio,
        ignore_non_standard=ignore_non_standard,
    )
    seq2pdb_mapping, residueList, pdbseq, chain, num_mismatches, num_matches = out
    if seq2pdb_mapping is None:
        return None, None, None, None, None
    residueList = list(residueList)
    atom_coordinates, _ = extract_coords_by_mapping(
        sequence, seq2pdb_mapping, residueList, atoms=atoms, bUseAlternativeAtoms=bUseAlternativeAtoms
    )
    res_ids = [r.get_id()[1] for r in residueList]
    return atom_coordinates, pdbseq, num_mismatches, num_matches, res_ids


def extract_atom_coords_n_mask_tensors(
    seq: Optional[str],
    pdb_path: str,
    atom_tys: List[str],
    warn: bool = True,
    return_res_ids: bool = True,
    remove_invalid_residues: bool = True,
    chain_id: Optional[str] = None,
    ignore_non_standard=True,
) -> Union[Tuple[Tensor, Tensor, Tensor, str], Tuple[Tensor, Tensor, str]]:
    """Extracts

    :param seq: sequence to map from
    :param pdb_path: pdb path to extract coordinates from
    :param atom_tys: atom types to extract coordinates for
    :param warn: warn if alignment has more than 5 mismatches
    :param return_res_ids: whether to return pdb residue ids from mapping
    :param remove_invalid_residues: whether to remove residues with invalid coordinates
    :param chain_id: id of the chain to extract (used if no seq. is given)
    :return: Tuple containing
        (1) coords: Tensor of shape (n,a,3) containing atom coordinates for
        each atom type (1..a) and each residue 1..n in the input sequence
        (2) mask: Tensor of shape (n,a) indicating whether valid coordinates
        were obtained for atom types for each atom in atom_tys (1..a) and each
        residue (1..n) in the input sequence
    """
    assert seq is not None, " must provide sequence to extract coords and masks"

    out = extract_coords_from_seq_n_pdb(
        sequence=seq,
        pdbfile=pdb_path,
        atoms=atom_tys,
        ignore_non_standard=ignore_non_standard,
    )

    atom_coords, pdbseq, numMisMatches, numMatches, res_ids = out
    atom_mask = None
    if atom_coords is not None:
        if warn and numMisMatches > 5:
            print(f"WARNING: got {numMisMatches} mismatches mapping seq. to pdb")
        atom_coords, atom_mask = _get_coord_n_mask_tensors(atom_coords, atom_tys)
    # if remove_invalid_residues and atom_coords is not None:
    #    if remove_invalid_residues:
    #        tmp = torch.any(atom_mask, dim=-1)
    #        atom_coords = atom_coords[tmp]
    #        atom_mask = atom_mask[tmp]
    #        seq = "".join([s for i, s in enumerate(seq) if tmp[i]])

    if return_res_ids:
        if res_ids is not None:
            res_ids = torch.tensor(res_ids).long()
            res_ids = res_ids - res_ids[0]
        if res_ids is None or torch.any(res_ids < 0):
            print("[WARNING] : bad residue ids")
            return [None] * 4
        return atom_coords, atom_mask, res_ids, seq
    return atom_coords, atom_mask, seq


def _get_coord_n_mask_tensors(atom_coords: List[Dict[str, np.ndarray]], atom_tys: List[str]) -> Tuple[Tensor, Tensor]:
    """Retrieves coord and mask tensors from output of extract_coords_from_seq_n_pdb(...).

    :param atom_coords: List of dictionaries. each dict mapping from atom type to atom coordinates.
    :param atom_tys: the atom types to extract coordinates for.
    :return: Tuple containing
        (1) coords: Tensor of shape (n,a,3) containing atom coordinates for
        each atom type (1...a) and each residue 1..n in the input sequence.
        (2) mask: Tensor of shape (n,a) indicating whether valid coordinates
        were obtained for atom types for each atom in atom_tys (1..a) and each
        residue (1...n) in the input sequence.
    """
    n_res, n_atoms = len(atom_coords), len(atom_tys)
    coords, mask = torch.zeros(n_res, n_atoms, 3), torch.zeros(n_res, n_atoms)
    for i, res in enumerate(atom_coords):
        for atom_pos, atom_ty in enumerate(atom_tys):
            if res[atom_ty] is None:
                continue
            coords[i, atom_pos] = torch.tensor([res[atom_ty][j] for j in range(3)])
            mask[i, atom_pos] = 1
    return coords, mask.bool()
