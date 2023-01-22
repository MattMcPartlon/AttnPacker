"""Input for protein-based learning model
"""
from __future__ import annotations

import inspect
import os

try:
    from functools import cached_property  # noqa
except:  # noqa
    from cached_property import cached_property  # noqa
from typing import Optional, Dict, List, Union, Any, Tuple

import torch
from torch import Tensor

from protein_learning.common.rigids import Rigids
from protein_learning.common.helpers import exists, safe_to_device, default
from protein_learning.common.io.pdb_utils import (
    extract_atom_coords_n_mask_tensors,
    extract_pdb_seq_from_pdb_file,
)
from protein_learning.common.io.sequence_utils import load_fasta_file
from protein_learning.common.io.pdb_io import write_pdb
from protein_learning.common.protein_constants import BB_ATOMS, SC_ATOMS, AA_TO_INDEX
from protein_learning.protein_utils.align.per_residue import impute_beta_carbon
from protein_learning.protein_utils.sidechain_utils import (
    get_symmetric_residue_keys_n_indices,
    per_residue_chi_indices_n_mask,
)
from protein_learning.protein_utils.align.kabsch_align import _calc_kabsch_rot_n_trans  # noqa
from protein_learning.protein_utils.sidechain_utils import align_symmetric_sidechains
from einops import rearrange  # noqa
from protein_learning.common.io.dssp_utils import get_ss_from_pdb_and_seq, encode_sec_structure
from protein_learning.common.io.extract_cdrs import extract_cdr_posns

BB_ATOM_SET, SC_ATOM_SET = set(BB_ATOMS), set(SC_ATOMS)


def check_consistent(chain: Chain):
    assert len(chain.seq) == chain.atom_masks.shape[0]
    assert chain.atom_coords.shape[:2] == chain.atom_masks.shape[:2]
    assert len(chain.atom_tys) == chain.atom_masks.shape[1]
    assert len(chain.res_ids) == len(chain.seq)
    if exists(chain.sec_struct):
        assert len(chain.sec_struct) == len(chain.seq)


class Chain:
    """Represents a protein (or batch of proteins)"""

    def __init__(
            self,
            atom_coords: Tensor,
            atom_masks: Tensor,
            atom_tys: List[str],
            seq: str,
            name: str,
            res_ids: Optional[Union[Tensor, List[Tensor]]] = None,
            sec_struct: Optional[str] = None,
    ):
        """
        :param atom_coords: (n,a,3) Tensor of coordinates
         for n residues and a atom types
        :param atom_masks: (n,a) Tensor storing True iff atom coord.
         for residue i, atom j is valid
        (e.g. indicator of whether atom_coords[i,j] is valid.
        :param atom_tys: String atom type for corresponding
         coordinate (must have length a)
        :param seq: protein primary sequence
        :param name: name of the protein
        :param res_ids: (Optional) mapping from residue i to
         residue position
         (res_ids[i] = position of residue i in chain)
        """

        self.atom_coords = atom_coords
        self.atom_masks = atom_masks
        self.atom_tys = atom_tys
        self.seq = seq
        self._name = name
        self.res_ids = default(
            res_ids,
            [torch.arange(len(seq), device=atom_coords.device)]
        )
        self._crop = None
        self._rigids = None
        self.sec_struct = sec_struct

    @classmethod
    def FromPDBAndSeq(
            cls,
            pdb_path: str,
            seq: str,
            atom_tys: Optional[List[str]],
            remove_invalid_residues: bool = False,
            ignore_non_standard: bool = True,
            load_ss: bool = True,
            missing_seq: bool = False
    ) -> Chain:
        """Generates a `Chain` object for a single chain given
        a pdb file and (optional) sequence.

        When sequence is provided, the (sub-sequence) of the chain which
        best matches the input sequence will be extracted from the PDB file.

        If no sequence is provided, then the first chain in the pdb file
        is used to construct the protein
        """
        if missing_seq:
            seq = safe_load_sequence(None, pdb_path)
        coords, mask, seq = extract_atom_coords_n_mask_tensors(
            seq=seq,
            pdb_path=pdb_path,
            atom_tys=atom_tys,
            remove_invalid_residues=remove_invalid_residues,
            ignore_non_standard=ignore_non_standard,
            return_res_ids=False
        )
        sec_struct, res_ids = None, torch.arange(len(seq))
        if load_ss and seq is not None:
            sec_struct = get_ss_from_pdb_and_seq(pdbfile=pdb_path, seq=seq)

        return cls(
            atom_coords=coords,
            atom_masks=mask,
            seq=seq,
            name=pdb_path,
            atom_tys=atom_tys,
            res_ids=res_ids,
            sec_struct=sec_struct,
        )

    def to(self, device: Any) -> Chain:
        """Places coords and mask on given device"""
        self.atom_coords = self.atom_coords.to(device)
        self.atom_masks = self.atom_masks.to(device)
        self.res_ids = safe_to_device(self.res_ids, device)
        return self

    def crop(self, start: int, end: int) -> Chain:

        """Crop the protein from start..end"""
        assert end <= len(self.seq)
        self.seq = self.seq[start:end]
        self.atom_coords = self.atom_coords[..., start:end, :, :].clone()
        self.atom_masks = self.atom_masks[..., start:end, :].clone()
        if exists(self.sec_struct):
            self.sec_struct = self.sec_struct[start:end]
        self._crop = (start, end)
        return self

    def clone(self) -> Chain:
        """Clone the Chain"""
        chain = Chain(
            atom_coords=self.atom_coords.detach().clone(),
            atom_masks=self.atom_masks.clone(),
            atom_tys=self.atom_tys,
            seq="".join([s for s in self.seq]),
            name=self._name,
            res_ids=self.res_ids.clone(),
            sec_struct=self.sec_struct,
        )
        chain._crop = self._crop
        return chain

    @property
    def device(self):
        """Get device where coord/mask/res_id's are stored"""
        return self.atom_masks.device

    @property
    def name(self):
        """protein's pdb"""
        name = os.path.basename(self._name)
        return name[:-4] if name.endswith(".pdb") else name

    @property
    def full_coords_n_mask(self) -> Tuple[Tensor, Tensor]:
        """Get full atom coordinates and corresponding atom masks"""
        return self.atom_coords, self.atom_masks

    @property
    def valid_residue_mask(self) -> Tensor:
        """mask for all residues with valid backbone atoms"""
        return torch.all(self.bb_atom_mask, dim=-1)

    @property
    def secondary_structure_encoding(self) -> Tensor:
        """Label encoding of secondary structure"""
        assert exists(self.sec_struct)
        return encode_sec_structure(self.sec_struct)

    @cached_property
    def atom_ty_set(self):
        """Set of atom types with valid coordinates for this protein"""
        return set(self.atom_tys)

    @cached_property
    def atom_positions(self) -> Dict[str, int]:
        """Mapping from atom type to atom position"""
        return {a: i for i, a in enumerate(self.atom_tys)}

    @cached_property
    def bb_atom_tys(self) -> List[str]:
        """List of BB atoms present in this protein"""
        return list(filter(lambda x: x in BB_ATOM_SET, self.atom_tys))

    @cached_property
    def sc_atom_tys(self) -> List[str]:
        """List of side chain atoms present in this protein"""
        sc_atoms = list(filter(lambda x: x in SC_ATOM_SET, self.atom_tys))
        return sc_atoms

    @property
    def bb_atom_coords(self):
        """Coordinates of backbone atoms"""
        return self.get_atom_coords(atom_tys=self.bb_atom_tys)

    @property
    def bb_atom_mask(self):
        """Coordiantes of sidechain atoms"""
        return self.get_atom_masks(atom_tys=self.bb_atom_tys)

    @property
    def sc_atom_coords(self):
        """Coordiantes of sidechain atoms"""
        return self.get_atom_coords(atom_tys=self.sc_atom_tys)

    @property
    def sc_atom_mask(self):
        """Coordiantes of sidechain atoms"""
        return self.get_atom_masks(atom_tys=self.sc_atom_tys)

    @cached_property
    def symmetric_sc_data(self):
        """gets residue positions and atom indices for residues with symmetric sidechains"""
        return get_symmetric_residue_keys_n_indices(self.seq)

    @cached_property
    def sc_dihedral_data(self):
        """Gets residue and sidechain positions for chi1-4 dihedrals"""
        mask = self.get_atom_masks(self.sc_atom_tys)
        return per_residue_chi_indices_n_mask(coord_mask=mask, seq=self.seq)

    def get_atom_coords(
            self,
            atom_tys: Optional[Union[str, List[str]]] = None,
            coords: Optional[Tensor] = None
    ) -> Tensor:
        """Gets the atom coordinates for the given atom types
        Returns:
            - Tensor of shape (...,n,3) if atom_tys is a string
            - Tensor of shape(...,n,a,3) if atom_tys is a list, where a is the
             number of atom_tys given.
            - All atom coordinates if atom_tys is None
        """
        coords = default(coords, self.atom_coords)
        if atom_tys is None:
            return coords
        atom_posns = self.atom_positions
        return coords[..., atom_posns[atom_tys], :] if isinstance(atom_tys, str) else \
            torch.cat([coords[..., atom_posns[ty], :].unsqueeze(-2) for ty in atom_tys], dim=-2)

    def get_atom_masks(
            self,
            atom_tys: Optional[Union[str, List[str]]] = None,
            mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Gets the atom masks for the given atom types

        Returns:
            - Tensor of shape (...,n) if atom_tys is a string
            - Tensor of shape(...,n,a) if atom_tys is a list, where a is the
             number of atom_tys given.
            - All atom coordinate mask if atom_tys is None
        """
        if atom_tys is None:
            return self.atom_masks
        atom_posns = self.atom_positions
        mask = default(mask, self.atom_masks)
        if not exists(atom_tys):
            return mask
        elif isinstance(atom_tys, str):
            return mask[..., atom_posns[atom_tys]]
        else:
            masks = [mask[..., atom_posns[ty]] for ty in atom_tys]
            return torch.cat([m.unsqueeze(-1) for m in masks], dim=-1)

    def get_atom_coords_n_masks(
            self,
            atom_tys: Optional[Union[str, List[str]]] = None,
            coords: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """Gets the atom coordinates and masks for the given atom types

        Returns:
            - Tuple of tensors with shapes (...,n,3), (...,n) if atom_tys is a string
            - Tuple of tensors with shapes (...,n,3), (...,n) if atom_tys is a list,
             where a is the number of atom types given.
            - All atom coordinates and all atom coord. masks if atom_tys is None
        """
        coords = self.get_atom_coords(atom_tys=atom_tys, coords=coords)
        masks = self.get_atom_masks(atom_tys=atom_tys)
        return coords, masks

    def align_symmetric_scs(
            self,
            target: Optional[Chain] = None,
            target_coords: Optional[Tensor] = None,
            overwrite: bool = True,
    ):
        """Aligns the sidechain atoms of this chain to those of the target chain.

        For residues with side-chain symmetries, the coordinates of atoms constituting those
        symmetries will be swapped to minimize side-chain RMSD between this protein and the
        target structure.
        """
        atom_tys = self.sc_atom_tys
        assert exists(target) ^ exists(target_coords)
        assert len(self.sc_atom_tys) == len(SC_ATOMS)
        assert (all([x == y for x, y in zip(atom_tys, SC_ATOMS)])), \
            f"ERROR: must have same order"
        target_sc_coords = target.get_atom_coords(atom_tys) \
            if exists(target) else target_coords
        target_sc_coords = target_sc_coords.squeeze(0)
        sc_coords = self.get_atom_coords(atom_tys)
        assert sc_coords.shape == target_sc_coords.shape
        sc_mask = self.get_atom_masks(atom_tys)
        sc_coords, _ = align_symmetric_sidechains(
            sc_coords.unsqueeze(0),
            target_sc_coords.unsqueeze(0),
            sc_mask.unsqueeze(0),
            self.symmetric_sc_data,
        )
        assert sc_coords.shape == self.sc_atom_coords.shape
        if overwrite:
            self.overwrite_coords(sc_coords.squeeze(), atom_tys)
        return sc_coords, target_coords, sc_mask

    def to_pdb(
            self,
            path: str,
            atom_tys: Optional[List[str]] = None,
            write_ss: bool = False,
            chain_label: str = "A",
            res_idxs: Optional[Tensor] = None,
    ) -> None:
        """saves this protein to a .pdb file"""
        assert len(res_idxs) == len(self)
        atom_tys = default(atom_tys, self.atom_tys)
        res_idxs,coords = default(res_idxs, self.res_ids), self.atom_coords
        res_idxs, coords = map(lambda x :x.detach().cpu().numpy(),(res_idxs,coords))
        coord_dicts, residue_indices, seq = [],[], ""
        for seq_idx, res_idx in enumerate(res_idxs):
            atoms = {}
            for atom in atom_tys:
                atom_idx = self.atom_positions[atom]
                if self.atom_masks[seq_idx, atom_idx]:
                    atoms[atom] = coords[seq_idx, atom_idx].tolist()
            if len(atoms) > 0:
                seq += self.seq[seq_idx]
                coord_dicts.append(atoms)
                residue_indices.append(res_idx)

            ss = None
            if exists(self.sec_struct) and write_ss:
                for i in chain_indices:
                    ss += self.sec_struct[i]
            write_pdb(coord_dicts, seq=_seq, out_path=path, chain_ids=chain_ids, ss=None, res_idxs=residue_indices)

    def impute_cb(self, override: bool = False, exists_ok: bool = False) -> Tuple[Tensor, Tensor]:
        """Impute CB atom position"""
        assert all([x in self.atom_ty_set for x in ["N", "CA", "C"]])
        coords = self.atom_coords
        n, a, c = coords.shape
        bb_coords, bb_mask = self.get_atom_coords_n_masks(atom_tys=["N", "CA", "C"], coords=coords)
        cb_mask, cb_coords = torch.all(bb_mask, dim=-1), torch.zeros(n, 1, 3, device=coords.device)
        cb_coords[cb_mask] = impute_beta_carbon(bb_coords[cb_mask]).unsqueeze(-2)
        if override:
            self.add_atoms(cb_coords, cb_mask.unsqueeze(-1), ["CB"], exists_ok=exists_ok)
        return cb_coords, cb_mask.unsqueeze(-1)

    def add_atoms(self, atom_coords: Tensor, atom_masks: Tensor, atom_tys: List[str],
                  exists_ok: bool = False) -> Chain:
        """Add atoms and respective coordinates to the protein"""
        atom_exists = any([atom_ty in self.atom_tys for atom_ty in atom_tys])
        if not exists_ok:
            assert not atom_exists
        assert atom_coords.ndim == self.atom_coords.ndim, f"{atom_coords.shape},{self.atom_coords.shape}"
        assert atom_masks.ndim == self.atom_masks.ndim, f"{atom_masks.shape},{self.atom_masks.shape}"
        assert atom_masks.shape[0] == self.atom_masks.shape[0], f"{atom_masks.shape[0]},{self.atom_masks.shape}"
        assert atom_coords.shape[0] == self.atom_coords.shape[0], f"{atom_coords.shape}, {self.atom_coords.shape}"

        new_atom_tys, curr_atom_tys = [x for x in self.atom_tys], self.atom_ty_set
        new_coords, new_masks = torch.clone(self.atom_coords), torch.clone(self.atom_masks)
        for i, atom in enumerate(atom_tys):
            if atom in curr_atom_tys:
                atom_pos = self.atom_positions[atom]
                new_coords[..., atom_pos, :] = atom_coords[..., i, :]
                new_masks[..., atom_pos] = atom_masks[..., i]
            else:
                new_atom_tys.append(atom)
                new_coords = torch.cat((new_coords, atom_coords[..., i, :].unsqueeze(-2)), dim=-2)
                new_masks = torch.cat((new_masks, atom_masks[..., i].unsqueeze(-1)), dim=-1)

        self.atom_coords = new_coords
        self.atom_masks = new_masks
        self.atom_tys = new_atom_tys
        self.__clear_cache__()
        return self

    def overwrite_coords(self, coords, atom_tys):
        """Overwrite coordinates for given atom types"""
        assert coords.shape[-2] == len(atom_tys)
        assert coords.ndim == self.atom_coords.ndim
        self._rigids = None
        for i, atom_ty in enumerate(atom_tys):
            atom_idx = self.atom_positions[atom_ty]
            self.atom_coords[..., atom_idx, :] = coords[..., i, :]

    def __clear_cache__(self):
        cache_keys = [
            name for name, value in inspect.getmembers(Chain)
            if isinstance(value, cached_property)
        ]
        for key in cache_keys:
            if key in self.__dict__:
                del self.__dict__[key]

    def __getitem__(self, atoms):
        return self.get_atom_coords(atoms)

    def __len__(self):
        return len(self.seq)

    def restrict_to(self, indices: Tensor) -> Chain:
        """Get protein restricted to given indices"""
        assert isinstance(indices, Tensor)
        sec_struct = None
        if exists(self.sec_struct):
            sec_struct = "".join([self.seq[i.item()] for i in indices]),
        return Chain(
            atom_coords=self.atom_coords[indices],
            atom_masks=self.atom_masks[indices],
            seq="".join([self.seq[i.item()] for i in indices]),
            atom_tys=self.atom_tys,
            name=self._name,
            res_ids=self.res_ids[indices],
            sec_struct=sec_struct,
        )

    @property
    def rigids(self) -> Rigids:
        """Get rigids for this protein"""
        if self._rigids is None:
            self._rigids = Rigids.RigidFromBackbone(self[["N", "CA", "C"]].unsqueeze(0))
        return self._rigids

    @cached_property
    def seq_encoding(self):
        """Encoding of protein sequence"""
        return torch.tensor([AA_TO_INDEX[r] for r in self.seq]).long().to(self.device)


def safe_load_sequence(seq_path: Optional[str], pdb_path: str, ignore_non_std: bool = True) -> str:
    """Loads sequence, either from fasta or given pdb file"""
    if exists(seq_path):
        return load_fasta_file(seq_path)
    pdbseqs, residueLists, chains = extract_pdb_seq_from_pdb_file(pdb_path, ignore_non_standard=ignore_non_std)
    return pdbseqs[0]
