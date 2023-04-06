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
from protein_learning.common.protein_constants import BB_ATOMS, SC_ATOMS, AA_TO_INDEX, ALL_ATOMS
from protein_learning.protein_utils.align.per_residue import impute_beta_carbon


from protein_learning.protein_utils.align.kabsch_align import _calc_kabsch_rot_n_trans  # noqa
from einops import rearrange  # noqa
from protein_learning.common.io.dssp_utils import get_ss_from_pdb_and_seq, encode_sec_structure
from protein_learning.common.io.extract_cdrs import extract_cdr_posns

BB_ATOM_SET, SC_ATOM_SET = set(BB_ATOMS), set(SC_ATOMS)


class Protein:
    """Represents a protein (or batch of proteins)"""

    def __init__(
        self,
        atom_coords: Tensor,
        atom_masks: Tensor,
        atom_tys: List[str],
        seq: str,
        name: str,
        res_ids: Optional[Union[Tensor, List[Tensor]]] = None,
        chain_ids: Optional[Tensor] = None,
        chain_indices: Optional[List[Tensor]] = None,
        chain_names: Optional[List[str]] = None,
        sec_struct: Optional[str] = None,
        cdrs: Optional[Dict] = None,
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
        :param chain_ids: (Optional) List of ID's for each input chain
        :param chain_indices: (Optional) List of tensors where,
        e.g. i=chain_indices[0][7] is the position of
        residue 7 in chain 0. i.e. atom_coords[i] = atom
        coordinates of this residue
        :param chain_names: (Optional) names to give input chains
        """
        assert atom_coords.shape[:2] == atom_masks.shape[:2]
        assert len(seq) == atom_masks.shape[0]
        assert len(atom_tys) == atom_masks.shape[1]
        assert atom_coords.shape[0] == len(seq), f"{atom_coords.shape},{len(seq)}"
        self.atom_coords = atom_coords
        self.atom_masks = atom_masks
        self.atom_tys = atom_tys
        self.seq = seq
        self._name = name
        if exists(res_ids):
            res_ids = [res_ids] if torch.is_tensor(res_ids) else res_ids
        self.res_ids = default(res_ids, [torch.arange(len(seq), device=atom_coords.device)])
        assert sum(map(len, self.res_ids)) == len(seq), f"{sum(map(len, self.res_ids))}, {len(seq)}"
        self.chain_indices = default(chain_indices, [torch.arange(len(seq), device=atom_coords.device)])
        assert sum(map(len, self.chain_indices)) == len(seq)
        self.chain_ids = default(chain_ids, torch.zeros(len(seq), device=atom_coords.device))
        assert len(self.chain_ids) == len(seq)
        self.crops, self.input_partition = None, self.chain_indices
        self.chain_names = default(chain_names, [self.name] * len(self.chain_indices))
        self._check_chain_indices()
        self._rigids = None
        if exists(sec_struct):
            assert len(sec_struct) == len(seq), f"{len(sec_struct)},{len(seq)}"
        self.sec_struct = sec_struct
        self.cdrs = cdrs  # dict with keys "heavy" and "light" mapping to 0-indexed cdr posns
        self.replica = 0

    def _check_chain_indices(self):
        if self.chain_indices is None:
            return
        to_set = lambda tnsr: set([x.item() for x in tnsr])
        idx_set = to_set(torch.cat(self.chain_indices).cpu())
        expected_idxs = to_set(torch.arange(len(self)))
        assert len(idx_set) == len(self), f"{len(idx_set)},{len(self)}"
        assert idx_set == expected_idxs, f"{self.chain_indices},{expected_idxs - idx_set}\n{idx_set - expected_idxs}"

    def chain_seq(self, chain_idx: int) -> str:
        """Sequence for chain at given index"""
        idxs = self.chain_indices[chain_idx]
        return "".join([self.seq[i] for i in idxs.cpu().numpy()])

    @classmethod
    def make_antibody(
        cls,
        light: str,
        heavy: str,
        antigen: Optional[str],
        atom_tys: List[str],
        load_ss=True,
        seqs: Optional[List[str]] = None,
    ):
        """Make an antibody (and possibly add antigen as separate chain

        All pdb params (light, heavy, and antigen) must be of form
        <name>_<CHAIN_ID>.pdb, and must contain only a single chain
        """
        seqs = default(seqs, [None] * 3)
        _heavy, _light, _ag = heavy, light, antigen
        assert heavy.endswith(".pdb"), f"{light},{heavy}"
        heavy, light, ag = map(
            lambda x: Protein.FromPDBAndSeq(
                pdb_path=x[0],
                seq=x[1],
                atom_tys=atom_tys,
                remove_invalid_residues=False,
                ignore_non_standard=True,
                ignore_res_ids=True,
                load_ss=load_ss,
                missing_seq=not exists(x[1]),
            )
            if exists(x[0])
            else None,
            zip((_heavy, _light, _ag), (seqs)),
        )
        # get cdr domain defs for heavy/light chains
        hc_len, lc_len, ag_len = len(heavy), 0, 0
        h_chain, l_chain, ag_chain = _heavy[-5], None, None
        if exists(light):
            lc_len = len(light)
            l_chain = _light[-5]
            try:
                light_cdrs = extract_cdr_posns(_light, [], [l_chain])[1][l_chain]
            except:
                light_cdrs = extract_cdr_posns(_light, [], [l_chain.lower()])[1][l_chain.lower()]
            light_atom_coords = light.atom_coords
            light_seq, light_ss = light.seq, light.sec_struct
            light_atom_masks = light.atom_masks
            light_res_ids = light.res_ids[0]
            light_cdrs = [(s + len(heavy), e + len(heavy)) for (s, e) in light_cdrs]
            light_name = light.name
        else:
            light_cdrs = [(-1, -1) for _ in range(3)]
            light_atom_coords, light_seq, light_ss = torch.ones(0), "", ""
            light_atom_masks = torch.ones(0)
            light_res_ids = torch.ones(0)
            light_name = "none"
        try:
            heavy_cdrs = extract_cdr_posns(_heavy, [h_chain], [])[0][h_chain]
        except:
            heavy_cdrs = extract_cdr_posns(_heavy, [h_chain.lower()], [])[0][h_chain.lower()]
        # build single chain for antibody
        atom_coords = torch.cat((heavy.atom_coords, light_atom_coords), dim=0)
        atom_masks = torch.cat((heavy.atom_masks, light_atom_masks), dim=0)
        res_ids = [torch.cat((heavy.res_ids[0], light_res_ids))]
        sec_struct = None
        if exists(heavy.sec_struct):
            heavy_ss = [x for x in heavy.sec_struct]
            for s, e in heavy_cdrs:
                for i in range(e - s + 1):
                    heavy_ss[s + i] = "C"
            sec_struct = "".join(heavy_ss) + light_ss
        ab = Protein(
            atom_coords=atom_coords,
            atom_masks=atom_masks.bool(),
            seq=heavy.seq + light_seq,
            atom_tys=atom_tys,
            name=f"H_{heavy.name}_L_{light_name}",
            res_ids=res_ids,
            chain_names=[f"H_{heavy.name}_L_{light_name}"],
            sec_struct=sec_struct,
            cdrs=dict(heavy=heavy_cdrs, light=light_cdrs),
        )
        if exists(ag):
            ag_chain, ag_len = _ag[-5], len(ag)
        ab = ab.add_chain(ag) if exists(ag) else ab
        ab.chain_labels = (h_chain, l_chain, ag_chain)
        ab.chain_lens = (hc_len, lc_len, ag_len)
        return ab

    @property
    def chain_1_name(self):
        assert self.is_complex
        if self.is_antibody:
            return self.chain_names[0][2:6]
        return self.chain_names[0][:4]

    @classmethod
    def FromPDB(cls, pdb_path: str) -> Protein:
        return Protein.FromPDBAndSeq(pdb_path=pdb_path, seq=None, missing_seq=True, atom_tys=ALL_ATOMS, load_ss=False)

    @classmethod
    def FromPDBAndSeq(
        cls,
        pdb_path: str,
        seq: str,
        atom_tys: Optional[List[str]],
        remove_invalid_residues: bool = False,  # TODO
        ignore_non_standard: bool = True,
        load_ss: bool = True,
        ignore_res_ids: bool = True,  # TODO: made change
        missing_seq: bool = False,
    ) -> Protein:
        """Generates a `Protein` object for a single chain given
        a pdb file and (optional) sequence.

        When sequence is provided, the (sub-sequence) of the chain which
        best matches the input sequence will be extracted from the PDB file.

        If no sequence is provided, then the first chain in the pdb file
        is used to construct the protein
        """
        # TODO: ss is causing errors since it may not be same len as coords/seq
        if missing_seq:
            seq = safe_load_sequence(None, pdb_path)
        coords, mask, seq = extract_atom_coords_n_mask_tensors(
            seq=seq,
            pdb_path=pdb_path,
            atom_tys=atom_tys,
            remove_invalid_residues=remove_invalid_residues,
            ignore_non_standard=ignore_non_standard,
            return_res_ids=False,  # TODO
        )
        sec_struct, res_ids = None, None
        if load_ss and seq is not None:
            sec_struct = get_ss_from_pdb_and_seq(pdbfile=pdb_path, seq=seq)
        if ignore_res_ids:
            res_ids = torch.arange(len(seq))

        return cls(
            atom_coords=coords,
            atom_masks=mask,
            seq=seq,
            name=pdb_path,
            atom_tys=atom_tys,
            res_ids=res_ids,
            sec_struct=sec_struct,
        )

    def to(self, device: Any) -> Protein:
        """Places coords and mask on given device"""
        self.atom_coords = self.atom_coords.to(device)
        self.atom_masks = self.atom_masks.to(device)
        self.res_ids = safe_to_device(self.res_ids, device)
        self.chain_ids = self.chain_ids.to(device)
        self.chain_indices = safe_to_device(self.chain_indices, device)
        return self

    def crop(self, start: int, end: int) -> Protein:
        """Crop the protein from start..end"""
        assert end <= len(self.seq), f"end: {end}, len(seq): {len(self.seq)} len(self): {len(self)}"
        self.seq = self.seq[start:end]
        self.atom_coords = self.atom_coords[..., start:end, :, :].clone()
        self.atom_masks = self.atom_masks[..., start:end, :].clone()
        self.crops = (start, end)
        self.chain_ids = self.chain_ids[start:end].clone()

        if exists(self.sec_struct):
            self.sec_struct = self.sec_struct[start:end]

        # per-chain crop
        chain_indices, res_ids, curr_start = [], [], 0
        for i, chain in enumerate(self.chain_indices):
            e = min(end, curr_start + len(chain))
            s = max(start, curr_start)
            _s, _e = s - curr_start, e - curr_start
            if start < end:
                chain_indices.append(self.chain_indices[i][_s:_e].clone() - start)
                res_ids.append(self.res_ids[i][_s:_e].clone())
            curr_start += len(chain)
        self.chain_indices = chain_indices
        self.res_ids = res_ids
        self._check_chain_indices()

        return self

    @property
    def is_antibody(self):
        return exists(self.cdrs)

    def clone(self) -> Protein:
        """Clone the protien"""
        ptn = Protein(
            atom_coords=self.atom_coords.detach().clone(),
            atom_masks=self.atom_masks.clone(),
            atom_tys=self.atom_tys,
            seq="".join([s for s in self.seq]),
            name=self._name,
            res_ids=[x.clone() for x in self.res_ids],
            chain_ids=self.chain_ids.clone(),
            chain_indices=[x.clone() for x in self.chain_indices],
            chain_names=self.chain_names,
            sec_struct=self.sec_struct,
            cdrs=self.cdrs,
        )
        if exists(self.input_partition):
            ptn.set_input_partition([x.clone() for x in self.input_partition])
        ptn.replica = self.replica
        return ptn

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

    @property
    def is_complex(self):
        """Whether this protein is a complex"""
        return len(self.chain_indices) > 1

    @property
    def n_chains(self):
        """Number of chains in this protein"""
        return len(self.chain_indices)

    def set_input_partition(self, partition: List[Tensor]) -> None:
        """Set partition used in cropping/ generating this protein

        (Should only be referenced in data loader --
        More advanced feature used for generating esm embeddings)
        """
        assert len(partition) == self.n_chains
        self.input_partition = partition

    def make_complex(self, partition: List[Tensor]) -> None:
        """Make the protein a complex"""
        assert not self.is_complex
        assert isinstance(partition, list)
        assert sum([len(x) for x in partition]) == len(self)
        partition = safe_to_device(partition, device=self.device)
        self.chain_indices = [x.clone() for x in partition]
        self.chain_ids = torch.zeros(sum(map(len, partition)), device=self.device)
        for i, part in enumerate(partition):
            self.chain_ids[part] = i
        self.res_ids = [self.res_ids[0][p] for p in partition]
        assert sum(map(len, self.res_ids)) == sum(map(len, partition))

        # remap input partition
        if exists(self.input_partition):
            part = self.input_partition[0]
            self.input_partition = [part[cidxs] for cidxs in self.chain_indices]

        # duplicate chain names
        self.chain_names = self.chain_names * 2
        self._check_chain_indices()

    def get_chain_coords(self):
        """List of per-chain residue coordinates"""
        assert self.chain_indices is not None
        return [self.atom_coords[..., self.chain_indices[i], :, :] for i in range(len(self.chain_indices))]

    def get_atom_coords(
        self, atom_tys: Optional[Union[str, List[str]]] = None, coords: Optional[Tensor] = None
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
        return (
            coords[..., atom_posns[atom_tys], :]
            if isinstance(atom_tys, str)
            else torch.cat([coords[..., atom_posns[ty], :].unsqueeze(-2) for ty in atom_tys], dim=-2)
        )

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
        self, atom_tys: Optional[Union[str, List[str]]] = None, coords: Optional[Tensor] = None
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

    @cached_property
    def is_homodimer(self):
        """Whether this protein is a homodimer"""
        if not self.is_complex:
            return False
        if len(self.chain_indices) != 2:
            return False
        c1, c2 = self.chain_indices
        if len(c1) != len(c2):
            return False
        return all([self.seq[c1[i]] == self.seq[c2[i]] for i in range(len(c1))])

    def kabsch_align_coords(
        self,
        target: Optional[Protein],
        coords: Optional[Tensor] = None,
        atom_ty: Optional[str] = None,
        overwrite: bool = True,
    ) -> Tensor:
        """Aligns this proteins coordinates to other_coords
        :param target: protein to align to
        :param overwrite: overwrite this proteins coordinates
        :param coords: coordinates to use for alignment (if target not given)
        :param atom_ty: atom types to align on (if none given, all atom types are assumed).
        :return: the aligned coordinates
        """
        assert exists(target) ^ exists(coords)
        atom_ty = default(atom_ty, "CA")
        align_to = target.get_atom_coords(atom_ty) if exists(target) else coords
        align_from = self.get_atom_coords(atom_ty)
        mask = self.get_atom_masks(atom_ty)
        reshape = lambda x: x[0].unsqueeze(0) if x[0].ndim < x[1] else x[0]
        align_to, align_from, mask = map(reshape, ((align_to, 3), (align_from, 3), (mask, 2)))
        assert align_to.ndim == 3 and align_to.shape == align_from.shape, f"{align_to.shape},{align_from.shape}"
        rot_to, mean_to, mean_from = _calc_kabsch_rot_n_trans(
            align_to, align_from, mask=self.get_atom_masks(atom_ty).unsqueeze(0)
        )
        coords = rearrange(self.atom_coords, "n a c -> () (n a) c")
        aligned_from = torch.matmul(coords - mean_from, rot_to) + mean_to
        aligned_from = rearrange(aligned_from, "b (n a) c -> (b n) a c", n=self.atom_coords.shape[0])
        if overwrite:
            self.overwrite_coords(aligned_from, self.atom_tys)
        return aligned_from

    def to_pdb(
        self,
        path: str,
        coords: Tensor = None,
        atom_tys: List[str] = None,
        seq: str = None,
        chain_indices: Optional[List[Tensor]] = None,
        chain_idx: Optional[int] = None,
        write_ss: bool = False,
        chain_labels=None,
        res_idxs: Optional[List[List[int]]] = None,
        beta=None,
    ) -> None:
        """saves this protein to a .pdb file"""
        atom_tys = default(atom_tys, self.atom_tys)
        coords = default(coords, self.atom_coords)
        seq = default(seq, self.seq)
        coord_dicts = []
        assert coords.ndim == 3
        coords = coords.detach().cpu().numpy()
        chain_ids = []
        chain_labels = default(chain_labels, "ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        chain_indices = default(chain_indices, self.chain_indices)
        chain_indices = [chain_indices[chain_idx]] if exists(chain_idx) else chain_indices
        for _chain_idx, chain_indices in enumerate(chain_indices):
            _seq, residue_indices = "", []
            _res_idxs = res_idxs[_chain_idx] if exists(res_idxs) else list(range(len(chain_indices)))
            for idx, i in enumerate(chain_indices.detach().cpu().numpy()):
                atoms = {}
                for atom in atom_tys:
                    atom_idx = self.atom_positions[atom]
                    if self.atom_masks[i, atom_idx]:
                        atoms[atom] = coords[i, atom_idx].tolist()
                if len(atoms) > 0:
                    _seq += seq[i]
                    _idx = default(chain_idx, _chain_idx)
                    chain_ids.append(chain_labels[_idx])
                    coord_dicts.append(atoms)
                    residue_indices.append(_res_idxs[idx])

            ss = None
            if exists(self.sec_struct) and write_ss:
                for i in chain_indices:
                    ss += self.sec_struct[i]
            write_pdb(
                coord_dicts, seq=_seq, out_path=path, chain_ids=chain_ids, ss=None, res_idxs=residue_indices, beta=beta
            )

    def impute_cb(self, override: bool = False, exists_ok: bool = False) -> Tuple[Tensor, Tensor]:
        """Impute CB atom position"""
        assert all([x in self.atom_ty_set for x in ["N", "CA", "C"]])
        coords = self.atom_coords
        n, a, c = coords.shape
        bb_coords, bb_mask = self.get_atom_coords_n_masks(atom_tys=["N", "CA", "C"], coords=coords)
        cb_mask, cb_coords = torch.all(bb_mask, dim=-1), torch.zeros(n, 1, 3, device=coords.device)
        # gly_mask = torch.tensor([1 if s != "G" else 0 for s in self.seq]).bool()
        # cb_mask = cb_mask & gly_mask
        cb_coords[cb_mask] = impute_beta_carbon(bb_coords[cb_mask]).unsqueeze(-2)
        if override:
            self.add_atoms(cb_coords, cb_mask.unsqueeze(-1), ["CB"], exists_ok=exists_ok)
        return cb_coords, cb_mask.unsqueeze(-1)

    def add_atoms(
        self, atom_coords: Tensor, atom_masks: Tensor, atom_tys: List[str], exists_ok: bool = False
    ) -> Protein:
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

    def swap_chains(self, order: Optional[List[int]] = None) -> Protein:
        """Swap underlying chains in the given order"""
        assert self.is_complex
        order = default(order, list(reversed(range(len(self.chain_indices)))))
        chain_idxs = [self.chain_indices[i] for i in order]
        res_ids = [self.res_ids[i] for i in order]
        tensors = (self.atom_coords, self.atom_masks, self.chain_ids)
        out = map(lambda x: torch.cat([x[cidxs] for cidxs in chain_idxs], dim=0), tensors)
        atom_coords, atom_masks, chain_ids = out
        idx_order = torch.cat(chain_idxs, dim=0).detach().cpu().numpy()
        seq = "".join([self.seq[i] for i in idx_order])
        new_chain_indices, cum_sum = [], 0
        for i, idxs in enumerate(self.chain_indices):
            new_chain_indices.append(torch.arange(len(idxs)) + cum_sum)
            cum_sum += len(idxs)

        sec_struct = None
        if exists(self.sec_struct):
            sec_struct = "".join([self.sec_struct[i] for i in idx_order])

        swapped_protein = Protein(
            atom_coords=atom_coords,
            atom_masks=atom_masks,
            atom_tys=self.atom_tys,
            seq=seq,
            name=self._name,
            res_ids=res_ids,
            chain_ids=chain_ids,
            chain_indices=new_chain_indices,
            chain_names=[self.chain_names[i] for i in order],
            sec_struct=sec_struct,
            cdrs=self.cdrs,
        ).to(self.device)
        return swapped_protein

    def overwrite_coords(self, coords, atom_tys):
        """Overwrite coordinates for given atom types"""
        assert coords.shape[-2] == len(atom_tys)
        assert coords.ndim == self.atom_coords.ndim
        self._rigids = None
        for i, atom_ty in enumerate(atom_tys):
            atom_idx = self.atom_positions[atom_ty]
            self.atom_coords[..., atom_idx, :] = coords[..., i, :]

    def __clear_cache__(self):
        cache_keys = [name for name, value in inspect.getmembers(Protein) if isinstance(value, cached_property)]
        for key in cache_keys:
            if key in self.__dict__:
                del self.__dict__[key]

    def __getitem__(self, atoms):
        return self.get_atom_coords(atoms)

    def __len__(self):
        return len(self.seq)

    def restrict_to(self, indices: List[Tensor]) -> Protein:
        """Get protein restricted to given indices"""
        assert isinstance(indices, list)
        assert len(indices) == 2, "more than 2 chains not supported :/"
        i1, i2 = indices
        full_indices = torch.cat(indices, dim=0)
        mask = torch.zeros(len(self))
        mask[full_indices] = 1
        split_mask = [mask[: len(self.res_ids[0])], mask[len(self.res_ids[0]) :]]
        chain_indices = [torch.arange(len(i1)), torch.arange(len(i1), len(i1) + len(i2))]
        res_ids = [self.res_ids[i][split_mask[i].bool()] for i in range(2)]
        assert sum(map(len, res_ids)) == sum(map(len, indices))
        assert sum(map(len, chain_indices)) == sum(map(len, indices))

        sec_struct = None
        if exists(self.sec_struct):
            sec_struct = "".join([self.sec_struct[i] for i in full_indices])

        return Protein(
            atom_coords=self.atom_coords[full_indices],
            atom_masks=self.atom_masks[full_indices],
            seq="".join([self.seq[i.item()] for i in full_indices]),
            atom_tys=self.atom_tys,
            name=self._name,
            res_ids=res_ids,
            chain_ids=self.chain_ids[full_indices],
            chain_names=self.chain_names,
            chain_indices=chain_indices,
            sec_struct=sec_struct,
            cdrs=self.cdrs,
        )

    def add_chain(self, other: Protein) -> Protein:
        """Add a chain to this protein"""
        assert not other.is_complex
        atom_coords = torch.cat((self.atom_coords, other.atom_coords), dim=0)
        atom_masks = torch.cat((self.atom_masks, other.atom_masks), dim=0)
        res_ids = self.res_ids + other.res_ids
        chain_ids = torch.cat((self.chain_ids, (self.n_chains + 1) * torch.ones(len(other))))
        chain_indices = self.chain_indices + [len(self) + torch.arange(len(other))]
        sec_struct = None
        if exists(self.sec_struct) and exists(other.sec_struct):
            sec_struct = self.sec_struct + other.sec_struct
        return Protein(
            atom_coords=atom_coords,
            atom_masks=atom_masks,
            seq=self.seq + other.seq,
            atom_tys=self.atom_tys,
            name=self._name,
            res_ids=res_ids,
            chain_indices=chain_indices,
            chain_names=self.chain_names + other.chain_names,
            sec_struct=sec_struct,
            chain_ids=chain_ids,
            cdrs=self.cdrs,
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

    @property
    def all_res_ids(self):
        """Get (flattened) list of residue ids"""
        return torch.cat(self.res_ids, dim=-1)

    @cached_property
    def cdr_mask(self):
        """Mask[0][i] indicates if residue i is part of heavy cdr
        Mask[1][i] indicates if residue i is part of light cdr
        """
        assert self.is_antibody
        mask = torch.zeros((len(self), 2), device=self.device)
        for _, (s, e) in enumerate(self.cdrs["heavy"]):
            if e >= s > 0:
                mask[s : e + 1, 0] = 1
        if "light" in self.cdrs:
            for _, (s, e) in enumerate(self.cdrs["light"]):
                if e >= s > 0:
                    mask[s : e + 1, 1] = 1
        return mask

    def from_coords_n_seq(self, coords: Tensor, seq: Optional[str] = None) -> Protein:
        return Protein(
            atom_coords=coords,
            atom_masks=self.atom_masks,
            atom_tys=self.atom_tys,
            seq=default(seq, self.seq),
            name=self._name,
            res_ids=self.res_ids,
            chain_ids=self.chain_ids,
            chain_indices=self.chain_indices,
            chain_names=self.chain_names,
            sec_struct=self.sec_struct,
            cdrs=self.cdrs,
        )


def safe_load_sequence(seq_path: Optional[str], pdb_path: str, ignore_non_std: bool = True) -> str:
    """Loads sequence, either from fasta or given pdb file"""
    if exists(seq_path):
        return load_fasta_file(seq_path)
    pdbseqs, residueLists, chains = extract_pdb_seq_from_pdb_file(pdb_path, ignore_non_standard=ignore_non_std)
    return pdbseqs[0]
