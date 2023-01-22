"""Model Output"""

from __future__ import annotations

from typing import Optional, List, Union, Tuple, Any

import torch
from einops import rearrange  # noqa
from torch import Tensor

from protein_learning.common.data.data_types.model_input import ModelInput
from protein_learning.common.data.data_types.protein import Protein
from protein_learning.common.helpers import default
from protein_learning.protein_utils.align.kabsch_align import kabsch_align
from protein_learning.protein_utils.sidechains.sidechain_utils import align_symmetric_sidechains



class ModelOutput:
    """Model Output"""

    def __init__(
        self,
        predicted_coords: Tensor,
        scalar_output: Tensor,
        pair_output: Tensor,
        model_input: ModelInput,
        predicted_atom_tys: Optional[List[str]] = None,
        extra: Any = None,
    ):
        self.predicted_coords = predicted_coords
        self.predicted_atom_tys = predicted_atom_tys
        self.scalar_output = scalar_output
        self.pair_output = pair_output
        self.model_input = model_input
        self.extra = extra
        self.native = model_input.native
        self.align_symmetric_scs()

    def set_native_protein(self, protein: Protein):
        """Set the native protein"""
        self.native = protein
        self.align_symmetric_scs()

    @property
    def native_protein(self) -> Protein:
        """Input native protein"""
        return self.native

    @property
    def decoy_protein(self) -> Protein:
        """Input decoy protein"""
        return self.model_input.decoy

    @property
    def valid_residue_mask(self) -> Tensor:
        """Mask indicating which residues are valid
        e.g. (atom coords defined for both native and decoy)
        """
        return self.native_protein.valid_residue_mask & self.decoy_protein.valid_residue_mask

    @property
    def num_valid_residues(self) -> int:
        """Number of valid residues"""
        valid_mask = self.valid_residue_mask
        return valid_mask[valid_mask].numel()

    @property
    def predicted_sc_coords(self) -> Tensor:
        """predicted side chain coordinates"""
        pred, native = self.predicted_coords.squeeze(), self.native.atom_coords.squeeze()
        if pred.shape == native.shape:
            return self.native.get_atom_coords(
                atom_tys=self.native.sc_atom_tys, coords=self.predicted_coords.squeeze(0)
            ).unsqueeze(0)
        elif self.predicted_coords.shape[-2] == len(self.native.sc_atom_tys):
            return self.predicted_coords
        else:
            raise Exception("Not implemented!")

    def get_atom_coords(
        self,
        native: bool = False,
        decoy: bool = False,
        atom_tys: Optional[Union[str, List[str]]] = None,
        coords: Optional[Tensor] = None,
    ) -> Tensor:
        """Get native or decoy atom coordinates for given atom types"""
        assert native ^ decoy
        protein = self.native_protein if native else self.decoy_protein
        coords = default(coords, protein.atom_coords)
        return protein.get_atom_coords(atom_tys=atom_tys, coords=coords)

    def get_atom_mask(
        self,
        native: bool = False,
        decoy: bool = False,
        atom_tys: Optional[Union[str, List[str]]] = None,
    ) -> Tensor:
        """Get native or decoy atom masks for given atom types"""
        assert native ^ decoy
        protein = self.native_protein if native else self.decoy_protein
        return protein.get_atom_masks(atom_tys=atom_tys)

    def get_pred_and_native_coords_and_mask(
        self,
        atom_tys: Optional[List[str]] = None,
        align_by_kabsch: bool = False,
    ) -> Tuple[Tensor, ...]:
        """Gets coordinates and masks from model output for given atom types

        Returns:
            (1) Predicted coords for given atom types (b,n,a,3)
            (2) Native Coords for given atom types (b,n,a,3)
            (3) coord mask (b,n,a)
        """
        pred_coords = self.predicted_coords
        native_coords = self.get_atom_coords(native=True, atom_tys=atom_tys).unsqueeze(0)
        pred_coords = self.get_atom_coords(decoy=True, atom_tys=atom_tys, coords=pred_coords)
        native_mask = self.get_atom_mask(native=True, atom_tys=atom_tys).unsqueeze(0)
        pred_mask = self.get_atom_mask(decoy=True, atom_tys=atom_tys).unsqueeze(0)
        joint_mask = torch.logical_and(native_mask, pred_mask)
        assert native_coords.shape == pred_coords.shape
        assert native_mask.shape == pred_mask.shape
        assert native_mask.shape == pred_coords.shape[:3]
        if align_by_kabsch:
            tmp, native_coords, mask = map(
                lambda x: rearrange(x, "b n a ... -> b (n a) ..."), (pred_coords, native_coords, joint_mask)
            )
            _, native_coords = kabsch_align(align_to=tmp, align_from=native_coords, mask=mask)
            native_coords = rearrange(native_coords, "b (n a) c -> b n a c", n=pred_coords.shape[1])

        return pred_coords, native_coords, joint_mask

    def align_symmetric_scs(self):
        """Overwrite native coordinates so that sc atoms are swapped to minimize rmsd"""
        if len(self.native.sc_atom_tys) <= 1:  # nothing to align
            return
        native_seq = self.native_protein.seq_encoding.unsqueeze(0)
        native_coords = self.native_protein.atom_coords.unsqueeze(0)
        atom_mask = self.native_protein.atom_masks.unsqueeze(0)
        nat_aligned = align_symmetric_sidechains(
            native_coords=native_coords, predicted_coords=self.predicted_coords, atom_mask=atom_mask, native_seq=native_seq
        )
        self.native_protein.atom_coords = nat_aligned.squeeze(0)

    @property
    def seq_len(self):
        """Sequence length"""
        return len(self.native_protein.seq)

    def detach(self)->ModelOutput:
        return ModelOutput(
            scalar_output=self.scalar_output.detach(),
            predicted_coords=self.predicted_coords.detach(),
            pair_output= self.pair_output.detach(),
            model_input=self.model_input,
            extra={k : v.detach() if torch.is_tensor(v) else v for k,v in default(self.extra,dict()).items()},
        )

    def __getattr__(self, attr):
        """Called only if this class does not have the given attribute"""
        try:
            if isinstance(self.extra, dict):
                return self.extra[attr]
            return getattr(self.extra, attr)
        except:  # noqa
            raise AttributeError(f"No attribute {attr} found for this class")
