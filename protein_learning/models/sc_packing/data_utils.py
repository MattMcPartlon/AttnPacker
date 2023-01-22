from __future__ import annotations

from abc import abstractmethod
from typing import Any, Optional

import torch
from einops import repeat, rearrange  # noqa
from torch import Tensor

from protein_learning.common.data.data_types.model_input import ExtraInput
from protein_learning.common.data.data_types.protein import Protein
from protein_learning.features.feature_config import InputFeatureConfig
from protein_learning.features.feature_generator import FeatureGenerator, get_input_features
from protein_learning.features.input_features import InputFeatures
from protein_learning.protein_utils.sidechain_utils import (
    swap_symmetric_atoms,
    per_residue_chi_indices_n_mask,
    get_chi_pi_periodic_mask,
    get_symmetric_residue_keys_n_indices,
    get_sc_dihedral,
)


def augment(decoy_protein: Protein, native_protein: Protein) -> Any:
    """Augment Model Input

    In addition to standard data, we will add
    (1) alt_truth atom positions
    (2) native sc-dihedral info
        - chi masks
        - chi_i atom indices
        - native chi angles
        - chi_i pi-periodic mask
    """
    seq, native_coords = native_protein.seq, native_protein.atom_coords
    residue_mask = torch.logical_and(native_protein.bb_atom_mask, decoy_protein.bb_atom_mask)
    residue_mask = torch.all(residue_mask, dim=-1)
    # compute alt-truth atom positions
    symm_keys, symm_indices = get_symmetric_residue_keys_n_indices(seq)
    alt_truth_coords = swap_symmetric_atoms(native_coords, (symm_keys, symm_indices))
    # compute dihedral data
    coord_mask = torch.logical_and(native_protein.atom_masks, decoy_protein.atom_masks)
    chi_indices, chi_mask = per_residue_chi_indices_n_mask(
        coord_mask=coord_mask, seq=seq
    )
    chi_pi_periodic = get_chi_pi_periodic_mask(seq=seq)
    chi_angles = get_sc_dihedral(native_coords, chi_mask, chi_indices)
    chi_mask[:, ~residue_mask] = False

    return ExtraSCInput(
        alt_truth_atom_coords=alt_truth_coords,
        chi_indices=chi_indices,
        chi_mask=chi_mask,
        chi_pi_periodic=chi_pi_periodic,
        native_chi_angles=chi_angles
    )


class ExtraSCInput(ExtraInput):
    """Extra input passed to ModelInput when data entries are loaded"""

    def __init__(
            self,
            alt_truth_atom_coords: Tensor,
            chi_indices: Optional[Tensor],
            chi_mask: Optional[Tensor],
            chi_pi_periodic: Optional[Tensor],
            native_chi_angles: Optional[Tensor],
    ):
        super(ExtraSCInput, self).__init__()
        n = alt_truth_atom_coords.shape[0]
        self.alt_truth_atom_coords = alt_truth_atom_coords
        self.chi_mask = chi_mask
        assert chi_mask.shape == (4, n), f"{chi_mask.shape}"
        self.chi_indices = chi_indices
        assert chi_indices.shape == (4, n, 4), f"{chi_indices.shape}"
        self.chi_pi_periodic = chi_pi_periodic  # (4,n)
        assert chi_pi_periodic.shape == (4, n), f"{chi_pi_periodic.shape}"
        self.native_chi_angles = native_chi_angles
        assert native_chi_angles.shape == (4, n), f"{native_chi_angles.shape}"

    def get_sc_dihedral(self, coords: Tensor) -> Tensor:
        """Get chi1-chi4 dihedral angles of given coordinates"""
        return get_sc_dihedral(
            coords=coords,
            chi_mask=self.chi_mask,
            chi_indices=self.chi_indices
        )

    def crop(self, start: int, end: int) -> ExtraSCInput:
        """crops instance variables to start:end"""
        self.alt_truth_atom_coords = self.alt_truth_atom_coords[start:end]
        self.chi_mask = self.chi_mask[:, start:end]
        self.chi_indices = self.chi_indices[:, start:end]
        self.chi_pi_periodic = self.chi_pi_periodic[:, start:end]
        self.native_chi_angles = self.native_chi_angles[:, start:end]
        return self

    def to(self, device: Any) -> ExtraSCInput:
        """Maps instance variables to given device"""
        self.alt_truth_atom_coords = self.alt_truth_atom_coords.to(device)
        self.chi_mask = self.chi_mask.to(device)
        self.chi_indices = self.chi_indices.to(device)
        self.chi_pi_periodic = self.chi_pi_periodic.to(device)
        self.native_chi_angles = self.native_chi_angles.to(device)
        return self

    def align_symm_atom_coords(self, decoy_coords: Tensor, native_coords: Tensor):
        """Gets alignment of native atom coordinates
        to given decoy coordinates. This method uses the alt_truth atom
        positions of the native to determine which atoms to swap/keep.
        """
        alt_truth = self.alt_truth_atom_coords.unsqueeze(0)
        assert alt_truth.shape[1:] == decoy_coords.shape[1:], \
            f"{alt_truth.shape},{decoy_coords.shape}"
        unswapped = torch.sum(torch.square(decoy_coords - native_coords), dim=(-1, -2))
        swapped = torch.sum(torch.square(decoy_coords - alt_truth), dim=(-1, -2))
        aligned_coords = native_coords.detach().clone()
        swap_mask = swapped < unswapped
        aligned_coords[swap_mask] = alt_truth[swap_mask]
        return aligned_coords


class SCFeatureGenerator(FeatureGenerator):
    def __init__(
            self,
            config: InputFeatureConfig,
    ):
        super(SCFeatureGenerator, self).__init__(config)

    @abstractmethod
    def generate_features(
            self,
            protein: Protein,
            extra: Optional[ExtraInput] = None
    ) -> InputFeatures:
        feats = get_input_features(
            seq=protein.seq,
            coords=protein.atom_coords,
            res_ids=protein.res_ids,
            atom_ty_to_coord_idx=protein.atom_positions,
            config=self.config
        )
        return InputFeatures(features=feats, batch_size=1, length=len(protein)).maybe_add_batch()
