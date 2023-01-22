"""Model Input data type"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, Union, List, Dict

from torch import Tensor

from protein_learning.common.data.data_types.protein import Protein
from protein_learning.common.helpers import exists, default
from protein_learning.features.input_features import InputFeatures


class ExtraInput(ABC):
    """Extra information to augment ModelInput with

    Must define "crop" and "to"
    """

    def __init__(self):
        pass

    @abstractmethod
    def crop(self, start, end) -> ExtraInput:
        """Crop data from start..end"""
        pass

    @abstractmethod
    def to(self, device: Any) -> ExtraInput:
        """Place data on given device"""
        pass

    def extra_res_feats(self, protein: Protein) -> Optional[Tensor]:  # noqa
        """Get extra residue features (optinoal)"""
        return None

    def extra_pair_feats(self, protein: Protein) -> Optional[Tensor]:  # noqa
        """Get extra pair features (optinoal)"""
        return None

    def extra_feats(self, protein: Protein) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """Get extra features (optinoal)"""
        return self.extra_res_feats(protein), self.extra_pair_feats(protein)

    def extra_feat_descs(self, protein: Protein):
        return dict(), dict()

    def intra_chain_mask_kwargs(self, ids_to_mask: Tensor) -> Dict:  # noqa
        """Extra keyword args for intra-chain masks"""
        return {}

    def inter_chain_mask_kwargs(self, ids_to_mask: Tensor) -> Dict:  # noqa
        """Extra keyword args for inter-chain masks"""
        return {}

    def load_extra_mask_inputs(
            self,
            seq_mask: Optional[Tensor],
            feat_mask: Optional[Tensor],
            inter_chain_mask: Optional[Tensor]
    ):
        """Load any extra inputs conditional on generated masks"""
        pass


class ModelInput:
    """Input for protein-based learning model"""

    def __init__(
            self,
            decoy: Protein,
            native: Optional[Protein] = None,
            input_features: Optional[InputFeatures] = None,
            extra: Optional[ExtraInput] = None,
    ):
        self.decoy, self.native = decoy, native
        self.input_features = input_features
        self.extra = extra
        # store positions of input crop (if applicable)
        self.crop_posns = None

    def crop(self, max_len, bounds = None) -> ModelInput:
        """Randomly crop model input to max_len"""
        start, end = 0, len(self.decoy.seq)
        start = random.randint(0, (end - max_len)) if end > max_len else start
        end = min(end, start + max_len)
        start, end = default(bounds,(start,end))
        self.crop_posns = (start, end)
        self.input_features = self.input_features.crop(start=start, end=end)
        self.decoy = self.decoy.crop(start, end)
        self.native = self.native.crop(start, end)
        self.extra = self.extra.crop(start, end) if exists(self.extra) else None
        return self

    def to(self, device: Any) -> ModelInput:
        """Places all data on given device"""
        self.decoy = self.decoy.to(device)

        self.input_features = self.input_features.to(device) if \
            exists(self.input_features) else None

        self.native = self.native.to(device) if \
            exists(self.native) else None

        self.extra = self.extra.to(device) if \
            exists(self.extra) else None
        return self

    def _get_protein(self, native: bool = False, decoy: bool = False) -> Protein:
        """Gets native or decoy protein"""
        assert native ^ decoy
        return self.native if native else self.decoy

    def bb_atom_tys(self, native: bool = False, decoy: bool = False):
        """Backbone atom types for native or decoy"""
        return self._get_protein(native=native, decoy=decoy).bb_atom_tys

    def sc_atom_tys(self, native: bool = False, decoy: bool = False):
        """Side-chain atom types for native or decoy"""
        return self._get_protein(native=native, decoy=decoy).sc_atom_tys

    def get_atom_coords(
            self,
            atom_tys: Union[str, List[str]],
            native: bool = False,
            decoy: bool = False,
            coords: Optional[Tensor] = None
    ) -> Tensor:
        """Gets the atom coordinates for the given atom types

        Returns:
            Tensor of shape (...,n,3) if atom_tys is a string,
            otherwise a tensor of shape (...,n,a,3) where a
            is the number of atom_tys given.
        """
        protein = self._get_protein(native=native, decoy=decoy)
        coords = default(coords, protein.atom_coords)
        return protein.get_atom_coords(atom_tys=atom_tys, coords=coords)

    def get_atom_masks(
            self,
            atom_tys: Union[str, List[str]],
            native: bool = False,
            decoy: bool = False,
            mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Gets the atom masks for the given atom types

        Returns: Tensor of shape (...,n) if atom_tys is a string,
        otherwise a tensor of shape (...,n,a) where a is the number
        of atom_tys given.
        """
        ptn = self._get_protein(native=native, decoy=decoy)
        return ptn.get_atom_masks(atom_tys, mask=default(mask, ptn.atom_masks))

    def get_atom_coords_n_masks(
            self,
            atom_tys: Union[str, List[str]],
            native: bool = False,
            decoy: bool = False,
            coords: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """Get coords and masks for given atom types"""
        coords = self.get_atom_coords(native=native, decoy=decoy, coords=coords, atom_tys=atom_tys)
        masks = self.get_atom_masks(native=native, decoy=decoy, atom_tys=atom_tys)
        return coords, masks

    @property
    def device(self):
        """Get device storing relevant input data"""
        return self.native.device

    def n_residue(self, decoy: bool = False, native: bool = False) -> int:
        """Number of residues in decoy/native protein"""
        assert native ^ decoy
        return len(self.native) if native else len(self.decoy)

    def __getattr__(self, attr):
        """Called only if this class does not have the given attribute"""
        try:
            return getattr(self.extra, attr)
        except:
            raise AttributeError(f"No attribute {attr} found for this class")
