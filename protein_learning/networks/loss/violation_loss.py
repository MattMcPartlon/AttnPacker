"""Modules For Computing Violation Loss

(1) InterResidueVDWRepulsiveLoss
(2) BackboneBondLenDeviationLoss
(3) BackboneAngleDeviationLoss
"""
import math
from typing import Dict, Any, Optional

import torch
import torch.nn.functional as F  # noqa
from einops import repeat, rearrange  # noqa
from torch import nn, Tensor

from protein_learning.common.protein_constants import (
    ALL_ATOM_POSNS,
    ALL_ATOM_VDW_RADII,
    BOND_ANGLES,
    BOND_LENS,
    BOND_LEN_TOL,
    BOND_ANGLE_TOL,
    BOND_LEN_OFFSET,
    BOND_ANGLE_OFFSET,
)
from protein_learning.networks.loss.utils import (
    outer_sum,
    outer_prod,
    to_rel_coord,
)


def masked_mean(tensor, mask, dim=-1, keepdim=False):
    """Masked mean"""
    if not exists(mask):
        return torch.mean(tensor, dim=dim, keepdim=keepdim)
    mask = mask.bool() if mask.dtype != torch.bool else mask
    diff_len = len(tensor.shape) - len(mask.shape)
    mask = mask[(..., *((None,) * diff_len))]  # noqa
    tensor.masked_fill_(~mask, 0.)
    total_el = mask.sum(dim=dim, keepdim=keepdim)
    mean = tensor.sum(dim=dim, keepdim=keepdim) / total_el.clamp(min=1.)
    mean.masked_fill_(total_el == 0, 0.)
    return mean


def safe_norm(x, dim=-1, keepdim=False, eps=1e-8) -> Tensor:
    """Takes gradient-safe norem"""
    return torch.sqrt(torch.sum(torch.square(x) + eps, dim=dim, keepdim=keepdim))


def exists(x: Any) -> bool:
    """Returns whether x is not None"""
    return x is not None


def default(x: Any, y: Any) -> Any:
    """Returns x if it exists, else y"""
    return x if exists(x) else y


def _init_table() -> Tensor:
    """Comput Van Der Waals atom radii table for given sequence length"""
    vdw_radii = torch.zeros(len(ALL_ATOM_POSNS))
    for atom_ty, atom_pos in ALL_ATOM_POSNS.items():
        vdw_radii[atom_pos] = ALL_ATOM_VDW_RADII[atom_ty]
    return vdw_radii


def _compute_atom_ty_mapping(
        atom_ty_order: Dict[str, int],
        device: Any = "cpu"
) -> Tensor:
    """Given a mapping from atom type to atom position, computes a mapping
    from the given atom type order to the implicit atom type order used by
    modules in this file.

    :param atom_ty_order: map from atom type (str) to atom position (int)
    in atom coordinate tensor.
    :return: mapping from given atom ordering to implicit atom ordering
    """
    mapping = torch.zeros(len(atom_ty_order), device=device).long()
    for atom_ty, atom_pos in atom_ty_order.items():
        mapping[atom_pos] = ALL_ATOM_POSNS[atom_ty]
    return mapping


class InterResidueVDWRepulsiveLoss(nn.Module):
    """(approximate) Van Der Waals repulsive loss between atom pairs in
    separate residues

    For parameter shape descriptions, b is batch size, n is the sequence length,
    and a is the number of atom types.

    For each input atom pair in distinct residues, the pairwise distance is
    subtracted from the sum of the respective Van Der Waals radii.
    ReLU of this difference is squared and summed for each atom to compute the
    final loss for each atom.
    """

    def __init__(self, tol: float = 0.25, reduce_mean: bool = True):  # noqa
        """

        :param tol : Tolerance to use when computing clashes.
        """
        super(InterResidueVDWRepulsiveLoss, self).__init__()
        self._vdw_table = _init_table()
        self.tol = tol
        self.reduce_mean = reduce_mean

    def vdw_table(self, coords: Tensor, mapping: Tensor) -> Tensor:
        """Gets table of Van Der Waals radii for atom types

        :param coords: used to determine table shape and device. shape: (b,n,a,3)
        :param mapping : mapping for atom types to compute sum for
        :return: (b,n,a) list of van der waals radii for each atom type
        """
        b, n, device = coords.shape[0], coords.shape[1], coords.device
        if self._vdw_table.device != device:
            self._vdw_table = self._vdw_table.to(device)
        return repeat(self._vdw_table[mapping], "i-> b n i", b=b, n=n)

    def forward(
            self,
            atom_coords: Tensor,
            atom_ty_order: Optional[Dict[str, int]] = None,
            atom_coord_mask: Optional[Tensor] = None,
            reduce: bool = True,
            reduce_mean=True,
    ) -> Tensor:
        """Compute Van Der Waals repulsize loss

        :param atom_coords: Atom cordinates of shape (b,n,a,3)
        :param atom_ty_order : (Optional) order of atom types in forward input.
        If the order is different than that of ALL_ATOM_POSNS, then this parameter should
        not be None.
        Ex. Coordinate and mask tensors passed to forward should have shape
        (b,n,a,3) and (b,n,a) resp. if the coords[_,_,1] denotes CA atoms, then
        atom_ty_order["CA"] should be set to 1.
        :param atom_coord_mask: (Optional) valid atom mask of shape (b,n,a)
        :param reduce: whether to return sum of violations

        :return: tensor of shape (b,n,a) giving the Van Der Waals repulsive loss for each atom
        (if reduce not specified) else sum of the Van Der Waals energies
        """
        b, n, a, device = *atom_coords.shape[:3], atom_coords.device  # noqa

        # permute coordinates and mask to match pre-defined ordering
        mapping = _compute_atom_ty_mapping(atom_ty_order, atom_coords.device)

        # pair mask will mask out invalid coordinates and atom pairs with
        # sequence separation <= 1
        rng = torch.arange(n, device=atom_coords.device)
        pair_mask = torch.abs(outer_sum(rng, -rng)) > 1
        pair_mask = repeat(pair_mask, " i j -> () (i x) (j y)", x=a, y=a)

        if exists(atom_coord_mask):
            atom_coord_mask = rearrange(atom_coord_mask, "b n a -> b (n a)")
            pair_mask = outer_prod(atom_coord_mask.float()) * pair_mask

        # compute inter-atom distances
        rel_atom_coords = rearrange(atom_coords, "b n a c -> b (n a) c")
        atom_dists = safe_norm(to_rel_coord(rel_atom_coords), dim=-1)

        # compute vdw sums between valid atom pairs
        vdw_radii = rearrange(self.vdw_table(coords=atom_coords, mapping=mapping), "b n i -> b (n i)")
        vdw_sums = outer_sum(vdw_radii)
        # determine volations
        violations = F.relu(vdw_sums + self.tol - atom_dists)
        viols = torch.sum(torch.square(violations) * pair_mask, dim=-1) / 2
        if reduce:
            denom = viols[viols > 0].numel() if reduce_mean else 1
            return torch.sum(viols) / max(1, denom)
        return rearrange(viols, "b (n a) -> b n a", n=n)


class BackboneBondLenDeviationLoss(nn.Module):
    """Penalizes deviations from ideal backbone bond lengths

    For parameter shape descriptions, b is batch size, n is the sequence length,
    and a is the number of atom types.

    (Optionally) Distance deviations between consecutive CA atoms
    can be included

    New length can easily be added by appending to
    BOND_LENS
    BOND_LEN_OFFSET
    BOND_LEN_TOL

    just call the "_register_pair(...)" method on the corresponding atoms
    and it will magically appear in the loss.
    """

    def __init__(
            self,
            include_c_n_bond: bool = True,
            include_n_ca_bond: bool = True,
            include_ca_c_bond: bool = True,
            include_ca_ca_pseudo_bond: bool = True,
    ):
        """
        :param include_c_n_bond: whether peptide bond length deviations should be included
        :param include_n_ca_bond: whether N_i-CA_i bond length deviations should be included
        :param include_ca_c_bond: whether CA_i-C_i bond length deviations should be included
        :param include_ca_ca_pseudo_bond: whether CA_i - CA_{i+1} deviations should be included
        """
        super(BackboneBondLenDeviationLoss, self).__init__()
        self.atom_tys = []
        self.offsets, self.ideal_lens, self.tols = [], [], []
        if include_c_n_bond:
            self._register_pair("C", "N")
        if include_n_ca_bond:
            self._register_pair("N", "CA")
        if include_ca_c_bond:
            self._register_pair("CA", "C")
        if include_ca_ca_pseudo_bond:
            self._register_pair("CA", "CA")

    def _register_pair(self, a1_ty: str, a2_ty: str):
        key = (a1_ty, a2_ty)
        self.atom_tys.append(key)
        self.offsets.append(BOND_LEN_OFFSET[key])
        self.ideal_lens.append(BOND_LENS[key])
        self.tols.append(BOND_LEN_TOL[key] ** 2)

    def forward(
            self,
            atom_coords: Tensor,
            atom_ty_order: Optional[Dict[str, int]] = None,
            atom_coord_mask: Optional[Tensor] = None,
            residue_indices: Optional[Tensor] = None,
            bonded_mask: Optional[Tensor] = None,
    ):
        """Compute bond length (squared) deviation loss

        :param atom_coords: Atom cordinates of shape (b,n,a,3)
        :param atom_coord_mask: (Optional) valid atom mask of shape (b,n,a)
        :param atom_ty_order : (Optional) order of atom types in forward input.
        If the order is different than that of ALL_ATOM_POSNS, then this parameter should
        not be None.
        Ex. Coordinate and mask tensors passed to forward should have shape
        (b,n,a,3) and (b,n,a) resp. if the coords[_,_,1] denotes CA atoms, then
        atom_ty_order["CA"] should be set to 1.
        :param residue_indices: (Optional) residue_indices[_,_,i] gives the sequence position of residue i.
        :return: mean of deviation loss summed over each atom type.
        """

        b, n = atom_coords.shape[:2]
        atom_ty_order = default(atom_ty_order, ALL_ATOM_POSNS)
        loss = 0  # store sum of loss
        for idx, offset in enumerate(self.offsets):
            # compute loss for specific atom pair
            first_idx, second_idx = map(lambda a: atom_ty_order[a], self.atom_tys[idx])
            valid_mask = torch.ones(b, n - offset, device=atom_coords.device).bool()
            first_atoms, second_atoms = atom_coords[:, :, first_idx], atom_coords[:, :, second_idx]
            if offset == 1:
                deviations = safe_norm(first_atoms[:, :-1] - second_atoms[:, 1:])
                if exists(residue_indices):
                    valid_mask = valid_mask & ((residue_indices[:, 1:] - residue_indices[:, :-1]) <= 1)
                if exists(atom_coord_mask):
                    valid_mask = valid_mask & (atom_coord_mask[:, :-1, first_idx] & atom_coord_mask[:, 1:, second_idx])
                if exists(bonded_mask):
                    valid_mask = bonded_mask & valid_mask
            else:
                deviations = safe_norm(first_atoms - second_atoms)
                if exists(atom_coord_mask):
                    valid_mask = valid_mask & (atom_coord_mask[..., first_idx] & atom_coord_mask[..., second_idx])

            deviation = F.relu(torch.square(deviations - self.ideal_lens[idx]) - self.tols[idx])
            deviation[deviation > 0] = torch.sqrt(deviation[deviation > 0])
            loss = loss + masked_mean(deviation, valid_mask, dim=-1)

        # loss has shape (b,)
        return loss / len(self.atom_tys)


class BackboneAngleDeviationLoss(nn.Module):
    """Penalizes deviations from ideal backbone bond angles

    For parameter shape descriptions, b is batch size, n is the sequence length,
    and a is the number of atom types.

    New angles can easily be added by appending to
    BOND_ANGLE_OFFSET
    BOND_ANGLES
    BOND_ANGLE_TOL

    just call the "_register_triple(...)" method on the corresponding atoms
    and it will magically appear in the loss.
    """

    def __init__(
            self,
            include_n_ca_c_angle: bool = True,
            include_ca_c_n_angle: bool = True,
            include_c_n_ca_bond: bool = True,
    ):
        super(BackboneAngleDeviationLoss, self).__init__()
        self.atom_tys = []
        self.offsets, self.ideal_angles, self.tols = [], [], []
        if include_n_ca_c_angle:
            self._register_triple("N", "CA", "C")
        if include_ca_c_n_angle:
            self._register_triple("CA", "C", "N")
        if include_c_n_ca_bond:
            self._register_triple("C", "N", "CA")

    def _register_triple(self, a1_ty: str, a2_ty: str, a3_ty: str):
        key = (a1_ty, a2_ty, a3_ty)
        self.atom_tys.append(key)
        self.offsets.append(BOND_ANGLE_OFFSET[key])
        self.ideal_angles.append(math.cos(BOND_ANGLES[key] * math.pi / 180))
        self.tols.append((BOND_ANGLE_TOL[key] * math.pi / 180) ** 2)

    def forward(
            self,
            atom_coords: Tensor,
            atom_ty_order: Optional[Dict[str, int]] = None,
            atom_coord_mask: Optional[Tensor] = None,
            residue_indices: Optional[Tensor] = None,
            bonded_mask: Optional[Tensor] = None,
    ):
        """Compute bond length (squared) deviation loss

        :param atom_coords: Atom cordinates of shape (b,n,a,3)
        :param atom_coord_mask: (Optional) valid atom mask of shape (b,n,a)
        :param atom_ty_order : (Optional) order of atom types in forward input.
        If the order is different than that of ALL_ATOM_POSNS, then this parameter should
        not be None.
        Ex. Coordinate and mask tensors passed to forward should have shape
        (b,n,a,3) and (b,n,a) resp. if the coords[_,_,1] denotes CA atoms, then
        atom_ty_order["CA"] should be set to 1.
        :param residue_indices: (Optional) residue_indices[_,_,i] gives the sequence position of residue i.
        :return: mean of deviation loss summed over each angle type.
        """
        b, n = atom_coords.shape[:2]
        atom_ty_order = default(atom_ty_order, ALL_ATOM_POSNS)
        loss = 0  # store sum of loss
        for idx, (o1, o2, o3) in enumerate(self.offsets):
            # indices of each atom type
            i1, i2, i3 = map(lambda a: atom_ty_order[a], self.atom_tys[idx])
            # mask to apply to loss
            has_offset = any((o1, o2, o3))
            valid_mask = torch.ones(b, n - int(has_offset), device=atom_coords.device).bool()
            m1, m2, m3 = map(lambda i: atom_coord_mask[:, :, i], (i1, i2, i3)) if \
                exists(atom_coord_mask) else [None] * 3

            # coordinates for each atom type
            a1, a2, a3 = map(lambda i: atom_coords[:, :, i], (i1, i2, i3))
            if has_offset:
                a1, a2, a3 = map(lambda x: x[0][:, 1:] if x[1] else x[0][:, :-1], [(a1, o1), (a2, o2), (a3, o3)])
                if exists(atom_coord_mask):
                    m1, m2, m3 = map(lambda x: x[0][:, 1:] if x[1] else x[0][:, :-1], [(m1, o1), (m2, o2), (m3, o3)])
                    valid_mask = valid_mask & m1 & m2 & m3
                if exists(residue_indices):
                    valid_mask = valid_mask & ((residue_indices[:, 1:] - residue_indices[:, :-1]) <= 1)
                if exists(bonded_mask):
                    valid_mask = bonded_mask & valid_mask
            else:
                if exists(atom_coord_mask):
                    valid_mask = valid_mask & m1 & m2 & m3

            b01, b02 = map(lambda x: x / safe_norm(x, dim=-1, keepdim=True), (a1 - a2, a3 - a2))
            cos_theta = torch.clamp(torch.sum(b01 * b02, dim=-1), -1, 1)
            ideal_angles = torch.abs((torch.ones_like(cos_theta) * self.ideal_angles[idx])) * torch.sign(cos_theta)
            dev = F.relu(torch.square(cos_theta - ideal_angles.detach()) - self.tols[idx])
            angle_loss = masked_mean(dev, valid_mask, dim=-1)
            loss = loss + (angle_loss if angle_loss > 0 else 0)

        # loss has shape (b,)
        return loss


class IntraResidueDistance(nn.Module):
    def __init__(self):
        super(IntraResidueDistance, self).__init__()

    def forward(self, predicted_coords, actual_coords, valid_res_mask, atom_mask):
        to_intra_dists = lambda x: safe_norm(rearrange("b n a c -> b n a () c") -
                                             rearrange("b n a c -> b n () a c"),
                                             dim=-1)
        atom_mask = atom_mask.float()
        intra_mask = torch.einsum("b n i, b n j -> b n i j", atom_mask, atom_mask)
        intra_mask[~valid_res_mask] = 0
        deviations = torch.square(to_intra_dists(predicted_coords) - to_intra_dists(actual_coords))
        loss = torch.sum(deviations * intra_mask, dim=(-1, -2))
        totals = torch.clamp_min(torch.sum(intra_mask, dim=(-1, -2)), 1)
