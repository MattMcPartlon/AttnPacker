"""Coordinate Based Loss Functions"""
import random
from typing import Optional, List, Callable, Dict

import torch
from einops import rearrange, repeat  # noqa
from torch import nn, Tensor

from protein_learning.common.data.data_types.model_output import ModelOutput
from protein_learning.common.global_constants import get_logger
from protein_learning.common.helpers import exists, safe_norm, default
from protein_learning.common.rigids import Rigids
from protein_learning.networks.common.utils import calc_tm_torch, exists
from protein_learning.networks.loss.utils import get_loss_func, get_tm_scale
from protein_learning.networks.loss.violation_loss import (
    BackboneAngleDeviationLoss,
    BackboneBondLenDeviationLoss,
    InterResidueVDWRepulsiveLoss,
)
from protein_learning.protein_utils.align.kabsch_align import kabsch_align

logger = get_logger(__name__)

"""Constants and Helpers"""
BETA = 0.25
MAX_DIST_CLAMP, MIN_DIST_CLAMP = 10, 1e-6
MAX_COORD_CLAMP, MIN_COORD_CLAMP = 10, 1e-6
flatten_coords = lambda crds: rearrange(crds, "b n a c-> b (n a) c")
to_res_rel_coords = lambda x, y: rearrange(x, "b n a c -> b n a () c") - rearrange(y, "b n a c -> b n () a c")
BB_LDDT_THRESHOLDS, SC_LDDT_THRESHOLDS = [0.5, 1, 2, 4], [0.25, 0.5, 1, 2]
to_rel_dev_coords = lambda x: rearrange(x, "b n a ... -> b (n a) () ...") - rearrange(x, "b n a ... -> b () (n a) ...")
max_value = lambda x: torch.finfo(x.dtype).max  # noqa


def per_residue_mean(x: Tensor, mask: Tensor) -> Tensor:
    """Takes a (masked) mean over the atom axis for each residue,
    then takes a mean over the residue axis.
    :param x: shape (b,n,a,*) -> (batch, res, atom, *) (must have 4 dimensions)
    :param mask: shape (b,n,a) -> (batch, res, atom)
    mask[b,i,a] indicates if the atom is present for a given residue.
    :return : mean of (masaked) per-residue means
    """
    if mask is None:
        return torch.mean(x)
    x = x.unsqueeze(0) if x.ndim == 3 else x
    atoms_per_res = mask.sum(dim=-1)
    retain_mask = atoms_per_res > 0
    x = x.masked_fill(~mask.unsqueeze(-1), value=0.0)
    per_res_mean = x.sum(dim=(-1, -2)) / torch.clamp_min(atoms_per_res, 1.0)
    return torch.sum(per_res_mean / torch.sum(retain_mask, dim=-1, keepdim=True))


class ViolationLoss(nn.Module):
    """VDW+BondLen+BondAngle Loss"""

    def __init__(
        self,
        bond_len_wt: float = 1,
        bond_angle_wt: float = 3,
        vdw_wt: float = 1,
        vdw_mean: bool = True,
        vdw_tol: float = 0.25,
    ):
        super(ViolationLoss, self).__init__()
        self.vdw = InterResidueVDWRepulsiveLoss(tol=vdw_tol, reduce_mean=vdw_mean)
        self.bond_len = BackboneBondLenDeviationLoss()
        self.bond_angle = BackboneAngleDeviationLoss()
        self.bond_len_wt = bond_len_wt
        self.bond_angle_wt = bond_angle_wt
        self.vdw_wt = vdw_wt

    def forward_from_output(
        self, output: ModelOutput, baseline: bool = False, vdw_reduce_mean: bool = True, weighted: bool = True, **kwargs
    ) -> Dict[str, Tensor]:
        if not baseline:
            coords = output.predicted_coords
        else:
            coords = output.native_protein.atom_coords.unsqueeze(0)
        posns = output.native_protein.atom_positions
        mask = output.native_protein.atom_masks.unsqueeze(0)
        indices = output.decoy_protein.all_res_ids.unsqueeze(0)
        ca_native = output.native_protein["CA"]
        bonded_mask = safe_norm(ca_native[:-1] - ca_native[1:], dim=-1).squeeze() < 3.9

        vdw_loss = self.vdw.forward(
            atom_coords=coords,
            atom_ty_order=posns,
            atom_coord_mask=mask,
            reduce=True,
            reduce_mean=vdw_reduce_mean,
        )

        bond_len_loss = self.bond_len.forward(
            atom_coords=coords,
            atom_ty_order=posns,
            atom_coord_mask=mask,
            residue_indices=indices,
            bonded_mask=bonded_mask.unsqueeze(0),
        )

        bond_angle_loss = self.bond_angle.forward(
            atom_coords=coords,
            atom_ty_order=posns,
            atom_coord_mask=mask,
            residue_indices=indices,
            bonded_mask=bonded_mask.unsqueeze(0),
        )
        bond_wt, vdw_wt, ang_wt = self.bond_len_wt, self.vdw_wt, self.bond_angle_wt
        if not weighted:
            bond_wt, vdw_wt, ang_wt = 1, 1, 1
        return dict(
            bond_angle=bond_angle_loss * ang_wt,
            bond_len=bond_len_loss * bond_wt,
            vdw=vdw_loss * vdw_wt,
        )


class TMLoss(nn.Module):
    """Coordinate Loss proportional to (negative) TM-Score"""

    def __init__(self, eps=1e-8):
        super(TMLoss, self).__init__()
        self.eps = eps

    def forward(
        self, predicted_coords: torch.Tensor, actual_coords: torch.Tensor, align: bool = True, reduce: bool = True
    ) -> torch.Tensor:  # noqa
        """Compute TM-Loss

        b: batch dimension
        n: sequence/residue dimension

        :param predicted_coords: Predicted Coordinates shape : (b,n,3)
        :param actual_coords: Actual Coordinates, shape: (b,n,3)
        :param align: Whether to align the coordinates before computing loss
        If align is set to true, then score is approximated using kabsch-alignment

        :return: (negative) TM-Score
        """
        if align:
            _, actual_coords = kabsch_align(align_to=predicted_coords, align_from=actual_coords)
        dev = torch.norm((predicted_coords - actual_coords) + self.eps, dim=-1)
        assert dev.shape[0] == actual_coords.shape[0]
        return -calc_tm_torch(dev, reduce=reduce)

    def forward_from_output(
        self,
        output: ModelOutput,
        reduce: bool = True,
        align: bool = True,
        atom_tys: List[str] = None,
        baseline: bool = False,
    ) -> Tensor:
        """Run forward from ModelOutput Object"""
        loss_input = output.get_pred_and_native_coords_and_mask(atom_tys, align_by_kabsch=align)
        pred_coords, native_coords, mask = loss_input
        if baseline:
            pred_coords = output.model_input.get_atom_coords(atom_tys, decoy=True).unsqueeze(0)
        loss = 0
        for i in range(pred_coords.shape[0]):
            loss = loss + self.forward(
                pred_coords[i][mask[i]].unsqueeze(0),
                native_coords[i][mask[i]].unsqueeze(0),
                align=baseline,
                reduce=reduce,
            )
        return loss


class CoordDeviationLoss(nn.Module):
    """
    Computes the loss between two sets of coordinates by minimizing
    l_p distance, for p = 1, 2, ...
    """

    def __init__(
        self,
        p: int = 1,
        atom_tys: Optional[List[str]] = None,
        loss_fn: Optional[Callable] = None,
    ):
        super(CoordDeviationLoss, self).__init__()
        def_loss = get_loss_func(p, beta=BETA, min_clamp=MIN_COORD_CLAMP, max_clamp=MAX_COORD_CLAMP, reduction="none")
        self.loss_fn = default(loss_fn, def_loss)
        self.atom_tys = atom_tys

    def forward(
        self, predicted_coords: Tensor, actual_coords: Tensor, coord_mask: Tensor, align: bool = True
    ) -> Tensor:
        """Compute L_p deviation loss

        b: batch dimension
        n: sequence/residue dimension
        a: atom type

        :param predicted_coords: predicted coordinates, shape (b,n,a,3)
        :param actual_coords: actual coordinates, shape (b,n,a,3)
        :param coord_mask: mask indicating which atoms to compute deviations for,
        shape b,n,a,3
        :param align: Whether to align the coordinates before computing loss
        :return: Mean L_p deviation loss.
        """
        if align:
            predicted, actual = map(lambda x: rearrange(x, "b n a c -> (b a) n c"), (predicted_coords, actual_coords))
            mask = None
            if exists(coord_mask):
                mask = rearrange(coord_mask, "b n a  -> (b a) n ")
            predicted, actual = kabsch_align(align_to=predicted, align_from=actual, mask=mask)
            a = predicted_coords.shape[-2]
            predicted_coords, actual_coords = map(
                lambda x: rearrange(x, "(b a) n c -> b n a c", a=a), (predicted, actual)
            )
        deviations = self.loss_fn(predicted_coords, actual_coords)
        if exists(coord_mask):
            assert coord_mask.shape == deviations.shape[:3], f"{coord_mask.shape},{deviations.shape}"
        return per_residue_mean(deviations, mask=coord_mask)

    def forward_from_output(self, output: ModelOutput, align: bool = True, atom_tys=None) -> Tensor:
        """Run forward from ModelOutput Object"""
        atom_tys = default(atom_tys, self.atom_tys)
        loss_input = output.get_pred_and_native_coords_and_mask(atom_tys, align_by_kabsch=align)
        pred_coords, native_coords, mask = loss_input
        return self.forward(pred_coords, native_coords, mask, align=False)


class CoordinateRelDevLoss(nn.Module):
    def __init__(self, d_cutoff: float = 25):
        super(CoordinateRelDevLoss, self).__init__()
        self.d_cutoff = d_cutoff
        self.loss_fn = get_loss_func(p=1, reduction="none", min_clamp=MIN_COORD_CLAMP, max_clamp=MAX_COORD_CLAMP)

    def forward(
        self, predicted_coords: Tensor, actual_coords: Tensor, mask: Optional[Tensor] = None, reduce: bool = True
    ) -> Tensor:
        """
        :param predicted_coords: shape (b,n,a,3)
        :param actual_coords: shape (b,n,a,3)
        :param mask: shape (b,n,a)
        :return: coordinate-wise l1 loss on relative coordinates for each predicted atom (similar to FAPE loss)
        """
        b = predicted_coords.shape[0]
        pred_rel_coords, actual_rel_coords = map(
            lambda x: rearrange(x, "b n m c -> (b n) m c"),
            (to_rel_dev_coords(predicted_coords), to_rel_dev_coords(actual_coords)),
        )

        dist_mask = torch.norm(actual_rel_coords) < self.d_cutoff if exists(self.d_cutoff) else True
        if exists(mask):
            mask = to_rel_dev_coords(mask) & dist_mask

        pred_rel_coords, actual_rel_coords = kabsch_align(
            pred_rel_coords, actual_rel_coords, apply_translation=False, mask=mask
        )

        pred_rel_coords, actual_rel_coords = map(
            lambda x: rearrange(x, "(b n) m c -> b n m c", b=b), (pred_rel_coords, actual_rel_coords)
        )
        if exists(mask):
            return torch.mean(self.loss_fn(pred_rel_coords, actual_rel_coords)[mask])
        else:
            return torch.mean(self.loss_fn(pred_rel_coords, actual_rel_coords))


class FAPELoss(nn.Module):
    """FAPE Loss"""

    def __init__(
        self,
        d_clamp: float = 10,
        eps: float = 1e-8,
        scale: float = 10,
        clamp_prob: float = 0.9,
        intra_res: bool = False,
    ):
        """Clamped FAPE loss

        :param d_clamp: maximum distance value allowed for loss - all values larger will be clamped to d_clamp
        :param eps: tolerance factor for computing norms (so gradient is defined in sqrt)
        :param scale: (Inverse) Amount to scale loss by (usually equal tho d_clamp)
        :param clamp_prob: Probability with which loss values are clamped [0,1]. If this
        value is not 1, then a (1-clamp_prob) fraction of samples will not have clamping applied to
        distance deviations
        """
        super(FAPELoss, self).__init__()
        self._d_clamp, self.eps, self.scale = d_clamp, eps, scale
        self.clamp_prob = clamp_prob
        self.clamp = lambda diffs: torch.clamp_max(
            diffs, self._d_clamp if random.uniform(0, 1) < self.clamp_prob else max_value(diffs)
        )
        self.intra_res = intra_res

    def forward(
        self,
        pred_coords: Tensor,
        true_coords: Tensor,
        pred_rigids: Optional[Rigids] = None,
        true_rigids: Optional[Rigids] = None,
        coord_mask: Optional[Tensor] = None,
        reduce: bool = True,
        clamp_val: Optional[float] = None,
        mask_fill: float = 0,
    ) -> Tensor:
        """Compute FAPE Loss

        :param pred_coords: tensor of shape (b,n,a,3)
        :param true_coords: tensor of shape (b,n,a,3)
        :param pred_rigids: (Optional) predicted rigid transformation
        :param true_rigids: (Optional) rigid transformation computed on native structure.
        If missing, the transformation is computed assuming true_points[:,:,:3]
        are N,CA, and C coordinates
        :param coord_mask: (Optional) tensor of shape (b,n,a) indicating which atom coordinates
        to compute loss for
        :param reduce: whether to output mean of loss (True), or return loss for each input coordinate
        :param : clamp_val: value to use for clamping (optional)

        :return: FAPE loss between predicted and true coordinates
        """
        b, n, a = true_coords.shape[:3]
        assert pred_coords.ndim == true_coords.ndim == 4

        true_rigids = true_rigids if exists(true_rigids) else Rigids.RigidFromBackbone(true_coords)
        pred_rigids = pred_rigids if exists(pred_rigids) else Rigids.RigidFromBackbone(pred_coords)

        if not self.intra_res:
            pred_coords, true_coords = map(lambda x: repeat(x, "b n a c -> b () (n a) c"), (pred_coords, true_coords))

        # rotate and translate coordinates into local frame
        true_coords_local = true_rigids.apply_inverse(true_coords).detach()
        pred_coords_local = pred_rigids.apply_inverse(pred_coords)

        # compute deviations of predicted and actual coords.
        unclamped_diffs = safe_norm(pred_coords_local - true_coords_local, dim=-1, eps=self.eps)
        diffs = torch.clamp_max(unclamped_diffs, clamp_val) if exists(clamp_val) else self.clamp(unclamped_diffs)

        if exists(coord_mask):
            residue_mask = torch.any(coord_mask, dim=-1, keepdim=True)
            if not self.intra_res:
                coord_mask = repeat(coord_mask, "b n a -> b m (n a)", m=n)
            coord_mask = coord_mask.masked_fill(~residue_mask, False)

        if reduce:
            return (1 / self.scale) * torch.mean(diffs[coord_mask] if exists(coord_mask) else diffs)

        return (1 / self.scale) * (diffs.masked_fill(~coord_mask, mask_fill) if exists(coord_mask) else diffs)

    def forward_from_output(
        self, output: ModelOutput, reduce: bool = True, atom_tys: List[str] = None, mask_fill: float = 0
    ) -> Tensor:
        """Run forward from ModelOutput Object"""
        pred_rigids = output.pred_rigids
        # assert exists(pred_rigids)
        true_rigids = output.native_protein.rigids
        # logger.info(f"computing FAPE loss - predicted rigids {exists(pred_rigids)}, true_rigids: {exists(true_rigids)}")
        loss_input = output.get_pred_and_native_coords_and_mask(atom_tys=atom_tys, align_by_kabsch=False)
        pred_coords, native_coords, mask = loss_input
        return self.forward(
            pred_coords=pred_coords,
            true_coords=native_coords.detach(),
            pred_rigids=pred_rigids,
            true_rigids=true_rigids.detach_all(),
            coord_mask=mask,
            reduce=reduce,
            mask_fill=mask_fill,
        )


class DistanceInvLoss(nn.Module):
    """Distance Based Loss similar to TM-Score on pairwise dists."""

    def __init__(self):
        super(DistanceInvLoss, self).__init__()

    @staticmethod
    def forward(
        predicted_coords: Tensor,
        actal_coords: Tensor,
        coord_mask: Tensor,
        reduce: bool = True,
    ) -> Tensor:
        """Compute Loss"""
        assert predicted_coords.shape == actal_coords.shape
        assert coord_mask.shape == predicted_coords.shape[:3]
        n_res = predicted_coords.shape[1]
        predicted_coords, actal_coords, coord_mask = map(
            lambda x: rearrange(x, "b n a ... -> b (n a) ..."), (predicted_coords, actal_coords, coord_mask)
        )
        pred_dists, native_dists = map(
            lambda x: torch.cdist(x, x),
            (
                predicted_coords,
                actal_coords,
            ),
        )
        prox = 1 / (1 + (torch.square((pred_dists - native_dists) / get_tm_scale(n_res))))
        loss = -prox
        mask = torch.einsum("b i, b j -> b i j", coord_mask, coord_mask)
        if reduce:
            return torch.mean(loss[mask]) if exists(mask) else torch.mean(loss)  # noqa
        return loss.masked_fill(~mask, 0) if exists(mask) else loss  # noqa

    def forward_from_output(self, output: ModelOutput, reduce: bool = True, atom_tys: List[str] = None) -> Tensor:
        """Run forward from ModelOutput Object"""
        loss_input = output.get_pred_and_native_coords_and_mask(atom_tys=atom_tys, align_by_kabsch=False)
        pred_coords, native_coords, mask = loss_input
        return self.forward(pred_coords, native_coords, coord_mask=mask, reduce=reduce)


class LDDTProxLoss(nn.Module):
    """Differentiable Approximation of LDDT score for two sets of coordinates"""

    def __init__(self, weight=1, DMAX=15):
        super(LDDTProxLoss, self).__init__()
        self.activation = nn.Softplus(beta=20)
        self.abs_func = get_loss_func(p=1, beta=0.25, reduction="none")
        self.DMAX = DMAX
        self.weight = weight

    def forward(self, predicted_coords: Tensor, actual_coords: Tensor, coord_mask: Tensor) -> Tensor:
        """Compute (Approximate) LDDT-score

        :param predicted_coords:
        :param actual_coords:
        :param coord_mask:
        :return:

        thresholds = SC_LDDT_THRESHOLDS if output.has_sc else BB_LDDT_THRESHOLDS
        if baseline:
            aln = output.baseline_coords()
            base_coords = aln.baseline_coords[aln.baseline_mask]
            pred_dists = torch.cdist(base_coords, base_coords)
            native_coords = aln.native_coords[aln.native_mask]
            native_dists = torch.cdist(native_coords, native_coords)
        else:
            pred_dists = output.get_atom_dists(predicted=True)
            native_dists = output.get_atom_dists(native=True)

        mask = output.get_initial_nbr_mask(self.DMAX)
        assert dim(mask) == 3, f"{mask.shape}"
        num_nbrs = torch.clamp_min(torch.sum(mask.float(), dim=-1, keepdim=True), 1)
        dists = self.abs_func(pred_dists, native_dists.detach()).reshape_as(mask)
        loss = 0
        for threshold in thresholds:
            intermediate = self.activation(dists - threshold)
            loss = loss - torch.exp(-(intermediate ** 2))
        loss = loss * (1 / num_nbrs)
        return torch.sum(loss[mask]) / (len(thresholds) * mask.shape[1])
        """
        raise Exception("Not yet implemented")


class CenterOfMassLoss(nn.Module):
    def __init__(self):
        super(CenterOfMassLoss, self).__init__()

    def forward(self, pred_chains: List[Tensor], native_chains: List[Tensor]) -> Tensor:
        assert pred_chains[0].ndim == 2, f"{pred_chains[0].ndim}"
        com_pred = torch.cat(list(map(lambda x: torch.mean(x, dim=0, keepdim=True), pred_chains)), dim=0)
        com_native = torch.cat(list(map(lambda x: torch.mean(x, dim=0, keepdim=True), native_chains)), dim=0)
        d_pred, d_native = map(lambda x: torch.cdist(x, x), (com_pred, com_native))
        loss = torch.clamp_max(0.05 * (d_pred - d_native + 4), 0) ** 2
        return torch.sum(loss / max(1, len(pred_chains) * (len(pred_chains) - 1)))

    def forward_from_output(self, output: ModelOutput) -> Tensor:
        chain_indices = output.decoy_protein.chain_indices
        loss_input = output.get_pred_and_native_coords_and_mask(atom_tys=["CA"], align_by_kabsch=False)
        pred_coords, native_coords, mask = loss_input
        pred_chains, native_chains = [], []
        for idxs in chain_indices:
            pred_chains.append(pred_coords.squeeze()[idxs])
            native_chains.append(native_coords.squeeze()[idxs])
        return self.forward(pred_chains, native_chains)
