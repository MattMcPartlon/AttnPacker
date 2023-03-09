"""Side Chain Loss Functions"""
import math

import torch
from torch import nn, Tensor
from typing import Optional
from protein_learning.common.data.data_types.model_output import ModelOutput
from protein_learning.common.helpers import masked_mean
from protein_learning.common.helpers import exists
import protein_learning.common.protein_constants as pc
from protein_learning.protein_utils.sidechains.sidechain_rigid_utils import atom37_to_torsion_angles
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()


@typechecked
def get_torsion_angles_n_mask(
    coords: TensorType["batch", "seq", 37, 3],
    seq: TensorType["batch", "seq"],
    atom_mask: TensorType["batch", "seq", 37],
):
    ptn = dict(aatype=seq, all_atom_positions=coords, all_atom_mask=atom_mask)
    ptn = atom37_to_torsion_angles(ptn)
    angles = ptn["torsion_angles_sin_cos"]
    return angles, ptn["torsion_angles_mask"]


class SideChainDihedralLoss(nn.Module):
    """
    Loss on predicted sidechain dihedral angles
    """

    def __init__(self, *args, **kwargs):  # noqa
        super(SideChainDihedralLoss, self).__init__()
        stack_list = lambda lst: torch.stack([torch.tensor(x) for x in lst], dim=0)
        self.register_buffer(
            "chi_pi_periodic_mask",
            stack_list(pc.CHI_PI_PERIODIC_LIST).bool(),
        )
        self.register_buffer("chi_angle_mask", stack_list(pc.CHI_ANGLES_MASK_LIST).bool())
        self.loss_fn = torch.nn.SmoothL1Loss(reduction=None, beta=0.1)

    @typechecked
    def forward(
        self,
        sequence: TensorType["batch", "seq"],
        unnormalized_angles: TensorType["batch", "seq", "no_angles", 2],  # un-normalized predicted angles
        native_coords: TensorType["batch", "seq", 37, 3],
        atom_mask: TensorType["batch", "seq", 37, torch.bool],
    ):
        assert torch.max(sequence) < self.chi_angle_mask.shape[0], f"{torch.max(sequence)},{self.chi_angle_mask.shape}"
        # b,n,7,2 and b,n,7
        a_gt, a_mask = get_torsion_angles_n_mask(native_coords, sequence, atom_mask)
        chi_pi_periodic_mask = self.chi_pi_periodic_mask[sequence]  # b,n,4
        a_gt, a_mask = a_gt[..., -4:, :], a_mask[..., -4:]
        a_alt_gt = a_gt.clone()
        assert chi_pi_periodic_mask.shape == a_mask.shape, f"{chi_pi_periodic_mask.shape},{a_mask.shape}"
        assert chi_pi_periodic_mask.shape == a_gt.shape[:3], f"{chi_pi_periodic_mask.shape},{a_gt.shape}"
        a_alt_gt[chi_pi_periodic_mask] = a_alt_gt[chi_pi_periodic_mask] * -1
        a_alt_gt, a_gt = map(lambda x: x / (torch.norm(x, dim=-1, keepdim=True) + 1e-8), (a_alt_gt, a_gt))

        not_nan_mask = ~torch.any(torch.isnan(a_gt), dim=-1)
        a_mask = self.chi_angle_mask[sequence] & a_mask.bool() & not_nan_mask
        return self._forward(a=unnormalized_angles[..., -4:, :], a_gt=a_gt, a_alt_gt=a_alt_gt, mask=a_mask)

    @typechecked
    def _forward(
        self,
        a: TensorType["batch", "seq", "no_angles", 2],
        a_gt: TensorType["batch", "seq", "no_angles", 2],
        a_alt_gt: TensorType["batch", "seq", "no_angles", 2],
        mask: TensorType["batch", "seq", "no_angles", torch.bool],
    ):
        safe_norm = lambda x, kd=False: torch.sqrt(1e-8 + torch.sum(torch.square(x), keepdim=kd, dim=-1))

        norm = safe_norm(a)
        a = a / norm.unsqueeze(-1)

        a, a_gt, a_alt_gt = map(lambda x: x[mask], (a, a_gt, a_alt_gt))
        diff_norm_gt = safe_norm(a - a_gt)
        diff_norm_alt_gt = safe_norm(a - a_alt_gt)
        min_diff = torch.minimum(diff_norm_gt**2, diff_norm_alt_gt**2)

        l_torsion = torch.mean(min_diff)
        l_angle_norm = torch.mean(torch.square(norm - 1))

        return l_torsion, l_angle_norm


class SideChainDeviationLoss(nn.Module):
    """
    Loss on predicted sidechain residue RMSD's
    """

    def __init__(self, p=2, *args, **kwargs):  # noqa
        super(SideChainDeviationLoss, self).__init__()
        self.p = p

    @typechecked
    def forward(
        self,
        predicted_coords: TensorType["batch", "seq", 33, 3],
        actual_coords: TensorType["batch", "seq", 33, 3],
        sc_atom_mask: TensorType["batch", "seq", 33, torch.bool],
    ) -> Tensor:
        """Per-Residue Side-Chain RMSD Loss
        :param predicted_coords: predicted side-chain coordinates (b,n,33,3)
        :param actual_coords: true side-chain coordinates (b,n,33,3)
        :param sc_atom_mask: side-chain atom mask. Side-chain atoms are assumed to
        be in the order given SC_ATOMS list (see common/protein_constants.py).
        :return: Average residue side-chain RMSD
        """
        deviations = torch.square(predicted_coords - actual_coords)
        residue_mask = torch.any(sc_atom_mask, dim=-1)
        if self.p == 2:
            per_res_mse = masked_mean(torch.sum(deviations, dim=-1), sc_atom_mask, dim=-1)
            return torch.mean(torch.sqrt(per_res_mse[residue_mask] + 1e-8))
        elif self.p == 1:
            deviations = torch.sum(torch.sqrt(deviations + 1e-8), dim=-1)
            return torch.mean(masked_mean(deviations, sc_atom_mask, dim=-1)[residue_mask])
        else:
            raise Exception("Not Implemented", self.p)

    def forward_from_output(self, output: ModelOutput, **kwargs) -> Tensor:
        """Run forward from ModelOutput Object"""
        pred, nat, mask = output.get_pred_and_native_coords_and_mask()
        assert (nat.ndim == pred.ndim == 4) and mask.ndim == 3
        return self.forward(
            predicted_coords=pred[..., 4:, :], actual_coords=nat[..., 4:, :], sc_atom_mask=mask[..., 4:]
        )
