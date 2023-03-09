"""Default Loss Function for protein learning"""

from enum import Enum
from typing import Optional, List, Tuple, Union

from torch import nn

from protein_learning.networks.common.net_utils import exists, default
from protein_learning.networks.loss.pair_loss import PairDistLossNet
from protein_learning.networks.loss.residue_loss import SequenceRecoveryLossNet, PredLDDTLossNet
from protein_learning.networks.loss.side_chain_loss import SideChainDeviationLoss


class LossTy(Enum):
    """Names for each loss type"""

    FAPE = "fape"
    PLDDT = "plddt"
    NSR = "nsr"
    TM = "tm"
    DIST_INV = "dist-inv"
    PAIR_DIST = "pair-dist"
    COM = "com"
    PAE = "pae"
    VIOL = "violation"
    SC_RMSD = "sc-rmsd"
    RES_FAPE = "res-fape"


class LossConfig:
    """Reconstruction Loss function"""

    def __init__(
        self,
        res_dim: int,
        pair_dim: int,
        output_atom_tys: List[str],
        # Loss Weights
        fape_wt: Optional[float] = None,
        inter_fape_scale: Optional[float] = None,
        plddt_wt: Optional[float] = None,
        nsr_wt: Optional[float] = None,
        tm_wt: Optional[float] = None,
        dist_inv_wt: Optional[float] = None,
        pair_dist_wt: Optional[float] = None,
        com_wt: Optional[float] = None,
        pae_wt: Optional[float] = None,
        violation_wt: Optional[float] = None,
        sc_rmsd_wt: Optional[float] = None,
        sc_rmsd_p: int = 1,
        res_fape_wt: Optional[float] = None,
        # Atom Types to use for each loss
        pair_dist_atom_tys: Optional[List[str]] = None,
        fape_atom_tys: Optional[List[str]] = None,
        tm_atom_tys: Optional[List[str]] = None,
        dist_inv_atom_tys: Optional[List[str]] = None,
        plddt_atom_tys: Optional[List[str]] = None,
        pae_atom_tys: Optional[List[str]] = None,
        # Discretizatoin options
        plddt_bins: int = 10,
        pair_dist_step: float = 0.4,
        pair_dist_max_dist: float = 16,
        pae_max_dist: float = 16,
        pae_step: float = 0.5,
        # violation loss specific
        vdw_wt: float = 1,
        bond_len_wt: float = 1,
        bond_angle_wt: float = 3,
        viol_schedule: Optional[Union[Tuple[float], List[float]]] = None,
        # relative weight for masked regions
        mask_rel_wt: float = 1,
        include_viol_after: int = 0,
        include_nsr_after: int = 0,
        include_plddt_after: int = 0,
        include_pae_after: int = 0,
    ):
        # dims
        self.res_dim = res_dim
        self.pair_dim = pair_dim
        self.output_atoms = default(output_atom_tys, [])
        self.coord_dim = len(self.output_atoms)

        # weights
        self.fape_wt = fape_wt
        self.inter_fape_scale = default(inter_fape_scale, 1.0)
        self.plddt_wt = plddt_wt
        self.nsr_wt = nsr_wt
        self.tm_wt = tm_wt
        self.dist_inv_wt = dist_inv_wt
        self.pair_dist_wt = pair_dist_wt
        self.com_wt = com_wt
        self.pae_wt = pae_wt
        self.violation_wt = violation_wt
        self.sc_rmsd_wt = sc_rmsd_wt
        self.sc_rmsd_p = sc_rmsd_p
        self.res_fape_wt = res_fape_wt

        # atom tys
        self.default_atom = None
        if exists(output_atom_tys):
            self.default_atom = "CA" if "CA" in self.output_atoms else self.output_atoms[0]
        self.pair_dist_atom_tys = default(pair_dist_atom_tys, [self.default_atom] * 2)
        self.fape_atom_tys = default(fape_atom_tys, output_atom_tys)
        self.dist_inv_atom_tys = default(dist_inv_atom_tys, [self.default_atom] * 2)
        self.plddt_atom_tys = default(plddt_atom_tys, [self.default_atom])
        self.tm_atom_tys = default(tm_atom_tys, [self.default_atom])
        self.pae_atom_tys = default(pae_atom_tys, [self.default_atom])

        # discretization
        self.plddt_bins = plddt_bins
        self.pair_dist_step = pair_dist_step
        self.pae_max_dist = pae_max_dist
        self.pair_dist_max_dist = pair_dist_max_dist
        self.pae_step = pae_step

        # violation loss specific
        self.vdw_wt = vdw_wt
        self.bond_len_wt = bond_len_wt
        self.bond_angle_wt = bond_angle_wt
        self.ramp_viol_wt = exists(viol_schedule)
        self.viol_schedule = viol_schedule
        self.include_viol_after = include_viol_after

        # mask_rel_weight
        self.mask_rel_wt = mask_rel_wt

        # nsr/plddt
        self.include_nsr_after = include_nsr_after
        self.include_plddt_after = include_plddt_after
        self.include_pae_after = include_pae_after

    def get_losses_n_weights(self):
        """Get loss function nn.ModuleDict and loss weights"""
        loss_fns, loss_wts = nn.ModuleDict(), {}

        def _register_loss(loss, name, wt):
            name = name if isinstance(name, str) else name.value
            if exists(wt):
                loss_fns[name] = loss
                loss_wts[name] = wt

        _register_loss(SequenceRecoveryLossNet(self.res_dim, hidden_layers=1), LossTy.NSR, self.nsr_wt)
        _register_loss(
            PairDistLossNet(
                dim_in=self.pair_dim,
                atom_tys=self.pair_dist_atom_tys,
                use_hidden=True,
                d_max=self.pair_dist_max_dist,
                step=self.pair_dist_step,
            ),
            LossTy.PAIR_DIST,
            self.pair_dist_wt,
        )

        _register_loss(
            PredLDDTLossNet(self.res_dim, n_hidden_layers=1, n_bins=self.plddt_bins, atom_ty=self.plddt_atom_tys[0]),
            LossTy.PLDDT,
            self.plddt_wt,
        )

        _register_loss(SideChainDeviationLoss(p=self.sc_rmsd_p), LossTy.SC_RMSD, self.sc_rmsd_wt)

        return loss_fns, loss_wts
