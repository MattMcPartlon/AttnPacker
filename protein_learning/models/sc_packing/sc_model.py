"""Side-chain Packing Model"""
from protein_learning.common.data.data_types.model_loss import ModelLoss
from protein_learning.common.data.data_types.model_output import ModelOutput
from protein_learning.common.global_config import GlobalConfig
from protein_learning.features.input_embedding import InputEmbedding
from typing import List
import torch
from protein_learning.common.global_constants import get_logger
from protein_learning.networks.loss.pair_loss import PairDistLossNet
from protein_learning.networks.loss.side_chain_loss import SideChainDihedralLoss, SideChainDeviationLoss
from protein_learning.common.protein_constants import ALL_ATOM_POSNS
from protein_learning.networks.structure_net.structure_net import StructureNet

logger = get_logger(__name__)


class SCPacker(StructureNet):
    """VAE for protein imputation"""

    def __init__(
            self,
            model_config: GlobalConfig,
            input_embedding: InputEmbedding,
            scalar_dim_hidden: int,
            pair_dim_hidden: int,
            coord_dim_hidden: int,
            evoformer_scalar_heads_n_dim: List[int],
            evoformer_pair_heads_n_dim: List[int],
            tfn_heads: int,
            tfn_head_dims: List[int],
            evoformer_depth: int = 6,
            tfn_depth: int = 6,
            use_dist_sim: bool = False,
            append_rel_dist: bool = False,
            use_coord_layernorm: bool = False
    ):
        super(SCPacker, self).__init__(
            model_config=model_config,
            input_embedding=input_embedding,
            scalar_dim_hidden=scalar_dim_hidden,
            pair_dim_hidden=pair_dim_hidden,
            coord_dim_hidden=coord_dim_hidden,
            evoformer_scalar_heads_n_dim=evoformer_scalar_heads_n_dim,
            evoformer_pair_heads_n_dim=evoformer_pair_heads_n_dim,
            tfn_heads=tfn_heads,
            evoformer_depth=evoformer_depth,
            tfn_depth=tfn_depth,
            tfn_head_dims=tfn_head_dims,
            use_dist_sim=use_dist_sim,
            append_rel_dist=append_rel_dist,
            use_coord_layernorm=use_coord_layernorm,
        )
        # loss
        pair_atom_tys = "CB CB CA CG CA CG1 CA CG2 CA OG1".split(" ")
        self.pair_loss = PairDistLossNet(dim_in=pair_dim_hidden, atom_tys=pair_atom_tys)
        self.sc_rmsd_loss = SideChainDeviationLoss()
        self.dihedral_loss = SideChainDihedralLoss()

    def augment_output(self, model_input, coords, scalar_out, evo_scalar, neighbor_info, pair):
        return {"neighbor_info": neighbor_info, "evo_scalar": evo_scalar}

    def compute_loss(self, output: ModelOutput, **kwargs) -> ModelLoss:
        """Compute model loss"""
        logger.info("computing sample loss")
        loss = ModelLoss()
        native = output.model_input.native
        decoy = output.model_input.decoy
        extra = output.model_input.extra
        full_pred = torch.cat((decoy.bb_atom_coords.unsqueeze(0), output.predicted_coords), dim=-2)
        valid_res_mask = torch.all(native.bb_atom_mask.unsqueeze(0), dim=-1)
        pair_mask = torch.einsum("b i, b j -> b i j", valid_res_mask.float(), valid_res_mask.float())

        # pair loss
        loss.add_loss(
            loss=self.pair_loss.forward(
                pair_output=output.pair_output,
                atom_ty_map=ALL_ATOM_POSNS,
                atom_coords=native.atom_coords.unsqueeze(0),
                atom_masks=native.atom_masks.unsqueeze(0),
                pair_mask=pair_mask,
            ),
            loss_name="pair-dist",
            loss_weight=1.,
        )

        # side-chain rmsd
        aligned_native = extra.align_symm_atom_coords(full_pred, native.atom_coords.unsqueeze(0))  # noqa
        loss.add_loss(
            loss=self.sc_rmsd_loss.forward(
                predicted_coords=output.predicted_coords,
                actual_coords=aligned_native[:, :, 4:],
                sc_atom_mask=native.sc_atom_mask.unsqueeze(0),
            ),
            loss_weight=1,
            loss_name="sc-rmsd",
        )

        # side-chain dihedral
        pred_chis = extra.get_sc_dihedral(full_pred)  # noqa
        loss.add_loss(
            loss=self.dihedral_loss.forward(
                pred_chis=pred_chis,
                actual_chis=extra.native_chi_angles,  # noqa
                chi_mask=extra.chi_mask,  # noqa
                chi_pi_periodic=extra.chi_pi_periodic,  # noqa
            ),
            loss_name="sc_dihedral",
            loss_weight=0.,

        )

        return loss
