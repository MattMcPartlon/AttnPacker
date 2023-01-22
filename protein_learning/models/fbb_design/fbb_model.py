"""Model for Fixed backbone design"""
from typing import Tuple, Dict, Any, Optional, List

import numpy as np
import torch
from einops.layers.torch import Rearrange 
from torch import nn, Tensor
from protein_learning.common.data.data_types.model_input import ModelInput
from protein_learning.common.data.data_types.model_loss import ModelLoss
from protein_learning.common.data.data_types.model_output import ModelOutput
from protein_learning.common.global_constants import get_logger
from protein_learning.common.helpers import exists, default
from protein_learning.common.rigids import Rigids
from protein_learning.features.input_embedding import InputEmbedding
from protein_learning.models.model_abc.structure_model import StructureModel
from protein_learning.networks.geometric_gt.geom_gt_config import GeomGTConfig
from protein_learning.networks.geometric_gt.geometric_graph_transformer import GraphTransformer
from protein_learning.networks.se3_transformer.se3_transformer_config import SE3TransformerConfig
from protein_learning.networks.se3_transformer.tfn_transformer import TFNTransformer
from protein_learning.networks.fa_coord.coords_from_angles import FACoordModule
from protein_learning.networks.loss.side_chain_loss import SideChainDihedralLoss, SideChainDeviationLoss
from protein_learning.protein_utils.sidechains.project_sidechains import atom37_to_torsion_angles, safe_normalize

logger = get_logger(__name__)


def get_model(gt_configs, se3_config):
    if exists(se3_config):
        return TFNTransformer(config=se3_config)
    return GraphTransformer(gt_configs[-1])


class FBBDesigner(StructureModel):  # noqa
    """Model for fixed backbone design"""

    def __init__(
        self,
        gt_configs: List[GeomGTConfig],
        se3_config: Optional[SE3TransformerConfig],
        input_embedding: InputEmbedding,
        loss_fn: nn.Module,
        coord_dim_out: int,
        predict_from_angles: bool = True,
        torsion_loss_weight: float = 0.2,
        use_torsion_loss: bool = True,
    ):
        super(FBBDesigner, self).__init__(
            model=get_model(gt_configs, se3_config),
            input_embedding=input_embedding,
            node_dim_hidden=gt_configs[0].node_dim,
            pair_dim_hidden=gt_configs[0].pair_dim,
        )
        self.torsion_loss_weight = torsion_loss_weight
        self.use_tfn = exists(se3_config)
        
        self.loss_fn = loss_fn


        assert len(gt_configs)>0
        self.pre_structure = GraphTransformer(gt_configs[0]) 
        logger.info(f"using pre-structure embedding : {exists(self.pre_structure)}")
        c = gt_configs[0]
        self.node_dim_in, self.pair_dim_in = c.node_dim, c.pair_dim
        
        self.pair_norm = nn.LayerNorm(c.pair_dim)

        self.to_structure_node = (
            nn.Sequential(nn.LayerNorm(c.node_dim), nn.Linear(c.node_dim, c.node_dim))
            if len(gt_configs) > 0
            else nn.Identity()
        )

        self.to_structure_pair = (
            nn.Sequential(nn.LayerNorm(c.pair_dim), nn.Linear(c.pair_dim, c.pair_dim))
            if len(gt_configs) > 0
            else nn.Identity()
        )
        self.predict_from_angles = predict_from_angles
        self.dihedral_loss = None
        if use_torsion_loss or self.predict_from_angles:
            self.dihedral_loss = SideChainDihedralLoss()
        
        if self.predict_from_angles:
            structure_node_dim = se3_config.scalar_dims()[0] if exists(se3_config) else gt_configs[-1].node_dim
            self.to_points_norm_pre = nn.LayerNorm(c.node_dim)
            self.to_points_norm_post = nn.LayerNorm(structure_node_dim)
            self.to_out_node = nn.Sequential(nn.Linear(c.node_dim+structure_node_dim,structure_node_dim),nn.LayerNorm(structure_node_dim))
            
            self.to_points = FACoordModule(
                node_dim_in= c.node_dim+structure_node_dim,
                dim_hidden = c.node_dim+ structure_node_dim,
                replace_bb=True,
                )
        elif not self.use_tfn:
            self.scalar_norm = nn.LayerNorm(c.node_dim)
            self.to_points = (
                nn.Sequential(
                    nn.LayerNorm(c.node_dim),
                    nn.Linear(c.node_dim, 3 * coord_dim_out, bias=False),
                    Rearrange("b n (d c) -> b n d c", c=3),
                )
                if len(gt_configs) > 1
                else None
            )
        else: 
            self.scalar_norm = nn.LayerNorm(se3_config.scalar_dims()[0])
            self.to_points = None

        self.pre_structure_pair = None
        self.pre_structure_node = None
        self.per_res_sc_rmsd_loss = SideChainDeviationLoss(p=2)

    def set_model_parallel(self, device_indices: List):
        """first index is always device of input and feat embeddings"""
        self.device_indices = device_indices
        devices = list(map(lambda x: f"cuda:{x}", device_indices))
        self.model.to(devices[0])
        if exists(self.pre_structure):
            self.pre_structure.to(devices[-1])
        self.secondary_device = devices[-1]
        self.main_device = devices[0]

    def get_input_res_n_pair_feats(self, sample: ModelInput):
        """Input residue and pair features"""
        node, pair = self.input_embedding(sample.input_features)
        return self.residue_project_in(node), self.pair_project_in(pair)

    def get_coords_and_angles(
        self,
        sequence: Tensor,
        bb_coords:Tensor,
        residue_feats: Tensor,
        rigids,
        CA_posn: int = 1
        ):
        """Get coordinates from output"""
        if self.predict_from_angles:
            assert sequence.ndim == 2

            preds = self.to_points(
                residue_feats = residue_feats,
                seq_encoding = sequence,
                coords = bb_coords,
                rigids = None,
                angles = None,
            )
            return [preds[k] for k in "positions angles unnormalized_angles".split()]
        else:
            if not exists(rigids):
                rigids = Rigids.RigidFromBackbone(bb_coords)
            # predict from node features
            local_points = self.to_points(residue_feats)
            global_points = rigids.apply(local_points)
            global_points[:, :, :4] = bb_coords
            # place points in global frame by applying rigids
            return global_points, None, None


    def get_feats_for_recycle(self, model_out: Any, model_in: Dict) -> Tuple[Tensor, Tensor, Tensor]:
        raise Exception("Recycling not performed!!")

    def get_forward_kwargs(
        self,
        model_input: ModelInput,
        residue_feats: Tensor,
        pair_feats: Tensor,
        model_output: Optional[Dict] = None,
        last_cycle: bool = True,
    ) -> Dict:
        """set up for structure module"""
        res_mask = model_input.decoy.valid_residue_mask
        mask_frac = len(res_mask[res_mask]) / len(res_mask)
        if len(res_mask[res_mask] / len(res_mask)) < 1:
            print(f"[WARNING] : {np.round(100 - 100 * mask_frac, 2)}% of backbone is missing!")

        if exists(self.pre_structure):
            main_device = residue_feats.device
            sec_device = default(self.secondary_device, residue_feats.device)
            net_kwargs = dict(
                node_feats=residue_feats.to(sec_device),
                pair_feats=pair_feats.to(sec_device),
                res_mask=res_mask.unsqueeze(0).to(sec_device),
            )
            residue_feats, pair_feats, *_ = self.pre_structure(**net_kwargs)
            residue_feats, pair_feats = map(lambda x: x.to(main_device), (residue_feats, pair_feats))

        self.pre_structure_pair = pair_feats
        self.pre_structure_node = residue_feats

        if self.use_tfn:
            return dict(
                node_feats=self.to_structure_node(residue_feats),
                pair_feats=self.to_structure_pair(pair_feats),
                coords=model_input.decoy.atom_coords.unsqueeze(0),
            )
        else:
            return dict(
                node_feats=self.to_structure_node(residue_feats),
                pair_feats=self.to_structure_pair(pair_feats),
                true_rigids=model_input.true_rigids,
                rigids=model_input.decoy.rigids, 
                res_mask=res_mask.unsqueeze(0),
                compute_aux=last_cycle,
                update_rigids = False,
            )

    def get_model_output(self, model_input: ModelInput, fwd_output: Any, fwd_input: Dict, **kwargs) -> ModelOutput:
        """Get model output"""
        aux_loss, angle_loss, rigids = None, None, None
        sequence = model_input.native.seq_encoding.unsqueeze(0)
        atom_mask = model_input.native.atom_masks.unsqueeze(0)
        if self.use_tfn:
            node, pred_coords = fwd_output
        else:
            node, _, rigids, aux_loss = fwd_output

        if self.predict_from_angles:
            residue_feats = torch.cat(
                (
                    self.to_points_norm_pre(self.pre_structure_node),
                    self.to_points_norm_post(node)
                ),
                dim=-1
            )
            residue_out = self.to_out_node(residue_feats)
        else:
            residue_feats,residue_out = node, self.scalar_norm(node)

        if self.predict_from_angles or (not self.use_tfn):
            coords, angles, unnormalized_angles = self.get_coords_and_angles(
                sequence = sequence,
                bb_coords = model_input.native.atom_coords[...,:4,:].unsqueeze(0),
                residue_feats = residue_feats,
                rigids = None,
                )
        else:
            coords = pred_coords
            coords[...,:4,:] = model_input.native.atom_coords[...,:4,:].unsqueeze(0)
            ptn = dict(aatype=sequence, all_atom_positions=pred_coords, all_atom_mask=atom_mask)
            angles = safe_normalize(atom37_to_torsion_angles(ptn)["torsion_angles_sin_cos"])
            unnormalized_angles = None



        return ModelOutput(
            predicted_coords=coords,
            scalar_output=residue_out,
            pair_output=self.pair_norm(self.pre_structure_pair),
            model_input=model_input,
            predicted_atom_tys=None,
            extra=dict(
                pred_rigids=rigids,
                fape_aux=aux_loss,
                chain_indices=model_input.decoy.chain_indices,
                angles = angles,
                unnormalized_angles = unnormalized_angles
            ),
        )

    def finish_forward(self):
        """Finish forward pass"""
        self.pre_structure_pair = None
        self.pre_structure_node = None

    def compute_loss(self, output: ModelOutput, **kwargs) -> ModelLoss:
        """Model loss calculation"""
        loss = self.loss_fn(output, **kwargs)
        if output.fape_aux is not None:
            loss.add_loss(loss=output.fape_aux, loss_name="fape-aux", loss_weight=0.2)

        if output.angles is not None:
            torsion_loss, norm_loss = self.dihedral_loss(
                sequence = output.native_protein.seq_encoding.unsqueeze(0),
                unnormalized_angles = default(output.unnormalized_angles,output.angles),
                native_coords = output.native_protein.atom_coords.unsqueeze(0),
                atom_mask = output.native_protein.atom_masks.unsqueeze(0),
            )
            loss.add_loss(loss=torsion_loss, loss_name="sc-torsion", loss_weight=self.torsion_loss_weight)
            if exists(output.unnormalized_angles):
                loss.add_loss(loss = norm_loss, loss_name="torsion-norm",loss_weight=0.01)

        if "compute_zero_wt_loss" in kwargs:
            if kwargs["compute_zero_wt_loss"]:
                prmsd = self.per_res_sc_rmsd_loss.forward_from_output(output)
                loss.add_loss(loss=prmsd, loss_name="sc-per-res-rmsd", loss_weight=0)

        return loss


