"""Model for Fixed backbone design"""
from typing import Tuple, Dict, Any, Optional, List

import numpy as np
from torch import nn, Tensor
from protein_learning.common.data.data_types.model_input import ModelInput
from protein_learning.common.data.data_types.model_loss import ModelLoss
from protein_learning.common.data.data_types.model_output import ModelOutput
from protein_learning.common.global_constants import get_logger
from protein_learning.common.helpers import exists
from protein_learning.features.input_embedding import InputEmbedding
from protein_learning.models.model_abc.structure_model import StructureModel
from protein_learning.networks.geometric_gt.geom_gt_config import GeomGTConfig
from protein_learning.networks.geometric_gt.geometric_graph_transformer import GraphTransformer
from protein_learning.networks.se3_transformer.se3_transformer_config import SE3TransformerConfig
from protein_learning.networks.se3_transformer.tfn_transformer import TFNTransformer

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
        basis_dir: Optional[str] = None,
        **kwargs,
    ):
        super(FBBDesigner, self).__init__(
            model=TFNTransformer(config=se3_config),
            input_embedding=input_embedding,
            node_dim_hidden=gt_configs[0].node_dim,
            pair_dim_hidden=gt_configs[0].pair_dim,
        )
        self.use_tfn = exists(se3_config)

        self.loss_fn = loss_fn

        assert len(gt_configs) > 0
        self.pre_structure = GraphTransformer(gt_configs[0])
        logger.info(f"using pre-structure embedding : {exists(self.pre_structure)}")
        c = gt_configs[0]
        self.node_dim_in, self.pair_dim_in = c.node_dim, c.pair_dim

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

        self.scalar_norm = nn.LayerNorm(se3_config.scalar_dims()[0])
        self.pair_norm = nn.LayerNorm(c.pair_dim)
        self.pre_structure_pair = None
        self.pre_structure_node = None
        self.basis_dir = basis_dir

    def get_input_res_n_pair_feats(self, sample: ModelInput):
        """Input residue and pair features"""
        node, pair = self.input_embedding(sample.input_features)
        return self.residue_project_in(node), self.pair_project_in(pair)

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
            main_device, sec_device = residue_feats.device, residue_feats.device
            net_kwargs = dict(
                node_feats=residue_feats.to(sec_device),
                pair_feats=pair_feats.to(sec_device),
                res_mask=res_mask.unsqueeze(0).to(sec_device),
            )
            residue_feats, pair_feats, *_ = self.pre_structure(**net_kwargs)
            residue_feats, pair_feats = map(lambda x: x.to(main_device), (residue_feats, pair_feats))

        self.pre_structure_pair = pair_feats
        self.pre_structure_node = residue_feats

        return dict(
            node_feats=self.to_structure_node(residue_feats),
            pair_feats=self.to_structure_pair(pair_feats),
            coords=model_input.decoy.atom_coords.unsqueeze(0),
            basis_dir=self.basis_dir,
        )

    def get_model_output(self, model_input: ModelInput, fwd_output: Any, fwd_input: Dict, **kwargs) -> ModelOutput:
        """Get model output"""
        node, pred_coords = fwd_output
        pred_coords[..., :4, :] = model_input.native.atom_coords[..., :4, :].unsqueeze(0)

        return ModelOutput(
            predicted_coords=pred_coords,
            scalar_output=self.scalar_norm(node),
            pair_output=self.pair_norm(self.pre_structure_pair),
            model_input=model_input,
            predicted_atom_tys=None,
            extra=dict(
                pred_rigids=None,
                fape_aux=None,
                chain_indices=model_input.decoy.chain_indices,
                angles=None,
                unnormalized_angles=None,
            ),
        )

    def finish_forward(self):
        """Finish forward pass"""
        self.pre_structure_pair = None
        self.pre_structure_node = None

    def compute_loss(self, output: ModelOutput, **kwargs) -> ModelLoss:
        raise Exception("Only Inference is Supported")
