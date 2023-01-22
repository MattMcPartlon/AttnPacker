"""Model for Complex Design"""
from typing import Tuple, Dict, Any, Optional, List

import numpy as np
import torch
from einops.layers.torch import Rearrange  # noqa
from torch import nn, Tensor

from protein_learning.common.data.data_types.model_input import ModelInput
from protein_learning.common.data.data_types.model_loss import ModelLoss
from protein_learning.common.data.data_types.model_output import ModelOutput
from protein_learning.common.global_constants import get_logger
from protein_learning.common.helpers import exists
from protein_learning.common.rigids import Rigids
from protein_learning.features.input_embedding import InputEmbedding
from protein_learning.models.model_abc.structure_model import StructureModel
from protein_learning.models.utils.esm_embedder import ESMInputEmbedder
from protein_learning.networks.geometric_gt.geom_gt_config import GeomGTConfig
from protein_learning.networks.geometric_gt.geometric_graph_transformer import GraphTransformer

logger = get_logger(__name__)


class ComplexDesigner(StructureModel):  # noqa
    """Model for designing complexes"""

    def __init__(
            self,
            gt_configs: List[GeomGTConfig],
            input_embedding: InputEmbedding,
            loss_fn: nn.Module,
            coord_dim_out: int,
            n_cycles: int = 1,
            use_cycles: int = -1,
            complex_weight: float = 1,
            esm_embedder: Optional[ESMInputEmbedder] = None,
    ):
        super(ComplexDesigner, self).__init__(
            model=GraphTransformer(
                gt_configs[-1]
            ),
            input_embedding=input_embedding,
            node_dim_hidden=gt_configs[-1].node_dim,
            pair_dim_hidden=gt_configs[-1].pair_dim,
            n_cycles=n_cycles,
            use_cycles=use_cycles,
        )
        c = gt_configs[0]
        self.loss_fn = loss_fn
        self.scalar_norm = nn.LayerNorm(c.node_dim)
        self.pair_norm = nn.LayerNorm(c.pair_dim)
        self.complex_weight = complex_weight
        self.esm_embedder = esm_embedder
        logger.info(f"using esm : {exists(self.esm_embedder)}")

        self.pre_structure = GraphTransformer(c) \
            if len(gt_configs) > 1 else None
        logger.info(f"using pre-structure embedding : {exists(self.pre_structure)}")

        self.to_structure_node = nn.Sequential(
            nn.LayerNorm(c.node_dim),
            nn.Linear(c.node_dim, c.node_dim)
        ) if len(gt_configs) > 1 else nn.Identity()

        self.to_structure_pair = nn.Sequential(
            nn.LayerNorm(c.pair_dim),
            nn.Linear(c.pair_dim, c.pair_dim)
        ) if len(gt_configs) > 1 else nn.Identity()

        self.to_points = nn.Sequential(
            nn.LayerNorm(c.node_dim),
            nn.Linear(c.node_dim, 3 * coord_dim_out, bias=False),
            Rearrange("b n (d c) -> b n d c", c=3),
        )

    def get_input_res_n_pair_feats(self, sample: ModelInput):
        """Input residue and pair features"""
        node, pair = self.input_embedding(sample.input_features)
        node, pair = self.residue_project_in(node), self.pair_project_in(pair)
        if exists(self.esm_embedder):
            assert hasattr(sample, "esm_feats")
            node, pair = self.esm_embedder(
                node_feats=node,
                pair_feats=pair,
                feat_gen_kwargs=sample.esm_feats
            )
        return node, pair

    def get_coords(
            self,
            node_feats: Tensor,
            rigids: Optional[Rigids],
            CA_posn: int = 1
    ) -> Tensor:
        """Get coordinates from output"""
        # predict from node features
        local_points = self.to_points(node_feats)
        local_points[:, :, CA_posn] = torch.zeros_like(local_points[:, :, 0])
        # place points in global frame by applying rigids
        return rigids.apply(local_points)

    def get_feats_for_recycle(
            self,
            model_out: Any,
            model_in: Dict
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Get residue pair and coordinate features from
        output (and input) of model.forward()

        (1) Residue features (b,n,d_res)
        (2) Pair Features (b,n,n,d_pair)
        (3) Coordinates (b,n,a,3)
        """
        node_feats, pair_feats, rigids, aux_loss = model_out
        assert exists(rigids)
        coords = self.get_coords(node_feats, rigids, CA_posn=1)
        return node_feats, pair_feats, coords

    def get_forward_kwargs(
            self,
            model_input: ModelInput,
            residue_feats: Tensor,
            pair_feats: Tensor,
            model_output: Optional[Dict] = None,
            last_cycle: bool = True,
    ) -> Dict:
        """set up for structure module"""
        res_mask = model_input.native.valid_residue_mask
        mask_frac = len(res_mask[res_mask]) / len(res_mask)
        if mask_frac < 1:
            print(f"[WARNING] : {np.round(100 - 100 * mask_frac, 2)}% of backbone is missing!")

        net_kwargs = dict(
            node_feats=residue_feats,
            pair_feats=pair_feats,
            res_mask=res_mask.unsqueeze(0),
        )
        if exists(self.pre_structure):
            residue_feats, pair_feats, *_ = self.pre_structure(
                **net_kwargs
            )
        assert hasattr(model_input.extra, "decoy_rigids")
        return dict(
            node_feats=self.to_structure_node(residue_feats),
            pair_feats=self.to_structure_pair(pair_feats),
            true_rigids=model_input.true_rigids,
            rigids=model_input.extra.decoy_rigids,  # noqa
            res_mask=res_mask.unsqueeze(0),
            compute_aux=last_cycle,
        )

    def get_model_output(
            self,
            model_input: ModelInput,
            fwd_output: Any,
            fwd_input: Dict,
            **kwargs
    ) -> ModelOutput:
        """Get model output"""
        node, pair, rigids, aux_loss = fwd_output
        return ModelOutput(
            predicted_coords=self.get_coords(node, rigids=rigids),
            scalar_output=self.scalar_norm(node),
            pair_output=self.pair_norm(pair),
            model_input=model_input,
            predicted_atom_tys=None,
            extra=dict(
                pred_rigids=rigids,
                fape_aux=aux_loss,
                chain_indices=model_input.decoy.chain_indices,
            )
        )

    def finish_forward(self):
        """Finish forward pass"""
        pass

    def compute_loss(self, output: ModelOutput, **kwargs) -> ModelLoss:
        """Model loss calculation"""
        loss_scale = 1
        if not output.model_input.native.is_complex:
            loss_scale = self.complex_weight
        loss = self.loss_fn(output, **kwargs)
        if output.fape_aux is not None:
            loss.add_loss(loss=output.fape_aux, loss_name="fape-aux")
        loss.scale(loss_scale)
        return loss
