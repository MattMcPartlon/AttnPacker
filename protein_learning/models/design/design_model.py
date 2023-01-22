"""Design Model"""

from typing import Optional, Any, Tuple, Dict

import torch
from einops.layers.torch import Rearrange  # noqa
from torch import nn, Tensor

from protein_learning.common.data.data_types.model_input import ModelInput
from protein_learning.common.data.data_types.model_loss import ModelLoss
from protein_learning.common.data.data_types.model_output import ModelOutput
from protein_learning.common.global_constants import get_logger
from protein_learning.common.helpers import maybe_add_batch
from protein_learning.common.protein_constants import SC_ATOMS
from protein_learning.features.input_embedding import InputEmbedding
from protein_learning.models.model_abc.structure_model import StructureModel
from protein_learning.models.utils.esm_embedder import ESMInputEmbedder
from protein_learning.networks.common.helpers.neighbor_utils import get_neighbor_info
from protein_learning.networks.evoformer.evoformer import Evoformer, EvoformerConfig
from protein_learning.networks.se3_transformer.se3_transformer_config import SE3TransformerConfig
from protein_learning.networks.se3_transformer.tfn_transformer import TFNTransformer
from protein_learning.common.helpers import exists

logger = get_logger(__name__)


class Designer(StructureModel):
    """Protein Design Model"""

    def __init__(
            self,
            input_embedding: InputEmbedding,
            loss_fn: nn.Module,
            evo_config: Optional[EvoformerConfig],
            se3_trans_config: Optional[SE3TransformerConfig],
            freeze_evo: bool = False,
            freeze_tfn: bool = False,
            esm_embedder: Optional[ESMInputEmbedder] = None,
    ):
        super(Designer, self).__init__(
            model=TFNTransformer(se3_trans_config, freeze=freeze_tfn),
            input_embedding=input_embedding,
            node_dim_hidden=evo_config.node_dim,
            pair_dim_hidden=evo_config.edge_dim,
        )
        self.evoformer = Evoformer(evo_config, freeze=freeze_evo)

        node_dim, pair_dim = evo_config.node_dim, evo_config.edge_dim
        self.loss_fn = loss_fn

        self.node_norm = nn.LayerNorm(node_dim)
        self.pair_norm = nn.LayerNorm(pair_dim)

        self.esm_embedder = esm_embedder

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

    def get_feats_for_recycle(
            self,
            model_out: Any,
            model_in: Dict
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Should not be called"""
        raise Exception("Not implemented!")

    def get_forward_kwargs(
            self,
            model_input: ModelInput,
            residue_feats: Tensor,
            pair_feats: Tensor,
            model_output: Optional = None,
            last_cycle: bool = True,
    ) -> Dict:
        """Get input for structure module"""
        CA = maybe_add_batch(model_input.get_atom_coords(["CA"], decoy=True), 3)
        N_C_O = maybe_add_batch(model_input.get_atom_coords(["N", "C", "O"], decoy=True), 3)
        nbr_info = get_neighbor_info(CA.squeeze(-2), max_radius=16, top_k=16)
        residue_feats, pair_feats = self.evoformer(residue_feats, pair_feats, adj_mask=nbr_info.full_mask)
        feats = {"0": residue_feats, "1": N_C_O - CA}
        return dict(feats=feats, edges=pair_feats, neighbor_info=nbr_info)

    def get_model_output(
            self,
            model_input: ModelInput,
            fwd_output: Any,
            fwd_input: Dict, **kwargs
    ) -> ModelOutput:
        """Get Model output object
        """
        CA = fwd_input["neighbor_info"].coords
        assert torch.allclose(CA.squeeze(), model_input.decoy["CA"])
        return ModelOutput(
            predicted_coords=CA.unsqueeze(-2) + fwd_output["1"],
            scalar_output=self.node_norm(fwd_output["0"].squeeze(-1)),
            pair_output=self.pair_norm(fwd_input["edges"]),
            model_input=model_input,
            predicted_atom_tys=SC_ATOMS,
        )

    def compute_loss(self, output: ModelOutput, **kwargs) -> ModelLoss:
        """Compute model loss"""
        return self.loss_fn(output, **kwargs)
