"""Structure Prediction Model"""
from abc import abstractmethod
from typing import Dict, Any, List

import torch
from torch import nn, Tensor

from protein_learning.common.data.data_types.model_input import ModelInput
from protein_learning.common.data.data_types.model_loss import ModelLoss
from protein_learning.common.data.data_types.model_output import ModelOutput
from protein_learning.common.global_constants import get_logger
from protein_learning.features.input_embedding import InputEmbedding

logger = get_logger(__name__)


class ProteinModel(nn.Module):
    """Base class for all protein learning models"""

    def __init__(
        self,
        model: nn.Module,
        input_embedding: InputEmbedding,
        node_dim_hidden: int,
        pair_dim_hidden: int,
        project_in: bool = True,
    ):
        super(ProteinModel, self).__init__()
        self.model = model
        self.input_embedding = input_embedding
        s_in, p_in = self.input_embedding.dims
        self.node_in, self.pair_in = s_in, p_in
        # input projections
        self.project_in = project_in
        if project_in:
            self.residue_project_in = nn.Linear(s_in, node_dim_hidden)
            self.pair_project_in = nn.Linear(p_in, pair_dim_hidden)

        self.device_indices = None
        self.secondary_device = None
        self.main_device = None

    def forward(self, sample: ModelInput, **kwargs) -> ModelOutput:
        """Run the model"""
        # get input features
        residue_feats, pair_feats = self.input_embedding(sample.input_features)
        if self.project_in:
            residue_feats = self.residue_project_in(residue_feats)
            pair_feats = (self.pair_project_in(pair_feats),)

        fwd_kwargs = self.get_forward_kwargs(
            model_input=sample,
            residue_feats=residue_feats,
            pair_feats=pair_feats,
        )

        model_output = self.get_model_output(
            model_input=sample,
            fwd_output=self.model(**fwd_kwargs),
            fwd_input=fwd_kwargs,
        )

        self.finish_forward()
        return model_output

    def set_model_parallel(self, device_indices: List):
        """first index is always device of input and feat embeddings"""
        devices = ["cpu"]
        if torch.cuda.is_available():
            self.device_indices = device_indices
            devices = list(map(lambda x: f"cuda:{x}", device_indices))
        self.secondary_device = devices[-1]
        self.main_device = devices[0]

    @abstractmethod
    def get_forward_kwargs(
        self,
        model_input: ModelInput,
        residue_feats: Tensor,
        pair_feats: Tensor,
        **kwargs,
    ) -> Dict:
        """Get keyword arguments for protein model forward pass

        Params:
            model_input: moel input representing training sample
            residue_feats: (linearly projected) residue features from input embedding
            pair_feats: (linearly projected) pair features from input embedding

        Return: All input kwargs needed for structure module
        """
        pass

    @abstractmethod
    def get_model_output(self, model_input: ModelInput, fwd_output: Any, fwd_input: Dict, **kwargs) -> ModelOutput:
        """Get Model output object from

        (1) output of model forward
        (2) input kwargs of forward pass
        (3) model input
        """
        pass

    @abstractmethod
    def compute_loss(self, output: ModelOutput, **kwargs) -> ModelLoss:
        """Compute the loss"""
        pass

    def finish_forward(self) -> None:
        """Called at the very end of the forward pass
        This is the place to clear any state information you may have saved, e.g.
        """
        pass

    def get_loss_fn(self):
        return getattr(self, "loss_fn", None)


def _log_names_n_shapes(x, tab=""):
    for k, v in x.items():
        if isinstance(v, dict):
            _log_names_n_shapes(v, tab=tab + "    ")
        else:
            shape = v.shape if torch.is_tensor(v) else None
            logger.info(f"{k} : {shape}, ty: {type(v)}")
