"""Structure Prediction Model"""
from abc import abstractmethod
from typing import Dict, Any, Tuple, Optional

import numpy as np
import torch
from torch import nn, Tensor

from protein_learning.common.data.data_types.model_input import ModelInput
from protein_learning.common.data.data_types.model_loss import ModelLoss
from protein_learning.common.data.data_types.model_output import ModelOutput
from protein_learning.common.global_constants import get_logger
from protein_learning.features.input_embedding import InputEmbedding
from protein_learning.models.model_abc.protein_model import ProteinModel
from protein_learning.networks.loss.loss_config import LossTy
import numpy as np
from protein_learning.features.feature_config import FeatureName
from torch.utils.data import WeightedRandomSampler

logger = get_logger(__name__)


class StructureModel(ProteinModel):
    """Evoformer + Coord prediction network"""

    def __init__(
            self,
            model: nn.Module,
            input_embedding: InputEmbedding,
            node_dim_hidden: int,
            pair_dim_hidden: int,
            # Recycling Options
            n_cycles: int = -1,
            use_cycles: int = -1,
            project_in: bool = True,
            n_dist_bins_for_recycle=15,
    ):
        super(StructureModel, self).__init__(
            model=model,
            input_embedding=input_embedding,
            node_dim_hidden=node_dim_hidden,
            pair_dim_hidden=pair_dim_hidden,
            project_in=project_in,
        )
        self.n_dist_bins_for_recycle = n_dist_bins_for_recycle
        self.n_cycles, self.use_cycles = max(1, n_cycles), use_cycles
        if n_cycles > 1:
            self.ca_emb = nn.Linear(n_dist_bins_for_recycle, pair_dim_hidden)
            self.residue_recycle_proj = nn.Sequential(
                nn.LayerNorm(node_dim_hidden),
                nn.Linear(node_dim_hidden, node_dim_hidden),
            )
            self.pair_recycle_proj = nn.Sequential(
                nn.LayerNorm(pair_dim_hidden),
                nn.Linear(pair_dim_hidden, pair_dim_hidden),
            )

    def forward(self, sample: ModelInput, use_cycles=None, **kwargs) -> ModelOutput:
        """Run the model"""
        # get input features
        init_residue_feats, init_pair_feats = self.get_input_res_n_pair_feats(sample)
        b, n, device = *init_residue_feats.shape[:-1], init_pair_feats.device  # noqa
        residue_feats, pair_feats, model_out, fwd_kwargs = 0, 0, None, None
        n_cycles = np.random.randint(1, self.n_cycles + 1)
        use_cycles = self.use_cycles if use_cycles is None else use_cycles
        n_cycles = n_cycles if use_cycles <= 0 else use_cycles
        # print(f"using {n_cycles} cycles")
        for cycle in range(n_cycles):
            last_iter = cycle + 1 == n_cycles
            with torch.set_grad_enabled(mode=(last_iter and self.training)):
                res_in = init_residue_feats + residue_feats
                pair_in = init_pair_feats + pair_feats
                if not last_iter:
                    res_in, pair_in = res_in.detach().clone(), pair_in.detach().clone()
                fwd_kwargs = self.get_forward_kwargs(
                    model_input=sample,
                    residue_feats=res_in,
                    pair_feats=pair_in,
                    model_output=model_out,
                    last_cycle=cycle == n_cycles - 1,
                )

                model_out = self.model(
                    **fwd_kwargs
                )

            if cycle < n_cycles - 1:
                with torch.set_grad_enabled(mode=(cycle + 2 == n_cycles and self.training)):
                    residue_feats, pair_feats, coords = self.get_feats_for_recycle(
                        model_out, fwd_kwargs
                    )
                    residue_feats = self.residue_recycle_proj(residue_feats.detach())
                    pair_feats = self.pair_recycle_proj(pair_feats.detach())
                    pair_feats = pair_feats + self.embed_dists(coords.detach())
                    if cycle + 2 <= n_cycles:
                        residue_feats, pair_feats = residue_feats.detach(), pair_feats.detach()

        model_out = self.get_model_output(sample, model_out, fwd_kwargs)
        self.finish_forward()
        return model_out

    def sample(
        self, 
        model_input: ModelInput, 
        temperature: float=0.1, 
        use_cycles=None, 
        **kwargs
        ):
        # get sequence mask

        seq_mask = model_input.input_features.seq_mask
        seq_mask = seq_mask.unsqueeze(0) if seq_mask.ndim==1 else seq_mask
        seq_feat = model_input.input_features[FeatureName.RES_TY]
        curr_seq = seq_feat.get_encoded_data().clone()

        b,n = seq_mask.shape
        design_idxs = [torch.arange(len(seq_mask[i]))[seq_mask[i]] for i in range(b)]
        max_iter = max(map(len,design_idxs))
        seq_loss_fn = self.get_loss_fn()[LossTy.NSR]

        for design_iter in range(max_iter):
            output = self.forward(
                model_input,
                use_cycles=use_cycles,
                **kwargs
            )
        
            # get residue logits
            probs = torch.softmax(seq_loss_fn.get_predicted_logits(output.scalar_output)[...,:20]/temperature, dim=-1)
            # sample 
            for i in range(b):
                sample_idx = min(design_iter,len(design_idxs[i])-1)
                seq_idx = design_idxs[i][sample_idx]
                pred_label = list(WeightedRandomSampler(probs[i,seq_idx],1))[0]
                class_vec = torch.zeros_like(curr_seq[i,seq_idx])
                class_vec[pred_label] = 1
                curr_seq[i,seq_idx] = class_vec

            #update feature
            seq_feat.encoded_data = curr_seq
            model_input.input_features[FeatureName.RES_TY] = seq_feat

            

            





        


        

    def get_input_res_n_pair_feats(self, sample: ModelInput):
        """Input residue and pair features"""
        residue_feats, pair_feats = self.input_embedding(sample.input_features)
        return self.residue_project_in(residue_feats), self.pair_project_in(pair_feats)

    @abstractmethod
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
        pass

    @abstractmethod
    def get_forward_kwargs(
            self,
            model_input: ModelInput,
            residue_feats: Tensor,
            pair_feats: Tensor,
            model_output: Optional = None,
            last_cycle: bool = True,
    ) -> Dict:
        """Get keyword arguments for protein model forward pass"""
        pass

    @abstractmethod
    def get_model_output(
            self,
            model_input: ModelInput,
            fwd_output: Any,
            fwd_input: Dict, **kwargs
    ) -> ModelOutput:
        """Get Model output object
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

    def embed_dists(self, coords: Tensor) -> Tensor:
        """Embed CA distances"""
        b, n = coords.shape[:2]
        idx = min(coords.shape[2] - 1, 1)
        CA = coords[:, :, idx]
        CA_dists = torch.cdist(CA, CA)
        CA_dists = CA_dists * (self.n_dist_bins_for_recycle / 21)
        CA_dists = torch.clamp(CA_dists, 0, self.n_dist_bins_for_recycle - 1).long()
        d_bins = torch.nn.functional.one_hot(CA_dists.detach(), self.n_dist_bins_for_recycle)
        return self.ca_emb(d_bins.float()).reshape(b, n, n, -1)
