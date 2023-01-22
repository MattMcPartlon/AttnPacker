import torch
from protein_learning.networks.common.helpers.neighbor_utils import NeighborInfo
from protein_learning.networks.tfn.repr.fiber import Fiber, chunk_fiber
from protein_learning.networks.tfn.tfn import ConvSE3
from protein_learning.networks.common.equivariant.fiber_units import FiberLinear
from typing import Tuple, Dict, Optional
from protein_learning.networks.se3_transformer.attention.se3_attention import SE3Attention
from protein_learning.networks.se3_transformer.se3_attention_config import SE3AttentionConfig
from protein_learning.networks.tfn.tfn_config import TFNConfig


# attention
class TFNAttention(SE3Attention):
    def __init__(
            self,
            fiber_in: Fiber,
            config: SE3AttentionConfig,
            tfn_config: TFNConfig,
            share_keys_and_values: bool = False,
            linear_project_keys: bool = False,
    ):
        """SE(3)-Equivariant attention using TFNs

        :param fiber_in: description of input fiber
        :param config: configuration for SE3Attention
        :param tfn_config: config for TFNs
        :param share_keys_and_values: whether to compute separate key and value features.
        :param linear_project_keys: whether to use  alinear projection to compute keys
        """

        super().__init__(fiber_in=fiber_in, config=config)
        assert not (share_keys_and_values and linear_project_keys)
        self.linear_project_keys = linear_project_keys
        self.share_keys_and_values = share_keys_and_values

        # determine hidden dimensions
        qk_hidden = self.hidden_fiber.scale(2) if linear_project_keys else self.hidden_fiber
        kv_hidden = self.hidden_fiber.scale(2) if not linear_project_keys else self.hidden_fiber
        kv_hidden = self.hidden_fiber if share_keys_and_values else kv_hidden
        # get key, query, and value funcs
        self.tfn_config = tfn_config.override(
            pool=False,
            self_interaction=False,
            edge_dim=self.augmented_edge_dim,
            fiber_in=self.fiber_in,
            fiber_out=kv_hidden
        )
        self.to_kv = ConvSE3(self.tfn_config)
        self.to_q = FiberLinear(fiber_in=self.fiber_in, fiber_out=qk_hidden)

    def get_qkv(
            self,
            features: Dict[str, torch.Tensor],
            edge_info: Tuple[torch.Tensor, NeighborInfo],
            basis: Dict[str, torch.Tensor],
            global_feats: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], ...]:
        if global_feats is not None:
            raise Exception("global feats not yet supported!")
        kv, q = self.to_kv(features=features, edge_info=edge_info, basis=basis), self.to_q(features)
        split_kv = not (self.linear_project_keys or self.share_keys_and_values)
        k, v = (kv, kv) if not split_kv else chunk_fiber(kv, 2)
        q, k = (q, k) if split_kv else chunk_fiber(q, 2)
        return q, k, v
