"""Input Embedding with ESM-1b or ESM-MSA features"""
from __future__ import annotations

from typing import Dict, Optional, Any, Tuple

import torch
from einops import repeat, rearrange  # noqa
from einops.layers.torch import Rearrange  # noqa
from torch import nn, Tensor

from protein_learning.common.helpers import exists
from protein_learning.networks.attention.node_attention import NodeUpdateBlock
from protein_learning.models.utils.esm_input import ESMFeatGen

ESM_1B_NODE_REPR_DIM, ESM_1B_PAIR_REPR_DIM = 1280, 660
ESM_MSA_NODE_REPR_DIM, ESM_MSA_PAIR_REPR_DIM = 768, 144
ESM_1B_NODE_REPR_IDX, ESM_MSA_NODE_REPR_IDX = 33, 12

class MissingParam(nn.Module):  # noqa
    """Representation of missing parameter"""

    def __init__(self, dim):
        super(MissingParam, self).__init__()
        self.dim = dim
        self.param = nn.Parameter(
            self.random_std_gaussian(dim)
        )
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, b: Optional[int] = None, mult: Optional[int] = None):
        """Get parameter or b*mult copies of it"""
        val = self.param * self.scale
        if exists(b):
            val = repeat(val, "i -> b i", b=b)
        if exists(mult):
            val = repeat(val, "... i -> ... m i", m=mult)
        return val

    @staticmethod
    def random_std_gaussian(dim):
        """random gaussian(ish) tensor with mean 0 and variance 1

        meant to look like the output of layernorm
        """
        x = torch.randn(dim)
        x = (x - torch.mean(x)) / torch.std(x)
        return x.detach()


class ESMInputEmbedder(nn.Module):  # noqa
    """Embeds input node and pair features by concatenating with ESM model
    output representations (for node features) and embedded attention matrices (for
    pair features).

    the esm node representation is passed through a single attention layer,
    and the esm attention logits are passed through a linear projection before
    concatenation with the given input features.

    The input features (independent of ESM output) should have shape:
        Node: (b,n,node_dim_in)
        Pair: (b,n,n,pair_dim_in)

    The output will have shape:
        Node: (b,n,node_dim_out)
        Pair: (b,n,n,pair_dim_out)

    The process (in pseudocode):
        esm_embed(node_feats, pair_feats)
            esm_node, esm_pair = self.esm_model(input)
            esm_node = self.esm_node_embed(esm_node) # attention
            esm_pair = self.esm_pair_embed(esm_pair) # linear
            node_out = self.node_project_out(concat(node_feats,esm_node)) # (b,n,node_dim_out)
            pair_out = self.pair_project_out(concat(pair_feats,esm_pair)) # (b,n,n,pair_dim_out)

    """

    def __init__(
            self,
            use_esm_1b: bool,
            use_esm_msa: bool,
            node_dim_in: int,
            pair_dim_in: int,
            node_dim_out: int,
            pair_dim_out: int,
            feat_gen: ESMFeatGen,
            esm_node_emb_dim: int = 256,
            esm_pair_emb_dim: int = 128,
            allow_missing: bool = True,
    ):
        super(ESMInputEmbedder, self).__init__()
        assert use_esm_1b ^ use_esm_msa, "must use exactly one of esm1b or esm_msa"
        self.feat_gen = feat_gen

        self.use_esm_1b, self.use_esm_msa = use_esm_1b, use_esm_msa
        self.node_dim_in, self.node_dim_out = node_dim_in, node_dim_out
        self.pair_dim_in, self.pair_dim_out = pair_dim_in, pair_dim_out

        # esm feature embeddings
        esm_node_repr_dim = ESM_1B_NODE_REPR_DIM if use_esm_1b else ESM_MSA_NODE_REPR_DIM
        esm_pair_repr_dim = ESM_1B_PAIR_REPR_DIM if use_esm_1b else ESM_MSA_PAIR_REPR_DIM

        # embedding for concatenation of input and esm features
        self.esm_node_norm = nn.LayerNorm(esm_node_repr_dim)
        self.esm_pair_norm = nn.LayerNorm(esm_pair_repr_dim)
        self.joint_node = nn.Linear(node_dim_in + esm_node_repr_dim, node_dim_out)
        self.joint_pair = nn.Linear(pair_dim_in + esm_pair_repr_dim, pair_dim_out)

        # fill values for missing features
        self.allow_missing = allow_missing
        if allow_missing:
            self.missing_esm_node = MissingParam(esm_node_emb_dim)
            self.missing_esm_pair = MissingParam(esm_pair_emb_dim)

    def forward(
            self,
            node_feats: Tensor,
            pair_feats: Tensor,
            feat_gen_kwargs: Dict,
    ):
        """Joint embedding of input and esm node/pair features"""
        # Sanity Checks

        assert node_feats.ndim == 3
        assert pair_feats.ndim == 4
        esm_output = self.feat_gen.get_esm_feats(**feat_gen_kwargs)

        b, n, device = *node_feats.shape[:2], node_feats.device  # noqa
        missing_indices = esm_output["missing_indices"]
        assert self.allow_missing or not exists(missing_indices)
        missing_indices = missing_indices.to(device) if exists(missing_indices) else None
        n_missing = len(missing_indices) if exists(missing_indices) else 0
        n_esm = n - n_missing

        if not exists(esm_output["node"]):
            assert exists(missing_indices) and len(missing_indices) == n, \
                f"{type(missing_indices)},{n_missing}"

        # embeddings for esm output
        esm_node_feats, esm_pair_feats = None, None
        if exists(esm_output["node"]):
            esm_node_feats, esm_pair_feats = self.get_esm_feat_embeddings(esm_output, device=device)
            assert esm_node_feats.shape[1] == n_esm, \
                f"expected {n_esm} feats, num in feats: {n}, " \
                f"esm node : {esm_node_feats.shape}. num missing in feats : {n_missing}"

        # create full esm features by infill with missing
        if exists(missing_indices):
            esm_node_feats, esm_pair_feats = self.fill_missing(
                missing_indices,
                esm_node=esm_node_feats,
                esm_pair=esm_pair_feats,
                n_feats=n,
                batch_size=b,
                device=device
            )
        assert esm_node_feats.shape[-2] == n, f"{n}, {esm_node_feats.shape}"
        assert esm_pair_feats.shape[-2] == n, f"{n}, {esm_pair_feats.shape}"

        # concat with input node and pair features and project out
        out_node = self.joint_node(
            torch.cat(
                (esm_node_feats, node_feats), dim=-1
            )
        )
        out_pair = self.joint_pair(
            torch.cat(
                (esm_pair_feats, pair_feats), dim=-1
            )
        )
        return out_node, out_pair

    def fill_missing(
            self,
            missing_indices,
            esm_node: Optional[Tensor],
            esm_pair: Optional[Tensor],
            n_feats: int,
            batch_size: int,
            device: Any,
    ):
        """Infill missing information"""
        b, n, n_missing = batch_size, n_feats, len(missing_indices)
        missing_node_feats = self.missing_esm_node(b, n_missing)
        n_missing_pair = int(2*n_missing * (n - n_missing) + n_missing ** 2)
        missing_pair_feats = self.missing_esm_pair(b, n_missing_pair)
        missing_node_mask, missing_pair_mask = self._get_missing_masks(n, missing_indices)
        assert missing_pair_mask[missing_pair_mask].numel() == n_missing_pair, \
            f"expected: {n_missing_pair}, got : {missing_pair_mask[missing_pair_mask].numel()}"
        # tensors for full features (esm+missing)
        _node_feats = torch.zeros(b, n, missing_node_feats.shape[-1], device=device)
        _pair_feats = torch.zeros(b, n, n, missing_pair_feats.shape[-1], device=device)
        # fill node features
        _node_feats[:, missing_node_mask] = missing_node_feats
        _pair_feats[:, missing_pair_mask] = missing_pair_feats
        if exists(esm_node):
            assert exists(esm_pair)
            _node_feats[:, ~missing_node_mask] = esm_node
            _pair_feats[:, ~missing_pair_mask] = rearrange(esm_pair, "b n m d -> b (n m) d")
        return _node_feats, _pair_feats

    def get_esm_feat_embeddings(self, esm_output: Dict, device: Any) -> Tuple[Tensor, Tensor]:
        """Embed esm output features"""
        return self.esm_node_norm(esm_output["node"].to(device)), \
               self.esm_pair_norm(esm_output["pair"].to(device))

    @staticmethod
    def _get_missing_masks(n: int, missing_indices: Tensor) -> Tuple[Tensor, Tensor]:
        device = missing_indices.device
        # missing node mask
        missing_node_mask = torch.zeros(n, device=device)
        missing_node_mask[missing_indices] = 1
        missing_node_mask = missing_node_mask.bool()
        # missing pair mask
        missing_pair_mask = torch.zeros(n, n, device=device)
        missing_pair_mask[missing_indices, :] = missing_pair_mask[:, missing_indices] = 1
        missing_pair_mask = missing_pair_mask.bool()

        return missing_node_mask, missing_pair_mask

    @staticmethod
    def _get_esm_input_embs(use_esm_1b: bool, node_out=256, pair_out=128):
        # ESM pair embedding
        pair_repr_dim = ESM_1B_PAIR_REPR_DIM if use_esm_1b else ESM_MSA_PAIR_REPR_DIM
        esm_pair_emb = nn.Linear(pair_repr_dim, pair_out)

        # ESM node embedding
        esm_node_emb = NodeUpdateBlock(
            node_dim=256,
            pair_dim=None,
            ff_mult=4,
            use_ipa=False,
            dropout=0,
            dim_out=node_out,
            dim_head=32,
            heads=8,
        )

        return esm_node_emb, esm_pair_emb
