from math import sqrt
import torch
import torch.nn.functional as F  # noqa
from einops import rearrange, repeat  # noqa
from torch import einsum
from torch import nn
from abc import abstractmethod

from protein_learning.networks.common.helpers.neighbor_utils import NeighborInfo
from protein_learning.networks.common.helpers.torch_utils import batched_index_select, safe_norm, safe_cat
from protein_learning.networks.common.utils import exists
from protein_learning.networks.se3_transformer.se3_attention_config import SE3AttentionConfig
from protein_learning.networks.tfn.repr.fiber import Fiber, to_order, cast_fiber
from protein_learning.networks.common.equivariant.linear import VNLinear
from protein_learning.networks.common.equivariant.fiber_units import FiberLinear
from protein_learning.networks.common.constants import DIST_SCALE
from typing import Tuple, Dict, Optional
from protein_learning.networks.common.attention_utils import (
    compute_hidden_coords,
    get_rel_dists,
    get_degree_scale_for_attn,
    get_similarity,
    get_attn_weights_from_sim,
    SimilarityType,
    AttentionType,
)


# TODO : mask features (e.g. edge attn).


class SE3Attention(nn.Module):
    def __init__(
            self,
            fiber_in: Fiber,
            config: SE3AttentionConfig,
    ):
        """SE(3)-Equivariant Attention Layer

        All subclasses must accept parameters "fiber_in" and "config" as keyword arguments.

        :param fiber_in: input fiber
        :param config: SE3Attention Config
        """
        super(SE3Attention, self).__init__()
        self.config, self.fiber_in = config, fiber_in
        self.sim_ty = SimilarityType.DISTANCE if config.use_dist_sim else SimilarityType.DOT_PROD
        self.attn_ty = AttentionType.SHARED if config.share_attn_weights else AttentionType.PER_TY
        self.shared_scale = None
        if self.attn_ty == AttentionType.SHARED:
            self.shared_scale = sqrt(1 / 3) if self.sim_ty == SimilarityType.DISTANCE else sqrt(1 / 2)

        # hidden fiber will map degree to heads*dim_head
        self.hidden_fiber = Fiber(
            {deg: config.heads[deg] * config.dim_heads[deg] for deg in fiber_in.degrees}
        )

        # compute dimension of augmented edges
        edge_hidden = config.edge_dim + (2 * fiber_in[1] if config.append_hidden_dist else 0)
        self.augmented_edge_dim = edge_hidden + (config.num_dist_conv_filters
                                                 if config.use_dist_conv else 0)

        assert set(fiber_in.degrees) == set(config.dim_heads.degrees)
        assert set(config.heads.degrees) == set(fiber_in.degrees)

        self.degree_scale = {
            deg: get_degree_scale_for_attn(
                to_order(deg), config.dim_heads[deg], sim_ty=self.sim_ty
            )
            for deg in self.hidden_fiber.degrees
        }

        if config.attend_self:
            self.to_self_kv = nn.ModuleDict({
                str(deg): VNLinear(
                    dim_in=fiber_in[deg], dim_out=2 * self.hidden_fiber[deg])
                for deg in fiber_in.degrees
            })
            if config.append_edge_attn:
                self.to_self_loop = nn.Linear(fiber_in[0], self.hidden_fiber[0], bias=False)

        if config.use_null_kv:
            self.null_keys, self.null_values = nn.ParameterDict(), nn.ParameterDict()
            for deg in fiber_in.degrees:
                h, d = config.heads[deg], config.dim_heads[deg]
                self.null_keys[str(deg)] = nn.Parameter(torch.zeros(h, d, to_order(deg)))
                self.null_values[str(deg)] = nn.Parameter(torch.zeros(h, d, to_order(deg)))
            if config.append_edge_attn:
                self.null_edge = nn.Parameter(torch.zeros(config.heads[0], config.dim_heads[0]))

        if config.learn_head_weights:
            head_weights = {str(deg): torch.log(torch.exp(torch.ones(1, dim, 1, 1)) - 1) for
                            deg, dim in config.heads.items}
            self.head_weights = nn.ParameterDict({k: nn.Parameter(v) for k, v in head_weights.items()})

        self.to_bias = None
        if config.pair_bias:
            self.to_bias = nn.Linear(config.edge_dim, config.heads[0], bias=False)
            self.self_bias = nn.Parameter(torch.zeros(config.heads[0]))
            self.null_bias = nn.Parameter(torch.zeros(config.heads[0]))

        fiber_out_dims = [self.hidden_fiber[0], self.hidden_fiber[1]]
        self.to_edge_v = None
        if config.append_edge_attn:
            fiber_out_dims[0] += self.hidden_fiber[0]
            self.to_edge_v = nn.Linear(config.edge_dim, self.hidden_fiber[0], bias=False)

        if config.append_norm:
            fiber_out_dims[0] += self.hidden_fiber[1]

        self.fiber_out = cast_fiber(dims=tuple(fiber_out_dims))
        self.to_out = FiberLinear(self.fiber_out, self.fiber_in)

    def forward(
            self,
            features: Dict[str, torch.Tensor],
            edge_info: Tuple[torch.Tensor, NeighborInfo],
            basis: Dict[str, torch.Tensor],
            global_feats: Optional[torch.Tensor] = None,  # noqa
    ):
        config, (edges, neighbor_info) = self.config, edge_info
        edge_feats, neighbor_mask, initial_coords = edges, neighbor_info.mask, neighbor_info.coords.detach()
        # reshape adj mask to account for head dimension
        edge_feats = self.augment_edges(edges=edges, features=features, neighbor_info=neighbor_info)
        if exists(neighbor_mask):
            neighbor_mask = rearrange(neighbor_mask, 'b i j -> b () i j')
            # get queries, keys, and values
        queries, keys, values = self.get_qkv(
            features=features,
            edge_info=(edge_feats, neighbor_info),
            basis=basis,
            global_feats=global_feats,
        )

        # compute bias
        bias = rearrange(self.to_bias(edges), "b n N h -> b h n N") if exists(self.to_bias) else None

        degree_sim, degree_vals = {}, {}

        for degree in features.keys():
            h = config.heads[int(degree)]
            q, k, v = map(lambda x: x[degree], (queries, keys, values))
            k, v = map(lambda x: rearrange(x, 'b i j (h d) m -> b h i j d m', h=h), (k, v))
            q = rearrange(q, 'b i (h d) m -> b h i d m', h=h)

            # augment queries keys and values
            q, k, v = self.augment_qkv(feats=features[degree], q=q, k=k, v=v, degree=int(degree))
            # save values
            degree_vals[degree] = v
            # get head weight(s)
            if config.learn_head_weights:
                weight = F.softplus(self.head_weights[str(degree)]) if degree == 1 else 1
            else:
                weight = 1

            # compute sim(q_i, k_j)
            degree_sim[degree] = get_similarity(keys=k,
                                                queries=q,
                                                sim_ty=self.sim_ty,
                                                bias=bias if to_order(int(degree)) == 1 else None,
                                                initial_coords=initial_coords,
                                                scale=self.degree_scale[int(degree)] * weight,
                                                dist_scale=DIST_SCALE,
                                                )

        # compute attention for each feature type
        attn_wts = get_attn_weights_from_sim(
            sims=degree_sim,
            neighbor_mask=neighbor_mask,
            attn_ty=self.attn_ty,
            shared_scale=self.shared_scale
        )
        outputs = {}
        for degree, val in degree_vals.items():
            outputs[degree] = einsum('b h i j, b h i j d m -> b h i d m', attn_wts[degree], val)

        if config.append_edge_attn:
            outputs['0'] = self.append_edge_attn(
                feats=features,
                edges=edges,
                output=outputs['0'],
                attn_wts=attn_wts['0']
            )

        for degree in outputs:
            outputs[degree] = rearrange(outputs[degree], 'b h n d m -> b n (h d) m')

        if config.append_norm:
            hidden_coords = compute_hidden_coords(initial_coords=initial_coords, coord_feats=outputs['1'])
            coord_norms = safe_norm(hidden_coords, dim=-1, keepdim=True) * DIST_SCALE
            outputs['0'] = torch.cat((outputs['0'], coord_norms), dim=-2)

        return self.to_out(outputs)

    @abstractmethod
    def get_qkv(
            self,
            features: Dict[str, torch.Tensor],
            edge_info: Tuple[torch.Tensor, NeighborInfo],
            basis: Dict[str, torch.Tensor],
            global_feats: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], ...]:
        """Get query, key, and value tensors for each feature type.

        In param/output descriptions,
        b: represents the batch size
        n: sequence length
        i(m) : hidden dimension of type m input features
        h: number of attention heads
        d: attention head dimension
        m: degree of the corresponding feature type (1 for scalar, 3 for point)
        N: number of neighbors

        :param features: dict mapping type (e.g. "0" or "1") to hidden features of shape (b,n,i(m),m)
        :param edge_info: tuple containing edge feature tensor and neighbor info instance
        :param basis: equivariant basis
        :param global_feats: global features
        :return: queries, keys, and values for each input feature type.

        queries: shape (b,n,h*d,m)
        keys: shape (b,n,N,h*d,m)
        values: shape (b,n,N,h*d,m)
        """
        pass

    def augment_qkv(
            self,
            feats: torch.Tensor,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            degree: int) -> Tuple[torch.Tensor, ...]:
        config, degree_key = self.config, str(degree)
        if config.attend_self:
            self_k, self_v = self.to_self_kv[degree_key](feats).chunk(2, dim=-2)
            self_k, self_v = map(
                lambda t: rearrange(t, 'b n (h d) m -> b h n () d m', h=config.heads[degree]),
                (self_k, self_v)
            )
            k = torch.cat((self_k, k), dim=3) if k is not None else k
            v = torch.cat((self_v, v), dim=3)

        if config.use_null_kv:
            null_k, null_v = map(lambda t: t[degree_key], (self.null_keys, self.null_values))
            null_k, null_v = map(lambda t: repeat(t, 'h d m -> b h i () d m', b=v.shape[0], i=v.shape[2]),
                                 (null_k, null_v))
            k = torch.cat((null_k, k), dim=3) if k is not None else k
            v = torch.cat((null_v, v), dim=3)

        return q, k, v

    def augment_edges(
            self,
            edges: torch.Tensor,
            features: Dict[str, torch.Tensor],
            neighbor_info: NeighborInfo
    ) -> torch.Tensor:

        config, edge_feats = self.config, edges
        # (Optional) Augment edge features
        if config.append_hidden_dist:
            rel_dist1 = get_rel_dists(neighbor_info.coords, coord_feats=features["1"], normalize_dists=False)
            rel_dist2 = get_rel_dists(neighbor_info.coords, coord_feats=features["1"], normalize_dists=True)
            rel_dist = torch.cat((rel_dist1 * DIST_SCALE, rel_dist2), dim=-1)
            rel_dist = batched_index_select(rel_dist, neighbor_info.indices, dim=2)
            edge_feats = safe_cat(edge_feats, rel_dist, dim=-1)

        return edge_feats

    def append_edge_attn(
            self,
            feats: Dict[str, torch.Tensor],
            edges: torch.Tensor,
            output: torch.Tensor,
            attn_wts: torch.Tensor
    ) -> torch.Tensor:
        config = self.config
        b, n = edges.shape[:2]
        edge_vals = rearrange(self.to_edge_v(edges),
                              "b i j (h d)-> b h i j d", h=config.heads[0])
        if config.attend_self:
            self_loop = rearrange(self.to_self_loop(feats['0'].squeeze(-1)),
                                  "b n (h d)-> b h n () d", h=config.heads[0])
            edge_vals = torch.cat((self_loop, edge_vals), dim=-2)
        if config.use_null_kv:
            null_edge = repeat(self.null_edge, "h d-> b h n () d", b=b, n=n)
            edge_vals = torch.cat((null_edge, edge_vals), dim=-2)

        edge_attn = einsum('b h i j e, b h i j -> b h i e',
                           edge_vals, attn_wts)
        return torch.cat((output, edge_attn.unsqueeze(-1)), dim=-2)
