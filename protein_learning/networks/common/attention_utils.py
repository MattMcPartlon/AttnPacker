import torch
from torch import Tensor, einsum
from protein_learning.networks.common.helpers.torch_utils import to_rel_pos, ndim, get_max_neg_value, safe_norm
from einops import rearrange
from math import sqrt
from protein_learning.networks.common.utils import exists
from enum import Enum
from typing import Dict, Optional

"""Helper Methods for Equivariant Attention

In all doc strings, we use
b : batch dimension
n : sequence dimension
d : hidden dimension
h : number of attention heads
N : neighbor dimension (= number of neighbors per residue)
"""


class SimilarityType(Enum):
    DISTANCE = "distance"
    DOT_PROD = "dot_prod"


class AttentionType(Enum):
    SHARED = "shared"
    PER_TY = "per_ty"


def compute_hidden_coords(initial_coords: Tensor, coord_feats: Tensor) -> Tensor:
    """
    Determines the relative positions of hidden coordinates

    :param initial_coords: shape (b,...,n,3)
    :param coord_feats: shape (b,...,n,d,3)
    :return: tensor with same shape as coord_feats
    """
    assert ndim(initial_coords) == 3
    n_empty_dims = ndim(coord_feats) - 4
    einstring = f"... n c -> ... {' '.join(['()'] * n_empty_dims)} n () c"
    return rearrange(initial_coords, einstring) + coord_feats


def compute_hidden_rel_coords(initial_coords: Tensor, coord_feats: Tensor) -> Tensor:
    """
    Determines the relative positions of hidden coordinates

    :param initial_coords: shape (b,...,n,3)
    :param coord_feats: shape (b,...,n,d,3)
    :return: tensor of shape (b,...,n,n,d,3)
    """
    assert not initial_coords.requires_grad
    hidden_coords = compute_hidden_coords(initial_coords=initial_coords, coord_feats=coord_feats)
    return to_rel_pos(hidden_coords)


def get_rel_dists(initial_coords: Tensor, coord_feats: Tensor, normalize_dists: bool) -> Tensor:
    """
    Appends the relative distance between hidden coordinates i_k and j_k to the
    edge features, where k = 1..d indexes the hidden dimension of coordinate features

    eg:
        coord_feats[...,i,:] has shape (6,3), then 6 is the hidden dimension
        we will append 6 total distances to the edge features

    :param initial_coords: shape (...,n,3) where h is hidden dimension
    :param coord_feats: shape (...,n,d,3)
    :param normalize_dists : scale the pairwise distances s.t. the l2 norm is sqrt(d)/2
    (reccomended).
    :return: Tensor of shape (...,n,n,d) where d is the coordinate feature hidden dimension
    """
    assert coord_feats.shape[-1] == 3, \
        f"coordinate features must have 3 as trailing dimension, got shape {coord_feats.shape}"

    hidden_rel_coords = compute_hidden_rel_coords(
        initial_coords=initial_coords, coord_feats=coord_feats
    )
    pairwise_dists = torch.norm(hidden_rel_coords, dim=-1)
    s = sqrt(coord_feats.shape[-2]) / 2
    scale = safe_norm(pairwise_dists, dim=-1, keepdim=True) / s if normalize_dists else None
    return pairwise_dists / scale if exists(scale) else pairwise_dists


def get_degree_scale_for_attn(feat_degree: int, dim_head: int, sim_ty: SimilarityType) -> float:
    """
    :param feat_degree: 1 for scalar features and 3 for coordinate features.
    :param dim_head: head dimension
    :param sim_ty: similarity type being computed
    :return: the scale to apply to similarity logits.
    """
    if feat_degree == 1:
        return 1 / sqrt(dim_head)
    elif feat_degree == 3:
        if sim_ty == SimilarityType.DISTANCE:
            return 2 / sqrt(9 * dim_head)
        else:
            return 1 / sqrt(3 * dim_head)
    else:
        raise Exception("Not implemented")


def get_similarity(
        keys: Tensor,
        queries: Tensor,
        sim_ty: SimilarityType,
        bias: Optional[Tensor] = None,
        initial_coords: Optional[Tensor] = None,
        scale: Optional[float] = None,
        dist_scale: Optional[float] = 0.1,

) -> Tensor:
    """Computes Multihead similarity scores (used to derive attention weights).

    For shape specs, m is feature degree (1 or 3).

    :param keys: Tensor of shape (b,h,n,N,d,m)
    :param queries: Tensor of shape (b,h,n,d,m)
    :param bias: Tensor of shape (b,h,n,N)
    :param sim_ty:
    :param initial_coords: Tensor of shape (b,n,3)
    :param scale:
    :param dist_scale: amount to scale relative distances by before computing
    attention logits.
    :return: tensor of shape (b,h,
    """
    q, k = queries, keys
    sim, feat_degree = None, keys.shape[-1]

    if feat_degree == 3 and sim_ty == SimilarityType.DISTANCE:
        if initial_coords is None:
            raise Exception("must include initial coords if using Distance Similarity!")

    if feat_degree == 1 or sim_ty == SimilarityType.DOT_PROD:
        sim = einsum('b h i d m, b h i j d m -> b h i j', q, k)

    if feat_degree == 3:
        if sim_ty == SimilarityType.DISTANCE:
            coords = rearrange(initial_coords, "b n c -> b () n () () c").detach()
            qp, kp = rearrange(q, "b h i d m -> b h i () d m") + coords, k + coords
            sim = -1 * torch.sum(torch.square((qp - kp) * dist_scale), dim=(-1, -2))

    sim = sim * scale if exists(scale) else sim
    # apply the bias
    if exists(bias):
        num_left_pad = sim.shape[-1] - bias.shape[-1]
        sim[:, :, :, num_left_pad:] = sim[:, :, :, num_left_pad:] + bias

    return sim


def get_attn_weights_from_sim(
        sims: Dict[str, Tensor],
        neighbor_mask: Optional[Tensor],
        attn_ty: AttentionType,
        shared_scale: Optional[float] = None,
) -> Dict[str, Tensor]:
    """Computes attention weights for input feature types

    :param sims: dict mapping feature dim to similarity tensor of shape (b,h,n,N)
    :param neighbor_mask: neighbor adjacency mask of shape (b,h,n,N)
    :param attn_ty: attention type to use (shared or per-type)
    :param shared_scale: optional scale to apply similarity logits before performing softmax.
    :return: dict mapping feature degree to tensor of attention weights (b,h,n,N)
    """
    sim_keys = list(sims.keys())
    if attn_ty == AttentionType.SHARED:
        sim_sum = sum([v for v in sims.values()])
        sims = {sim_keys[0]: sim_sum}

    out = {}
    for feat_dim, sim in sims.items():
        # mask attention for pairs that are not neighbors
        if exists(neighbor_mask):
            num_left_pad = sim.shape[-1] - neighbor_mask.shape[-1]
            sim[:, :, :, num_left_pad:] = sim[:, :, :, num_left_pad:].masked_fill(
                ~neighbor_mask, get_max_neg_value(sim)
            )
        sim = sim * shared_scale if exists(shared_scale) else sim
        out[feat_dim] = torch.softmax(sim, dim=-1)

    # shared setting
    if attn_ty == AttentionType.SHARED:
        return {k: out[sim_keys[0]] for k in sim_keys}
    # non-shared setting
    return out
