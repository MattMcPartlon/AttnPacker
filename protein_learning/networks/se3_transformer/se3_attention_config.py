from __future__ import annotations
from typing import Optional, NamedTuple
from protein_learning.networks.common.utils import update_named_tuple
from protein_learning.networks.tfn.repr.fiber import Fiber, cast_fiber


class SE3AttentionConfig(NamedTuple):
    """Params for SE3 attention layer

    :param heads: number of heads to use for hidden features (scalar and point) e.g.
    (10,5) means "use 10 heads for scalar feats and 5 for points" an equal
    number of heads should be used if shared attention is enabled (share_attn_weights = True)

    :param dim_heads: dimensions to sue for each head (scalar_dim, point_dim)

    :param edge_dim: dimension of edge features

    :param global_feats_dim : dimension of any global features (not yet implemented)

    :param attend_self : whether to include attention between point i and point i

    :param use_null_kv : use null keys and values (similar to applying head weights)

    :param share_attn_weights: share attention weights between scalar and point features.

    :param use_dist_sim: use distance based similarity for computing point affinities.

    :param learn_head_weights: learn separate weights for each attention head

    :param use_coord_attn: if false, attention logits for scalar vals will also be used for points.

    :param append_edge_attn: append (linearly projected) edge features to output weighted by
    scalar attention scores. edge features mapped to tensor of shape (heads[0], dim_heads[0]) if True.

    :param use_dist_conv: append the output of a distance based convolution kernel to edge features.
    This requires a fair bit of extra memory, but usually improves results.

    :param pairwise_dist_conv : learn separate convolution kernels for each pair of pair features.

    :param num_dist_conv_filters: number of kernels to learn.

    :param pair_bias: bias overall attention weights via a linear projection of edge features

    :param append_norm: append point norms to output before final projection

    :param append_hidden_dist : append distance between hidden points to edge features
    """

    heads: Fiber = cast_fiber((10, 10))
    dim_heads: Fiber = cast_fiber((20, 4))
    edge_dim: int = None
    global_feats_dim: Optional[int] = None
    attend_self: bool = True
    use_null_kv: bool = True
    share_attn_weights: bool = True
    use_dist_sim: bool = False
    learn_head_weights: bool = False
    use_coord_attn: bool = True
    append_edge_attn: bool = False
    use_dist_conv: bool = False
    pairwise_dist_conv: bool = False
    num_dist_conv_filters: int = 16
    pair_bias: bool = True
    append_norm: bool = False
    append_hidden_dist: bool = False

    def override(self, **override) -> SE3AttentionConfig:
        return update_named_tuple(SE3AttentionConfig, self, **override)
