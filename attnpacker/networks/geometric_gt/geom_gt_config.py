"""Config class for Geometric GT"""
from argparse import Namespace
from typing import Union, List, NamedTuple, Dict, Any


class GeomGTConfig(NamedTuple):
    """Graph Transformer config"""
    node_dim: int
    pair_dim: int
    depth: int
    use_ipa: bool
    node_update_kwargs: Dict[str, Any]
    pair_update_kwargs: Dict[str, Any]
    share_weights: bool
    scale_rigid_update: bool


def get_configs(
        node_dim: int,
        pair_dim: int,
        opts: Namespace,
) -> Union[GeomGTConfig, List[GeomGTConfig]]:
    depth = opts.depth
    share_weights = list(map(bool, opts.share_weights))
    use_ipa = list(map(bool, opts.use_ipa))

    node_kwargs = lambda i: dict(
        dropout=opts.node_dropout[i],
        heads=opts.node_heads[i],
        dim_query=opts.node_dim_query[i],
        dim_head=opts.node_dim_query[i],
        dim_value=opts.node_dim_value[i],
        dim_query_point=opts.node_dim_query_point[i],
        dim_value_point=opts.node_dim_value_point[i],
        use_dist_attn=bool(opts.use_dist_attn[i])
    )

    pair_kwargs = lambda i: dict(
        dropout=opts.pair_dropout[i],
        heads=opts.pair_heads[i],
        dim_head=opts.pair_dim_head[i],
        do_tri_mul=bool(opts.do_tri_mul[i]),
        do_tri_attn=bool(opts.do_tri_attn[i]),
        do_pair_outer=bool(opts.do_pair_outer[i])
    )

    gtc = lambda i: GeomGTConfig(
        node_dim=node_dim,
        pair_dim=pair_dim,
        use_ipa=use_ipa[i],
        depth=depth[i],
        node_update_kwargs=node_kwargs(i),
        pair_update_kwargs=pair_kwargs(i),
        share_weights=share_weights[i],
        scale_rigid_update=opts.scale_rigid_update
    )

    if len(depth) > 1:
        return [gtc(i) for i in range(len(depth))]
    return gtc(0)


def add_gt_options(parser):
    """Add feature options to parser"""
    gt_options = parser.add_argument_group("gt_args")
    gt_options.add_argument("--scale_rigid_update", action="store_true")
    gt_options.add_argument("--do_pair_outer", nargs="+", default=[1] * 3, type=int)
    # encoder/ decoder node kwargs
    gt_options.add_argument("--node_dropout", type=float, default=[0, 0], nargs="+")
    gt_options.add_argument("--pair_dropout", type=float, default=[0, 0], nargs="+")
    kwargs = "use_ipa depth share_weights use_dist_attn".split()
    node_kwargs = "heads dim_query dim_value dim_query_point dim_value_point".split()
    node_kwargs = ["node_" + x for x in node_kwargs]
    kwargs += node_kwargs
    pair_kwargs = "pair_heads pair_dim_head do_tri_mul do_tri_attn".split()
    kwargs += pair_kwargs
    for opt in kwargs:
        gt_options.add_argument(f"--{opt}", type=int, nargs="+", default=None)

    return parser, gt_options
