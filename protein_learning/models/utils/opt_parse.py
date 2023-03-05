from argparse import Namespace

from protein_learning.common.helpers import default, exists

namespace_2_dict = lambda x: vars(x) if isinstance(x, Namespace) else x

def add_esm_options(parser):
    esm_args = parser.add_argument_group("esm_args")
    esm_args.add_argument("--use_esm_prob", type=float, default=1)
    esm_args.add_argument("--use_esm_both_chain_prob", type=float, default=1)
    esm_args.add_argument("--use_esm_1b", action="store_true")
    esm_args.add_argument("--use_esm_msa", action="store_true")
    esm_args.add_argument("--esm_gpu_idx", type=int, default=None)
    esm_args.add_argument("--esm_msa_dir", default=None)
    esm_args.add_argument("--max_msa_seqs", type=int, default=128)
    return parser, esm_args

def add_default_loss_options(parser):
    # Loss options
    loss_options = parser.add_argument_group("loss_args")

    # PER-LOSS WEIGHTS
    loss_tys = "fape plddt nsr tm dist_inv pair_dist com pae violation sc_rmsd res_fape".split(" ")
    for loss_ty in loss_tys:
        loss_options.add_argument(f"--{loss_ty}_wt", help=f"weight for {loss_ty} loss", default=None, type=float)
    loss_options.add_argument(
        f"--inter_fape_scale", help=f"weight to scale intra fape loss by", default=None, type=float
    )

    # PER-LOSS ATOM TYPES
    loss_tys = "fape plddt tm dist_inv pair_dist pae".split(" ")
    for loss_ty in loss_tys:
        loss_options.add_argument(
            f"--{loss_ty}_atom_tys", help=f"atom types for {loss_ty} loss", default=None, nargs="+"
        )

    # PER-LOSS DISCRETIZATION
    loss_options.add_argument("--plddt_bins", default=10, type=int)
    loss_options.add_argument("--pair_dist_step", default=0.4, type=float)
    loss_options.add_argument("--pair_dist_max_dist", default=20, type=float)
    loss_options.add_argument("--pae_max_dist", default=16, type=float)
    loss_options.add_argument("--pae_step", default=0.5, type=float)
    loss_options.add_argument("--include_viol_after", type=int, default=-1)
    loss_options.add_argument("--vdw_wt", type=float, default=1)
    loss_options.add_argument("--bond_len_wt", type=float, default=1)
    loss_options.add_argument("--bond_angle_wt", type=float, default=3)
    loss_options.add_argument("--viol_schedule", type=float, nargs="+", default=None)
    loss_options.add_argument("--mask_rel_wt", type=float, default=1)
    loss_options.add_argument("--sc_rmsd_p",type=int,default=1)
    for name in "pae plddt nsr".split():
        loss_options.add_argument(f"--include_{name}_after", default=0, type=int)

    return parser, loss_options


def add_intra_chain_mask_options(parser):
    """Add intra-chain mask options to parser"""
    # Mask Options
    mask_options = parser.add_argument_group("intra_chain_mask_args")
    mask_tys = "spatial contiguous random no full cdr".split(" ")
    mask_tys += "interface_full true_interface true_inverse_interface".split()
    mask_tys += "interface inverse_interface inverse_interface_full".split()
    for mask_ty in mask_tys:
        mask_options.add_argument(f"--{mask_ty}_mask_weight", help=f"weight for {mask_ty} mask", default=0, type=float)
    # spatial mask
    mask_options.add_argument("--spatial_mask_top_k", type=int, default=30)
    mask_options.add_argument("--spatial_mask_max_radius", type=float, default=12.0)
    # contiguous mask
    mask_options.add_argument("--contiguous_mask_min_len", type=int, default=5)
    mask_options.add_argument("--contiguous_mask_max_len", type=int, default=60)
    # random mask
    mask_options.add_argument("--random_mask_min_p", type=float, default=5)
    mask_options.add_argument("--random_mask_max_p", type=float, default=60)
    # overall mask size options (as fraction of input protein length)
    mask_options.add_argument("--max_mask_frac", type=float, default=0.3)
    mask_options.add_argument("--min_mask_frac", type=float, default=0)
    # interface masking
    mask_options.add_argument("--interface_mask_min_frac", type=float, default=0)
    mask_options.add_argument("--interface_mask_max_frac", type=float, default=0)
    mask_options.add_argument("--inverse_interface_mask_min_frac", type=float, default=0)
    mask_options.add_argument("--inverse_interface_mask_max_frac", type=float, default=0)

    return parser, mask_options


def add_inter_chain_mask_options(parser):
    """Add inter-chain mask options to parser"""
    # Mask Options
    mask_options = parser.add_argument_group("inter_chain_mask_args")
    mask_tys = "random one_to_all full no".split(" ")
    for mask_ty in mask_tys:
        mask_options.add_argument(
            f"--inter_{mask_ty}_mask_weight", help=f"weight for {mask_ty} mask", default=0, type=float
        )
    return parser, mask_options


def add_chain_partition_options(parser):
    """Add chain partition options to parser"""
    partition_options = parser.add_argument_group("chain_partition_args")
    partition_tys = "linear bilinear identity".split(" ")
    for partition_ty in partition_tys:
        partition_options.add_argument(
            f"--{partition_ty}_partition_weight",
            help=f"weight for {partition_ty} chain partition",
            default=0,
            type=float,
        )
    partition_options.add_argument("--linear_n_classes", nargs="+", default=[2, 2], type=int)
    partition_options.add_argument("--partition_min_frac", type=float, default=0)
    partition_options.add_argument("--partition_min_len", type=int, default=10)
    partition_options.add_argument("--bilinear_max_len", type=int, default=50)

    return parser, partition_options


def add_feature_options(parser):
    """Add feature options to parser"""
    feature_options = parser.add_argument_group("feature_args")
    feature_options.add_argument("--embed_sec_struct", action="store_true")
    feature_options.add_argument("--sec_struct_embed_dim", type=int, default=0)
    feature_options.add_argument("--joint_embed_res_pair_rel_sep_embed_dim", default=48, type=int)

    ty_opts = (
        "quat_encode_rel_ori encode_local_rel_coords "
        "joint_embed_res_pair_rel_sep rbf_encode_rel_distance "
        "rbf_encode_centrality one_hot_rel_chain fourier_encode_bb_dihedral include_sc_dihedral"
        " fourier_encode_tr_rosetta_ori".split(" ")
    )
    for opt in ty_opts:
        feature_options.add_argument(f"--{opt}", action="store_true")

    # types that can be embedded or one-hot encoded
    tys = "res_rel_pos res_ty bb_dihedral rel_sep centrality rel_dist tr_rosetta_ori".split(" ")
    for ty in tys:
        feature_options.add_argument(f"--embed_{ty}", action="store_true")
        feature_options.add_argument(f"--one_hot_{ty}", action="store_true")
        feature_options.add_argument(f"--{ty}_embed_dim", type=int, default=-1)
        if ty not in "res_ty rel_sep".split():
            feature_options.add_argument(f"--{ty}_encode_dim", type=int, default=-1)

    feature_options.add_argument("--rel_dist_bounds", nargs="+", type=float, default=[2.5, 16.5])
    feature_options.add_argument("--rel_dist_atom_tys", nargs="+", default="CA CA N CA".split())
    feature_options.add_argument("--res_ty_corrupt_prob", default=0, type=float)
    feature_options.add_argument("--sc_dihedral_noise", default=[0,0], type=float, nargs="+")
    # feature_options.add_argument("--coord_noise", default=0, type=float)
    return parser, feature_options


def add_transformer_options(parser, name=None, suffix=None, include_pair: bool = True, include_coord: bool = True):
    name = default(name, "transformer_args")
    suffix = f"{suffix}_" if exists(suffix) else ""
    transformer_options = parser.add_argument_group(name)

    transformer_options.add_argument(f"--{suffix}scalar_heads", default=8, type=int)
    transformer_options.add_argument(f"--{suffix}scalar_head_dim", default=16, type=int)
    if include_pair:
        transformer_options.add_argument(f"--{suffix}pair_heads", default=4, type=int)
        transformer_options.add_argument(f"--{suffix}pair_head_dim", default=20, type=int)
    if include_coord:
        transformer_options.add_argument(f"--{suffix}coord_head_dim", default=16, type=int)
    return parser, transformer_options


def add_feat_gen_options(parser):
    """Masked feature generation arguments"""
    feat_gen_args = parser.add_argument_group("feat_gen_args")
    feat_gen_args.add_argument("--mask_feats", action="store_true")
    feat_gen_args.add_argument("--mask_seq", action="store_true")
    feat_gen_args.add_argument("--mask_feat_n_seq_indep_prob", default=0, type=float)
    return parser, feat_gen_args


def add_egnn_options(parser):
    group = parser.add_argument_group("egnn_args")
    # group.add_argument(name='--depth', type=int, default=3)
    # group.add_argument(name='--node_dim', type=int, default=120)
    # group.add_argument(name='--pair_dim', type=int, default=120)
    # group.add_argument(name='--coord_dim_in', type=int, default=4)
    group.add_argument("--coord_dim_hidden", type=int, default=None)
    group.add_argument("--coord_dim_out", type=int, default=None)
    group.add_argument("--coord_scale", type=float, default=0.1)
    group.add_argument("--m_dim", type=int, default=16)
    group.add_argument("--dropout", type=float, default=0.0)
    group.add_argument("--norm_rel_coords", action="store_true")
    group.add_argument("--use_nearest", action="store_true")
    group.add_argument("--max_radius", type=float, default=20)
    group.add_argument("--top_k", type=int, default=32)
    group.add_argument("--lin_proj_coords", action="store_true")
    group.add_argument("--use_rezero", action="store_true")
    return parser, group


def add_tfn_options(parser):
    group = parser.add_argument_group("tfn_args")
    # group.add_argument(name='--fiber_in', type=int, nargs="+", default=[100, 4])
    group.add_argument("--fiber_out", type=int, nargs="+", default=[100, 3])
    group.add_argument("--fiber_hidden", type=int, nargs="+", default=[128, 16])
    group.add_argument("--heads", type=int, nargs="+", default=[10, 10])
    group.add_argument("--dim_heads", type=int, nargs="+", default=[20, 4])
    # group.add_argument(name='--edge_dim', type=int, default=100)
    # group.add_argument(name='--depth', type=int, default=6)
    group.add_argument("--conv_in_layers", type=int, default=1)
    group.add_argument("--conv_out_layers", type=int, default=1)
    # group.add_argument(name='--linear_proj_keys', type=bool, default=False)
    group.add_argument("--hidden_mult", type=int, default=2)
    group.add_argument("--radial_mult", type=float, default=2)
    #TODO: may want to double check
    
    group.add_argument("--attn_ty", type=str, default="tfn")
    group.add_argument("--use_coord_layernorm", action="store_true")
    return parser, group


def add_stats_options(parser):
    stats_group = parser.add_argument_group("stats_args")
    tys = "nsr lddt violation rmsd contact mae distance".split()
    for ty in tys:
        stats_group.add_argument(f"--get_{ty}_stats", action="store_true", help=f"get statistics for {ty} data")
    return parser, stats_group


def add_flag_args(parser):
    """Add flag args to parser"""
    flag_args = parser.add_argument_group("flag_args")
    flag_args.add_argument("--include_complex_flags", action="store_true")
    flag_args.add_argument("--include_block_adjs", action="store_true")
    flag_args.add_argument("--include_terminal_flags", action="store_true")
    flag_args.add_argument("--include_bond_flags", action="store_true")
    flag_args.add_argument("--include_ss_one_hot", action="store_true")
    flag_args.add_argument("--include_contact_prob", default=0, type=float)
    flag_args.add_argument("--include_interface_prob", default=0, type=float)
    flag_args.add_argument("--contact_fracs", default=[0, 0], type=float, nargs="+")
    flag_args.add_argument("--interface_fracs", default=[0, 0], type=float, nargs="+")
    flag_args.add_argument("--contact_thresh", default=0, type=float)
    flag_args.add_argument("--include_both_interface_prob", default=0.5, type=float)
    flag_args.add_argument("--num_interface", type=float, default=None)
    flag_args.add_argument("--num_contact", type=float, default=None)
    flag_args.add_argument("--random_interface", type=int, default=None)
    flag_args.add_argument("--cdr_interface", action="store_true")
    flag_args.add_argument("--random_contact", type=int, default=None)

    return parser, flag_args
