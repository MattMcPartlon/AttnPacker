import os
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from typing import Dict, Optional, NamedTuple, List

import numpy as np

from common.constants import NAME, PATH, EXT, START_TIME
from common.constants import STATS, MODELS, PARAMS, LOGS, CHECKPOINTS
from data.data_io import load_npy
from helpers.utils import default
from scoring.loss.atom_loss import ATOM_LOSS_TY_DESCS
from scoring.loss.coord_loss import CoordLossTys, COORD_LOSS_TY_DESCS

"""
Arg Categories

- data I/O: 
    paths and such

- Run Settings: 
    how/ when to load save model, how often to test/validate, #gpu's, max sequence length, etc

-Training Settings
    learning rate, batch size, warmup, epochs, lr-scheduler, gradient clipping

-Loss Settings
    loss types for coord/atom, loss relative weights, etc.

- Model Settings
    model type (SE3, En, EGNN, AF2), depth, neighbors, lots more.

- Atom Settings (for model)
    dimensions of atom features in model (in, hidden, out, attn heads, etc)

- Coord Settings
    dimensions of coord features, attn dims, etc.

- Edge Settings
    whether to use edge features, and hidden dim.

- SE3 Transformer Settings 
    If using SE3 Model, extra options availabe

- Feature Generation Settings
    use rosetta score terms, include dssp

-Embedding and Encoding options
    options for feature generation from model pdbs
    embedding dimensions and flag for different features

- Experimental options
    untested options to experiment with.
    (should probably toggle "raise_exceptions" flag if using any of these options")


"""


def get_config(arg_list):
    """Command line parser for global model configuration"""
    parser = ArgumentParser(description="Global Configuration Settings",  # noqa
                            epilog='Global configuration settings for protein learning model',
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('model_name',
                        help='name to use for model output. output will be saved as '
                             'os.path.join(out_root,<ty>,name_{date_time}.<suffix>)')

    parser.add_argument('train_list',
                        help='list of samples to train on. Must be formatted as : [dataset] [target] [model_pdb] '
                             'where \ndataset: is the folder containing the target models \ntarget: is the name'
                             ' of the target, and directory containing pdb files\n model_pdb: is the pdb for '
                             'the model.\n All paths are taken relative to the data_root argument')

    parser.add_argument('out_root',
                        help="root directory to store output in used for storing model statistics,"
                             " saving the model, and checkpointing the model.",
                        type=str,
                        )

    # I/O #
    data_group = parser.add_argument_group("data (I/O)")

    data_group.add_argument('--native_folder',
                            help='Name of folder containing native pdb files.',
                            type=str,
                            default=None)

    data_group.add_argument('--decoy_folder',
                            help='Name of folder containing decoy pdb files',
                            type=str,
                            default=None)

    data_group.add_argument('--valid_list',
                            help='path to validation list - should use same format as train_list argument',
                            default=None,
                            type=str)

    data_group.add_argument('--test_list',
                            help='path to test list - should use same format as train_list argument',
                            default=None,
                            type=str)

    # Run Options #

    run_settings = parser.add_argument_group('Run Settings')

    run_settings.add_argument('--data_workers',
                              help="number of workers to use in loading data",
                              type=int,
                              default=4)

    run_settings.add_argument('-g', '--gpu_indices',
                              help="gpu index/indices to run on",
                              type=str,
                              nargs='+',
                              default=["0"]
                              )

    run_settings.add_argument('--save_every',
                              help="frequency (num batches) at which model state is saved",
                              type=int,
                              default=50)

    run_settings.add_argument('--checkpoint_every',
                              help="frequency (num batches) at which model state is saved. "
                                   "use -1 to disable",
                              type=int,
                              default=2000)

    run_settings.add_argument('--validate_every',
                              help="frequency (num batches) at which to run model on validation set",
                              type=int,
                              default=250)

    run_settings.add_argument('--test_every',
                              help="frequency (num batches) at which to run model on test set",
                              type=int,
                              default=250)

    run_settings.add_argument('--max_seq_len',
                              help='maximum (sequence) length to run through the model'
                                   'proteins with more residues than this value will have their'
                                   'sequence and coordinates randomly cropped.',
                              default=300,
                              type=int)

    # Training Settings #

    training_settings = parser.add_argument_group("Training Settings")
    training_settings.add_argument('-b', '--batch_size',
                                   help="batch size",
                                   type=int,
                                   default=64)

    training_settings.add_argument('-e', '--epochs',
                                   help="max number of epochs to run for",
                                   type=int,
                                   default=10)

    training_settings.add_argument('--learning_rate',
                                   help="learning rate to use during training",
                                   type=float,
                                   default=1e-4)

    training_settings.add_argument('--decrease_lr_every',
                                   help="decrease learning rate by 1/2 every decrease_lr_every many epochs."
                                        "If this parameter is not positive, no update will be performed.",
                                   default=0, type=int)

    training_settings.add_argument('--weight_decay',
                                   help="weight decay parameter for l2 regularization",
                                   type=float,
                                   default=0)

    training_settings.add_argument('--grad_norm_clip',
                                   type=float,
                                   default=1.,
                                   help="scale gradients so that cumulative norm is at most this value")

    training_settings.set_defaults(clip_grad_per_sample=False)
    training_settings.add_argument('--clip_grad_per_sample',
                                   action='store_true',
                                   help="clip gradients in each sample (rather than entire batch)")

    training_settings.set_defaults(use_re_zero=True)
    training_settings.add_argument('--no_use_re_zero',
                                   action='store_false',
                                   dest='use_re_zero',
                                   help="do not use reZero residual scheme")

    # Loss Settings #

    loss_settings = parser.add_argument_group("Loss Settings")
    # coordinate loss types
    coord_loss_desc = "\n".join([x + " : " + y for (x, y) in COORD_LOSS_TY_DESCS])
    loss_settings.add_argument('--coord_loss',
                               nargs='+',
                               type=str,
                               default=[CoordLossTys.COORD_L1],
                               help=f"list of coordinate loss types. Options include \n{coord_loss_desc}"
                               )

    loss_settings.add_argument('--coord_loss_weights',
                               nargs='+',
                               type=float,
                               default=[1.],
                               help="weight of each coordinate auxillary loss type"
                               )

    # atom loss types

    atom_loss_desc = "\n".join([x + " : " + y for (x, y) in ATOM_LOSS_TY_DESCS])
    loss_settings.add_argument('--residue_loss',
                               nargs='+',
                               type=str,
                               default=[],
                               help=f"list of residue loss types. Options include \n{atom_loss_desc}"
                               )

    loss_settings.add_argument('--residue_loss_weights',
                               nargs='+',
                               type=float,
                               default=[],
                               help="weight of each coordinate loss type"
                               )

    #pair loss
    loss_settings.add_argument('--pair_loss',
                               nargs='+',
                               type=str,
                               default=[],
                               help=f"list of pair loss types. Options include \n{atom_loss_desc}"
                               )

    loss_settings.add_argument('--pair_loss_weights',
                               nargs='+',
                               type=float,
                               default=[],
                               help="weight of each coordinate loss type"
                               )

    # Generic options applicable to all refinement models #
    loss_settings.set_defaults(predict_chi_angles=False)
    loss_settings.add_argument("--predict_chi_angles", action="store_true")
    loss_settings.add_argument("--angle_norm_weight", default=0.01, type=float)
    loss_settings.add_argument("--pred_chi_weight", default=1, type=float)
    loss_settings.add_argument("--num_residual_chi_blocks", default=2, type=int)

    model_settings = parser.add_argument_group("Model Settings")
    model_settings.add_argument('--model',
                                help='model to use for coord refinemet (SE3 or En)',
                                default='SE3',
                                type=str)

    model_settings.set_defaults(predict_sidechains=False)
    model_settings.add_argument("--predict_sidechains",
                                action="store_true",
                                )
    model_settings.add_argument("--sc_atom_mode",
                                type=str,
                                default="all")

    model_settings.add_argument('--load_state',
                                help='load from previous state',
                                action='store_true',
                                dest='load_state')

    model_settings.add_argument('--load_checkpoint',
                                help='load model checkpoint',
                                action='store_true',
                                dest='load_checkpoint')

    model_settings.add_argument('--checkpoint_idx',
                                help='index of model checkpoint to load (default most recent)',
                                default=-1,
                                type=int,
                                )

    model_settings.add_argument('--atom_tys',
                                type=str,
                                help='atom types to use for model refinement (any subset of CA, N, CB, C)',
                                nargs='+',
                                default=["N", "CA", "CB", "C"],
                                )
    model_settings.add_argument('--predict_atom_tys',
                                type=str,
                                help='atom types to use for model refinement (any subset of CA, N, CB, C)',
                                nargs='+',
                                default=None,
                                )
    model_settings.add_argument('--num_replicas', type=int, default=1,
                                help="number of refinement model replicas to use - input passed through all "
                                     "replicas iteratively")

    model_settings.add_argument('-d', '--depth',
                                help="transformer depth",
                                type=int,
                                default=3)

    model_settings.add_argument('--num_neighbors',
                                help="number of sparse neighbors to attend to",
                                type=int,
                                default=8)

    model_settings.add_argument('--valid_radius',
                                help="radius (in angstroms) s.t. two atoms within the given"
                                     " radius are considered neighbors",
                                type=int,
                                default=15)

    model_settings.add_argument('--rmsd_cutoff',
                                help="cutoff (in angstroms) for maximum allowed rmsd between model and native. "
                                     "Training and validation samples with initial RMSD > cutoff will be ignored",
                                type=int,
                                default=10)

    model_settings.add_argument('--tm_cutoff',
                                help="cutoff (in angstroms) for minimum allowed tm between model and native. "
                                     "Training and validation samples with initial tm < cutoff will be ignored",
                                type=float,
                                default=0.0)

    model_settings.add_argument('--refine_iters', default=1, type=int,
                                help="max number of iterations to (recursively) run model for")

    model_settings.add_argument('--structure_iters', default=1, type=int,
                                help="max number of iterations to (recursively) run structure module for")

    model_settings.set_defaults(rotary_position=False)
    model_settings.add_argument('--rotary_position',
                                action='store_true',
                                help="apply rotary (fourier) positional encodings in attention")

    model_settings.set_defaults(rotary_rel_dist=False)
    model_settings.add_argument('--rotary_rel_dist',
                                action='store_true',
                                help="apply rotary (fourier) encodings in attention")

    model_settings.set_defaults(ty0_bias=False)
    model_settings.add_argument('--ty0_bias', action='store_true',
                                help="whether to add bias term to scalar feature linear projections")

    model_settings.set_defaults(ty0_layernorm=False)
    model_settings.add_argument('--ty0_layernorm', action='store_true',
                                help="whether to use layernorm or SE3Norm (default) for scalar features")

    model_settings.set_defaults(append_rel_dist=False)
    model_settings.add_argument('--append_rel_dist',
                                action='store_true',
                                help="append relative distance to attention linear out projection")

    model_settings.set_defaults(append_edge_attn=False)
    model_settings.add_argument('--append_edge_attn',
                                action='store_true',
                                help="append attention weighted mean of edge features to attention output projection")

    model_settings.set_defaults(append_norm=False)
    model_settings.add_argument('--append_norm',
                                action='store_true',
                                help="append attention weighted mean of edge features to attention output projection")

    model_settings.set_defaults(pair_bias=False)
    model_settings.add_argument("--pair_bias",
                                action="store_true",
                                dest="pair_bias")
    model_settings.add_argument('--init_eps', type=float, default=None,
                                help="initialization scale to use for all linear projections."
                                     "if none given, 1/sqrt(dim_in) is used.")

    model_settings.set_defaults(recycle_structure_state=False)
    model_settings.add_argument('--recycle_structure_state',
                                dest="recycle_structure_state",
                                action='store_true',
                                help="recycle structure state at each iteration")

    model_settings.set_defaults(recycle_pair_state=True)
    model_settings.add_argument('--no_recycle_pair_state',
                                dest="recycle_pair_state",
                                action='store_false',
                                help="recycle pair state at each iteration")

    model_settings.set_defaults(recycle_atom_state=True)
    model_settings.add_argument('--no_recycle_atom_state',
                                action='store_false',
                                dest='recycle_atom_state',
                                help="recycle atom state at each iteration")

    # Parameters for Transformer atom (ty0) features #

    atom_feat_settings = parser.add_argument_group('Scalar (atom) settings')

    atom_feat_settings.add_argument('--atom_dim_in',
                                    type=int,
                                    default=128,
                                    help="dimension of atom feature input to transformer model")

    atom_feat_settings.add_argument('--atom_heads',
                                    help="number of transformer heads to use for scalar atom features",
                                    type=int,
                                    default=6)

    atom_feat_settings.add_argument('--dim_atom_head',
                                    help="dimension to use for atom feature transformer heads",
                                    type=int,
                                    default=48)

    atom_feat_settings.add_argument('--atom_dim_hidden',
                                    help="hidden dimension to use for (scalar) atom features",
                                    type=int,
                                    default=196)

    atom_feat_settings.add_argument('--atom_dim_out',
                                    help="output dimension to use for  atom features (input"
                                         "dimension used if not specified)",
                                    type=int,
                                    default=None)

    # Parameters for Transformer coordinate (ty1) features #

    coord_feat_settings = parser.add_argument_group('Coordinate settings')

    coord_feat_settings.add_argument('-n', '--coord_heads',
                                     help="number of transformer heads to use for coordinate features"
                                          "defaults to atom_heads",
                                     type=int,
                                     default=None)

    coord_feat_settings.add_argument('-l', '--dim_coord_head',
                                     help="dimension to use for atom feature transformer heads"
                                          "defaults to dim_atom_head",
                                     type=int,
                                     default=None)

    coord_feat_settings.add_argument('-i', '--coord_dim_hidden',
                                     help="hidden dimension to use for coord features"
                                          "defaults to atom_dim_hidden",
                                     type=int,
                                     default=None)
    coord_feat_settings.add_argument('--coord_dim_out',
                                     help=" dimension to use for coord output features"
                                          "defaults to 4",
                                     type=int,
                                     default=4)

    coord_feat_settings.set_defaults(use_rel_coord_input=False)
    coord_feat_settings.add_argument('--use_rel_coord_input',
                                     action='store_true',
                                     help="whether to use CA-X relative position as"
                                          " input to se3 transformer")

    coord_feat_settings.add_argument('--rel_coord_dropout',
                                     help="dropout rate for relative coordinate input",
                                     default=0, type=float)

    coord_feat_settings.set_defaults(use_coord_attn=True)
    coord_feat_settings.add_argument('--no_use_coord_attn',
                                     action='store_false',
                                     dest='use_coord_attn',
                                     help='if set to false, only ty0 '
                                          'attention will be used to update '
                                          'coordinate features.'
                                     )
    # Parameters for Transformer coordinate edge features #

    edge_feat_settings = parser.add_argument_group('Edge Settings')
    edge_feat_settings.set_defaults(use_edge_feats=True)
    edge_feat_settings.add_argument('--no_use_edge_feats', action='store_false', dest='use_edge_feats')
    edge_feat_settings.add_argument('--edge_hidden', type=int, default=128)

    # Parameters for global attention
    global_attn_settings = parser.add_argument_group('Global Attention Settings')
    global_attn_settings.set_defaults(use_global_attn=False)
    global_attn_settings.add_argument('--use_global_attn',
                                      action='store_true',
                                      help="whether or not to use global attention")
    global_attn_settings.add_argument('--global_feat_dim',
                                      type=int,
                                      default=None,
                                      help="dimension of global features for global attention")

    global_attn_settings.set_defaults(use_super_node=False)
    global_attn_settings.add_argument("--use_super_node",
                                      action="store_true", )
    # SE3 Transformer model related parameters #

    se3_settings = parser.add_argument_group('SE3 Transformer Settings')

    se3_settings.add_argument('--chunks',
                              help="number of chunks to split into for atention and convolution operations"
                                   "used to reduce gpu memory when training model - chunks = 3 will reduce"
                                   "memory up to 30 percent",
                              type=int,
                              default=3
                              )
    se3_settings.add_argument('--conv_in_layers',
                              help="number of (additional) pre-convolutional layers to use in SE3 model input",
                              type=int,
                              default=0)

    se3_settings.add_argument('--conv_out_layers',
                              help="number of (additional) post-convolutional layers to use in SE3 model output",
                              type=int,
                              default=0)
    se3_settings.set_defaults(differentiable_coords=False)
    se3_settings.add_argument('--differentiable_coords', action='store_true',
                              help="whether or not coordinate basis should be treated as differentiable -"
                                   "If using coord_l1_pairwise_dev for coord. loss, probly a good idea. ")

    se3_settings.set_defaults(reversible=False)
    se3_settings.add_argument('--reversible',
                              help="use reversible networks (helps with memory) (default = False)",
                              action='store_true',
                              dest='reversible')

    se3_settings.set_defaults(linear_proj_keys=False)
    se3_settings.add_argument('--linear_proj_keys',
                              action='store_true',
                              help='whether to use a linear projection for attention keys (saves a lot on space)')

    se3_settings.set_defaults(radial_embed_rel_dist=False)
    se3_settings.add_argument('--radial_embed_rel_dist',
                              action='store_true',
                              help='whether to embed relative distances as input to ConvSE3 radial kernel function')

    se3_settings.set_defaults(one_hot_rel_dist=False)
    se3_settings.add_argument('--one_hot_rel_dist',
                              action='store_true',
                              help='whether to one-hot encode relative distances as input to ConvSE3 '
                                   'radial kernel function')

    se3_settings.set_defaults(use_tfn_q=False)
    se3_settings.add_argument('--use_tfn_q',
                              action='store_true',
                              help="use a tensor field network to produce queries (rather than linear proj.)")

    se3_settings.set_defaults(share_attn_weights=False)
    se3_settings.add_argument('--share_attn_weights',
                              action='store_true',
                              help="if specified, attention weights for each feature degree"
                                   "are summed - i.e. attention weights will be the same for"
                                   "each degree.")

    se3_settings.set_defaults(learn_head_weights=False)
    se3_settings.add_argument('--learn_head_weights',
                              action='store_true',
                              help="learn degree-specific weights for attention heads")

    # Feature generation options - all features are generated from pdb files
    feature_gen = parser.add_argument_group("feature generation settings")
    feature_gen.set_defaults(one_hot_features=False)
    feature_gen.add_argument("--one_hot_features",
                             action="store_true",
                             dest="one_hot_features"
                             )
    feature_gen.set_defaults(include_dssp=True)
    feature_gen.add_argument('--no_include_dssp',
                             help="whether or not to include dssp features (sec. structure, relative SASA)",
                             dest='include_dssp',
                             action='store_false')

    feature_gen.set_defaults(use_rosetta_score_terms=False)
    feature_gen.add_argument('--use_rosetta_score_terms', action='store_true',
                             help="whether to use rosetta to generate centroid scores as atom/edge features")

    feature_gen.set_defaults(pre_embed_rosetta_score_terms=True)
    feature_gen.add_argument('--no_pre_embed_rosetta_score_terms', action='store_false',
                             help="rosetta score terms may not be well distributed, in an attempt to fix this,"
                                  "there is an option to pre-embed (linear project) these terms before"
                                  "concatenating to other features",
                             dest='pre_embed_rosetta_score_terms')

    # Residue Feature Embedding Options

    embedding_opts = parser.add_argument_group("Embedding Settings")
    res_embedding_opts = embedding_opts.add_argument_group("Residue Embedding Options")

    # residue embedding
    res_embedding_opts.set_defaults(embed_res_ty=False)
    res_embedding_opts.add_argument('--embed_res_ty', action='store_true', help="")
    res_embedding_opts.add_argument('--res_ty_embed_dim', type=int, default=32, help="")

    # embed node centrality
    res_embedding_opts.set_defaults(embed_centrality=False)
    res_embedding_opts.add_argument("--embed_centrality", action="store_true")
    res_embedding_opts.add_argument("--centrality_embed_dim", type=int, default=6)

    # embed rel solv acc.
    res_embedding_opts.set_defaults(embed_rel_solv_acc=False)
    res_embedding_opts.add_argument('--embed_rel_solv_acc', action='store_true', help="")
    res_embedding_opts.add_argument('--rel_solv_acc_embed_bins', type=int, default=10, help="")
    res_embedding_opts.add_argument('--rel_solv_acc_embed_dim', type=int, default=8, help="")

    # dihedral
    res_embedding_opts.set_defaults(embed_dihedral=True)
    res_embedding_opts.add_argument('--no_embed_dihedral', action='store_false',
                                    dest='embed_dihedral',
                                    help="don't embed phi and psi dihedral angles")
    res_embedding_opts.add_argument('--dihedral_embed_bins', type=int, default=48, help="")
    res_embedding_opts.add_argument('--dihedral_embed_dim', type=int, default=6, help="")

    res_embedding_opts.set_defaults(fourier_encode_dihedral=False)
    res_embedding_opts.add_argument('--fourier_encode_dihedral', action='store_true',
                                    dest='fourier_encode_dihedral',
                                    help=" fourier encode phi and psi dihedral angles")
    res_embedding_opts.add_argument('--dihedral_fourier_feats', type=int, default=1, help="")

    # relative sequence position
    res_embedding_opts.set_defaults(embed_rel_pos=True)
    res_embedding_opts.add_argument('--no_embed_rel_pos',
                                    action='store_false',
                                    dest='embed_rel_pos',
                                    help='whether to embed relative seq. position')
    res_embedding_opts.add_argument('--rel_pos_embed_bins', type=int, default=10,
                                    help="number of bins to use for rel pos embedding")
    res_embedding_opts.add_argument('--rel_pos_embed_dim', type=int, default=5,
                                    help="dimension of relative position embedding")

    # secondary structure
    res_embedding_opts.set_defaults(embed_sec_struct=False)
    res_embedding_opts.add_argument('--embed_sec_struct',
                                    action='store_true',
                                    help='whether to embed secondary structure (include_dssp must be enabled)')
    res_embedding_opts.add_argument('--sec_struct_embed_dim', type=int, default=10,
                                    help="dimension of secondary structure embedding")

    res_embedding_opts.set_defaults(embed_cen_dist=False)
    res_embedding_opts.add_argument('--embed_cen_dist', action='store_true',
                                    help="embeds (discretized) distance from atom to protein coordinate"
                                         "centroid. provides information about atom dist. from protein core.")
    res_embedding_opts.add_argument('--cen_dist_embed_dim', type=int, default=5)

    res_embedding_opts.set_defaults(embed_ca_dihedral=False)
    res_embedding_opts.add_argument('--embed_ca_dihedral',
                                    action='store_true',
                                    help="embed dihedral angles between each 4 consecutive CA atoms"
                                    )
    res_embedding_opts.add_argument('--ca_dihedral_embed_dim',
                                    help="dimenion of ca dihedral embedding",
                                    default=6,
                                    type=int)

    # Residue Pair Feature Options #

    res_pair_opts = parser.add_argument_group("Residue Pair Embedding + Encoding Options")

    res_pair_opts.set_defaults(embed_res_pairs=False)
    res_pair_opts.add_argument('--embed_res_pairs',
                               action='store_true',
                               help="use residue pair embedding in edge features")
    res_pair_opts.add_argument('--res_pair_embed_dim',
                               type=int,
                               default=32,
                               help="embedding dimension for residue pair edge features")

    res_pair_opts.set_defaults(joint_embed_res_pair_n_rel_sep=False)
    res_pair_opts.add_argument('--joint_embed_res_pair_n_rel_sep',
                               action='store_true',
                               help='embed relative separation and residue pair jointly'
                                    'dimension defaults to max of res_pair_embed_dim and'
                                    'rel_sep_embed_dim.')

    res_pair_opts.add_argument('--max_embed_dist', type=float, default=16,
                               help="maximum value of relative distance to embed")
    res_pair_opts.add_argument('--min_embed_dist', type=float, default=2,
                               help="minimum value of relative distance to embed")
    res_pair_opts.add_argument('--num_dist_bins', type=int, default=48,
                               help='number of bins to discretize distances to')
    res_pair_opts.add_argument('--rel_dist_embed_dim', type=int, default=32,
                               help='dimension of relative distance embedding')
    res_pair_opts.add_argument("--rel_dist_atom_tys",
                               help="atom types to embed relative distances for",
                               nargs="+",
                               default=["CA", "CA"])

    res_pair_opts.set_defaults(fourier_encode_rel_dist=False)
    res_pair_opts.add_argument('--fourier_encode_rel_dist', action='store_true',
                               help='whether to apply fourier encoding to relative inter atom distances')
    res_pair_opts.add_argument('--num_fourier_rel_dist_feats', type=int, default=4,
                               help='number of fourier features in encoding (each feature has size 2)')

    res_pair_opts.set_defaults(fourier_encode_rel_sep=False)
    res_pair_opts.add_argument("--fourier_encode_rel_sep", action='store_true', dest='fourier_encode_rel_sep',
                               help="whether to include fourier encodings for relative seq. separation")
    res_pair_opts.add_argument("--num_fourier_rel_sep_feats", type=int, default=4, help="")

    res_pair_opts.set_defaults(embed_rel_sep=True)
    res_pair_opts.add_argument('--no_embed_rel_sep', action='store_false', dest='embed_rel_sep',
                               help="whether to embed relative separation")
    res_pair_opts.add_argument('--rel_sep_embed_dim', type=int, default=8,
                               help="relative separation embed dim")
    res_pair_opts.set_defaults(use_small_sep_bins=False)
    res_pair_opts.add_argument("--use_small_sep_bins", action="store_true", dest="use_small_sep_bins")

    # Tr-Rosetta orientation features
    res_pair_opts.set_defaults(use_tr_rosetta_ori_features=True)
    res_pair_opts.add_argument('--no_use_tr_rosetta_ori_features',
                               help="disable tr_rosetta orientation features",
                               dest='use_tr_rosetta_ori_features',
                               action='store_false')
    res_pair_opts.add_argument('--ori_embed_bins', type=int, default=48,
                               help="number of bins to discretize orientation features into")
    res_pair_opts.add_argument('--ori_embed_dim', type=int, default=6,
                               help="dimension of Tr-Rosetta orientation feature embedding")

    # Experimental Options (not thoroughly tested) #

    experimental_settings = parser.add_argument_group('Experimental Options')

    experimental_settings.add_argument('--attn_ty', default="se3", type=str)
    experimental_settings.add_argument('--use_dist_conv', action="store_true", default=False)
    experimental_settings.add_argument('--pairwise_dist_conv', action="store_true", default=False)
    experimental_settings.add_argument('--num_dist_conv_filters', default=32, type=int)
    experimental_settings.add_argument('--structure_dropout', default=0.15, type=float)
    experimental_settings.add_argument('--radial_dropout', default=0.0, type=float)
    experimental_settings.add_argument('--radial_mult', default=1, type=float)
    experimental_settings.set_defaults(radial_compress=False)
    experimental_settings.add_argument('--radial_compress', action="store_true")

    experimental_settings.set_defaults(use_pre_transformer=False)
    experimental_settings.add_argument('--use_pre_transformer',
                                       help="feature extraction on atom features with vanilla transformer",
                                       action='store_true',
                                       dest='use_pre_transformer')
    experimental_settings.add_argument("--pre_transformer_node_dropout", default=0.15, type=float)
    experimental_settings.add_argument("--pre_transformer_edge_dropout", default=0.25, type=float)

    experimental_settings.add_argument('--pre_transformer_gpu_index',
                                       type=int,
                                       default=None,
                                       help="alternate gpu index for pre - transformer")

    experimental_settings.set_defaults(pre_transformer_edge_consensus=False)
    experimental_settings.add_argument("--pre_transformer_edge_consensus",
                                       action="store_true")

    experimental_settings.set_defaults(use_nbr_attn=False)
    experimental_settings.add_argument('--use_nbr_attn',
                                       action="store_true")

    experimental_settings.set_defaults(mask_unsampled_edges=False)
    experimental_settings.add_argument('--mask_unsampled_edges',
                                       action="store_true")

    experimental_settings.set_defaults(symmetrize_edges=False)
    experimental_settings.add_argument('--symmetrize_edges',
                                       action="store_true")
    experimental_settings.add_argument('--pre_transformer_max_radius',
                                       default=24,
                                       type=float)
    experimental_settings.add_argument('--pre_transformer_max_nbrs',
                                       default=50,
                                       type=int)
    experimental_settings.add_argument('--pre_transformer_edge_attn_heads',
                                       default=4,
                                       type=int)
    experimental_settings.add_argument('--pre_transformer_edge_attn_dim',
                                       default=32,
                                       type=int)
    experimental_settings.set_defaults(use_pre_transformer_update_bias=False)
    experimental_settings.add_argument('--use_pre_transformer_update_bias',
                                       action="store_true")

    experimental_settings.add_argument('--pre_transformer_heads',
                                       type=int,
                                       default=8, help="")

    experimental_settings.add_argument('--pre_transformer_dim_head', type=int, default=32,
                                       help="head dimension")

    experimental_settings.add_argument('--pre_transformer_dim_hidden', type=int, default=256,
                                       help="hidden dimension")

    experimental_settings.set_defaults(pre_transformer_edge_attn=True)
    experimental_settings.add_argument('--no_use_pre_transformer_edge_attn',
                                       action='store_false',
                                       dest='pre_transformer_edge_attn'
                                       )

    experimental_settings.add_argument('--pre_transformer_depth', type=int, default=4)

    experimental_settings.set_defaults(coord_norm_seq=False)
    experimental_settings.add_argument('--coord_norm_seq', action='store_true',
                                       help="whether to normalize coordinate features over sequence dimension"
                                            "in this case, the mean norm over all coord features is 1 (rather than)"
                                            "mean 1 norm per-residue")

    experimental_settings.set_defaults(weighted_out=False)
    experimental_settings.add_argument('--weighted_out',
                                       action='store_true',
                                       dest="weighted_out",
                                       help="use atom features to predict the norm of model coordinates")

    experimental_settings.set_defaults(raise_exceptions=False)
    experimental_settings.add_argument('--raise_exceptions',
                                       action='store_true',
                                       dest='raise_exceptions',
                                       help="whether or not to raise exceptions thrown during training (bad input or"
                                            " bad data will crash code if set to true")

    experimental_settings.set_defaults(random_sample_tri_attn=False)
    experimental_settings.add_argument('--random_sample_tri_attn',
                                       action='store_true',
                                       dest="random_sample_tri_attn")
    experimental_settings.set_defaults(reuse_tri_edges=False)
    experimental_settings.add_argument('--reuse_tri_edges',
                                       action='store_true',
                                       dest="reuse_tri_edges")
    experimental_settings.add_argument('--tri_attn_n_points_per_sample',
                                       type=int,
                                       default=8)
    experimental_settings.add_argument('--tri_attn_n_nbrs_per_sample',
                                       type=int,
                                       default=16
                                       )
    experimental_settings.set_defaults(edge_attn_residual=True)
    experimental_settings.add_argument('--no_edge_attn_residual',
                                       action='store_false',
                                       dest='edge_attn_residual',
                                       )
    experimental_settings.set_defaults(checkpoint_tri_attn=True)
    experimental_settings.add_argument('--no_checkpoint_tri_attn',
                                       action='store_false',
                                       dest='checkpoint_tri_attn',
                                       )

    experimental_settings.set_defaults(checkpoint_tfn=True)
    experimental_settings.add_argument('--no_checkpoint_tfn',
                                       action='store_false',
                                       dest='checkpoint_tfn',
                                       )
    experimental_settings.set_defaults(locality_aware=True)
    experimental_settings.add_argument('--no_locality_aware',
                                       action='store_false',
                                       dest='locality_aware',
                                       )

    args = parser.parse_args(arg_list)
    # set some input-dependent defaults
    args.atom_dim_out = default(args.atom_dim_out, args.atom_dim_in)
    args.coord_dim_hidden = default(args.coord_dim_hidden, args.atom_dim_hidden)
    args.coord_heads = default(args.coord_heads, args.atom_heads)
    args.dim_coord_head = default(args.dim_coord_head, args.dim_atom_head)
    args.pre_transformer_gpu_index = default(args.pre_transformer_gpu_index, args.gpu_indices[0])
    args.predict_atom_tys = default(args.predict_atom_tys, args.atom_tys)
    if not args.load_state:
        args.name = args.name + f"_{START_TIME}"
    return args


best_defaults = dict(
    clip_grad_per_sample=True,
    learn_head_weights=False,
    share_attn_weights=False,
)

MAX_VALIDATION_SAMPLES = 200
MAX_TEST_SAMPLES = 200


class GlobalConfig(NamedTuple):
    """Data for global model configuration.
    For description of parameters see `get_args(...)` above
    """
    """Mandatory Params"""
    model_name: str
    train_list: str
    out_root: str
    """Data relevant params"""
    native_folder: Optional[str]
    decoy_folder: Optional[str]
    valid_list: Optional[str]
    test_list: Optional[str]
    """Run Settings"""
    data_workers: int
    gpu_indices: int
    validate_every: int
    checkpoint_every: int
    save_every: int
    validate_every: int
    test_every: int
    max_seq_len: int
    """Training Settings"""
    batch_size: int
    epochs: int
    decrease_lr_every: int
    learning_rate: float
    weight_decay: float
    grad_norm_clip: float
    clip_grad_per_sample: bool
    use_rezero: bool
    """Loss"""
    coord_loss: List[str]
    coord_loss_weights: List[float]
    residue_loss: List[str]
    residue_loss_weights: List[float]
    pair_loss: List[str]
    pair_loss_weights: List[str]


    def get_dir(self, ty):
        info = self.out_dirs[ty]
        return os.path.join(info[PATH], info[NAME] + info[EXT])

    @property
    def use_atom_loss(self):
        return len(self.atom_loss_weights) > 0

    @property
    def default_atom_ty(self):
        atom_tys = self.atom_tys
        if not self.predict_sidechains:
            if "CA" in atom_tys:
                return 'CA'
            elif "CB" in atom_tys:
                return "CB"
            else:
                return atom_tys[0]
        else:
            return "CG" if "CG" in self.atom_tys else self.atom_tys[0]


def parse_arg_file(path):
    if path is None or not os.path.exists(path):
        return
    args = []
    with open(path, 'r') as f:
        for x in f:
            line = x.strip()
            if len(line.strip()) > 0 and not line.startswith("#"):
                arg = line.split(' ')
                for a in arg:
                    args.append(a)
    return args


def make_args(arg_file=None) -> RefineArgs:
    args = parse_arg_file(arg_file)
    args = args if args is not None else sys.argv[1:]
    parsed_args = get_args(args)
    # transfer into refiners args object
    return RefineArgs(out_dirs=add_dirs(parsed_args), **vars(parsed_args))


def add_dirs(args) -> Dict[str, Dict[str, str]]:
    base = args.out_root
    os.makedirs(base, exist_ok=True)
    tys = [STATS, MODELS, CHECKPOINTS, PARAMS, LOGS]
    exts = ['.npy', '.tar', '.tar', '.npy', '.log']
    out_dirs = {k: {PATH: None, NAME: args.name, EXT: ext} for k, ext in zip(tys, exts)}
    for ty in tys:
        out_dirs[ty][PATH] = os.path.join(base, ty)
        os.makedirs(out_dirs[ty][PATH], exist_ok=True)
    return out_dirs


def get_members(r):
    return {attr: getattr(r, attr) for attr in dir(r) if
            not callable(getattr(r, attr)) and not attr.startswith("__")}


def print_args(args: RefineArgs):
    mems = args._asdict()
    for k, v in mems.items():
        if k == 'note':
            print(f"{k} {' '.join(v)}")
        else:
            print(f"{k} : {v}")


def load_refine_args(arg_path, curr_args: Optional[RefineArgs], **override) -> RefineArgs:
    load_kwargs = load_npy(arg_path)
    if curr_args is not None:
        curr_args = curr_args._asdict()
        curr_args.update(load_kwargs)
    else:
        curr_args = load_kwargs
    curr_args['load_state'] = True
    curr_args.update(override)
    return RefineArgs(**curr_args)


def save_refine_args(args: RefineArgs):
    np.save(args.get_dir(PARAMS), args._asdict())
