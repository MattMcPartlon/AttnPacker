"""Denovo Complex Design"""
import os

# make sure cuda devices are listed according to PCI_BUS_ID before any torch modules are loaded

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'

from protein_learning.models.complex_design.eval.complex_design_stats import ComplexDesignStats
import torch
from protein_learning.models.complex_design.masked_design_model import ComplexDesignConfig, ComplexDesigner
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from protein_learning.models.complex_design.complex_design_utils import (
    augment,
    ExtraComplexDesign,
    add_augment_args,
)
from protein_learning.features.input_embedding import InputEmbedding
from protein_learning.common.global_constants import get_logger
from protein_learning.models.utils.model_io import (
    load_n_save_args,
    print_args,
    get_args_n_groups,
)
from protein_learning.models.utils.opt_parse import add_intra_chain_mask_options, add_inter_chain_mask_options, \
    add_chain_partition_options, add_feat_gen_options
from protein_learning.networks.loss.complex_reconstruction_loss import ComplexReconstructionLoss
from protein_learning.common.data.datasets.protein_dataset import ProteinDataset
from functools import partial
from protein_learning.models.complex_design.train_masked_model import (
    info,
    __setup__,
    DEFAULT_ATOM_TYS,
    get_parser as _get_parser_main,
    get_feature_gen,
    get_input_feature_config,
    get_val_feat_gen,
    transform as cb_transform,
)

torch.set_printoptions(precision=3)
logger = get_logger(__name__)
CONFIG_DIR = "/mnt/local/mmcpartlon/design/models/params"


def get_global_n_local_configs(model_name, suffix="masked_design"):
    """Get path to global and local config"""
    global_config = f"{model_name}.npy"
    local_config = f"{model_name}_{suffix}.npy"
    return map(lambda x: os.path.join(CONFIG_DIR, x),
               (global_config, local_config))


def get_args():
    """Get arguments for complex design"""
    parser = ArgumentParser(description="Evaluate Complex Design Model",  # noqa
                            epilog='',
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('global_config_path',
                        help="path to global config args")
    parser.add_argument("stats_path", type=str, default=None)
    parser.add_argument("--pdb_dir", type=str, default=None)
    parser.add_argument("--raise_exceptions", action="store_true")
    parser.add_argument("--n_decoys", type=int, default=100)
    global_override = parser.add_argument_group("global_override")
    global_override.add_argument('--gpu_indices', nargs="+", default=None)
    global_override.add_argument('--max_len', type=int, default=400)
    global_override.add_argument('--train_root', type=str, default=None)
    global_override.add_argument('--train_list', type=str, default=None)
    global_override.add_argument('--train_seq_folder', type=str, default=None)
    global_override.add_argument('--train_native_folder', type=str, default=None)

    # augmentation args
    parser, augment_args = add_augment_args(parser)  # augment_args
    # add masking options
    parser, intra_chain_mask_group = add_intra_chain_mask_options(parser)  # intra_chain_mask_args
    parser, inter_chain_mask_group = add_inter_chain_mask_options(parser)  # inter_chain_mask_args
    # add chain partition options
    parser, partition_group = add_chain_partition_options(parser)  # chain_partition_args
    parser, feat_gen_args = add_feat_gen_options(parser)  # feat_gen_args

    return get_args_n_groups(parser)


def get_defaults(global_config_path):
    parser = _get_parser_main()
    return get_args_n_groups(parser, arg_list=[global_config_path])


if __name__ == "__main__":
    __setup__()
    info("RUNNING COMPLEX DESIGN EVALUATION")

    # Set up model args
    eval_args, eval_groups = get_args()
    assert eval_args.stats_path.endswith(".npy")
    main_args, main_groups = get_defaults(eval_args.global_config_path)
    global_override, model_override = vars(eval_groups["global_override"]), dict()
    config, main_args, main_arg_groups = load_n_save_args(
        main_args,
        main_groups,
        main_args,
        suffix="masked_design",
        global_override=global_override,
        model_override=model_override,
        model_config_path=eval_args.global_config_path,
        save=False,
        force_load=True,
    )
    info("Loaded config and model args")
    print_args(config=config, args=main_args, arg_groups=main_arg_groups)
    info("Eval args")
    print_args(config=None, args=None, arg_groups=eval_groups)

    # copy over the arguments
    main_arg_groups["intra_chain_mask_args"] = eval_groups["intra_chain_mask_args"]
    main_arg_groups["inter_chain_mask_args"] = eval_groups["inter_chain_mask_args"]
    main_arg_groups["chain_partition_args"] = eval_groups["chain_partition_args"]
    main_arg_groups["augment_args"] = eval_groups["augment_args"]
    main_arg_groups["feat_gen_args"] = eval_groups["feat_gen_args"]
    arg_groups = main_arg_groups

    # set up feature generator
    feature_config = get_input_feature_config(arg_groups)
    feat_gen = get_feature_gen(main_args, arg_groups, feature_config)
    val_feat_gen = get_val_feat_gen(arg_groups, feature_config)

    augment_fn = partial(
        augment,
        kls=partial(ExtraComplexDesign, **vars(arg_groups["augment_args"]))
    )

    # set up datasets
    train_data, val_data, test_data = get_datasets(
        dataset=ProteinDataset,
        global_config=config,
        augment_fn=augment_fn,
        atom_tys=DEFAULT_ATOM_TYS,
        feature_gen=[feat_gen, val_feat_gen, val_feat_gen],
        filter_fn=lambda x, y: len(x) > main_args.min_len_cutoff,
        transform_fn=cb_transform,
        train_on_bounded_complex=main_args.train_on_bounded_complex
    )

    # set up model
    model = ComplexDesigner(
        input_embedding=InputEmbedding(feature_config).to(config.device),
        global_config=config,
        precompute_rigids=main_args.precompute_rigids,
        loss_fn=ComplexReconstructionLoss(
            pair_dim=main_args.pair_dim_hidden,
            scalar_dim=main_args.scalar_dim_hidden,
            atom_tys=DEFAULT_ATOM_TYS,
            **vars(arg_groups["loss_args"])
        ),
        model_config=ComplexDesignConfig(
            coord_dim_out=len(DEFAULT_ATOM_TYS),
            **vars(arg_groups["complex_model_args"])
        ),
        n_cycles=main_args.n_cycles,
        pair_noise_dim=main_args.pair_noise_dim,
        scalar_noise_dim=main_args.scalar_noise_dim,
        use_cycles=main_args.n_cycles
    )

    stats = ComplexDesignStats(
        config=config,
        model=model,
        max_samples=eval_args.n_decoys,
        pdb_dir=eval_args.pdb_dir,
        raise_exceptions=eval_args.raise_exceptions
    )

    stats.evaluate_dataset(train_data)
    os.makedirs(os.path.dirname(eval_args.stats_path), exist_ok=True)
    stats.save(eval_args.stats_path)
