import os

# make sure cuda devices are listed according to PCI_BUS_ID beofre any torch modules are loaded
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'

import gc
import torch
from protein_learning.common.global_config import (
    GlobalConfig,
    make_config,
    load_config,
    print_config,
    save_config,
    parse_arg_file,
    load_npy,
)
import numpy as np

from protein_learning.models.sc_packing.sc_model import SCPacker
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from protein_learning.common.helpers import exists
from protein_learning.features.feature_config import InputFeatureConfig
from protein_learning.features.input_embedding import InputEmbedding
from protein_learning.training.trainer import Trainer
from protein_learning.models.sc_packing.data_utils import augment, SCFeatureGenerator
from protein_learning.common.data.datasets.chain_dataset import ChainDataset
from typing import Optional, Tuple
import sys
from protein_learning.common.global_constants import get_logger

logger = get_logger(__name__)


def get_args():
    print("getting args")
    arg_file = sys.argv[1] if len(sys.argv) == 2 else None
    logger.info(f"Parsing arguments from {arg_file}")
    arg_list = parse_arg_file(arg_file) if exists(arg_file) else sys.argv[1:]
    parser = ArgumentParser(description="SC-Packing",  # noqa
                            epilog='',
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('model_config',
                        help="path to global config args")

    parser.add_argument('--scalar_dim_hidden',
                        help="scalar hidden dimension",
                        default=128, type=int
                        )

    parser.add_argument('--pair_dim_hidden',
                        help="pair hidden dimension",
                        default=128, type=int
                        )

    parser.add_argument('--coord_dim_hidden',
                        help="coordinate hidden dimension",
                        default=16, type=int
                        )

    parser.add_argument('--evoformer_scalar_heads_n_dim',
                        help="number of heads and head dimension for evoformer"
                             " scalar features",
                        nargs="+", default=[10, 16], type=int
                        )

    parser.add_argument('--evoformer_pair_heads_n_dim',
                        help="number of heads for evoformer scalar features",
                        nargs="+", default=[4, 32], type=int
                        )

    parser.add_argument('--tfn_heads',
                        help="number of heads for TFN-Transformer",
                        default=10, type=int
                        )

    parser.add_argument('--tfn_head_dims',
                        help="number of heads for IPA (scalar, coord)",
                        default=(16, 4), type=int, nargs="+"
                        )

    parser.add_argument('--tfn_depth',
                        help="number of heads for IPA (scalar, coord)",
                        default=3, type=int
                        )

    parser.add_argument('--evoformer_depth',
                        help="number of heads for IPA (scalar, coord)",
                        default=3, type=int
                        )
    parser.set_defaults(use_dist_sim=False)
    parser.add_argument("--use_dist_sim", action="store_true", help="use dist. similarity TFN Transformer")
    parser.set_defaults(use_coord_layernorm=False)
    parser.add_argument("--use_coord_layernorm", action="store_true", help="use layernorm for TFN Transformer")
    parser.set_defaults(append_rel_dist=False)
    parser.add_argument("--append_rel_dist", action="store_true", help="append rel. dist in TFN Transformer")

    return parser.parse_args(arg_list)


def load_sc_args(config: GlobalConfig):  # noqa
    path = os.path.join(config.param_directory, config.name + "_scp.npy")
    return load_npy(path)


def save_sc_args(config: GlobalConfig, args):  # noqa
    path = os.path.join(config.param_directory, config.name + "_scp.npy")
    np.save(path, args)


def print_sc_args(args):  # noqa
    for k, v in args.items():
        print(f"{k} : {v}")


def get_datasets(c: GlobalConfig) -> Tuple[ChainDataset, Optional[ChainDataset], Optional[ChainDataset]]:
    dataset = lambda lst, nat, decoy, seq, samples=-1: ChainDataset(
        model_list=lst,
        native_folder=nat,
        decoy_folder=decoy,
        seq_folder=seq,
        max_samples=samples,
        raise_exceptions=c.raise_exceptions,
        feat_gen=feat_gen,
        augment_fn=augment
    ) if exists(lst) else None

    train_data = dataset(c.train_list, c.train_native_folder,
                         c.train_native_folder, c.train_seq_folder)
    valid_data = dataset(c.val_list, c.val_native_folder, c.val_native_folder,
                         c.val_seq_folder, c.max_val_samples)
    test_data = dataset(c.test_list, c.test_native_folder, c.test_native_folder,
                        c.test_seq_folder, c.max_test_samples, )

    return train_data, valid_data, test_data


if __name__ == "__main__":
    print("RUNNING SC-PACK")
    # sometimes needed -- unclear why ?
    gc.collect()
    torch.cuda.empty_cache()

    model_args = vars(get_args())
    config = make_config(model_args["model_config"])
    model_args.update({"model_config": config})
    if config.load_state:
        print("[INFO] Loading Previous Model...")
        override = dict()
        config = load_config(config, **override)
        vae_args = load_sc_args(config)

    print("[INFO] Global Model Config : ")
    print_config(config)
    save_config(config)
    print("[INFO] VAE specific args : ")
    print_sc_args(model_args)
    save_sc_args(config, model_args)

    # set up feature generator
    feature_config = InputFeatureConfig()
    feat_gen = SCFeatureGenerator(
        config=feature_config,  # use defaults
    )

    model = SCPacker(
        input_embedding=InputEmbedding(feature_config).to(config.device),
        **model_args,
    )

    train, val, test = get_datasets(c=config)

    trainer = Trainer(
        config=config,
        model=model,
        train_data=train,
        valid_data=val,
        test_data=test,
    )
    detect_anomoly = False
    with torch.autograd.set_detect_anomaly(detect_anomoly):
        trainer.train()
