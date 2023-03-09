"""Denovo Complex Design"""
import os

# make sure cuda devices are listed according to PCI_BUS_ID before any torch modules are loaded
from typing import Union, Optional, Callable, Tuple, List

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
import sys
from abc import abstractmethod
import gc
import torch
from protein_learning.features.input_embedding import InputEmbedding
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from protein_learning.features.feature_config import InputFeatureConfig
from protein_learning.features.feature_generator import FeatureGenerator

# from protein_learning.training.trainer import Trainer
from protein_learning.common.global_constants import get_logger, _set_jit_fusion_options  # noqa
from protein_learning.models.utils.model_io import (
    load_n_save_args,
    print_args,
    get_args_n_groups,
    load_args_for_eval,
)
from protein_learning.models.utils.opt_parse import (
    add_default_loss_options,
    add_inter_chain_mask_options,
    add_intra_chain_mask_options,
    add_chain_partition_options,
    add_feature_options,
    add_feat_gen_options,
    add_stats_options,
)

from protein_learning.features.default_feature_generator import DefaultFeatureGenerator
from protein_learning.common.data.datasets.protein_dataset import ProteinDataset
from protein_learning.common.global_config import GlobalConfig
from protein_learning.common.helpers import exists, default
from protein_learning.models.model_abc.protein_model import ProteinModel

# from protein_learning.assessment.model_eval.model_evaluator import ModelEvaluator, StatsConfig

# from protein_learning.assessment.model_eval.genetic_alg.ga_evaluator import GAEvaluator
# from protein_learning.assessment.model_eval.genetic_alg.utils import add_ga_args
import numpy as np

# _set_jit_fusion_options()
torch.set_printoptions(precision=3)
logger = get_logger(__name__)
DEFAULT_ATOM_TYS = "N CA C CB".split()


class Trainer:
    def __init__(*args, **kwargs):
        pass
        # trainer class removed, only inference supported


def info(msg):  # noqa
    print(f"[INFO] : {msg}")


def __setup__():  # noqa
    info(f"PID : {os.getpid()}")
    info(f"GID : {os.getgid()}")
    gc.collect()
    torch.cuda.empty_cache()


def get_datasets(
    dataset,
    global_config: GlobalConfig,
    augment_fn: Union[Optional[Callable], List[Optional[Callable]]],
    feature_gen: Union[FeatureGenerator, List[FeatureGenerator]],
    **kwargs,
) -> Tuple[ProteinDataset, Optional[ProteinDataset], Optional[ProteinDataset]]:
    """Load train/valid/test datasets"""
    c = global_config
    feature_gens = feature_gen if isinstance(feature_gen, list) else [feature_gen] * 3
    augment_fns = augment_fn if isinstance(augment_fn, list) else [augment_fn] * 3
    _dataset = (
        lambda lst, nat, decoy, seq, gen, samples, aug_fn: dataset(
            model_list=lst,
            native_folder=nat,
            decoy_folder=decoy,
            seq_folder=seq,
            raise_exceptions=c.raise_exceptions,
            feat_gen=gen,
            augment_fn=aug_fn,
            crop_len=c.max_len,
            **kwargs,
        )
        if exists(lst)
        else None
    )

    train_data = _dataset(
        c.train_list,
        c.train_native_folder,
        c.train_decoy_folder,
        c.train_seq_folder,
        feature_gens[0],
        -1,
        augment_fns[0],
    )
    valid_data = _dataset(
        c.val_list,
        c.val_native_folder,
        c.val_decoy_folder,
        c.val_seq_folder,
        feature_gens[1],
        c.max_val_samples,
        augment_fns[1],
    )
    test_data = _dataset(
        c.test_list,
        c.test_native_folder,
        c.test_decoy_folder,
        c.test_seq_folder,
        feature_gens[2],
        c.max_test_samples,
        augment_fns[2],
    )

    return train_data, valid_data, test_data


def get_feature_gen(arg_groups, feature_config, apply_masks) -> DefaultFeatureGenerator:
    """Get Masked Feature Generator"""
    return DefaultFeatureGenerator(
        config=feature_config,
        intra_chain_mask_kwargs=vars(arg_groups["intra_chain_mask_args"]),
        inter_chain_mask_kwargs=vars(arg_groups["inter_chain_mask_args"]),
        **vars(arg_groups["feat_gen_args"]),
        apply_masks=apply_masks,
    )


def get_input_feature_config(
    arg_groups,
    pad_embeddings: bool = False,
    extra_pair_feat_dim: int = 0,
    extra_res_feat_dim: int = 0,
) -> InputFeatureConfig:
    """Get input feature configuration"""
    return InputFeatureConfig(
        extra_residue_dim=extra_res_feat_dim,
        extra_pair_dim=extra_pair_feat_dim,
        pad_embeddings=pad_embeddings,
        **vars(arg_groups["feature_args"]),
    )


def add_feat_gen_groups(parser):
    # masked feature generation
    parser, feat_gen_args = add_feat_gen_options(parser)  # feat_gen_args
    # add masking options
    parser, intra_chain_mask_group = add_intra_chain_mask_options(parser)  # intra_chain_mask_args
    parser, inter_chain_mask_group = add_inter_chain_mask_options(parser)  # inter_chain_mask_args
    # add chain partition options
    parser, partition_group = add_chain_partition_options(parser)  # chain_partition_args
    return parser


def get_default_parser():
    """Get arguments for protein learning model"""
    parser = ArgumentParser(
        description="Train options for Protein Learning Model",  # noqa
        epilog="",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    # Complex-Design-Specific arguments
    parser.add_argument("model_config", help="path to global config args")
    parser.add_argument("--node_dim_hidden", type=int, default=128)
    parser.add_argument("--pair_dim_hidden", type=int, default=128)
    parser, feature_group = add_feature_options(parser)  # feature_args
    # add (default) loss options for each loss type
    parser, loss_group = add_default_loss_options(parser)  # loss_args
    return add_feat_gen_groups(parser)


def get_default_parser_for_eval():
    """Get arguments for complex design"""
    parser = ArgumentParser(description="", epilog="", formatter_class=ArgumentDefaultsHelpFormatter)  # noqa

    # Complex-Design-Specific arguments
    parser.add_argument("--raise_exceptions", action="store_true")
    parser.add_argument("--model_config_path", type=str)
    parser.add_argument("--global_config_path", type=str)
    parser.add_argument("--stats_dir", help="directory to store model stats in", default=None)
    parser.add_argument("--pdb_dir", help="directory to store pdbs in (optional)", default=None)
    parser.add_argument("--gpu_indices", help="gpu index to run on", type=str, nargs="+", default=["0"])
    parser.add_argument("--eval_decoy_folder")
    parser.add_argument("--eval_native_folder")
    parser.add_argument("--eval_target_list")
    parser.add_argument("--eval_seq_folder")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--n_replicas", type=int, default=1)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--max_len", type=int, default=500)
    # parser.add_argument("--do_ga", action="store_true")
    # add_ga_args(parser)
    parser, _ = add_stats_options(parser)
    add_feat_gen_groups(parser)
    return parser


def default_global_override_for_eval(args):
    return dict(
        train_decoy_folder=args.eval_decoy_folder,
        train_native_folder=args.eval_native_folder,
        train_list=args.eval_target_list,
        train_seq_folder=args.eval_seq_folder,
        raise_exceptions=args.raise_exceptions,
        gpu_indices=args.gpu_indices,
        max_len=args.max_len,
    )


class TrainABC:
    """Train a model"""

    def __init__(self, skip_init=False):
        self.do_eval = True
        """
        self.do_eval=True
        eval_parser = get_default_parser_for_eval()
        self.add_extra_cmd_line_options_for_eval(eval_parser)
        self.eval_args, self.eval_groups = get_args_n_groups(eval_parser)
        train_parser = self.add_extra_cmd_line_options(get_default_parser())
        (train_args, train_groups) = get_args_n_groups(train_parser, ["none"])
        global_override = default_global_override_for_eval(self.eval_args)
        global_override.update(self.global_override_eval)


        print(f"global override\n {global_override}")
        print(f"model override\n {self.model_override_eval}")

        self.config, self.args, self.arg_groups = load_args_for_eval(
            global_config_path=self.eval_args.global_config_path,
            model_config_path=self.eval_args.model_config_path,
            model_override=self.model_override_eval,
            global_override=global_override,
            default_model_args=train_args,
            default_model_arg_groups=train_groups,
        )
        self.pad_embeddings = self.args.mask_seq or self.args.mask_feats
        self._setup()
        """
        pass

    def _setup(self):
        pass

    @property
    def load_native_ss(self):
        return False

    @property
    def input_atom_tys(self) -> List:
        """List of atom types to load for input"""
        return ["N", "CA", "C", "CB"]

    @property
    def output_atom_tys(self) -> List:
        """List of atom types to predict coords for"""
        return ["N", "CA", "C", "CB"]

    @property
    def apply_masks(self):
        """Whether to apply masks to input"""
        return self.pad_embeddings

    @property
    def model_name(self):
        """Name of the model"""
        return "protein_model"

    @property
    def extra_res_feat_dim(self):
        """Extra feature dimension for residue features"""
        return 0

    @property
    def extra_pair_feat_dim(self):
        """dimension for extra pair features"""
        return 0

    @property
    def global_override(self):
        """kwargs to override in global config for training"""
        return dict()

    @property
    def model_override(self):
        """Kwargs to override in model config for training"""
        return dict()

    @property
    def global_override_eval(self):
        """kwargs to override in global config for eval"""
        return dict()

    @property
    def model_override_eval(self):
        """Kwargs to override in model config for eval"""
        return dict()

    def add_extra_cmd_line_options(self, parser):
        """Add extra command line options to parse for during training"""
        return parser

    def add_extra_cmd_line_options_for_eval(self, parser):
        """Add extra command line options to parse for during evaluation"""
        return parser

    def input_augment_fn(self):
        """Augment modelinput object with extra data"""
        return lambda *args, **kwargs: None

    @property
    def dataset_transform_fn(self):
        """transform fn to pass to dataset"""
        return None

    @property
    def dataset_filter_fn(self):
        """filter fn to pass to dataset"""
        return None

    def get_dataset(self):
        """get dataset"""
        return None

    def get_post_model_init_fn(self):
        """fn to apply to model after initialization
        (e.g. nn.init.kaiming_uniform_(...))
        """
        return None

    @abstractmethod
    def get_model(self, input_emb: InputEmbedding) -> ProteinModel:
        """Get the model to use"""
        pass

    @property
    def allow_missing_nn_modules(self):
        return False

    def get_val_feat_gen(self, arg_groups, feature_config, apply_masks):
        return None

    def get_val_aug_fn(self):
        return None

    def get_trainer(self):
        __setup__()
        config, args, arg_groups = self.config, self.args, self.arg_groups
        print("########### Training Args ##############")
        print_args(config, args, arg_groups)
        if self.do_eval:
            print("########### Eval Args ##############")
            print_args(config, self.eval_args, self.eval_groups)

        # set up feature generator
        feature_config = get_input_feature_config(
            arg_groups,
            pad_embeddings=self.pad_embeddings,
            extra_pair_feat_dim=self.extra_pair_feat_dim,
            extra_res_feat_dim=self.extra_res_feat_dim,
        )
        self.feature_config = feature_config  # noqa

        feat_gen = get_feature_gen(arg_groups, feature_config, apply_masks=self.apply_masks)
        self.feat_gen = feat_gen  # noqa
        fgs = [feat_gen] * 3
        val_feat_gen = self.get_val_feat_gen(arg_groups, feature_config, apply_masks=self.apply_masks)
        if exists(val_feat_gen):
            fgs = [feat_gen, val_feat_gen, val_feat_gen]

        aug_fns = [self.input_augment_fn] * 3
        val_aug_fn = self.get_val_aug_fn()
        if exists(val_aug_fn):
            aug_fns[1], aug_fns[2] = val_aug_fn, val_aug_fn

        # set up datasets
        train_data, val_data, test_data = get_datasets(
            dataset=default(self.get_dataset(), ProteinDataset),
            global_config=config,
            augment_fn=aug_fns,
            atom_tys=self.input_atom_tys,
            feature_gen=fgs,
            filter_fn=self.dataset_filter_fn,
            transform_fn=self.dataset_transform_fn,
            load_native_ss=self.load_native_ss,
        )

        input_embedding = InputEmbedding(feature_config).to(config.device)

        return Trainer(
            config=config,
            model=self.get_model(input_embedding),
            train_data=train_data,
            valid_data=val_data,
            test_data=test_data,
            post_init_fn=self.get_post_model_init_fn(),
            allow_missing_modules=self.allow_missing_nn_modules,
            load_optim=self.load_optim,
        )

    @property
    def load_optim(self):
        """Whether to load trainer optimizer
        If lr or loss weights change, best to set to False in implementing class
        """
        return True

    def train(self):
        """Train the model"""
        trainer = self.get_trainer()
        trainer.train()

    @property
    def eval_kwargs(self):
        return dict()

    def eval(self):
        """Evaluate the model"""
        raise Exception("removed, only inference supported")

    def run(self, detect_anomoly=False):
        """Run train or eval"""
        try:
            with torch.autograd.set_detect_anomaly(detect_anomoly):
                if self.do_eval:
                    self.eval()
                else:
                    self.train()
        except Exception as e:
            ty = "eval" if self.do_eval else "training"
            print(f"[ERROR] Caught exception {e} during {ty}")
            raise e
