"""Global Model configuration"""
import os
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from typing import Optional, NamedTuple, List

import numpy as np
import torch

from protein_learning.common.global_constants import (
    START_TIME, CHECKPOINTS, LOGS, PARAMS, MODELS, STATS
)
from protein_learning.common.helpers import exists, default
from protein_learning.common.io.utils import parse_arg_file, load_npy


def get_config(arg_list):
    """gets args to pass to ModelConfig"""
    parser = ArgumentParser(description="Protein Learning Model",  # noqa
                            epilog='trains Equivariant Transformer on model refinement task',
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--name',
                        help='name to use for model output. output will be saved as '
                             'os.path.join(out_root,<ty>,name_{date_time}.<suffix>)',
                        type=str,
                        default="model"
                        )

    parser.add_argument('--out_root',
                        help="root directory to store output in used for storing model statistics,"
                             " saving the model,"
                             " and checkpointing the model.",
                        type=str,
                        default='./output'
                        )

    parser.add_argument('--train_decoy_folder',
                        help='directory for loading decoys to train on',
                        type=str,
                        )

    parser.add_argument('--val_decoy_folder',
                        help='directory for loading decoys to validate on',
                        type=str,
                        )

    parser.add_argument('--test_decoy_folder',
                        help='directory for loading decoys to test on',
                        type=str,
                        )

    parser.add_argument('--train_list',
                        help='list of samples to train on. Must be formatted as : [dataset] [target] [model_pdb] '
                             'where \ndataset: is the folder containing the target models \ntarget: is the name'
                             ' of the target, and directory containing pdb files\n model_pdb: is the pdb for '
                             'the model.\n All paths are taken relative to the data_root argument',
                        type=str, default=None)

    parser.add_argument('--val_list',
                        help='list of samples to validate on on. (see --train_list)',
                        type=str, default=None)

    parser.add_argument('--test_list',
                        help='list of samples to validate on on. (see --train_list)',
                        type=str, default=None)

    parser.add_argument('--train_native_folder',
                        help='full path to folder storing native pdbs for --train_list samples',
                        type=str, default=None)

    parser.add_argument('--val_native_folder',
                        help='full path to folder storing native pdbs for --val_list samples',
                        type=str, default=None)

    parser.add_argument('--test_native_folder',
                        help='full path to folder storing native pdbs for --test_list samples',
                        type=str, default=None)

    parser.add_argument('--train_seq_folder',
                        help='full path to folder storing native sequences for --train_list samples',
                        type=str, default=None)

    parser.add_argument('--val_seq_folder',
                        help='full path to folder storing native sequences for --val_list samples',
                        type=str, default=None)

    parser.add_argument('--test_seq_folder',
                        help='full path to folder storing native sequences for --test_list samples',
                        type=str, default=None)

    parser.add_argument('--load_state',
                        help='load from previous state', action='store_true', dest='load_state')

    parser.set_defaults(raise_exceptions=False)
    parser.add_argument('--raise_exceptions',
                        help='whether to raise exceptions', action='store_true', dest='raise_exceptions')

    parser.add_argument('--config_path',
                        help='path to load config from if --load_state specified',
                        type=str,
                        default=None)

    parser.add_argument('--checkpoint_idx',
                        help=' (optional) index of model checkpoint to load\n'
                             'loads most recent model state if value is negative',
                        default=-1, type=int)

    # Run Options #

    run_settings = parser.add_argument_group('Run Settings')

    run_settings.add_argument('--data_workers',
                              help="number of workers to use in loading data",
                              type=int,
                              default=4)

    run_settings.set_defaults(shuffle=True)
    run_settings.add_argument('--no_shuffle',
                              help="whether or not to shuffle the train/validation data",
                              action='store_false',
                              dest='shuffle'
                              )

    run_settings.add_argument('-g', '--gpu_indices',
                              help="gpu index to run on",
                              type=str,
                              nargs='+',
                              default=["0"]
                              )

    run_settings.add_argument('--checkpoint_every',
                              help="frequency (num batches) at which a hard copy of model is saved\n"
                                   "use negative value to disable checkpointing model",
                              type=int,
                              default=1000)

    run_settings.add_argument('--save_every',
                              help="frequency (num batches) at which model state is saved",
                              type=int,
                              default=50)

    run_settings.add_argument('--validate_every',
                              help="frequency (num batches) at which to run model on validation set",
                              type=int,
                              default=250)

    run_settings.add_argument('--test_every',
                              help="frequency (num batches) at which to run model on test set",
                              type=int,
                              default=250)

    run_settings.add_argument('--max_val_samples',
                              help="maximum number of samples to validate on",
                              type=int, default=300)

    run_settings.add_argument('--max_test_samples',
                              help="maximum number of samples to validate on",
                              type=int, default=300)

    run_settings.add_argument('--max_len',
                              help='maximum (sequence) length to run through the model'
                                   'proteins with more residues than this value will have their'
                                   'sequence and coordinates randomly cropped.',
                              default=300,
                              type=int)

    # Training Settings #

    training_settings = parser.add_argument_group("Training Settings")
    training_settings.add_argument('-b', '--batch_size',
                                   help="batch size", type=int, default=32)

    training_settings.add_argument('-e', '--epochs',
                                   help="max number of epochs to run for",
                                   type=int, default=10)

    training_settings.add_argument('--decrease_lr_by',
                                   help="decrease learning rate by this factor every --decrease_lr_every epochs",
                                   default=1, type=float)

    training_settings.add_argument('--decrease_lr_every',
                                   help="decrease learning rate by --decrease_lr_by every --decrease_lr_every\n"
                                        "many epochs. If this parameter is not positive, no update will be performed.",
                                   default=0, type=int)

    training_settings.add_argument('--lr',
                                   help="learning rate to use during training",
                                   type=float,
                                   default=1e-4)

    training_settings.add_argument('--weight_decay',
                                   help="weight decay parameter for l2 regularization",
                                   type=float,
                                   default=0)

    training_settings.add_argument('--perturb_wts',
                                   help="perturb model weights using this value as variance",
                                   type=float,
                                   default=0)

    training_settings.add_argument('--grad_norm_clip',
                                   type=float,
                                   default=3,
                                   help="scale gradients so that cumulative norm is at most this value")
    training_settings.set_defaults(average_batch=True)
    training_settings.add_argument('--no_average_batch',
                                   action="store_false",
                                   dest="average_batch",
                                   )

    training_settings.add_argument('--example_clip',
                                   type=float,
                                   default=1.0,
                                   )

    training_settings.set_defaults(clip_grad_per_sample=True)
    training_settings.add_argument('--no_clip_grad_per_sample',
                                   action='store_false',
                                   help="clip gradients in each sample (rather than entire batch)",
                                   dest="clip_grad_per_sample"
                                   )

    training_settings.set_defaults(use_re_zero=True)
    training_settings.add_argument('--no_use_re_zero',
                                   action='store_false',
                                   dest='use_re_zero',
                                   help="do not use reZero residual scheme")

    training_settings.add_argument('--rmsd_cutoff',
                                   help="cutoff (in angstroms) for maximum allowed rmsd between model and native. "
                                        "Training and validation samples with initial RMSD > cutoff will be ignored",
                                   type=int,
                                   default=10)

    training_settings.add_argument('--tm_cutoff',
                                   help="cutoff (in angstroms) for minimum allowed tm between model and native. "
                                        "Training and validation samples with initial tm < cutoff will be ignored",
                                   type=float,
                                   default=0.0)

    # Loss Settings #

    loss_settings = parser.add_argument_group("Loss Settings")
    # coordinate loss types
    loss_settings.add_argument('--coord_loss_tys',
                               nargs='+',
                               type=str,
                               default=[],
                               help=f"list of coordinate loss types."
                               )

    loss_settings.add_argument('--coord_loss_weights',
                               nargs='+',
                               type=float,
                               default=[],
                               help="weight of each coordinate loss type"
                               )

    loss_settings.add_argument('--atom_loss_tys',
                               nargs='+',
                               type=str,
                               default=[],
                               help=f"list of atom loss types"
                               )

    loss_settings.add_argument('--atom_loss_weights',
                               nargs='+',
                               type=float,
                               default=[],
                               help="weight of each atom loss type"
                               )

    loss_settings.add_argument('--pair_loss_tys',
                               nargs='+',
                               type=str,
                               default=[],
                               help=f"list of pair loss types."
                               )

    loss_settings.add_argument('--pair_loss_weights',
                               nargs='+',
                               type=float,
                               default=[],
                               help="weight of each pair loss type"
                               )

    model_settings = parser.add_argument_group("Model Settings")

    model_settings.add_argument("--se3_configs",
                                nargs="+",
                                help="path to file conttaining se3 attention config(s)",
                                default=[],
                                )
    model_settings.add_argument("--evoformer_configs",
                                nargs="+",
                                help="path to file conttaining evoformer config(s)",
                                default=[],
                                )
    model_settings.add_argument("--ipa_configs",
                                nargs="+",
                                help="path to file conttaining evoformer config(s)",
                                default=[],
                                )

    # new options
    parser.set_defaults(save_pdbs=False)
    parser.add_argument("--save_pdbs", action="store_true")
    parser.add_argument("--pdb_dir", default="./", type=str)

    args = parser.parse_args(arg_list)
    if not args.load_state:
        args.name = args.name + f"_{START_TIME}"
    return args


class GlobalConfig(NamedTuple):
    """Global model configuration parameters"""
    name: str
    out_root: str
    checkpoint_idx: int
    load_state: bool
    config_path: str
    raise_exceptions: bool
    # train/validate/test info
    train_decoy_folder: str
    train_list: str
    train_seq_folder: str
    train_native_folder: str
    val_decoy_folder: str
    val_list: str
    val_seq_folder: str
    val_native_folder: str
    test_decoy_folder: str
    test_list: str
    test_seq_folder: str
    test_native_folder: str

    # Training/data loading settings
    data_workers: int
    shuffle: bool
    gpu_indices: List[str]
    checkpoint_every: int
    save_every: int
    validate_every: int
    test_every: int
    max_val_samples: int
    max_test_samples: int
    max_len: int
    epochs: int
    batch_size: int
    decrease_lr_by: float
    decrease_lr_every: int
    lr: float
    weight_decay: float
    grad_norm_clip: float
    clip_grad_per_sample: bool
    use_re_zero: bool
    tm_cutoff: float
    rmsd_cutoff: float
    average_batch: bool
    example_clip: float
    perturb_wts: float

    # Loss settings
    coord_loss_tys: List[str]
    atom_loss_tys: List[str]
    pair_loss_tys: List[str]
    coord_loss_weights: List[float]
    pair_loss_weights: List[float]
    atom_loss_weights: List[float]

    # model configs
    se3_configs: List[str]
    evoformer_configs: List[str]
    ipa_configs: List[str]

    # new options
    save_pdbs: bool = False
    pdb_dir: str = "./"

    @property
    def load_from_checkpoint(self):
        """Whether to load model from checkpoint"""
        return self.load_state and self.checkpoint_idx >= 0

    @property
    def learning_rate(self):
        """learning rate"""
        return self.lr

    @property
    def store_model_checkpoints(self):
        """Whether to store mhard checkpoints of model during training"""
        return self.checkpoint_every > 0

    def _get_dir(self, ty):
        path = os.path.join(self.out_root, ty)
        os.makedirs(path, exist_ok=True)
        return path

    @property
    def model_load_path(self):
        """Gets directory to load model from"""
        assert self.load_state
        if self.load_from_checkpoint:
            return self.checkpoint_path(self.checkpoint_idx)
        return self.save_path

    @property
    def checkpoint_directory(self):
        """Gets directory to store model checkpoints to"""
        return self._get_dir(CHECKPOINTS)

    def checkpoint_path(self, idx):
        """Gets directory to store model checkpoints to"""
        check_dir = self._get_dir(CHECKPOINTS)
        return os.path.join(check_dir, f"{self.name}_cp_{idx}.tar")

    @property
    def log_directory(self):
        """Gets directory to store model logging info to"""
        return self._get_dir(LOGS)

    @property
    def log_path(self):
        """Gets path to store model logging info to"""
        return os.path.join(self.log_directory, self.name + ".log")

    @property
    def device(self):
        """Gets the model device"""
        if int(self.gpu_indices[0]) < 0:
            return "cpu"
        if torch.cuda.is_available():
            return f"cuda:{self.gpu_indices[0]}"
        return "cpu"

    @property
    def stats_directory(self):
        """Gets directory to store model stats to"""
        return self._get_dir(STATS)

    @property
    def stats_path(self):
        """Gets path to store model stats to"""
        return os.path.join(self.stats_directory, self.name + ".npy")

    @property
    def save_directory(self):
        """Gets directory to save model to"""
        return self._get_dir(MODELS)

    @property
    def save_path(self):
        """Gets path to save model to"""
        return os.path.join(self.save_directory, self.name + ".tar")

    @property
    def param_directory(self):
        """Gets directory to store model params to"""
        return self._get_dir(PARAMS)

    @property
    def param_path(self):
        """Gets path to store model params to"""
        return os.path.join(self.param_directory, self.name + ".npy")


def make_config(arg_file=None) -> GlobalConfig:
    """Make a ModelConfig Object from arg file or command line"""
    args = parse_arg_file(arg_file) if exists(arg_file) else sys.argv[1:]
    return GlobalConfig(**vars(get_config(args)))


def print_config(config: GlobalConfig):
    """Prints the config"""
    mems = config._asdict()  # noqa
    for k, v in mems.items():
        print(f"{k} : {v}")


def load_config(curr_config: Optional[GlobalConfig] = None, config_path: Optional[str] = None,
                **override) -> GlobalConfig:
    """Load a ModelConfig
    :param curr_config: a ModelConfig
    :param config_path: path to config file
    :param override: any arguments to override
    :return: model config with specified arguments overridden
    """
    if exists(curr_config):
        config_path = default(curr_config.param_path, config_path)
    else:
        assert exists(config_path)
    load_kwargs = load_npy(config_path)
    load_kwargs['load_state'] = True
    load_kwargs['config_path'] = config_path
    load_kwargs.update(override)
    return GlobalConfig(**load_kwargs)


def save_config(config: GlobalConfig):
    """Saves config"""
    os.makedirs(config.param_directory, exist_ok=True)
    np.save(config.param_path, config._asdict())  # noqa
