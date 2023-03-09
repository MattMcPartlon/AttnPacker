"""Training class for design models"""
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
from abc import abstractmethod
from functools import partial
from typing import List

from protein_learning.common.protein_constants import ALL_ATOMS
from protein_learning.features.input_embedding import InputEmbedding
from protein_learning.features.masking.partition import (
    ChainPartitionGenerator,
)
from protein_learning.models.utils.feature_flags import FeatureFlagGen
from protein_learning.models.model_abc.train import TrainABC
from protein_learning.models.utils.dataset_augment_fns import impute_cb, partition_chain

# from protein_learning.models.utils.esm_embedder import ESMInputEmbedder
# from protein_learning.models.utils.esm_input import ESMFeatGen
from protein_learning.models.utils.opt_parse import add_flag_args, add_esm_options


class TrainDesign(TrainABC):
    """Train Masked Design Model"""

    def __init__(self, skip_init=False):
        super(TrainDesign, self).__init__(skip_init=skip_init)

    @abstractmethod
    def _add_extra_cmd_line_options(self, parser):
        pass

    @abstractmethod
    def _add_extra_cmd_line_options_for_eval(self, parser):
        pass

    @abstractmethod
    def _model_override_eval(self):
        """Kwargs to override in model config for eval"""
        pass

    @property
    def global_override_eval(self):
        """kwargs to override in global config for eval"""
        return dict(data_workers=4)

    @property
    def model_override_eval(self):
        """Kwargs to override in model config for eval"""
        override = dict(
            use_cycles=self.eval_args.use_cycles,
            partition_chains=self.eval_args.partition_chains,
            train_on_bounded_complex=self.eval_args.train_on_bounded_complex,
            min_len_cutoff=self.eval_args.min_len_cutoff,
        )
        override.update(**vars(self.eval_groups["esm_args"]))
        override.update(**vars(self.eval_groups["flag_args"]))
        override.update(**self._model_override_eval())
        override.update(**vars(self.eval_groups["feat_gen_args"]))
        override.update(**vars(self.eval_groups["intra_chain_mask_args"]))
        override.update(**vars(self.eval_groups["chain_partition_args"]))
        override.update(**vars(self.eval_groups["inter_chain_mask_args"]))
        return override

    def add_extra_cmd_line_options(self, parser):
        """Extra options to parse for"""
        parser.add_argument("--n_cycles", default=1, type=int)
        parser.add_argument("--partition_chains", action="store_true")
        parser.add_argument("--predict_sc_atoms", action="store_true")
        parser.add_argument("--include_init_rigids", action="store_true")
        parser.add_argument("--train_on_bounded_complex", action="store_true")
        parser.add_argument("--min_len_cutoff", type=int, default=60)
        parser, _ = add_esm_options(parser)
        parser, _ = add_flag_args(parser)
        self._add_extra_cmd_line_options(parser)
        return parser

    def add_extra_cmd_line_options_for_eval(self, parser):
        parser.add_argument("--use_cycles", default=1, type=int)
        parser.add_argument("--partition_chains", action="store_true")
        parser.add_argument("--train_on_bounded_complex", action="store_true")
        parser.add_argument("--min_len_cutoff", type=int, default=60)
        esm_args = parser.add_argument_group("esm_args")
        esm_args.add_argument("--use_esm_prob", type=float, default=0)
        esm_args.add_argument("--use_esm_both_chain_prob", type=float, default=0)
        esm_args.add_argument("--esm_gpu_idx", default=0)
        flag_args = parser.add_argument_group("flag_args")
        flag_args.add_argument("--include_contact_prob", type=float, default=0)
        flag_args.add_argument("--include_interface_prob", type=float, default=0)
        flag_args.add_argument("--contact_fracs", type=float, default=[0, 0], nargs="+")
        flag_args.add_argument("--interface_fracs", type=float, default=0, nargs="+")
        flag_args.add_argument("--include_both_interface_prob", type=float, default=0)
        flag_args.add_argument("--num_interface", type=float, default=None)
        flag_args.add_argument("--num_contact", type=float, default=None)
        flag_args.add_argument("--random_interface", type=int, default=None)
        flag_args.add_argument("--random_contact", type=int, default=None)
        self._add_extra_cmd_line_options_for_eval(parser)
        return parser

    @property
    def input_atom_tys(self) -> List:
        """Atom types to load in input"""
        return ALL_ATOMS if self.args.predict_sc_atoms else ["N", "CA", "C", "CB"]

    @property
    def output_atom_tys(self) -> List:
        """Atom types to predict"""
        return ALL_ATOMS if self.args.predict_sc_atoms else ["N", "CA", "C", "CB"]

    @property
    def load_native_ss(self):
        return True

    @property
    def extra_res_feat_dim(self):
        """Residue flag dim"""
        return self.flag_gen.flag_dims[0] if self.flag_gen is not None else 0

    @property
    def use_flags(self):
        return True

    @property
    def extra_pair_feat_dim(self):
        """pair flag dim"""
        return self.flag_gen.flag_dims[1] if self.flag_gen is not None else 0

    def _setup(self):
        self.flag_gen = FeatureFlagGen(**vars(self.arg_groups["flag_args"])) if self.use_flags else None
        self.partition_gen = None
        if self.args.partition_chains:
            self.partition_gen = ChainPartitionGenerator(
                strat_n_weight_kwargs=vars(self.arg_groups["chain_partition_args"])
            )
        self.esm_embedder = None
        if self.args.use_esm_1b or self.args.use_esm_msa:
            esm_feat_gen = ESMFeatGen(
                use_esm_msa=self.args.use_esm_msa,
                use_esm_1b=self.args.use_esm_1b,
                esm_gpu_idx=self.args.esm_gpu_idx,
                msa_dir=self.args.esm_msa_dir,
                max_msa_seqs=self.args.max_msa_seqs,
            )
            self.esm_embedder = ESMInputEmbedder(
                use_esm_1b=self.args.use_esm_1b,
                use_esm_msa=self.args.use_esm_msa,
                node_dim_in=self.args.node_dim_hidden,
                pair_dim_in=self.args.pair_dim_hidden,
                node_dim_out=self.args.node_dim_hidden,
                pair_dim_out=self.args.pair_dim_hidden,
                feat_gen=esm_feat_gen,
            )

    @property
    def dataset_transform_fn(self):
        """Transform native and decoy input"""
        return partial(
            _transform, partition_gen=self.partition_gen, train_on_bounded=self.args.train_on_bounded_complex
        )

    @property
    def dataset_filter_fn(self):
        """Filter dataset entries"""
        min_len = self.args.min_len_cutoff
        return lambda x, y, _min_len=min_len: len(x) > _min_len

    @abstractmethod
    def get_model(self, input_embedding: InputEmbedding):
        """Get Model"""
        pass


def _transform(decoy, native, train_on_bounded: bool = False, partition_gen=None):
    assert len(native) == len(decoy)
    if train_on_bounded:
        decoy = native.clone()
    decoy, native = impute_cb(decoy, native)
    if partition_gen is not None:
        decoy, native = partition_chain(decoy, native, partition_gen)
    return decoy, native
