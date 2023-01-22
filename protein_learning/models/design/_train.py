"""Training class for design models"""
from abc import abstractmethod
from functools import partial
from typing import List

from protein_learning.common.protein_constants import ALL_ATOMS
from protein_learning.features.input_embedding import InputEmbedding
from protein_learning.models.masked_design.masked_design_utils import augment
from protein_learning.models.model_abc.train import TrainABC
from protein_learning.models.utils.esm_embedder import ESMInputEmbedder
from protein_learning.models.utils.esm_input import ESMFeatGen
from protein_learning.networks.evoformer.evoformer_config import EvoformerConfig
from protein_learning.networks.se3_transformer.se3_transformer_config import SE3TransformerConfig


class TrainDesign(TrainABC):
    """Train Masked Design Model"""

    def __init__(self):
        super(TrainDesign, self).__init__()

    def add_extra_cmd_line_options(self, parser):
        """Add extra optiona at command line"""
        model_options = parser.add_argument_group("model_options")
        model_options.add_argument('--coord_dim_hidden',
                                   help="coordinate hidden dimension",
                                   default=16, type=int
                                   )

        model_options.add_argument('--tfn_depth',
                                   help="number of heads for IPA (scalar, coord)",
                                   default=3, type=int
                                   )

        model_options.add_argument('--evoformer_depth',
                                   help="number of heads for IPA (scalar, coord)",
                                   default=3, type=int
                                   )

        esm_args = parser.add_argument_group("esm_args")
        esm_args.add_argument("--use_esm_prob", type=float, default=1)
        esm_args.add_argument("--use_esm_both_chain_prob", type=float, default=1)
        esm_args.add_argument("--use_esm_1b", action="store_true")
        esm_args.add_argument("--use_esm_msa", action="store_true")
        esm_args.add_argument("--esm_gpu_idx", type=int, default=None)
        esm_args.add_argument("--esm_msa_dir", default=None)
        esm_args.add_argument("--max_msa_seqs", type=int, default=128)

    @property
    def model_name(self):
        """Name of model"""
        return "inverse_folding"

    @property
    def global_override(self):
        """kwargs to override in global config"""
        return dict()

    @property
    def model_override(self):
        """kwargs to override in global model config"""
        return dict()

    @property
    def input_atom_tys(self) -> List:
        """Atom types to load in input"""
        return ALL_ATOMS

    @property
    def output_atom_tys(self) -> List:
        """Atom types to predict"""
        return ALL_ATOMS

    @property
    def extra_res_feat_dim(self):
        """Residue flag dim"""
        return 0

    @property
    def extra_pair_feat_dim(self):
        """pair flag dim"""
        return 0

    @property
    def apply_masks(self):
        """Whether to apply masks to input"""
        return True

    @property
    def input_augment_fn(self):
        """Augment model input"""
        return partial(
            augment,
            precompute_rigids=False,
            flag_gen=None,
            use_esm=(self.esm_embedder is not None),
            esm_chain_selector=_esm_chain_select_fn,
        )

    # TODO: FIx seq. corruption

    def _setup(self):
        args = self.args
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

        self.evo_config = EvoformerConfig(
            node_dim=args.node_dim_hidden,
            edge_dim=args.pair_dim_hidden,
            depth=args.evoformer_depth,
        )

        self.tfn_config = SE3TransformerConfig(
            fiber_in=(args.node_dim_hidden, 3),
            fiber_out=(args.node_dim_hidden, len(self.output_atom_tys)),
            fiber_hidden=(args.node_dim_hidden, args.coord_dim_hidden),
            edge_dim=args.pair_dim_hidden,
            depth=args.tfn_depth,
        )

    @property
    def dataset_transform_fn(self):
        """Transform native and decoy input"""
        return _transform

    @property
    def dataset_filter_fn(self):
        """Filter dataset entries"""
        return True

    @abstractmethod
    def get_model(self, input_embedding: InputEmbedding):
        """Get Model"""
        pass


def _esm_chain_select_fn(*args, **kwargs):
    return [True]


def _transform(decoy, native, *args, **kwargs):
    return native.clone(), native
