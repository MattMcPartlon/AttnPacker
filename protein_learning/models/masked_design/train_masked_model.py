"""Train Masked Design Model"""
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'

from functools import partial

import numpy as np

from protein_learning.features.input_embedding import InputEmbedding
from protein_learning.models.masked_design.design_train import TrainDesign
from protein_learning.models.masked_design.masked_design_utils import augment
from protein_learning.models.masked_design.masked_design_model import ComplexDesigner
from protein_learning.networks.geometric_gt.geom_gt_config import (
    add_gt_options, get_configs as get_gt_configs
)
from protein_learning.networks.loss.loss_fn import LossConfig, DefaultLossFunc
from protein_learning.common.helpers import default


class Train(TrainDesign):
    """Train Masked Design Model"""

    def __init__(self):
        super(Train, self).__init__()

    @property
    def model_name(self):
        """Name of model"""
        return "masked_design"

    @property
    def global_override(self):
        """kwargs to override in global config"""
        return {}  # dict(data_workers=4)

    @property
    def model_override(self):
        """kwargs to override in global model config"""
        return dict()#contiguous_mask_max_len=100,random_mask_min_p=0.0,random_mask_max_p=1,max_mask_frac=0.5,)

    def _model_override_eval(self):
        """Kwargs to override in model config for eval"""
        return dict()

    def _add_extra_cmd_line_options(self, parser):
        add_gt_options(parser)

    def _add_extra_cmd_line_options_for_eval(self, parser):
        return parser

    @property
    def apply_masks(self):
        """Whether to apply masks to input"""
        return True

    @property
    def input_augment_fn(self):
        """Augment model input"""
        return partial(
            augment,
            precompute_rigids=self.args.include_init_rigids,
            flag_gen=self.flag_gen,
            use_esm=(self.esm_embedder is not None),
            esm_chain_selector=partial(
                _esm_chain_select_fn,
                select_both_prob=self.args.use_esm_both_chain_prob,
                use_esm_prob=self.args.use_esm_prob,
            )
        )

    def get_model(self, input_embedding: InputEmbedding):
        """Get DesignVAE model"""
        config, args, arg_groups = self.config, self.args, self.arg_groups
        extra_kwargs = {}
        if self.do_eval:
            extra_kwargs["use_cycles"] = max(1, self.args.n_cycles)
        return ComplexDesigner(
            input_embedding=input_embedding,
            loss_fn=DefaultLossFunc(
                LossConfig(
                    res_dim=args.node_dim_hidden,
                    pair_dim=args.pair_dim_hidden,
                    output_atom_tys=self.output_atom_tys,
                    **vars(arg_groups["loss_args"])
                )
            ),
            n_cycles=args.n_cycles,
            gt_configs=get_gt_configs(
                node_dim=args.node_dim_hidden,
                pair_dim=args.pair_dim_hidden,
                opts=arg_groups["gt_args"],
            ),
            coord_dim_out=len(self.output_atom_tys),
            esm_embedder=self.esm_embedder,
            **extra_kwargs,
        )


def _esm_chain_select_fn(
        ptn,
        is_complex,
        select_both_prob: float = 1,
        use_esm_prob: float = 1
):
    chain_selection = [False] * len(ptn.chain_indices)
    if np.random.uniform() < use_esm_prob:
        largest_idx = np.argmax(list(map(len, ptn.chain_indices)))
        chain_selection[largest_idx] = True
        if np.random.uniform() < select_both_prob:
            chain_selection = [True] * len(ptn.chain_indices)
    return chain_selection


if __name__ == "__main__":
    x = Train()
    ty = "Training" if not x.do_eval else "Evaluation"
    print(f"[INFO] Beginning {ty} for Masked Design Model")
    x.run(detect_anomoly=False)
