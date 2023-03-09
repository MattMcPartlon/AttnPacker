"""Train Masked Design Model"""
from __future__ import annotations
import os

# eb47fd6 dec 22
# 3cb9de3 dec 28
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
from functools import partial

import numpy as np
import torch
from protein_learning.features.input_embedding import InputEmbedding
from protein_learning.models.fbb_design.train_utils import TrainDesign
from protein_learning.networks.geometric_gt.geom_gt_config import add_gt_options, get_configs as get_gt_configs
from protein_learning.networks.loss.loss_fn import LossConfig, DefaultLossFunc
from protein_learning.features.default_feature_generator import DefaultFeatureGenerator
from protein_learning.common.data.data_types.protein import Protein
from protein_learning.models.fbb_design.fbb_model import FBBDesigner
from protein_learning.networks.se3_transformer.se3_transformer_config import SE3TransformerConfig, add_se3_options
from protein_learning.models.utils.dataset_augment_fns import impute_cb
from protein_learning.common.data.data_types.model_input import ExtraInput
from protein_learning.common.data.data_types.protein import Protein
from protein_learning.common.protein_constants import AA_TO_INDEX
from protein_learning.common.rigids import Rigids
from torch import Tensor
from typing import Any
from protein_learning.common.helpers import exists


class ExtraFBB(ExtraInput):
    """Store Encoded Native Sequence"""

    def __init__(
        self,
        native_seq_enc: Tensor,
        true_rigids: Rigids,
        decoy_rigids: Rigids,
        native_protein: Protein,
        decoy_protein: Protein,
    ):
        super(ExtraFBB, self).__init__()
        self.true_rigids = true_rigids
        self.decoy_rigids = decoy_rigids
        self.native = native_protein
        self.decoy = decoy_protein
        self.native_seq_enc = native_seq_enc if native_seq_enc.ndim == 2 else native_seq_enc.unsqueeze(0)

    def crop(self, start, end) -> ExtraFBB:
        """Crop native seq. encoding"""
        self.native_seq_enc = self.native_seq_enc[:, start:end]
        self.true_rigids = self.true_rigids.crop(start, end)
        self.decoy_rigids = self.decoy_rigids.crop(start, end)
        return self

    def to(self, device: Any) -> ExtraFBB:
        """Send native sequence encoding to device"""
        self.native_seq_enc = self.native_seq_enc.to(device)
        self.true_rigids = self.true_rigids.to(device)
        self.decoy_rigids = self.decoy_rigids.to(device)
        return self


def _augment(
    decoy_protein: Protein,
    native_protein: Protein,
) -> ExtraFBB:
    """Augment function for storing native seq. encoding in ModelInput object"""
    seq = native_protein.seq
    native_seq_enc = [AA_TO_INDEX[r] for r in seq]
    return ExtraFBB(
        torch.tensor(native_seq_enc).long(),
        true_rigids=native_protein.rigids,
        decoy_protein=decoy_protein,
        native_protein=native_protein,
        decoy_rigids=decoy_protein.rigids,
    )


class Train(TrainDesign):
    """Train Docking Model"""

    def __init__(self):
        super(Train, self).__init__()

    @property
    def use_flags(self):
        return False

    @property
    def model_name(self):
        """Name of model"""
        return "fbb_design"

    @property
    def load_native_ss(self):
        return False

    @property
    def load_optim(self):
        return True

    @property
    def global_override(self):
        """kwargs to override in global config"""
        return dict(max_len=440, lr=0.0005)  # ,checkpoint_idx=14)#raise_exceptions=False, max_len=388)

    @property
    def model_override(self):
        """kwargs to override in global model config"""
        return dict()

    def _model_override_eval(self):
        """Kwargs to override in model config for eval"""
        return dict()

    def _add_extra_cmd_line_options(self, parser):
        parser.add_argument("--use_tfn", action="store_true")
        parser.add_argument("--coord_noise", default=0, type=float)
        parser.add_argument("--torsion_loss_weight", default=0.2, type=float)
        parser.add_argument("--no_predict_from_angles", action="store_false", dest="predict_from_angles")
        parser.add_argument("--predict_bb", action="store_true")
        add_gt_options(parser)
        add_se3_options(parser)

        return parser

    def _add_extra_cmd_line_options_for_eval(self, parser):
        return parser

    @property
    def dataset_transform_fn(self):  # add noise + impute cb
        """Transform native and decoy input"""
        return partial(_transform, noise=self.args.coord_noise, eval=self.do_eval)

    @property
    def apply_masks(self):
        """Whether to apply masks to input"""
        return True

    def get_val_feat_gen(self, arg_groups, feature_config, apply_masks):  # TODO
        return DefaultFeatureGenerator(
            config=feature_config,
            intra_chain_mask_kwargs=vars(arg_groups["intra_chain_mask_args"]),
            inter_chain_mask_kwargs=vars(arg_groups["inter_chain_mask_args"]),
            **vars(arg_groups["feat_gen_args"]),
            apply_masks=apply_masks,
        )

    def get_model(self, input_embedding: InputEmbedding):
        """Get Docking model"""
        config, args, arg_groups = self.config, self.args, self.arg_groups
        se3_args = self.arg_groups["se3_args"]
        return FBBDesigner(
            input_embedding=input_embedding,
            loss_fn=DefaultLossFunc(
                LossConfig(
                    res_dim=args.node_dim_hidden,
                    pair_dim=args.pair_dim_hidden,
                    output_atom_tys=self.output_atom_tys,
                    **vars(arg_groups["loss_args"]),
                )
            ),
            gt_configs=get_gt_configs(
                node_dim=args.node_dim_hidden,
                pair_dim=args.pair_dim_hidden,
                opts=arg_groups["gt_args"],
            ),
            se3_config=SE3TransformerConfig(
                fiber_in=se3_args.fiber_in,
                fiber_hidden=se3_args.fiber_hidden,
                fiber_out=se3_args.fiber_out,
                heads=se3_args.se3_heads,
                dim_heads=se3_args.se3_dim_heads,
                edge_dim=se3_args.se3_edge_dim,
                depth=se3_args.se3_depth,
            )
            if self.args.use_tfn
            else None,
        )

    @property
    def input_augment_fn(self):
        """Augment model input"""
        return _augment


def _transform(decoy, native, noise=0, eval=False):
    assert len(native) == len(decoy)
    decoy, native = impute_cb(decoy, native)
    decoy.atom_coords += torch.randn_like(decoy.atom_coords) * noise
    return decoy, native


if __name__ == "__main__":
    x = Train()
    ty = "Training" if not x.do_eval else "Evaluation"
    print(f"[INFO] Beginning {ty} for Masked Design Model")
    x.run(detect_anomoly=False)
