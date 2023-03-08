import torch
import os
from protein_learning.common.data.data_types.protein import Protein
from protein_learning.common.protein_constants import AA_TO_SC_ATOMS, BB_ATOMS, NAT_AA_SET

import protein_learning.models.model_abc.train as sc
from protein_learning.models.utils.model_io import (
    get_args_n_groups,
    load_args_for_eval,
)
from protein_learning.models.fbb_design.train import Train as SCPTrain, _augment
import protein_learning.common.protein_constants as pc
from protein_learning.features.input_embedding import InputEmbedding
from protein_learning.models.utils.dataset_augment_fns import impute_cb
from protein_learning.common.data.data_types.model_input import ModelInput
from typing import Optional, Union
from torch import Tensor
import traceback


def fill_atom_masks(protein: Protein, overwrite: bool = False) -> Protein:
    seq, atom_tys = protein.seq, protein.atom_tys
    bb_atom_set = set(BB_ATOMS)
    mask = torch.ones(len(seq), len(atom_tys))
    for i, a in enumerate(atom_tys):
        if a in bb_atom_set:
            continue
        for idx, s in enumerate(seq):
            if s not in NAT_AA_SET or a not in AA_TO_SC_ATOMS[s]:
                mask[idx, i] = 0
    if not overwrite:
        protein.atom_masks = protein.atom_masks & mask.bool()
    else:
        protein.atom_masks = mask.bool()
    return protein


def fill_missing_coords(protein: Protein) -> Protein:
    ca_coords = protein["CA"]
    atom_masks = protein.atom_masks
    for i in range(len(protein.atom_tys)):
        assert atom_masks[:, i].ndim == 1
        msk = ~atom_masks[:, i]
        protein.atom_coords[msk, i] = ca_coords[msk]
    return protein


def set_canonical_coords_n_masks(protein: Protein, overwrite: bool = False):
    return fill_missing_coords(fill_atom_masks(protein, overwrite=overwrite))


def exists(x):
    return x is not None


def default(x, y):
    return x if exists(x) else y


def _parse_args(arg_stream):
    args = []
    for x in arg_stream:  # noqa
        line = x.strip()
        if len(line.strip()) > 0 and not line.startswith("#"):
            arg = line.split(" ")
            for a in arg:
                args.append(a)
    return args


def parse_args(arg_path=None, arg_list=None, arg_string=None):
    if exists(arg_list):
        return arg_list
    elif exists(arg_path):
        with open(arg_path, "r") as f:
            return _parse_args(f)
    elif exists(arg_string):
        return _parse_args(arg_string.split("\n"))
    else:
        raise Exception("All inputs are None!")


def make_predicted_protein(model_out, seq: Optional[Union[str, Tensor]] = None) -> Protein:
    """Constructs predicted protein"""
    if torch.is_tensor(seq):
        seq = "".join([pc.INDEX_TO_AA_ONE[x.item()] for x in seq.squeeze()])
    coords = model_out.predicted_coords.squeeze(0).detach().clone()
    x = model_out.decoy_protein.clone()
    pred_protein = Protein(
        atom_coords=coords,
        atom_masks=x.atom_masks,
        atom_tys=x.atom_tys,
        seq=default(seq, x.seq),
        name=x._name,
        res_ids=x.res_ids,
        chain_ids=x.chain_ids,
        chain_indices=x.chain_indices,
        chain_names=x.chain_names,
        sec_struct=x.sec_struct,
        cdrs=x.cdrs,
    )
    pred_protein = set_canonical_coords_n_masks(pred_protein, overwrite=True)
    return pred_protein


INFERENCE_ARGS = """
# whether todesign sequence
--mask_seq
# Don't mask any residues in input sequence
--no_mask_weight 1
--inter_no_mask_weight 1
"""
BASIS_PATH = os.path.dirname(__file__)
NAME_TO_BASIS_CACHE = {
    "fbb_design_21_12_2022_16:00:06": os.path.join(BASIS_PATH, ".basis_cache", "rx7"),  # rx7 - 1
    "fbb_design_21_12_2022_16:07:51": os.path.join(BASIS_PATH, ".basis_cache", "rx7"),  # rx7 - 2
    "fbb_design_21_12_2022_15:57:43": os.path.join(BASIS_PATH, ".basis_cache", "rx11"),  # rx11
}


class Inference:
    def __init__(
        self,
        model_n_config_root: str,
        model_name: str,
    ):
        self.model_name = model_name
        self.trainer = SCPTrain()
        self.resource_root = model_n_config_root
        self.load_inference_args()
        self.trainer.pad_embeddings = self.args.mask_seq or self.args.mask_feats
        self.trainer._setup()

    def load_inference_args(self):
        eval_parser = sc.get_default_parser_for_eval()
        self.trainer.add_extra_cmd_line_options_for_eval(eval_parser)
        # default inference args
        arg_list = parse_args(arg_string=INFERENCE_ARGS)
        eval_args, eval_groups = get_args_n_groups(eval_parser, arg_list)
        self.trainer.eval_args = eval_args
        self.trainer.eval_groups = eval_groups

        train_parser = self.trainer.add_extra_cmd_line_options(sc.get_default_parser())
        train_args, train_groups = get_args_n_groups(train_parser, ["none"])  # defaults only

        global_override = sc.default_global_override_for_eval(eval_args)
        # args that should always be overridden
        global_override.update(self.trainer.global_override_eval)
        global_override.update(
            dict(
                raise_exceptions=True,
                out_root=self.resource_root,
                # config_path = GLOBAL_CONFIG_PATH,
            )
        )
        rr = self.resource_root
        config, args, arg_groups = load_args_for_eval(
            global_config_path=os.path.join(rr, "params", f"{self.model_name}.npy"),
            model_config_path=os.path.join(rr, "params", f"{self.model_name}_fbb_design.npy"),
            model_override=dict(**self.trainer.model_override_eval, model_config="none"),
            global_override=global_override,
            default_model_args=train_args,
            default_model_arg_groups=train_groups,
        )

        self.config, self.args, self.arg_groups = config, args, arg_groups
        self.trainer.config = config
        self.trainer.args = args
        self.trainer.arg_groups = arg_groups

        return config, args, arg_groups

    def _init_model(self):
        # set up feature generator
        feature_config = sc.get_input_feature_config(
            self.arg_groups,
            pad_embeddings=self.trainer.pad_embeddings,
            extra_pair_feat_dim=self.trainer.extra_pair_feat_dim,
            extra_res_feat_dim=self.trainer.extra_res_feat_dim,
        )
        self.feature_config = feature_config  # noqa
        feat_gen = sc.get_feature_gen(self.arg_groups, feature_config, apply_masks=self.trainer.apply_masks)
        self.feat_gen = feat_gen  # noqa
        self.input_embedding = InputEmbedding(feature_config)
        self.model = self.trainer.get_model(self.input_embedding)

    def get_model(self):
        self._init_model()
        model_path = os.path.join(self.resource_root, "models", f"{self.model_name}.tar")
        checkpoint = torch.load(model_path, map_location="cpu")
        try:
            self.model.load_state_dict(checkpoint["model"], strict=True)
        except:  # noqa
            # print(traceback.format_exc())
            self.model.load_state_dict(checkpoint["model"], strict=False)

        self.model.basis_dir = NAME_TO_BASIS_CACHE[self.model_name]
        return self.model

    def load_example(self, pdb_path, fasta_path=None):
        protein = Protein.FromPDBAndSeq(
            pdb_path=pdb_path,
            seq=fasta_path,
            atom_tys=pc.ALL_ATOMS,
            missing_seq=fasta_path is None,
            load_ss=False,
        )
        protein, _ = impute_cb(protein, protein)
        extra = _augment(protein, protein)
        protein = set_canonical_coords_n_masks(protein)

        return ModelInput(
            decoy=protein,
            native=protein,
            input_features=self.feat_gen.generate_features(
                protein,
                extra=extra,
                # seq_mask=seq_mask,
            ),
            extra=extra,
        )
