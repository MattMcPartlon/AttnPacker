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
from protein_learning.common.data.data_types.model_output import ModelOutput
from typing import Optional, Union
from torch import Tensor
import traceback
from copy import deepcopy


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


def format_prediction(model, model_in, model_out, device="cpu"):
    # masked residue positions
    seq_mask = default(model_in.input_features.seq_mask, torch.zeros(len(model_in.decoy)).bool())

    # coords, pLDDT, and Sequence
    res_feats = model_out.scalar_output
    pred_coords = model_out.predicted_coords
    pred_seq_labels = model_out.decoy_protein.seq_encoding
    pred_seq_logits = None

    # Predicted pLDDT
    plddt_head = model.loss_fn.loss_fns["plddt"]
    pred_plddt = plddt_head.get_expected_value(res_feats)
    pred_plddt = pred_plddt

    # Predicted Sequence
    if "nsr" in model.loss_fn:
        pred_seq_logits = model.loss_fn.loss_fns["nsr"].get_predicted_logits(res_feats)
        pred_seq_labels = torch.argmax(pred_seq_logits, dim=-1)

    out = dict(
        pred_coords=pred_coords,
        pred_seq_labels=pred_seq_labels,
        pred_seq_logits=pred_seq_logits,
        pred_plddt=pred_plddt,
        res_output=res_feats,
        pair_output=model_out.pair_output,
        design_mask=seq_mask,
        model_out=model_out.detach(),
        model_in=model_in,
        seq="".join([pc.INDEX_TO_AA_ONE[x.item()] for x in pred_seq_labels.squeeze()]),
    )
    fn = lambda x: x.detach().to(device).squeeze() if torch.is_tensor(x) else x
    return {k: fn(v) for k, v in out.items()}


def chunk_inference(
    model,
    sample: ModelInput,
    max_len: int = 500,
    device="cpu",
):
    seq_len = len(sample.decoy.seq)
    crops, crop_offset = get_inference_crops(max_len, seq_len=seq_len)
    output = []

    for crop in crops:
        sample.native.__clear_cache__()
        sample.decoy.__clear_cache__()
        sample_copy = deepcopy(sample)
        croppy = sample_copy.to(device).crop(max_len, bounds=crop)  # punny
        output.append(model(croppy))

    device = output[0].scalar_output.device
    scalar_out_shape = output[0].scalar_output.shape
    pair_dim = output[0].pair_output.shape[-1]
    predicted_coords = torch.zeros(1, seq_len, 37, 3, device=device)
    scalar_output = torch.zeros(1, seq_len, scalar_out_shape[-1], device=device)
    pair_output = torch.zeros(1, seq_len, seq_len, pair_dim, device=device)

    quats, translations, angles = None, None, None
    if exists(output[0].angles):
        angles = torch.zeros(1, seq_len, *output[0].angles.shape[2:], device=device)
    if exists(output[0].pred_rigids):
        quats = torch.zeros(1, seq_len, *output[0].pred_rigids.quaternions.shape[2:], device=device)
        translations = torch.zeros(1, seq_len, *output[0].pred_rigids.translations.shape[2:], device=device)

    for i in range(len(crops)):
        offset = 0 if i == 0 else crop_offset // 2
        start, end = crops[i]
        curr_out = output[i]
        scalar_output[:, start + offset : end] = curr_out.scalar_output[:, offset:]
        predicted_coords[:, start + offset : end] = curr_out.predicted_coords[:, offset:]
        pair_output[:, start + offset : end, start + offset : end] = curr_out.pair_output[:, offset:, offset:]
        if exists(curr_out.pred_rigids):
            translations[:, start + offset : end] = curr_out.pred_rigids.translations[:, offset:]
            quats[:, start + offset : end] = curr_out.pred_rigids.quaternions[:, offset:]
        if exists(curr_out.angles):
            angles[:, start + offset : end] = curr_out.angles[:, offset:]

    # overwrite model out features
    return ModelOutput(
        scalar_output=scalar_output,
        predicted_coords=predicted_coords,
        pair_output=pair_output,
        model_input=sample.to(device),
        extra=dict(
            pred_rigids=None, chain_indices=sample.decoy.chain_indices, angles=angles, unnormalized_angles=angles
        ),
    )


"""Helper Methods"""


def get_inference_crops(max_len, seq_len):
    crop_offset = min(max_len // 2, 200)
    crops, start = [(0, max_len)], max_len - crop_offset
    while start < seq_len - crop_offset:
        crop = (start, min(seq_len, start + max_len))
        crops.append(crop)
        start += max_len - crop_offset
    crops = crops[:-1]
    crops.append((seq_len - max_len, seq_len))
    return crops, crop_offset


def _convert_tensor(x: torch.Tensor):
    x = x.squeeze(0).detach().cpu()
    if x.dtype in [torch.float32, torch.double, torch.float64, torch.float]:
        x = x.float()
    return x.numpy()


def _convert_value(value):
    if torch.is_tensor(value):
        return _convert_tensor(value)
    if isinstance(value, list):
        return [_convert_value(x) for x in value]
    if isinstance(value, tuple):
        return tuple([_convert_value(x) for x in value])
    return value


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
    "fbb_design_05_04_2023_17:56:21": os.path.join(BASIS_PATH, ".basis_cache", "rx7"),  # +rot conditioning, noise 0.05
    "fbb_design_05_04_2023_17:56:52": os.path.join(BASIS_PATH, ".basis_cache", "rx7"),  # +rot conditioning
}


class Inference:
    def __init__(
        self,
        model_n_config_root: str,
        use_design_variant: bool = False,
        use_rotamer_conditioning: bool = False,
    ):
        self.model = None
        if not use_rotamer_conditioning:
            if use_design_variant:
                self.model_name = "fbb_design_21_12_2022_16:07:51"
            else:
                self.model_name = "fbb_design_21_12_2022_15:57:43"
        else:
            # Note: can swap with fbb_design_05_04_2023_17:56:21, which was trained with
            # backbone Gaussian noise
            self.model_name = "fbb_design_05_04_2023_17:56:52"
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
        """Should only be called once - sort of like a lazy property"""
        print("[INFO] Initializing AttnPacker Model")
        if exists(self.model):
            return self.model
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
        model_path = os.path.join(self.resource_root, "models", f"{self.model_name}.tar")
        checkpoint = torch.load(model_path, map_location="cpu")
        try:
            self.model.load_state_dict(checkpoint["model"], strict=True)
        except:  # noqa
            # print(traceback.format_exc())
            self.model.load_state_dict(checkpoint["model"], strict=False)
        self.model = self.model.eval()
        self.model.basis_dir = NAME_TO_BASIS_CACHE[self.model_name]

    def get_model(self, device="cpu"):
        self._init_model()
        return self.model

    def load_example(self, pdb_path, fasta_path=None, seq_mask=None, dihedral_mask=None, protein=None):
        protein = (
            Protein.FromPDBAndSeq(
                pdb_path=pdb_path,
                seq=fasta_path,
                atom_tys=pc.ALL_ATOMS,
                missing_seq=fasta_path is None,
                load_ss=False,
            )
            if (not exists(protein))
            else protein
        )
        protein, _ = impute_cb(protein, protein)
        extra = _augment(protein, protein)
        protein = set_canonical_coords_n_masks(protein)

        return ModelInput(
            decoy=protein,
            native=protein.clone(),
            input_features=self.feat_gen.generate_features(
                protein,
                extra=extra,
                seq_mask=seq_mask,
                dihedral_mask=dihedral_mask,
            ),
            extra=extra,
        ).to(self.device)

    def to(self, device):
        self.model = self.get_model().to(device)
        return self

    @property
    def device(self):
        return next(self.model.parameters()).device

    def infer(self, pdb_path, fasta_path=None, seq_mask=None, dihedral_mask=None, format=True, chunk_size: int = 1e9):
        model = self.get_model()
        with torch.no_grad():
            model_input = self.load_example(pdb_path, fasta_path, seq_mask=seq_mask, dihedral_mask=dihedral_mask)
            if len(model_input.decoy) <= chunk_size:
                model_output = model(model_input, use_cycles=1)
            else:
                model_output = chunk_inference(model, model_input, max_len=chunk_size, device=self.device)
            if format:
                return format_prediction(model, model_input, model_output)
            return model_output
