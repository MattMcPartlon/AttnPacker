"""Compute Design Model statistics
"""
import os
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from protein_learning.assessment.metrics import (
    calculate_sequence_identity,
    calculate_perplexity,
    calculate_unnormalized_confusion,
    calculate_average_entropy,
    per_residue_neighbor_counts,
)
from protein_learning.common.data.data_types.model_output import ModelOutput
from protein_learning.common.data.data_types.model_input import ModelInput
from protein_learning.common.data.data_types.protein import Protein
from protein_learning.common.global_config import GlobalConfig
from protein_learning.common.helpers import exists, res_labels_to_seq
from protein_learning.common.protein_constants import (
    INDEX_TO_AA_ONE, AA_TO_SC_ATOMS, ALL_ATOMS, ALL_ATOM_POSNS
)
from protein_learning.features.feature_config import FeatureName
from protein_learning.features.feature_utils import res_ty_encoding
from protein_learning.models.design.design_model import Designer
from protein_learning.models.model_abc.protein_model import ProteinModel
from protein_learning.models.utils.model_eval import (
    ModelStats,
    to_npy,
)
from protein_learning.networks.loss.residue_loss import SequenceRecoveryLossNet


class DesignStats(ModelStats):
    """Caclulate and store statistics for design model"""

    def __init__(
            self,
            config: GlobalConfig,
            model: ProteinModel,
            max_samples: int = -1,
            autoregressive: bool = False,
            pdb_dir: Optional[str] = None,
            ignore_mask: bool = False,
            model_path: Optional[str]=None,
    ):
        super(DesignStats, self).__init__(
            model=model, config=config,
            max_samples=max_samples,
            use_custom_forward=autoregressive,
            model_path=model_path,
        )
        self.pdb_dir = pdb_dir
        self.save_pdbs = exists(pdb_dir)
        self.ignore_mask = ignore_mask

    def _init_data(self):
        return dict(seqs=[], perplexity=[], entropy=[], confusion=[], recovery=[], names=[],
                    logits=[], true_labels=[], input_seqs=[], nbr_counts=[], global_masks=[], pred_labels=[],
                    seq_masks=[])

    def log_stats(self, model_out: ModelOutput, model: ProteinModel):
        """Log statistics for single input sample"""
        # nsr_feats_normed, nsr_feats_unnormed = model_out.extra["nsr_scalar"], model_out.scalar_output
        nsr_loss_fn = model.loss_fn.loss_fns["nsr"]
        """
        _nsr_feats = None
        best_rec = 0
        for nsr_feats in [nsr_feats_normed, nsr_feats_unnormed]:
            mask = model_out.valid_residue_mask
            mask = mask.unsqueeze(0) if mask.ndim == 1 else mask
            true_labels = model_out.model_input.native_seq_enc[mask]
            logits = nsr_loss_fn.get_predicted_logits(nsr_feats)
            pred_labels = torch.argmax(logits, dim=-1)[mask]
            rec = to_npy(calculate_sequence_identity(pred_labels, true_labels))
            if rec > best_rec:
                best_rec = rec
                _nsr_feats = nsr_feats

        nsr_feats = _nsr_feats
        """
        nsr_feats = model_out.extra["nsr_scalar"]
        mask = model_out.valid_residue_mask
        mask = mask.unsqueeze(0) if mask.ndim == 1 else mask
        if self.ignore_mask:
            mask = torch.ones_like(mask).bool()
        true_labels = model_out.model_input.native_seq_enc[mask]
        labels = to_npy(true_labels).squeeze()
        _logits = nsr_loss_fn.get_predicted_logits(nsr_feats)
        logits = _logits[mask]
        _pred_labels = torch.argmax(_logits, dim=-1)
        pred_labels = _pred_labels[mask]
        assert true_labels.ndim == pred_labels.ndim

        perp = to_npy(calculate_perplexity(pred_aa_logits=logits, true_labels=true_labels))
        ent = to_npy(calculate_average_entropy(pred_aa_logits=logits))
        rec = to_npy(calculate_sequence_identity(pred_labels, true_labels))
        conf = to_npy(calculate_unnormalized_confusion(pred_labels=pred_labels, true_labels=true_labels))
        input_seq = model_out.model_input.input_features["res_ty"].get_encoded_data()
        nbr_counts = per_residue_neighbor_counts(model_out.native_protein["CA"])
        native, decoy = model_out.native_protein, model_out.decoy_protein
        print(f"target: {native.name}, rec : {np.round(rec, 3)}, len : {len(native)}")
        self.data["names"].append(native.name)
        self.data["input_seqs"].append(to_npy(input_seq))
        self.data["seqs"].append(labels)
        self.data["perplexity"].append(perp)
        self.data["entropy"].append(ent)
        self.data["recovery"].append(rec)
        self.data["confusion"].append(conf)
        self.data["logits"].append(to_npy(logits))
        self.data["true_labels"].append(to_npy(true_labels.squeeze()))
        self.data["nbr_counts"].append(to_npy(nbr_counts))
        self.data["global_masks"].append(mask)
        self.data["pred_labels"].append(pred_labels)
        self.data["seq_masks"].append(input_seq >= 20)

        if self.save_pdbs:
            pred_protein = make_predicted_protein(model_out, _pred_labels)
            pred_protein.to_pdb(path=os.path.join(self.pdb_dir, model_out.native_protein.name + ".pdb"))

    def custom_forward(self, model: Designer, sample: ModelInput, decode_frac=0.1) -> ModelOutput:
        """Autoregressive-style forward"""
        print("decoding", sample.decoy.name)
        return ar_decode(model, sample, decode_frac=decode_frac, device=self.config.device)


def make_predicted_protein(model_out: ModelOutput, pred_seq_ids: Tensor) -> Protein:
    """Constructs predicted protein"""
    pred_bb_coords = model_out.native_protein.bb_atom_coords.detach()
    pred_sc = model_out.predicted_coords.detach().squeeze()
    pred_coords = torch.cat((pred_bb_coords, pred_sc), dim=-2)
    seq = res_labels_to_seq(pred_seq_ids)
    mask = make_atom_masks(
        seq=seq, res_mask=model_out.native_protein.valid_residue_mask.squeeze(),
        bb_mask=model_out.native_protein.bb_atom_mask.squeeze(),
    )
    return Protein(
        atom_coords=pred_coords,
        atom_masks=mask.to(pred_coords.device),
        atom_tys=model_out.native_protein.atom_tys,
        seq=seq,
        name=model_out.native_protein.name,
        res_ids=model_out.native_protein.res_ids,
    )


def make_atom_masks(seq, res_mask, bb_mask):
    mask = torch.zeros(len(seq), len(ALL_ATOMS)).to(bb_mask.device).float()
    mask[:, :4] = bb_mask.float()

    for i, aa in enumerate(seq):
        if res_mask[i]:
            for atom_ty in AA_TO_SC_ATOMS[aa]:
                mask[i, ALL_ATOM_POSNS[atom_ty]] = 1
    return mask.bool()


def ar_decode(model: Designer, sample: ModelInput, decode_frac: float = 0.05, device="cpu") -> ModelOutput:
    """Auto-regressive decoder"""
    nsr_loss: SequenceRecoveryLossNet = model.loss_fn["nsr"]
    sample = sample.to(device)
    n = sample.n_residue(decoy=True)
    unpredicted_mask = torch.ones(n, device=device).bool()
    res_indices = torch.arange(n, device=device)
    predicted_sequence = [s for s in "".join(["-"] * n)]
    for i in range(int(1 / decode_frac)):
        output: ModelOutput = model(sample)
        predicted_logits = nsr_loss.get_predicted_logits(residue_feats=output.nsr_scalar)
        predicted_probs = torch.exp(torch.log_softmax(predicted_logits, dim=-1)).squeeze()

        # set maximum likelihood labels to the predicted identity
        masked_probs = predicted_probs[unpredicted_mask]
        masked_indices = res_indices[unpredicted_mask]
        max_values, max_indices = torch.max(masked_probs, dim=-1)

        # filter out by top decode_frac probability
        top_k = min(max_values.numel(), int(len(predicted_sequence) * decode_frac))
        order = torch.argsort(max_values, descending=True)[:top_k]
        print(f"top k {top_k}, seq len {n}")
        print("max vals :", max_values[order])
        seq_indices = masked_indices[order].cpu().numpy()
        aa_indices = max_indices[order].cpu().numpy()
        assert torch.all(unpredicted_mask[seq_indices])
        unpredicted_mask[seq_indices] = False
        for masked_idx, seq_idx in enumerate(seq_indices):
            predicted_sequence[seq_idx] = INDEX_TO_AA_ONE[aa_indices[masked_idx]]

        # update sample input features
        new_seq_feature = res_ty_encoding(seq="".join(predicted_sequence))
        sample.input_features[FeatureName.RES_TY.value] = new_seq_feature
        sample = sample.to(device)
        print(predicted_sequence)

    return model(sample)
