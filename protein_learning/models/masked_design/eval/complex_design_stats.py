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
    per_residue_neighbor_counts,
)
from protein_learning.common.data.data_types.model_output import ModelOutput
from protein_learning.common.data.data_types.protein import Protein
from protein_learning.common.global_config import GlobalConfig
from protein_learning.common.helpers import exists, res_labels_to_seq
from protein_learning.models.utils.model_eval import (
    ModelStats,
    to_npy,
)
from protein_learning.networks.structure_net.structure_netv2 import StructureNetV2
from protein_learning.protein_utils.align.kabsch_align import kabsch_align


class ComplexDesignStats(ModelStats):
    """Caclulate and store statistics for design model"""

    def __init__(
            self,
            config: GlobalConfig,
            model: StructureNetV2,
            max_samples: int = -1,
            pdb_dir: Optional[str] = None,
            raise_exceptions: bool = True,
    ):
        super(ComplexDesignStats, self).__init__(
            model=model,
            config=config,
            max_samples=max_samples,
            use_custom_forward=False,
            raise_exceptions=raise_exceptions,
        )
        self.pdb_dir = pdb_dir
        self.save_pdbs = exists(pdb_dir)

    def _init_data(self):
        return dict(seqs=[],
                    perplexity=[],
                    recovery=[],
                    names=[],
                    true_labels=[],
                    input_seqs=[],
                    global_masks=[],
                    pred_labels=[],
                    partitions=[],
                    seq_masks=[],
                    feat_masks=[],
                    pred_coords=[],
                    native_coords=[],
                    )

    def log_stats(self, model_out: ModelOutput, model: StructureNetV2):
        """Log statistics for single input sample"""
        scalar_out = model_out.scalar_output
        if "nsr" in model.loss_fn.loss_fns:
            nsr_loss_fn = model.loss_fn.loss_fns["nsr"].get_predicted_logits
        else:
            nsr_loss_fn = lambda x: torch.randn(*x.shape[:2], 20).to(x.device)
        mask = model_out.valid_residue_mask
        mask = mask.unsqueeze(0) if mask.ndim == 1 else mask
        true_labels = model_out.model_input.native_seq_enc[mask]
        labels = to_npy(true_labels).squeeze()
        _logits = nsr_loss_fn(scalar_out)
        logits = _logits[mask]
        _pred_labels = torch.argmax(_logits, dim=-1)
        pred_labels = _pred_labels[mask]
        assert true_labels.ndim == pred_labels.ndim

        perp = to_npy(calculate_perplexity(pred_aa_logits=logits, true_labels=true_labels))
        rec = to_npy(calculate_sequence_identity(pred_labels, true_labels))
        input_seq = model_out.model_input.input_features["res_ty"].get_encoded_data()
        nbr_counts = per_residue_neighbor_counts(model_out.native_protein["CA"])
        native, decoy = model_out.native_protein, model_out.decoy_protein
        print(f"target: {native.name}, rec : {np.round(rec, 3)}, len : {len(native)}")
        self.data["names"].append(native.chain_names)
        self.data["input_seqs"].append(to_npy(input_seq))
        self.data["seqs"].append(labels)
        self.data["perplexity"].append(perp)
        self.data["recovery"].append(rec)
        self.data["true_labels"].append(to_npy(true_labels.squeeze()))
        # self.data["nbr_counts"].append(to_npy(nbr_counts))
        self.data["global_masks"].append(mask)
        self.data["pred_labels"].append(pred_labels)
        self.data["seq_masks"].append(input_seq >= 20)
        self.data["feat_masks"].append(model_out.model_input.input_features.masks[1])
        self.data["partitions"].append(model_out.decoy_protein.chain_indices)
        self.data["pred_coords"].append(to_npy(model_out.predicted_coords))
        self.data["native_coords"].append(to_npy(model_out.native_protein.atom_coords))


        if self.save_pdbs:
            pred_protein = make_predicted_protein(model_out, _pred_labels)
            chain_names = model_out.decoy_protein.chain_names
            atom_tys = ["CA"]
            pred_protein.to_pdb(
                path=os.path.join(self.pdb_dir, "decoy_" + "_".join(chain_names) + ".pdb"),
                atom_tys=atom_tys,
            )
            model_out.native_protein.to_pdb(
                path=os.path.join(self.pdb_dir, "native_" + "_".join(chain_names) + ".pdb"),
                atom_tys=atom_tys,
                coords=model_out.native_protein.atom_coords.squeeze(),
                chain_indices=model_out.decoy_protein.chain_indices
            )


def make_predicted_protein(model_out: ModelOutput, pred_seq_ids: Tensor) -> Protein:
    """Constructs predicted protein"""
    pred_coords = model_out.predicted_coords.detach().cpu().squeeze()  # n x 4 x 3
    native_ca = model_out.native_protein.get_atom_coords("CA")  # n x 3
    pred_ca = pred_coords[:, 1]
    _, pred_ca = kabsch_align(native_ca.unsqueeze(0),
                              pred_ca.to(native_ca.device).unsqueeze(0)
                              )
    pred_coords[:, 1] = pred_ca.to(pred_coords.device)

    return Protein(
        atom_coords=pred_coords,
        atom_masks=model_out.native_protein.atom_masks,
        atom_tys=model_out.native_protein.atom_tys,
        seq=res_labels_to_seq(torch.clamp_max(pred_seq_ids, 19)),
        name=model_out.native_protein.name,
        res_ids=model_out.native_protein.res_ids,
        chain_ids=model_out.decoy_protein.chain_ids,
        chain_indices=model_out.decoy_protein.chain_indices,
    )
