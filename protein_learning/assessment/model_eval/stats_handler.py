"""Get stats from ModelOut object"""
from typing import Dict, Optional

import torch
from torch import Tensor

from protein_learning.assessment.metrics import (
    calculate_perplexity,
    get_inter_chain_contacts,
    compute_coord_rmsd,
    compute_coord_lddt,
    compute_interface_rmsd,
    get_contact_recall,
    get_contact_precision,
)
from protein_learning.common.data.data_types.model_output import ModelOutput
from protein_learning.common.protein_constants import INDEX_TO_AA_ONE, AA_TO_INDEX
from protein_learning.networks.loss.loss_fn import DefaultLossFunc, LossTy
from protein_learning.networks.loss.residue_loss import SequenceRecoveryLossNet
from protein_learning.networks.loss.coord_loss import ViolationLoss
from protein_learning.common.helpers import exists
from protein_learning.protein_utils.align.kabsch_align import kabsch_align


class LossHandler:
    """
    (1) NSR
        (logits, acc, perp)
    (2) Violation
        (bond-len, bond-angle, vdw-repulsive)
    (3) RMSD
        (interface, complex, chain)
    (4) Contacts
        (precision and recall: interface, chain)
    (5) LDDT
        - pLDDT vs LDDT
        - complex pLDDT
        - interface LDDT
    (6) Aligned Error
        -
    (7) Predicted PW Distance
    """

    def __init__(self, loss_fn: DefaultLossFunc):
        self.loss_fn = loss_fn

    @staticmethod
    def get_interface_mask(model_out: ModelOutput, contact_thresh: float, mask: Optional[Tensor]) -> Tensor:
        """Mask indicating which residues are on interface"""
        chain_indices = model_out.decoy_protein.chain_indices
        assert len(chain_indices) == 2
        contacts = get_inter_chain_contacts(
            model_out.native["CA"],
            chain_indices,
            atom_mask=model_out.valid_residue_mask,
            contact_thresh=contact_thresh
        )
        if exists(mask):
            contacts[~mask, :] = False
            contacts[:, ~mask] = False
        return torch.any(contacts, dim=-1)

    def get_contact_data(self, model_out: ModelOutput):
        """
        (4) Contacts
        (precision and recall: interface, chain)
        """
        out = {}
        pred_ca, native_ca, ca_mask = model_out.get_pred_and_native_coords_and_mask(atom_tys=["CA"])
        pred_ca, native_ca, ca_mask = map(lambda x: x.squeeze(), (pred_ca, native_ca, ca_mask))
        chain_indices = model_out.decoy_protein.chain_indices
        out["chain-contact-precision"] = {i: {} for i in range(len(chain_indices))}
        out["chain-contact-recall"] = {i: {} for i in range(len(chain_indices))}
        for c_idx, indices in enumerate(chain_indices):
            for sep in [(5, 1e5), (11, 1e5), (23, 1e5)]:
                if len(indices) > (sep[0] * 2):
                    _msk = ca_mask[indices]
                    out["chain-contact-precision"][c_idx][sep[0] + 1] = get_contact_precision(
                        pred_coords=pred_ca[indices][_msk],
                        actual_coords=native_ca[indices][_msk],
                        min_sep=sep[0],
                        max_sep=int(sep[1]),
                        threshold=8
                    )
                    out["chain-contact-recall"][c_idx][sep[0] + 1] = get_contact_recall(
                        pred_coords=pred_ca[indices][_msk],
                        actual_coords=native_ca[indices][_msk],
                        min_sep=sep[0],
                        max_sep=int(sep[1]),
                        threshold=8
                    )

        if len(chain_indices) > 1:
            out["interface-contact-precision"] = {}
            out["interface-contact-recall"] = {}
            out["true_interface_contacts"] = {}
            out["n_contacts"] = {}
            for threshold in [6, 8, 12]:
                interface_mask = self.get_interface_mask(model_out, contact_thresh=threshold, mask=ca_mask)
                out["true_interface_contacts"][threshold] = get_inter_chain_contacts(
                    coords=native_ca,
                    partition=chain_indices,
                    atom_mask=ca_mask,
                    contact_thresh=threshold,
                )
                n_contacts = torch.sum(out["true_interface_contacts"][threshold].float())
                out["n_contacts"][threshold] = n_contacts
                out["interface-contact-precision"][threshold] = get_contact_precision(
                    pred_coords=pred_ca,
                    actual_coords=native_ca,
                    min_sep=0,
                    max_sep=10000,
                    partition=chain_indices,
                    threshold=threshold,
                ) if n_contacts > 0 else 0
                out["interface-contact-recall"][threshold] = get_contact_recall(
                    pred_coords=pred_ca[interface_mask],
                    actual_coords=native_ca[interface_mask],
                    min_sep=0,
                    max_sep=10000,
                    partition=chain_indices,
                    threshold=threshold,
                ) if n_contacts > 0 else 0

        return out

    @staticmethod
    def _get_dist_data(model_out, loss_fn):
        # (b,n,n,# atom pairs,d)
        logits, masks = loss_fn.forward_from_output(model_out, return_logits=True)
        probs = torch.softmax(logits, dim=-1)
        bins = loss_fn.bins(logits.device)
        expected = torch.sum(probs * bins, dim=-1)

        return dict(
            expected={
                aty: expected[..., i] for i, aty in enumerate(loss_fn.atom_tys[:1])
            },
            probs={
                aty: probs[..., i, :] for i, aty in enumerate(loss_fn.atom_tys[:1])
            },
            masks={
                aty: masks[..., i] for i, aty in enumerate(loss_fn.atom_tys[:1])
            },
            bins=bins,
        )

    def get_mae_data(self, model_out: ModelOutput) -> Dict:
        """Get Mean Aligned Error data"""
        pae_loss = self.loss_fn[LossTy.PAE]
        return self._get_dist_data(model_out, pae_loss)

    def get_predicted_dist_data(self, model_out: ModelOutput) -> Dict:
        """Get predicted distance data"""
        dist_loss = self.loss_fn[LossTy.PAIR_DIST]
        return self._get_dist_data(model_out, dist_loss)

    def get_lddt_data(self, model_out: ModelOutput) -> Dict:
        """
        - pLDDT vs LDDT
        - complex pLDDT
        - interface LDDT
        """
        out = {}
        pred_ca, native_ca, ca_mask = model_out.get_pred_and_native_coords_and_mask(atom_tys=["CA"])
        pred_ca, native_ca, ca_mask = map(lambda x: x.squeeze(), (pred_ca, native_ca, ca_mask))
        chain_indices = model_out.decoy_protein.chain_indices

        pair_mask = torch.einsum("i,j->ij", ca_mask, ca_mask)

        #out["plddt"] = compute_coord_lddt(
        #    predicted_coords=pred_ca,
        #    actual_coords=native_ca,
        #    cutoff=15.,
        #    per_residue=True,
        #    pair_mask=pair_mask
        #)
        pred_plddt = None
        if exists(self.loss_fn) and LossTy.PLDDT in self.loss_fn:
            plddt_net = self.loss_fn[LossTy.PLDDT]
            pred_plddt = plddt_net.get_expected_value(model_out.scalar_output)
            out["pred-plddt"] = pred_plddt
            out["plddt"], out["plddt-mask"] = plddt_net.forward_from_output(model_out,return_true_prop=True)[1:]


        if len(chain_indices) > 1:
            out["interface-plddt"], out["pred-interface-plddt"] = {}, {}
            for threshold in [8, 10]:
                interface_mask = self.get_interface_mask(model_out, contact_thresh=threshold, mask=ca_mask)
                out["interface-plddt"][threshold] = compute_coord_lddt(
                    predicted_coords=pred_ca[interface_mask],
                    actual_coords=native_ca[interface_mask],
                    cutoff=15,
                )
                if exists(self.loss_fn) and LossTy.PLDDT in self.loss_fn:
                    out["pred-interface-plddt"][threshold] = pred_plddt[interface_mask]
            out["per-chain"], out["pred-per-chain"] = {}, {}
            for c_idx, indices in enumerate(chain_indices):
                pair_mask = torch.einsum("i,j->ij", ca_mask[indices], ca_mask[indices])
                out["per-chain"][c_idx] = compute_coord_lddt(
                    predicted_coords=pred_ca[indices],
                    actual_coords=native_ca[indices],
                    cutoff=15,
                    pair_mask=pair_mask
                )
                if exists(self.loss_fn) and LossTy.PLDDT in self.loss_fn:
                    out["pred-per-chain"][c_idx] = pred_plddt[indices]
        return out

    def get_nsr_data(self, model_out: ModelOutput) -> Dict:
        """Native Sequence Recovery Data"""
        loss_fn: SequenceRecoveryLossNet = self.loss_fn[LossTy.NSR]
        logits = loss_fn.get_predicted_logits(residue_feats=model_out.scalar_output).squeeze()
        pred_labels = torch.argmax(logits, dim=-1)
        pred_seq = "".join([
            INDEX_TO_AA_ONE[label.item()] if label.item() in INDEX_TO_AA_ONE else "-"
            for label in pred_labels
        ])
        native_seq = model_out.native_protein.seq
        true_labels = torch.tensor([
            AA_TO_INDEX[s] if s in AA_TO_INDEX else 20 for i, s in enumerate(native_seq)
        ], device=logits.device).long()

        correct_preds = pred_labels == true_labels
        acc = correct_preds[correct_preds].numel() / max(1, correct_preds.numel())  # noqa
        seq_mask = model_out.model_input.input_features.seq_mask.to(correct_preds.device)  # noqa
        correct_preds = correct_preds[seq_mask]  # noqa
        if correct_preds.numel() > 0:
            masked_acc = correct_preds[correct_preds].numel() / max(1, correct_preds.numel())  # noqa
        else:
            masked_acc = 0

        perp = calculate_perplexity(logits, true_labels)
        masked_perp = calculate_perplexity(logits[seq_mask], true_labels[seq_mask])
        return dict(
            seq_mask=seq_mask,
            true_labels=true_labels,
            pred_labels=pred_labels,
            pred_seq=pred_seq,
            true_seq=model_out.native_protein.seq,
            logits=logits,
            acc=acc,
            masked_acc=masked_acc,
            perplexity=perp,
            masked_perplexity=masked_perp,
        )

    def get_violation_data(self, model_out: ModelOutput) -> Dict:
        """Violation data for native and redicted proteins"""
        if exists(self.loss_fn):
            viol_loss = self.loss_fn[LossTy.VIOL]
        else:
            viol_loss = ViolationLoss(bond_len_wt=1, bond_angle_wt=1, vdw_mean=False)

        pred_viols = viol_loss.forward_from_output(
            output=model_out,
            baseline=False,
            vdw_reduce_mean=False,
            weighted=False,
        )
        true_viols = viol_loss.forward_from_output(
            output=model_out,
            baseline=True,
            vdw_reduce_mean=False,
            weighted=False,
        )
        pred_viols = {f"pred-{k}": v for k, v in pred_viols.items()}
        pred_viols.update({f"true-{k}": v for k, v in true_viols.items()})
        return pred_viols

    def get_rmsd_data(self, model_out: ModelOutput) -> Dict:
        """
        - decoy to native chain rmsd
        - Intra-chain RMSD
        - Complex RMSD
        - Interface RMSD
        """
        out = {}
        pred_ca, native_ca, ca_mask = model_out.get_pred_and_native_coords_and_mask(atom_tys=["CA"])
        pred_ca, native_ca, ca_mask = map(lambda x: x.squeeze(), (pred_ca, native_ca, ca_mask))
        decoy_ca = model_out.decoy_protein["CA"].squeeze()
        masked_pred_ca, masked_native_ca, masked_decoy_ca = map(lambda x: x[ca_mask], (pred_ca, native_ca, decoy_ca))

        # complex rmsd
        out['complex'] = compute_coord_rmsd(
            predicted_coords=masked_pred_ca,
            actual_coords=masked_native_ca,
            atom_mask=None,
            align=True,
        )
        if exists(model_out.model_input.input_features):
            feat_mask = model_out.model_input.input_features.feat_mask
            if feat_mask is not None:
                if torch.any(feat_mask):
                    msk = feat_mask & ca_mask
                    out["masked"] = compute_coord_rmsd(
                        predicted_coords=pred_ca[msk],
                        actual_coords=native_ca[msk],
                        atom_mask=None,
                        align=True,
                    )

        chain_indices = model_out.decoy_protein.chain_indices
        # interface RMSD at thresholds 6, 8, 10 ,12
        if len(chain_indices) > 1:
            out["interface"] = {
                thresh: compute_interface_rmsd(
                    predicted_coords=pred_ca,
                    actual_coords=native_ca,
                    chain_indices=chain_indices,
                    atom_mask=ca_mask,
                    contact_thresh=thresh
                )
                for thresh in [6, 8, 10, 12]
            }
            out["interface_mask"] = {
                t: self.get_interface_mask(model_out, contact_thresh=t, mask=ca_mask)
                for t in [6, 8, 10, 12]
            }
        per_chain = {i: {} for i in range(len(chain_indices))}
        # per-chain
        for c_idx, indices in enumerate(chain_indices):
            # pred. to native rmsd
            per_chain[c_idx][f'pred-to-native'] = compute_coord_rmsd(
                predicted_coords=pred_ca[indices],
                actual_coords=native_ca[indices],
                atom_mask=ca_mask[indices],
                align=True,
            )
            # native to decoy rmsd
            per_chain[c_idx][f'input-to-native'] = compute_coord_rmsd(
                predicted_coords=decoy_ca[indices],
                actual_coords=native_ca[indices],
                atom_mask=ca_mask[indices],
                align=True,
            )
            # pred. to decoy rmsd
            per_chain[c_idx][f'pred-to-input'] = compute_coord_rmsd(
                predicted_coords=pred_ca[indices],
                actual_coords=decoy_ca[indices],
                atom_mask=ca_mask[indices],
                align=True,
            )
        out["per-chain"] = per_chain

        if model_out.decoy_protein.is_antibody:
            out["cdr"] = self.get_ab_stats(model_out)
        return out

    def get_ab_stats(self, model_out: ModelOutput):
        assert model_out.native.is_antibody
        # get masks for cdr regions
        cdr_masks = []
        ab_chain = model_out.native.chain_indices[0]
        res_mask = model_out.valid_residue_mask.squeeze()[ab_chain]
        full_mask = torch.zeros(len(ab_chain), device=ab_chain.device)
        for (s, e) in model_out.decoy_protein.cdrs["heavy"]:
            msk = torch.zeros(len(ab_chain), device=ab_chain.device)
            msk[s:e + 1] = 1
            full_mask[s:e + 1] = 1
            cdr_masks.append(msk.bool() & res_mask)
        full_mask = full_mask.bool()
        # align antibody chain coords
        out = {}
        for mask, name in ((res_mask, "framework"), (full_mask, "cdr")):
            pred_coords = model_out.predicted_coords[..., 1, :].squeeze()[ab_chain].unsqueeze(0)
            native_coords = model_out.native["CA"].squeeze()[ab_chain].unsqueeze(0)
            aln_pred, aln_native = kabsch_align(align_to=pred_coords, align_from=native_coords,
                                                mask=mask.unsqueeze(0))
            aln_pred, aln_native = map(lambda x: x.squeeze(), (aln_pred, aln_native))
            devs = torch.sum(torch.square(aln_pred - aln_native), dim=-1).squeeze()
            cdr_rmsd = []
            for i in range(3):
                cdr_rmsd.append(torch.sqrt(torch.mean(devs[cdr_masks[i]])))
            out[name] = cdr_rmsd
        out["masks"] = cdr_masks
        return out
