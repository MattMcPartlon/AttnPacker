"""Reconstruction Loss Functi0n"""
import random
from typing import Optional, List, Tuple, Union

import numpy as np
import torch
from einops import repeat, rearrange  # noqa
from torch import nn, Tensor

from protein_learning.common.data.data_types.model_loss import ModelLoss
from protein_learning.common.data.data_types.model_output import ModelOutput
from protein_learning.common.global_constants import get_logger
from protein_learning.common.helpers import exists, default, get_max_val, get_min_val
from protein_learning.features.masking.masking_utils import (
    get_chain_masks
)
from protein_learning.networks.loss.loss_config import LossConfig, LossTy

logger = get_logger(__name__)


class DefaultLossFunc(nn.Module):  # noqa
    """Reconstruction Loss function"""

    def __init__(self, config: LossConfig):
        super(DefaultLossFunc, self).__init__()
        self.config = config
        self.loss_fns, self.loss_wts = config.get_losses_n_weights()
        self.unmask_wt = 1 / config.mask_rel_wt
        self.viol_wt_dict = dict(
            bond_angle=config.bond_angle_wt,
            bond_len=config.bond_len_wt,
            vdw=config.vdw_wt,
        )

    def __getitem__(self, item: Union[str, LossTy]):
        if isinstance(item, LossTy):
            item = item.value
        assert item in self.loss_fns
        return self.loss_fns[item]

    def __contains__(self, item):
        if isinstance(item, LossTy):
            item = item.value
        return item in self.loss_fns

    def update_viol_wt(self, batch_idx: int) -> None:
        """Update violation loss weight (in ramped case)"""
        if self.config.ramp_viol_wt:
            min_wt, max_wt, step, step_every = self.config.viol_schedule
            if max_wt > 0 and step > 0 and step_every > 0:
                x = (batch_idx + 1) // step_every
                self.loss_wts["violation"] = min(max_wt, step * x + min_wt)

    def forward(
            self,
            model_out: ModelOutput,
            compute_zero_wt_loss,
            batch_idx: int,
    ):
        """Compute model loss"""
        self.update_viol_wt(batch_idx)
        is_homodimer = model_out.decoy_protein.is_homodimer
        note = f"complex : {model_out.native_protein.is_complex}, "
        feat_mask = model_out.model_input.input_features.feat_mask
        seq_mask = model_out.model_input.input_features.seq_mask
        note += f"num masked features {feat_mask[feat_mask].numel()} "
        note += f"num masked seq {seq_mask[seq_mask].numel()}\n"
        fape_clamp = (0, 10 if random.random() < 0.9 else 1e2)
        native_ptn = model_out.native_protein

        loss = self._forward(
            model_out=model_out,
            fape_clamp=fape_clamp,
            compute_zero_wt_loss=compute_zero_wt_loss,
            include_violation_loss=batch_idx >= self.config.include_viol_after >= 0,
            batch_idx=batch_idx,
        )

        alt_loss = None
        if model_out.decoy_protein.is_homodimer:
            alt_native = model_out.native.swap_chains()
            print([len(x) for x in model_out.decoy_protein.chain_indices])
            model_out.set_native_protein(alt_native)

            alt_loss = self._forward(
                model_out=model_out,
                fape_clamp=fape_clamp,
                compute_zero_wt_loss=compute_zero_wt_loss,
                include_violation_loss=batch_idx >= self.config.include_viol_after >= 0,
                batch_idx=batch_idx
            )

        if exists(alt_loss):
            alt_val, val = map(lambda x: np.round(x, 3), (alt_loss.float_value, loss.float_value))
            if val < alt_val:
                model_out.set_native_protein(native_ptn)
            loss = min(alt_loss, loss)
            note += f", homodimer : {is_homodimer}, val: {val}, alt_val : {alt_val}"
        loss.add_note(note)
        return loss

    def _forward(
            self,
            model_out: ModelOutput,
            fape_clamp: Tuple,
            compute_zero_wt_loss: bool = False,
            include_violation_loss: bool = False,
            batch_idx: int = -1,
    ):
        """Compute model loss (helper)"""
        # get the chain ids
        chain_indices = model_out.decoy_protein.chain_indices
        true_rigids = model_out.native_protein.rigids
        native_seq_enc = model_out.native.seq_encoding.unsqueeze(0)
        # collect all loss information
        n_res = model_out.seq_len
        chain_masks = get_chain_masks(n_res, chain_indices=chain_indices)
        model_loss = ModelLoss(seq_len=model_out.seq_len, pdb=model_out.native_protein.name)

        infeats, device = model_out.model_input.input_features, native_seq_enc.device
        _seq_mask, _feat_mask = infeats.seq_mask.to(device), infeats.seq_mask.to(device)

        """
        if compute_zero_wt_loss:
            try:
                pca, nca, msk = model_out.get_native_and_pred_coords_and_mask(atom_tys=["CA"], align_by_kabsch=True)
                dev = torch.sum(torch.square(pca - nca), dim=-1)
                rmsd = torch.sqrt(torch.mean(dev[:, msk.squeeze()]))
                fm = _feat_mask & msk.squeeze()
                masked_rmsd = torch.sqrt(torch.mean(dev[:, fm]))
                model_loss.add_loss(rmsd, loss_weight=0, loss_name="rmsd")
                model_loss.add_loss(masked_rmsd, loss_weight=0, loss_name="masked-rmsd")
            except Exception as e:
                print(f"[ERROR] computing RMSD : {e}")
                pass
        """

        for loss_name, loss_fn in self.loss_fns.items():
            intra_clamp, inter_clamp, pair_mask = None, None, None
            pair_loss, res_loss, intra_loss_scale, inter_loss_scale = None, None, 1, 1
            reshape_fn = lambda x: x

            loss_wt = self.loss_wts[loss_name]
            if loss_wt == 0 and not compute_zero_wt_loss:
                continue
            logger.info(f"computing {loss_name} loss")

            """Pairwise Loss"""
            # FAPE
            if loss_name == LossTy.FAPE.value:
                pair_loss = loss_fn.forward_from_output(
                    model_out, reduce=False, atom_tys=self.config.fape_atom_tys,
                    mask_fill=-1,
                )
                pair_mask = pair_loss >= 0
                m, inter_clamp = len(self.config.fape_atom_tys), (0, 30)
                intra_clamp = fape_clamp
                reshape_fn = lambda x, _m=m: repeat(x, "i j -> () i (j m)", m=_m)
                intra_loss_scale, inter_loss_scale = (1 / 10), self.config.inter_fape_scale * 1 / 15

            # Predicted Distance
            if loss_name == LossTy.PAIR_DIST.value:
                logits = None
                if "evo_pair" in model_out.extra:
                    logits = model_out.extra["evo_pair"]
                pair_loss, pair_mask = loss_fn.forward_from_output(model_out, reduce=False, pair_feats=logits)
                m = len(loss_fn.atom_tys)
                reshape_fn = lambda x, _m=m: repeat(x, "i j -> () i j m", m=_m)

            # Inverse Distance Loss
            if loss_name == LossTy.DIST_INV.value:
                pair_loss = loss_fn.forward_from_output(
                    model_out, reduce=False, atom_tys=self.config.dist_inv_atom_tys
                )
                m = len(self.config.dist_inv_atom_tys)
                reshape_fn = lambda x, _m=m: repeat(x, "i j -> () (i a) (j b)", a=_m, b=_m)

            """Per-Residue Loss"""
            # Native Seq. Recovery
            if loss_name == LossTy.NSR.value:
                if self.config.include_nsr_after <= batch_idx:
                    res_loss = loss_fn.forward_from_output(
                        model_out, reduce=False, native_seq_enc=native_seq_enc
                    )
                    res_loss[:, ~_seq_mask] = res_loss[:, ~_seq_mask] * self.unmask_wt
                    res_loss = torch.mean(res_loss)
                else:
                    continue

            if loss_name == LossTy.PAE.value:
                if self.config.include_pae_after <= batch_idx:
                    res_loss = loss_fn.forward_from_output(
                        model_out,
                        reduce=True,
                        true_rigids=true_rigids
                    )
                else:
                    continue

            if loss_name == LossTy.COM.value:
                res_loss = loss_fn.forward_from_output(model_out)

            # TM score
            if loss_name == LossTy.TM.value:  # always reduce
                res_loss = loss_fn.forward_from_output(model_out, atom_tys=self.config.tm_atom_tys)

            # Predicted pLDDT
            if loss_name == LossTy.PLDDT.value:
                if self.config.include_plddt_after <= batch_idx:
                    res_loss = loss_fn.forward_from_output(output=model_out, reduce=True)
                else:
                    continue

            # SC-RMSD
            if loss_name == LossTy.SC_RMSD.value:
                res_loss = loss_fn.forward_from_output(output=model_out, reduce=True)

            if loss_name == LossTy.RES_FAPE.value:
                res_loss = loss_fn.forward_from_output(output=model_out, reduce=True)


            # Violation
            if loss_name == LossTy.VIOL.value:
                if not include_violation_loss:
                    continue
                baseline = None
                if compute_zero_wt_loss:
                    print(f"[INFO] violation loss weight : {loss_wt}")
                    baseline = loss_fn.forward_from_output(model_out, baseline=True, weighted=False)
                tmp = loss_fn.forward_from_output(model_out, baseline=False, weighted=False)
                baseline = default(baseline, tmp)
                for k, v in tmp.items():
                    tmp_wt = loss_wt if v > 0 else 0  # will get nan in grad if loss not > 0
                    vwt = self.viol_wt_dict[k] * tmp_wt
                    model_loss.add_loss(loss=v, loss_name=f"viol-{k}", baseline=baseline[k], loss_weight=vwt)
                continue

            if exists(res_loss):
                model_loss.add_loss(loss=res_loss, loss_weight=loss_wt, loss_name=loss_name)

            if exists(pair_loss):
                if model_out.decoy_protein.is_complex:
                    intra_loss, inter_loss = get_complex_pair_loss(
                        loss=pair_loss,
                        n_res=n_res,
                        chain_indices=chain_indices,
                        reshape_fn=reshape_fn,
                        pair_mask=pair_mask,
                        chain_masks=chain_masks,
                        inter_chain_clamp=inter_clamp,
                        intra_chain_clamp=intra_clamp,
                        intra_loss_scale=intra_loss_scale,
                        inter_loss_scale=inter_loss_scale
                    )
                    model_loss.add_loss(loss=intra_loss, loss_weight=loss_wt, loss_name=f"intra-{loss_name}")
                    model_loss.add_loss(loss=inter_loss, loss_weight=loss_wt, loss_name=f"inter-{loss_name}")
                else:
                    intra_clamp = default(intra_clamp, (get_min_val(pair_loss), get_max_val(pair_loss)))
                    pair_mask = default(pair_mask, torch.ones_like(pair_loss).bool())
                    pair_loss = torch.clamp(pair_loss[pair_mask], *intra_clamp) * intra_loss_scale
                    model_loss.add_loss(loss=torch.mean(pair_loss), loss_weight=loss_wt, loss_name="intra-" + loss_name)

            assert exists(res_loss) or exists(pair_loss), f"no loss computed for {loss_name}"

        return model_loss


def get_complex_pair_loss(
        loss: Tensor,
        n_res: int,
        chain_indices: List[Tensor],
        reshape_fn=lambda x: x,
        pair_mask: Optional[Tensor] = None,
        chain_masks: Tuple[List[Tensor], List[Tensor], Tensor] = None,
        inter_chain_clamp: Optional[Tuple[float, float]] = None,
        intra_chain_clamp: Optional[Tuple[float, float]] = None,
        intra_loss_scale: float = 1,
        inter_loss_scale: float = 1,
) -> Tuple[Tensor, Tensor]:
    """Get pairwise loss for protein complex"""
    single_chain_masks, chain_pair_masks, inter_chain_mask = chain_masks if exists(chain_masks) else \
        get_chain_masks(n_res=n_res, chain_indices=chain_indices)
    mn, mx = get_min_val(loss), get_max_val(loss)
    intra_chain_clamp, inter_chain_clamp = map(lambda x: default(x, (mn, mx)),
                                               (intra_chain_clamp, inter_chain_clamp))
    intra_chain_loss, inter_chain_loss = 0, None
    total_mask = inter_chain_mask
    for chain_pair_mask in chain_pair_masks:
        total_mask = torch.logical_or(chain_pair_mask, total_mask)
        loss_mask = reshape_fn(chain_pair_mask)
        loss_mask = loss_mask & default(pair_mask, loss_mask)
        clamped_loss = torch.clamp(loss[loss_mask], *intra_chain_clamp)
        intra_chain_loss = intra_chain_loss + torch.mean(clamped_loss)  # noqa
    intra_chain_loss = intra_loss_scale * intra_chain_loss / len(chain_pair_masks)
    # get inter-chain loss
    inter_chain_mask = reshape_fn(inter_chain_mask)
    inter_chain_mask = inter_chain_mask & default(pair_mask, inter_chain_mask)
    inter_chain_loss = inter_loss_scale * torch.mean(torch.clamp(loss[inter_chain_mask], *inter_chain_clamp))

    return intra_chain_loss, inter_chain_loss
