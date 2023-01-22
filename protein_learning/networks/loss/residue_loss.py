"""Networks for Computing Residue Feature Loss"""
from abc import abstractmethod
from typing import Optional

import torch
import torch.nn.functional as F  # noqa
from einops import rearrange  # noqa
from torch import Tensor, nn

from protein_learning.assessment.metrics import compute_coord_lddt
from protein_learning.common.data.data_types.model_output import ModelOutput
from protein_learning.common.helpers import exists, default
from protein_learning.common.rigids import Rigids
from protein_learning.networks.loss.utils import FeedForward
from protein_learning.networks.loss.utils import (
    softmax_cross_entropy,
)
from protein_learning.common.protein_constants import DISTAL_ATOM_MASK_TENSOR, FUNCTIONAL_ATOM_MASK_TENSOR, ALL_ATOMS


class SequenceRecoveryLossNet(nn.Module):  # noqa
    """Loss for predicted residue type

    Output residue features are converted to residue type predictions via a
    feed-formard network. Cross Entropy loss between predicted labels and
    true labels is averaged to obtain the output.
    """

    def __init__(
        self,
        dim_in: int,
        n_labels: int = 21,
        hidden_layers: int = 2,
        pre_norm: bool = False,
    ):
        """Native Sequence Recovery Loss

        :param dim_in: residue feature dimension
        :param n_labels:number of underlying residue labels
        :param hidden_layers: number of hidden layers to use in feed forward network
        """
        super(SequenceRecoveryLossNet, self).__init__()
        self.net = FeedForward(dim_in=dim_in, dim_out=n_labels, n_hidden_layers=hidden_layers, pre_norm=pre_norm)
        self.loss_fn = softmax_cross_entropy

    def forward(
        self,
        residue_feats: Tensor,
        true_labels: Tensor,
        mask: Optional[Tensor] = None,
        reduce: bool = True,
    ) -> Tensor:
        """Compute Native Sequence Recovery Loss

        :param residue_feats: Residue features of shape (b,n,d) where d is the feature dimension
        :param true_labels: LongTensor of shape (b,n) storing residue class labels
        :param mask: residue mask of shape (b,n) indicating which residues to compute loss on.
        :param reduce : take mean of loss (iff reduce)
        :return: cross entropy loss of predicted and true labels.
        """
        assert residue_feats.shape[:2] == true_labels.shape, f"{residue_feats.shape},{true_labels.shape}"
        if exists(mask):
            assert mask.shape == true_labels.shape, f"{mask.shape},{true_labels.shape}"
        labels = torch.nn.functional.one_hot(true_labels, 21)
        logits = self.get_predicted_logits(residue_feats)
        ce = self.loss_fn(logits, labels)

        if reduce:
            return torch.mean(ce[mask]) if exists(mask) else torch.mean(ce)
        return ce.masked_fill(~mask, 0) if exists(mask) else ce

    def get_predicted_logits(self, residue_feats: Tensor) -> Tensor:
        """Get predicted logits from residue features"""
        return self.net(residue_feats)

    def predict_classes(self, residue_feats: Tensor) -> Tensor:
        """Get predicted class labels from residue features"""
        return torch.argmax(self.get_predicted_logits(residue_feats), dim=-1)

    def get_acc(self, residue_feats: Tensor, true_labels: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Get label prediction accuracy"""
        pred_labels = self.predict_classes(residue_feats)
        assert pred_labels.shape == true_labels.shape
        correct_preds = pred_labels == true_labels
        correct_preds = correct_preds[mask] if exists(mask) else correct_preds  # noqa
        return torch.mean(correct_preds.float(), dim=(-1))

    def forward_from_output(
        self, output: ModelOutput, reduce: bool = True, native_seq_enc: Optional[Tensor] = None, **kwargs  # noqa
    ) -> Tensor:
        """Run forward from ModelOutput Object"""
        if not (exists(native_seq_enc)):
            if not (hasattr(output, "native_seq_enc") or hasattr(output.model_input, "native_seq_enc")):
                raise Exception("output or input must have native_seq_enc as attribute!")
            native_seq_enc = default(
                getattr(output, "native_seq_enc", None), getattr(output.model_input, "native_seq_enc", None)
            )

        return self.forward(
            residue_feats=output.scalar_output,
            true_labels=native_seq_enc,
            mask=output.valid_residue_mask.unsqueeze(0),
            reduce=reduce,
        )


class ResiduePropertyLossNet(nn.Module):  # noqa
    """Loss for (binned) residue properties"""

    def __init__(
        self,
        dim_in,
        min_val: float,
        max_val: float,
        step: float,
        n_hidden_layers: int = 1,
        smoothing: float = 0,
    ):
        super().__init__()

        bins = torch.arange(min_val, max_val + step, step=step)

        self.net = FeedForward(dim_in=dim_in, dim_out=bins.numel() - 1, pre_norm=True, n_hidden_layers=n_hidden_layers)

        self._bins = rearrange((bins[:-1] + bins[1:]) / 2, "i -> () () i")
        self.n_bins = self._bins.numel()
        self.loss_fn = softmax_cross_entropy
        self.smoothing = smoothing

    def get_bins(self, device) -> torch.Tensor:
        """Gets the underlying bins"""
        if self._bins.device != device:
            self._bins = self._bins.to(device)
        return self._bins

    def get_logits(self, residue_feats):
        """Get predicted logits"""
        return self.net(residue_feats)

    def get_expected_value(self, residue_feats) -> torch.Tensor:
        """computes expected value for underlying property
        :return: tensor of pLDDT scores with shape (b, n)
        """
        logits = self.get_logits(residue_feats)
        return torch.sum(F.softmax(logits, dim=-1) * self.get_bins(logits.device), dim=-1)

    @abstractmethod
    def get_true_property(self, predicted_coords: Tensor, actual_coords: Tensor, **kwargs):
        """get true property"""
        pass

    def property_to_labels(self, prop: Tensor) -> Tensor:
        """Convert property values to one-hot labels"""
        nearest_bins = torch.abs(prop - self.get_bins(prop.device))
        return F.one_hot(torch.argmin(nearest_bins, dim=-1), self.n_bins)

    def forward(
        self,
        residue_feats: Tensor,
        predicted_coords: Tensor,
        actual_coords: Tensor,
        mask: Optional[Tensor] = None,
        reduce: bool = True,
        **prop_kwargs,
    ) -> Tensor:
        """Compute Predicted LDDT Loss
        :param residue_feats: residue features of shape (b,n,d) where d = self.dim_in
        :param predicted_coords: predicted coordinates of shape (b,n,3) used in ground-truth LDDT calculation
        :param actual_coords: actual coordinates of shape (b,n,3) used in ground-truth LDDT calculation
        :param mask: residue mask of shape (b,n) indicating which residues to compute LDDT scores for
        :param reduce : take mean of loss (iff reduce)
        :return: cross entropy loss between predicted logits and true LDDT labels.
        """
        assert residue_feats.ndim == predicted_coords.ndim == actual_coords.ndim == 3
        # compute true property and class labels
        true_prop = self.get_true_property(
            predicted_coords=predicted_coords, actual_coords=actual_coords, **prop_kwargs
        )
        assert true_prop.ndim == 3, f"{true_prop.shape}"
        true_labels = self.property_to_labels(true_prop).detach()
        # get predicted logits
        pred_prop_logits = self.get_logits(residue_feats)
        ce = self.loss_fn(pred_prop_logits, true_labels)

        if reduce:
            return torch.mean(ce[mask]) if exists(mask) else torch.mean(ce)
        return ce.masked_fill(~mask, 0) if exists(mask) else ce

    @abstractmethod
    def forward_from_output(self, output: ModelOutput, reduce: bool = True, **kwargs) -> Tensor:
        """Run forward from ModelOutput Object"""
        pass


class PredLDDTLossNet(ResiduePropertyLossNet):  # noqa
    """Loss for LDDT Prediction"""

    def __init__(
        self,
        dim_in,
        n_bins: int = 20,
        n_hidden_layers: int = 1,
        atom_ty: Optional[str] = "CA",
        smoothing: float = 0,
    ):
        """
        :param dim_in: residue feature dimension
        :param n_bins: number of lddt bins to project residue features into
        """
        super().__init__(
            dim_in=dim_in, min_val=0, max_val=1, step=1 / n_bins, n_hidden_layers=n_hidden_layers, smoothing=smoothing
        )
        self.atom_ty = atom_ty
        
        self.cutoff, self.ts = 15.0, [0.5,1,2,4]
        if self.atom_ty.lower() in ["distal", "functional"]:
            mask = DISTAL_ATOM_MASK_TENSOR if self.atom_ty.lower() == "distal" else FUNCTIONAL_ATOM_MASK_TENSOR
            self.register_buffer("mask", mask)
        else:
            self.mask = None

    def get_true_property(
        self,
        predicted_coords: Tensor,
        actual_coords: Tensor,
        pred_rigids: Rigids = None,
    ) -> Tensor:
        """pLDDT score between two lists of coordinates"""
        
        with torch.no_grad():
            return compute_coord_lddt(
                predicted_coords=predicted_coords,
                actual_coords=actual_coords,
                cutoff=self.cutoff,
                per_residue=True,
                pred_rigids=pred_rigids,
                thresholds = self.ts,
            ).unsqueeze(-1)

    def forward_from_output(self, output: ModelOutput, reduce: bool = True, use_rigids=False, return_true_prop=False, **kwargs) -> Tensor:
        """Run forward from ModelOutput Object"""
        if self.atom_ty.lower() in ["distal", "functional"]:
            seq = output.native_protein.seq_encoding.unsqueeze(0)
            mask, atom_tys = self.mask[seq], ALL_ATOMS
        else:
            mask = output.get_atom_mask(native=True, atom_tys=[self.atom_ty])
            atom_tys = [self.atom_ty]
            mask = mask.unsqueeze(0) if mask.ndim == 2 else mask

        loss_input = output.get_pred_and_native_coords_and_mask(atom_tys, align_by_kabsch=False)
        pred_coords, native_coords, atom_mask = loss_input
        mask = atom_mask & mask
        residue_feats = output.scalar_output
        pred_coords, native_coords = map(
            lambda x: x[mask].unsqueeze(0), (pred_coords, native_coords)
        )
        residue_feats = residue_feats[torch.any(mask,dim=-1)].unsqueeze(0)

        loss= self.forward(
            residue_feats=residue_feats,
            predicted_coords=pred_coords,
            actual_coords=native_coords,
            mask=mask[mask].unsqueeze(0),
            reduce=reduce,
        )
        true_prop = self.get_true_property(pred_coords,native_coords)
        return (loss, true_prop, torch.any(mask,dim=-1)) if return_true_prop else loss
