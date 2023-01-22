"""Pair Loss Functions and Networks"""
from typing import Dict, Optional, Tuple, Union
from typing import List

import torch
import torch.nn.functional as F  # noqa
from einops import rearrange  # noqa
from einops.layers.torch import Rearrange  # noqa
from torch import nn, Tensor

from protein_learning.common.data.data_types.model_output import ModelOutput
from protein_learning.common.helpers import exists, safe_norm, default
from protein_learning.common.rigids import Rigids
from protein_learning.networks.loss.utils import softmax_cross_entropy, partition


class PairDistLossNet(nn.Module):  # noqa
    """
    This module is used to predict pairwise distances (for given atom types) from
    output pair features. The predictions are compared to the true distances and
    cross entropy loss is used to derive the final output.
    A shallow FeedForward network is used to obtain predicted distance logits.
    """

    def __init__(
            self,
            dim_in: int,
            atom_tys: List[str],
            step: float = 0.4,
            d_min: float = 2.5,
            d_max: float = 20,
            use_hidden: bool = False,
            smoothing: float = 0,
    ):
        """
        :param dim_in: pair feature dimension
        :param atom_tys: atom types to compute loss for - should be given as
        a list [a_1,a_2,a_3,...,a_2k]. distances between atoms a_2i and a_{2i+1} will
        be predicted... i.e. the atom pairs are (a1,a2),...(a_{2k-1}, a_2k)
        :param step: step size between predicted distances
        :param d_min: minimum distance to predict
        :param d_max: maximum distance to predict
        :param use_hidden : whether to add a hidden layer in distance logit prediction network
        **An extra bin for distances greater than d_max is also appended***
        """
        super().__init__()
        assert len(atom_tys) % 2 == 0, f"must have even number of atom types, got: {atom_tys}"
        self.step, self.d_min, self.d_max = step, d_min, d_max
        self._bins = torch.arange(self.d_min, self.d_max + 2 * step, step=step)
        self.atom_ty_set = set(atom_tys)
        self.num_pair_preds = len(atom_tys) // 2
        self.atom_tys = [(x, y) for x, y in partition(atom_tys, chunk=2)]
        dim_hidden = self.num_pair_preds * self._bins.numel()
        self.net = nn.Sequential(
            nn.LayerNorm(dim_in),
            nn.Linear(dim_in, dim_hidden),
            nn.GELU() if use_hidden else nn.Identity(),
            nn.Linear(dim_hidden, dim_hidden) if use_hidden else nn.Identity(),
            Rearrange("b n m (p d) -> b n m p d", p=len(self.atom_tys))
        )
        self.loss_fn = softmax_cross_entropy
        self.smoothing = smoothing

    def bins(self, device) -> Tensor:
        """Gets the bins used to define the predicted distances"""
        # makes sure devices match
        if self._bins.device == device:
            return self._bins
        self._bins = self._bins.to(device)
        return self._bins

    def _to_labels(self, dists: Tensor) -> Tensor:
        """Convert native distances to one-hot labels"""
        dists = torch.clamp(dists, self.d_min, self.d_max + self.step) - self.d_min
        labels = torch.round(dists / self.step).long()
        return F.one_hot(labels, num_classes=self._bins.numel())

    def _get_true_dists_n_masks(self, atom_ty_map: Dict[str, int],
                                atom_coords: Tensor,
                                atom_masks: Optional[Tensor] = None,
                                pair_mask: Optional[Tensor] = None,
                                ) -> Tuple[Tensor, Optional[Tensor]]:
        a1_a2_dists, a1_a2_masks = [], []
        with torch.no_grad():
            for (a1, a2) in self.atom_tys:
                a1_pos, a2_pos = atom_ty_map[a1], atom_ty_map[a2]
                a1_coords, a2_coords = atom_coords[:, :, a1_pos], atom_coords[:, :, a2_pos]

                # add mask
                if exists(atom_masks):
                    a1_mask, a2_mask = atom_masks[:, :, a1_pos], atom_masks[:, :, a2_pos]
                    a1_a2_mask = torch.einsum("b i, b j -> b i j", a1_mask, a2_mask)
                    a1_a2_mask = torch.logical_and(a1_a2_mask, pair_mask) if exists(pair_mask) else a1_a2_mask
                    a1_a2_masks.append(a1_a2_mask)

                a1_a2_dist = torch.cdist(a1_coords, a2_coords)
                a1_a2_dists.append(a1_a2_dist)
            full_dists = torch.cat([x.unsqueeze(-1) for x in a1_a2_dists], dim=-1)
            full_mask = None
            if exists(atom_masks):
                full_mask = torch.cat([x.unsqueeze(-1) for x in a1_a2_masks], dim=-1)
        return full_dists.detach(), full_mask

    def forward(
            self,
            pair_output: Tensor,
            atom_ty_map: Dict[str, int],
            atom_coords: Tensor,
            atom_masks: Optional[Tensor] = None,
            pair_mask: Optional[Tensor] = None,
            reduce: bool = True,
            return_logits: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        :param pair_output: Output pair features
        :param atom_ty_map: Dict mapping from atom type to atom position in coordinate tensor
        :param atom_coords: atom coordinates of shape (b,n,a,3) (where atom_ty_map[atom_ty] indexes a dimension)
        :param atom_masks: coordinate mask of shape (b,n,a)
        :param pair_mask: (Optional) mask indicating which pair features to predict distances for (b,n,n)
        :param reduce : whether to take mean of output or return raw scores
        :return: Average cross entropy loss of predictions over all atom types
        """
        assert atom_coords.ndim == 4
        # get predictions
        full_dists, full_mask = self._get_true_dists_n_masks(
            atom_ty_map=atom_ty_map,
            atom_coords=atom_coords,
            atom_masks=atom_masks,
            pair_mask=pair_mask,
        )
        if return_logits:
            return self.net(pair_output), full_mask
        labels = self._to_labels(full_dists)
        if reduce:
            return torch.mean(self.loss_fn(self.net(pair_output)[full_mask], labels[full_mask]))
        else:
            return self.loss_fn(self.net(pair_output), labels), full_mask

    def forward_from_output(
            self,
            output: ModelOutput,
            reduce: bool = True,
            pair_feats: Optional[Tensor] = None,
            return_logits: bool = False,
            **kwargs
    ) -> Union[
        Tensor, Tuple[Tensor, Tensor]]:
        """Run forward from ModelOutput Object"""
        atom_tys = list(self.atom_ty_set)
        atom_ty_map = {a: i for i, a in enumerate(atom_tys)}
        pair_mask = getattr(output, "pair_mask", None)
        native_atom_coords = output.get_atom_coords(native=True, atom_tys=atom_tys).unsqueeze(0)
        native_atom_mask = output.get_atom_mask(native=True, atom_tys=atom_tys).unsqueeze(0)

        return self.forward(
            pair_output=default(pair_feats, output.pair_output),
            atom_ty_map=atom_ty_map,
            atom_coords=native_atom_coords,
            atom_masks=native_atom_mask,
            pair_mask=pair_mask,
            reduce=reduce,
            return_logits=return_logits,
        )


class PredictedAlignedErrorLossNet(nn.Module):
    """Predicted Aligned Error
    """

    def __init__(
            self,
            dim_in: int,
            atom_tys: List[str],
            step: float = 0.5,
            d_max: float = 12,
            smoothing: float = 0,
    ):
        """
        :param dim_in: pair feature dimension
        :param atom_tys: atom types to compute loss for - should be given as
        a list [a_1,a_2,a_3,...,a_2k]. distances between atoms a_2i and a_{2i+1} will
        be predicted... i.e. the atom pairs are (a1,a2),...(a_{2k-1}, a_2k)
        :param step: step size between predicted distances
        :param d_min: minimum distance to predict
        :param d_max: maximum distance to predict
        :param use_hidden : whether to add a hidden layer in distance logit prediction network
        **An extra bin for distances greater than d_max is also appended***
        """
        super().__init__()
        self.step, self.d_min, self.d_max = step, -d_max, d_max
        self._bins = torch.arange(self.d_min - step, self.d_max + 2 * step, step=step)
        self._bins = (self._bins[:-1] + self._bins[1:]) / 2
        self.atom_tys = atom_tys
        self.net = nn.Sequential(
            nn.LayerNorm(dim_in),
            nn.Linear(dim_in, 128),
            nn.GELU(),
            nn.Linear(128, self._bins.numel() * len(self.atom_tys)),
            Rearrange("b n m (p d) -> b n m p d", p=len(self.atom_tys))
        )
        self.loss_fn = softmax_cross_entropy
        self.smoothing = smoothing
        self._bins = self._bins[(None,) * 4]

    def bins(self, device) -> Tensor:
        """Gets the bins used to define the predicted distances"""
        # makes sure devices match
        if self._bins.device == device:
            return self._bins
        self._bins = self._bins.to(device)
        return self._bins

    def _to_labels(self, dists: Tensor) -> Tensor:
        """Convert native distances to one-hot labels"""
        bins = self.bins(dists.device)
        nearest_bins = torch.abs(dists.unsqueeze(-1) - bins)
        labels = F.one_hot(torch.argmin(nearest_bins, dim=-1), bins.numel())
        return labels

    def _get_true_dists_n_masks(self, atom_ty_map: Dict[str, int],
                                true_coords: Tensor,
                                pred_coords: Tensor,
                                true_rigids: Optional[Rigids],
                                pred_rigids: Optional[Rigids],
                                atom_masks: Optional[Tensor] = None,
                                pair_mask: Optional[Tensor] = None,
                                ) -> Tuple[Tensor, Optional[Tensor]]:
        dists, masks = [], []
        with torch.no_grad():
            for aty in self.atom_tys:
                a_pos = atom_ty_map[aty]
                c_true, c_pred = true_coords[:, :, a_pos], pred_coords[:, :, a_pos]
                if exists(pred_rigids):
                    assert exists(true_rigids)
                    c_true, c_pred = map(lambda x: rearrange(x, "b n c-> b () n c"), (c_true, c_pred))
                    rel_true = true_rigids.apply_inverse(c_true)
                    rel_pred = pred_rigids.apply_inverse(c_pred)
                    a_dists = safe_norm(rel_pred, dim=-1) - safe_norm(rel_true, dim=-1)
                else:
                    a_dists = torch.cdist(c_pred, c_pred) - torch.cdist(c_true, c_true)

                dists.append(a_dists)

                # add mask
                if exists(atom_masks):
                    mask = atom_masks[:, :, a_pos]
                    mask = torch.einsum("b i, b j -> b i j", mask, mask)
                    mask = torch.logical_and(mask, pair_mask) if exists(pair_mask) else mask
                    masks.append(mask)

            full_dists = torch.cat([x.unsqueeze(-1) for x in dists], dim=-1)
            full_mask = None
            if exists(atom_masks):
                full_mask = torch.cat([x.unsqueeze(-1) for x in masks], dim=-1)
        return full_dists.detach(), full_mask

    def forward(
            self,
            pair_output: Tensor,
            atom_ty_map: Dict[str, int],
            true_coords: Tensor,
            pred_coords: Tensor,
            true_rigids: Optional[Rigids],
            pred_rigids: Optional[Rigids],
            atom_masks: Optional[Tensor] = None,
            pair_mask: Optional[Tensor] = None,
            reduce: bool = True,
            return_logits: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        :param pair_output: Output pair features
        :param atom_ty_map: Dict mapping from atom type to atom position in coordinate tensor
        :param atom_coords: atom coordinates of shape (b,n,a,3) (where atom_ty_map[atom_ty] indexes a dimension)
        :param atom_masks: coordinate mask of shape (b,n,a)
        :param pair_mask: (Optional) mask indicating which pair features to predict distances for (b,n,n)
        :param reduce : whether to take mean of output or return raw scores
        :return: Average cross entropy loss of predictions over all atom types
        """
        assert true_coords.ndim == 4
        # get predictions
        full_dists, full_mask = self._get_true_dists_n_masks(
            atom_ty_map=atom_ty_map,
            true_rigids=true_rigids,
            true_coords=true_coords,
            pred_rigids=pred_rigids,
            pred_coords=pred_coords,
            atom_masks=atom_masks,
            pair_mask=pair_mask,
        )
        if return_logits:
            return self.net(pair_output), full_mask

        labels = self._to_labels(full_dists)
        if reduce:
            return torch.mean(self.loss_fn(self.net(pair_output)[full_mask], labels[full_mask]))
        else:
            return self.loss_fn(self.net(pair_output), labels), full_mask

    def forward_from_output(
            self,
            output: ModelOutput,
            true_rigids: Optional[Rigids] = None,
            reduce: bool = True,
            return_logits: bool = False,
            use_rigids=False,
            **kwargs
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Run forward from ModelOutput Object"""
        atom_tys = self.atom_tys
        atom_ty_map = {a: i for i, a in enumerate(atom_tys)}
        pair_mask = getattr(output, "pair_mask", None)
        loss_input = output.get_pred_and_native_coords_and_mask(atom_tys=atom_tys, align_by_kabsch=False)
        pred_coords, native_coords, mask = loss_input
        pred_rigids = default(
            getattr(output, "pred_rigids", None),
            getattr(output, "rigids", None)
        )
        if not exists(true_rigids):
            true_rigids = default(
                getattr(output, 'true_rigids', None),
                getattr(output.model_input, "true_rigids", None)
            )
        native_atom_mask = output.get_atom_mask(native=True, atom_tys=atom_tys).unsqueeze(0)

        return self.forward(
            pair_output=output.pair_output,
            atom_ty_map=atom_ty_map,
            true_rigids=true_rigids if use_rigids else None,
            true_coords=native_coords,
            pred_rigids=pred_rigids if use_rigids else None,
            pred_coords=pred_coords ,
            atom_masks=native_atom_mask,
            pair_mask=pair_mask,
            reduce=reduce,
            return_logits=return_logits,
        )
