"""Feature representation
"""
from __future__ import annotations

from typing import Any, Optional, List, Union

import torch
from torch import Tensor

from protein_learning.common.helpers import safe_to_device, exists, default
from protein_learning.features.feature_config import FeatureTy, FeatureEmbeddingTy


class Feature:
    """Represents a single feature
    """

    def __init__(
            self,
            raw_data: Any,
            encoded_data: Any,
            name: str,
            ty: FeatureTy,
            dtype: torch.dtype = torch.float32,
            n_classes: Optional[int] = None,
            raw_mask_value: Optional[Any] = None,
            encoding_ty: Optional[Union[List[FeatureEmbeddingTy], FeatureEmbeddingTy]] = None,
            **kwargs,
    ):
        """
        :param raw_data: raw feature data (e.g. string AA sequence)
        :param encoded_data: encoded feature data (e.g. AA sequence (str) -> AA ids (int))
        :param name: feature name (e.g. "relative_orientation"
        :param ty: feature tye (coord, scalar, pair)
        :param dtype: feature dtype (e.g. torch.float32, torch.long)
        :param n_classes: number of classes for embedding (e.g. one-hot, nn.Embedding)
        :param raw_mask_value: value to use when masking raw data for this feature.
        """
        self.raw_data, self.encoded_data = raw_data, encoded_data
        self.name = name
        self.ty, self.dtye = ty, dtype
        self.n_classes = n_classes
        self.masked = False
        self.raw_mask_value = raw_mask_value
        self.encoding_ty = encoding_ty
        self.kwargs = kwargs

    def to(self, device: Any) -> Feature:
        """maps underlying data to given device
        :param device: the device to map to
        """
        self.raw_data = safe_to_device(self.raw_data, device)
        self.encoded_data = safe_to_device(self.encoded_data, device)
        self.raw_mask_value = safe_to_device(self.raw_mask_value, device)
        return self

    def get_raw_data(self):
        """Returns features raw data"""
        if self.masked:
            if not exists(self.raw_mask_value):
                raise Exception(f"feature {self.name} has been masked,"
                                f"and no mask_value param was specified."
                                f" raw data is not safe to use!")
        return self.raw_data

    def get_encoded_data(self):
        """Returns features encoded data"""
        return self.encoded_data.long()

    @property
    def encoded_shape(self):
        """Returns shape of underlying encoded data object"""
        return self.encoded_data.shape if torch.is_tensor(self.encoded_data) else None

    @property
    def raw_shape(self):
        """Returns shape of underlying raw data object"""
        return self.raw_data.shape if torch.is_tensor(self.raw_data) else None

    def __len__(self):
        data = default(self.encoded_data, self.raw_data)
        if exists(data):
            idx = 1 if self._has_batch_dim(data) else 0
            return data.shape[idx]
        raise Exception(f"no encoded data found for feature {self.name}")

    def add_batch(self, other: Feature):
        """Adds the other features to this feature"""
        raise Exception("not implemented")

    def _maybe_add_batch(self, feat: Any) -> Any:
        if not exists(feat) or not torch.is_tensor(feat):
            return feat
        return feat if self._has_batch_dim(feat) else feat.unsqueeze(0)

    def maybe_add_batch(self) -> Feature:
        """Adds batch dimension if not present"""
        self.raw_data = self._maybe_add_batch(self.raw_data)
        self.encoded_data = self._maybe_add_batch(self.encoded_data)
        return self

    def _has_batch_dim(self, feat):
        if not exists(feat) or not torch.is_tensor(feat):
            return False
        if self.ty == FeatureTy.PAIR:
            if feat.ndim == 3:
                return False
            assert feat.ndim == 4, f"unexpected feature dimension :" \
                                   f" {feat.ndim}, {self.name}, {feat.shape}"
            return True
        if self.ty == FeatureTy.RESIDUE:
            if feat.ndim == 2:
                return False
            assert feat.ndim == 3, \
                f"[{self.name}] expected feature dimension 3, got shape : {feat.shape}"
            return True
        raise Exception("this line should be unreachable!")

    def _crop(self, feat: Any, start: int, end: int) -> Any:
        if not exists(feat) or not torch.is_tensor(feat):
            return feat
        if self.ty == FeatureTy.PAIR:
            return feat[..., start:end, start:end, :]
        elif self.ty == FeatureTy.RESIDUE:
            return feat[..., start:end, :]
        else:
            raise Exception("not implemented")

    def crop(self, start, end) -> Feature:
        """crop the feature from start..end"""
        self.raw_data = self._crop(self.raw_data, start, end)
        self.encoded_data = self._crop(self.encoded_data, start, end)
        return self

    def apply_mask(self, mask: Tensor):
        """Apply mask to feature information"""
        assert mask.shape[0] == len(self)
        if self._has_batch_dim(default(self.encoded_data, self.raw_data)):
            mask = mask.unsqueeze(0)

        if exists(self.encoded_data):
            assert self.encoded_shape[:mask.ndim] == mask.shape, \
                f"[Apply Mask] encoded data shape mismatch {self.name}:" \
                f" {self.encoded_shape}, {mask.shape}"
            self.encoded_data[mask] = self.n_classes

        if exists(self.raw_mask_value):
            assert exists(self.raw_data), "Missing raw data, but have mask value!"
            assert self.raw_data.shape[:mask.ndim] == mask.shape, \
                f"[Apply Mask] raw data shape mismatch {self.name}:" \
                f" {self.raw_data.shape}, {mask.shape}"
            self.raw_data[mask] = self.raw_mask_value

        self.masked = True

    def set_encoding_type(self, ty: FeatureEmbeddingTy):
        self.encoding_ty = ty

    @property
    def device(self):
        """Get device on which feature is stored"""
        return self.encoded_data.device if exists(self.encoded_data) else self.raw_data.device
