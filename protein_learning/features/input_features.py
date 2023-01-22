"""Helper methods for generating model input features
"""
from __future__ import annotations

from enum import Enum
from math import pi
from typing import Dict

import torch
from einops import rearrange, repeat  # noqa

from protein_learning.features.feature import Feature

PI = pi + 1e-10


class InputFeatures:
    """Dictionary-like wrapper for ProteinModel input features"""

    def __init__(
            self,
            features: Dict[str, Feature],
            batch_size: int,
            length: int,
            **extra
    ):
        """
        :param features: Dict of input features
        :param batch_size: batch size (should be same for all features)
        :param length: the length of the input features  (e.g. number of residues -
        should be same for all features)
        :param masks: (Optional) sequence and feature masks applied to input features
        and input sequence encodings.
        """
        self.features = features
        self._length = length
        self._batch_size = batch_size
        self.crop_posns = None
        self.extra = extra

    @property
    def length(self) -> int:
        """Get the input length"""
        return self._length

    @property
    def batch_size(self) -> int:
        """Get the batch size"""
        return self._batch_size

    def crop(self, start, end) -> InputFeatures:
        """crop underlying features"""
        self.crop_posns = (start, end)
        self._length = end - start
        self.features = {k: v.crop(start, end) for k, v in self.features.items()}
        return self

    def maybe_add_batch(self) -> InputFeatures:
        """Add batch dimension if not present"""
        self.features = {k: v.maybe_add_batch() for k, v in self.features.items()}
        return self

    def to(self, device: str) -> InputFeatures:
        """Send input features to specified device"""
        self.features = {k: v.to(device) for k, v in self.features.items()}
        return self

    def add_batch(self, features: Dict[str, Feature]):
        """Add batch of features to the input"""
        raise Exception("not yet implemented!")

    def items(self):
        """feature dict items"""
        return self.features.items()

    def keys(self):
        """feature dict keys"""
        return self.features.keys()

    def values(self):
        """feature dict values"""
        return self.features.values()

    def __getitem__(self, item):
        if isinstance(item, Enum):
            item = item.value
        return self.features[item]

    def __setitem__(self, key: str, value: Feature):
        assert isinstance(value, Feature)
        self.features[key] = value

    def __contains__(self, item):
        if isinstance(item, Enum):
            item = item.value
        return item in self.features

    @property
    def seq_mask(self):
        """Get sequence mask appied to input"""
        msk = None
        if "seq_mask" in self.extra:
            msk = self.extra["seq_mask"]
        return msk if msk is not None else torch.zeros(self.length).bool()

    @property
    def feat_mask(self):
        """Get feature mask appied to input"""
        msk = None
        if "feat_mask" in self.extra:
            msk = self.extra["feat_mask"]
        return msk if msk is not None else torch.zeros(self.length).bool()

    @property
    def inter_chain_mask(self):
        """Get inter-chain pair mask applied to input"""
        msk = None
        if "inter_chain_mask" in self.extra:
            msk = self.extra["inter_chain_mask"]
        return msk if msk is not None else torch.zeros(self.length).bool()
