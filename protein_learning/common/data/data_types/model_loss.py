"""Model Loss"""

from __future__ import annotations

import math
from math import log2
from typing import Optional, Union, Any

import numpy as np
import torch
from torch import Tensor

from protein_learning.common.helpers import default, exists

ZERO_LOSS = dict(raw_loss=0, loss_val=0, loss_weight=0, baseline=0)


class ModelLoss:
    """Tracks model loss"""

    def __init__(
            self,
            seq_len: Optional[int] = None,
            pdb: Optional[str] = None,
            scale_by_seq_len: bool = True
    ):
        self.loss_dict = {}
        self.seq_len = seq_len
        self.pdb = pdb
        self.scale_by_seq_len = scale_by_seq_len
        self.note = None

    def add_loss(
            self,
            loss: Tensor,
            loss_weight: float = 1,
            baseline: Union[Tensor, float] = 0,
            loss_name: Optional[str] = None,
    ):
        """Add loss term"""
        loss_name = default(loss_name, f"loss_{len(self.loss_dict)}")
        assert loss.numel() == 1, f"[{loss_name}] expected scalar loss"
        assert loss_name not in self.loss_dict
        tmp = self.loss_dict[loss_name] if loss_name in self.loss_dict else ZERO_LOSS
        self.loss_dict[loss_name] = dict(
            raw_loss=loss + tmp["raw_loss"],
            loss_val=loss * loss_weight + tmp["loss_val"],
            loss_weight=loss_weight + tmp["loss_weight"],
            baseline=baseline + tmp["baseline"]
        )

    def merge_loss(self, other: ModelLoss) -> ModelLoss:
        """Merge other loss values with this loss"""
        for loss_name in self.loss_dict:
            if loss_name in other.loss_dict:
                self.loss_dict[loss_name] = self.loss_dict[loss_name] + other.loss_dict[loss_name]
            else:
                self.loss_dict[loss_name] = other.loss_dict[loss_name]
        return self

    @staticmethod
    def print_loss_val(name, baseline, actual, val):
        """Format and print values for loss"""
        item = lambda x: np.round(x.detach().cpu().item() if torch.is_tensor(x) else x, 4)
        print(f"[{name}] : baseline : {item(baseline)}, "
              f"actual : {item(actual)}, loss_val : {item(val)}")

    @property
    def non_zero_loss_names(self):
        """Names of loss values with non-zero weight"""
        return [k for k, v in self.loss_dict.items() if v["loss_weight"] != 0]

    @property
    def non_zero_loss_vals(self):
        """Loss values with non-zero weight"""
        return [v["loss_val"] for v in self.loss_dict.values() if v["loss_weight"] != 0]

    @staticmethod
    def _valid_loss(val: Any) -> bool:
        if not torch.is_tensor(val):
            return False
        # definitely a tensor
        if val.numel() > 1:
            return False
        # definitely a one element tensor
        return not torch.isnan(val)

    def get_loss(self) -> Tensor:
        """Gets weighted loss value for backprop"""
        scale = log2(default(self.seq_len, math.e)) if self.scale_by_seq_len else 1
        return sum(filter(self._valid_loss, [scale * v for v in self.non_zero_loss_vals]))

    @property
    def float_value(self):
        """Get the (weighted) model loss as a float"""
        return sum(v["loss_val"].detach().item() for v in self.loss_dict.values() if v["loss_weight"] != 0 and not torch.isnan(v["loss_val"]))

    def delete_loss(self, name: str):
        """Delete loss by name"""
        del self.loss_dict[name]

    def scale(self, scale: float):
        """Scale all loss values"""
        for k in self.loss_dict:
            self.loss_dict[k]["loss_val"] = scale * self.loss_dict[k]["loss_val"]

    @staticmethod
    def _get_val(other: Union[float, ModelLoss]):
        return other if isinstance(other, float) else other.float_value

    """Can Treat as a dict, and compare with other loss
    e.g. when computing loss for a homodimer, one must consider 
    the native conformation after permuting each chain. 
    In this case, the loss is computed twice, and one can simply
    take 
        homo_loss = min(loss1,loss2) 
    as the final loss object.
    """

    def __contains__(self, item):
        return item in self.loss_dict

    def __getitem__(self, item):
        return self.loss_dict[item]

    def __setitem__(self, key, value):
        self.loss_dict[key] = value

    def __le__(self, other: Union[float, ModelLoss]) -> bool:
        return self.float_value <= self._get_val(other)

    def __lt__(self, other: Union[float, ModelLoss]) -> bool:
        return self.float_value < self._get_val(other)

    def __ge__(self, other: Union[float, ModelLoss]) -> bool:
        return self.float_value >= self._get_val(other)

    def __gt__(self, other: Union[float, ModelLoss]) -> bool:
        return self.float_value > self._get_val(other)

    def __eq__(self, other: Union[float, ModelLoss]) -> bool:
        return self.float_value == self._get_val(other)

    def add_note(self, note):
        """Add a note (when printing loss)"""
        self.note = default(self.note, "")
        self.note += note

    def display_loss(self):
        """Print loss values for each term"""
        if exists(self.seq_len):
            print(f"model pdb : {self.pdb}, sequence length : {self.seq_len}")
        for name, vals in self.loss_dict.items():
            self.print_loss_val(
                name,
                *list(map(lambda x: vals[x], "baseline raw_loss loss_val".split()))
            )
        self.print_loss_val("total", 0, np.round(self.float_value, 4), np.round(self.float_value, 4))
        if exists(self.note):
            print("[NOTE]", self.note)
