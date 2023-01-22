"""Input for protein-based learning model
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor

from protein_learning.common.protein_constants import BB_ATOMS, SC_ATOMS

try:
    from functools import cached_property  # noqa
except:  # noqa
    from cached_property import cached_property  # noqa

SC_ATOM_SET, BB_ATOM_SET = set(BB_ATOMS), set(SC_ATOMS)


def detect_interface(x1: Tensor, x2: Tensor) -> Tensor:
    assert x1.ndim == x2.ndim == 2
    centroid = torch.mean(x1, dim=0, keepdim=True)
    ds = torch.cdist(x2, centroid)
    interface_order = torch.argsort(ds.squeeze())
    assert interface_order.ndim == 1 and interface_order.numel() == x2.shape[0]
    return interface_order


def get_interface_segment(interface_order: Tensor, max_res: int) -> Tuple[int, int]:
    n_res = interface_order.numel()
    max_res = min(max_res, n_res)
    nearest_res = interface_order[0]
    return (nearest_res - max_res // 2, nearest_res + max_res // 2 - ((max_res + 1) % 2))
