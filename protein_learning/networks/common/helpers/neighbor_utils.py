from __future__ import annotations

from typing import List

import torch
from einops import rearrange, repeat  # noqa

from protein_learning.networks.common.helpers.torch_utils import batched_index_select
from protein_learning.networks.common.helpers.torch_utils import to_rel_pos


def safe_to_device(x, device):
    return x.to(device) if torch.is_tensor(x) else x


class NeighborInfo:

    def __init__(self,
                 indices: torch.LongTensor,
                 mask: torch.BoolTensor,
                 rel_dists: torch.Tensor,
                 rel_pos: torch.Tensor,
                 coords: torch.Tensor,
                 max_radius: float,
                 top_k: int,
                 full_mask: torch.Tensor,
                 ):
        self.indices = indices
        self.mask = mask
        self.rel_dists = rel_dists
        self.rel_pos = rel_pos
        self.coords = coords
        self.max_radius = max_radius
        self.top_k = top_k
        self.full_mask = full_mask

    @property
    def _attrs(self) -> List[str]:
        return [attr for attr in self.__dict__.keys() if not attr.startswith("_")]

    @property
    def device(self):
        return self.coords.device

    def to_device(self, device) -> NeighborInfo:
        for key in self._attrs:
            setattr(self, key, safe_to_device(getattr(self, key, None), device))
        return self

    def __getitem__(self, item) -> NeighborInfo:
        batch_attrs = {}
        for key in self._attrs:
            val = getattr(self, key)
            batch_attrs[key] = val[item].unsqueeze(0) if torch.is_tensor(val) else val
        return NeighborInfo(**batch_attrs)


def get_neighbor_info(
        coords: torch.Tensor,
        max_radius: float,
        top_k: int,
        exclude_self: bool = True,
) -> NeighborInfo:
    """Gets the k nearest neighbors for each input coordinate

    Args:
        coords: b x n x 3 matrix of coordinates
        max_radius: maximum radius under which a pair of coordinates are considered neighbors
        top_k: maximum number of neighbors per coordinate (if the number of neighbors within the given
        radius exceeds top_k, only the top_k closest paris are returned).
        exclude_self: whether to include an edge from node i to node i (if False)

    Returns:
        A Tuple of tensors
        (1) neighbor_indices: A b x n x top_k matrix of neighbor indices, where
        neighbor_indices[b][i] gives the indices of nearest neighbors for coordinate i
        (2) neighbor_mask: A b x n x top_k boolean tensor such that neighbor_mask[b][i][j]
        is true if and only if i and j are neighbors
        (3) rel_dists: A b x n x top_k matrix of relative distances between all pairs of coordinates
        (4) rel_pos: A b x n x top_k x 3 matrix of relative coordinates. i.e. rel_coords[b][i][j]
        equals coords[b][i]-coords[b][j].
    """
    with torch.no_grad():
        assert len(coords.shape) == 3, f"got shape {coords.shape}, expected shape to have length 3"
        #coords = (coords - torch.mean(coords, dim=1, keepdim=True)).detach().clone()
        b, n, device = *coords.shape[:2], coords.device  # noqa

        # masks and helpers
        exclude_self_mask = rearrange(~torch.eye(n, dtype=torch.bool, device=device).bool(), 'i j -> () i j')  # noqa

        indices = repeat(torch.arange(n, device=device), 'i -> b j i', b=b, j=n)

        # exclude edge of token to itself
        if exclude_self:
            indices = indices.masked_select(exclude_self_mask).reshape(b, n, n - 1)

        rel_pos = to_rel_pos(coords)
        if exclude_self:
            rel_pos = rel_pos.masked_select(exclude_self_mask[..., None]).reshape(b, n, n - 1, 3)
        rel_dist = rel_pos.norm(dim=-1)

        total_neighbors = int(min(top_k, n - 1))
        dist_values, nearest_indices = rel_dist.topk(total_neighbors, dim=-1, largest=False)

        neighbor_mask = dist_values <= max_radius
        neighbor_rel_dist = batched_index_select(rel_dist, nearest_indices, dim=2)
        neighbor_rel_pos = batched_index_select(rel_pos, nearest_indices, dim=2)
        neighbor_indices = batched_index_select(indices, nearest_indices, dim=2)
        full_neighbor_mask = rel_dist <= dist_values[..., -1].reshape(b, n, 1)
        full_neighbor_mask = torch.logical_and(full_neighbor_mask, rel_dist <= max_radius)  # noqa
        return NeighborInfo(
            indices=neighbor_indices,
            mask=neighbor_mask,
            full_mask=full_neighbor_mask,
            rel_dists=neighbor_rel_dist.detach(),
            rel_pos=neighbor_rel_pos.detach(),
            coords=coords.detach(),
            max_radius=max_radius,
            top_k=total_neighbors
        )
