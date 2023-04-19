import torch
from torch import nn, Tensor
from scp_postprocess.of_rigid_utils import Rigid, Rotation
from typing import Optional, Dict, Tuple
from scp_postprocess.of_utils import Linear
from scp_postprocess.helpers import default, exists
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()

from scp_postprocess.sidechain_rigid_utils import (
    frames_and_literature_positions_to_atom37_pos,
    torsion_angles_to_frames,
    restype_rigid_group_default_frame,
    restype_atom37_to_rigid_group,
    restype_atom37_mask,
    restype_atom37_rigid_group_positions,
)


def masked_mean(mask, value, dim, eps=1e-4):
    mask = mask.expand(*value.shape)
    return torch.sum(mask * value, dim=dim) / (eps + torch.sum(mask, dim=dim))


class AngleResnetBlock(nn.Module):
    def __init__(self, c_hidden):
        """
        Args:
            c_hidden:
                Hidden channel dimension
        """
        super(AngleResnetBlock, self).__init__()

        self.c_hidden = c_hidden

        self.linear_1 = Linear(self.c_hidden, self.c_hidden, init="relu")
        self.linear_2 = Linear(self.c_hidden, self.c_hidden, init="final")

        self.relu = nn.ReLU()

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        s_initial = a

        a = self.relu(a)
        a = self.linear_1(a)
        a = self.relu(a)
        a = self.linear_2(a)

        return a + s_initial


class AngleResnet(nn.Module):
    """
    Implements Algorithm 20, lines 11-14
    """

    def __init__(self, c_in, c_hidden, no_blocks, no_angles, epsilon):
        """
        Args:
            c_in:
                Input channel dimension
            c_hidden:
                Hidden channel dimension
            no_blocks:
                Number of resnet blocks
            no_angles:
                Number of torsion angles to generate
            epsilon:
                Small constant for normalization
        """
        super(AngleResnet, self).__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_blocks = no_blocks
        self.no_angles = no_angles
        self.eps = epsilon

        self.linear_in = Linear(self.c_in, self.c_hidden)
        self.linear_initial = Linear(self.c_in, self.c_hidden)

        self.layers = nn.ModuleList()
        for _ in range(self.no_blocks):
            layer = AngleResnetBlock(c_hidden=self.c_hidden)
            self.layers.append(layer)

        self.linear_out = Linear(self.c_hidden, self.no_angles * 2)
        self.relu = nn.ReLU()

    def forward(self, s: torch.Tensor, s_initial: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s:
                [*, C_hidden] single embedding
            s_initial:
                [*, C_hidden] single embedding as of the start of the
                StructureModule
        Returns:
            [*, no_angles, 2] predicted angles
        """
        # NOTE: The ReLU's applied to the inputs are absent from the supplement
        # pseudocode but present in the source. For maximal compatibility with
        # the pretrained weights, I'm going with the source.

        # [*, C_hidden]
        s_initial = self.relu(s_initial)
        s_initial = self.linear_initial(s_initial)
        s = self.relu(s)
        s = self.linear_in(s)
        s = s + s_initial

        for l in self.layers:
            s = l(s)

        s = self.relu(s)

        # [*, no_angles * 2]
        s = self.linear_out(s)

        # [*, no_angles, 2]
        s = s.view(s.shape[:-1] + (-1, 2))

        unnormalized_s = s
        norm_denom = torch.sqrt(
            torch.clamp(
                torch.sum(s**2, dim=-1, keepdim=True),
                min=self.eps,
            )
        )
        s = s / norm_denom

        return unnormalized_s, s


class FACoordModule(nn.Module):
    """Get Full atom coordinates from residue features, pair features, and rigids"""

    def __init__(
        self,
        node_dim_in: Optional[int] = None,
        num_angles: int = 7,
        dim_hidden: int = 128,
        num_blocks: int = 2,
        epsilon=1e-8,
        predict_angles: bool = True,
        use_persistent_buffers: bool = False,
        replace_bb: bool = True,
    ):
        super(FACoordModule, self).__init__()
        self.node_dim_in = node_dim_in
        self.num_angles = num_angles
        self.dim_hidden = dim_hidden
        self.epsilon = epsilon
        self.num_blocks = num_blocks
        self.angle_resnet = (
            AngleResnet(
                self.node_dim_in,
                self.dim_hidden,
                self.num_blocks,
                self.num_angles,
                self.epsilon,
            )
            if predict_angles
            else None
        )
        self.use_persistent_buffers = use_persistent_buffers
        self.replace_bb = replace_bb

    def _init_residue_constants(self, float_dtype, device):
        if not hasattr(self, "default_frames"):
            self.register_buffer(
                "default_frames",
                torch.tensor(
                    restype_rigid_group_default_frame.clone().tolist(),
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=self.use_persistent_buffers,
            )
        if not hasattr(self, "group_idx"):
            self.register_buffer(
                "group_idx",
                torch.tensor(
                    restype_atom37_to_rigid_group.clone().tolist(),
                    device=device,
                    requires_grad=False,
                ),
                persistent=self.use_persistent_buffers,
            )
        if not hasattr(self, "atom_mask"):
            self.register_buffer(
                "atom_mask",
                torch.tensor(
                    restype_atom37_mask.clone().tolist(),
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=self.use_persistent_buffers,
            )
        if not hasattr(self, "lit_positions"):
            self.register_buffer(
                "lit_positions",
                torch.tensor(
                    restype_atom37_rigid_group_positions.clone().tolist(),
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=self.use_persistent_buffers,
            )

    def torsion_angles_to_frames(self, r, alpha, f):
        # Lazily initialize the residue constants on the correct device
        self._init_residue_constants(alpha.dtype, alpha.device)
        # Separated purely to make testing less annoying
        return torsion_angles_to_frames(r, alpha, f, self.default_frames)

    def frames_and_literature_positions_to_atom37_pos(self, r, f):  # [*, N, 8]  # [*, N]
        # Lazily initialize the residue constants on the correct device
        self._init_residue_constants(r.get_rots().dtype, r.get_rots().device)
        return frames_and_literature_positions_to_atom37_pos(
            r,
            f,
            self.default_frames,
            self.group_idx,
            self.atom_mask,
            self.lit_positions,
        )

    @typechecked
    def forward(
        self,
        residue_feats: Optional[TensorType["batch", "seq", "hidden"]],
        seq_encoding: TensorType["batch", "seq"],
        coords: Optional[TensorType["batch", "seq", "atoms", 3]],
        rigids: Optional[Rigid],
        angles: Optional[TensorType["batch", "seq", "no_angles", 2]] = None,
    ) -> Dict[str, Tensor]:
        # NOTE: rigids expected to be scaled (unit==angstrom)
        # AND stop grad on rigid rotation should be called *after* this forward pass
        assert exists(rigids) or exists(coords)
        assert (not self.replace_bb) or exists(coords)
        assert seq_encoding.ndim == 2
        if exists(coords):
            assert coords.ndim == 4, f"{coords.shape}"

        seq_encoding = torch.clamp_max(seq_encoding.clone(), 20)

        if not exists(rigids):
            N, CA, C, *_ = coords.unbind(dim=-2)
            rigids = Rigid.make_transform_from_reference(N, CA, C)
        if not exists(angles):
            unnormalized_angles, angles = self.angle_resnet(residue_feats, residue_feats)
        else:
            unnormalized_angles = angles

        all_rigids = Rigid(
            Rotation(rot_mats=rigids.get_rots().get_rot_mats(), quats=None),
            rigids.get_trans(),
        )
        all_rigids = self.torsion_angles_to_frames(
            all_rigids,
            angles,
            seq_encoding,
        )
        all_coords = self.frames_and_literature_positions_to_atom37_pos(
            all_rigids,
            seq_encoding,
        )
        if self.replace_bb:  # replace backbone coordinates with ground truth
            all_coords[..., :4, :] = coords[..., :4, :]
        preds = {
            # "frames": rigids.to_tensor_7(),
            # "sidechain_frames": all_rigids.to_tensor_4x4(),
            "unnormalized_angles": unnormalized_angles,
            "angles": angles,
            "positions": all_coords,
        }

        return preds
