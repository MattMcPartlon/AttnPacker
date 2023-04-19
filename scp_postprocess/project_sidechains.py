"""
Functions to project side-chain atom predictions onto continuous rotamer library
Finds sidechain rotamers minimizing RMSD to atom37 representation of coordinates.
"""
import torch
from torch import Tensor, nn
from scp_postprocess.of_rigid_utils import Rigid
from scp_postprocess.coords_from_angles import FACoordModule
import scp_postprocess.protein_constants as pc
from einops import rearrange, repeat  # noqa
from scp_postprocess.helpers import default, safe_normalize, safe_norm
import torch.nn.functional as F  # noqa
from typing import Union, Optional, List, Tuple, Any, Dict
from scp_postprocess.sidechain_rigid_utils import atom37_to_torsion_angles
from scp_postprocess.sidechain_utils import align_symmetric_sidechains
from scp_postprocess.rigids import Rigids as SimpleRigids

try:
    from functools import cached_property  # noqa
except:  # noqa
    from cached_property import cached_property  # noqa
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()


def masked_mean(tensor: Tensor, mask: Optional[Tensor], dim: Union[Tuple, int] = -1, keepdim: bool = False):
    """Performs a masked mean over a given dimension

    Parameters:
        tensor: tensor to apply mean to
        mask: mask to use for calculating mean
        dim: dimension to extract mean over
        keepdim: keep the dimension where mean is taken
    Returns:
        masked mean of tensor, according to mask, along dimension dim.
    """
    if mask is None:
        return torch.mean(tensor, dim=dim, keepdim=keepdim)
    assert tensor.ndim == mask.ndim
    assert tensor.shape[dim] == mask.shape[dim] if isinstance(dim, int) else True
    tensor = torch.masked_fill(tensor, ~mask, 0)
    total_el = mask.sum(dim=dim, keepdim=keepdim)
    mean = tensor.sum(dim=dim, keepdim=keepdim) / total_el.clamp(min=1.0)
    return mean.masked_fill(total_el == 0, 0.0)


@typechecked
def align_starting_coords(
    coords: TensorType["batch", "res", "atom", 3],
    sequence: TensorType["batch", "res", torch.long],
    atom_mask: TensorType["batch", "res", "atom", torch.bool],
) -> TensorType["batch", "res", "atom", 3]:
    """Align coordinates before dihedral projection

    This is done to avoid "mirrored" looking conformations
    for residues such as valine or leucine when the terminal O and OH are flipped.
    """
    coord_module = FACoordModule(
        use_persistent_buffers=True,
        predict_angles=False,
        replace_bb=True,
    )
    N, CA, C, *_ = coords.unbind(dim=-2)
    rigids = Rigid.make_transform_from_reference(N, CA, C).detach()

    # consider initial conformations
    ptn = dict(aatype=sequence, all_atom_positions=coords, all_atom_mask=atom_mask)
    torsion_info = atom37_to_torsion_angles(ptn)
    angles = safe_normalize(torsion_info["torsion_angles_sin_cos"].detach())

    # for residues like LEU and VAL, the terminal groups can be flipped
    swapped_coords = swap_symmetric_atoms(coords.clone(), sequence)
    swapped_ptn = dict(aatype=sequence, all_atom_positions=swapped_coords, all_atom_mask=atom_mask)
    swapped_torsion_info = atom37_to_torsion_angles(swapped_ptn)
    swapped_angles = safe_normalize(swapped_torsion_info["torsion_angles_sin_cos"].detach())

    # map from dihedrals to coordinates
    to_coords = lambda x: coord_module.forward(
        seq_encoding=sequence,
        residue_feats=None,
        coords=coords,
        rigids=rigids,
        angles=x,
    )["positions"]

    # per-res rmsd between x and y
    prms = lambda x, y: compute_sc_rmsd(
        coords=x,
        target_coords=y,
        atom_mask=atom_mask,
        per_residue=True,
        sequence=sequence,
    )

    # take coordinates of lower rmsd conformations
    coords_proj = to_coords(angles)
    swapped_coords_proj = to_coords(swapped_angles)

    prms_initial = prms(coords_proj, coords)
    prms_mirrored = prms(swapped_coords_proj, swapped_coords)
    swapped_better = prms_initial > prms_mirrored
    coords[swapped_better] = swapped_coords[swapped_better]
    return coords


def compute_sc_rmsd(
    coords: TensorType["batch", "res", "atom", 3],
    target_coords: TensorType["batch", "res", "atom", 3],
    atom_mask: TensorType["batch", "res", "atom", torch.bool],
    sequence: TensorType["batch", "res", torch.long],
    per_residue: bool,
    align_coords: bool = False,
    bb_tol: float = 0.25,
) -> Union[TensorType["batch", "res"], TensorType["batch"]]:
    """Compute Side-Chain RMSD

    Parameters:
        per_residue:
            whether to mean side-chain RMSD for entire chain, or for each individual residue.

    NOTE:
        Assumes atoms are given in the order defined by pc.ATOM_TYS
    """
    aln_target_coords, aln_coords = target_coords.clone(), coords.clone()
    diff_bbs = torch.sum(torch.square(aln_target_coords[..., :3, :] - aln_coords[..., :3, :]))
    # print(target_coords[0,:5,:10],coords[0,:5,:10])
    if align_coords or diff_bbs > bb_tol:
        # align ground-truth and native based on per-residue backbone frames
        # rotate each residue using backbone N-CA C-CA displacements to define coordinate system
        target_frames, frames = map(lambda x: SimpleRigids.RigidFromBackbone(x[..., :3, :]), (target_coords, coords))
        shape = target_coords.shape
        aln_target_coords = target_frames.apply_inverse(target_coords)
        aln_coords = frames.apply_inverse(coords)
        assert aln_target_coords.shape == coords.shape == shape
    # swap symmetric side chain atoms in ground-truth
    aligned_target = align_symmetric_sidechains(
        native_coords=aln_target_coords,
        predicted_coords=aln_coords,
        atom_mask=atom_mask,
        native_seq=sequence,
    )

    # compute per-residue side chain RMSD (b, n, a, c -> b, n, a)
    atom_deviations = torch.sum(torch.square(aligned_target - aln_coords), dim=-1)
    # disregard backbone atoms
    per_residue_rmsd = torch.sqrt(masked_mean(atom_deviations[..., 4:], mask=atom_mask[..., 4:], dim=-1))
    res_mask = torch.any(atom_mask[..., 4:], dim=-1)
    if not torch.any(res_mask):
        print("[WARNING] no side-chain atoms detected!")
    return per_residue_rmsd if per_residue else torch.mean(per_residue_rmsd[res_mask], dim=-1)


def rmsd_loss(predicted_coords, target_coords, atom_mask):
    """Compute RMSD loss"""
    coord_deviations = torch.sum(torch.square(predicted_coords - target_coords), dim=-1)[atom_mask]
    return torch.sqrt(torch.clamp_min(torch.mean(coord_deviations), 1e-10))


def item(x: Any):
    if torch.is_tensor(x):
        if len(x) <= 1:
            return x.item()
    return x


def compute_pairwise_vdw_table_and_mask(
    sequence: TensorType["batch", "res", torch.long],
    atom_mask: TensorType["batch", "res", 37, torch.bool],
    global_allowance: float = 0.1,
    global_tol_frac: Optional[float] = None,
    hbond_allowance: float = 0.6,
):
    """
    Special Cases:
        - only atoms separated by at least 4 bonds are considered
        - Backbone-Backbone atom interactions are ignored
        - potential disulfide bridges SG-SG in cysteines are ignored
        - An allowance is subtracted from all atom pairs where one atom is an H-bond donor
        and the other atom is an H-bond acceptor

    Parameters:
        sequence: encoding of sequence (according to pc.AA_IDX_MAP)
        atom_mask: (according to pc.ALL_ATOM_POSITIONS)
        global_allowance: reduce the sum of vdw radii by this amount
        hbond_allowance: reduce the sum of of vdw radii by this amount, when one of the atoms is
        a hydrogen bond donor, and the other is a hydrogen bond acceptor.
        (See e.g. : https://pubmed.ncbi.nlm.nih.gov/9672047/ )

    Returns:
        (1) Table of the form
            T[i,j] = rVDW[i] + rVDW[j] - allowance[i,j]
        (2) Mask of the form:
            M[i,j] = True if and only if steric overlap should be computed for pair (i,j)

    """

    b, n, a, device = *atom_mask.shape[:3], atom_mask.device  # noqa (batch, res, atoms)
    # (a,) vdw radii for each atom 1..a
    atom_vdw_radii = [pc.VAN_DER_WAALS_RADII[atom[:1]] - (global_allowance / 2) for atom in pc.ALL_ATOMS[:a]]
    atom_vdw_radii = repeat(torch.tensor(atom_vdw_radii, device=device), "a -> b n a", b=b, n=n)

    # mapping of each atom to residue index for masking (b,n,a)
    atom_res_indices = repeat(torch.arange(n, device=device), "i -> b i a ", b=b, a=a)

    # apply atom mask (should save a lot of memory :)) -> (N,), where N is number of valid atoms
    all_atom_vdw_radii = atom_vdw_radii[atom_mask]  # N,
    all_atom_res_indices = atom_res_indices[atom_mask]  # N,

    # (1) Calculate initial pairwise VDW sums and pairwise mask
    to_pairwise_diffs = lambda x, y=None: (
        rearrange(x, "N ...-> N () ...") - rearrange(default(y, x), "N ...-> () N ...")
    )
    pairwise_vdw_sums = to_pairwise_diffs(all_atom_vdw_radii, -1 * all_atom_vdw_radii)  # N,N
    pairwise_vdw_sums = pairwise_vdw_sums * default(global_tol_frac, 1)  # allowance for clashes

    # keep only inter-residue atom pairs, excluding lower triangle
    vdw_pair_mask = to_pairwise_diffs(all_atom_res_indices) > 0  # N,N

    # (2) mask out BB-BB interactions
    bb_atom_types = set(pc.BB_ATOMS + ["CB"])  # CB placement is completely determined by given backbone conformation
    # bb_atom_types = set(pc.ALL_ATOMS).intersection(set(bb_atom_types))
    bb_atom_posns = [pc.ALL_ATOM_POSNS[atom] for atom in bb_atom_types]
    bb_mask = torch.zeros_like(atom_mask).float()
    # backbone atoms have 1, all others set to 0
    bb_mask[:, :, bb_atom_posns] = 1
    bb_mask = bb_mask[atom_mask]
    bb_pair_mask = bb_mask.unsqueeze(1) * bb_mask.unsqueeze(0)
    # update vdw pair mask
    vdw_pair_mask = vdw_pair_mask & ~bb_pair_mask.bool()

    # (3) Correct with tolerance for H-Bonds
    hbd, hba = pc.HBOND_DONOR_TENSOR, pc.HBOND_ACCEPTOR_TENSOR
    hbd, hba = map(lambda x: repeat(x.to(device), "a -> () n a", n=n)[atom_mask], (hbd, hba))
    hbond_allowance_mask = torch.einsum("i,j->ij", hbd, hba).bool()
    hbond_allowance_mask = torch.logical_or(hbond_allowance_mask, hbond_allowance_mask.T)
    pairwise_vdw_sums[hbond_allowance_mask] = pairwise_vdw_sums[hbond_allowance_mask] - hbond_allowance

    # (4) now for proline, CD is bonded with BB nitrogen - clashes should be ignored
    # between this and next residue
    is_proline = sequence == pc.AA_INDEX_MAP["P"]
    before_or_after_proline = is_proline.clone().float()
    before_or_after_proline[:, 1:] += is_proline[:, :-1]
    before_or_after_proline[:, :-1] += is_proline[:, 1:]
    before_or_after_proline = before_or_after_proline.bool().unsqueeze(-1)

    # mask the PRO CD -> next or prev. backbone interactions
    pro_mask = torch.zeros_like(atom_mask).float()
    CD_mask, bb_mask = torch.zeros_like(atom_mask), torch.zeros_like(atom_mask)
    CD_mask[..., pc.ALL_ATOM_POSNS["CD"]], bb_mask[..., :4] = True, True

    pro_mask[before_or_after_proline & bb_mask] = 1
    pro_mask[is_proline.unsqueeze(-1) & CD_mask] = 1
    pro_mask = torch.einsum("i,j->ij", pro_mask[atom_mask], pro_mask[atom_mask]).bool()
    vdw_pair_mask = vdw_pair_mask & ~pro_mask

    # Finally, do not penalize disulfide bridges
    SG_mask = torch.zeros_like(atom_mask)
    SG_mask[..., pc.ALL_ATOM_POSNS["SG"]] = True
    SG_mask = torch.einsum("i,j->ij", SG_mask[atom_mask], SG_mask[atom_mask]).bool()
    vdw_pair_mask = vdw_pair_mask & ~SG_mask

    # return updated vdw sums and pair mask
    return pairwise_vdw_sums, vdw_pair_mask


def _compute_steric_loss(
    atom_coords: TensorType["batch", "res", "atom", 3],
    atom_mask: TensorType["batch", "res", "atom", torch.bool],
    vdw_table: TensorType["N", "M"],  # N is total number of atoms
    vdw_mask: TensorType["N", "M", torch.bool],
    return_clashing_pairs: bool = False,
    masked_reduce: bool = True,
    reduction: str = "sum",
    p: int = 2,
    tol: float = 0.0,
    nbr_indices: Optional[Tensor] = None,
) -> Union[Tensor, Tuple[Tensor, List]]:
    # pairwise distances
    all_atom_coords = atom_coords[atom_mask]  # N, 3
    relative_dists = safe_norm(
        rearrange(all_atom_coords, "N ...-> N () ...") - rearrange(all_atom_coords, "N ...-> () N ..."), dim=-1
    )

    vdw_loss = F.relu(vdw_table - relative_dists, inplace=False) ** p
    # conside only inter-residue interactions, between residue_i and residue_j where i>j
    vdw_loss[~vdw_mask] = 0
    violation_mask = vdw_loss > 0

    clashing_pairs = None
    if return_clashing_pairs:
        b, n, a, device = *atom_mask.shape[:3], atom_mask.device  # noqa (batch, res, atoms)
        # mapping of each atom to residue index for masking (b,n,a)
        atom_res_indices = repeat(torch.arange(n, device=device), "i -> b i a ", b=b, a=a)
        all_atom_res_indices = atom_res_indices[atom_mask]  # N,
        # mapping from residue to indices of each atom
        res_atom_indices = repeat(torch.arange(a, device=device), "i -> b n i", b=b, n=n)
        all_atom_atom_indices = res_atom_indices[atom_mask]  # N,
        clashing_pairs = []
        for i, j in zip(*torch.where(violation_mask)):
            pos_i, pos_j = map(lambda x: (all_atom_res_indices[x], all_atom_atom_indices[x]), (i, j))
            pos_i, pos_j = map(lambda x: (x[0].item(), pc.ALL_ATOMS[x[1].item()]), (pos_i, pos_j))
            d_ij, vdw_sum_ij = relative_dists[i, j], vdw_table[i, j] + tol
            clashing_pairs.append((pos_i, pos_j, round(d_ij.item(), 3), round(vdw_sum_ij.item(), 3)))

    if masked_reduce:
        vdw_loss = vdw_loss[violation_mask]

    zero_loss = relative_dists.new_zeros(1)
    if reduction == "none":
        loss = vdw_loss
    elif reduction == "mean":
        loss = torch.mean(vdw_loss) if len(vdw_loss) > 0 else zero_loss  # grad can be nan o.w.
    elif reduction == "sum":
        loss = torch.sum(vdw_loss) if len(vdw_loss) > 0 else zero_loss  # grad can be nan o.w.
    else:
        raise Exception(f"reduction {reduction} not in ['mean','sum','none']")

    return (loss, clashing_pairs) if return_clashing_pairs else loss


def steric_loss(
    atom_coords: TensorType["batch", "res", "atom", 3],
    atom_mask: TensorType["batch", "res", "atom", torch.bool],
    sequence: Optional[TensorType["batch", "res", torch.bool]],
    return_clashing_pairs: bool = False,
    hbond_allowance: float = 0.6,
    global_allowance: float = 0,
    steric_tol_frac: float = 0.9,
    masked_reduce: bool = True,
    reduction: str = "sum",
    p: int = 2,
) -> Union[float, Tensor]:
    """Inter-Residue Steric clash loss.

    Parameters:
        sequence:
            encoding of sequence (according to pc.AA_IDX_MAP)
        atom_mask:
            (according to pc.ALL_ATOM_POSITIONS)
        global_allowance:
            reduce the sum of vdw radii by this amount
        hbond_allowance:
            reduce the sum of of vdw radii by this amount, when one of the atoms is
            a hydrogen bond donor, and the other is a hydrogen bond acceptor.
            (See e.g. : https://pubmed.ncbi.nlm.nih.gov/9672047/ )
        masked_reduce:
            whether to reduce over only atoms pairs violating
            steric constraints
        reduction:
            reduction to apply to output ["mean","sum","none"]
        p:
            return reduction(loss) ** p
    """
    table, mask = compute_pairwise_vdw_table_and_mask(
        sequence=sequence,
        atom_mask=atom_mask,
        global_allowance=global_allowance,
        hbond_allowance=hbond_allowance,
        global_tol_frac=steric_tol_frac,
    )
    return _compute_steric_loss(
        atom_coords=atom_coords,
        atom_mask=atom_mask,
        vdw_mask=mask,
        vdw_table=table,
        masked_reduce=masked_reduce,
        reduction=reduction,
        p=p,
        return_clashing_pairs=return_clashing_pairs,
        tol=global_allowance,
    )


def compute_clashes(atom_coords: Tensor, atom_mask: Tensor, sequence: Tensor, **kwargs) -> Tuple[int, int]:
    """Returns tuple : (number of clashes, number of valid atom pairs)"""
    unreduced_loss, clash_pairs = steric_loss(
        atom_coords=atom_coords,
        atom_mask=atom_mask,
        reduction="none",
        masked_reduce=False,
        p=1,
        return_clashing_pairs=True,
        sequence=sequence,
        **kwargs,
    )
    num_clashes = torch.sum(unreduced_loss > 0).item()
    # compute number of atom pairs - backbone-backbone pairs are ignored
    num_atoms = torch.sum(atom_mask).item()
    num_bb_atoms = torch.sum(atom_mask[..., :4]).item()
    choose_2 = lambda x: x * (x - 1) / 2

    return dict(
        steric=dict(
            num_clashes=num_clashes,
            num_atom_pairs=int(choose_2(num_atoms) - choose_2(num_bb_atoms)),
            clashing_pairs=clash_pairs,
            energy=torch.sum(unreduced_loss),
        )
    )


class RotamerProjection(nn.Module):  # noqa
    """convert side chain dihedral angles to coordinates"""

    @typechecked
    def __init__(
        self,
        atom_coords: TensorType["batch", "res", "atom", 3],
        sequence: TensorType["batch", "res", torch.long],
        atom_mask: TensorType["batch", "res", "atom", torch.bool],
        use_input_backbone: bool = True,
    ) -> None:
        """
        Parameters:
            atom_coords: backbone and side chain atom coordinates
            sequence: residue sequence
            atom_mask: mask indicating, for each residue, which atoms exist
            use_input_backbone: whether to use the backbone given as input, or also
            impute backbone atom coordinates by iotimizing bb-hihedral angles.
        """
        super(RotamerProjection, self).__init__()
        atom_coords = align_starting_coords(atom_coords.clone(), sequence, atom_mask)
        self.sequence = sequence
        self.atom_coords = atom_coords.detach()
        self.atom_mask = atom_mask
        # rigid frame defined by (fixed) backbone atoms
        N, CA, C, *_ = atom_coords.unbind(dim=-2)
        self.bb_rigids = Rigid.make_transform_from_reference(N, CA, C).detach()
        # differentiable mapping from side chain dihedrals to side chain atom coordinates
        self.coord_module = FACoordModule(
            use_persistent_buffers=True,
            predict_angles=False,
            replace_bb=True,
        )
        # Set initial dihedrals to those computed from initial s.c. atoms
        ptn = dict(aatype=sequence, all_atom_positions=atom_coords, all_atom_mask=atom_mask)
        torsion_info = atom37_to_torsion_angles(ptn)
        self.initial_angles = safe_normalize(torsion_info["torsion_angles_sin_cos"].detach())
        self.angle_mask = torsion_info["torsion_angles_mask"].detach().bool()
        self.initial_angle_alt_gt = safe_normalize(torsion_info["alt_torsion_angles_sin_cos"].detach())

        # angle parameter to optimize over
        self.dihedrals = nn.Parameter(self.initial_angles.clone(), requires_grad=True)
        self._dihedral_loss = SideChainDihedralLoss()

    def dihedral_deviation_loss(self):
        norm_angles = safe_normalize(self.dihedrals)
        angle_dev_loss, _ = self._dihedral_loss._forward(
            a=norm_angles,
            a_gt=self.initial_angles,
            a_alt_gt=self.initial_angle_alt_gt,
            mask=self.angle_mask,
        )
        return angle_dev_loss

    def forward(self) -> Tensor:
        """impute atom coordinates from self.dihedrals"""
        angles = safe_normalize(self.dihedrals)
        coord_dat = self.coord_module.forward(
            seq_encoding=self.sequence,
            residue_feats=None,
            coords=self.atom_coords,
            rigids=self.bb_rigids,
            angles=angles,
        )
        return coord_dat["positions"]

    def normalize_dihedrals(self):
        self.dihedrals = nn.Parameter(safe_normalize(self.dihedrals.data), requires_grad=True)


class SingleSeqStericLoss(nn.Module):
    """Steric loss function

    Pre-computes vdw data. Use when computing on same sequence mutliple times
    """

    def __init__(
        self,
        atom_mask: TensorType["batch", "res", "atom", torch.bool],
        sequence: Optional[TensorType["batch", "res", torch.bool]],
        hbond_allowance: float = 0.6,
        global_allowance: float = 0.1,
        global_tol_frac: float = 1,
        reduction: str = "sum",
        p: int = 2,
    ):
        super(SingleSeqStericLoss, self).__init__()
        self.atom_mask = atom_mask
        self.sequence = sequence
        self.hbond_allowance = hbond_allowance
        self.global_allowance = global_allowance
        self.reduction = reduction
        self.global_tol_frac = global_tol_frac
        self.p = p

    @cached_property
    def table_and_mask(self):
        table, mask = compute_pairwise_vdw_table_and_mask(
            sequence=self.sequence,
            atom_mask=self.atom_mask,
            global_allowance=self.global_allowance,
            hbond_allowance=self.hbond_allowance,
            global_tol_frac=self.global_tol_frac,
        )
        return table, mask

    def forward(self, coords: TensorType["batch", "res", "atoms", 3]) -> Tensor:
        table, mask = self.table_and_mask
        return _compute_steric_loss(
            atom_coords=coords,
            atom_mask=self.atom_mask,
            vdw_mask=mask,
            vdw_table=table,
            masked_reduce=True,
            reduction=self.reduction,
            p=self.p,
        )


@typechecked
def get_losses(
    predicted_coords: TensorType["batch", "res", "atom", 3],
    target_coords: TensorType["batch", "res", "atom", 3],
    atom_mask: TensorType["batch", "res", "atom"],
    sequence: TensorType["batch", "res"],
    compute_steric: bool = True,
    steric_loss_fn: Optional[Union[MemEfficientSingleSeqStericLoss, SingleSeqStericLoss]] = None,
    rotamer_proj: RotamerProjection = None,
):
    """Gets rmsd and steric loss"""
    rmsd = compute_sc_rmsd(
        coords=predicted_coords,
        target_coords=target_coords,
        atom_mask=atom_mask,
        per_residue=False,
        align_coords=False,
        sequence=sequence,
    )
    steric = (
        steric_loss_fn(
            coords=predicted_coords,
        )
        if compute_steric
        else torch.zeros(1)
    )

    dihedral_dev_loss = torch.zeros(1)
    if exists(rotamer_proj):
        dihedral_dev_loss = rotamer_proj.dihedral_deviation_loss()

    return rmsd, steric, dihedral_dev_loss


def print_verbose(msg, verbose):
    if verbose:
        print(msg)


@typechecked
def project_onto_rotamers(
    atom_coords: TensorType["batch", "res", "atom", 3],
    sequence: TensorType["batch", "res", torch.long],
    atom_mask: TensorType["batch", "res", "atom", torch.bool],
    optim_repeats: int = 2,
    use_input_backbone: bool = True,
    steric_clash_weight: Optional[Union[float, List[float]]] = 0,
    optim_kwargs: Optional[Dict] = None,
    steric_loss_kwargs: Optional[Dict] = None,
    device="cpu",
    optim_ty: str = "LBFGS",
    max_optim_iters: int = 400,
    torsion_deviation_loss_wt: float = 0.0,
    verbose: bool = True,
) -> Tuple[TensorType["batch", "res", "atom", 3], TensorType["batch", "res", 7, 2]]:  # coords and dihedrals
    """Project residue sidechain coordinates to nearest rotamer

    Parameters:
        atom_coords:
            backbone and side chain atom coordinates
        sequence:
            encoding of residue sequence
        atom_mask:
            mask indicating, for each residue, which atoms exist
        use_input_backbone:
            whether to use the backbone given as input, or also
            impute backbone atom coordinates by optimizing bb-hihedral angles.
        steric_clash_weight:
            weight(s) to use for steric clash loss term. Either float or list of floats
            with same length as optim_repeats
        optim_repeats:
            number of times to run optimization loop (Recommended >=2)
        optim_kwargs:
            keyword arguments to pass to torch.LBFGS optimizer (default recommended)
        steric_loss_kwargs:
            keyword arguments to pass to SingleSequenceStericLoss (default recommended)
        torsion_deviation_loss_wt:
            penalize deviation of projected torsion angles from the initial torsion angles
            derived from input coordinates.
    """
    _optim_kwargs = dict()
    if optim_ty == "LBFGS":
        _optim_kwargs = dict(lr=0.001, max_iter=max_optim_iters, line_search_fn="strong_wolfe")
        _optim_kwargs.update(default(optim_kwargs, dict()))
    atom_coords, sequence, atom_mask = map(lambda x: x.to(device), (atom_coords, sequence, atom_mask))

    module = RotamerProjection(
        atom_coords=atom_coords,
        sequence=sequence,
        atom_mask=atom_mask,
        use_input_backbone=use_input_backbone,
    ).to(device)

    print_verbose(f"[fn: project_onto_rotamers] : Using device {device}", verbose)
    steric_clash_weight = default(steric_clash_weight, 0)
    steric_loss_fn = (
        MemEfficientSingleSeqStericLoss(
            atom_mask=atom_mask, sequence=sequence, atom_coords=atom_coords, **default(steric_loss_kwargs, dict())
        )
        if steric_clash_weight != 0
        else None
    )

    print_verbose("[INFO] Beginning rotamer projection", verbose)
    initial_rmsd, initial_steric, dev_loss = get_losses(
        predicted_coords=module(),
        target_coords=atom_coords,
        atom_mask=atom_mask,
        steric_loss_fn=steric_loss_fn,
        compute_steric=exists(steric_loss_fn),
        sequence=sequence,
        rotamer_proj=module,
    )

    update_str = (
        f"[INFO] Initial loss values\n"
        f"   [RMSD loss] = {round(initial_rmsd.item(), 3)}\n"
        f"   [Steric loss] = {round(initial_steric.item(), 3)}\n"
        f"   [Angle Dev. loss] = {round(dev_loss.item(), 3)}\n"
    )
    print_verbose(update_str, verbose)

    # set steric clash weights for each optim. repeat
    steric_clash_weights = (
        steric_clash_weight if isinstance(steric_clash_weight, list) else [steric_clash_weight] * optim_repeats
    )
    steric_clash_weights = steric_clash_weights + [steric_clash_weights[-1]] * (
        optim_repeats - len(steric_clash_weights)
    )

    previous_loss = 0
    for eval_iter in range(optim_repeats):
        module.normalize_dihedrals()
        if optim_ty == "LBFGS":
            opt = torch.optim.LBFGS(module.parameters(), **_optim_kwargs)
        else:
            opt = torch.optim.Adam(module.parameters(), **_optim_kwargs)
        _optim_kwargs["lr"] = _optim_kwargs["lr"] * 0.9
        steric_clash_weight = steric_clash_weights[eval_iter]
        print_verbose(f"beginning iter: {eval_iter}, steric weight: {steric_clash_weight}", verbose)

        def loss_closure():
            """For LBFGS"""
            opt.zero_grad()
            rmsd_loss_val, steric_loss_val, dihedral_dev = get_losses(
                predicted_coords=module(),
                target_coords=atom_coords,
                atom_mask=atom_mask,
                compute_steric=steric_clash_weight != 0,
                steric_loss_fn=steric_loss_fn,
                sequence=sequence,
                rotamer_proj=module,
            )
            loss_val = (
                rmsd_loss_val + (steric_loss_val * steric_clash_weight) + (dihedral_dev * torsion_deviation_loss_wt)
            )
            loss_val.backward()
            return loss_val

        if optim_ty == "LBFGS":
            opt.step(loss_closure)
        else:
            for _ in range(max_optim_iters):
                curr_loss = loss_closure()
                opt.step()
            if torch.abs((previous_loss - curr_loss)) < 0.01:
                break
            previous_loss = curr_loss

    final_coords = module()
    final_rmsd, final_steric, final_dev = get_losses(
        predicted_coords=final_coords,
        target_coords=atom_coords,
        atom_mask=atom_mask,
        steric_loss_fn=steric_loss_fn,
        compute_steric=exists(steric_loss_fn),
        sequence=sequence,
        rotamer_proj=module,
    )

    update_str = (
        f"[INFO] Final Loss Values\n"
        f"   [RMSD loss] = {round(final_rmsd.item(), 3)}\n"
        f"   [Steric loss] = {round(final_steric.item(), 3)}\n"
        f"   [Angle Dev. loss] = {round(final_dev.item(), 3)}\n"
    )
    print_verbose(update_str, verbose)
    return final_coords, module.dihedrals


# TODO: Add iterative projection (project onto rotamers, average with initial, repeat)


@typechecked
def iterative_project_onto_rotamers(
    atom_coords: TensorType["batch", "res", "atom", 3],
    sequence: TensorType["batch", "res", torch.long],
    atom_mask: TensorType["batch", "res", "atom", torch.bool],
    iterative_refine_max_iters: int = 1,
    alphas: Union[float, List[float]] = 0.5,
    override_atoms: Optional[List[str]] = None,
    ignore_atoms: Optional[List[str]] = None,
    override_alpha=0.25,
    device="cpu",
    *args,
    **kwargs,
) -> Tuple[TensorType["batch", "res", "atom", 3], TensorType["batch", "res", 7, 2]]:  # coords and dihedrals
    """Iteratively Project residue sidechain coordinates to nearest rotamer

    Calls project_onto_rotamers(*args,**kwargs) 'iterative_refine_max_iters' times, between each
    round, the current coordinate prediction is averaged with the coordinate predictions
    of the previous round as (alpha*prev) + (1-alpha)*current

    Atom types in 'override_atoms' will be set to projected rotamer (not averaged).
    """
    print("[iterative_project_onto_rotamers] Iteratively Relaxing Sidechain Coordinates")

    atom_coords, sequence, atom_mask = map(lambda x: x.to(device), (atom_coords, sequence, atom_mask))
    prev_atom_coords = atom_coords.clone()
    alphas = alphas if isinstance(alphas, list) else alphas

    ignore_mask = torch.zeros_like(sequence).bool()
    for atom_ty in default(ignore_atoms, []):
        atom_ty_mask = sequence == pc.AA_TO_INDEX[atom_ty]
        ignore_mask[atom_ty_mask] = True

    for project_iter in range(iterative_refine_max_iters):
        print(f"[iterative_project_onto_rotamers] Beginning Round {project_iter}")
        coords, dihedrals = project_onto_rotamers(
            *args,
            atom_coords=prev_atom_coords,
            sequence=sequence,
            atom_mask=atom_mask,
            device=device,
            **kwargs,
        )
        aligned_next = align_symmetric_sidechains(
            native_coords=coords.detach(),
            predicted_coords=prev_atom_coords.detach(),
            atom_mask=atom_mask,
            native_seq=sequence,
        )
        alpha = alphas[min(len(alphas) - 1, project_iter)]

        prev_atom_coords[~ignore_mask] = (
            alpha * prev_atom_coords[~ignore_mask].detach() + (1 - alpha) * aligned_next.detach()[~ignore_mask]
        )

    for atom_ty in default(override_atoms, []):
        atom_ty_mask = sequence == pc.AA_TO_INDEX[atom_ty]
        x0, xt = prev_atom_coords[atom_ty_mask], aligned_next[atom_ty_mask]
        prev_atom_coords[atom_ty_mask] = override_alpha * x0 + (1 - override_alpha) * xt
    return prev_atom_coords, dihedrals


class MemEfficientSingleSeqStericLoss(nn.Module):
    """More efficient version of SingleSeqStericLoss

    sometimes efficiency comes at the cost of 'understandability' :()
    """

    def __init__(
        self,
        atom_mask: TensorType["batch", "res", "atom", torch.bool],
        sequence: Optional[TensorType["batch", "res", torch.bool]],
        atom_coords: Optional[TensorType["batch", "res", "atom", 3]] = None,
        hbond_allowance: float = 0.6,
        global_allowance: float = 0.1,
        global_tol_frac: float = 1,
        reduction: str = "sum",
        p: int = 2,
        top_k: int = 32,
    ):
        super(MemEfficientSingleSeqStericLoss, self).__init__()
        self.atom_mask = atom_mask
        self.sequence = sequence
        self.hbond_allowance = hbond_allowance
        self.global_allowance = global_allowance
        self.reduction = reduction
        self.global_tol_frac = global_tol_frac
        self.p = p
        valid_coords = atom_coords[atom_mask].unsqueeze(0)
        self.nbr_indices = get_neighbor_info(valid_coords, max_radius=1e10, top_k=top_k).indices

    @cached_property
    def table_and_mask(self):
        table, mask = compute_pairwise_vdw_table_and_mask(
            sequence=self.sequence,
            atom_mask=self.atom_mask,
            global_allowance=self.global_allowance,
            hbond_allowance=self.hbond_allowance,
            global_tol_frac=self.global_tol_frac,
        )
        table = batched_index_select(table, self.nbr_indices.squeeze())
        mask = batched_index_select(mask, self.nbr_indices.squeeze())
        return table, mask

    def forward(self, coords: TensorType["batch", "res", "atoms", 3]) -> Tensor:
        table, mask = self.table_and_mask
        masked_coords = coords[self.atom_mask]
        nbr_coords = masked_coords.unsqueeze(-2) - batched_index_select(
            masked_coords, self.nbr_indices.squeeze(), dim=0
        )
        relative_dists = safe_norm(nbr_coords, dim=-1)

        vdw_loss = F.relu(table - relative_dists, inplace=False) ** self.p
        # conside only inter-residue interactions, between residue_i and residue_j where i>j
        vdw_loss[~mask] = 0
        vdw_loss = vdw_loss[vdw_loss > 0]
        zero_loss = relative_dists.new_zeros(1)
        if self.reduction == "none":
            loss = vdw_loss
        elif self.reduction == "mean":
            loss = torch.mean(vdw_loss) if len(vdw_loss) > 0 else zero_loss  # grad can be nan o.w.
        elif self.reduction == "sum":
            loss = torch.sum(vdw_loss) if len(vdw_loss) > 0 else zero_loss  # grad can be nan o.w.
        else:
            raise Exception(f"reduction {self.reduction} not in ['mean','sum','none']")

        return loss
