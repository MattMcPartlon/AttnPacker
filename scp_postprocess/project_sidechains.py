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

"""Helper Functions"""
def masked_mean(
    tensor: Tensor, 
    mask: Optional[Tensor], 
    dim: Union[Tuple,int]=-1, 
    keepdim:bool=False
    ):
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
    mean = tensor.sum(dim=dim, keepdim=keepdim) / total_el.clamp(min=1.)
    return mean.masked_fill(total_el == 0, 0.)

def compute_sc_rmsd(
    coords: TensorType["batch","res", "atom", 3],
    target_coords: TensorType["batch","res", "atom", 3],
    atom_mask: TensorType["batch","res", "atom", torch.bool],
    sequence: TensorType["batch","res", torch.long],
    per_residue: bool,
    align_coords = False,
    )-> Union[TensorType["batch","res"],TensorType["batch"]]:
    """Compute Side-Chain RMSD
    
    Parameters:
        per_residue:
            whether to mean side-chain RMSD for entire chain, or for each individual residue.

    NOTE:
        Assumes atoms are given in the order defined by pc.ATOM_TYS
    """
    aln_target_coords, aln_coords = target_coords.clone(), coords.clone()
    if align_coords:
        #align ground-truth and native based on per-residue backbone frames
        # rotate each residue using backbone xN-CA C-CA displacements to define coordinate system
        target_frames, frames = map(lambda x: SimpleRigids.RigidFromBackbone(x[...,:3,:]), (target_coords,coords))
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
    atom_deviations = torch.sum(torch.square(aligned_target-aln_coords),dim=-1)
    # disregard backbone atoms
    per_residue_rmsd = torch.sqrt(masked_mean(atom_deviations[...,4:],mask=atom_mask[...,4:],dim=-1))
    res_mask = torch.any(atom_mask[...,4:],dim=-1)
    return per_residue_rmsd if per_residue else torch.mean(per_residue_rmsd[res_mask],dim=-1)

def rmsd_loss(predicted_coords, target_coords, atom_mask):
    """Compute RMSD loss"""
    coord_deviations = torch.sum(torch.square(predicted_coords - target_coords), dim=-1)[atom_mask]
    return torch.sqrt(torch.clamp_min(torch.mean(coord_deviations), 1e-10))

def item(x : Any):
    if torch.is_tensor(x):
        if len(x) == 0:
            return x.item()


def compute_pairwise_vdw_table_and_mask(
        sequence: TensorType["batch","res",torch.long], 
        atom_mask: TensorType["batch","res",37, torch.bool], 
        global_allowance: float = 0.1, 
        hbond_allowance: float = 0.6
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
                T[i,j] = rVDW[i] + rVDW[j] – allowance[i,j]
            (2) Mask of the form:
                M[i,j] = True if and only if steric overlap should be computed for pair (i,j)

        """

        b, n, a, device = *atom_mask.shape[:3], atom_mask.device  # noqa (batch, res, atoms)
        # (a,) vdw radii for each atom 1..a
        atom_vdw_radii = [pc.VAN_DER_WAALS_RADII[atom[:1]] - (global_allowance/2) for atom in pc.ALL_ATOMS[:a]]
        atom_vdw_radii = repeat(torch.tensor(atom_vdw_radii, device=device), "a -> b n a", b=b, n=n)

        # mapping of each atom to residue index for masking (b,n,a)
        atom_res_indices = repeat(torch.arange(n,device=device), "i -> b i a ", b=b, a=a)

        # apply atom mask (should save a lot of memory :)) -> (N,), where N is number of valid atoms
        all_atom_vdw_radii = atom_vdw_radii[atom_mask]  # N,
        all_atom_res_indices = atom_res_indices[atom_mask]  # N,

        # (1) Calculate initial pairwise VDW sums and pairwise mask
        to_pairwise_diffs = lambda x, y=None: (
            rearrange(x, "N ...-> N () ...") - rearrange(default(y, x), "N ...-> () N ...")
        )
        pairwise_vdw_sums = to_pairwise_diffs(all_atom_vdw_radii, -1 * all_atom_vdw_radii)  # N,N
        #keep only inter-residue atom pairs, excluding lower triangle
        vdw_pair_mask = to_pairwise_diffs(all_atom_res_indices) > 0  # N,N

        # (2) mask out BB-BB interactions
        bb_atom_types = pc.BB_ATOMS + ["CB"] # CB placement is completely determined by given backbone conformation
        bb_atom_types = set(pc.ALL_ATOMS).intersection(set(bb_atom_types))
        bb_atom_posns = [pc.ALL_ATOMS.index(atom) for atom in bb_atom_types]
        bb_mask = torch.zeros_like(atom_mask).float()
        # backbone atoms have 1, all others set to 0
        bb_mask[:, :, bb_atom_posns] = 1
        bb_mask = bb_mask[atom_mask]  # indicates which atoms are backbone
        bb_pair_mask = (bb_mask.unsqueeze(-1) - bb_mask.unsqueeze(0)) == 0
        #update vdw pair mask
        vdw_pair_mask = vdw_pair_mask & ~bb_pair_mask
        

        # (3) Correct with tolerance for H-Bonds
        hbd, hba = pc.HBOND_DONOR_TENSOR, pc.HBOND_ACCEPTOR_TENSOR
        hbd, hba = map(lambda x: repeat(x.to(device),"a -> () n a",n=n)[atom_mask], (hbd, hba))
        hbond_allowance_mask = torch.einsum("i,j->ij", hbd,hba).bool() 
        hbond_allowance_mask = torch.logical_or(hbond_allowance_mask, hbond_allowance_mask.T)
        pairwise_vdw_sums[hbond_allowance_mask] = pairwise_vdw_sums[hbond_allowance_mask] - hbond_allowance

        # (4) now for proline, CD is bonded with BB nitrogen - clashes should be ignored 
        # between this and next residue
        is_proline = sequence == pc.AA_INDEX_MAP["P"]
        before_or_after_proline = is_proline.clone().float()
        before_or_after_proline[:,1:] +=  is_proline[:,:-1]
        before_or_after_proline[:,:-1] += is_proline[:,1:]
        before_or_after_proline = before_or_after_proline.bool().unsqueeze(-1)

        #mask the PRO CD -> next or prev. backbone interactions
        pro_mask = torch.zeros_like(atom_mask).float()
        CD_mask, bb_mask = torch.zeros_like(atom_mask),torch.zeros_like(atom_mask)
        CD_mask[...,pc.ALL_ATOM_POSNS["CD"]], bb_mask[...,:4] = True, True

        pro_mask[before_or_after_proline & bb_mask] = 1
        pro_mask[is_proline.unsqueeze(-1)& CD_mask] = 1
        pro_mask = torch.einsum("i,j->ij", pro_mask[atom_mask],pro_mask[atom_mask]).bool()
        vdw_pair_mask = vdw_pair_mask & ~pro_mask

        #Finally, do not penalize disulfide bridges
        SG_mask = torch.zeros_like(atom_mask)
        SG_mask[...,pc.ALL_ATOM_POSNS["SG"]] = True
        SG_mask = torch.einsum("i,j->ij", SG_mask[atom_mask],SG_mask[atom_mask]).bool()
        vdw_pair_mask = vdw_pair_mask & ~SG_mask

        # return updated vdw sums and pair mask
        return pairwise_vdw_sums, vdw_pair_mask

def _compute_steric_loss(
    atom_coords: TensorType["batch","res","atom",3],
    atom_mask:  TensorType["batch","res","atom",torch.bool],
    vdw_table: TensorType["N","N"], # N is total number of atoms
    vdw_mask: TensorType["N","N",torch.bool],
    return_clashing_pairs: bool = False,
    masked_reduce: bool = True,
    reduction: str = "sum",  
    p: int = 2, 
    tol: float = 0.1,
    )->Union[Tensor,Tuple[Tensor,List]]:
        # pairwise distances
        all_atom_coords = atom_coords[atom_mask]  # N, 3
        relative_dists = safe_norm(rearrange(all_atom_coords, "N ...-> N () ...") - \
            rearrange(all_atom_coords, "N ...-> () N ..."), dim=-1)
                
        vdw_loss = F.relu(vdw_table - relative_dists, inplace=False) ** p
        # conside only inter-residue interactions, between residue_i and residue_j where i>j
        vdw_loss[~vdw_mask] = 0
        violation_mask = vdw_loss > 0

        clashing_pairs = None
        if return_clashing_pairs:
            b, n, a, device = *atom_mask.shape[:3], atom_mask.device  # noqa (batch, res, atoms)
            # mapping of each atom to residue index for masking (b,n,a)
            atom_res_indices = repeat(torch.arange(n,device=device), "i -> b i a ", b=b, a=a)
            all_atom_res_indices = atom_res_indices[atom_mask]  # N,
            # mapping from residue to indices of each atom
            res_atom_indices = repeat(torch.arange(a,device=device), "i -> b n i",b=b,n=n)
            all_atom_atom_indices = res_atom_indices[atom_mask] # N,
            clashing_pairs = []
            for i,j in zip(*torch.where(violation_mask)):
                pos_i, pos_j = map(lambda x: (all_atom_res_indices[x], all_atom_atom_indices[x]),(i,j))
                pos_i, pos_j = map(lambda x: (x[0].item(), pc.ALL_ATOMS[x[1].item()]),(pos_i, pos_j))
                d_ij, vdw_sum_ij = relative_dists[i,j], vdw_table[i,j]+tol
                clashing_pairs.append((pos_i, pos_j, round(d_ij.item(),3), round(vdw_sum_ij.item(),3)))

        if masked_reduce:
            vdw_loss = vdw_loss[violation_mask]

        zero_loss = relative_dists.new_zeros(1)
        if reduction == "none":
            loss = vdw_loss
        elif reduction == "mean":
            loss =  torch.mean(vdw_loss) if len(vdw_loss) > 0 else zero_loss  # grad can be nan o.w.
        elif reduction == "sum":
            loss = torch.sum(vdw_loss) if len(vdw_loss) > 0 else zero_loss  # grad can be nan o.w.
        else:
            raise Exception(f"reduction {reduction} not in ['mean','sum','none']")

        return (loss, clashing_pairs) if return_clashing_pairs else loss


def steric_loss(
        atom_coords: TensorType["batch","res","atom",3],
        atom_mask:  TensorType["batch","res","atom",torch.bool],
        sequence: Optional[TensorType["batch","res",torch.bool]],
        return_clashing_pairs: bool = False,
        hbond_allowance: float = 0.6,
        global_allowance: float = 0.1,
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

        
def compute_clashes(atom_coords: Tensor, atom_mask: Tensor, sequence:Tensor, **kwargs) -> Tuple[int, int]:
    """Returns tuple : (number of clashes, number of valid atom pairs)"""
    unreduced_loss, clash_pairs = steric_loss(
        atom_coords=atom_coords, 
        atom_mask=atom_mask, 
        reduction="none",
        masked_reduce=False,
        p=1, 
        return_clashing_pairs=True, 
        sequence=sequence, 
        **kwargs
    )
    num_clashes = torch.sum(unreduced_loss > 0).item()
    # compute number of atom pairs - backbone-backbone pairs are ignored
    num_atoms = torch.sum(atom_mask).item()
    num_bb_atoms = torch.sum(atom_mask[...,:4]).item()
    choose_2 = lambda x: x*(x-1)/2

    return dict(
        steric = dict(
            num_clashes = num_clashes,
            num_atom_pairs = int(choose_2(num_atoms)-choose_2(num_bb_atoms)),
            clashing_pairs = clash_pairs,
        )
    )
    

class RotamerProjection(nn.Module):  # noqa
    """convert side chain dihedral angles to coordinates"""
    @typechecked
    def __init__(
        self, 
        atom_coords: TensorType["batch","res","atom",3], 
        sequence: TensorType["batch","res", torch.long], 
        atom_mask: TensorType["batch","res","atom",torch.bool], 
        use_input_backbone: bool = False
        )-> None:
        """
        Parameters:
            atom_coords: backbone and side chain atom coordinates 
            sequence: residue sequence
            atom_mask: mask indicating, for each residue, which atoms exist
            use_input_backbone: whether to use the backbone given as input, or also
            impute backbone atom coordinates by iotimizing bb-hihedral angles.
        """
        super(RotamerProjection, self).__init__()

        # Set initial dihedrals to those computed from initial s.c. atoms
        ptn = dict(aatype=sequence, all_atom_positions=atom_coords, all_atom_mask=atom_mask)
        initial_dihedrals = atom37_to_torsion_angles(ptn)["torsion_angles_sin_cos"].detach()
        
        # initial angles, and angle parameter to optimize over
        self.initial_angles = initial_dihedrals.clone()
        self.dihedrals = nn.Parameter(initial_dihedrals.clone(), requires_grad=True)

        # rigid frame defined by (fixed) backbone atoms
        N, CA, C, *_ = atom_coords.unbind(dim=-2)
        self.bb_rigids = Rigid.make_transform_from_reference(N, CA, C).detach()
        # differentiable mapping from side chain dihedrals to side chain atom coordinates
        self.coord_module = FACoordModule(
            use_persistent_buffers=True,
            predict_angles=False,
            replace_bb = use_input_backbone
        )

        self.sequence = sequence
        self.atom_coords = atom_coords.detach()

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
        self.dihedrals = nn.Parameter(safe_normalize(self.dihedrals.data),requires_grad=True)

class SingleSeqStericLoss(nn.Module):
    """Steric loss function 

    Pre-computes vdw data. Use when computing on same sequence mutliple times
    """
    def __init__(
        self,
        atom_mask:  TensorType["batch","res","atom",torch.bool],
        sequence: Optional[TensorType["batch","res",torch.bool]],
        hbond_allowance: float = 0.6,
        global_allowance: float = 0.1,
        reduction: str = "sum", 
        p: int = 2, 
    ):
        super(SingleSeqStericLoss, self).__init__()
        self.atom_mask = atom_mask
        self.sequence = sequence
        self.hbond_allowance = hbond_allowance
        self.global_allowance = global_allowance
        self.reduction = reduction
        self.p=p
    
    @cached_property
    def table_and_mask(self):
        return compute_pairwise_vdw_table_and_mask(
            sequence=self.sequence,
            atom_mask=self.atom_mask,
            global_allowance=self.global_allowance,
            hbond_allowance=self.hbond_allowance,
        )

    def forward(self, coords: TensorType["batch","res","atoms",3])->Tensor:
        table,mask = self.table_and_mask
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
    predicted_coords: TensorType["batch","res","atom",3],
    target_coords: TensorType["batch","res","atom",3],
    atom_mask: TensorType["batch","res","atom"],
    compute_steric: bool = True,
    steric_loss_fn: Optional[SingleSeqStericLoss] = None,
    ):
    """Gets rmsd and steric loss"""
    rmsd = rmsd_loss(
        predicted_coords=predicted_coords, 
        target_coords=target_coords, 
        atom_mask=atom_mask
    )
    steric = (
        steric_loss_fn(
            coords=predicted_coords,
        )
        if compute_steric
        else 0
    )
    return rmsd, steric

@typechecked
def project_onto_rotamers(
    atom_coords: TensorType["batch","res","atom",3], 
    sequence: TensorType["batch","res", torch.long], 
    atom_mask: TensorType["batch","res","atom",torch.bool], 
    optim_repeats: int = 2,
    use_input_backbone: bool = True,
    steric_clash_weight: Union[float, List[float]] = 0,
    optim_kwargs: Optional[Dict] = None,
    steric_loss_kwargs: Optional[Dict] = None,
    use_cuda: bool = True,
)-> Tuple[TensorType["batch","res","atom",3],TensorType["batch","res",7,2]]: # coords and dihedrals
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
    
    """
    _optim_kwargs = dict(max_iter=150, lr=1e-3, line_search_fn="strong_wolfe")
    _optim_kwargs.update(default(optim_kwargs, dict()))

    device = "cuda:0" if (use_cuda and torch.cuda.is_available()) else "cpu"
    
    atom_coords, sequence, atom_mask = map(lambda x: x.to(device), (atom_coords, sequence, atom_mask))

    module = RotamerProjection(
        atom_coords=atom_coords,
        sequence=sequence,
        atom_mask=atom_mask,
        use_input_backbone=use_input_backbone,
    ).to(device)

    steric_loss_fn = SingleSeqStericLoss(
        atom_mask=atom_mask,
        sequence=sequence,
        **default(steric_loss_kwargs,dict())
    )


    print("[INFO] Beginning rotamer projection")
    initial_rmsd, initial_steric = get_losses(
        predicted_coords=module(),
        target_coords=atom_coords,
        atom_mask=atom_mask,
        steric_loss_fn=steric_loss_fn,
        compute_steric=True,
    )

    update_str = (
        f"[INFO] Initial loss values\n"
        f"   [RMSD loss] = {round(initial_rmsd.item(), 3)}\n"
        f"   [Steric loss] = {round(initial_steric.item(), 3)}\n"
    )
    print(update_str)

    #set steric clash weights for each optim. repeat
    steric_clash_weights = (
        [steric_clash_weight] * optim_repeats if isinstance(steric_clash_weight, float) else steric_clash_weight
    )
    steric_clash_weights = steric_clash_weights + [steric_clash_weights[-1]] * (
        optim_repeats - len(steric_clash_weights)
    )

    for eval_iter in range(optim_repeats):
        module.normalize_dihedrals()
        opt = torch.optim.LBFGS(module.parameters(), **_optim_kwargs)
        _optim_kwargs["lr"] = _optim_kwargs["lr"]*0.5
        steric_clash_weight = steric_clash_weights[eval_iter]
        print(f"beginning iter: {eval_iter}, steric weight: {steric_clash_weight}")

        def loss_closure():
            """For LBFGS"""
            opt.zero_grad()
            rmsd_loss_val, steric_loss_val = get_losses(
                predicted_coords=module(),
                target_coords=atom_coords,
                atom_mask=atom_mask,
                compute_steric=steric_clash_weight != 0,
                steric_loss_fn=steric_loss_fn,
            )
            loss_val = rmsd_loss_val + (steric_loss_val * steric_clash_weight)
            loss_val.backward()
            return loss_val
        
        opt.step(loss_closure)

    final_coords = module()
    final_rmsd, final_steric = get_losses(
        predicted_coords=final_coords,
        target_coords=atom_coords,
        atom_mask=atom_mask,
        steric_loss_fn=steric_loss_fn,
    )

    update_str = (
        f"[INFO] Final Loss Values\n"
        f"   [RMSD loss] = {round(final_rmsd.item(), 3)}\n"
        f"   [Steric loss] = {round(final_steric.item(), 3)}\n"
    )
    print(update_str)
    return final_coords, module.dihedrals
