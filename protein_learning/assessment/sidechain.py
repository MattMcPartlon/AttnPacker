from protein_learning.protein_utils.sidechains.project_sidechains import (
    compute_sc_rmsd,
    compute_clashes
)
from protein_learning.common.data.data_types.protein import Protein
import torch
import protein_learning.common.protein_constants as pc
from protein_learning.common.helpers import safe_normalize, default, masked_mean
from protein_learning.common.data.datasets.utils import set_canonical_coords_n_masks
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
from protein_learning.protein_utils.sidechains.sidechain_rigid_utils import atom37_to_torsion_angles
from protein_learning.protein_utils.sidechains.sidechain_utils import align_symmetric_sidechains, get_sc_dihedral
from typing import List
import numpy as np
import math
from protein_learning.protein_utils.align.kabsch_align import kabsch_align
PI = math.pi
patch_typeguard()


@typechecked
def compute_degree_centrality(
    coords: TensorType[...,"res","atom",3],
    atom_ty: str ="CB",
    dist_thresh: float = 10,
    ):
    atom_pos = pc.ALL_ATOM_POSNS[atom_ty]
    atom_coords = coords[...,atom_pos,:]
    pairwise_dists = torch.cdist(atom_coords,atom_coords)
    return torch.sum(pairwise_dists<dist_thresh,dim=-1)-1

@typechecked
def gather_dihedral_info(
    predicted_coords: TensorType["batch","res","atom",3], 
    target_coords: TensorType["batch","res","atom",3],
    atom_mask: TensorType["batch","res","atom"],
    sequence: TensorType["batch","res", torch.long],
):
    out = dict()
    dihedral_info = lambda crds : get_sc_dihedral(
        coords = crds, seq=sequence, atom_mask=atom_mask, return_unit_vec=True, return_mask=True
    )
    dihedral_pred, chi_mask = dihedral_info(predicted_coords)
    dihedral_target, _ = dihedral_info(target_coords)

    dihedral_pred, dihedral_target = map(lambda x : safe_normalize(x),(dihedral_pred, dihedral_target))

    #account for pi-peridicity of residues such as ASP, GLU, PHE, TYR
    periodic_mask = torch.tensor(pc.CHI_PI_PERIODIC_LIST,device=sequence.device)[sequence].bool()
    alt_target = dihedral_target.clone()
    alt_target[periodic_mask] = -alt_target[periodic_mask]

    angle_diff = lambda a1,a2 : torch.sum(torch.square(a1-a2),dim=-1)
    diff, alt_diff = angle_diff(dihedral_pred, dihedral_target), angle_diff(dihedral_pred, alt_target)
    # swap for the unit-vec that is closer to the predicted for these residues
    dihedral_target[diff>alt_diff] = alt_target[diff>alt_diff]    

    #convert unit vec (sin,cos) to radians
    target_angle, pred_angle = map(
        lambda x: torch.atan2(*x.unbind(-1)),
        (
            dihedral_target, 
            dihedral_pred,
        )
    )

    out["predicted-angles"] = pred_angle
    out["target-angles"] = target_angle
    out["mask"] = chi_mask.bool()

    # compute angle MAE of angles (in radians)
    angle_diff = target_angle - pred_angle 
    # account for periodicity in the angle difference
    angle_diff[angle_diff>PI] = angle_diff[angle_diff>PI] - 2*PI
    angle_diff[angle_diff<-PI] = angle_diff[angle_diff<-PI] + 2*PI
    out["mae"] = angle_diff
    return out

@typechecked
def gather_dihedral_info_of(
    predicted_coords: TensorType["batch","res","atom",3], 
    target_coords: TensorType["batch","res","atom",3],
    atom_mask: TensorType["batch","res","atom"],
    sequence: TensorType["batch","res", torch.long],
):
    out = dict()
    
    # get dihedrals of predicted coordinates
    dihedral_info = lambda crds : atom37_to_torsion_angles(
        dict(aatype=sequence, all_atom_positions=crds, all_atom_mask=atom_mask)
    )
    dihedral_pred, dihedral_target = map(dihedral_info, (predicted_coords,target_coords))
    
    #account for pi-peridicity of residues such as ASP, GLU, PHE, TYR
    angle_diff = lambda key : torch.sum(torch.square(dihedral_pred["torsion_angles_sin_cos"]-dihedral_target[key]),dim=-1)
    diff, alt_diff = angle_diff("torsion_angles_sin_cos"), angle_diff("alt_torsion_angles_sin_cos")
    # swap for the unit-vec that is closer to the predicted for these residues
    dihedral_target["torsion_angles_sin_cos"][diff>alt_diff] = dihedral_target["alt_torsion_angles_sin_cos"][diff>alt_diff]

    #convert unit vec (sin,cos) to radians
    target_angle, pred_angle = map(
        lambda x: torch.atan2(*x.unbind(-1)),
        (
            dihedral_target["torsion_angles_sin_cos"][...,-4:,:], 
            dihedral_pred["torsion_angles_sin_cos"][...,-4:,:]
        )
    )
    out["predicted-angles"] = pred_angle
    out["target-angles"] = target_angle
    out["mask"] = dihedral_target["torsion_angles_mask"][...,-4:].bool()

    # compute angle MAE of angles (in radians)
    angle_diff = target_angle - pred_angle 
    # account for periodicity in the angle difference
    angle_diff[angle_diff>PI] = angle_diff[angle_diff>PI] - 2*PI
    angle_diff[angle_diff<-PI] = angle_diff[angle_diff<-PI] + 2*PI
    out["mae"] = angle_diff
    return out



@typechecked
def gather_target_stats(
    predicted_coords: TensorType["batch","res","atom",3], 
    target_coords: TensorType["batch","res","atom",3],
    atom_mask: TensorType["batch","res","atom"],
    sequence: TensorType["batch","res", torch.long],
    steric_tol_fracs: List[float] = None,
    steric_tol_allowance: float = 0,
    use_openfold_dihedral_calc: bool = False,
):
    steric_tol_fracs = default(steric_tol_fracs,np.linspace(0.4,1,7))
    output = dict(rmsd=dict())
    # per - residue RMSD and mask
    output["rmsd"]["per-res"] = compute_sc_rmsd(
        coords = predicted_coords,
        target_coords= target_coords,
        atom_mask=atom_mask,
        sequence=sequence,
        per_residue=True,
    )
    output["rmsd"]["per-res-mask"] = torch.any(atom_mask[...,4:],dim=-1)

    # number of CB neighbors in 10A ball
    output["centrality"] = compute_degree_centrality(target_coords)

    # steric clash info
    output["clash-info"] = {
        str(round(100*tol_frac)): 
        compute_clashes(
            predicted_coords,
            atom_mask=atom_mask,
            sequence=sequence,
            global_allowance = steric_tol_allowance,
            steric_tol_frac = tol_frac,
        ) for tol_frac in steric_tol_fracs
    }

    # compute dihedral stats
    dihedral_fn = gather_dihedral_info_of if use_openfold_dihedral_calc else gather_dihedral_info
    output["dihedral"] = dihedral_fn(
        predicted_coords=predicted_coords,
        target_coords=target_coords,
        atom_mask=atom_mask,
        sequence=sequence,
    )

    #backbone RMSD (If comparing native and non-native backbone)
    ca_native, ca_target = map(lambda x: x[...,1,:].clone(),(target_coords,predicted_coords))
    # need to add small epsilon here so that matrix for SVD is non-singular
    ca_native, ca_target = kabsch_align(align_to=ca_target,align_from=ca_native+torch.randn_like(ca_native)*1e-7)
    output["ca_rmsd"] = torch.sqrt(torch.mean(torch.sum(torch.square(ca_native - ca_target),dim=-1)))
    return output

def map_tensor_fn_onto_iterable(data, fn):
    if isinstance(data,dict):
        return {k: map_tensor_fn_onto_iterable(v,fn) for k,v in data.items()}
    if isinstance(data,list):
        return [map_tensor_fn_onto_iterable(d,fn) for d in data]
    if isinstance(data,tuple):
        return tuple([map_tensor_fn_onto_iterable(d,fn) for d in data])    
    return fn(data)

def assess_sidechains(
    target_pdb_path: str,
    decoy_pdb_path: str,
    device="cpu",
    steric_tol_allowance: float = 0.05,
    steric_tol_fracs: List[float]=None,
    use_openfold_dihedral_calc: bool = True,
):
    device = device if torch.cuda.is_available() else "cpu"
    decoy_protein = Protein.FromPDBAndSeq(
        pdb_path = decoy_pdb_path,
        seq=None,
        atom_tys = pc.ALL_ATOMS,
        missing_seq = True,
        load_ss=False,
    )

    target_protein = Protein.FromPDBAndSeq(
        pdb_path = target_pdb_path,
        seq=decoy_protein.seq, # align with sequence of decoy
        atom_tys = pc.ALL_ATOMS,
        load_ss=False,
    )

    decoy_protein, target_protein = map(
        lambda x: set_canonical_coords_n_masks(x).to(device),
        (decoy_protein, target_protein)
    )
    
    #make sure we are comparing only on valid atoms
    decoy_protein.atom_masks = decoy_protein.atom_masks & target_protein.atom_masks
    target_protein.atom_masks = decoy_protein.atom_masks & target_protein.atom_masks

    # NOTE: Coordinates of CB in GLY are set to CA by default
    target_protein.atom_coords = align_symmetric_sidechains(
        target_protein.atom_coords.unsqueeze(0),
        predicted_coords=decoy_protein.atom_coords.unsqueeze(0),
        native_seq=decoy_protein.seq_encoding.unsqueeze(0),
        atom_mask=decoy_protein.atom_masks.unsqueeze(0)
    ).squeeze(0)

    stats = gather_target_stats(
        predicted_coords=decoy_protein.atom_coords.unsqueeze(0),
        target_coords=target_protein.atom_coords.unsqueeze(0),
        sequence=decoy_protein.seq_encoding.unsqueeze(0),
        atom_mask=decoy_protein.atom_masks.unsqueeze(0),
        steric_tol_fracs=steric_tol_fracs,
        steric_tol_allowance=steric_tol_allowance,
        use_openfold_dihedral_calc = use_openfold_dihedral_calc,
    )
    #add sequence and some meta-data
    stats["sequence"] = target_protein.seq
    stats["sequence-encoding"] = target_protein.seq_encoding
    stats["target-pdb"] = target_pdb_path
    stats["decoy-pdb"] = decoy_pdb_path

    
    #place data on cpu so we can save dict without pickling errors
    return map_tensor_fn_onto_iterable(
        stats,
        lambda x: x.detach().cpu().squeeze(0) if torch.is_tensor(x) else x
    )

def summarize(assess_stats):
    angle_diffs = torch.abs(assess_stats["dihedral"]["mae"]*(180/PI))
    mean_mae = masked_mean(torch.abs(angle_diffs),assess_stats["dihedral"]["mask"],dim=0)
    has_chi_mask = torch.any(assess_stats["dihedral"]["mask"],dim=-1)
    all_lt_20 = torch.sum(angle_diffs < 20, dim=-1) == 4
    mae_sr = torch.sum(all_lt_20[has_chi_mask])/torch.sum(has_chi_mask.float())
    mean_per_res_rmsd = torch.mean(assess_stats["rmsd"]["per-res"][has_chi_mask])
    dihedral_counts = torch.sum(assess_stats["dihedral"]["mask"],dim=0)
    steric_ks = "num_clashes num_atom_pairs energy".split()
    return dict(
        mean_mae = mean_mae,
        mae_sr = mae_sr,
        rmsd = mean_per_res_rmsd,
        dihedral_counts=dihedral_counts,
        clash_info = {
            tol:{
                k: assess_stats['clash-info'][tol]["steric"][k] for k in steric_ks
                } 
            for tol in assess_stats['clash-info']
        },
        seq_len = len(has_chi_mask),
        num_sc = torch.sum(has_chi_mask).item(),
        ca_rmsd = assess_stats["ca_rmsd"]
    )
