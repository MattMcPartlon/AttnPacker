import os
import sys
import subprocess

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
import torch
import numpy as np

torch.set_printoptions(
    precision=3,
    threshold=None,
    edgeitems=None,
    linewidth=None,
    profile=None,
    sci_mode=False,
)
from protein_learning.assessment.sidechain import assess_sidechains
from protein_learning.common.helpers import masked_mean
from protein_learning.protein_utils.sidechains.project_sidechains import (
    project_onto_rotamers,
    compute_sc_rmsd,
    compute_clashes,
)
from protein_learning.protein_utils.sidechains.sidechain_utils import align_symmetric_sidechains, swap_symmetric_atoms
from protein_learning.common.data.data_types.protein import Protein
import torch
import protein_learning.common.protein_constants as pc
from protein_learning.common.data.datasets.utils import set_canonical_coords_n_masks
from protein_learning.common.io.pdb_io import write_pdb
import math
from protein_learning.common.protein_constants import DISTAL_ATOM_MASK_TENSOR, AA_INDEX_MAP
from protein_learning.assessment.metrics import compute_coord_lddt
from multiprocessing import Pool
from functools import partial
from collections import defaultdict


def calc_native_plddt(tgt_stats):
    seq = torch.tensor([AA_INDEX_MAP[x] for x in tgt_stats["metadata"]["native_seq"]]).long()
    mask = DISTAL_ATOM_MASK_TENSOR[seq]
    pred, actual, c_mask = map(lambda x: tgt_stats["metadata"][x], "actual_coords decoy_coords coord_mask".split())
    pred, actual, c_mask = map(lambda x: torch.from_numpy(x), (pred, actual, c_mask))
    mask = c_mask & mask
    pred, actual = map(lambda x: x[mask].unsqueeze(0), (pred, actual))
    return compute_coord_lddt(predicted_coords=pred, actual_coords=actual, per_residue=True)


def masked_nsr_tgt(tgt_stats):
    nsr_stats = tgt_stats["nsr"]
    mask, true_labels, pred_labels = [nsr_stats[x] for x in "seq_mask true_labels pred_labels".split()]
    true_labels, pred_labels = map(lambda x: x[mask], (true_labels, pred_labels))
    return np.sum(true_labels == pred_labels) / len(pred_labels)


def summary_stats(values):
    qs = np.quantile(values, q=(0.25, 0.5, 0.75))
    m = np.mean(values)
    s = np.std(values)
    return qs.tolist() + [m] + [s]


def masked_nsr(stats):
    return summary_stats([masked_nsr_tgt(s) for s in stats])


def find_target_pdb(pdb_dir, target):
    decoy_pdb = None
    for tmp in os.listdir(pdb_dir):
        if tmp.startswith(target):
            if tmp.startswith(target + "-D"):
                continue
            if tmp.startswith(target + "s"):
                continue
            decoy_pdb = os.path.join(pdb_dir, tmp)
            break
    return decoy_pdb


def get_method_stats_fn(pdb_dir, targets, native_dir, overwrite=False):
    fix_pdb_names(pdb_dir)
    save_path = os.path.join(pdb_dir, "stats.npy")
    if os.path.exists(save_path) and (not overwrite):
        method_stats = np.load(save_path, allow_pickle=True).item()
    else:
        method_stats = {}
    for target in targets:
        try:
            if target in method_stats and not overwrite:
                continue
            native_pdb = os.path.join(native_dir, target + ".pdb")
            decoy_pdb = find_target_pdb(pdb_dir, target)
            if decoy_pdb is None:
                continue
            if not os.path.exists(native_pdb):
                print(f"        no path for native {native_pdb}")
                continue
            if not os.path.exists(decoy_pdb):
                # print(f"        no path for decoy {decoy_pdb}")
                continue
            method_stats[target] = assess_sidechains(
                target_pdb_path=native_pdb, decoy_pdb_path=decoy_pdb, steric_tol_allowance=0.01
            )
        except Exception as e:
            print("#################################")
            print(f"caught {e}", target + "\n", decoy_pdb + "\n", native_pdb + "\n")
    print(f"finished {pdb_dir}")
    np.save(save_path, method_stats)


def get_stats_for_tgt(assess_stats):
    angle_diffs = torch.abs(assess_stats["dihedral"]["mae"] * (180 / math.pi))
    mean_mae = masked_mean(torch.abs(angle_diffs), assess_stats["dihedral"]["mask"], dim=0)
    has_chi_mask = torch.any(assess_stats["dihedral"]["mask"], dim=-1)
    all_lt_20 = torch.sum(angle_diffs < 20, dim=-1) == 4
    mae_sr = (torch.sum(all_lt_20[has_chi_mask]) / has_chi_mask[has_chi_mask].numel(),)
    mean_per_res_rmsd = torch.mean(assess_stats["rmsd"]["per-res"][has_chi_mask])
    dihedral_counts = torch.sum(assess_stats["dihedral"]["mask"], dim=0)
    steric_ks = "num_clashes num_atom_pairs energy".split()
    # print([k for k in assess_stats['clash-info']["40"]])
    return dict(
        mean_mae=mean_mae,
        mae_sr=mae_sr,
        rmsd=mean_per_res_rmsd,
        dihedral_counts=dihedral_counts,
        clash_info={
            tol: {k: assess_stats["clash-info"][tol]["steric"][k] for k in steric_ks}
            for tol in assess_stats["clash-info"]
        },
        seq_len=len(has_chi_mask),
        res_len=torch.sum(has_chi_mask).item(),
        ca_rmsd=None,
    )


def safe_fn(args, fn, ret):
    try:
        fn(args)
    except:
        print(f"caught exception {args}... terminating")
        return ret


def get_all_terminal_subdirectories(root):
    out = []
    found_subdir = False
    for x in os.listdir(root):
        if x.startswith("."):
            continue
        nxt = os.path.join(root, x)
        if os.path.isdir(nxt):
            out = out + get_all_terminal_subdirectories(nxt)
            found_subdir = True
    if not found_subdir:
        return [root]
    return out


def get_all_pdb_folders(root, methods):
    subdirs = []
    for x in os.listdir(root):
        if x not in methods:
            continue
        subdirs = subdirs + get_all_terminal_subdirectories(os.path.join(root, x))
    pdb_fldrs = []
    for fldr in subdirs:
        for x in os.listdir(fldr):
            if "pdb" in x:
                pdb_fldrs.append(fldr)
                break
    return pdb_fldrs


def get_stats_for_tgt(assess_stats):
    angle_diffs = torch.abs(assess_stats["dihedral"]["mae"] * (180 / torch.pi))
    mean_mae = masked_mean(torch.abs(angle_diffs), assess_stats["dihedral"]["mask"], dim=0)
    has_chi_mask = torch.any(assess_stats["dihedral"]["mask"], dim=-1)
    all_lt_20 = torch.sum(angle_diffs < 20, dim=-1) == 4
    mae_sr = (torch.sum(all_lt_20[has_chi_mask]) / has_chi_mask[has_chi_mask].numel(),)
    mean_per_res_rmsd = torch.mean(assess_stats["rmsd"]["per-res"][has_chi_mask])
    dihedral_counts = torch.sum(assess_stats["dihedral"]["mask"], dim=0)
    clash_info = {}
    for tol in assess_stats["clash-info"]:
        clash_info[f"num_clashes_{tol}"] = assess_stats["clash-info"][tol]["steric"]["num_clashes"]

    return dict(
        mean_mae=mean_mae,
        mae_sr=mae_sr,
        rmsd=mean_per_res_rmsd,
        dihedral_counts=dihedral_counts,
        seq_len=len(has_chi_mask),
        res_len=torch.sum(has_chi_mask).item(),
        ca_rmsd=assess_stats["ca_rmsd"],
        **clash_info,
    )


def get_per_res_stats_for_target(assess_stats, surface=False, core=False):
    angle_diffs = torch.abs(assess_stats["dihedral"]["mae"] * (180 / torch.pi))
    has_chi_mask = torch.any(assess_stats["dihedral"]["mask"], dim=-1)
    all_lt_20 = torch.sum(angle_diffs < 20, dim=-1) == 4

    seq = assess_stats["sequence-encoding"]
    seq[~has_chi_mask] = -1
    per_res_mae, per_res_rmsd, per_res_sr = dict(), dict(), dict()

    for i in range(20):
        msk = assess_stats["sequence-encoding"] == i
        if torch.any(msk):
            per_res_mae[i] = angle_diffs[msk]
            per_res_rmsd[i] = assess_stats["rmsd"]["per-res"][msk]
            per_res_sr[i] = all_lt_20[msk]

    return dict(
        per_res_mae=per_res_mae,
        per_res_rmsd=per_res_rmsd,
        per_res_sr=per_res_sr,
        seq_len=len(has_chi_mask),
        res_len=torch.sum(has_chi_mask).item(),
        ca_rmsd=assess_stats["ca_rmsd"],
    )


def get_per_res_csv_row(pdb_fldr, method, benchmark, targets, min_len=2, ca_rmsd_thresh=2.5):
    stats_path = os.path.join(pdb_fldr, "stats.npy")
    assert os.path.exists(stats_path), f"{stats_path}"
    _method_stats, method_stats = np.load(stats_path, allow_pickle=True).item(), dict()

    # filter and convert format
    for tgt in targets:
        if tgt not in _method_stats:
            continue
        method_stats[tgt] = get_stats_for_tgt(_method_stats[tgt])

    if len(_method_stats) < min_len:
        return None

    # convert format again
    stat_keys = "per_res_mae per_res_rmsd per_res_sr".split()
    tmp = {k: defaultdict(list) for k in stat_keys}
    for tgt in method_stats:
        for k in stat_keys:
            for res_idx in method_stats[tgt][k]:
                tmp[k][res_idx].append(method_stats[tgt][k][res_idx])

    # average over all values
    header = ["method", "bm", "eval_ty", "N"] + ["rmsd", "surface", "core", "chi_1", "chi_2", "chi_3", "chi_4"]
    rows = []
    for k in stat_keys:
        row = [
            method,
            benchmark,
        ]
        for res_idx in tmp[k]:
            pass


def get_csv_row(pdb_fldr, method, benchmark, targets, min_len=2, ca_rmsd_thresh=2.5):
    stats_path = os.path.join(pdb_fldr, "stats.npy")
    assert os.path.exists(stats_path), f"{stats_path}"
    _method_stats, method_stats = np.load(stats_path, allow_pickle=True).item(), dict()

    # filter
    for tgt in targets:
        if tgt not in _method_stats:
            continue
        method_stats[tgt] = get_stats_for_tgt(_method_stats[tgt])

    if len(method_stats) < min_len:
        return None

    # convert format
    ks = "mae_sr mean_mae rmsd dihedral_counts res_len ca_rmsd".split()
    steric_ks = [f"num_clashes_frac_{r}" for r in map(str, (50, 60, 70, 80, 90, 100))]
    targets = list(method_stats.keys())
    tmp = defaultdict(list)
    for target in targets:
        for k in ks + steric_ks:
            tmp[k].append(method_stats[target][k])

    total_resi = sum([x for x in tmp["res_len"]])
    total_dihedrals = torch.sum(torch.stack(tmp["dihedral_counts"], dim=0), dim=0)
    total_mae = torch.sum(torch.stack([x * y for x, y in zip(tmp["mean_mae"], tmp["dihedral_counts"])], dim=0), dim=0)
    row = {}
    row["mae_sr"] = sum([x[0].item() * r for x, r in zip(tmp["mae_sr"], tmp["res_len"])]) / total_resi
    mae = total_mae / total_dihedrals
    for i in range(4):
        row[f"chi_{i+1}_mae"] = mae[i].item()
    row["rmsd"] = sum([x.item() * y for x, y in zip(tmp["rmsd"], tmp["res_len"])]) / total_resi
    for k in steric_ks:
        for q in (0.5, 0.75):
            row[k + f"_q_{round(100*q)}"] = np.quantile(tmp[k], q=q)
        row[k + f"_mean"] = np.mean(tmp[k])
    ks = sorted([k for k in row])
    eval_ty = os.path.basename(pdb_fldr)
    header = ["method", "bm", "eval_ty", "N"] + ks
    row = [method, benchmark, eval_ty, len(method_stats)] + [row[k] for k in ks]
    return header, row


def fix_pdb_names(fldr):
    for x in os.listdir(fldr):
        if not x.endswith(".pdb.pdb"):
            continue
        from_path = os.path.join(fldr, x)
        to_path = os.path.join(fldr, x[:-4])
        if os.path.exists(to_path):
            subprocess.call(f"rm {to_path}", shell=True)
            if os.path.exists(os.path.join(fldr, "stats.npy")):
                subprocess.call(f"rm {os.path.join(fldr,'stats.npy')}", shell=True)
        subprocess.call(f"mv {from_path} {to_path}", shell=True)


if __name__ == "__main__":
    RESULT_ROOT = "/mnt/local/mmcpartlon/sidechain_packing/standard_results/"
    methods = [x for x in os.listdir(RESULT_ROOT) if not x.startswith(".") and "npy" not in x]
    TARGETS = [x[:-4] for x in os.listdir(os.path.join(RESULT_ROOT, "native", "casp_all")) if x.endswith("pdb")]
    NATIVE_DIR = os.path.join(RESULT_ROOT, "native", "casp_all")
    fn = partial(get_method_stats_fn, targets=TARGETS, native_dir=NATIVE_DIR, overwrite=False)
    sf = partial(safe_fn, fn=fn, ret={})
    print("Gathering results for:")
    for m in methods:
        print(f"    {m}")
    # methods=["rx7_2"]
    pdb_fldrs = get_all_pdb_folders(RESULT_ROOT, methods)
    for x in pdb_fldrs:
        print(x)

    with Pool(40) as p:
        sts = p.map(sf, pdb_fldrs)
