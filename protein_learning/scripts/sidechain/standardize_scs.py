import sys
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
import subprocess
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from multiprocessing import Pool
import torch
from functools import partial

torch.set_printoptions(
    precision=3,
    threshold=None,
    edgeitems=None,
    linewidth=None,
    profile=None,
    sci_mode=False,
)
import numpy as np
from protein_learning.protein_utils.sidechains.project_sidechains import (
    project_onto_rotamers,
    iterative_project_onto_rotamers,
    default,
)
from typing import Optional, List, Union
from protein_learning.common.data.data_types.protein import Protein
import protein_learning.common.protein_constants as pc
from protein_learning.common.data.datasets.utils import set_canonical_coords_n_masks
import traceback
import math


RESULT_ROOT = "/mnt/local/mmcpartlon/design_with_plv2/sidechain_packing/"


def load_stats(method, res_fldr):
    stat_root = os.path.join(RESULT_ROOT, method, "stats", res_fldr)
    stat_file = [x for x in os.listdir(stat_root) if x.endswith("stats.npy")][0]
    return np.load(os.path.join(stat_root, stat_file), allow_pickle=True)


def standardize_pdbs_for_method(method, res_fldr):
    try:
        pdb_dir = os.path.join(RESULT_ROOT, method, "stats", res_fldr, "pdbs")
        return standardize_pdbs(pdb_dir)
    except:
        os.makedirs(os.path.join(RESULT_ROOT, method, "stats", res_fldr, "pdbs"), exist_ok=True)
        _pdb_dir = os.path.join(RESULT_ROOT, method, "stats", res_fldr)
        tmp = [x for x in os.listdir(_pdb_dir) if "npy" not in x and x.startswith("fbb")][0]
        subprocess.call(f"cp -r {os.path.join(_pdb_dir,tmp)} {pdb_dir}/", shell=True)
    return standardize_pdbs(pdb_dir)


def standardize_pdbs(pdb_dir):
    ignore = ". native decoy project".split()
    sub_folders = [x for x in os.listdir(pdb_dir) if not any([x.startswith(y) for y in ignore])]
    assert len(sub_folders) == 1, os.listdir(pdb_dir)
    native_names, decoy_names = [], []
    pdb_root = os.path.join(pdb_dir, sub_folders[0])
    all_pdbs = os.listdir(pdb_root)
    native_names = [x for x in all_pdbs if "native" in x]
    decoy_names = [x for x in all_pdbs if "native" not in x and x.endswith("pdb")]
    for names, new_dir in zip((native_names, decoy_names), ("native", "decoy")):
        save_dir = os.path.join(pdb_dir, new_dir)
        print(f"saving {new_dir} pdbs to : {save_dir}")
        os.makedirs(save_dir, exist_ok=True)
        from_paths = [os.path.join(pdb_root, x) for x in names]
        to_paths = [os.path.join(save_dir, x.split("_")[0] + ".pdb") for x in names]
        for to_path, from_path in zip(to_paths, from_paths):
            subprocess.call(f"cp {from_path} {to_path}", shell=True)
    return os.path.join(pdb_dir, "decoy")


"""
        projected_coords, projected_dihedrals = iterative_project_onto_rotamers(
            iterative_refine_max_iters = 2,
            override_atoms=["ASN","ARG","GLU","GLN","TRP","ASP","VAL"],
            override_alpha = 0.35,
            alphas = [0.75,0.65],
            atom_coords = se3_protein.atom_coords.unsqueeze(0).clone(),
            atom_mask = se3_protein.atom_masks.unsqueeze(0),
            sequence = se3_protein.seq_encoding.unsqueeze(0),
            optim_repeats = 4,
            steric_clash_weight = [0.1],
            steric_loss_kwargs = dict(global_tol_frac=0.9, reduction = "sum", p=2),
            optim_kwargs = dict(max_iter=200, lr=1e-3,line_search_fn="strong_wolfe"),
            use_input_backbone = True,
            torsion_deviation_loss_wt = 0.5,
            
        )
"""


def project_rotamers(
    target: str,
    pdb_dir: str,
    save_dir: str,
    steric_weight: List[float],
    torsion_loss_wt: float,
    steric_tol_allowance: float = 0.0,
    steric_tol_frac: Optional[float] = None,
    iterative_refine_max_iters: int = 1,
    alphas: Union[float, List[float]] = 0,
    device="cpu",
):
    os.makedirs(save_dir, exist_ok=True)
    if os.path.exists(os.path.join(save_dir, target)):
        return
    decoy = Protein.FromPDBAndSeq(
        pdb_path=os.path.join(pdb_dir, target),
        seq=None,
        atom_tys=pc.ALL_ATOMS,
        missing_seq=True,
    )
    decoy = set_canonical_coords_n_masks(decoy)
    print(f"Beginning inference on {decoy.name}, seq length : {len(decoy)}")

    projected_coords, _ = iterative_project_onto_rotamers(
        iterative_refine_max_iters=iterative_refine_max_iters,
        torsion_deviation_loss_wt=torsion_loss_wt,
        atom_coords=decoy.atom_coords.unsqueeze(0).clone(),
        atom_mask=decoy.atom_masks.unsqueeze(0),
        sequence=decoy.seq_encoding.unsqueeze(0),
        optim_repeats=max(4, len(steric_weight)),
        steric_clash_weight=steric_weight,
        steric_loss_kwargs=dict(
            global_allowance=steric_tol_allowance,
            reduction="sum",
            p=2,
            global_tol_frac=steric_tol_frac,
        ),
        optim_kwargs=dict(max_iter=400, lr=1e-3, line_search_fn="strong_wolfe"),
        use_input_backbone=True,
        device=device,
        alphas=alphas,
        override_atoms=["ASN", "ARG", "GLU", "GLN", "TRP", "ASP"],
        ignore_atoms=["LEU", "VAL"],
        override_alpha=0.5,
    )

    # save the new (projected) pdb
    decoy.to_pdb(
        path=os.path.join(save_dir, target),
        coords=projected_coords.squeeze(),
    )


if __name__ == "__main__":
    parser = ArgumentParser(
        description="standardize side chain pdbs after inference (For use on RX machines only)",  # noqa
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--method", help="root directory with method statistics")
    parser.add_argument(
        "--steric_weight",
        help="weight to use for steric loss during each round of minimization",
        default=[0.01, 0.05, 0.1],
        nargs="+",
        type=float,
    )
    parser.add_argument(
        "--steric_tol_frac",
        help="penalize clashes when distance is less that ((vdw_i+vdw_j) - --steric_tol_allowance)*steric_tol_frac",
        default=0.9,
        type=float,
    )
    parser.add_argument(
        "--steric_tol_allowance",
        help=(
            "penalize clashes when distance is less that (vdw_i+vdw_j)-steric_tol_allowance "
            "Note: stacks on --steric_tol_frac"
        ),
        default=0.0,
        type=float,
    )
    parser.add_argument(
        "--device",
        help="device to run on",
        default="cuda:0",
        type=str,
    )
    parser.add_argument(
        "--max_threads",
        help="number of threads to use",
        default=2,
        type=int,
    )
    parser.add_argument("--folders", help="folders to search in", default=None, nargs="+")
    parser.add_argument(
        "--alphas", help="set only if using iterative projection scheme", type=float, default=[0], nargs="+"
    )
    parser.add_argument(
        "--torsion_loss_wt",
        help="weight to use for devistion in initial/projected dihedral angle loss",
        default=0.5,
        type=float,
    )
    parser.add_argument(
        "--out_fldr",
        help="name of folder to save to (default: projected_{100 * --steric_tol_frac})",
        default=None,
    )

    args = parser.parse_args(sys.argv[1:])
    STAT_ROOT = os.path.join(RESULT_ROOT, args.method, "stats")
    folders = os.listdir(STAT_ROOT)
    folders = args.folders if args.folders is not None else folders
    for res_fldr in folders:
        if not os.path.isdir(os.path.join(STAT_ROOT, res_fldr)):
            print("skipping {res_fldr}")
            continue
        print(f"starting {args.method}, {res_fldr}")
        stats = load_stats(args.method, res_fldr)
        pdb_root = standardize_pdbs_for_method(args.method, res_fldr)

        print(f"pdb root: {pdb_root}", os.path.dirname(pdb_root))
        suff = str(round(args.steric_tol_frac * 100))
        save_root = os.path.join(os.path.dirname(pdb_root), default(args.out_fldr, "projected" + suff))

        def fn(target):
            try:
                project_rotamers(
                    target,
                    pdb_dir=pdb_root,
                    save_dir=save_root,
                    device=args.device,
                    steric_weight=args.steric_weight,
                    torsion_loss_wt=args.torsion_loss_wt,
                    iterative_refine_max_iters=len(args.alphas),
                    alphas=args.alphas,
                )
            except:
                traceback.print_exc()
                return target  # likely failure due to OOM, retry these targets after
            return None

        targets = [x for x in os.listdir(pdb_root) if x.endswith("pdb")]
        for i in range(math.ceil(math.log2(max(2, args.max_threads)))):
            n_threads = max(1, args.max_threads // (2**i))
            targets = [x for x in targets if x is not None]
            with Pool(n_threads) as p:
                targets = p.map(fn, targets)
