#!usr/bin/env python
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'
import numpy as np
from functools import partial
import subprocess
import random

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import sys
# Python
from pyrosetta import *  # noqa
import pyrosetta as prs  # noqa
from pyrosetta.toolbox import cleanATOM  # noqa
# Core Includes
from rosetta.core.pack.task import TaskFactory  # noqa
from rosetta.core.pack.task import operation  # noqa
from rosetta.core.select.movemap import *  # noqa

# Protocol Includes
from rosetta.protocols import minimization_packing as pack_min  # noqa
from rosetta.protocols.antibody import *  # noqa
from rosetta.protocols.loops import *  # noqa
from rosetta.protocols.relax import FastRelax  # noqa

from pyrosetta import init, Pose, get_fa_scorefxn, standard_packer_task, pose_from_file  # noqa
from multiprocessing import Pool  # noqa

SWITCH = SwitchResidueTypeSetMover("centroid")  # noqa
SWITCH2 = SwitchResidueTypeSetMover("fa_standard")  # noqa
SIDECHAIN_CONSTRAINT_SD = 0.1

init('-out:levels core.conformation.Conformation:error '
     'core.io.pose_from_sfr.PoseFromSFRBuilder:error '
     'core.pack.pack_missing_sidechains:error '
     'core.pack.dunbrack.RotamerLibrary:error '
     'core.scoring.etable:error '
     'basic.io.database:error '
     'core.chemical.GlobalResidueTypeSet:error '
     'core.scoring:error '
     'basic.random:error '
     'protocols.relax:error '
     'core.optimization:error '
     'core.pack:error '
     'core.import_pose:error '
     '-use_input_sc '
     '-input_ab_scheme '
     #whether to constrain side chain coordinates
     #'-relaxcoord_constrain_sidechains '
     #f'-relax:coord_cst_stdev {SIDECHAIN_CONSTRAINT_SD} '
     'AHo_Scheme '
     '-ignore_waters 1 '
     '-ignore_unrecognized_res '
     '-ignore_zero_occupancy false '
     '-load_PDB_components false '
     '-relax:default_repeats 5 '  # change this to run more relax repeats
     '-no_fconfig '
     '-jumps:no_chainbreak_in_relax 1' #Cchange to 0 if chain breaks are allowed
     )


def load_list(list_path):
    """Loads a target list (one entry per line)"""
    with open(list_path, "r") as lst:
        return [line.strip() for line in lst]  # noqa


def get_pose_coords(pose, atom_ty="CA"):
    coords = []
    for i in range(pose.total_residue()):
        atom_coords = pose.residue(i + 1).xyz(atom_ty)
        coords.append(np.array(atom_coords))
    return np.array(coords)


def add_ca_dist_cons(pose, sfxn, wt=0, d_max=1e6):
    if wt > 0:
        sfxn.set_weight(prs.rosetta.core.scoring.atom_pair_constraint, wt)
        # add distance constraints
        # prs.rosetta.core.id.AtomID(2,i)
        crds = get_pose_coords(pose)
        atom_id = lambda i: prs.rosetta.core.id.AtomID(2, i)
        dists = np.linalg.norm((crds[:, None, :] - crds[None, :, :]) + 1e-5, axis=-1)
        print(dists.shape)
        func = lambda i, j, sigma=1: prs.rosetta.core.scoring.func.HarmonicFunc(dists[i - 1, j - 1], 1)
        constraint = lambda i, j: prs.rosetta.core.scoring.constraints.AtomPairConstraint(atom_id(i), atom_id(j),
                                                                                          func(i, j))

        for i in range(pose.total_residue()):
            for j in range(pose.total_residue()):
                if abs(i - j) > 4 and dists[i, j] < d_max:
                    pose.add_constraint(constraint(i + 1, j + 1))


def merge_pdbs(pdb1, pdb2, out_path):
    lines = []
    for pdb in (pdb1, pdb2):
        with open(pdb, "r+") as pin:
            for x in pin:
                if not any([x.strip().startswith(y) for y in "REMARK END TER HEADER MODEL".split()]):
                    lines.append(x)
        lines.append("TER\n")
    lines = lines[:-1]
    lines.append("END")
    with open(out_path, "w+") as pout:
        for line in lines:
            pout.write(line)


def idealize(pose):
    mover = prs.rosetta.protocols.idealize.IdealizeMover()
    mover.fast(True)
    mover.coordinate_constraint_weight(0.1)
    mover.report_CA_rmsd(True)
    mover.impose_constraints(True)
    mover.atom_pair_constraint_weight(0.25)
    ideal_pose = pose.clone()
    # switch.apply(start_pose)
    mover.apply(ideal_pose)
    return ideal_pose


def init_pose_for_relax(pdb_name, pdb_folder, tmp_dir="./", ignore_scs=False):
    _clean_pdb = f"clean_{random.randint(0, int(1e9))}" + pdb_name + ".pdb"
    pdb_in = os.path.join(pdb_folder, pdb_name + ".pdb")
    if not os.path.exists(pdb_in):
        print("[WARNING] no pdb found for target:", pdb_name, pdb_in, "\nSkipping...")
        return None
    clean_pdb = os.path.join(tmp_dir, _clean_pdb)
    cleanATOM(pdb_in, out_file=clean_pdb)
    pose = Pose()
    pose_from_file(pose, clean_pdb)

    if ignore_scs:
        SWITCH.apply(pose)
        SWITCH2.apply(pose)
    try:
        os.remove(clean_pdb)
    except:  # noqa
        print(f"[WARNING] Unable to remove cleaned pdb file : {clean_pdb}")
    return pose


def run_relax(
        pdb_name,
        pdb_folder=None,
        n_decoys=1,
        relax_iters=300,
        design_seq=False,
        ignore_scs=False,
        out_root="./",
        ignore_existing=True,
        ca_cons_wt=0,
):
    """
    Run FastRelax on input pdb
    """
    pdb_out = os.path.join(out_root, "relaxed_" + pdb_name)
    if ignore_existing and os.path.exists(pdb_out + "_best.pdb"):
        return 0

    pdb_folder = "" if pdb_folder is None else pdb_folder
    pose = init_pose_for_relax(
        pdb_name=pdb_name,
        pdb_folder=pdb_folder,
        tmp_dir=out_root,
        ignore_scs=ignore_scs,
    )
    if pose is None:
        return 0
    original_pose = pose.clone()
    tf = TaskFactory()
    tf.push_back(operation.InitializeFromCommandline())
    if not design_seq:
        tf.push_back(operation.RestrictToRepacking())  # disables design
    #packer = pack_min.PackRotamersMover()
    #packer.task_factory(tf)
    scorefxn = get_fa_scorefxn()
    best_score, best_pose = 1e10, original_pose
    for decoy in range(n_decoys):
        scorefxn = get_fa_scorefxn()
        test_pose = Pose()
        test_pose.assign(pose)
        add_ca_dist_cons(test_pose, sfxn=scorefxn, wt=ca_cons_wt)
        fr = FastRelax()
        fr.set_scorefxn(scorefxn)
        fr.max_iter(relax_iters)
        # packer task
        print(f'\n{pdb_name} Pre relax score (iter = {decoy + 1}):', scorefxn(test_pose))
        fr.apply(test_pose)
        score = scorefxn(test_pose)
        print(f'{pdb_name} Post packing score (iter = {decoy + 1}):', score)
        print(f'{pdb_name} RMSD : {prs.rosetta.core.scoring.CA_rmsd(original_pose, test_pose)}')

        if score < best_score:
            best_score = score
            best_pose = best_pose.assign(test_pose)
        test_pose.dump_pdb(pdb_out + f"_{decoy + 1}.pdb")
    best_pose.dump_pdb(pdb_out + "_best.pdb")
    return scorefxn(best_pose)


def get_args(arg_list):
    parser = ArgumentParser(description=" FastRelax",  # noqa
                            epilog='Run rosetta FastRelax protocol. Each relaxed '
                                   'pdb will be saved in --out_root\n'
                                   'pdbs are saved as <out_root>/relaxed_<target>_{decoy #}.pdb '
                                   '(up to # decoys).\n'
                                   'A copy is made for the lowest energy target, '
                                   'and saved as <out_root>/relaxed_<target>_best.pdb.',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--is_ab_ag', action='store_true')
    parser.add_argument('--pdb_folder',
                        help='path to folder with native pdbs')
    parser.add_argument('--pdb_list',
                        help='path to list of pdbs to run on. If no list specified,'
                             ' all pdbs in <pdb_folder> will be relaxed',
                        default=None,
                        )
    parser.add_argument("--n_cpus", type=int, default=1,
                        help="number of cpus to run on (in parallel)")
    parser.add_argument("--out_root",
                        default="./",
                        type=str,
                        help="path to directory to store relaxed pdbs in")
    parser.add_argument("--relax_iters", default=300, type=int,
                        help="maximum number of minimization steps in relax energy minimization protocol")
    parser.add_argument("--ignore_scs", action="store_true", help="ignore input side-chains")
    parser.add_argument("--n_decoys", default=3, type=int, help="Number of decoys to generate per-target")
    parser.add_argument("--design_seq", action="store_true", help="Design sequence during relax")
    parser.add_argument("--ignore_existing", action="store_true")
    parser.add_argument("--ca_cons_wt", type=float, default=0)

    return parser.parse_args(arg_list)


args = get_args(sys.argv[1:])
if args.pdb_list is not None:
    model_list = load_list(args.pdb_list)
else:
    model_list = [x for x in os.listdir(args.pdb_folder) if x.endswith("pdb")]
os.makedirs(args.out_root, exist_ok=True)

to_pdb_file = lambda x, fldr=args.pdb_folder: os.path.join(fldr, x + ("" if x.endswith(".pdb") else ".pdb"))
pdb_folder = args.pdb_folder


def copy(x, y):
    subprocess.call(f"cp {x} {y}", shell=True)


if args.is_ab_ag:
    H_chains, L_chains, A_chains = map(lambda i: [x.strip().split()[i] for x in model_list], range(3))
    Ag_model_list, HL_model_list = A_chains, []
    # merge heavy and light chains
    paired_fldr = os.path.join(args.pdb_folder, "paired_HL")
    pdb_folder = paired_fldr
    os.makedirs(paired_fldr, exist_ok=True)
    for H, L in zip(H_chains, L_chains):
        if "none" in L.lower():
            copy(to_pdb_file(H), to_pdb_file(f"{H[:6]}_paired", fldr=paired_fldr))
        else:
            merge_pdbs(
                to_pdb_file(H),
                to_pdb_file(L),
                to_pdb_file(f"{H[:6]}_paired", fldr=paired_fldr)
            )
        HL_model_list.append(f"{H[:6]}_paired")
    for A in A_chains:
        copy(to_pdb_file(A), to_pdb_file(A, fldr=paired_fldr))
    model_list = A_chains + HL_model_list

strip_pdb = lambda pdb: pdb[:-4] if pdb.endswith(".pdb") else pdb
pdb_names = list(map(strip_pdb, model_list))
print(f"pdb names : {pdb_names}")

relax_fn = partial(
    run_relax,
    pdb_folder=pdb_folder,
    n_decoys=args.n_decoys,
    relax_iters=args.relax_iters,
    design_seq=args.design_seq,
    ignore_scs=args.ignore_scs,
    out_root=args.out_root,
    ignore_existing=args.ignore_existing,
    ca_cons_wt=args.ca_cons_wt,
)


def _safe_run(arg):
    try:
        relax_fn(arg)
    except:
        print(f"caught exception running {arg}")


with Pool(args.n_cpus) as p:
    p.map(_safe_run, pdb_names)
