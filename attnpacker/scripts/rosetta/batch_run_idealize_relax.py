#!usr/bin/env python
import os
import numpy as np
from functools import partial

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'

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
     'AHo_Scheme '
     '-ignore_unrecognized_res '
     '-ignore_zero_occupancy false '
     '-load_PDB_components false '
     '-no_fconfig'
     )


def info(msg: str):
    print(f"[INFO]: {msg}")


def load_list(list_path):
    """Loads a target list (one entry per line)"""
    with open(list_path, "r") as lst:
        return [line.strip() for line in lst]  # noqa


def idealize_pose(pose):
    info("Beginning idealize pose")
    mover = prs.rosetta.protocols.idealize.IdealizeMover()
    mover.fast(True)
    mover.coordinate_constraint_weight(0.1)
    mover.report_CA_rmsd(True)
    mover.impose_constraints(True)
    mover.atom_pair_constraint_weight(0.25)
    mover.apply(pose)
    info("Finished idealize pose")
    return pose

def show_score(pose,sfxn,msg=""):
    info("scoring pose")
    print(msg)
    sfxn.show(pose)


def init_pose_for_relax(pdb_name, pdb_folder, tmp_dir="./", ignore_scs=False):
    _clean_pdb = f"clean_{np.random.randint(int(1e9))}" + pdb_name + ".pdb"
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
        n_decoys=3,
        relax_iters=300,
        design_seq=False,
        ignore_scs=False,
        out_root="./"
):
    """
    Run FastRelax on input pdb
    """
    pdb_folder = "" if pdb_folder is None else pdb_folder
    pose = init_pose_for_relax(
        pdb_name=pdb_name,
        pdb_folder=pdb_folder,
        tmp_dir=out_root,
        ignore_scs=ignore_scs,
    )
    pdb_out = os.path.join(out_root, "relaxed_" + pdb_name)
    scorefxn = get_fa_scorefxn()
    original_pose = pose.clone()
    tf = TaskFactory()
    tf.push_back(operation.InitializeFromCommandline())
    if not design_seq:
        tf.push_back(operation.RestrictToRepacking())  # disables design
    packer = pack_min.PackRotamersMover()
    packer.task_factory(tf)
    fr = FastRelax()
    fr.set_scorefxn(scorefxn)
    fr.max_iter(relax_iters)
    best_score, best_pose = 1e10, original_pose
    for decoy in range(n_decoys):
        test_pose = Pose()
        test_pose.assign(pose)
        # packer task
        print(f'\n{pdb_name} Pre relax score (iter = {decoy + 1}):', scorefxn(test_pose))
        fr.apply(test_pose)
        score = scorefxn(test_pose)
        print(f'{pdb_name} Post packing score (iter = {decoy + 1}):', score)
        test_pose.pdb_info().name('relaxed')  # for PyMOLMover

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
    parser.add_argument('pdb_folder',
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

    return parser.parse_args(arg_list)


args = get_args(sys.argv[1:])
if args.pdb_list is not None:
    model_list = load_list(args.pdb_list)
else:
    model_list = os.listdir(args.pdb_folder)
os.makedirs(args.out_root, exist_ok=True)

strip_pdb = lambda pdb: pdb[:-4] if pdb.endswith(".pdb") else pdb
pdb_names = list(map(strip_pdb, model_list))
print(f"pdb names : {pdb_names}")
relax_fn = partial(
    run_relax,
    pdb_folder=args.pdb_folder,
    n_decoys=args.n_decoys,
    relax_iters=args.relax_iters,
    design_seq=args.design_seq,
    ignore_scs=args.ignore_scs,
    out_root=args.out_root,
)

with Pool(args.n_cpus) as p:
    p.map(relax_fn, pdb_names)
