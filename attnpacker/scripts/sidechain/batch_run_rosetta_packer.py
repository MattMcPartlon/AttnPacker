#!usr/bin/env python
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import sys
from pyrosetta import *
import pyrosetta.rosetta as rosetta
from functools import partial
import time

init(
    "-out:levels core.conformation.Conformation:error "
    "core.pack.pack_missing_sidechains:error "
    "core.pack.dunbrack.RotamerLibrary:error "
    "core.scoring.etable:error "
    "-packing:repack_only "
    "-ex1 -ex2 -ex3 -ex4 "
    "-multi_cool_annealer 5 "
    "-no_his_his_pairE "
    "-linmem_ig 1"
)
from pyrosetta import init, Pose, get_fa_scorefxn, standard_packer_task, pose_from_file  # noqa
from pyrosetta.rosetta import core, protocols
import numpy as np
from multiprocessing import Pool


def load_list(rt, lst):
    out = []
    with open(os.path.join(rt, lst), "r") as f:
        for x in f:
            if len(x) > 3:
                out.append(x.strip()[:-4] if x.endswith("pdb") else x.strip())
    return out


def packer_task(pdb_out, pdb_in, n_decoys=1):
    """
    Demonstrates the syntax necessary for basic usage of the PackerTask object
    performs demonstrative sidechain packing and selected design
    using  <pose>  and writes structures to PDB files if  <PDB_out>
    is True

    """
    pose = Pose()
    pose_from_file(pose, pdb_in)
    uid = "res_file" + str(time.time_ns())
    tmp = "./tmp/"
    os.makedirs(tmp, exist_ok=True)
    resfile = os.path.join(tmp, uid)
    pyrosetta.toolbox.generate_resfile.generate_resfile_from_pdb(
        pdb_in, resfile, pack=True, design=True, input_sc=False, freeze=[], specific={}
    )

    best_score, best_pose = 1e10, Pose()
    scorefxn = get_fa_scorefxn()
    for _ in range(n_decoys):
        test_pose = Pose()
        test_pose.assign(pose)
        # packer task
        pose_packer = standard_packer_task(test_pose)
        rosetta.core.pack.task.parse_resfile(test_pose, pose_packer, resfile)
        pose_packer.restrict_to_repacking()  # turns off design
        # pose_packer.or_include_current(True)  # considers original conformation
        packmover = protocols.minimization_packing.PackRotamersMover(scorefxn, pose_packer)

        scorefxn(pose)  # to prevent verbose output on the next line
        print("\nPre packing score:", scorefxn(test_pose))
        packmover.apply(test_pose)
        print("Post packing score:", scorefxn(test_pose))
        score = scorefxn(test_pose)
        if score < best_score:
            best_score = score
            best_pose = best_pose.assign(test_pose)

    best_pose.dump_pdb(pdb_out)


def get_args(arg_list):
    parser = ArgumentParser(
        description=" Rosetta Pack",  # noqa
        epilog="run rosetta fixed backbone packing protocl",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "data_root",
        help="root directory for data loading",
        type=str,
        default=None,
    )

    parser.add_argument("list", help="list of pdbs to run on (list name relative to data_root)")

    parser.add_argument("native_folder", help="name of folder with native pdbs (relative to data_root)")

    parser.add_argument("--start", default=0, type=int)

    parser.add_argument("--end", default=1000, type=int)

    parser.add_argument("--out_root", default=None, type=str, help="root directory to store stats output")

    parser.add_argument("--resfile", default=None, type=str)

    return parser.parse_args(arg_list)


def get_args(arg_list):
    parser = ArgumentParser(
        description=" Rosetta Pack",  # noqa
        epilog="run rosetta fixed backbone packing protocl",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "data_root",
        help="root directory for data loading",
        type=str,
        default=None,
    )

    parser.add_argument("list", help="list of pdbs to run on (list name relative to data_root)")

    parser.add_argument("native_folder", help="name of folder with native pdbs (relative to data_root)")

    parser.add_argument("--start", default=0, type=int)

    parser.add_argument("--end", default=1000, type=int)

    parser.add_argument("--out_root", default=None, type=str, help="root directory to store stats output")

    parser.add_argument("--resfile", default=None, type=str)
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--n_decoys", type=int, default=1)

    return parser.parse_args(arg_list)


args = get_args(sys.argv[1:])
model_list = load_list(args.data_root, args.list)[args.start : args.end]
os.makedirs(args.out_root, exist_ok=True)
arg_list = []
for i, pdb in enumerate(model_list):
    pdb = pdb if pdb.endswith(".pdb") else pdb + ".pdb"
    pdb_in = os.path.join(args.data_root, args.native_folder, pdb)
    pdb_out = os.path.join(args.out_root, pdb)
    arg_list.append((pdb_out, pdb_in))


def safe_run(arg):
    try:
        packer_task(*arg, n_decoys=args.n_decoys)
    except:
        print(f"skipping {arg}")


with Pool(args.threads) as p:
    p.map(safe_run, arg_list)
