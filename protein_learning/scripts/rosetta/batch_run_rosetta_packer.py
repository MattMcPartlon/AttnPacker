#!usr/bin/env python
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from pyrosetta import *
import pyrosetta.rosetta as rosetta

""" SC-packing
init('-out:levels core.conformation.Conformation:error '
     'core.pack.pack_missing_sidechains:error '
     'core.pack.dunbrack.RotamerLibrary:error '
     'core.scoring.etable:error '
     '-packing:repack_only '
     '-ex1 -ex2 -ex3 -ex4 '
     '-multi_cool_annealer 5 '
     '-no_his_his_pairE '
     '-linmem_ig 1'
     )
"""
# fixed bb design
init('-out:levels core.conformation.Conformation:error '
     'core.pack.pack_missing_sidechains:error '
     'core.pack.dunbrack.RotamerLibrary:error '
     'core.scoring.etable:error '
     '-ex1 -ex2 -ex3 -ex4 '
     '-multi_cool_annealer 5 '
     '-linmem_ig 1 '
     )
from pyrosetta import init, Pose, get_fa_scorefxn, standard_packer_task, pose_from_file  # noqa
from pyrosetta.rosetta import protocols
from protein_learning.scripts.utils import load_list
import numpy as np


def packer_task(pose, pdb_out, pdb_in, n_decoys=1):
    """
    Demonstrates the syntax necessary for basic usage of the PackerTask object
                performs demonstrative sidechain packing and selected design
                using  <pose>  and writes structures to PDB files if  <PDB_out>
                is True

    """
    uid = "res_file" + str(np.random.randint(0, 10000000))
    tmp = "./tmp/"
    os.makedirs(tmp, exist_ok=True)
    resfile = os.path.join(tmp, uid)
    pyrosetta.toolbox.generate_resfile.generate_resfile_from_pdb(pdb_in, resfile, pack=True, design=True,
                                                                 input_sc=False, freeze=[], specific={})

    best_score, best_pose = 1e10, Pose()
    scorefxn = get_fa_scorefxn()
    for _ in range(n_decoys):
        test_pose = Pose()
        test_pose.assign(pose)
        # packer task
        pose_packer = standard_packer_task(test_pose)

        print(os.path.exists(resfile))

        rosetta.core.pack.task.parse_resfile(test_pose, pose_packer, resfile)
        # pose_packer.restrict_to_repacking()  # turns off design
        # pose_packer.or_include_current(True)  # considers original conformation
        packmover = protocols.minimization_packing.PackRotamersMover(scorefxn, pose_packer)

        scorefxn(pose)  # to prevent verbose output on the next line
        print('\nPre packing score:', scorefxn(test_pose))
        packmover.apply(test_pose)
        print('Post packing score:', scorefxn(test_pose))
        test_pose.pdb_info().name('packed')  # for PyMOLMover
        score = scorefxn(test_pose)
        if score < best_score:
            best_score = score
            best_pose = best_pose.assign(test_pose)

    print("designed_seq", best_pose.sequence())
    print("og_seq", pose.sequence())
    design = best_pose.sequence()
    og = pose.sequence()
    rec = [1 if d == o else 0 for d, o in zip(design, og)]
    best_pose.dump_pdb(pdb_out)
    return sum(rec) / len(og)


def get_args(arg_list):
    parser = ArgumentParser(description=" Rosetta Pack",  # noqa
                            epilog='run rosetta fixed backbone packing protocl',
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('list',
                        help='path to list of pdbs to run on'
                        )

    parser.add_argument('pdb_folder',
                        help='path to folder with native pdbs (relative to data_root)')

    parser.add_argument('--start',
                        default=0,
                        type=int)

    parser.add_argument('--end',
                        default=1000,
                        type=int)

    parser.add_argument("--out_root",
                        default=None,
                        type=str,
                        help="root directory to store stats output")

    parser.add_argument("--resfile",
                        default=None,
                        type=str)

    return parser.parse_args(arg_list)


args = get_args(sys.argv[1:])
model_list = load_list(args.list)[args.start:args.end]
os.makedirs(args.out_root, exist_ok=True)
for i, pdb in enumerate(model_list):
    pdb = pdb if pdb.endswith(".pdb") else pdb + ".pdb"
    pdb_in = os.path.join(args.data_root, args.native_folder, pdb)
    pdb_out = os.path.join(args.out_root, pdb)
    pose = Pose()
    pose_from_file(pose, pdb_in)
    rec = packer_task(pose, pdb_out, pdb_in, 2)
    print(f"pdb_in#{pdb_in}#rec#{rec}")
