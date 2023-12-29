#!usr/bin/env python
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import sys
from pyrosetta import *
import pyrosetta.rosetta as rosetta
from functools import partial
import time
from random import shuffle

#tmp = "/mnt/local/mmcpartlon/TrainTestData/test_data/CASP/AF2_for_eval/AF2_pdbs_lt_600"
#tgts = [x[:-4] for x in os.listdir(tmp)]

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
from pyrosetta import init, Pose, get_fa_scorefxn, standard_packer_task, pose_from_file, pose_from_pdb  # noqa
from pyrosetta.rosetta import core, protocols
import numpy as np
from multiprocessing import Pool


def compute_energy(pdb_path):
    #if not any([x in pdb_path for x in tgts]):
    #    print(f"skipping {pdb_path}")
    #    return None,None
    try:
        pose = pose_from_pdb(pdb_path)
        sf = get_fa_scorefxn()
        out= os.path.basename(pdb_path),sf(pose)
        print(out)
        return out
    except Exception as e:
        raise e


def get_args(arg_list):
    parser = ArgumentParser(description=" Score pdbs",  # noqa
                            epilog='compute rosetta ref2015 energy',
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--pdb_folder',
                        help='root directory for data loading',
                        type=str,
                        default=None,
                        )
    parser.add_argument('--n_threads',
                    help='number of threads to use',
                    type=int,
                    default=30,
                    )

    return parser.parse_args(arg_list)


if __name__ == "__main__":
    args = get_args(sys.argv[1:])



    pdbs = list(filter(lambda x: x.endswith(".pdb") and "best" in x,os.listdir(args.pdb_folder)))
    paths = [os.path.join(args.pdb_folder,x) for x in pdbs]
    shuffle(paths)
    def safe_run(pdb_path):
        try:
            return compute_energy(pdb_path)
        except Exception as e:
            print(f"skipping {pdb_path}")
            return None,None

    with Pool(args.n_threads) as p:
        pool_result = p.map_async(compute_energy,paths)
        # wait 5 minutes for every worker to finish
        try:
            pool_result.wait(timeout=120)
        except:
            pass
    print(pool_result)
    # once the timeout has finished we can try to get the results    
    scores=pool_result.get(timeout=10)
    print(scores)


    print("FINISHED!")
    
    with open(os.path.join(args.pdb_folder,"scores.txt"),"w+") as f:
        for name,score in scores:
            f.write(f"{name},{score}\n")

    exit(1)
