import os
# make sure cuda devices are listed according to PCI_BUS_ID beofre any torch modules are loaded
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import subprocess

def load_list(rt,lst):
    out = []
    with open(os.path.join(rt,lst),"r") as f:
        for x in f:
            if len(x)>3:
                out.append(x.strip()[:-4] if x.endswith("pdb") else x.strip())
    return out
            


def get_args(arg_list):
    parser = ArgumentParser(description=" SCWRL4",  # noqa
                            epilog='runs SCWRL4 on list of pdb targets',
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('scwrl_path',
                        help='path to scwrl',
                        type=str,
                        )

    parser.add_argument('data_root',
                        help='root directory for data loading',
                        type=str,
                        )

    parser.add_argument('list',
                        help='list of pdbs to run on (list name relative to data_root)'
                        )

    parser.add_argument('native_folder',
                        help='name of folder with native pdbs (relative to data_root)')

    parser.add_argument('--start',
                        default=0,
                        type=int)

    parser.add_argument('--end',
                        default=1000,
                        type=int)

    parser.add_argument("--out_root",
                        default=None,
                        type=str,
                        help="root directory to store scwrl output pdbs")

    return parser.parse_args(arg_list)


if __name__ == "__main__":
    args = get_args(sys.argv[1:])
    model_list = load_list(args.data_root, args.list)[args.start:args.end]
    os.makedirs(args.out_root, exist_ok=True)

    for idx, pdb in enumerate(model_list):
        pdb = pdb if pdb.endswith(".pdb") else pdb + ".pdb"
        pdb_in = os.path.join(args.data_root, args.native_folder, pdb)
        assert os.path.exists(pdb_in), f"{pdb_in}"
        pdb_out = os.path.join(args.out_root, pdb)
        cmd = " ".join([args.scwrl_path, '-i', pdb_in, '-o', pdb_out])
        out = subprocess.run(cmd, capture_output=True, shell=True, check=True)
        if idx % (len(model_list) // 10) == 0:
            print(f"Progress {idx}/{len(model_list)}")
