import os
import sys
from protein_learning.protein_utils.sidechains.project_sidechains import (
    project_onto_rotamers,
)
from protein_learning.common.data.data_types.protein import Protein
import protein_learning.common.protein_constants as pc
from protein_learning.common.data.datasets.utils import set_canonical_coords_n_masks
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import time


def project_pdb(
    pdb_path_in: str,
    pdb_path_out: str = None,
    steric_wt: float = 0.1,
    steric_tol_allowance=0.05,
    steric_tol_frac=0.9,
    steric_hbond_allowance: float = 0.6,
    max_optim_iters: int = 200,
    torsion_loss_wt: float = 0.25,
    device: str = "cpu",
):
    protein = Protein.FromPDBAndSeq(
        pdb_path=pdb_path_in,
        seq=None,
        atom_tys=pc.ALL_ATOMS,
        missing_seq=True,
    )
    protein = set_canonical_coords_n_masks(protein)

    projected_coords, _ = project_onto_rotamers(
        atom_coords=protein.atom_coords.unsqueeze(0).clone(),
        atom_mask=protein.atom_masks.unsqueeze(0),
        sequence=protein.seq_encoding.unsqueeze(0),
        optim_repeats=1,
        steric_clash_weight=steric_wt,
        steric_loss_kwargs=dict(
            global_tol_frac=steric_tol_frac,
            hbond_allowance=steric_hbond_allowance,
            global_allowance=steric_tol_allowance,
            reduction="sum",
            p=2,
        ),
        optim_kwargs=dict(max_iter=max_optim_iters, lr=1e-3, line_search_fn="strong_wolfe"),
        use_input_backbone=True,
        torsion_deviation_loss_wt=torsion_loss_wt,  # penalize torsion angle deviations
        device=device,
    )

    if pdb_path_out is None:
        pdb_path_out = os.path.join(os.path.dirname(pdb_path_in), f"post-processed-{os.path.basename(pdb_path_in)}")
    print(f"Saving to: {pdb_path_out}")
    protein.to_pdb(
        path=pdb_path_out,
        coords=projected_coords.squeeze(),
    )


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Project Protein Sidechains onto Continuous Rotamer and Minimize Steric Clashes",  # noqa
        epilog="",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("pdb_path_in", help="path to input pdb")
    parser.add_argument(
        "--pdb_path_out", help="path to save projected pdb to (dafaults to post-processed-<input pdb name>.pdb"
    )
    parser.add_argument("--steric_wt", help="weight to use for steric clash loss", default=0.2, type=float)
    parser.add_argument(
        "--steric_tol_allowance", help="subtract this number from all atom vdW radii", default=0.05, type=float
    )
    parser.add_argument(
        "--steric_tol_frac", help="set vdW radii to steric_tol_frac*vdW(atom_type)", default=0.925, type=float
    )
    parser.add_argument(
        "--steric_hbond_allowance",
        help="subtract this number from the sum of vdW radii for hydrogen bond donor/acceptor pairs",
        default=0.6,
        type=float,
    )
    parser.add_argument(
        "--max_optim_iters",
        help="maximum number of iterations to run optimization procedure for",
        type=int,
        default=250,
    )
    parser.add_argument(
        "--torsion_loss_wt",
        help="penalize average deviaiton from initial dihedral angles with this weight",
        type=float,
        default=0,
    )
    parser.add_argument("--device", help="device to use when running this procedure", type=str, default="cpu")
    args = parser.parse_args(sys.argv[1:])
    start = time.time()
    project_pdb(**vars(args))
    print(f"Finished in {round(time.time()-start,3)} seconds")
