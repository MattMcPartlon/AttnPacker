# Source Code for AttnPacker 

This repo contains code for AttnPacker

Pre-trained models, and PDB files used to generate all results are available at https://zenodo.org/record/7713779#.ZApZHezMIVU

UPDATE (04/19/2023): AttnPacker+Design now supports conditioning on partial sequence *and* rotamers. Pre trained models are
available at https://zenodo.org/record/7843977#.ZEAWqezML0o. The main inference notebook has been updated to reflect this option.

# Install


```
$ git clone git@github.com:MattMcPartlon/AttnPacker.git
$ conda create -n attnpacker python=3.8
$ conda activate attnpacker
$ pip install -r ./AttnPacker/requirements.txt
```

Note: The default pytorch installation may not include GPU support. Since this is often system-specific it is left to the user to change this.


# Examples

Inference with AttnPacker is outlined in `attnpacker/examples/inference.ipynb`. This includes examples for sequence design, side-chain post processing 
per-residue confidence prediction and more. A notbook with examples specific to sampling is available at `attnpacker/examples/sampling.ipynb`. Additional examples are outlined below.


## Run Post-Process Procedure on a PDB File
```
usage: post_process.py [-h] [--pdb_path_out PDB_PATH_OUT] [--steric_wt STERIC_WT] [--steric_tol_allowance STERIC_TOL_ALLOWANCE]
                                  [--steric_tol_frac STERIC_TOL_FRAC] [--steric_hbond_allowance STERIC_HBOND_ALLOWANCE]
                                  [--max_optim_iters MAX_OPTIM_ITERS] [--torsion_loss_wt TORSION_LOSS_WT] [--device DEVICE]
                                  pdb_path_in

Project Protein Sidechains onto Continuous Rotamer and Minimize Steric Clashes

positional arguments:
  pdb_path_in           path to input pdb

optional arguments:
  -h, --help            show this help message and exit
  --pdb_path_out PDB_PATH_OUT
                        path to save projected pdb to (defaults to post-processed-<input pdb name>.pdb (default: None)
  --steric_wt STERIC_WT
                        weight to use for steric clash loss (default: 0.2)
  --steric_tol_allowance STERIC_TOL_ALLOWANCE
                        subtract this number from all atom vdW radii (default: 0.05)
  --steric_tol_frac STERIC_TOL_FRAC
                        set vdW radii to steric_tol_frac*vdW(atom_type) (default: 0.9)
  --steric_hbond_allowance STERIC_HBOND_ALLOWANCE
                        subtract this number from the sum of vdW radii for hydrogen bond donor/acceptor pairs (default: 0.6)
  --max_optim_iters MAX_OPTIM_ITERS
                        maximum number of iterations to run optimization procedure for (default: 250)
  --torsion_loss_wt TORSION_LOSS_WT
                        penalize average deviaiton from initial dihedral angles with this weight (default: 0)
  --device DEVICE       device to use when running this procedure (default: cpu)

(py38)[mmcpartlon@raptorx11 AttnPacker]$ python attnpacker/examples/post_process.py ./attnpacker/examples/pdbs/T1057-predicted.pdb --steric_tol_allowance 0 --steric_tol_frac 0.95 --max_optim_iters 200 --device cuda:0

[fn: project_onto_rotamers] : Using device cuda:0
[INFO] Beginning rotamer projection
[INFO] Initial loss values
   [RMSD loss] = 0.103
   [Steric loss] = 0.035
   [Angle Dev. loss] = 0.0

beginning iter: 0, steric weight: 0.2
[INFO] Final Loss Values
   [RMSD loss] = 0.064
   [Steric loss] = 0.002
   [Angle Dev. loss] = 0.001

Saving to: ./attnpacker/examples/pdbs/post-processed-T1057-predicted.pdb
Finished in 3.32 seconds
```

## Compare Side-Chain prediction with native structure

```python
from attnpacker.assessment.sidechain import assess_sidechains, summarize
import pprint
predicted_pdb = "./pdbs/post-processed-T1080-predicted.pdb"
target_pdb = "./pdbs/T1080.pdb"
res_level_stats = assess_sidechains(target_pdb, predicted_pdb, steric_tol_fracs = [1,0.9,0.8])
target_level_stats = summarize(assessment_stats)
print(pprint.pformat(target_level_stats))
```
Output:
```
{'ca_rmsd': tensor(    0.000),
 'clash_info': {'100': {'energy': tensor(2.010),
                        'num_atom_pairs': 308580,
                        'num_clashes': 16},
                '80': {'energy': tensor(0.),
                       'num_atom_pairs': 308580,
                       'num_clashes': 0},
                '90': {'energy': tensor(0.),
                       'num_atom_pairs': 308580,
                       'num_clashes': 0}},
 'dihedral_counts': tensor([98, 57, 12,  7]),
 'mae_sr': tensor(0.520),
 'mean_mae': tensor([28.504, 22.006, 73.557, 45.663]),
 'num_sc': 98,
 'rmsd': tensor(0.743),
 'seq_len': 133}
```

In the example above, `assessment_stats` contains residue level information regarding dihedral MAE, RMSD, clashing atom pairs, etc. The `summarize` function produces target-level statistics by averaging over all residues with at least two side-chain atoms. For this target, a total of 138 residues were analyzed, and 98 had at least two side chain atoms (i.e. were not Glycine or Alanine).

# Code Organization

## common

The common folder contains protein specific constants such as residue names, atom types, side-chain dihedral
information, etc. It also contains functionality for working with sequence and pdb files, data-loading, and model
configuration settings.

### common/data

This sub-folder contains base classes for protein model input and output (`model_data.py`)
as well as protein datasets. The *dataset* subfolder contains PyTorch datasets for training a protein-learning model
which are compatible with PyTorch's DataLoader.

## networks
Implementation of

- Tensor Field Networks
- TFN-Transformer
- EvoFormer

### networks/loss

This folder contains loss functions for residue, pair, and coordinate features

## features

This folder contains functions for computing input features.

The `InputEmbedding` class (`features/input_embedding.py`) can be used to generate and embed all input features.

The `FeatureGenerator` class (`features/generator.py`) is passed to a `ProteinDataset` instance to generate input
features during training. Options for which input features to include can be found in `features/feature_config.py`. You
can also subclass `FeatureGenerator` and/or `InputEmbedding` to obtain any additional functionality.

## assesment

Code used to compare predicted side-chain packing with ground truth can be found here (`sidechains.py`). 
This will generate per-residue statistics such as:
- Side-Chain RMSD
- Dihedral MAE
- Side Chain Dihedral Accuracy
- Number of Cb neighbors for each residue
- Steric clash information



