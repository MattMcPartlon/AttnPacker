# AttnPacker 

Pre-trained models, and PDB files used to generate all results is available at  https://zenodo.org/record/7559358#.Y83tYuzMI0Q

TODO: Examples and Inference

This repo contains code for AttnPacker

# Examples

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
                        path to save projected pdb to (dafaults to post-processed-<input pdb name>.pdb (default: None)
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


(py38)[mmcpartlon@raptorx11 AttnPacker]$ python protein_learning/examples/post_process.py ./protein_learning/examples/pdbs/T1057-predicted.pdb --steric_tol_allowance 0 --steric_tol_frac 0.95 --max_optim_iters 200 --device cuda:0

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

Saving to: ./protein_learning/examples/pdbs/post-processed-T1057-predicted.pdb
Finished in 3.32 seconds
```


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



