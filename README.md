# AttnPacker 

Pre-trained models, and PDB files used to generate all results is available at  https://zenodo.org/record/7559358#.Y83tYuzMI0Q

TODO: Examples and Inference

This repo contains code for AttnPacker

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



