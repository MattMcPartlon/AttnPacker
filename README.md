# Protein Learning

This repo contains building-blocks for machine learning-based protein models. A description of each subfolder is given
below.

## common

The common folder contains protein specific constants such as residue names, atom types, side-chain dihedral
information, etc. It also contains functionality for working with sequence and pdb files, data-loading, and model
configuration settings.

### common/data

This sub-folder contains base classes for protein model input and output (`model_data.py`)
as well as protein datasets. The *dataset* subfolder contains PyTorch datasets for training a protein-learning model
which are compatible with PyTorch's DataLoader.

## networks

This folder provides implementation of several Machine Learning architectures relevant to Protein
Folding/Design/Refinement, and more, including

- Tensor Field Networks
- SE(3)-Transformer
- EvoFormer
- Invariant Point Attention
- EGNN

### networks/loss

This folder contains loss functions for residue, pair, and coordinate features

#### Pair Loss

- Predicted Distance between Atom types (cross-entropy)

#### Residue Loss

- Native Sequence Recovery
- pLDDT

#### Coord Loss

- RMSD
- TM
- BB-Dihedral
- Distance
- FAPE

#### Side-chain Loss

- Side-chain RMSD
- Side Chain Dihedral

#### Violation

- VDW-Repulsive Energy
- Bond Length
- Bond Angle

## features

This folder contains functions for computing common input features to a protein specific machine learning model.

The `InputEmbedding` class (`features/input_embedding.py`) can be used to generate and embed common protein input
features such as

- back bone dihedral angles
- pairwise distance
- relative sequence position
- relative sequence separation
- TrRosetta Orientation features

and more.

The `FeatureGenerator` class (`features/generator.py`) is passed to a `ProteinDataset` instance to generate input
features during training. Options for which input features to include can be found in `features/feature_config.py`. You
can also subclass `FeatureGenerator` and/or `InputEmbedding` to obtain any additional functionality.

## models

The models folder contains four protein-learning models for

- side-chain packing
- design
- refinement
- multi-dimensional-scaling (MDS)

Each model defines a Dataset, Feature Generator, and torch.nn.Module for performing the associated task. Each model also
serves as an example of how to use the framework of this code base.

Each model subclasses `model_abc.py`. ANy protein-learning model must implement the abstract methods defined in this
class to be compatible with a `Trainer` object (defined in `training/trainer.py`).

## training

This folder houses the `Trainer` class used to train, validate, test, save, and load a protein learning model.

## protein_utils

This folder contains functionality for computing

- alignments of two proteins (e.g. kabsch alignment)
- dihedral and angle information (e.g. TrRosetta dihedral and angle or backbone dihedral )

## assesment

Code used to compare predicted side-chain packing with ground truth can be found here (`sidechains.py`). 
This will generate per-residue statistics such as:
- Side-Chain RMSD
- Dihedral MAE
- Side Chain Dihedral Accuracy
- Number of Cb neighbors for each residue
- Steric clash information



