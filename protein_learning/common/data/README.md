# Data

This folder contains two sub-folders

## datasets
- Pytorch-native Dataset and DataLoader for working with (pdb-formatted) protein data sets.
- abstract base classes and concrete implementations for data loading

## data_types
Four main data types used by all *ProteinLearning* models
- protein.py (`Protein`)
- model_input.py (`ModelInput`)
- model_output.py (`ModelOutput`)
- model_loss.py (`ModelLoss`)

### Protein
The protein class is the primary representation of pdb data. The class supports any number of chains.
A `Protein` object contains information about
- Coordinates
- Primary Sequence
- Atom Types
- Chains
- Residue Indices

Functionality for retrieving coordinates and masks given atom types are provided.

Some useful methods include:
- `FromPDBAndSeq(...) -> Protein`
  - Generates a `Protein` object for a single chain given a pdb file and (optional) sequence.
  when sequence is provided, the (sub-sequence) of the chain which best matches the input
  sequence will be extracted from the PDB file. If no sequence is provided, then the 
  first chain in the pdb file is used to construct the protein
- `to(device:Any) -> Protein`
  - places all tensor objects in the protein on the given device and returns the protein.
  
- `crop(start:int, end:int)->Protein` 
    - crop the protein (sequence, coordinates, etc) to the contiguous segment defined by [start,end).
    - works for both single chains and complexes
- `kabsch_align_coords(...)->Tensor`
  - Align the coordinates of this protein to another protein or specified coordinates
  - optionally overwrite the protein objects coordinates
- `to_pdb(...)`
  - write the protein to a pdb file
  
Some useful Properties include
- `rigids` (cached property)
  - get `Rigid` object representing local rigid frames for each protein residue
- `seq_encoding` (chached property)
  - sequence encoding as integer label tensor


### Model Input

An instance of the `ModelInput` class is passed to the forward method of any
ProteinLearning model. The class is initialized with two

