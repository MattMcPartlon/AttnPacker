# Features

This folder contains functionality for 

- Creating, Defining, and Storing Features
  - `feature.py`, `input_features.py`
- Feature Generation
  - `feature_generator.py`
- Input Feature Embedding 
  - `input_embedding.py`
- Feature Masking
  - `./masking/...`

Raw features are generated in functions defined in `feature_utils.py`

## Code Design and Rationale
Generating features is a pain. Choosing the right representation can be even worse. 
To help alleviate this, feature extraction of several standard features
are implemented in this folder, and these can be derived directly from `Protein` objects.

To make things even more convenient, an instance of the `FeatureConfig` class (`feature_config.py`)
can be passed to a `FeatureGenerator` and `InputEmbedding` instance, yielding user-specified 
representations and embeddings of all supported feature types.

Features can be one-hot encoded, embedded (`nn.Embedding`), fourier encoded (sin + cos with freq.) or RBF-Encoded. The encoding 
dimension can be specified in the config file. 

Currently (05-28-2022) We have support for:

### Residue Features
- Relative Sequence Position
- Degree Centrality
- backbone dihedral
- Residue Type
- Flags

### Pair Features
- Relative Sequence Separation
- Relative Distance
  - user specified atom pairs
- TrRosetta Orientation
- Relative Orientation (Quaternion, Continuous)
- Relative Coordinates (pairwise relative coordinates in local invariant residue frame)
- Relative Chain
- Flags


## Example

Suppose you want to generate a 

```python
config = InputFeatureConfig(
    
    pad_embeddings= False,
    
    # Residue Type (SCALAR)
    # One-hot with 21 bins (20AA + gap)
    embed_res_ty = False,
    res_ty_embed_dim = 0,
    one_hot_res_ty = True,

    # Residue Relative Position (SCALAR)
    # One-Hot with 10 bins
    embed_res_rel_pos = False,
    res_rel_pos_embed_dim = 6,
    res_rel_pos_bins = 10, # one-hot dim
    one_hot_res_rel_pos = True,

    # BB Dihedral (SCALAR)
    # Sin and Cos encoding
    embed_bb_dihedral = False,
    bb_dihedral_embed_dim = 0,
    fourier_encode_bb_dihedral = True,
    n_bb_dihedral_fourier_feats = 2,
    one_hot_bb_dihedral = False,
    bb_dihedral_bins = 36, # one-hot dim

    # Centrality (SCALAR)
    # One-Hot with 6 bins
    embed_centrality = False,
    centrality_embed_bins = 6, # one-hot dim
    centrality_embed_dim = None,
    one_hot_centrality = True,

    # Relative Separation (PAIR)
    # One-Hot, using SMALL_SEP_BINS
    embed_rel_sep = False,
    rel_sep_embed_dim = 32,
    one_hot_rel_sep = False,
    rel_sep_embed_bins = len(SMALL_SEP_BINS), # one-hot dim

    # Relative Distance (PAIR)
    # One-Hot distance 
    embed_rel_dist = False,
    rel_dist_embed_dim = 16, # one-hot dim
    one_hot_rel_dist = True,
    rbf_encode_rel_distance = False,
    rel_dist_rbf_radii = DEFAULT_PW_DIST_RADII,
    # atom_i and atom_{i+1} will be used 
    # to generate distance features
    rel_dist_atom_tys = ["CA", "CA", "N", "CA"],
    rel_dist_embed_bins = 32, 

    # trRosetta Orientation (PAIR)
    embed_tr_rosetta_ori = True,
    tr_rosetta_ori_embed_dim = 6,
    tr_rosetta_ori_embed_bins = 36,
    fourier_encode_tr_rosetta_ori = True,
    tr_rosetta_fourier_feats = 2,
    one_hot_tr_rosetta_ori = False,

    # Joint Embedding for Pair and Sep (PAIR)
    joint_embed_res_pair_rel_sep = True,
    joint_embed_res_pair_rel_sep_embed_dim = 48,
)

```