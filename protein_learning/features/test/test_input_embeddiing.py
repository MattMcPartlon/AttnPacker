"""Tests for input embedding"""
from protein_learning.features.feature_config import InputFeatureConfig, AA_ALPHABET
from protein_learning.features.feature_generator import DefaultFeatureGenerator
from protein_learning.features.input_embedding import InputEmbedding
from protein_learning.common.data.data_types.protein import Protein
import torch

config_kwargs = dict(
    one_hot_res_ty=True,
    embed_res_rel_pos=True,
    fourier_encode_bb_dihedral=True,
    n_bb_dihedral_fourier_feats=4,
    embed_centrality=True,
    one_hot_centrality=True,
    one_hot_rel_sep=True,
)
JOINT_CONFIG = InputFeatureConfig(

    pad_embeddings=True,
    joint_embed_res_pair_rel_sep=True,
    **config_kwargs
)

SIMPLE_CONFIG = InputFeatureConfig(
    **config_kwargs
)

SIMPLE_CONFIG_W_PAD = InputFeatureConfig(
    pad_embeddings=True,
    **config_kwargs
)

SIMPLE_CONFIG_W_ID_ENCs = InputFeatureConfig(
    quat_encode_rel_ori=True,
    encode_local_rel_coords=True,
    **config_kwargs,
)

n_res, seq, atoms = len(AA_ALPHABET) - 1, AA_ALPHABET[:-1], "N CA C CB".split()
TEST_PROTEIN = Protein(
    atom_coords=torch.randn(n_res, len(atoms), 3),
    atom_masks=torch.ones(n_res, len(atoms)).bool(),
    atom_tys=atoms,
    seq=seq,
    name="TEST",
)


# simple res and pair features
def test_feat_gen_dims_without_pad():
    config = SIMPLE_CONFIG
    # one-hot res_ty + embed rel pos + fourier bb dihedral + cen embed + cen one-hot
    scalar_size = len(AA_ALPHABET) + config.res_rel_pos_embed_dim + 3 * 8
    scalar_size += config.centrality_embed_dim + config.centrality_encode_dim
    pair_size = len(config.rel_sep_encode_bins)
    input_emb = InputEmbedding(feature_config=config)
    assert input_emb.scalar_dim == scalar_size, \
        f"actual:{input_emb.scalar_dim}, expected: {scalar_size}"

    assert input_emb.pair_dim == pair_size, \
        f"actual:{input_emb.pair_dim}, expected: {pair_size}"


# simple res and pair features
def test_feat_gen_dims_with_pad():
    config = SIMPLE_CONFIG_W_PAD
    # one-hot res_ty + embed rel pos + fourier bb dihedral + cen embed + cen one-hot
    scalar_size = len(AA_ALPHABET) + config.res_rel_pos_embed_dim + 3 * 8
    scalar_size += config.centrality_embed_dim + config.centrality_encode_dim
    scalar_pad = 1 + 1  # just one-hots should be padded
    pair_size = len(config.rel_sep_encode_bins)
    input_emb = InputEmbedding(feature_config=config)
    assert input_emb.scalar_dim == scalar_size + scalar_pad, \
        f"actual:{input_emb.scalar_dim}, expected: {scalar_size + scalar_pad}"

    assert input_emb.pair_dim == pair_size + 1, \
        f"actual:{input_emb.pair_dim}, expected: {pair_size + 1}"


# simple res and pair features
def test_feat_gen_dims_with_joint_emb():
    config = JOINT_CONFIG
    # one-hot res_ty + embed rel pos + fourier bb dihedral + cen embed + cen one-hot
    scalar_size = len(AA_ALPHABET) + config.res_rel_pos_embed_dim + 3 * 8
    scalar_size += config.centrality_embed_dim + config.centrality_encode_dim
    scalar_pad = 1 + 1  # just one-hots should be padded
    pair_size = len(config.rel_sep_encode_bins) + config.joint_embed_res_pair_rel_sep_embed_dim
    input_emb = InputEmbedding(feature_config=config)
    assert input_emb.scalar_dim == scalar_size + scalar_pad, \
        f"actual:{input_emb.scalar_dim}, expected: {scalar_size + scalar_pad}"

    assert input_emb.pair_dim == pair_size + 1, \
        f"actual:{input_emb.pair_dim}, expected: {pair_size + 1}"


# simple res and pair features
def test_feat_gen_dims_with_id_encs():
    config = SIMPLE_CONFIG_W_ID_ENCs
    # one-hot res_ty + embed rel pos + fourier bb dihedral + cen embed + cen one-hot
    scalar_size = len(AA_ALPHABET) + config.res_rel_pos_embed_dim + 3 * 8
    scalar_size += config.centrality_embed_dim + config.centrality_encode_dim
    scalar_pad = 0
    pair_size = len(config.rel_sep_encode_bins) + 7
    input_emb = InputEmbedding(feature_config=config)
    assert input_emb.scalar_dim == scalar_size + scalar_pad, \
        f"actual:{input_emb.scalar_dim}, expected: {scalar_size + scalar_pad}"

    assert input_emb.pair_dim == pair_size, \
        f"actual:{input_emb.pair_dim}, expected: {pair_size}"


def test_feature_generation():
    config = SIMPLE_CONFIG
    # one-hot res_ty + embed rel pos + fourier bb dihedral + cen embed + cen one-hot
    input_emb = InputEmbedding(feature_config=config)
    feat_gen = DefaultFeatureGenerator(config)
    features = feat_gen.generate_features(protein=TEST_PROTEIN)
    for desc in config.descriptors:
        assert desc.name in features, f"{desc.name.value}"

    res_feats = input_emb.get_scalar_input(features, (1, n_res))
    pair_feats = input_emb.get_pair_input(features, (1, n_res, n_res))

    assert res_feats.shape[-1] == input_emb.scalar_dim, f"{res_feats.shape}, {input_emb.scalar_dim}"
    assert pair_feats.shape[-1] == input_emb.pair_dim, f"{pair_feats.shape}, {input_emb.pair_dim}"


def test_feature_generation_multiple_feats():
    kwargs = config_kwargs.copy()
    kwargs["one_hot_rel_sep"] = False
    config = InputFeatureConfig(
        joint_embed_res_pair_rel_sep=True,
        quat_encode_rel_ori=True,
        encode_local_rel_coords=True,
        pad_embeddings=True,
        **kwargs
    )
    # one-hot res_ty + embed rel pos + fourier bb dihedral + cen embed + cen one-hot
    input_emb = InputEmbedding(feature_config=config)
    feat_gen = DefaultFeatureGenerator(config)
    features = feat_gen.generate_features(protein=TEST_PROTEIN)
    for desc in config.descriptors:
        assert desc.name in features, f"{desc.name.value}"

    res_feats = input_emb.get_scalar_input(features, (1, n_res))
    pair_feats = input_emb.get_pair_input(features, (1, n_res, n_res))

    assert pair_feats.shape[-1] == 7 + 48
    assert res_feats.shape[-1] == input_emb.scalar_dim, f"{res_feats.shape}, {input_emb.scalar_dim}"
    assert pair_feats.shape[-1] == input_emb.pair_dim, f"{pair_feats.shape}, {input_emb.pair_dim}"
