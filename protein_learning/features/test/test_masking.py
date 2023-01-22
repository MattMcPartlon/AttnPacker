"""Tests for feature masking"""
from protein_learning.features.feature_config import (
    InputFeatureConfig, AA_ALPHABET, FeatureName, FeatureTy, FeatureEmbeddingTy
)
from protein_learning.features.feature_generator import DefaultFeatureGenerator
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
CONFIG = InputFeatureConfig(
    pad_embeddings=True,
    quat_encode_rel_ori=True,
    encode_local_rel_coords=True,
    joint_embed_res_pair_rel_sep=True,
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


def test_scalar_masking():
    feat_gen = DefaultFeatureGenerator(config=CONFIG)
    feats = feat_gen.generate_features(protein=TEST_PROTEIN)

    expected_names = [FeatureName.RES_TY, FeatureName.CENTRALITY, FeatureName.REL_POS, FeatureName.BB_DIHEDRAL]
    actual_names = []
    for feat_desc in CONFIG.descriptors:
        if feat_desc.ty == FeatureTy.RESIDUE:
            actual_names.append(feat_desc.name)
        else:
            continue
        scalar_mask = torch.randn(n_res) > 0
        feat, val = feats[feat_desc.name], feat_desc.encode_dim
        feat.apply_mask(scalar_mask)
        assert torch.all(feat.encoded_data[:, scalar_mask] == val), f"{feat.name}"
        assert torch.all(feat.encoded_data[:, ~scalar_mask] < val), f"{feat.name}"
    assert set(actual_names) == set(expected_names)


def test_pair_masking():
    feat_gen = DefaultFeatureGenerator(config=CONFIG)
    feats = feat_gen.generate_features(protein=TEST_PROTEIN)

    expected_names = [FeatureName.REL_ORI, FeatureName.REL_COORD, FeatureName.REL_SEP]
    actual_names = []
    for feat_desc in CONFIG.descriptors:
        if feat_desc.ty == FeatureTy.PAIR:
            assert feat_desc.name not in actual_names
            actual_names.append(feat_desc.name)
        else:
            continue
        NONE, ONEHOT = FeatureEmbeddingTy.NONE, FeatureEmbeddingTy.ONEHOT
        assert NONE in feat_desc.embed_tys or ONEHOT in feat_desc.embed_tys

        pair_mask = torch.randn(n_res, n_res) > 0
        pair_mask = torch.logical_or(pair_mask, torch.eye(n_res).bool())  # noqa
        feat = feats[feat_desc.name]
        if FeatureEmbeddingTy.NONE in feat_desc.embed_tys:
            val = feat.raw_mask_value
            data = feat.get_raw_data()
        else:
            val = feat_desc.encode_dim
            data = feat.get_encoded_data()
        feat.apply_mask(pair_mask)
        assert torch.all(data[:, pair_mask] == val), f"{feat.name}"
        # if feat.name != FeatureName.REL_ORI:
        assert torch.all(torch.any(data[:, ~pair_mask] != val, dim=-1)), f"{feat.name}"
    assert set(actual_names) == set(expected_names)
