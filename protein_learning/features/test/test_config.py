"""Tests for feature config"""
from protein_learning.features.feature_config import InputFeatureConfig, FeatureName, FeatureEmbeddingTy


def test_scalar():
    """Test scalar features with single embed/encode option enabled"""
    config = InputFeatureConfig(
        one_hot_res_ty=True,
        embed_res_rel_pos=True,
        one_hot_bb_dihedral=True,
        fourier_encode_bb_dihedral=True,
        n_bb_dihedral_fourier_feats=4,
        embed_centrality=True,
        one_hot_centrality=True,
        rbf_encode_centrality=True
    )
    assert config.include_res_ty
    assert config.include_rel_pos
    assert config.include_bb_dihedral
    assert config.include_centrality

    descriptors = config.descriptors

    assert len(descriptors) == 4, f"{[x.name for x in descriptors]}, {len(descriptors)}"
    for desc in descriptors:
        if desc.name == FeatureName.CENTRALITY:
            expected_embed_tys = {FeatureEmbeddingTy.ONEHOT, FeatureEmbeddingTy.EMBED, FeatureEmbeddingTy.RBF}
            assert set(desc.embed_tys) == expected_embed_tys
        elif desc.name == FeatureName.RES_TY:
            assert set(desc.embed_tys) == {FeatureEmbeddingTy.ONEHOT}
        elif desc.name == FeatureName.REL_POS:
            assert desc.embed_tys == [FeatureEmbeddingTy.EMBED]
        elif desc.name == FeatureName.BB_DIHEDRAL:
            assert set(desc.embed_tys) == {FeatureEmbeddingTy.ONEHOT, FeatureEmbeddingTy.FOURIER}
            assert desc.n_fourier_feats == 4
        else:
            assert False, f"unexpected feature {desc.name}"
