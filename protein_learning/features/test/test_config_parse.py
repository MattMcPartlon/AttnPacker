"""Tests for parsing feature config"""
from protein_learning.models.utils.opt_parse import add_feature_options
from protein_learning.models.model_abc.train import get_input_feature_config
from argparse import ArgumentParser
from protein_learning.models.utils.model_io import get_args_n_groups


def test_feat_config_parse():
    opts = "--one_hot_res_ty --one_hot_rel_dist --rel_dist_bounds 0 20 " \
           "--quat_encode_rel_ori --one_hot_rel_sep " \
           "--rel_dist_encode_dim 40 --encode_local_rel_coords".split()
    parser = ArgumentParser()
    parser, feat_group = add_feature_options(parser)
    args, groups = get_args_n_groups(parser, opts)
    config = get_input_feature_config(groups, pad_embeddings=True)

    assert config.one_hot_rel_sep
    assert config.one_hot_rel_dist
    assert not config.one_hot_rel_chain
    assert config.pad_embeddings
    assert not config.one_hot_tr_rosetta_ori
    assert config.rel_dist_encode_dim == 40
    assert not config.one_hot_centrality
    assert not config.embed_rel_sep
    assert config.one_hot_res_ty
    assert config.quat_encode_rel_ori
    assert config.encode_local_rel_coords
    assert config.rel_dist_bounds == [0, 20]
