import torch
from protein_learning.models.utils.feature_flags import (
    get_ss_blocks,
    get_block_adj_and_ori_mats
)


def test_get_ss_blocks1():
    ss1 = "HHHEEECCC"
    ss_blocks, block_labels = get_ss_blocks(ss1)
    assert len(ss_blocks) == len(block_labels)
    expected_labels = ["H", "E", "C"]
    assert block_labels == expected_labels, \
        f"expected: {expected_labels}, got: {block_labels}"
    expected_blocks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    assert ss_blocks == expected_blocks, f"got: {ss_blocks}, expected: {expected_blocks}"


def test_get_ss_blocks2():
    ss2 = "EHHEEECCCHCHC"
    ss_blocks, block_labels = get_ss_blocks(ss2)
    assert len(ss_blocks) == len(block_labels)
    assert sum(map(len, ss_blocks)) == len(ss2)
    expected_labels = ["E", "H", "E", "C", "H", "C", "H", "C"]
    assert block_labels == expected_labels, \
        f"expected: {expected_labels}, got: {block_labels}"
    expected_blocks = [[0], [1, 2], [3, 4, 5], [6, 7, 8], [9], [10], [11], [12]]
    assert ss_blocks == expected_blocks, f"got: {ss_blocks}, expected: {expected_blocks}"


def check_group_adjs(g1, g2, mat, val):
    """Check group adjacencies are equal to val"""
    for i in g1:
        for j in g2:
            if not mat[i, j] == val:
                return False
    return True


def test_get_block_adj_and_ori_mats_all_adj():
    ss1 = "HHHEEECCC"
    coords = torch.zeros(len(ss1), 3)
    # every group should be adjacent
    adj_mat, ori_mat = get_block_adj_and_ori_mats(ss1, ca_coords=coords, include_loop_adjs=True)
    ss_blocks, block_labels = get_ss_blocks(ss1)
    for group_idx1, group1 in enumerate(ss_blocks):
        for group_idx2, group2 in enumerate(ss_blocks):
            val = 1 if group_idx2 != group_idx1 else 0
            if not check_group_adjs(group1, group2, adj_mat, val):
                assert False


def test_get_block_adj_and_ori_mats_no_adj():
    ss1 = "HHHEEECCC"
    coords = torch.zeros(len(ss1), 3)
    ss_blocks, block_labels = get_ss_blocks(ss1)
    ss_blocks = list(map(torch.tensor, ss_blocks))
    for i, block_ids in enumerate(ss_blocks):
        coords[block_ids] = i * 10
    # no group should be adjacent
    adj_mat, ori_mat = get_block_adj_and_ori_mats(ss1, ca_coords=coords, include_loop_adjs=True)
    ss_blocks, block_labels = get_ss_blocks(ss1)
    for group_idx1, group1 in enumerate(ss_blocks):
        for group_idx2, group2 in enumerate(ss_blocks):
            val = 0
            if not check_group_adjs(group1, group2, adj_mat, val):
                err_string = f"group 1: {group1}, group2: {group2}\n"
                err_string += f"coords for \ngroup 1: {coords[group1]}, \ngroup2:{coords[group2]}\n"
                group_dists = torch.cdist(coords[group1], coords[group2])
                err_string += f"group1/group2 dists: {group_dists}"
                assert False, err_string
