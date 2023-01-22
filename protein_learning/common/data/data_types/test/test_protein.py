import torch

from protein_learning.common.data.data_types.protein import Protein


def test_crop():
    # single chain crop
    n = 100
    ptn = Protein(
        atom_coords=torch.ones(n, 4, 3),
        atom_masks=torch.ones(n, 4).bool(),
        seq="".join(["A"] * n),
        atom_tys="N CA C CB".split(),
        name="test",
        sec_struct="".join(["H"] * n),
    )
    ptn = ptn.crop(30, 70)
    assert torch.allclose(ptn.chain_indices[0].float(), torch.arange(40).float())
    assert torch.allclose(ptn.res_ids[0].float(), 30 + torch.arange(40).float())

    # multiple chain crop
    n = 100
    ptn = Protein(
        atom_coords=torch.ones(n, 4, 3),
        atom_masks=torch.ones(n, 4).bool(),
        seq="".join(["A"] * n),
        atom_tys="N CA C CB".split(),
        name="test",
        sec_struct="".join(["H"] * n),
    )
    ptn.make_complex(partition=[torch.arange(30),torch.arange(70)+30])
    ptn = ptn.restrict_to(indices=[torch.arange(20),torch.arange(50)+40])
    print("chain")
    print(ptn.chain_indices)
    print("res ids")
    print(ptn.res_ids)
