import torch
from torch import Tensor
from typing import Optional, List
from protein_learning.models.utils.feature_flags import (
    get_bond_flags,
    get_inter_chain_contacts,
    get_contact_flags,
    get_interface_flags
)
from einops import repeat  # noqa
from protein_learning.common.data.data_types.protein import Protein
from protein_learning.common.helpers import default, exists  # noqa


def _get_protein_from_ca(
        coords: Tensor,
        ss: Optional[str] = None,
        partition: Optional[List[Tensor]] = None
) -> Protein:
    assert coords.ndim == 3
    assert coords.shape[1] == 1
    if exists(partition):
        assert sum(map(len, partition)) == coords.shape[1]
    return Protein(
        atom_coords=coords,
        atom_masks=torch.ones_like(coords)[:, :, 0].bool(),
        atom_tys=["CA"],
        seq="".join(["A"] * coords.shape[0]),
        name="CA-PTN",
        sec_struct=ss,
        chain_indices=partition,
    )


def test_get_bond_flags():
    n = 10
    scale = 3.8 / (3 ** (1 / 2))
    coords = torch.arange(n).float() * scale
    coords = repeat(coords, "i-> i c", c=3)
    bond_flags = get_bond_flags(_get_protein_from_ca(coords.unsqueeze(1))).squeeze()
    assert torch.sum(bond_flags) == 2 * (n - 1)
    for i, j in zip(range(n - 1), range(1, n)):
        assert bond_flags[i, j] == 1
        assert bond_flags[j, i] == 1

    n = 10
    scale = 3.8 / (3 ** (1 / 2))
    coords = torch.arange(n).float() * scale
    coords[n // 2] = 0
    coords = repeat(coords, "i-> i c", c=3)
    bond_flags = get_bond_flags(_get_protein_from_ca(coords.unsqueeze(1))).squeeze()
    assert torch.sum(bond_flags) == 2 * (n - 1) - 4
    for i, j in zip(range(n - 1), range(1, n)):
        val = 1 if (i != n // 2 - 1 and i != n // 2) else 0
        assert bond_flags[i, j] == val
        assert bond_flags[j, i] == val


def test_get_inter_chain_contacts():
    # No contacts should be recovered for single chain
    n = 2 * 5
    assert n % 2 == 0
    single_chain = _get_protein_from_ca(torch.ones(n, 1, 3))
    partition = [torch.arange(10)]
    contacts = get_inter_chain_contacts(single_chain["CA"], partition=partition).float()
    assert torch.sum(contacts) == 0, f"{torch.sum(contacts)}"

    # all residues between each chain should be in contact
    n = 10
    single_chain = _get_protein_from_ca(torch.ones(n, 1, 3))
    partition = [torch.arange(n // 2), n // 2 + torch.arange(n // 2)]
    contacts = get_inter_chain_contacts(single_chain["CA"], partition=partition).float()
    assert torch.sum(contacts) == 2 * ((n // 2) ** 2), f"{torch.sum(contacts)}"


def test_get_inter_chain_contact_flags():
    # No contacts should be recovered for single chain
    n = 2 * 5
    assert n % 2 == 0
    single_chain = _get_protein_from_ca(torch.ones(n, 1, 3))
    partition = [torch.arange(10)]
    flags = get_contact_flags(
        protein=single_chain,
        include_contact_prob=1,
        contact_threshold=10,
        contact_fracs=(1, 1),
        chain_indices=partition
    )
    assert torch.sum(flags) == 0, f"{torch.sum(flags)}"

    # all residues between each chain should be in contact
    n = 10
    partition = [torch.arange(n // 2), n // 2 + torch.arange(n // 2)]
    flags = get_contact_flags(
        protein=single_chain,
        include_contact_prob=1,
        contact_threshold=10,
        contact_fracs=(1, 1),
        chain_indices=partition
    )
    assert torch.sum(flags) == 2 * ((n // 2) ** 2), f"{torch.sum(flags)}"

    # no residues in contact, but two chains
    n = 10
    partition = [torch.arange(n // 2), n // 2 + torch.arange(n // 2)]
    coords = torch.ones(n, 1, 3)
    coords[:n // 2] = 0
    coords[n // 2:] = 10
    single_chain = _get_protein_from_ca(coords)
    flags = get_contact_flags(
        protein=single_chain,
        include_contact_prob=1,
        contact_threshold=10,
        contact_fracs=(1, 1),
        chain_indices=partition
    )
    assert torch.sum(flags) == 0, f"{torch.sum(flags)}"


def test_get_interface_res():
    # No contacts should be recovered for single chain
    n = 2 * 5
    assert n % 2 == 0
    single_chain = _get_protein_from_ca(torch.ones(n, 1, 3))
    partition = [torch.arange(10)]
    flags = get_interface_flags(
        protein=single_chain,
        include_prob=1,
        contact_thresh=10,
        interface_fracs=(1, 1),
        include_both_prob=1,
        chain_indices=partition,
    )
    assert torch.sum(flags) == 0, f"{torch.sum(flags)}"

    # all residues between each chain should be in contact
    n = 10
    partition = [torch.arange(n // 2), n // 2 + torch.arange(n // 2)]
    flags = get_interface_flags(
        protein=single_chain,
        include_prob=1,
        contact_thresh=10,
        interface_fracs=(1, 1),
        include_both_prob=1,
        chain_indices=partition,
    )
    assert torch.sum(flags) == 10, f"{torch.sum(flags)}"

    # no residues in contact, but two chains
    n = 10
    partition = [torch.arange(n // 2), n // 2 + torch.arange(n // 2)]
    coords = torch.ones(n, 1, 3)
    coords[:n // 2] = 0
    coords[n // 2:] = 10
    single_chain = _get_protein_from_ca(coords)
    flags = get_interface_flags(
        protein=single_chain,
        include_prob=1,
        contact_thresh=10,
        interface_fracs=(1, 1),
        include_both_prob=1,
        chain_indices=partition,
    )
    assert torch.sum(flags) == 0, f"{torch.sum(flags)}"
