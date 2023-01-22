import random
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
from einops import repeat  # noqa
from torch import Tensor

from protein_learning.common.data.data_types.protein import Protein
from protein_learning.common.helpers import default, exists
from protein_learning.features.masking.masking_utils import get_partition_mask
from protein_learning.common.io.utils import load_npy
from collections import defaultdict

protein_to_name = lambda protein: protein.chain_1_name

DEFAULT_SS_THRESHOLDS = {
    ("H", "H"): 7, ("H", "E"): 7, ("H", "C"): 4,
    ("E", "E"): 5, ("E", "H"): 7, ("E", "C"): 4,
    ("C", "C"): 4, ("C", "H"): 4, ("C", "E"): 4
}

DEFAULT_SS_LABELS = {
    ("H", "H"): 1, ("H", "E"): 2, ("H", "C"): 0,
    ("E", "E"): 3, ("E", "H"): 2, ("E", "C"): 0,
    ("C", "C"): 0, ("C", "H"): 0, ("C", "E"): 0
}

ERROR, INFO = "[FLAG-GEN-ERROR]", "[FLAG-GEN-INFO]"


class FeatureFlagGen:
    """Generates flag features for node/pair input"""
    name_to_index = defaultdict(int)

    def __init__(
            self,
            include_complex_flags: bool,
            include_terminal_flags: bool,
            include_bond_flags: bool,
            include_block_adjs: bool,
            include_ss_one_hot: bool,
            include_contact_prob: float = 0,
            include_interface_prob: float = 0,
            contact_fracs: Optional[Tuple[float, float]] = (0, 0),
            interface_fracs: Optional[Tuple[float, float]] = (0, 0),
            contact_thresh: float = 12,
            include_both_interface_prob: float = 1,
            one_hot_block_adjs: bool = True,
            num_interface: Optional[float] = None,
            num_contact: Optional[float] = None,
            random_interface: Optional[int] = None,
            cdr_interface: Optional[bool] = False,
            random_contact: Optional[int] = None,

    ):
        # Flags to include
        self.include_complex_flags = include_complex_flags
        self.include_terminal_flags = include_terminal_flags
        self.include_bond_flags = include_bond_flags
        self.include_contact_flags = include_contact_prob > 0
        self.include_interface_flags = include_interface_prob > 0
        self.include_block_adjs = include_block_adjs
        self.include_ss_one_hot = include_ss_one_hot
        self.one_hot_block_adjs = one_hot_block_adjs

        # probability
        self.include_contact_prob = include_contact_prob
        self.contact_fracs = contact_fracs
        self.include_interface_prob = include_interface_prob
        self.interface_fracs = interface_fracs
        self.contact_thresh = contact_thresh
        self.include_both_interface_prob = include_both_interface_prob
        self.include_terminal_flags = include_terminal_flags
        self.num_interface = num_interface if 0 < num_interface < 1 else int(num_interface)
        self.num_contact = num_contact if 0 < num_contact < 1 else int(num_contact)
        self.random_interface = random_interface
        self.random_contact = random_contact
        self.ignore_inter_chain_ss = False
        self.include_all_bonded = False
        self.contact_n_interface_dict = None
        self.cdr_interface = cdr_interface
        self.res_feat_descs, self.pair_feat_descs = dict(), dict()

    def load_contact_n_interface_dict(self, path):
        self.contact_n_interface_dict = load_npy(path)

    def gen_flags(self, native_protein: Protein, decoy_protein: Protein,
                  ) -> Optional[Tuple[Optional[Tensor], Optional[Tensor]]]:
        """Generate flag features"""
        res_flags, pair_flags, n = [], [], len(decoy_protein)

        curr_idx, res_offset, pair_offset = -1,0,0
        if exists(self.contact_n_interface_dict):
            curr_idx = decoy_protein.replica

        if self.include_terminal_flags:
            t_flags = get_terminal_flags(decoy_protein)
            res_flags.append(t_flags)
            self.res_feat_descs["terminal_flags"] = (res_offset, t_flags.shape[-1])
            res_offset += t_flags.shape[-1]

        if self.include_complex_flags:
            complex_flags = get_complex_flag(native_protein)
            pair_comp_flag = repeat(complex_flags, "n i -> n m i", m=len(decoy_protein))
            res_flags.append(complex_flags)
            pair_flags.append(pair_comp_flag)
            self.res_feat_descs["complex_flags"] = (res_offset, complex_flags.shape[-1])
            res_offset += complex_flags.shape[-1]

        if self.include_bond_flags:
            pair_flags.append(get_bond_flags(native_protein, all_bonded=self.include_all_bonded))

        if self.include_contact_flags:
            if not exists(self.contact_n_interface_dict):
                pair_flags.append(
                    get_contact_flags(
                        protein=native_protein,
                        include_contact_prob=self.include_contact_prob,
                        contact_threshold=self.contact_thresh,
                        contact_fracs=self.contact_fracs,
                        chain_indices=decoy_protein.chain_indices,
                        num_contact=self.num_contact,
                        random_contact=self.random_contact,
                    )
                )
            else:
                pair_flags.append(
                    get_contact_flags_from_dict(
                        protein=decoy_protein,
                        data=self.contact_n_interface_dict,
                        index=curr_idx
                    )
                )

        if self.include_interface_flags:
            if not exists(self.contact_n_interface_dict):
                i_flags = get_interface_flags(
                        protein=native_protein,
                        include_prob=self.include_interface_prob,
                        contact_thresh=self.contact_thresh,
                        interface_fracs=self.interface_fracs,
                        include_both_prob=self.include_both_interface_prob,
                        chain_indices=decoy_protein.chain_indices,
                        num_interface=self.num_interface,
                        cdr_interface=self.cdr_interface,
                    )
                res_flags.append(i_flags)
            else:
                i_flags = get_interface_flags_from_dict(
                        decoy_protein,
                        data=self.contact_n_interface_dict,
                        index=curr_idx
                    )
                res_flags.append(i_flags)

            self.res_feat_descs["interface_flags"] = (res_offset, i_flags.shape[-1])
            res_offset += i_flags.shape[-1]

        if self.include_block_adjs:
            block_adjs, block_oris = get_block_adj_and_ori_mats(
                sec_struc=native_protein.sec_struct,
                ca_coords=native_protein["CA"],
                dist_thresh=None,
                use_adj_labels=self.one_hot_block_adjs
            )

            if self.ignore_inter_chain_ss:
                if decoy_protein.is_complex:
                    pmask = get_partition_mask(len(decoy_protein), decoy_protein.chain_indices)
                    block_adjs[pmask], block_oris[pmask] = 0, 0

            if self.one_hot_block_adjs:
                block_adjs = torch.nn.functional.one_hot(block_adjs.long(), 4)
            else:
                block_adjs = block_adjs.unsqueeze(-1)
            _feats = torch.cat(
                (block_adjs, block_oris.unsqueeze(-1)), dim=-1
            )
            pair_flags.append(_feats)

        if self.include_ss_one_hot:
            enc = native_protein.secondary_structure_encoding
            assert torch.max(enc) <= 2, f"{torch.max(enc)}"
            enc = torch.nn.functional.one_hot(enc, 3)
            assert enc.ndim == 2
            res_flags.append(enc)

        if exists(self.contact_n_interface_dict):
            self._update_contact_dict_state(protein=decoy_protein)

        res_flags, pair_flags = map(lambda x: torch.cat(x, dim=-1) if len(x) > 0 else None, (res_flags, pair_flags))
        res_flags, pair_flags = res_flags.detach().float(), pair_flags.detach().float()
        res_dim, pair_dim = self.flag_dims
        assert res_flags.shape[-1] == res_dim, f"{res_flags.shape[-1]},{res_dim}"
        assert pair_flags.shape[-1] == pair_dim, f"{pair_flags.shape[-1]},{pair_dim}"
        return res_flags, pair_flags

    def _update_contact_dict_state(self, protein: Protein):
        try:
            key = protein_to_name(protein)
            curr_idx = protein.replica
            l1 = len(self.contact_n_interface_dict[key]["interface"])
            l2 = len(self.contact_n_interface_dict[key]["contacts"])
            n_entries = max(l1, l2)
            self.name_to_index[key] = (curr_idx + 1) % n_entries
            print(f"{INFO} loaded contact and interface info for"
                  f" {key}, index : {self.name_to_index[key] - 1}\n")
        except Exception as e:
            print(f"{ERROR} {e} updating contact dict state")

    @property
    def flag_dims(self):  # noqa
        node_flag_dim = int(self.include_interface_flags) + \
                        2 * int(self.include_complex_flags) + \
                        2 * int(self.include_terminal_flags) + \
                        3 * int(self.include_ss_one_hot)

        block_adj_scale = 5 if self.one_hot_block_adjs else 2
        pair_flag_dim = int(self.include_bond_flags) + \
                        int(self.include_contact_flags) + \
                        2 * int(self.include_complex_flags) + \
                        block_adj_scale * int(self.include_block_adjs)

        return node_flag_dim, pair_flag_dim


def get_complex_flag(protein: Protein, is_complex: bool = None):
    """binary flag indicating if protein is a complex"""
    is_complex = default(is_complex, protein.is_complex)
    complex_flags = torch.zeros(len(protein), 2)
    if is_complex:
        complex_flags[:, 0] = 1
    else:
        complex_flags[:, 1] = 1
    return complex_flags


def get_terminal_flags(protein: Protein) -> Tensor:
    """Flags for n and c terminus"""
    n = len(protein)
    term_flags = torch.zeros(n, 2)
    N_terms = [(x[0], 1) for x in protein.chain_indices]
    C_terms = [(x[-1], 0) for x in protein.chain_indices]
    keys = N_terms + C_terms
    for (tidx, pos) in keys:
        term_flags[tidx, pos] = 1
    return term_flags


def get_bond_flags(protein: Protein, coords: Optional[Tensor] = None, all_bonded=False) -> Tensor:
    """pair feature indicating if (i,j) is bonded"""
    CA = default(coords, protein.get_atom_coords("CA"))
    assert CA.ndim == 2
    n = CA.shape[0]
    bonds = torch.norm(CA[:-1] - CA[1:], dim=-1) < 3.9  # noqa
    if all_bonded:
        bonds = torch.logical_or(bonds, ~bonds)  # noqa
        if len(protein.chain_indices) > 1:
            bonds[len(protein.chain_indices[0]) - 1] = False
    bonds = bonds.float()
    tmp = torch.zeros(n, n)
    tmp[torch.arange(n - 1), 1 + torch.arange(n - 1)] = bonds
    tmp[1 + torch.arange(n - 1), torch.arange(n - 1)] = bonds
    return tmp.unsqueeze(-1)


def _load_dict_entry(protein: Protein, data: Dict, index: int, key: str):
    try:
        assert protein.is_complex
        k = protein_to_name(protein)
        assert k in data, f"key {k} not in contact/interface dict"
        if "sizes" in data[k]:
            c1, c2 = protein.chain_indices
            sizes = data[k]["sizes"]
            assert len(c1) == sizes[0] and len(c2) == sizes[1], \
                f"got chain lens: ({len(c1)},{len(c2)}), expected: {sizes}"
        index = index % len(data[k][key])
        return data[k][key][index]
    except Exception as e:
        print(f"{ERROR} {e} loading contact dict enrty for protein {protein_to_name(protein)}")
        raise e


def get_contact_flags_from_dict(protein: Protein, data: Dict, index: int) -> Tensor:
    try:
        n = len(protein)
        contact_mat = torch.zeros(n, n)
        contacts = _load_dict_entry(protein, data, index, "contacts")
        for (i, j) in contacts:
            pair = [i, j]
            np.random.shuffle(pair)
            contact_mat[pair[0], pair[1]] = 1
        print(f"loaded {len(contacts)} contacts for target {protein_to_name(protein)} (replica : {index})")
        return contact_mat.unsqueeze(-1)
    except Exception as e:
        print(f"{ERROR} {e} loading contacts for {protein_to_name(protein)}")
        raise e


def get_interface_flags_from_dict(protein: Protein, data: Dict, index: int) -> Tensor:
    try:
        n = len(protein)
        interface_flags = torch.zeros(n)
        interface = _load_dict_entry(protein, data, index, "interface")
        for i in interface:
            interface_flags[i] = 1
        print(f"loaded {len(interface)} interface residues for target {protein_to_name(protein)} (replica : {index})\n"
              f"interface: {interface}\nchain lengths : {list(map(len, protein.chain_indices))}")
        return interface_flags.unsqueeze(-1)
    except Exception as e:
        print(f"{ERROR} {e} loading interface res data for protein {protein_to_name(protein)}")
        raise e


def get_contact_flags(
        protein: Protein,
        include_contact_prob: float,
        contact_threshold: float,
        contact_fracs: Tuple[float, float],
        chain_indices: Optional[List[Tensor]],
        num_contact: Optional[int] = None,
        random_contact: Optional[int] = None,
) -> Tensor:
    """Get contact indicator matrix between two chains"""
    assert protein.is_complex or exists(chain_indices)
    n = len(protein)
    contacts = torch.zeros(n, n)
    if include_contact_prob > 0 and len(chain_indices) >= 2:
        if random.random() <= include_contact_prob:
            contacts = get_inter_chain_contacts(
                protein.get_atom_coords("CA"),
                partition=default(chain_indices, protein.chain_indices),
                contact_thresh=contact_threshold,
            ).float()
            atom_mask = protein.valid_residue_mask.float()
            pair_mask = torch.einsum("i,j->ij", atom_mask, atom_mask).bool()
            contacts[~pair_mask] = 0
            if not exists(num_contact) and not exists(random_contact):
                contact_frac = np.random.uniform(*contact_fracs)
                keep = torch.rand_like(contacts) < contact_frac
                contacts = keep.float() * contacts  # noqa
            elif exists(num_contact) and num_contact > 0:
                n_con = np.random.geometric(num_contact, 1)[0] if 0 < num_contact < 1 else num_contact
                n_contacts = contacts[contacts > 0].numel()
                n_con = min(n_contacts, n_con)
                contact_mask = torch.zeros(n_contacts)
                contact_mask[:n_con] = 1
                contacts[contacts > 0] = contact_mask[torch.randperm(n_contacts)]
            elif exists(random_contact):
                print(f"randomly setting {random_contact} contacts")
                assert exists(random_contact) and random_contact > 0
                c1, c2 = chain_indices[0], chain_indices[1]
                contacts = torch.zeros_like(contacts.float())
                c1, c2 = map(lambda x: x.detach().cpu().numpy(), (c1, c2))
                rc1, rc2 = map(
                    lambda x: np.random.choice(x, size=random_contact, replace=True), (c1, c2)
                )
                rc1, rc2 = map(lambda x: x if isinstance(x, np.ndarray) else [x], (rc1, rc2))
                for r1, r2 in zip(rc1, rc2):
                    contacts[r1, r2] = 1.
            else:
                # must be 0 contact case
                if exists(num_contact):
                    assert num_contact == 0
                contacts[:, :] = 0

    return contacts.unsqueeze(-1)


def get_interface_flags(
        protein: Protein,
        include_prob: float,
        contact_thresh: float,
        interface_fracs: Tuple[float, float],
        include_both_prob: float,
        chain_indices: Optional[List[Tensor]] = None,
        num_interface: Optional[int] = None,
        random_interface: Optional[int] = None,
        cdr_interface: Optional[bool] = False,
) -> Tensor:
    """Residue feature indicating if res is part of complex interface"""
    assert protein.is_complex or exists(chain_indices)
    chain_indices = default(chain_indices, protein.chain_indices)
    n = len(protein)
    interface_flags = torch.zeros(n)
    if cdr_interface:
        assert protein.is_antibody
        cdrs = protein.cdr_mask.sum(dim=-1)
        cdrs = (cdrs * torch.rand_like(cdrs)) > 0.5
        return cdrs.float().unsqueeze(-1)

    if include_prob > 0 and len(chain_indices) >= 2:
        if random.random() < include_prob:
            contacts = get_inter_chain_contacts(
                protein.get_atom_coords("CA"),
                partition=chain_indices,
                contact_thresh=contact_thresh,
            )
            atom_mask = protein.valid_residue_mask.float()
            pair_mask = torch.einsum("i,j->ij", atom_mask, atom_mask).bool()
            contacts[~pair_mask] = False
            is_pocket = torch.any(contacts, dim=-1)
            if not exists(num_interface):
                is_pocket = is_pocket.float()
                pocket_frac = np.random.uniform(*interface_fracs)
                pocket_res = (torch.rand_like(is_pocket) < pocket_frac).float()  # noqa
                pocket_res = is_pocket * pocket_res
            else:

                c1, c2 = chain_indices[0], chain_indices[1]
                cp1, cp2 = c1[is_pocket[c1]], c2[is_pocket[c2]]
                cp1, cp2 = map(lambda x: x.detach().cpu().numpy(), (cp1, cp2))
                if len(cp1) > 0:
                    n_int = np.random.geometric(num_interface) - 1 if 0 < num_interface < 1 else num_interface
                    cp1 = np.random.choice(cp1, size=min(len(cp1), n_int), replace=False)
                if len(cp2) > 0:
                    n_int = np.random.geometric(num_interface) - 1 if 0 < num_interface < 1 else num_interface
                    cp2 = np.random.choice(cp2, size=min(len(cp2), n_int), replace=False)
                pocket_res = torch.zeros_like(is_pocket.float())
                pocket_res[cp1] = 1
                pocket_res[cp2] = 1
            if exists(random_interface) and random_interface > 0:
                print("random interface!")
                part = chain_indices[np.random.randint(0, 2)]
                pocket_res[part] = 0
                rand_pocket = np.random.choice(part.numpy(), size=random_interface, replace=False)
                rand_pocket = rand_pocket if isinstance(rand_pocket, np.ndarray) else [rand_pocket]
                for idx in rand_pocket:
                    pocket_res[idx] = 1.
            elif random.random() > include_both_prob:
                part = chain_indices[np.random.randint(0, 2)]
                pocket_res[part] = 0
            else:
                pass
            interface_flags = pocket_res

    return interface_flags.unsqueeze(-1)


def get_inter_chain_contacts(
        native_ca: Tensor,
        partition: List[Tensor],
        contact_thresh: float = 12,
) -> Tensor:
    """Get contact flags for pair features"""
    contacts = torch.cdist(native_ca.squeeze(), native_ca.squeeze())
    part_mask = get_partition_mask(n_res=sum([len(p) for p in partition]), partition=partition)
    contacts[~part_mask] = contact_thresh + 1
    return contacts < contact_thresh  # noqa


def get_interface_scores(
        partition: List[Tensor],
        coords: Tensor,
        sigma: float = 3
) -> Tensor:
    """Assign score to each residue based on proximity to interface"""
    assert coords.ndim == 2, f"{coords.ndim}"
    assert len(coords) == sum([len(x) for x in partition]), \
        f"{len(coords)}, {sum([len(x) for x in partition])}"
    if len(partition) == 1:
        return torch.ones(len(partition[0]))
    c1, c2 = coords[partition[0]], coords[partition[1]]
    scores = 1 / (1 + torch.square(torch.cdist(c1, c2) / sigma))
    s1, s2 = torch.sum(scores, dim=1), torch.sum(scores, dim=0)
    s1, s2 = map(lambda x: torch.exp(sigma * x / torch.max(x)), (s1, s2))
    return torch.cat((s1, s2))


def get_ss_blocks(sec_struc: str) -> Tuple[List[List[int]], List[str]]:
    """Partition secondary structure into contiguous blocks

    Returns:
            ss_blocks
            where ss_blocks[i] = index of residues constituting block i
            ss_labels
            where ss_labels[i] = the secondary structure label of residues in block i
    """
    ss_blocks, ss_block, block_labels = [], [0], []
    for i in range(1, len(sec_struc)):
        if sec_struc[i] == sec_struc[i - 1]:
            ss_block.append(i)
        else:
            ss_blocks.append(ss_block)
            block_labels.append(sec_struc[i - 1])
            ss_block = [i]
    ss_blocks.append(ss_block)
    block_labels.append(sec_struc[-1])
    return ss_blocks, block_labels


def get_block_adj_and_ori_mats(
        sec_struc: str,
        ca_coords: Tensor,
        dist_thresh: Optional[Dict[Tuple[str, str], float]] = None,
        include_self_loop: bool = False,
        include_loop_adjs: bool = False,
        use_adj_labels: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Get ss block adjacency and orientation information"""
    dist_thresh = default(dist_thresh, DEFAULT_SS_THRESHOLDS)
    ss_blocks, block_labels = get_ss_blocks(sec_struc)
    ss_blocks = list(map(torch.tensor, ss_blocks))
    n = sum(map(len, ss_blocks))
    adj_mat, ori_mat = torch.zeros(n, n), torch.zeros(n, n)
    for i, idxs_i in enumerate(ss_blocks):
        for j, idxs_j in enumerate(ss_blocks):
            if block_labels[i] == "C" or block_labels[j] == "C":
                if not include_loop_adjs:
                    continue
            if i == j and not include_self_loop:
                continue
            pair_key = (block_labels[i], block_labels[j])
            dist = torch.min(torch.cdist(ca_coords[idxs_i], ca_coords[idxs_j]))
            thresh = dist_thresh[pair_key]
            if dist < thresh:
                # adjacency information
                _idxs_i = repeat(idxs_i, "i-> b i", b=len(idxs_j))
                _idxs_j = repeat(idxs_j, "j-> j b", b=len(idxs_i))
                label = DEFAULT_SS_LABELS[pair_key] if use_adj_labels else 1
                adj_mat[_idxs_i, _idxs_j] = label
                # orientation information
                u, v = map(lambda idxs: ca_coords[idxs[-1]] - ca_coords[idxs[0]], (idxs_i, idxs_j))
                u_dot_v = torch.sum(u * v)
                ori_mat[_idxs_i, _idxs_j] = 1 if u_dot_v > 0 else -1

    return adj_mat, ori_mat

