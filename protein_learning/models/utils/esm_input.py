from __future__ import annotations

import os
import string
from typing import Optional, Tuple, List, Union, Dict

import esm  # noqa
import numpy as np
import torch.cuda
from Bio import SeqIO  # noqa
from einops import repeat, rearrange  # noqa
from scipy.spatial.distance import cdist
from torch import Tensor

from protein_learning.common.helpers import exists, default

# This is an efficient way to delete lowercase characters and insertion characters from a string
deletekeys = dict.fromkeys(string.ascii_lowercase)
deletekeys["."] = None
deletekeys["*"] = None
translation = str.maketrans(deletekeys)


def read_sequence(filename: str) -> Tuple[str, str]:
    """ Reads the first (reference) sequences from a fasta or MSA file."""
    record = next(SeqIO.parse(filename, "fasta"))
    return record.description, str(record.seq)


def remove_insertions(sequence: str) -> str:
    """ Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. """
    return sequence.translate(translation)


def read_msa(msa_dir: str, filename: Optional[str], ignore_missing: bool = False) -> Optional[List[Tuple[str, str]]]:
    """ Reads the sequences from an MSA file, automatically removes insertions."""
    if exists(filename):
        assert filename.endswith(".a3m")
    if not exists(filename) or not os.path.exists(os.path.join(msa_dir, filename)):
        if ignore_missing:
            return None
        raise Exception(f"MSA path {filename} does not exist!")
    return [
        (
            record.description,
            remove_insertions(str(record.seq))
        )
        for record in SeqIO.parse(filename, "fasta")
    ]


def crop_msa(inputs: List[Tuple[str, str]], idxs: Tensor) -> List[Tuple[str, str]]:
    """Crop MSA inputs from s to e"""
    assert len(inputs) > 0
    if len(inputs[0][1]) <= len(idxs):
        return inputs
    return [(x[0], "".join([x[1][i] for i in idxs])) for x in inputs]


def greedy_select(msa: List[Tuple[str, str]], num_seqs: int, mode: str = "max") -> List[Tuple[str, str]]:
    """Select sequences from the MSA to maximize the hamming distance.
    Alternatively, can use hhfilter
    """
    assert mode in ("max", "min")
    if len(msa) <= num_seqs:
        return msa

    array = np.array([list(seq) for _, seq in msa], dtype=np.bytes_).view(np.uint8)

    optfunc = np.argmax if mode == "max" else np.argmin
    all_indices = np.arange(len(msa))
    indices = [0]
    pairwise_distances = np.zeros((0, len(msa)))
    for _ in range(num_seqs - 1):
        dist = cdist(array[indices[-1:]], array, "hamming")
        pairwise_distances = np.concatenate([pairwise_distances, dist])
        shifted_distance = np.delete(pairwise_distances, indices, axis=1).mean(0)
        shifted_index = optfunc(shifted_distance)
        index = np.delete(all_indices, indices)[shifted_index]
        indices.append(index)
    indices = sorted(indices)
    return [msa[idx] for idx in indices]


def _chain_pair_masks(chain_indices: List[Tensor]) -> List[Tensor, Tensor]:
    assert len(chain_indices) == 2
    n_total = sum(map(len, chain_indices))
    masks = []
    for idxs in chain_indices:
        mask = torch.zeros(n_total, device=idxs.device)
        mask[idxs] = 1
        masks.append(torch.einsum("i,j->ij", mask, mask).bool())
    return masks


class ESMFeatGen:
    """Generate features from ESM-1b or ESM-MSA

    see: https://github.com/facebookresearch/esm
    """

    def __init__(
            self,
            use_esm_msa: bool = False,
            use_esm_1b: bool = False,
            esm_gpu_idx: Optional[int] = None,
            msa_dir: Optional[str] = None,
            max_msa_seqs: Optional[int] = 128,
    ):
        self.use_esm_msa, self.use_esm_1b = use_esm_msa, use_esm_1b
        self.msa_dir = msa_dir
        self.esm_device = f"cuda:{esm_gpu_idx}" if torch.cuda.is_available() else "cpu"
        self.max_msa_seqs = max_msa_seqs

        assert (use_esm_msa ^ use_esm_1b), f"must specify exactly one of esm-1b or esm-msa"

        if self.use_esm_msa:
            esm_model, esm_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
        else:
            esm_model, esm_alphabet = esm.pretrained.esm1b_t33_650M_UR50S()

        self.esm_model = esm_model.eval().to(self.esm_device)
        self.esm_batch_converter = esm_alphabet.get_batch_converter()
        # keys for getting node and pair feats from esm output
        self.esm_node_key = "representations"
        self.esm_node_repr_idx = 33 if use_esm_1b else 12
        self.esm_pair_key = "attentions" if use_esm_1b else "row_attentions"

        # remove start/end tokens prior to embedding
        self.repr_offsets = (1, 1) if use_esm_1b else (1, 0)



    def get_esm_feats(
            self,
            chain_indices: List[Tensor],
            seqs: Optional[Union[str, List[str]]] = None,
            seq_masks: Optional[Union[Tensor, List[Tensor]]] = None,
            msa_files: Optional[Union[str, List[str]]] = None,
            msa_crops: Optional[List[Tensor]] = None,
    ) -> Dict:
        """Get ESM features

        output dict has keys:
            "node":esm node features (b,n,node_dim)
            "pair":esm pair features (b,n,n,pair_dim)
        """
        cast_list = lambda x: x if isinstance(x, list) else [x]
        with torch.no_grad():
            if self.use_esm_msa and exists(msa_files):
                msa_files = cast_list(msa_files)
                msa_crops = default(msa_crops, [None] * len(msa_files))
                assert len(msa_files) == len(chain_indices)
                out = self._get_esm_msa_embs(msa_files, msa_crops)
            elif self.use_esm_1b and exists(seqs):
                seqs = cast_list(seqs)
                assert len(seqs) == len(chain_indices)
                seq_masks = default(seq_masks, [None] * len(seqs))
                out = self._get_esm_1b_embs(seqs, seq_masks)
            else:
                out = [None] * len(chain_indices)
        return self.merge_output(out, chain_indices)

    def _get_esm_msa_embs(
            self,
            msa_files: List[Optional[str]],
            msa_crops: List[Optional[Tensor]],
    ) -> List[Optional[Dict]]:
        output = []
        for msa_file, msa_crop in zip(msa_files, msa_crops):
            inputs = read_msa(msa_dir=self.msa_dir, filename=msa_file, ignore_missing=True)
            if not exists(inputs):
                output.append(None)
                continue
            if exists(msa_crop):
                inputs = crop_msa(inputs, msa_crop)
            inputs = greedy_select(inputs, num_seqs=self.max_msa_seqs)
            _, _, batch_tokens = self.esm_batch_converter([inputs])
            batch_tokens = batch_tokens.to(self.esm_device)
            out = self.esm_model(batch_tokens, repr_layers=[12], return_contacts=True)
            out = dict(
                node=out[self.esm_node_key][self.esm_node_repr_idx].cpu(),
                pair=out[self.esm_pair_key].cpu()
            )
            output.append(out)
        return output

    def _get_esm_1b_embs(self, seqs: List[Tensor], seq_masks: List[Optional[Tensor]]) -> List[Optional[Dict]]:
        output = []
        assert len(seqs) == len(seq_masks)
        for seq, mask in zip(seqs, seq_masks):
            if not exists(seq):
                output.append(None)
                continue
            if exists(mask):
                seq = " ".join(["<mask>" if mask[i] else seq[i] for i in range(len(seq))])
            _, _, batch_tokens = self.esm_batch_converter([("tmp", seq)])
            out = self.esm_model(
                batch_tokens.to(self.esm_device),
                repr_layers=[33],
                return_contacts=True
            )
            out = dict(
                node=out[self.esm_node_key][self.esm_node_repr_idx].cpu(),
                pair=out[self.esm_pair_key].cpu()
            )
            output.append(out)
        return output

    def merge_output(self, output: List[Dict], chain_indices: List[Tensor]) -> Dict:
        """Merge ESM output (for multiple chains)"""
        # Missing output
        missing_indices, node_feats, pair_feats = [], [], []
        for out, indices in zip(output, chain_indices):
            if out is None:
                missing_indices.append(indices)
                continue
            # get node features
            node_repr = out["node"][:, 0] if self.use_esm_msa else out["node"]
            node_feats.append(node_repr[:, 1:len(indices) + 1])

            # get pair features
            pair_repr = rearrange(out["pair"], "b l h n m -> b n m (l h)")
            pair_feats.append(pair_repr[:, 1:len(indices) + 1, 1:len(indices) + 1])

        # merge features
        missing_indices = torch.cat(missing_indices) if len(missing_indices) > 0 else None
        node_feats = torch.cat(node_feats, dim=-2) if len(node_feats) > 0 else None
        # more tedius concatenation with pair feats
        if len(pair_feats) <= 1:
            pair_feats = pair_feats[0] if len(pair_feats) == 1 else None
        else:
            # combine in block-diag form
            masks = _chain_pair_masks(chain_indices)
            n, b = sum(map(len, chain_indices)), pair_feats[0].shape[0]
            _pair_feats = torch.zeros(b, n, n, pair_feats[0].shape[-1], device=pair_feats[0].device)
            for feats, mask in zip(pair_feats, masks):
                _pair_feats[:, mask] = rearrange(feats, "b n m d -> b (n m) d")
            pair_feats = _pair_feats

        return {"node": node_feats, "pair": pair_feats, "missing_indices": missing_indices}
