"""Feature Generator ABC and Default implementation"""
from __future__ import annotations

from abc import abstractmethod
from typing import Optional, Dict

import torch
from torch import Tensor
from protein_learning.common.transforms import matrix_to_quaternion
from protein_learning.common.data.data_types.model_input import ExtraInput
from protein_learning.common.data.data_types.protein import Protein
from protein_learning.common.helpers import rotation_from_3_points, exists
from protein_learning.features.feature import Feature
from protein_learning.features.feature_config import InputFeatureConfig, FeatureTy, FeatureName
from protein_learning.features.feature_utils import res_ty_encoding, rel_pos_encoding, bb_dihedral_encoding, \
    degree_centrality_encoding, rel_sep_encoding, tr_rosetta_ori_encoding, rel_dist_encoding, local_rel_coords, \
    rel_ori_encoding, rel_chain_encoding, extra_encoding, sc_dihedral_encoding
from protein_learning.features.input_features import InputFeatures
from protein_learning.protein_utils.dihedral.orientation_utils import get_bb_dihedral, get_tr_rosetta_orientation_mats


class FeatureGenerator:
    """Feature Generator"""

    def __init__(
            self,
            config: InputFeatureConfig,
    ):
        self.config = config

    @abstractmethod
    def generate_features(
            self, protein: Protein,
            extra: Optional[ExtraInput] = None
    ) -> InputFeatures:
        """Generate input features"""
        pass


class DefaultFeatureGenerator(FeatureGenerator):
    """Default Feature Generator"""

    def __init__(
            self,
            config: InputFeatureConfig,
    ):
        super(DefaultFeatureGenerator, self).__init__(config)

    def generate_features(
            self,
            protein: Protein,
            extra: Optional[ExtraInput] = None,
    ) -> InputFeatures:
        """Generate input features"""
        feats = get_input_features(
            protein,
            config=self.config,
        )
        return InputFeatures(
            features=feats, batch_size=1, length=len(protein)
        ).maybe_add_batch()


def get_input_features(
        protein: Protein,
        config: InputFeatureConfig,
        extra_residue: Optional[Tensor] = None,
        extra_pair: Optional[Tensor] = None,
) -> Dict[str, Feature]:
    """
    :return: Dict mapping feature name to corresponding feature
    """
    coords, res_ids, seq = protein.atom_coords, protein.res_ids, protein.seq
    atom_ty_to_coord_idx, chain_ids = protein.atom_positions, protein.chain_ids
    atom_coords = lambda atom_ty: coords[..., atom_ty_to_coord_idx[atom_ty], :]

    feats = {}

    def add_feat(*features):
        """Adds feature to feats dict
        """
        for i, feat in enumerate(features):
            feats[feat.name] = feat

    if config.include_res_ty:
        add_feat(
            res_ty_encoding(
                seq=seq,
                corrupt_prob=config.res_ty_corrupt_prob
            )
        )

    if config.embed_sec_struct:
        assert exists(protein.sec_struct)
        add_feat(
            Feature(
                raw_data=protein.sec_struct,
                encoded_data=protein.secondary_structure_encoding.unsqueeze(-1),
                name=FeatureName.SS.value,
                ty=FeatureTy.RESIDUE,
                n_classes=3,
                dtype=torch.long,
            )
        )

    if config.include_rel_pos:
        add_feat(
            rel_pos_encoding(
                res_ids=res_ids,
                n_classes=config.res_rel_pos_encode_dim
            ),
        )

    if config.include_bb_dihedral:
        N, CA, C = [atom_coords(ty) for ty in ["N", "CA", "C"]]
        dihedrals = get_bb_dihedral(N=N, CA=CA, C=C)
        add_feat(
            bb_dihedral_encoding(
                bb_dihedrals=dihedrals,
                encode=True,
                n_classes=config.bb_dihedral_encode_dim,
            ),
        )

    if config.include_centrality:
        add_feat(
            degree_centrality_encoding(
                coords=atom_coords("CB" if "CB" in atom_ty_to_coord_idx else "CA"),
                chain_indices=protein.chain_indices,
                n_classes=config.centrality_encode_dim,
            ),
        )

    if config.include_rel_sep:
        add_feat(
            rel_sep_encoding(
                res_ids=res_ids,
                sep_bins=config.rel_sep_encode_bins
            ),
        )

    if config.include_tr_ori:
        N, CA, CB = [atom_coords(ty) for ty in ["N", "CA", "CB"]]
        phi, psi, omega = get_tr_rosetta_orientation_mats(N=N, CA=CA, CB=CB)
        add_feat(
            tr_rosetta_ori_encoding(
                tr_angles=(phi, psi, omega),
                encode=True,
                n_classes=config.tr_rosetta_ori_encode_dim,
            ),
        )

    if config.include_rel_dist:
        rel_dists = []
        for (a1, a2) in config.rel_dist_atom_pairs:
            c1, c2 = atom_coords(a1), atom_coords(a2)
            rel_dists.append(torch.cdist(c1, c2).unsqueeze(-1))
        add_feat(
            rel_dist_encoding(
                rel_dists=torch.cat(rel_dists, dim=-1),
                n_classes=config.rel_dist_encode_dim,
                dist_bounds=config.rel_dist_bounds,
            ),
        )

    quats = None
    if config.encode_local_rel_coords or config.quat_encode_rel_ori:
        assert coords.ndim == 3, \
            f"expected shape with 3 dimensions, got {coords.shape}"
        N, CA, C = [atom_coords(ty) for ty in ["N", "CA", "C"]]
        rots = rotation_from_3_points(N, CA, C)
        quats = matrix_to_quaternion(rots.reshape(coords.shape[0], 3, 3))

    if config.encode_local_rel_coords:
        add_feat(
            local_rel_coords(
                ca_coords=atom_coords("CA"), quats=quats
            ),
        )

    if config.quat_encode_rel_ori:
        add_feat(
            rel_ori_encoding(quats=quats),
        )

    if config.one_hot_rel_chain:
        assert exists(chain_ids)
        add_feat(
            rel_chain_encoding(chain_ids),
        )

    if config.has_extra:
        if config.extra_residue_dim > 0:
            add_feat(
                extra_encoding(extra_residue, FeatureTy.RESIDUE),
            )
        if config.extra_pair_dim > 0:
            add_feat(
                extra_encoding(extra_pair, FeatureTy.PAIR),
            )
    
    if config.include_sc_dihedral:
        assert coords.shape[-2] == 37
        noise,sc_coords=0,coords
        if config.sc_dihedral_noise[1] > 0:
            sc_coords = coords.clone()
            scale = config.sc_dihedral_noise[0]+torch.rand_like(coords)*(config.sc_dihedral_noise[1]-config.sc_dihedral_noise[0])
            noise = torch.randn_like(sc_coords)* scale
            noise[...,:4,:] = 0
            sc_coords = sc_coords+noise

        add_feat(
            sc_dihedral_encoding(coords=sc_coords,sequence_enc=protein.seq_encoding,mask=protein.atom_masks)
        )

    return feats
