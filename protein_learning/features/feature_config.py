"""Configuration for Input Feature Generation
"""
from __future__ import annotations

from enum import Enum
from typing import NamedTuple, List, Tuple, Union, Optional
import numpy as np 
from protein_learning.common.protein_constants import AA_ALPHABET
from protein_learning.features.constants import SMALL_SEP_BINS, DEFAULT_PW_DIST_RADII


class FeatureDescriptor(NamedTuple):
    """Descriptor for a given feature"""
    name: FeatureName
    ty: FeatureTy
    encode_dim: int
    embed_dim: int
    embed_tys: List[FeatureEmbeddingTy]
    # Optional - for embeddings
    mult: Optional[int] = 1
    rbf_sigma: Optional[float] = 4
    rbf_radii: Optional[List[float]] = None
    n_fourier_feats: Optional[int] = None


class InputFeatureConfig(NamedTuple):
    """Config Input Features"""
    # Global settings
    # pad embeddings with an extra bin (n_classes+1) - can be used
    # in conjunction with sequence or coordinate masking, e.g.
    pad_embeddings: bool = False
    # add noise to one hot encodings?
    one_hot_noise_sigma: float = 0  # set>0 to have this amount of random noise added

    # Residue Type (SCALAR)
    embed_res_ty: bool = False
    one_hot_res_ty: bool = False
    res_ty_embed_dim: int = 32
    res_ty_corrupt_prob: float = 0

    # Residue Relative Position (SCALAR)
    embed_res_rel_pos: bool = False
    one_hot_res_rel_pos: bool = False
    res_rel_pos_embed_dim: int = 6
    res_rel_pos_encode_dim: int = 10

    # BB Dihedral (SCALAR)
    embed_bb_dihedral: bool = False
    one_hot_bb_dihedral: bool = False
    fourier_encode_bb_dihedral: bool = False
    bb_dihedral_embed_dim: int = 6
    bb_dihedral_encode_dim: int = 36
    n_bb_dihedral_fourier_feats: int = 2

    # SC Dihedral (SCALAR)
    include_sc_dihedral: bool = False
    sc_dihedral_noise: List[float] = [0,0]

    # Centrality (SCALAR)
    embed_centrality: bool = False
    one_hot_centrality: bool = False
    rbf_encode_centrality: bool = False
    centrality_encode_dim: int = 6
    centrality_embed_dim: int = 6
    centrality_rbf_radii: List[float] = [6, 12, 18, 24, 30, 36]

    # Secondary Structure
    embed_sec_struct: bool = False
    sec_struct_embed_dim: int = 16

    # Relative Separation (PAIR)
    embed_rel_sep: bool = False
    one_hot_rel_sep: bool = False
    rel_sep_embed_dim: int = 32
    rel_sep_encode_bins: List[int] = SMALL_SEP_BINS

    # Relative Distance (PAIR)
    embed_rel_dist: bool = False
    one_hot_rel_dist: bool = False
    rbf_encode_rel_distance: bool = False
    rel_dist_embed_dim: int = 16
    rel_dist_encode_dim: int = 32
    rel_dist_bounds: Tuple[float, float] = (2.5, 16.5)
    rel_dist_rbf_radii: List[float] = DEFAULT_PW_DIST_RADII
    rel_dist_atom_tys: List[str] = ["CA", "CA", "N", "CA"]

    # trRosetta Orientation (PAIR)
    embed_tr_rosetta_ori: bool = False
    one_hot_tr_rosetta_ori: bool = False
    fourier_encode_tr_rosetta_ori: bool = False
    tr_rosetta_ori_embed_dim: int = 6
    tr_rosetta_ori_encode_dim: int = 36
    tr_rosetta_fourier_feats: int = 2

    # Joint Embedding for Pair and Sep (PAIR)
    joint_embed_res_pair_rel_sep: bool = False
    joint_embed_res_pair_rel_sep_embed_dim: int = 48

    # Invariant relative orientation features
    quat_encode_rel_ori: bool = False
    encode_local_rel_coords: bool = False

    # relative chain embedding
    one_hot_rel_chain: bool = False

    # Extra
    extra_residue_dim: int = 0
    extra_pair_dim: int = 0
    coord_noise: float = 0

    @property
    def include_res_ty(self):
        """Whether to include residue type features"""
        return self.joint_embed_res_pair_rel_sep \
               or self.embed_res_ty \
               or self.one_hot_res_ty  # noqa

    @property
    def include_rel_pos(self):
        """Whether to include residue position features"""
        return self.embed_res_rel_pos or self.one_hot_res_rel_pos

    @property
    def include_bb_dihedral(self):
        """Whether to include bb dihedral features"""
        return self.embed_bb_dihedral or self.one_hot_bb_dihedral or \
               self.fourier_encode_bb_dihedral  # noqa

    @property
    def include_centrality(self):
        """Whether to include centrality features"""
        return self.embed_centrality or self.one_hot_centrality or self.rbf_encode_centrality

    @property
    def include_rel_sep(self):
        """Whether to include relative separation features"""
        return self.one_hot_rel_sep or self.embed_rel_sep \
               or self.joint_embed_res_pair_rel_sep  # noqa

    @property
    def include_rel_dist(self):
        """Whether to include relative distance features"""
        return self.one_hot_rel_dist or self.embed_rel_dist or self.rbf_encode_rel_distance

    @property
    def rel_dist_atom_pairs(self) -> List[Tuple[str, str]]:
        """atom pairs to use for relative distance"""
        tys = self.rel_dist_atom_tys
        return list(zip(tys[::2], tys[1::2]))

    @property
    def include_tr_ori(self):
        """Whether to include TrRosetta orientation features

        Note: CB atom must be present to sue these features - can add
        "impute_cb" flag to data loader if this atom is not available.
        """
        return self.embed_tr_rosetta_ori or self.fourier_encode_tr_rosetta_ori \
               or self.one_hot_tr_rosetta_ori  # noqa

    @property
    def include_rel_ori(self):
        """Whether to include relative orientation information
        e.g. quaternion of (pairwise) relative orientation matrices.
        """
        return self.quat_encode_rel_ori

    @property
    def has_extra(self):
        """Whether the input features include flags"""
        return self.extra_residue_dim > 0 or self.extra_pair_dim > 0

    @property
    def descriptors(self) -> List[FeatureDescriptor]:
        """Get a list of FeatureDescriptors
        For all features specified by the config
        """
        descriptors = []

        if self.embed_sec_struct:
            descriptors.append(
                FeatureDescriptor(
                    name=FeatureName.SS,
                    ty=FeatureTy.RESIDUE,
                    encode_dim=3,
                    embed_dim=self.sec_struct_embed_dim,
                    embed_tys=[FeatureEmbeddingTy.EMBED]
                )
            )

        if self.include_res_ty:
            descriptors.append(
                FeatureDescriptor(
                    name=FeatureName.RES_TY,
                    ty=FeatureTy.RESIDUE,
                    encode_dim=len(AA_ALPHABET),
                    embed_dim=self.res_ty_embed_dim,
                    embed_tys=_get_encoding_tys(
                        embed=self.embed_res_ty,
                        one_hot=self.one_hot_res_ty
                    )

                )

            )

        if self.include_rel_pos:
            descriptors.append(
                FeatureDescriptor(
                    name=FeatureName.REL_POS,
                    ty=FeatureTy.RESIDUE,
                    encode_dim=self.res_rel_pos_encode_dim,
                    embed_dim=self.res_rel_pos_embed_dim,
                    embed_tys=_get_encoding_tys(
                        embed=self.embed_res_rel_pos,
                        one_hot=self.one_hot_res_rel_pos
                    )
                )

            )

        if self.include_bb_dihedral:
            descriptors.append(
                FeatureDescriptor(
                    name=FeatureName.BB_DIHEDRAL,
                    ty=FeatureTy.RESIDUE,
                    encode_dim=self.bb_dihedral_encode_dim,
                    embed_dim=self.bb_dihedral_embed_dim,
                    embed_tys=_get_encoding_tys(
                        embed=self.embed_bb_dihedral,
                        one_hot=self.one_hot_bb_dihedral,
                        fourier=self.fourier_encode_bb_dihedral,
                    ),
                    mult=3,
                    n_fourier_feats=self.n_bb_dihedral_fourier_feats
                ),
            )

        if self.include_centrality:
            descriptors.append(
                FeatureDescriptor(
                    name=FeatureName.CENTRALITY,
                    ty=FeatureTy.RESIDUE,
                    encode_dim=self.centrality_encode_dim,
                    embed_dim=self.centrality_embed_dim,
                    embed_tys=_get_encoding_tys(
                        embed=self.embed_centrality,
                        one_hot=self.one_hot_centrality,
                        rbf=self.rbf_encode_centrality,
                    ),
                    rbf_radii=self.centrality_rbf_radii,
                ),
            )

        if self.include_rel_sep:
            descriptors.append(
                FeatureDescriptor(
                    name=FeatureName.REL_SEP,
                    ty=FeatureTy.PAIR,
                    encode_dim=len(self.rel_sep_encode_bins),
                    embed_dim=self.rel_sep_embed_dim,
                    embed_tys=_get_encoding_tys(
                        embed=self.embed_rel_sep,
                        one_hot=self.one_hot_rel_sep,
                    ),
                ),
            )

        if self.include_rel_dist:
            descriptors.append(
                FeatureDescriptor(
                    name=FeatureName.REL_DIST,
                    ty=FeatureTy.PAIR,
                    encode_dim=self.rel_dist_encode_dim,
                    embed_dim=self.rel_dist_embed_dim,
                    embed_tys=_get_encoding_tys(
                        embed=self.embed_rel_dist,
                        one_hot=self.one_hot_rel_dist,
                        rbf=self.rbf_encode_rel_distance,
                    ),
                    mult=len(self.rel_dist_atom_tys) // 2,
                    rbf_sigma=np.mean([(x-y) for x,y in zip(self.rel_dist_rbf_radii[1:],self.rel_dist_rbf_radii[:-1])]),
                    rbf_radii=self.rel_dist_rbf_radii,
                ),
            )

        if self.include_tr_ori:
            descriptors.append(
                FeatureDescriptor(
                    name=FeatureName.TR_ORI,
                    ty=FeatureTy.PAIR,
                    encode_dim=self.tr_rosetta_ori_encode_dim,
                    embed_dim=self.tr_rosetta_ori_embed_dim,
                    embed_tys=_get_encoding_tys(
                        embed=self.embed_tr_rosetta_ori,
                        one_hot=self.one_hot_tr_rosetta_ori,
                        fourier=self.fourier_encode_tr_rosetta_ori,
                    ),
                    mult=3,
                    n_fourier_feats=self.tr_rosetta_fourier_feats,

                ),
            )

        if self.quat_encode_rel_ori:
            descriptors.append(
                FeatureDescriptor(
                    name=FeatureName.REL_ORI,
                    ty=FeatureTy.PAIR,
                    encode_dim=4,
                    embed_dim=4,
                    embed_tys=[FeatureEmbeddingTy.NONE]
                ),
            )

        if self.encode_local_rel_coords:
            descriptors.append(
                FeatureDescriptor(
                    name=FeatureName.REL_COORD,
                    ty=FeatureTy.PAIR,
                    encode_dim=3,
                    embed_dim=3,
                    embed_tys=[FeatureEmbeddingTy.NONE]
                ),
            )

        if self.one_hot_rel_chain:
            descriptors.append(
                FeatureDescriptor(
                    name=FeatureName.REL_CHAIN,
                    ty=FeatureTy.PAIR,
                    encode_dim=5,
                    embed_dim=5,
                    embed_tys=[FeatureEmbeddingTy.ONEHOT]
                ),
            )

        if self.extra_pair_dim > 0:
            descriptors.append(
                FeatureDescriptor(
                    name=FeatureName.EXTRA_PAIR,
                    ty=FeatureTy.PAIR,
                    encode_dim=self.extra_pair_dim,
                    embed_dim=self.extra_pair_dim,
                    embed_tys=[FeatureEmbeddingTy.NONE]
                ),
            )

        if self.extra_residue_dim > 0:
            descriptors.append(
                FeatureDescriptor(
                    name=FeatureName.EXTRA_RES,
                    ty=FeatureTy.RESIDUE,
                    encode_dim=self.extra_residue_dim,
                    embed_dim=self.extra_residue_dim,
                    embed_tys=[FeatureEmbeddingTy.NONE]
                ),
            )

        if self.include_sc_dihedral:
            descriptors.append(
                FeatureDescriptor(
                    name=FeatureName.SC_DIHEDRAL,
                    ty=FeatureTy.RESIDUE,
                    embed_tys=[FeatureEmbeddingTy.NONE],
                    encode_dim=21,
                    embed_dim=21,
                ),
            )

        return descriptors

    def include_feat(self, name: Union[str, FeatureName]) -> bool:
        """Whether the config specifies the given feature"""
        name = name if isinstance(name, str) else name.value
        if name == FeatureName.REL_POS.value:
            return self.include_rel_pos
        if name == FeatureName.REL_SEP.value:
            return self.include_rel_sep
        if name == FeatureName.REL_DIST.value:
            return self.include_rel_dist
        if name == FeatureName.BB_DIHEDRAL.value:
            return self.include_bb_dihedral
        if name == FeatureName.CENTRALITY.value:
            return self.include_centrality
        if name == FeatureName.RES_TY.value:
            return self.include_res_ty
        if name == FeatureName.TR_ORI.value:
            return self.include_tr_ori
        if name == FeatureName.REL_ORI:
            return self.include_rel_ori
        if name == FeatureName.REL_COORD:
            return self.encode_local_rel_coords
        if name == FeatureName.REL_CHAIN:
            return self.one_hot_rel_chain
        if name == FeatureName.EXTRA_RES:
            return self.extra_residue_dim > 0
        if name == FeatureName.EXTRA_PAIR:
            return self.extra_pair_dim > 0
        if name == FeatureName.SC_DIHEDRAL:
            return self.include_sc_dihedral


class FeatureTy(Enum):
    """Feature Type flag
    """
    RESIDUE, COORD, PAIR = 1, 2, 3


class FeatureEmbeddingTy(Enum):
    """Feature Type flag
    """
    RBF, ONEHOT, EMBED, FOURIER, NONE = 0, 1, 2, 3, 4


def _get_encoding_tys(one_hot=False, embed=False, rbf=False, fourier=False) -> List[FeatureEmbeddingTy]:
    """Encoding type for feature"""
    encoding_tys = []
    if one_hot:
        encoding_tys.append(FeatureEmbeddingTy.ONEHOT)
    if embed:
        encoding_tys.append(FeatureEmbeddingTy.EMBED)
    if rbf:
        encoding_tys.append(FeatureEmbeddingTy.RBF)
    if fourier:
        encoding_tys.append(FeatureEmbeddingTy.FOURIER)
    return encoding_tys


class FeatureName(Enum):
    """Identifiers for each feature type
    """
    REL_POS = "rel_pos"
    REL_SEP = "rel_sep"
    REL_DIST = "rel_dist"
    BB_DIHEDRAL = "bb_dihedral"
    CENTRALITY = "centrality"
    RES_TY = "res_ty"
    TR_ORI = "tr_ori"
    REL_ORI = "rel_ori"
    REL_COORD = "rel_coord"
    REL_CHAIN = "rel_chain"
    EXTRA_RES = "extra_res"
    EXTRA_PAIR = "extra_pair"
    SS = "sec_struct"
    SC_DIHEDRAL="sc_dihedral"


FEATURE_NAMES = [
    FeatureName.REL_POS,
    FeatureName.REL_SEP,
    FeatureName.REL_DIST,
    FeatureName.BB_DIHEDRAL,
    FeatureName.CENTRALITY,
    FeatureName.RES_TY,
    FeatureName.TR_ORI,
    FeatureName.REL_ORI,
    FeatureName.REL_COORD,
    FeatureName.REL_CHAIN,
    FeatureName.EXTRA_RES,
    FeatureName.EXTRA_PAIR,
    FeatureName.SS,
    FeatureName.SC_DIHEDRAL,
]
