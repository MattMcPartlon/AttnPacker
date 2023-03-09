"""Input Feature Embedding
"""
from typing import Tuple, Optional, List, Union, Dict

import torch
from einops import rearrange, repeat  # noqa
from torch import Tensor, nn
from torch.nn.functional import one_hot as to_one_hot

from protein_learning.common.global_constants import get_logger
from protein_learning.common.helpers import exists, default
from protein_learning.common.protein_constants import AA_ALPHABET
from protein_learning.features.feature import Feature
from protein_learning.features.feature_config import (
    InputFeatureConfig,
    FeatureTy,
    FeatureName,
    FeatureEmbeddingTy,
    FEATURE_NAMES,
)
from protein_learning.features.feature_utils import fourier_encode
from protein_learning.features.input_features import InputFeatures

logger = get_logger(__name__)


class RawFeatEncoding(nn.Module):  # noqa
    """Raw feature encoding (identity encoding)"""

    def __init__(self, embed_dim):
        super(RawFeatEncoding, self).__init__()
        self.embed_dim = embed_dim

    @property
    def embedding_dim(self) -> int:
        """Raw feature dimension"""
        return self.embed_dim

    def forward(self, feat: Feature) -> Tensor:  # noqa
        """Pass along the raw feature data"""
        return feat.get_raw_data()


class FeatEmbedding(nn.Module):  # noqa
    """One Hot encoding (wrapped, so it can be used in module dict)"""

    def __init__(self, num_classes, embed_dim, mult=1):
        super(FeatEmbedding, self).__init__()
        self.mult, self.offsets = mult, None
        self.embed_dim, self.num_classes = embed_dim, num_classes
        self.embedding = nn.Embedding(mult * num_classes, embed_dim)

    @property
    def embedding_dim(self) -> int:
        """flattened shape of embedding"""
        return self.mult * self.embed_dim

    def get_offsets(self, feat: Tensor):
        """values to shift bins by for case of multiple embeddings"""
        if not exists(self.offsets):
            offsets = [i * self.num_classes for i in range(self.mult)]
            self.offsets = torch.tensor(offsets, device=feat.device, dtype=torch.long)
        return self.offsets

    def forward(self, feat: Feature):
        """Embed the feature"""
        to_emb = feat.get_encoded_data()
        assert to_emb.shape[-1] == self.mult, f"{to_emb.shape},{self.mult},{feat.name}"
        max_token = torch.max(to_emb + self.get_offsets(to_emb))
        min_token = torch.min(to_emb + self.get_offsets(to_emb))
        expected_max_token = self.mult * self.num_classes
        assert min_token >= 0, f"{feat.name}:{max_token},{expected_max_token}"
        assert max_token < expected_max_token, f"{feat.name}:{max_token},{expected_max_token}"
        return self.embedding(to_emb + self.get_offsets(to_emb))


class FeatOneHotEncoding(nn.Module):  # noqa
    """One Hot encoding (wrapped, so it can be used in module dict)"""

    def __init__(self, num_classes, mult: int = 1, std_noise: float = 0):
        """One hot feature encoding

        :param num_classes: number of classes to encode
        :param mult: trailing dimension of input features
        :param std_noise: [Optional] noise standard deviation (added to encodings)
        """
        super(FeatOneHotEncoding, self).__init__()
        self.mult, self.num_classes = mult, num_classes
        self.std_noise = std_noise

    @property
    def embedding_dim(self) -> int:
        """flattened shape of embedding"""
        return self.mult * self.num_classes

    def forward(self, feat: Feature):
        """One hot encode the features"""
        to_hot = feat.get_encoded_data()
        assert to_hot.shape[-1] == self.mult, f"{to_hot.shape},{self.mult},{feat.name}"
        assert torch.min(to_hot) >= 0, f"{feat.name}:{torch.min(to_hot)},{self.num_classes}"
        assert torch.max(to_hot) < self.num_classes, f"{feat.name}:{torch.max(to_hot)},{self.num_classes}"
        encoding = to_one_hot(to_hot, self.num_classes)
        return encoding + torch.randn_like(encoding.float()) * self.std_noise


class RBFEncoding(nn.Module):  # noqa
    """RBF Encoding (wrapped, so it can be used in module dict)"""

    def __init__(self, radii: List[float], mult: int = 1, sigma: float = 4):
        """RBF feature encoding"""
        super(RBFEncoding, self).__init__()
        radii = torch.tensor(radii).float()
        self.radii = repeat(radii, "r -> m r", m=mult)
        self.mult = mult
        self.sigma_sq = default(sigma**2, torch.mean(radii[:-1] - radii[1:]))
        self.embedding_dim = radii.numel() * self.mult

    def forward(self, feat: Feature):
        """RBF - encode the features"""
        raw_data = feat.get_raw_data()
        assert raw_data.shape[-1] == self.mult, f"{raw_data.shape},{self.mult},{feat.name}"
        raw_data = raw_data.unsqueeze(-1)
        shape_diff = raw_data.ndim - self.radii.ndim
        radii = self.radii[(None,) * shape_diff].to(raw_data.device)
        rbf = torch.exp(-torch.clamp_max(torch.square((radii - raw_data) / self.sigma_sq), 50))
        return rbf


class FeatFourierEncoding(nn.Module):  # noqa
    """Fourier (sin and cos) encoding (wrapped so it can be used in module dict)"""

    def __init__(self, n_feats, include_self=False, mult: int = 1):  # noqa
        super(FeatFourierEncoding, self).__init__()
        self.n_feats, self.include_self, self.mult = n_feats, include_self, mult
        self.num_embeddings = 2 * self.n_feats + (1 if include_self else 0)

    @property
    def embedding_dim(self) -> int:
        """flattened shape of embedding"""
        return self.mult * self.num_embeddings

    def forward(self, feat: Feature):
        """fourier encode the features"""
        with torch.no_grad():
            to_encode = feat.get_raw_data()
            assert to_encode.shape[-1] == self.mult, f"{to_encode.shape},{self.mult},{feat.name}"
            return fourier_encode(feat.get_raw_data(), num_encodings=self.n_feats, include_self=self.include_self)


def count_embedding_dim(embeddings) -> int:
    """sums output dimension of list of embeddings"""
    return sum([e.embedding_dim for e in embeddings])


def get_embeddings(
    fourier_feats: int = None,
    n_classes: int = None,
    embed_dim: int = None,
    mult: int = 1,
    rbf_radii: Optional[List[float]] = None,
    rbf_sigma: Optional[float] = 4,
    std_noise: float = 0,
    embed_tys: List[FeatureEmbeddingTy] = None,
) -> nn.ModuleDict:
    """Gets embedding dict for input"""
    embeddings = nn.ModuleDict()
    if FeatureEmbeddingTy.EMBED in embed_tys:
        embeddings["emb"] = FeatEmbedding(n_classes, embed_dim, mult=mult)
    if FeatureEmbeddingTy.ONEHOT in embed_tys:
        embeddings["one_hot"] = FeatOneHotEncoding(num_classes=n_classes, mult=mult, std_noise=std_noise)
    if FeatureEmbeddingTy.FOURIER in embed_tys:
        embeddings["fourier"] = FeatFourierEncoding(n_feats=fourier_feats, mult=mult)
    if FeatureEmbeddingTy.NONE in embed_tys:
        embeddings["identity"] = RawFeatEncoding(embed_dim)
    if FeatureEmbeddingTy.RBF in embed_tys:
        assert exists(rbf_radii), "must include radii with rbf encoding!"
        embeddings["rbf"] = RBFEncoding(rbf_radii, mult=mult, sigma=rbf_sigma)

    return embeddings


class InputEmbedding(nn.Module):  # noqa
    """Input Embedding"""

    def __init__(self, feature_config: InputFeatureConfig):
        super(InputEmbedding, self).__init__()
        self.config = feature_config
        dims, embeddings = self._feat_dims_n_embeddings(feature_config)
        self.scalar_dim, self.pair_dim = dims
        self.scalar_embeddings, self.pair_embeddings = embeddings
        # _print_embedding_info(self.scalar_embeddings, self.pair_embeddings, self.scalar_dim, self.pair_dim)

    def forward(self, feats: InputFeatures) -> Tuple[Tensor, Tensor]:
        """Get pair and scalar input features"""
        leading_shape = (feats.batch_size, feats.length)
        scalar_feats = self.get_scalar_input(feats, leading_shape)
        pair_feats = self.get_pair_input(feats, (*leading_shape, leading_shape[-1]))  # noqa
        # logger.info(f"scalar shape : {scalar_feats.shape}, expected : {self.scalar_dim}")
        # logger.info(f"pair shape : {pair_feats.shape}, expected : {self.pair_dim}")
        return scalar_feats.float(), pair_feats.float()

    @property
    def dims(self) -> Tuple[int, int]:
        """Scalar and Pair Feature dimension"""
        return self.scalar_dim, self.pair_dim

    @staticmethod
    def _feat_dims_n_embeddings(
        config: InputFeatureConfig,
    ) -> Tuple[Tuple[int, int], Tuple[nn.ModuleDict, nn.ModuleDict]]:
        """gets scalar and pair input dimensions as well as embedding/encoding
        functions for each input feature.

        Feature types can be found in features/input_features.py
        """
        pad = 1 if config.pad_embeddings else 0
        scalar_dim, pair_dim = 0, 0
        scalar_embeddings, pair_embeddings = nn.ModuleDict(), nn.ModuleDict()
        get_embed_dict = lambda ty: scalar_embeddings if ty == FeatureTy.RESIDUE else pair_embeddings
        for descriptor in config.descriptors:
            name = descriptor.name.value
            # print(f"[INFO] adding embeddings for {name}:"
            #      f" {[e.value for e in descriptor.embed_tys]}")
            embed_dict = get_embed_dict(descriptor.ty)
            embed_dict[name] = get_embeddings(
                embed_tys=descriptor.embed_tys,
                embed_dim=descriptor.embed_dim,
                n_classes=descriptor.encode_dim + pad,
                std_noise=config.one_hot_noise_sigma,
                fourier_feats=descriptor.n_fourier_feats,
                rbf_sigma=descriptor.rbf_sigma,
                rbf_radii=descriptor.rbf_radii,
                mult=descriptor.mult,
            )

        if config.joint_embed_res_pair_rel_sep:
            name = "joint_pair_n_sep"
            logger.info(f"adding embeddings for {name}")
            pair_dim -= 2 * config.joint_embed_res_pair_rel_sep_embed_dim
            pair_embeddings[name] = nn.ModuleDict()
            pair_embeddings[name][FeatureName.REL_SEP.value] = FeatEmbedding(
                len(config.rel_sep_encode_bins) + pad, config.joint_embed_res_pair_rel_sep_embed_dim
            )
            pair_embeddings[name][FeatureName.RES_TY.value + "_a"] = FeatEmbedding(
                len(AA_ALPHABET) + pad, config.joint_embed_res_pair_rel_sep_embed_dim
            )
            pair_embeddings[name][FeatureName.RES_TY.value + "_b"] = FeatEmbedding(
                len(AA_ALPHABET) + pad, config.joint_embed_res_pair_rel_sep_embed_dim
            )

        scalar_dim += sum([count_embedding_dim(e.values()) for e in scalar_embeddings.values()])
        pair_dim += sum([count_embedding_dim(e.values()) for e in pair_embeddings.values()])
        return (scalar_dim, pair_dim), (scalar_embeddings, pair_embeddings)

    def get_feat_dict(
        self,
        features: InputFeatures,
        n: int,
        feat_names: Optional[List[FeatureName]] = None,
        only_scalar: bool = False,
        only_pair: bool = False,
    ):
        feat_dict = {}
        feat_names = default(feat_names, FEATURE_NAMES)
        for feat_id in feat_names:
            if feat_id.value not in features:
                continue
            is_scalar = feat_id.value in self.scalar_embeddings
            is_pair = feat_id.value in self.pair_embeddings
            if not (is_pair or is_scalar):
                continue
            if (only_pair and is_scalar) or (only_scalar and is_pair):
                continue
            embs = self.scalar_embeddings if is_scalar else self.pair_embeddings
            feat = features[feat_id.value]
            feat_embeddings = embs[feat_id.value]
            leading_shape = (1, n) if is_scalar else (1, n, n)
            if feat.name == FeatureName.REL_SEP.value:
                rs_bins = len(self.config.rel_sep_encode_bins) + int(self.config.pad_embeddings)
                rel_sep = features[FeatureName.REL_SEP.value].get_encoded_data()
                one_hot = torch.nn.functional.one_hot(rel_sep, rs_bins)
                feat_dict[FeatureName.REL_SEP.value] = one_hot.reshape(1, n, n, -1)
                continue
            for emb_name, emb in feat_embeddings.items():
                emb_feat = emb(feat).reshape(*leading_shape, -1)
                feat_dict[feat.name] = emb_feat
        return feat_dict

    def get_scalar_input(
        self,
        features: InputFeatures,
        leading_shape: Tuple[int, int],
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
        """Get scalar input"""
        scalar_feats, feat_dict = [], {}
        for feat_id in FEATURE_NAMES:
            if feat_id.value not in features:
                continue
            feat_name, feat = feat_id.value, features[feat_id.value]
            if feat.ty == FeatureTy.RESIDUE:
                if feat.name not in self.scalar_embeddings:
                    continue
                feat_embeddings = self.scalar_embeddings[feat.name]
                for emb_name, emb in feat_embeddings.items():
                    emb_feat = emb(feat).reshape(*leading_shape, -1)
                    scalar_feats.append(emb_feat)
        scalar_feats = torch.cat(scalar_feats, dim=-1) if len(scalar_feats) > 0 else None
        return scalar_feats

    def get_pair_input(
        self,
        features: InputFeatures,
        leading_shape: Tuple[int, int, int],
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
        """Get pair input"""
        pair_feats, feat_dict = [], {}
        b, n = leading_shape[:2]
        for feat_id in FEATURE_NAMES:
            if feat_id.value not in features:
                continue
            feat_name, feat = feat_id.value, features[feat_id.value]
            if feat.ty == FeatureTy.PAIR:
                if feat.name not in self.pair_embeddings:
                    continue
                feat_embeddings = self.pair_embeddings[feat.name]
                for emb_name, emb in feat_embeddings.items():
                    emb_feat = emb(feat).reshape(*leading_shape, -1)
                    pair_feats.append(emb_feat)

        # optional joint embedding
        if "joint_pair_n_sep" in self.pair_embeddings:
            joint_embs = self.pair_embeddings["joint_pair_n_sep"]
            res_ty = features[FeatureName.RES_TY.value]
            sep = features[FeatureName.REL_SEP.value]
            emb_sep = joint_embs[FeatureName.REL_SEP.value](sep).reshape(*leading_shape, -1)
            emb_a = joint_embs[FeatureName.RES_TY.value + "_a"](res_ty).reshape(*leading_shape[:-1], -1)
            emb_b = joint_embs[FeatureName.RES_TY.value + "_b"](res_ty).reshape(*leading_shape[:-1], -1)
            joint_emb = (
                rearrange(emb_a, "... n d-> ... n () d") + rearrange(emb_b, "... n d-> ... () n d") + emb_sep
            )  # noqa
            pair_feats.append(joint_emb)

        pair_feats = torch.cat(pair_feats, dim=-1) if len(pair_feats) > 0 else None
        return pair_feats


def _print_embedding_info(scalar_embs, pair_embs, scalar_dim, pair_dim):
    tab = "----"
    print("------------------------------------------------------")
    print(f"[INFO] Input feature embeddings\n")

    def print_embs(embs):
        for feat_ty in embs:
            print(f"{tab}{tab}{feat_ty}")
            for emb_ty in embs[feat_ty]:
                print(f"{tab}{tab}{tab}{emb_ty} : {embs[feat_ty][emb_ty].embedding_dim}")
        print()

    print(f"{tab}[INFO] SCALAR (dim = {scalar_dim})\n")
    print_embs(scalar_embs)
    print(f"{tab}[INFO] PAIR (dim = {pair_dim})\n")
    print_embs(pair_embs)
    print("------------------------------------------------------")
