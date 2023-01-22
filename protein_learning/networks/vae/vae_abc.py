"""Encoder/Decoder and VAE Base Classes"""
from abc import abstractmethod
from typing import Tuple, Dict, Any, Optional

import torch
from torch import nn, Tensor

from protein_learning.common.global_constants import get_logger
from protein_learning.common.helpers import default, exists
from protein_learning.common.rigids import Rigids
from protein_learning.networks.common.net_utils import SplitLinear, LearnedOuter

logger = get_logger(__name__)


class EncoderABC(nn.Module):
    """Encoder Base class"""

    def __init__(self):
        super(EncoderABC, self).__init__()

    @abstractmethod
    def encode(
            self,
            node_feats: Tensor,
            pair_feats: Tensor,
            mask: Tensor,
            **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        """Encode a sample
        :param node_feats: scalar features sampled from latent distribution
        (b,n,d_latent)
        :param pair_feats: pair features of shape (b,n,n,d_pair)
        :param mask: mask of shape (b,n)
        :param kwargs: additional key word arguments
        :return: node and pair encodings
        """
        pass


class DecoderABC(nn.Module):
    """Decoder base class"""

    def __init__(self):
        super(DecoderABC, self).__init__()

    @abstractmethod
    def decode(
            self,
            node_feats: Tensor,
            pair_feats: Tensor,
            mask: Tensor,
            **kwargs,
    ) -> Tuple[Tensor, Tensor, Tensor, Rigids, Tensor]:
        """Decode a sample
        :param node_feats: scalar features sampled from latent distribution
        (b,n,d_latent)
        :param pair_feats: pair features of shape (b,n,n,d_pair)
        :param mask: mask of shape (b,n)
        :param kwargs: additional key word arguments
        :return: decoded node and pair features, any extra info
        """
        pass

    @abstractmethod
    def get_coords(
            self,
            node_feats: Tensor,
            rigids: Optional[Rigids] = None,
            CA_posn: int = 1
    ) -> Tensor:
        """Get coordinates from output"""
        pass


class VAE(nn.Module):  # noqa
    """Variational Autoencoder"""

    def __init__(
            self,
            encoder: EncoderABC,
            decoder: DecoderABC,
            node_dim_hidden: int,
            pair_dim_hidden: int,
            latent_dim: int,
            extra_node_dim: int = None,
            extra_pair_dim: int = None,

    ):
        super(VAE, self).__init__()
        self.extra_node_feature_dim = default(extra_node_dim, 0)
        self.extra_pair_feature_dim = default(extra_pair_dim, 0)
        self.latent_dim = latent_dim
        self.encoder, self.decoder = encoder, decoder

        self.latent_pair_pre_norm = nn.LayerNorm(pair_dim_hidden)
        self.latent_node_pre_norm = nn.LayerNorm(node_dim_hidden)
        self.hidden_to_latent = SplitLinear(
            pair_dim_hidden + node_dim_hidden, 3 * self.latent_dim, chunks=3
        )

        self.node_to_pair = LearnedOuter(
            dim_in=self.latent_dim,
            dim_out=2 * self.latent_dim,
        )

        self.node_latent_to_hidden = nn.Linear(
            latent_dim + self.extra_node_feature_dim, node_dim_hidden
        )

        self.pair_latent_to_hidden = nn.Linear(
            2 * latent_dim + self.extra_pair_feature_dim, pair_dim_hidden
        )

    def forward(
            self,
            node_feats: Tensor,
            pair_feats: Tensor,
            mask: Tensor,
            encoder_kwargs: Optional[Dict[str, Any]],
            decoder_kwargs: Optional[Dict[str, Any]],
            sample_kwargs: Optional[Dict[str, Any]],
    ):
        """Run the model"""
        kwargs = (encoder_kwargs, decoder_kwargs, sample_kwargs)
        encoder_kwargs, decoder_kwargs, sample_kwargs = \
            map(lambda x: default(x, {}), kwargs)

        # encode features
        encoded_node, encoded_pair = self.encoder.encode(
            node_feats=node_feats,
            pair_feats=pair_feats,
            mask=mask,
            **encoder_kwargs
        )

        # concat and norm to get hidden feats for latent proj.
        hidden_repr = torch.cat(
            (
                self.latent_node_pre_norm(encoded_node),
                self.latent_pair_pre_norm(torch.mean(encoded_pair, dim=-2))
            ),
            dim=-1
        )

        # project to latent space
        mu, log_var, mu_gate = self.hidden_to_latent(hidden_repr)
        mu = torch.sigmoid(mu_gate) * mu

        # sample latnt space
        node_sample, pair_sample = self.sample(
            mu=mu, log_var=log_var,
            **sample_kwargs
        )

        # decode latent sample
        node_feats, pair_feats, coords, rigids, aux_loss = self.decoder.decode(
            node_feats=self.node_latent_to_hidden(node_sample),
            pair_feats=self.pair_latent_to_hidden(pair_sample),
            mask=mask,
            **decoder_kwargs
        )

        # return decoded sample feats and kld
        return node_feats, pair_feats, coords, rigids, aux_loss, self.get_kld(mu, log_var)

    @staticmethod
    def get_kld(
            mu_node: Tensor,
            log_var_node: Tensor,
    ) -> Tensor:
        """Get kld"""
        return kl_diag_gaussian(mu_node, log_var_node)

    def sample(
            self,
            mu: Tensor,
            log_var: Tensor,
            node_mask: Tensor = None,
            pair_mask: Tensor = None,  # noqa
            node_feats: Tensor = None,
            pair_feats: Tensor = None,
    ) -> Tuple[Tensor, Tensor]:
        """Get a sample from encoder latent distribution
        """

        if exists(node_mask):
            mu = mu.masked_fill(node_mask.unsqueeze((-1)), 0)
            log_var = log_var.masked_fill(node_mask.unsqueeze((-1)), 0)

        node_sample = sample_diag_gaussian(mu, log_var)
        pair_sample = self.node_to_pair(node_sample)

        return add_feats_to_samples(
            node_sample=node_sample,
            pair_sample=pair_sample,
            node_feats=node_feats,
            pair_feats=pair_feats,
        )


def kl_diag_gaussian(mu: Tensor, log_var: Tensor) -> Tensor:
    """KLD of diagonal unit gaussian"""
    diff = torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=-1)
    return torch.mean(-0.5 * diff)


def add_feats_to_samples(
        node_sample: Tensor,
        pair_sample: Tensor,
        node_feats: Optional[Tensor] = None,
        pair_feats: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Add additional node and pair features to sample"""
    if exists(node_feats):
        node_sample = torch.cat((node_sample, node_feats), dim=-1)
    if exists(pair_feats):
        pair_sample = torch.cat((pair_sample, pair_feats), dim=-1)
    return node_sample, pair_sample


def sample_diag_gaussian(mu: Tensor, log_var: Tensor):
    """Sample from a multivariate gaussian
    with diagonal covariance matrix"""
    std = torch.exp(0.5 * log_var)
    return mu + torch.randn_like(mu).detach() * std
