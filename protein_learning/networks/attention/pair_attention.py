"""Functions for performing Pair feature updates via triangle multiplication and attention"""
from typing import Optional

import torch
import torch.utils.checkpoint as checkpoint
from einops import rearrange  # noqa
from torch import nn, einsum, Tensor

from protein_learning.networks.common.net_utils import Residual, exists, default, LearnedOuter, Transition, get_min_val
from protein_learning.networks.common.jit_scripts import Fuser

List = nn.ModuleList  # noqa
tri_outgoing = lambda a, b: torch.einsum("b i k d, b j k d -> b i j d ", a, b)
tri_incoming = lambda a, b: torch.einsum("b k i d, b k j d -> b i j d ", a, b)


class TriangleMul(nn.Module):  # noqa
    """Global Traingle Multiplication"""

    def __init__(
        self,
        dim_in,
        incoming: bool,
        dim_hidden=128,
        residual: Optional[nn.Module] = None,
        checkpoint_outer: bool = False,
    ):
        super().__init__()
        self.pre_norm = nn.LayerNorm(dim_in)
        self.to_feats_n_gates = nn.Linear(dim_in, 4 * dim_hidden)
        self.gate_out_proj = nn.Linear(dim_in, dim_in)
        self.to_out = nn.Sequential(nn.LayerNorm(dim_hidden), nn.Linear(dim_hidden, dim_in))
        self.func = tri_outgoing if not incoming else tri_incoming
        self.residual = default(residual, lambda x, res: x)
        self.checkpoint_outer = checkpoint_outer
        self.sigmoid, self.sigmoid_gate = Fuser().sigmoid, Fuser().sigmoid_gate

    def forward(self, edges, mask) -> Tensor:
        """Perform Forward Pass"""
        normed_edges = self.pre_norm(edges)
        feats, gates = self.to_feats_n_gates(normed_edges).chunk(2, -1)
        to_feats, from_feats = self.sigmoid_gate(feats, gates).chunk(2, -1)
        out_gate = self.sigmoid(self.gate_out_proj(normed_edges))
        if exists(mask):
            mask = ~mask.unsqueeze(-1)
            assert mask.shape[:3] == to_feats.shape[:3]
            to_feats, from_feats, out_gate = map(lambda x: x.masked_fill(mask, 0), (to_feats, from_feats, out_gate))

        args = (out_gate, to_feats, from_feats, edges)
        checkpoint_outer = self.checkpoint_outer and self.training
        return checkpoint.checkpoint(self._to_out, *args) if checkpoint_outer else self._to_out(*args)

    def _to_out(self, out_gate, to_feats, from_feats, edges):
        # to_feats, from_feats = map(lambda x: x.half(), (to_feats, from_feats))
        out = out_gate * self.to_out(self.func(to_feats, from_feats))
        return self.residual(out, edges)


class TriangleAttn(nn.Module):  # noqa
    """Triangle Attention"""

    def __init__(
        self,
        dim_in,
        starting: bool,
        dim_head=32,
        heads=4,
        residual: Optional[nn.Module] = None,
        checkpoint_attn: bool = False,
    ):
        super().__init__()
        self.pre_norm = nn.LayerNorm(dim_in)
        self.to_qkv = nn.Linear(dim_in, dim_head * heads * 3, bias=False)
        self.to_b = nn.Linear(dim_in, heads, bias=False)
        self.to_g = nn.Linear(dim_in, dim_head * heads)
        self.heads = heads
        self.scale = dim_head ** -(1 / 2)
        self.to_out = nn.Linear(dim_head * heads, dim_in)
        self.starting = starting
        self.residual = default(residual, lambda x, res: x)
        self.attn_fn = self.attn_starting if starting else self.attn_ending
        self.checkpoint_attn = checkpoint_attn

    def forward(self, edges: Tensor, mask: Optional[Tensor]) -> Tensor:
        """Triangle Attention at starting/ending node

        :param edges: Edge features of shape (b,n,n,d) where b is the batch dimension,
        and d is the feature dimension
        :param mask: (Optional) Boolean tensor of shape (b,n,n)
        :return: Triangle Attention features
        """

        normed_edges = self.pre_norm(edges)
        q, k, v = self.to_qkv(normed_edges).chunk(3, -1)
        g = self.to_g(normed_edges)
        b = self.to_b(normed_edges)
        q, k, v, b, g = map(lambda x: rearrange(x, "b n m (h d) -> b h n m d", h=self.heads), (q, k, v, b, g))
        b = b.squeeze(-1)
        args = (q, k, b, v, g, mask)
        checkpoint_attn = self.checkpoint_attn and self.training
        output = checkpoint.checkpoint(self.attn_fn, *args) if checkpoint_attn else self.attn_fn(*args)
        out = self.to_out(rearrange(output, "b h i j d -> b i j (h d)"))
        return self.residual(out, edges)

    def attn_starting(self, q, k, b, v, g, mask: Tensor):
        if q.shape[2] > 600 and not self.training:  # slower, but more memory efficient
            return self._attn_starting_loop(q, k, b, v, g, mask)
        return self._attn_starting(q, k, b, v, g, mask)

    def attn_ending(self, q, k, b, v, g, mask: Tensor):
        if q.shape[2] > 600 and not self.training:  # slower, but more memory efficient
            return self._attn_ending_loop(q, k, b, v, g, mask)
        return self._attn_ending(q, k, b, v, g, mask)

    def _attn_starting_loop(self, q, k, b, v, g, mask: Tensor):
        out = []
        for h in range(q.shape[1]):
            sim = torch.einsum("b i j d, b i k d -> b i j k", q[:, h, ...], k[:, h, ...]) * self.scale
            sim = sim + rearrange(b[:, h, ...], "b j k -> b () j k")
            _g = g[:, h, ...]
            if exists(mask):
                attn_mask = ~rearrange(mask, "b i j -> b i j ()")
                sim, _g = map(lambda x: x.masked_fill(attn_mask, get_min_val(sim)), (sim, _g))

            attn = torch.softmax(sim, dim=-1)
            out.append(torch.sigmoid(_g) * einsum("...i j k,... i k d -> ... i j d", attn, v[:, h, ...]))
        return torch.stack(out, dim=1)

    def _attn_ending_loop(self, q, k, b, v, g, mask: Tensor):
        out = []
        for h in range(q.shape[1]):
            sim = torch.einsum("b i j d, b k j d -> b i j k", q[:, h, ...], k[:, h, ...]) * self.scale
            sim = sim + rearrange(b[:, h, ...], "b i k -> b k () i")
            _g = g[:, h, ...]
            if exists(mask):
                attn_mask = ~rearrange(mask, "b i j -> b i j ()")
                sim, _g = map(lambda x: x.masked_fill(attn_mask, get_min_val(sim)), (sim, _g))
            attn = torch.softmax(sim, dim=-1)
            out.append(torch.sigmoid(_g) * einsum("...i j k,... k j d -> ... i j d", attn, v[:, h, ...]))
        return torch.stack(out, dim=1)

    def _attn_starting(self, q, k, b, v, g, mask: Tensor):
        sim = torch.einsum("b h i j d, b h i k d -> b h i j k", q, k) * self.scale
        sim = sim + rearrange(b, "b h j k -> b h () j k")
        if exists(mask):
            attn_mask = ~rearrange(mask, "b i j -> b () i j ()")
            sim, g = map(lambda x: x.masked_fill(attn_mask, get_min_val(sim)), (sim, g))
        attn = torch.softmax(sim, dim=-1)
        return torch.sigmoid(g) * einsum("...i j k,... i k d -> ... i j d", attn, v)

    def _attn_ending(self, q, k, b, v, g, mask: Tensor):
        sim = torch.einsum("b h i j d, b h k j d -> b h i j k", q, k) * self.scale
        sim = sim + rearrange(b, "b h i k -> b h k () i")
        if exists(mask):
            attn_mask = ~rearrange(mask, "b i j -> b () i j ()")
            sim, g = map(lambda x: x.masked_fill(attn_mask, get_min_val(sim)), (sim, g))
        attn = torch.softmax(sim, dim=-1)
        return torch.sigmoid(g) * einsum("...i j k,... k j d -> ... i j d", attn, v)


class PairUpdateBlock(nn.Module):  # noqa
    """Perform Triangle updates for Pair Features"""

    def __init__(
        self,
        pair_dim: int,
        node_dim: Optional[int] = None,
        heads: int = 4,
        dim_head: Optional[int] = None,
        dropout: float = 0,
        tri_mul_dim: Optional[int] = None,
        do_checkpoint: bool = True,
        ff_mult: int = 2,
        do_tri_mul: bool = True,
        do_tri_attn: bool = True,
        do_pair_outer: bool = True,
    ):
        super(PairUpdateBlock, self).__init__()
        Dropout = lambda: nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        TriMul = lambda incoming: List(
            [
                TriangleMul(
                    pair_dim,
                    incoming=incoming,
                    dim_hidden=default(tri_mul_dim, pair_dim),
                )
                if do_tri_mul
                else None,
                Residual() if do_tri_mul else None,
                Dropout() if do_tri_mul else None,
            ]
        )
        TriAttn = lambda starting: List(
            [
                TriangleAttn(
                    dim_in=pair_dim,
                    dim_head=default(dim_head, pair_dim // heads),
                    heads=heads,
                    starting=starting,
                )
                if do_tri_attn
                else None,
                Residual() if do_tri_attn else None,
                Dropout() if do_tri_attn else None,
            ]
        )
        do_pair_outer = do_pair_outer and node_dim > 0
        self.layers = List(
            [
                List(
                    [
                        LearnedOuter(dim_in=node_dim, dim_out=pair_dim) if do_pair_outer else None,
                        Residual() if do_pair_outer else None,
                    ]
                ),
                TriMul(False),
                TriMul(True),
                TriAttn(True),
                TriAttn(False),
            ]
        )
        self.transition = nn.Identity()
        if do_pair_outer or do_tri_mul or do_tri_attn:
            self.transition = Transition(dim_in=pair_dim, mult=ff_mult, residual=Residual())
        self.do_checkpoint = do_checkpoint

    def forward(self, node_feats: Tensor, pair_feats: Tensor, mask: Optional[Tensor]) -> Tensor:
        """Perform Pair Feature Update"""
        do_checkpoint = self.do_checkpoint and self.training
        apply_fn = lambda fn, *args: checkpoint.checkpoint(fn, *args) if do_checkpoint else fn(*args)
        outer, outer_res = self.layers[0]
        if exists(outer):
            pair_feats = outer_res(outer(node_feats), pair_feats)
        for i, (net, residual, dropout) in enumerate(self.layers[1:]):  # noqa
            if not exists(net):
                continue
            out = apply_fn(net.forward, pair_feats, mask)
            pair_feats = residual(dropout(out), pair_feats)
        return apply_fn(self.transition.forward, pair_feats)
