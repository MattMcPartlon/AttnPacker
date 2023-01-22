from collections import namedtuple
from itertools import product
from typing import Dict, Tuple, Union

import torch
from torch import nn

from protein_learning.networks.common.utils import default

FiberEl = namedtuple('FiberEl', ['degrees', 'dim'])


def uniq(arr):
    return list({el: True for el in arr}.keys())


class Fiber(nn.Module):
    """Describes a mapping from feature types to dimensions

    Concrete Example:
        An equivariant linear layer mapping x in R^(5 x 3) to x in R^(10 x 3)
        is achieved by specifying
        fiber_in = Fiber({1: 5}), and fiber_out = Fiber({1: 10})

    Note:
        The name is taken from the equivariant NN literature, originally
        appearing in https://arxiv.org/pdf/1807.02547.pdf,
        "3D Steerable CNNs: Learning Rotationally Equivariant Features in
         Volumetric Data"

    """

    DEGREE, DIM = 0, 1

    def __init__(
            self,
            structure
    ):
        super().__init__()
        if isinstance(structure, dict):
            structure = list(structure.items())
        self.structure = structure

    @property
    def dims(self):
        """Dimension of the mapping

        Example:
            Fiber({3:10}).dims = (10,)
        """
        return uniq(map(lambda t: t[1], self.structure))

    @property
    def degrees(self):
        """Degree of points being mapped

        Example:
            Fiber({1:5, 3:10}).degrees = (1,3)
        """
        return list(map(lambda t: t[0], self.structure))

    @property
    def items(self):
        return map(lambda t: (t[self.DEGREE], t[self.DIM]), self.structure)

    @staticmethod
    def create(num_degrees, dim):
        dim_tuple = dim if isinstance(dim, tuple) else ((dim,) * num_degrees)
        return Fiber([FiberEl(degree, dim) for degree, dim in zip(range(num_degrees), dim_tuple)])

    def scale(self, mult: int):
        return Fiber([(deg, dim * mult) for deg, dim in self.items])

    def __getitem__(self, degree):
        """Returns the dimension corresponding to degree
        """
        return dict(self.structure)[degree]

    def __iter__(self):
        return iter(self.structure)

    def __mul__(self, other: 'Fiber'):
        return product(self.structure, other.structure)

    def __and__(self, other: 'Fiber'):
        out = []
        for degree, dim in self:
            if degree in other.degrees:
                dim_out = other[degree]
                out.append((degree, dim, dim_out))
        return out

    def __or__(self, other):
        return list(zip(self, other))

    def has_ty(self, ty):
        return int(ty) in self.degrees

    @property
    def n_degrees(self):
        return len(self.degrees)

    @staticmethod
    def as_fiber(elt, n_degrees):
        Fiber.create(n_degrees, elt) if isinstance(elt, int) else elt
        assert isinstance(elt, Fiber)
        return elt

    @staticmethod
    def to_order(deg):
        return 2 * deg + 1

    def __str__(self):
        return str(self.structure)


def default_degree_map(fiber_from: Fiber, fiber_to: Fiber) -> torch.Tensor:
    """Returns a degree mapping between all pairs of input and output degrees
    """
    d1, d2 = max(fiber_from.degrees), max(fiber_to.degrees)
    return torch.ones((d1, d2)).bool()


def filter_fiber_product(fiber_from: Fiber, fiber_to: Fiber, degree_map=None):
    """Filters mappings between input and output degrees

    In TFN's and other equivariant convolutional architectures, degree i features
    can be mapped to degree j features for i!=j. In this case, the desired mapping
    can be specified with a boolean matrix, and the fiber product will be filtered
    to reflect the desired mapping.

    :param fiber_from: The fiber being mapped from
    :param fiber_to: The fiber being mapped to
    :param degree_map: boolean matrix indicating which degrees to map to and from.
    i.e. map degree i features to degree j features if and only if degree_map[i,j]

    :return: fiber product of fiber_in and fiber_out such that ((i, k1),(j,k2))
    is listed iff degree_map[i,j]
    """

    dmat = default(degree_map, default_degree_map(fiber_from, fiber_to))
    key = fiber_to.DEGREE
    return filter(lambda x, _dmat=dmat: _dmat[x[0][key], x[1][key]], fiber_from * fiber_to)


def default_tymap(fiber_in: Fiber, fiber_out: Fiber):
    return torch.ones((fiber_in.n_degrees, fiber_out.n_degrees)).bool()


# main class
def cast_fiber(dims: Union[int, Tuple, Fiber, Dict], degrees: int = 1) -> Union[Fiber, None]:
    if dims is None:
        return None
    if isinstance(dims, int):
        return Fiber([(i, dims) for i in range(degrees)])
    if isinstance(dims, tuple) or isinstance(dims, list):
        return Fiber([(i, d) for i, d in enumerate(dims)])
    return Fiber(list(dims.items()))  # noqa


def to_order(degree):
    return 2 * degree + 1


def chunk_fiber(feats: Dict[str, torch.Tensor], n_chunks: int, dim: int = -2):
    out = [{} for _ in range(n_chunks)]
    for degree in feats:
        chunks = feats[degree].chunk(n_chunks, dim=dim)
        for i in range(len(chunks)):
            out[i][degree] = chunks[i]
    return out
