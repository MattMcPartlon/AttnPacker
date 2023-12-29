from abc import ABC, abstractmethod, abstractproperty
from typing import Union, Tuple


class NetConfig:
    """Network configuration"""

    def __init__(self):
        pass

    @property
    @abstractmethod
    def scalar_dims(self) -> Tuple[int, int, int]:
        """in, hidden, and out dimensions of scalar features"""
        pass

    @property
    @abstractmethod
    def pair_dims(self) -> Tuple[int, int, int]:
        """in, hidden, and out dimensions of pair features"""
        pass

    @property
    def coord_dims(self) -> Tuple[int, int, int]:
        """in, hidden, and out dimensions of coordinate features"""
        return -1, -1, -1

    def _select_dims(self, dims: Tuple[int, ...], scalar: bool = False, pair: bool = False, coord=False
                     ) -> Union[Tuple[int, ...], int]:
        out = []
        if scalar:
            out.append(dims[0])
        if pair:
            out.append(dims[1])
        if coord:
            out.append(dims[2])
        return tuple(out) if len(out) > 1 else out[0]

    def dim_in(self, scalar: bool = False, pair: bool = False, coord=False) -> Union[Tuple[int, ...], int]:
        """Input dimension of scalar, pair and coord features"""
        dims = (self.scalar_dims[0], self.pair_dims[0], self.coord_dims[0])
        return self._select_dims(dims, scalar, pair, coord)

    def dim_hidden(self, scalar: bool = False, pair: bool = False, coord=False) -> Union[Tuple[int, ...], int]:
        """hidden dimension of scalar, pair and coord features"""
        dims = (self.scalar_dims[1], self.pair_dims[1], self.coord_dims[1])
        return self._select_dims(dims, scalar, pair, coord)

    def dim_out(self, scalar: bool = False, pair: bool = False, coord=False) -> Union[Tuple[int, ...], int]:
        """output dimension of scalar, pair and coord features"""
        dims = (self.scalar_dims[2], self.pair_dims[2], self.coord_dims[2])
        return self._select_dims(dims, scalar, pair, coord)
