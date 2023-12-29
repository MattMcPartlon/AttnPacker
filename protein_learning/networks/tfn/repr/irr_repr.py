import os
from math import pi
from pathlib import Path

import numpy as np
import torch
from torch import sin, cos, atan2, acos

from attnpacker.networks.common.helpers.torch_utils import cast_torch_tensor
from attnpacker.networks.common.utils import default
from attnpacker.networks.tfn.repr.fiber import to_order
from attnpacker.networks.tfn.repr.spherical_harmonics import get_spherical_harmonics

DATA_PATH = path = Path(os.path.dirname(__file__)) / 'data'

try:
    path = DATA_PATH / 'J_dense.pt'
    Jd = torch.load(str(path))
except: # noqa
    path = DATA_PATH / 'J_dense.npy'
    Jd_np = np.load(str(path), allow_pickle=True)
    Jd = list(map(torch.from_numpy, Jd_np))


def wigner_d_matrix(degree, alpha, beta, gamma, dtype=None, device=None):
    """Create wigner D matrices for batch of ZYZ Euler anglers for degree l."""
    J = Jd[degree].type(dtype).to(device)
    order = to_order(degree)
    x_a = z_rot_mat(alpha, degree)
    x_b = z_rot_mat(beta, degree)
    x_c = z_rot_mat(gamma, degree)
    res = x_a @ J @ x_b @ J @ x_c
    return res.view(order, order)


def z_rot_mat(angle, l):
    device, dtype = angle.device, angle.dtype
    order = to_order(l)
    m = angle.new_zeros((order, order))
    inds = torch.arange(0, order, 1, dtype=torch.long, device=device)
    reversed_inds = torch.arange(2 * l, -1, -1, dtype=torch.long, device=device)
    frequencies = torch.arange(l, -l - 1, -1, dtype=dtype, device=device)[None]

    m[inds, reversed_inds] = sin(frequencies * angle[None])
    m[inds, inds] = cos(frequencies * angle[None])
    return m


def irr_repr(order, alpha, beta, gamma, dtype=None):
    """
    irreducible representation of SO3
    - compatible with compose and spherical_harmonics
    """
    cast_ = cast_torch_tensor(lambda t: t)
    dtype = default(dtype, torch.get_default_dtype())
    alpha, beta, gamma = map(cast_, (alpha, beta, gamma))
    return wigner_d_matrix(order, alpha, beta, gamma, dtype=dtype)


@cast_torch_tensor
def rot_z(gamma):
    '''
    Rotation around Z axis
    '''
    return torch.tensor([
        [cos(gamma), -sin(gamma), 0],
        [sin(gamma), cos(gamma), 0],
        [0, 0, 1]
    ], dtype=gamma.dtype)


@cast_torch_tensor
def rot_y(beta):
    '''
    Rotation around Y axis
    '''
    return torch.tensor([
        [cos(beta), 0, sin(beta)],
        [0, 1, 0],
        [-sin(beta), 0, cos(beta)]
    ], dtype=beta.dtype)


@cast_torch_tensor
def x_to_alpha_beta(x):
    '''
    Convert point (x, y, z) on the sphere into (alpha, beta)
    '''
    x = x / torch.norm(x)
    beta = acos(x[2])
    alpha = atan2(x[1], x[0])
    return (alpha, beta)


def rot(alpha, beta, gamma):
    '''
    ZYZ Euler angles rotation
    '''
    return rot_z(alpha) @ rot_y(beta) @ rot_z(gamma)


def compose(a1, b1, c1, a2, b2, c2):
    """
    (a, b, c) = (a1, b1, c1) composed with (a2, b2, c2)
    """
    comp = rot(a1, b1, c1) @ rot(a2, b2, c2)
    xyz = comp @ torch.tensor([0, 0, 1.])
    a, b = x_to_alpha_beta(xyz)
    rotz = rot(0, -b, -a) @ comp
    c = atan2(rotz[1, 0], rotz[0, 0])
    return a, b, c


def spherical_harmonics(order, alpha, beta, dtype=None):
    return get_spherical_harmonics(order, theta=(pi - beta), phi=alpha)
