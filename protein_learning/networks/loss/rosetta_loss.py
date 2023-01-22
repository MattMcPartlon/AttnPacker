import numpy as np
import torch

DEFAULT_DTYPE = torch.float32
torch.set_default_dtype(DEFAULT_DTYPE)
from pyrosetta import *
from pyrosetta.rosetta import *
from pyrosetta.rosetta.numeric import xyzVector_double_t as xyz_vec

from protein_learning.common.protein_constants import N_BB_ATOMS
from protein_learning.common.helpers import default
from protein_learning.protein_utils.dihedral.orientation_utils import signed_dihedral_4, get_bb_dihedral

DSSP = protocols.moves.DsspMover()  # noqa
init(
    '-out:levels core.scoring.dssp:error'
    ' core.scoring.saxs.FormFactorManager:error'
    ' protocols.DsspMover:error'
    ' basic.io.database:error'
    ' core.scoring.saxs.SAXSEnergy:error')

fa_score_types = "fa_atr fa_rep fa_sol lk_ball_wtd fa_elec hbond_bb_sc " \
                 "hbond_sc p_aa_pp rama_prepro omega fa_dun".split(" ")

# uncomment any of the terms below to activate
# note that cart bonded terms take a long time to compute
cen_score_types = [
    # ("env",1),
    # "cbeta",
    "cenpack",
    "cen_env_smooth",
    "cen_pair_smooth",
    "cbeta_smooth",
    # ("saxs_cen_score",0.1),
    # "cbeta_smooth",
    "rama2b",
    # "ch_bond_bb_bb",
    # "cen_pair_motifs",
    # "hbond_sr_bb",
    # "hbond_lr_bb",
    # "dslfc_cb_dst",
    # "dslfc_cb_dih",
    # "dslfc_ang",
    # "DFIRE",
    # "cart_bonded",
    "geom_sol",
    # "dunbrack_constraint",
    # "cart_bonded_length",
    # "cart_bonded_angle",
    # "cart_bonded_torsion",
    # ("neigh_vect", 1),
    # ("neigh_count", 1),
    # "pair",
    # "vdw",
    # "hs_pair"
]

# change to fa_score_types if using full atom mode (not backbone)
BACKBONE = True
score_terms = cen_score_types if BACKBONE else fa_score_types
score_types = [(s, 1.0) if not isinstance(s, tuple) else s for s in score_terms]


def init_fa_sfxn():
    fa_sfxn = ScoreFunction()  # pyrosetta.create_score_function("score1")
    for ty, wt in fa_score_types:
        st = getattr(core.scoring.ScoreType, ty)
        fa_sfxn.set_weight(st, wt)
    return fa_sfxn


def init_cen_sfxn():
    cen_sfxn = ScoreFunction()  # pyrosetta.create_score_function("score1")
    for ty, wt in score_types:
        st = getattr(core.scoring.ScoreType, ty)
        cen_sfxn.set_weight(st, wt)
    return cen_sfxn


def _to_numpy(x, dtype=np.float64) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return x


def get_pose_from_seq_n_coords(seq, coords=None):
    pose = pose_from_sequence(seq)
    return set_pose_coords(pose, coords)


def set_pose_coords(pose, coords):
    seq = pose.sequence()
    if coords is None:
        return pose
    _coords = map(lambda x: _to_numpy(x), coords)
    N, CA, CB, C, O = _coords  # noqa
    for i in range(len(coords) // N_BB_ATOMS):
        pose.residue(i + 1).set_xyz('N', xyz_vec(*N[i]))
        pose.residue(i + 1).set_xyz('CA', xyz_vec(*CA[i]))
        if seq[i] != 'G':
            pose.residue(i + 1).set_xyz('CB', xyz_vec(*CB[i]))
        pose.residue(i + 1).set_xyz('C', xyz_vec(*C[i]))
        pose.residue(i + 1).set_xyz('O', xyz_vec(*O[i]))
    # pose will not update internal coords from explicit
    # coordinate change - only happens for bb torsion change
    pose.set_phi(1, pose.phi(1))
    return pose


def get_dssp(seq, coords, pose=None):
    pose = pose if pose is not None else get_pose_from_seq_n_coords(seq, coords)
    phi, psi, omega = get_bb_dihedral(N=coords[0], CA=coords[1], C=coords[3])
    DSSP.apply(pose)
    ss = pose.secstruct()
    return phi, psi, omega, ss


def safe_cast(arr):
    nans = torch.any(torch.isnan(arr)).to(arr.device)
    infs = torch.any(torch.isinf(arr)).to(arr.device)
    arr[torch.logical_or(nans, infs)] = 0
    return arr


def rosetta_to_3Dtensor(arr, device='cpu', dtype=DEFAULT_DTYPE):
    # rosetta score array is a numpy array of tuples.
    # we must first convert to a numpy float array before casting to tensor
    np_arr = np.array([[list(x) for x in y] for y in arr], dtype=np.float32)
    return safe_cast(torch.tensor(np_arr, device=device, dtype=dtype))


def rosetta_to_2Dtensor(arr, device='cpu', dtype=DEFAULT_DTYPE):
    np_arr = np.array([list(x) for x in arr], dtype=np.float32)
    return safe_cast(torch.tensor(np_arr, device=device, dtype=dtype))


def _get_residue_ens(pose, device='cpu', dtype=torch.float32):
    residue_total_ens = pyrosetta.bindings.energies.residue_total_energies_array(pose.energies())
    residue_pair_ens = pyrosetta.bindings.energies.residue_pair_energies_array(pose.energies())
    return rosetta_to_2Dtensor(residue_total_ens, device=device, dtype=dtype), \
           rosetta_to_3Dtensor(residue_pair_ens, device=device, dtype=dtype)  # noqa


class RosettaScore:
    def __init__(self, backbone=True):
        self.sfxn = init_cen_sfxn() if backbone else init_fa_sfxn()
        self.use_cen = backbone
        self.switch = SwitchResidueTypeSetMover("centroid")  # noqa
        self._pose = pose_from_sequence('AAA')
        tmp_res, tmp_pair = self.get_energies('AAAA', None)
        self.num_res_score_terms = tmp_res.shape[-1]
        self.num_res_pair_score_terms = tmp_pair.shape[-1]

    def _get_pose(self, seq, coords, pdb=None):
        if pdb is not None:
            pose = pose_from_pdb(pdb)
        else:
            if seq != self._pose.sequence():
                pose = get_pose_from_seq_n_coords(seq, coords)
            else:
                pose = set_pose_coords(self._pose, coords)
        if self.use_cen:
            self.switch.apply(pose)  # change to centroid
        self._pose = pose
        return pose

    def get_energies(self, seq, coords, device=None, dtype=None, pdb=None):
        _device = 'cpu' if coords is None else coords[0].device
        device = default(device, _device)
        pose = self._get_pose(seq, coords.squeeze() if torch.is_tensor(coords) else coords, pdb=pdb)
        self.sfxn(pose)  # populate pose energies object with scores
        res, pair = _get_residue_ens(pose, device=device, dtype=dtype)
        return res, pair

    def get_energies_and_dssp(self, seq, coords, device=None, dtype=torch.float32, pdb=None):
        device = default(device, coords[0].device)
        dtype = default(dtype, coords[0].dtype)
        pose = self._get_pose(seq, coords, pdb=pdb)
        self.sfxn(pose)  # populate pose energies object with scores
        ens_single, ens_pair = _get_residue_ens(pose, device=device, dtype=dtype)
        phi, psi, omega, ss = get_dssp(seq, coords, pose=pose)
        return ens_single, ens_pair, phi, psi, omega, ss

    def get_dssp(self, seq, coords, pose=None):
        pose = pose if pose is not None else self._get_pose(seq, coords)
        phi, psi, omega = get_bb_dihedral(N=coords[0], CA=coords[1], C=coords[3])
        DSSP.apply(pose)
        ss = pose.secstruct()
        return phi, psi, omega, ss
