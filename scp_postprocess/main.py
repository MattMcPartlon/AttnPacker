from scp_postprocess.project_sidechains import (
    project_onto_rotamers,
    compute_sc_rmsd,
    compute_clashes
)
from protein_learning.common.data.data_types.protein import Protein
import torch
import protein_learning.common.protein_constants as pc
from protein_learning.common.data.datasets.utils import set_canonical_coords_n_masks