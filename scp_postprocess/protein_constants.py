import torch

AA_ALPHABET = "".join(
    ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
)
RES_TYPES = [x for x in AA_ALPHABET]
AA_ALPHABET += "X-"
AA_INDEX_MAP = {aa: i for i, aa in enumerate(AA_ALPHABET)}
N_AMINO_ACID_KEYS = 22
BB_ATOMS = ["N", "CA", "C", "O"]
SC_ATOMS = [
    "CE3",
    "CZ",
    "SD",
    "CD1",
    "CB",
    "NH1",
    "OG1",
    "CE1",
    "OE1",
    "CZ2",
    "OH",
    "CG",
    "CZ3",
    "NE",
    "CH2",
    "OD1",
    "NH2",
    "ND2",
    "OG",
    "CG2",
    "OE2",
    "CD2",
    "ND1",
    "NE2",
    "NZ",
    "CD",
    "CE2",
    "CE",
    "OD2",
    "SG",
    "NE1",
    "CG1",
    "OXT",
]
SC_ATOM_POSNS = {a: i for i, a in enumerate(SC_ATOMS)}

AA3LetterCode = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "ASX",
    "CYS",
    "GLU",
    "GLN",
    "GLX",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "UNK",
    "GAP",
]
AA1LetterCode = [
    "A",
    "R",
    "N",
    "D",
    "B",
    "C",
    "E",
    "Q",
    "Z",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
    "X",
    "-",
]

NAT_AA_SET = set(AA1LetterCode[:-1])
VALID_AA_3_LETTER = set(AA3LetterCode)
VALID_AA_1_LETTER = set(AA1LetterCode)

ALL_ATOMS = BB_ATOMS + SC_ATOMS

ALL_ATOM_POSNS = {a: i for i, a in enumerate(ALL_ATOMS)}

THREE_TO_ONE = {three: one for three, one in zip(AA3LetterCode, AA1LetterCode)}
ONE_TO_THREE = {one: three for three, one in THREE_TO_ONE.items()}
N_BB_ATOMS = len(BB_ATOMS)
N_SEC_STRUCT_KEYS = 9
SS_KEY_MAP = {"S": 1, "H": 2, "T": 3, "I": 4, "E": 5, "G": 6, "L": 7, "B": 8, "-": 0}


def update_letters(x, is_three=True):
    mapping = THREE_TO_ONE if is_three else ONE_TO_THREE
    x.update({mapping[res]: value for res, value in x.items()})
    return x


AA_TO_INDEX = {
    "ALA": 0,
    "ARG": 1,
    "ASN": 2,
    "ASP": 3,
    "CYS": 4,
    "GLN": 5,
    "GLU": 6,
    "GLY": 7,
    "HIS": 8,
    "ILE": 9,
    "LEU": 10,
    "LYS": 11,
    "MET": 12,
    "PHE": 13,
    "PRO": 14,
    "SER": 15,
    "THR": 16,
    "TRP": 17,
    "TYR": 18,
    "VAL": 19,
    "UNK": 20,
}
AA_TO_INDEX.update({THREE_TO_ONE[aa]: v for aa, v in AA_TO_INDEX.items()})
INDEX_TO_AA_ONE = {AA_TO_INDEX[a]: a for a in AA_TO_INDEX.keys() if len(a) == 1}
INDEX_TO_AA_THREE = {AA_TO_INDEX[a]: a for a in AA_TO_INDEX.keys() if len(a) == 3}

AA_TO_DISTAL = {
    "ALA": "CB",
    "ARG": "CG",
    "ASN": "CG",
    "ASP": "CG",
    "CYS": "SG",
    "GLN": "CG",
    "GLU": "CG",
    "HIS": "CG1",
    "ILE": "CG",
    "LEU": "CG",
    "LYS": "CD",
    "MET": "CG",
    "PHE": "CG",
    "PRO": "CG",
    "SER": "OG",
    "THR": "OG1",
    "TRP": "CG",
    "TYR": "CG",
    "VAL": "CG2",
    "GLY": "CA",
}
AA_TO_DISTAL = update_letters(AA_TO_DISTAL)


AA_TO_FUNCTIONAL = {
    "ALA": "CB",
    "ARG": "NH2",
    "ASN": "ND2",
    "ASP": "OD2",
    "CYS": "SG",
    "GLN": "NE2",
    "GLU": "OE2",
    "HIS": "NE2",
    "ILE": "CD1",
    "LEU": "CD2",
    "LYS": "NZ",
    "MET": "CE",
    "PHE": "CZ",
    "PRO": "CG",
    "SER": "OG",
    "THR": "OG1",
    "TRP": "CH2",
    "TYR": "CZ",
    "VAL": "CG2",
    "GLY": "CA",
}
AA_TO_FUNCTIONAL = update_letters(AA_TO_FUNCTIONAL)

DISTAL_ATOM_MASK_TENSOR = []
FUNCTIONAL_ATOM_MASK_TENSOR = []
for r in RES_TYPES:
    fa_mask, da_mask = torch.zeros(37), torch.zeros(37)
    fa_mask[ALL_ATOM_POSNS[AA_TO_FUNCTIONAL[r]]] = 1  # functional atom index
    da_mask[ALL_ATOM_POSNS[AA_TO_DISTAL[r]]] = 1  # distal atom index
    DISTAL_ATOM_MASK_TENSOR.append(da_mask)
    FUNCTIONAL_ATOM_MASK_TENSOR.append(fa_mask)
DISTAL_ATOM_MASK_TENSOR = torch.stack(DISTAL_ATOM_MASK_TENSOR, dim=0).bool()
FUNCTIONAL_ATOM_MASK_TENSOR = torch.stack(FUNCTIONAL_ATOM_MASK_TENSOR, dim=0).bool()


AA_TO_SC_ATOMS = {
    "MET": ["CB", "CE", "CG", "SD"],
    "ILE": ["CB", "CD1", "CG1", "CG2"],
    "LEU": ["CB", "CD1", "CD2", "CG"],
    "VAL": ["CB", "CG1", "CG2"],
    "THR": ["CB", "CG2", "OG1"],
    "ALA": ["CB"],
    "ARG": ["CB", "CD", "CG", "CZ", "NE", "NH1", "NH2"],
    "SER": ["CB", "OG"],
    "LYS": ["CB", "CD", "CE", "CG", "NZ"],
    "HIS": ["CB", "CD2", "CE1", "CG", "ND1", "NE2"],
    "GLU": ["CB", "CD", "CG", "OE1", "OE2"],
    "ASP": ["CB", "CG", "OD1", "OD2"],
    "PRO": ["CB", "CD", "CG"],
    "GLN": ["CB", "CD", "CG", "NE2", "OE1"],
    "TYR": ["CB", "CD1", "CD2", "CE1", "CE2", "CG", "CZ", "OH"],
    "TRP": ["CB", "CD1", "CD2", "CE2", "CE3", "CG", "CH2", "CZ2", "CZ3", "NE1"],
    "CYS": ["CB", "SG"],
    "ASN": ["CB", "CG", "ND2", "OD1"],
    "PHE": ["CB", "CD1", "CD2", "CE1", "CE2", "CG", "CZ"],
    "GLY": [],
}
AA_TO_SC_ATOMS = update_letters(AA_TO_SC_ATOMS)

HBOND_DONORS = "OG OG1 NE2 ND1 ND2 NZ NE NH1 NH2 NE1 OH N".split()
HBOND_ACCEPTORS = "ND1 NE2 OE1 OE2 OD1 OD2 OH OG OG1 O".split()

HBOND_DONOR_TENSOR = torch.zeros(37)
HBOND_ACCEPTOR_TENSOR = torch.zeros(37)
for atom in HBOND_DONORS:
    HBOND_DONOR_TENSOR[ALL_ATOM_POSNS[atom]] = 1
for atom in HBOND_ACCEPTORS:
    HBOND_ACCEPTOR_TENSOR[ALL_ATOM_POSNS[atom]] = 1


# HBOND_DONORS = [("SER","OG"),("THR","OG1"),("GLN","NE2"),("HIS","ND1"),("HIS","NE2"),("LYS","NZ"),("ARG","NE"),("ARG","NH1"),("ARG","NH2"),("TRP","NE1"),("TYR","OH")]
# HBOND_ACCEPTORS = [("SER","OG"),("THR","OG1"),("GLU","OE1"),("GLU","OE2"),("ASP","OD1"),("ASP","OD2"),("GLN","OE1"),("HIS","ND1"),("HIS","NE2"),("TYR","OH"),("ASN","OD2")]


to_posns = lambda chi: {THREE_TO_ONE[res]: [ALL_ATOM_POSNS[atom] for atom in chi[res]] for res in chi}


def to_chi_posns(chi):
    return {THREE_TO_ONE[res]: [ALL_ATOM_POSNS[atom] for atom in chi[res]] for res in chi}


CHI1 = {
    "PRO": ["N", "CA", "CB", "CG"],
    "THR": ["N", "CA", "CB", "OG1"],
    "VAL": ["N", "CA", "CB", "CG1"],
    "LYS": ["N", "CA", "CB", "CG"],
    "LEU": ["N", "CA", "CB", "CG"],
    "CYS": ["N", "CA", "CB", "SG"],
    "HIS": ["N", "CA", "CB", "CG"],
    "MET": ["N", "CA", "CB", "CG"],
    "ARG": ["N", "CA", "CB", "CG"],
    "SER": ["N", "CA", "CB", "OG"],
    "TYR": ["N", "CA", "CB", "CG"],
    "PHE": ["N", "CA", "CB", "CG"],
    "TRP": ["N", "CA", "CB", "CG"],
    "GLN": ["N", "CA", "CB", "CG"],
    "ASN": ["N", "CA", "CB", "CG"],
    "ASP": ["N", "CA", "CB", "CG"],
    "ILE": ["N", "CA", "CB", "CG1"],
    "GLU": ["N", "CA", "CB", "CG"],
}
chi1_atom_posns = to_chi_posns(CHI1)

CHI2 = {
    "ARG": ["CA", "CB", "CG", "CD"],
    "ASN": ["CA", "CB", "CG", "OD1"],
    "ASP": ["CA", "CB", "CG", "OD1"],
    "GLN": ["CA", "CB", "CG", "CD"],
    "GLU": ["CA", "CB", "CG", "CD"],
    "HIS": ["CA", "CB", "CG", "ND1"],
    "ILE": ["CA", "CB", "CG1", "CD"],
    "LEU": ["CA", "CB", "CG", "CD1"],
    "LYS": ["CA", "CB", "CG", "CD"],
    "MET": ["CA", "CB", "CG", "SD"],
    "PHE": ["CA", "CB", "CG", "CD1"],
    "PRO": ["CA", "CB", "CG", "CD"],
    "TRP": ["CA", "CB", "CG", "CD1"],
    "TYR": ["CA", "CB", "CG", "CD1"],
}
chi2_atom_posns = to_chi_posns(CHI2)

CHI3 = {
    "ARG": ["CB", "CG", "CD", "NE"],
    "GLN": ["CB", "CG", "CD", "OE1"],
    "GLU": ["CB", "CG", "CD", "OE1"],
    "LYS": ["CB", "CG", "CD", "CE"],
    "MET": ["CB", "CG", "SD", "CE"],
}
chi3_atom_posns = to_chi_posns(CHI3)

CHI4 = {"ARG": ["CG", "CD", "NE", "CZ"], "LYS": ["CG", "CD", "CE", "NZ"]}
chi4_atom_posns = to_chi_posns(CHI4)

CHI5 = {"ARG": ["CD", "NE", "CZ", "NH1"]}
chi5_atom_posns = to_chi_posns(CHI5)


# Format: The list for each AA type contains chi1, chi2, chi3, chi4 in
# this order (or a relevant subset from chi1 onwards). ALA and GLY don't have
# chi angles so their chi angle lists are empty.
RES_TY_TO_CHI_GROUP_ATOMS = {
    "ALA": [],
    # Chi5 in arginine is always 0 +- 5 degrees, so ignore it.
    "ARG": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "NE"],
        ["CG", "CD", "NE", "CZ"],
    ],
    "ASN": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "OD1"]],
    "ASP": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "OD1"]],
    "CYS": [["N", "CA", "CB", "SG"]],
    "GLN": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "OE1"],
    ],
    "GLU": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "OE1"],
    ],
    "GLY": [],
    "HIS": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "ND1"]],
    "ILE": [["N", "CA", "CB", "CG1"], ["CA", "CB", "CG1", "CD1"]],
    "LEU": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "LYS": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "CE"],
        ["CG", "CD", "CE", "NZ"],
    ],
    "MET": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "SD"],
        ["CB", "CG", "SD", "CE"],
    ],
    "PHE": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "PRO": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"]],
    "SER": [["N", "CA", "CB", "OG"]],
    "THR": [["N", "CA", "CB", "OG1"]],
    "TRP": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "TYR": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "VAL": [["N", "CA", "CB", "CG1"]],
}
RES_TY_TO_CHI_GROUP_ATOMS = update_letters(RES_TY_TO_CHI_GROUP_ATOMS)

RES_TY_TO_ALL_CHI_MASK_TENSOR = torch.zeros(len(RES_TYPES), 37)
for i, aa in enumerate(RES_TYPES):
    for group in RES_TY_TO_CHI_GROUP_ATOMS[aa]:
        for atom in group:
            RES_TY_TO_ALL_CHI_MASK_TENSOR[i, ALL_ATOM_POSNS[atom]] = 1


CHI_PI_PERIODIC = {
    "ALA": [0.0, 0.0, 0.0, 0.0],
    "ARG": [0.0, 0.0, 0.0, 0.0],
    "ASN": [0.0, 0.0, 0.0, 0.0],
    "ASP": [0.0, 1.0, 0.0, 0.0],
    "CYS": [0.0, 0.0, 0.0, 0.0],
    "GLN": [0.0, 0.0, 0.0, 0.0],
    "GLU": [0.0, 0.0, 1.0, 0.0],
    "GLY": [0.0, 0.0, 0.0, 0.0],
    "HIS": [0.0, 0.0, 0.0, 0.0],
    "ILE": [0.0, 0.0, 0.0, 0.0],
    "LEU": [0.0, 0.0, 0.0, 0.0],
    "LYS": [0.0, 0.0, 0.0, 0.0],
    "MET": [0.0, 0.0, 0.0, 0.0],
    "PHE": [0.0, 1.0, 0.0, 0.0],
    "PRO": [0.0, 0.0, 0.0, 0.0],
    "SER": [0.0, 0.0, 0.0, 0.0],
    "THR": [0.0, 0.0, 0.0, 0.0],
    "TRP": [0.0, 0.0, 0.0, 0.0],
    "TYR": [0.0, 1.0, 0.0, 0.0],
    "VAL": [0.0, 0.0, 0.0, 0.0],
    "UNK": [0.0, 0.0, 0.0, 0.0],
}
CHI_PI_PERIODIC = update_letters(CHI_PI_PERIODIC)
CHI_PI_PERIODIC_LIST = [CHI_PI_PERIODIC[r] for r in RES_TYPES]

CHI_ANGLES_MASK = {
    "ALA": [0.0, 0.0, 0.0, 0.0],  # ALA
    "ARG": [1.0, 1.0, 1.0, 1.0],  # ARG
    "ASN": [1.0, 1.0, 0.0, 0.0],  # ASN
    "ASP": [1.0, 1.0, 0.0, 0.0],  # ASP
    "CYS": [1.0, 0.0, 0.0, 0.0],  # CYS
    "GLN": [1.0, 1.0, 1.0, 0.0],  # GLN
    "GLU": [1.0, 1.0, 1.0, 0.0],  # GLU
    "GLY": [0.0, 0.0, 0.0, 0.0],  # GLY
    "HIS": [1.0, 1.0, 0.0, 0.0],  # HIS
    "ILE": [1.0, 1.0, 0.0, 0.0],  # ILE
    "LEU": [1.0, 1.0, 0.0, 0.0],  # LEU
    "LYS": [1.0, 1.0, 1.0, 1.0],  # LYS
    "MET": [1.0, 1.0, 1.0, 0.0],  # MET
    "PHE": [1.0, 1.0, 0.0, 0.0],  # PHE
    "PRO": [1.0, 1.0, 0.0, 0.0],  # PRO
    "SER": [1.0, 0.0, 0.0, 0.0],  # SER
    "THR": [1.0, 0.0, 0.0, 0.0],  # THR
    "TRP": [1.0, 1.0, 0.0, 0.0],  # TRP
    "TYR": [1.0, 1.0, 0.0, 0.0],  # TYR
    "VAL": [1.0, 0.0, 0.0, 0.0],  # VAL
}
CHI_ANGLES_MASK = update_letters(CHI_ANGLES_MASK)
CHI_ANGLES_MASK_LIST = [CHI_ANGLES_MASK[r] for r in RES_TYPES]
# Naming swaps for ambiguous atom names.
# Due to symmetries in the amino acids the naming of atoms is ambiguous in
# 4 of the 20 amino acids.
# Check LEU, ARG, and VAL
SYMM_SC_RESIDUES = "ASP GLU PHE TYR ARG LEU VAL".split()
RESIDUE_ATOM_RENAMING_SWAPS = {
    "ASP": [["OD1", "OD2"], ["OD1", "OD1"]],
    "GLU": [["OE1", "OE2"], ["OE1", "OE1"]],
    "PHE": [["CD1", "CD2"], ["CE1", "CE2"]],
    "TYR": [["CD1", "CD2"], ["CE1", "CE2"]],
    "ARG": [["NH1", "NH2"], ["NH1", "NH1"]],
    "LEU": [["CD1", "CD2"], ["CD1", "CD1"]],
    "VAL": [["CG1", "CG2"], ["CG1", "CG1"]],
}
for res in RES_TYPES:
    if res not in RESIDUE_ATOM_RENAMING_SWAPS:
        # identity swap
        RESIDUE_ATOM_RENAMING_SWAPS[ONE_TO_THREE[res]] = [["CA", "CA"], ["CA", "CA"]]
SYMM_SC_RES_TYPE_SET = set(SYMM_SC_RESIDUES)
SYMM_SC_RES_ATOMS = [RESIDUE_ATOM_RENAMING_SWAPS[res] for res in SYMM_SC_RESIDUES]

RES_TO_LEFT_SYMM_SC_ATOM_MASK = []
RES_TO_RIGHT_SYMM_SC_ATOM_MASK = []

for res in RES_TYPES:
    res3 = ONE_TO_THREE[res]
    left_atoms, right_atoms = torch.zeros(37, 2), torch.zeros(37, 2)
    for i, [a_left, a_right] in enumerate(RESIDUE_ATOM_RENAMING_SWAPS[res3]):
        left_atoms[ALL_ATOM_POSNS[a_left], i] = 1
        right_atoms[ALL_ATOM_POSNS[a_right], i] = 1
    RES_TO_LEFT_SYMM_SC_ATOM_MASK.append(left_atoms)
    RES_TO_RIGHT_SYMM_SC_ATOM_MASK.append(right_atoms)

# mask_left[seq,i], mask_right[seq,i] are atoms that should be swapped
# shapes here are (N_RES, 37, 2)
RES_TO_LEFT_SYMM_SC_ATOM_MASK = torch.stack(RES_TO_LEFT_SYMM_SC_ATOM_MASK, dim=0).bool()
RES_TO_RIGHT_SYMM_SC_ATOM_MASK = torch.stack(RES_TO_RIGHT_SYMM_SC_ATOM_MASK, dim=0).bool()


class DSSPKeys:
    SS = 2
    AA = 1
    REL_ASA = 3
    PHI = 4
    PSI = 5
    SS_key_map = {"S": 1, "H": 2, "T": 3, "I": 4, "E": 5, "G": 6, "L": 7, "B": 8, "-": 0}


# FROM https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2810841/
BOND_LENS = {("N", "C"): 1.33, ("CA", "CA"): 3.8, ("CA", "C"): 1.51, ("N", "CA"): 1.46}
BOND_LENS.update({(k[1], k[0]): v for k, v in BOND_LENS.items()})
BOND_LEN_SIGMA = 3
BOND_LEN_TOL = {("N", "C"): 0.01, ("CA", "CA"): 0.016, ("CA", "C"): 0.01, ("N", "CA"): 0.01}
BOND_LEN_TOL = {k: v * BOND_LEN_SIGMA for k, v in BOND_LEN_TOL.items()}
BOND_LEN_TOL.update({(k[1], k[0]): v for k, v in BOND_LEN_TOL.items()})
BOND_LEN_OFFSET = {("N", "C"): 1, ("CA", "CA"): 1, ("CA", "C"): 0, ("N", "CA"): 0}
BOND_LEN_OFFSET.update({(k[1], k[0]): v for k, v in BOND_LEN_OFFSET.items()})

BOND_ANGLE_SIGMA = 4
BOND_ANGLES = {("N", "CA", "C"): 111, ("CA", "C", "N"): 117.2, ("C", "N", "CA"): 121.7}
BOND_ANGLE_TOL = {("N", "CA", "C"): 2.8, ("CA", "C", "N"): 2.0, ("C", "N", "CA"): 1.8}
BOND_ANGLE_TOL = {k: v * BOND_ANGLE_SIGMA for k, v in BOND_ANGLE_TOL.items()}
BOND_ANGLE_OFFSET = {("N", "CA", "C"): (0, 0, 0), ("CA", "C", "N"): (0, 0, 1), ("C", "N", "CA"): (0, 1, 1)}

VDW_RADIUS = dict(C=1.7, O=1.52, N=1.55, S=1.8)
ALL_ATOM_VDW_RADII = {}
for atom in ALL_ATOMS:
    for ty in VDW_RADIUS:
        if ty in atom:
            ALL_ATOM_VDW_RADII[atom] = VDW_RADIUS[ty]

INTRA_RES_BONDS = dict(
    A={("CA", "N", 1.458), ("C", "O", 1.231), ("C", "CA", 1.523), ("CA", "CB", 1.522)},
    R={
        ("CA", "N", 1.458),
        ("CD", "NE", 1.454),
        ("CA", "CB", 1.522),
        ("CZ", "NH1", 1.315),
        ("CB", "CG", 1.52),
        ("CD", "CG", 1.485),
        ("C", "CA", 1.523),
        ("CZ", "NE", 1.347),
        ("CZ", "NH2", 1.322),
        ("C", "O", 1.231),
    },
    N={
        ("CA", "N", 1.458),
        ("CB", "CG", 1.504),
        ("CG", "OD1", 1.236),
        ("C", "CA", 1.523),
        ("CG", "ND2", 1.309),
        ("C", "O", 1.231),
        ("CA", "CB", 1.518),
    },
    D={
        ("CA", "N", 1.458),
        ("CG", "OD1", 1.208),
        ("CB", "CG", 1.523),
        ("C", "CA", 1.523),
        ("CA", "CB", 1.531),
        ("C", "O", 1.231),
        ("CG", "OD2", 1.208),
    },
    C={("CA", "N", 1.458), ("CB", "SG", 1.809), ("C", "CA", 1.523), ("CA", "CB", 1.529), ("C", "O", 1.231)},
    Q={
        ("CA", "N", 1.458),
        ("CB", "CG", 1.519),
        ("CD", "NE2", 1.328),
        ("CD", "CG", 1.517),
        ("CD", "OE1", 1.234),
        ("C", "CA", 1.523),
        ("CA", "CB", 1.531),
        ("C", "O", 1.231),
    },
    E={
        ("CA", "N", 1.458),
        ("CD", "OE2", 1.209),
        ("CD", "CG", 1.503),
        ("CD", "OE1", 1.208),
        ("CB", "CG", 1.522),
        ("C", "CA", 1.523),
        ("CA", "CB", 1.53),
        ("C", "O", 1.231),
    },
    G={("CA", "N", 1.458), ("C", "O", 1.231), ("C", "CA", 1.523)},
    H={
        ("CA", "N", 1.458),
        ("CG", "ND1", 1.379),
        ("CD2", "NE2", 1.373),
        ("CE1", "NE2", 1.32),
        ("CE1", "ND1", 1.322),
        ("C", "CA", 1.523),
        ("CD2", "CG", 1.354),
        ("CA", "CB", 1.532),
        ("C", "O", 1.231),
        ("CB", "CG", 1.497),
    },
    I={
        ("CA", "N", 1.458),
        ("CA", "CB", 1.54),
        ("CB", "CG1", 1.531),
        ("CB", "CG2", 1.521),
        ("C", "CA", 1.523),
        ("CD1", "CG1", 1.512),
        ("C", "O", 1.231),
    },
    L={
        ("CD1", "CG", 1.523),
        ("CA", "N", 1.458),
        ("CB", "CG", 1.534),
        ("C", "O", 1.231),
        ("CD2", "CG", 1.521),
        ("C", "CA", 1.523),
        ("CA", "CB", 1.534),
    },
    K={
        ("CA", "N", 1.458),
        ("CD", "CG", 1.521),
        ("CD", "CE", 1.522),
        ("CB", "CG", 1.523),
        ("C", "CA", 1.523),
        ("CA", "CB", 1.53),
        ("C", "O", 1.231),
        ("CE", "NZ", 1.488),
    },
    M={
        ("CG", "SD", 1.804),
        ("CA", "N", 1.458),
        ("CB", "CG", 1.522),
        ("CE", "SD", 1.79),
        ("C", "CA", 1.523),
        ("CA", "CB", 1.527),
        ("C", "O", 1.231),
    },
    F={
        ("CE1", "CZ", 1.379),
        ("CA", "N", 1.458),
        ("CD1", "CG", 1.387),
        ("CD2", "CE2", 1.381),
        ("CB", "CG", 1.502),
        ("CD2", "CG", 1.387),
        ("CE2", "CZ", 1.38),
        ("C", "CA", 1.523),
        ("CD1", "CE1", 1.382),
        ("CA", "CB", 1.53),
        ("C", "O", 1.231),
    },
    P={
        ("CA", "N", 1.458),
        ("CB", "CG", 1.491),
        ("C", "CA", 1.523),
        ("CA", "CB", 1.532),
        ("C", "O", 1.231),
        ("CD", "CG", 1.506),
        ("CD", "N", 1.444),
    },
    S={("CA", "N", 1.458), ("C", "CA", 1.523), ("CB", "OG", 1.401), ("C", "O", 1.231), ("CA", "CB", 1.516)},
    T={
        ("CA", "N", 1.458),
        ("CA", "CB", 1.54),
        ("CB", "OG1", 1.434),
        ("CB", "CG2", 1.521),
        ("C", "CA", 1.523),
        ("C", "O", 1.231),
    },
    W={
        ("CA", "N", 1.458),
        ("CD2", "CE2", 1.397),
        ("CE3", "CZ3", 1.39),
        ("CB", "CG", 1.499),
        ("CD2", "CG", 1.448),
        ("CD2", "CE3", 1.4),
        ("CH2", "CZ3", 1.372),
        ("CD1", "NE1", 1.373),
        ("C", "CA", 1.523),
        ("CE2", "CZ2", 1.386),
        ("CA", "CB", 1.53),
        ("C", "O", 1.231),
        ("CE2", "NE1", 1.372),
        ("CD1", "CG", 1.363),
        ("CH2", "CZ2", 1.395),
    },
    Y={
        ("CB", "CG", 1.513),
        ("CA", "N", 1.458),
        ("CD1", "CG", 1.387),
        ("CD2", "CE2", 1.381),
        ("CD2", "CG", 1.387),
        ("CE2", "CZ", 1.38),
        ("CE1", "CZ", 1.39),
        ("C", "CA", 1.523),
        ("CD1", "CE1", 1.382),
        ("CA", "CB", 1.53),
        ("CZ", "OH", 1.376),
        ("C", "O", 1.231),
    },
    V={
        ("CA", "N", 1.458),
        ("CA", "CB", 1.54),
        ("C", "O", 1.2),
        ("CB", "CG2", 1.521),
        ("C", "CA", 1.523),
        ("CB", "CG1", 1.521),
    },
)

# Virtual bonds include CA_{i}-CA+{i+1}, and
VIRTUAL_BONDS = dict()

# Taken from: https://github.com/aqlaboratory/openfold/blob/main/openfold/np/residue_constants.py
# Atoms positions relative to the 8 rigid groups, defined by the pre-omega, phi,
# psi and chi angles:
# 0: 'backbone group',
# 1: 'pre-omega-group', (empty)
# 2: 'phi-group', (currently empty, because it defines only hydrogens)
# 3: 'psi-group',
# 4,5,6,7: 'chi1,2,3,4-group'
# The atom positions are relative to the axis-end-atom of the corresponding
# rotation axis. The x-axis is in direction of the rotation axis, and the y-axis
# is defined such that the dihedral-angle-definiting atom (the last entry in
# chi_angles_atoms above) is in the xy-plane (with a positive y-coordinate).
# format: [atomname, group_idx, rel_position]
RIGID_GROUP_ATOM_POSITIONS = {
    "ALA": [
        ["N", 0, (-0.525, 1.363, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, -0.000, -0.000)],
        ["CB", 0, (-0.529, -0.774, -1.205)],
        ["O", 3, (0.627, 1.062, 0.000)],
    ],
    "ARG": [
        ["N", 0, (-0.524, 1.362, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, -0.000, -0.000)],
        ["CB", 0, (-0.524, -0.778, -1.209)],
        ["O", 3, (0.626, 1.062, 0.000)],
        ["CG", 4, (0.616, 1.390, -0.000)],
        ["CD", 5, (0.564, 1.414, 0.000)],
        ["NE", 6, (0.539, 1.357, -0.000)],
        ["NH1", 7, (0.206, 2.301, 0.000)],
        ["NH2", 7, (2.078, 0.978, -0.000)],
        ["CZ", 7, (0.758, 1.093, -0.000)],
    ],
    "ASN": [
        ["N", 0, (-0.536, 1.357, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, -0.000, -0.000)],
        ["CB", 0, (-0.531, -0.787, -1.200)],
        ["O", 3, (0.625, 1.062, 0.000)],
        ["CG", 4, (0.584, 1.399, 0.000)],
        ["ND2", 5, (0.593, -1.188, 0.001)],
        ["OD1", 5, (0.633, 1.059, 0.000)],
    ],
    "ASP": [
        ["N", 0, (-0.525, 1.362, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.527, 0.000, -0.000)],
        ["CB", 0, (-0.526, -0.778, -1.208)],
        ["O", 3, (0.626, 1.062, -0.000)],
        ["CG", 4, (0.593, 1.398, -0.000)],
        ["OD1", 5, (0.610, 1.091, 0.000)],
        ["OD2", 5, (0.592, -1.101, -0.003)],
    ],
    "CYS": [
        ["N", 0, (-0.522, 1.362, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.524, 0.000, 0.000)],
        ["CB", 0, (-0.519, -0.773, -1.212)],
        ["O", 3, (0.625, 1.062, -0.000)],
        ["SG", 4, (0.728, 1.653, 0.000)],
    ],
    "GLN": [
        ["N", 0, (-0.526, 1.361, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, 0.000, 0.000)],
        ["CB", 0, (-0.525, -0.779, -1.207)],
        ["O", 3, (0.626, 1.062, -0.000)],
        ["CG", 4, (0.615, 1.393, 0.000)],
        ["CD", 5, (0.587, 1.399, -0.000)],
        ["NE2", 6, (0.593, -1.189, -0.001)],
        ["OE1", 6, (0.634, 1.060, 0.000)],
    ],
    "GLU": [
        ["N", 0, (-0.528, 1.361, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, -0.000, -0.000)],
        ["CB", 0, (-0.526, -0.781, -1.207)],
        ["O", 3, (0.626, 1.062, 0.000)],
        ["CG", 4, (0.615, 1.392, 0.000)],
        ["CD", 5, (0.600, 1.397, 0.000)],
        ["OE1", 6, (0.607, 1.095, -0.000)],
        ["OE2", 6, (0.589, -1.104, -0.001)],
    ],
    "GLY": [
        ["N", 0, (-0.572, 1.337, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.517, -0.000, -0.000)],
        ["O", 3, (0.626, 1.062, -0.000)],
    ],
    "HIS": [
        ["N", 0, (-0.527, 1.360, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, 0.000, 0.000)],
        ["CB", 0, (-0.525, -0.778, -1.208)],
        ["O", 3, (0.625, 1.063, 0.000)],
        ["CG", 4, (0.600, 1.370, -0.000)],
        ["CD2", 5, (0.889, -1.021, 0.003)],
        ["ND1", 5, (0.744, 1.160, -0.000)],
        ["CE1", 5, (2.030, 0.851, 0.002)],
        ["NE2", 5, (2.145, -0.466, 0.004)],
    ],
    "ILE": [
        ["N", 0, (-0.493, 1.373, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.527, -0.000, -0.000)],
        ["CB", 0, (-0.536, -0.793, -1.213)],
        ["O", 3, (0.627, 1.062, -0.000)],
        ["CG1", 4, (0.534, 1.437, -0.000)],
        ["CG2", 4, (0.540, -0.785, -1.199)],
        ["CD1", 5, (0.619, 1.391, 0.000)],
    ],
    "LEU": [
        ["N", 0, (-0.520, 1.363, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, -0.000, -0.000)],
        ["CB", 0, (-0.522, -0.773, -1.214)],
        ["O", 3, (0.625, 1.063, -0.000)],
        ["CG", 4, (0.678, 1.371, 0.000)],
        ["CD1", 5, (0.530, 1.430, -0.000)],
        ["CD2", 5, (0.535, -0.774, 1.200)],
    ],
    "LYS": [
        ["N", 0, (-0.526, 1.362, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, 0.000, 0.000)],
        ["CB", 0, (-0.524, -0.778, -1.208)],
        ["O", 3, (0.626, 1.062, -0.000)],
        ["CG", 4, (0.619, 1.390, 0.000)],
        ["CD", 5, (0.559, 1.417, 0.000)],
        ["CE", 6, (0.560, 1.416, 0.000)],
        ["NZ", 7, (0.554, 1.387, 0.000)],
    ],
    "MET": [
        ["N", 0, (-0.521, 1.364, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, 0.000, 0.000)],
        ["CB", 0, (-0.523, -0.776, -1.210)],
        ["O", 3, (0.625, 1.062, -0.000)],
        ["CG", 4, (0.613, 1.391, -0.000)],
        ["SD", 5, (0.703, 1.695, 0.000)],
        ["CE", 6, (0.320, 1.786, -0.000)],
    ],
    "PHE": [
        ["N", 0, (-0.518, 1.363, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.524, 0.000, -0.000)],
        ["CB", 0, (-0.525, -0.776, -1.212)],
        ["O", 3, (0.626, 1.062, -0.000)],
        ["CG", 4, (0.607, 1.377, 0.000)],
        ["CD1", 5, (0.709, 1.195, -0.000)],
        ["CD2", 5, (0.706, -1.196, 0.000)],
        ["CE1", 5, (2.102, 1.198, -0.000)],
        ["CE2", 5, (2.098, -1.201, -0.000)],
        ["CZ", 5, (2.794, -0.003, -0.001)],
    ],
    "PRO": [
        ["N", 0, (-0.566, 1.351, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.527, -0.000, 0.000)],
        ["CB", 0, (-0.546, -0.611, -1.293)],
        ["O", 3, (0.621, 1.066, 0.000)],
        ["CG", 4, (0.382, 1.445, 0.0)],
        # ['CD', 5, (0.427, 1.440, 0.0)],
        ["CD", 5, (0.477, 1.424, 0.0)],  # manually made angle 2 degrees larger
    ],
    "SER": [
        ["N", 0, (-0.529, 1.360, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, -0.000, -0.000)],
        ["CB", 0, (-0.518, -0.777, -1.211)],
        ["O", 3, (0.626, 1.062, -0.000)],
        ["OG", 4, (0.503, 1.325, 0.000)],
    ],
    "THR": [
        ["N", 0, (-0.517, 1.364, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, 0.000, -0.000)],
        ["CB", 0, (-0.516, -0.793, -1.215)],
        ["O", 3, (0.626, 1.062, 0.000)],
        ["CG2", 4, (0.550, -0.718, -1.228)],
        ["OG1", 4, (0.472, 1.353, 0.000)],
    ],
    "TRP": [
        ["N", 0, (-0.521, 1.363, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, -0.000, 0.000)],
        ["CB", 0, (-0.523, -0.776, -1.212)],
        ["O", 3, (0.627, 1.062, 0.000)],
        ["CG", 4, (0.609, 1.370, -0.000)],
        ["CD1", 5, (0.824, 1.091, 0.000)],
        ["CD2", 5, (0.854, -1.148, -0.005)],
        ["CE2", 5, (2.186, -0.678, -0.007)],
        ["CE3", 5, (0.622, -2.530, -0.007)],
        ["NE1", 5, (2.140, 0.690, -0.004)],
        ["CH2", 5, (3.028, -2.890, -0.013)],
        ["CZ2", 5, (3.283, -1.543, -0.011)],
        ["CZ3", 5, (1.715, -3.389, -0.011)],
    ],
    "TYR": [
        ["N", 0, (-0.522, 1.362, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.524, -0.000, -0.000)],
        ["CB", 0, (-0.522, -0.776, -1.213)],
        ["O", 3, (0.627, 1.062, -0.000)],
        ["CG", 4, (0.607, 1.382, -0.000)],
        ["CD1", 5, (0.716, 1.195, -0.000)],
        ["CD2", 5, (0.713, -1.194, -0.001)],
        ["CE1", 5, (2.107, 1.200, -0.002)],
        ["CE2", 5, (2.104, -1.201, -0.003)],
        ["OH", 5, (4.168, -0.002, -0.005)],
        ["CZ", 5, (2.791, -0.001, -0.003)],
    ],
    "VAL": [
        ["N", 0, (-0.494, 1.373, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.527, -0.000, -0.000)],
        ["CB", 0, (-0.533, -0.795, -1.213)],
        ["O", 3, (0.627, 1.062, -0.000)],
        ["CG1", 4, (0.540, 1.429, -0.000)],
        ["CG2", 4, (0.533, -0.776, 1.203)],
    ],
}
# order the rigid group positions by atom index
tmp = RIGID_GROUP_ATOM_POSITIONS.copy()
for res, pos_n_groups in RIGID_GROUP_ATOM_POSITIONS.items():
    tmp[res] = sorted([v for v in pos_n_groups], key=lambda x: ALL_ATOM_POSNS[x[0]])
RIGID_GROUP_ATOM_POSITIONS = tmp

# Van der Waals radii [Angstrom] of the atoms (from Wikipedia)
VAN_DER_WAALS_RADII = {
    "C": 1.7,
    "N": 1.55,
    "O": 1.52,
    "S": 1.8,
}
