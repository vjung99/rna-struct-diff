import os

PROJECT_PATH = os.environ.get("PROJECT_PATH")

DATA_PATH = os.environ.get("DATA_PATH")

X3DNA_PATH = os.environ.get("X3DNA")

ETERNAFOLD_PATH = os.environ.get("ETERNAFOLD")


# process_data.py fills absent slots with this; rotated per-conformer, so compare norms, not components
FILL_VALUE = 1e-5

# fill norm is FILL_VALUE*sqrt(3) (1.7e-5); nearest real atom 0.16
FILL_TOL = 1e-3

DISTANCE_EPS = 0.001


# https://github.com/chaitjo/geometric-rna-design/blob/main/src/constants.py
RNA_ATOMS = [
    'P', "C5'", "O5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'",
    'N1',
    'C2',
    'O2', 'N2',
    'N3',
    'C4', 'O4', 'N4',
    'C5',
    'C6',
    'O6', 'N6',
    'N7',
    'C8',
    'N9',
    'OP1', 'OP2',
]

ATOM_NAMES = RNA_ATOMS

NUM_ATOM_SLOTS = len(RNA_ATOMS)

# Slot index of the C3' frame origin that all other slots are expressed relative to
C3P_SLOT = RNA_ATOMS.index("C3'")

ATOM_INDEX = {name: i for i, name in enumerate(RNA_ATOMS)}

RNA_NUCLEOTIDES = ['A', 'C', 'G', 'U', '_']

PURINES = ["A", "G"]

PYRIMIDINES = ["C", "U"]

LETTER_TO_NUM = dict(zip(
    RNA_NUCLEOTIDES,
    list(range(len(RNA_NUCLEOTIDES)))
))

NUM_TO_LETTER = {v: k for k, v in LETTER_TO_NUM.items()}

DOTBRACKET_TO_NUM = {
    '.': 0,
    '(': 1,
    ')': 2
}


SEPARATION_TOKEN = 'S'
MASK_TOKEN = 'M'
PAD_TOKEN = 'P'

NUCLEOTIDES = 'ACGUN'
NUM_BASES = len(NUCLEOTIDES)

TOKENS = NUCLEOTIDES + SEPARATION_TOKEN + MASK_TOKEN + PAD_TOKEN
VOCAB_SIZE = len(TOKENS)

BASE_TO_INT = {t: i for i, t in enumerate(TOKENS)}
INT_TO_BASE = {v: k for k, v in BASE_TO_INT.items()}

PAD_IDX = BASE_TO_INT[PAD_TOKEN]
MASK_IDX = BASE_TO_INT[MASK_TOKEN]
SEPARATION_IDX = BASE_TO_INT[SEPARATION_TOKEN]

# Alias so the model code can spell the atom dimension the way the irreps do
N_ATOMS = NUM_ATOM_SLOTS

# Measured geometric maximum distance for atoms
CHEM_MAX_RADIUS = 10

RMSD_THRESHOLD = 2.0
TM_THRESHOLD = 0.45
GDT_THRESHOLD = 0.50
