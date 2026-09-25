'''

We want to build an e3nn euclidian neural network
Input and output are is 1x0e + 1x1o + N_ATOMSx1o (heavy-atom slots per nucleotide).

'''
import os
from glob import glob

import numpy as np
import polars as pl
from biopandas.pdb import PandasPdb
from tqdm import tqdm

MIN_RESIDUES, MAX_RESIDUES = 11, 2000
# Measured geometric maximum distance for atoms
CHEM_MAX_RADIUS = 10

# TOKEN Definition
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
DATASET_DIR = "./dataset/BGSU__M__All__All__4_0__pdb_4_55"
SKIP_DIR = "./dataset/skipped"

# Heavy atoms following description in Ribodiff sec 4.1
# https://github.com/chaitjo/geometric-rna-design/blob/main/src/constants.py
ATOM_NAMES = [
    "P",
    "C5'",
    "O5'",
    "C4'",
    "O4'",
    "C3'",
    "O3'",
    "C2'",
    "O2'",
    "C1'",
    "N1",
    "C2",
    "O2",
    "N2",
    "N3",
    "C4",
    "O4",
    "N4",
    "C5",
    "C6",
    "O6",
    "N6",
    "N7",
    "C8",
    "N9",
    "OP1",
    "OP2",
]
N_ATOMS = len( ATOM_NAMES )

ATOM_TO_SLOT = {n: i for i, n in enumerate(ATOM_NAMES)}

BASE_ATOMS = {
    "A": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'",
          "N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"],
    "C": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'",
          "N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"],
    "G": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'",
          "N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"],
    "U": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'",
          "N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6"],
}


def get_pdb_id(path):
    return os.path.basename(path).split("_")[1].lstrip("0")


def process_pdb(path):
    df = pl.from_pandas(PandasPdb().read_pdb(path).df["ATOM"])

    df = df.filter(
        pl.col("atom_name").str.strip_chars().is_in(ATOM_NAMES)
    )
    
    # Remove altLocs
    df = df.unique(
        subset=["chain_id", "residue_number", "insertion", "atom_name"],
        keep="first", maintain_order=True,
    )

    SX = (
        df.filter(pl.col("atom_name") == "C3'")
        .unique(
            subset=["chain_id", "residue_number", "insertion"],
            keep="first", maintain_order=True,
        )
        .select(
            "chain_id",
            "residue_number",
            "insertion",
            "residue_name",
            "x_coord",
            "y_coord",
            "z_coord",
        )
        .with_row_index("residue_index")
    )

    n_modified_nucleotides = SX.filter(~pl.col("residue_name").str.strip_chars().is_in(list(BASE_TO_INT))).height
    R = SX.height
    #NOTE: Some PDB files have modified nucleotides. i.e. PSU, 5MC, We choose to drop them
    if n_modified_nucleotides > 0 or R > MAX_RESIDUES or R < MIN_RESIDUES:
        return None


    df_recentered = (
        df.join(
            SX.select("chain_id", "residue_number", "insertion", "residue_index"),
            on=["chain_id", "residue_number", "insertion"],
            how="inner",
        )
        .with_columns(
            pl.col("atom_name")
            .replace_strict({n: i for i, n in enumerate(ATOM_NAMES)})
            .alias("atom_idx")
        )
        .select("residue_index", "atom_idx", "x_coord", "y_coord", "z_coord")
    )

    residue_names = np.full(MAX_RESIDUES, "", dtype="<U8")
    residue_names[:R] = [str(name).strip() for name in SX["residue_name"]]

    S_row = np.full(MAX_RESIDUES, -1, dtype=np.int8)
    S_row[:R] = [BASE_TO_INT.get(name.strip(), -1) for name in SX["residue_name"]]

    X_row = np.full((MAX_RESIDUES, 3), np.nan, dtype=np.float32)
    X_row[:R] = SX.select("x_coord", "y_coord", "z_coord").to_numpy()

    V_row = np.full((MAX_RESIDUES, N_ATOMS, 3), np.nan, dtype=np.float32)
    V_row[
        df_recentered["residue_index"].to_numpy(),
        df_recentered['atom_idx'].to_numpy(),
    ] = df_recentered.select("x_coord", "y_coord", "z_coord").to_numpy()
    V_row -= X_row[:, None, :]

    # If an atom is in a geometrically and chemically imposible location set to nan
    V_row[np.linalg.norm(V_row, axis=-1) > CHEM_MAX_RADIUS] = np.nan

    return (S_row, X_row, V_row, residue_names)


def build_dataset():
    files = sorted(glob(os.path.join(DATASET_DIR, "*.pdb")))
    n = len(files)
    print(f"{n} pdb files found")

    S = np.full((n, MAX_RESIDUES), -1, dtype=np.int8)
    X = np.full((n, MAX_RESIDUES, 3), np.nan, dtype=np.float32)
    V = np.full((n, MAX_RESIDUES, N_ATOMS, 3), np.nan, dtype=np.float32)
    residue_names = np.full((n, MAX_RESIDUES), "", dtype="<U8")

    os.makedirs(SKIP_DIR, exist_ok=True)
    kept = []
    ids_full = []
    for i, path in enumerate(tqdm(files, unit="file")):
        sample = None
        pdb_id = get_pdb_id(path)

        sample = process_pdb(path)
        ids_full.append(pdb_id)

        if sample is not None:
            S[i], X[i], V[i], residue_names[i] = sample
            kept.append(i)


    print(f"kept {len(kept)}, skipped {n - len(kept)}")
    S, X, V, residue_names = S[kept], X[kept], V[kept], residue_names[kept]
    ids = [ids_full[i] for i in kept]

    # Centering and scaling
    X_centered = np.nanmean(X, axis=1, keepdims=True)
    X -= X_centered

    x_scale = np.nanstd(X)
    X /= x_scale

    v_scale = np.nanstd(V)
    V /= v_scale

    np.savez_compressed(
        "./dataset/dataset.npz",
        S=S,
        X=X,
        V=V,
        ids=np.array(ids, dtype="U9"),
        files=np.array([os.path.basename(p) for p in np.asarray(files)[kept]], dtype="U40"),
        residue_names=residue_names,
        x_scale=x_scale,
        v_scale=v_scale,
    )

    return S, X, V, x_scale, v_scale


if __name__ == "__main__":
    S, X, V, x_scale, v_scale = build_dataset()
    print("S:", S.shape, S.dtype)
    print("X:", X.shape, X.dtype)
    print("V:", V.shape, V.dtype)
    print(f'X Scale {x_scale}, V Scale {v_scale}')

