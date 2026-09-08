'''

We want to build an e3nn euclidian neural network
Input and output are is 1x0e + 1x1o + 24x1o

'''
import os
import shutil
from glob import glob

import numpy as np
import polars as pl
from biopandas.pdb import PandasPdb
from tqdm import tqdm

MIN_RESIDUES, MAX_RESIDUES, MAX_ATOMS_PER_RESIDUE = 40, 150, 24
BASE_TO_INT = {"A": 0, "C": 1, "G": 2, "U": 3}
INT_TO_BASE = {v: k for k, v in BASE_TO_INT.items()}
DATASET_DIR = "./dataset/BGSU__M__All__All__4_0__pdb_4_55"
SKIP_DIR = "./dataset/oversize"


def pdb_id(path):
    return os.path.basename(path).split("_")[1].lstrip("0")


def process_pdb(path):
    df = pl.from_pandas(PandasPdb().read_pdb(path).df["ATOM"])
    modified_nucleotide = False

    SX = (
        df.filter(pl.col("atom_name") == "C3'")
        .unique(subset=["chain_id", "residue_number"], keep="first", maintain_order=True)
        .select(
            "chain_id",
            "residue_number",
            "residue_name",
            "x_coord",
            "y_coord",
            "z_coord",
        )
        .with_row_index("residue_index")
    )

    #NOTE: Some PDB files have modified nucleotides. i.e. PSU, 5MC
    if (
        SX.filter(~pl.col("residue_name").str.strip_chars().is_in(list(BASE_TO_INT))).height
        > 0
    ):
        modified_nucleotide = True

    R = SX.height
    if R == 0 or R > MAX_RESIDUES or R < MIN_RESIDUES:
        return None

    df_recentered = (
        df.join(
            SX.select("chain_id", "residue_number", "residue_index"),
            on=["chain_id", "residue_number"],
            how="inner",
        )
        .with_columns(
            (pl.col("atom_number") - pl.col("atom_number").min().over("residue_index")).alias(
                "atom_slot"
            )
        )
        .filter(pl.col("atom_slot") < MAX_ATOMS_PER_RESIDUE)
        .select("residue_index", "atom_slot", "x_coord", "y_coord", "z_coord")
    )

    residue_names = np.full(MAX_RESIDUES, "", dtype="<U8")
    residue_names[:R] = [str(name).strip() for name in SX["residue_name"]]

    S_row = np.full(MAX_RESIDUES, -1, dtype=np.int8)
    S_row[:R] = [BASE_TO_INT.get(name.strip(), -1) for name in SX["residue_name"]]

    X_row = np.full((MAX_RESIDUES, 3), np.nan, dtype=np.float32)
    X_row[:R] = SX.select("x_coord", "y_coord", "z_coord").to_numpy()

    V_row = np.full((MAX_RESIDUES, MAX_ATOMS_PER_RESIDUE, 3), np.nan, dtype=np.float32)
    V_row[
        df_recentered["residue_index"].to_numpy(),
        df_recentered["atom_slot"].to_numpy(),
    ] = df_recentered.select("x_coord", "y_coord", "z_coord").to_numpy()
    V_row -= X_row[:, None, :]

    return (S_row, X_row, V_row, residue_names), modified_nucleotide


def build_dataset():
    files = sorted(glob(os.path.join(DATASET_DIR, "*.pdb")))
    n = len(files)
    print(f"{n} pdb files found")

    S = np.full((n, MAX_RESIDUES), -1, dtype=np.int8)
    X = np.full((n, MAX_RESIDUES, 3), np.nan, dtype=np.float32)
    V = np.full((n, MAX_RESIDUES, MAX_ATOMS_PER_RESIDUE, 3), np.nan, dtype=np.float32)
    residue_names = np.full((n, MAX_RESIDUES), "", dtype="<U8")

    os.makedirs(SKIP_DIR, exist_ok=True)
    kept = []
    ids_full = []
    for i, path in enumerate(tqdm(files, unit="file")):
        modified_nucleotide = False
        sample = None

        res = process_pdb(path)
        if res is not None:
            sample, modified_nucleotide = res[0], res[1]

        ids_full.append(pdb_id(path))

        if sample is None:
            shutil.move(path, os.path.join(SKIP_DIR, os.path.basename(path)))
        else:
            S[i], X[i], V[i], residue_names[i] = sample

        if not modified_nucleotide:
            kept.append(i)

    # Centering and scaling
    X_centered = np.nanmean(X, axis=1, keepdims=True)   
    X = X - X_centered

    x_scale = np.nanstd(X)                        
    X = X / x_scale

    v_scale = np.nanstd(V)                        
    V = V / v_scale

    print(f"kept {len(kept)}, skipped {n - len(kept)}")
    np.savez(
        "./dataset/dataset_full.npz",
        S=S,
        X=X,
        V=V,
        ids=np.array(ids_full, dtype="U9"),
        residue_names=residue_names,
        x_scale=x_scale,
        v_scale=v_scale,
    )

    S, X, V, residue_names = S[kept], X[kept], V[kept], residue_names[kept]
    ids = [ids_full[i] for i in kept]
    np.savez(
        "./dataset/dataset.npz",
        S=S,
        X=X,
        V=V,
        ids=np.array(ids, dtype="U9"),
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
