"""RF2NA PDB output -> our (L, 27, 3) tensor convention."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from src.constants import RNA_ATOMS, RNA_NUCLEOTIDES

SLOT_OF_NAME = {n: i for i, n in enumerate(RNA_ATOMS)}
N_SLOTS = len(RNA_ATOMS)


def read_pdb(path):
    """RF2NA PDB -> (tokens (L,), x (L,3), v (L,27,3), mask (L,27))."""
    res, pos = [], {}          # resid -> {atom_name: xyz}
    for line in open(path):
        if not line.startswith("ATOM"):
            continue
        name = line[12:16].strip()
        if name not in SLOT_OF_NAME:
            continue
        key = line[22:27]
        if key not in pos:
            res.append((key, line[17:20].strip()))
            pos[key] = {}
        pos[key][name] = (float(line[30:38]), float(line[38:46]), float(line[46:54]))

    L = len(res)
    tokens = np.array([RNA_NUCLEOTIDES.index(b) if b in RNA_NUCLEOTIDES else 4 for _, b in res], dtype=np.int64)
    x = np.full((L, 3), np.nan, dtype=np.float32)
    v = np.full((L, N_SLOTS, 3), np.nan, dtype=np.float32)
    mask = np.zeros((L, N_SLOTS), dtype=bool)

    for i, (key, _) in enumerate(res):
        atoms = pos[key]
        if "C3'" not in atoms:                       # frame origin absent -> residue unusable
            continue
        x[i] = atoms["C3'"]
        for name, xyz in atoms.items():
            s = SLOT_OF_NAME[name]
            v[i, s] = np.asarray(xyz, dtype=np.float32) - x[i]
            mask[i, s] = True
    return tokens, x, v, mask


def save(path, tokens, x, v, mask, **extra):
    np.savez_compressed(path, tokens=tokens, x=x, v=v, mask=mask, **extra)


def load(path):
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}
