"""Helpers for data/processed.pt -- one entry per unique RNA sequence, each entry holding
every observed structure of that sequence.

Coordinates are raw Angstroms in src.constants.RNA_ATOMS slot order."""

import numpy as np
import torch

from src.constants import (
    ATOM_NAMES,
    C3P_SLOT,
    FILL_TOL,
    N_ATOMS,
    NUM_ATOM_SLOTS,
    PAD_IDX,
    CHEM_MAX_RADIUS,
    RNA_NUCLEOTIDES
)


def rmsd_matrix(coords):
    """All-pairs superposed RMSD between the rows of ``coords``.

    ``coords`` is ``(C, L, 3)``; returns a ``(C, C)`` matrix. The optimal rotation for
    every pair is obtained in closed form from the 3x3 cross-covariance (Kabsch), so no
    per-pair iteration is needed.
    """
    C, L, _ = coords.shape
    A = np.asarray(coords, dtype=np.float64)
    A = A - A.mean(axis=1, keepdims=True)

    P = A.transpose(0, 2, 1).reshape(C * 3, L)
    G = (P @ P.T).reshape(C, 3, C, 3).transpose(0, 2, 1, 3)   # block [i, j] = A_i^T A_j

    U, s, Vt = np.linalg.svd(G.reshape(-1, 3, 3))
    det = np.sign(np.linalg.det(Vt.transpose(0, 2, 1) @ U.transpose(0, 2, 1)))
    trace = s[:, 0] + s[:, 1] + det * s[:, 2]                  # optimal superposition

    nrm2 = (A ** 2).sum(axis=(1, 2))
    msd = nrm2[:, None] + nrm2[None, :] - 2 * trace.reshape(C, C)
    return np.sqrt(np.maximum(msd, 0.0) / L)


class Sequence:
    """A unique RNA sequence together with all of its observed 3D structures.

    ``len(seq)`` is the number of structures (what ``__getitem__`` and iteration index);
    ``seq.length`` is the number of residues.
    """

    def __init__(self, struct) -> None:
        self.sequence = struct['sequence']
        self.pdb_ids = list(struct['id_list'])
        self.atom_positions = np.stack(struct['coords_list'])   # (C, L, 27, 3)
        self.secondary_structures = list(struct['sec_struct_list'])
        self.sasa = list(struct['sasa_list'])
        self.rfam = list(struct['rfam_list'])
        self.eq_class = list(struct['eq_class_list'])
        self.type = list(struct['type_list'])
        self.rmsds = struct['rmsds_list']
        self.cluster_seqid = struct['cluster_seqid0.8']
        self.cluster_structsim = struct['cluster_structsim0.45']

        self.n_structures = len(self.pdb_ids)
        self.length = len(self.sequence)

        assert self.atom_positions.shape[1] == self.length, "coords/sequence length mismatch"
        assert self.atom_positions.shape[2] == NUM_ATOM_SLOTS, "atom slot count mismatch"

        self.tokens = np.array([RNA_NUCLEOTIDES.index(ch) for ch in self.sequence], dtype=np.int64)
        self.offsets = self.atom_positions - self.atom_positions[:, :, C3P_SLOT, None, :]
        self.atom_mask = (np.linalg.norm(self.atom_positions, axis=-1) > FILL_TOL) & (
            np.linalg.norm(self.offsets, axis=-1) < CHEM_MAX_RADIUS
        )

    def __len__(self):
        return self.n_structures

    def __repr__(self):
        return (f"Sequence(len={self.length}, structures={self.n_structures}, "
                f"cluster_structsim={self.cluster_structsim})")

    def get_tokenized_sequence(self, start_residue_idx=0, end_residue_idx=None):
        """Residue tokens of [start, end); ``None`` end means "to the end"."""
        return self.tokens[start_residue_idx:end_residue_idx].tolist()

    def get_tertiary_structure(self, idx, start_residue_idx=0, end_residue_idx=None):
        """(span, 3) C3' origins and (span, 27, 3) atom offsets of one structure."""
        window = slice(start_residue_idx, end_residue_idx)
        return self.atom_positions[idx, window, C3P_SLOT, :], self.offsets[idx, window]

    def atom_mask_window(self, idx, start_residue_idx=0, end_residue_idx=None):
        """(span, 27, 1) presence mask for the same window."""
        return self.atom_mask[idx, start_residue_idx:end_residue_idx, :, None]

    def get_secondary_structure(self, idx):
        return self.secondary_structures[idx]

    def get_random_sample(self, rng, crop_len=256):
        """(s, x, v, atom_mask), each ``crop_len`` rows, for one random structure."""
        structure_idx = int(rng.integers(0, self.n_structures))
        if self.length > crop_len:
            start_idx = int(rng.integers(0, self.length - crop_len + 1))
            span = crop_len
        else:
            start_idx, span = 0, self.length

        s = np.full(crop_len, PAD_IDX, dtype=np.int64)
        x = np.zeros((crop_len, 3), dtype=np.float32)
        v = np.zeros((crop_len, N_ATOMS * 3), dtype=np.float32)
        atom_mask = np.zeros((crop_len, N_ATOMS, 3), dtype=bool)

        s[:span] = self.get_tokenized_sequence(start_idx, start_idx + span)
        x[:span] = self.atom_positions[structure_idx, start_idx:start_idx + span, C3P_SLOT, :]
        v[:span] = self.offsets[structure_idx, start_idx:start_idx + span].reshape(span, -1)
        atom_mask[:span] = self.atom_mask_window(structure_idx, start_idx, start_idx + span)

        v.reshape(crop_len, N_ATOMS, 3)[~atom_mask[..., 0]] = 0.0

        return s, x, v, atom_mask

    # ---- analysis ----------------------------------------------------------------

    def pairwise_rmsd(self, slot=C3P_SLOT):
        """(C, C) superposed RMSD between this sequence's conformers, on ``slot``."""
        return rmsd_matrix(self.atom_positions[:, :, slot, :])

    def neighbor_counts(self, radii):
        """Neighbour counts on C3' within each radius, across conformers.

        Returns ``{radius: [count, ...]}`` with one entry per (structure, residue);
        each count excludes the residue itself.
        """
        from scipy.spatial import cKDTree

        out = {float(r): [] for r in radii}
        for i in range(self.n_structures):
            points = self.atom_positions[i, :, C3P_SLOT, :]
            tree = cKDTree(points)
            for r in radii:
                out[float(r)].extend(
                    (tree.query_ball_point(points, r, return_length=True) - 1).tolist()
                )
        return out

    @staticmethod
    def load_sequences_from_file(path):
        """Load every unique sequence from a ``processed.pt`` checkpoint."""
        data = torch.load(path, weights_only=False)
        return [Sequence(v) for v in data.values()]


def atom_offset_stats(sequences):
    """Pooled |atom - C3'| statistics per atom slot.

    Returns mean, std, min, max, count.
    """
    total = np.zeros(NUM_ATOM_SLOTS)
    total_sq = np.zeros(NUM_ATOM_SLOTS)
    count = np.zeros(NUM_ATOM_SLOTS)
    lo = np.full(NUM_ATOM_SLOTS, np.inf)
    hi = np.full(NUM_ATOM_SLOTS, -np.inf)

    for seq in sequences:
        dist = np.linalg.norm(seq.offsets, axis=-1).astype(np.float64)
        mask = seq.atom_mask
        dist = np.where(mask, dist, np.nan)
        total += np.nansum(dist, axis=(0, 1))
        total_sq += np.nansum(dist ** 2, axis=(0, 1))
        count += mask.sum(axis=(0, 1))

        lo = np.fmin(lo, np.min(np.where(mask, dist, np.inf), axis=(0, 1)))
        hi = np.fmax(hi, np.max(np.where(mask, dist, -np.inf), axis=(0, 1)))

    mean = total / count
    var = np.maximum(total_sq / count - mean ** 2, 0.0)
    return mean, np.sqrt(var), lo, hi, count


def pooled_c3p(sequences):
    """(N, 3) every residue's C3' position."""
    return np.concatenate(
        [seq.atom_positions[:, :, C3P_SLOT, :].reshape(-1, 3) for seq in sequences], axis=0
    )


def atom_position_stats(sequences):
    """Pooled mean/std of every present atom's x/y/z offset from its C3', in Angstroms.

    Returns mean, std, count.
    """
    n = 0
    total = np.zeros(3)
    total_sq = np.zeros(3)

    for seq in sequences:
        v = seq.offsets[seq.atom_mask].astype(np.float64)
        n += len(v)
        total += v.sum(axis=0)
        total_sq += (v ** 2).sum(axis=0)

    mean = total / n
    return mean, np.sqrt(np.maximum(total_sq / n - mean ** 2, 0.0)), n
