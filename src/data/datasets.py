import random

import numpy as np
from torch.utils.data import Dataset, Sampler

from dataset import PAD_IDX


class RNACoGenerationDataset(Dataset):
    def __init__(self, s, x, v, crop_len = 256, seed=0) -> None:
        self.N, self.L = s.shape
        self.crop_len = crop_len
        self.rng = np.random.default_rng(seed)

        self.s, self.x, self.v = s, np.nan_to_num(x), v.reshape(self.N, self.L, -1)

        self.token_mask = np.where(self.s == -1, 0, 1)
        self.atom_mask = np.where(np.isnan(self.v), 0, 1)
        self.v = np.nan_to_num(self.v)

        self.s = np.where(self.s == -1, PAD_IDX, self.s)

        print(f"Dataset Loaded {self.s.shape=} {self.x.shape=} {self.v.shape=}")

    def __len__(self):
        return len(self.s)

    def __getitem__(self, index):
        len = (self.s[index, :] != PAD_IDX).sum()
        if len > self.crop_len:
            end_idx = self.rng.integers(self.crop_len, len)
            start_idx = end_idx - self.crop_len
        else:
            start_idx, end_idx = 0, self.crop_len

        return (
            self.s[index, start_idx:end_idx],
            self.x[index, start_idx:end_idx, :],
            self.v[index, start_idx:end_idx, :],
            self.token_mask[index, start_idx:end_idx],
            self.atom_mask[index, start_idx:end_idx, :],
        )

# Taken from https://github.com/chaitjo/geometric-rna-design/blob/main/src/data/dataset.py
# Added sequence groups (only one identical sequence can appear per epoch)
class BatchSampler(Sampler):
    """Custom batch sampler for efficient GPU memory utilization with variable-length RNAs.

    This sampler groups RNA structures into batches such that the total number of nucleotides
    (nodes) in each batch does not exceed a specified maximum. This is critical for graph
    neural networks where memory consumption scales with the total number of nodes rather
    than the number of graphs in a batch.

    The sampler handles structures of varying lengths by:
        - Grouping small/medium structures together up to max_nodes_batch total nodes
        - Processing large structures (> max_nodes_batch) individually if they fit within
          max_nodes_sample
        - Filtering out structures exceeding max_nodes_sample

    This approach maximizes GPU utilization while preventing out-of-memory errors and
    ensures efficient training on datasets with heterogeneous sequence lengths.

    Credit: Adapted from https://github.com/jingraham/neurips19-graph-protein-design

    Args:
        node_counts (array-like): Array of nucleotide counts for each RNA structure in
            the dataset. Used to determine batch composition.
        max_nodes_batch (int): Maximum total number of nucleotides allowed in a batch
            when combining multiple structures. Defaults to 3000.
        max_nodes_sample (int): Maximum number of nucleotides for a single structure.
            Structures with more nucleotides than this are excluded from training.
            Defaults to 5000.
        shuffle (bool): Whether to shuffle the order of batches. Set to True for training
            and False for evaluation. Defaults to True.
    """

    def __init__(self, node_counts, max_nodes_batch=3000, max_nodes_sample=5000, shuffle=True, seq_group= None, seed = 0):
        self.groups = None
        if seq_group is not None:
          self.groups = {}
          for i, g in enumerate(np.asarray(seq_group)):
            self.groups.setdefault(int(g), []).append(i)
          self.epoch = 0

        self.seed = seed
        self.node_counts = node_counts
        self.shuffle = shuffle
        self.max_nodes_batch = max_nodes_batch
        self.max_nodes_sample = max_nodes_sample

        self._form_batches()

    def set_epoch(self, epoch):
        if self.groups is not None:
          self.epoch = epoch
          self._form_batches()

    def _form_batches(self):
        """Construct batches by greedily grouping samples to maximize node count per batch.

        This method uses a greedy algorithm to pack RNA structures into batches:
            1. Optionally shuffle the dataset indices
            2. Iterate through structures in order
            3. For each structure, add it to the current batch if it fits within max_nodes_batch
            4. When adding the next structure would exceed the limit, start a new batch
            5. Continue until all structures are assigned to batches

        The resulting batches are stored in self.batches as a list of lists, where each
        inner list contains dataset indices for structures in that batch.
        """
        if self.groups is not None:
          rng = np.random.default_rng(self.seed + self.epoch)

          keys = list(self.groups)
          picks = [self.groups[k][int(rng.integers(len(self.groups[k])))] for k in keys]

          max_nodes = min(self.max_nodes_batch, self.max_nodes_sample)
          self.idx = [i for i in picks if self.node_counts[i] <= max_nodes]
          self.batches_single = [
              [i] for i in picks if max_nodes < self.node_counts[i] <= self.max_nodes_sample
          ]
        else:
          rng = np.random.default_rng(self.seed)

          # no grouping: every row is a candidate, as in the original gRNAde sampler
          max_nodes = min(self.max_nodes_batch, self.max_nodes_sample)
          self.idx = [i for i in range(len(self.node_counts)) if self.node_counts[i] <= max_nodes]
          self.batches_single = [
              [i] for i in range(len(self.node_counts))
              if max_nodes < self.node_counts[i] <= self.max_nodes_sample
          ]

        self.batches = []
        if self.shuffle:
            rng.shuffle(self.idx)

        idx = self.idx
        while idx:
            batch = []
            n_nodes = 0
            while idx and n_nodes + self.node_counts[idx[0]] <= self.max_nodes_batch:
                next_idx, idx = idx[0], idx[1:]
                n_nodes += self.node_counts[next_idx]
                batch.append(next_idx)
            self.batches.append(batch)

        if len(self.batches_single) > 0:
            self.batches += self.batches_single
            if self.shuffle:
                random.shuffle(self.batches)

    def __len__(self):
        """Return the total number of batches.

        Returns:
            int: Number of batches that will be yielded during iteration
        """
        if not self.batches:
            self._form_batches()
        return len(self.batches)

    def __iter__(self):
        """Iterate over batches, yielding lists of dataset indices.

        Yields:
            list: List of integer dataset indices for each batch. Each yielded list
                contains indices of RNA structures that should be collated together.
        """
        if not self.batches:
            self._form_batches()
        yield from self.batches
