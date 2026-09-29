import numpy as np
from torch.utils.data import Dataset, Sampler

from src.data.sec_utils import Sequence


class RNACoGenerationDataset(Dataset):
    """One item per unique sequence, padded/cropped to ``crop_len``; a random structure
    and window per draw, so one epoch sees one conformer each."""

    def __init__(self, sequences: list[Sequence], crop_len=256, seed=0) -> None:
        self.sequences = sequences
        self.crop_len = crop_len
        self.rng = np.random.default_rng(seed)

        self.node_counts = np.minimum(
            np.array([seq.length for seq in sequences]), crop_len
        )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return self.sequences[index].get_random_sample(self.rng, self.crop_len)

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

    def __init__(self, node_counts, max_nodes_batch=3000, max_nodes_sample=5000, shuffle=True, seed=0):
        self.seed = seed
        self.node_counts = node_counts
        self.shuffle = shuffle
        self.max_nodes_batch = max_nodes_batch
        self.max_nodes_sample = max_nodes_sample
        self.epoch = 0

        self._form_batches()

    def set_epoch(self, epoch):
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
        rng = np.random.default_rng(self.seed + self.epoch)

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
            if not batch:                       # rows larger than max_nodes_batch
                batch, idx = [idx[0]], idx[1:]
            self.batches.append(batch)

        if len(self.batches_single) > 0:
            self.batches += self.batches_single
            if self.shuffle:
                rng.shuffle(self.batches)

    def __len__(self):
        """Return the total number of batches.

        Returns:
            int: Number of batches that will be yielded during iteration
        """
        return len(self.batches)

    def __iter__(self):
        """Iterate over batches, yielding lists of dataset indices.

        Yields:
            list: List of integer dataset indices for each batch. Each yielded list
                contains indices of RNA structures that should be collated together.
        """
        yield from self.batches
