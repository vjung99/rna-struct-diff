import numpy as np
from torch.utils.data import Dataset

class RNACoGenerationDataset(Dataset):
    def __init__(self, s, x, v) -> None:
        self.N, self.L = s.shape

        self.s, self.x, self.v = s, np.nan_to_num(x), v.reshape(self.N, self.L, -1)

        self.token_mask = np.where(self.s == -1, 0, 1)
        self.atom_mask = np.where(np.isnan(self.v), 0, 1)
        self.v = np.nan_to_num(self.v)

        self.s = np.where(
            self.s == -1, np.max(s) + 1, self.s
        )  # NOTE: Pad idx set to max + 1

        print(f"Dataset Loaded {self.s.shape=} {self.x.shape=} {self.v.shape=}")

    def __len__(self):
        return len(self.s)

    def __getitem__(self, index):
        return (
            self.s[index, :],
            self.x[index, :, :],
            self.v[index, :, :],
            self.token_mask[index, :],
            self.atom_mask[index, :, :],
        )
