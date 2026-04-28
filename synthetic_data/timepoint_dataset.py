from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class TimePointDataset(Dataset):
    def __init__(self, path):
        self.files = list(Path(path).glob("*.npz"))

    def __len__(self):
        return len(self.files)

    def kp_to_mask(self, kp, L):
        mask = np.zeros(L)
        mask[kp] = 1
        return mask

    def build_match_mask(self, kp, kp_w, L):
        """
        Creates [L, L] matrix (1_G)
        """
        mask = np.zeros((L, L))
        for i, j in zip(kp, kp_w):
            if i < L and j < L:
                mask[i, j] = 1
        return mask

    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True)

        abp = data["abp"].astype(np.float32)
        abp_w = data["abp_warped"].astype(np.float32)

        kp = data["kp"]
        kp_w = data["kp_warped"]

        L = len(abp)

        kp_mask = self.kp_to_mask(kp, L)
        kp_w_mask = self.kp_to_mask(kp_w, L)

        match_mask = self.build_match_mask(kp, kp_w, L)

        return {
            "x": torch.from_numpy(abp),
            "x_w": torch.from_numpy(abp_w),
            "kp": torch.from_numpy(kp_mask),
            "kp_w": torch.from_numpy(kp_w_mask),
            "match_mask": torch.from_numpy(match_mask)
        }