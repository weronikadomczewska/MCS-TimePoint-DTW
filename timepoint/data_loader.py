import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path


class NPZLoader(Dataset):
    def __init__(self, path, transform=None, use_signal="abp"):
        self.files = list(Path(path).glob("*.npz"))
        self.transform = transform
        self.use_signal = use_signal  # "abp" or "cbfv"

    def __len__(self):
        return len(self.files)

    def kp_to_mask(self, kp, L):
        mask = torch.zeros(L)
        kp = kp[kp < L]  # safety
        mask[kp] = 1.0
        return mask

    def build_match_mask(self, kp, kp_w, L):
        mask = torch.zeros((L, L))
        for i, j in zip(kp, kp_w):
            if i < L and j < L:
                mask[i, j] = 1.0
        return mask

    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True)

        # --- select signal ---
        if self.use_signal == "abp":
            x = torch.from_numpy(data["abp"]).float()
            x_w = torch.from_numpy(data["abp_warped"]).float()
            kp = torch.from_numpy(data["abp_kp"]).long()
            kp_w = torch.from_numpy(data["abp_kp_warped"]).long()

        else:
            x = torch.from_numpy(data["cbfv"]).float()
            x_w = torch.from_numpy(data["cbfv_warped"]).float()
            kp = torch.from_numpy(data["cbfv_kp"]).long()
            kp_w = torch.from_numpy(data["cbfv_kp_warped"]).long()

        L = len(x)

        # --- masks ---
        kp_mask = self.kp_to_mask(kp, L)
        kp_w_mask = self.kp_to_mask(kp_w, L)
        match_mask = self.build_match_mask(kp, kp_w, L)

        # --- optional transform ---
        if self.transform:
            x = self.transform(x)
            x_w = self.transform(x_w)

        return {
            "x": x,
            "x_w": x_w,
            "kp": kp_mask,
            "kp_w": kp_w_mask,
            "match_mask": match_mask,
        }
