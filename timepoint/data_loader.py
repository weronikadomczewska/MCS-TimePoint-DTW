import torch
import numpy as np
import os


def random_crop(signal, window):
    L = len(signal)

    if L <= window:
        return signal, 0  # start=0

    start = np.random.randint(0, L - window)
    return signal[start : start + window], start


def kp_to_mask(kp_indices, length):
    mask = np.zeros(length, dtype=np.float32)
    kp_indices = kp_indices[(kp_indices >= 0) & (kp_indices < length)]
    mask[kp_indices] = 1.0
    return mask


def process_signal(x, x_w, kp, kp_w, window):
    # crop
    x, start = random_crop(x, window)
    x_w = x_w[start : start + window]

    # keypoints → crop + shift
    kp = kp[(kp >= start) & (kp < start + window)] - start
    kp_w = kp_w[(kp_w >= start) & (kp_w < start + window)] - start

    # maska (WAŻNE)
    kp_mask = kp_to_mask(kp, window)
    kp_w_mask = kp_to_mask(kp_w, window)

    return x, x_w, kp_mask, kp_w_mask


class NPZLoader(torch.utils.data.Dataset):
    def __init__(self, data_dir, use_signal="abp", window=1024):
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".npz")]
        self.use_signal = use_signal
        self.window = window

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])

        if self.use_signal == "abp":
            x = data["abp"]
            x_w = data["abp_warped"]
            kp = data["abp_kp"]
            kp_w = data["abp_kp_warped"]

        else:
            x = data["cbfv"]
            x_w = data["cbfv_warped"]
            kp = data["cbfv_kp"]
            kp_w = data["cbfv_kp_warped"]

        x, x_w, kp_mask, kp_w_mask = process_signal(x, x_w, kp, kp_w, self.window)

        # 🔴 FULL match_mask (ale dla MAŁEGO window!)
        match_mask = np.outer(kp_mask, kp_w_mask).astype(np.float32)

        return {
            "x": torch.tensor(x, dtype=torch.float32),
            "x_w": torch.tensor(x_w, dtype=torch.float32),
            "kp": torch.tensor(kp_mask),
            "kp_w": torch.tensor(kp_w_mask),
            "match_mask": torch.tensor(match_mask, dtype=torch.float32),
        }
