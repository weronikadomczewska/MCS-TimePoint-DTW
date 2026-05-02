import numpy as np
import torch


def _patched_solve(B, A):
    X = torch.linalg.solve(A, B)
    return X, None  # mimic old behavior (solution, LU)


torch.solve = _patched_solve
import torch.nn.functional as F
from libcpab import Cpab

"""
https://github.com/SkafteNicki/libcpab
"""


class CPABWarper:
    def __init__(self, tess_size=[16], device="cpu", backend="pytorch"):
        """
        CPAB-based time warper for 1D signals.

        tess_size: number of segments (paper uses ~16)
        """
        self.device = device

        self.T = Cpab(tess_size=tess_size, backend=backend, device=device)

    def _create_grid(self, L):
        """Create normalized grid [0,1]"""
        return torch.linspace(0, 1, L, device=self.device).unsqueeze(0)

    def _warp_signal(self, signal, grid_t):
        signal = np.asarray(signal)
        grid_t = grid_t.squeeze().cpu().numpy()

        L = len(signal)

        # grid_t jest w [0,1] → skalujemy do indeksów
        grid_idx = grid_t * (L - 1)

        # interpolacja 1D
        warped = np.interp(grid_idx, np.arange(L), signal)

        return warped

    def sample_theta(self, batch_size=1, scale=0.1):
        thetas = [self.T.sample_transformation() for _ in range(batch_size)]
        theta = torch.cat(thetas, dim=0)
        return scale * theta

    def warp(self, signal, theta=None):
        """
        Warp signal with CPAB.

        Returns:
        - warped signal
        - transformed grid
        """
        L = len(signal)

        if theta is None:
            theta = self.sample_theta(scale=0.05)

        grid = self._create_grid(L)
        grid_t = self.T.transform_grid(grid, theta)

        warped = self._warp_signal(signal, grid_t)

        return warped, grid_t

    def warp_keypoints(self, keypoints, grid_t, L):
        """
        Warp keypoints using the same CPAB transformation.
        """
        kp = np.array(keypoints)

        # normalize to [0,1]
        kp_norm = kp / (L - 1)

        grid_t_np = grid_t.squeeze().cpu().numpy()

        kp_warped = np.interp(kp_norm, np.linspace(0, 1, L), grid_t_np)

        kp_warped = (kp_warped * (L - 1)).astype(int)

        # clip to valid range
        kp_warped = np.clip(kp_warped, 0, L - 1)

        return kp_warped
