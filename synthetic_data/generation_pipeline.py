import torch


def _patched_solve(B, A):
    X = torch.linalg.solve(A, B)
    return X, None


torch.solve = _patched_solve


import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from signal_generators import CBFVSignalGenerator
from cpab import CPABWarper

from scipy.signal import butter, filtfilt

def highpass_filter(signal, fs, cutoff=0.5, order=3):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    return filtfilt(b, a, signal)


def extract_keypoints(signal, fs):
    peaks, _ = find_peaks(signal, distance=int(fs * 0.4))
    troughs, _ = find_peaks(-signal, distance=int(fs * 0.4))
    return np.sort(np.concatenate([peaks, troughs]))


def plot_all(
    t,
    abp,
    cbfv,
    abp_warped,
    cbfv_warped,
    abp_kp,
    abp_kp_warped,
    cbfv_kp,
    cbfv_kp_warped,
    save_path=None,
):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # --- ABP ---
    ax1.plot(t, abp, color="red", label="ABP original")
    ax1.plot(t, abp_warped, color="darkred", alpha=0.6, label="ABP warped")

    ax1.scatter(t[abp_kp], abp[abp_kp], color="black", s=15)
    ax1.scatter(t[abp_kp_warped], abp_warped[abp_kp_warped], color="gray", s=15)

    ax1.set_ylabel("ABP (mmHg)")
    ax1.set_title("ABP")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # --- CBFV ---
    ax2.plot(t, cbfv, color="blue", alpha=0.7, label="CBFV original")
    ax2.plot(t, cbfv_warped, color="navy", alpha=0.6, label="CBFV warped")

    ax2.scatter(t[cbfv_kp], cbfv[cbfv_kp], color="black", s=15)
    ax2.scatter(t[cbfv_kp_warped], cbfv_warped[cbfv_kp_warped], color="gray", s=15)

    ax2.set_ylabel("CBFV (cm/s)")
    ax2.set_xlabel("Time (s)")
    ax2.set_title("CBFV")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def generate_cpab_dataset(
    n_samples=5000,
    save_dir="dataset_cpab",
    plot_dir="plots_cpab",
    fs=100,
    duration=10,
    save_plots=True,
):
    os.makedirs(save_dir, exist_ok=True)
    if save_plots:
        os.makedirs(plot_dir, exist_ok=True)

    generator = CBFVSignalGenerator(fs=fs, duration=duration)
    warper = CPABWarper(tess_size=[16])

    for i in range(n_samples):
        t, abp, cbfv, metadata = generator.generate()

        # --- FILTER ---
        abp_filt = highpass_filter(abp, fs).copy()
        cbfv_filt = highpass_filter(cbfv, fs).copy()

        # --- KEYPOINTS ON FILTERED ---
        abp_kp = extract_keypoints(abp_filt, fs)
        cbfv_kp = extract_keypoints(cbfv_filt, fs)

        # # --- 2. Extract keypoints ---
        # abp_kp = extract_keypoints(abp, fs)
        # cbfv_kp = extract_keypoints(cbfv, fs)

        # --- 3. CPAB transform (SAME theta!) ---
        theta = warper.sample_theta()

        abp_warped, grid_t = warper.warp(abp_filt, theta)
        cbfv_warped, _ = warper.warp(cbfv_filt, theta)

        # abp_warped, grid_t = warper.warp(abp, theta)
        # cbfv_warped, _ = warper.warp(cbfv, theta)

        abp_kp_warped = warper.warp_keypoints(abp_kp, grid_t, len(abp))
        cbfv_kp_warped = warper.warp_keypoints(cbfv_kp, grid_t, len(cbfv))

        # --- 4. Save ---
        np.savez_compressed(
            os.path.join(save_dir, f"sample_{i}.npz"),
            # signals
            t=t,
            abp=abp_filt,
            cbfv=cbfv_filt,
            abp_warped=abp_warped,
            cbfv_warped=cbfv_warped,
            # keypoints
            abp_kp=abp_kp,
            cbfv_kp=cbfv_kp,
            abp_kp_warped=abp_kp_warped,
            cbfv_kp_warped=cbfv_kp_warped,
            # cpab
            theta=theta.squeeze().cpu().numpy(),
            grid_t=grid_t.squeeze().cpu().numpy(),
            # metadata
            metadata=str(metadata),
        )

        # --- 5. Plot ---
        if save_plots and i < 50:
            plot_all(
                t,
                abp,
                cbfv,
                abp_warped,
                cbfv_warped,
                abp_kp,
                abp_kp_warped,
                cbfv_kp,
                cbfv_kp_warped,
                save_path=os.path.join(plot_dir, f"sample_{i}.png"),
            )

        if i % 100 == 0:
            print(f"[INFO] {i}/{n_samples}")

    print("DONE")


if __name__ == "__main__":
    generate_cpab_dataset(
        n_samples=6, save_dir="data_cpab", plot_dir="plots_cpab", fs=100, duration=10
    )
