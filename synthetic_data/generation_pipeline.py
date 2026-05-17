import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

import torch

from signal_generators import ABPGenerator, CBFVMaderModel, LogNormalCBFV
from cpab import CPABWarper


# --- monkey patch (po importach) ---
def _patched_solve(B, A):
    X = torch.linalg.solve(A, B)
    return X, None


torch.solve = _patched_solve  # noqa


def extract_keypoints(signal, fs):
    """
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
    """
    peaks, _ = find_peaks(
        signal, distance=int(fs * 0.2), height=np.mean(signal) + 0.5 * np.std(signal)
    )
    troughs, _ = find_peaks(-signal, distance=int(fs * 0.2))
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
    n_samples=500,
    base_dir="dataset_cpab",
    plot_dir="plots_cpab",
    fs=100,
    duration=10,
    save_plots=True,
):
    # =========================
    # FOLDERY
    # =========================
    dir_mader = os.path.join(base_dir, "mader")
    dir_log = os.path.join(base_dir, "lognormal")

    os.makedirs(dir_mader, exist_ok=True)
    os.makedirs(dir_log, exist_ok=True)

    if save_plots:
        os.makedirs(plot_dir, exist_ok=True)

    # =========================
    # GENERATORY
    # =========================
    mader = CBFVMaderModel()
    lognorm = LogNormalCBFV(fs)

    warper = CPABWarper(tess_size=[16])

    for i in range(n_samples):
        abp_gen = ABPGenerator(fs=fs)
        # =========================
        # 1. ABP
        # =========================
        t, abp = abp_gen.generate_abp_signal(duration_sec=duration)

        # =========================
        # 2. KEYPOINTY ABP
        # =========================
        # abp_filt = highpass_filter(abp, fs)
        abp_kp = extract_keypoints(abp, fs)

        # =========================
        # 3. CBFV – MADER
        # =========================
        cbfv_mader = mader.simulate(abp, fs)
        # cbfv_mader_filt = highpass_filter(cbfv_mader, fs)
        cbfv_mader_kp = extract_keypoints(cbfv_mader, fs)

        # =========================
        # 4. CBFV – LOGNORMAL
        # =========================
        peaks, _ = find_peaks(abp, distance=int(fs * 0.4))
        beats = t[peaks]

        cbfv_log = lognorm.generate(t, beats, abp=abp)
        # cbfv_log_filt = highpass_filter(cbfv_log, fs)
        cbfv_log_kp = extract_keypoints(cbfv_log, fs)

        # =========================
        # 5. CPAB (ten sam theta!)
        # =========================

        scale = np.random.uniform(0.05, 0.15)
        theta = warper.sample_theta(scale=scale)

        # --- MADER ---
        abp_warped_m, grid_t = warper.warp(abp.copy(), theta)
        cbfv_warped_m, _ = warper.warp(cbfv_mader.copy(), theta)

        abp_kp_warped_m = warper.warp_keypoints(abp_kp.copy(), grid_t, len(abp))
        cbfv_kp_warped_m = warper.warp_keypoints(cbfv_mader_kp.copy(), grid_t, len(abp))

        # --- LOGNORMAL ---
        abp_warped_l, grid_t = warper.warp(abp.copy(), theta)
        cbfv_warped_l, _ = warper.warp(cbfv_log.copy(), theta)

        abp_kp_warped_l = warper.warp_keypoints(abp_kp.copy(), grid_t, len(abp))
        cbfv_kp_warped_l = warper.warp_keypoints(cbfv_log_kp.copy(), grid_t, len(abp))

        # =========================
        # 6. SAVE
        # =========================

        # --- MADER ---
        np.savez_compressed(
            os.path.join(dir_mader, f"sample_{i}.npz"),
            t=t,
            abp=abp,
            cbfv=cbfv_mader,
            abp_warped=abp_warped_m,
            cbfv_warped=cbfv_warped_m,
            abp_kp=abp_kp,
            cbfv_kp=cbfv_mader_kp,
            abp_kp_warped=abp_kp_warped_m,
            cbfv_kp_warped=cbfv_kp_warped_m,
            theta=theta.squeeze().cpu().numpy(),
            grid_t=grid_t.squeeze().cpu().numpy(),
            model="mader",
        )

        # --- LOGNORMAL ---
        np.savez_compressed(
            os.path.join(dir_log, f"sample_{i}.npz"),
            t=t,
            abp=abp,
            cbfv=cbfv_log,
            abp_warped=abp_warped_l,
            cbfv_warped=cbfv_warped_l,
            abp_kp=abp_kp,
            cbfv_kp=cbfv_log_kp,
            abp_kp_warped=abp_kp_warped_l,
            cbfv_kp_warped=cbfv_kp_warped_l,
            theta=theta.squeeze().cpu().numpy(),
            grid_t=grid_t.squeeze().cpu().numpy(),
            model="lognormal",
        )

        # =========================
        # 7. PLOT
        # =========================
        if save_plots and i < 20:
            plot_all(
                t,
                abp,
                cbfv_mader,
                abp_warped_m,
                cbfv_warped_m,
                abp_kp,
                abp_kp_warped_m,
                cbfv_mader_kp,
                cbfv_kp_warped_m,
                save_path=os.path.join(plot_dir, f"mader_{i}.png"),
            )

            plot_all(
                t,
                abp,
                cbfv_log,
                abp_warped_l,
                cbfv_warped_l,
                abp_kp,
                abp_kp_warped_l,
                cbfv_log_kp,
                cbfv_kp_warped_l,
                save_path=os.path.join(plot_dir, f"log_{i}.png"),
            )

        if i % 50 == 0:
            print(f"[INFO] {i}/{n_samples}")

    print("DONE")


if __name__ == "__main__":
    generate_cpab_dataset(
        n_samples=1000,
        base_dir="dataset_synthetic_cpab",
        plot_dir="plots_dataset_synthetic_cpab",
        fs=200,
        duration=10,
    )
