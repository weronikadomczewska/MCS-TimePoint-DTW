import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import json
from scipy.signal import butter, filtfilt, find_peaks
from synthetic_data.cpab import CPABWarper
import numpy as np


def get_cbfv_channel(row):
    if row["FV_PRAWA_MCA.3"] == 1:
        return "fv_r"
    elif row["FV_LEWA_MCA.3"] == 1:
        return "fv_l"
    return None


def extract_keypoints(signal, fs):
    peaks, _ = find_peaks(signal, distance=int(fs * 0.4))
    troughs, _ = find_peaks(-signal, distance=int(fs * 0.4))
    return np.sort(np.concatenate([peaks, troughs]))


def interpolate_signal(x):
    x = pd.Series(x)
    return x.interpolate().bfill().ffill().values


def extract_signals(df, cbfv_channel):
    df = df.copy()
    df.columns = df.columns.str.strip()

    t = pd.to_numeric(df["DateTime"].astype(str).str.replace(",", "."), errors="coerce")
    t = (t - t.iloc[0]) * 24 * 3600

    abp = pd.to_numeric(
        df["abp_finger[mm_Hg]"].astype(str).str.replace(",", "."), errors="coerce"
    ).values

    cbfv = pd.to_numeric(
        df["fv_r" if cbfv_channel == "fv_r" else "fv_l"]
        .astype(str)
        .str.replace(",", "."),
        errors="coerce",
    ).values

    abp = interpolate_signal(abp)
    cbfv = interpolate_signal(cbfv)

    L = min(len(t), len(abp), len(cbfv))
    return t[:L], abp[:L], cbfv[:L]


def bandpass_filter(signal, fs, low=0.5, high=10, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal).copy()


def normalize(x):
    return (x - np.mean(x)) / (np.std(x) + 1e-8)


def plot_sample(t, abp, abp_w, cbfv, cbfv_w, abp_kp, abp_kp_w, save_path=None):
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    ax[0].plot(t, abp, label="ABP")
    ax[0].plot(t, abp_w, alpha=0.6, label="ABP warped")
    ax[0].scatter(t[abp_kp], abp[abp_kp], s=10)
    ax[0].scatter(t[abp_kp_w], abp_w[abp_kp_w], s=10)
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(t, cbfv, label="CBFV")
    ax[1].plot(t, cbfv_w, alpha=0.6, label="CBFV warped")
    ax[1].legend()
    ax[1].grid()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


config = json.load(open("config/config.json"))

root = Path(config["data_folder_path"])
finetuning_data_path = Path(config["finetuning_data_path"])

cbfv_config = pd.read_csv("config/cbfv_config.csv")
cbfv_config["cbfv_channel"] = cbfv_config.apply(get_cbfv_channel, axis=1)
cbfv_config = cbfv_config[cbfv_config["cbfv_channel"].notna()]

meta_map = dict(zip(cbfv_config["patient_id"], cbfv_config["cbfv_channel"]))

fs = 200
warper = CPABWarper(tess_size=[16])

counters = {"baseline": 0, "position": 0}

for patient_dir in root.iterdir():
    if not patient_dir.is_dir():
        continue

    patient_id = patient_dir.name

    if patient_id not in meta_map:
        continue

    cbfv_channel = meta_map[patient_id]

    baseline_path = patient_dir / "baseline.csv"
    position_path = patient_dir / "zmiana_pozycji.csv"

    if not baseline_path.exists() or not position_path.exists():
        continue

    print(f"[OK] {patient_id}")

    datasets = {
        "baseline": pd.read_csv(baseline_path, sep=";"),
        "position": pd.read_csv(position_path, sep=";"),
    }

    for tag, df in datasets.items():
        t, abp, cbfv = extract_signals(df, cbfv_channel)

        if len(abp) < fs * 5:  # min 5 sekund
            continue

        # --- filtering ---
        abp = bandpass_filter(abp, fs)
        cbfv = bandpass_filter(cbfv, fs)

        # --- normalization ---
        abp = normalize(abp)
        cbfv = normalize(cbfv)

        # --- keypoints ---
        abp_kp = extract_keypoints(abp, fs)
        if len(abp_kp) < 10:
            continue

        # --- CPAB (ważne: mniejsze deformacje dla długich sygnałów) ---
        scale = np.random.uniform(0.03, 0.08)
        theta = warper.sample_theta(scale=scale)

        abp_w, grid_t = warper.warp(abp, theta)
        cbfv_w, _ = warper.warp(cbfv, theta)

        abp_kp_w = warper.warp_keypoints(abp_kp, grid_t, len(abp))

        # --- SAVE ---
        idx = counters[tag]
        save_dir = finetuning_data_path / tag
        save_dir.mkdir(parents=True, exist_ok=True)

        path = save_dir / f"sample_{idx}.npz"

        np.savez_compressed(
            path,
            t=t.astype(np.float32),
            abp=abp.astype(np.float32),
            abp_warped=abp_w.astype(np.float32),
            cbfv=cbfv.astype(np.float32),
            cbfv_warped=cbfv_w.astype(np.float32),
            abp_kp=abp_kp.astype(np.int32),
            abp_kp_warped=abp_kp_w.astype(np.int32),
            theta=theta.squeeze().cpu().numpy(),
            grid_t=grid_t.squeeze().cpu().numpy(),
            patient_id=patient_id,
            condition=tag,
        )


print("DONE")
