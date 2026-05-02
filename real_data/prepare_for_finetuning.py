import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import json
from scipy.signal import butter, filtfilt, find_peaks
from synthetic_data.cpab import CPABWarper
import numpy as np
import os


def get_cbfv_channel(row):
    left = row["FV_LEWA_MCA.3"]
    right = row["FV_PRAWA_MCA.3"]

    if right == 1:
        return "fv_r"
    elif left == 1:
        return "fv_l"
    else:
        return None


def extract_keypoints(signal, fs):
    peaks, _ = find_peaks(signal, distance=int(fs * 0.4))
    troughs, _ = find_peaks(-signal, distance=int(fs * 0.4))
    return np.sort(np.concatenate([peaks, troughs]))


def interpolate_signal(x):
    x = pd.Series(x)

    # interpolacja liniowa
    x = x.interpolate(method="linear")

    # fallback (gdy NaN na początku/końcu)
    x = x.bfill().ffill()

    return x.values


def convert_datetime(df):
    dt_numeric = pd.to_numeric(
        df["DateTime"].astype(str).str.replace(",", "."), errors="coerce"
    )

    dt = pd.to_datetime(dt_numeric, origin="1899-12-30", unit="D")

    t = (dt - dt.iloc[0]).dt.total_seconds()

    return t.values


def extract_signals(df, cbfv_channel):
    """
    Extracts time (in seconds), ABP and CBFV signals from dataframe.

    Args:
        df: pandas DataFrame (baseline or position_change)
        cbfv_channel: "fv_r" or "fv_l"

    Returns:
        t: np.array [L] (seconds)
        abp: np.array [L]
        cbfv: np.array [L]
    """

    # --- clean column names (important!) ---
    df = df.copy()
    df.columns = df.columns.str.strip()

    # --- TIME (Excel → seconds) ---
    t = pd.to_numeric(df["DateTime"].astype(str).str.replace(",", "."), errors="coerce")

    t = (t - t.iloc[0]) * 24 * 3600  # days → seconds

    # --- ABP ---
    abp = pd.to_numeric(
        df["abp_finger[mm_Hg]"].astype(str).str.replace(",", "."), errors="coerce"
    ).values

    # --- CBFV ---
    if cbfv_channel == "fv_r":
        cbfv = pd.to_numeric(
            df["fv_r"].astype(str).str.replace(",", "."), errors="coerce"
        ).values
    elif cbfv_channel == "fv_l":
        cbfv = pd.to_numeric(
            df["fv_l"].astype(str).str.replace(",", "."), errors="coerce"
        ).values
    else:
        raise ValueError(f"Invalid cbfv_channel: {cbfv_channel}")

    # --- interpolacja ---
    abp = interpolate_signal(abp)
    cbfv = interpolate_signal(cbfv)

    # --- ensure equal length ---
    L = min(len(t), len(abp), len(cbfv))

    return t[:L], abp[:L], cbfv[:L]


import matplotlib.pyplot as plt
import numpy as np
import os


def plot_sample(
    abp, abp_w, cbfv, cbfv_w, abp_kp, abp_kp_w, fs, patient_id, tag, idx, save_dir=None
):
    t = np.arange(len(abp)) / fs

    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # --- ABP ---
    ax[0].plot(t, abp, color="red", label="ABP")
    ax[0].plot(t, abp_w, color="darkred", alpha=0.6, label="ABP warped")

    ax[0].scatter(t[abp_kp], abp[abp_kp], color="black", s=15, label="kp")
    ax[0].scatter(t[abp_kp_w], abp_w[abp_kp_w], color="gray", s=15, label="kp warped")

    ax[0].set_title(f"ABP | {patient_id} | {tag}")
    ax[0].legend()
    ax[0].grid()

    # --- CBFV ---
    ax[1].plot(t, cbfv, color="blue", label="CBFV")
    ax[1].plot(t, cbfv_w, color="navy", alpha=0.6, label="CBFV warped")

    ax[1].set_title("CBFV")
    ax[1].legend()
    ax[1].grid()

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{patient_id}_{tag}_{idx}.png")
        plt.savefig(path, dpi=150)
        plt.close()
    else:
        plt.show()


def segment(signal, window, step):
    return [signal[i : i + window] for i in range(0, len(signal) - window, step)]


def bandpass_filter(signal, fs, low=0.5, high=10, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal).copy()


def normalize(x):
    return (x - np.mean(x)) / (np.std(x) + 1e-8)


config_path = Path("config/config.json")
config = json.load(open(config_path))

real_data_path = config["data_folder_path"]

cbfv_config_path = "config/cbfv_config.csv"

cbfv_config = pd.read_csv(cbfv_config_path)

cbfv_config["cbfv_channel"] = cbfv_config.apply(get_cbfv_channel, axis=1)

# filtering out patients with no cbfv
cbfv_config = cbfv_config[cbfv_config["cbfv_channel"].notna()]

meta_map = dict(zip(cbfv_config["patient_id"], cbfv_config["cbfv_channel"]))

root = Path(real_data_path)

finetuning_data_path = Path(config["finetuning_data_path"])

fs = 100
window = 1000
step = 500

warper = CPABWarper(tess_size=[16])

counters = {"baseline": 0, "position": 0}


plot_dir = "plots_real"
os.makedirs(plot_dir, exist_ok=True)

for patient_dir in root.iterdir():
    if not patient_dir.is_dir():
        continue

    patient_id = patient_dir.name

    if patient_id not in meta_map:
        print(f"[SKIP] {patient_id} not in metadata")
        continue

    cbfv_channel = meta_map[patient_id]

    if pd.isna(cbfv_channel):
        print(f"[SKIP] {patient_id} no valid CBFV channel")
        continue

    baseline_path = patient_dir / "baseline.csv"
    position_path = patient_dir / "zmiana_pozycji.csv"

    if not baseline_path.exists() or not position_path.exists():
        print(f"[SKIP] {patient_id} missing files")
        continue

    print(f"[OK] {patient_id}")

    baseline_df = pd.read_csv(baseline_path, sep=";")
    position_df = pd.read_csv(position_path, sep=";")

    datasets = {"baseline": baseline_df, "position": position_df}

    for tag, df in datasets.items():
        # --- extract ---
        t, abp, cbfv = extract_signals(df, cbfv_channel)

        if len(abp) < window:
            print(f"  [SKIP {tag}] too short")
            continue

        print(abp)

        # --- filtering ---
        abp = bandpass_filter(abp, fs)
        cbfv = bandpass_filter(cbfv, fs)

        # --- normalization ---
        abp = normalize(abp)
        cbfv = normalize(cbfv)

        # --- segmentation ---
        abp_segments = segment(abp, window, step)
        cbfv_segments = segment(cbfv, window, step)

        # print(abp_segments)

        for abp_seg, cbfv_seg in zip(abp_segments, cbfv_segments):
            print(f"{tag} len signal seg:", len(abp_seg))
            if len(abp_seg) != window:
                continue

            # --- keypoints ---
            abp_kp = extract_keypoints(abp_seg, fs)
            print(f"{tag} len signal:", len(abp_kp))
            if len(abp_kp) < 5:
                continue

            # --- CPAB ---
            theta = warper.sample_theta()

            abp_w, grid_t = warper.warp(abp_seg, theta)
            cbfv_w, _ = warper.warp(cbfv_seg, theta)

            abp_kp_w = warper.warp_keypoints(abp_kp, grid_t, len(abp_seg))

            # --- SAVE ---
            idx = counters[tag]

            save_dir = finetuning_data_path / tag
            save_dir.mkdir(parents=True, exist_ok=True)

            finetuning_path = save_dir / f"sample_{idx}.npz"

            print(f"[SAVING] {finetuning_path}")

            if idx < 10:  # 🔥 tylko kilka na start
                plot_sample(
                    abp_seg,
                    abp_w,
                    cbfv_seg,
                    cbfv_w,
                    abp_kp,
                    abp_kp_w,
                    fs,
                    patient_id,
                    tag,
                    idx,
                    save_dir="debug_plots",
                )

            np.savez_compressed(
                finetuning_path,
                abp=abp_seg.astype(np.float32),
                abp_warped=abp_w.astype(np.float32),
                cbfv=cbfv_seg.astype(np.float32),
                cbfv_warped=cbfv_w.astype(np.float32),
                abp_kp=abp_kp.astype(np.int32),
                abp_kp_warped=abp_kp_w.astype(np.int32),
                theta=theta.squeeze().cpu().numpy(),
                grid_t=grid_t.squeeze().cpu().numpy(),
                patient_id=patient_id,
                condition=tag,
            )

            counters[tag] += 1

        print(f"  {tag}: total saved = {counters[tag]}")

print("DONE")
