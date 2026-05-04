import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dtaidistance import dtw


# -----------------------------
# ⚙️ CONFIG
# -----------------------------
BASE_PATH = "../data_for_finetuning"
FS = 200

EXCLUDE_FILES = {
    "sample_PAC02.npz",
    "sample_PAC03.npz",
    "sample_PAC33.npz",
    "sample_PAC38.npz",
    "sample_PAC39.npz",
}


# -----------------------------
# 🔧 CORE: compute delays per cycle
# -----------------------------
def compute_cycle_delays(abp, cbfv, abp_kp, fs=200):
    delays = []

    print("Signal length:", len(abp))
    print("Num keypoints:", len(abp_kp))

    for i in range(len(abp_kp) - 1):
        start = abp_kp[i]
        end = abp_kp[i + 1]

        x = abp[start:end]
        y = cbfv[start:end]

        if len(x) < 10 or len(y) < 10:
            continue

        path = dtw.warping_path(x, y)

        lags = [(j - i) / fs for i, j in path]

        delays.append(np.mean(lags))

    return delays


# -----------------------------
# 🔧 PROCESS ONE FOLDER
# -----------------------------
def process_split(split_name):
    folder = os.path.join(BASE_PATH, split_name)
    files = sorted([f for f in os.listdir(folder) if f.endswith(".npz")])

    all_delays = []
    per_file_stats = []

    for f in files:
        if f in EXCLUDE_FILES:
            continue

        path = os.path.join(folder, f)
        data = np.load(path)

        abp = data["abp"]
        cbfv = data["cbfv"]
        abp_kp = data["abp_kp"]

        if len(abp_kp) < 5:
            continue

        delays = compute_cycle_delays(abp, cbfv, abp_kp, FS)

        if len(delays) == 0:
            continue

        all_delays.extend(delays)

        per_file_stats.append(
            {
                "file": f,
                "mean_delay": np.mean(delays),
                "std_delay": np.std(delays),
                "num_cycles": len(delays),
            }
        )

    return np.array(all_delays), pd.DataFrame(per_file_stats)


# -----------------------------
# 🚀 RUN
# -----------------------------
baseline_delays, baseline_df = process_split("baseline")
position_delays, position_df = process_split("position")


# -----------------------------
# 📊 GLOBAL SUMMARY
# -----------------------------
print("\n=== GLOBAL RESULTS ===")

print("\nBASELINE:")
print("Mean delay:", np.mean(baseline_delays))
print("Std delay:", np.std(baseline_delays))

print("\nPOSITION:")
print("Mean delay:", np.mean(position_delays))
print("Std delay:", np.std(position_delays))


# -----------------------------
# 📊 PER-FILE SUMMARY
# -----------------------------
print("\n=== PER FILE (BASELINE) ===")
print(baseline_df.head())

print("\n=== PER FILE (POSITION) ===")
print(position_df.head())


# -----------------------------
# 📈 HISTOGRAM
# -----------------------------
plt.figure(figsize=(8, 5))

plt.hist(baseline_delays, bins=40, alpha=0.5, label="baseline")
plt.hist(position_delays, bins=40, alpha=0.5, label="position")

plt.xlabel("Delay (seconds)")
plt.ylabel("Count")
plt.title("Cycle-wise ABP → CBFV delay")
plt.legend()
plt.grid()
plt.show()


# -----------------------------
# 📈 BOXPLOT (very good for thesis)
# -----------------------------
plt.figure()

plt.boxplot([baseline_delays, position_delays], labels=["baseline", "position"])

plt.ylabel("Delay (seconds)")
plt.title("Delay comparison")
plt.grid()
plt.show()


# -----------------------------
# 📈 OPTIONAL: visualize one sample
# -----------------------------
def plot_example(path):
    data = np.load(path)

    abp = data["abp"]
    cbfv = data["cbfv"]
    abp_kp = data["abp_kp"]

    plt.figure(figsize=(12, 4))
    plt.plot(abp, label="ABP")
    plt.plot(cbfv, label="CBFV")

    plt.scatter(abp_kp, abp[abp_kp], s=10)

    plt.legend()
    plt.title("Example signal with cycles")
    plt.grid()
    plt.show()


# example usage:
# plot_example("../finetuning_data/baseline/sample_0.npz")
