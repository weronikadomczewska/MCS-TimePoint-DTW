import numpy as np
import matplotlib.pyplot as plt

# Create time axis for one full cardiac cycle (1.0 second = 60 bpm)
t = np.linspace(0, 1.0, 500)

# Synthetic pulsatile signal (CBFV-like morphology)
signal = (
    1.2 * np.exp(-(((t - 0.1) / 0.05) ** 2))  # F1: Systolic upstroke at 100ms
    + 0.7 * np.exp(-(((t - 0.25) / 0.08) ** 2))  # F2: Tidal wave
    + 0.4 * np.exp(-(((t - 0.45) / 0.12) ** 2))  # F3: Dicrotic wave / Diastolic flow
)

# --- PHYSIOLOGICAL SCALING ---
# Normal ranges based on Krejza et al. (1999)
EDV_target = 45.0  # End-Diastolic Velocity [cm/s] (Normal: 30-60)
PSV_target = 110.0  # Peak Systolic Velocity [cm/s] (Normal: 70-140)

# Normalize baseline from 0 to 1
signal_norm = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))

# Scale to physiological boundaries [cm/s]
signal_cbfv = EDV_target + signal_norm * (PSV_target - EDV_target)

# --- PLOTTING ---
plt.figure(figsize=(10, 6))
plt.plot(t * 1000, signal_cbfv, color="#8B0000", linewidth=2.5, label="CBFV waveform")

# Mark keypoints (peaks)
peaks_t = [0.1, 0.25, 0.45]  # Time in seconds
labels = ["F1 (PSV)", "F2", "F3"]

for p, l in zip(peaks_t, labels):
    # Find exact Y value at given Time
    idx = np.argmin(np.abs(t - p))
    y = signal_cbfv[idx]

    plt.scatter(p * 1000, y, color="black", s=50, zorder=5)
    plt.text(p * 1000 + 15, y + 2, l, fontsize=11, fontweight="bold")

# Add algorithm sanity check thresholds
plt.axhline(
    y=140,
    color="red",
    linestyle="--",
    alpha=0.6,
    label="Max artifact threshold (140 cm/s)",
)
plt.axhline(
    y=30,
    color="blue",
    linestyle="--",
    alpha=0.6,
    label="Min dropout threshold (30 cm/s)",
)

# Labels and limits
plt.title("Example CBFV waveform", fontsize=14)
plt.xlabel("Time [ms]", fontsize=12)
plt.ylabel("CBFV [cm/s]", fontsize=12)

# Set Y-axis from 0 to 220 to clearly show the boundaries
plt.ylim(0, 220)
plt.xlim(0, 1000)

plt.grid(True, linestyle=":", alpha=0.7)
plt.legend(loc="upper right")

plt.tight_layout()
plt.savefig("cbfv_plot_physiological.png", dpi=300)
plt.show()
