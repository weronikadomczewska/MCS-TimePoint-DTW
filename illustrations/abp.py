import numpy as np
import matplotlib.pyplot as plt

# Time axis
t = np.linspace(0, 0.9, 500)

# Synthetic ABP-like waveform
signal = (
    60
    + 45 * np.exp(-(((t - 0.25) / 0.08) ** 2))  # systolic peak
    - 10 * np.exp(-(((t - 0.15) / 0.05) ** 2))  # early dip
    + 8 * np.exp(-(((t - 0.55) / 0.12) ** 2))  # diastolic bump
)

# Plot
plt.figure()
plt.plot(t, signal)

# Key points (approximate positions)
points = {
    "SPO": (0.12, signal[np.argmin(np.abs(t - 0.12))]),
    "SPP": (0.25, signal[np.argmin(np.abs(t - 0.25))]),
    "DN": (0.45, signal[np.argmin(np.abs(t - 0.45))]),
    "DPP": (0.55, signal[np.argmin(np.abs(t - 0.55))]),
    "DPE": (0.85, signal[np.argmin(np.abs(t - 0.85))]),
}

# Plot points
for name, (x, y) in points.items():
    plt.scatter(x, y)

# Annotations
plt.annotate(
    "Systolic phase onset (SPO)",
    xy=points["SPO"],
    xytext=(0.02, 55),
    arrowprops=dict(arrowstyle="->"),
)

plt.annotate(
    "Systolic phase peak (SPP)",
    xy=points["SPP"],
    xytext=(0.3, 100),
    arrowprops=dict(arrowstyle="->"),
)

plt.annotate(
    "Dicrotic notch (DN)",
    xy=points["DN"],
    xytext=(0.5, 75),
    arrowprops=dict(arrowstyle="->"),
)

plt.annotate(
    "Diastolic phase peak (DPP)",
    xy=points["DPP"],
    xytext=(0.6, 70),
    arrowprops=dict(arrowstyle="->"),
)

plt.annotate(
    "Diastolic phase endpoint (DPE)",
    xy=points["DPE"],
    xytext=(0.65, 55),
    arrowprops=dict(arrowstyle="->"),
)

# Labels
plt.title("Example ABP cardiac cycle")
plt.xlabel("Time (s)")
plt.ylabel("Pressure (mmHg)")

plt.tight_layout()
plt.savefig("abp_cycle.png", dpi=300)
plt.show()
