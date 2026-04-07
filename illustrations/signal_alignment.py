import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from fastdtw import fastdtw

# ---------------------------
# 1. Generate synthetic signals
# ---------------------------
np.random.seed(0)
t = np.linspace(0, 10, 100)

# Base signal
signal1 = np.sin(t) + 0.3 * np.random.randn(len(t))

# Warped version (nonlinear shift)
t_warped = t + 0.5 * np.sin(0.5 * t)
interp = interp1d(t, signal1, kind="linear", fill_value="extrapolate")
signal2 = interp(t_warped) + 0.3 * np.random.randn(len(t))

# ---------------------------
# 2. DTW alignment
# ---------------------------
distance, path = fastdtw(signal1, signal2)

# Extract aligned signals
aligned_1 = []
aligned_2 = []

for i, j in path:
    aligned_1.append(signal1[i])
    aligned_2.append(signal2[j])

aligned_1 = np.array(aligned_1)
aligned_2 = np.array(aligned_2)

# ---------------------------
# 3. Plot results
# ---------------------------
plt.figure(figsize=(10, 6))

# Original signals
plt.subplot(2, 1, 1)
plt.plot(signal1, label="Signal 1")
plt.plot(signal2, label="Signal 2")
plt.title("Original signals")
plt.legend()
plt.grid(True)

# Aligned signals
plt.subplot(2, 1, 2)
plt.plot(aligned_1, label="Aligned signal 1")
plt.plot(aligned_2, label="Aligned signal 2")
plt.title("Aligned with DTW")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
