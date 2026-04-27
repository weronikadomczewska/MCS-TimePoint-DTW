import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os


# =========================
# TIMELINE
# =========================
class TimelineGenerator:
    def __init__(self, duration, min_hr=60, max_hr=90):
        self.duration = duration
        self.min_hr = min_hr
        self.max_hr = max_hr

    def generate(self):
        mean_hr = random.randint(self.min_hr, self.max_hr)
        avg_interval = 60.0 / mean_hr

        times = []
        t = 0

        while t < self.duration:
            interval = avg_interval + np.random.normal(0, 0.05)
            t += max(interval, 0.3)  # avoid negative/too small intervals
            times.append(t)

        return np.array(times)


# =========================
# ABP GENERATOR
# =========================
class ABPGenerator:
    def __init__(self, fs, params=None):
        self.fs = fs
        self.params = params

    def gaussian(self, t, a, b, c):
        return a * np.exp(-((t - b) ** 2) / (2 * c**2))

    def generate_wave(self, t_local):
        if self.params is None:
            params = [
                (
                    np.random.uniform(30, 50),
                    np.random.uniform(0.2, 0.3),
                    np.random.uniform(0.04, 0.08),
                ),
                (
                    np.random.uniform(10, 25),
                    np.random.uniform(0.3, 0.4),
                    np.random.uniform(0.05, 0.1),
                ),
                (
                    np.random.uniform(5, 15),
                    np.random.uniform(0.4, 0.55),
                    np.random.uniform(0.08, 0.15),
                ),
            ]
        else:
            params = self.params

        wave = np.zeros_like(t_local)
        for a, b, c in params:
            wave += self.gaussian(t_local, a, b, c)

        return wave


# =========================
# CA MODEL
# =========================
class CAModel:
    def __init__(self, dt, a=None, b=None, c=None, d=None, M=None):
        self.dt = dt

        self.a = a if a is not None else np.random.uniform(0.2, 0.4)
        self.b = b if b is not None else np.random.uniform(0.05, 0.2)
        self.c = c if c is not None else np.random.uniform(0.7, 1.2)
        self.d = d if d is not None else np.random.uniform(0.01, 0.1)
        self.M = M if M is not None else np.random.uniform(0.8, 1.5)

        self.v1 = 0.0
        self.v2 = 0.0
        self.V_bas = np.random.uniform(40, 70)

    def step(self, p):
        dv1 = (
            -(self.a + self.b + self.c) * self.v1
            + (self.c - self.d) * self.v2
            + (self.a + self.b) * p
        )

        dv2 = -self.b * self.v1 - self.d * self.v2 + self.b * p

        self.v1 += self.dt * dv1
        self.v2 += self.dt * dv2

        V_dyn = self.M * (p - self.v1)
        return self.V_bas + V_dyn


# =========================
# MAIN GENERATOR
# =========================
class CBFVSignalGenerator:
    def __init__(
        self,
        fs=100,
        duration=10,
        hr_range=None,
        noise_std=None,
        ca_params=None,
        abp_params=None,
    ):
        self.fs = fs
        self.duration = duration

        self.hr_range = hr_range if hr_range else (55, 110)
        self.noise_std = noise_std

        self.abp_gen = ABPGenerator(fs, params=abp_params)
        self.ca_params = ca_params

        # FIX: initialize timeline
        self.timeline = TimelineGenerator(
            duration=duration, min_hr=self.hr_range[0], max_hr=self.hr_range[1]
        )

    def generate(self):
        t = np.linspace(0, self.duration, int(self.duration * self.fs))
        abp = np.zeros_like(t)

        beats = self.timeline.generate()

        # --- ABP ---
        for i, bt in enumerate(beats):
            next_bt = beats[i + 1] if i < len(beats) - 1 else self.duration
            dt = max(next_bt - bt, 1e-3)

            mask = (t >= bt) & (t < next_bt)

            t_local = (t[mask] - bt) / dt

            wave = self.abp_gen.generate_wave(t_local)

            baseline = 80 if bt < self.duration / 2 else 60
            abp[mask] += wave + baseline

        # --- noise (random per sample) ---
        if self.noise_std is None:
            abp_noise = np.random.uniform(1.0, 3.0)
            cbfv_noise = np.random.uniform(1.5, 4.0)
        else:
            abp_noise, cbfv_noise = self.noise_std

        abp += np.random.normal(0, abp_noise, len(abp))

        # --- CBFV ---
        ca = CAModel(dt=1 / self.fs, **(self.ca_params if self.ca_params else {}))
        cbfv = np.zeros_like(abp)

        for i in range(len(abp)):
            cbfv[i] = ca.step(abp[i])

        cbfv += np.random.normal(0, cbfv_noise, len(cbfv))

        return t, abp, cbfv


# =========================
# KEYPOINTS
# =========================
def extract_keypoints(signal, fs):
    peaks, _ = find_peaks(signal, distance=int(fs * 0.4))
    troughs, _ = find_peaks(-signal, distance=int(fs * 0.4))
    return np.sort(np.concatenate([peaks, troughs]))


# =========================
# VISUALIZATION
# =========================
def plot_signals(t, abp, cbfv, abp_kp=None, cbfv_kp=None):
    fig, ax1 = plt.subplots(figsize=(12, 5))

    ax1.plot(t, abp, color="red", label="ABP")
    if abp_kp is not None:
        ax1.scatter(t[abp_kp], abp[abp_kp], color="darkred", s=30)

    ax2 = ax1.twinx()
    ax2.plot(t, cbfv, color="blue", label="CBFV")
    if cbfv_kp is not None:
        ax2.scatter(t[cbfv_kp], cbfv[cbfv_kp], color="navy", s=30)

    plt.title("ABP → CBFV with keypoints")
    plt.grid(True)
    plt.show()


# =========================
# DATASET
# =========================
def generate_dataset(n_samples=5000, save_dir="data", fs=100, duration=10):
    os.makedirs(save_dir, exist_ok=True)

    gen = CBFVSignalGenerator(fs=fs, duration=duration)

    for i in range(n_samples):
        t, abp, cbfv = gen.generate()

        abp_kp = extract_keypoints(abp, fs)
        cbfv_kp = extract_keypoints(cbfv, fs)

        np.savez_compressed(
            os.path.join(save_dir, f"sample_{i}.npz"),
            t=t,
            abp=abp,
            cbfv=cbfv,
            abp_kp=abp_kp,
            cbfv_kp=cbfv_kp,
        )

        if i % 100 == 0:
            print(f"Generated {i}/{n_samples}")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    # generate_dataset(n_samples=5, duration=10)

    # quick test
    # gen = CBFVSignalGenerator()
    # t, abp, cbfv = gen.generate()
    # plot_signals(t, abp, cbfv,
    #              extract_keypoints(abp, 100),
    #              extract_keypoints(cbfv, 100))

    data = np.load("data/sample_1.npz")

    t = data["t"]
    abp = data["abp"]
    cbfv = data["cbfv"]
    abp_kp = data["abp_kp"]
    cbfv_kp = data["cbfv_kp"]

    plot_signals(
        t, abp, cbfv, extract_keypoints(abp, 100), extract_keypoints(cbfv, 100)
    )

    print(abp.shape)
    print(abp_kp[:10])
