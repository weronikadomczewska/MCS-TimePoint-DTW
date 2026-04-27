import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


class TimelineGenerator:
    """
    Generates shared timeline for ABP and CBFV signals.
    """

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
            t += interval
            times.append(t)

        return np.array(times)


class ABPGenerator:
    """
    Generates ABP signal as 3 Gaussian functions.
    Liu et al.
    """

    def __init__(self, fs):
        self.fs = fs

    def gaussian(self, t, a, b, c):
        return a * np.exp(-((t - b) ** 2) / (2 * c**2))

    def generate_wave(self, t_local):
        params = [
            (40, 0.25, 0.05),
            (20, 0.35, 0.07),
            (10, 0.45, 0.1),
        ]

        wave = np.zeros_like(t_local)
        for a, b, c in params:
            wave += self.gaussian(t_local, a, b, c)

        return wave


class CAModel:
    """

    Mahdi et al.
    """

    def __init__(self, dt):
        self.dt = dt

        self.a = 0.25
        self.b = 0.1
        self.c = 0.9
        self.M = 1.0
        self.d = 0.05

        self.v1 = 0.0
        self.v2 = 0.0

        self.V_bas = 50.0

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
    """
    Mahdi et al.
    """

    def __init__(self, fs=100, duration=10):
        self.fs = fs
        self.duration = duration

        self.timeline = TimelineGenerator(duration)
        self.abp_gen = ABPGenerator(fs)

    def generate(self):
        t = np.linspace(0, self.duration, int(self.duration * self.fs))

        abp = np.zeros_like(t)

        beats = self.timeline.generate()

        # --- ABP generation ---
        for i, bt in enumerate(beats):
            next_bt = beats[i + 1] if i < len(beats) - 1 else self.duration
            dt = next_bt - bt

            t_local = t - bt
            mask = (t >= bt) & (t < next_bt)

            t_norm = t_local / dt
            t_norm[~mask] = 0

            wave = self.abp_gen.generate_wave(t_norm)

            # baseline change (orthostatic stress)
            if bt < self.duration / 2:
                baseline = 80
            else:
                baseline = 60

            abp[mask] += wave[mask] + baseline

        # noise from normal distribution
        abp += np.random.normal(0, 1.5, len(abp))

        # --- CBFV generation FROM ABP ---
        ca = CAModel(dt=1 / self.fs)
        cbfv = np.zeros_like(abp)

        for i in range(len(abp)):
            cbfv[i] = ca.step(abp[i])

        cbfv += np.random.normal(0, 2.0, len(cbfv))

        return t, abp, cbfv


# =========================
# VISUALIZATION
# =========================
def plot_signals(t, abp, cbfv, abp_kp=None, cbfv_kp=None):
    fig, ax1 = plt.subplots(figsize=(12, 5))

    # --- ABP ---
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("ABP (mmHg)", color="red")
    (line_abp,) = ax1.plot(t, abp, color="red", alpha=0.7, label="ABP")

    if abp_kp is not None:
        ax1.scatter(
            t[abp_kp],
            abp[abp_kp],
            color="darkred",
            s=30,
            label="ABP keypoints",
            zorder=5,
        )

    ax1.tick_params(axis="y", labelcolor="red")

    # --- CBFV ---
    ax2 = ax1.twinx()
    ax2.set_ylabel("CBFV (cm/s)", color="blue")
    (line_cbfv,) = ax2.plot(t, cbfv, color="blue", alpha=0.7, label="CBFV")

    if cbfv_kp is not None:
        ax2.scatter(
            t[cbfv_kp],
            cbfv[cbfv_kp],
            color="navy",
            s=30,
            label="CBFV keypoints",
            zorder=5,
        )

    ax2.tick_params(axis="y", labelcolor="blue")

    # --- Combined legend ---
    handles = [line_abp, line_cbfv]
    labels = ["ABP", "CBFV"]

    if abp_kp is not None:
        handles.append(
            plt.Line2D([], [], marker="o", color="darkred", linestyle="None")
        )
        labels.append("ABP keypoints")

    if cbfv_kp is not None:
        handles.append(plt.Line2D([], [], marker="o", color="navy", linestyle="None"))
        labels.append("CBFV keypoints")

    ax1.legend(handles, labels, loc="upper right")

    plt.title("ABP → CBFV with extracted keypoints")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def extract_keypoints(signal, fs):
    """
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
    """
    peaks, _ = find_peaks(signal, distance=int(fs * 0.4))
    troughs, _ = find_peaks(-signal, distance=int(fs * 0.4))

    keypoints = np.sort(np.concatenate([peaks, troughs]))
    return keypoints


if __name__ == "__main__":
    gen = CBFVSignalGenerator(fs=100, duration=10)
    t, abp, cbfv = gen.generate()

    abp_kp = extract_keypoints(abp, fs=100)
    cbfv_kp = extract_keypoints(cbfv, fs=100)

    plot_signals(t, abp, cbfv, abp_kp, cbfv_kp)
