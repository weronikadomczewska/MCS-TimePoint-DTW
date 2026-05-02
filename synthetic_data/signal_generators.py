import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os
import pandas as pd


class ABPGenerator:
    """
    https://doi.org/10.1504/IJMIC.2019.103651
    """

    def __init__(self, fs):
        self.fs = fs
        self.last_params = None

        # base parameters of artificial patient for whole recording
        self.b1_base = np.random.uniform(0.2, 0.25)
        self.b2_base = self.b1_base + np.random.uniform(0.06, 0.1)
        self.b3_base = self.b2_base + np.random.uniform(0.15, 0.25)

        self.a1_base = np.random.uniform(50, 70)
        self.a2_base = np.random.uniform(10, 25)
        self.a3_base = np.random.uniform(5, 10)

        self.c1_base = np.random.uniform(0.02, 0.04) 
        self.c2_base = np.random.uniform(0.06, 0.1)
        self.c3_base = np.random.uniform(0.1, 0.18)


        self.params = [
            (self.a1_base, self.b1_base, self.c1_base),
            (self.a2_base, self.b2_base, self.c2_base),
            (self.a3_base, self.b3_base, self.c3_base),
        ]

    def gaussian(self, t, a, b, c):
        return a * np.exp(-((t - b) ** 2) / (2*c**2))

    def generate_wave(self, t_local):
        # adding micro-fluctuations in between beats

        a1 = self.a1_base * np.random.uniform(0.94, 1.06)
        a2 = self.a2_base * np.random.uniform(0.93, 1.07)
        a3 = self.a3_base * np.random.uniform(0.95, 1.05)

        b1 = self.b1_base * np.random.uniform(0.98, 1.02)
        b2 = self.b2_base * np.random.uniform(0.98, 1.02)
        b3 = self.b3_base * np.random.uniform(0.98, 1.02)

        c1 = self.c1_base * np.random.uniform(0.98, 1.02)
        c2 = self.c2_base * np.random.uniform(0.98, 1.02)
        c3 = self.c3_base * np.random.uniform(0.98, 1.02)

        self.last_params = [(a1, b1, c1), (a2, b2, c2), (a3, b3, c3)]

        g = (
            self.gaussian(t_local, a1, b1, c1)
            + self.gaussian(t_local, a2, b2, c2)
            + self.gaussian(t_local, a3, b3, c3)
        )

        # including linear trend
        d = (g[-1] - g[0]) / (len(t_local) - 1)  # slope

        b = g[0] - d  # intercept

        n = np.arange(len(t_local))

        trend = d * n + b

        G = g + trend

        # return g.astype(np.float32)
        return G.astype(np.float32)
    
    def generate_abp_signal(self, duration_sec, hr_mean=70):
        abp = []
        time = 0.0

        while time < duration_sec:

            hr = np.random.normal(hr_mean, 5)
            hr = np.clip(hr, 50, 120)

            beat_duration = 60.0 / hr
            beat_samples = int(self.fs * beat_duration)

            if beat_samples < 10:
                continue

            t_local = np.linspace(0, 1, beat_samples)

            wave = self.generate_wave(t_local)
            diastolic_offset = np.random.normal(0, 1.5)
            wave += diastolic_offset

            if len(abp) > 0:
                last_val = abp[-1][-1]
                wave = wave - wave[0] + last_val

            abp.append(wave)
            time += beat_duration

        abp = np.concatenate(abp)

        target_len = int(self.fs * duration_sec)
        abp = abp[:target_len]

        t = np.arange(len(abp)) / self.fs

        abp = abp - np.min(abp)
        abp = abp / (np.max(abp) + 1e-8)

        # ensuring physiological range
        mean_bp = np.random.uniform(85, 120)
        pulse_pressure = np.random.uniform(35, 55)

        diastolic = mean_bp - pulse_pressure / 2

        abp = abp * pulse_pressure + diastolic

        return t, abp.astype(np.float32)


class CBFVMaderModel:
    """
    https://link.springer.com/article/10.1007/s10439-014-1220-4
    """

    def __init__(self, a=0.25, b=0.1, c=0.9, M=1.0, V_bas=57.0):
        self.a = a
        self.b = b
        self.c = c
        self.M = M
        self.V_bas = V_bas

        # states
        self.v1 = None
        self.v2 = None

    # Autoregulation curve
    def f_aut(self, p):
        return 2.03e-6 * p**3 - 6.02e-4 * p**2 + 5.94e-2 * p - 1.95

    # d(p)
    def d_of_p(self, p):
        f = self.f_aut(p)

        fs = p

        denom = self.M * self.c * fs - (self.a + self.c) * f

        if np.abs(denom) < 1e-8:
            denom = np.sign(denom) * 1e-8 if denom != 0 else 1e-8

        return (self.b * self.c * f) / denom

    # Initial state initialization
    def initialize(self, p0):
        d = self.d_of_p(p0)

        denom = self.b * self.c + (self.a + self.c) * d

        self.v1 = p0 * (self.b * self.c + self.a * d) / denom
        self.v2 = p0 * (self.b * self.c) / denom

    def step(self, p, dt):
        if self.v1 is None:
            self.initialize(p)

        d = self.d_of_p(p)

        dv1 = (
            -(self.a + self.b + self.c) * self.v1
            + (self.c - d) * self.v2
            + (self.a + self.b) * p
        )

        dv2 = -self.b * self.v1 - d * self.v2 + self.b * p

        self.v1 += dv1 * dt
        self.v2 += dv2 * dt

        Vdyn = self.M * (p - self.v1)
        Vmca = self.V_bas + Vdyn

        return Vmca

    def simulate(self, p_signal, fs):
        dt = 1.0 / fs
        out = np.zeros_like(p_signal, dtype=np.float32)

        self.initialize(p_signal[0])

        for i, p in enumerate(p_signal):
            out[i] = self.step(p, dt)

        return out
        
    def generate_cbfv_from_abp(self, abp, fs):

        n = len(abp)

        window = int(1.0 * fs)
        map_signal = pd.Series(abp).rolling(window, min_periods=1).mean().values

        model = CBFVMaderModel()

        cbfv_slow = model.simulate(map_signal, fs)

        # delay
        delay = int(0.12 * fs)

        cbfv_slow_delayed = np.zeros_like(cbfv_slow)
        cbfv_slow_delayed[delay:] = cbfv_slow[:-delay]
        cbfv_slow_delayed[:delay] = cbfv_slow[0]

        # taking pulsation from ABP
        abp_hp = abp - map_signal

        # amplitude 
        cbfv_pulse = 0.3 * abp_hp

        cbfv = cbfv_slow_delayed + cbfv_pulse

        return cbfv.astype(np.float32)
    
class LogNormalCBFV:
    def __init__(self, fs):
        self.fs = fs

    def lognormal_pulse(self, t, mu, sigma):
        t = np.maximum(t, 1e-4)
        return np.exp(-((np.log(t) - mu) ** 2) / (2 * sigma**2))

    def generate(self, t, beats, abp=None):
        # cbfv = np.zeros_like(t)
        baseline_global = np.random.uniform(55, 60)
        cbfv = np.ones_like(t) * baseline_global

        for i, bt in enumerate(beats):
            next_bt = beats[i + 1] if i < len(beats) - 1 else t[-1]
            duration = next_bt - bt

            mask = (t >= bt) & (t < next_bt)
            if not np.any(mask):
                continue

            t_local = (t[mask] - bt) / duration

            # --- shape ---
            mu = np.random.uniform(-1.0, -0.8)
            sigma = np.random.uniform(0.3, 0.5)

            pulse = self.lognormal_pulse(t_local, mu, sigma)
            
            # normalisation to [-1, 1]
            pulse = pulse - np.mean(pulse)
            pulse = pulse / (np.max(np.abs(pulse)) + 1e-8)

            amplitude = np.random.uniform(8, 12)

            pulse = pulse * amplitude

            #baseline - middle of the range
            baseline = np.random.uniform(55, 60)

            cbfv[mask] = baseline + pulse

        # delay according to ABP
        delay = int(0.1 * self.fs)
        cbfv_delayed = np.zeros_like(cbfv)
        cbfv_delayed[delay:] = cbfv[:-delay]
        cbfv_delayed[:delay] = cbfv[0]

        return cbfv_delayed.astype(np.float32)


def extract_keypoints(signal, fs):
    """
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
    """
    peaks, _ = find_peaks(signal, distance=int(fs * 0.4), height=np.mean(signal) + 0.5 * np.std(signal))
    troughs, _ = find_peaks(-signal, distance=int(fs * 0.4))
    return np.sort(np.concatenate([peaks, troughs]))


def plot_signals(t, abp, cbfv, abp_kp=None, cbfv_kp=None, save_path=None, show=False):
    fig, ax1 = plt.subplots(figsize=(12, 5))

    color_abp = "red"
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("ABP (mmHg)", color=color_abp)

    (line_abp,) = ax1.plot(t, abp, color=color_abp, label="ABP")

    if abp_kp is not None:
        scatter_abp = ax1.scatter(
            t[abp_kp], abp[abp_kp], color="darkred", s=25, label="ABP keypoints"
        )

    ax1.tick_params(axis="y", labelcolor=color_abp)

    ax2 = ax1.twinx()
    color_cbfv = "blue"
    ax2.set_ylabel("CBFV (cm/s)", color=color_cbfv)

    (line_cbfv,) = ax2.plot(t, cbfv, color=color_cbfv, alpha=0.5, label="CBFV")

    if cbfv_kp is not None:
        scatter_cbfv = ax2.scatter(
            t[cbfv_kp], cbfv[cbfv_kp], color="navy", s=25, label="CBFV keypoints"
        )

    ax2.tick_params(axis="y", labelcolor=color_cbfv)

    handles = [line_abp, line_cbfv]
    labels = ["ABP", "CBFV"]

    if abp_kp is not None:
        handles.append(scatter_abp)
        labels.append("ABP keypoints")

    if cbfv_kp is not None:
        handles.append(scatter_cbfv)
        labels.append("CBFV keypoints")

    ax1.legend(handles, labels, loc="upper right")

    plt.title("ABP → CBFV with keypoints")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)

    if show:
        plt.show()

    plt.close()


def plot_signals_stacked(
    t, abp, cbfv, abp_kp=None, cbfv_kp=None, save_path=None, show=True
):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    ax1.plot(t, abp, color="red", alpha=0.8, label="ABP")
    if abp_kp is not None:
        ax1.scatter(
            t[abp_kp], abp[abp_kp], color="darkred", s=25, label="ABP keypoints"
        )

    ax1.set_ylabel("ABP (mmHg)")
    ax1.set_title("Arterial Blood Pressure (ABP)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(t, cbfv, color="blue", alpha=0.6, label="CBFV")
    if cbfv_kp is not None:
        ax2.scatter(
            t[cbfv_kp], cbfv[cbfv_kp], color="navy", s=25, label="CBFV keypoints"
        )

    ax2.set_ylabel("CBFV (cm/s)")
    ax2.set_xlabel("Time (s)")
    ax2.set_title("Cerebral Blood Flow Velocity (CBFV)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()

    if save_path is not None:
        import os

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)

    if show:
        plt.show()

    plt.close()


if __name__ == "__main__":
    fs = 200

    # --- ABP ---
    abp_gen = ABPGenerator(fs=fs)
    t, abp = abp_gen.generate_abp_signal(duration_sec=10)

    peaks, _ = find_peaks(abp, distance=int(fs * 0.4))
    beats = t[peaks]


    cbfv_gen = LogNormalCBFV(fs)
    cbfv = cbfv_gen.generate(t, beats, abp=abp)

    cbfv_gen = CBFVMaderModel(fs)
    cbfv = cbfv_gen.generate_cbfv_from_abp(abp, fs)

    abp_kp = extract_keypoints(abp, fs)
    cbfv_kp = extract_keypoints(cbfv, fs)

    plot_signals_stacked(
        t,
        abp,
        cbfv,
        abp_kp=abp_kp,
        cbfv_kp=cbfv_kp,
    )

