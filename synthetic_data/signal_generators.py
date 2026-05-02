import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os
import pandas as pd


class TimelineGenerator:
    def __init__(self, duration, min_hr=60, max_hr=90):
        self.duration = duration
        self.min_hr = min_hr
        self.max_hr = max_hr
        self.mean_hr = None

    def generate(self):
        self.mean_hr = random.randint(self.min_hr, self.max_hr)
        avg_interval = 60.0 / self.mean_hr

        times = []
        t = 0

        while t < self.duration:
            interval = avg_interval + np.random.normal(0, 0.05)
            t += max(interval, 0.3)  # avoid negative/too small intervals
            times.append(t)

        return np.array(times)


class ABPGenerator:
    """
    https://doi.org/10.1504/IJMIC.2019.103651
    """

    def __init__(self, fs):
        self.fs = fs
        self.last_params = None

        # base parameters of artificial patient for whole recording
        self.b1_base = np.random.uniform(0.15, 0.25)
        self.b2_base = self.b1_base + np.random.uniform(0.05, 0.1)
        self.b3_base = self.b2_base + np.random.uniform(0.05, 0.15)

        self.a1_base = np.random.uniform(30, 59)
        self.a2_base = np.random.uniform(10, 25)
        self.a3_base = np.random.uniform(5, 15)

        self.c1_base = np.random.uniform(0.03, 0.06)
        self.c2_base = np.random.uniform(0.05, 0.1)
        self.c3_base = np.random.uniform(0.08, 0.15)

        self.params = [
            (self.a1_base, self.b1_base, self.c1_base),
            (self.a2_base, self.b2_base, self.c2_base),
            (self.a3_base, self.b3_base, self.c3_base),
        ]

    def gaussian(self, t, a, b, c):
        return a * np.exp(-((t - b) ** 2) / (c**2))

    def generate_wave(self, t_local):
        # adding micro-fluctuations in between beats

        a1 = self.a1_base * np.random.uniform(0.97, 1.03)
        a2 = self.a2_base * np.random.uniform(0.97, 1.03)
        a3 = self.a3_base * np.random.uniform(0.97, 1.03)

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

        return G.astype(np.float32)


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

        self.abp_gen = ABPGenerator(fs)
        self.ca_params = ca_params if ca_params else {}
        self.timeline = TimelineGenerator(
            duration=duration, min_hr=self.hr_range[0], max_hr=self.hr_range[1]
        )

    def asymmetric_pulse(self, t_local, mfv_beat):
        """
        Generuje asymetryczną falę TCD (Log-Normal).
        Strome narastanie skurczowe i powolny spadek rozkurczowy.
        """
        # Unikamy log(0)
        t_shifted = t_local + 0.05

        # Funkcja Log-Normal tworząca fizjologiczny profil uderzenia
        wave = (1.0 / t_shifted) * np.exp(
            -((np.log(t_shifted) - (-1.0)) ** 2) / (2 * 0.4**2)
        )
        wave_norm = (wave - wave.min()) / (wave.max() - wave.min() + 1e-6)

        # Dopasowujemy falę do wyliczonego średniego przepływu
        baseline = mfv_beat * 0.65  # Typowy przepływ rozkurczowy
        amplitude = mfv_beat * 1.6  # Typowy pik skurczowy

        return baseline + wave_norm * (amplitude - baseline)

    def generate(self):
        t = np.linspace(0, self.duration, int(self.duration * self.fs))
        abp = np.zeros_like(t)
        cbfv = np.zeros_like(t)

        beats = self.timeline.generate()
        ca_model = CBFVMaderModel(**self.ca_params)

        for i, bt in enumerate(beats):
            next_bt = beats[i + 1] if i < len(beats) - 1 else self.duration
            dt = max(next_bt - bt, 1e-3)
            mask = (t >= bt) & (t < next_bt)

            if not np.any(mask):
                continue

            # 1. Generujemy falę ciśnienia (ABP)
            t_local = (t[mask] - bt) / dt
            wave_abp = self.abp_gen.generate_wave(t_local)

            # Skok ciśnienia (symulacja ortostatyczna w połowie czasu)
            baseline_p = 90 if bt < self.duration / 2 else 60
            abp_current = wave_abp + baseline_p
            abp[mask] += abp_current

            # 2. Wyliczamy Średnie Ciśnienie Tętnicze (MAP) uderzenia
            map_beat = np.mean(abp_current)

            # 3. Mader 2014 Model - wylicza docelowy średni przepływ mózgowy (MFV) na podstawie MAP
            mfv_beat = ca_model.step(map_beat, dt)

            # 4. Rysujemy asymetryczną, fizjologiczną falę CBFV opartą na wyliczonym MFV
            delay = int(0.12 * self.fs)  # Opóźnienie pulsu krew-mózg (~120ms)

            # Przesunięcie masek, by fala przepływu była delikatnie opóźniona względem ciśnienia
            cbfv_wave = self.asymmetric_pulse(t_local, mfv_beat)

            start_idx = np.where(mask)[0][0] + delay
            end_idx = start_idx + len(cbfv_wave)

            if start_idx >= len(cbfv):
                continue

            valid_len = len(cbfv) - start_idx

            if valid_len <= 0:
                continue

            if not np.any(mask):
                continue

            if end_idx <= len(cbfv):
                cbfv[start_idx:end_idx] += cbfv_wave
            else:
                # Obcięcie dla ostatniego uderzenia poza wykresem
                cbfv[start_idx:] = cbfv_wave[: len(cbfv) - start_idx]

        # Wypełnienie początkowego opóźnienia
        # cbfv[cbfv == 0] = ca_model.V_bas
        mask_zero = cbfv == 0
        cbfv[mask_zero] = np.nan

        # interpolacja
        cbfv = pd.Series(cbfv).interpolate().bfill().ffill().values

        # Szum pomiarowy
        abp_noise, cbfv_noise = (
            self.noise_std
            if self.noise_std
            else (np.random.uniform(1.0, 3.0), np.random.uniform(1.5, 4.0))
        )
        abp += np.random.normal(0, abp_noise, len(abp))
        cbfv += np.random.normal(0, cbfv_noise, len(cbfv))
        cbfv = np.clip(cbfv, 20, 150)

        metadata = {
            "abp_params": self.abp_gen.last_params,
            # "ca_params": ca_model.params,
            "mean_hr": self.timeline.mean_hr,
            "noise": {"abp": abp_noise, "cbfv": cbfv_noise},
        }

        return t, abp, cbfv, metadata


def extract_keypoints(signal, fs):
    """
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
    """
    peaks, _ = find_peaks(signal, distance=int(fs * 0.4))
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


def plot_signals_stacked_with_trend(
    t, abp, cbfv, fs, window_sec=2, save_path=None, show=True
):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

    # Obliczenie rozmiaru okna w próbkach (np. 10 sekund * 200 Hz = 2000 próbek)
    window_samples = int(window_sec * fs)

    # Wyliczenie średniej kroczącej (trendu)
    abp_trend = pd.Series(abp).rolling(window=window_samples, center=True).mean()
    cbfv_trend = pd.Series(cbfv).rolling(window=window_samples, center=True).mean()

    # --- Wykres 1: ABP ---
    # Rysujemy surowy sygnał w tle (jasny, przy 5 minutach to będzie gęsty "pasek")
    ax1.plot(t, abp, color="red", alpha=0.15, label="Surowe ABP (200 Hz)")
    # Rysujemy wyraźną linię trendu
    ax1.plot(
        t,
        abp_trend,
        color="darkred",
        linewidth=2.5,
        label=f"Trend ABP (okno {window_sec}s)",
    )

    ax1.set_ylabel("ABP (mmHg)")
    ax1.set_title(f"Arterial Blood Pressure (ABP) w 5-minutowym zapisie")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")

    # --- Wykres 2: CBFV ---
    ax2.plot(t, cbfv, color="blue", alpha=0.15, label="Surowe CBFV (200 Hz)")
    ax2.plot(
        t,
        cbfv_trend,
        color="navy",
        linewidth=2.5,
        label=f"Trend CBFV (okno {window_sec}s)",
    )

    ax2.set_ylabel("CBFV (cm/s)")
    ax2.set_xlabel("Czas [s]")
    ax2.set_title(f"Cerebral Blood Flow Velocity (CBFV) w 5-minutowym zapisie")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)

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


def generate_dataset(
    n_samples=5000, save_dir="synthetic_dataset", plot_dir="plots", fs=100, duration=10
):
    os.makedirs(save_dir, exist_ok=True)

    gen = CBFVSignalGenerator(fs=fs, duration=duration)

    for i in range(n_samples):
        t, abp, cbfv, metadata = gen.generate()

        abp_kp = extract_keypoints(abp, fs)
        cbfv_kp = extract_keypoints(cbfv, fs)

        np.savez_compressed(
            os.path.join(save_dir, f"sample_{i}.npz"),
            t=t,
            abp=abp,
            cbfv=cbfv,
            abp_kp=abp_kp,
            cbfv_kp=cbfv_kp,
            metadata=metadata,
        )

        # --- SAVE PLOT (optional: only some) ---
        if i < 50:
            plot_signals(
                t,
                abp,
                cbfv,
                abp_kp,
                cbfv_kp,
                save_path=os.path.join(plot_dir, f"sample_{i}.png"),
                show=False,
            )

            plot_signals_stacked(
                t,
                abp,
                cbfv,
                abp_kp,
                cbfv_kp,
                save_path=os.path.join(plot_dir, f"sample_{i}_stacked.png"),
                show=False,
            )

        if i % 100 == 0:
            print(f"Generated {i}/{n_samples}")

    return metadata


# if __name__ == "__main__":
#     # test
#     # gen = CBFVSignalGenerator()
#     # t, abp, cbfv, metadata = gen.generate()
#     # plot_signals(t, abp, cbfv,
#     #              extract_keypoints(abp, 100),
#     #              extract_keypoints(cbfv, 100))

#     data = np.load("synthetic_dataset/sample_4.npz", allow_pickle=True)

#     t = data["t"]
#     abp = data["abp"]
#     cbfv = data["cbfv"]
#     abp_kp = data["abp_kp"]
#     cbfv_kp = data["cbfv_kp"]

#     plot_signals(
#         t, abp, cbfv, extract_keypoints(abp, 100), extract_keypoints(cbfv, 100)
#     )

#     print(abp.shape)
#     print(abp_kp[:10])
#     print(metadata)
