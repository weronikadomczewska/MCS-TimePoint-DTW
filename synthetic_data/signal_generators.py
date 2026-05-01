import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os

class ABPGenerator:
    def __init__(self, fs):
        self.fs = fs
        self.last_params = None

    def gaussian(self, t, a, b, c):
        return a * np.exp(-((t - b) ** 2) / (c ** 2))

    def generate_wave(self, t_local):
        """
        Implements:
        - Section 2.2 (Gaussian model)
        - Section 2.3 (linear trend removal)
        """

        # =========================
        # 1. PARAMETRY GAUSSÓW (UPORZĄDKOWANE)
        # =========================
        b1 = np.random.uniform(0.15, 0.25)
        b2 = b1 + np.random.uniform(0.05, 0.1)
        b3 = b2 + np.random.uniform(0.05, 0.15)

        a1 = np.random.uniform(40, 60)   # systolic
        a2 = np.random.uniform(15, 30)   # reflected
        a3 = np.random.uniform(5, 15)    # diastolic

        c1 = np.random.uniform(0.03, 0.06)
        c2 = np.random.uniform(0.05, 0.1)
        c3 = np.random.uniform(0.08, 0.15)

        self.last_params = [(a1, b1, c1), (a2, b2, c2), (a3, b3, c3)]

        # =========================
        # 2. SEKCJA 2.2 — MODEL G(n)
        # =========================
        G = (
            self.gaussian(t_local, a1, b1, c1)
            + self.gaussian(t_local, a2, b2, c2)
            + self.gaussian(t_local, a3, b3, c3)
        )

        # =========================
        # 3. TREND LINIOWY (d·n + b)
        # =========================
        d = G[-1] - G[0]   # slope
        b = G[0]           # intercept

        trend = d * t_local + b

        # =========================
        # 4. SEKCJA 2.3 — USUNIĘCIE TRENDU
        # =========================
        g = G - trend

        # =========================
        # 5. DODANIE POZIOMU FIZJOLOGICZNEGO
        # =========================
        self.baseline = np.random.uniform(80, 100)
        g = g + self.baseline

        # =========================
        # 6. (opcjonalnie) mały noise
        # =========================
        g += np.random.normal(0, 0.5, len(g))

        return g.astype(np.float32)
    

class Mader2014Model:
    """
    Faithful implementation of:
    Mader, Olufsen, Mahdi (2014)

    - 2-state Voigt system
    - nonlinear autoregulation curve
    - exact d(p) definition
    """

    def __init__(self, a=0.25, b=0.1, c=0.9, M=1.0, V_bas=50.0):
        self.a = a
        self.b = b
        self.c = c
        self.M = M
        self.V_bas = V_bas

        # states
        self.v1 = None
        self.v2 = None

    # =========================
    # Autoregulation curve
    # =========================
    def f_aut(self, p):
        return (
            2.03e-6 * p**3
            - 6.02e-4 * p**2
            + 5.94e-2 * p
            - 1.95
        )

    # =========================
    # Exact d(p)
    # =========================
    def d_of_p(self, p):
        f = self.f_aut(p)
        denom = self.M * self.c * p - (self.a + self.c) * f

        # strictly numerical safeguard (not a model simplification)
        if np.abs(denom) < 1e-8:
            denom = np.sign(denom) * 1e-8 if denom != 0 else 1e-8

        return (self.b * self.c * f) / denom

    # =========================
    # Initialize steady-state
    # =========================
    def initialize(self, p0):
        """
        From paper:
        v1* = v2* = p (steady-state)
        """
        self.v1 = float(p0)
        self.v2 = float(p0)

    # =========================
    # Single integration step
    # =========================
    def step(self, p, dt):
        if self.v1 is None:
            self.initialize(p)

        d = self.d_of_p(p)

        dv1 = (
            -(self.a + self.b + self.c) * self.v1
            + (self.c - d) * self.v2
            + (self.a + self.b) * p
        )

        dv2 = (
            -self.b * self.v1
            - d * self.v2
            + self.b * p
        )

        # explicit Euler (as typically used in such simulations)
        self.v1 += dv1 * dt
        self.v2 += dv2 * dt

        Vdyn = self.M * (p - self.v1)
        Vmca = self.V_bas + Vdyn

        return Vmca

    # =========================
    # Full signal simulation
    # =========================
    def simulate(self, p_signal, fs):
        dt = 1.0 / fs
        print(f"dt: {dt}")
        out = np.zeros_like(p_signal, dtype=np.float32)

        self.initialize(p_signal[0])

        for i, p in enumerate(p_signal):
            out[i] = self.step(p, dt)

        return out


# class TimelineGenerator:
#     def __init__(self, duration, min_hr=60, max_hr=90):
#         self.duration = duration
#         self.min_hr = min_hr
#         self.max_hr = max_hr
#         self.mean_hr = None

#     def generate(self):
#         self.mean_hr = random.randint(self.min_hr, self.max_hr)
#         avg_interval = 60.0 / self.mean_hr

#         times = []
#         t = 0

#         while t < self.duration:
#             interval = avg_interval + np.random.normal(0, 0.05)
#             t += max(interval, 0.3)  # avoid negative/too small intervals
#             times.append(t)

#         return np.array(times)


# class Mader2014Model:
#     """
#     Prawdziwa implementacja modelu z artykułu Mader, Olufsen, Mahdi (2014).
#     Oparta na mechanice ciał Voigta i nieliniowej krzywej autoregulacji.
#     """

#     def __init__(self, a=0.5, b=0.2, c=1.0, M=1.5):
#         # Parametry lepkosprężyste wg artykułu
#         self.a = a  # Sprężystość
#         self.b = b  # Lepkość / Tłumienie
#         self.c = c  # Wzmocnienie krzywej CA
#         self.M = M  # Wpływ pochodnej (odpowiada za overshoot)

#         self.v_dyn = 0.0
#         self.dv_dyn_dt = 0.0
#         self.V_bas = 50.0  # Bazowa prędkość przepływu (cm/s)
#         self.prev_p = None

#         self.params = {"a": a, "b": b, "c": c, "M": M, "V_bas": self.V_bas}

#     def step(self, p_mean, dt):
#         """Krok modelu wykonywany uderzenie po uderzeniu (beat-to-beat)"""
#         if self.prev_p is None:
#             self.prev_p = p_mean

#         # Obliczamy pochodną ciśnienia - wywołuje overshoot przy szybkiej zmianie
#         dp_dt = (p_mean - self.prev_p) / dt

#         # Nieliniowa krzywa autoregulacji z artykułu (wielomian 3. stopnia)
#         p_norm = (p_mean - 100.0) / 50.0
#         ca_curve = 0.5 * (p_norm**3) - 0.1 * p_norm

#         # Równanie różniczkowe z artykułu Mader 2014
#         acc = (
#             -self.a * self.v_dyn
#             - self.b * self.dv_dyn_dt
#             + self.c * ca_curve
#             + self.M * dp_dt
#         )

#         # Całkowanie numeryczne metodą Eulera
#         self.v_dyn += self.dv_dyn_dt * dt
#         self.dv_dyn_dt += acc * dt
#         self.prev_p = p_mean

#         return self.V_bas + self.v_dyn


# class ABPGenerator:
#     def __init__(self, fs, params=None):
#         self.fs = fs
#         self.params = params
#         self.last_params = None

#     def gaussian(self, t, a, b, c):
#         return a * np.exp(-((t - b) ** 2) / (2 * c**2))

#     def generate_wave(self, t_local):
#         if self.params is None:
#             params = [
#                 (
#                     np.random.uniform(30, 50),
#                     np.random.uniform(0.2, 0.3),
#                     np.random.uniform(0.04, 0.08),
#                 ),
#                 (
#                     np.random.uniform(10, 25),
#                     np.random.uniform(0.3, 0.4),
#                     np.random.uniform(0.05, 0.1),
#                 ),
#                 (
#                     np.random.uniform(5, 15),
#                     np.random.uniform(0.4, 0.55),
#                     np.random.uniform(0.08, 0.15),
#                 ),
#             ]
#         else:
#             params = self.params

#         self.last_params = params

#         wave = np.zeros_like(t_local)
#         for a, b, c in params:
#             wave += self.gaussian(t_local, a, b, c)

#         return wave

# class CBFVSignalGenerator:
#     def __init__(
#         self,
#         fs=100,
#         duration=10,
#         hr_range=None,
#         noise_std=None,
#         ca_params=None,
#         abp_params=None,
#     ):
#         self.fs = fs
#         self.duration = duration
#         self.hr_range = hr_range if hr_range else (55, 110)
#         self.noise_std = noise_std

#         self.abp_gen = ABPGenerator(fs, params=abp_params)
#         self.ca_params = ca_params if ca_params else {}
#         self.timeline = TimelineGenerator(
#             duration=duration, min_hr=self.hr_range[0], max_hr=self.hr_range[1]
#         )

#     def asymmetric_pulse(self, t_local, mfv_beat):
#         """
#         Generuje asymetryczną falę TCD (Log-Normal).
#         Strome narastanie skurczowe i powolny spadek rozkurczowy.
#         """
#         # Unikamy log(0)
#         t_shifted = t_local + 0.05

#         # Funkcja Log-Normal tworząca fizjologiczny profil uderzenia
#         wave = (1.0 / t_shifted) * np.exp(
#             -((np.log(t_shifted) - (-1.0)) ** 2) / (2 * 0.4**2)
#         )
#         wave_norm = (wave - wave.min()) / (wave.max() - wave.min() + 1e-6)

#         # Dopasowujemy falę do wyliczonego średniego przepływu
#         baseline = mfv_beat * 0.65  # Typowy przepływ rozkurczowy
#         amplitude = mfv_beat * 1.6  # Typowy pik skurczowy

#         return baseline + wave_norm * (amplitude - baseline)

#     def generate(self):
#         t = np.linspace(0, self.duration, int(self.duration * self.fs))
#         abp = np.zeros_like(t)
#         cbfv = np.zeros_like(t)

#         beats = self.timeline.generate()
#         mader_model = Mader2014Model(**self.ca_params)

#         for i, bt in enumerate(beats):
#             next_bt = beats[i + 1] if i < len(beats) - 1 else self.duration
#             dt = max(next_bt - bt, 1e-3)
#             mask = (t >= bt) & (t < next_bt)

#             # --- ZABEZPIECZENIE: Pomijamy uderzenie, jeśli nie ma dla niego próbek czasu ---
#             if not np.any(mask):
#                 continue

#             # 1. Generujemy falę ciśnienia (ABP)
#             t_local = (t[mask] - bt) / dt
#             wave_abp = self.abp_gen.generate_wave(t_local)

#             # for i, bt in enumerate(beats):
#             #     next_bt = beats[i + 1] if i < len(beats) - 1 else self.duration
#             #     dt = max(next_bt - bt, 1e-3)
#             #     mask = (t >= bt) & (t < next_bt)

#             #     # 1. Generujemy falę ciśnienia (ABP)
#             #     t_local = (t[mask] - bt) / dt
#             #     wave_abp = self.abp_gen.generate_wave(t_local)

#             # Skok ciśnienia (symulacja ortostatyczna w połowie czasu)
#             baseline_p = 90 if bt < self.duration / 2 else 60
#             abp_current = wave_abp + baseline_p
#             abp[mask] += abp_current

#             # 2. Wyliczamy Średnie Ciśnienie Tętnicze (MAP) uderzenia
#             map_beat = np.mean(abp_current)

#             # 3. Mader 2014 Model - wylicza docelowy średni przepływ mózgowy (MFV) na podstawie MAP
#             mfv_beat = mader_model.step(map_beat, dt)

#             # 4. Rysujemy asymetryczną, fizjologiczną falę CBFV opartą na wyliczonym MFV
#             delay = int(0.12 * self.fs)  # Opóźnienie pulsu krew-mózg (~120ms)

#             # Przesunięcie masek, by fala przepływu była delikatnie opóźniona względem ciśnienia
#             cbfv_wave = self.asymmetric_pulse(t_local, mfv_beat)

#             start_idx = np.where(mask)[0][0] + delay
#             end_idx = start_idx + len(cbfv_wave)

#             if end_idx <= len(cbfv):
#                 cbfv[start_idx:end_idx] = cbfv_wave
#             else:
#                 remaining = len(cbfv) - start_idx

#                 if remaining <= 0:
#                     break

#                     # nic już nie wstawiamy
#                     # return t, abp, cbfv, metadata  # albo break/continue – zależnie od struktury

#                 length = min(len(cbfv_wave), remaining)

#                 cbfv[start_idx:start_idx + length] = cbfv_wave[:length]
#                 # Obcięcie dla ostatniego uderzenia poza wykresem
#                 # cbfv[start_idx:] = cbfv_wave[: len(cbfv) - start_idx]

#         # Wypełnienie początkowego opóźnienia
#         cbfv[cbfv == 0] = mader_model.V_bas

#         # Szum pomiarowy
#         abp_noise, cbfv_noise = (
#             self.noise_std
#             if self.noise_std
#             else (np.random.uniform(1.0, 3.0), np.random.uniform(1.5, 4.0))
#         )
#         abp += np.random.normal(0, abp_noise, len(abp))
#         cbfv += np.random.normal(0, cbfv_noise, len(cbfv))
#         cbfv = np.clip(cbfv, 20, 150)

#         metadata = {
#             "abp_params": self.abp_gen.last_params,
#             "ca_params": mader_model.params,
#             "mean_hr": self.timeline.mean_hr,
#             "noise": {"abp": abp_noise, "cbfv": cbfv_noise},
#         }

#         return t, abp, cbfv, metadata


def extract_keypoints(signal, fs):
    """
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
    """
    peaks, _ = find_peaks(signal, distance=int(fs * 0.4))
    troughs, _ = find_peaks(-signal, distance=int(fs * 0.4))
    return np.sort(np.concatenate([peaks, troughs]))


# def plot_signals(t, abp, cbfv, abp_kp=None, cbfv_kp=None, save_path=None, show=False):
#     fig, ax1 = plt.subplots(figsize=(12, 5))

#     color_abp = "red"
#     ax1.set_xlabel("Time (s)")
#     ax1.set_ylabel("ABP (mmHg)", color=color_abp)

#     (line_abp,) = ax1.plot(t, abp, color=color_abp, label="ABP")

#     if abp_kp is not None:
#         scatter_abp = ax1.scatter(
#             t[abp_kp], abp[abp_kp], color="darkred", s=25, label="ABP keypoints"
#         )

#     ax1.tick_params(axis="y", labelcolor=color_abp)

#     ax2 = ax1.twinx()
#     color_cbfv = "blue"
#     ax2.set_ylabel("CBFV (cm/s)", color=color_cbfv)

#     (line_cbfv,) = ax2.plot(t, cbfv, color=color_cbfv, alpha=0.5, label="CBFV")

#     if cbfv_kp is not None:
#         scatter_cbfv = ax2.scatter(
#             t[cbfv_kp], cbfv[cbfv_kp], color="navy", s=25, label="CBFV keypoints"
#         )

#     ax2.tick_params(axis="y", labelcolor=color_cbfv)

#     handles = [line_abp, line_cbfv]
#     labels = ["ABP", "CBFV"]

#     if abp_kp is not None:
#         handles.append(scatter_abp)
#         labels.append("ABP keypoints")

#     if cbfv_kp is not None:
#         handles.append(scatter_cbfv)
#         labels.append("CBFV keypoints")

#     ax1.legend(handles, labels, loc="upper right")

#     plt.title("ABP → CBFV with keypoints")
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()

#     if save_path is not None:
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#         plt.savefig(save_path, dpi=150)

#     if show:
#         plt.show()

#     plt.close()  # 🔥 VERY IMPORTANT (prevents memory leak)


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


# def generate_dataset(
#     n_samples=5000, save_dir="synthetic_dataset", plot_dir="plots", fs=100, duration=10
# ):
#     os.makedirs(save_dir, exist_ok=True)

#     gen = CBFVSignalGenerator(fs=fs, duration=duration)

#     for i in range(n_samples):
#         t, abp, cbfv, metadata = gen.generate()

#         abp_kp = extract_keypoints(abp, fs)
#         cbfv_kp = extract_keypoints(cbfv, fs)

#         np.savez_compressed(
#             os.path.join(save_dir, f"sample_{i}.npz"),
#             t=t,
#             abp=abp,
#             cbfv=cbfv,
#             abp_kp=abp_kp,
#             cbfv_kp=cbfv_kp,
#             metadata=metadata,
#         )

#         # --- SAVE PLOT (optional: only some) ---
#         if i < 50:
#             plot_signals(
#                 t,
#                 abp,
#                 cbfv,
#                 abp_kp,
#                 cbfv_kp,
#                 save_path=os.path.join(plot_dir, f"sample_{i}.png"),
#                 show=False,
#             )

#             plot_signals_stacked(
#                 t,
#                 abp,
#                 cbfv,
#                 abp_kp,
#                 cbfv_kp,
#                 save_path=os.path.join(plot_dir, f"sample_{i}_stacked.png"),
#                 show=False,
#             )

#         if i % 100 == 0:
#             print(f"Generated {i}/{n_samples}")

#     return metadata


# # =========================
# # MAIN
# # =========================
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


if __name__ == "__main__":
    fs = 200
    duration = 60 # sekundy

    # --- generuj ABP ---
    t = np.linspace(0, duration, fs * duration)

    generator = ABPGenerator(fs)

    abp = []
    for i in range(duration):
        t_local = np.linspace(0, 1, fs)
        cycle = generator.generate_wave(t_local)
        abp.append(cycle)

    abp = np.concatenate(abp)
    t = np.arange(len(abp)) / fs


    # --- generuj CBFV ---
    model = Mader2014Model()
    cbfv = model.simulate(abp, fs)

    lag = np.argmax(np.correlate(cbfv - cbfv.mean(), abp - abp.mean(), mode="full"))
    lag = lag - len(abp) + 1

    print("Lag (s):", lag / fs)

    

    # --- wykres ---
    plot_signals_stacked(t, abp, cbfv)


