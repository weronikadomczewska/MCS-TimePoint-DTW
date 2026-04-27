import numpy as np
import random
import matplotlib.pyplot as plt


class TimelineGenerator:
    """
    Generates shared timeline for ABP and CBFV signals.
    """

    def __init__(self, duration: int, min_hr: int = 60, max_hr: int = 90):
        self.duration = duration  # Duration of the signal in seconds
        self.min_hr = min_hr
        self.max_hr = max_hr

    def _get_random_average_hr(self) -> int:
        """Generates a random average heart rate (HR) value within the specified range."""
        return random.randint(self.min_hr, self.max_hr)

    def _generate_hrv_noise(self, num_beats: int, mean_hr: int, hrv_std: float = 0.05):
        """
        Generates heart rate variability (HRV) noise for a given number of heartbeats.
        Draws from a normal distribution centered around the average interval between beats.
        """
        avg_interval = 60.0 / mean_hr
        hrv_noise = np.random.normal(0, hrv_std, num_beats)
        return avg_interval + hrv_noise

    def generate_heartbeats_timeline(self, mean_hr: int) -> np.ndarray:
        """
        Generates a timeline of heartbeats based on the specified duration and average heart rate.
        """
        num_beats = int((self.duration / 60.0) * mean_hr) + 5  # adding spare signals
        hrv_noise = self._generate_hrv_noise(num_beats, mean_hr)
        beat_times = np.cumsum(hrv_noise)
        return beat_times[beat_times < self.duration]


class ViscoelasticCAModel:
    """
    Cerebral autoregulation model based on:
    Mader et al. (2014)

    Maps ABP → mean CBFV using viscoelastic dynamics.
    """

    def __init__(self, dt=0.01):
        # model parameters (reasonable defaults from paper)
        self.a = 0.25
        self.b = 0.1
        self.c = 0.9
        self.M = 1.0

        # pressure-dependent term (simplified for now)
        self.d = 0.05

        # states
        self.v1 = 0.0
        self.v2 = 0.0

        self.dt = dt

        # baseline flow
        self.V_bas = 50.0

    def update_beat(self, current_abp, beat_dt):
        """
        Integrates model over one heartbeat interval.
        """

        steps = max(1, int(beat_dt / self.dt))
        dt = beat_dt / steps

        for _ in range(steps):
            p = current_abp

            dv1 = (
                -(self.a + self.b + self.c) * self.v1
                + (self.c - self.d) * self.v2
                + (self.a + self.b) * p
            )
            dv2 = -self.b * self.v1 - self.d * self.v2 + self.b * p

            self.v1 += dt * dv1
            self.v2 += dt * dv2

        V_dyn = self.M * (current_abp - self.v1)

        return self.V_bas + V_dyn


class PhysiologicalSignalGenerator:
    def __init__(
        self, fs: int = 100, duration: int = 30, min_hr: int = 60, max_hr: int = 90
    ):
        self.fs = fs
        self.duration = duration
        self.min_hr = min_hr
        self.max_hr = max_hr
        self.timeline_gen = TimelineGenerator(
            duration=duration, min_hr=min_hr, max_hr=max_hr
        )

    def _generate_heartbeats_timeline(
        self, mean_hr: int = 60, min_hr: int = 60, max_hr: int = 90
    ) -> np.ndarray:
        """Generates a timeline of heartbeats (with HRV) based on the average heart rate."""
        return self.timeline_gen.generate_heartbeats_timeline(mean_hr)

    def gaussian_component(self, t, a, b, c):
        """
        Computes a single Gaussian kernel function.
        Wektor 't' jest już w sekundach, więc nie dzielimy przez fs!
        Formula: a * exp(- ((t - b)^2) / (2 * c^2))
        """
        return a * np.exp(-((t - b) ** 2) / (2 * c**2))

    def lognorm_component(self, t, alpha, beta, gamma):
        """
        Funkcja Log-Normalna dokładnie według Równania (4) z artykułu.
        t: czas lokalny w sekundach (odpowiednik n/fs)
        alpha, beta, gamma: parametry modelu
        """
        wave = np.zeros_like(t)
        valid = t > 1e-6  # Zabezpieczenie przed dzieleniem przez 0

        # Współczynnik z przodu: alpha / (sqrt(2*pi) * gamma * t)
        coeff = alpha / (np.sqrt(2 * np.pi) * gamma * t[valid])

        # Wnętrze exponenty: - (ln(t / beta))^2 / (2 * gamma^2)
        exponent = -((np.log(t[valid] / beta)) ** 2) / (2 * gamma**2)

        wave[valid] = coeff * np.exp(exponent)
        return wave

    def get_dynamic_cbfv_wave(self, t_local, peak_amp):
        """
        Zwraca TYLKO falę pulsującą CBFV (bez linii bazowej).
        """
        lognorm_params = [
            (peak_amp * 0.71, -1.2, 0.25),
            (peak_amp * 0.29, -0.8, 0.35),
        ]

        cbfv_wave = np.zeros_like(t_local)
        for a, b, c in lognorm_params:
            cbfv_wave += self.lognorm_component(t_local, a, b, c)

        return cbfv_wave

    def get_abp_wave(self, t_local, k=0.0, b_base=80.0):
        """
        https://ieeexplore.ieee.org/document/8926447
        f(n, theta) + B(n, psi).
        k - slope for the single wave
        b_base - ABP at the beginning of the wave
        """
        # 1. B(n, psi): Obliczamy linię bazową jako trend liniowy
        baseline = k * t_local + b_base

        # 2. f(n, theta): Generujemy model morfologiczny (np. 3 lub 4 funkcje Gaussa)
        gaussian_params = [
            (40, 0.25, 0.05),  # a1, b1, c1 (Szczyt skurczowy)
            (20, 0.35, 0.07),  # a2, b2, c2 (Wcięcie dykrotyczne)
            (10, 0.45, 0.1),  # a3, b3, c3 (Fala rozkurczowa)
        ]

        f_theta = np.zeros_like(t_local)
        for a, b, c in gaussian_params:
            f_theta += self.gaussian_component(t_local, a, b, c)

        # 3. Zwracamy kompletną falę: f(n, theta) + B(n, psi)
        return f_theta + baseline

    def get_cbfv_wave(self, t_local):
        """Returns a single CBFV wave based on Log-Normal components."""
        # Parameters for typical 80/40 cm/s flow wave
        lognorm_params = [
            (25, -1.2, 0.25),  # Rapid rise
            (10, -0.8, 0.35),  # Slower decay
        ]

        cbfv_wave = np.zeros_like(t_local)
        for a, b, c in lognorm_params:
            cbfv_wave += self.lognorm_component(t_local, a, b, c)
        return cbfv_wave

    def generate_window(self):
        heartbeats = self._generate_heartbeats_timeline()
        total_samples = int(self.duration * self.fs)
        t = np.linspace(0, self.duration, total_samples, endpoint=False)

        # 1. Inicjalizujemy puste tablice TYLKO na pulsację
        abp_pulsatile = np.zeros_like(t)
        cbfv_pulsatile = np.zeros_like(t)

        # 2. Inicjalizujemy ciągłe linie bazowe (domyślne wartości na krawędzie)
        abp_baseline = np.full_like(t, 80.0)
        cbfv_baseline = np.full_like(t, 40.0)

        abp_keypoints = []
        cbfv_keypoints = []
        delay_cbfv = 0.1

        ca_model = ViscoelasticCAModel()

        for i, beat_time in enumerate(heartbeats):
            # Określamy, do kiedy trwa to uderzenie
            next_beat = (
                heartbeats[i + 1] if i < (len(heartbeats) - 1) else self.duration
            )
            dt = next_beat - beat_time

            # --- A. GENEROWANIE ABP ---
            if beat_time < 20.0:
                current_abp_diastolic = 80.0
                abp_systolic_amp = 40.0
            else:
                current_abp_diastolic = 60.0
                abp_systolic_amp = 30.0

            # Wypełniamy linię bazową ABP tylko dla czasu trwania tego uderzenia
            abp_baseline[(t >= beat_time) & (t < next_beat)] = current_abp_diastolic

            # Dodajemy falkę ABP (wymuszamy b_base=0.0, bo dodaliśmy bazę wyżej!)
            abp_wave = self.get_abp_wave(t - beat_time, k=0.0, b_base=0.0)
            abp_pulsatile += abp_wave
            abp_keypoints.append(int((beat_time + 0.25) * self.fs))

            # --- B. NIELINIOWE WYLICZENIE CBFV ---
            current_mean_abp = current_abp_diastolic + (abp_systolic_amp / 3.0)
            cbfv_mean = ca_model.update_beat(current_mean_abp, dt)

            current_cbfv_base = cbfv_mean * 0.6
            cbfv_peak_amp = (cbfv_mean - current_cbfv_base) * 2.5

            # Wypełniamy linię bazową CBFV (z uwzględnieniem fizjologicznego opóźnienia!)
            cbfv_baseline[
                (t >= beat_time + delay_cbfv) & (t < next_beat + delay_cbfv)
            ] = current_cbfv_base

            # Dodajemy falkę CBFV (tylko pulsacja)
            cbfv_wave = self.get_dynamic_cbfv_wave(
                t - (beat_time + delay_cbfv), peak_amp=cbfv_peak_amp
            )
            cbfv_pulsatile += cbfv_wave
            cbfv_keypoints.append(int((beat_time + delay_cbfv + 0.20) * self.fs))

        # --- C. OSTATECZNE SUMOWANIE ---
        # Sygnał = linia bazowa + pulsacja + szum (teraz nic nie ma prawa zniknąć!)
        abp_signal = abp_baseline + abp_pulsatile + np.random.normal(0, 1.5, len(t))
        cbfv_signal = cbfv_baseline + cbfv_pulsatile + np.random.normal(0, 2.0, len(t))

        max_idx = len(abp_signal)
        abp_keypoints_safe = [kp for kp in abp_keypoints if kp < max_idx]
        cbfv_keypoints_safe = [kp for kp in cbfv_keypoints if kp < max_idx]

        return (
            abp_signal,
            np.array(abp_keypoints_safe),
            cbfv_signal,
            np.array(cbfv_keypoints_safe),
        )

    def visualize_signals(self, abp_signal, abp_keypoints, cbfv_signal, cbfv_keypoints):
        """Visualizes the generated ABP and CBFV signals along with their keypoints."""

        time_axis = np.arange(len(abp_signal)) / self.fs

        plt.figure(figsize=(12, 6))

        plt.subplot(2, 1, 1)
        plt.plot(time_axis, abp_signal, label="ABP Signal", color="red")
        plt.scatter(
            abp_keypoints / self.fs,
            abp_signal[abp_keypoints],
            color="blue",
            label="ABP Keypoints",
        )
        plt.title("Simulated Arterial Blood Pressure (ABP) Signal")
        plt.xlabel("Time (s)")
        plt.ylabel("Pressure (mmHg)")
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(time_axis, cbfv_signal, label="CBFV Signal", color="green")
        plt.scatter(
            cbfv_keypoints / self.fs,
            cbfv_signal[cbfv_keypoints],
            color="orange",
            label="CBFV Keypoints",
        )
        plt.title("Simulated Cerebral Blood Flow Velocity (CBFV) Signal")
        plt.xlabel("Time (s)")
        plt.ylabel("Velocity (cm/s)")
        plt.legend()

        plt.tight_layout()
        plt.show()

    def visualize_signals_combined(
        self, abp_signal, abp_keypoints, cbfv_signal, cbfv_keypoints
    ):
        """Visualizes the generated ABP and CBFV signals on a single plot with dual Y-axes."""

        time_axis = np.arange(len(abp_signal)) / self.fs

        fig, ax1 = plt.subplots(figsize=(14, 6))

        # --- Oś Y1: Sygnał ABP ---
        color_abp = "tab:red"
        ax1.set_xlabel("Time (s)", fontsize=12)
        ax1.set_ylabel("Pressure (mmHg)", color=color_abp, fontsize=12)

        # Rysowanie linii ciągłej i keypointów dla ABP
        (line_abp,) = ax1.plot(
            time_axis, abp_signal, label="ABP Signal", color=color_abp, alpha=0.7
        )
        scatter_abp = ax1.scatter(
            abp_keypoints / self.fs,
            abp_signal[abp_keypoints],
            color="darkred",
            label="ABP Keypoints",
            zorder=5,
        )
        ax1.tick_params(axis="y", labelcolor=color_abp)

        # --- Oś Y2: Sygnał CBFV ---
        # Tworzymy drugą oś Y współdzielącą tę samą oś X
        ax2 = ax1.twinx()

        color_cbfv = "tab:blue"
        ax2.set_ylabel("Velocity (cm/s)", color=color_cbfv, fontsize=12)

        # Rysowanie linii ciągłej i keypointów dla CBFV
        (line_cbfv,) = ax2.plot(
            time_axis, cbfv_signal, label="CBFV Signal", color=color_cbfv, alpha=0.7
        )
        scatter_cbfv = ax2.scatter(
            cbfv_keypoints / self.fs,
            cbfv_signal[cbfv_keypoints],
            color="darkblue",
            label="CBFV Keypoints",
            zorder=5,
        )
        ax2.tick_params(axis="y", labelcolor=color_cbfv)

        # --- Wspólna Legenda ---
        # Zbieramy wszystkie obiekty, aby wrzucić je do jednej czytelnej legendy
        lines = [line_abp, scatter_abp, line_cbfv, scatter_cbfv]
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="upper right")

        plt.title(
            "Simulated ABP and CBFV Signals with Orthostatic Stress Adaptation",
            fontsize=14,
        )

        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    signal_gen = PhysiologicalSignalGenerator(fs=100, duration=10, min_hr=60, max_hr=90)
    abp_signal, abp_keypoints, cbfv_signal, cbfv_keypoints = (
        signal_gen.generate_window()
    )
    signal_gen.visualize_signals(abp_signal, abp_keypoints, cbfv_signal, cbfv_keypoints)
    signal_gen.visualize_signals_combined(
        abp_signal, abp_keypoints, cbfv_signal, cbfv_keypoints
    )

print("ABP length:", len(abp_signal))
print("CBFV length:", len(cbfv_signal))
