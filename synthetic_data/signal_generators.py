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

    def lognorm_component(self, t, a, b, c):
        """
        Bezpieczna funkcja rozkładu log-normalnego dla szeregów czasowych.
        t: czas lokalny (t - beat_time)
        a: amplituda (wysokość piku)
        b: parametr mu (przesunięcie w skali logarytmicznej)
        c: parametr sigma (szerokość i asymetria fali)
        """
        # Przygotowujemy pustą falę samych zer o odpowiednim rozmiarze
        wave = np.zeros_like(t)

        # Wybieramy TYLKO te indeksy, gdzie czas jest dodatni (po uderzeniu serca)
        # Używamy drobnego marginesu 1e-6, żeby uniknąć np.log(0)
        valid = t > 1e-6

        # Obliczamy falę Log-Normal tylko na wycinku z dodatnim czasem.
        # Wzór: a * exp( - (ln(t) - b)^2 / (2 * c^2) )
        wave[valid] = a * np.exp(-((np.log(t[valid]) - b) ** 2) / (2 * c**2))

        return wave

    def get_abp_wave(self, t_local):
        """Returns a single ABP wave based on a mixture on Gaussian components."""
        # Parameters for typical 120/80 mmHg pressure wave
        gaussian_params = [
            (40, 0.25, 0.05),  # Systolic peak
            (20, 0.35, 0.07),  # Dicrotic notch
            (10, 0.45, 0.1),  # Diastolic wave
        ]

        abp_wave = np.zeros_like(t_local)
        for a, b, c in gaussian_params:
            abp_wave += self.gaussian_component(t_local, a, b, c)
        return abp_wave

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
        """Main function that generates the ABP and CBFV signals along with their keypoints."""
        # 1. Wektor ZDARZEŃ (ok. 29 elementów - same sekundy uderzeń)
        heartbeats = self._generate_heartbeats_timeline()

        # 2. Ciągła OŚ CZASU (3000 elementów dla 30s przy 100Hz)
        total_samples = self.duration * self.fs
        t = np.linspace(0, self.duration, total_samples, endpoint=False)

        # 3. Sygnały inicjujemy na bazie osi czasu (mają po 3000 elementów)
        abp_signal = np.full_like(t, 80.0)  # Baseline (diastolic)
        cbfv_signal = np.full_like(t, 40.0)  # Baseline

        abp_keypoints = []
        cbfv_keypoints = []

        delay_cbfv = 0.1  # Delay of blood flow

        # 4. Dodajemy fale Gaussa używając gęstej osi 't'
        for beat_time in heartbeats:
            abp_wave = self.get_abp_wave(t - beat_time)
            abp_signal += abp_wave
            abp_keypoints.append(int((beat_time + 0.25) * self.fs))  # Systolic peak

            cbfv_wave = self.get_cbfv_wave(t - (beat_time + delay_cbfv))
            cbfv_signal += cbfv_wave
            cbfv_keypoints.append(
                int((beat_time + delay_cbfv + 0.20) * self.fs)
            )  # CBFV peak

        # 5. Szum pomiarowy dla całych 3000 elementów
        abp_signal += np.random.normal(0, 1.5, len(abp_signal))
        cbfv_signal += np.random.normal(0, 2.0, len(cbfv_signal))

        # 6. ZABEZPIECZENIE (Rozwiązuje IndexError przy rysowaniu)
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


# if __name__ == "__main__":
#     signal_gen = PhysiologicalSignalGenerator(fs=100, duration=10, min_hr=60, max_hr=90)
#     abp_signal, abp_keypoints, cbfv_signal, cbfv_keypoints = (
#         signal_gen.generate_window()
#     )
#     signal_gen.visualize_signals(abp_signal, abp_keypoints, cbfv_signal, cbfv_keypoints)
