from dtaidistance import dtw
from scipy.stats import zscore
import numpy as np
import matplotlib.pyplot as plt

def analyze_dtw_morphological_delay(abp_signal, cbfv_signal, fs=100):
    # Normalisation - different units
    abp_norm = zscore(abp_signal)
    cbfv_norm = zscore(cbfv_signal)
    
    # 2. Sakoe-Chiba window DTW
    window_size = int(0.5 * fs) 
    
    distance, paths = dtw.warping_paths(abp_norm, cbfv_norm, window=window_size)
    best_path = dtw.best_path(paths)
    
    path_diffs = []
    for i, j in best_path:

        diff_seconds = (j - i) / fs
        path_diffs.append(diff_seconds)
        
    path_diffs = np.array(path_diffs)
    
    mean_dtw_delay = np.mean(path_diffs)
    std_dtw_delay = np.std(path_diffs)
    
    return path_diffs, mean_dtw_delay, std_dtw_delay

def plot_delay_trend(window_times, mean_delays, std_delays=None, event_times=None):
    """
    Rysuje wykres trendu opóźnienia ABP-CBFV w trakcie dłuższego badania.
    
    :param window_times: Czas środka każdego analizowanego okna (oś X)
    :param mean_delays: Średnie opóźnienie DTW dla danego okna w sekundach
    :param std_delays: (Opcjonalnie) Odchylenie standardowe opóźnienia
    :param event_times: (Opcjonalnie) Lista momentów (w sekundach), kiedy pacjent zmieniał pozycję (np. wstawał)
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(window_times, mean_delays, marker='o', color='purple', linewidth=2, label="Średnie przesunięcie (Delay)")

    if std_delays is not None:
        ax.fill_between(window_times, 
                        np.array(mean_delays) - np.array(std_delays), 
                        np.array(mean_delays) + np.array(std_delays), 
                        color='purple', alpha=0.2, label="Zmienność (Std Dev)")

    # position change events
    if event_times:
        for et in event_times:
            ax.axvline(x=et, color='black', linestyle=':', alpha=0.7)
        # Trik na dodanie tylko jednej legendy dla wszystkich linii pionowych
        ax.axvline(x=event_times[0], color='black', linestyle=':', alpha=0.7, label="Zmiana pozycji (Sit-to-stand)")

    ax.set_title("Autoregulation assesment", fontsize=14)
    ax.set_xlabel("Duration (s)", fontsize=12)
    ax.set_ylabel("Delay (s)", fontsize=12)
    
    ax.axhline(0, color='gray', linewidth=1)
    
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_dtw_alignment(abp_window, cbfv_window, path, fs=100, title="Wizualizacja przesunięcia DTW (Warping Path)"):
    """
    Rysuje dwa sygnały i szare linie łączące odpowiadające sobie punkty według ścieżki DTW.
    """
    time_axis = np.arange(len(abp_window)) / fs

    # Normalizacja Z-score tylko do celów wizualizacji (żeby nałożyły się na jednej osi Y)
    abp_norm = zscore(abp_window)
    cbfv_norm = zscore(cbfv_window)

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(time_axis, abp_norm, label='ABP (Wejście)', color='red', linewidth=2)
    ax.plot(time_axis, cbfv_norm, label='CBFV (Wyjście)', color='blue', linewidth=2)

    for i, j in path[::3]:
        ax.plot([time_axis[i], time_axis[j]], [abp_norm[i], cbfv_norm[j]], 
                color='gray', linestyle='--', alpha=0.4)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Czas (s)", fontsize=12)
    ax.set_ylabel("Znormalizowana Amplituda (Z-score)", fontsize=12)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()