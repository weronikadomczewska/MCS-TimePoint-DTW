import torch
import torch.nn.functional as F
import librosa  # pip install librosa
from utils.timepoint_utils import apply_nms_and_select


def align_signals_with_timepoint(model, signal_ABP, signal_CBFV, threshold=0.5):
    """
    Wykonuje pełen proces dopasowania TimePoint DTW na dwóch sygnałach.
    """
    # 1. ZAMROŻENIE MODELU (Bardzo ważne - tryb ewaluacji!)
    model.eval()

    with torch.no_grad():
        # 2. Przejście przez model
        # Uwaga: zakładamy wymiary [Batch=1, Channels=1, Length=3000]
        scores_ABP, _, desc_ABP_dense = model(signal_ABP)
        scores_CBFV, _, desc_CBFV_dense = model(signal_CBFV)

        # 3. Ekstrakcja Punktów Kluczowych (NMS)
        # Używamy funkcji, którą napisaliśmy wcześniej
        mask_ABP, _ = apply_nms_and_select(scores_ABP, threshold=threshold)
        mask_CBFV, _ = apply_nms_and_select(scores_CBFV, threshold=threshold)

        # Wyciągamy same indeksy w czasie (np. punkt nr 50, 150, 240)
        kp_ABP = torch.nonzero(mask_ABP[0]).squeeze()
        kp_CBFV = torch.nonzero(mask_CBFV[0]).squeeze()

        # 4. Pobranie deskryptorów TYLKO dla punktów kluczowych
        # Wycinamy odpowiednie kolumny z gęstej macierzy deskryptorów
        # Otrzymujemy kształty: [N, 256] oraz [M, 256]
        desc_ABP = desc_ABP_dense[0, :, kp_ABP].T
        desc_CBFV = desc_CBFV_dense[0, :, kp_CBFV].T

        # 5. OBLICZENIE MACIERZY KOSZTÓW (Cost Matrix)
        # Ponieważ w treningu używaliśmy podobieństwa kosinusowego,
        # odległością (kosztem) będzie 1 minus to podobieństwo.
        desc_ABP_norm = F.normalize(desc_ABP, p=2, dim=-1)
        desc_CBFV_norm = F.normalize(desc_CBFV, p=2, dim=-1)

        # Mnożenie macierzy: cos_sim = [N, 256] x [256, M] -> [N, M]
        cos_sim_matrix = torch.mm(desc_ABP_norm, desc_CBFV_norm.T)
        cost_matrix = (
            1.0 - cos_sim_matrix
        )  # Zmiana podobieństwa na koszt (0 = idealne, 2 = tragiczne)

        # 6. WYKONANIE ALGORYTMU DTW (za pomocą Librosy)
        cost_matrix_np = cost_matrix.cpu().numpy()

        # DTW zwraca D (macierz skumulowanych kosztów) oraz wp (warping path)
        D, wp = librosa.sequence.dtw(C=cost_matrix_np)

        # Librosa zwraca ścieżkę od końca do początku, więc ją odwracamy
        wp = wp[::-1]

        return wp, kp_ABP.cpu().numpy(), kp_CBFV.cpu().numpy()
