import torch


def extract_keypoint_descriptors(feature_map: torch.Tensor, kp_indices: torch.Tensor):
    """
    Wyciąga deskryptory tylko w miejscach występowania punktów kluczowych.

    Argumenty:
    - feature_map: Pełne wyjście z Enkodera. Kształt: [Batch, Dim, L] (np. 32, 64, 3000)
    - kp_indices: Indeksy szczytów. Kształt: [Batch, N] (np. 32, 30)

    Zwraca:
    - D: Wyselekcjonowane deskryptory. Kształt: [Batch, N, Dim]
    """
    Batch, Dim, L = feature_map.shape
    _, N = kp_indices.shape

    # Przekształcamy indeksy, by pasowały do wymiarów funkcji gather
    # Musimy powielić indeksy dla każdego wymiaru cech (Dim)
    indices_expanded = kp_indices.unsqueeze(1).expand(Batch, Dim, N)

    # Wyciągamy (gather) tylko wartości cech podanymi indeksami
    extracted_D = torch.gather(feature_map, dim=2, index=indices_expanded)

    # Transponujemy, by otrzymać format oczekiwany przez funkcję straty: [Batch, N, Dim]
    extracted_D = extracted_D.transpose(1, 2)

    return extracted_D


def apply_nms_and_select(scores, nms_window=5, threshold=None, top_k=None):
    """
    Aplikuje Non-Maximum Suppression (NMS) oraz wybiera punkty kluczowe.

    Argumenty:
    - scores: Tensor z prawdopodobieństwami z dekodera [Batch, L]
    - nms_window: Rozmiar okna NMS (domyślnie 5 z artykułu)
    - threshold: Próg prawdopodobieństwa (np. 0.5)
    - top_k: Wybór K najlepszych punktów. (Wybierz 'threshold' LUB 'top_k')

    Zwraca:
    - selected_kps: Binarny tensor (1 dla punktu, 0 dla braku) [Batch, L]
    """
    # 1. NMS za pomocą Max Pooling
    # Dodajemy sztuczny wymiar kanału dla MaxPool1d: [Batch, 1, L]
    scores_unsqueeze = scores.unsqueeze(1)

    # Przesuwamy okno o 5. Padding = window // 2 zapewnia zachowanie rozmiaru
    pad = nms_window // 2
    max_pooled = F.max_pool1d(
        scores_unsqueeze, kernel_size=nms_window, stride=1, padding=pad
    )

    # Punkt przeżywa NMS tylko wtedy, gdy jest lokalnym maksimem
    # (jego oryginalna wartość jest równa wartości po MaxPool)
    is_local_max = scores_unsqueeze == max_pooled

    # Zerujemy wszystko, co nie przeszło NMS
    nms_scores = scores_unsqueeze * is_local_max
    nms_scores = nms_scores.squeeze(1)  # Powrót do [Batch, L]

    # 2. Wybór punktów (Selection)
    selected_kps = torch.zeros_like(nms_scores)

    if top_k is not None:
        # Opcja A: Top-K (Wybieramy dokładnie K punktów z najwyższym wynikiem)
        _, topk_indices = torch.topk(nms_scores, k=top_k, dim=1)
        selected_kps.scatter_(1, topk_indices, 1.0)

    elif threshold is not None:
        # Opcja B: Threshold (Wybieramy wszystkie punkty powyżej zadanego progu)
        selected_kps = (nms_scores >= threshold).float()

    else:
        raise ValueError("Musisz podać 'threshold' albo 'top_k'")

    return selected_kps, nms_scores
