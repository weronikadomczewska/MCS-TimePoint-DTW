import torch
import torch.nn.functional as F 
import numpy as np 

def get_topk_in_original_order(X_desc, X_probas, K):
    """
    Get the descriptors of the top K keypoints from X_keypoints without changing their original order.

    Args:
        X_desc (torch.Tensor): The descriptors associated with keypoints, shape [N, C, L].
        X_keypoints (torch.Tensor): Tensor of keypoint probabilities, shape [N, L].
        K (int): Number of top elements to select per sample.

    Returns:
        X_topk (torch.Tensor): Tensor containing the descriptors of the top K keypoints per sample, shape [N, C, K].
    """
    N, C, L = X_desc.shape
    assert X_probas.shape == (N, L), "X_keypoints must have shape (N, L)"
    
    device = X_probas.device
    if K >= L:
        return X_probas, X_desc

    # Get the indices of the top K values per sample
    topk_values, topk_indices = torch.topk(X_probas, K, dim=1)
    # topk_indices: shape [N, K]
    
    # Sort the indices per sample to maintain original order
    sorted_topk_indices, _ = torch.sort(topk_indices, dim=1)
    # sorted_topk_indices: shape [N, K]
    
    # Expand indices for gathering
    indices_expanded = sorted_topk_indices.unsqueeze(1).expand(-1, C, -1)  # Shape: [N, C, K]
    
    # Gather descriptors along the L dimension (time steps)
    X_topk = torch.gather(X_desc, dim=2, index=indices_expanded)  # Shape: [N, C, K]
    
    return sorted_topk_indices, X_topk




def non_maximum_suppression(detection_prob, window_size=7):
    """
    Apply non-maximum suppression to the detection map.

    Args:
        detection_map: Tensor of shape [N, L].
        window_size: Size of the window for NMS.
        threshold: Detection threshold.

    Returns:
        keypoints: Tensor of shape [N, L], boolean mask of keypoints after NMS.
    """
    # NMS
    if isinstance(detection_prob, np.ndarray):
        detection_prob = torch.from_numpy(detection_prob)
    # prepare input
    N, L = detection_prob.shape
    # (1, L' < L)
    pooled, pooled_idx = F.max_pool1d(detection_prob, kernel_size=window_size,
                                      stride=window_size, padding=window_size // 2,
                                      return_indices=True)

    # Squeeze dim=1 from proba, make our life easier if only one sample
    if len(pooled.shape) == 3:
        detection_prob = detection_prob.squeeze()
        pooled_idx = pooled_idx.squeeze()
    # pooled_idx array of ints, turn to bool
    zero_out = torch.ones_like(detection_prob)
    for i in range(N):
        zero_out[i, pooled_idx[i]] = 0
    # zero out everything but max pooled
    detection_prob[zero_out.type(torch.bool)] = 0
    return detection_prob





# import torch


# def extract_keypoint_descriptors(feature_map: torch.Tensor, kp_indices: torch.Tensor):
#     """
#     Wyciąga deskryptory tylko w miejscach występowania punktów kluczowych.

#     Argumenty:
#     - feature_map: Pełne wyjście z Enkodera. Kształt: [Batch, Dim, L] (np. 32, 64, 3000)
#     - kp_indices: Indeksy szczytów. Kształt: [Batch, N] (np. 32, 30)

#     Zwraca:
#     - D: Wyselekcjonowane deskryptory. Kształt: [Batch, N, Dim]
#     """
#     Batch, Dim, L = feature_map.shape
#     _, N = kp_indices.shape

#     # Przekształcamy indeksy, by pasowały do wymiarów funkcji gather
#     # Musimy powielić indeksy dla każdego wymiaru cech (Dim)
#     indices_expanded = kp_indices.unsqueeze(1).expand(Batch, Dim, N)

#     # Wyciągamy (gather) tylko wartości cech podanymi indeksami
#     extracted_D = torch.gather(feature_map, dim=2, index=indices_expanded)

#     # Transponujemy, by otrzymać format oczekiwany przez funkcję straty: [Batch, N, Dim]
#     extracted_D = extracted_D.transpose(1, 2)

#     return extracted_D


# def apply_nms_and_select(scores, nms_window=5, threshold=None, top_k=None):
#     """
#     Aplikuje Non-Maximum Suppression (NMS) oraz wybiera punkty kluczowe.

#     Argumenty:
#     - scores: Tensor z prawdopodobieństwami z dekodera [Batch, L]
#     - nms_window: Rozmiar okna NMS (domyślnie 5 z artykułu)
#     - threshold: Próg prawdopodobieństwa (np. 0.5)
#     - top_k: Wybór K najlepszych punktów. (Wybierz 'threshold' LUB 'top_k')

#     Zwraca:
#     - selected_kps: Binarny tensor (1 dla punktu, 0 dla braku) [Batch, L]
#     """
#     # 1. NMS za pomocą Max Pooling
#     # Dodajemy sztuczny wymiar kanału dla MaxPool1d: [Batch, 1, L]
#     scores_unsqueeze = scores.unsqueeze(1)

#     # Przesuwamy okno o 5. Padding = window // 2 zapewnia zachowanie rozmiaru
#     pad = nms_window // 2
#     max_pooled = F.max_pool1d(
#         scores_unsqueeze, kernel_size=nms_window, stride=1, padding=pad
#     )

#     # Punkt przeżywa NMS tylko wtedy, gdy jest lokalnym maksimem
#     # (jego oryginalna wartość jest równa wartości po MaxPool)
#     is_local_max = scores_unsqueeze == max_pooled

#     # Zerujemy wszystko, co nie przeszło NMS
#     nms_scores = scores_unsqueeze * is_local_max
#     nms_scores = nms_scores.squeeze(1)  # Powrót do [Batch, L]

#     # 2. Wybór punktów (Selection)
#     selected_kps = torch.zeros_like(nms_scores)

#     if top_k is not None:
#         # Opcja A: Top-K (Wybieramy dokładnie K punktów z najwyższym wynikiem)
#         _, topk_indices = torch.topk(nms_scores, k=top_k, dim=1)
#         selected_kps.scatter_(1, topk_indices, 1.0)

#     elif threshold is not None:
#         # Opcja B: Threshold (Wybieramy wszystkie punkty powyżej zadanego progu)
#         selected_kps = (nms_scores >= threshold).float()

#     else:
#         raise ValueError("Musisz podać 'threshold' albo 'top_k'")

#     return selected_kps, nms_scores
