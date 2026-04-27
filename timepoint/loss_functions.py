import torch
import torch.nn as nn
import torch.nn.functional as F

# https://github.com/BGU-CS-VIL/TimePoint/blob/main/TimePoint/models/wtconv1d.py


class TimePointKeypointLoss(nn.Module):
    """
    Funkcja straty dla detekcji punktów kluczowych.
    Klasyfikacja binarna dla każdego punktu w czasie (obecność szczytu: 1, brak: 0).
    """

    def __init__(self, use_logits=True):
        super(TimePointKeypointLoss, self).__init__()

        # ASPEKT INŻYNIERSKI (Stabilność numeryczna):
        # Wzór mówi o prawdopodobieństwie 's_t', co sugerowałoby użycie nn.BCELoss().
        # Jednak w PyTorchu dobrą praktyką jest, by ostatnia warstwa sieci NIE miała
        # aktywacji Sigmoid, lecz by sieć wypluwała surowe liczby (logits).
        # nn.BCEWithLogitsLoss łączy Sigmoid i BCELoss w jedną, stabilną matematycznie operację,
        # co zapobiega błędom typu NaN (Not a Number) podczas treningu.

        if use_logits:
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = nn.BCELoss()

    def forward(self, S_pred: torch.Tensor, Y_true: torch.Tensor):
        """
        Argumenty:
        - S_pred: Przewidywania modelu. Kształt: [Batch, L] (np. 32, 3000)
        - Y_true: Ground-truth. Wektor zer i jedynek. Kształt: [Batch, L]

        Zwraca:
        - Skalar: uśredniona strata BCE.
        """
        # Upewniamy się, że ground-truth jest wektorem zmiennoprzecinkowym (wymóg BCE)
        Y_true = Y_true.float()

        # Wyliczenie straty i automatyczne uśrednienie (-1/L * sum(...))
        loss = self.loss_fn(S_pred, Y_true)

        return loss


class TimePointDescriptorLoss(nn.Module):
    def __init__(self, mp=1.0, mn=0.1):
        super().__init__()
        self.mp = mp
        self.mn = mn

    def forward(self, D: torch.Tensor, D_prime: torch.Tensor, match_mask: torch.Tensor):
        """
        D: Deskryptory z oryginalnego sygnału [Batch, N, Dim]
        D_prime: Deskryptory z zaburzonego sygnału [Batch, N, Dim]
        match_mask: Funkcja wskaźnikowa 1_G [Batch, N, N]
        """
        # cos(Di, D'j) - Podobieństwo kosinusowe
        D_norm = F.normalize(D, p=2, dim=-1)
        D_prime_norm = F.normalize(D_prime, p=2, dim=-1)
        cos_sim = torch.bmm(D_norm, D_prime_norm.transpose(1, 2))  # [Batch, N, N]

        # match_mask to matematyczne 1_G((i,j))
        # max(0, mp - cos(Di, D'j))^2
        pos_loss = match_mask * (torch.relu(self.mp - cos_sim) ** 2)

        # (1 - 1_G((i,j))) * max(0, cos(Di, D'j) - mn)^2
        neg_mask = 1.0 - match_mask
        neg_loss = neg_mask * (torch.relu(cos_sim - self.mn) ** 2)

        # Uśrednianie po N^2 (1/N^2 ze wzoru)
        total_loss = pos_loss + neg_loss
        return total_loss.mean()


class TimePointOverallLoss(nn.Module):
    """
    Całkowita funkcja straty (Overall Loss) dla modelu TimePoint.
    Łączy błąd detekcji (dla sygnału oryginalnego i zaburzonego) oraz błąd dopasowania deskryptorów.
    """

    def __init__(self, mp=1.0, mn=0.1, lambda_desc=1.0):
        super().__init__()
        # Inicjalizujemy nasze dwie wcześniejsze funkcje straty
        self.kp_loss_fn = TimePointKeypointLoss(use_logits=True)
        self.desc_loss_fn = TimePointDescriptorLoss(mp=mp, mn=mn)

        # ASPEKT INŻYNIERSKI: Mimo że we wzorze nie ma jawnych wag,
        # w praktyce głębokiego uczenia zawsze warto dodać mnożnik (lambda).
        # Czasami strata na deskryptorach (L_desc) dominuje błąd detekcji (L_kp),
        # co psuje trening. 'lambda_desc' pozwala Ci to balansować w Zadaniu 6.
        self.lambda_desc = lambda_desc

    def forward(
        self,
        S_logits: torch.Tensor,
        Y_true: torch.Tensor,
        S_prime_logits: torch.Tensor,
        Y_prime_true: torch.Tensor,
        D: torch.Tensor,
        D_prime: torch.Tensor,
        match_mask: torch.Tensor,
    ):
        """
        Argumenty:
        - S_logits, Y_true: Logits predykcji i ground-truth dla ORYGINALNEGO sygnału.
        - S_prime_logits, Y_prime_true: Logits i ground-truth dla ZABURZONEGO sygnału.
        - D, D_prime: Wyselekcjonowane deskryptory (kształt: [Batch, N, Dim]).
        - match_mask: Maska dopasowań punktów kluczowych 1_G.

        Zwraca:
        - total_loss: Skalar, ostateczny błąd dla optymalizatora.
        - loss_dict: Słownik z rozbiciem na składowe (przydatne do wykresów w TensorBoard/WandB).
        """

        # 1. Błąd detekcji punktów kluczowych w sygnale oryginalnym (X)
        loss_kp_orig = self.kp_loss_fn(S_logits, Y_true)

        # 2. Błąd detekcji punktów kluczowych w sygnale zaburzonym (X')
        loss_kp_warped = self.kp_loss_fn(S_prime_logits, Y_prime_true)

        # 3. Błąd deskryptorów (podobieństwo kosinusowe na marginesach)
        loss_desc = self.desc_loss_fn(D, D_prime, match_mask)

        # 4. Suma (Zgodnie ze wzorem z artykułu)
        total_loss = loss_kp_orig + loss_kp_warped + (self.lambda_desc * loss_desc)

        # Zwracamy też poszczególne straty, żebyś mogła je monitorować w czasie uczenia!
        loss_dict = {
            "total_loss": total_loss.item(),
            "loss_kp_orig": loss_kp_orig.item(),
            "loss_kp_warped": loss_kp_warped.item(),
            "loss_desc": loss_desc.item(),
        }

        return total_loss, loss_dict
