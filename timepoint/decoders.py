import torch
import torch.nn as nn


class KeypointDecoder(nn.Module):
    def __init__(self, d_enc=256, cell_size=8):
        super().__init__()
        self.cell_size = cell_size

        # Warstwa konwolucyjna mapująca D_enc -> 8 (cell_size)
        # Używamy kernel_size=3 i padding=1, aby długość L' pozostała bez zmian
        self.conv = nn.Conv1d(
            in_channels=d_enc, out_channels=cell_size, kernel_size=3, padding=1
        )

    def forward(self, F_features):
        """
        Argumenty:
        - F_features: Mapa cech z enkodera. Kształt: [Batch, D_enc, L']

        Zwraca:
        - scores: Prawdopodobieństwa po sigmoidzie [Batch, L]
        - logits: Wartości przed sigmoidem (do funkcji straty) [Batch, L]
        """
        B, D, L_prime = F_features.shape

        # 1. Redukcja cech: R^{D_enc x L'} -> R^{8 x L'}
        x = self.conv(F_features)  # Kształt: [Batch, 8, L']

        # 2. Reshape do oryginalnej długości sygnału L (gdzie L = L' * 8)
        # Analogiczne do 1D PixelShuffle.
        # Najpierw zmieniamy wymiary na [Batch, L', 8], aby kolejne 8 komórek czasu
        # ułożyło się po kolei dla każdego punktu przestrzennego L'.
        x = x.permute(0, 2, 1).contiguous()

        # Następnie spłaszczamy wymiary, by uzyskać sygnał 1D o długości L
        logits = x.view(B, -1)  # Kształt: [Batch, L]

        # 3. Aktywacja Sigmoid (prawdopodobieństwa s_t w przedziale [0, 1])
        scores = torch.sigmoid(logits)

        return scores, logits


class DescriptorDecoder(nn.Module):
    def __init__(self, d_enc=256, d_desc=256, upsample_factor=8):
        super().__init__()

        # 1. Warstwa konwolucyjna mapująca cechy z enkodera (F) do przestrzeni deskryptorów
        # Używamy padding=1, aby nie zmienić długości sygnału (zostaje L')
        self.conv = nn.Conv1d(
            in_channels=d_enc, out_channels=d_desc, kernel_size=3, padding=1
        )

        # 2. Operator upsamplingu (L' -> L)
        # Ponieważ enkoder zmniejszył sygnał 8-krotnie, musimy go powiększyć 8-krotnie.
        # Używamy interpolacji liniowej (standard dla sygnałów 1D).
        self.upsample = nn.Upsample(
            scale_factor=upsample_factor, mode="linear", align_corners=False
        )

    def forward(self, F_features):
        """
        Argumenty:
        - F_features: Mapa cech z Shared Encoder. Kształt: [Batch, D_enc, L']

        Zwraca:
        - F_desc: Znormalizowana, gęsta macierz deskryptorów. Kształt: [Batch, D_desc, L]
        """
        # Mapowanie cech [Batch, D_enc, L'] -> [Batch, D_desc, L']
        x = self.conv(F_features)

        # Upsampling [Batch, D_desc, L'] -> [Batch, D_desc, L]
        x = self.upsample(x)

        # L2 Normalization (aby wektory leżały na "jednostkowej hipersferze")
        # Normalizujemy wzdłuż wymiaru D_desc (dim=1), tak aby wektor cech
        # dla KAŻDEGO punktu w czasie 't' miał długość euklidesową równą 1.
        F_desc = F.normalize(x, p=2, dim=1)

        return F_desc
