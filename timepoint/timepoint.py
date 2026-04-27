from encoder import SharedEncoder
from decoders import KeypointDecoder, DescriptorDecoder
import torch.nn as nn


class TimePointModel(nn.Module):
    def __init__(self, in_channels=1, d_enc=256, d_desc=256):
        super().__init__()
        # 1. Rdzeń: Współdzielony Enkoder (zmniejsza L do L')
        self.shared_encoder = SharedEncoder(in_channels=in_channels, d_enc=d_enc)

        # 2. Ramię pierwsze: Dekoder Punktów Kluczowych (Zwraca do L)
        self.kp_decoder = KeypointDecoder(d_enc=d_enc, cell_size=8)

        # 3. Ramię drugie: Dekoder Deskryptorów (Zwraca do L)
        self.desc_decoder = DescriptorDecoder(
            d_enc=d_enc, d_desc=d_desc, upsample_factor=8
        )

    def forward(self, x):
        """Przepływ sygnału (np. zaszumionego ABP) przez całą sieć."""
        # Krok 1: Ekstrakcja cech bazowych
        F_features = self.shared_encoder(x)

        # Krok 2: Detekcja, gdzie są punkty kluczowe
        kp_scores, kp_logits = self.kp_decoder(F_features)

        # Krok 3: Wygenerowanie "odcisków palców" dla każdego punktu w czasie
        F_desc = self.desc_decoder(F_features)

        return kp_scores, kp_logits, F_desc
