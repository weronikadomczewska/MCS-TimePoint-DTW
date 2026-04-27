import torch
from libcpab import Cpab


class TimePointCPABWarper:
    """
    Klasa generująca zaburzenia czasu na podstawie 1D Difeomorfizmów (CPAB).
    Zgodna z architekturą samonadzorowaną opisaną w artykule TimePoint.
    """

    def __init__(self, tess_size=5, sigma=0.5, device="cpu"):
        """
        Argumenty:
        - tess_size: Liczba segmentów komórek CPAB. Artykuł sugeruje wartości rzędu 4-8.
                     Im więcej segmentów, tym bardziej złożone "powyginanie" czasu.
        - sigma: Skala wariancji dla losowania parametrów theta.
                 Kontroluje "agresywność" zniekształcenia (0.0 to brak zmian).
        """
        self.device = device
        self.sigma = sigma

        # Inicjalizacja rdzenia libcpab (wymiar 1D dla szeregów czasowych)
        self.cpab = Cpab(tess_size=tess_size, backend="pytorch", device=device, ndim=1)

    def generate_warped_pair(self, signal: torch.Tensor):
        """
        Pobiera oryginalny sygnał X i zwraca jego zaburzoną wersję X'
        oraz mapowanie indeksów czasu.

        Argumenty:
        - signal: Tensor 1D o kształcie [L] reprezentujący pojedynczy sygnał.

        Zwraca:
        - warped_signal: Sygnał po transformacji czasu X' [L].
        - time_mapping: Wektor mówiący, skąd wzięła się dana próbka.
        """
        L = signal.shape[-1]

        # 1. Przygotowanie wejścia do formatu wymaganego przez libcpab [Batch, Channels, Length]
        if signal.dim() == 1:
            signal_input = signal.unsqueeze(0).unsqueeze(0).to(self.device)
        else:
            signal_input = signal.to(self.device)

        # 2. Losowanie nieliniowej deformacji (theta)
        # sample_transformation zwraca losowe wartości z rozkładu normalnego.
        # Mnożymy przez naszą sigmę, by kontrolować siłę deformacji.
        theta = self.cpab.sample_transformation(1) * self.sigma

        # 3. Zastosowanie deformacji na sygnale (Wygenerowanie X')
        warped_signal = self.cpab.transform_data(signal_input, theta, outsize=(L,))

        # 4. Magia TimePoint: Generowanie mapowania siatki czasu (Grid Mapping)
        # Tworzymy znormalizowaną siatkę czasu (od 0.0 do 1.0)
        grid = self.cpab.uniform_meshgrid([L]).to(self.device)

        # Aplikujemy to samo przekształcenie theta na siatkę czasu.
        # Zwróci to informację, gdzie fizycznie przesunęły się punkty bazowe.
        warped_grid = self.cpab.transform_grid(grid, theta)

        # Wracamy do formatu 1D dla łatwiejszej manipulacji na zewnątrz
        return warped_signal.squeeze(), warped_grid.squeeze(), theta

    def get_matching_indices(
        self, warped_grid: torch.Tensor, original_indices: torch.Tensor, L: int
    ):
        """
        Funkcja pomocnicza dla Zadania 2 / 6:
        Przelicza oryginalne indeksy punktów kluczowych na nowe indeksy po deformacji.
        """
        # warped_grid jest w skali [0, 1]. Mnożymy przez L-1, aby dostać indeksy tablicy [0, L-1]
        grid_indices = warped_grid * (L - 1)

        mapped_indices = []
        for idx in original_indices:
            # Szukamy, gdzie w nowej, zdeformowanej siatce wylądował nasz oryginalny indeks
            # Używamy najprostszego zaokrąglenia do najbliższego sąsiada
            distances = torch.abs(grid_indices - float(idx))
            new_idx = torch.argmin(distances)
            mapped_indices.append(new_idx.item())

        return torch.tensor(mapped_indices, dtype=torch.long)
