# import torch
import torch.nn as nn
from layers import ConvBlock1D, WTConvBlock1D

# https://github.com/BGU-CS-VIL/TimePoint/blob/main/TimePoint/models/wtconv1d.py


class SharedEncoder(nn.Module):
    def __init__(
        self, input_channels=1, dims=[64, 64, 128, 128], stride=2, wt_levels=[3, 3, 3]
    ):
        super().__init__()
        self.stride = stride
        self.layer1 = ConvBlock1D(input_channels, dims[0], stride=1, padding="same")
        self.layer2 = WTConvBlock1D(
            dims[0], dims[1], stride=self.stride, wt_levels=wt_levels[0]
        )  # stride=2 to downsample
        self.layer3 = WTConvBlock1D(
            dims[1], dims[2], stride=self.stride, wt_levels=wt_levels[1]
        )
        self.layer4 = WTConvBlock1D(
            dims[2], dims[3], stride=self.stride, wt_levels=wt_levels[2]
        )

    def forward(self, x):
        # Input x: [N, C, L]
        x = self.layer1(x)  # [N, base_channels, L]
        x = self.layer2(x)  # [N, base_channels, L/2]
        x = self.layer3(x)  # [N, base_channels*2, L/4]
        x = self.layer4(x)  # [N, base_channels*2, L/8]
        return x  # Feature map of size L/8


if __name__ == "__main__":
    import torch

    print("🔍 Testing SharedEncoder...")

    try:
        model = SharedEncoder()
        print("✅ Encoder initialized")

        # dummy input: [batch, channels, length]
        x = torch.randn(2, 1, 1000)

        y = model(x)
        print("✅ Forward pass OK")
        print("Output shape:", y.shape)

    except Exception as e:
        print("ERROR during encoder test:")
