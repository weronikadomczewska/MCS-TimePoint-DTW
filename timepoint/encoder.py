# import torch
import torch.nn as nn
from layers import ConvBlock1D, WTConvBlock1D

# https://github.com/BGU-CS-VIL/TimePoint/blob/main/TimePoint/models/wtconv1d.py


class SharedEncoder(nn.Module):
    def __init__(self, input_channels=1, dims=[64, 64, 128, 128]):
        super().__init__()
        self.input_layer = ConvBlock1D(input_channels, dims[0], stride=1)
        self.layer2 = ConvBlock1D(dims[0], dims[1], stride=2)
        self.layer3 = ConvBlock1D(dims[1], dims[2], stride=2)
        self.output_layer = ConvBlock1D(dims[2], dims[3], stride=2)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.output_layer(x)
        return x
