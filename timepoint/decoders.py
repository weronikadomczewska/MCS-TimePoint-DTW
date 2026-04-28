import torch
import torch.nn as nn
import torch.nn.functional as F


class KeypointDecoder(nn.Module):
    """
    Detector Head for predicting keypoint probability map.
    """
    def __init__(self, input_channels, cell_size=8):
        super().__init__()
        self.cell_size = cell_size
        self.conv = nn.Conv1d(input_channels, cell_size + 1, kernel_size=1)  # Output channels: cell_size + 1 (dustbin)

    def forward(self, x):
        # x: [N, C, L/8]
        N, C, Lc = x.shape  # Lc = L/8
        x = self.conv(x)     # [N, cell_size + 1, Lc]

        # Reshape to [N, cell_size + 1, Lc]
        # Softmax over the cell_size + 1 channels (including dustbin)
        x = F.softmax(x, dim=1)
        # Remove dustbin (last channel)
        x = x[:, :-1, :]  # [N, cell_size, Lc]

        # Reshape to [N, 1, L]
        x = x.permute(0, 2, 1).reshape(N, 1, Lc * self.cell_size)

        return x  # Keypoint probability map of size [N, 1, L]


class DescriptorDecoder(nn.Module):
    """
    Descriptor Head for generating feature descriptors.
    """
    def __init__(self, input_channels, descriptor_dim=256):
        super().__init__()
        self.conv = nn.Conv1d(input_channels, descriptor_dim, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=8, mode='linear', align_corners=False)

    def forward(self, x):
        # x: [N, C, L/8]
        x = self.conv(x)  # [N, descriptor_dim, L/8]
        x = self.upsample(x)  # [N, descriptor_dim, L]
        #x = F.normalize(x, p=2, dim=1)  # L2 norm along channel dimension, now performed at loss.
        return x  

if __name__ == "__main__":
    import torch

    print("🔍 Testing decoders...")

    try:
        # simulate encoder output: [B, C, L/8]
        B, C, L = 2, 128, 1000
        x = torch.randn(B, C, L // 8)

        print("Input shape:", x.shape)

        # --- Keypoint decoder ---
        kp_decoder = KeypointDecoder(C)
        kp_out = kp_decoder(x)

        print("✅ KeypointDecoder OK")
        print("Keypoint output shape:", kp_out.shape)  # expected [B, 1, L]

        # --- Descriptor decoder ---
        desc_decoder = DescriptorDecoder(C)
        desc_out = desc_decoder(x)

        print("✅ DescriptorDecoder OK")
        print("Descriptor output shape:", desc_out.shape)  # expected [B, D, L]

    except Exception as e:
        print("❌ ERROR in decoders:")
        import traceback
        traceback.print_exc()