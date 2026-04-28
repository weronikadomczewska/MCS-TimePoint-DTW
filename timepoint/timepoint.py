from encoder import SharedEncoder
from decoders import KeypointDecoder, DescriptorDecoder
import torch.nn as nn
import torch
from utils.timepoint_utils import get_topk_in_original_order, non_maximum_suppression

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class TimePointModel(nn.Module):
    def __init__(self, input_channels=1, encoder_dims=[64,64,128,128], descriptor_dim=256):
        super().__init__()

        self.encoder = SharedEncoder(input_channels, encoder_dims)

        encoder_output_channels = encoder_dims[-1]

        self.detector_head = KeypointDecoder(encoder_output_channels)
        self.descriptor_head = DescriptorDecoder(encoder_output_channels, descriptor_dim)

    def forward(self, x):
        N, C, L = x.shape

        features = self.encoder(x)

        # --- Keypoints ---
        S_scores = self.detector_head(features)   # [B,1,L]
        S_scores = S_scores[:, :, :L]
        S_scores = S_scores.squeeze(1)            # [B,L]

        # --- Descriptors ---
        D = self.descriptor_head(features)        # [B,D,L]
        D = D[:, :, :L]
        D = D.permute(0, 2, 1)                    # [B,L,D]

        return S_scores, D
    
    def get_topk_points(self, x, kp_percent=1, nms_window=5):
        N, C, L = x.shape

        features = self.encoder(x)

        detection_proba = self.detector_head(features)[:, :, :L]
        descriptors = self.descriptor_head(features)[:, :, :L]

        detection_proba = detection_proba.squeeze(1)

        detection_proba = non_maximum_suppression(
            detection_proba, window_size=nms_window
        )

        if kp_percent < 1:
            num_kp = int(kp_percent * L)

            sorted_topk_indices, detection_proba = get_topk_in_original_order(
                detection_proba, detection_proba, K=num_kp
            )
        else:
            sorted_topk_indices = torch.arange(L)

        return sorted_topk_indices, detection_proba, descriptors
    

if __name__ == "__main__":
    import torch

    print("🔍 Testing TimePointModel...")

    try:
        model = TimePointModel()
        print("✅ Model initialized")

        # dummy input [B, C, L]
        x = torch.randn(2, 1, 1000)

        S, D = model(x)

        print("✅ Forward pass OK")
        print("Keypoint map shape:", S.shape)   # [B, L]
        print("Descriptor shape:", D.shape)     # [B, L, D]

    except Exception as e:
        print("❌ ERROR in TimePointModel:")
        import traceback
        traceback.print_exc()