import torch
import torch.nn as nn
import torch.nn.functional as F

# https://github.com/BGU-CS-VIL/TimePoint/blob/main/TimePoint/models/wtconv1d.py


class TimePointKeypointLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.BCELoss()

    def forward(self, S_logits: torch.Tensor, Y_true: torch.Tensor):
        Y_true = Y_true.float()
        S_prob = torch.sigmoid(S_logits)

        loss = self.loss_fn(S_prob, Y_true)
        return loss


class TimePointDescriptorLoss(nn.Module):
    def __init__(self, mp=1.0, mn=0.1):
        """
        mp: positive margin (target similarity)
        mn: negative margin (maximum allowed similarity)
        """
        super().__init__()
        self.mp = mp
        self.mn = mn

    def forward(self, D, D_prime, match_mask):
        """
        D:         [B, N, D]  - descriptors (original)
        D_prime:   [B, N, D]  - descriptors (warped)
        match_mask:[B, N, N]  - 1 if (i,j) is matching pair

        returns: scalar loss
        """

        # normalize descriptors
        D = F.normalize(D, p=2, dim=-1)
        D_prime = F.normalize(D_prime, p=2, dim=-1)

        # cosine similarity matrix
        # shape: [B, N, N]
        sim = torch.bmm(D, D_prime.transpose(1, 2))

        # positive loss 
        pos_loss = match_mask * (F.relu(self.mp - sim) ** 2)

        # negative loss 
        neg_mask = 1.0 - match_mask
        neg_loss = neg_mask * (F.relu(sim - self.mn) ** 2)

        # combine 
        loss = pos_loss + neg_loss

        return loss.mean()


class TimePointOverallLoss(nn.Module):
    def __init__(self, mp=1.0, mn=0.1, lambda_desc=1.0):
        super().__init__()

        self.kp_loss_fn = TimePointKeypointLoss()
        self.desc_loss_fn = TimePointDescriptorLoss(mp=mp, mn=mn)

        self.lambda_desc = lambda_desc

    def forward(
        self,
        S_logits,
        Y_true,
        S_prime_logits,
        Y_prime_true,
        D,
        D_prime,
        match_mask,
    ):
        # Keypoint losses (with sigmoid inside)
        loss_kp_orig = self.kp_loss_fn(S_logits, Y_true)
        loss_kp_warped = self.kp_loss_fn(S_prime_logits, Y_prime_true)

        # Descriptor loss
        loss_desc = self.desc_loss_fn(D, D_prime, match_mask)

        # Total loss
        total_loss = loss_kp_orig + loss_kp_warped + self.lambda_desc * loss_desc

        return total_loss, {
            "total": total_loss.item(),
            "kp_orig": loss_kp_orig.item(),
            "kp_warped": loss_kp_warped.item(),
            "desc": loss_desc.item(),
        }
