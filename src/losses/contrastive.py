import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, e1: torch.Tensor, e2: torch.Tensor, label: torch.Tensor):
        # label: 1 for genuine (similar), 0 for dissimilar
        dist_sq = torch.sum((e1 - e2) ** 2, dim=1)
        pos_loss = label * dist_sq
        neg_loss = (1 - label) * torch.clamp(self.margin - torch.sqrt(dist_sq + 1e-8), min=0.0) ** 2
        loss = torch.mean(pos_loss + neg_loss)
        return loss
