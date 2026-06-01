import torch
import torch.nn as nn


class TripletLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.loss = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, a, p, n):
        return self.loss(a, p, n)
