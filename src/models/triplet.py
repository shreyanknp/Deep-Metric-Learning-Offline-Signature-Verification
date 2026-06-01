import torch
import torch.nn as nn


class TripletNetwork(nn.Module):
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, a, p, n):
        ea = self.encoder(a)
        ep = self.encoder(p)
        en = self.encoder(n)
        return ea, ep, en
