import torch
import torch.nn as nn
from .backbones import SmallCNN


class SiameseNetwork(nn.Module):
    def __init__(self, encoder: nn.Module = None):
        super().__init__()
        self.encoder = encoder or SmallCNN()

    def forward(self, x1, x2):
        e1 = self.encoder(x1)
        e2 = self.encoder(x2)
        return e1, e2
