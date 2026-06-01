from typing import Callable
import torch
import torch.nn as nn
import torchvision.models as models


class SmallCNN(nn.Module):
    def __init__(self, emb_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(64, emb_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ResNet18Embed(nn.Module):
    def __init__(self, emb_dim: int = 512, pretrained: bool = False):
        super().__init__()
        self.backbone = models.resnet18(pretrained=pretrained)
        # adjust first conv to single-channel
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_feats, emb_dim)

    def forward(self, x):
        return self.backbone(x)
