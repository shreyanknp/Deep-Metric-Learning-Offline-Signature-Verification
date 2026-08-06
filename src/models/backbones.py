import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class GeMPool(nn.Module):
    """Generalised Mean Pooling with learnable exponent p (p≈3 at init)."""

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p   = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), 1
        ).pow(1.0 / self.p)


class MultiScaleResNet34(nn.Module):
    """ResNet34 with multi-scale GeM pooling for grayscale signature images.

    Splits at the layer3/layer4 boundary, GeM-pools each branch independently
    (256-dim + 512-dim = 768-dim concat), then projects through an
    InsightFace-style BN→Drop→FC→BN head to an EMB_DIM-dimensional embedding.

    The first conv is adapted to single-channel by averaging the three ImageNet
    RGB weight planes — retains pretrained features without discarding colour info.
    """

    def __init__(self, emb_dim: int = 256):
        super().__init__()
        try:
            from torchvision.models import ResNet34_Weights
            net = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        except (ImportError, AttributeError):
            net = models.resnet34(pretrained=True)

        w = net.conv1.weight.data
        net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        net.conv1.weight.data = w.mean(dim=1, keepdim=True)

        self.stem   = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3   # 256 ch
        self.layer4 = net.layer4   # 512 ch
        self.gem3   = GeMPool(p=3)
        self.gem4   = GeMPool(p=3)
        self.head   = nn.Sequential(
            nn.BatchNorm1d(768),
            nn.Dropout(0.4),
            nn.Linear(768, emb_dim),
            nn.BatchNorm1d(emb_dim, affine=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x  = self.stem(x)
        x  = self.layer1(x)
        x  = self.layer2(x)
        x3 = self.layer3(x)
        x4 = self.layer4(x3)
        f  = torch.cat([self.gem3(x3).flatten(1), self.gem4(x4).flatten(1)], dim=1)
        return self.head(f)
