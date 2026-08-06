import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SubCenterArcFace(nn.Module):
    """Sub-Center ArcFace margin loss (Deng et al., ECCV 2020).

    K sub-centres per class let the model tolerate intra-class variation
    (fast vs. careful signing, pen pressure, etc.) without contaminating the
    main cluster centre. K=3 is the validated default for signature tasks.

    Margin curriculum: call _update_margin(m) at the start of each epoch to
    ramp m from MARGIN_START to MARGIN_END over RAMP_EPOCHS.

    Args:
        emb_dim:   backbone output dimension (must match MultiScaleResNet34.emb_dim)
        n_classes: number of training writers
        K:         sub-centres per class
        scale:     logit temperature s (64.0 is standard for well-normalised embeddings)
        margin:    initial additive angular margin m (radians)
    """

    def __init__(
        self,
        emb_dim: int,
        n_classes: int,
        K: int = 3,
        scale: float = 64.0,
        margin: float = 0.5,
    ):
        super().__init__()
        self.K      = K
        self.scale  = scale
        self.weight = nn.Parameter(torch.FloatTensor(n_classes * K, emb_dim))
        nn.init.xavier_uniform_(self.weight)
        self._update_margin(margin)

    def _update_margin(self, m: float) -> None:
        self.margin = m
        self.cos_m  = math.cos(m)
        self.sin_m  = math.sin(m)
        self.th     = math.cos(math.pi - m)
        self.mm     = math.sin(math.pi - m) * m

    def forward(
        self,
        emb: torch.Tensor,
        labels: torch.Tensor,
        reduction: str = 'mean',
    ) -> torch.Tensor:
        emb_n   = F.normalize(emb,         dim=1)
        W_n     = F.normalize(self.weight,  dim=1)
        cos_all = F.linear(emb_n, W_n)                                 # (B, n_classes*K)
        cos_all = cos_all.view(-1, cos_all.shape[1] // self.K, self.K)
        cos_t, _ = cos_all.max(dim=2)                                   # (B, n_classes)

        sin_t  = torch.sqrt((1 - cos_t**2).clamp(min=1e-6))
        cos_tm = cos_t * self.cos_m - sin_t * self.sin_m
        cos_tm = torch.where(cos_t > self.th, cos_tm, cos_t - self.mm)

        one_hot = torch.zeros_like(cos_t)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        logits = (one_hot * cos_tm + (1 - one_hot) * cos_t) * self.scale
        return F.cross_entropy(logits, labels.long(), reduction=reduction)
