import io
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torchvision.transforms as T
from PIL import Image
from flask import Flask, request, jsonify, render_template

# ── paths ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
WEIGHTS_PATH = PROJECT_ROOT / 'models' / 'v8_backbone.pth'
EMB_DIM      = 256
IMG_SIZE     = 224
TTA_N        = 4
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── architecture (must match notebook 08) ─────────────────────────────────
class GeMPool(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p   = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.adaptive_avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), 1
        ).pow(1.0 / self.p)


class MultiScaleResNet34(nn.Module):
    def __init__(self, emb_dim=256):
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
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.gem3   = GeMPool(p=3)
        self.gem4   = GeMPool(p=3)
        self.head   = nn.Sequential(
            nn.BatchNorm1d(768),
            nn.Dropout(0.4),
            nn.Linear(768, emb_dim),
            nn.BatchNorm1d(emb_dim, affine=False),
        )

    def forward(self, x):
        x  = self.stem(x)
        x  = self.layer1(x)
        x  = self.layer2(x)
        x3 = self.layer3(x)
        x4 = self.layer4(x3)
        f  = torch.cat([self.gem3(x3).flatten(1), self.gem4(x4).flatten(1)], dim=1)
        return self.head(f)


# ── load backbone ──────────────────────────────────────────────────────────
backbone    = MultiScaleResNet34(EMB_DIM).to(DEVICE)
MODEL_READY = False

if WEIGHTS_PATH.exists():
    backbone.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    backbone.eval()
    MODEL_READY = True
    print(f'[OK] Backbone loaded — {WEIGHTS_PATH.name}  ({DEVICE})')
else:
    print(f'[WARNING] Weights not found at {WEIGHTS_PATH}')
    print('          Run notebook 08 and save the model first.')

# ── transforms ─────────────────────────────────────────────────────────────
_tta_tfm = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.RandomAffine(degrees=5, translate=(0.03, 0.03), scale=(0.96, 1.04), fill=255),
    T.RandomPerspective(distortion_scale=0.1, p=0.5, fill=255),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5]),
])


@torch.no_grad()
def embed(pil_img, n_views=TTA_N):
    img   = pil_img.convert('L')
    views = torch.stack([_tta_tfm(img) for _ in range(n_views)]).to(DEVICE)
    embs  = F.normalize(backbone(views), dim=1)
    return F.normalize(embs.mean(0), dim=0)


# ── Flask ──────────────────────────────────────────────────────────────────
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html', model_ready=MODEL_READY)


@app.route('/verify', methods=['POST'])
def verify():
    if not MODEL_READY:
        return jsonify({'error': 'Model not loaded. Run notebook 08 and save v8_backbone.pth first.'}), 503

    f1 = request.files.get('image1')
    f2 = request.files.get('image2')
    if not f1 or not f2:
        return jsonify({'error': 'Both image1 and image2 are required.'}), 400

    try:
        threshold = float(request.form.get('threshold', 0.70))
        threshold = max(0.0, min(1.0, threshold))
    except ValueError:
        threshold = 0.70

    img1 = Image.open(io.BytesIO(f1.read()))
    img2 = Image.open(io.BytesIO(f2.read()))

    emb1 = embed(img1)
    emb2 = embed(img2)
    sim  = float((emb1 * emb2).sum().item())
    sim  = round(sim, 4)

    verdict = 'same' if sim >= threshold else 'different'
    # map cosine similarity [-1,1] → confidence [0,1]
    confidence = (sim + 1.0) / 2.0

    return jsonify({
        'similarity':  sim,
        'verdict':     verdict,
        'threshold':   threshold,
        'confidence':  f'{confidence:.1%}',
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000, use_reloader=False)
