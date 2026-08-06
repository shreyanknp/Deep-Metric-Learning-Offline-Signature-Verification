import io, sys
from pathlib import Path

# Allow `from src.models import ...` when running from within src/frontend/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from flask import Flask, request, jsonify, render_template

from src.models import MultiScaleResNet34

WEIGHTS_PATH = PROJECT_ROOT / 'models' / 'v8_backbone.pth'
EMB_DIM      = 256
IMG_SIZE     = 224
TTA_N        = 4
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    print('          Run train.py (or notebook 08) and save the model first.')

# ── TTA transform ──────────────────────────────────────────────────────────
_tta_tfm = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.RandomAffine(degrees=5, translate=(0.03, 0.03), scale=(0.96, 1.04), fill=255),
    T.RandomPerspective(distortion_scale=0.1, p=0.5, fill=255),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5]),
])


@torch.no_grad()
def embed(pil_img, n_views: int = TTA_N) -> torch.Tensor:
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
        return jsonify({'error': 'Model not loaded. Run train.py or notebook 08 first.'}), 503

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
    sim  = round(float((emb1 * emb2).sum().item()), 4)

    verdict    = 'same' if sim >= threshold else 'different'
    confidence = (sim + 1.0) / 2.0

    return jsonify({
        'similarity':  sim,
        'verdict':     verdict,
        'threshold':   threshold,
        'confidence':  f'{confidence:.1%}',
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000, use_reloader=False)
