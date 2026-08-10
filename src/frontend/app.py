import io, sys, base64
from pathlib import Path
import numpy as np
import cv2

# Allow `from src.models import ...` when running from within src/frontend/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageOps
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

# ── Signature extraction ───────────────────────────────────────────────────
def extract_signature(pil_img: 'Image.Image') -> 'Image.Image':
    """Remove background and isolate the signature region.

    Pipeline:
      1. Grayscale
      2. Gaussian blur  — suppress texture / camera noise
      3. Adaptive threshold — dark ink → black, background → white
                             handles coloured paper and uneven lighting
      4. Morphological close — fill small gaps in ink strokes
      5. Bounding-box crop around ink with 5% padding
      6. Return white-background / black-ink PIL image
    """
    pil_img = ImageOps.exif_transpose(pil_img)  # honour camera rotation EXIF tag
    gray = np.array(pil_img.convert('L'))

    # Denoise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive threshold: ink (dark) → 0, background (light) → 255
    binary = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=15, C=8,
    )

    # Morphological close to reconnect broken strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Find ink pixels (value = 0) and compute bounding box
    ink = np.where(closed == 0)
    if ink[0].size == 0:
        # Nothing found — return grayscale original as fallback
        return pil_img.convert('L')

    y_min, y_max = int(ink[0].min()), int(ink[0].max())
    x_min, x_max = int(ink[1].min()), int(ink[1].max())

    # 5% padding around the detected ink region
    h, w = gray.shape
    pad_y = max(10, int((y_max - y_min) * 0.05))
    pad_x = max(10, int((x_max - x_min) * 0.05))
    y_min = max(0, y_min - pad_y)
    y_max = min(h, y_max + pad_y)
    x_min = max(0, x_min - pad_x)
    x_max = min(w, x_max + pad_x)

    cropped = closed[y_min:y_max, x_min:x_max]
    return Image.fromarray(cropped)


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
    img   = extract_signature(pil_img)   # isolate ink region, white bg
    views = torch.stack([_tta_tfm(img) for _ in range(n_views)]).to(DEVICE)
    embs  = F.normalize(backbone(views), dim=1)
    return F.normalize(embs.mean(0), dim=0)


# ── Flask ──────────────────────────────────────────────────────────────────
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html', model_ready=MODEL_READY)


@app.route('/preprocess', methods=['POST'])
def preprocess_preview():
    f = request.files.get('image')
    if not f:
        return jsonify({'error': 'No image provided'}), 400
    pil_img   = Image.open(io.BytesIO(f.read()))
    processed = extract_signature(pil_img)
    buf = io.BytesIO()
    processed.save(buf, format='PNG')
    b64 = base64.b64encode(buf.getvalue()).decode()
    return jsonify({'image': 'data:image/png;base64,' + b64})


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
