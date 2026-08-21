# Deep Metric Learning Framework for Offline Signature Verification

This is my Master's thesis project — a complete pipeline for offline handwritten signature verification using deep metric learning. The system learns writer-discriminative embeddings that can distinguish genuine signatures from skilled forgeries, even for writers it has never seen during training.

The core idea: instead of training a binary classifier, I train a backbone to produce compact 256-dimensional embeddings where signatures from the same writer cluster together and signatures from different writers (or forgers) are pushed apart. At inference time, similarity is measured purely by cosine distance — no retraining needed for new writers.

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Datasets                                       │
│   CEDAR  ·  GPDS-150  ·  SigComp2011  ·  SigComp2009  ·  Private       │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │ scan_*() → unified DataFrame
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        Training Pipeline                                 │
│                                                                          │
│   Phase 1 ─ ArcFace Classification (40 epochs, all params)              │
│       ↓                                                                  │
│   Phase 2 ─ Backbone Fine-tuning   (20 epochs, stem+L1-3 frozen)        │
│       ↓                                                                  │
│   Phase 3 ─ Batch-Hard Triplet     (15 epochs, genuine vs forgery)      │
│                                                                          │
│               saves → models/v8_backbone.pth                            │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
     ┌──────────────┐  ┌────────────┐  ┌──────────────────┐
     │ evaluate.py  │  │ Notebooks  │  │  Flask Demo App  │
     │ (pairwise +  │  │ (08_*.ipynb│  │  localhost:5000  │
     │  prototype)  │  │  analysis) │  │                  │
     └──────────────┘  └────────────┘  └──────────────────┘
```

---

## Repository Structure

```
.
├── train.py                        # Main training entry point
├── evaluate.py                     # Evaluation entry point
├── requirements.txt
│
├── configs/
│   ├── v8.yaml                     # Primary config (used for training/eval)
│   ├── cedar_baseline.yaml
│   ├── multidataset_train.yaml
│   └── ...
│
├── src/
│   ├── data/
│   │   ├── scanners.py             # One scanner per dataset → unified DataFrame
│   │   ├── datasets.py             # SignatureClassDataset, TripletMiningDataset
│   │   ├── splits.py               # Deterministic 70/10/20 writer-level split
│   │   ├── transforms.py           # Train / eval / TTA torchvision pipelines
│   │   ├── samplers.py             # EMA-weighted balanced sampler, structured triplet sampler
│   │   └── pairs.py                # PairDataset / TripletDataset (legacy)
│   │
│   ├── models/
│   │   ├── backbones.py            # MultiScaleResNet34 (primary model)
│   │   ├── heads.py                # SubCenterArcFace (K=3)
│   │   ├── siamese.py              # Legacy SiameseNetwork
│   │   └── triplet.py              # Legacy TripletNetwork
│   │
│   ├── losses/
│   │   ├── triplet.py              # Batch-hard triplet loss (cosine)
│   │   └── contrastive.py          # Legacy contrastive loss
│   │
│   ├── training/
│   │   └── trainer.py              # run_phase1, run_phase2, run_phase3
│   │
│   ├── evaluation/
│   │   ├── embedder.py             # TTA embedding cache builder
│   │   └── evaluator.py           # Pair generation, pairwise eval, prototype eval
│   │
│   ├── metrics/
│   │   └── verification.py         # EER, AUC, FAR/FRR
│   │
│   ├── utils/
│   │   ├── config.py               # YAML config loader
│   │   ├── logging.py              # Logger setup
│   │   └── seed.py                 # Reproducibility seed
│   │
│   └── frontend/
│       ├── app.py                  # Flask app (port 5000)
│       ├── streamlit_app.py        # Legacy Streamlit interface
│       └── templates/index.html    # Single-page JS frontend
│
├── notebooks/
│   ├── 08_improved_eval.ipynb      # Full v8 experiment (current)
│   ├── 07_multi_eval.ipynb
│   ├── 06_multi_eval.ipynb
│   ├── ...
│   └── _write_nb_v8.py             # Script that generates 08_improved_eval.ipynb
│
├── models/
│   ├── v8_backbone.pth             # Trained backbone weights
│   └── v8_archead.pth              # Trained ArcFace head weights
│
└── datasets/                       # Place all datasets here (see below)
```

---

## Model Architecture

### MultiScaleResNet34

The backbone branches ResNet34 at the `layer3/layer4` boundary and GeM-pools each branch independently. Concatenating mid-level (stroke texture) and high-level (global shape) features in a single embedding makes the model more robust to variations in signing speed and pressure.

```
Input — 1 × 224 × 224 (grayscale)
        │
        ▼
┌───────────────────────────────────────────┐
│  Stem: Conv1(1→64, 7×7) → BN → ReLU → MaxPool  │
├───────────────────────────────────────────┤
│  Layer 1  (64ch,  3 residual blocks)      │
│  Layer 2  (128ch, 4 residual blocks)      │
│  Layer 3  (256ch, 6 residual blocks) ─────┼──▶ GeM Pool ──▶ 256-dim
│  Layer 4  (512ch, 3 residual blocks) ─────┼──▶ GeM Pool ──▶ 512-dim
└───────────────────────────────────────────┘          │
                                                  Concat (768-dim)
                                                        │
                                                        ▼
                                          ┌─────────────────────────────┐
                                          │  BN(768)                    │
                                          │  Dropout(0.4)               │
                                          │  Linear(768 → 256)          │
                                          │  BN(256, affine=False)      │
                                          └─────────────────────────────┘
                                                        │
                                                 256-dim embedding
                                                  (L2 normalised)
```

The first conv is adapted for grayscale by averaging the three pretrained ImageNet RGB weight planes — this preserves pretrained features without discarding information.

### SubCenterArcFace (Training Head)

During training, embeddings are classified using SubCenterArcFace with K=3 sub-centres per writer. Multiple sub-centres capture natural within-writer signing variation (e.g. fast vs. careful signing). The margin is warmed up linearly from 0.10 to 0.70 over the first 20 epochs to stabilise early training.

---

## Training Pipeline

```
Phase 1 ── Full-network ArcFace  (40 epochs)
│
│   Data:     All 5 datasets, training writers only, genuine images only
│   Sampler:  EMA-weighted balanced sampler — harder samples drawn more
│             frequently; equal weighting across datasets regardless of size
│   Loss:     SubCenterArcFace  (K=3, scale=64, margin 0.10→0.70)
│   Optim:    Adam, LR=1e-3, weight decay=1e-4, gradient clip=1.0
│   Schedule: Linear warmup (3 epochs) → CosineAnnealingLR
│
▼
Phase 2 ── Backbone Fine-tuning  (20 epochs)
│
│   Frozen:   stem, layer1, layer2, layer3
│   Trainable: layer4, GeM pools, embedding head, ArcFace head
│   LR:       backbone 2e-4 / head 5e-4
│
▼
Phase 3 ── Batch-Hard Triplet Mining  (15 epochs)
│
│   Data:     GPDS-150 + CEDAR + Private (writers with genuine AND forgery)
│   Batches:  10 writers × (3 genuine + 2 forgery) per batch
│   Loss:     Batch-hard cosine triplet, margin=0.3
│             Hardest positive = farthest same-writer genuine
│             Hardest negative = closest same-writer forgery
│   LR:       5e-5, all backbone params unfrozen
│
▼
Output:  models/v8_backbone.pth
         models/v8_archead.pth
```

---

## Inference / Demo App

At inference, the ArcFace head is discarded. Verification is done purely with cosine similarity between TTA-averaged embeddings.

```
Upload Signature 1 ───┐
                      ├──▶ OpenCV Preprocessing ──▶ TTA Embed (n=4 views)
Upload Signature 2 ───┘                                      │
                                                             ▼
                                                    Cosine Similarity
                                                       [-1.0 … 1.0]
                                                             │
                                                   threshold = 0.70
                                                             │
                                          ┌──────────────────┴──────────────────┐
                                          ▼                                      ▼
                                   ✅ Same Signer                         ❌ Different Signer
```

### OpenCV Preprocessing

Raw images (especially camera photos) go through an ink-extraction step before embedding:

```
Raw Image (any background)
        │
        ▼
  Grayscale conversion
        │
        ▼
  Gaussian Blur (5×5)   ← suppresses texture and camera noise
        │
        ▼
  Adaptive Gaussian Threshold   ← handles coloured paper / uneven lighting
  (blockSize=15, C=8)           dark ink → 0, background → 255
        │
        ▼
  Morphological Close (3×3)    ← reconnects broken ink strokes
        │
        ▼
  Bounding-box crop around ink  ← 5% padding on each side
        │
        ▼
  Grayscale binary image (ink on white)   → sent to backbone
```

---

## Datasets

Download each dataset and place it under the `datasets/` folder with the following structure:

```
datasets/
├── CEDAR/
│   ├── full_org/           # genuine signatures: original_NNN_N.png
│   └── full_forg/          # forgeries: forgeries_NNN_N.png
│
├── GPDS150/
│   └── train/
│       ├── genuine/        # c-NNN-MM.png   (NNN = writer, MM = sample)
│       └── forge/          # cf-NNN-MM.png
│
├── SigComp2009-training/
│   └── NISDCC-offline-all-001-051-6g/   # genuine only, 51 writers
│
├── SigComp2009-evaluation/
│   ├── 6b_NFIgenuines/genuines/         # NFI-AAABBBCCC.png
│   └── 6b_NFIforgeries/forgeries/       # NFI-AAABBBCCC.png
│
├── sigComp2011-trainingSet/
│   └── trainingSet/OfflineSignatures/
│       └── Dutch/TrainingSet/
│           ├── Offline Genuine/
│           └── Offline Forgeries/
│
└── private_signature_verification/
    ├── full_org/
    └── full_forg/
```

The scanner functions in `src/data/scanners.py` expect exactly this layout. The `scan_*` functions each take the dataset root folder as an argument and return a unified DataFrame with columns `[path, writer_uid, label, dataset]`.

**Dataset sources:**
- CEDAR: [Link from the Computer Science dept, SUNY Buffalo]
- GPDS-150: [GPDS Research Group, University of Las Palmas]
- SigComp 2009/2011: [ICDAR competition archives]

---

## Setup

```bash
# Clone the repository
git clone <repo-url>
cd "Deep Metric Learning Framework Offline Signature Verification"

# Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Requirements:**
- Python 3.10+
- PyTorch 2.x (CUDA recommended for training; CPU works for inference)
- torchvision, numpy, pandas, scikit-learn, Pillow, opencv-python, PyYAML, tqdm, matplotlib, Flask

If you have a CUDA GPU, install the matching `torch` + `torchvision` from [pytorch.org](https://pytorch.org/get-started/locally/) before running `pip install -r requirements.txt`.

---

## Training

Make sure all datasets are in place, then run:

```bash
python train.py --config configs/v8.yaml
```

Key options:

```bash
python train.py \
  --config configs/v8.yaml \
  --ds-root datasets \       # root folder containing all dataset subfolders
  --out-dir models \         # where to save checkpoints
  --device cuda              # or cpu
```

Training runs all three phases sequentially. Checkpoints are saved after each phase as `v8_backbone_phN.pth`. The final backbone is saved as `models/v8_backbone.pth`.

**Expected training time** (approximate, on a single GPU):
- Phase 1: ~3–4 hours on an RTX 3080
- Phase 2: ~1–1.5 hours
- Phase 3: ~45 minutes

If you just want to use the pre-trained weights, download `v8_backbone.pth` and place it in the `models/` folder — training is not required to run the demo.

---

## Evaluation

Run evaluation against the held-out test writers (20% writer-level split, fixed seed):

```bash
python evaluate.py \
  --checkpoint models/v8_backbone.pth \
  --ds-root datasets \
  --config configs/v8.yaml
```

This runs two evaluation protocols per dataset:

1. **Pairwise**: 10,000 sampled pairs (genuine-genuine, genuine-forgery, cross-writer genuine), cosine similarity → EER + AUC
2. **Prototype**: enroll k=6 genuine images per writer as a mean prototype, score all remaining samples against it → EER + AUC

Datasets evaluated: CEDAR, GPDS-150, SigComp2011, SigComp2009-evaluation, Private.

To evaluate a specific dataset only:

```bash
python evaluate.py --checkpoint models/v8_backbone.pth --dataset cedar
```

---

## Demo App

The Flask app provides a browser-based interface for comparing two signatures:

```bash
python src/frontend/app.py
```

Then open `http://localhost:5000` in your browser.

**Usage:**
1. Upload or photograph Signature 1 and Signature 2 using the upload zones
2. Each upload zone has a **File** button (pick from disk) and a **Camera** button (take a live photo)
3. After uploading, a preprocessed preview shows what the model actually sees — useful for checking whether the signature was extracted correctly
4. Click **Verify Signatures**
5. The result shows the cosine similarity score and a verdict: *Same Signer* or *Different Signer*

The threshold slider (default 0.70) controls the decision boundary. Increase it to make the system stricter (fewer false accepts), decrease it to accept more variation.

**Model weights** must be present at `models/v8_backbone.pth`. If the file is missing, the app will start but verification will return a 503 error.

---

## Notebooks

The `notebooks/` folder contains the full experimental history as Jupyter notebooks. Each notebook is generated by a `_write_nb_vN.py` script (never edited directly) to keep a clean record of each experiment iteration.

| Notebook | Description |
|---|---|
| `01_pipeline_test.ipynb` | Initial sanity check of the data pipeline with a small CNN |
| `02_arcface_cedar.ipynb` | First ArcFace experiment, CEDAR dataset only |
| `03_arcface_multidataset.ipynb` | Extended to all four datasets |
| `04_arcface_v5.ipynb` | SubCenterArcFace + multi-scale GeM pooling introduced |
| `05_arcface_v5.ipynb` | Extended ablation run |
| `06_multi_eval.ipynb` | Multi-dataset evaluation protocol added |
| `07_multi_eval.ipynb` | Evaluation improvements |
| `08_improved_eval.ipynb` | Full v8 pipeline: balanced sampler, triplet Phase 3, prototype+TTA eval |

To regenerate the current notebook from scratch:

```bash
python notebooks/_write_nb_v8.py
jupyter notebook notebooks/08_improved_eval.ipynb
```

Pre-executed versions (with cell outputs) are in `datalab_executes/` — these were run on Kaggle with GPU.

---

## Configuration

All hyperparameters are controlled through YAML config files in `configs/`. The active config is `configs/v8.yaml`:

```yaml
# Embedding
emb_dim: 256
img_size: 224

# SubCenterArcFace
k_sub:  3
scale:  64.0
margin_start: 0.10
margin_end:   0.70

# Phase 1
p1_epochs:     40
p1_lr:         1.0e-3
p1_warmup:     3
batch_size:    64

# Phase 2
p2_epochs:     20
p2_lr_backbone: 2.0e-4
p2_lr_head:     5.0e-4

# Phase 3
p3_epochs:     15
p3_lr:         5.0e-5
p3_margin:     0.30
p3_writers_per_batch: 10
p3_genuine_per_writer: 3
p3_forgery_per_writer: 2

# Evaluation
tta_n:    8
k_enroll: 6
n_pairs:  10000
```

---

## Project Notes

- The ArcFace head (`v8_archead.pth`) is only needed if you want to resume training from a checkpoint. It is not used during inference or evaluation.
- SigComp2009-training (51 writers, genuine only) is used for training only and has no evaluation split — it has no forgeries.
- SigComp2009-evaluation (28 writers) is strictly evaluation-only and is never seen during training.
- The private dataset follows the same file naming convention as CEDAR.
- All writer splits are deterministic (seed=42), so the same train/val/test partition is always reproduced.
- For camera uploads, the app mirrors the preview image so it matches what you see in the live viewfinder. The actual image sent to the model is not mirrored — the preprocessing pipeline handles orientation via EXIF transpose.
