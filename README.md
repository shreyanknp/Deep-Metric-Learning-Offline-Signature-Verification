# Deep Metric Learning Framework for Offline Signature Verification

This repository provides a clean, modular PyTorch framework for offline
handwritten signature verification using deep metric learning.

Goal: train siamese or triplet networks on signature datasets (CEDAR, GPDS,
BHSig260) and evaluate cross-dataset generalisation.

Project layout:

- `src/` : core code (data, models, losses, metrics, utils)
- `configs/` : example YAML experiment configs
- `old/` : reference code and notebooks (read-only)
- `data/raw/` : placeholder paths for datasets (not included)
- `checkpoints/`, `logs/`, `outputs/` : runtime artifacts (gitignored)

Quick start (after installing requirements and placing datasets):

```
python train.py --config configs/cedar_baseline.yaml
python evaluate.py --config configs/gpds_train_cedar_test.yaml --checkpoint checkpoints/model.pt
```

Datasets placeholders:

- `data/raw/CEDAR/`
- `data/raw/GPDS/`
- `data/raw/BHSig260/Hindi/`
- `data/raw/BHSig260/Bengali/`

See `configs/` for example experiment settings.
