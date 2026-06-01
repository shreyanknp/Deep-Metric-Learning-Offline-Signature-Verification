import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import numpy as np
from src.utils.config import load_config
from src.data.datasets import SignatureDataset
from src.data.transforms import get_default_transforms
from src.models.backbones import SmallCNN, ResNet18Embed
from src.models.siamese import SiameseNetwork
from src.metrics.verification import compute_metrics


def build_model(cfg):
    emb_dim = cfg.get("model", {}).get("emb_dim", 128)
    b = cfg.get("model", {}).get("backbone", "small")
    if b == "resnet18":
        backbone = ResNet18Embed(emb_dim=emb_dim)
    else:
        backbone = SmallCNN(emb_dim=emb_dim)
    model = SiameseNetwork(encoder=backbone)
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    args = p.parse_args()
    cfg = load_config(args.config)

    data_root = cfg.get("data", {}).get("root", "data/raw/CEDAR")
    ds = SignatureDataset(data_root, source=cfg.get("data", {}).get("source", "CEDAR"), transform=get_default_transforms(cfg.get("data", {}).get("size", 224)))
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    model = build_model(cfg)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    scores = []
    labels = []
    with torch.no_grad():
        # simple evaluation: compute embedding for each sample and compare pairwise
        embeddings = []
        writers = []
        for img, label, wid, src in dl:
            emb = model.encoder(img)
            embeddings.append(emb.numpy()[0])
            labels.append(int(label.item()))
            writers.append(wid)
        emb = np.vstack(embeddings)
        # build pairwise scores and labels (for demo, compare all pairs)
        n = emb.shape[0]
        for i in range(n):
            for j in range(i+1, n):
                score = -np.linalg.norm(emb[i] - emb[j])  # higher = more similar
                same = 1 if writers[i] == writers[j] else 0
                scores.append(score)
                labels.append(same)

    import numpy as _np
    metrics = compute_metrics(_np.array(labels), _np.array(scores))
    print(metrics)


if __name__ == "__main__":
    main()
