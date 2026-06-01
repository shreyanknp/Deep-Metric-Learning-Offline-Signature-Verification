import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.logging import get_logger
from src.data.datasets import SignatureDataset
from src.data.transforms import get_default_transforms
from src.models.backbones import SmallCNN, ResNet18Embed
from src.models.siamese import SiameseNetwork
from src.losses.contrastive import ContrastiveLoss


def build_model(cfg):
    emb_dim = cfg.get("model", {}).get("emb_dim", 128)
    backbone = None
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
    args = p.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    logger = get_logger("train")

    # dataset
    data_root = cfg.get("data", {}).get("root", "data/raw/CEDAR")
    ds = SignatureDataset(data_root, source=cfg.get("data", {}).get("source", "CEDAR"), transform=get_default_transforms(cfg.get("data", {}).get("size", 224)))
    dl = DataLoader(ds, batch_size=cfg.get("train", {}).get("batch_size", 16), shuffle=True)

    model = build_model(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.get("train", {}).get("lr", 1e-4))
    criterion = ContrastiveLoss(margin=cfg.get("loss", {}).get("margin", 1.0))

    epochs = cfg.get("train", {}).get("epochs", 1)
    ckpt_dir = Path(cfg.get("output", {}).get("ckpt_dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    for ep in range(epochs):
        for batch in dl:
            imgs, labels, writer, source = batch
            # simple self-contrastive step: pair consecutive images
            imgs = imgs.to(device)
            labels = labels.to(device).float()
            if imgs.size(0) < 2:
                continue
            x1 = imgs[0::2]
            x2 = imgs[1::2]
            lab = labels[0::2]
            e1, e2 = model(x1, x2)
            loss = criterion(e1, e2, lab)
            opt.zero_grad()
            loss.backward()
            opt.step()
        logger.info(f"Epoch {ep+1}/{epochs} loss={loss.item():.4f}")
        torch.save(model.state_dict(), ckpt_dir / f"model_ep{ep+1}.pt")


if __name__ == "__main__":
    main()
