from pathlib import Path
from typing import List, Optional, Tuple, Dict
import re
from PIL import Image
import torch
from torch.utils.data import Dataset


class SignatureDataset(Dataset):
    """Generic signature dataset loader.

    Expects folder with `genuine` and `forgery` subfolders or files named
    with patterns that include writer and sample ids.
    """

    def __init__(self, root: str, source: str = "CEDAR", transform=None):
        self.root = Path(root)
        self.source = source
        self.transform = transform
        self.items = []  # (path, writer_id, label)
        self._scan()

    def _scan(self):
        # support structured folders or flat files
        org = self.root / "full_org"
        forg = self.root / "full_forg"
        if org.exists() and forg.exists():
            for p in org.glob("*.png"):
                self.items.append((str(p), None, "genuine"))
            for p in forg.glob("*.png"):
                self.items.append((str(p), None, "forgery"))
            return

        # fallback: scan any png and try to parse writer/sample
        pat_org = re.compile(r"original_(\d+)_(\d+)\.png", re.IGNORECASE)
        pat_forg = re.compile(r"forgeries_(\d+)_(\d+)\.png", re.IGNORECASE)
        for p in self.root.rglob("*.png"):
            m1 = pat_org.search(p.name)
            m2 = pat_forg.search(p.name)
            if m1:
                self.items.append((str(p), int(m1.group(1)), "genuine"))
            elif m2:
                self.items.append((str(p), int(m2.group(1)), "forgery"))
            else:
                self.items.append((str(p), None, "unknown"))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, writer_id, label = self.items[idx]
        img = Image.open(path).convert("L")
        if self.transform:
            img = self.transform(img)
        target = 1 if label == "genuine" else 0
        return img, target, writer_id, self.source
