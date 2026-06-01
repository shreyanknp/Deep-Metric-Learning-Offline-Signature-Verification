from typing import List, Tuple, Dict, Any
import random
import torch
from torch.utils.data import Dataset


class PairDataset(Dataset):
    """Generates siamese pairs from a SignatureDataset or precomputed list.
    Each item: (img1, img2, label) where label=1 for genuine pair, 0 for forgery.
    """

    def __init__(self, items: List[Tuple[Any, int, str]], transform=None, n_pairs: int = 10000):
        # items: list of (path_or_tensor, writer_id, label_str)
        self.items = items
        self.transform = transform
        self.by_writer = {}
        for it in items:
            _, wid, _ = it
            self.by_writer.setdefault(wid, []).append(it)
        self.n_pairs = n_pairs

    def __len__(self):
        return self.n_pairs

    def __getitem__(self, idx):
        # sample genuine or forgery pair
        if random.random() < 0.5:
            # genuine: same writer
            wid = random.choice([w for w in self.by_writer.keys() if w is not None])
            a, b = random.sample(self.by_writer[wid], 2)
            img1 = a[0]
            img2 = b[0]
            label = 1
        else:
            # forgery: pick genuine vs forgery or different writers
            wids = [w for w in self.by_writer.keys() if w is not None]
            wa = random.choice(wids)
            wb = random.choice(wids)
            a = random.choice(self.by_writer[wa])
            b = random.choice(self.by_writer[wb])
            img1 = a[0]
            img2 = b[0]
            label = 0
        return img1, img2, label


class TripletDataset(Dataset):
    """Simple triplet generation: anchor, positive (same writer), negative (different writer)."""

    def __init__(self, items: List[Tuple[Any, int, str]], n_triplets: int = 10000):
        self.items = items
        self.by_writer = {}
        for it in items:
            _, wid, _ = it
            self.by_writer.setdefault(wid, []).append(it)
        self.wids = [w for w in self.by_writer.keys() if w is not None]
        self.n_triplets = n_triplets

    def __len__(self):
        return self.n_triplets

    def __getitem__(self, idx):
        wa = random.choice(self.wids)
        a, p = random.sample(self.by_writer[wa], 2)
        wb = random.choice([w for w in self.wids if w != wa])
        n = random.choice(self.by_writer[wb])
        return a[0], p[0], n[0]
