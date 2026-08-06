from typing import Dict, List, Set
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image


class SignatureClassDataset(Dataset):
    """Genuine-only classification dataset for Phase 1 / Phase 2 ArcFace training.

    Filters `df` to training writers and genuine images only.
    Returns (img_tensor, class_idx, sample_idx).

    sample_idx is used by make_balanced_mining_loader to update the
    per-sample EMA loss estimate for hard-mining weighting.
    The 'dataset' column is used for balanced sampling across datasets.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        writer_to_class: Dict[str, int],
        writer_uid_set: Set[str],
        transform=None,
    ):
        sub = df[
            df['writer_uid'].isin(writer_uid_set) & (df['label'] == 'genuine')
        ]
        self.records         = sub[['path', 'writer_uid', 'dataset']].reset_index(drop=True)
        self.writer_to_class = writer_to_class
        self.transform       = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        row = self.records.iloc[idx]
        img = Image.open(row['path']).convert('L')
        if self.transform:
            img = self.transform(img)
        return img, self.writer_to_class[row['writer_uid']], idx


class TripletMiningDataset(Dataset):
    """Dataset for Phase 3 batch-hard triplet mining.

    Pre-indexes genuine and forgery paths per writer so StructuredBatchSampler
    can efficiently build structured batches at yield time.
    Eligible writers: those with ≥ 2 genuine AND ≥ 1 forgery image.

    Returns (img_tensor, writer_uid_str, is_genuine_int).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        writer_uid_set: Set[str],
        transform=None,
    ):
        sub  = df[df['writer_uid'].isin(writer_uid_set)]
        gen  = sub[sub['label'] == 'genuine'].groupby('writer_uid')['path'].apply(list)
        forg = sub[sub['label'] == 'forgery'].groupby('writer_uid')['path'].apply(list)
        eligible = [
            w for w in gen.index
            if w in forg.index and len(forg[w]) >= 1 and len(gen[w]) >= 2
        ]

        self.items:    List[tuple]        = []
        self.gen_idx:  Dict[str, List[int]] = {}
        self.forg_idx: Dict[str, List[int]] = {}

        for w in eligible:
            gi0 = len(self.items)
            for p in gen[w]:  self.items.append((p, w, 1))
            gi1 = len(self.items)
            for p in forg[w]: self.items.append((p, w, 0))
            fi1 = len(self.items)
            self.gen_idx[w]  = list(range(gi0, gi1))
            self.forg_idx[w] = list(range(gi1, fi1))

        self.eligible  = eligible
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, wid, lbl = self.items[idx]
        img = Image.open(path).convert('L')
        if self.transform:
            img = self.transform(img)
        return img, wid, lbl


def collate_triplet(batch):
    """Custom collate for TripletMiningDataset — keeps writer UIDs as a plain list."""
    imgs = torch.stack([b[0] for b in batch])
    wids = [b[1] for b in batch]
    lbls = [int(b[2]) for b in batch]
    return imgs, wids, lbls
