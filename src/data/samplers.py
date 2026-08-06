from typing import List
import numpy as np
import torch
from torch.utils.data import Sampler, WeightedRandomSampler, DataLoader

from .datasets import SignatureClassDataset, TripletMiningDataset, collate_triplet


class StructuredBatchSampler(Sampler):
    """Yields structured batches for Phase 3 triplet mining.

    Each batch contains N writers × (N_gen genuine + N_forg forgery).
    Re-seeds per epoch (pass seed=BASE+epoch) so writers appear in different
    order each epoch without re-instantiating the sampler.

    This guarantees every genuine anchor in a batch has at least (N_gen−1)
    genuine positives and N_forg forgery negatives from the same writer,
    making batch-hard mining well-conditioned.
    """

    def __init__(
        self,
        dataset: TripletMiningDataset,
        n_writers: int,
        n_gen: int,
        n_forg: int,
        seed: int = 42,
    ):
        self.n_w      = n_writers
        self.n_g      = n_gen
        self.n_f      = n_forg
        self.seed     = seed
        self.eligible = dataset.eligible
        self.gen_idx  = {w: np.array(dataset.gen_idx[w])  for w in self.eligible}
        self.forg_idx = {w: np.array(dataset.forg_idx[w]) for w in self.eligible}

    def __len__(self) -> int:
        return len(self.eligible) // self.n_w

    def __iter__(self):
        rng  = np.random.default_rng(self.seed)
        shuf = rng.permutation(self.eligible)
        for i in range(0, len(shuf) - self.n_w + 1, self.n_w):
            batch_idx: List[int] = []
            for w in shuf[i:i + self.n_w]:
                gi = rng.choice(
                    self.gen_idx[w],
                    size=min(self.n_g, len(self.gen_idx[w])),
                    replace=False,
                )
                fi = rng.choice(
                    self.forg_idx[w],
                    size=min(self.n_f, len(self.forg_idx[w])),
                    replace=True,
                )
                batch_idx.extend(gi.tolist())
                batch_idx.extend(fi.tolist())
            yield batch_idx


def make_balanced_mining_loader(
    train_ds: SignatureClassDataset,
    ema_weights: torch.Tensor,
    batch_size: int,
    num_workers: int = 0,
) -> DataLoader:
    """DataLoader where each dataset contributes equal total weight per batch.

    Within each dataset's quota, per-sample EMA loss weights determine draw
    probability (harder samples drawn more often). This prevents the private
    dataset (87% of raw images) from dominating the batch distribution.
    """
    datasets  = train_ds.records['dataset'].values
    unique_ds = np.unique(datasets)
    ema_np    = ema_weights.numpy()
    combined  = np.zeros(len(train_ds))

    for ds in unique_ds:
        mask   = datasets == ds
        ema_ds = ema_np[mask]
        combined[mask] = (ema_ds / ema_ds.sum()) / len(unique_ds)

    weights = torch.tensor(combined / combined.sum(), dtype=torch.float)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
