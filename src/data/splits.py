from typing import Set, Tuple
import numpy as np


def split_writers(
    writer_uids,
    seed: int = 42,
    train_frac: float = 0.70,
    val_frac: float = 0.10,
) -> Tuple[Set[str], Set[str], Set[str]]:
    """Deterministic writer-level 70/10/20 train/val/test split.

    Shuffles writer UIDs with a fixed seed, then cuts at the fractions.
    The same seed always produces the same split regardless of input order.

    Returns:
        (train_set, val_set, test_set) — disjoint sets of writer UIDs
    """
    arr = np.array(sorted(writer_uids))
    rng = np.random.default_rng(seed)
    rng.shuffle(arr)
    n    = len(arr)
    n_tr = int(round(n * train_frac))
    n_v  = int(round(n * val_frac))
    return (
        set(arr[:n_tr]),
        set(arr[n_tr:n_tr + n_v]),
        set(arr[n_tr + n_v:]),
    )
