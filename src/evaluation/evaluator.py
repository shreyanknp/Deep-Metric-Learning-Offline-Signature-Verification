from typing import Dict, Set, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.metrics.verification import compute_metrics

_NAN = {'eer': float('nan'), 'auc': float('nan'), 'accuracy': float('nan')}
_EMPTY = (np.array([]), np.array([]))


def _build_pools(
    df: pd.DataFrame,
    wid_set: Set[str],
) -> Tuple[Dict, Dict]:
    gen, forg = {}, {}
    for wid, grp in df[df['writer_uid'].isin(wid_set)].groupby('writer_uid'):
        g = grp[grp['label'] == 'genuine']['path'].tolist()
        f = grp[grp['label'] == 'forgery']['path'].tolist()
        if g: gen[wid]  = g
        if f: forg[wid] = f
    return gen, forg


def generate_pairs(
    df: pd.DataFrame,
    wid_set: Set[str],
    n_pairs: int = 10000,
    seed: int = 1,
    neg_mix: float = 0.8,
) -> pd.DataFrame:
    """Sample balanced genuine / forgery pairs for pairwise evaluation.

    Args:
        neg_mix: fraction of negatives that are same-writer skilled forgeries
                 (remainder are random cross-writer genuine pairs)
    """
    gen, forg = _build_pools(df, wid_set)
    writers   = sorted(gen)
    wf        = sorted(set(gen) & set(forg))
    rng       = np.random.default_rng(seed)
    n_pos     = n_pairs // 2
    n_neg     = n_pairs - n_pos
    n_sf      = int(round(n_neg * neg_mix))
    n_rnd     = n_neg - n_sf
    rows      = []

    for _ in range(n_pos):
        w = rng.choice(writers);  g = gen[w]
        if len(g) < 2: continue
        i, j = rng.choice(len(g), 2, replace=False)
        rows.append({'path_a': g[i], 'path_b': g[j], 'label': 1})

    for _ in range(n_sf):
        if not wf: break
        w = rng.choice(wf);  g = gen[w];  f = forg[w]
        rows.append({'path_a': g[rng.integers(len(g))],
                     'path_b': f[rng.integers(len(f))], 'label': 0})

    for _ in range(n_rnd):
        if len(writers) < 2: break
        w1, w2 = rng.choice(writers, 2, replace=False)
        rows.append({'path_a': gen[w1][rng.integers(len(gen[w1]))],
                     'path_b': gen[w2][rng.integers(len(gen[w2]))], 'label': 0})

    return pd.DataFrame(rows)


def run_eval_pairwise(
    name: str,
    df: pd.DataFrame,
    test_wids: Set[str],
    cache: Dict[str, torch.Tensor],
    n_pairs: int = 10000,
) -> Tuple[dict, float, np.ndarray, np.ndarray]:
    """Pairwise cosine similarity evaluation using the TTA embedding cache."""
    if not test_wids:
        print(f'=== {name}: SKIPPED (no test writers)')
        return _NAN, float('nan'), *_EMPTY

    pairs_df = generate_pairs(df, test_wids, n_pairs=n_pairs, seed=1)
    valid    = pairs_df[pairs_df['path_a'].isin(cache) & pairs_df['path_b'].isin(cache)]
    if len(valid) < 10:
        print(f'=== {name}: too few valid pairs ({len(valid)})')
        return _NAN, float('nan'), *_EMPTY

    embs_a = torch.stack([cache[p] for p in valid['path_a']])
    embs_b = torch.stack([cache[p] for p in valid['path_b']])
    sims   = (embs_a * embs_b).sum(dim=1).numpy()
    ys     = valid['label'].values
    res    = compute_metrics(ys, sims)
    gap    = float(sims[ys == 1].mean() - sims[ys == 0].mean())
    print(f'  [pairwise+TTA]  EER={res["eer"]:.2%}  AUC={res["auc"]:.4f}  '
          f'Acc={res["accuracy"]:.2%}  gap={gap:.4f}')
    return res, gap, sims, ys


def run_eval_prototype(
    name: str,
    df: pd.DataFrame,
    test_wids: Set[str],
    cache: Dict[str, torch.Tensor],
    n_enroll: int = 6,
) -> Tuple[dict, float, np.ndarray, np.ndarray]:
    """Prototype evaluation: build one writer prototype from n_enroll genuine images.

    Simulates real-world enrollment where a known set of genuine samples are
    used to create a stable writer representation before verification begins.
    """
    if not test_wids:
        print(f'=== {name}: SKIPPED (no test writers)')
        return _NAN, float('nan'), *_EMPTY

    scores, ys = [], []
    n_valid    = 0

    for wid in sorted(test_wids):
        sub        = df[df['writer_uid'] == wid]
        gen_paths  = [p for p in sub[sub['label'] == 'genuine']['path'].tolist() if p in cache]
        forg_paths = [p for p in sub[sub['label'] == 'forgery']['path'].tolist() if p in cache]
        if len(gen_paths) < n_enroll + 1:
            continue
        proto = F.normalize(
            torch.stack([cache[p] for p in gen_paths[:n_enroll]]).mean(0), dim=0
        )
        for p in gen_paths[n_enroll:]:
            scores.append((proto * cache[p]).sum().item()); ys.append(1)
        for p in forg_paths:
            scores.append((proto * cache[p]).sum().item()); ys.append(0)
        n_valid += 1

    scores = np.array(scores); ys = np.array(ys)
    if len(scores) < 10:
        return _NAN, float('nan'), scores, ys

    res = compute_metrics(ys, scores)
    gap = float(scores[ys == 1].mean() - scores[ys == 0].mean())
    print(f'  [prototype+TTA] EER={res["eer"]:.2%}  AUC={res["auc"]:.4f}  '
          f'Acc={res["accuracy"]:.2%}  gap={gap:.4f}  '
          f'(enroll={n_enroll}, writers={n_valid})')
    return res, gap, scores, ys
