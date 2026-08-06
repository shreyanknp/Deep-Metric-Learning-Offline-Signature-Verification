from typing import Dict, Set
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm
import pandas as pd


def build_embedding_cache(
    df: pd.DataFrame,
    wid_set: Set[str],
    backbone: torch.nn.Module,
    device: torch.device,
    tfm,
    n_views: int = 8,
    imgs_per_batch: int = 8,
) -> Dict[str, torch.Tensor]:
    """Pre-embed all unique images for a writer set using Test-Time Augmentation.

    Each image is forward-passed n_views times with random augmentation.
    The n_views embeddings are averaged and L2-normalised, giving a more
    stable representation than a single deterministic pass.

    Batches imgs_per_batch images together (each with n_views views) to
    amortise GPU launch overhead: one forward pass = imgs_per_batch × n_views
    images on GPU.

    Args:
        df:             DataFrame with 'writer_uid' and 'path' columns
        wid_set:        writer UIDs to embed
        backbone:       model — set to eval mode internally
        device:         target device
        tfm:            stochastic transform (get_tta_transforms())
        n_views:        TTA views per image (8 used in v8)
        imgs_per_batch: images per GPU forward pass

    Returns:
        {path: L2-normalised embedding tensor (D,)}
    """
    backbone.eval()
    paths: list = df[df['writer_uid'].isin(wid_set)]['path'].unique().tolist()
    cache: Dict[str, torch.Tensor] = {}

    with torch.no_grad():
        for i in tqdm(range(0, len(paths), imgs_per_batch), desc='TTA embed', leave=False):
            batch_paths = paths[i:i + imgs_per_batch]
            views = []
            for p in batch_paths:
                img = Image.open(p).convert('L')
                views.extend(tfm(img) for _ in range(n_views))
            views_t = torch.stack(views).to(device)                          # (B*n_views, C, H, W)
            embs    = F.normalize(backbone(views_t), dim=1)                  # (B*n_views, D)
            embs    = embs.view(len(batch_paths), n_views, -1).mean(dim=1)   # (B, D)
            embs    = F.normalize(embs, dim=1).cpu()
            for j, p in enumerate(batch_paths):
                cache[p] = embs[j]

    return cache
