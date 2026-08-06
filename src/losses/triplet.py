import torch
import torch.nn.functional as F


def batch_hard_triplet_loss(
    emb: torch.Tensor,
    wids: list,
    is_gen_list: list,
    margin: float = 0.3,
) -> torch.Tensor:
    """Batch-hard triplet loss for signature verification.

    For every genuine anchor in the batch, mines:
    - hardest positive: maximum cosine distance to another genuine from the same writer
    - hardest negative: minimum cosine distance to a forgery from the same writer

    Requires each batch to contain multiple genuine AND forgery images per writer
    (use StructuredBatchSampler to guarantee this).

    Args:
        emb:         raw embeddings (B, D) — normalised internally
        wids:        writer UID strings, length B
        is_gen_list: 1=genuine, 0=forgery, length B
        margin:      triplet margin (0.3 works well after ArcFace pre-training)

    Returns:
        scalar mean loss over all valid anchor triplets
    """
    n     = len(emb)
    emb_n = F.normalize(emb, dim=1)
    cdist = (1 - emb_n @ emb_n.t()).clamp(min=0.0)   # cosine distances (B, B)

    is_gen   = [bool(x) for x in is_gen_list]
    is_gen_t = torch.tensor(is_gen, dtype=torch.bool, device=emb.device)

    same_w = torch.zeros(n, n, dtype=torch.bool, device=emb.device)
    for i in range(n):
        for j in range(n):
            same_w[i, j] = (wids[i] == wids[j])

    total = torch.zeros(1, device=emb.device)
    count = 0
    for i in range(n):
        if not is_gen[i]:
            continue
        pos_mask    = same_w[i] & is_gen_t
        pos_mask[i] = False                         # exclude self
        neg_mask    = same_w[i] & (~is_gen_t)
        if not pos_mask.any() or not neg_mask.any():
            continue
        hp    = cdist[i][pos_mask].max()
        hn    = cdist[i][neg_mask].min()
        total = total + F.relu(hp - hn + margin)
        count += 1

    return total.squeeze() / max(count, 1)
