from typing import Tuple
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.models.heads import SubCenterArcFace
from src.data.datasets import TripletMiningDataset, collate_triplet
from src.data.samplers import make_balanced_mining_loader, StructuredBatchSampler
from src.losses.triplet import batch_hard_triplet_loss


def run_phase1(
    backbone: torch.nn.Module,
    arc_head: SubCenterArcFace,
    train_ds,
    n_epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    warmup_epochs: int,
    margin_start: float,
    margin_end: float,
    ramp_epochs: int,
    grad_clip: float,
    device: torch.device,
    num_classes: int,
    k_sub: int,
    seed: int = 42,
) -> Tuple[dict, torch.Tensor]:
    """Phase 1: full-network training with balanced sampler and margin curriculum.

    Returns:
        history:         {'loss': [...], 'acc': [...]} per epoch
        sample_loss_ema: per-sample EMA loss tensor (needed by Phase 2 sampler)
    """
    n_samples       = len(train_ds)
    sample_loss_ema = torch.ones(n_samples)
    params          = list(backbone.parameters()) + list(arc_head.parameters())
    optimizer       = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    scheduler       = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(n_epochs - warmup_epochs, 1), eta_min=1e-6
    )
    history = {'loss': [], 'acc': []}

    for epoch in range(n_epochs):
        if epoch < warmup_epochs:
            for g in optimizer.param_groups:
                g['lr'] = lr * (epoch + 1) / warmup_epochs

        m = margin_start + (margin_end - margin_start) * min(epoch, ramp_epochs) / ramp_epochs
        arc_head._update_margin(m)

        loader = make_balanced_mining_loader(train_ds, sample_loss_ema, batch_size)
        backbone.train(); arc_head.train()
        ep_loss = 0.0; correct = 0; total = 0; n_b = 0

        for imgs, labels, idxs in tqdm(loader, desc=f'P1 {epoch+1:02d}/{n_epochs}', leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            emb      = backbone(imgs)
            per_loss = arc_head(emb, labels, reduction='none')
            loss     = per_loss.mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
            optimizer.step()

            with torch.no_grad():
                idx_cpu = idxs.cpu()
                sample_loss_ema[idx_cpu] = (
                    0.7 * sample_loss_ema[idx_cpu] + 0.3 * per_loss.detach().cpu()
                )
                cos_all = F.linear(
                    F.normalize(emb, dim=1), F.normalize(arc_head.weight, dim=1)
                )
                logits  = cos_all.view(-1, num_classes, k_sub).max(dim=2).values * arc_head.scale
                correct += (logits.argmax(1) == labels.long()).sum().item()
                total   += labels.size(0)

            ep_loss += loss.item(); n_b += 1

        if epoch >= warmup_epochs:
            scheduler.step()

        avg = ep_loss / n_b; acc = correct / total
        history['loss'].append(avg); history['acc'].append(acc)
        print(f'P1 E{epoch+1:02d}/{n_epochs}  loss={avg:.4f}  acc={acc:.3f}  '
              f'lr={optimizer.param_groups[0]["lr"]:.2e}  margin={m:.3f}')

    return history, sample_loss_ema


def run_phase2(
    backbone: torch.nn.Module,
    arc_head: SubCenterArcFace,
    train_ds,
    sample_loss_ema: torch.Tensor,
    n_epochs: int,
    batch_size: int,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float,
    grad_clip: float,
    device: torch.device,
    num_classes: int,
    k_sub: int,
) -> dict:
    """Phase 2: fine-tune layer4 + GeM + head with stem/layer1-3 frozen."""
    for name, p in backbone.named_parameters():
        p.requires_grad = not any(
            name.startswith(s) for s in ['stem.', 'layer1', 'layer2', 'layer3']
        )
    trainable  = [p for p in backbone.parameters() if p.requires_grad]
    params_all = trainable + list(arc_head.parameters())
    n_frozen   = sum(1 for p in backbone.parameters() if not p.requires_grad)
    print(f'Frozen: {n_frozen}  | Trainable: {len(trainable)}')

    optimizer = torch.optim.Adam([
        {'params': trainable,                    'lr': backbone_lr},
        {'params': list(arc_head.parameters()),  'lr': head_lr},
    ], weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)
    history   = {'loss': [], 'acc': []}

    for epoch in range(n_epochs):
        loader = make_balanced_mining_loader(train_ds, sample_loss_ema, batch_size)
        backbone.train(); arc_head.train()
        ep_loss = 0.0; correct = 0; total = 0; n_b = 0

        for imgs, labels, idxs in tqdm(loader, desc=f'P2 {epoch+1:02d}/{n_epochs}', leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            emb      = backbone(imgs)
            per_loss = arc_head(emb, labels, reduction='none')
            loss     = per_loss.mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params_all, grad_clip)
            optimizer.step()

            with torch.no_grad():
                idx_cpu = idxs.cpu()
                sample_loss_ema[idx_cpu] = (
                    0.7 * sample_loss_ema[idx_cpu] + 0.3 * per_loss.detach().cpu()
                )
                cos_all = F.linear(
                    F.normalize(emb, dim=1), F.normalize(arc_head.weight, dim=1)
                )
                logits  = cos_all.view(-1, num_classes, k_sub).max(dim=2).values * arc_head.scale
                correct += (logits.argmax(1) == labels.long()).sum().item()
                total   += labels.size(0)

            ep_loss += loss.item(); n_b += 1

        scheduler.step()
        avg = ep_loss / n_b; acc = correct / total
        history['loss'].append(avg); history['acc'].append(acc)
        print(f'P2 E{epoch+1:02d}/{n_epochs}  loss={avg:.4f}  acc={acc:.3f}  '
              f'lr={scheduler.get_last_lr()[0]:.2e}')

    return history


def run_phase3(
    backbone: torch.nn.Module,
    triplet_ds: TripletMiningDataset,
    n_epochs: int,
    lr: float,
    weight_decay: float,
    margin: float,
    grad_clip: float,
    n_writers: int,
    n_gen: int,
    n_forg: int,
    device: torch.device,
    seed: int = 42,
) -> dict:
    """Phase 3: batch-hard triplet mining with structured writer batches."""
    for p in backbone.parameters():
        p.requires_grad = True

    params    = list(backbone.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-7)
    history   = {'loss': []}

    for epoch in range(n_epochs):
        sampler = StructuredBatchSampler(triplet_ds, n_writers, n_gen, n_forg, seed=seed + epoch)
        loader  = DataLoader(triplet_ds, batch_sampler=sampler,
                             collate_fn=collate_triplet, num_workers=0)
        backbone.train()
        ep_loss = 0.0; n_b = 0

        for imgs, wids_b, lbls_b in tqdm(loader, desc=f'P3 {epoch+1:02d}/{n_epochs}', leave=False):
            imgs = imgs.to(device)
            emb  = backbone(imgs)
            loss = batch_hard_triplet_loss(emb, wids_b, lbls_b, margin=margin)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
            optimizer.step()
            ep_loss += loss.item(); n_b += 1

        scheduler.step()
        avg = ep_loss / n_b if n_b else float('nan')
        history['loss'].append(avg)
        print(f'P3 E{epoch+1:02d}/{n_epochs}  triplet_loss={avg:.4f}  '
              f'lr={scheduler.get_last_lr()[0]:.2e}')

    return history
