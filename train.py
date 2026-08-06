"""Train the v8 MultiScaleResNet34 + SubCenterArcFace model.

Usage:
    python train.py
    python train.py --config configs/v8.yaml --data-root datasets/
    python train.py --device cuda --seed 0
"""
import argparse
from pathlib import Path
import pandas as pd
import torch

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.logging import get_logger
from src.data import (
    scan_cedar, scan_gpds150, scan_sigcomp2011, scan_sigcomp2009, scan_private,
    split_writers, SignatureClassDataset, TripletMiningDataset,
    get_train_transforms,
)
from src.models import MultiScaleResNet34, SubCenterArcFace
from src.training import run_phase1, run_phase2, run_phase3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Train v8 signature verification model')
    p.add_argument('--config',    default='configs/v8.yaml',  help='YAML config path')
    p.add_argument('--data-root', default=None,               help='Override datasets/ root')
    p.add_argument('--device',    default=None,               help='cuda | cpu | auto')
    p.add_argument('--seed',      type=int, default=None,     help='Override random seed')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = load_config(args.config)
    log  = get_logger('train')

    seed = args.seed if args.seed is not None else cfg.get('seed', 42)
    set_seed(seed)

    dev_str = args.device or cfg.get('device', 'auto')
    device  = (torch.device('cuda' if torch.cuda.is_available() else 'cpu')
               if dev_str == 'auto' else torch.device(dev_str))
    log.info(f'Device: {device}  |  Seed: {seed}')

    ds_root  = Path(args.data_root or 'datasets')
    img_size = cfg['data']['img_size']
    emb_dim  = cfg['model']['emb_dim']
    k_sub    = cfg['model']['k_sub']
    scale    = cfg['model']['scale']
    tcfg     = cfg['training']
    ecfg     = cfg.get('evaluation', {})
    out_dir  = Path(cfg['output']['models_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── scan ─────────────────────────────────────────────────────────────────
    log.info('Scanning datasets...')
    df_cedar = scan_cedar(ds_root / 'CEDAR')
    df_gpds  = scan_gpds150(ds_root / 'GPDS150')
    df_sc11  = scan_sigcomp2011(ds_root / 'sigComp2011-trainingSet')
    df_sc09  = scan_sigcomp2009(ds_root / 'SigComp2009-training')
    df_prv   = scan_private(ds_root / 'private_signature_verification')
    df_all   = pd.concat([df_cedar, df_gpds, df_sc11, df_sc09, df_prv], ignore_index=True)

    for name, df in [('CEDAR', df_cedar), ('GPDS150', df_gpds), ('SigComp2011', df_sc11),
                     ('SigComp2009', df_sc09), ('Private', df_prv)]:
        gen  = (df['label'] == 'genuine').sum()
        forg = (df['label'] == 'forgery').sum()
        log.info(f'{name:12s}  writers={df["writer_uid"].nunique():4d}  '
                 f'genuine={gen:6d}  forgery={forg:6d}')

    # ── splits ────────────────────────────────────────────────────────────────
    cedar_train, _, _ = split_writers(df_cedar['writer_uid'].unique(), seed=seed)
    gpds_train,  _, _ = split_writers(df_gpds['writer_uid'].unique(),  seed=seed)
    sc11_train,  _, _ = split_writers(df_sc11['writer_uid'].unique(),  seed=seed)
    prv_train,   _, _ = split_writers(df_prv['writer_uid'].unique(),   seed=seed)
    sc09_train        = set(df_sc09['writer_uid'].unique())

    all_train_wids  = cedar_train | gpds_train | sc11_train | sc09_train | prv_train
    sorted_train    = sorted(all_train_wids)
    writer_to_class = {wid: i for i, wid in enumerate(sorted_train)}
    num_classes     = len(sorted_train)
    log.info(f'Training classes: {num_classes}')

    # ── datasets ──────────────────────────────────────────────────────────────
    train_tfm = get_train_transforms(img_size)
    train_ds  = SignatureClassDataset(df_all, writer_to_class, all_train_wids, transform=train_tfm)
    log.info(f'Training samples: {len(train_ds):,}')

    p3_wids    = gpds_train | cedar_train | prv_train
    df_p3      = df_all[df_all['writer_uid'].isin(p3_wids)]
    triplet_ds = TripletMiningDataset(df_p3, p3_wids, transform=train_tfm)
    log.info(f'Phase 3 writers: {len(triplet_ds.eligible)}')

    # ── model ─────────────────────────────────────────────────────────────────
    set_seed(seed)
    backbone = MultiScaleResNet34(emb_dim).to(device)
    arc_head = SubCenterArcFace(emb_dim, num_classes, K=k_sub, scale=scale, margin=0.1).to(device)
    n_bb = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    n_hd = sum(p.numel() for p in arc_head.parameters()  if p.requires_grad)
    log.info(f'Backbone: {n_bb:,} params  |  ArcFace: {n_hd:,} params')

    # ── phase 1 ───────────────────────────────────────────────────────────────
    p1 = tcfg['phase1']
    log.info('Phase 1: full-network ArcFace training')
    _, sample_loss_ema = run_phase1(
        backbone=backbone, arc_head=arc_head, train_ds=train_ds,
        n_epochs=p1['epochs'], batch_size=tcfg['batch_size'],
        lr=p1['lr'], weight_decay=p1['weight_decay'],
        warmup_epochs=p1['warmup_epochs'],
        margin_start=p1['margin_start'], margin_end=p1['margin_end'],
        ramp_epochs=p1['ramp_epochs'], grad_clip=p1['grad_clip'],
        device=device, num_classes=num_classes, k_sub=k_sub, seed=seed,
    )

    # ── phase 2 ───────────────────────────────────────────────────────────────
    p2 = tcfg['phase2']
    log.info('Phase 2: fine-tuning (layer4 + head, frozen stem/layer1-3)')
    run_phase2(
        backbone=backbone, arc_head=arc_head, train_ds=train_ds,
        sample_loss_ema=sample_loss_ema,
        n_epochs=p2['epochs'], batch_size=tcfg['batch_size'],
        backbone_lr=p2['backbone_lr'], head_lr=p2['head_lr'],
        weight_decay=p2['weight_decay'], grad_clip=p2['grad_clip'],
        device=device, num_classes=num_classes, k_sub=k_sub,
    )

    # ── phase 3 ───────────────────────────────────────────────────────────────
    p3 = tcfg['phase3']
    log.info('Phase 3: batch-hard triplet mining')
    run_phase3(
        backbone=backbone, triplet_ds=triplet_ds,
        n_epochs=p3['epochs'], lr=p3['lr'], weight_decay=p3['weight_decay'],
        margin=p3['margin'], grad_clip=p3['grad_clip'],
        n_writers=p3['n_writers'], n_gen=p3['n_genuine'], n_forg=p3['n_forgery'],
        device=device, seed=seed,
    )

    # ── save ──────────────────────────────────────────────────────────────────
    bb_path = out_dir / cfg['output']['backbone_name']
    hd_path = out_dir / cfg['output']['archead_name']
    torch.save(backbone.state_dict(), bb_path)
    torch.save(arc_head.state_dict(), hd_path)
    log.info(f'Backbone → {bb_path}  ({bb_path.stat().st_size / 1e6:.1f} MB)')
    log.info(f'ArcFace  → {hd_path}  ({hd_path.stat().st_size / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
