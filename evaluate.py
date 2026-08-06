"""Evaluate a trained backbone on all datasets using pairwise and prototype protocols.

Usage:
    python evaluate.py --checkpoint models/v8_backbone.pth
    python evaluate.py --checkpoint models/v8_backbone.pth --config configs/v8.yaml
    python evaluate.py --checkpoint models/v8_backbone.pth --datasets cedar gpds150
"""
import argparse
from pathlib import Path
import pandas as pd
import torch

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.logging import get_logger
from src.data import (
    scan_cedar, scan_gpds150, scan_sigcomp2011, scan_sigcomp2009, scan_sigcomp2009_eval, scan_private,
    split_writers, get_tta_transforms,
)
from src.models import MultiScaleResNet34
from src.evaluation import build_embedding_cache, run_eval_pairwise, run_eval_prototype
from src.metrics.verification import compute_metrics
import numpy as np


_DATASET_NAMES = ['cedar', 'gpds150', 'sigcomp2011', 'sigcomp2009', 'private']


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Evaluate v8 signature verification model')
    p.add_argument('--checkpoint', required=True, help='Path to backbone .pth weights')
    p.add_argument('--config',     default='configs/v8.yaml')
    p.add_argument('--data-root',  default=None)
    p.add_argument('--device',     default=None)
    p.add_argument('--datasets',   nargs='+', default=_DATASET_NAMES,
                   choices=_DATASET_NAMES, help='Datasets to evaluate')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = load_config(args.config)
    log  = get_logger('evaluate')

    seed = cfg.get('seed', 42)
    set_seed(seed)

    dev_str = args.device or cfg.get('device', 'auto')
    device  = (torch.device('cuda' if torch.cuda.is_available() else 'cpu')
               if dev_str == 'auto' else torch.device(dev_str))

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    emb_dim  = cfg['model']['emb_dim']
    img_size = cfg['data']['img_size']
    ecfg     = cfg.get('evaluation', {})
    tta_n    = ecfg.get('tta_n', 8)
    k_enroll = ecfg.get('k_enroll', 6)
    n_pairs  = ecfg.get('n_pairs', 10000)
    ds_root  = Path(args.data_root or 'datasets')

    # ── model ─────────────────────────────────────────────────────────────────
    backbone = MultiScaleResNet34(emb_dim).to(device)
    backbone.load_state_dict(torch.load(ckpt_path, map_location=device))
    backbone.eval()
    log.info(f'Loaded backbone from {ckpt_path}')

    tta_tfm = get_tta_transforms(img_size)

    # ── scan + split ──────────────────────────────────────────────────────────
    log.info('Scanning datasets...')
    all_results: dict = {}

    eval_sets = []
    if 'cedar' in args.datasets:
        df_cedar = scan_cedar(ds_root / 'CEDAR')
        _, _, cedar_test = split_writers(df_cedar['writer_uid'].unique(), seed=seed)
        eval_sets.append(('CEDAR', df_cedar, cedar_test))

    if 'gpds150' in args.datasets:
        df_gpds = scan_gpds150(ds_root / 'GPDS150')
        _, _, gpds_test = split_writers(df_gpds['writer_uid'].unique(), seed=seed)
        eval_sets.append(('GPDS150', df_gpds, gpds_test))

    if 'sigcomp2011' in args.datasets:
        df_sc11 = scan_sigcomp2011(ds_root / 'sigComp2011-trainingSet')
        _, _, sc11_test = split_writers(df_sc11['writer_uid'].unique(), seed=seed)
        eval_sets.append(('SigComp2011', df_sc11, sc11_test))

    if 'sigcomp2009' in args.datasets:
        df_sc09e = scan_sigcomp2009_eval(ds_root / 'SigComp2009-evaluation')
        # all 28 writers are evaluation-only (never seen during training)
        sc09e_wids = set(df_sc09e['writer_uid'].unique())
        if sc09e_wids:
            eval_sets.append(('SigComp2009', df_sc09e, sc09e_wids))
        else:
            log.warning('SigComp2009-evaluation: no writers with both genuine and forgery found')

    if 'private' in args.datasets:
        df_prv = scan_private(ds_root / 'private_signature_verification')
        _, _, prv_test = split_writers(df_prv['writer_uid'].unique(), seed=seed)
        eval_sets.append(('Private', df_prv, prv_test))

    # ── evaluate ──────────────────────────────────────────────────────────────
    all_proto_sims, all_proto_ys = [], []

    for name, df, test_wids in eval_sets:
        log.info(f'Building {name} embedding cache (TTA={tta_n})...')
        cache = build_embedding_cache(df, test_wids, backbone, device, tta_tfm, n_views=tta_n)
        print(f'=== {name} (n_test_writers={len(test_wids)}) ===')
        pairs_n = 3000 if name == 'SigComp2011' else n_pairs
        res_pw,  gap_pw,  sims_pw,  y_pw  = run_eval_pairwise( name, df, test_wids, cache, n_pairs=pairs_n)
        res_pro, gap_pro, sims_pro, y_pro = run_eval_prototype(name, df, test_wids, cache, n_enroll=k_enroll)
        all_results[name] = {'pairwise': res_pw, 'prototype': res_pro}
        if len(sims_pro):
            all_proto_sims.append(sims_pro); all_proto_ys.append(y_pro)

    # ── combined ──────────────────────────────────────────────────────────────
    if len(all_proto_sims) > 1:
        sims_c = np.concatenate(all_proto_sims); y_c = np.concatenate(all_proto_ys)
        res_c  = compute_metrics(y_c, sims_c)
        gap_c  = float(sims_c[y_c == 1].mean() - sims_c[y_c == 0].mean())
        print(f'\n=== Combined prototype+TTA ({len(sims_c):,} pairs) ===')
        print(f'  EER={res_c["eer"]:.2%}  AUC={res_c["auc"]:.4f}  '
              f'Acc={res_c["accuracy"]:.2%}  gap={gap_c:.4f}')

    # ── summary table ─────────────────────────────────────────────────────────
    print('\n' + '─' * 78)
    print(f'{"Dataset":<14} {"EER pair+TTA":>12} {"AUC pair":>9} {"EER proto+TTA":>13} {"AUC proto":>9}')
    print('─' * 78)
    for name, r in all_results.items():
        pw  = r['pairwise']; pro = r['prototype']
        eer_pw  = f'{pw["eer"]:.2%}'  if pw["eer"] == pw["eer"]  else 'N/A'
        eer_pro = f'{pro["eer"]:.2%}' if pro["eer"] == pro["eer"] else 'N/A'
        auc_pw  = f'{pw["auc"]:.4f}'  if pw["auc"] == pw["auc"]  else 'N/A'
        auc_pro = f'{pro["auc"]:.4f}' if pro["auc"] == pro["auc"] else 'N/A'
        print(f'{name:<14} {eer_pw:>12} {auc_pw:>9} {eer_pro:>13} {auc_pro:>9}')
    print('─' * 78)


if __name__ == '__main__':
    main()
