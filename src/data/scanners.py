"""Dataset scanners — one function per dataset, each returns a uniform DataFrame.

Columns: path (str), writer_uid (str), label ('genuine'|'forgery'), dataset (str).

Writer UID prefixes:
  cedar_NNN   GPDS150: gpds_NNN   SigComp2011: sc11d_NNN / sc11c_NNN
  sc09_NNN    sc09e_NNN (eval)    private: prv_NNNN
"""
import re
from pathlib import Path
import pandas as pd

_COLS = ['path', 'writer_uid', 'label', 'dataset']


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=_COLS)


def scan_cedar(root: str | Path) -> pd.DataFrame:
    root  = Path(root)
    pat_o = re.compile(r'^original_(\d+)_(\d+)\.png$',  re.I)
    pat_f = re.compile(r'^forgeries_(\d+)_(\d+)\.png$', re.I)
    rows  = []
    for fp in (root / 'full_org').iterdir():
        m = pat_o.match(fp.name)
        if m:
            rows.append({'path': str(fp), 'writer_uid': f'cedar_{int(m.group(1)):03d}',
                         'label': 'genuine', 'dataset': 'cedar'})
    for fp in (root / 'full_forg').iterdir():
        m = pat_f.match(fp.name)
        if m:
            rows.append({'path': str(fp), 'writer_uid': f'cedar_{int(m.group(1)):03d}',
                         'label': 'forgery', 'dataset': 'cedar'})
    return pd.DataFrame(rows, columns=_COLS)


def scan_gpds150(root: str | Path) -> pd.DataFrame:
    root  = Path(root)
    pat_g = re.compile(r'^c-(\d+)-',  re.I)
    pat_f = re.compile(r'^cf-(\d+)-', re.I)
    rows  = []
    for fp in (root / 'train' / 'genuine').iterdir():
        m = pat_g.match(fp.name)
        if m:
            rows.append({'path': str(fp), 'writer_uid': f'gpds_{int(m.group(1)):03d}',
                         'label': 'genuine', 'dataset': 'gpds150'})
    for fp in (root / 'train' / 'forge').iterdir():
        m = pat_f.match(fp.name)
        if m:
            rows.append({'path': str(fp), 'writer_uid': f'gpds_{int(m.group(1)):03d}',
                         'label': 'forgery', 'dataset': 'gpds150'})
    return pd.DataFrame(rows, columns=_COLS)


def scan_sigcomp2011(root: str | Path) -> pd.DataFrame:
    """SigComp2011 forgery fix: 7-digit filename {2d}{2d}{3d}_attempt.
    pat_forg captures the last 3 digits (target writer), not the full prefix.
    """
    root     = Path(root)
    pat_gen  = re.compile(r'^(\d{3})_\d+\.', re.I)
    pat_forg = re.compile(r'^\d{4}(\d{3})_\d+\.', re.I)
    rows     = []

    base = root / 'OfflineSignatures'
    if not base.exists():
        deeper = [
            d / 'OfflineSignatures'
            for d in root.iterdir()
            if d.is_dir() and (d / 'OfflineSignatures').exists()
        ]
        base = deeper[0] if deeper else None
    if base is None:
        return _empty()

    for lang, prefix in [('Dutch', 'sc11d'), ('Chinese', 'sc11c')]:
        ts = base / lang / 'TrainingSet'
        if not ts.exists():
            continue
        for fp in (ts / 'Offline Genuine').iterdir():
            m = pat_gen.match(fp.name)
            if m:
                rows.append({'path': str(fp), 'writer_uid': f'{prefix}_{m.group(1)}',
                             'label': 'genuine', 'dataset': 'sc11'})
        forg_dir = ts / 'Offline Forgeries'
        if forg_dir.exists():
            for fp in forg_dir.iterdir():
                m = pat_forg.match(fp.name)
                if m:
                    rows.append({'path': str(fp), 'writer_uid': f'{prefix}_{m.group(1)}',
                                 'label': 'forgery', 'dataset': 'sc11'})
    return pd.DataFrame(rows, columns=_COLS)


def scan_sigcomp2009(root: str | Path) -> pd.DataFrame:
    """SigComp2009: genuine only — used for training, no test split."""
    root   = Path(root)
    pat    = re.compile(r'^NISDCC-(\d+)_', re.I)
    rows   = []
    inner  = root / 'NISDCC-offline-all-001-051-6g' / 'NISDCC-offline-all-001-051-6g'
    target = inner if inner.exists() else root
    for fp in target.iterdir():
        if not fp.is_file():
            continue
        m = pat.match(fp.name)
        if m:
            rows.append({'path': str(fp), 'writer_uid': f'sc09_{int(m.group(1)):03d}',
                         'label': 'genuine', 'dataset': 'sc09'})
    return pd.DataFrame(rows, columns=_COLS)


def scan_sigcomp2009_eval(root: str | Path) -> pd.DataFrame:
    """SigComp2009-evaluation: NFI pattern with both genuines and forgeries.

    Only returns writers that have BOTH genuine AND forgery images (28 writers).
    These are evaluation-only writers — never used for training.

    Filename pattern: NFI-{AAA}{BB}{CCC}.png
      AAA = 3-digit writer ID  BB = 2-digit session/forger  CCC = 3-digit sample
    """
    root = Path(root)
    pat  = re.compile(r'^NFI-(\d{3})\d{5}\.png$', re.I)
    rows = []

    gen_dir  = root / '6b_NFIgenuines' / 'genuines'
    forg_dir = root / '6b_NFIforgeries' / 'forgeries'

    if not gen_dir.exists() or not forg_dir.exists():
        return _empty()

    gen_rows: list[dict]  = []
    forg_rows: list[dict] = []

    for fp in gen_dir.iterdir():
        m = pat.match(fp.name)
        if m:
            gen_rows.append({'path': str(fp), 'writer_uid': f'sc09e_{m.group(1)}',
                             'label': 'genuine', 'dataset': 'sc09eval'})

    for fp in forg_dir.iterdir():
        m = pat.match(fp.name)
        if m:
            forg_rows.append({'path': str(fp), 'writer_uid': f'sc09e_{m.group(1)}',
                              'label': 'forgery', 'dataset': 'sc09eval'})

    gen_writers  = {r['writer_uid'] for r in gen_rows}
    forg_writers = {r['writer_uid'] for r in forg_rows}
    both = gen_writers & forg_writers

    rows = [r for r in gen_rows  if r['writer_uid'] in both]
    rows += [r for r in forg_rows if r['writer_uid'] in both]
    return pd.DataFrame(rows, columns=_COLS)


def scan_private(root: str | Path) -> pd.DataFrame:
    """Private dataset: original_{id}_{n}.jpg / forgeries_{id}_{n}.jpg."""
    root     = Path(root)
    pat_gen  = re.compile(r'^original_(\d+)_\d+\.jpg$',  re.I)
    pat_forg = re.compile(r'^forgeries_(\d+)_\d+\.jpg$', re.I)
    rows     = []
    for fp in (root / 'full_org').iterdir():
        m = pat_gen.match(fp.name)
        if m:
            rows.append({'path': str(fp), 'writer_uid': f'prv_{m.group(1)}',
                         'label': 'genuine', 'dataset': 'private'})
    for fp in (root / 'full_forg').iterdir():
        m = pat_forg.match(fp.name)
        if m:
            rows.append({'path': str(fp), 'writer_uid': f'prv_{m.group(1)}',
                         'label': 'forgery', 'dataset': 'private'})
    return pd.DataFrame(rows, columns=_COLS)
