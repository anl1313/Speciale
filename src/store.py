'''Results store for the Monte Carlo study.

Layout (one directory per design cell; nothing is ever overwritten):
    results/<design>/<cell_id>/cell.json                  cell specification (design parameters, seed, target R)
    results/<design>/<cell_id>/runs.jsonl                 one line per run: commit, versions, machine, reps, time
    results/<design>/<cell_id>/part-<first>-<last>-<stamp>.parquet   replication-level rows

cell_id is a hash of the cell specification and the root seed, so the same cell always maps to the same
directory. A run only computes replications not already stored (resumable); parts are written atomically per
chunk, so an interruption costs at most one chunk. load() refuses duplicated replication keys.
'''
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from importlib import metadata as ilm
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get('MC_RESULTS', Path(__file__).resolve().parents[1] / 'results'))
KEY = ['rep', 'est', 'basis']
PACKAGES = ('numpy', 'scipy', 'scikit-learn', 'pandas', 'joblib', 'pyarrow')


def cell_id(spec, seed):
    blob = json.dumps(dict(spec=spec, seed=seed), sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def cell_dir(design, cid, root=None):
    return Path(root or ROOT) / design / cid


def environment():
    '''Code version and software: Git commit (with a dirty flag), package versions, Python, OS, BLAS.'''
    repo = Path(__file__).resolve().parents[1]
    git = lambda *a: subprocess.run(['git', *a], cwd=repo, capture_output=True, text=True).stdout.strip()
    try:
        commit, dirty = git('rev-parse', 'HEAD'), bool(git('status', '--porcelain', '--untracked-files=no'))
    except OSError:
        commit, dirty = 'unknown', None
    versions = {}
    for p in PACKAGES:
        try:
            versions[p] = ilm.version(p)
        except ilm.PackageNotFoundError:
            versions[p] = None
    try:
        blas = {k: v.get('name') for k, v in np.show_config(mode='dicts')['Build Dependencies'].items()
                if k in ('blas', 'lapack')}
    except Exception:  # noqa: BLE001 (older numpy)
        blas = None
    return dict(commit=commit or 'unknown', dirty=dirty, python=sys.version.split()[0], platform=platform.platform(),
                machine=platform.node(), packages=versions, blas=blas)


def parts(design, cid, root=None):
    return sorted(cell_dir(design, cid, root).glob('part-*.parquet'))


def load(design, cid=None, root=None):
    '''All stored rows of one cell (or of every cell of a design). Raises on duplicated replication keys.'''
    base = Path(root or ROOT) / design
    cids = [cid] if cid else sorted(p.name for p in base.iterdir() if p.is_dir()) if base.exists() else []
    frames = [pd.read_parquet(f) for c in cids for f in parts(design, c, root)]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    dup = df.duplicated(['cell', *KEY], keep=False)
    if dup.any():
        raise ValueError(f'{dup.sum()} duplicated rows on {["cell", *KEY]} in {design}: '
                         f'{df.loc[dup, ["cell", *KEY]].drop_duplicates().head().to_dict("records")}')
    mixed = df.groupby('cell').commit.nunique()
    if (mixed > 1).any():
        print(f'warning: cells {list(mixed[mixed > 1].index)} contain rows from more than one commit')
    return df


def done(design, cid, root=None):
    '''Replication ids already stored (a replication is written whole, so presence means complete).'''
    ps = parts(design, cid, root)
    return set() if not ps else set(pd.concat([pd.read_parquet(f, columns=['rep']) for f in ps]).rep.unique())


def _write_part(df, path):
    tmp = path.with_suffix('.tmp')
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)  # atomic: a crash leaves no half-written part


def run_cell(design, spec, build, R=500, seed=90, n_jobs=-1, chunk=50, root=None, verbose=0):
    '''Run the missing replications of one cell and store them. build(spec) -> (rep_fn, kw, consts):
    consts are cell-level columns (e.g. the naive bias) added to every row. Returns all stored rows.'''
    import mc
    cid = cell_id(spec, seed)
    cdir = cell_dir(design, cid, root)
    cdir.mkdir(parents=True, exist_ok=True)
    cfile = cdir / 'cell.json'
    if cfile.exists():
        old = json.loads(cfile.read_text())
        if old['spec'] != json.loads(json.dumps(spec, default=str)):
            raise ValueError(f'{cfile} holds a different specification')
    else:
        cfile.write_text(json.dumps(dict(design=design, cell=cid, spec=spec, seed=seed), indent=2, default=str))
    todo = [i for i in range(R) if i not in done(design, cid, root)]
    if not todo:
        return load(design, cid, root)
    rep_fn, kw, consts = build(spec)
    env = environment()
    t0 = time.time()
    for j in range(0, len(todo), chunk):
        reps = todo[j:j + chunk]
        df = mc.run(rep_fn, seed=seed, n_jobs=n_jobs, reps=reps, verbose=verbose, **kw)
        params = {(f'p_{k}' if k in df.columns else k): (json.dumps(v) if isinstance(v, (list, dict)) else v)
                  for k, v in spec.items()}  # cell parameters as columns (prefixed only on a name clash)
        df = df.assign(design=design, cell=cid, commit=env['commit'], **consts, **params)
        _write_part(df, cdir / f'part-{reps[0]:05d}-{reps[-1]:05d}-{time.strftime("%Y%m%dT%H%M%S")}.parquet')
    with open(cdir / 'runs.jsonl', 'a') as fh:
        fh.write(json.dumps(env | dict(reps=[todo[0], todo[-1]], n_reps=len(todo), n_jobs=n_jobs,
                                       seconds=round(time.time() - t0, 1), finished=time.strftime('%Y-%m-%d %H:%M:%S')),
                            default=str) + '\n')
    return load(design, cid, root)


def export_csv(design, path=None, root=None):
    '''Optional CSV copy of all rows of a design (Parquet stays the primary store).'''
    df = load(design, root=root)
    path = Path(path or Path(root or ROOT) / f'{design}.csv')
    if path.exists():
        raise FileExistsError(f'{path} exists; choose another path')
    df.to_csv(path, index=False)
    return path
