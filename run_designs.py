'''Run Monte Carlo designs from a fresh Python process and store every cell (see src/store.py).

    python run_designs.py D2                      # all cells of D2, R = 500, 4 workers
    python run_designs.py D5 --cells 0 1 --R 200  # selected cells
    python run_designs.py D5 --dry-run            # list cells and stored progress, run nothing
    python run_designs.py D5 --time               # time one replication of the most expensive cell

Rerunning the same command continues where it stopped; finished replications are never recomputed.
'''
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
import registry  # noqa: E402
import store  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('design', choices=[f'D{i}' for i in range(1, 9)])
    ap.add_argument('--R', type=int, default=500)
    ap.add_argument('--seed', type=int, default=90)
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--chunk', type=int, default=50)
    ap.add_argument('--cells', type=int, nargs='*')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--time', action='store_true')
    a = ap.parse_args(argv)

    specs = registry.cells(a.design)
    idx = a.cells if a.cells is not None else range(len(specs))
    build = registry.build(a.design)

    if a.time:
        spec = specs[max(idx, key=lambda i: specs[i].get('N', 0) * specs[i].get('J', 1))]
        rep_fn, kw, _ = build(spec)
        t0 = time.time()
        rep_fn(np.random.default_rng(0), **kw)
        s = time.time() - t0
        print(f'{a.design} {spec}: {s:.1f} s per replication; '
              f'about {s * a.R * len(idx) / a.jobs / 3600:.2f} h for {len(idx)} cells x R = {a.R} on {a.jobs} workers '
              f'(upper bound: this is the most expensive cell)')
        return

    env = store.environment()
    if env['dirty'] and not a.dry_run:
        print('warning: uncommitted changes in tracked files; stored rows will carry a commit that does not '
              'contain the code that produced them. Commit first for the final run.', flush=True)
    for i in idx:
        spec, cid = specs[i], store.cell_id(specs[i], a.seed)
        n_done = len(store.done(a.design, cid))
        print(f'[{a.design} cell {i}] {cid} {spec}: {n_done}/{a.R} stored', flush=True)
        if a.dry_run or n_done >= a.R:
            continue
        t0 = time.time()
        df = store.run_cell(a.design, spec, build, R=a.R, seed=a.seed, n_jobs=a.jobs, chunk=a.chunk)
        bad = (df.status != 'ok').sum()
        print(f'    done in {(time.time() - t0) / 60:.1f} min; {len(df)} rows, {bad} not ok', flush=True)


if __name__ == '__main__':
    main()
