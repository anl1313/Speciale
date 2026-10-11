'''Audit of the extreme D5 draws reported on 10 Oct (reps 100, 170, 296, 98, 492), in their own cells.

For each replication: DML-DiD with the default OLS post-Lasso refit ('bcch') and the separately labelled ridge
refit ('bcch_rr', alpha = 1e-6), run twice to check reproducibility. Prints and saves theta-hat per basis with the
numerical diagnostics (condition numbers of the refit designs, propensity range, largest weight, warnings).

    python audit/d5_reps.py            -> audit/d5_reps_<machine>.csv
'''
import platform
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import dgp  # noqa: E402
import mc  # noqa: E402
import store  # noqa: E402

CASES = [(100, 2000, 100, 'l'), (296, 2000, 100, 'l'), (170, 2000, 100, 'g'), (98, 2000, 100, 'both'),
         (492, 500, 25, 'l')]  # (rep, N, J, hit) from legacy/simulations_vol5_outputs.ipynb
METHODS = ('bcch', ('bcch_rr', dict(method='bcch', refit='ridge')))
COLS = ['rep', 'N', 'hit', 'run', 'est', 'basis', 'theta', 'se', 'cond_l', 'cond_g', 'rankdef_l', 'rankdef_g', 'g_min', 'g_max', 'w_max',
        'nwarn', 'nsel_g', 'nsel_l', 'status']


def main():
    env = store.environment()
    print({k: env[k] for k in ('python', 'platform', 'packages', 'blas')})
    out = []
    for rep, N, J, hit in CASES:
        for run in (1, 2):
            df = mc.run(mc.replicate, reps=[rep], n_jobs=1, sim=dgp.sim_cat, bases=dgp.bases_cat, N=N, J=J, hit=hit,
                        n_ref=10, methods=METHODS, extras=('oracle',))
            out.append(df.assign(N=N, hit=hit, run=run))
    res = pd.concat(out)[COLS]
    path = ROOT / 'audit' / f'd5_reps_{platform.node() or "machine"}.csv'
    res.to_csv(path, index=False)
    pd.set_option('display.width', 220)
    show = res[res.basis.isin(['A0', 'bad']) | (res.est == 'oracle')]
    print(show.round(4).to_string(index=False))
    same = res.pivot_table(index=['rep', 'est', 'basis'], columns='run', values='theta')
    print('\nmax |run 1 - run 2| over all estimates:', float((same[1] - same[2]).abs().max()))
    print('extreme estimates (|theta - 3| > 10):', int(((res.theta - 3).abs() > 10).sum()))
    print('saved', path)


if __name__ == '__main__':
    main()
