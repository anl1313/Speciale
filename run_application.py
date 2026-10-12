'''Section 6: rebuild Chang's (2020) Table 2 DML-DiD (Lasso) estimate step by step and store the table.

    python run_application.py              # all steps, B = 100 splits, 4 workers (about 30 min on 4 cores)
    python run_application.py --B 10       # quick look
Writes results/application/decomposition-<B>-<stamp>.parquet and a JSON file with the run metadata.
'''
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
import application as ap  # noqa: E402
import store  # noqa: E402


def main(argv=None):
    a = argparse.ArgumentParser()
    a.add_argument('--B', type=int, default=100)
    a.add_argument('--jobs', type=int, default=4)
    a.add_argument('--steps', nargs='*', default=list(ap.STEPS))
    a = a.parse_args(argv)
    env = store.environment()
    if env['dirty']:
        print('warning: uncommitted changes in tracked files; commit first for the final run', flush=True)
    t0 = time.time()
    df = ap.decomposition(B=a.B, n_jobs=a.jobs, steps={k: ap.STEPS[k] for k in a.steps})
    out = ROOT / 'results' / 'application'
    out.mkdir(parents=True, exist_ok=True)
    stem = f'decomposition-{a.B}-{time.strftime("%Y%m%dT%H%M%S")}'
    df.assign(commit=env['commit']).to_parquet(out / f'{stem}.parquet', index=False)
    (out / f'{stem}.json').write_text(json.dumps(env | dict(B=a.B, jobs=a.jobs, seconds=round(time.time() - t0)),
                                                 indent=2, default=str))
    pd.set_option('display.width', 200)
    print(f"Chang (2020), Table 2, DML-DiD (Lasso): {ap.TABLE2['dml_lasso'][0]} ({ap.TABLE2['dml_lasso'][1]})")
    print(df[['step', 'description', 'theta', 'se', 'change', 'sd_splits', 'kept', 'nwarn']].round(3).to_string(index=False))
    print('saved', out / f'{stem}.parquet')


if __name__ == '__main__':
    main()
