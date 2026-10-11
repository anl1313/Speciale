'''Monte Carlo: one replication, diagnostics (Lemma 4, Prop. 10), parallel runs, summaries with MCSEs.

Every estimator call is wrapped: a failure becomes a row with status='error' and the message, never a value.
OR-DiD that cannot be computed gets status='infeasible'. Summaries use status == 'ok' rows and report the counts.
'''
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import dmldid as dd
from dgp import theta0, sim_rcs

Z95 = 1.959963984540054
DIAG = ('g_min', 'g_max', 'n_clip', 'w_max', 'cond_g', 'cond_l', 'nwarn', 'rankdef_g', 'rankdef_l')


def l2(f, axis=None):
    '''Sample analogue of ||f||_{P,2}.'''
    return np.sqrt(np.mean(f ** 2, axis=axis))


def lemma4(f, t, clip=dd.eps):
    '''Lemma 4 on the test draw t: B1 and first-step errors, averaged over folds.'''
    gt, lt = np.clip(f['gt'], clip, 1 - clip), np.array(f['lt'])  # (K, n_test)
    dg, dl = gt - t['g0'], lt - t['l0']
    nsel = np.mean(f['nsel'], axis=0)
    return dict(B1=np.mean(dg * dl / (t['g0'].mean() * (1 - gt))),
                err_g=l2(dg, axis=1).mean(), err_l=l2(dl, axis=1).mean(),
                nsel_g=nsel[0], nsel_l=nsel[1], kappa=np.minimum(f['ghat'], 1 - f['ghat']).min())


def prop10(D, dY, fa, f0, folds):
    '''Prop. 10, exact: theta(A) - theta(A0) = c1 (propensity) + c2 (outcome) + c3 (cross).'''
    dg, dl = fa['ghat'] - f0['ghat'], fa['ellhat'] - f0['ellhat']
    v = (1 - fa['ghat']) * (1 - f0['ghat'])
    c = (-(1 - D) * dg / v * (dY - f0['ellhat']),
         -(D - f0['ghat']) / (1 - f0['ghat']) * dl,
         (1 - D) * dg / v * dl)
    return {f'c{j + 1}': np.mean([(cj / f0['phat'])[k].mean() for _, k in folds]) for j, cj in enumerate(c)}


def _call(fn, *a, **k):
    '''Run one estimator; a failure is recorded, never turned into an estimate.'''
    t0 = time.perf_counter()
    try:
        out = dict(fn(*a, **k))
        out.setdefault('status', 'ok')
        out['error'] = ''
    except Exception as e:  # noqa: BLE001 (recorded, not hidden)
        out = dict(theta=np.nan, se=np.nan, status='error', error=f'{type(e).__name__}: {e}'[:300])
    out['runtime'] = time.perf_counter() - t0
    return out


def _row(est, basis, out, **extra):
    r = dict(est=est, basis=basis, theta=out.get('theta', np.nan), se=out.get('se', np.nan),
             status=out['status'], error=out['error'], runtime=out['runtime'])
    return r | out.get('diag', {}) | extra


def _specs(methods, fs):
    '''methods: names ('bcch') or (label, options) pairs, e.g. ('bcch_rr', dict(method='bcch', refit='ridge')).'''
    specs = {}
    for m in methods:
        label, spec = (m, dict(method=m)) if isinstance(m, str) else (m[0], dict(m[1]))
        specs[label] = fs | spec
    return specs


def replicate(rng, sim, bases, N, K=5, methods=('bcch',), n_test=5000, fs={},
              extras=('oracle', 'unadjusted', 'ordid'), **par):
    '''One draw: DML-DiD for every method x basis (same folds), plus the benchmarks in extras.'''
    try:
        d, t = sim(rng, N, **par), sim(np.random.default_rng(999), n_test, **par)
        B = bases(rng, d['X'], t['X'], **par)
        folds = dd.make_folds(N, K, rng)
    except Exception as e:  # noqa: BLE001
        return [dict(est='replication', basis=None, theta=np.nan, se=np.nan, status='error',
                     error=f'{type(e).__name__}: {e}'[:300], runtime=0.)]
    D, dY = d['D'], d['dY']
    rows = []
    for label, spec in _specs(methods, fs).items():
        fits = {b: _call(dd.dmldid, Q, D, dY, folds, Qt=Qt, **spec) for b, (Q, Qt) in B.items()}
        f0 = fits.get('A0')
        for b, f in fits.items():
            extra = dict(method=spec['method'])
            if f['status'] == 'ok':
                extra |= lemma4(f, t, spec.get('clip', dd.eps))
                if len(B) > 1 and f0 is not None and f0['status'] == 'ok':
                    extra |= prop10(D, dY, f, f0, folds)
            rows.append(_row(label, b, f, **extra))
    bench = {'oracle': lambda: dd.oracle(D, dY, d['g0'], d['l0'], folds),
             'unadjusted': lambda: dd.unadjusted(D, dY),
             'ordid': lambda: dd.ordid(B['A0'][0], D, dY),
             'abadie': lambda: dd.abadie(B['A0'][0], D, dY, **fs)}
    return rows + [_row(e, None, _call(bench[e]), method=e) for e in extras]


def replicate_rcs(rng, Qs, cg, cl, D=None, lam=.5, sigma=1., K=5, method='bcch', fs={}):
    '''D8: real covariates, dictionaries Qs = {name: Q} (first = A0, where cg, cl are sparse).'''
    d = sim_rcs(rng, next(iter(Qs.values())), cg, cl, D, lam, sigma)
    folds = dd.make_folds(len(d['D']), K, rng)
    rows = [_row(method, b, _call(dd.dmldid_rcs, Q, d['D'], d['T'], d['Y'], folds, method, **fs), method=method)
            for b, Q in Qs.items()]
    m = [d['Y'][(d['D'] == i) & (d['T'] == j)].mean() for i in (1, 0) for j in (1, 0)]
    unadj = _call(lambda: dict(theta=m[0] - m[1] - m[2] + m[3], se=np.nan))
    return rows + [_row('unadjusted', None, unadj, method='unadjusted')]


def run(rep, R=500, seed=90, n_jobs=-1, reps=None, verbose=0, **kw):
    '''Replications `reps` (default 0..R-1) of rep in parallel, one row per estimator x basis x replication.
    Replication i uses SeedSequence(seed, spawn_key=(i,)) = SeedSequence(seed).spawn(R)[i], so any subset of
    replications can be run or rerun on its own.'''
    reps = list(range(R)) if reps is None else [int(i) for i in reps]
    res = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(rep)(np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(i,))), **kw) for i in reps)
    df = pd.DataFrame([r | dict(rep=i, seed=seed) for i, rows in zip(reps, res) for r in rows])
    return add_inference(df)


def add_inference(df):
    '''Replication-level truth, error, 95% interval and coverage indicator (NaN without a usable SE).'''
    df = df.copy()
    ok = (df.status == 'ok') & np.isfinite(df.se)
    df['theta0'] = theta0
    df['err'] = df.theta - theta0
    df['ci_lo'] = (df.theta - Z95 * df.se).where(ok)
    df['ci_hi'] = (df.theta + Z95 * df.se).where(ok)
    df['cover'] = ((df.ci_lo <= theta0) & (theta0 <= df.ci_hi)).astype(float).where(ok)
    return df


def summary(df, by=('est',)):
    '''Per group: sampling quantities of theta-hat (bias, sd, rmse, coverage, mean SE) over the status == 'ok'
    replications, each with its Monte Carlo standard error (mcse_*: uncertainty of the reported summary itself,
    which shrinks with R; sd is the sampling spread of the estimator, which does not). Counts of failures included.'''
    d = df if 'status' in df else df.assign(status=np.where(df.theta.notna(), 'ok', 'error'))
    if 'cover' not in d:
        d = add_inference(d)
    by = list(by)
    counts = d.groupby(by, dropna=False).status.value_counts().unstack(fill_value=0)
    counts = counts.reindex(columns=['ok', 'error', 'infeasible'], fill_value=0).add_prefix('n_')
    d = d[d.status == 'ok'].copy()
    for c in ('c1', 'c2', 'c3'):
        if c in d:
            d[c] = d[c].abs()
    g = d.groupby(by, dropna=False)
    n, sd, rmse = g.theta.count(), g.theta.std(), g.err.apply(l2)
    cov, ncov = g.cover.mean(), g.cover.count()
    out = pd.DataFrame(dict(
        n=n, bias=g.err.mean(), mcse_bias=sd / np.sqrt(n), sd=sd, mcse_sd=sd / np.sqrt(2 * (n - 1)),
        rmse=rmse, mcse_rmse=g.err.apply(lambda e: np.std(e ** 2, ddof=1) / (2 * l2(e) * np.sqrt(len(e)))),
        cover=cov, mcse_cover=np.sqrt(cov * (1 - cov) / ncov), mean_se=g.se.mean(), se_sd=g.se.mean() / sd,
        med_bias=g.err.median(), sd_rob=g.theta.apply(lambda t: (t.quantile(.75) - t.quantile(.25)) / 1.349)))
    out['bias_sd'] = out.bias / out.sd
    extra = [c for c in ('naive', 'B1', 'err_g', 'err_l', 'nsel_g', 'nsel_l', 'kappa', 'c1', 'c2', 'c3') if c in d]
    return out.join(g[extra].mean()).join(counts, how='outer')


def diag_quantiles(df, by=('est',), cols=DIAG, q=(.5, .9, .99)):
    '''Quantiles of the per-replication numerical diagnostics (status == 'ok' rows).'''
    d = df[df.status == 'ok']
    cols = [c for c in cols if c in d]
    return d.groupby(list(by), dropna=False)[cols].quantile(list(q)).unstack()


def ranges(df, by=(), exclude=('full',)):
    '''Studentized range R(A) over the reference-coded bases (Sec. 4.2), per replication, summarized by group.
    'full' coding is excluded: it is not an invertible p x p reparametrization of A0's dictionary.'''
    d = df[(df.status == 'ok') & df.basis.notna() & ~df.basis.isin(exclude)]
    key = [*by, 'est', 'rep']
    g = d.groupby(key).theta
    R = (g.max() - g.min()) / d[d.basis == 'A0'].set_index(key).se
    return R.groupby(level=[*by, 'est']).describe(percentiles=[.5, .9])[['count', 'mean', '50%', '90%']]


def slopes(s, group=('est', 'a', 'cos')):
    '''Slope of log|bias/sd| on log N (Prop. 7). s: summary(...) with N in the index or columns.'''
    s = s.reset_index()
    return s.groupby(list(group)).apply(lambda z: np.polyfit(np.log(z.N), np.log(abs(z.bias_sd)), 1)[0])
