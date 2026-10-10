import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import dmldid as dd
from dgp import theta0, sim_rcs


def l2(f, axis=None):
    '''Sample analogue of ||f||_{P,2}.'''
    return np.sqrt(np.mean(f ** 2, axis=axis))


def lemma4(f, t):
    '''Lemma 4 on the test draw t: B1 and first-step errors, averaged over folds.'''
    gt, lt = np.clip(f['gt'], dd.eps, 1 - dd.eps), np.array(f['lt'])  # (K, n_test)
    dg, dl = gt - t['g0'], lt - t['l0']
    nsel = np.mean(f['nsel'], axis=0)
    return dict(B1=np.mean(dg * dl / (t['g0'].mean() * (1 - gt))),
                err_g=l2(dg, axis=1).mean(), err_l=l2(dl, axis=1).mean(),
                nsel_g=nsel[0], nsel_l=nsel[1], kappa=np.minimum(f['ghat'], 1 - f['ghat']).min())


def prop10(D, dY, fa, f0, folds):
    """
    args: 
    D (array): treatment assignment info
    dY (array): Y(1) - Y(0)
    fa (dict): fitted model with basis A
    f0 (dict): fitted model with basis A0
    folds (list): list of tuples containing train and test indices for cross-validation

    output: dict with c1, c2, c3 values

    description:
    Function that computes the components of the decomposition in Proposition 10,
    which quantifies the difference between the estimated treatment effect using 
    basis A and the true treatment effect
    """
    '''Prop. 10, exact: theta(A) - theta(A0) = c1 (propensity) + c2 (outcome) + c3 (cross).'''
    # delta g and delta ell
    dg = fa['ghat'] - f0['ghat'] 
    dl = fa['ellhat'] - f0['ellhat']
    # compute c1, c2, c3 as averages over folds
    v = (1 - fa['ghat']) * (1 - f0['ghat']) # denominator
    c = (-(1 - D)*dg / v * (dY - f0['ellhat']), 
         -(D - f0['ghat']) / (1 - f0['ghat']) * dl,
         (1 - D) * dg / v * dl)
    return {f'c{j + 1}': np.mean([(cj / f0['phat'])[k].mean() for _, k in folds]) for j, cj in enumerate(c)}


def replicate(rng, sim, bases, N, K=5, methods=('bcch',), n_test=5000, fs={}, **par):
    '''One draw: DML-DiD for every method x basis (same folds), plus oracle, unadjusted and OR-DiD.
    fs: first-step options, e.g. dict(loadings=False).'''
    d, t = sim(rng, N, **par), sim(np.random.default_rng(999), n_test, **par)
    B = bases(rng, d['X'], t['X'], **par)
    D, dY = d['D'], d['dY']
    folds = dd.make_folds(N, K, rng)
    rows = []
    for m in methods:
        fits = {b: dd.dmldid(Q, D, dY, folds, m, Qt, **fs) for b, (Q, Qt) in B.items()}
        f0 = fits['A0']
        rows += [dict(est=m, basis=b, theta=f['theta'], se=f['se']) | lemma4(f, t) | prop10(D, dY, f, f0, folds)
                 for b, f in fits.items()]
    rows += [dict(est='oracle') | dd.oracle(D, dY, d['g0'], d['l0'], folds),
             dict(est='unadjusted') | dd.unadjusted(D, dY),
             dict(est='ordid') | dd.ordid(B['A0'][0], D, dY)]
    return rows


def replicate_rcs(rng, Qs, cg, cl, D=None, lam=.5, sigma=1., K=5, method='bcch', fs={}):
    '''D8: real covariates, dictionaries Qs = {name: Q} (first = A0, where cg, cl are sparse).'''
    d = sim_rcs(rng, next(iter(Qs.values())), cg, cl, D, lam, sigma)
    folds = dd.make_folds(len(d['D']), K, rng)
    rows = [dict(est=method, basis=b) | dd.dmldid_rcs(Q, d['D'], d['T'], d['Y'], folds, method, **fs)
            for b, Q in Qs.items()]
    m = [d['Y'][(d['D'] == i) & (d['T'] == j)].mean() for i in (1, 0) for j in (1, 0)]
    return rows + [dict(est='unadjusted', theta=m[0] - m[1] - m[2] + m[3], se=np.nan)]


def run(rep, R=500, seed=90, n_jobs=-1, **kw):
    '''R replications of rep (replicate or replicate_rcs) in parallel; one row per estimator x basis x replication.'''
    seeds = np.random.SeedSequence(seed).spawn(R)
    res = Parallel(n_jobs=n_jobs)(delayed(rep)(np.random.default_rng(s), **kw) for s in seeds)
    return pd.DataFrame([r | dict(rep=i) for i, rows in enumerate(res) for r in rows])


def summary(df, by=('est',)):
    d = df.dropna(subset=['theta']).copy()
    d['err'] = d.theta - theta0
    d['cover'] = (d.err.abs() <= 1.96 * d.se).astype(float).where(d.se.notna())
    for c in ('c1', 'c2', 'c3'):
        if c in d:
            d[c] = d[c].abs()
    g = d.groupby(list(by), dropna=False)
    out = pd.DataFrame(dict(n=g.theta.count(), bias=g.err.mean(), sd=g.theta.std(),
                            rmse=g.err.apply(l2), cover=g.cover.mean(), se_sd=g.se.mean() / g.theta.std()))
    out['bias_sd'] = out.bias / out.sd
    out['med_bias'] = g.err.median()
    out['sd_rob'] = g.theta.apply(lambda t: (t.quantile(.75) - t.quantile(.25)) / 1.349)
    extra = [c for c in ('naive', 'B1', 'err_g', 'err_l', 'nsel_g', 'nsel_l', 'kappa', 'c1', 'c2', 'c3') if c in d]
    return out.join(g[extra].mean())


def ranges(df, by=(), exclude=('full',)):
    '''Studentized range R(A) over the bases (Sec. 4.2), per replication, summarized by group.'''
    d = df[df.basis.notna() & ~df.basis.isin(exclude)]
    key = [*by, 'est', 'rep']
    g = d.groupby(key).theta
    R = (g.max() - g.min()) / d[d.basis == 'A0'].set_index(key).se
    return R.groupby(level=[*by, 'est']).describe(percentiles=[.5, .9])[['mean', '50%', '90%']]


def slopes(s, group=('est', 'a', 'cos')):
    '''Slope of log|bias/sd| on log N (Prop. 7); decay path: about (1 - a)/(2a) for 1/2 < a < 1.'''
    s = s.reset_index()
    return s.groupby(list(group)).apply(lambda z: np.polyfit(np.log(z.N), np.log(abs(z.bias_sd)), 1)[0])
