'''Design registry D1-D8: each design is a list of cell specifications (JSON-serialisable dicts) plus a builder
that turns a specification into (replication function, keyword arguments, cell-level constant columns).
The grids reproduce the 10 Oct notebook (legacy/simulations_vol5_outputs.ipynb) exactly.

    D1 chang     Chang (2020) DGP (l10 = 1), and the same gamma with a trend 1 + X'gamma
    D2 density   decay path gamma_j ~ j^-a, unit norm, delta = gamma
    D3 align     D2 with cos(gamma, delta) in {1, .5, 0}
    D4 nuis2x2   propensity / trend sparse or dense (core + equal-weight tail)
    D5 refcat    reference-category coding; J fixed (5) vs growing (N/20); hit = g, l, both
    D6 overlap   D5 machinery, J = 10, propensity index scaled
    D7 centring  cubic dictionary centred at the mean, 0, 1, sample median, sample mean
    D8 seq_rcs   repeated cross sections on Sequeira's covariates (real D), Chang's N = 1084 sample
'''
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

import dgp
import mc

DATA = Path(__file__).resolve().parents[1] / 'data' / 'Bribes_Regression.dta'
NAMES = dict(D1='chang', D2='density', D3='align', D4='nuis2x2', D5='refcat', D6='overlap', D7='centring',
             D8='seq_rcs')
EXTRAS = ['oracle', 'unadjusted', 'ordid']


def _code_gamma(p):
    return np.r_[np.arange(5, 0, -1) / 5, np.zeros(p - 5)]  # Chang's replication code


def cells(design):
    if design == 'D1':
        return ([dict(variant='chang', N=200, p=100, K=K, trend=False, sigma=.1, sigma3=.1) for K in (2, 5)] +
                [dict(variant='trend', N=N, p=N // 2, K=5, trend=True, sigma=float(np.sqrt(2.2)), sigma3=.1)
                 for N in (200, 1000, 2000)])
    if design == 'D2':
        return [dict(N=N, a=a, cos=1., ratio=.5) for N in (250, 500, 1000, 2000) for a in (2., 1., .75, .5)]
    if design == 'D3':
        return [dict(N=N, a=a, cos=c, ratio=.5) for N in (500, 1000) for a in (1., .5) for c in (1., .5, 0.)]
    if design == 'D4':
        return [dict(N=1000, p=500, dense_g=i, dense_l=j, tau2=.8) for i, j in product((False, True), repeat=2)]
    if design == 'D5':
        return [dict(N=N, J=J, rule=rule, hit=h, n_ref=10) for N in (500, 1000, 2000)
                for rule, J in (('fixed', 5), ('growing', N // 20)) for h in ('g', 'l', 'both')]
    if design == 'D6':
        return [dict(N=1000, J=10, scale=s, hit='both', n_ref=10) for s in (.5, 1., 2., 3.)]
    if design == 'D7':
        return [dict(N=N) for N in (500, 1000, 2000, 4000)]
    if design == 'D8':
        return [dict(D='real', s=5, seed_g=1, seed_l=2)]
    raise KeyError(design)


def build(design):
    '''Builder for one design: spec -> (rep_fn, kw, consts).'''
    def gauss(spec):
        if design == 'D1':
            g = _code_gamma(spec['p'])
            d = g.copy() if spec['trend'] else np.zeros(spec['p'])
            kw = dict(N=spec['N'], K=spec['K'], sigma=spec['sigma'], sigma3=spec['sigma3'],
                      extras=EXTRAS + ['abadie'])
        elif design == 'D4':
            g, d = (dgp.core_tail(spec['p'], tau2=spec['tau2'], dense=spec[k]) for k in ('dense_g', 'dense_l'))
            kw = dict(N=spec['N'])
        else:
            g, d = dgp.coefs(int(spec['ratio'] * spec['N']), 'decay', spec['a'], spec['cos'])
            kw = dict(N=spec['N'])
        return mc.replicate, dict(sim=dgp.sim_gauss, bases=dgp.bases_gauss, gamma=g, delta=d) | kw, \
            dict(naive=float(dgp.naive_bias(g, d)))

    def cat(spec):
        par = {k: spec[k] for k in ('N', 'J', 'hit', 'n_ref')} | ({'scale': spec['scale']} if 'scale' in spec else {})
        return mc.replicate, dict(sim=dgp.sim_cat, bases=dgp.bases_cat) | par, {}

    def centre(spec):
        return mc.replicate, dict(sim=dgp.sim_center, bases=dgp.bases_center, N=spec['N']), {}

    def seq(spec):
        X, D, lam = sequeira()
        S = dgp.resolve(dgp.poly(X, X.mean(0)))
        Qs = {k: dgp.poly(X, c)[:, S] for k, c in (('A0', X.mean(0)), ('raw', 0.), ('median', np.median(X, 0)))}
        cg = dgp.sparse_coefs(Qs['A0'], spec['s'], seed=spec['seed_g'])
        cl = dgp.sparse_coefs(Qs['A0'], spec['s'], seed=spec['seed_l'])
        return mc.replicate_rcs, dict(Qs=Qs, cg=cg, cl=cl, D=D if spec['D'] == 'real' else None, lam=lam), {}

    return {'D1': gauss, 'D2': gauss, 'D3': gauss, 'D4': gauss, 'D5': cat, 'D6': cat, 'D7': centre,
            'D8': seq}[design]


def sequeira(path=DATA):
    '''Chang's Table 2 sample: complete cases on his columns (N = 1084). X: his nine covariates minus post_2008
    (= T, simulated in D8). Weekday and monitor enter as numeric codes, as in his R code.'''
    keep = ['lba', 'tariff_change_2008', 'post_2008', 'tariff2007', 'lvalue_tonnage', 'differentiated', 'agri',
            'perishable', 'dfs', 'day_w_arrival', 'monitor', 'psi', 'rsa', 'clear_agent', 'hc_group', 'hc_4digits']
    df = pd.read_stata(path, convert_categoricals=False)[keep].dropna()
    return df[keep[3:11]].to_numpy(float), df.tariff_change_2008.to_numpy().astype(int), float(df.post_2008.mean())
