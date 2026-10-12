'''DML-DiD (Chang 2020) with Lasso first steps, the benchmarks, and per-fit diagnostics.

Changes from the 10 Oct version (estimator definitions unchanged unless an option is set):
  * liblinear is seeded (random_state=SEED); it shuffles data and was unseeded before.
  * convergence and numerical warnings are counted per fit instead of being silenced.
  * first_step returns (predict, info); info holds nsel, nwarn and the condition number of the refit design.
  * options, all off by default: post=False (penalized fit, no refit), refit='ridge' (ridge post-Lasso refit,
    a separately labelled estimator), clip (propensity clip, default 1e-6 as before).
  * the unpenalized linear (re)fit is MinNormOLS: identical to LinearRegression when the design has full rank;
    when it is rank-deficient (OLS not identified) it returns the minimum-norm solution with an explicit
    singular-value cutoff, so the result no longer depends on the LAPACK build. See audit/d5_extreme_reps.md.
'''
import warnings
import numpy as np
from numpy import linalg as la
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (Lasso, LassoCV, LogisticRegression,
                                  LogisticRegressionCV, Ridge, RidgeCV)
from sklearn.model_selection import KFold

warnings.filterwarnings('ignore', category=FutureWarning)  # sklearn API deprecation notices only
warnings.filterwarnings('ignore', message='.*penalty.*')    # sklearn 1.8: 'penalty' deprecation notes
eps = 1e-6  # default propensity clip
SEED = 0    # random_state for liblinear



class MinNormOLS:
    '''OLS with intercept. A rank-deficient design gets the minimum-norm least-squares solution, with singular
    values below rcond * (largest) treated as zero. rank_ records the numerical rank.'''
    def __init__(self, rcond=1e-10):
        self.rcond = rcond

    def fit(self, X, y):
        mx, my = X.mean(0), y.mean()
        self.coef_, _, self.rank_, _ = la.lstsq(X - mx, y - my, rcond=self.rcond)
        self.intercept_ = my - mx @ self.coef_
        return self

    def predict(self, X):
        return self.intercept_ + X @ self.coef_


# unpenalized / CV-tuned fits: models_[method][binary]
models_ = {'ols':   (MinNormOLS,
                     lambda: LogisticRegression(C=1e4, max_iter=5000)),
           'cv':    (lambda: LassoCV(cv=5),
                     lambda: LogisticRegressionCV(Cs=10, cv=5, penalty='l1', solver='liblinear', random_state=SEED,
                                                  intercept_scaling=100, scoring='neg_log_loss')),
           'ridge': (lambda: RidgeCV(alphas=np.logspace(-2, 5, 30)),
                     lambda: LogisticRegressionCV(Cs=10, cv=5, scoring='neg_log_loss', max_iter=2000))}


def G(z):
    return 1.0 / (1.0 + np.exp(-z))


def standardize(Q):
    mu, sd = Q.mean(0), Q.std(0)
    sd[sd < 1e-12] = np.inf  # constant columns -> 0
    return lambda Z: (Z - mu) / sd


def _fit(m, X, y):
    '''Fit m and count convergence / numerical warnings (they were silenced before).'''
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        m.fit(X, y)
    return m, sum(issubclass(x.category, (ConvergenceWarning, RuntimeWarning)) for x in w)


def _nanmax(x):
    x = np.asarray(x, float)
    return np.nan if np.all(np.isnan(x)) else np.nanmax(x)


def bcch(X, y, q, loadings=True):
    '''BCCH (AME): pilot penalty from y - ybar, final penalty from the pilot residuals.
    Penalty on b_j is q psi_j, psi_j = sqrt(mean(e^2 X_j^2)); loadings=False uses max_j psi_j for all j.
    q = c Phi^-1(1 - alpha/2p) / sqrt(n). Returns the final Lasso (fitted on X / psi), psi, no. of warnings.'''
    def lasso(e):
        psi = np.sqrt(np.mean(X ** 2 * e[:, None] ** 2, axis=0))
        psi = np.where(psi > 1e-12, psi if loadings else psi.max(), np.inf)
        m, nw = _fit(Lasso(alpha=q), X / psi, y)
        return m, psi, nw
    m, psi, n1 = lasso(y - y.mean())
    m, psi, n2 = lasso(y - m.predict(X / psi))
    return m, psi, n1 + n2


def first_step(Q, y, binary, method='bcch', c=1.1, alpha=0.05, loadings=True, post=True, refit='ols',
               refit_alpha=1e-6):
    '''Fit g (binary: logit) or ell (linear) on standardized Q.
    bcch: plug-in (logit) Lasso; post=True refits on the support: refit='ols' (default; logit refit C=1e4)
    or refit='ridge' (linear refit Ridge(refit_alpha); logit refit unchanged).
    Returns: predict function, dict(nsel, nwarn, cond) with cond the condition number of the refit design.'''
    s = standardize(Q)
    X = s(Q)
    n, p = X.shape
    S, nw = np.arange(p), 0
    if method == 'bcch':
        q = c * norm.ppf(1 - alpha / (2 * p)) / np.sqrt(n)
        if binary:
            m, nw = _fit(LogisticRegression(penalty='l1', C=2 / (n * q), solver='liblinear', intercept_scaling=100,
                                            random_state=SEED), X, y)
            f = lambda Z: m.predict_proba(Z)[:, 1]
        else:
            m, psi, nw = bcch(X, y, q, loadings)
            f = lambda Z: m.predict(Z / psi)
        S = np.flatnonzero(m.coef_)
        if not post:
            return (lambda Z: f(s(Z))), dict(nsel=len(S), nwarn=nw, cond=np.nan, rankdef=0)
        if len(S) == 0:
            return (lambda Z: np.full(len(Z), y.mean())), dict(nsel=0, nwarn=nw, cond=np.nan, rankdef=0)
        method = 'refit_ridge' if refit == 'ridge' else 'ols'
    model = Ridge(alpha=refit_alpha) if method == 'refit_ridge' and not binary else \
        models_['ols' if method == 'refit_ridge' else method][binary]()
    m, nw2 = _fit(model, X[:, S], y)
    refitted = method in ('ols', 'refit_ridge')
    cond = la.cond(X[:, S]) if refitted else np.nan
    rankdef = int(la.matrix_rank(X[:, S]) < len(S)) if refitted else 0
    predict = (lambda Z: m.predict_proba(s(Z)[:, S])[:, 1]) if binary else (lambda Z: m.predict(s(Z)[:, S]))
    return predict, dict(nsel=int(np.sum(m.coef_ != 0)), nwarn=nw + nw2, cond=cond, rankdef=rankdef)


def make_folds(N, K, rng):
    '''K random folds as a list of (auxiliary sample I_k^c, fold I_k).'''
    return list(KFold(K, shuffle=True, random_state=int(rng.integers(2 ** 31))).split(np.empty(N)))


def fold_mean(x, folds):
    m = np.empty(len(x))
    for _, k in folds:
        m[k] = x[k].mean()
    return m


def estimate(psi, D, phat, folds, corr=0.):
    '''theta = average over folds of the fold mean of psi (Def. 2c);
    Chang's variance with G_p = -theta/phat (+ corr: the G_lambda term for RCS).'''
    theta = np.mean([psi[k].mean() for _, k in folds])
    s = psi - theta - theta * (D - phat) / phat + corr
    return dict(theta=theta, se=np.sqrt(np.mean([np.mean(s[k] ** 2) for _, k in folds]) / len(D)))


def _diag(graw, clip, w, info_g, info_l):
    return dict(g_min=graw.min(), g_max=graw.max(), n_clip=int(np.sum((graw < clip) | (graw > 1 - clip))),
                w_max=np.abs(w).max(), cond_g=_nanmax([i['cond'] for i in info_g]),
                cond_l=_nanmax([i['cond'] for i in info_l]), nwarn=sum(i['nwarn'] for i in info_g + info_l),
                rankdef_g=sum(i['rankdef'] for i in info_g), rankdef_l=sum(i['rankdef'] for i in info_l))


def dmldid(Q, D, dY, folds, method='bcch', Qt=None, clip=eps, **fs):
    '''DML-DiD, repeated outcomes, eq. (6.1). dY = Y(1) - Y(0).
    Qt: test dictionary; first steps are also evaluated there (for Lemma 4). fs: passed to first_step.'''
    N = len(D)
    ghat, ellhat = np.empty(N), np.empty(N)
    res, ig, il = dict(gt=[], lt=[], nsel=[]), [], []
    for train, k in folds:
        g, i_g = first_step(Q[train], D[train], True, method, **fs)
        c = train[D[train] == 0]  # controls in I_k^c
        ell, i_l = first_step(Q[c], dY[c], False, method, **fs)
        ghat[k], ellhat[k] = g(Q[k]), ell(Q[k])
        ig.append(i_g)
        il.append(i_l)
        res['nsel'].append((i_g['nsel'], i_l['nsel']))
        if Qt is not None:
            res['gt'].append(g(Qt))
            res['lt'].append(ell(Qt))
    graw = ghat.copy()
    ghat, phat = np.clip(ghat, clip, 1 - clip), fold_mean(D, folds)
    w = (D - ghat) / (phat * (1 - ghat))
    return res | estimate((dY - ellhat) * w, D, phat, folds) | dict(
        ghat=ghat, ellhat=ellhat, phat=phat, diag=_diag(graw, clip, w, ig, il))


def _cut(x, on):
    '''Chang's score trimming (replication code only): keep values below |min|.'''
    return x[x < abs(x.min())] if on else x


def dmldid_rcs(Q, D, T, Y, folds, method='bcch', clip=eps, trim=None, score_trim=False, lam_aux=False, **fs):
    '''DML-DiD, repeated cross sections, eq. (6.2); G_lambda = fold mean of d psi/d lambda (Chang, Online Appendix).
    Defaults give the paper's estimator. Options reproducing departures in Chang's replication code:
      trim=(lo, hi)  use only fold observations with lo < g-hat < hi (p-hat and theta_k over those);
      score_trim     drop score values >= |min| before averaging, in the estimate and the variance;
      lam_aux        lambda-hat from the auxiliary sample instead of the fold.'''
    N = len(D)
    ghat, ellhat, laux = np.empty(N), np.empty(N), np.empty(N)
    ig, il = [], []
    for train, k in folds:
        g, i_g = first_step(Q[train], D[train], True, method, **fs)
        c = train[D[train] == 0]
        laux[k] = T[train].mean()
        ell, i_l = first_step(Q[c], (T[c] - T[train].mean()) * Y[c], False, method, **fs)
        ghat[k], ellhat[k] = g(Q[k]), ell(Q[k])
        ig.append(i_g)
        il.append(i_l)
    graw = ghat.copy()
    ghat = np.clip(ghat, clip, 1 - clip)
    lam = laux if lam_aux else fold_mean(T, folds)
    h = lam * (1 - lam)
    parts, thetas = [], []
    for _, k in folds:
        m = k if trim is None else k[(ghat[k] > trim[0]) & (ghat[k] < trim[1])]
        p = D[m].mean()
        w, r = (D[m] - ghat[m]) / (p * (1 - ghat[m])), (T[m] - lam[m]) * Y[m] - ellhat[m]
        s = r * w / h[m]
        thetas.append(_cut(s, score_trim).mean())
        parts.append((m, p, w, r, s))
    theta = np.mean(thetas)
    var = []
    for m, p, w, r, s in parts:
        dpsi = -(Y[m] + r * (1 - 2 * lam[m]) / h[m]) * w / h[m]
        v = s - theta - theta * (D[m] - p) / p + dpsi.mean() * (T[m] - lam[m])
        var.append(np.mean(_cut(v, score_trim) ** 2))
    w_all = (D - ghat) / (fold_mean(D, folds) * (1 - ghat))
    return dict(theta=theta, se=np.sqrt(np.mean(var) / N),
                diag=_diag(graw, clip, w_all, ig, il) | dict(kept=sum(len(x[0]) for x in parts) / N))


def oracle(D, dY, g0, l0, folds):
    '''Eq. (6.1) with the true nuisances: the sampling noise O_k alone.'''
    phat = fold_mean(D, folds)
    return estimate((dY - l0) * (D - g0) / (phat * (1 - g0)), D, phat, folds)


def abadie(Q, D, dY, method='bcch', clip=eps, **fs):
    '''Abadie (2005), eq. (3): penalized first step (no refit) on the full sample, no orthogonal correction.
    Chang (2020), Fig. 1, left panel. No standard error.'''
    g, _ = first_step(Q, D, True, method, **(fs | dict(post=False)))
    ghat = np.clip(g(Q), clip, 1 - clip)
    return dict(theta=np.mean(dY / D.mean() * (D - ghat) / (1 - ghat)), se=np.nan)


def unadjusted(D, dY):
    a, b = dY[D == 1], dY[D == 0]
    return dict(theta=a.mean() - b.mean(), se=np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))


def ordid(Q, D, dY):
    '''Outcome-regression DiD: OLS of dY on Q among controls, impute for the treated.
    Infeasible (status) when the regressors exceed 90% of the controls.'''
    X0, X1 = (np.c_[np.ones((D == v).sum()), Q[D == v]] for v in (0, 1))
    n0, k = X0.shape
    if k > 0.9 * n0:
        return dict(theta=np.nan, se=np.nan, status='infeasible')
    b = la.lstsq(X0, dY[D == 0], rcond=None)[0]
    r0, r1, x1 = dY[D == 0] - X0 @ b, dY[D == 1] - X1 @ b, X1.mean(0)
    V = r0 @ r0 / (n0 - k) * la.pinv(X0.T @ X0)
    return dict(theta=r1.mean(), se=np.sqrt(r1.var(ddof=1) / len(r1) + x1 @ V @ x1))
