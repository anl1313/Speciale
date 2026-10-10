import warnings
import numpy as np
from numpy import linalg as la
from scipy.stats import norm
from sklearn.linear_model import (Lasso, LassoCV, LinearRegression, LogisticRegression,
                                  LogisticRegressionCV, RidgeCV)
from sklearn.model_selection import KFold


warnings.filterwarnings('ignore')
eps = 1e-6  # propensity clip

# 1. model dictionary 
# unpenalized / tuned-by-CV fits: models_[method][binary]
models_ = {'ols':   (LinearRegression,
                    lambda: LogisticRegression(C=1e4, max_iter=5000)),
          'cv':    (lambda: LassoCV(cv=5),
                    lambda: LogisticRegressionCV(Cs=10, cv=5, penalty='l1', solver='liblinear',
                                                 intercept_scaling=100, scoring='neg_log_loss')),
          'ridge': (lambda: RidgeCV(alphas=np.logspace(-2, 5, 30)),
                    lambda: LogisticRegressionCV(Cs=10, cv=5, scoring='neg_log_loss', max_iter=2000))}

# 2. link function
def G(z):
    return 1.0 / (1.0 + np.exp(-z))

# 3. standardizer 
def standardize(Q):
    mu, sd = Q.mean(0), Q.std(0)
    sd[sd < 1e-12] = np.inf  # constant columns -> 0
    return lambda Z: (Z - mu) / sd


def bcch(X, y, q, loadings=True):
    '''BCCH (AME): pilot penalty from y - ybar, final penalty from the pilot residuals.
    Penalty on b_j is q psi_j, psi_j = sqrt(mean(e^2 X_j^2)); loadings=False uses max_j psi_j for all j
    (the AME slides). q = c Phi^-1(1 - alpha/2p) / sqrt(n). Returns the selected support.'''
    def lasso(e):
        psi = np.sqrt(np.mean(X ** 2 * e[:, None] ** 2, axis=0))
        psi = np.where(psi > 1e-12, psi if loadings else psi.max(), np.inf)
        m = Lasso(alpha=q).fit(X / psi, y)
        return m.predict(X / psi), np.flatnonzero(m.coef_)
    return lasso(y - lasso(y - y.mean())[0])[1]


def first_step(Q, y, binary, method='bcch', c=1.1, alpha=0.05, loadings=True):
    '''Fit g (binary: logit) or ell (linear) on standardized Q.
    bcch: plug-in (logit) Lasso, then unpenalized refit on the support (post-Lasso).
    Returns: predict function, no. of nonzero coefficients.'''
    s = standardize(Q)
    X = s(Q)
    n, p = X.shape
    S = np.arange(p)
    if method == 'bcch':
        q = c * norm.ppf(1 - alpha / (2 * p)) / np.sqrt(n)
        S = (np.flatnonzero(LogisticRegression(penalty='l1', C=2 / (n * q), solver='liblinear',
                                               intercept_scaling=100).fit(X, y).coef_)
             if binary else bcch(X, y, q, loadings))
        if len(S) == 0:
            return lambda Z: np.full(len(Z), y.mean()), 0
        method = 'ols'
    m = models_[method][binary]().fit(X[:, S], y)
    predict = (lambda Z: m.predict_proba(s(Z)[:, S])[:, 1]) if binary else (lambda Z: m.predict(s(Z)[:, S]))
    return predict, int(np.sum(m.coef_ != 0))


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


def dmldid(Q, D, dY, folds, method='bcch', Qt=None, **fs):
    '''DML-DiD, repeated outcomes, eq. (6.1). dY = Y(1) - Y(0).
    Qt: test dictionary; first steps are also evaluated there (for Lemma 4). fs: passed to first_step.'''
    N = len(D)
    ghat, ellhat = np.empty(N), np.empty(N)
    res = dict(gt=[], lt=[], nsel=[])
    for train, k in folds:
        g, ng = first_step(Q[train], D[train], True, method, **fs)
        c = train[D[train] == 0]  # controls in I_k^c
        ell, nl = first_step(Q[c], dY[c], False, method, **fs)
        ghat[k], ellhat[k] = g(Q[k]), ell(Q[k])
        res['nsel'].append((ng, nl))
        if Qt is not None:
            res['gt'].append(g(Qt))
            res['lt'].append(ell(Qt))
    ghat, phat = np.clip(ghat, eps, 1 - eps), fold_mean(D, folds)
    psi = (dY - ellhat) * (D - ghat) / (phat * (1 - ghat))
    return res | estimate(psi, D, phat, folds) | dict(ghat=ghat, ellhat=ellhat, phat=phat)


def dmldid_rcs(Q, D, T, Y, folds, method='bcch', **fs):
    '''DML-DiD, repeated cross sections, eq. (6.2); G_lambda with the nuisances held fixed.'''
    N = len(D)
    ghat, ellhat = np.empty(N), np.empty(N)
    for train, k in folds:
        g, _ = first_step(Q[train], D[train], True, method, **fs)
        c = train[D[train] == 0]
        ell, _ = first_step(Q[c], (T[c] - T[train].mean()) * Y[c], False, method, **fs)
        ghat[k], ellhat[k] = g(Q[k]), ell(Q[k])
    ghat, phat, lamhat = np.clip(ghat, eps, 1 - eps), fold_mean(D, folds), fold_mean(T, folds)
    h, w, r = lamhat * (1 - lamhat), (D - ghat) / (phat * (1 - ghat)), (T - lamhat) * Y - ellhat
    dpsi = -(Y + r * (1 - 2 * lamhat) / h) * w / h  # d psi / d lambda
    return estimate(r * w / h, D, phat, folds, corr=fold_mean(dpsi, folds) * (T - lamhat))


def oracle(D, dY, g0, l0, folds):
    '''Eq. (6.1) with the true nuisances: the sampling noise O_k alone.'''
    phat = fold_mean(D, folds)
    return estimate((dY - l0) * (D - g0) / (phat * (1 - g0)), D, phat, folds)


def unadjusted(D, dY):
    a, b = dY[D == 1], dY[D == 0]
    return dict(theta=a.mean() - b.mean(), se=np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))


def ordid(Q, D, dY):
    '''Outcome-regression DiD (= imputation in the 2x2): OLS of dY on Q among controls, impute treated.'''
    X0, X1 = (np.c_[np.ones((D == v).sum()), Q[D == v]] for v in (0, 1))
    n0, k = X0.shape
    if k > 0.9 * n0:
        return dict(theta=np.nan, se=np.nan)
    b = la.lstsq(X0, dY[D == 0], rcond=None)[0]
    r0, r1, x1 = dY[D == 0] - X0 @ b, dY[D == 1] - X1 @ b, X1.mean(0)
    V = r0 @ r0 / (n0 - k) * la.pinv(X0.T @ X0)
    return dict(theta=r1.mean(), se=np.sqrt(r1.var(ddof=1) / len(r1) + x1 @ V @ x1))
