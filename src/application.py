'''Section 6: Chang's (2020) Table 2 DML-DiD (Lasso) estimate on Sequeira (2016), rebuilt step by step.

chang_dml mirrors Table 2/DMLML.R of his replication package (github.com/NengChiehChang/Diff-in-Diff):
K = 2 halves, B = 100 random splits averaged, glmnet Lasso / logit Lasso with 10-fold CV and lambda.1se,
p = his nine covariates (post_2008 first). Each option undoes one departure from the estimator of his Theorem 3.1:
    intercept=False  predictions omit the fitted intercepts (his code: p %*% fit$beta)
    trim=(.03, .97)  only fold observations with g-hat in the band enter p-hat, theta_k and the variance
    score_trim       score values >= |min| are dropped (estimate and variance)
    lam_aux          lambda-hat from the auxiliary half instead of the fold
The first configuration in STEPS is his code; each later one removes one departure.
glmnet's CV folds and R's random splits cannot be reproduced in Python, so the first step matches Table 2 only
approximately (reported -5.222, se 2.647).
'''
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.special import expit
from sklearn.linear_model import Lasso, LassoCV, LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import KFold

import dmldid as dd
from registry import DATA

COLS = ['post_2008', 'tariff2007', 'lvalue_tonnage', 'differentiated', 'agri', 'perishable', 'dfs', 'day_w_arrival',
        'monitor']  # his p = X[, 4:12]; weekday and monitor enter as Stata codes (Friday = 1 ... Wednesday = 7)
CASES = ['lba', 'tariff_change_2008', 'psi', 'rsa', 'clear_agent', 'hc_group', 'hc_4digits']  # also in his filter
TABLE2 = dict(sequeira=(-2.928, .944), abadie_kernel=(-8.168, 3.072), dml_kernel=(-6.998, 3.752),
              abadie_lasso=(-6.432, 2.737), dml_lasso=(-5.222, 2.647))

STEPS = {  # name: (description, options); each step removes one departure from Theorem 3.1
    'chang_code':   ("Chang's replication code", dict()),
    'intercept':    ('+ intercepts in the predictions', dict(intercept=True)),
    'no_scoretrim': ('+ no score trimming', dict(intercept=True, score_trim=False)),
    'no_band':      ('+ no propensity band (Theorem 3.1, lambda from aux. half)',
                     dict(intercept=True, score_trim=False, trim=None)),
    'paper':        ('+ fold lambda: the estimator of Theorem 3.1', dict(intercept=True, score_trim=False, trim=None,
                                                                      lam_aux=False)),
    'no_T':         ('+ post_2008 (= T) removed from the covariates', dict(intercept=True, score_trim=False, trim=None,
                                                                         lam_aux=False, drop_T=True)),
    'bcch':         ('+ BCCH plug-in post-Lasso first steps (this thesis)',
                     dict(intercept=True, score_trim=False, trim=None, lam_aux=False, drop_T=True, learner='bcch')),
}


def load(path=DATA):
    '''Chang's sample: complete cases on every column of his X (N = 1084). Returns Y, D, T, X (his nine columns).'''
    df = pd.read_stata(path, convert_categoricals=False)[CASES[:2] + COLS + CASES[2:]].dropna()
    return (df.lba.to_numpy(float), df.tariff_change_2008.to_numpy(int), df.post_2008.to_numpy(float),
            df[COLS].to_numpy(float))


def _path(X, y, n=100, ratio=1e-4):
    lmax = np.max(np.abs(X.T @ (y - y.mean()))) / len(y)
    return np.geomspace(lmax, lmax * ratio, n)  # glmnet default: 100 values down to 1e-4 lambda_max (n > p)


def _one_se(err, lams):
    '''err: (folds, lambdas) CV error. Largest lambda within one standard error of the minimum (lambda.1se).'''
    m, se = err.mean(0), err.std(0, ddof=1) / np.sqrt(err.shape[0])
    j = np.argmin(m)
    return lams[np.flatnonzero(m <= m[j] + se[j])[0]]  # lams descending: first index = largest lambda


def _glmnet_like(X, y, binary, seed):
    '''Standardised Lasso / logit Lasso, 10-fold CV, lambda.1se. Returns (linear predictor with intercept,
    linear predictor without it as in Chang's code).'''
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-12] = np.inf
    Z, n = (X - mu) / sd, len(y)
    lams, cv = _path(Z, y), KFold(10, shuffle=True, random_state=seed)
    if binary:
        m = LogisticRegressionCV(Cs=1 / (.9 * n * lams), cv=cv, penalty='l1', solver='liblinear', random_state=0,
                                 intercept_scaling=100, scoring='neg_log_loss').fit(Z, y)
        lam = _one_se(-2 * m.scores_[1], lams)
        f = LogisticRegression(penalty='l1', C=1 / (n * lam), solver='liblinear', random_state=0,
                               intercept_scaling=100).fit(Z, y)
        b, b0 = f.coef_[0], f.intercept_[0]
    else:
        m = LassoCV(alphas=lams, cv=cv).fit(Z, y)
        lam = _one_se(m.mse_path_.T, m.alphas_)
        f = Lasso(alpha=lam).fit(Z, y)
        b, b0 = f.coef_, f.intercept_
    beta = b / sd                    # original scale
    a0 = b0 - mu @ beta              # original-scale intercept, the term Chang's predictions leave out
    return (lambda W: a0 + W @ beta), (lambda W: W @ beta)


def _nuisances(X, Y, D, T, train, fold, learner, intercept, seed):
    lam_aux = T[train].mean()
    c = train[D[train] == 0]
    if learner == 'bcch':
        g, _ = dd.first_step(X[train], D[train], True, 'bcch')
        ell, _ = dd.first_step(X[c], (T[c] - lam_aux) * Y[c], False, 'bcch')
        return g(X[fold]), ell(X[fold]), lam_aux
    g_full, g_bug = _glmnet_like(X[train], D[train].astype(float), True, seed)
    l_full, l_bug = _glmnet_like(X[c], (T[c] - lam_aux) * Y[c], False, seed)
    gi, li = (g_full, l_full) if intercept else (g_bug, l_bug)
    return expit(gi(X[fold])), li(X[fold]), lam_aux


def _split(b, X, Y, D, T, opt, seed):
    N = len(Y)
    idx = np.random.default_rng([seed, b]).permutation(N)
    halves = (idx[:N // 2], idx[N // 2:])
    out = []
    for q in (0, 1):
        fold, train = halves[q], halves[1 - q]
        g, ell, lam_aux = _nuisances(X, Y, D, T, train, fold, opt['learner'], opt['intercept'], 333)
        keep = np.ones(len(fold), bool) if opt['trim'] is None else (g > opt['trim'][0]) & (g < opt['trim'][1])
        f, g, ell = fold[keep], g[keep], ell[keep]
        lam = lam_aux if opt['lam_aux'] else T[fold].mean()
        p = D[f].mean()
        s = ((T[f] - lam) * Y[f] - ell) * (D[f] - g) / (1 - g) / (lam * (1 - lam)) / p
        out.append(dict(theta=dd._cut(s, opt['score_trim']).mean(), s=s, f=f, g=g, lam=lam, p=p, kept=keep.mean()))
    return out


def chang_dml(Y, D, T, X, B=100, trim=(.03, .97), score_trim=True, intercept=False, lam_aux=True, drop_T=False,
              learner='cv1se', seed=123, n_jobs=-1):
    '''DML-DiD for repeated cross sections as in Chang's Table 2 code (defaults) or with departures removed.
    SE as in his code: sqrt(mean over splits and halves of mean(s^2)) / sqrt(N), with his analytic G_lambda.'''
    if drop_T:
        X = X[:, 1:]
    opt = dict(trim=trim, score_trim=score_trim, intercept=intercept, lam_aux=lam_aux, learner=learner)
    res = Parallel(n_jobs=n_jobs)(delayed(_split)(b, X, Y, D, T, opt, seed) for b in range(B))
    thetas = np.array([np.mean([h['theta'] for h in r]) for r in res])
    theta = thetas.mean()
    var = []
    for r in res:
        for h in r:
            f, g, lam, p = h['f'], h['g'], h['lam'], h['p']
            G = -(1 - 2 * lam) * theta / (lam * (1 - lam)) - np.mean(Y[f] * (D[f] - g) / (1 - g) / (lam * (1 - lam)) / p)
            v = h['s'] - D[f] * theta / p + G * (T[f] - lam)
            var.append(np.mean(dd._cut(v, score_trim) ** 2))
    return dict(theta=theta, se=np.sqrt(np.mean(var)) / np.sqrt(len(Y)), sd_splits=thetas.std(ddof=1),
                kept=np.mean([h['kept'] for r in res for h in r]), B=B)


def decomposition(B=100, n_jobs=-1, steps=STEPS):
    '''Table for Section 6: one row per step, each removing one departure from Theorem 3.1.'''
    Y, D, T, X = load()
    rows = []
    for name, (label, o) in steps.items():
        r = chang_dml(Y, D, T, X, B=B, n_jobs=n_jobs, **o)
        rows.append(dict(step=name, description=label, **r))
    df = pd.DataFrame(rows)
    df['change'] = df.theta.diff()
    return df
