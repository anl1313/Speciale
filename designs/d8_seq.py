import numpy as np
import diagnostics as diag
import runs as runs
import pandas as pd
import estimators as est
from scipy.special import expit


def resolve(Q, order=None, tol=1e-8):
    """Collinearity resolution: keep columns greedily in `order` unless in span(1, kept). Different orders,
    different resolutions (KMR's first normalization choice)."""
    Z = Q - Q.mean(0)
    basis, keep = [], []
    for j in (range(Q.shape[1]) if order is None else order):
        v = Z[:, j].copy()
        for _ in range(2):
            for b in basis:
                v -= (b @ v) * b
        nv = np.linalg.norm(v)
        if nv > tol * max(np.linalg.norm(Z[:, j]), 1e-300):
            basis.append(v / nv)
            keep.append(j)
    return np.sort(keep)


def dmldid_rcs(Q, D, T, Y, folds, learner="plugin"):
    """Chang's repeated-cross-section score; p_k, lambda_k on the fold; G_lambda with eta held fixed."""
    fg, fl = learners[learner]
    N = len(D)
    g, l = np.empty(N), np.empty(N)
    for tr, te in folds:
        pg, _ = fg(Q[tr], D[tr])
        c0 = tr[D[tr] == 0]
        pl, _ = fl(Q[c0], (T[c0] - T[tr].mean()) * Y[c0])
        g[te], l[te] = pg(Q[te]), pl(Q[te])
    g, pk, lk = np.clip(g, eps, 1 - eps), est._fold_p(D, folds), est._fold_p(T, folds)
    h, w = lk * (1 - lk), (D - g) / (pk * (1 - g))
    s = ((T - lk) * Y - l) * w / h
    ds = -Y * w / h - ((T - lk) * Y - l) * w * (1 - 2 * lk) / h ** 2
    th = np.mean([s[te].mean() for _, te in folds])
    v = np.mean([((s[te] - th - th * (D[te] - pk[te]) / pk[te] + ds[te].mean() * (T[te] - lk[te])) ** 2).mean()
                 for _, te in folds])
    return th, np.sqrt(v / N)


def rcs_rep(rng, X, builders, cg, cl, D=None, lam=0.5, sigma=1.0, K=5, learner="plugin"):
    """Real X; T ~ Bernoulli(lam) independent; Y with known theta0 and trend sparse in A0 (first builder).
    D: real treatment (keeps real overlap) or None to simulate from the A0 propensity."""
    Qs = {k: b(X) for k, b in builders.items()}
    Q0 = next(iter(Qs.values()))
    if D is None:
        idx = Q0 @ cg
        D = rng.binomial(1, expit(idx - idx.mean()))
    T = rng.binomial(1, lam, len(X))
    Y = 0.5 * Q0 @ cl + 0.5 * D + T * (1 + Q0 @ cl) + D * T * theta0 + sigma * rng.standard_normal(len(X))
    folds = est.make_folds(len(X), K, rng)
    res = {k: dmldid_rcs(Q, D, T, Y, folds, learner) for k, Q in Qs.items()}
    ths, se0 = np.array([v[0] for v in res.values()]), next(iter(res.values()))[1]
    rows = [est._row(f"dml_{learner}", v, basis=k, R=(ths.max() - ths.min()) / se0) for k, v in res.items()]
    cell = [Y[(D == d) & (T == t)].mean() for d in (1, 0) for t in (1, 0)]
    return rows + [dict(est="unadjusted", basis="-", theta=cell[0] - cell[1] - cell[2] + cell[3], se=np.nan)]


def sparse_coefs(Q, s=5, scale=1.0, seed=0):
    """s random columns, random signs, scaled so the index has sd = scale."""
    r = np.random.default_rng(seed)
    c = np.zeros(Q.shape[1])
    idx = r.choice(Q.shape[1], s, replace=False)
    c[idx] = r.choice([-1.0, 1.0], s) / Q[:, idx].std(0)
    return c * scale / (Q @ c).std()


def design_d8(X, builders, D=None, s=5, scale_g=1.0, scale_l=1.0, lam=0.5, R=500, **kw):
    Q0 = next(iter(builders.values()))(X)
    cg, cl = sparse_coefs(Q0, s, scale_g, 1), sparse_coefs(Q0, s, scale_l, 2)
    return runs.run(rcs_rep, R, X=X, builders=builders, cg=cg, cl=cl, D=D, lam=lam, **kw)


