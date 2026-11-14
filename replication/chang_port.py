"""Python port of Chang's (2020) Table 2 procedure for DML-DiD with Lasso first steps.

It follows "Table 2/DMLML.R" in Chang's replication repository step by step:
K = 2 folds, penalties chosen by 10-fold cross-validation with the
one-standard-error rule (cv.glmnet's lambda.1se), no refit, 100 random splits,
and the mean over the splits as the estimate. Three steps of his code can be
switched off:
    intercept=False    his code predicts g and l2 WITHOUT the fitted intercepts
    drop_scores=True   his code discards scores s >= |min(s)| in each fold
    trim="drop"        his code drops observations with g outside (0.03, 0.97)

The random splits are not the ones R draws, so the numbers differ a little from
his (-5.222). For the exact numbers use replication/chang_variants.R.

Run from the repository root:   python replication/chang_port.py [number of splits]
"""
import sys, time, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, lasso_path
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")
seed = 666                                                            # the seed of the notebook


def logit(x):
    return 1/(1 + np.exp(-x))


# ---------------------------------------------------------------- Chang's sample, N = 1084
raw = pd.read_stata("Bribes_Regression.dta", convert_categoricals=False)
chang_cols = ["post_2008","tariff2007","lvalue_tonnage","differentiated","agri","perishable","dfs","day_w_arrival","monitor"]
other_cols = ["psi","rsa","clear_agent","hc_group","hc_4digits"]
sample = raw[["lba","tariff_change_2008","post_2008"]+chang_cols[1:]+other_cols].dropna().reset_index(drop=True)
Y = sample["lba"].to_numpy(float); D = sample["tariff_change_2008"].to_numpy(float); T = sample["post_2008"].to_numpy(float)
X_chang = sample[chang_cols].to_numpy(float)

# ---------------------------------------------------------------- cv.glmnet in Python
def path_fit(Z, y, lams, family):
    """Lasso fits for a decreasing grid of penalties. Returns intercepts (L,) and coefficients (L, p) for standardised Z."""
    n = len(y)
    warnings.filterwarnings("ignore")                                 # newer scikit-learn versions warn about penalty="l1"
    if family == "gaussian":
        _, coefs, _ = lasso_path(Z, y - y.mean(), alphas=lams)        # (p, L), same order as lams
        return np.full(len(lams), y.mean()), coefs.T
    model = LogisticRegression(penalty="l1", solver="saga", warm_start=True, max_iter=2000, tol=1e-5)
    a, B = [], []
    for lam in lams:                                                  # warm start along the path
        model.C = 1/(n*lam)
        model.fit(Z, y)
        a.append(model.intercept_[0]); B.append(model.coef_[0].copy())
    return np.array(a), np.array(B)

def deviance(y, eta, family):
    if family == "gaussian":
        return np.mean((y - eta)**2)
    pr = np.clip(logit(eta), 1e-12, 1 - 1e-12)
    return -2*np.mean(y*np.log(pr) + (1 - y)*np.log(1 - pr))

def cv_lasso(X, y, family, rng, n_lambda=100, n_folds=10):
    """Lasso with the penalty chosen by 10-fold cross-validation, the way cv.glmnet does it.
    Returns {'1se': (intercept, coefs), 'min': (intercept, coefs)} on the raw scale of X."""
    n = len(y)
    mean, sd = X.mean(0), X.std(0); sd[sd == 0] = 1.0
    Z = (X - mean)/sd
    lam_max = np.max(np.abs(Z.T @ (y - y.mean())))/n                  # smallest penalty that sets every coefficient to zero
    lams = lam_max*np.logspace(0, -4, n_lambda)                       # glmnet's grid: down to 1e-4 of it
    folds = np.array_split(rng.permutation(n), n_folds)
    dev = np.empty((n_folds, n_lambda))
    for f, test in enumerate(folds):
        train = np.setdiff1d(np.arange(n), test)
        a, B = path_fit(Z[train], y[train], lams, family)
        for j in range(n_lambda):
            dev[f, j] = deviance(y[test], a[j] + Z[test] @ B[j], family)
    cvm, cvsd = dev.mean(0), dev.std(0, ddof=1)/np.sqrt(n_folds)
    j_min = int(np.argmin(cvm))
    j_1se = int(np.flatnonzero(cvm <= cvm[j_min] + cvsd[j_min])[0])   # largest penalty within one SE of the best
    a, B = path_fit(Z, y, lams[:max(j_min, j_1se) + 1], family)       # final fit on all the data
    out = {}
    for name, j in (("1se", j_1se), ("min", j_min)):
        b = B[j]/sd
        out[name] = (a[j] - mean @ b, b)
    return out

# ---------------------------------------------------------------- Chang's Table 2 procedure
def chang_one_split(X, D, T, Y, rng, intercept=False, drop_scores=True, rule="1se", trim="drop"):
    N = len(Y)
    perm = rng.permutation(N)
    folds = [perm[:N//2], perm[N//2:]]                                # K = 2
    parts, n_sel = [], []
    for q in range(2):
        ev, aux = folds[q], folds[1 - q]
        a1, b1 = cv_lasso(X[aux], D[aux], "binomial", rng)[rule]
        if not intercept: a1 = 0.0                                    # his code uses fit$beta only
        g = logit(a1 + X[ev] @ b1)
        if trim == "drop":
            ev = ev[(g > 0.03) & (g < 0.97)]; g = logit(a1 + X[ev] @ b1)
        elif trim == "clip":
            g = np.clip(g, 0.03, 0.97)
        lam = T[aux].mean()
        ctrl = aux[D[aux] == 0]
        a2, b2 = cv_lasso(X[ctrl], (T[ctrl] - lam)*Y[ctrl], "gaussian", rng)[rule]
        if not intercept: a2 = 0.0
        l2 = a2 + X[ev] @ b2
        p_hat = D[ev].mean()
        w = (D[ev] - g)/(1 - g)/(lam*(1 - lam))/p_hat
        parts.append((ev, lam, p_hat, w, l2))
        n_sel.append((np.sum(b1 != 0), np.sum(b2 != 0)))
    return parts, np.mean(n_sel, axis=0)

def chang_estimate(X, D, T, Y, B=100, n_jobs=-1, **opts):
    children = np.random.SeedSequence([seed, 61]).spawn(B)
    res = Parallel(n_jobs=n_jobs)(delayed(chang_one_split)(X, D, T, Y, np.random.default_rng(c), **opts) for c in children)
    drop = opts.get("drop_scores", True)
    def keep(s): return s[s < abs(s.min())] if drop else s           # his line: s = s[which(s < abs(min(s)))]
    theta_l, n_drop = [], []
    for parts, _ in res:
        th = []
        for ev, lam, p_hat, w, l2 in parts:
            s = ((T[ev] - lam)*Y[ev] - l2)*w
            th.append(keep(s).mean()); n_drop.append(len(s) - len(keep(s)))
        theta_l.append(np.mean(th))
    theta = np.mean(theta_l)                                          # MEAN over the B splits
    var_l = []
    for parts, _ in res:
        v = []
        for ev, lam, p_hat, w, l2 in parts:
            G = -(1 - 2*lam)*theta/(lam*(1 - lam)) - np.mean(Y[ev]*w)
            s = ((T[ev] - lam)*Y[ev] - l2)*w - D[ev]*theta/p_hat + G*(T[ev] - lam)
            v.append(np.mean(keep(s)**2))
        var_l.append(np.mean(v))
    return {"theta": theta, "se": np.sqrt(np.mean(var_l)/len(Y)), "median split": np.median(theta_l),
            "min split": np.min(theta_l), "max split": np.max(theta_l),
            "sel g": np.mean([r[1][0] for r in res]), "sel l": np.mean([r[1][1] for r in res]), "dropped per fold": np.mean(n_drop)}

if __name__ == "__main__":
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    rows = {}
    for name, opts in [("Chang as coded", dict()),
                       ("keep all scores", dict(drop_scores=False)),
                       ("add intercepts", dict(intercept=True)),
                       ("intercepts, keep all scores", dict(intercept=True, drop_scores=False)),
                       ("intercepts, all scores, clip", dict(intercept=True, drop_scores=False, trim="clip")),
                       ("intercepts, all scores, clip, lambda.min",
                        dict(intercept=True, drop_scores=False, trim="clip", rule="min"))]:
        t0 = time.time()
        rows[name] = chang_estimate(X_chang, D, T, Y, B=B, **opts)
        print(f"{name}: {time.time() - t0:.0f} s", flush=True)
    table = pd.DataFrame(rows).T.round(3)
    print(table.to_string())
    table.to_csv("replication/chang_port.csv")
