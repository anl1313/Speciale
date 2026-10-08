import warnings
import first_steps as fs
import numpy as np
from sklearn.model_selection import KFold
warnings.filterwarnings("ignore")
theta0= 3 #match chang
eps = 1e-6 # error tolerance

learners = {"plugin": (fs.logit_plugin, fs.lasso_plugin), 
            "cv": (fs.logit_cv, fs.lasso_cv), 
            "ridge": (fs.logit_ridge, fs.ridge),
            "mle": (fs.logit_mle, fs.ols), 
            "const": (fs.const, fs.const)}

def make_folds(N, K, rng):
    """
    args: 
    N = sample size, 
    K = number of folds, 
    rng = some random number generator

    output: list of K tuples (train_index, test_index)

    description: 
    Function that returns a list of K tuples, 
    each containing the train and test indices for K-fold cross-validation. 
    The folds are shuffled and the random state is set using the provided random number generator.
    """
    return list(KFold(K, 
                      shuffle = True, 
                      random_state = int(rng.integers(2 ** 31))).split(np.empty(N)))


def _fold_p(D, folds):
    """
    args: 
    D (any): treatment vector, 
    folds (list): list of (train, test) tuples.
    
    output: 
    pk (array with floats): vector of fold-specific treatment probabilities for each observation.
    
    description:
    Function that returns a vector of fold-specific treatment probabilities for each observation."""
    # a. laver et tomt array til at gemme resultater
    pk = np.empty(len(D))
    # b. looper over hvert fold og tager gennemsnittet af D
    for _, te in folds:
        pk[te] = D[te].mean()
    return pk


def _theta_se(s, D, pk, folds):
    """
    args:
    s (array): score vector
    D (array): treatment vector
    pk (array): fold-specific treatment probabilities
    folds (list): list of (train, test) tuples
    
    output:
    th (float): estimated treatment effect,
    se (float): standard error of the estimated treatment effect.
    
    description:
    Function that returns the estimated treatment effect and its standard error.
    """
    # a. beregner gns af s over folds
    th = np.mean([s[te].mean() for _, te in folds])
    # b. beregner varians af s over folds 
    # s -th -(th*(D - pk)/pk) = s - th - th*(D-pk)/pk
    v = np.mean([((s[te] - th - th * (D[te] - pk[te]) / pk[te]) ** 2).mean() for _, te in folds])
    return th, np.sqrt(v / len(D))


def dmldid(Q, D, dY, folds, learner="plugin", Qt=None):
    """
    args:
    Q (array): covariate matrix
    D (array): treatment vector
    dY (array): outcome vector
    folds (list): list of (train, test) tuples
    learner (str): name of the learner to use for estimating nuisance parameters
    Qt (array): optional test dictionary for the measurement layer

    output:
    th (float): estimated treatment effect
    se (float): standard error of the estimated treatment effect
    out (dict): dictionary containing estimated nuisance parameters and selected supports

    description:
    Function that implements the cross-fitted DML-DiD estimator for repeated outcomes.
    """
    # a. vælger g og l 
    fg, fl = learners[learner]
    N = len(D)
    # b. laver tomme arrays til resultater
    g, l = np.empty(N), np.empty(N)
    out = dict(gt=[], lt=[], sel=[])
    # c. looper over hvert fold og estimerer g og l
    for tr, te in folds:
        pg, ng = fg(Q[tr], D[tr])
        c0 = tr[D[tr] == 0]
        pl, nl = fl(Q[c0], dY[c0])
        g[te], l[te] = pg(Q[te]), pl(Q[te])
        out["sel"].append((ng, nl))
        if Qt is not None:
            out["gt"].append(pg(Qt))
            out["lt"].append(pl(Qt))
    g, pk = np.clip(g, eps, 1 - eps), _fold_p(D, folds)
    th, se = _theta_se((dY - l) * (D - g) / (pk * (1 - g)), D, pk, folds)
    return th, se, dict(out, g=g, l=l, pk=pk)


def oracle(D, dY, g0, l0, folds):
    pk = _fold_p(D, folds)
    return _theta_se((dY - l0) * (D - g0) / (pk * (1 - g0)), D, pk, folds)


def unadjusted(D, dY):
    a, b = dY[D == 1], dY[D == 0]
    return a.mean() - b.mean(), np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))


def or_did(Q, D, dY):
    """Outcome-regression DiD (= imputation estimator in the 2x2): OLS of dY on Q among controls, impute treated."""
    X0, X1 = (np.c_[np.ones((D == v).sum()), Q[D == v]] for v in (0, 1))
    n0, k = X0.shape
    if k > 0.9 * n0:
        return np.nan, np.nan
    b = np.linalg.lstsq(X0, dY[D == 0], rcond=None)[0]
    r0, r1 = dY[D == 0] - X0 @ b, dY[D == 1] - X1 @ b
    x1 = X1.mean(0)
    V = r0 @ r0 / (n0 - k) * np.linalg.pinv(X0.T @ X0)
    return r1.mean(), np.sqrt(r1.var(ddof=1) / len(r1) + x1 @ V @ x1)

def _row(est, th_se, **kw):
    return dict(est = est, theta=th_se[0], se=th_se[1], **kw)
