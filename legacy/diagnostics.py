import numpy as np
import estimators as est
from scipy.special import expit
eps = 1e-6 # error tolerance
theta0 = 3 #match chang

def L2_norm(x):
    """
    args: 
    x (array): input array

    output:
    float: L2 norm of the input array

    description: 
    Function that returns the L2 norm of the input array x.
    """
    return np.sqrt(np.mean((x)**2))

def measure(info, g0t, l0t):
    """
    args: 
    info (dict): dictionary with info on first step estimates
    g0t: test-set truth 
    l0t: test-set truth

    output: dict with keys B1, err_g, err_l, sel_g, sel_l, kappa
    
    description:
    Lemma 4 on a large test draw: product bias B1 and L2 first-step errors, averaged over folds.
    """
    # a. p0 = mean of g0t
    p0 = g0t.mean()
    # b. clipping g
    gt = [np.clip(a, eps, 1 - eps) for a in info["gt"]]
    # c. get info of selected supports
    sel = np.array(info["sel"], float)
    # d. return dictionary with bias, L2 errors, support selection, and kappa
    return dict(
        # 1. bias
        B1 = np.mean([np.mean((a - g0t) * (b - l0t) / (p0 * (1 - a))) for a, b in zip(gt, info["lt"])]),
        # 2. L2 fejl
        err_g = np.mean([L2_norm(a - g0t) for a in gt]),
        err_l = np.mean([L2_norm(b - l0t) for b in info["lt"]]),
        # 3. support selection
        sel_g = sel[:, 0].mean(), sel_l=sel[:, 1].mean(),
        # 4. kappa = min(g, 1-g) over folds
        kappa = np.minimum(info["g"], 1 - info["g"]).min())


def decompose(D, dY, fa, f0, folds):
    """
    args:
    D (array): treatment vector with dim N
    dY (array): outcome vector with dim N
    fa (dict): dictionary with first step estimates for basis A
    f0 (dict): dictionary with first step estimates for basis A0
    folds (list): list of (train, test) tuples

    output:
    c1, c2, c3 (list): lists of the three terms in the decomposition of the bias

    description:
    Function that returns the three terms in the decomposition of the 
    bias for the DML-DiD estimator.
    """
    dg = fa["g"] - f0["g"]  # delta g
    dl = fa["l"] - f0["l"] # delta l
    vt = (1 - fa["g"]) * (1 - f0["g"]) # denominator for c1 and c3
    pk = f0["pk"] # fold-specific treatment probabilities
    terms = (-(1 - D) * dg / vt * (dY - f0["l"]), # c1
            -(D - f0["g"]) / (1 - f0["g"]) * dl, # c2
            (1 - D) * dg / vt * dl) # c3
    return [np.mean([(t / pk)[te].mean() for _, te in folds]) for t in terms]


def compare_bases(bases, D, dY, folds, 
                  learner, g0t = None, l0t = None, tests = None):
    """
    args:
    bases (dict): dictionary with basis matrices
    D (array): treatment vector
    dY (array): outcome vector
    folds (list): list of (train, test) tuples
    learner (str): name of the learner to use for estimating nuisance parameters
    g0t (array): optional test-set truth for g
    l0t (array): optional test-set truth for l
    tests (dict): optional dictionary with test dictionaries for each basis

    output:
    rows (list): list of dictionaries with bias,
    L2 errors, support selection, and kappa for each basis

    description:
    Function that compares the performance of different bases for the DML-DiD estimator.
    Same folds and tuning rule in every basis.
    first key is A0.
    """
    # a. fit DML-DiD for each basis
    fits = {k: est.dmldid(Q, D, dY, folds, learner, 
                          None if tests is None else tests[k]) for k, Q in bases.items()}
    th0, se0, f0 = next(iter(fits.values()))
    ths = np.array([v[0] for v in fits.values()])
    rows = []
    # b. decompose bias for each basis
    for k, (th, se, f) in fits.items():
        c1, c2, c3 = decompose(D, dY, f, f0, folds)
        r = est._row(f"dml_{learner}", (th, se), basis=k, c1=c1, c2=c2, c3=c3, R=(ths.max() - ths.min()) / se0)
        rows.append(r | (measure(f, g0t, l0t) if tests is not None else {}))
    return rows