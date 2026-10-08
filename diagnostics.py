import numpy as np
import estimators as est
import diagnostics as diag
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

def coefs(p, shape = "decay", par=1.0, cos=1.0, seed=0):
    """
    args: 
    p (int): number of coefficients
    shape (str): "decay" or "equal"
    par (float): parameter for the shape of the coefficients
    cos (float): cosine of the angle between gamma and delta
    seed (int): random seed for reproducibility

    output:
    gamma (array): coefficient vector for the treatment model
    delta (array): coefficient vector for the outcome model
    
    description:
    Function that returns two coefficient vectors gamma and delta with unit norm,
    one on the same support as the other. 'decay': gamma_j ∝ j^-par. 
    'equal': 1/sqrt(s) on the first s = par coordinates.
    """
    u = np.arange(1, p + 1) ** -float(par) if shape == "decay" else (np.arange(p) < par).astype(float)
    u /= np.linalg.norm(u)
    if cos == 1:
        return u, u.copy()
    sg = np.random.default_rng(seed).permutation(np.resize([1.0, -1.0], p))
    w = u * sg
    w -= (w @ u) * u
    return u, cos * u + np.sqrt(1 - cos ** 2) * w / np.linalg.norm(w)


def core_tail(p, s=5, tau2= 0.8, dense = False):
    """
    args: 
    p (int): number of coefs
    s (int): number of non-zero coefs in the core
    tau2 (float): proportion of the norm in the tail
    dense (bool): whether to have a dense tail or not

    output:
    v (array): coefficient vector with unit norm
    description:
    Function that returns a coefficient vector v with unit norm,
    sparse core on first s coordinates.
    If dense, then the share tau2 of the norm sits in an equal-weight tail.
    """
    # a. initialize vector v with zeros
    v = np.zeros(p)
    # b. set first s number of coordinates to 1/sqrt(s)
    v[:s] = 1 / np.sqrt(s)
    # i. if dense, set the remaining coordinates to sqrt(tau2 / (p - s))
    if dense:
        v *= np.sqrt(1 - tau2)
        v[s:] = np.sqrt(tau2 / (p - s))
    return v


def naive_bias(gamma, delta, a=0.0):
    """
    args: 
    gamma (array): coefficient vector for the treatment model
    delta (array): coefficient vector for the outcome model
    a (float): intercept for the treatment model

    output:
    float: naive bias of the DML-DiD estimator
    description:
    Function that returns the bias of the DML-DiD estimator when using the naive estimator.
    X ~ N(0, I): 
    E[Lambda'] gamma'delta / (p0 (1 - p0)).
    """
    z, w = np.polynomial.hermite_e.hermegauss(80)
    w = w / w.sum()
    h = expit(a + np.linalg.norm(gamma) * z)
    p0 = w @ h
    return w @ (h * (1 - h)) * (gamma @ delta) / (p0 * (1 - p0))

def gauss_rep(rng, 
              N, 
              gamma, 
              delta, a=0.0, 
              sigma=1.0, 
              sigma3=0.0, 
              K=5, 
              learners=("plugin",), 
              support = None,
              n_test=5000):
    """
    args:
    rng: random number generator
    N (int): number of observations
    gamma (array): coefficient vector for the treatment model
    delta (array): coefficient vector for the outcome model
    a (float): intercept for the treatment model
    sigma (float): standard deviation of the noise in the outcome model
    sigma3 (float): standard deviation of the noise in the treatment effect
    K (int): number of folds for cross-validation
    learners (tuple): tuple of learners to use for estimating nuisance parameters
    support (array): optional array of indices for the support of the coefficients
    n_test (int): number of observations in the test set
    
    output:
    rows (list): list of dictionaries with bias, L2 errors, 
    support selection, and kappa for each learner
    
    description:
    function that generates a dataset 
    and evaluates the performance of different learners for the DML-DiD estimator.
    """
    p = len(gamma)
    X = rng.standard_normal((N, p))
    g0 = expit(a + X @ gamma)
    l0= 1 + X @ delta
    D = rng.binomial(1, g0)
    dY = l0 + sigma * rng.standard_normal(N) + D * (theta0 + sigma3 * rng.standard_normal(N))
    Xt = np.random.default_rng(999).standard_normal((n_test, p))
    g0t, l0t = expit(a + Xt @ gamma), 1 + Xt @ delta
    folds = est.make_folds(N, K, rng)
    rows = []
    for lr in learners:
        th, se, f = est.dmldid(X, D, dY, folds, lr, Xt)
        rows.append(est._row(f"dml_{lr}", (th, se), **diag.measure(f, g0t, l0t)))
    rows += [est._row("oracle", est.oracle(D, dY, g0, l0, folds)), 
             est._row("unadjusted", est.unadjusted(D, dY)),
             est._row("or_did", est.or_did(X, D, dY))]
    if support is not None and len(support) < 0.3 * N:
        rows.append(est._row("oracle_support", 
                    est.dmldid(X[:, support], D, dY, folds, "mle")[:2]))
    return rows