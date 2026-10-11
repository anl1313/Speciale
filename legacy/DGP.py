import numpy as np
import runs as runs
import diagnostics as diag
import pandas as pd
import estimators as est
from scipy.special import expit
theta0 = 3.0

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
    # 1. generate the first coefficient vector u
    u = np.arange(1, p + 1) ** -float(par) if shape == "decay" else (np.arange(p) < par).astype(float)
    # 2. normalize u to have unit norm
    u /= np.linalg.norm(u)
    # 3. generate the second coefficient vector v
    # i. if alignment is 1, return u and a copy of u
    if cos == 1:
        return u, u.copy()
    # ii. if alignment is not 1, generate a random vector w and orthogonalize it to u
    sg = np.random.default_rng(seed).permutation(np.resize([1.0, -1.0], p))
    w = u * sg
    w -= (w @ u) * u
    # 4. return u and a linear combination of u and w that has the desired cosine with u
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
    # 1. Gauss-Hermite quadrature points and weights
    z, w = np.polynomial.hermite_e.hermegauss(80)
    # 2. scale weights to sum to 1
    w = w / w.sum()
    # 3. compute the naive bias using the quadrature points and weights
    h = expit(a + np.linalg.norm(gamma) * z)
    p0 = w @ h
    # 4. return the naive bias
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
    # 1. define dimension of p
    p = len(gamma)
    # 2. generate covariates X, treatment probabilities g0, and outcome means l0
    X = rng.standard_normal((N, p))
    g0 = expit(a + X @ gamma)
    l0= 1 + X @ delta
    # 3. generate treatment D and outcome dY
    D = rng.binomial(1, g0)
    dY = l0 + sigma * rng.standard_normal(N) + D * (theta0 + sigma3 * rng.standard_normal(N))
    # 4. generate test data
    Xt = np.random.default_rng(999).standard_normal((n_test, p))
    g0t, l0t = expit(a + Xt @ gamma), 1 + Xt @ delta
    folds = est.make_folds(N, K, rng)
    rows = []
    # 5. loop over learners and evaluate performance
    for lr in learners:
        th, se, f = est.dmldid(X, D, dY, folds, lr, Xt) # i. dml-did
        rows.append(est._row(f"dml_{lr}", (th, se), **diag.measure(f, g0t, l0t)))
    rows += [est._row("oracle", est.oracle(D, dY, g0, l0, folds)), # ii. oracle
             est._row("unadjusted", est.unadjusted(D, dY)), # iii. unadjusted
             est._row("or_did", est.or_did(X, D, dY))] # iv. or-did
    if support is not None and len(support) < 0.3 * N:
        rows.append(est._row("oracle_support", # v. oracle with support
                    est.dmldid(X[:, support], D, dY, folds, "mle")[:2]))
    return rows