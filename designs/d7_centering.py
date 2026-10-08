import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from scipy.special import expit
import runs as runs
import DGP as dgp
import diagnostics as diag
import estimators as est
theta0 = 3.0

def poly(X, c):
    """
    args:
    X (array): input array with shape (N, k)
    c (array): centering vector with shape (k,)

    output: 
    centered polynomial featuresup to degree 3 of a covariate matrix X
    """
    # drop column with 1s 
    return PolynomialFeatures(3, include_bias = False).fit_transform(X - c)
 
def center_rep(rng, N, 
               k=9, scale=1.0, 
               sigma=1.0, K=5, learner="plugin", n_test=5000):
    """
    args:
    rng (numpy.random.Generator): random number generator
    N (int): number of observations
    k (int): number of covariates
    scale (float): scale parameter for the logistic function
    sigma (float): standard deviation of the noise
    K (int): number of folds for cross-validation
    learner (str): type of learner to use
    n_test (int): number of test observations

    output:
    a dataframe with sim results.
    X_j = 1 + 6*Beta(1, 4): 
    skewed, mean 2.2 != median. 
    Truth sparse in the cubic dictionary centred at the mean.

    """
    # 1. define mean
    m = 2.2
    # 2. define a function to draw the data
    def draw(r, n):
        X = 1 + 6 * r.beta(1, 4, (n, k)) #i. draw covars according to 1+ 6*Beta(1, 4, shape=(n, k))
        Z = X - m # ii. center at the mean
        # 0,5*Z1 - 0,4*Z2 + 0,25*Z1*Z3
        g0 = expit(scale*(0.5*Z[:, 0]-0.4*Z[:, 1] + 0.25*Z[:, 0]*Z[:, 2])) 
        # 1 + 0,5*Z1 - 0,4*Z2 + 0,25*Z1*Z3 + 0,3*Z4^2 + 0,05*Z4*Z5*Z6
        l0 = 1 + 0.5*Z[:, 0] + 0.25*Z[:, 0]*Z[:, 2] + 0.3*Z[:, 3]**2 + 0.05*Z[:, 3]*Z[:, 4]*Z[:, 5] 
        return X, g0, l0
    # 3. draw data
    X, g0, l0 = draw(rng, N)
    D = rng.binomial(1, g0)
    dY = l0 + sigma * rng.standard_normal(N) + D * theta0
    Xt, g0t, l0t = draw(np.random.default_rng(999), n_test)
    # 4. define centering 
    cents = {"A0": np.full(k, m), 
             "raw": np.zeros(k), 
             "lower": np.ones(k), 
             "median": np.median(X, 0),
             "sample_mean": X.mean(0)}
    folds = est.make_folds(N, K, rng)
    # 5. compare different centering approaches
    rows = diag.compare_bases({c: poly(X, v) for c, v in cents.items()}, D, dY, folds, learner, g0t, l0t,
                         {c: poly(Xt, v) for c, v in cents.items()})
    return rows + [est._row("or_did", 
                            est.or_did(poly(X, m), D, dY), basis="any"), 
                            est._row("oracle", 
                                     est.oracle(D, dY, g0, l0, folds))]
 
 
def design_d7(N_grid=(500, 1000, 2000, 4000), R=500, **kw):
    """
    args:
    N_grid (tuple): grid of sample sizes
    R (int): number of repetitions

    output:
    dataframe with simulation results for each combination of N and repetition

    description:
    runs estimator with centering at the mean, median, 
    lower bound, and sample mean of the covariates.
    """
    return pd.concat([runs.run(center_rep, R, N=N, **kw).assign(N=N) for N in N_grid])