import warnings
import numpy as np
import sklearn
from scipy.stats import norm
from sklearn.linear_model import (Lasso, LassoCV, LinearRegression, LogisticRegression, 
                                  LogisticRegressionCV, RidgeCV)


warnings.filterwarnings("ignore")
theta0= 3 #match chang
eps = 1e-6 # error tolerance
_SK18 = tuple(int(v) for v in sklearn.__version__.split(".")[:2]) >= (1, 8)
 

def _lr(kind, crossval=False, **kw):
    """
    args: 
    kind (str): "l1", "l2", "none"
    crossval (bool): whether to use cross-validation
    
    output: LogisticRegression or LogisticRegressionCV object

    description: Function that returns a logistic regression
    model based on the specified kind and cross-validation option.
    """
    if crossval:
        return (LogisticRegressionCV(l1_ratios=[float(kind == "l1")], **kw) if _SK18
                else LogisticRegressionCV(penalty=kind, **kw))
    if kind == "none":
        return LogisticRegression(C=np.inf, **kw) if _SK18 else LogisticRegression(penalty=None, **kw)
    return LogisticRegression(l1_ratio=float(kind == "l1"), **kw) if _SK18 else LogisticRegression(penalty=kind, **kw)

def _scaler(Q):
    """
    args: Q (n x p) matrix

    output: standardized Q

    description: Function that returns a standardization function for the input matrix Q. 
    """
    mu, sd = Q.mean(0), Q.std(0)
    sd[sd < 1e-12] = np.inf
    return lambda Z: (Z - mu) / sd


def _post(fit, Q, y, S, fallback):
    """
    args:
    fit: function to fit the model
    Q: input matrix
    y: response variable
    S: selected support
    fallback: fallback value if S is empty

    output: function that predicts using the fitted model
    
    description: Refit on the selected support S (post-Lasso, hdm default); constant if S is empty.
    """
    if len(S) == 0:
        return lambda Z: np.full(len(Z), fallback)
    m = fit(Q[:, S], y)
    return lambda Z: m(Z[:, S])


def _ols_fit(Q, y):
    """
    args:
    Q: (n x p) input matrix
    y: (n x 1) response variable

    output: function that predicts using the fitted linear regression model

    """
    return LinearRegression().fit(Q, y).predict


def _logit_fit(Q, d):
    """ 
    args: 
    Q (n x p) matrix, 
    d (n x 1) binary response var

    output: function that predicts probabilities using logistic regression

    description: Fit a logistic regression model 
    to the input matrix Q and binary response d.
    """
    m = _lr("l2", C=1e4, max_iter=5000).fit(Q, d)  # near-unpenalized, guards against separation
    return lambda Z: m.predict_proba(Z)[:, 1]


def lasso_plugin(Q, y, c = 1.1, iters = 5, post = True):
    """
    args:
    Q: (n x p) input matrix
    y: (n x 1) response variable
    c: constant for penalty level (adjusted to BCH rule)
    iters: number of iterations for penalty loadings
    post: whether to use post-Lasso OLS for predictions

    output: function that predicts using the fitted Lasso model, number of selected variables

    description: 
    BCH plug-in is calculated using the formula:
    lambda = 2c sqrt(n) Phi^-1(1 - gamma/2p), 
    with gamma = 0.1/log n, iterated penalty loadings.
    
    post = True: loadings updated with, and predictions from, post-Lasso OLS.
    
    """
    n, p = Q.shape
    alpha = c * norm.ppf(1 - 0.1 / np.log(n) / (2 * p)) / np.sqrt(n)
    Qc, e = Q - Q.mean(0), y - y.mean()
    for _ in range(iters):
        psi = np.sqrt((Qc ** 2 * e[:, None] ** 2).mean(0))
        psi[psi < 1e-12] = np.inf
        m = Lasso(alpha=alpha, max_iter=5000).fit(Q / psi, y)
        S = np.flatnonzero(m.coef_)
        pred = _post(_ols_fit, Q, y, S, y.mean()) if post else (lambda Z, m=m, psi=psi: m.predict(Z / psi))
        e = y - pred(Q)
    return pred, len(S)


def logit_plugin(Q, d, c=1.1, post=True):
    """
    lambda = (c/2) sqrt(n) Phi^-1(1 - gamma/2p) on standardized columns, post-logit.
    
    """
    # 1. get shape
    n, p = Q.shape
    # 2. calc penalty follwoing bch rule
    lam = c / 2 * np.sqrt(n) * norm.ppf(1 - 0.1 / np.log(n) / (2 * p))
    # 3.fit logit model
    s = _scaler(Q) # i. scale columns of Q using scaler
    m = _lr("l1",
            C=1 / lam, 
            solver="liblinear", 
            intercept_scaling = 100).fit(s(Q), d) #ii. fit logistic regression with l1 penalty
    S = np.flatnonzero(m.coef_[0]) # iii. get support of non-zero coefficients
    # 4. return predictions and number of selected variables
    if post:
        f = _post(_logit_fit, s(Q), d, S, d.mean())
        return (lambda Z: f(s(Z))), len(S)
    return (lambda Z: m.predict_proba(s(Z))[:, 1]), len(S)


def lasso_cv(Q, y):
    """
    args:
    Q: (n x p) input matrix
    y: (n x 1) response variable

    output: function that predicts using the fitted Lasso model, 
    number of selected variables

    """
    s = _scaler(Q) # i. scale columns of Q using scaler
    m = LassoCV(cv=5, max_iter=5000).fit(s(Q), y) # ii. fit Lasso model with cross-validation
    return (lambda Z: m.predict(s(Z))), int((m.coef_ != 0).sum()) 


def logit_cv(Q, d):
    """
    args:
    Q: (n x p) input matrix
    d: (n x 1) binary response variable

    output: function that predicts probabilities using the 
    fitted logistic regression model,
    number of selected variables using l1 penalty and cross-validation
    """
    s = _scaler(Q)
    m = _lr("l1", 
            crossval=True, 
            Cs=10, 
            cv=5, 
            solver="liblinear", 
            intercept_scaling = 100,
            scoring = "neg_log_loss").fit(s(Q), d)
    return (lambda Z: m.predict_proba(s(Z))[:, 1]), int((m.coef_ != 0).sum())


def ridge(Q, y):
    """
    args:
    Q: (n x p) input matrix
    y: (n x 1) response variable

    output: function that predicts using the fitted ridge regression model, 
    number of selected variables
    """
    s = _scaler(Q)
    m = RidgeCV(alphas=np.logspace(-2, 5, 30)).fit(s(Q), y)
    return (lambda Z: m.predict(s(Z))), Q.shape[1]


def logit_ridge(Q, d):
    """
    args:
    Q: (n x p) input matrix
    d: (n x 1) binary response variable
    
    output:
    function that predicts probabilities using the 
    fitted ridge logistic regression model, number of selected variables
    """
    s = _scaler(Q)
    m = _lr("l2", crossval=True, Cs=10, cv=5, scoring="neg_log_loss", max_iter=2000).fit(s(Q), d)
    return (lambda Z: m.predict_proba(s(Z))[:, 1]), Q.shape[1]


def ols(Q, y):
    """
    args:
    Q: (n x p) input matrix
    y: (n x 1) response variable

    output: function that predicts using the fitted linear regression model, 
    number of selected variables
    """
    return LinearRegression().fit(Q, y).predict, Q.shape[1]


def logit_mle(Q, d):
    """
    args:
    Q: (n x p) input matrix
    d: (n x 1) binary response variable

    output: function that predicts probabilities using the 
    fitted logistic regression model, number of selected variables
    """
    s = _scaler(Q)
    m = _lr("none", max_iter=5000).fit(s(Q), d)
    return (lambda Z: m.predict_proba(s(Z))[:, 1]), Q.shape[1]


def const(Q,y):
    """
    args:
    Q: (n x p) input matrix
    y: response variable

    output: 
    function that predicts the mean of y
    """
    return (lambda Z: np.full(len(Z), y.mean())), 0

