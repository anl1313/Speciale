import numpy as np
from numpy import linalg as la
from sklearn.preprocessing import PolynomialFeatures
from dmldid import G

theta0 = 3.  # Chang (2020)

# Each design: sim(rng, N, **par) -> dict(X, D, dY, g0, l0)
#              bases(rng, X, Xt, **par) -> {name: (Q, Qt)}, first = A0; centres etc. from the training X


# ---------- Gaussian linear index: D1-D4
def sim_gauss(rng, N, gamma, delta, a=0., sigma=1., sigma3=0., **_):
    '''Chang (2020, Sec. 4) with a trend: dY = 1 + X'delta + D (theta0 + e3) + e.
    delta = 0, sigma = sigma3 = .1 is Chang's DGP (l10 = 1).'''
    X = rng.standard_normal((N, len(gamma)))
    g0, l0 = G(a + X @ gamma), 1 + X @ delta
    D = rng.binomial(1, g0)
    dY = l0 + D * (theta0 + sigma3 * rng.standard_normal(N)) + sigma * rng.standard_normal(N)
    return dict(X=X, D=D, dY=dY, g0=g0, l0=l0)


def bases_gauss(rng, X, Xt, **_):
    return {'A0': (X, Xt)}


def coefs(p, shape='decay', par=1., cos=1., seed=0):
    '''Unit-norm gamma: decay gamma_j ~ j^-par, or equal weights on the first par coords.
    delta: same support and magnitude profile, cos(gamma, delta) = cos.'''
    u = np.arange(1, p + 1) ** -float(par) if shape == 'decay' else (np.arange(p) < par).astype(float)
    u /= la.norm(u)
    if cos == 1:
        return u, u.copy()
    w = u * np.random.default_rng(seed).permutation(np.resize([1., -1.], p))
    w -= (w @ u) * u
    return u, cos * u + np.sqrt(1 - cos ** 2) * w / la.norm(w)


def core_tail(p, s=5, tau2=.8, dense=False):
    '''Unit norm: equal weights on the first s coords; if dense, share tau2 moved to an equal-weight tail.'''
    v = np.zeros(p)
    v[:s] = 1 / np.sqrt(s)
    if dense:
        v *= np.sqrt(1 - tau2)
        v[s:] = np.sqrt(tau2 / (p - s))
    return v


def naive_bias(gamma, delta, a=0.):
    '''Unadjusted-DiD bias = Prop. 8 limit, by Stein (X ~ N(0, I)): E[G'] gamma'delta / (p0 (1 - p0)).'''
    z, w = np.polynomial.hermite_e.hermegauss(80)
    h = G(a + la.norm(gamma) * z)
    w = w / w.sum()
    p0 = w @ h
    return w @ (h * (1 - h)) * (gamma @ delta) / (p0 * (1 - p0))


# ---------- reference categories: D5-D6
def sim_cat(rng, N, B=5, J=5, hit='both', mu=1.2, scale=1., pc=5, sigma=1., **_):
    '''B categorical covariates: level 0 (.3), level 1 (.2), J-2 small levels (.5 in total).
    Level 1 shifts the index by +-mu: 1-sparse with reference 0, (J-1)-sparse with reference 1.
    hit: which nuisance depends on the categoricals ('g', 'l', 'both'). X = [C, Xc].'''
    C = rng.choice(J, (N, B), p=np.r_[.3, .2, np.full(J - 2, .5 / (J - 2))])
    Xc = rng.standard_normal((N, pc))
    f = (C == 1) @ (mu * (-1.) ** np.arange(B))
    g0 = G(scale * (f * (hit != 'l') + .5 * Xc[:, 0]))
    l0 = 1 + f * (hit != 'g') + .5 * Xc[:, 0]
    D = rng.binomial(1, g0)
    return dict(X=np.c_[C, Xc], D=D, dY=l0 + D * theta0 + sigma * rng.standard_normal(N), g0=g0, l0=l0)


def dummies(C, J, ref=None):
    '''Reference coding (drop level ref[b] in block b); ref=None: all J dummies (full coding).'''
    E = np.eye(J)
    return np.hstack([E[c] if ref is None else np.delete(E[c], ref[b], 1) for b, c in enumerate(C.T)])


def bases_cat(rng, X, Xt, B=5, J=5, n_ref=10, **_):
    refs = {'A0': np.zeros(B, int), 'bad': np.ones(B, int)}
    refs |= {f'r{i}': rng.integers(0, J, B) for i in range(n_ref - 2)} | {'full': None}
    Q = lambda X, r: np.c_[X[:, B:], dummies(X[:, :B].astype(int), J, r)]
    return {k: (Q(X, r), Q(Xt, r)) for k, r in refs.items()}


# ---------- centring before polynomial expansion: D7
def sim_center(rng, N, k=9, scale=1., sigma=1., **_):
    '''X_j = 1 + 6 Beta(1, 4): mean 2.2, median ~1.95. Truth sparse in the cubic dictionary centred at the mean.'''
    X = 1 + 6 * rng.beta(1, 4, (N, k))
    Z = X - 2.2
    g0 = G(scale * (.5 * Z[:, 0] - .4 * Z[:, 1] + .25 * Z[:, 0] * Z[:, 2]))
    l0 = 1 + .5 * Z[:, 0] + .25 * Z[:, 0] * Z[:, 2] + .3 * Z[:, 3] ** 2 + .05 * Z[:, 3] * Z[:, 4] * Z[:, 5]
    D = rng.binomial(1, g0)
    return dict(X=X, D=D, dY=l0 + D * theta0 + sigma * rng.standard_normal(N), g0=g0, l0=l0)


def poly(X, c, deg=3):
    return PolynomialFeatures(deg, include_bias=False).fit_transform(X - c)


def bases_center(rng, X, Xt, **_):
    cents = {'A0': 2.2, 'raw': 0., 'lower': 1., 'median': np.median(X, 0), 'mean': X.mean(0)}
    return {k: (poly(X, c), poly(Xt, c)) for k, c in cents.items()}


# ---------- repeated cross sections on real covariates: D8
def sim_rcs(rng, Q0, cg, cl, D=None, lam=.5, sigma=1.):
    '''T ~ Bernoulli(lam) independent of (X, D) (RPC(i)); trend 1 + Q0 cl, sparse in A0.
    D: real treatment (keeps the real overlap) or None (simulated from G(Q0 cg)).'''
    N = len(Q0)
    if D is None:
        z = Q0 @ cg
        D = rng.binomial(1, G(z - z.mean()))
    T = rng.binomial(1, lam, N)
    Y = .5 * Q0 @ cl + .5 * D + T * (1 + Q0 @ cl + D * theta0) + sigma * rng.standard_normal(N)
    return dict(D=D, T=T, Y=Y)


def sparse_coefs(Q, s=5, scale=1., seed=0):
    '''s random columns, random signs, index sd = scale.'''
    r = np.random.default_rng(seed)
    c, j = np.zeros(Q.shape[1]), r.choice(Q.shape[1], s, replace=False)
    c[j] = r.choice([-1., 1.], s) / Q[:, j].std(0)
    return c * scale / (Q @ c).std()


def resolve(Q, order=None, tol=1e-8):
    '''Collinearity resolution: keep columns in `order` unless in span(1, kept).
    Different orders, different resolutions (KMR's first normalisation choice).'''
    Z, basis, keep = Q - Q.mean(0), [], []
    for j in range(Q.shape[1]) if order is None else order:
        v = Z[:, j].copy()
        for _ in range(2):
            for b in basis:
                v -= (b @ v) * b
        if la.norm(v) > tol * max(la.norm(Z[:, j]), 1e-300):
            basis.append(v / la.norm(v))
            keep.append(j)
    return np.sort(keep)

