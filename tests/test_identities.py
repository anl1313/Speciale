'''The two exact identities the interpretation rests on: Lemma 4 and Proposition 10.'''
import numpy as np

import dgp
import dmldid as dd
import mc


def test_prop10_identity_holds_exactly():
    rng = np.random.default_rng(1)
    d = dgp.sim_cat(rng, 400, J=5, hit='both')
    B = dgp.bases_cat(rng, d['X'], d['X'][:5], J=5, n_ref=4)
    folds = dd.make_folds(400, 5, rng)
    fits = {k: dd.dmldid(Q, d['D'], d['dY'], folds) for k, (Q, _) in B.items()}
    for k, f in fits.items():
        c = mc.prop10(d['D'], d['dY'], f, fits['A0'], folds)
        assert abs(f['theta'] - fits['A0']['theta'] - sum(c.values())) < 1e-10


def test_lemma4_identity_population():
    '''E[psi(W, theta0, p0, g, l)] = B1(g, l) for fixed (wrong) nuisances; checked by a large-n average.'''
    rng = np.random.default_rng(2)
    n = 400_000
    X = rng.standard_normal((n, 3))
    g0, l0 = dd.G(X[:, 0]), 1 + X[:, 0] + X[:, 1]
    D = rng.binomial(1, g0)
    dY = l0 + 3 * D + rng.standard_normal(n)
    p0 = g0.mean()
    g, l = dd.G(.5 * X[:, 0] + .3), 1 + .5 * X[:, 0]
    s = (dY - l) * (D - g) / (p0 * (1 - g)) - dY * (D - g0) / (p0 * (1 - g0))  # psi(g, l) - psi(g0, 0), theta0 cancels
    b1 = np.mean((g - g0) * (l - l0) / (p0 * (1 - g)))
    assert abs(s.mean() - b1) < 4 * s.std() / np.sqrt(n)
