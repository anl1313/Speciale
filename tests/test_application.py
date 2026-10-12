'''Section 6 reconstruction: lambda.1se rule, the intercept switch, and that every step runs.'''
import numpy as np

import application as ap


def test_one_se_picks_largest_lambda_within_one_se():
    lams = np.array([4., 2., 1., .5])
    err = np.array([[5., 3.1, 3., 3.2], [5., 2.9, 3., 3.2], [5., 3.0, 3., 3.2]])
    assert ap._one_se(err, lams) == 2.  # min at lambda = 1 (mean 3.0); lambda = 2 has mean 3.0 <= 3.0 + se


def test_intercept_switch_only_shifts_the_linear_predictor():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 4)) + 5
    y = X[:, 0] + rng.standard_normal(300)
    full, bug, _ = ap._glmnet_like(X, y, False, seed=1)
    diff = full(X) - bug(X)
    assert np.allclose(diff, diff[0]) and abs(diff[0]) > 1  # the omitted term is the intercept


def test_steps_run():
    Y, D, T, X = ap.load()
    assert len(Y) == 1084
    for name in ('chang_code', 'bcch'):
        r = ap.chang_dml(Y, D, T, X, B=1, n_jobs=1, **ap.STEPS[name][1])
        assert np.isfinite(r['theta']) and r['se'] > 0 and r['nwarn'] == 0
