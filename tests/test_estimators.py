'''Estimator behaviour: known-parameter recovery, SE calibration, coverage arithmetic, failures, reproducibility.'''
import numpy as np
import pandas as pd
import pytest

import dgp
import dmldid as dd
import dmldid_v5 as old
import mc

GAMMA, DELTA = np.array([1., .5, 0, 0, 0]), np.array([1., 0, .5, 0, 0])


@pytest.fixture(scope='module')
def lowdim():
    '''p = 5, unpenalized first steps (correctly specified), R = 80.'''
    return mc.run(mc.replicate, R=80, n_jobs=1, sim=dgp.sim_gauss, bases=dgp.bases_gauss, N=1000,
                  gamma=GAMMA, delta=DELTA, methods=('ols',), n_test=500)


def test_known_parameter_recovery(lowdim):
    s = mc.summary(lowdim)
    for e in ('ols', 'oracle', 'ordid'):
        assert abs(s.loc[e, 'bias']) < 4 * s.loc[e, 'mcse_bias'], e
    assert abs(s.loc['unadjusted', 'bias']) > 10 * s.loc['unadjusted', 'mcse_bias']  # confounded by design


def test_se_calibration_and_coverage(lowdim):
    s = mc.summary(lowdim)
    for e in ('ols', 'oracle'):
        assert .7 < s.loc[e, 'se_sd'] < 1.4, e
        assert abs(s.loc[e, 'cover'] - .95) < 4 * np.sqrt(.95 * .05 / s.loc[e, 'n']), e


def test_coverage_and_mcse_arithmetic():
    df = pd.DataFrame(dict(est='x', basis=None, theta=[3., 3.5, 2.0, np.nan], se=[.1, .1, 1., np.nan],
                           status=['ok', 'ok', 'ok', 'error'], rep=range(4)))
    df = mc.add_inference(df)
    assert df.cover.tolist()[:3] == [1., 0., 1.] and np.isnan(df.cover.iloc[3])
    s = mc.summary(df).loc['x']
    assert s.n == 3 and s.n_error == 1
    assert np.isclose(s.bias, np.mean([0., .5, -1.])) and np.isclose(s.cover, 2 / 3)
    assert np.isclose(s.mcse_cover, np.sqrt(2 / 3 / 3 / 3))


def test_failures_are_recorded_not_dropped():
    def bad_sim(rng, N, **_):
        raise RuntimeError('boom')
    rows = mc.replicate(np.random.default_rng(0), bad_sim, dgp.bases_gauss, 100)
    assert rows[0]['status'] == 'error' and 'boom' in rows[0]['error']

    def nan_bases(rng, X, Xt, **_):
        return {'A0': (np.full_like(X, np.nan), Xt)}
    rows = pd.DataFrame(mc.replicate(np.random.default_rng(0), dgp.sim_gauss, nan_bases, 200, gamma=GAMMA,
                                     delta=DELTA, extras=('oracle',)))
    dml = rows[rows.est == 'bcch'].iloc[0]
    assert dml.status == 'error' and np.isnan(dml.theta)
    assert rows[rows.est == 'oracle'].status.iloc[0] == 'ok'


def test_ordid_infeasible_is_flagged():
    rows = pd.DataFrame(mc.replicate(np.random.default_rng(0), dgp.sim_gauss, dgp.bases_gauss, 200,
                                     gamma=np.r_[GAMMA, np.zeros(95)], delta=np.r_[DELTA, np.zeros(95)],
                                     extras=('ordid',), n_test=200))
    assert rows[rows.est == 'ordid'].status.iloc[0] == 'infeasible'


def test_reproducible_and_subset_equals_full_run():
    kw = dict(sim=dgp.sim_cat, bases=dgp.bases_cat, N=300, J=5, hit='both', n_ref=3, n_test=300)
    a = mc.run(mc.replicate, R=4, n_jobs=1, **kw)
    b = mc.run(mc.replicate, R=4, n_jobs=1, **kw)
    c = mc.run(mc.replicate, n_jobs=1, reps=[2], **kw)
    cols = ['rep', 'est', 'basis', 'theta', 'se']
    pd.testing.assert_frame_equal(a[cols], b[cols])
    pd.testing.assert_frame_equal(a[a.rep == 2][cols].reset_index(drop=True), c[cols].reset_index(drop=True))
    assert np.array_equal(np.random.SeedSequence(90).spawn(5)[3].generate_state(4),
                          np.random.SeedSequence(90, spawn_key=(3,)).generate_state(4))


def test_update_changes_only_what_it_declares():
    '''Against the frozen 10 Oct estimator: identical estimates wherever liblinear is not involved.'''
    rng = np.random.default_rng(7)
    d = dgp.sim_gauss(rng, 600, np.r_[GAMMA, np.zeros(45)], np.r_[DELTA, np.zeros(45)])
    folds = dd.make_folds(600, 5, rng)
    for m in ('ols', 'ridge'):
        assert np.isclose(dd.dmldid(d['X'], d['D'], d['dY'], folds, m)['theta'],
                          old.dmldid(d['X'], d['D'], d['dY'], folds, m)['theta'], rtol=1e-8)
    new, ref = dd.dmldid(d['X'], d['D'], d['dY'], folds), old.dmldid(d['X'], d['D'], d['dY'], folds)
    assert np.allclose(new['ellhat'], ref['ellhat'], rtol=1e-8)  # bcch outcome step unchanged (full rank)


def test_min_norm_ols_on_duplicated_columns():
    '''The D5 failure: two identical selected columns. LinearRegression and MinNormOLS agree at full rank;
    with a duplicate, MinNormOLS splits the coefficient and predicts sensibly where the columns differ.'''
    rng = np.random.default_rng(8)
    X = rng.standard_normal((300, 3))
    y = X @ [1., 2., 0.] + rng.standard_normal(300)
    from sklearn.linear_model import LinearRegression
    assert np.allclose(dd.MinNormOLS().fit(X, y).predict(X), LinearRegression().fit(X, y).predict(X))
    Xd = np.c_[X, X[:, 1]]
    m = dd.MinNormOLS().fit(Xd, y)
    assert m.rank_ == 3 and np.isclose(m.coef_[1], m.coef_[3])
    assert np.abs(m.predict(np.c_[X, np.zeros(300)])).max() < 20


def test_ridge_refit_is_a_separately_labelled_estimator():
    rows = pd.DataFrame(mc.replicate(np.random.default_rng(0), dgp.sim_gauss, dgp.bases_gauss, 400,
                                     gamma=np.r_[GAMMA, np.zeros(45)], delta=np.r_[DELTA, np.zeros(45)],
                                     methods=('bcch', ('bcch_rr', dict(method='bcch', refit='ridge'))),
                                     extras=(), n_test=300))
    assert set(rows.est) == {'bcch', 'bcch_rr'}
    assert {'cond_l', 'cond_g', 'w_max', 'g_min', 'nwarn'} <= set(rows.columns)
    assert np.isfinite(rows.cond_l).all()
