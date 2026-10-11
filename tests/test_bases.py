'''Basis facts used in D5 and D7: supports, ranks, column spaces.'''
import numpy as np

import dgp


def _rank(M):
    M = M - M.mean(0)
    M = M[:, M.std(0) > 0] / M[:, M.std(0) > 0].std(0)
    return np.linalg.matrix_rank(np.c_[np.ones(len(M)), M], tol=1e-8 * len(M))


def test_reference_coding_support_one_vs_J_minus_one():
    for J in (5, 25):
        C = np.repeat(np.arange(J), 40)
        f = 1.2 * (C == 1)  # the D5 truth in one block
        for ref, expected in ((0, 1), (1, J - 1), (J - 1, 1)):
            X = np.c_[np.ones(len(C)), np.delete(np.eye(J)[C], ref, 1)]
            b = np.linalg.lstsq(X, f, rcond=None)[0]
            assert np.sum(np.abs(b[1:]) > 1e-10) == expected


def test_full_coding_is_rank_deficient_by_one_per_block():
    rng = np.random.default_rng(3)
    d = dgp.sim_cat(rng, 3000, B=5, J=8)
    B = dgp.bases_cat(rng, d['X'], d['X'][:5], B=5, J=8, n_ref=3)
    QA, QF = (np.c_[np.ones(3000), B[k][0]] for k in ('A0', 'full'))
    assert np.linalg.matrix_rank(QA) == QA.shape[1]
    assert QF.shape[1] - np.linalg.matrix_rank(QF) == 5


def test_reference_codings_share_column_space():
    rng = np.random.default_rng(4)
    d = dgp.sim_cat(rng, 3000, J=6)
    B = dgp.bases_cat(rng, d['X'], d['X'][:5], J=6, n_ref=4)
    for k in ('bad', 'full'):
        assert _rank(np.c_[B['A0'][0], B[k][0]]) == _rank(B['A0'][0])


def test_centring_is_invertible_on_full_polynomial_dictionary():
    rng = np.random.default_rng(5)
    X = 1 + 6 * rng.beta(1, 4, (2000, 3))
    Q0, Q1 = dgp.poly(X, 2.2), dgp.poly(X, np.median(X, 0))
    assert _rank(np.c_[Q0, Q1]) == _rank(Q0) == Q0.shape[1] + 1


def test_centring_inflates_support_by_a_bounded_factor():
    '''D7 outcome truth: 4 terms when centred at the mean; at most sum(2^deg - 1) = 14 when not centred.'''
    rng = np.random.default_rng(6)
    X = 1 + 6 * rng.beta(1, 4, (4000, 6))
    Z = X - 2.2
    l0 = .5 * Z[:, 0] + .25 * Z[:, 0] * Z[:, 2] + .3 * Z[:, 3] ** 2 + .05 * Z[:, 3] * Z[:, 4] * Z[:, 5]
    nz = {}
    for k, c in (('A0', 2.2), ('raw', 0.)):
        Q = np.c_[np.ones(len(X)), dgp.poly(X, c)]
        b = np.linalg.lstsq(Q, l0, rcond=None)[0][1:]
        nz[k] = int(np.sum(np.abs(b) > 1e-6 * np.abs(b).max()))
    assert nz['A0'] == 4 and 4 < nz['raw'] <= 14
