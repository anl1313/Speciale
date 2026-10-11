'''Results store: round trip, resumption without duplication, duplicate and specification guards.'''
import shutil

import numpy as np
import pandas as pd
import pytest

import dgp
import mc
import store

SPEC = dict(N=300, p=10)


def build(spec):
    g = np.r_[1., .5, np.zeros(spec['p'] - 2)]
    return mc.replicate, dict(sim=dgp.sim_gauss, bases=dgp.bases_gauss, N=spec['N'], gamma=g, delta=g,
                              n_test=200), dict(naive=0.)


def test_round_trip_and_resume(tmp_path):
    df = store.run_cell('T', SPEC, build, R=6, n_jobs=1, chunk=4, root=tmp_path)
    cid = store.cell_id(SPEC, 90)
    assert sorted(df.rep.unique()) == list(range(6)) and len(store.parts('T', cid, tmp_path)) == 2
    again = store.load('T', cid, tmp_path)
    pd.testing.assert_frame_equal(df, again)
    df2 = store.run_cell('T', SPEC, build, R=8, n_jobs=1, chunk=4, root=tmp_path)  # only reps 6, 7 are new
    assert sorted(df2.rep.unique()) == list(range(8)) and len(store.parts('T', cid, tmp_path)) == 3
    fresh = mc.run(build(SPEC)[0], reps=[6, 7], n_jobs=1, **build(SPEC)[1])
    pd.testing.assert_series_equal(df2[df2.rep >= 6].theta.reset_index(drop=True), fresh.theta, check_names=False)
    assert {'commit', 'cell', 'design', 'N', 'p', 'seed', 'status', 'cover'} <= set(df2.columns)
    assert (tmp_path / 'T' / cid / 'runs.jsonl').read_text().count('\n') == 2


def test_duplicates_and_spec_mismatch_are_refused(tmp_path):
    store.run_cell('T', SPEC, build, R=2, n_jobs=1, root=tmp_path)
    cid = store.cell_id(SPEC, 90)
    p = store.parts('T', cid, tmp_path)[0]
    shutil.copy(p, p.with_name('part-copy.parquet'))
    with pytest.raises(ValueError, match='duplicated'):
        store.load('T', cid, tmp_path)
    cj = tmp_path / 'T' / cid / 'cell.json'
    cj.write_text(cj.read_text().replace('"N": 300', '"N": 301'))
    with pytest.raises(ValueError, match='different specification'):
        store.run_cell('T', SPEC, build, R=3, n_jobs=1, root=tmp_path)


def test_csv_export_does_not_overwrite(tmp_path):
    store.run_cell('T', SPEC, build, R=2, n_jobs=1, root=tmp_path)
    path = store.export_csv('T', root=tmp_path)
    with pytest.raises(FileExistsError):
        store.export_csv('T', root=tmp_path)
    assert len(pd.read_csv(path)) > 0
