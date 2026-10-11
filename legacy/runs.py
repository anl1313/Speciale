import numpy as np
import pandas as pd
import estimators as est
import diagnostics as diag
from joblib import Parallel, delayed
theta0 = 3.0
eps = 1e-6


def run(rep, R = 500, seed = 90, n_jobs=-1, **kw):
    """
    args:
    rep: replication function
    R (int): number of repetitions
    seed (int): random seed
    n_jobs (int): number of parallel jobs
    kw: additional keyword arguments for the simulation function

    output: pd.DataFrame with simulation results

    description:
    Function that runs a simulation for a given repetition number,
    using the specified random seed and number of repetitions. 
    The simulation is run in parallel
    and the results are returned as a pandas DataFrame
    """
    seeds = np.random.SeedSequence(seed).spawn(R)
    res = Parallel(n_jobs = n_jobs)(delayed(rep)(np.random.default_rng(s), **kw) for s in seeds)
    return pd.DataFrame([dict(r, rep=i) for i, rows in enumerate(res) for r in rows])

def summarize(df, by=("est",), theta0=theta0):
    """
    args:
    df (dataframe): input dataframe with simulation results
    by (tuple): columns to group by
    theta0 (float): true parameter value

    output: pd.DataFrame with summarized simulation results
    """

    d = df.dropna(subset=["theta"]).copy()
    # a. error in theta estimate
    d["err"] = d.theta - theta0
    # b. coverage dummy
    d["cover"] = np.where(d.se.notna(),
                        (d.err.abs() <= 1.96 * d.se).astype(float), np.nan)
    # c. find absulute values of c1, c2, c3 if they exist
    for c in ("c1", "c2", "c3"):
        if c in d:
            d[c] = d[c].abs()
    g = d.groupby(list(by))
    # d. return summary statistics for each group
    out = pd.DataFrame(dict(bias = g.err.mean(), 
                            sd = g.theta.std(), 
                            cover = g.cover.mean(),
                            rmse = g.err.apply(lambda e: np.sqrt((e ** 2).mean())), 
                            se_sd = g.se.mean() / g.theta.std()))
    out["bias_sd"] = out.bias / out.sd

    extra = [c for c in ("naive", "B1", "err_g", 
                         "err_l", "sel_g", "sel_l", 
                         "kappa", "R", "c1", "c2", "c3") if c in d]
    return out.join(g[extra].mean())


def growth_slopes(summary, group=("est", "par", "cos")):
    """
    args: 
    summary (dataframe): input dataframe with summarized simulation results
    group (tuple): columns to group by
    output: pd.DataFrame with slopes of log|bias|/sd on log N
    description:
    Function that returns the slopes of log|bias|/sd on log N for each group
    Decay path: about (1 - a)/(2a) for 1/2 < a < 1, <= 0 for a >= 1.
    """
    s = summary.reset_index()
    return s.groupby(list(group)).apply(lambda x: np.polyfit(np.log(x.N), 
                                                             np.log(np.abs(x.bias_sd)), 1)[0])


def range_summary(df, by):
    """
    args: 
    df (dataframe): input dataframe with simulation results
    by (tuple): columns to group by
    
    output: pd.DataFrame with summary statistics for each group
    
    description:
    Function that returns summary statistics for each group in the input dataframe.
    """
    r = df.dropna(subset=["R"]).groupby(list(by) + ["rep"]).R.first().groupby(list(by))
    return pd.DataFrame(dict(R_mean=r.mean(), R_p90=r.quantile(0.9)))