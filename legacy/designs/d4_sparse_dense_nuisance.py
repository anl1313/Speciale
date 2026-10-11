import pandas as pd
import DGP as dgp
import runs as runs
import numpy as np

def design_d4(N = 1000, ratio = 0.5, tau2 = 0.8, r2 = 0.5, R = 500, **kw):
    """
    args: 
    N (int): sample size
    ratio (float): ratio of p to N
    tau2 (float): proportion of signal in the tail
    r2 (float): proportion of variance explained by the model
    R (int): number of replications

    output: pd.DataFrame with simulation results

    description:
    2x2 design: sparse/dense treatment model x sparse/dense outcome model.
    Sparse = the signal sits in a sparse core.
    dense =  the signal sits in a dense tail.
    """
    p, out = int(ratio * N), []
    for dg in (False, True):
        for dl in (False, True):
            g = dgp.core_tail(p, tau2 = tau2, dense = dg) 
            d = dgp.core_tail(p, tau2 = tau2, dense = dl)
            df = runs.run(dgp.gauss_rep, 
                          R, 
                          N=N,
                          gamma = g, delta = d, 
                          sigma = np.sqrt((1 - r2) / r2), **kw)
            out.append(df.assign(dense_g = dg,
                                 dense_l = dl, 
                                 naive = dgp.naive_bias(g, d)))
    return pd.concat(out)