import numpy as np
import runs as runs
import diagnostics as diag
import pandas as pd

def design_d2(N_grid=(250, 500, 1000, 2000), 
              shape="decay", pars=(2.0, 1.0, 0.75, 0.5), 
              cos=(1.0,), 
              ratio=0.5,
              r2=0.5, 
              R=500, **kw):
    """
    args:
    N_grid (tuple): grid of sample sizes
    shape (str): "decay" or "equal", determines the shape of the coefficients
    pars (tuple): parameters for the coefficient shapes
    cos (tuple): cosine values for the coefficients
    ratio (float): ratio of p to N
    r2 (float): R-squared value for the trend
    R (int): number of replications
    kw (dict): additional keyword arguments for the runs.run function

    output: pd.DataFrame with simulation results

    description:
    Density path, unit norms. 
    "shape" can be either "decay" or "equal".
    "decay": pars are decay rates a rate boundary a = 1, Prop 8 at a = 1/2
    "equal": pars are alpha with s = N^alpha (alpha >= 1: s = p). D2: cos=(1, .5, 0).
    """
    # 1. initialize an empty list to store results
    out = []
    # a. loop over sample sizes in N_grid
    for N in N_grid:
        p = int(ratio * N)
        for par in pars:
            q = par if shape == "decay" else (p if par >= 1 else int(np.clip(np.ceil(N ** par), 
                                                                             2, p)))
            sup = None if shape == "decay" else np.arange(q)
            for c in cos:
                g, d = diag.coefs(p, shape, q, c)
                df = runs.run(diag.gauss_rep, 
                              R, 
                              N = N, 
                              gamma = g, 
                              delta = d, 
                              sigma = np.sqrt((1 - r2) / r2), 
                              support = sup, **kw)
                out.append(df.assign(N=N, 
                                     p=p, 
                                     shape=shape, 
                                     par=par,
                                     cos = c, 
                                     naive = diag.naive_bias(g, d)))
    return pd.concat(out)