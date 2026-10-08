import numpy as np
import runs as runs
import pandas as pd
import DGP as dgp


def design_d1(R=500, Ks=(2, 5), N_trend=(200, 1000, 2000), r2=0.5, **kw):
    """
    args:
    R (int): number of reps
    Ks (tuple): grid of K values
    N_trend (tuple): grid of N values for the trend design
    r2 (float): R^2 of the trend design

    output: pd.DataFrame with simulation results
    
    description: design that matches Chang's DGP and a trend design with the same gamma.
    (a) Chang's exact DGP, N=200, p=100, l10 = 1, sd .1, both gamma versions and K.
    (b) Same gamma with trend 1 + X'gamma (trend R^2 = r2), p = N/2.
    """
    # 1. define gamma functions
    gam = lambda p, v: np.r_[np.arange(5, 0, -1) / 5 
                             if v == "code" 
                             else 1 / np.arange(1, 6), 
                             np.zeros(p - 5)]
    
    out = [runs.run(dgp.gauss_rep, 
                    R, 
                    N=200, gamma=gam(100, v), 
                    delta=np.zeros(100), 
                    sigma=0.1, 
                    sigma3=0.1, 
                    K=K,
                    support=np.arange(5), **kw).assign(design="chang", 
                                                  gamma=v, K=K, N=200)
           for v in ("code", "paper") for K in Ks]
    for N in N_trend:
        g = gam(N // 2, "code")
        out.append(runs.run(dgp.gauss_rep, 
                       R, 
                       N=N, 
                       gamma=g, 
                       delta=g, 
                       sigma=np.linalg.norm(g) * np.sqrt((1 - r2) / r2),
                       sigma3=0.1, 
                       K=5, 
                       support=np.arange(5), **kw).assign(design="trend", 
                                                          gamma="code", 
                                                          K=5, 
                                                          N=N))
    return pd.concat(out)