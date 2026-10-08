import numpy as np
import pandas as pd
import diagnostics as diag
import runs as runs
import estimators as est
from scipy.special import expit
theta0 = 3.0

def dummies(C, J, ref = None):
    """
    args: 
    C (array): categorical variable with shape (N, B)
    J (int): number of levels in the categorical variable
    ref (array): reference levels for each block, shape (B,). If None, all levels are included.

    output: 
    array of dummies with shape (N, B*(J-1)) if ref is not None, else (N, B*J)

    description:
    Function that returns the dummies for a categorical variable C with J levels.
    If ref is not None, the dummies are reference-coded with respect to
    the levels specified in ref. If ref is None, all levels are included in the dummies.
    """
    E = np.eye(J)
    return np.hstack([E[C[:, b]] if ref is None else np.delete(E[C[:, b]], ref[b], 1) for b in range(C.shape[1])])


def cat_rep(rng, N, 
            B = 5, 
            J = 5,
            hit = "both", 
            mu = 1.2, 
            scale = 1.0,
            pc = 5, 
            sigma = 1.0, 
            n_ref=10, 
            K=5, learner = "plugin",
            n_test = 5000):
    """
    args:
    rng (np.random.Generator): random number generator
    N (int): number of observations
    B (int): number of blocks
    J (int): number of levels in each block
    hit (str): which nuisance depends on the blocks.
    mu (float): the amount by which level 1 shifts the index.
    scale (float): the scale of the propensity score.
    pc (int): the number of confounding variables.
    sigma (float): the standard deviation of the noise.
    n_ref (int): the number of reference levels to consider.
    K (int): the number of folds for cross-validation.
    learner (str): the machine learning algorithm to use.
    n_test (int): the number of test observations.

    output:
    list of dicts: each dict contains the results for a single repetition of the simulation.

    description:
    Function that simulates a categorical treatment variable with B blocks and J levels,
    where each block has a different set of levels.
    Per block: level 0 (prob .3), level 1 (prob .2), J-2 small levels (prob .5 in total). Level 1 shifts the
    index by +-mu: 1-sparse with reference 0 or any small level, (J-1)-sparse with reference 1 ('bad').
    hit: which nuisance depends on the blocks.
    
    """
    # 1. set the probabilities for each level in each block
    probs = np.r_[0.3, 0.2, 
                  np.full(J - 2, 0.5 / (J - 2))]
    # 2. set the signs for each level in each block
    sgn = mu * (-1.0) ** np.arange(B)

    # 3. draw categoricals and calc nuisances
    def draw(r, n):
        C = r.choice(J, (n, B), p=probs)
        Xc = r.standard_normal((n, pc))
        f = (C == 1) @ sgn
        g0 = expit(scale * (f * (hit in ("g", "both")) + 0.5 * Xc[:, 0]))
        l0 = 1 + f * (hit in ("l", "both")) + 0.5 * Xc[:, 0]
        return C, Xc, g0, l0
    C, Xc, g0, l0 = draw(rng, N)
    # 4. draw treatment and outcome
    D = rng.binomial(1, g0)
    dY = l0 + sigma * rng.standard_normal(N) + D * theta0
    Ct, Xct, g0t, l0t = draw(np.random.default_rng(999), n_test)
    # 5. draw reference levels for each block
    refs = {"A0": np.zeros(B, int),
            "bad": np.ones(B, int)}
    refs |= {f"r{i}": rng.integers(0, J, B) for i in range(n_ref - 2)}
    # 6. make folds and run the simulation
    folds = est.make_folds(N, K, rng)
    rows = diag.compare_bases({k: np.c_[Xc, dummies(C, J, r)] for k, r in refs.items()}, D, dY, folds, learner,
                         g0t, l0t, {k: np.c_[Xct, dummies(Ct, J, r)] for k, r in refs.items()})
    rows.append(est._row(f"dml_{learner}", est.dmldid(np.c_[Xc, dummies(C, J)], D, dY, folds, learner)[:2], basis="full"))
    rows.append(est._row("oracle", est.oracle(D, dY, g0, l0, folds), basis="-"))
    return rows

# design 5: number of ref levels fixed vs growing with N, for both nuisances
def design_d5(N_grid=(500, 1000, 2000), hits=("g", "l", "both"), R=500, **kw):
    """
    args:
    N_grid (tuple): grid of sample sizes
    hits (tuple): which nuisance functions depend on the blocks
    R (int): number of repetitions

    output:
    DataFrame: results of the simulation for each combination of N, hit, and repetition

    description:
    Function that runs the simulation for different sample sizes and nuisance functions.
    The number of reference levels is fixed (bounded sparsity inflation)
    vs growing with N (inflation growing with p).
    """
    rules = {"fixed": lambda N: 5, 
             "growing": lambda N: max(5, N // 20)}
    return pd.concat([runs.run(cat_rep, 
                               R, N = N,
                               J = rule(N), 
                               hit = h, **kw).assign(N=N, 
                                                   J=rule(N), 
                                                   J_rule=jr, 
                                                   hit=h)
                      for N in N_grid for jr, rule in rules.items() for h in hits])

# design 6: ref levels fixed, scale the propensity index; R and |c1| against realized kappa
def design_d6(scales=(0.5, 1.0, 2.0, 3.0), N=1000, J=10, R=500, **kw):
    """
    args:
    scales (tuple): grid of scales for the propensity index
    N (int): sample size
    J (int): number of levels in the categorical variable
    R (int): number of repetitions
    
    output: dataframe with realized kappa, R, and |c1| for each scale

    description:
    Function that runs the simulation for different scales of the propensity index.
    The number of reference levels is fixed (bounded sparsity inflation).

    """
    return pd.concat([runs.run(cat_rep, 
                               R, N=N, 
                               J = J, 
                               scale = s, 
                               hit = "both", **kw).assign(scale = s) for s in scales])
