# Audit: extreme D5 estimates (10 Oct run)

## What was reported
In `legacy/simulations_vol5_outputs.ipynb`, D5 contained draws with |theta-hat - 3| > 10:
reps 100 and 296 (growing J, hit = l, N = 2000; theta-hat 7.4e12 and 1.6e11), 170 (growing, g, 2000; -5.8e11),
98 (growing, both, 2000; -2.9e10), 492 (growing, l, N = 500; 18.4 with se 15.5).

## What was checked (11 Oct, `audit/d5_reps.py`, Linux, numpy 2.4.4, scikit-learn 1.8.0, scipy-openblas)
* Reps 100, 296, 170, 98 reproduce the same selections and propensity range as in the notebook, but finite
  estimates (3.04, 2.98, 2.93, 2.96). Two runs give identical results (difference 0.0).
* In each of these four, the outcome post-Lasso refit design is numerically singular: condition number
  1.3e16, 4.8e15, 3.2e15, 2.0e16. Rep 100, fold 2: the Lasso selected two dummies (block 0 level 81, block 1
  level 36) that each occur once among the 808 training controls, for the same unit: identical columns.
* OLS is not identified on that support. With `LinearRegression` the coefficients on the two columns are
  decided by whether LAPACK's rank cutoff (machine epsilon times the largest singular value) removes the smallest
  singular value (3.1e-15 against 27.6, i.e. right at the cutoff). Removed: minimum-norm, finite predictions.
  Kept: coefficients of order 1e15 with opposite signs, which explode wherever the two columns differ.
  That depends on the LAPACK build, which explains why the draws explode on one machine and not on another.
  (Machine-dependence is the explanation consistent with all evidence; it is not confirmed until the script has
  been run on the machine that produced the original run.)
* Rep 492 is different: well-conditioned refits (cond 1.15), but the largest fitted propensity is 0.9997, so one
  control weight is about 6,100. The estimate (18.4) and its standard error (15.5) are both large and the interval
  covers 3. This is a statistically unstable estimator under weak overlap, not a numerical failure.

## Change made
The unpenalized linear (re)fit is now `MinNormOLS`: the minimum-norm least-squares solution with an explicit
relative cutoff of 1e-10. It equals OLS whenever the design has full rank (tested), and makes the rank-deficient
case deterministic across machines. Each fit records `cond_*` and `rankdef_*`, so the frequency of rank-deficient
refits is reported in the final run. A ridge refit is available as a separately labelled estimator
(`refit='ridge'`); it is not used by default.

## Open
* Run `python audit/d5_reps.py` on the original machine and compare with `audit/d5_reps_vm.csv`.
* Singletons are selected because the BCCH loadings are data-dependent; levels with one or two observations
  violate the bounded-loading conditions behind the plug-in penalty. Growing J (N/20 levels) produces such levels
  by construction. To be discussed as a property of the design, not a coding error.
