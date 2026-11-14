# Recreating Chang (2020), Table 2, DML-DiD with Lasso first steps

Chang reports -5.222 (SE 2.647). His script `Table 2/DMLML.R`
(https://github.com/NengChiehChang/Diff-in-Diff) reproduces both numbers exactly
if one line is added before the call `DMLML(Y,D,p,T)`:

    RNGkind(sample.kind = "Rounding")   # the sampling algorithm of R < 3.6.0

With the current default of R the same script draws other random splits and
gives -5.893 (SE 2.575).

## Files

- `chang_variants.R`: the steps of his script, with switches for the three steps
  that the paper does not describe. Run `Rscript replication/chang_variants.R`
  from the repository root (packages `glmnet`, `readstata13`; two minutes).
  Output in `chang_variants.csv`.
- `chang_port.py`: the same procedure in Python, for use in the notebook. Its
  random splits are not R's, so its numbers differ somewhat from the table below.

## Result of `chang_variants.R` (100 splits, K = 2, his seeds)

| Variant | Estimate | SE | Min split | Max split | Mean of fitted g | Scores discarded per fold |
|---|---|---|---|---|---|---|
| Chang as coded | -5.222 | 2.647 | -13.71 | 7.62 | 0.265 | 5.3 |
| keep all scores | 4.308 | 4.592 | -5.35 | 18.34 | 0.265 | 0 |
| add the fitted intercepts | -12.105 | 1.711 | -15.53 | -9.17 | 0.170 | 1.6 |
| intercepts, keep all scores | -10.091 | 2.377 | -14.64 | 0.47 | 0.170 | 0 |
| ... and clip g instead of dropping observations | -10.084 | 2.430 | -14.64 | 0.47 | 0.162 | 0 |
| ... and lambda.min instead of lambda.1se | -3.227 | 3.666 | -9.46 | 32.66 | 0.171 | 0 |

The share of treated shipments is 0.162. The cross-validated Lasso selects on
average 3.7 of the nine covariates for the propensity score and 2.0 for the
outcome (6.8 and 7.0 with lambda.min).

## Result of `chang_port.py` (100 splits, K = 2, seed 666)

| Variant | Estimate | SE | Min split | Max split |
|---|---|---|---|---|
| Chang as coded | -5.945 | 2.539 | -14.10 | 3.47 |
| keep all scores | 3.317 | 4.485 | -6.76 | 14.23 |
| add the fitted intercepts | -11.959 | 1.665 | -15.14 | -8.93 |
| intercepts, keep all scores | -9.900 | 2.286 | -12.83 | -5.30 |
| ... and clip g instead of dropping observations | -9.885 | 2.361 | -12.83 | -5.30 |
| ... and lambda.min instead of lambda.1se | -3.549 | 3.070 | -9.13 | 5.74 |

The port selects on average 3.8 and 2.0 covariates and discards 5.1 scores per
fold, against 3.7, 2.0 and 5.3 in R. Its first row is close to what Chang's own
script gives with R's current random numbers (-5.893).
