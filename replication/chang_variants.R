# Chang (2020), Table 2, DML-DiD with Lasso first steps: his procedure with switches.
#
# This script repeats the steps of "Table 2/DMLML.R" in Chang's replication
# repository and lets three of them be switched off, one at a time:
#   intercept  = FALSE : predict g and l2 without the intercept that glmnet fitted (his code)
#   dropscores = TRUE  : discard scores s >= |min(s)| in each fold before averaging (his code)
#   trim       = "drop": drop observations with fitted propensity outside (0.03, 0.97) (his code)
# With all three at his settings the result is his published -5.222 (SE 2.647).
#
# Run from the repository root:   Rscript replication/chang_variants.R
# Needs the packages glmnet and readstata13. About two minutes.

suppressMessages({library(glmnet); library(readstata13)})
# R >= 3.6 draws random samples differently from the R that Chang used. This line
# restores the old algorithm, so that the 100 random splits are his.
suppressWarnings(RNGkind(sample.kind = "Rounding"))

args <- commandArgs(TRUE)
data <- suppressWarnings(read.dta13(if (length(args)) args[1] else "Bribes_Regression.dta"))

# ---- sample and covariates, exactly as in DMLML.R -----------------------------
Y <- data$lba; D <- data$tariff_change_2008; T <- data$post_2008
ca <- sapply(2:8,  function(j) data$clear_agent == j)
hg <- sapply(2:15, function(j) data$hc_group == j)
X <- cbind(data$post_2008, data$tariff2007, data$lvalue_tonnage, data$differentiated, data$agri,
           data$perishable, data$dfs, data$day_w_arrival, data$monitor, data$psi, data$rsa,
           data$post_2008, ca, hg, data$hc_4digits)
W <- cbind(Y, D, T, X); W <- W[complete.cases(W), ]      # complete cases of all 34 columns: N = 1084
Y <- W[, 1]; D <- W[, 2]; T <- W[, 3]
p <- W[, 4:12]                                            # ... but only the first nine enter the Lasso
N <- length(Y)

run <- function(intercept = FALSE, dropscores = TRUE, rule = "1se", trim = "drop", B = 100, k = 2) {
  set.seed(123); random <- sample(1:1000, B)              # his seeds for the B splits
  keep <- function(s) if (dropscores) s[which(s < abs(min(s)))] else s
  pieces <- list(); thetabar <- c(); gm <- c(); ndrop <- c(); nz <- NULL
  for (l in 1:B) {
    set.seed(random[l]); train_ind <- sample(seq_len(N), size = floor(N/k))
    K <- rbind((1:N)[train_ind], (1:N)[-train_ind]); th <- c()
    for (q in 1:k) {                                      # K[q,] = evaluation fold, -K[q,] = auxiliary sample
      set.seed(333)
      CV  <- cv.glmnet(p[-K[q,],], D[-K[q,]], family = "binomial", alpha = 1)
      fit <- glmnet(p[-K[q,],], D[-K[q,]], family = "binomial", alpha = 1,
                    lambda = if (rule == "1se") CV$lambda.1se else CV$lambda.min)
      b1 <- as.numeric(fit$beta); a1 <- if (intercept) as.numeric(fit$a0) else 0
      ghat <- 1/(1 + exp(-(a1 + p[K[q,],] %*% b1)))
      index1 <- if (trim == "drop") K[q,][which(ghat < 0.97 & ghat > 0.03)] else K[q,]
      ghat <- 1/(1 + exp(-(a1 + p[index1,] %*% b1)))
      if (trim == "clip") ghat <- pmin(pmax(ghat, 0.03), 0.97)
      lambda <- mean(T[-K[q,]]); index <- which(D[-K[q,]] == 0)
      YY <- Y[-K[q,]][index]; TT <- T[-K[q,]][index]; XX <- p[-K[q,],][index,]
      set.seed(333)
      CV2  <- cv.glmnet(XX, (TT - lambda)*YY, family = "gaussian", alpha = 1)
      fit2 <- glmnet(XX, (TT - lambda)*YY, family = "gaussian", alpha = 1,
                     lambda = if (rule == "1se") CV2$lambda.1se else CV2$lambda.min)
      b2 <- as.numeric(fit2$beta); a2 <- if (intercept) as.numeric(fit2$a0) else 0
      ell <- a2 + p[index1,] %*% b2
      w <- (D[index1] - ghat)/(1 - ghat)/(lambda*(1 - lambda))/mean(D[index1])
      s <- ((T[index1] - lambda)*Y[index1] - ell)*w
      th[q] <- mean(keep(s))
      gm <- c(gm, mean(ghat)); ndrop <- c(ndrop, length(s) - length(keep(s)))
      nz <- rbind(nz, c(sum(b1 != 0), sum(b2 != 0)))
      pieces[[length(pieces) + 1]] <- list(i = index1, w = w, ell = ell, lambda = lambda, l = l)
    }
    thetabar[l] <- mean(th)
  }
  theta <- mean(thetabar)                                 # the estimate: MEAN over the B splits
  v <- sapply(pieces, function(P) { i <- P$i              # his variance, with the same scores discarded
    G <- -(1 - 2*P$lambda)*theta/(P$lambda*(1 - P$lambda)) - mean(Y[i]*P$w)
    s <- ((T[i] - P$lambda)*Y[i] - P$ell)*P$w - D[i]*theta/mean(D[i]) + G*(T[i] - P$lambda)
    mean(keep(s)^2) })
  var_l <- tapply(v, sapply(pieces, function(P) P$l), mean)
  c(theta = theta, se = sqrt(mean(var_l))/sqrt(N), median = median(thetabar), min = min(thetabar),
    max = max(thetabar), mean_ghat = mean(gm), sel_g = mean(nz[, 1]), sel_l = mean(nz[, 2]), dropped = mean(ndrop))
}

res <- rbind(
  "Chang as coded"                          = run(),
  "keep all scores"                         = run(dropscores = FALSE),
  "add intercepts"                          = run(intercept = TRUE),
  "intercepts, keep all scores"             = run(intercept = TRUE, dropscores = FALSE),
  "intercepts, all scores, clip"            = run(intercept = TRUE, dropscores = FALSE, trim = "clip"),
  "intercepts, all scores, clip, lambda.min" = run(intercept = TRUE, dropscores = FALSE, trim = "clip", rule = "min"))
options(width = 200)
print(round(res, 3))
write.csv(round(res, 4), "replication/chang_variants.csv")
