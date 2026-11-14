"""Build the D4 LaTeX tables from the result files in the repository root.

    python tables/build_d4.py

Reads   d4_D4a_<share>_N<N>.csv, d4_D4a_<share>_N<N>_forced.csv,
        d4_D4b_N<N>.csv,          d4_D4b_N<N>_forced.csv
Writes  tables/d4a_table.tex      (D4a at the largest N, one block per share)
        tables/d4a_byN_table.tex  (D4a across N: bias and coverage of three codings, range)
        tables/d4b_table.tex      (D4b, one block per N)
The tables need \\usepackage{booktabs} in the preamble.
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "tables")
theta0 = 1.0                      # ATT in D4
a0 = -0.25                        # logit intercept in D4
shares = ["0.05", "0.15", "0.30"]
Ns = (1000, 2000, 4000)


def logit(x):
    return 1/(1 + np.exp(-x))


def load(name):
    """A D4 cell: the penalised codings and, merged in, the forced-in coding."""
    d = pd.read_csv(os.path.join(ROOT, name + ".csv"))
    f = pd.read_csv(os.path.join(ROOT, name + "_forced.csv"))
    # same seeds, so the coding-free columns must agree; otherwise the files do not belong together
    assert len(d) == len(f) and np.allclose(d.theta_oracle, f.theta_oracle), name
    return pd.concat([d, f[[c for c in f.columns if c not in d.columns]]], axis=1)


def summ(th, se=None):
    out = {"bias": th.mean() - theta0, "sd": th.std(ddof=1)}
    out["mcse"] = out["sd"]/np.sqrt(len(th))
    if se is not None:
        out["sesd"] = se.mean()/out["sd"]
        out["cov"] = np.mean(np.abs(th - theta0) <= 1.96*se)
    return out


def fmt(x, d=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    if round(x, d) == 0:
        x = 0.0
    s = f"{x:.{d}f}"
    return ("$-$" + s[1:]) if s.startswith("-") else s


def population_d4a(share, C=20):
    """Unadjusted bias (Remark) and the Corollary's prediction for reference 0."""
    pi = np.full(C, (1 - share)/(C - 1)); pi[0] = share
    gam, dlt = np.zeros(C), np.zeros(C)
    gam[0], dlt[0], gam[1], dlt[2] = 1.0, 1.0, 1.0, 1.0
    g, l = logit(a0 + gam), 1.0 + dlt
    p0 = pi @ g
    unadj = (pi @ (g*l) - p0*(pi @ l))/(p0*(1 - p0))
    pool = [0] + list(range(3, C))
    w = pi[0]/pi[pool].sum()
    g_pool = pi[pool] @ g[pool]/pi[pool].sum()
    cor = pi[0]*(1 - w)*(g[0] - g[3])*(l[0] - l[3])/(p0*(1 - g_pool))
    return unadj, cor


def coding_rows(df, codings, labels):
    """One table row per coding: post-Lasso, then the same fits without the refit."""
    rows = []
    for cd, lab in zip(codings, labels):
        s = summ(df[f"theta_post_{cd}"], df[f"se_post_{cd}"])
        l = summ(df[f"theta_lasso_{cd}"], df[f"se_lasso_{cd}"])
        forced = cd.startswith("forced")
        rows.append([lab, fmt(s["bias"]), fmt(s["mcse"]), fmt(s["sd"]), fmt(s["sesd"], 2), fmt(s["cov"], 2),
                     fmt(df[f"B1_post_{cd}"].mean()), "--" if forced else fmt(df[f"pool_{cd}"].mean()),
                     fmt(l["bias"]), fmt(l["cov"], 2),
                     "all" if forced else fmt(df[f"selg_{cd}"].mean(), 1),
                     "all" if forced else fmt(df[f"sell_{cd}"].mean(), 1)])
    return rows


def other_rows(df, or_col):
    o = summ(df[or_col])
    u = summ(df["theta_unadj"], df["se_unadj"])
    return [["OR-DiD", fmt(o["bias"]), fmt(o["mcse"]), fmt(o["sd"])] + ["--"]*8,
            ["Unadjusted DiD", fmt(u["bias"]), fmt(u["mcse"]), fmt(u["sd"]), fmt(u["sesd"], 2), fmt(u["cov"], 2)] + ["--"]*6]


HEAD = [r"\begin{tabular}{l rrrrr rr rr rr}", r"\toprule",
        r" & \multicolumn{5}{c}{Post-Lasso} & & & \multicolumn{2}{c}{No refit} & \multicolumn{2}{c}{Selected} \\",
        r"\cmidrule(lr){2-6} \cmidrule(lr){9-10} \cmidrule(lr){11-12}",
        r" & Bias & MCSE & SD & SE/SD & Cov. & $B_1$ & Pool & Bias & Cov. & $\hat g$ & $\hat\ell$ \\",
        r"\midrule"]


def table_d4a(N=4000):
    codings = ["ref3", "ref1", "ref2", "ref0", "forced_ref3"]
    labels = ["Reference 3 (typical)", "Reference 1 (selection only)", "Reference 2 (trend only)",
              "Reference 0 (both)", "Dummies forced in"]
    lines = list(HEAD)
    for i, sh in enumerate(shares):
        df = load(f"d4_D4a_{sh}_N{N}")
        unadj, cor = population_d4a(float(sh))
        if i:
            lines.append(r"\addlinespace")
        lines.append(rf"\multicolumn{{12}}{{l}}{{\emph{{$\pi_0={sh}$: unadjusted bias {unadj:.3f}, "
                     rf"Corollary for reference 0: {cor:.3f}}}}} \\")
        for row in coding_rows(df, codings, labels) + other_rows(df, "theta_or_ref3"):
            lines.append(r"\quad " + " & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT, "d4a_table.tex"), "w").write("\n".join(lines) + "\n")


def table_d4a_byN():
    lines = [r"\begin{tabular}{ll rr rr rr r}", r"\toprule",
             r" & & \multicolumn{2}{c}{Reference 3} & \multicolumn{2}{c}{Reference 0} & \multicolumn{2}{c}{Forced in} & \\",
             r"\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8}",
             r"$\pi_0$ & $N$ & Bias & Cov. & Bias & Cov. & Bias & Cov. & Range \\", r"\midrule"]
    refs = ["ref3", "ref1", "ref2", "ref0"]
    for i, sh in enumerate(shares):
        if i:
            lines.append(r"\addlinespace")
        for N in Ns:
            df = load(f"d4_D4a_{sh}_N{N}")
            row = [sh if N == Ns[0] else "", str(N)]
            for cd in ("ref3", "ref0", "forced_ref3"):
                s = summ(df[f"theta_post_{cd}"], df[f"se_post_{cd}"])
                row += [fmt(s["bias"]), fmt(s["cov"], 2)]
            th = df[[f"theta_post_{c}" for c in refs]]
            rng = (th.max(axis=1) - th.min(axis=1))/df["se_post_ref3"]      # studentised range, per replication
            row.append(fmt(rng.median(), 2))
            lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT, "d4a_byN_table.tex"), "w").write("\n".join(lines) + "\n")


def table_d4b():
    codings = ["thresholds", "ref0", "forced_ref0"]
    labels = ["Thresholds", "Dummies, reference 0", "Dummies forced in"]
    lines = list(HEAD)
    for i, N in enumerate(Ns):
        df = load(f"d4_D4b_N{N}")
        rng = (df.theta_post_ref0 - df.theta_post_thresholds).abs()/df.se_post_thresholds
        if i:
            lines.append(r"\addlinespace")
        lines.append(rf"\multicolumn{{12}}{{l}}{{\emph{{$N={N}$: median studentised range {rng.median():.2f}}}}} \\")
        for row in coding_rows(df, codings, labels) + other_rows(df, "theta_or_ref0"):
            lines.append(r"\quad " + " & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT, "d4b_table.tex"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    table_d4a()
    table_d4a_byN()
    table_d4b()
    print("R per cell:", {len(load(f"d4_D4a_{sh}_N{N}")) for sh in shares for N in Ns} | {len(load(f"d4_D4b_N{N}")) for N in Ns})
    for name in ("d4a_table.tex", "d4a_byN_table.tex", "d4b_table.tex"):
        print("wrote", os.path.join("tables", name))
