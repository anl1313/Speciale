import os, numpy as np, pandas as pd
os.chdir("/home/user/speciale")
OUT = "/home/user/speciale/tables/"
seed = 666                                   # the seed in your notebook
p_max = 2000
theta0_d23 = 1.0
cells_d23 = [(0.7, 0.7), (2.0, 2.0), (0.8, 1.5)]
Ns = (500, 1000, 2000, 4000)

def coefs_d23(p, alpha_g, alpha_l, signs, cell_id):     # exactly as in your notebook
    j = np.arange(1, p + 1)
    if signs == "aligned":
        s, t = np.ones(p), np.ones(p)
    else:
        rng_sign = np.random.default_rng([seed, 123, cell_id])
        s = rng_sign.choice([-1.0, 1.0], size=p_max)[:p]
        t = rng_sign.choice([-1.0, 1.0], size=p_max)[:p]
    gamma = s * j ** (-alpha_g); delta = t * j ** (-alpha_l)
    return gamma / np.linalg.norm(gamma), delta / np.linalg.norm(delta)

def fmt(x, d=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    if round(x, d) == 0:
        x = 0.0
    s = f"{x:.{d}f}"
    return ("$-$" + s[1:]) if s.startswith("-") else s

def load(N, ag, al, signs):
    return pd.read_csv(f"d23_N{N}_a{ag}_{al}_{signs}.csv")

def main_table(SIGNS, path):
    lines = [r"\begin{tabular}{ll rr " + "rr " * len(Ns) + "r r}", r"\toprule",
             r" & & & & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{$N={N}$}}" for N in Ns) + r" & \multicolumn{2}{c}{Slope} \\",
             " ".join(rf"\cmidrule(lr){{{5 + 2 * i}-{6 + 2 * i}}}" for i in range(len(Ns)))
             + rf" \cmidrule(lr){{{5 + 2 * len(Ns)}-{6 + 2 * len(Ns)}}}",
             r"$(\alpha_g,\alpha_\ell)$ & Signs & $\gamma'\delta$ & Unadj. & " + " & ".join(["Bias & Cov."] * len(Ns))
             + r" & Est. & Pred. \\", r"\midrule"]
    max_mcse = 0.0
    for p, (kind, title) in enumerate([("post", "A. DML-DiD, post-Lasso"), ("lasso", "B. DML-DiD, Lasso (no refit)")]):
        if p > 0:
            lines.append(r"\addlinespace")
        lines.append(rf"\multicolumn{{{6 + 2 * len(Ns)}}}{{l}}{{\emph{{{title}}}}} \\")
        for cell_id, (ag, al) in enumerate(cells_d23):
            for signs in SIGNS:
                gamma, delta = coefs_d23(Ns[-1] // 2, ag, al, signs, cell_id)
                unadj = load(Ns[-1], ag, al, signs)["theta_unadj"].mean() - theta0_d23
                row = [f"$({ag:g},\\,{al:g})$", signs, fmt(gamma @ delta, 2), fmt(unadj, 2)]
                biases = []
                for N in Ns:
                    df = load(N, ag, al, signs)
                    est, se = df[f"theta_{kind}"], df[f"se_{kind}"]
                    bias = est.mean() - theta0_d23
                    biases.append(bias)
                    max_mcse = max(max_mcse, est.std(ddof=1) / np.sqrt(len(df)))
                    row += [fmt(bias), fmt(np.mean(np.abs(est - theta0_d23) <= 1.96 * se), 2)]
                if signs == "aligned":
                    slope = np.polyfit(np.log(Ns), np.log(np.sqrt(Ns) * np.abs(biases)), 1)[0]
                    pred = (1 / (2 * ag) + 1 / (2 * al) - 1) / 2
                    row += [fmt(slope, 2), fmt(pred, 2)]
                else:
                    row += ["--", "--"]
                lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(path, "w").write("\n".join(lines) + "\n")
    return max_mcse

def formula_table(SIGNS, path):
    cols = [("post", "theta"), ("post", "B1_unclipped"), ("post", "full"), ("post", "both_omit"),
            ("lasso", "theta"), ("lasso", "B1_unclipped"), ("lasso", "inner"), ("lasso", "both_omit")]
    lines = [r"\begin{tabular}{ll r rrrr rrrr}", r"\toprule",
             r" & & & \multicolumn{4}{c}{Post-Lasso} & \multicolumn{4}{c}{Lasso (no refit)} \\",
             r"\cmidrule(lr){4-7} \cmidrule(lr){8-11}",
             r"$(\alpha_g,\alpha_\ell)$ & Signs & $N$ & Bias & $B_1$ & Formula & Both omit & Bias & $B_1$ & Inner & Both omit \\",
             r"\midrule"]
    for ag, al in cells_d23:
        for signs in SIGNS:
            for N in (Ns[0], Ns[-1]):
                df = load(N, ag, al, signs)
                vals = [df[f"theta_{k}"].mean() - theta0_d23 if n == "theta" else df[f"{n}_{k}"].mean() for k, n in cols]
                lines.append(f"$({ag:g},\\,{al:g})$ & {signs} & {N} & " + " & ".join(fmt(v) for v in vals) + r" \\")
        lines.append(r"\addlinespace")
    lines = lines[:-1] + [r"\bottomrule", r"\end{tabular}"]
    open(path, "w").write("\n".join(lines) + "\n")

m1 = main_table(("aligned",), OUT + "d23_table_main.tex")
m2 = main_table(("aligned", "random"), OUT + "d23_table_full.tex")
formula_table(("aligned", "random"), OUT + "d23_formula_table.tex")
print("max MCSE of the bias, aligned:", round(m1, 3), "| all cells:", round(m2, 3))
print("R per cell:", {len(load(N, 0.7, 0.7, 'aligned')) for N in Ns})
