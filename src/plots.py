"""Figures from stored Monte Carlo results (status == 'ok' rows): Chang-style histograms, Lemma 4 scatter,
summary paths along N, and estimates by basis with the Prop. 10 terms."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from dgp import theta0

def _ok(df):
    return df[df.status == 'ok'] if 'status' in df else df


plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})


def _grid(nr, nc, w=2.6, h=2.2, **kw):
    return plt.subplots(nr, nc, figsize=(w * nc, h * nr), squeeze=False, **kw)


def hist(df, col='est', row=None, cols=None, rows=None, z=False, bins=40, names={}):
    '''Chang (2020, Fig. 1-2): histograms of theta-hat over replications, a panel per (row, col); x shared within rows.
    Dashed: the normal limit Chang's theorem implies, N(theta0, mean se^2). Mass outside the 0.5-99.5% range is
    piled on the edges. z=True: (theta-hat - theta0)/se against N(0, 1). names: panel labels, e.g. {'bcch': 'DML-DiD'}.'''
    df = _ok(df)
    cols = cols or list(df[col].dropna().unique())
    rows = rows or ([None] if row is None else list(df[row].dropna().unique()))
    c = 0. if z else theta0
    fig, axs = _grid(len(rows), len(cols), sharey='row')
    for i, r in enumerate(rows):
        d = df[df[col].isin(cols) & ((df[row] == r) if row else True)]
        d = d.assign(v=(d.theta - theta0) / d.se if z else d.theta)
        lo, hi = np.nanquantile(d.v, [.005, .995])
        lo, hi = min(lo, c - (hi - c) / 3), max(hi, c + (c - lo) / 3)  # keep the truth in view
        x = np.linspace(lo, hi, 300)
        for j, k in enumerate(cols):
            ax, dk = axs[i, j], d[d[col] == k]
            ax.hist(dk.v.dropna().clip(lo, hi), bins=bins, range=(lo, hi), density=True, color='0.75', edgecolor='white')
            ax.axvline(c, color='k', lw=1)
            if dk.se.notna().any():
                ax.plot(x, norm.pdf(x, c, 1. if z else dk.se.mean()), 'k--', lw=1)
            ax.set_title(names.get(k, k) if row is None else f'{r}: {names.get(k, k)}')
    return fig


def lemma4(df, col='est', cols=None, names={}):
    '''Realized error theta-hat - theta0 against B1 (Lemma 4): the 45-degree line up to the O_k noise.
    Axes cover 99% of the draws; the heavy-tail draws beyond are left out.'''
    df = _ok(df)
    cols = cols or list(df.loc[df.B1.notna(), col].unique())
    fig, axs = _grid(1, len(cols), sharex=True, sharey=True)
    for ax, k in zip(axs[0], cols):
        dk = df[df[col] == k]
        m = np.nanquantile(np.abs(np.r_[dk.B1, dk.theta - theta0]), .99)
        ax.scatter(dk.B1, dk.theta - theta0, s=4, color='0.3', alpha=.4)
        ax.plot([-m, m], [-m, m], 'k-', lw=.8)
        ax.set(xlim=(-m, m), ylim=(-m, m), title=names.get(k, k), xlabel='$B_1$', ylabel=r'$\hat\theta-\theta_0$')
    return fig


def path(s, y='bias_sd', hue='a', x='N', panel=None, logx=True):
    '''A summary column along N, one line per hue (e.g. bias/sd on the D2 decay path: Prop. 7).
    s: mc.summary(...).reset_index(). Error bars +-2 MCSE for bias and cover.'''
    panels = [None] if panel is None else list(s[panel].unique())
    fig, axs = _grid(1, len(panels), w=3.2, sharey=True)
    for ax, pv in zip(axs[0], panels):
        sp = s if panel is None else s[s[panel] == pv]
        for h, sh in sp.groupby(hue):
            e = 2 * sh[f'mcse_{y}'] if f'mcse_{y}' in sh else None
            ax.errorbar(sh[x], sh[y], yerr=e, marker='o', ms=3, lw=1, capsize=2, label=f'{hue} = {h}')
        if y == 'cover':
            ax.axhline(.95, color='k', lw=.8, ls=':')
        ax.set(xscale='log' if logx else 'linear', xlabel=x, ylabel=y, title='' if panel is None else f'{panel} = {pv}')
        ax.set_xticks(sorted(sp[x].unique()), [str(v) for v in sorted(sp[x].unique())])
        ax.minorticks_off()
    axs[0, -1].legend(frameon=False, loc='center left', bbox_to_anchor=(1, .5))
    return fig


def bases(df, est='bcch', order=None):
    '''Left: theta-hat by basis; the spread across bases is R(A). Right: mean |c1|, |c2|, |c3| (Prop. 10).'''
    d = _ok(df)[lambda x: x.est == est]
    order = order or list(d.basis.unique())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.5, 2.6))
    a1.boxplot([d.loc[d.basis == b, 'theta'] for b in order], showfliers=False, medianprops=dict(color='k'))
    a1.set_xticks(range(1, len(order) + 1), order)
    a1.axhline(theta0, color='k', lw=1)
    a1.set_ylabel(r'$\hat\theta$')
    c = d.groupby('basis')[['c1', 'c2', 'c3']].apply(lambda v: v.abs().mean()).loc[order]
    c.plot.bar(stacked=True, ax=a2, color=['0.2', '0.55', '0.85'], width=.7, rot=0)
    a2.legend(['$|c_1|$ propensity', '$|c_2|$ outcome', '$|c_3|$ cross'], frameon=False)
    return fig
