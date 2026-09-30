"""Diagnostic plots (matplotlib is imported only when a plot is requested)."""
import numpy as np


def _plt():
    import matplotlib.pyplot as plt
    return plt


def gof(table, *, output=None, log=False, figsize=(9, 8)):
    """Four-panel goodness-of-fit plot: DV vs PRED, DV vs IPRED, CWRES vs TIME, CWRES vs PRED."""
    plt = _plt()
    k = np.ones(len(table['DV']), dtype=bool) if output is None else table['OUTPUT'] == output
    k &= table['CENS'] == 0
    fig, axes = plt.subplots(2, 2, figsize=figsize, layout='constrained')
    for ax, xname in zip(axes[0], ('PRED', 'IPRED')):
        x, y = table[xname][k], table['DV'][k]
        ax.scatter(x, y, s=12, alpha=.6, color='#0072B2')
        lim = [np.nanmin([x.min(), y.min()]), np.nanmax([x.max(), y.max()])]
        ax.plot(lim, lim, color='k', lw=1)
        ax.set_xlabel(xname); ax.set_ylabel('DV')
        if log:
            ax.set_xscale('log'); ax.set_yscale('log')
    for ax, xname in zip(axes[1], ('TIME', 'PRED')):
        ax.scatter(table[xname][k], table['CWRES'][k], s=12, alpha=.6, color='#D55E00')
        ax.axhline(0, color='k', lw=1)
        for v in (-2, 2):
            ax.axhline(v, color='grey', lw=.8, ls='--')
        ax.set_xlabel(xname); ax.set_ylabel('CWRES')
    return fig


def vpc(result, output=None, *, log=False, ax=None, observed=True):
    """Plot a vpc() result: observed percentiles (lines) and simulated percentile intervals (bands)."""
    plt = _plt()
    name = output or next(iter(result))
    e = result[name]
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 4.5), layout='constrained')
    else:
        fig = ax.figure
    t = e['bin_time']
    colors = ['#0072B2', '#D55E00', '#0072B2']
    for j, q in enumerate(e['quantiles']):
        ax.fill_between(t, e['lower'][:, j], e['upper'][:, j], color=colors[j % 3], alpha=.18, lw=0)
        ax.plot(t, e['observed'][:, j], color=colors[j % 3], lw=1.6, ls='-' if q == .5 else '--')
    if observed:
        ax.scatter(e['observations']['time'], e['observations']['dv'], s=8, color='grey', alpha=.45)
    if e.get('lloq') is not None and not e['prediction_corrected']:
        ax.axhline(e['lloq'], color='k', lw=.8, ls=':')
    ax.set_xlabel('Time'); ax.set_ylabel(('Prediction-corrected ' if e['prediction_corrected'] else '') + name)
    if log:
        ax.set_yscale('log')
    return fig


def individual_fits(table, ids=None, *, output=None, ncols=4, log=False):
    """Observed data with PRED and IPRED per subject."""
    plt = _plt()
    ids = list(dict.fromkeys(table['ID'].tolist())) if ids is None else list(ids)
    nrows = int(np.ceil(len(ids) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.2 * nrows), layout='constrained', squeeze=False)
    for ax, sid in zip(axes.ravel(), ids):
        k = table['ID'] == sid
        if output is not None:
            k &= table['OUTPUT'] == output
        order = np.argsort(table['TIME'][k])
        t = table['TIME'][k][order]
        ax.scatter(t, table['DV'][k][order], s=10, color='k')
        ax.plot(t, table['PRED'][k][order], color='#009E73', lw=1)
        ax.plot(t, table['IPRED'][k][order], color='#0072B2', lw=1.2)
        ax.set_title(str(sid), fontsize=9)
        if log:
            ax.set_yscale('log')
    for ax in axes.ravel()[len(ids):]:
        ax.axis('off')
    return fig
