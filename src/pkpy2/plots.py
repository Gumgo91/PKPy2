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


def vpc(result, output=None, *, log=False, ax=None, observed=True, legend=True):
    """Plot a vpc() result: observations (points), observed percentiles (lines) and the intervals of the
    simulated percentiles (bands); the median is orange, the other percentiles blue and dashed."""
    plt = _plt()
    name = output or next(iter(result))
    e = result[name]
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 4.5), layout='constrained')
    else:
        fig = ax.figure
    t = np.asarray(e['bin_time'])
    lower, upper, obs = (np.asarray(e[k], dtype=float) for k in ('lower', 'upper', 'observed'))
    if observed:
        ax.scatter(e['observations']['time'], e['observations']['dv'], s=7, color='#5F6B75', alpha=.45,
                   edgecolor='none', zorder=1)
    for j, q in enumerate(e['quantiles']):
        color = '#D55E00' if q == .5 else '#0072B2'
        ax.fill_between(t, lower[:, j], upper[:, j], color=color, alpha=.2, lw=0, zorder=2)
        ax.plot(t, obs[:, j], color=color, lw=1.5, ls='-' if q == .5 else '--', zorder=3)
    if e.get('lloq') is not None and not e['prediction_corrected']:
        ax.axhline(e['lloq'], color='k', lw=.8, ls=':')
    ax.set_xlabel('Time'); ax.set_ylabel(('Prediction-corrected ' if e['prediction_corrected'] else '') + name)
    if log:
        ax.set_yscale('log')
    if legend:
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        ax.legend(handles=[Line2D([], [], color='#D55E00', lw=1.5, label='Observed median'),
                           Line2D([], [], color='#0072B2', lw=1.5, ls='--', label='Observed percentiles'),
                           Patch(color='#5F6B75', alpha=.3, label='Simulated intervals')], frameon=False, fontsize=8)
    return fig


def individual_fits(table, ids=None, *, output=None, ncols=4, log=False, figsize=None):
    """Observed data with PRED and IPRED per subject."""
    plt = _plt()
    ids = list(dict.fromkeys(table['ID'].tolist())) if ids is None else list(ids)
    nrows = int(np.ceil(len(ids) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize or (2.6 * ncols, 2.2 * nrows + .4), layout='constrained',
                             squeeze=False)
    for ax, sid in zip(axes.ravel(), ids):
        k = table['ID'] == sid
        if output is not None:
            k &= table['OUTPUT'] == output
        order = np.argsort(table['TIME'][k])
        t = table['TIME'][k][order]
        ax.scatter(t, table['DV'][k][order], s=10, color='k', label='Observed')
        ax.plot(t, table['PRED'][k][order], color='#009E73', lw=1, label='PRED')
        ax.plot(t, table['IPRED'][k][order], color='#0072B2', lw=1.2, label='IPRED')
        ax.set_title(f'ID {int(sid) if float(sid).is_integer() else sid}', fontsize=9)
        if log:
            ax.set_yscale('log')
    for ax in axes.ravel()[len(ids):]:
        ax.axis('off')
    fig.supxlabel('Time', fontsize=10)
    fig.supylabel(output or 'DV', fontsize=10)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='outside upper center', ncol=3, frameon=False)
    return fig
