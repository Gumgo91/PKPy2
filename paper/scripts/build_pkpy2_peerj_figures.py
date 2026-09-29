"""Build the PeerJ revision figures for PKPy2 (larger type, 600 dpi, revised layouts).

All figures are drawn at their final printed width (6.9 in) so that the point sizes set
here are the printed sizes. Every text element is checked to be at least MIN_PT.

Figure order follows first citation in the revised manuscript:
 1 workflow, 2 recovery vs PKPy/Gaussian control, 3 PKPy2 interval coverage,
 4 clinical datasets across software, 5 simulation comparison with established software,
 6 prediction-call speedup.
"""
from pathlib import Path
import csv
import json
import math
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / 'output/pkpy2_development'
PAPER = ROOT / 'docs/pkpy2_paper'
CMP = ROOT / 'output/pkpy2_software_comparison'
OUT = PAPER / 'figures_peerj_revision'

WIDTH = 174 / 25.4   # Springer single-column width (174 mm); height at most 234 mm
CLIP = 50.0   # differences beyond ±CLIP % are drawn at the limit and labelled with their value
MIN_PT = 10.0
BLUE, ORANGE, GREEN, PURPLE, GRAY = '#0072B2', '#D55E00', '#009E73', '#CC79A7', '#5F6B75'
INK = '#17212B'
GRID = '#E4E7EA'
PARAM_LABEL = {'theta_CL': r'$CL$', 'theta_V': r'$V$', 'omega_CL': r'$\omega^2_{CL}$',
               'omega_V': r'$\omega^2_{V}$', 'sigma_prop': r'$\sigma_{prop}$'}
DEV_METHODS = [('pkpy', 'PKPy', GRAY, 's'), ('gaussian_tst', 'Gaussian two-stage', ORANGE, '^'),
               ('pkpy2', 'PKPy2', BLUE, 'o')]
SW_METHODS = [('pkpy2', 'PKPy2', BLUE, 'o'), ('nlmixr2_focei', 'nlmixr2 FOCEi', ORANGE, '^'),
              ('nlmixr2_saem', 'nlmixr2 SAEM', GREEN, 's'), ('saemix', 'saemix', PURPLE, 'D')]

plt.rcParams.update({
    'font.family': 'Arial', 'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial', 'mathtext.it': 'Arial:italic',
    'mathtext.bf': 'Arial:bold', 'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11,
    'xtick.labelsize': 10.5, 'ytick.labelsize': 10.5, 'legend.fontsize': 10.5,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .8,
    'xtick.major.width': .8, 'ytick.major.width': .8, 'lines.linewidth': 1.4,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'path', 'axes.unicode_minus': False,
    'mathtext.default': 'it'})


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def check_and_save(fig, name, report, pad=.04):
    fig.canvas.draw()
    sizes = [t.get_fontsize() for t in fig.findobj(matplotlib.text.Text) if t.get_text().strip() and t.get_visible()]
    assert min(sizes) >= MIN_PT, (name, min(sizes))
    OUT.mkdir(parents=True, exist_ok=True)
    opts = dict(facecolor='white', bbox_inches='tight', pad_inches=pad)
    fig.savefig(OUT / f'{name}.png', dpi=600, **opts)
    fig.savefig(OUT / f'{name}.pdf', **opts)
    fig.savefig(OUT / f'{name}.eps', **opts)
    plt.close(fig)
    # TIFF for submission: RGB, 8 bits per channel, LZW, 600 dpi (combination art).
    image = Image.open(OUT / f'{name}.png').convert('RGB')
    image.save(OUT / f'{name}.tif', compression='tiff_lzw', dpi=(600, 600))
    w, h = image.size[0] / 600, image.size[1] / 600
    assert w * 25.4 <= 174.5 and h * 25.4 <= 234, (name, w * 25.4, h * 25.4)
    report.append(dict(figure=name, width_mm=round(w * 25.4, 1), height_mm=round(h * 25.4, 1), dpi=600,
                       minimum_font_pt=float(min(sizes)), maximum_font_pt=float(max(sizes))))


# ------------------------------------------------------------------ Figure 1
def figure_workflow(report):
    H = 4.25
    fig = plt.figure(figsize=(WIDTH, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, WIDTH), ylim=(0, H))
    ax.axis('off')

    def box(x, y, w, h, text, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.08',
                                    lw=1.1, edgecolor=color, facecolor='white'))
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=11, color=INK, linespacing=1.35)

    def arrow(a, b, color):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=13, lw=1.2, color=color,
                                     shrinkA=0, shrinkB=0))

    ax.text(.08, H - .08, 'a   PKPy', fontsize=12, weight='bold', va='top', color=INK)
    ys = [3.20, 2.67, 2.14]
    mid = 2.67 + .21
    for y, label in zip(ys, ['Subject 1', 'Subject 2', 'Subject N']):
        box(.10, y, 1.25, .42, label, GRAY)
        arrow((1.42, y + .21), (2.02, mid + (y - 2.67) * .45), GRAY)
    box(2.10, mid - .50, 2.0, 1.0, 'Separate\nindividual estimates', GRAY)
    arrow((4.17, mid), (4.60, mid), GRAY)
    box(4.68, mid - .50, 2.10, 1.0, 'Geometric means\nand log-parameter\ncovariance', GRAY)
    ax.plot([.08, WIDTH - .08], [1.95, 1.95], lw=.8, color='#D5DCE2')

    ax.text(.08, 1.84, 'b   PKPy2', fontsize=12, weight='bold', va='top', color=INK)
    w, gap, y0, h = 1.48, .22, .30, 1.0
    labels = ['Subject data\nand recorded\ndose histories', 'Declared model:\nfixed and\nestimated terms',
              'Joint marginal-\nlikelihood\nestimation', 'Independent\nnumerical audit\nand intervals']
    for i, label in enumerate(labels):
        x = .10 + i * (w + gap)
        box(x, y0, w, h, label, BLUE)
        if i < 3:
            arrow((x + w + .03, y0 + h / 2), (x + w + gap - .03, y0 + h / 2), BLUE)
    check_and_save(fig, 'Figure_1', report, pad=0)


# ------------------------------------------------------------------ Figure 2
def figure_recovery(summary, report):
    names = ['theta_CL', 'theta_V', 'omega_CL', 'omega_V']
    fig, axes = plt.subplots(4, 2, figsize=(WIDTH, 8.6), layout='constrained')
    for row, name in enumerate(names):
        for col, metric in enumerate(['relative_bias_pct', 'relative_rmse_pct']):
            ax = axes[row, col]
            for j, (engine, label, color, marker) in enumerate(DEV_METHODS):
                data = [next(a for a in summary['primary_estimation'] if a['sampling'] == s and a['engine'] == engine
                             and a['parameter'] == name and a['population'] == 'numerically_accepted')
                        for s in ['rich', 'sparse']]
                xx = np.array([0, 1]) + (j - 1) * .12
                yerr = [1.96 * a['bias_mcse_pct'] for a in data] if col == 0 else None
                ax.errorbar(xx, [a[metric] for a in data], yerr=yerr, marker=marker, ms=6, lw=1.2, capsize=3,
                            color=color, label=label)
            ax.set_xticks([0, 1], ['Rich', 'Sparse'])
            ax.set_xlim(-.4, 1.4)
            ax.grid(axis='y', color=GRID, lw=.6)
            if col == 0:
                ax.axhline(0, color='#333333', lw=.8)
                ax.set_ylabel(PARAM_LABEL[name], rotation=0, ha='right', va='center', fontsize=12, labelpad=10)
            if row == 0:
                ax.set_title(['Relative bias (%)', 'Relative RMSE (%)'][col], fontsize=12, weight='bold')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside upper center', ncol=3, frameon=False)
    check_and_save(fig, 'Figure_2', report)


# ------------------------------------------------------------------ Figure 3
def figure_coverage(summary, report):
    names = list(PARAM_LABEL)
    fig, axes = plt.subplots(2, 1, figsize=(WIDTH, 6.6), layout='constrained')
    lower = []
    for j, sampling in enumerate(['rich', 'sparse']):
        rows = [next(a for a in summary['primary_intervals'] if a['sampling'] == sampling and a['parameter'] == n)
                for n in names]
        v = np.array([a['conditional_coverage']['rate'] for a in rows])
        ci = np.array([a['conditional_coverage']['interval'] for a in rows])
        lower.extend(ci[:, 0])
        x = np.arange(5) + (j - .5) * .2
        color, marker = [BLUE, ORANGE][j], ['o', 's'][j]
        axes[0].errorbar(x, v, yerr=np.vstack((v - ci[:, 0], ci[:, 1] - v)), fmt=marker, ms=6, capsize=3,
                         color=color, label=sampling.title())
        wmed = np.array([a['relative_width_median'] for a in rows])
        q = np.array([a['relative_width_quartiles'] for a in rows])
        axes[1].errorbar(x, wmed, yerr=np.vstack((wmed - q[:, 0], q[:, 1] - wmed)), fmt=marker, ms=6, capsize=3,
                         color=color)
    axes[0].axhline(.95, color=GRAY, ls='--', lw=1)
    axes[0].set_ylim(max(0, math.floor((min(lower) - .05) * 10) / 10), 1.03)
    axes[0].set_ylabel('Coverage')
    axes[1].set_ylabel('Width / true value')
    axes[0].set_title('a', loc='left', weight='bold')
    axes[1].set_title('b', loc='left', weight='bold')
    for ax in axes:
        ax.set_xticks(np.arange(5), [PARAM_LABEL[n] for n in names], fontsize=12)
        ax.set_xlim(-.5, 4.5)
        ax.grid(axis='y', color=GRID, lw=.6)
    axes[0].legend(frameon=False, loc='lower left', ncol=2)
    check_and_save(fig, 'Figure_3', report)


# ------------------------------------------------------------------ Figure 4
def figure_clinical(cmp, report):
    rows = cmp['clinical']['rows']
    panels = [('theophylline', 'a   Theophylline'), ('warfarin', 'b   Warfarin')]
    if '--tobramycin' in sys.argv:
        panels.append(('tobramycin', 'c   Tobramycin'))
    counts = [len([r for r in rows if r['dataset'] == d and r['nonmem'] is not None]) for d, _ in panels]
    fig, axes = plt.subplots(len(panels), 1, figsize=(WIDTH, 1.3 + 0.42 * sum(counts)), layout='constrained',
                             gridspec_kw=dict(height_ratios=counts))
    for ax, (dataset, title) in zip(axes, panels):
        rr = [r for r in rows if r['dataset'] == dataset and r['nonmem'] is not None]
        y = np.arange(len(rr))
        values = [r.get(m + '_diff_pct') for r in rr for m, *_ in SW_METHODS]
        inside = [v for v in values if v is not None and abs(v) <= CLIP]
        lo = min(-5, math.floor((min(inside) - 1) / 5) * 5)
        hi = max(5, math.ceil((max(inside) + 1) / 5) * 5)
        clipped = any(v is not None and abs(v) > CLIP for v in values)
        if clipped:
            lo, hi = min(lo, -10), CLIP + 19
        for j, (m, label, color, marker) in enumerate(SW_METHODS):
            xs = [r.get(m + '_diff_pct') for r in rr]
            keep = [i for i, v in enumerate(xs) if v is not None]
            if not keep:
                continue
            yy = y[keep] + (j - 1.5) * .17
            shown = [xs[i] if abs(xs[i]) <= CLIP else math.copysign(CLIP + 5, xs[i]) for i in keep]
            ax.scatter(shown, yy, marker=marker, s=46, color=color, label=label, zorder=3, edgecolor='white', linewidth=.4)
            for i, xv, yv in zip(keep, shown, yy):
                if abs(xs[i]) > CLIP:
                    ax.text(xv + 2.5, yv, f'{xs[i]:+.0f}', va='center', ha='left', fontsize=10, color=color)
        ax.axvline(0, color=INK, lw=.9)
        if clipped:
            ax.axvline(CLIP + 2, color=GRAY, lw=.8, ls=(0, (2, 2)))
        labels = []
        for r in rr:
            lab = r['label'].split(' (')[0]
            lab = {'K_a': 'Ka', 'σ_prop': r'$\sigma_{prop}$'}.get(lab, lab)
            labels.append(lab)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.grid(axis='x', color=GRID, lw=.6)
        ax.set_title(title, loc='left', weight='bold')
        ax.set_xlabel('Difference from NONMEM (%)')
        ax.set_xlim(lo, hi)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=9, integer=True))
        if clipped:
            ax.set_xticks([t for t in ax.get_xticks() if lo <= t <= CLIP])
            ax.set_xlim(lo, hi)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside upper center', ncol=4, frameon=False, handletextpad=.3,
               columnspacing=1.0)
    check_and_save(fig, 'Figure_4', report)


# ------------------------------------------------------------------ Figure 5
def figure_software(report):
    reps = list(csv.DictReader((CMP / 'comparison_replicates.csv').open(encoding='utf-8')))
    params = list(PARAM_LABEL)
    fig, axes = plt.subplots(len(params), 2, figsize=(WIDTH, 9.0), layout='constrained', sharey=True)
    for row, p in enumerate(params):
        extreme = 0.0
        for col, sampling in enumerate(['rich', 'sparse']):
            ax = axes[row, col]
            data, means = [], []
            for m, *_ in SW_METHODS:
                e = [100 * float(r['relative_error']) for r in reps
                     if r['parameter'] == p and r['sampling'] == sampling and r['method'] == m and r['returned'] == '1']
                data.append(e)
                means.append(np.mean(e))
                extreme = max(extreme, max(abs(v) for v in e))
            pos = np.arange(len(SW_METHODS))[::-1]
            bp = ax.boxplot(data, positions=pos, vert=False, widths=.55, patch_artist=True, showfliers=True,
                            flierprops=dict(marker='.', ms=3, alpha=.5), medianprops=dict(color=INK, lw=1.2))
            for patch, (_, _, color, _) in zip(bp['boxes'], SW_METHODS):
                patch.set(facecolor=color, alpha=.35, edgecolor=color)
            for k, (_, _, color, _) in enumerate(SW_METHODS):
                ax.plot(means[k], pos[k], marker='D', ms=5.5, color=color, markeredgecolor=INK, mew=.6, zorder=4)
            ax.axvline(0, color=INK, lw=.8)
            ax.grid(axis='x', color=GRID, lw=.6)
            ax.set_yticks(pos, [lab for _, lab, *_ in SW_METHODS])
            if row == 0:
                ax.set_title(['Rich sampling', 'Sparse sampling'][col], weight='bold')
            if col == 0:
                ax.set_ylabel(PARAM_LABEL[p], fontsize=12, labelpad=8)
            if row == len(params) - 1:
                ax.set_xlabel('Relative error (%)')
        for col in range(2):
            axes[row, col].set_xlim(-extreme * 1.06, extreme * 1.06)
    check_and_save(fig, 'Figure_5', report)


# ------------------------------------------------------------------ Figure 6
def figure_speed(speed, report):
    fig, ax = plt.subplots(figsize=(WIDTH, 4.4), layout='constrained')
    names = {'1cmt_iv': 'One-compartment IV', '1cmt_oral': 'One-compartment oral', '2cmt_iv': 'Two-compartment IV'}
    for j, model in enumerate(['1cmt_iv', '1cmt_oral', '2cmt_iv']):
        rows = [r for r in speed['rows'] if r['model'] == model]
        ax.plot([r['doses'] for r in rows], [r['speedup'] for r in rows], marker=['o', 's', '^'][j], ms=7,
                color=[BLUE, ORANGE, GREEN][j], label=names[model])
    ax.axhline(1, color=GRAY, lw=.9, ls='--')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set(xlabel='Number of dose events', ylabel='Speedup (direct / recurrence)')
    ax.set_xticks([1, 10, 100, 1000], ['1', '10', '100', '1,000'])
    ax.set_yticks([1, 10, 100, 1000], ['1', '10', '100', '1,000'])
    ax.minorticks_off()
    ax.grid(axis='both', which='major', lw=.6, color=GRID)
    ax.legend(frameon=False, loc='upper left')
    check_and_save(fig, 'Figure_6', report)


def main():
    summary = read(PAPER / 'analysis_summary.json')
    cmp = read(CMP / 'comparison_summary.json')
    speed = read(DEV / 'recurrence_benchmark.json')
    report = []
    figure_workflow(report)
    figure_recovery(summary, report)
    figure_coverage(summary, report)
    figure_clinical(cmp, report)
    figure_software(report)
    figure_speed(speed, report)
    (OUT / 'figure_checks.json').write_text(json.dumps(report, indent=1), encoding='utf-8')
    for r in report:
        print(r)


if __name__ == '__main__':
    main()
