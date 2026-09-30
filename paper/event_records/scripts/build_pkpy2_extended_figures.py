"""Figures of the event-record interface: Figure 1 (workflow, redrawn), Figure 6 (verification)
and Figure 7 (PK/PD application and uncertainty methods).

Style, sizes and checks are those of build_pkpy2_peerj_figures.py (174 mm width, 600 dpi,
text at least 10 pt). Data: output/pkpy2_extended_validation.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_pkpy2_peerj_figures import (WIDTH, BLUE, ORANGE, GREEN, PURPLE, GRAY, INK, GRID, OUT,  # noqa: E402
                                       check_and_save)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'packages/pkpy2/src'))
from pkpy2 import plots                                                                      # noqa: E402
V = ROOT / 'output/pkpy2_extended_validation'
ODE = {'mm_iv_multiple', 'mm_oral_ss', 'idr1', 'idr2', 'idr3', 'idr4', 'tmdd_full', 'tmdd_qss'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def panel(ax, letter):
    ax.set_title(letter, loc='left', weight='bold', fontsize=12)


# ------------------------------------------------------------------ Figure 1
def figure_workflow(report):
    H = 4.1
    fig = plt.figure(figsize=(WIDTH, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, WIDTH), ylim=(0, H))
    ax.axis('off')

    def box(x, y, w, h, text, color, size=11):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.08',
                                    lw=1.1, edgecolor=color, facecolor='white'))
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=size, color=INK, linespacing=1.3)

    def arrow(a, b, color):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=13, lw=1.2, color=color,
                                     shrinkA=0, shrinkB=0))

    ax.text(.08, H - .08, 'a   PKPy', fontsize=12, weight='bold', va='top', color=INK)
    top = H - .52
    ys = [top - .42, top - .95, top - 1.48]
    mid = ys[1] + .21
    for y, label in zip(ys, ['Subject 1', 'Subject 2', 'Subject N']):
        box(.10, y, 1.25, .42, label, GRAY)
        arrow((1.42, y + .21), (2.02, mid + (y - ys[1]) * .45), GRAY)
    box(2.10, mid - .45, 2.0, .9, 'Separate\nindividual estimates', GRAY)
    arrow((4.17, mid), (4.60, mid), GRAY)
    box(4.68, mid - .45, 2.10, .9, 'Geometric means\nand log-parameter\ncovariance', GRAY)
    sep = ys[2] - .22
    ax.plot([.08, WIDTH - .08], [sep, sep], lw=.8, color='#D5DCE2')

    ax.text(.08, sep - .11, 'b   PKPy2', fontsize=12, weight='bold', va='top', color=INK)
    w, gap, h = 1.53, .17, 1.28
    y0 = sep - .40 - h
    boxes = [('Event records', 'doses, infusions,\nsteady state,\ncovariates, BLQ data'),
             ('Declared model', 'structure or ODE,\nIIV and IOV,\nfixed values, bounds'),
             ('Joint estimation', 'marginal likelihood,\nindependent\nnumerical audit'),
             ('Diagnostics', 'CWRES, NPDE,\nVPC, intervals,\ncovariate search')]
    for i, (title, detail) in enumerate(boxes):
        x = .10 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle='round,pad=0,rounding_size=0.08',
                                    lw=1.1, edgecolor=BLUE, facecolor='white'))
        ax.text(x + w / 2, y0 + h - .21, title, ha='center', va='center', fontsize=11, weight='bold', color=INK)
        ax.text(x + w / 2, y0 + (h - .40) / 2, detail, ha='center', va='center', fontsize=10.5, color=INK,
                linespacing=1.3)
        if i < 3:
            arrow((x + w + .03, y0 + h / 2), (x + w + gap - .03, y0 + h / 2), BLUE)
    check_and_save(fig, 'Figure_1', report, pad=0)


# ------------------------------------------------------------------ Figure 6
def figure_verification(report):
    pred = read(V / 'prediction_agreement.json')
    lik = read(V / 'likelihood_agreement.json')
    eng = read(V / 'general_vs_classic.json')
    cmp = read(V / 'vs_nlmixr2.json') if (V / 'vs_nlmixr2.json').exists() else []
    wpd = read(V / 'warfarin_pkpd/exact_ofv.json') if (V / 'warfarin_pkpd/exact_ofv.json').exists() else []
    pairs = list(csv.DictReader((V / 'diagnostics/paired.csv').open()))
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 6.9), layout='constrained')
    rng = np.random.default_rng(3)

    # a: predictions against rxode2
    ax = axes[0, 0]
    for j, (group, color, marker) in enumerate([('Linear', BLUE, 'o'), ('ODE', ORANGE, 's')]):
        vals = [r['max_relative_difference'] for r in pred if (r['scenario'] in ODE) == (group == 'ODE')]
        ax.scatter(j + rng.uniform(-.13, .13, len(vals)), vals, s=34, color=color, marker=marker,
                   edgecolor='white', linewidth=.4, zorder=3)
        ax.text(j, 3e-5, f'n = {len(vals)}', ha='center', va='bottom', fontsize=10, color=INK)
    ax.set_yscale('log')
    ax.set_ylim(1e-14, 1e-4)
    ax.set_xticks([0, 1], ['Linear\nsystems', 'Nonlinear\nODE models'])
    ax.set_xlim(-.6, 1.6)
    ax.set_ylabel('Maximum relative difference')
    ax.grid(axis='y', color=GRID, lw=.6)
    panel(ax, 'a   Predictions vs rxode2')

    # b: marginal OFV against independent quadrature and the compact interface
    ax = axes[0, 1]
    groups = [('Independent\nquadrature', [abs(r['difference']) for r in lik], PURPLE, 'D'),
              ('Compact\ninterface', [abs(r['ofv_difference']) for r in eng], GREEN, '^')]
    for j, (label, vals, color, marker) in enumerate(groups):
        ax.scatter(j + rng.uniform(-.13, .13, len(vals)), vals, s=34, color=color, marker=marker,
                   edgecolor='white', linewidth=.4, zorder=3)
        ax.text(j, 2e-2, f'n = {len(vals)}', ha='center', va='bottom', fontsize=10, color=INK)
    ax.axhline(.05, color=GRAY, ls='--', lw=.9)
    ax.set_yscale('log')
    ax.set_ylim(1e-6, 1e-1)
    ax.set_xticks([0, 1], [g[0] for g in groups])
    ax.set_xlim(-.6, 1.6)
    ax.set_ylabel('|OFV difference|')
    ax.grid(axis='y', color=GRID, lw=.6)
    panel(ax, 'b   Marginal likelihood')

    # c: residual diagnostics against nlmixr2 at identical parameters
    ax = axes[1, 0]
    for q, color, marker in [('CWRES', BLUE, 'o'), ('IWRES', ORANGE, 's')]:
        a = np.array([[float(r['pkpy2']), float(r['nlmixr2'])] for r in pairs if r['quantity'] == q])
        ax.scatter(a[:, 1], a[:, 0], s=16, color=color, marker=marker, alpha=.8, label=q, edgecolor='none')
    lim = [-3.5, 3.5]
    ax.plot(lim, lim, color=INK, lw=.8)
    ax.set(xlim=lim, ylim=lim, xlabel='nlmixr2', ylabel='PKPy2')
    ax.legend(frameon=False, loc='upper left', handletextpad=.2)
    ax.grid(color=GRID, lw=.6)
    panel(ax, 'c   Residuals vs nlmixr2')

    # d: exact OFV at nlmixr2 estimates minus at PKPy2 estimates
    ax = axes[1, 1]
    names = {'infusion_block_covariates': 'Infusion, block Ω', 'oral_blq_m3': 'Oral, M3 censoring',
             'lognormal_residual': 'Log-normal error', 'michaelis_menten': 'Michaelis-Menten'}
    rows = [(names[r['scenario']], r['exact_ofv_difference'], ORANGE, '^') for r in cmp
            if r.get('exact_ofv_difference') is not None]
    base = next((r['ofv'] for r in wpd if r['source'] == 'PKPy2'), None)
    for r in wpd:
        if r['source'] == 'nlmixr2 focei':
            rows.append(('Warfarin PK/PD', r['ofv'] - base, ORANGE, '^'))
        if r['source'] == 'nlmixr2 saem':
            rows.append(('Warfarin PK/PD', r['ofv'] - base, GREEN, 's'))
    labels = list(dict.fromkeys(r[0] for r in rows))
    for label, value, color, marker in rows:
        y = labels.index(label)
        ax.scatter(value, y, s=46, color=color, marker=marker, edgecolor='white', linewidth=.4, zorder=3)
    ax.axvline(0, color=INK, lw=.8)
    ax.set_xscale('symlog', linthresh=1., linscale=.6)
    values = [r[1] for r in rows] or [1.]
    hi = max(1., max(values))
    lo = min(0., min(values))
    ticks = [t for t in (-100, -10, -1, 0, 1, 10, 100, 1000) if lo * 1.5 - .5 <= t <= hi * 1.5 + .5]
    ax.set_xticks(ticks, [f'{t:g}' for t in ticks])
    ax.set_xlim(min(lo * 1.8, -1.2), hi * 2.2)
    ax.minorticks_off()
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlabel('ΔOFV (nlmixr2 − PKPy2)')
    ax.grid(axis='x', color=GRID, lw=.6)
    handles = [plt.Line2D([], [], ls='', marker='^', color=ORANGE, label='FOCEi'),
               plt.Line2D([], [], ls='', marker='s', color=GREEN, label='SAEM')]
    ax.legend(handles=handles, frameon=False, loc='upper right')
    panel(ax, 'd   Estimates vs nlmixr2')
    check_and_save(fig, 'Figure_6', report)


# ------------------------------------------------------------------ Figure 7
def figure_application(report):
    diag = read(V / 'warfarin_pkpd/pkpy2_diagnostics.json')
    tools = read(V / 'tools/theophylline_intervals.json')
    fig = plt.figure(figsize=(WIDTH, 7.6), layout='constrained')
    top, bottom = fig.subfigures(2, 1, height_ratios=[1, 1.15])
    axes = top.subplots(1, 2)
    for ax, (name, title, ylabel) in zip(axes, [('CP', 'a   Warfarin concentration', 'Concentration (mg/L)'),
                                                ('R', 'b   Prothrombin complex activity', 'PCA (%)')]):
        # drawn by the plotting module of PKPy2 from the saved vpc() result
        plots.vpc({name: dict(diag['vpc'][name], observations=diag['observations'][name])}, name, ax=ax, legend=False)
        ax.set(xlabel='Time (h)', ylabel=ylabel)
        ax.grid(color=GRID, lw=.6)
        panel(ax, title)
    handles = [plt.Line2D([], [], color=ORANGE, lw=1.5, label='Observed median'),
               plt.Line2D([], [], color=BLUE, lw=1.5, ls='--', label='Observed 5th, 95th'),
               matplotlib.patches.Patch(color=GRAY, alpha=.3, label='Simulated 95% interval')]
    top.legend(handles=handles, loc='outside lower center', ncol=3, frameon=False, fontsize=10)
    ax = bottom.subplots(1, 1)
    keys = ['theta:CL', 'theta:V', 'theta:Ka', 'omega:CL', 'omega:V', 'omega:Ka', 'sigma:CP:proportional']
    labels = [r'$CL$', r'$V$', r'$K_a$', r'$\omega^2_{CL}$', r'$\omega^2_{V}$', r'$\omega^2_{Ka}$', r'$\sigma_{prop}$']
    methods = [('wald', 'Wald', BLUE, 'o'), ('sandwich', 'Sandwich', ORANGE, 's'), ('profile', 'Profile', PURPLE, 'D'),
               ('bootstrap', 'Bootstrap', GREEN, '^'), ('sir', 'SIR', GRAY, 'v')]
    for i, key in enumerate(keys):
        est = tools['wald'][key]['estimate']
        for j, (m, label, color, marker) in enumerate(methods):
            if m == 'profile':
                iv = tools['profile'].get(key, {}).get('interval')
            elif m == 'bootstrap':
                iv = tools.get('bootstrap', {}).get('summary', {}).get(key, {}).get('interval')
            elif m == 'sir':
                iv = tools['sir']['summary'][key]['interval']
            else:
                iv = tools[m][key]['interval']
            if iv is None or any(v is None for v in iv):
                continue
            y = i + (j - 2) * .15
            ax.plot([iv[0] / est, iv[1] / est], [y, y], color=color, lw=2.2, solid_capstyle='butt',
                    label=label if i == 0 else None)
            ax.plot(1, y, marker=marker, color=color, ms=4)
    ax.axvline(1, color=INK, lw=.8)
    ax.set_xlim(0, None)
    ax.set_yticks(range(len(keys)), labels)
    ax.invert_yaxis()
    ax.set_xlabel('95% interval relative to the estimate')
    ax.grid(axis='x', color=GRID, lw=.6)
    ax.legend(frameon=False, ncol=1, loc='upper right', handlelength=1.4)
    panel(ax, 'c   Theophylline')
    check_and_save(fig, 'Figure_7', report)


# ------------------------------------------------------------------ Supplementary Figure S1
def figure_s1(report):
    """Diagnostic plots of the theophylline example (Online Resource 1, Listing 2), drawn by pkpy2.plots from
    the saved diagnostics (validate_pkpy2_theophylline_example.py) and stacked as panels a and b."""
    from io import BytesIO
    from PIL import Image, ImageDraw, ImageFont
    with open(V / 'example_theophylline/diagnostics.csv', newline='') as f:
        rows = list(csv.DictReader(f))
    table = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    images = []
    for fig in (plots.gof(table, figsize=(WIDTH, 4.3)), plots.individual_fits(table, figsize=(WIDTH, 5.0))):
        buf = BytesIO()
        fig.savefig(buf, dpi=300, format='png')
        plt.close(fig)
        images.append(Image.open(buf).convert('RGB'))
    head = 70                                          # room for the panel letter above each panel
    canvas = Image.new('RGB', (max(im.width for im in images), sum(im.height + head for im in images)), 'white')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('arialbd.ttf', 50)       # 12 pt at 300 dpi
    y = 0
    for letter, im in zip('ab', images):
        draw.text((20, y + 5), letter, font=font, fill=INK)
        canvas.paste(im, (0, y + head))
        y += im.height + head
    canvas.save(OUT / 'Figure_S1.png', dpi=(300, 300))
    report.append(f"Figure_S1: {canvas.size[0]} x {canvas.size[1]} px at 300 dpi")


def main():
    report = []
    which = sys.argv[1:] or ['1', '6', '7', 'S1']
    if '1' in which:
        figure_workflow(report)
    if '6' in which:
        figure_verification(report)
    if '7' in which:
        figure_application(report)
    if 'S1' in which:
        figure_s1(report)
    for r in report:
        print(r)


if __name__ == '__main__':
    main()
