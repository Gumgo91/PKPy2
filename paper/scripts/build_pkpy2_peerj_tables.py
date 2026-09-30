"""Build the new PeerJ Tables 5 and 6 as Word files in the style of the submitted tables.

Table 5: clinical datasets, estimates from NONMEM, PKPy2, nlmixr2 (FOCEi, SAEM) and saemix.
Table 6: primary simulation, PKPy2 versus nlmixr2 (FOCEi, SAEM) and saemix.
The submitted Table_3.docx is used as the formatting template (landscape, 9 pt, shaded header).
"""
from pathlib import Path
import copy
import json
import sys

import docx
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parents[1]
CMP = ROOT / 'output/pkpy2_software_comparison'
TEMPLATE = Path(sys.argv[1])
OUT = Path(sys.argv[2])
METHODS = ['pkpy2', 'nlmixr2_focei', 'nlmixr2_saem', 'saemix']
HEAD = ['PKPy2', 'nlmixr2 FOCEi', 'nlmixr2 SAEM', 'saemix']
TOTAL_WIDTH = 13008


def set_cell(tc, text, bold=None):
    p = tc.findall(qn('w:p'))[0]
    for extra in tc.findall(qn('w:p'))[1:]:
        tc.remove(extra)
    runs = p.findall(qn('w:r'))
    for r in runs[1:]:
        p.remove(r)
    r = runs[0]
    r.find(qn('w:t')).text = text
    r.find(qn('w:t')).set(qn('xml:space'), 'preserve')
    rpr = r.find(qn('w:rPr'))
    b = rpr.find(qn('w:b'))
    if bold and b is None:
        rpr.insert(0, rpr.makeelement(qn('w:b'), {}))
    if bold is False and b is not None:
        rpr.remove(b)


def build(title, footnote, header, rows, widths, out):
    d = docx.Document(TEMPLATE)
    paras = d.paragraphs
    for p, text in [(paras[0], title), (paras[1], footnote)]:
        for r in p.runs[1:]:
            r._r.getparent().remove(r._r)
        p.runs[0].text = text
    tbl = d.tables[0]._tbl
    trs = tbl.findall(qn('w:tr'))
    head_tc = trs[0].findall(qn('w:tc'))[0]
    data_tc = trs[1].findall(qn('w:tc'))[0]
    head_tr, data_tr = copy.deepcopy(trs[0]), copy.deepcopy(trs[1])
    for tr in trs:
        tbl.remove(tr)
    grid = tbl.find(qn('w:tblGrid'))
    for g in list(grid):
        grid.remove(g)
    tw = [round(TOTAL_WIDTH * w / sum(widths)) for w in widths]
    for w in tw:
        g = grid.makeelement(qn('w:gridCol'), {qn('w:w'): str(w)})
        grid.append(g)

    def make_row(proto_tr, proto_tc, cells, bold=None, span=False, left=False):
        tr = copy.deepcopy(proto_tr)
        for tc in tr.findall(qn('w:tc')):
            tr.remove(tc)
        values = [cells[0]] if span else cells
        for i, text in enumerate(values):
            tc = copy.deepcopy(proto_tc)
            tcpr = tc.find(qn('w:tcPr'))
            width = sum(tw) if span else tw[i]
            tcpr.find(qn('w:tcW')).set(qn('w:w'), str(width))
            if span:
                gs = tcpr.makeelement(qn('w:gridSpan'), {qn('w:val'): str(len(tw))})
                tcpr.insert(1, gs)
                shd = tcpr.find(qn('w:shd'))
                if shd is None:
                    shd = tcpr.makeelement(qn('w:shd'), {})
                    tcpr.insert(2, shd)
                shd.set(qn('w:val'), 'clear'); shd.set(qn('w:color'), 'auto'); shd.set(qn('w:fill'), 'F2F2F2')
            if left or (i == 0 and not span):
                jc = tc.find(qn('w:p')).find(qn('w:pPr')).find(qn('w:jc'))
                if jc is not None and (span or left):
                    jc.set(qn('w:val'), 'left')
            set_cell(tc, text, bold)
            tr.append(tc)
        return tr

    tbl.append(make_row(head_tr, head_tc, header, bold=True))
    for row in rows:
        if isinstance(row, str):
            tbl.append(make_row(data_tr, data_tc, [row], bold=True, span=True, left=True))
        else:
            tbl.append(make_row(data_tr, data_tc, row, bold=False))
    d.save(out)


def f(x, digits):
    if x is None:
        return '—'
    x = 0.0 if round(x, digits) == 0 else x
    return f'{x:.{digits}f}'.replace('-', '−')


def table5(cmp):
    check = json.loads((CMP / 'clinical_likelihood_check.json').read_text(encoding='utf-8'))
    ofv = {(r['dataset'], r['method']): r['ofv'] for r in check}
    rows = []
    blocks = [('theophylline', 'Theophylline (12 subjects)'), ('warfarin', 'Warfarin (32 subjects)')]
    if '--tobramycin' in sys.argv:
        blocks.append(('tobramycin', 'Tobramycin (97 subjects)'))
    for dataset, head in blocks:
        rows.append(head)
        for r in cmp['clinical']['rows']:
            if r['dataset'] != dataset or r['nonmem'] is None:
                continue
            digits = 3 if r['nonmem'] >= 1 else 4
            cells = [r['label'], f(r['nonmem'], digits)]
            for m in METHODS:
                v = r.get(m)
                cells.append('—' if v is None else f"{f(v, digits)} ({f(r[m + '_diff_pct'], 1)})")
            rows.append(cells)
        rows.append(['Marginal OFV', f(ofv.get((dataset, 'nonmem')), 2)]
                    + [f(ofv.get((dataset, m)), 2) for m in METHODS])
    build('Table 5. Clinical datasets: estimates from NONMEM, PKPy2, nlmixr2 and saemix.',
          'Estimates with the relative difference from NONMEM (%) in parentheses. NONMEM, expert analyses [11]. CL and V are standardized '
          'to 70 kg. Marginal OFV: −2 log L of the declared model at each set of '
          'estimates, evaluated by independent adaptive Gauss-Hermite quadrature (Supplementary S8); for NONMEM, σ_prop (not reported) was '
          'set to its maximum-likelihood value. ' + (
          'Tobramycin was fitted with the expert-judgment constraints (WT exponent of V1 fixed at 1, V1 ≤ 10 L, V2 ≤ 30 L; Methods); '
          'nlmixr2 SAEM does not apply parameter bounds. saemix was not applied to warfarin or tobramycin. '
          if '--tobramycin' in sys.argv else 'saemix was not applied to warfarin. ') +
          '—, not available. '
          'Standard errors are given in Supplementary Table S13.',
          ['Parameter', 'NONMEM'] + HEAD, rows, [1.25, 1, 1.15, 1.15, 1.15, 1.15], OUT / 'Table_5.docx')


def table6(cmp):
    est = {(r['sampling'], r['method'], r['parameter']): r for r in cmp['estimation']}
    status = {(r['sampling'], r['method']): r for r in cmp['status']}
    cov = {(r['sampling'], r['method'], r['parameter']): r for r in cmp['coverage']}
    labels = {'theta_CL': 'CL', 'theta_V': 'V', 'omega_CL': 'ω²(CL)', 'omega_V': 'ω²(V)', 'sigma_prop': 'σ_prop'}
    rows = []
    for s, head in [('rich', 'Rich sampling (100 datasets)'), ('sparse', 'Sparse sampling (100 datasets)')]:
        rows.append(head)
        rows.append(['Estimates returned', ''] + [str(status[(s, m)]['returned']) for m in METHODS])
        for p, lab in labels.items():
            rows.append([lab, 'Bias (RMSE), %'] + [f"{f(est[(s, m, p)]['relative_bias_pct'], 2)} ({f(est[(s, m, p)]['relative_rmse_pct'], 2)})"
                                                    for m in METHODS])
        for p in ['theta_CL', 'theta_V']:
            rows.append([labels[p], '95% CI coverage, %'] + [
                '—' if cov[(s, m, p)]['rate'] is None else
                f"{100 * cov[(s, m, p)]['rate']:.0f} (n = {cov[(s, m, p)]['available']})" for m in METHODS])
        rows.append(['Median fit time, s', ''] + [f(status[(s, m)]['median_fit_seconds'], 1) for m in METHODS])
    build('Table 6. Primary simulation: PKPy2 compared with established open-source NLME software.',
          'Relative bias and RMSE (%) against the generating values CL = 4 L/h, V = 40 L, ω²(CL) = 0.09, ω²(V) = 0.04 and σ_prop = 0.15. '
          'Coverage refers to 95% Wald intervals on the log scale; n, datasets with an interval. PKPy2 fit times exclude the uncertainty '
          'calculation. Paired-bootstrap RMSE differences are given in Supplementary Table S12.',
          ['Parameter', 'Quantity'] + HEAD, rows, [1.1, 1.3, 1.1, 1.1, 1.1, 1.1], OUT / 'Table_6.docx')


SUP = str.maketrans('-0123456789', '⁻⁰¹²³⁴⁵⁶⁷⁸⁹')


def sci(x):
    mantissa, exponent = f'{x:.2e}'.split('e')
    return f'{mantissa} × 10{str(int(exponent)).translate(SUP)}'


def table2():
    ref = json.loads((ROOT / 'output/pkpy2_development/reference_v3/results.json').read_text(encoding='utf-8'))
    rows = [['Sparse' if c['sparse'] else 'Rich', 'Estimated' if c['covariate'] else 'Absent', sci(c['node_refinement_ofv_gap']),
             sci(c['engine_ofv_gap']), sci(c['maximum_coordinate_gap']), sci(100 * c['maximum_relative_coordinate_se_gap'])]
            for c in ref['cases']]
    build('Table 2. Independent quadrature comparisons in one-random-effect intravenous models.',
          'Coordinates are logarithms of positive parameters and untransformed covariate coefficients; interindividual variability uses '
          'log-SD coordinates. SE differences refer to these coordinates. Node refinement increased quadrature from 4,097 to 8,193 nodes.',
          ['Sampling', 'WT effect on CL', 'Node-refinement OFV gap', 'Same-point OFV gap', 'Maximum coordinate gap',
           'Maximum SE difference (%)'], rows, [1, 1, 1.2, 1.2, 1.2, 1.2], OUT / 'Table_2.docx')


def table4(cmp):
    rows = []
    for r in cmp['clinical']['rows']:
        if r['dataset'] != 'warfarin':
            continue
        rows.append([r['label'], 'NR' if r['nonmem'] is None else f(r['nonmem'], 4 if r['nonmem'] < 1 else 3), f(r['pkpy2'], 5),
                     'NR' if r['pkpy2_diff_pct'] is None else f(r['pkpy2_diff_pct'], 2),
                     'NR' if r['nonmem_rse'] is None else f(r['nonmem_rse'], 1), f(r['pkpy2_rse'], 1)])
    build('Table 4. Warfarin: comparison with the expert NONMEM analysis.',
          'NONMEM estimates and RSEs are from the expert analysis [11]. CL and V are standardized to 70 kg. Ka interindividual variance '
          '(0.5), additive variance (0.117), and WT exponents on CL and V (0.75 and 1) were fixed. NR, not reported.',
          ['Parameter', 'NONMEM estimate', 'PKPy2 estimate', 'Relative difference (%)', 'NONMEM RSE (%)', 'PKPy2 RSE (%)'],
          rows, [1.3, 1, 1, 1, 1, 1], OUT / 'Table_4.docx')


def table3(cmp):
    rows = []
    for r in cmp['clinical']['rows']:
        if r['dataset'] != 'theophylline':
            continue
        rows.append([r['label'], 'NR' if r['nonmem'] is None else f(r['nonmem'], 4), f(r['pkpy2'], 4),
                     'NR' if r['pkpy2_diff_pct'] is None else f(r['pkpy2_diff_pct'], 2),
                     'NR' if r['nonmem_rse'] is None else f(r['nonmem_rse'], 1), f(r['pkpy2_rse'], 1)])
    build('Table 3. Theophylline: comparison with the expert NONMEM analysis.',
          'NONMEM estimates and RSEs are from the expert analysis [11]; CL and V were reported per kg and are shown for 70 kg. '
          'NR, not reported.',
          ['Parameter', 'NONMEM estimate', 'PKPy2 estimate', 'Relative difference (%)', 'NONMEM RSE (%)', 'PKPy2 RSE (%)'],
          rows, [1.3, 1, 1, 1, 1, 1], OUT / 'Table_3.docx')


def retitle(name, title, footnote):
    """Copy a submitted table unchanged except for its title and footnote."""
    d = docx.Document(TEMPLATE.parent / name)
    paras = [p for p in d.paragraphs if p.text.strip()]
    assert len(paras) == 2 and paras[0].text.startswith(title.split(':')[0])
    for p, text in zip(paras, [title, footnote]):
        p.runs[0].text = text
        for r in p.runs[1:]:
            r._r.getparent().remove(r._r)
    d.save(OUT / name)


def table1():
    """Submitted Table 1 extended with the event-record interface (rows renamed, edited and added)."""
    d = docx.Document(TEMPLATE.parent / 'Table_1.docx')
    tbl = d.tables[0]

    def set_row(row, texts):
        for cell, text in zip(row.cells, texts):
            if text is None:
                continue
            runs = cell.paragraphs[0].runs
            runs[0].text = text
            for r in runs[1:]:
                r._r.getparent().remove(r._r)

    def row_named(name):
        return next(r for r in tbl.rows if r.cells[0].text == name)

    set_row(row_named('Basic structures'), [
        'Structural models', None,
        'The same four closed-form structures with oral lag; through event records, one- to three-compartment, zero-order, '
        'transit, parent-metabolite, Michaelis-Menten, target-mediated, effect-compartment and indirect-response models and '
        'user-defined ODEs'])
    set_row(row_named('Interindividual variability'), [
        'Random effects and residual error', None,
        'Diagonal or block Ω and interoccasion variability estimated jointly with additive, proportional, combined, or '
        'log-normal residual error for each output'])
    set_row(row_named('Fixed components'), [
        'Fixed components and bounds', None,
        'Explicit fixed or estimated θ, Ω, σ, and covariate coefficients, with optional lower and upper bounds'])
    set_row(row_named('Uncertainty evaluated here'), [
        None, None, 'Full local population information; sandwich, profile-likelihood, bootstrap, and SIR intervals'])
    set_row(row_named('Repeated dosing'), [
        'Dosing and observations', None,
        'Explicit dose histories and analytical superposition; NONMEM event records with infusions, steady state, additional '
        'doses, resets, several outputs, and censored observations'])
    anchor = row_named('Random effects and residual error')._tr
    for texts in (['Covariate models', 'Regression of individual estimates on covariates with forward selection',
                   'Power, exponential, linear, and categorical effects, including time-varying covariates, within the population '
                   'model; stepwise selection by likelihood-ratio tests'],
                  ['Diagnostics', 'Goodness-of-fit plots of the individual fits',
                   'PRED, IPRED, CWRES, NPDE, shrinkage, visual predictive checks, and diagnostic plots']):
        new_tr = copy.deepcopy(anchor)
        anchor.addnext(new_tr)
        anchor = new_tr
        row = next(r for r in tbl.rows if r._tr is new_tr)
        set_row(row, texts)
    d.save(OUT / 'Table_1.docx')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cmp = json.loads((CMP / 'comparison_summary.json').read_text(encoding='utf-8'))
    table1()
    table3(cmp)
    table2()
    table4(cmp)
    table5(cmp)
    table6(cmp)
    print('tables written to', OUT)


if __name__ == '__main__':
    main()
