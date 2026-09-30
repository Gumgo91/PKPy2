"""Build the PKPy2 supplement for the PeerJ revision from the current (v3) results.

All tables are computed from the saved fit records and summaries; the document is rendered
to PDF with PyMuPDF. Set INCLUDE_TOBRAMYCIN to add the tobramycin application.
Usage: python build_pkpy2_peerj_supplement.py <output.pdf>
"""
from collections import Counter
from pathlib import Path
import html
import io
import json
import math
import re
import sys

import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'docs/pkpy2_paper'
DEV = ROOT / 'output/pkpy2_development'
CMP = ROOT / 'output/pkpy2_software_comparison'
TOB = ROOT / 'output/pkpy2_tobramycin_v3'
OUT = Path(sys.argv[1])
INCLUDE_TOBRAMYCIN = '--tobramycin' in sys.argv
ARTICLE = dict(title='PKPy2: A Python framework for joint population pharmacokinetic estimation and uncertainty assessment',
               journal='Journal of Pharmacokinetics and Pharmacodynamics', authors='Hyunseung Kong, Inyoung Kim',
               corresponding='Inyoung Kim, Department of Defense Science, Korea National Defense University, Nonsan, Republic of Korea; inyoungkim@korea.kr')

LAB = {'pkpy2': 'PKPy2', 'nlmixr2_focei': 'nlmixr2 FOCEi', 'nlmixr2_saem': 'nlmixr2 SAEM', 'saemix': 'saemix', 'nonmem': 'NONMEM',
       'pkpy': 'PKPy', 'gaussian_tst': 'Gaussian TST'}
PAR = {'theta_CL': 'CL', 'theta_V': 'V', 'omega_CL': 'ω²(CL)', 'omega_V': 'ω²(V)', 'sigma_prop': 'σ_prop'}
MODEL = {'1cmt_oral': 'One-compartment oral', '2cmt_iv': 'Two-compartment IV', '2cmt_oral': 'Two-compartment oral'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def f(x, d=2):
    if x is None:
        return 'NR'
    x = 0.0 if round(x, d) == 0 else x
    return f'{x:.{d}f}'


def g(x, d=5):
    return 'NR' if x is None else f'{x:.{d}g}' if abs(x) >= 1e4 or abs(x) < 1e-3 else f'{x:.{d}f}'.rstrip('0').rstrip('.') if False else f'{x:.5f}'


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
                     + ['| ' + ' | '.join(map(str, r)) + ' |' for r in rows])


def intervals(fit):
    unc = fit.get('uncertainty') or {}
    return {r['coordinate']: r for r in unc.get('information', {}).get('data', {}).get('intervals', [])}


def route(fit):
    return 'Initial refinement' if (fit.get('estimation') or {}).get('saem_status') == 'not_needed' else 'SAEM then refinement'


def multimodal(fit):
    lx = (fit.get('estimation') or {}).get('laplace_exploration') or {}
    return len(lx.get('distinct_optima') or []) > 1


def paths():
    rows = Counter()
    flagged = Counter()
    for group, pattern, cond in [('Primary', 'confirmatory_v3', lambda r: r['sampling'].title()),
                                 ('Additional', 'secondary_v3', lambda r: MODEL[r['model']])]:
        for p in sorted((DEV / pattern).glob('*__[0-9][0-9][0-9].json')):
            rec = read(p)
            fit = rec['pkpy2']
            rows[(group, cond(rec), route(fit), fit['status'].title())] += 1
            flagged[(group, cond(rec))] += int(multimodal(fit))
    return rows, flagged


def unresolved_secondary():
    out = []
    for p in sorted((DEV / 'secondary_v3').glob('*__[0-9][0-9][0-9].json')):
        rec = read(p)
        fit = rec['pkpy2']
        if not fit['converged']:
            stages = fit['estimation'].get('refinement_stages') or []
            last = stages[-1] if stages else {}
            out.append(dict(model=rec['model'], replicate=rec['replicate'], message=fit['estimation'].get('refinement_message'),
                            score=last.get('projected_score_max')))
    return out


CLIN_LABEL = {'log_theta:CL': 'CL/F (L/h)', 'log_theta:V': 'V/F (L)', 'log_theta:Ka': 'Ka (h⁻¹)', 'log_theta:ALAG': 'ALAG (h)',
              'log_omega_sd:CL': 'ω²(CL)', 'log_omega_sd:V': 'ω²(V)', 'log_omega_sd:Ka': 'ω²(Ka)', 'log_sigma_prop': 'σ_prop',
              'log_sigma_add': 'σ_add (mg/L)', 'log_theta:V1': 'V1 (L)', 'log_theta:Q': 'Q (L/h)', 'log_theta:V2': 'V2 (L)',
              'covariate_coefficient:0': 'CLCR exponent', 'covariate_coefficient:1': 'WT exponent'}


def clinical_rows(name, fit):
    rows = []
    for coord, r in intervals(fit).items():
        label = CLIN_LABEL[coord]
        if name == 'Tobramycin' and coord == 'log_theta:CL':
            label = 'CL (L/h)'
        rows.append([name, label, f'{r["estimate"]:.5g}', f'{r["se"]:.3g}', f(r['rse_pct'], 1) if r['rse_pct'] is not None else 'NR',
                     f'{r["interval"][0]:.5g} to {r["interval"][1]:.5g}'])
    return rows


TOB_EXPERT = dict(CL=2.95, V1=4.59, Q=6.85, V2=13.2, b_clcr=0.236, b_wt=1.07, omega_CL=0.028)
TOB_SETTINGS = [
    ('1', 'None', 'start', ''),
    ('2', 'Expert NONMEM bounds', 'bounded_start', ''),
    ('3', 'WT exp. = 1; V2 ≤ 30 L', 'judgment_start', ''),
    ('4', 'Expert judgment', 'expert_start', ''),
]


def tobramycin_section(check):
    """S10: tobramycin fits without constraints, with the expert bounds, and with the expert-judgment constraints."""
    import csv
    rows = list(csv.DictReader((CMP / 'source/tobramycin_github.csv').read_text(encoding='utf-8').lstrip('#').splitlines()))
    ids = list(dict.fromkeys(r['ID'] for r in rows))
    tad, wt, clcr, nobs = [], [], [], []
    for i in ids:
        rr = [r for r in rows if r['ID'] == i]
        doses = [float(r['TIME']) for r in rr if r['EVID'] == '1']
        obs = [float(r['TIME']) for r in rr if r['EVID'] == '0']
        tad += [t - max(d for d in doses if d <= t) for t in obs]
        nobs.append(len(obs)); wt.append(float(rr[0]['WT'])); clcr.append(float(rr[0]['CLCR']))
    at25 = sum(abs(t - 2.5) < 1e-9 for t in tad)
    lw, lc = [math.log(x) for x in wt], [math.log(x) for x in clcr]
    mw, mc = sum(lw) / len(lw), sum(lc) / len(lc)
    corr = sum((a - mw) * (b - mc) for a, b in zip(lw, lc)) / math.sqrt(sum((a - mw) ** 2 for a in lw) * sum((b - mc) ** 2 for b in lc))

    def cells(fit):
        b = fit['coefficients']
        return [f(fit['theta']['CL']), f(fit['theta']['V1']), f(fit['theta']['Q']), f(fit['theta']['V2'], 1), f(b[0], 3),
                f(b[1], 2) if len(b) > 1 else '1 (fixed)', f(fit['omega']['CL'], 4)]
    s15 = []
    for no, label, stem, _ in TOB_SETTINGS:
        for k, start in [(1, 'Documented'), (2, 'Expert estimates')]:
            fit = read(TOB / f'{stem}_{k}.json')
            assert fit['converged'], (stem, k)
            optima = len(((fit.get('estimation') or {}).get('laplace_exploration') or {}).get('distinct_optima') or [])
            s15.append([no if k == 1 else '', label if k == 1 else '', start, f(fit['ofv'], 2)] + cells(fit) + [optima])
    nonmem = next(r for r in check if r['dataset'] == 'tobramycin' and r['method'] == 'nonmem')
    e = TOB_EXPERT
    s15.append(['', 'Expert NONMEM estimates [11]', '', f(nonmem['ofv'], 2), f(e['CL']), f(e['V1']), f(e['Q']), f(e['V2'], 1),
                f(e['b_clcr'], 3), f(e['b_wt'], 2), f(e['omega_CL'], 4), ''])
    t15 = table(['Setting', 'Constraints', 'Start', 'OFV', 'CL', 'V1', 'Q', 'V2', 'CLCR exp.', 'WT exp.', 'ω²(CL)', 'Laplace optima'], s15)

    prof = read(TOB / 'profile_v2.json')
    t16 = table(['V2 (L)', 'OFV', 'CL', 'V1', 'Q', 'CLCR exp.', 'WT exp.', 'ω²(CL)'],
                [[f'{r["V2"]:g}', f(r['ofv'], 2), f(r['CL']), f(r['V1']), f(r['Q']), f(r['b_clcr'], 3), f(r['b_wt'], 2), f(r['omega_CL'], 4)]
                 for r in prof['profile']])
    pv = [r['ofv'] for r in prof['profile']]
    assert all(r['b_wt'] < 0 for r in prof['profile'])

    r_rows = []
    for key, label in [('tobramycin', 'Published model, no bounds'), ('tobramycin_expert', 'Expert judgment')]:
        for m in ['nlmixr2_focei', 'nlmixr2_saem', 'saemix']:
            p = CMP / 'results/clinical' / f'{key}__{m}.json'
            if not p.exists():
                continue
            rec = read(p)
            th, b = rec['theta'], rec.get('coefficients') or {}
            r_rows.append([label, LAB[m], f(th['CL']), f(th['V1']), f(th['Q']), f(th['V2'], 1), f(b.get('b.clcr'), 3),
                           f(b['b.wt'], 2) if 'b.wt' in b and b['b.wt'] is not None and key == 'tobramycin' else '1 (fixed)',
                           f(rec['omega']['CL'], 4)])
    t17 = table(['Model', 'Program', 'CL', 'V1', 'Q', 'V2', 'CLCR exp.', 'WT exp.', 'ω²(CL)'], r_rows)
    fc = read(TOB / 'focei_check.json')
    pts = {p['pkpy2_fit']: p for p in fc['points']}
    ffit = fc['focei_fit_from_documented_start']
    u = read(TOB / 'start_1.json')
    ej = read(TOB / 'expert_start_1.json')
    th = ej['theta']
    assert th['V1'] < 10 * (1 - 1e-3) and th['V2'] < 30 * (1 - 1e-3) and th['CL'] < 20 and th['Q'] < 50
    assert 0 < ej['coefficients'][0] < 2 and ej['uncertainty']['status'] != 'boundary_estimate'

    return f'''
## S10. Tobramycin: fits without and with expert-judgment constraints

**Data and model.** The tobramycin dataset of the PKGPT repository [11] contains {len(ids)} patients, {sum(r['EVID'] == '1' for r in rows)} intravenous bolus doses, and {len(tad)} concentrations ({min(nobs)} to {max(nobs)} per patient). Of these concentrations, {at25} ({100 * at25 / len(tad):.0f}%) were measured 2.5 h after the preceding dose. Body weight ranged from {min(wt):.0f} to {max(wt):.0f} kg and was correlated with creatinine clearance (Pearson correlation of the logarithms, {corr:.2f}). The expert model (Table S6) is a two-compartment model with CL = θ₁(CLCR/58)^θ₂ exp(η), V1 = θ₃(WT/62)^θ₄, Q = θ₅ and V2 = θ₆. The documented initial values of the expert control stream were CL 3.4 L/h, V1 20.3 L, Q 5 L/h, V2 20 L, and both exponents 1, with ω²(CL) 0.1; σ_prop started at 0.2. Each setting was fitted from these values (with V1 = 8 L when V1 was restricted to 10 L) and from the expert estimates.

**Settings.** (1) The published model without bounds. (2) The $THETA bounds of the expert control stream: CL (0, 20) L/h, CLCR exponent (0, 2), V1 (0, 100) L, WT exponent (0, 2), Q (0, 50) L/h, V2 (0, 100) L. (3) The WT exponent fixed at 1 with V2 ≤ 30 L and the other expert bounds. (4) The expert-judgment constraints used in the main comparison: V1 proportional to body weight (WT exponent fixed at 1), a small central compartment (V1 ≤ 10 L), a peripheral volume consistent with the extracellular distribution of aminoglycosides (V2 ≤ 30 L), and the other expert bounds.

**Table S15. PKPy2 tobramycin estimates under each setting.**

{t15}

CL and Q in L/h, volumes in L, at CLCR 58 mL/min and WT 62 kg. OFV, exact marginal −2 log L reported by PKPy2. For the expert NONMEM estimates, the OFV was computed by the independent quadrature of S8 with σ_prop at its maximum-likelihood value. Laplace optima, number of distinct optima found by the Laplace exploration. All fits met the convergence criteria.

**Findings.** Without constraints, both starts converged to the same maximum-likelihood estimate (OFV {u['ofv']:.2f}), with a WT exponent of {u['coefficients'][1]:.1f} and V2 of {u['theta']['V2']:.0f} L. The independent 60-node Gauss-Hermite calculation used for the profile in Table S16 gave {prof['check']['independent']:.3f} at this estimate (PKPy2, {prof['check']['pkpy2_reported']:.3f}). Across V2 from {prof['profile'][0]['V2']:g} to {prof['profile'][-1]['V2']:g} L, the profile OFV varied by {max(pv) - min(pv):.1f} units, and every profile point had a negative WT exponent. With the expert bounds, the best solution lay on the bounds (WT exponent {read(TOB / 'bounded_start_1.json')['coefficients'][1]:.2f}, V2 {read(TOB / 'bounded_start_1.json')['theta']['V2']:.0f} L), and the fit started from the expert estimates stayed near them with an OFV {read(TOB / 'bounded_start_2.json')['ofv'] - read(TOB / 'bounded_start_1.json')['ofv']:.1f} units higher. The nlmixr2 FOCEi objective confirmed this ordering: {pts['bounded_start_1']['focei_objective']:.1f} at the bounded optimum and {pts['bounded_start_2']['focei_objective']:.1f} at the solution near the expert estimates (published NONMEM OFV, {fc['published_expert_nonmem_ofv']:.2f}); an nlmixr2 FOCEi fit with the expert bounds from the documented initial values stopped at a third solution (V1 {ffit['V1']:.1f} L, V2 {ffit['V2']:.1f} L, WT exponent {ffit['b_wt']:.2f}; objective {ffit['objective']:.1f}), {ffit['objective'] - pts['bounded_start_1']['focei_objective']:.1f} units above the FOCEi objective at the bounded PKPy2 optimum. Fixing the WT exponent and limiting V2 (setting 3) was not sufficient from the documented initial values, which reached V1 {read(TOB / 'judgment_start_1.json')['theta']['V1']:.1f} L with V2 on its bound. With the expert-judgment constraints (setting 4), both starts reached the same estimate, close to the expert estimates, with no constraint active; its local 95% intervals are given in Table S8.

**Table S16. Profile likelihood in V2 for the published model without constraints.**

{t16}

Independent calculation: 60-node Gauss-Hermite quadrature for the CL random effect, closed-form two-compartment predictions, and Nelder-Mead optimization of the other parameters from three starts at each V2.

**Table S17. nlmixr2 and saemix tobramycin estimates.**

{t17}

Without constraints, saemix used a proportional error model because it cannot fix the additive term. With the expert-judgment constraints, nlmixr2 SAEM applied the fixed WT exponent but not the bounds, and saemix was not applied.
'''


def markdown():
    summary = read(PAPER / 'analysis_summary.json')
    methods = read(PAPER / 'methods_audit.json')
    cmp = read(CMP / 'comparison_summary.json')
    check = read(CMP / 'clinical_likelihood_check.json')
    speed = read(DEV / 'recurrence_benchmark.json')
    theo = read(DEV / 'theophylline_v3/proportional.json')
    theo_comb = read(DEV / 'theophylline_v3/combined.json')
    warf1 = read(DEV / 'warfarin_v3/warfarin_start_1.json')
    warf2 = read(DEV / 'warfarin_v3/warfarin_start_2.json')
    env = cmp['clinical']['environment']
    records = [read(p) for p in (CMP / 'results').glob('*/*.json')]
    records = [r for r in records if r['dataset'] in ('theophylline', 'warfarin') or r.get('sampling') is not None
               or (INCLUDE_TOBRAMYCIN and r['dataset'] in ('tobramycin', 'tobramycin_expert'))]
    datasets = ['theophylline', 'warfarin'] + (['tobramycin'] if INCLUDE_TOBRAMYCIN else [])

    s1 = table(['Sampling', 'Method', 'Parameter', 'Set', 'n', 'Bias (%)', 'RMSE (%)'],
               [[r['sampling'].title(), LAB[r['engine']], PAR[r['parameter']], 'All' if r['population'] == 'all_returned_points' else 'Accepted',
                 r['n'], f(r['relative_bias_pct']), f(r['relative_rmse_pct'])] for r in summary['primary_estimation']])
    s2 = table(['Sampling', 'Parameter', 'Available / planned', 'Covered / available', 'Wilson 95% interval', 'Median relative width'],
               [[r['sampling'].title(), PAR[r['parameter']], f"{r['availability']['k']}/{r['availability']['n']}",
                 f"{r['conditional_coverage']['k']}/{r['conditional_coverage']['n']}",
                 f"{100 * r['conditional_coverage']['interval'][0]:.1f}-{100 * r['conditional_coverage']['interval'][1]:.1f}%",
                 f(r['relative_width_median'], 3)] for r in summary['primary_intervals']])
    s3 = table(['Sampling', 'Comparator', 'Parameter', 'Paired n', 'Difference (pp)', 'Bootstrap 95% interval (pp)'],
               [[r['sampling'].title(), LAB[r['comparator']], PAR[r['parameter']], r['n_pairs'], f(r['rmse_difference_pct']),
                 f"{f(r['bootstrap_95_interval'][0])} to {f(r['bootstrap_95_interval'][1])}"] for r in summary['paired_rmse']])

    def values(d):
        return '; '.join(f'{k} = {v:g}' for k, v in d.items() if k != 'ALAG')
    s4 = table(['Model', 'Structural truth', 'IIV variance truth', 'Structural start'],
               [[MODEL[s['model']], values(s['theta']), values(s['omega']), values(s['initial_theta'])] for s in methods['secondary_settings']])

    def plabel(name):
        if name.startswith('omega_'):
            return 'ω²(' + name.removeprefix('omega_') + ')'
        if name.startswith('theta_'):
            return name.removeprefix('theta_')
        return PAR.get(name, name)
    compact = {'1cmt_oral': '1-comp oral', '2cmt_iv': '2-comp IV', '2cmt_oral': '2-comp oral'}
    s5 = table(['Model', 'Accepted / planned', 'Parameter', 'n', 'Bias (%)', 'RMSE (%)'],
               [[compact[s['model']], f"{s['converged']}/{s['planned']}", plabel(r['parameter']), r['n'], f(r['relative_bias_pct']),
                 f(r['relative_rmse_pct'])] for s in summary['secondary'] for r in s['estimates']])
    unresolved = unresolved_secondary()
    if unresolved:
        s5_note = ' '.join(f"{MODEL[u['model']]} replicate {u['replicate']} did not pass the convergence assessment "
                           f"({u['message']}); it is reported as unresolved." for u in unresolved)
    else:
        s5_note = 'All 30 additional-model fits passed the convergence assessment.'

    models = [['Theophylline', 'One-compartment first-order absorption; FOCE-I; proportional error; IIV on CL, V and Ka; CL and V per kg; 12 subjects, 120 observations',
               'Same model; CL and V scaled by WT/70'],
              ['Warfarin', 'One-compartment first-order absorption with lag; FOCE-I; combined error with fixed additive variance 0.117; IIV on CL, fixed ω²(Ka) = 0.5; fixed WT exponents 0.75 (CL) and 1 (V); 32 subjects, 251 observations',
               'Same model']]
    if INCLUDE_TOBRAMYCIN:
        models.append(['Tobramycin', 'Two-compartment IV bolus; FOCE-I; combined error with fixed additive variance 1.85 × 10⁻⁶; IIV on CL; power effects of CLCR on CL (reference 58 mL/min) and of WT on V1 (reference 62 kg); 97 subjects, 236 observations, 858 doses',
                       'Same model; fitted without constraints and with the constraints described in S10'])
    s6 = table(['Dataset', 'Expert NONMEM analysis [11]', 'PKPy2 analysis'], models)
    s8_rows = clinical_rows('Theophylline', theo) + clinical_rows('Warfarin', warf1)
    if INCLUDE_TOBRAMYCIN:
        s8_rows += clinical_rows('Tobramycin', read(TOB / 'expert_start_1.json'))
    s8 = table(['Dataset', 'Parameter', 'Estimate', 'SE', 'RSE (%)', '95% interval'], s8_rows)
    s9 = table(['Parameter', 'Estimate', 'RSE (%)', '95% interval'],
               [[CLIN_LABEL[c], f'{r["estimate"]:.5g}', f(r['rse_pct'], 1), f'{r["interval"][0]:.5g} to {r["interval"][1]:.5g}']
                for c, r in intervals(theo_comb).items()])
    structural = [k for k in warf1['theta'] if not (k == 'ALAG' and False)]
    warf_diff = max(abs(warf2['theta'][k] / warf1['theta'][k] - 1) * 100 for k in warf1['theta'])
    s10 = table(['Model', 'Doses', 'Observations', 'Direct (μs)', 'Recurrence (μs)', 'Ratio'],
                [[r['model'], r['doses'], r['observations'], f(r['direct_seconds'] * 1e6), f(r['recurrence_seconds'] * 1e6), f(r['speedup'])]
                 for r in speed['rows']])
    path_counts, flagged = paths()
    s11 = table(['Experiment', 'Condition', 'Path after Laplace exploration', 'Final status', 'n'],
                [[g_, c, r, s, n] for (g_, c, r, s), n in sorted(path_counts.items())])
    n_flagged = sum(flagged.values())

    s12 = table(['Sampling', 'Parameter', 'Comparator', 'Pairs', 'PKPy2 − comparator (pp)', 'Bootstrap 95% interval'],
                [[r['sampling'].title(), PAR[r['parameter']], LAB[r['comparator']], r['n_pairs'], f(r['rmse_difference_pct']),
                  f"{f(r['bootstrap_95_interval'][0])} to {f(r['bootstrap_95_interval'][1])}"]
                 for s in ['rich', 'sparse'] for r in cmp['paired_rmse'] if r['sampling'] == s])
    s13 = table(['Dataset', 'Parameter', 'NONMEM', 'PKPy2', 'nlmixr2 FOCEi', 'nlmixr2 SAEM', 'saemix'],
                [[r['dataset'].title(), r['label'], f(r['nonmem_rse'], 1), f(r['pkpy2_rse'], 1), f(r['nlmixr2_focei_rse'], 1),
                  f(r['nlmixr2_saem_rse'], 1), 'n/a' if r['dataset'] == 'warfarin' else f(r['saemix_rse'], 1)]
                 for r in cmp['clinical']['rows'] if r['dataset'] in datasets])
    s14 = table(['Dataset', 'Estimates from', 'OFV', 'ΔOFV vs PKPy2', 'Node-refinement change'],
                [[r['dataset'].title(), LAB[r['method']] + (' (σ_prop profiled)' if r.get('sigma_prop_profiled') else ''), f(r['ofv'], 3),
                  f(r['delta_ofv_vs_pkpy2'], 3), f'{r["node_refinement_gap"]:.1e}']
                 for r in sorted(check, key=lambda r: (r['dataset'], r['ofv'])) if r['dataset'] in datasets])
    theo_chk = next(r for r in check if r['dataset'] == 'theophylline' and r['method'] == 'pkpy2')
    warf_chk = next(r for r in check if r['dataset'] == 'warfarin' and r['method'] == 'pkpy2')
    tob_text = ' Tobramycin estimates in Table S8 are those obtained with the expert-judgment constraints (S10).' if INCLUDE_TOBRAMYCIN else ''
    tob_section = tobramycin_section(check) if INCLUDE_TOBRAMYCIN else ''
    from pkpy2_extended_supplement import markdown as extended_markdown
    extended = extended_markdown()

    return f'''# Online Resource 1

**Article:** {ARTICLE['title']}

**Journal:** {ARTICLE['journal']}

**Authors:** {ARTICLE['authors']}

**Corresponding author:** {ARTICLE['corresponding']}

Numerical settings, complete simulation summaries, clinical reference details, the comparison with nlmixr2 and saemix, the tobramycin analyses, and the numerical methods and evaluation of the event-record interface. Notation follows the main article, and reference numbers refer to its reference list.

## S1. Protocols and numerical thresholds

The primary experiment contains 200 datasets and the additional-model experiment 30 datasets; their seeds are recorded in the protocols, which were fixed before execution.

Each fit used at most two workers. Estimation began with a deterministic Laplace exploration: the Laplace objective (twice the joint negative log density at the individual modes plus the log determinant of the individual Hessians, normal constants included) was minimized with L-BFGS-B and central-difference gradients from the starting values and from up to seven scrambled-Sobol perturbations of the typical values and covariate coefficients, alternately within ±log 3 and ±log 10, within 120 CPU seconds. Declared bounds were respected throughout. Exploration stopped early when the first four starts reached the same optimum within 0.1 OFV units; optima differing by more than this were recorded as distinct. The best Laplace point was then refined with the importance-sampled marginal likelihood: initial refinement within 20 CPU seconds, SAEM when required within 60 CPU seconds, and marginal refinement within 300 CPU seconds and at most 90 stages. Final acceptance required maximum absolute projected score ≤0.2 in both independent banks, their OFV difference ≤0.05, and every subject ESS ≥100. Reaching a budget limit was recorded as a separate termination reason.

The integration proposal was an equally weighted mixture of the population Gaussian and a subject-specific Gaussian, with importance weights computed using the full mixture density. Pilot moments used at least 2¹⁴ samples per subject, increasing to the subject's current integration power when higher. Pilot ESS below 50 or maximum normalized weight above 0.1 triggered verification of a conditional mode and its curvature; the proposal covariance was 2H_joint⁻¹ + 0.01Ω, where H_joint is the Hessian of the conditional negative log density. Marginal refinement began with 2¹² samples per subject, used L-BFGS stages of at most 40 iterations, and increased integration power for subjects showing integration disagreement, up to power 16. A failed stationarity audit increased all subjects to power 16. Final acceptance used two independent banks, each with 2¹⁶ = 65,536 samples per subject, so integration accuracy was assessed with samples independent of those used for optimization.

When required, SAEM used four chains and at most 400 exploration and 400 smoothing iterations within its CPU budget. The stochastic approximation weight was one during exploration and (m + 2)^(-0.7) at smoothing index m beginning at zero. Phase decisions were assessed every 25 iterations with a stability tolerance of 0.01. Eligible log-parameter means used regression updates and diagonal variances used conditional moments; the remaining coordinates, including residual terms, used a generalized L-BFGS-B M-step with at most six iterations, accepted when it improved the complete-data objective. SAEM was always followed by marginal refinement and the independent convergence assessment. The estimation paths are reported in Table S11.

Intervals used two independent banks of power 16, finite-difference steps 0.002 and 0.001, and the same saved estimate. Acceptance required positive information, relative Hessian changes across banks and steps ≤0.1, relative coordinate SE change ≤0.2, difference from the saved objective ≤0.05, ESS ≥100, and relative Hessian asymmetry ≤0.1. RSEs above 50% were flagged as large. Numerical floors were 1e-10 for prediction, 1e-12 for residual variance, and 1e-8 for interindividual variance.

The primary true values were CL = 4 L/h, V = 40 L, ω²(CL) = 0.09, ω²(V) = 0.04, and σ_prop = 0.15. Rich times were 0.25, 0.5, 1, 2, 4, 6, 8, 12, 18, and 24 h; sparse times were 1, 8, and 24 h. Each subject received 100 mg. Structural starting values were 80% of truth, initial estimated interindividual variances were 0.08, and initial proportional SD was 0.2. The Gaussian two-stage controls used the known generating residual SD.

## S2. Complete primary simulation summaries

**Table S1. Returned and numerically accepted population estimates.**

{s1}

All, every returned estimate, including unsuccessful optimizer states; Accepted, estimates meeting the method-specific numerical criteria. The Gaussian two-stage control (TST) used the true residual SD.

**Table S2. PKPy2 interval coverage and width.**

{s2}

Width is divided by the true parameter value.

**Table S3. Paired-bootstrap differences in relative RMSE.**

{s3}

PKPy2 minus comparator in percentage points (pp); negative values favor PKPy2. Intervals are from 10,000 paired bootstrap resamples of datasets accepted by both methods (seed 261001) and are unadjusted for multiplicity.

## S3. Additional model structures

**Table S4. Additional-model generating values and starting values.**

{s4}

Each structure used ten independently generated datasets with 30 subjects, a single 100 mg dose per subject, and samples at 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 18, 24, 36, and 48 h. Clearance and intercompartmental clearance are in L/h, volumes in L, and Ka in h⁻¹. IIV entries are diagonal log-parameter variances. Generating proportional SD was 0.15, starting proportional SD was 0.2, and every estimated IIV variance started at 0.08. Additive SD and oral lag were fixed at zero, with no covariates and F = 1. Structural starts were 80% of truth.

**Table S5. Additional-model parameter recovery among accepted fits.**

{s5}

{s5_note}

## S4. Clinical datasets and expert reference analyses

The expert NONMEM analyses are from the PKGPT study [11]; their published estimates and RSEs are used as reference values.

**Table S6. Models of the expert analyses and of PKPy2.**

{s6}

The expert theophylline model estimates CL and V per kg using mg/kg doses. For identical η values, multiplying CL and V by individual WT while multiplying dose by the same WT preserves concentrations exactly; PKPy2 implements this with CL70 and V70 and fixed (WT/70)¹ factors. The warfarin data retained all four zero observations. Initial values for CL, V, Ka and ALAG were (0.2, 10, 0.8, 0.5) for the primary warfarin fit and (0.1, 6, 2, 1) for a sensitivity fit; estimated CL variance started at 0.1 and proportional SD at 0.2. The first start was designated as primary before fitting.

**Table S7. Estimation and uncertainty in the expert analyses and in PKPy2.**

{table(['Component', 'Expert NONMEM [11]', 'PKPy2'], [['Estimation', 'FOCE-I', 'Importance-integrated marginal likelihood'], ['Uncertainty', 'NONMEM covariance step (RSEs as published)', 'Inverse observed marginal information, 2H_OFV⁻¹'], ['Reported quantities', 'Estimates; RSEs where published', 'Estimates, SEs, RSEs and 95% intervals (Table S8)']])}

**Table S8. PKPy2 clinical standard errors and 95% intervals.**

{s8}

Intervals are log-coordinate Wald intervals transformed to the reported scale. Interindividual effects are reported as variances and residual components as SDs.

**Table S9. Theophylline combined-error sensitivity analysis.**

{s9}

The combined-error model is a sensitivity analysis. Warfarin fits took {warf1['seconds']:.2f} s and {warf2['seconds']:.2f} s from the two starts, and the primary uncertainty assessment took {warf1['uncertainty']['seconds']:.2f} s; estimates from the two starts differed by at most {warf_diff:.3f}% and their OFVs by {abs(warf1['ofv'] - warf2['ofv']):.6f}.{tob_text}

## S5. Independent numerical calculations and prediction timing

The one-random-effect quadrature comparisons optimized with 4,097 nodes and were checked with 8,193 nodes. The independent code has its own prediction, likelihood, and finite-difference Hessian calculations and includes the Gaussian density constants.

**Table S10. Warm prediction-call microbenchmark.**

{s10}

Medians of seven alternating-order batches after compilation. With a single dose, recurrence was slightly slower than direct summation.

## S6. Computational environment and estimation paths

PKPy was run with its numerical model, objective, and optimizer unchanged; an unused plotting import was stubbed because its optional dependency was absent. The comparison covers PKPy's individual fitting and population summaries.

The calculation host was Windows 11, AMD Ryzen 5 5600 (6 cores, 12 threads), Python 3.13.5, NumPy 2.2.6, SciPy 1.16.1, Numba 0.65.1, and threadpoolctl 3.6.0. Each saved result records data fingerprints, numerical settings, and seeds, and can be restored only with matching input data.

**Table S11. Estimation paths in the simulation experiments.**

{s11}

Counts refer to the 200 primary and 30 additional datasets. The Laplace exploration reached more than one distinct optimum in {n_flagged} of these 230 fits.

## S7. Executable use of the declared-model interface

The accompanying examples/pkpy2_explicit_model.py creates synthetic observations for 30 subjects. After installing PKPy2 with Python 3.13, run the example with an unused output directory. It declares fixed volume and a fixed weight exponent for illustration, estimates clearance, clearance variability, and proportional residual SD, inspects the fit and uncertainty states, and restores the saved result with the same data. The model and result-handling core is shown in Listing 1; data generation and assertions are included in the complete executable file.

**Listing 1. Model declaration and numerical result handling (excerpt).**

```python
from pkpy2 import ModelSpec, Parameter as P, Covariate
from pkpy2 import fit, load_fit

specification = ModelSpec(
    '1cmt_iv',
    theta={{'CL': P(3.), 'V': P(40., fixed=True)}},
    omega={{'CL': P(.08)}},
    sigma_prop=P(.2), sigma_add=P(0., fixed=True),
    covariates=(Covariate('CL', 'WT', 70.,
                          P(.75, fixed=True)),),
)
result = fit(subjects, specification, seed=20260915, workers=2,
    saem_options={{'cpu_budget_seconds': 60.}},
    refinement_options={{'cpu_budget_seconds': 300.,
                        'max_stages': 90, 'analytic_non_eta': True}})
print(result.status, result.audit.get('passed'))
if result.converged:
    report = result.uncertainty(seed=20260916, workers=2,
                                power=16, step=.002)
    print(report['status'])
result.save(destination)
restored = load_fit(destination, subjects)
```

Here, subjects holds the generated Subject records and destination is a new JSON path. The complete example checks that fixed terms are preserved and that estimates, diagnostics, and intervals are restored.

## S8. Comparison with nlmixr2 and saemix

This section describes the comparison with nlmixr2 and saemix (Tables 5 and 6, Figs. 4 and 5).

**Data.** The observation records analysed by PKPy2 were exported unchanged to NONMEM-style event files (ID, TIME, AMT, DV, EVID, MDV, CMT, and covariates) for the 200 primary simulation datasets and the clinical datasets. Oral doses were placed in the depot compartment. A manifest of SHA-256 hashes links each exported file to its PKPy2 input.

**Models and starting values.** All programs used the models, fixed terms, and starting values of the PKPy2 analyses (S1 and S4). The residual variance was σ_prop² f² + σ_add² in every program (nlmixr2 addProp = "combined2").

**Program settings.** nlmixr2 {env["nlmixr2"]} (nlmixr2est {env["nlmixr2est"]}, rxode2 {env["rxode2"]}) was run with FOCEi and SAEM using default estimation and covariance settings (FOCEi: sandwich "r,s"; SAEM: stochastic approximation "sa", seed {env["seed"]}). The simulation and theophylline models used analytical linear-compartment solutions, and the warfarin model used ordinary differential equations for the absorption lag. saemix {env["saemix"]} was run with default iterations, seed {env["seed"]}, and standard errors from the linearized Fisher information; it was not applied to warfarin, whose model fixes nonzero variance components.{" For tobramycin, nlmixr2 used the expert-judgment constraints of S10; nlmixr2 SAEM does not apply parameter bounds, and saemix, which does not support bounds, was not applied. Tobramycin fits without constraints are reported in S10." if INCLUDE_TOBRAMYCIN else ""} All fits used {env["R"]} on the computer described in S6, with one rxode2 thread per fit and five concurrent processes. All {len(records)} external fits completed without errors.

**Metrics.** Relative bias, relative RMSE, Monte Carlo standard errors, Wilson intervals, and paired-bootstrap differences (10,000 resamples of datasets) were computed as in the primary analysis. Wald 95% intervals for CL and V were formed on the log scale from the reported standard errors (for saemix, SE divided by the estimate).

**Exact marginal likelihood at the clinical estimates.** Because the programs report objective functions under different approximations and constants, each set of clinical estimates was evaluated with the exact marginal OFV (normal constants included), computed by adaptive Gauss-Hermite quadrature in a separate Python implementation that shares no code with PKPy2. For each subject, the conditional mode was found by BFGS, the Hessian by central finite differences, and a product Gauss-Hermite rule was centred and scaled by the mode and the Cholesky factor of the inverse Hessian. For the published NONMEM estimates, which do not include σ_prop, σ_prop was set to its maximum-likelihood value. At the PKPy2 estimates, this calculation reproduced the OFV reported by PKPy2 ({f(theo_chk["ofv"], 3)} vs {f(theo_chk["pkpy2_reported_ofv"], 3)} for theophylline; {f(warf_chk["ofv"], 3)} vs {f(warf_chk["pkpy2_reported_ofv"], 3)} for warfarin).

**Table S12. Paired-bootstrap differences in relative RMSE between PKPy2 and nlmixr2 or saemix.**

{s12}

Negative values favor PKPy2. pp, percentage points.

**Table S13. Relative standard errors (%) of the clinical estimates.**

{s13}

NR, not reported; n/a, not applied. nlmixr2 RSEs of structural parameters are SEs of log-scale parameters.

**Table S14. Exact marginal OFV at the clinical estimates of each program.**

{s14}

Lower OFV indicates higher marginal likelihood.

## S9. Raw data files and codebook

The raw-data workbook (Online Resource 2) contains a README sheet and 18 data sheets, also provided as CSV files (Online Resource 3). S1 and S3 hold the simulated concentration-time records of the 200 primary and 30 additional-structure datasets with the true individual random effects. S2 and S4 hold per-dataset estimates, 95% intervals, coverage indicators, and fit times for every program. S5 and S6 hold the clinical analysis datasets and the clinical estimates of all programs. S7 to S9 hold the prediction checks, quadrature comparisons, and per-batch prediction timings. S10 to S18 hold the evaluation of the event-record interface (S12), including the simulated comparison datasets and the warfarin PK/PD dataset in NONMEM event format. The codebook (Online Resource 4) defines every variable, unit, and categorical code, including the numerically coded EVID, MDV, CMT, and 0/1 indicators.
{tob_section}{extended}'''


def caption_key(text, parent):
    """(key, parent): the key of a table caption ('Table S21') or sub-table caption ('Table S21(b)', following
    'Table S21'), otherwise None; `parent` is the last table caption seen, updated for the next call."""
    m = re.match(r'(Table S\d+|Listing \d+)\.|\(([a-z])\) ', text)
    if not m:
        return None, parent
    if m.group(1):
        return m.group(1), m.group(1)
    return f'{parent}({m.group(2)})', parent


def md_to_html(md, breaks=()):
    """`breaks`: captions (keys of caption_key) that start a new page."""
    def inline(t):
        t = html.escape(t)
        t = re.sub(r'\*\*(.+?)\*\*', lambda m: '<b>' + m.group(1) + '</b>', t)
        for token, sub in [('σ_prop', 'σ<sub>prop</sub>'), ('σ_add', 'σ<sub>add</sub>'), ('σ_log', 'σ<sub>log</sub>'),
                           ('H_joint', 'H<sub>joint</sub>'),
                           ('H_OFV', 'H<sub>OFV</sub>')]:
            t = t.replace(token, sub)
        t = re.sub(r'\^\(([^()]*)\)', lambda m: '<sup>' + m.group(1).replace('-', '−') + '</sup>', t)
        t = re.sub(r'\^(θ[₀-₉]+)', r'<sup>\1</sup>', t)
        return t
    out, lines, i, parent = [], md.splitlines(), 0, None
    while i < len(lines):
        line = lines[i].rstrip()
        if not line.strip():
            i += 1
            continue
        if line.startswith('```'):
            code = []
            i += 1
            while not lines[i].startswith('```'):
                code.append(html.escape(lines[i]))
                i += 1
            out.append('<pre>' + '\n'.join(code) + '</pre>')
            i += 1
            continue
        if line.startswith('# '):
            out.append(f'<h1>{inline(line[2:])}</h1>')
        elif line.startswith('## '):
            out.append(f'<h2>{inline(line[3:])}</h2>')
        elif line.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                cells = [c.strip() for c in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?', c) for c in cells):
                    rows.append(cells)
                i += 1
            head = ''.join(f'<th>{inline(c)}</th>' for c in rows[0])
            body = ''.join('<tr>' + ''.join(f'<td>{inline(c)}</td>' for c in r) + '</tr>' for r in rows[1:])
            out.append(f'<table><tr>{head}</tr>{body}</table>')
            continue
        else:
            cls = ' class="cap"' if line.startswith(('**Table', '**Listing')) else ''
            key, parent = caption_key(line.strip().removeprefix('**'), parent)
            if key in breaks:
                cls += ' style="page-break-before: always"'
            out.append(f'<p{cls}>{inline(line.strip())}</p>')
        i += 1
    css = ('* {font-family: serif;} body {font-size: 10.2pt; line-height: 1.4;} '
           'h1 {font-size: 16pt; font-weight: bold; margin: 0 0 10pt 0;} '
           'h2 {font-size: 12pt; font-weight: bold; margin: 12pt 0 6pt 0;} '
           'p {margin: 0 0 7pt 0;} p.cap {font-size: 9pt; margin-top: 4pt;} '
           'table {border-collapse: collapse; width: 100%; margin-bottom: 8pt;} '
           'th, td {font-size: 8.1pt; border-top: 0.5pt solid black; border-bottom: 0.5pt solid black; padding: 3pt 4pt; '
           'text-align: left; vertical-align: top;} th {font-weight: bold;} '
           'pre {font-family: monospace; font-size: 8pt; line-height: 1.3; margin: 0 0 8pt 0;}')
    return css, '<body>' + ''.join(out) + '</body>'


def render(md, path):
    """A caption in the last 80 pt of a page's text area moves to the next page with its table.

    Breaks are added one at a time in document order: a break moves only the content after it,
    so earlier decisions stay valid while later captions are re-examined in the new layout."""
    mediabox = pymupdf.paper_rect('a4')
    where = mediabox + (54, 48, -54, -59)
    breaks = set()
    while True:
        css, body = md_to_html(md, breaks)
        story = pymupdf.Story(html=body, user_css=css)
        buffer = io.BytesIO()
        writer = pymupdf.DocumentWriter(buffer)
        more = True
        while more:
            dev = writer.begin_page(mediabox)
            more, _ = story.place(where)
            story.draw(dev)
            writer.end_page()
        writer.close()
        doc = pymupdf.open('pdf', buffer.getvalue())
        late, parent = [], None
        for page in doc:
            for b in page.get_text('blocks', sort=True):
                key, parent = caption_key(b[4], parent)
                if key is not None and b[1] > where.y1 - 80 and key not in breaks:
                    late.append(key)
        if not late:
            break
        doc.close()
        breaks.add(late[0])
    gray = (0x56 / 255, 0x60 / 255, 0x6A / 255)
    for n, page in enumerate(doc, start=1):
        h, w = page.rect.height, page.rect.width
        page.insert_text((54, h - 33), 'PKPy2 | Online Resource 1', fontname='tiro', fontsize=8, color=gray)
        page.insert_text((w - 54 - pymupdf.get_text_length(str(n), fontname='tiro', fontsize=8), h - 33), str(n),
                         fontname='tiro', fontsize=8, color=gray)
    doc.save(str(path), garbage=3, deflate=True)
    doc.close()


def main():
    md = markdown()
    manuscript = ROOT / 'output/pkpy2_peerj_build/PKPy2_PeerJ_manuscript_revised.docx'
    if '--springer-references' in sys.argv:
        # reference numbers of the journal version (order of first citation in the revised manuscript)
        from pkpy2_references import manuscript_mapping, renumber, CITATION
        mapping = manuscript_mapping(manuscript)
        print('citations renumbered:', sorted({m.group(0) for m in CITATION.finditer(md)}))
        md = renumber(md, mapping)
    (PAPER / 'PKPy2_supplement_peerj_revision.md').write_text(md, encoding='utf-8')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    render(md, OUT)
    print('pages', pymupdf.open(OUT).page_count, OUT)


if __name__ == '__main__':
    main()
