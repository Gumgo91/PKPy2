"""Compute every result number quoted in the PKPy2 manuscript from the current (v3) records.

Prints `name: value` lines and writes docs/pkpy2_paper/manuscript_numbers.json, which the
manuscript revision script uses to update the quoted values.
"""
from pathlib import Path
import json
import math

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'docs/pkpy2_paper'
DEV = ROOT / 'output/pkpy2_development'
CMP = ROOT / 'output/pkpy2_software_comparison'


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


TOB = ROOT / 'output/pkpy2_tobramycin_v3'
TOB_EXPERT = dict(CL=2.95, V1=4.59, Q=6.85, V2=13.2, b_clcr=0.236, b_wt=1.07, omega_CL=0.028)


def tobramycin(check):
    """Numbers of the tobramycin application (unconstrained, expert bounds, expert judgment)."""
    import csv
    import numpy as np
    n = {}
    rows = list(csv.DictReader((CMP / 'source/tobramycin_github.csv').read_text(encoding='utf-8').lstrip('#').splitlines()))
    ids = list(dict.fromkeys(r['ID'] for r in rows))
    tad, wt, clcr = [], [], []
    for i in ids:
        rr = [r for r in rows if r['ID'] == i]
        doses = [float(r['TIME']) for r in rr if r['EVID'] == '1']
        tad += [float(r['TIME']) - max(d for d in doses if d <= float(r['TIME'])) for r in rr if r['EVID'] == '0']
        wt.append(float(rr[0]['WT'])); clcr.append(float(rr[0]['CLCR']))
    tad = np.array(tad)
    n['tob_subjects'] = len(ids); n['tob_obs'] = len(tad); n['tob_doses'] = sum(r['EVID'] == '1' for r in rows)
    n['tob_obs_at_2p5'] = int(np.sum(np.isclose(tad, 2.5)))
    n['tob_wt_range'] = [f'{min(wt):.0f}', f'{max(wt):.0f}']
    n['tob_corr_wt_clcr'] = f'{np.corrcoef(np.log(wt), np.log(clcr))[0, 1]:.2f}'

    def fitted(name):
        f = read(TOB / f'{name}.json')
        b = f['coefficients']
        return dict(ofv=f['ofv'], status=f['status'], CL=f['theta']['CL'], V1=f['theta']['V1'], Q=f['theta']['Q'], V2=f['theta']['V2'],
                    b_clcr=b[0], b_wt=b[1] if len(b) > 1 else 1.0, omega_CL=f['omega']['CL'],
                    optima=len(((f.get('estimation') or {}).get('laplace_exploration') or {}).get('distinct_optima') or []),
                    unc=(f.get('uncertainty') or {}).get('status'))
    fits = {k: fitted(k) for k in ['start_1', 'start_2', 'bounded_start_1', 'bounded_start_2', 'judgment_start_1', 'judgment_start_2',
                                   'expert_start_1', 'expert_start_2']}
    n['tob_fits'] = fits
    u = fits['start_1']
    n['tob_unc_ofv'] = f"{u['ofv']:.2f}"
    n['tob_unc_start_gap'] = f"{abs(u['ofv'] - fits['start_2']['ofv']):.3f}"
    n['tob_unc_bwt'] = f"{u['b_wt']:.1f}"; n['tob_unc_V2'] = f"{u['V2']:.0f}"; n['tob_unc_V1'] = f"{u['V1']:.1f}"
    n['tob_unc_optima'] = u['optima']
    ofv = {(r['dataset'], r['method']): r for r in check}
    n['tob_expert_ofv'] = f"{ofv[('tobramycin', 'nonmem')]['ofv']:.2f}"
    n['tob_expert_minus_unc'] = f"{ofv[('tobramycin', 'nonmem')]['ofv'] - u['ofv']:.1f}"
    b1, b2 = fits['bounded_start_1'], fits['bounded_start_2']
    n['tob_bounded_1'] = dict(ofv=f"{b1['ofv']:.2f}", V2=f"{b1['V2']:.0f}", b_wt=f"{b1['b_wt']:.2f}", V1=f"{b1['V1']:.1f}")
    n['tob_bounded_2'] = dict(ofv=f"{b2['ofv']:.2f}", max_diff=f"{max(abs(b2[k] / TOB_EXPERT[k] - 1) for k in TOB_EXPERT) * 100:.1f}")
    n['tob_bounded_gap'] = f"{b2['ofv'] - b1['ofv']:.1f}"
    j1 = fits['judgment_start_1']
    n['tob_judgment_1'] = dict(ofv=f"{j1['ofv']:.2f}", V1=f"{j1['V1']:.1f}", V2=f"{j1['V2']:.0f}", Q=f"{j1['Q']:.2f}")
    e1, e2 = fits['expert_start_1'], fits['expert_start_2']
    keys = ['CL', 'V1', 'Q', 'V2', 'b_clcr', 'omega_CL']
    n['tob_ej_diffs'] = {k: f"{100 * (e1[k] / TOB_EXPERT[k] - 1):.1f}" for k in keys}
    n['tob_ej_max'] = f"{max(abs(e[k] / TOB_EXPERT[k] - 1) for e in (e1, e2) for k in keys) * 100:.1f}"
    n['tob_ej_start_gap'] = f"{abs(e1['ofv'] - e2['ofv']):.3f}"
    n['tob_ej_ofv'] = f"{e1['ofv']:.2f}"
    n['tob_ej_minus_unc'] = f"{e1['ofv'] - u['ofv']:.1f}"
    n['tob_ej_any_on_bound'] = any(abs(e1[k] - hi) / hi < 1e-3 for k, hi in [('V1', 10.), ('V2', 30.), ('Q', 50.), ('CL', 20.)])
    cmp_rows = [r for r in read(CMP / 'comparison_summary.json')['clinical']['rows'] if r['dataset'] == 'tobramycin' and r['nonmem'] is not None]
    for m in ['nlmixr2_focei', 'nlmixr2_saem']:
        vals = [abs(r[m + '_diff_pct']) for r in cmp_rows if r[m + '_diff_pct'] is not None]
        n[f'tob_{m}_max'] = f'{max(vals):.1f}' if vals else None
    n['tob_saem_diffs'] = {r['label'].split(' (')[0]: f"{r['nlmixr2_saem_diff_pct']:.0f}" for r in cmp_rows}
    n['tob_ofv_focei_minus_pkpy2'] = f"{ofv[('tobramycin', 'nlmixr2_focei')]['ofv'] - ofv[('tobramycin', 'pkpy2')]['ofv']:.2f}"
    n['tob_ofv_saem_minus_pkpy2'] = f"{ofv[('tobramycin', 'nlmixr2_saem')]['ofv'] - ofv[('tobramycin', 'pkpy2')]['ofv']:.1f}"
    n['tob_ofv_nonmem_minus_pkpy2'] = f"{ofv[('tobramycin', 'nonmem')]['ofv'] - ofv[('tobramycin', 'pkpy2')]['ofv']:.2f}"
    prof = read(TOB / 'profile_v2.json')
    pv = [r['ofv'] for r in prof['profile']]
    n['tob_profile_range'] = f'{max(pv) - min(pv):.1f}'
    n['tob_profile_v2'] = [f"{prof['profile'][0]['V2']:g}", f"{prof['profile'][-1]['V2']:g}"]
    n['tob_profile_check'] = [f"{prof['check']['independent']:.3f}", f"{prof['check']['pkpy2_reported']:.3f}"]
    fc = read(TOB / 'focei_check.json')
    pts = {p['pkpy2_fit']: p['focei_objective'] for p in fc['points']}
    n['tob_focei_at_bounded_1'] = f"{pts['bounded_start_1']:.1f}"
    n['tob_focei_expert_published'] = f"{fc['published_expert_nonmem_ofv']:.1f}"
    return n


def main():
    s = read(PAPER / 'analysis_summary.json')
    cmp = read(CMP / 'comparison_summary.json')
    check = read(CMP / 'clinical_likelihood_check.json')
    ref = read(DEV / 'reference_v3/results.json')
    num = read(DEV / 'numerical_validation.json')
    speed = read(DEV / 'recurrence_benchmark.json')
    theo = read(DEV / 'theophylline_v3/proportional.json')
    n = {}

    def est(sampling, engine, param, key):
        return next(r for r in s['primary_estimation'] if r['sampling'] == sampling and r['engine'] == engine
                    and r['parameter'] == param and r['population'] == 'numerically_accepted')[key]
    for eng in ['pkpy', 'pkpy2', 'gaussian_tst']:
        for p in ['theta_CL', 'theta_V', 'omega_CL', 'omega_V']:
            n[f'sparse_rmse_{eng}_{p}'] = f"{est('sparse', eng, p, 'relative_rmse_pct'):.2f}"
            n[f'sparse_bias_{eng}_{p}'] = f"{est('sparse', eng, p, 'relative_bias_pct'):.2f}"
    for comp in ['pkpy', 'gaussian_tst']:
        r = next(r for r in s['paired_rmse'] if r['sampling'] == 'sparse' and r['parameter'] == 'omega_V' and r['comparator'] == comp)
        n[f'paired_{comp}_omegaV'] = f"{r['rmse_difference_pct']:.2f}"
        n[f'paired_{comp}_omegaV_lo'] = f"{r['bootstrap_95_interval'][0]:.2f}"
        n[f'paired_{comp}_omegaV_hi'] = f"{r['bootstrap_95_interval'][1]:.2f}"
    status = {r['sampling']: r for r in s['primary_status']}
    n['converged_rich'] = status['rich']['converged']
    n['converged_sparse'] = status['sparse']['converged']
    cov = {(r['sampling'], r['parameter']): round(100 * r['conditional_coverage']['rate']) for r in s['primary_intervals']}
    n['coverage_min'] = min(cov.values()); n['coverage_max'] = max(cov.values())
    for p in ['theta_CL', 'theta_V', 'omega_CL', 'omega_V', 'sigma_prop']:
        n[f'coverage_sparse_{p}'] = cov[('sparse', p)]
    sec = {x['model']: x for x in s['secondary']}
    for m, x in sec.items():
        n[f'secondary_converged_{m}'] = x['converged']
        for e in x['estimates']:
            n[f'secondary_rmse_{m}_{e["parameter"]}'] = f"{e['relative_rmse_pct']:.2f}"
    n['secondary_total_converged'] = sum(x['converged'] for x in sec.values())
    n['sec_1cmt_oral_struct_max'] = f"{max(e['relative_rmse_pct'] for e in sec['1cmt_oral']['estimates'] if e['parameter'].startswith('theta_')):.2f}"
    n['sec_2cmt_iv_struct_max'] = f"{max(e['relative_rmse_pct'] for e in sec['2cmt_iv']['estimates'] if e['parameter'].startswith('theta_')):.2f}"
    n['pred_max'] = f"{max(r['prediction_relative_scaled_error'] for r in num['independent_numerical_cases']):.2e}"
    n['sens_max'] = f"{max(r['sensitivity_relative_scaled_error'] for r in num['independent_numerical_cases']):.2e}"
    n['quad_same_point_ofv'] = f"{max(c['engine_ofv_gap'] for c in ref['cases']):.2e}"
    n['quad_coord_gap'] = f"{max(c['maximum_coordinate_gap'] for c in ref['cases']):.2e}"
    n['quad_se_gap_pct'] = f"{100 * max(c['maximum_relative_coordinate_se_gap'] for c in ref['cases']):.4f}"
    n['theo_CL'] = f"{theo['theta']['CL']:.4f}"; n['theo_V'] = f"{theo['theta']['V']:.4f}"; n['theo_Ka'] = f"{theo['theta']['Ka']:.4f}"
    rows = cmp['clinical']['rows']
    theo_rows = [r for r in rows if r['dataset'] == 'theophylline' and r['nonmem'] is not None]
    n['theo_max_diff'] = f"{max(abs(r['pkpy2_diff_pct']) for r in theo_rows):.2f}"
    n['theo_rse_omegaV_pkpy2'] = f"{next(r['pkpy2_rse'] for r in theo_rows if r['label'] == 'ω²(V)'):.1f}"
    warf_rows = [r for r in rows if r['dataset'] == 'warfarin' and r['nonmem'] is not None]
    n['warf_max_diff'] = f"{max(abs(r['pkpy2_diff_pct']) for r in warf_rows):.2f}"
    n['warf_max_param'] = max(warf_rows, key=lambda r: abs(r['pkpy2_diff_pct']))['label']
    tools = ['pkpy2', 'nlmixr2_focei', 'nlmixr2_saem', 'saemix']
    n['theo_tools_max_diff'] = f"{max(abs(r[m + '_diff_pct']) for r in theo_rows for m in tools if r[m + '_diff_pct'] is not None):.1f}"
    n['theo_tools_n'] = sum(1 for r in theo_rows for m in tools if r[m + '_diff_pct'] is not None)
    ofv = {(r['dataset'], r['method']): r['ofv'] for r in check}
    theo_ofvs = [v for (d, m), v in ofv.items() if d == 'theophylline']
    n['theo_ofv_spread'] = f"{max(theo_ofvs) - min(theo_ofvs):.2f}"
    n['theo_ofv_pkpy2'] = f"{ofv[('theophylline', 'pkpy2')]:.2f}"
    n['theo_ofv_nonmem'] = f"{ofv[('theophylline', 'nonmem')]:.2f}"
    n['warf_focei_max_diff'] = f"{max(abs(r['nlmixr2_focei_diff_pct']) for r in warf_rows):.1f}"
    n['warf_ofv_focei_minus_pkpy2'] = f"{ofv[('warfarin', 'nlmixr2_focei')] - ofv[('warfarin', 'pkpy2')]:.2f}"
    n['warf_ofv_saem_minus_pkpy2'] = f"{ofv[('warfarin', 'nlmixr2_saem')] - ofv[('warfarin', 'pkpy2')]:.1f}"
    n['warf_ofv_nonmem_minus_pkpy2'] = f"{ofv[('warfarin', 'nonmem')] - ofv[('warfarin', 'pkpy2')]:.2f}"
    n['theo_pkpy2_lowest'] = min(theo_ofvs) == ofv[('theophylline', 'pkpy2')]
    n['warf_pkpy2_lowest'] = min(v for (d, m), v in ofv.items() if d == 'warfarin') == ofv[('warfarin', 'pkpy2')]
    other = [v - ofv[(d, 'pkpy2')] for (d, m), v in ofv.items() if m not in ('pkpy2', 'nlmixr2_saem') and d in ('theophylline', 'warfarin')]
    n['others_within'] = f"{max(other):.2f}"
    est_c = {(r['sampling'], r['method'], r['parameter']): r for r in cmp['estimation']}
    stat_c = {(r['sampling'], r['method']): r for r in cmp['status']}
    cov_c = {(r['sampling'], r['method'], r['parameter']): r for r in cmp['coverage']}
    rich_diffs = [abs(est_c[('rich', m, p)]['relative_rmse_pct'] - est_c[('rich', 'pkpy2', p)]['relative_rmse_pct'])
                  for m in tools[1:] for p in ['theta_CL', 'theta_V', 'omega_CL', 'omega_V', 'sigma_prop']]
    n['rich_max_rmse_gap'] = f"{max(rich_diffs):.2f}"
    struct_diffs = [abs(est_c[(sm, m, p)]['relative_rmse_pct'] - est_c[(sm, 'pkpy2', p)]['relative_rmse_pct'])
                    for sm in ['rich', 'sparse'] for m in tools[1:] for p in ['theta_CL', 'theta_V']]
    n['struct_max_rmse_gap'] = f"{max(struct_diffs):.2f}"
    n['excluding_zero'] = [dict(sampling=r['sampling'], comparator=r['comparator'], parameter=r['parameter'],
                                diff=round(r['rmse_difference_pct'], 2), lo=round(r['bootstrap_95_interval'][0], 2),
                                hi=round(r['bootstrap_95_interval'][1], 2))
                           for r in cmp['paired_rmse'] if r['bootstrap_95_interval'][0] > 0 or r['bootstrap_95_interval'][1] < 0]
    n['sparse_bias_CL_pkpy2_cmp'] = f"{est_c[('sparse', 'pkpy2', 'theta_CL')]['relative_bias_pct']:.2f}"
    n['sparse_bias_CL_focei'] = f"{est_c[('sparse', 'nlmixr2_focei', 'theta_CL')]['relative_bias_pct']:.2f}"
    saem = {r['label'].split(' (')[0]: r['nlmixr2_saem_diff_pct'] for r in warf_rows}
    assert saem['Ka'] < 0 and saem['ALAG'] < 0
    n['warf_saem_ka_diff'] = f"{-saem['Ka']:.1f}"; n['warf_saem_alag_diff'] = f"{-saem['ALAG']:.1f}"
    n['sparse_rmse_omegaV_tools'] = {m: f"{est_c[('sparse', m, 'omega_V')]['relative_rmse_pct']:.2f}" for m in tools}
    covs = {m: [round(100 * cov_c[(sm, m, p)]['rate']) for sm in ['rich', 'sparse'] for p in ['theta_CL', 'theta_V']] for m in tools}
    n['coverage_tools'] = {m: [min(v), max(v)] for m, v in covs.items()}
    n['median_fit_seconds'] = {f'{sm}_{m}': round(stat_c[(sm, m)]['median_fit_seconds'], 1) for sm in ['rich', 'sparse'] for m in tools}
    n['median_uncertainty_seconds'] = {sm: round(stat_c[(sm, 'pkpy2')]['median_uncertainty_seconds'], 1) for sm in ['rich', 'sparse']}
    large = [r['speedup'] for r in speed['rows'] if r['doses'] == 1000]
    n['speed_1000'] = [f'{min(large):.0f}', f'{max(large):.0f}']
    # Estimation paths of the simulation fits.
    for group, folder in [('primary', 'confirmatory_v3'), ('secondary', 'secondary_v3')]:
        fits = [read(p)['pkpy2'] for p in sorted((DEV / folder).glob('*__[0-9][0-9][0-9].json'))]
        n[f'{group}_no_saem'] = sum((f.get('estimation') or {}).get('saem_status') == 'not_needed' for f in fits)
        n[f'{group}_multimodal'] = sum(len(((f.get('estimation') or {}).get('laplace_exploration') or {}).get('distinct_optima') or []) > 1
                                       for f in fits)
    n['theo_rse_omegaV_nonmem'] = f"{next(r['nonmem_rse'] for r in theo_rows if r['label'] == 'ω²(V)'):.1f}"
    rse_gap = sorted((abs(r['pkpy2_rse'] - r['nonmem_rse']), r['label']) for r in warf_rows
                     if r['pkpy2_rse'] is not None and r['nonmem_rse'] is not None)
    n['warf_rse_largest'] = [lab for _, lab in rse_gap[::-1][:2]]
    n.update(tobramycin(check))
    (PAPER / 'manuscript_numbers.json').write_text(json.dumps(n, indent=1, ensure_ascii=False), encoding='utf-8')
    for k, v in n.items():
        print(f'{k}: {v}')


if __name__ == '__main__':
    main()
