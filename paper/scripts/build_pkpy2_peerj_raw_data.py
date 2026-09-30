"""Assemble the PeerJ raw-data workbook, its codebook, and CSV copies of every sheet.

Every value is read from the saved study records; nothing is recomputed except
relative errors and interval-coverage indicators, whose definitions are given in the
codebook. Categorical variables are stored as text wherever possible; the remaining
numeric codes (EVID, MDV, CMT and 0/1 indicators) are defined in the codebook.
"""
from pathlib import Path
import json
import math
import shutil
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pkpy2_extended_manuscript import TITLE                     # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / 'output/pkpy2_development'
REF = ROOT / 'output/pkpy2_nonmem_reference'
CMP = ROOT / 'output/pkpy2_software_comparison'
PAPER = ROOT / 'docs/pkpy2_paper'
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'output/pkpy2_peerj_raw_data'

METHOD_LABEL = {'pkpy': 'PKPy', 'gaussian_tst': 'Gaussian two-stage control', 'pkpy2': 'PKPy2',
                'nlmixr2_focei': 'nlmixr2 FOCEi', 'nlmixr2_saem': 'nlmixr2 SAEM', 'saemix': 'saemix SAEM'}
Z = 1.959963984540054


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def dosing_rows(base, subjects, oral, eta_names=None, etas=None):
    rows = []
    for i, s in enumerate(subjects):
        common = dict(base, ID=s['sid'])
        if 'WT' in s:
            common['WT'] = s['WT']
        if etas is not None:
            common.update({f'ETA_{n}_TRUE': etas[i][k] for k, n in enumerate(eta_names)})
        rows.append(dict(common, TIME=0.0, AMT=s['dose'], DV=None, EVID=1, MDV=1, CMT=1))
        for t, y in zip(s['time'], s['obs']):
            rows.append(dict(common, TIME=t, AMT=None, DV=y, EVID=0, MDV=0, CMT=2 if oral else 1))
    return rows


def primary_records():
    return [(p, read(p)) for p in sorted((DEV / 'confirmatory_v3').glob('1cmt_iv__*__[0-9][0-9][0-9].json'))]


def sheet_primary_data(records):
    rows = []
    for path, rec in records:
        data = read(path.parent / rec['data_file'])
        subjects = [dict(sid=i + 1, time=t, obs=y, dose=data['dose'])
                    for i, (t, y) in enumerate(zip(data['time'], data['observations']))]
        base = dict(dataset=path.stem, model='1cmt_iv', sampling=rec['sampling'], replicate=rec['replicate'],
                    seed=rec['seed'])
        rows += dosing_rows(base, subjects, False, list(rec['truth']['omega']), data['eta'])
    return pd.DataFrame(rows)


def interval_map(fit):
    unc = fit.get('uncertainty') or {}
    if unc.get('status') != 'computed':
        return {}
    names = {'log_theta:CL': 'theta_CL', 'log_theta:V': 'theta_V', 'log_omega_sd:CL': 'omega_CL',
             'log_omega_sd:V': 'omega_V', 'log_sigma_prop': 'sigma_prop'}
    return {names[r['coordinate']]: r['interval'] for r in unc['information']['data']['intervals']
            if r['coordinate'] in names}


def sheet_primary_estimates(records):
    truth = dict(theta_CL=4.0, theta_V=40.0, omega_CL=0.09, omega_V=0.04, sigma_prop=0.15)
    rows = []

    def add(base, method, status, accepted, est, ci, seconds, ci_seconds=None, note=''):
        for p, t in truth.items():
            v = est.get(p)
            if v is None:
                continue
            lo, hi = ci.get(p, (None, None))
            rows.append(dict(base, method=METHOD_LABEL[method], fit_status=status, accepted=int(accepted),
                             parameter=p, estimate=v, true_value=t, relative_error=(v - t) / t,
                             ci95_lower=lo, ci95_upper=hi,
                             ci95_covers_truth=None if lo is None else int(lo <= t <= hi),
                             fit_seconds=seconds, uncertainty_seconds=ci_seconds, note=note))

    for path, rec in records:
        base = dict(dataset=path.stem, sampling=rec['sampling'], replicate=rec['replicate'], seed=rec['seed'])
        f = rec['pkpy2']
        add(base, 'pkpy2', f['status'], f['converged'],
            dict(theta_CL=f['theta']['CL'], theta_V=f['theta']['V'], omega_CL=f['omega']['CL'],
                 omega_V=f['omega']['V'], sigma_prop=f['sigma']['sigma_prop']),
            interval_map(f), f['seconds'], (f.get('uncertainty') or {}).get('seconds'))
        for m in ['pkpy', 'gaussian_tst']:
            g = rec[m]
            ok = g['successful_subjects'] == g['total_subjects']
            add(base, m, f"{g['successful_subjects']}/{g['total_subjects']} individual fits successful", ok,
                dict(theta_CL=g['theta']['CL'], theta_V=g['theta']['V'], omega_CL=g['omega']['CL'],
                     omega_V=g['omega']['V']), {}, g['seconds'],
                note='residual SD fixed at generating value 0.15' if m == 'gaussian_tst' else '')
        for m in ['nlmixr2_focei', 'nlmixr2_saem', 'saemix']:
            p = CMP / 'results/simulation' / f'{path.stem}__{m}.json'
            if not p.exists():
                raise SystemExit(f'missing comparison record {p}')
            r = read(p)
            if r['status'] != 'returned':
                add(base, m, 'error: ' + r.get('message', ''), False, {}, {}, r['seconds'])
                continue
            ci = {}
            for q, k in [('theta_CL', 'CL'), ('theta_V', 'V')]:
                se = r['se_log_theta'].get(k)
                if se is not None and math.isfinite(se) and se > 0:
                    ci[q] = (r['theta'][k] * math.exp(-Z * se), r['theta'][k] * math.exp(Z * se))
            add(base, m, 'returned', True,
                dict(theta_CL=r['theta']['CL'], theta_V=r['theta']['V'], omega_CL=r['omega']['CL'],
                     omega_V=r['omega']['V'], sigma_prop=r['sigma']['prop']), ci, r['seconds'],
                note='; '.join(r.get('warnings') or []))
    return pd.DataFrame(rows)


def sheet_additional():
    data_rows, est_rows = [], []
    for path in sorted((DEV / 'secondary_v3').glob('*__[0-9][0-9][0-9].json')):
        rec = read(path)
        data = read(path.parent / f'{path.stem}_data.json')
        subjects = [dict(sid=i + 1, time=t, obs=y, dose=100.0)
                    for i, (t, y) in enumerate(zip(data['time'], data['observations']))]
        base = dict(dataset=path.stem, model=rec['model'], replicate=rec['replicate'], seed=rec['seed'])
        eta_names = list(rec['truth']['omega'])
        for r in dosing_rows(base, subjects, 'oral' in rec['model'], eta_names, data['eta']):
            data_rows.append(r)
        f = rec['pkpy2']
        truth = rec['truth']
        for group, key in [('theta', k) for k in truth['theta'] if not (k == 'ALAG' and truth['theta'][k] == 0)] + \
                          [('omega', k) for k in truth['omega']] + [('sigma', 'sigma_prop')]:
            t = truth['sigma_prop'] if group == 'sigma' else truth[group][key]
            v = f['sigma']['sigma_prop'] if group == 'sigma' else f[group].get(key)
            est_rows.append(dict(base, method=METHOD_LABEL['pkpy2'], fit_status=f['status'], accepted=int(f['converged']),
                                 parameter=f'{group}_{key}' if group != 'sigma' else 'sigma_prop', estimate=v,
                                 true_value=t, relative_error=(v - t) / t if v is not None else None,
                                 fit_seconds=f['seconds']))
    return pd.DataFrame(data_rows), pd.DataFrame(est_rows)


def sheet_clinical_data():
    frames = []
    for name, source, oral_model in [('theophylline', DEV / 'theophylline_v3/data.json', True),
                                     ('warfarin', REF / 'warfarin_data.json', True)]:
        subjects = read(source)
        for s in subjects:
            s['WT'] = s['covariates']['WT']
        frames.append(pd.DataFrame(dosing_rows(dict(dataset=name), subjects, oral_model)))
    tob = pd.read_csv(CMP / 'data/tobramycin.csv', na_values='.')
    tob = tob[['ID', 'TIME', 'AMT', 'DV', 'EVID', 'MDV', 'CMT', 'WT', 'CLCR']]
    tob.insert(0, 'dataset', 'tobramycin')
    frames.append(tob)
    return pd.concat(frames, ignore_index=True)


def sheet_clinical_estimates():
    cmp = read(CMP / 'comparison_summary.json')['clinical']
    rows = []
    for r in cmp['rows']:
        for m, label in [('nonmem', 'NONMEM reference'), ('pkpy2', METHOD_LABEL['pkpy2']),
                         ('nlmixr2_focei', METHOD_LABEL['nlmixr2_focei']), ('nlmixr2_saem', METHOD_LABEL['nlmixr2_saem']),
                         ('saemix', METHOD_LABEL['saemix'])]:
            v = r.get(m)
            if v is None and m == 'saemix' and r['dataset'] == 'warfarin':
                note = 'not estimated: saemix cannot fix nonzero variance components required by the published model'
            elif v is None and m == 'saemix' and r['dataset'] == 'tobramycin':
                note = 'not estimated: saemix does not support parameter bounds'
            elif v is None:
                note = 'not reported'
            else:
                note = ''
            rows.append(dict(dataset=r['dataset'], parameter=r['label'], software=label, estimate=v,
                             rse_pct=r.get(m + '_rse'),
                             relative_difference_from_nonmem_pct=None if m == 'nonmem' else r.get(m + '_diff_pct'),
                             note=note))
    return pd.DataFrame(rows)


def sheet_numerical():
    n = read(DEV / 'numerical_validation.json')
    num = pd.DataFrame(n['independent_numerical_cases'])
    ref = read(DEV / 'reference_v3/results.json')
    quad = pd.DataFrame([dict(case=c['case'], sampling='sparse' if c['sparse'] else 'rich',
                              wt_effect_on_cl='estimated' if c['covariate'] else 'absent',
                              reference_ofv=c['reference_ofv'], pkpy2_ofv=c['fit']['ofv'],
                              node_refinement_ofv_gap=c['node_refinement_ofv_gap'], same_point_ofv_gap=c['engine_ofv_gap'],
                              maximum_coordinate_gap=c['maximum_coordinate_gap'],
                              hessian_step_relative_gap=c['hessian_step_relative_gap'],
                              maximum_relative_coordinate_se_gap=c['maximum_relative_coordinate_se_gap'],
                              passed=int(c['passed'])) for c in ref['cases']])
    return num, quad


def sheet_timing():
    speed = read(DEV / 'recurrence_benchmark.json')
    rows = []
    for r in speed['rows']:
        for method, values in r['timings'].items():
            for b, v in enumerate(values):
                rows.append(dict(model=r['model'], dose_events=r['doses'], observations=r['observations'],
                                 repetitions_per_batch=r['repetitions'], method=method, batch=b + 1,
                                 seconds_per_call=v))
    return pd.DataFrame(rows)


README = [
    ('Title', 'PKPy2 raw data: simulated and clinical datasets, per-replicate estimates, numerical checks and timings'),
    ('Article', TITLE),
    ('Journal', 'Journal of Pharmacokinetics and Pharmacodynamics'),
    ('Authors', 'Hyunseung Kong, Inyoung Kim'),
    ('Corresponding author', 'Inyoung Kim, Department of Defense Science, Korea National Defense University, Nonsan, Republic of Korea; inyoungkim@korea.kr'),
    ('Description', 'Raw data underlying all tables and figures of the PKPy2 manuscript. Each sheet is also provided as a CSV file with the same name. Variable definitions and codes are given in the accompanying codebook (PKPy2_codebook.xlsx).'),
    ('S1_primary_sim_data', 'Concentration-time records of the 200 primary simulation datasets (one-compartment IV bolus; 100 rich, 100 sparse), with the true individual random effects used for simulation.'),
    ('S2_primary_estimates', 'Per-dataset population estimates from PKPy2, PKPy, the Gaussian two-stage control, nlmixr2 FOCEi, nlmixr2 SAEM and saemix, with 95% intervals where available (Figures 2, 3 and 5; Tables S1-S3, 5).'),
    ('S3_additional_sim_data', 'Concentration-time records of the 30 additional-structure datasets (one-compartment oral, two-compartment IV, two-compartment oral).'),
    ('S4_additional_estimates', 'PKPy2 estimates for the 30 additional-structure datasets (Table S5).'),
    ('S5_clinical_data', 'Theophylline, warfarin and tobramycin analysis datasets in NONMEM-style event format, as analysed by every program.'),
    ('S6_clinical_estimates', 'Clinical parameter estimates from the published expert NONMEM analyses, PKPy2, nlmixr2 FOCEi, nlmixr2 SAEM and saemix (Tables 3-5, Figure 4). Tobramycin estimates are those obtained with the expert-judgment constraints (Supplementary S10).'),
    ('S7_prediction_checks', 'Maximum scaled discrepancies of PKPy2 predictions, sensitivities and scores against independent calculations for 48 conditions.'),
    ('S8_quadrature_checks', 'Independent Gaussian-quadrature comparisons of likelihoods, optima and standard errors (Table 2).'),
    ('S9_timing', 'Per-batch prediction-call timings for direct summation and recurrence (Table S10).'),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records = primary_records()
    add_data, add_est = sheet_additional()
    num, quad = sheet_numerical()
    sheets = {
        'S1_primary_sim_data': sheet_primary_data(records),
        'S2_primary_estimates': sheet_primary_estimates(records),
        'S3_additional_sim_data': add_data,
        'S4_additional_estimates': add_est,
        'S5_clinical_data': sheet_clinical_data(),
        'S6_clinical_estimates': sheet_clinical_estimates(),
        'S7_prediction_checks': num,
        'S8_quadrature_checks': quad,
        'S9_timing': sheet_timing(),
    }
    from pkpy2_extended_raw_data import sheets as extended_sheets, README as EXTENDED_README
    sheets.update(extended_sheets())
    README.extend(EXTENDED_README)
    csv_dir = OUT / 'PKPy2_raw_data_csv'
    if csv_dir.exists():
        shutil.rmtree(csv_dir)
    csv_dir.mkdir()
    with pd.ExcelWriter(OUT / 'PKPy2_raw_data.xlsx', engine='openpyxl') as xw:
        pd.DataFrame(README, columns=['Item', 'Description']).to_excel(xw, sheet_name='README', index=False)
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name, index=False)
            df.to_csv(csv_dir / f'{name}.csv', index=False, encoding='utf-8')
    for name, df in sheets.items():
        print(name, df.shape)
    shutil.make_archive(str(OUT / 'PKPy2_raw_data_csv'), 'zip', csv_dir)


if __name__ == '__main__':
    main()
