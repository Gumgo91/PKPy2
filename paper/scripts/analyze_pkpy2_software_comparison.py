"""Summarise the PKPy2 comparison with nlmixr2 (FOCEi, SAEM) and saemix.

Reads the saved PKPy2 primary-simulation records (confirmatory_v3), the external-software
records written by scripts/pkpy2_software_comparison.R, and the clinical reference
comparison. Metrics follow the primary PKPy2 analysis: relative bias with Monte Carlo SE,
relative RMSE, Wald 95% interval coverage with Wilson intervals, and paired-bootstrap
differences in relative RMSE (10,000 resamples of datasets returned by both methods).
"""
from pathlib import Path
import csv
import json
import math

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / 'output/pkpy2_development'
BASE = ROOT / 'output/pkpy2_software_comparison'
PAPER = ROOT / 'docs/pkpy2_paper'
METHODS = ['pkpy2', 'nlmixr2_focei', 'nlmixr2_saem', 'saemix']
LABELS = {'pkpy2': 'PKPy2', 'nlmixr2_focei': 'nlmixr2 FOCEi', 'nlmixr2_saem': 'nlmixr2 SAEM', 'saemix': 'saemix'}
PARAMS = ['theta_CL', 'theta_V', 'omega_CL', 'omega_V', 'sigma_prop']
TRUTH = {'theta_CL': 4.0, 'theta_V': 40.0, 'omega_CL': 0.09, 'omega_V': 0.04, 'sigma_prop': 0.15}
Z = 1.959963984540054
BOOT = 10_000


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def wilson(k, n):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return [c - h, c + h]


def pkpy2_record(rec):
    f = rec['pkpy2']
    ints = {r['coordinate']: r for r in f['uncertainty']['information']['data']['intervals']} \
        if f.get('uncertainty', {}).get('status') == 'computed' else {}
    est = dict(theta_CL=f['theta']['CL'], theta_V=f['theta']['V'], omega_CL=f['omega']['CL'],
               omega_V=f['omega']['V'], sigma_prop=f['sigma']['sigma_prop'])
    ci = {}
    for p, c in [('theta_CL', 'log_theta:CL'), ('theta_V', 'log_theta:V')]:
        if c in ints:
            ci[p] = ints[c]['interval']
    return dict(returned=True, converged=bool(f['converged']), estimates=est, intervals=ci,
                seconds=f['seconds'], ci_seconds=f.get('uncertainty', {}).get('seconds'))


def external_record(rec):
    if rec['status'] != 'returned':
        return dict(returned=False, converged=False, estimates={}, intervals={}, seconds=rec['seconds'],
                    message=rec.get('message'))
    th, om, sg = rec['theta'], rec['omega'], rec['sigma']
    est = dict(theta_CL=th['CL'], theta_V=th['V'], omega_CL=om['CL'], omega_V=om['V'], sigma_prop=sg['prop'])
    ci = {}
    for p, k in [('theta_CL', 'CL'), ('theta_V', 'V')]:
        se = (rec.get('se_log_theta') or {}).get(k)
        if se is not None and math.isfinite(se) and se > 0:
            ci[p] = [th[k] * math.exp(-Z * se), th[k] * math.exp(Z * se)]
    finite = all(v is not None and math.isfinite(v) and v > 0 for v in est.values())
    return dict(returned=finite, converged=finite, estimates=est, intervals=ci, seconds=rec['seconds'],
                covariance_available=bool(rec.get('covariance_available')), message=rec.get('message', ''),
                warnings=rec.get('warnings') or [])


def load_simulation():
    rows = {}
    for path in sorted((DEV / 'confirmatory_v3').glob('1cmt_iv__*__[0-9][0-9][0-9].json')):
        rec = read(path)
        rows[(path.stem, 'pkpy2')] = dict(sampling=rec['sampling'], replicate=rec['replicate'], seed=rec['seed'],
                                          **pkpy2_record(rec))
        for m in METHODS[1:]:
            ext = BASE / 'results/simulation' / f'{path.stem}__{m}.json'
            if ext.exists():
                rows[(path.stem, m)] = dict(sampling=rec['sampling'], replicate=rec['replicate'], seed=rec['seed'],
                                            **external_record(read(ext)))
    return rows


def summarise(rows):
    datasets = sorted({d for d, _ in rows})
    out = dict(estimation=[], coverage=[], status=[], paired_rmse=[])
    rng = np.random.default_rng(20260924)
    for sampling in ['rich', 'sparse']:
        ds = [d for d in datasets if rows[(d, 'pkpy2')]['sampling'] == sampling]
        boot_index = rng.integers(0, len(ds), size=(BOOT, len(ds)))
        for m in METHODS:
            recs = [rows.get((d, m)) for d in ds]
            present = [r for r in recs if r is not None]
            ok = [r for r in present if r['returned']]
            secs = [r['seconds'] for r in present]
            out['status'].append(dict(sampling=sampling, method=m, planned=len(ds), completed=len(present),
                                      returned=len(ok), converged=sum(r['converged'] for r in present),
                                      median_fit_seconds=float(np.median(secs)) if secs else None,
                                      median_uncertainty_seconds=float(np.median([r['ci_seconds'] for r in ok if r.get('ci_seconds')]))
                                      if m == 'pkpy2' else None,
                                      covariance_available=sum(bool(r.get('covariance_available', True)) for r in ok)))
            for p in PARAMS:
                e = np.array([(r['estimates'][p] - TRUTH[p]) / TRUTH[p] for r in ok])
                if len(e) == 0:
                    continue
                out['estimation'].append(dict(sampling=sampling, method=m, parameter=p, n=len(e),
                                              relative_bias_pct=100 * e.mean(),
                                              bias_mcse_pct=100 * e.std(ddof=1) / math.sqrt(len(e)),
                                              relative_rmse_pct=100 * math.sqrt((e ** 2).mean())))
            for p in ['theta_CL', 'theta_V']:
                with_ci = [r for r in ok if p in r['intervals']]
                k = sum(r['intervals'][p][0] <= TRUTH[p] <= r['intervals'][p][1] for r in with_ci)
                out['coverage'].append(dict(sampling=sampling, method=m, parameter=p, available=len(with_ci),
                                            planned=len(ds), covered=k,
                                            rate=k / len(with_ci) if with_ci else None,
                                            wilson=wilson(k, len(with_ci))))
        # paired bootstrap PKPy2 minus comparator on datasets returned by both
        for m in METHODS[1:]:
            both = [d for d in ds if rows.get((d, m), {}).get('returned') and rows[(d, 'pkpy2')]['returned']]
            if len(both) < 2:
                continue
            idx = rng.integers(0, len(both), size=(BOOT, len(both)))
            for p in PARAMS:
                a = np.array([(rows[(d, 'pkpy2')]['estimates'][p] - TRUTH[p]) / TRUTH[p] for d in both])
                b = np.array([(rows[(d, m)]['estimates'][p] - TRUTH[p]) / TRUTH[p] for d in both])
                diff = 100 * (math.sqrt((a ** 2).mean()) - math.sqrt((b ** 2).mean()))
                boot = 100 * (np.sqrt((a[idx] ** 2).mean(axis=1)) - np.sqrt((b[idx] ** 2).mean(axis=1)))
                out['paired_rmse'].append(dict(sampling=sampling, comparator=m, parameter=p, n_pairs=len(both),
                                               rmse_difference_pct=diff,
                                               bootstrap_95_interval=[float(np.quantile(boot, .025)),
                                                                      float(np.quantile(boot, .975))]))
    return out


# Published expert NONMEM estimates and RSEs (PKGPT article [11], Tables 2 and 3 and text).
# Theophylline CL and V are reported per kg and are multiplied by 70 kg. None = not reported.
# (label, group, key, PKPy2 coordinate, published estimate, published RSE %)
REFERENCE = {
    'theophylline': [
        ('CL/F (L/h)', 'theta', 'CL', 'log_theta:CL', 0.0404 * 70, 8.0),
        ('V/F (L)', 'theta', 'V', 'log_theta:V', 0.465 * 70, 4.3),
        ('Ka (h⁻¹)', 'theta', 'Ka', 'log_theta:Ka', 1.46, 21.4),
        ('ω²(CL)', 'omega', 'CL', 'log_omega_sd:CL', 0.0646, None),
        ('ω²(V)', 'omega', 'V', 'log_omega_sd:V', 0.0145, 42.2),
        ('ω²(Ka)', 'omega', 'Ka', 'log_omega_sd:Ka', 0.445, None),
        ('σ_prop', 'sigma', 'prop', 'log_sigma_prop', None, None)],
    'warfarin': [
        ('CL/F (L/h)', 'theta', 'CL', 'log_theta:CL', 0.135, 4.8),
        ('V/F (L)', 'theta', 'V', 'log_theta:V', 7.86, 3.0),
        ('Ka (h⁻¹)', 'theta', 'Ka', 'log_theta:Ka', 1.15, 30.5),
        ('ALAG (h)', 'theta', 'ALAG', 'log_theta:ALAG', 0.825, 8.6),
        ('ω²(CL)', 'omega', 'CL', 'log_omega_sd:CL', 0.0643, 37.3),
        ('σ_prop', 'sigma', 'prop', 'log_sigma_prop', None, None)],
    'tobramycin': [
        ('CL (L/h)', 'theta', 'CL', 'log_theta:CL', 2.95, None),
        ('V1 (L)', 'theta', 'V1', 'log_theta:V1', 4.59, None),
        ('Q (L/h)', 'theta', 'Q', 'log_theta:Q', 6.85, None),
        ('V2 (L)', 'theta', 'V2', 'log_theta:V2', 13.2, None),
        ('CLCR exponent', 'coef', 'b.clcr', 'covariate_coefficient:0', 0.236, None),
        ('ω²(CL)', 'omega', 'CL', 'log_omega_sd:CL', 0.028, None),
        ('σ_prop', 'sigma', 'prop', 'log_sigma_prop', None, None)],
}
PKPY2_CLINICAL = {'theophylline': DEV / 'theophylline_v3/proportional.json',
                  'warfarin': DEV / 'warfarin_v3/warfarin_start_1.json',
                  'tobramycin': ROOT / 'output/pkpy2_tobramycin_v3/expert_start_1.json'}
# Tobramycin is compared under the expert's modeling judgment (WT exponent on V1 fixed at 1,
# V1 <= 10 L, V2 <= 30 L, expert bounds); the R records of that model are 'tobramycin_expert'.
RECORD_KEY = {'tobramycin': 'tobramycin_expert'}


def pkpy2_clinical(dataset, group, key, coord):
    f = read(PKPY2_CLINICAL[dataset])
    ints = {r['coordinate']: r for r in (f.get('uncertainty') or {}).get('information', {}).get('data', {}).get('intervals', [])}
    if group == 'coef':
        v = f['coefficients'][int(coord.split(':')[1])]
    elif group == 'sigma':
        v = f['sigma']['sigma_prop']
    else:
        v = f[group][key]
    se = ints.get(coord, {}).get('se')
    if group == 'coef':
        rse = 100 * se / abs(v) if se is not None else None
    else:
        rse = ints.get(coord, {}).get('rse_pct')
    return v, rse


def clinical():
    rows = []
    records = {}
    for dataset in REFERENCE:
        for m in METHODS[1:]:
            path = BASE / 'results/clinical' / f'{RECORD_KEY.get(dataset, dataset)}__{m}.json'
            records[(dataset, m)] = read(path) if path.exists() else None
    for dataset, params in REFERENCE.items():
        for label, group, key, coord, ref, ref_rse in params:
            v, rse = pkpy2_clinical(dataset, group, key, coord)
            row = dict(dataset=dataset, label=label, coordinate=coord, nonmem=ref, nonmem_rse=ref_rse, pkpy2=v, pkpy2_rse=rse)
            for m in METHODS[1:]:
                rec = records[(dataset, m)]
                v = rse = None
                if rec and rec['status'] == 'returned':
                    if group == 'coef':
                        v = (rec.get('coefficients') or {}).get(key)
                        se = (rec.get('se_coefficients') or {}).get(key)
                        rse = 100 * se / abs(v) if se is not None and v else None
                    else:
                        v = (rec.get(group) or {}).get(key)
                        if group == 'theta':
                            se = (rec.get('se_log_theta') or {}).get(key)
                            rse = 100 * se if se is not None else None
                        elif group == 'sigma':
                            se = (rec.get('se_sigma') or {}).get(key)
                            rse = 100 * se / v if se is not None and v else None
                row[m] = v
                row[m + '_rse'] = rse
            for m in METHODS:
                v = row[m]
                row[m + '_diff_pct'] = 100 * (v - ref) / ref if v is not None and ref else None
            rows.append(row)
    env = next((rec['environment'] for rec in records.values() if rec), None)
    seconds = {f'{d}__{m}': rec['seconds'] for (d, m), rec in records.items() if rec}
    return dict(rows=rows, environment=env, seconds=seconds,
                objectives={f'{d}__{m}': rec.get('objective') for (d, m), rec in records.items() if rec})


def write_replicates(rows, path):
    fields = ['dataset', 'sampling', 'replicate', 'seed', 'method', 'returned', 'parameter', 'estimate', 'truth',
              'relative_error', 'ci_lower', 'ci_upper', 'ci_covers_truth', 'fit_seconds']
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for (d, m), r in sorted(rows.items()):
            for p in PARAMS:
                est = r['estimates'].get(p)
                ci = r['intervals'].get(p)
                w.writerow(dict(dataset=d, sampling=r['sampling'], replicate=r['replicate'], seed=r['seed'], method=m,
                                returned=int(r['returned']), parameter=p, estimate=est, truth=TRUTH[p],
                                relative_error=(est - TRUTH[p]) / TRUTH[p] if est is not None else None,
                                ci_lower=ci[0] if ci else None, ci_upper=ci[1] if ci else None,
                                ci_covers_truth=(int(ci[0] <= TRUTH[p] <= ci[1]) if ci else None),
                                fit_seconds=r['seconds']))


def main():
    rows = load_simulation()
    summary = summarise(rows)
    summary['clinical'] = clinical()
    summary['labels'] = LABELS
    summary['truth'] = TRUTH
    (BASE / 'comparison_summary.json').write_text(json.dumps(summary, indent=1), encoding='utf-8')
    write_replicates(rows, BASE / 'comparison_replicates.csv')
    for s in summary['status']:
        print(s)


if __name__ == '__main__':
    main()
