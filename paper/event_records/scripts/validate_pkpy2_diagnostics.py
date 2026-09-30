"""Diagnostics of PKPy2 compared with nlmixr2 at identical parameter values (theophylline).

Usage:
  python validate_pkpy2_diagnostics.py write     # data + PKPy2 estimates for R
  Rscript validate_pkpy2_diagnostics.R           # nlmixr2 posthoc EBEs, PRED, IPRED, IWRES, CWRES, NPDE
  python validate_pkpy2_diagnostics.py compare   # agreement summary
"""
from pathlib import Path
import csv
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'packages/pkpy2/src'))
import pkpy2                                     # noqa: E402

DEV = ROOT / 'output/pkpy2_development'
OUT = ROOT / 'output/pkpy2_extended_validation/diagnostics'


def classic_result():
    from pkpy2 import Subject, Parameter as P, Covariate, ModelSpec
    rows = json.loads((DEV / 'theophylline_v3/data.json').read_text())
    subjects = [Subject(r['sid'], np.array(r['time']), np.array(r['obs']), r['dose'], r['covariates']) for r in rows]
    est = json.loads((DEV / 'theophylline_v3/proportional.json').read_text())
    return subjects, pkpy2.load_fit(DEV / 'theophylline_v3/proportional.json', subjects), est


def write():
    OUT.mkdir(parents=True, exist_ok=True)
    subjects, result, est = classic_result()
    with (OUT / 'theophylline.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['ID', 'TIME', 'AMT', 'DV', 'EVID', 'CMT', 'WT'])
        for s in subjects:
            w.writerow([s.sid, 0, s.dose, '.', 1, 1, s.covariates['WT']])
            for t, y in zip(s.time, s.obs):
                w.writerow([s.sid, t, 0, y, 0, 2, s.covariates['WT']])
    (OUT / 'estimates.json').write_text(json.dumps(dict(theta=est['theta'], omega=est['omega'], sigma=est['sigma'])))
    print('written', OUT)


def npde_export():
    """Observed data and PKPy2 replicates in the npde package format, plus PKPy2's NPDE on the same replicates."""
    subjects, result, est = classic_result()
    sims = pkpy2.simulate(result, 1000, seed=20261001)
    table, _ = pkpy2.diagnostics(result, simulations=sims)
    with (OUT / 'npde_obs.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['ID', 'TIME', 'DV'])
        for i, t, y in zip(sims['ID'], sims['TIME'], sims['DV']):
            w.writerow([i, t, y])
    with (OUT / 'npde_sim.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['ID', 'TIME', 'DV'])
        for r in range(sims['SIM'].shape[0]):
            for i, t, y in zip(sims['ID'], sims['TIME'], sims['SIM'][r]):
                w.writerow([i, t, y])
    with (OUT / 'npde_pkpy2.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['ID', 'TIME', 'NPDE', 'PD'])
        for i, t, a, b in zip(table['ID'], table['TIME'], table['NPDE'], table['PD']):
            w.writerow([i, t, a, b])
    print('npde inputs written')


def compare():
    subjects, result, est = classic_result()
    table, summary = pkpy2.diagnostics(result, npde_samples=2000, cwres='diagonal', npde='cholesky_upper')
    ebe = pkpy2.individual_estimates(result)
    ref = list(csv.DictReader((OUT / 'nlmixr2_posthoc.csv').open()))
    key = lambda sid, t: (int(float(sid)), round(float(t), 6))
    ours = {key(i, t): k for k, (i, t) in enumerate(zip(table['ID'], table['TIME']))}
    pairs = {c: [] for c in ('PRED', 'IPRED', 'IWRES', 'CWRES', 'NPDE')}
    for r in ref:
        k = ours.get(key(r['ID'], r['TIME']))
        if k is None:
            continue
        for c in pairs:
            pairs[c].append((table[c][k], float(r[c])))
    with (OUT / 'paired.csv').open('w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['quantity', 'pkpy2', 'nlmixr2'])
        for c, v in pairs.items():
            for a, b in v:
                w.writerow([c, a, b])
    report = {}
    for c, v in pairs.items():
        a = np.array(v)
        ok = np.all(np.isfinite(a), axis=1)
        a = a[ok]
        report[c] = dict(n=int(len(a)), max_abs_difference=float(np.max(np.abs(a[:, 0] - a[:, 1]))),
                         correlation=float(np.corrcoef(a[:, 0], a[:, 1])[0, 1]),
                         mean_pkpy2=float(a[:, 0].mean()), mean_nlmixr2=float(a[:, 1].mean()),
                         var_pkpy2=float(a[:, 0].var(ddof=1)), var_nlmixr2=float(a[:, 1].var(ddof=1)))
    ebe_ref = {}
    for r in ref:
        ebe_ref[int(float(r['ID']))] = [float(r['eta.cl']), float(r['eta.v']), float(r['eta.ka'])]
    diffs = []
    for row in ebe:
        mine = [row['eta_mode'][n] for n in ('CL', 'V', 'Ka')]
        diffs.append(np.max(np.abs(np.array(mine) - np.array(ebe_ref[int(row['id'])]))))
    report['EBE'] = dict(n=len(diffs), max_abs_difference=float(max(diffs)))
    report['conventions'] = 'nlmixr2 conventions: CWRES standardized by its own variance; NPDE with U^-1'
    npde_ref = OUT / 'npde_package.csv'
    if npde_ref.exists():
        a = list(csv.DictReader((OUT / 'npde_pkpy2.csv').open()))
        b = list(csv.DictReader(npde_ref.open()))
        x = np.array([float(r['NPDE']) for r in a])
        y = np.array([float(r['npde']) for r in b])
        report['NPDE_vs_npde_package_same_replicates'] = dict(n=len(x), max_abs_difference=float(np.max(np.abs(x - y))),
                                                             correlation=float(np.corrcoef(x, y)[0, 1]))
    _, summary = pkpy2.diagnostics(result, npde_samples=2000)
    report['summary'] = summary
    (OUT.parent / 'diagnostics_agreement.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    {'write': write, 'compare': compare, 'npde': npde_export}[sys.argv[1]]()
