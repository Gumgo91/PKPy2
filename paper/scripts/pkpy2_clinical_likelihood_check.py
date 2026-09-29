"""Independent marginal-likelihood check of the clinical estimates from each program.

The exact (Gaussian) marginal OFV = -2 log L of each declared model is evaluated by adaptive
Gauss-Hermite quadrature at the estimates reported by PKPy2, nlmixr2 FOCEi, nlmixr2 SAEM,
saemix, and the published expert NONMEM analyses. This implementation shares no code with
PKPy2. Normal-density constants are included, as in PKPy2. The published NONMEM results do
not report the residual SD; for them, sigma_prop is set to the value that maximizes the
likelihood given the other published estimates.
"""
from pathlib import Path
import csv
import itertools
import json
import math

import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.optimize import minimize, minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / 'output/pkpy2_development'
REF = ROOT / 'output/pkpy2_nonmem_reference'
CMP = ROOT / 'output/pkpy2_software_comparison'
TOB = ROOT / 'output/pkpy2_tobramycin_v3'
TOBRA_ADD_SD = math.sqrt(1.85e-6)

# Published expert NONMEM estimates (PKGPT article, Tables 2 and 3 and text).
PUBLISHED = {
    'theophylline': dict(theta=dict(CL=0.0404 * 70, V=0.465 * 70, Ka=1.46), omega=dict(CL=0.0646, V=0.0145, Ka=0.445)),
    'warfarin': dict(theta=dict(CL=0.135, V=7.86, Ka=1.15, ALAG=0.825), omega=dict(CL=0.0643, Ka=0.5), sigma_add=math.sqrt(0.117)),
    'tobramycin': dict(theta=dict(CL=2.95, V1=4.59, Q=6.85, V2=13.2), omega=dict(CL=0.028),
                       coefficients=dict(b_clcr=0.236, b_wt=1.07), sigma_add=TOBRA_ADD_SD),
}


# ------------------------------------------------------------------ data
def oral_subjects(path):
    subs = json.loads(Path(path).read_text(encoding='utf-8'))
    return [dict(time=np.asarray(s['time']), obs=np.asarray(s['obs']), doses=[(0.0, s['dose'])], WT=s['covariates']['WT'])
            for s in subs]


def tobra_subjects():
    rows = list(csv.DictReader((CMP / 'source/tobramycin_github.csv').read_text(encoding='utf-8').lstrip('#').splitlines()))
    out = []
    for sid in dict.fromkeys(r['ID'] for r in rows):
        rr = [r for r in rows if r['ID'] == sid]
        obs = [r for r in rr if r['EVID'] == '0']
        out.append(dict(time=np.array([float(r['TIME']) for r in obs]), obs=np.array([float(r['DV']) for r in obs]),
                        doses=[(float(r['TIME']), float(r['AMT'])) for r in rr if r['EVID'] == '1'],
                        WT=float(rr[0]['WT']), CLCR=float(rr[0]['CLCR'])))
    return out


# ------------------------------------------------------------------ predictions
def oral_pred(s, ind, lag):
    k = ind['CL'] / ind['V']
    out = np.zeros_like(s['time'])
    for t0, amt in s['doses']:
        tt = np.maximum(s['time'] - t0 - lag, 0.0)
        out += amt * ind['Ka'] / (ind['V'] * (ind['Ka'] - k)) * (np.exp(-k * tt) - np.exp(-ind['Ka'] * tt))
    return out


def iv2_pred(s, ind):
    k10, k12, k21 = ind['CL'] / ind['V1'], ind['Q'] / ind['V1'], ind['Q'] / ind['V2']
    tot = k10 + k12 + k21
    root = math.sqrt(tot * tot - 4 * k10 * k21)
    a, b = (tot + root) / 2, (tot - root) / 2
    out = np.zeros_like(s['time'])
    for t0, amt in s['doses']:
        dt = s['time'] - t0
        m = dt >= 0
        out[m] += amt * ((a - k21) * np.exp(-a * dt[m]) + (k21 - b) * np.exp(-b * dt[m])) / ((a - b) * ind['V1'])
    return out


MODELS = {
    'theophylline': dict(subjects=lambda: oral_subjects(DEV / 'theophylline_v3/data.json'), eta=['CL', 'V', 'Ka']),
    'warfarin': dict(subjects=lambda: oral_subjects(REF / 'warfarin_data.json'), eta=['CL', 'Ka']),
    'tobramycin': dict(subjects=tobra_subjects, eta=['CL']),
}


def individual(dataset, s, p, eta):
    th = dict(p['theta'])
    for n, e in zip(MODELS[dataset]['eta'], eta):
        th[n] *= math.exp(e)
    if dataset == 'theophylline':
        th['CL'] *= s['WT'] / 70
        th['V'] *= s['WT'] / 70
    elif dataset == 'warfarin':
        th['CL'] *= (s['WT'] / 70) ** 0.75
        th['V'] *= s['WT'] / 70
    else:
        th['CL'] *= (s['CLCR'] / 58) ** p['coefficients']['b_clcr']
        th['V1'] *= (s['WT'] / 62) ** p['coefficients']['b_wt']
    return th


def predict(dataset, s, ind, p):
    if dataset.startswith('tobramycin'):
        return iv2_pred(s, ind)
    return oral_pred(s, ind, p['theta'].get('ALAG', 0.0))


def subject_ofv(dataset, s, p, nodes):
    names = MODELS[dataset]['eta']
    om = np.array([p['omega'][n] for n in names])

    def h(eta):
        f = predict(dataset, s, individual(dataset, s, p, eta), p)
        var = p['sigma_prop'] ** 2 * f ** 2 + p['sigma_add'] ** 2
        return (0.5 * np.sum(np.log(2 * math.pi * var) + (s['obs'] - f) ** 2 / var)
                + 0.5 * np.sum(np.log(2 * math.pi * om) + np.asarray(eta) ** 2 / om))

    mode = minimize(h, np.zeros(len(names)), method='BFGS', options=dict(gtol=1e-8)).x
    d, step = len(mode), 1e-4
    hess = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            ei, ej = np.eye(d)[i] * step, np.eye(d)[j] * step
            hess[i, j] = (h(mode + ei + ej) - h(mode + ei - ej) - h(mode - ei + ej) + h(mode - ei - ej)) / (4 * step * step)
    L = np.linalg.cholesky(np.linalg.inv(hess))
    z, wz = hermegauss(nodes)
    vals = np.array([-h(mode + L @ z[list(idx)]) + 0.5 * z[list(idx)] @ z[list(idx)] + np.sum(np.log(wz[list(idx)]))
                     for idx in itertools.product(range(nodes), repeat=d)])
    m = vals.max()
    return -2 * (m + math.log(np.exp(vals - m).sum()) + math.log(abs(np.linalg.det(L))))


NODES = {'theophylline': (9, 13), 'warfarin': (15, 21), 'tobramycin': (21, 31)}


def ofv(dataset, p, nodes, subs):
    return sum(subject_ofv(dataset, s, p, nodes) for s in subs)


# ------------------------------------------------------------------ estimates
def estimates():
    out = {}
    for dataset, path in [('theophylline', DEV / 'theophylline_v3/proportional.json'), ('warfarin', DEV / 'warfarin_v3/warfarin_start_1.json'),
                          ('tobramycin', TOB / 'expert_start_1.json')]:
        f = json.loads(path.read_text(encoding='utf-8'))
        p = dict(theta=f['theta'], omega=f['omega'], sigma_prop=f['sigma']['sigma_prop'], sigma_add=f['sigma']['sigma_add'],
                 reported_ofv=f['ofv'])
        if dataset.startswith('tobramycin'):
            p['coefficients'] = dict(b_clcr=f['coefficients'][0], b_wt=f['coefficients'][1])
        out[(dataset, 'pkpy2')] = p
        for m in ['nlmixr2_focei', 'nlmixr2_saem', 'saemix']:
            q = CMP / 'results/clinical' / f"{'tobramycin_expert' if dataset == 'tobramycin' else dataset}__{m}.json"
            if not q.exists():
                continue
            r = json.loads(q.read_text(encoding='utf-8'))
            if r['status'] != 'returned':
                continue
            omega = dict(r['omega'])
            if dataset == 'warfarin':
                omega.setdefault('Ka', 0.5)
            p = dict(theta=r['theta'], omega=omega, sigma_prop=r['sigma']['prop'],
                     sigma_add=r['sigma'].get('add', TOBRA_ADD_SD if dataset.startswith('tobramycin') else 0.0))
            if dataset.startswith('tobramycin'):
                c = r['coefficients']
                p['coefficients'] = dict(b_clcr=c['b.clcr'], b_wt=c.get('b.wt', 1.0))  # WT exponent fixed at 1 in the expert-judgment model
            out[(dataset, m)] = p
        if dataset not in PUBLISHED:
            continue
        pub = PUBLISHED[dataset]
        out[(dataset, 'nonmem')] = dict(theta=pub['theta'], omega=pub['omega'], coefficients=pub.get('coefficients'),
                                        sigma_prop=None, sigma_add=pub.get('sigma_add', 0.0))
    return out


def main():
    rows = []
    for (dataset, method), p in estimates().items():
        subs = MODELS[dataset]['subjects']()
        coarse, fine = NODES[dataset]
        profiled = p['sigma_prop'] is None
        if profiled:
            res = minimize_scalar(lambda ls: ofv(dataset, dict(p, sigma_prop=math.exp(ls)), coarse, subs),
                                  bounds=(math.log(0.02), math.log(1.0)), method='bounded', options=dict(xatol=1e-4))
            p = dict(p, sigma_prop=math.exp(res.x))
        a, b = ofv(dataset, p, coarse, subs), ofv(dataset, p, fine, subs)
        rows.append(dict(dataset=dataset, method=method, ofv=b, node_refinement_gap=abs(a - b),
                         sigma_prop_profiled=profiled, sigma_prop=p['sigma_prop'],
                         pkpy2_reported_ofv=p.get('reported_ofv')))
        print(dataset, method, round(b, 3), 'gap', f'{abs(a - b):.1e}', 'sigma', round(p['sigma_prop'], 4),
              '(profiled)' if profiled else '', 'reported', p.get('reported_ofv'), flush=True)
    for dataset in MODELS:
        base = next(r['ofv'] for r in rows if r['dataset'] == dataset and r['method'] == 'pkpy2')
        for r in rows:
            if r['dataset'] == dataset:
                r['delta_ofv_vs_pkpy2'] = r['ofv'] - base
    (CMP / 'clinical_likelihood_check.json').write_text(json.dumps(rows, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
