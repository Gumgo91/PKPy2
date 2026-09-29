"""Profile likelihood of the tobramycin model in V2, computed independently of PKPy2.

The marginal likelihood with one random effect (CL) is integrated by 60-node Gauss-Hermite
quadrature, vectorized over subjects; predictions use closed-form two-compartment bolus
superposition. For each fixed V2 the remaining parameters are re-optimized from several
starts. The value at the PKPy2 estimate is compared with PKPy2's reported OFV.
"""
from pathlib import Path
import csv
import json
import math

import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
CMP = ROOT / 'output/pkpy2_software_comparison'
OUT = ROOT / 'output/pkpy2_tobramycin_v3'
ADD = math.sqrt(1.85e-6)
Z, W = hermegauss(60)
LOGW = np.log(W / math.sqrt(2 * math.pi))


def load():
    rows = list(csv.DictReader((CMP / 'source/tobramycin_github.csv').read_text(encoding='utf-8').lstrip('#').splitlines()))
    subs = []
    for sid in dict.fromkeys(r['ID'] for r in rows):
        rr = [r for r in rows if r['ID'] == sid]
        t = np.array([float(r['TIME']) for r in rr if r['EVID'] == '0'])
        dt_ = np.array([float(r['TIME']) for r in rr if r['EVID'] == '1'])
        amt = np.array([float(r['AMT']) for r in rr if r['EVID'] == '1'])
        dt = t[:, None] - dt_[None, :]
        subs.append(dict(y=np.array([float(r['DV']) for r in rr if r['EVID'] == '0']), dt=np.where(dt >= 0, dt, 0.0),
                         mask=(dt >= 0).astype(float), amt=amt, lclcr=math.log(float(rr[0]['CLCR']) / 58),
                         lwt=math.log(float(rr[0]['WT']) / 62)))
    return subs


def ofv(p, subs):
    cl, v1, q, v2, bc, bw, om, sp = p
    total = 0.0
    for s in subs:
        cli = cl * math.exp(bc * s['lclcr']) * np.exp(math.sqrt(om) * Z)          # nodes
        v1i = v1 * math.exp(bw * s['lwt'])
        k10, k12, k21 = cli / v1i, q / v1i, q / v2
        tot = k10 + k12 + k21
        root = np.sqrt(tot * tot - 4 * k10 * k21)
        a, b = (tot + root) / 2, (tot - root) / 2
        dt = s['dt'][None]
        unit = ((a - k21)[:, None, None] * np.exp(-a[:, None, None] * dt) + (k21 - b)[:, None, None] * np.exp(-b[:, None, None] * dt)) \
            / ((a - b)[:, None, None] * v1i)
        f = (unit * s['mask'][None] * s['amt'][None, None, :]).sum(axis=2)          # nodes x obs
        var = sp * sp * f * f + ADD * ADD
        ll = -0.5 * (np.log(2 * math.pi * var) + (s['y'][None] - f) ** 2 / var).sum(axis=1) + LOGW
        m = ll.max()
        total += m + math.log(np.exp(ll - m).sum())
    return -2 * total


def fit_given_v2(v2, starts, subs, fixed_bw=None):
    best = None
    for s in starts:
        x0 = np.array([math.log(s[0]), math.log(s[1]), math.log(s[2]), s[4], s[5], math.log(s[6]), math.log(s[7])])

        def obj(x):
            bw = x[4] if fixed_bw is None else fixed_bw
            p = (math.exp(x[0]), math.exp(x[1]), math.exp(x[2]), v2, x[3], bw, math.exp(x[5]), math.exp(x[6]))
            return ofv(p, subs)
        r = minimize(obj, x0, method='Nelder-Mead', options=dict(maxiter=4000, maxfev=4000, xatol=1e-5, fatol=1e-6))
        r = minimize(obj, r.x, method='Nelder-Mead', options=dict(maxiter=3000, maxfev=3000, xatol=1e-6, fatol=1e-7))
        if best is None or r.fun < best[0]:
            x = r.x
            best = (r.fun, dict(CL=math.exp(x[0]), V1=math.exp(x[1]), Q=math.exp(x[2]), V2=v2, b_clcr=x[3],
                                b_wt=x[4] if fixed_bw is None else fixed_bw,
                                omega_CL=math.exp(x[5]), sigma_prop=math.exp(x[6])))
    return best


def main():
    import sys
    reduced = '--reduced' in sys.argv
    subs = load()
    f = json.loads((OUT / ('reduced_start_1.json' if reduced else 'start_1.json')).read_text(encoding='utf-8'))
    pk = (f['theta']['CL'], f['theta']['V1'], f['theta']['Q'], f['theta']['V2'], f['coefficients'][0], f['coefficients'][1],
          f['omega']['CL'], f['sigma']['sigma_prop'])
    check = dict(pkpy2_reported=f['ofv'], independent=ofv(pk, subs))
    print('OFV at PKPy2 estimate: reported %.3f, independent %.3f' % (check['pkpy2_reported'], check['independent']), flush=True)
    expert = (2.95, 4.59, 6.85, 13.2, 0.236, 1.07, 0.028, 0.22)
    starts = [pk, expert, (3.3, 16.0, 5.0, 7.0, 0.34, -7.4, 0.02, 0.22)]
    rows = []
    for v2 in [5, 7, 10, 13.2, 20, 30, 50, 100, 188, 300, 600]:
        val, p = fit_given_v2(v2, starts, subs, fixed_bw=1.0 if reduced else None)
        rows.append(dict(V2=v2, ofv=val, **{k: v for k, v in p.items() if k != 'V2'}))
        print('V2 %6.1f  OFV %.3f  CL %.3f V1 %.2f Q %.3f bCLCR %.3f bWT %.2f omega %.4f sigma %.4f' % (
            v2, val, p['CL'], p['V1'], p['Q'], p['b_clcr'], p['b_wt'], p['omega_CL'], p['sigma_prop']), flush=True)
    (OUT / ('profile_v2_reduced.json' if reduced else 'profile_v2.json')).write_text(json.dumps(dict(check=check, profile=rows), indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
